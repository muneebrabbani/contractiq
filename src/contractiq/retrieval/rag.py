from __future__ import annotations

from openai import OpenAI

from contractiq.config import settings
from contractiq.eval.pipeline import RagPipeline, RagResult
from contractiq.extraction.clause_chunking import load_all_clause_chunks
from contractiq.extraction.db import ContractRecord, get_session
from contractiq.retrieval.bm25_index import Bm25Index, build_bm25_index
from contractiq.retrieval.fusion import reciprocal_rank_fusion
from contractiq.retrieval.models import AnswerResult, Citation, RetrievedChunk
from contractiq.retrieval.reranker import rerank
from contractiq.retrieval.vector_store import dense_search

CANDIDATE_POOL_SIZE = 20
DEFAULT_K = 5

SYSTEM_PROMPT = (
    "You are a contract question-answering assistant for a telecom procurement team. "
    "Answer strictly and only using the provided context passages -- do not use outside "
    "knowledge and do not guess. If the passages do not contain enough information to "
    "answer, say so explicitly rather than speculating. Every factual claim in your answer "
    "must be followed by a citation in the exact format [document, section, page] using the "
    "labels given with each passage."
)

_bm25_index: Bm25Index | None = None


def _get_bm25_index() -> Bm25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = build_bm25_index(load_all_clause_chunks())
    return _bm25_index


def refresh_bm25_index() -> None:
    """Call after re-indexing so the in-memory BM25 index picks up new chunks."""
    global _bm25_index
    _bm25_index = None


def _build_where(
    segment: str | None, status: str | None, doc_ids: set[str] | None
) -> dict | None:
    clauses = []
    if segment is not None:
        clauses.append({"segment": segment})
    if status is not None:
        clauses.append({"status": status})
    if doc_ids is not None:
        clauses.append({"doc_id": {"$in": sorted(doc_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _resolve_allowed_doc_ids(
    segment: str | None, status: str | None, doc_ids: set[str] | None
) -> set[str] | None:
    """doc_ids narrows to a specific document set (e.g. a vendor named in the
    question); segment/status narrow by metadata. Both apply identically to
    the dense and BM25 legs so fusion never mixes a filtered leg with an
    unfiltered one -- combined here by intersection when both are given."""
    if segment is None and status is None:
        return doc_ids

    session = get_session()
    query = session.query(ContractRecord)
    if segment is not None:
        query = query.filter_by(segment=segment)
    if status is not None:
        query = query.filter_by(status=status)
    metadata_ids = {r.doc_id for r in query.all()}
    session.close()
    return metadata_ids if doc_ids is None else metadata_ids & doc_ids


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    segment: str | None = None,
    status: str | None = None,
    doc_ids: set[str] | None = None,
    client: OpenAI | None = None,
) -> list[RetrievedChunk]:
    """Dense (Chroma) + BM25 candidates, fused by RRF, reranked by a
    cross-encoder down to the final k. segment/status/doc_ids pre-filter both
    legs identically so fusion never mixes a filtered leg with an unfiltered
    one. doc_ids is for narrowing to a specific document (or set of related
    documents, e.g. a vendor's main agreement + its addenda) known ahead of
    the semantic search -- useful when the corpus has near-duplicate
    boilerplate across documents that a cross-encoder rerank alone can't
    reliably tell apart by content."""
    if client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "before running retrieval."
            )
        client = OpenAI(api_key=settings.openai_api_key)

    where = _build_where(segment, status, doc_ids)
    allowed_doc_ids = _resolve_allowed_doc_ids(segment, status, doc_ids)

    dense_results = dense_search(query, client, top_k=CANDIDATE_POOL_SIZE, where=where)
    bm25_results = _get_bm25_index().search(
        query, top_k=CANDIDATE_POOL_SIZE, allowed_doc_ids=allowed_doc_ids
    )

    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    return rerank(query, fused, top_k=k)


def _citation_label(chunk: RetrievedChunk) -> str:
    section = chunk.section_title or (
        f"Clause {chunk.clause_number}" if chunk.clause_number else "General"
    )
    return f"[{chunk.source_file}, {section}, p.{chunk.page}]"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = [
        f"Passage {i} {_citation_label(chunk)}:\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def answer(
    query: str,
    k: int = DEFAULT_K,
    segment: str | None = None,
    status: str | None = None,
    doc_ids: set[str] | None = None,
    retrieval_query: str | None = None,
) -> AnswerResult:
    """`retrieval_query`, if given, is used for the dense/BM25/rerank search
    instead of `query` -- `query` is still what's shown to the generation
    model and echoed in the prompt. Needed when `doc_ids` already narrows the
    corpus to one vendor's documents and `query` names that same vendor: once
    every candidate chunk is from that vendor, a name mentioned in the query
    text stops being a useful filter and starts being noise that dense/BM25/
    the cross-encoder all still weight heavily, so entity-identification
    boilerplate ("ASIF AND CO, a sole proprietorship of...") can out-rank the
    clause actually being asked about. See agents/rag_agent.py, which passes
    a vendor-name-stripped `retrieval_query` for exactly this case -- a real
    case observed where a termination question against a single vendor's
    documents returned zero termination clauses in the top-k."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running the answer pipeline."
        )
    client = OpenAI(api_key=settings.openai_api_key)
    retrieved = retrieve(
        retrieval_query or query, k=k, segment=segment, status=status, doc_ids=doc_ids, client=client
    )

    if not retrieved:
        return AnswerResult(
            answer="No relevant context was found in the indexed contracts.",
            citations=[],
            contexts=[],
        )

    context_block = _format_context(retrieved)
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context passages:\n\n{context_block}\n\nQuestion: {query}",
            },
        ],
    )
    answer_text = completion.choices[0].message.content or ""

    citations = [
        Citation(
            document=c.source_file,
            section=c.section_title,
            page=c.page,
            clause_number=c.clause_number,
        )
        for c in retrieved
    ]
    return AnswerResult(
        answer=answer_text, citations=citations, contexts=[c.text for c in retrieved]
    )


def make_eval_pipeline(k: int = DEFAULT_K) -> RagPipeline:
    """Adapter satisfying the eval harness's Callable[[str], RagResult]
    interface -- retrieve()/answer() stay unchanged, only this wrapper
    knows about the eval harness's narrower shape."""

    def pipeline(question: str) -> RagResult:
        result = answer(question, k=k)
        return RagResult(answer=result.answer, contexts=result.contexts)

    return pipeline
