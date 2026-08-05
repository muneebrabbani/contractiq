from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from contractiq.extraction.models import ClauseChunk
from contractiq.retrieval.models import RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Bm25Index:
    def __init__(self, chunks: list[ClauseChunk]):
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(
        self, query: str, top_k: int = 20, allowed_doc_ids: set[str] | None = None
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        pairs = list(zip(self.chunks, scores))
        if allowed_doc_ids is not None:
            pairs = [(c, s) for c, s in pairs if c.doc_id in allowed_doc_ids]

        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                clause_number=chunk.clause_number,
                section_title=chunk.section_title,
                page=chunk.page,
                text=chunk.text,
                score=float(score),
            )
            for chunk, score in ranked
            if score > 0
        ]


def build_bm25_index(chunks: list[ClauseChunk]) -> Bm25Index:
    return Bm25Index(chunks)
