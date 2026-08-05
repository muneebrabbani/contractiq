from __future__ import annotations

from contractiq.retrieval.models import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists by reciprocal rank position, not raw score
    -- dense cosine similarity and BM25 scores live on incomparable scales,
    so only where a chunk lands in each list's ranking is used, not how
    confident either retriever was."""
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    fused = sorted(chunk_by_id.values(), key=lambda c: rrf_scores[c.chunk_id], reverse=True)
    return [c.model_copy(update={"score": rrf_scores[c.chunk_id]}) for c in fused]
