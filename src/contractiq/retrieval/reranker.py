from __future__ import annotations

from sentence_transformers import CrossEncoder

from contractiq.retrieval.models import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    """Cross-encoder jointly scores (query, chunk) pairs -- much more
    accurate than comparing independently-encoded embeddings, but too
    expensive to run over the whole corpus, hence only applied to the
    already-fused shortlist."""
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)[:top_k]
    return [c.model_copy(update={"score": float(score)}) for c, score in ranked]
