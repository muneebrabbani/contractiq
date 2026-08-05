from __future__ import annotations

from openai import OpenAI

from contractiq.config import settings

BATCH_SIZE = 100


def embed_texts(texts: list[str], client: OpenAI, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []

    model = model or settings.retrieval_embedding_model
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
