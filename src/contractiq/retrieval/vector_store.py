from __future__ import annotations

import threading
from pathlib import Path

import chromadb
from openai import OpenAI

from contractiq.extraction.models import ClauseChunk
from contractiq.retrieval.embeddings import embed_texts
from contractiq.retrieval.models import RetrievedChunk

CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "contractiq_chunks"

_lock = threading.Lock()
_collections: dict[str, object] = {}


def get_collection(persist_dir: Path = CHROMA_DIR):
    """Cached per persist_dir. chromadb.PersistentClient's internal system
    registry isn't safe against concurrent construction for the same path
    (hit via the eval harness's ThreadPoolExecutor calling retrieve()/
    answer() concurrently) -- constructing it once under a lock avoids that
    race, and reusing the client is also just cheaper than reopening the
    persistent DB on every query."""
    key = str(persist_dir)
    if key not in _collections:
        with _lock:
            if key not in _collections:
                client = chromadb.PersistentClient(path=key)
                _collections[key] = client.get_or_create_collection(
                    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
    return _collections[key]


def _clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


def index_chunks(
    chunks_with_metadata: list[tuple[ClauseChunk, dict]],
    client: OpenAI,
    persist_dir: Path = CHROMA_DIR,
) -> int:
    """Embed and upsert chunks into the persistent Chroma collection.
    chunks_with_metadata pairs each chunk with extra filterable metadata
    (e.g. segment/status) that indexing.py resolves by joining on doc_id."""
    if not chunks_with_metadata:
        return 0

    collection = get_collection(persist_dir)
    texts = [chunk.text for chunk, _ in chunks_with_metadata]
    embeddings = embed_texts(texts, client)

    ids = [chunk.chunk_id for chunk, _ in chunks_with_metadata]
    metadatas = [_clean_metadata(meta) for _, meta in chunks_with_metadata]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(ids)


def dense_search(
    query: str,
    client: OpenAI,
    top_k: int = 20,
    where: dict | None = None,
    persist_dir: Path = CHROMA_DIR,
) -> list[RetrievedChunk]:
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query], client)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for chunk_id, text, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                source_file=meta.get("source_file", ""),
                clause_number=meta.get("clause_number"),
                section_title=meta.get("section_title"),
                page=meta.get("page", 0),
                text=text,
                score=1 - distance,  # cosine distance -> similarity, higher is better
            )
        )
    return chunks
