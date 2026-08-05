from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from contractiq.extraction.clause_chunking import load_all_clause_chunks

PROCESSED_DIR = Path("data/processed")


class RagResult(BaseModel):
    answer: str
    contexts: list[str]


RagPipeline = Callable[[str], RagResult]


def _keyword_score(question: str, text: str) -> int:
    question_words = set(re.findall(r"\w+", question.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    return len(question_words & text_words)


def make_stub_pipeline(input_dir: Path = PROCESSED_DIR) -> RagPipeline:
    """Trivial retrieval-plus-answer pipeline: naive keyword-overlap search
    over the already-chunked, already-redacted corpus, no embeddings and no
    LLM generation call, so it runs instantly and free of charge. Exists
    purely to prove the eval harness works end-to-end before a real
    retriever exists -- expect mediocre scores, that's not the point.
    """
    corpus = load_all_clause_chunks(input_dir)

    def stub_pipeline(question: str, top_k: int = 3) -> RagResult:
        if not corpus:
            return RagResult(answer="No relevant context found.", contexts=[])

        ranked = sorted(corpus, key=lambda c: _keyword_score(question, c.text), reverse=True)
        top = ranked[:top_k]
        contexts = [c.text for c in top]
        answer = top[0].text if top else "No relevant context found."
        return RagResult(answer=answer, contexts=contexts)

    return stub_pipeline
