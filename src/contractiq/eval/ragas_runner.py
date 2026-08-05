from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from ragas.embeddings.openai_provider import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, Faithfulness

from contractiq.config import settings
from contractiq.eval.models import EvalExample, EvalReport, EvalResult
from contractiq.eval.pipeline import RagPipeline

FAIL_THRESHOLD = 0.5


def _call_pipeline(example: EvalExample, rag_pipeline: RagPipeline) -> dict:
    result = rag_pipeline(example.question)
    return {
        "user_input": example.question,
        "response": result.answer,
        "retrieved_contexts": result.contexts,
        "reference": example.gold_answer,
    }


def _build_rows(
    examples: list[EvalExample], rag_pipeline: RagPipeline, max_workers: int
) -> list[dict]:
    if max_workers <= 1:
        return [_call_pipeline(e, rag_pipeline) for e in examples]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda e: _call_pipeline(e, rag_pipeline), examples))


def run_eval(
    examples: list[EvalExample], rag_pipeline: RagPipeline, max_workers: int = 4
) -> EvalReport:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running the eval harness (RAGAS uses it as the judge model)."
        )

    # ragas.metrics.collections requires an async client -- llm.agenerate()
    # raises TypeError against a sync OpenAI client.
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.chat_model, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=settings.embedding_model)

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecisionWithReference(llm=llm)

    rows = _build_rows(examples, rag_pipeline, max_workers)

    faithfulness_scores = faithfulness.batch_score(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in rows
        ]
    )
    relevancy_scores = answer_relevancy.batch_score(
        [{"user_input": r["user_input"], "response": r["response"]} for r in rows]
    )
    precision_scores = context_precision.batch_score(
        [
            {
                "user_input": r["user_input"],
                "reference": r["reference"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in rows
        ]
    )

    results = [
        EvalResult(
            id=example.id,
            question=example.question,
            gold_answer=example.gold_answer,
            generated_answer=row["response"],
            contexts=row["retrieved_contexts"],
            faithfulness=f.value,
            answer_relevancy=r.value,
            context_precision=c.value,
        )
        for example, row, f, r, c in zip(
            examples, rows, faithfulness_scores, relevancy_scores, precision_scores
        )
    ]

    n = len(results)
    return EvalReport(
        results=results,
        mean_faithfulness=sum(r.faithfulness for r in results) / n,
        mean_answer_relevancy=sum(r.answer_relevancy for r in results) / n,
        mean_context_precision=sum(r.context_precision for r in results) / n,
    )


def print_report(report: EvalReport, fail_threshold: float = FAIL_THRESHOLD) -> None:
    print("=" * 80)
    print(f"EVAL SUMMARY  ({len(report.results)} questions)")
    print("-" * 80)
    print(f"  mean faithfulness:      {report.mean_faithfulness:.3f}")
    print(f"  mean answer_relevancy:  {report.mean_answer_relevancy:.3f}")
    print(f"  mean context_precision: {report.mean_context_precision:.3f}")
    print()
    print("PER-QUESTION BREAKDOWN (worst first)")
    print("-" * 80)

    for r in sorted(report.results, key=lambda r: r.min_score):
        flag = " <-- FAIL" if r.min_score < fail_threshold else ""
        print(
            f"[{r.id}] faith={r.faithfulness:.2f} relev={r.answer_relevancy:.2f} "
            f"ctx_prec={r.context_precision:.2f}{flag}"
        )
        print(f"    Q: {r.question}")
        print(f"    gold:      {r.gold_answer}")
        print(f"    generated: {r.generated_answer}")
        print()


if __name__ == "__main__":
    from contractiq.eval.dataset import load_eval_set
    from contractiq.eval.pipeline import make_stub_pipeline

    report = run_eval(load_eval_set(), make_stub_pipeline())
    print_report(report)
