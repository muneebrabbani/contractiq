from contractiq.agents.drafting_agent import drafting_agent
from contractiq.agents.graph import run_supervisor
from contractiq.agents.models import (
    REVIEW_BANNER,
    AgentResponse,
    CompletenessReport,
    DraftClause,
    DraftResult,
    Intent,
    RouteDecision,
    RouteTrace,
)
from contractiq.agents.routing_tests import run_routing_tests
from contractiq.eval.pipeline import RagPipeline, RagResult


def make_eval_pipeline() -> RagPipeline:
    """Adapter satisfying the eval harness's Callable[[str], RagResult] --
    routes every question through the supervisor. Eval-set questions are
    all narrative, so they should classify to the rag agent and hit the
    same retrieval.answer() call the retrieval-only eval used."""

    def pipeline(question: str) -> RagResult:
        response = run_supervisor(question)
        return RagResult(answer=response.answer, contexts=response.contexts)

    return pipeline


__all__ = [
    "Intent",
    "RouteDecision",
    "RouteTrace",
    "AgentResponse",
    "DraftClause",
    "CompletenessReport",
    "DraftResult",
    "REVIEW_BANNER",
    "run_supervisor",
    "make_eval_pipeline",
    "run_routing_tests",
    "drafting_agent",
]
