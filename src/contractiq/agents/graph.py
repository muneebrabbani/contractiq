from __future__ import annotations

import datetime
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel

from contractiq.agents.classifier import classify_intent
from contractiq.agents.contextualize import contextualize_question
from contractiq.agents.drafting_agent import drafting_agent
from contractiq.agents.models import REVIEW_BANNER, AgentResponse, DraftResult, Intent, RouteTrace
from contractiq.agents.rag_agent import rag_agent
from contractiq.agents.sql_agent import sql_agent
from contractiq.config import settings
from contractiq.retrieval.models import Citation

logger = logging.getLogger(__name__)

_AGENT_BY_INTENT = {
    Intent.NARRATIVE.value: "rag",
    Intent.ANALYTICS.value: "sql",
    Intent.DRAFTING.value: "drafting",
    Intent.OUT_OF_SCOPE.value: "decline",
}

DECLINE_MESSAGE = (
    "I can only help with questions about your contracts -- reading clause content, "
    "running analytics over contract metadata, or drafting new agreements. Could you "
    "rephrase your question along those lines?"
)


class SupervisorState(TypedDict):
    question: str
    history: list[dict]
    contextualized_question: str
    active_doc_ids: set[str] | None  # sticky RAG document scope, carried in from the prior turn
    intent: str
    reasoning: str
    answer: str
    citations: list[Citation]
    contexts: list[str]
    sql: str | None
    draft: DraftResult | None


class _DraftingRequest(BaseModel):
    agreement_type: str
    business_brief: str


def _contextualize_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    contextualized = contextualize_question(state["question"], state["history"], client)
    if contextualized != state["question"]:
        logger.info(
            "contextualized question: raw=%r rewritten=%r", state["question"], contextualized
        )
    return {"contextualized_question": contextualized}


def _classify_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        decision = classify_intent(state["contextualized_question"], client)
        intent, reasoning = decision.intent, decision.reasoning
    except Exception:
        logger.exception("Intent classification failed for question: %r", state["question"])
        intent, reasoning = Intent.NARRATIVE, "Classification failed; defaulted to narrative."

    logger.info(
        "route decision: question=%r intent=%s reasoning=%r",
        state["question"],
        intent.value,
        reasoning,
    )
    return {"intent": intent.value, "reasoning": reasoning}


def _route_selector(state: SupervisorState) -> str:
    return state["intent"]


def _rag_node(state: SupervisorState) -> dict:
    result, doc_ids = rag_agent(
        state["contextualized_question"], fallback_doc_ids=state["active_doc_ids"]
    )
    if doc_ids:
        logger.info("rag document scope: doc_ids=%s", sorted(doc_ids))
    return {
        "answer": result.answer,
        "citations": result.citations,
        "contexts": result.contexts,
        # only overwrite the sticky scope when this turn actually resolved
        # one -- an ambiguous/unfiltered turn shouldn't erase what the
        # session already established.
        "active_doc_ids": doc_ids if doc_ids else state["active_doc_ids"],
    }


def _sql_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    # Uses the contextualized question, not the raw one: classify already
    # routes based on the contextualized text, so a follow-up like "when was
    # the addendum on the main contract" (raw) needs "the main contract"
    # resolved to a specific vendor before it ever reaches SQL generation --
    # otherwise the agent has nothing to filter on and generates no query.
    answer_text, sql = sql_agent(state["contextualized_question"], client)
    return {"answer": answer_text, "sql": sql}


def _extract_drafting_request(question: str, client: OpenAI) -> _DraftingRequest:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the agreement type (e.g. MSA, NDA, SOW) and a short business "
                    'brief from the user\'s drafting request. If no agreement type is stated, '
                    'use "MSA" as a reasonable default. The business brief should be the '
                    "rest of the request, verbatim or lightly trimmed -- do not add details "
                    "not present in the request."
                ),
            },
            {"role": "user", "content": question},
        ],
        response_format=_DraftingRequest,
    )
    return completion.choices[0].message.parsed


def _decline_node(state: SupervisorState) -> dict:
    return {"answer": DECLINE_MESSAGE}


def _drafting_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        request = _extract_drafting_request(state["question"], client)
    except Exception:
        logger.exception("Failed to parse drafting request from question: %r", state["question"])
        return {"answer": "Could not parse a drafting request from the question."}

    draft = drafting_agent(request.agreement_type, request.business_brief)

    summary = (
        f"Draft assembled for {draft.agreement_type}: "
        f"{len(draft.completeness.present)}/{len(draft.completeness.required)} required "
        f"clauses drafted from precedent."
    )
    if draft.completeness.missing:
        summary += f" Missing: {', '.join(c.value for c in draft.completeness.missing)}."
    summary += f" Saved to {draft.docx_path}. {REVIEW_BANNER}."

    return {"answer": summary, "draft": draft}


def build_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("contextualize", _contextualize_node)
    graph.add_node("classify", _classify_node)
    graph.add_node("rag", _rag_node)
    graph.add_node("sql", _sql_node)
    graph.add_node("drafting", _drafting_node)
    graph.add_node("decline", _decline_node)

    graph.set_entry_point("contextualize")
    graph.add_edge("contextualize", "classify")
    graph.add_conditional_edges("classify", _route_selector, _AGENT_BY_INTENT)
    graph.add_edge("rag", END)
    graph.add_edge("sql", END)
    graph.add_edge("drafting", END)
    graph.add_edge("decline", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_supervisor(
    question: str,
    history: list[dict] | None = None,
    active_doc_ids: set[str] | None = None,
) -> AgentResponse:
    """`history` is the prior conversation turns (each `{"role", "content"}`,
    oldest first, NOT including `question` itself) -- pass None or [] for a
    stateless call (e.g. eval/routing-test callers, which must keep scoring
    against the single-turn baseline). Only the last few turns are actually
    used (contextualize.py's HISTORY_WINDOW); the intent classifier, RAG
    retrieval/vendor-hint extraction, and the SQL agent all see the
    history-resolved question, not the raw one -- classify routes off the
    resolved text, so SQL must see the same text it was routed on, or a
    follow-up like "the main contract" reaches SQL generation with nothing
    to filter on. Drafting still sees the raw question.

    `active_doc_ids` is the RAG document scope a prior turn in this session
    already resolved to (see `AgentResponse.active_doc_ids` on the previous
    call's return value) -- pass it back in on the next call so a follow-up
    that doesn't re-name the vendor still stays scoped to it, as a fallback
    behind contextualization rather than instead of it."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running the supervisor."
        )

    graph = _get_graph()
    result = graph.invoke(
        {
            "question": question,
            "history": history or [],
            "contextualized_question": "",
            "active_doc_ids": active_doc_ids,
            "intent": "",
            "reasoning": "",
            "answer": "",
            "citations": [],
            "contexts": [],
            "sql": None,
            "draft": None,
        }
    )

    intent = Intent(result["intent"])
    trace = RouteTrace(
        question=question,
        contextualized_question=result["contextualized_question"],
        intent=intent,
        reasoning=result["reasoning"],
        agent=_AGENT_BY_INTENT[intent.value],
        timestamp=datetime.datetime.utcnow(),
    )

    return AgentResponse(
        answer=result["answer"],
        agent=_AGENT_BY_INTENT[intent.value],
        citations=result.get("citations") or [],
        contexts=result.get("contexts") or [],
        sql=result.get("sql"),
        draft=result.get("draft"),
        active_doc_ids=result.get("active_doc_ids"),
        trace=trace,
    )
