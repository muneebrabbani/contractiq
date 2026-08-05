from __future__ import annotations

import datetime
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel

from contractiq.agents.classifier import classify_intent
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


def _classify_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        decision = classify_intent(state["question"], client)
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
    result = rag_agent(state["question"])
    return {"answer": result.answer, "citations": result.citations, "contexts": result.contexts}


def _sql_node(state: SupervisorState) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    answer_text, sql = sql_agent(state["question"], client)
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
    graph.add_node("classify", _classify_node)
    graph.add_node("rag", _rag_node)
    graph.add_node("sql", _sql_node)
    graph.add_node("drafting", _drafting_node)
    graph.add_node("decline", _decline_node)

    graph.set_entry_point("classify")
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


def run_supervisor(question: str) -> AgentResponse:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running the supervisor."
        )

    graph = _get_graph()
    result = graph.invoke(
        {
            "question": question,
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
        trace=trace,
    )
