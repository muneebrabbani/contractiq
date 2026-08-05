from __future__ import annotations

from openai import OpenAI

from contractiq.agents.models import RouteDecision
from contractiq.config import settings

SYSTEM_PROMPT = (
    "You classify questions into exactly one of four intents. This assistant only handles "
    "questions about a telecom contract corpus -- reading contract text (in one document or "
    "across many), running analytics over contract metadata, or drafting new agreements.\n\n"
    "narrative: the question needs contract TEXT read and interpreted -- clause content, "
    "terms, obligations, definitions. This includes questions about a single contract AND "
    "questions that span or aggregate across the whole corpus's text, as long as answering "
    "means reading/comparing clause content rather than querying structured fields.\n"
    'Examples: "What does the termination clause say?", "Summarize the payment terms in '
    'the Acme contract.", "Who are the signatories on the MSA?", "What clauses are common '
    'across all our contracts?", "Which contracts include an indemnification clause?", '
    '"How do our contracts typically handle renewal?"\n\n'
    "analytics: the question needs COUNTING, SUMMING, or FILTERING over the structured "
    "contract metadata table (vendor, segment, status, value, dates) -- not reading any "
    "contract's clause text.\n"
    'Examples: "How many contracts are active in the Network Equipment segment?", '
    '"What\'s the total value of contracts expiring this quarter?", "List vendors with '
    'contracts over $1M."\n\n'
    "drafting: the question asks to CREATE a new contract or clause, not answer about "
    "existing ones.\n"
    'Examples: "Draft an NDA for a new vendor.", "Create a service agreement for X."\n\n'
    "out_of_scope: the question has NO connection to the contract corpus at all -- small "
    "talk, general knowledge, weather, math, jokes, or anything unrelated to contracts. "
    "A question that mentions contracts, clauses, vendors, or 'all our agreements' -- even "
    "if it spans the whole corpus, is oddly phrased, or doesn't neatly match a narrative/"
    "analytics example above -- is NEVER out_of_scope: pick whichever of narrative/analytics/"
    "drafting fits best instead. Reserve out_of_scope strictly for questions that aren't "
    "about this assistant's subject matter in any way.\n"
    'Examples: "How is the weather today?", "What\'s 2+2?", "Tell me a joke."\n\n'
    "Pick the single best-fitting intent and explain your reasoning briefly so the choice "
    "can be audited later."
)


def classify_intent(question: str, client: OpenAI) -> RouteDecision:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format=RouteDecision,
    )
    return completion.choices[0].message.parsed
