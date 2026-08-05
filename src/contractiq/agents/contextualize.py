from __future__ import annotations

import logging

from openai import OpenAI

from contractiq.config import settings

logger = logging.getLogger(__name__)

# How many prior chat turns (user+assistant messages, not pairs) to feed into
# the rewrite. Kept small deliberately: this system's prompts already spend
# most of their budget on retrieved passages (rag.py), and older turns are
# more likely to drag in a stale vendor/topic than to help resolve the new
# question. 6 (3 exchanges) rather than 4 -- 4 turned out to scroll the
# established vendor/document out of the window after just one follow-up.
HISTORY_WINDOW = 6

# The first version of this prompt told the model not to "add any fact, name,
# or detail not already present in the conversation" -- meant to stop it
# inventing things, but in practice the model read carrying the established
# vendor/document name forward as itself an addition, and left follow-ups
# like "what about price?" or "the contract" unresolved (bare, no vendor),
# which then meant retrieval hit unrelated boilerplate contracts instead of
# declining or erroring loudly. The fix is making entity carry-over an
# explicit requirement, not just "don't hallucinate", backed by worked
# examples of exactly this pattern.
SYSTEM_PROMPT = (
    "Rewrite the new question into a fully self-contained, standalone "
    "question -- one answerable by someone with no access to the "
    "conversation history.\n\n"
    "Resolve every implicit reference to what was actually being discussed: "
    "pronouns (\"it\", \"that\"), short follow-ups (\"what about X?\", \"and "
    "the price?\"), and generic phrases (\"the contract\", \"the vendor\", "
    "\"the agreement\") MUST be replaced with the specific vendor, document, "
    "or entity name established earlier in the conversation. Carrying "
    "forward a name or detail that was already stated earlier is REQUIRED, "
    "not a forbidden addition -- only genuinely new facts that were never "
    "mentioned anywhere in the conversation are off-limits.\n\n"
    "Do not answer the question. If the new question already names its own "
    "subject explicitly, return it unchanged.\n\n"
    "Example:\n"
    "Conversation so far:\n"
    "user: What does the termination clause say in the Asif & Co agreement?\n"
    "assistant: The Asif & Co agreement requires 30 days' notice to terminate.\n"
    "New question: What about the price?\n"
    "Standalone question: What is the price in the Asif & Co agreement?\n\n"
    "Example:\n"
    "Conversation so far:\n"
    "user: List termination clauses for Nextcom Links.\n"
    "assistant: [clause text for Nextcom Links]\n"
    "New question: Is there any price mentioned in the contract?\n"
    "Standalone question: Is there any price mentioned in the Nextcom Links "
    "contract?"
)


def contextualize_question(question: str, history: list[dict], client: OpenAI) -> str:
    """Rewrites a follow-up question into a standalone one using the last few
    turns of chat history. Returns `question` unchanged if there's no history
    to draw on, or if the rewrite call fails."""
    if not history:
        return question

    trimmed = history[-HISTORY_WINDOW:]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in trimmed)

    try:
        completion = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,  # rewriting is resolution, not creative writing -- the
            # same follow-up against the same history should resolve to the same
            # standalone question every time, not vary with sampling noise.
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{transcript}\n\n"
                        f"New question: {question}\n\n"
                        "Standalone question:"
                    ),
                },
            ],
        )
        rewritten = (completion.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception:
        logger.exception("Question contextualization failed for question: %r", question)
        return question
