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
#
# A second, later fix: the model over-corrected on "carry forward established
# detail" and started restating a contract's full formal title/description
# whenever one had come up earlier in the conversation -- e.g. rewriting
# "termination clauses for Asif & Co" into "...for Asif & Co for the
# Provisioning of Nationwide Civil Works Services" (that trailing clause is
# the contract's official agreement_title, not anything the user asked for).
# That's harmless for a human reader but actively harmful for retrieval: this
# corpus has ~17 near-identical Civil Works contracts sharing almost that
# exact title, so restating it drags in generic/boilerplate-matching passages
# and can bury the clause actually being asked about (see rag_agent.py's
# vendor-name stripping, which addresses a related but distinct case -- the
# vendor's own *name* repeated in the query -- and doesn't touch a restated
# formal title). The fix here is instructing minimal, not maximal,
# carry-over: resolve the reference with the shortest identifier that keeps
# the question unambiguous (typically just the vendor name), never the full
# formal title or description, even if that title was mentioned earlier.
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
    "When resolving a reference, use the SHORTEST identifier that keeps the "
    "question unambiguous -- almost always just the vendor/company name. "
    "Do NOT restate a contract's full formal title or description (e.g. "
    "\"Agreement for Provisioning of Nationwide Civil Works Services\") even "
    "if it was mentioned earlier in the conversation -- that's bulk the "
    "question doesn't need, not a missing reference to resolve.\n\n"
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
    "contract?\n\n"
    "Example (do not over-elaborate the reference):\n"
    "Conversation so far:\n"
    "user: What is the Asif & Co agreement about?\n"
    "assistant: It's the Agreement for Provisioning of Nationwide Civil "
    "Works Services between TES and Asif & Co.\n"
    "New question: mention the termination clauses for Asif & Co main "
    "contract\n"
    "Standalone question: What are the termination clauses in the Asif & Co "
    "agreement?"
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
