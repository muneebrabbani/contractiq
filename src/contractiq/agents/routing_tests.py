from __future__ import annotations

from contractiq.agents.graph import run_supervisor
from contractiq.agents.models import Intent

# (question, expected_intent or None if genuinely ambiguous -- no single
# correct route, so we only observe and print the decision for review)
CASES: list[tuple[str, Intent | None]] = [
    ("What does the termination clause say in the Acme contract?", Intent.NARRATIVE),
    ("How many contracts are currently in the system?", Intent.ANALYTICS),
    ("Tell me about our Network Equipment contracts.", None),
    ("Draft an NDA for a new vendor.", Intent.DRAFTING),
    ("How is the weather in Islamabad today?", Intent.OUT_OF_SCOPE),
    ("List clauses which are common in all contracts.", Intent.NARRATIVE),
]


def run_routing_tests() -> bool:
    all_passed = True
    for question, expected in CASES:
        response = run_supervisor(question)
        actual = response.trace.intent
        if expected is None:
            status = "OBSERVE"
        elif actual == expected:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False

        print(f"[{status}] {question!r}")
        print(f"    routed to: {actual.value} (agent={response.trace.agent})")
        print(f"    reasoning: {response.trace.reasoning}")
        if expected is not None:
            print(f"    expected:  {expected.value}")
        print()

    return all_passed


if __name__ == "__main__":
    passed = run_routing_tests()
    print("ALL PASS" if passed else "SOME FAILED")
