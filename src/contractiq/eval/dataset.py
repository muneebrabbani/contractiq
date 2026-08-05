from __future__ import annotations

from pathlib import Path

from contractiq.eval.models import EvalExample

EVAL_SET_PATH = Path("eval_data/eval_set.jsonl")

SEED_EXAMPLES: list[EvalExample] = [
    EvalExample(
        id="eval-001",
        question="What is the notice period required to terminate the agreement?",
        gold_answer="Either party may terminate the agreement upon 30 days written notice.",
        source_document="MSA_Acme.docx",
        source_section="Clause 1.1",
        category="obligation",
    ),
    EvalExample(
        id="eval-002",
        question="Which state's law governs this contract?",
        gold_answer="The agreement is governed by the laws of the State of New York.",
        source_document="MSA_Acme.docx",
        source_section="Section 2 - Governing Law",
        category="clause-lookup",
    ),
    EvalExample(
        id="eval-003",
        question="What is the total contract value and how is it paid?",
        gold_answer="The total contract value is $1,250,000.00, payable Net 45.",
        source_document="MSA_Acme.docx",
        source_section="Section 3 - Payment Terms",
        category="value",
    ),
    EvalExample(
        id="eval-004",
        question="Who should notices be addressed to at the Vendor?",
        gold_answer="Notices to the Vendor should be sent Attn: General Counsel.",
        source_document="MSA_Acme.docx",
        source_section="Section 4 - Notices",
        category="clause-lookup",
        notes="Gold answer intentionally omits the redacted street address.",
    ),
    EvalExample(
        id="eval-005",
        question="What is the late delivery penalty?",
        gold_answer="The penalty for late delivery is 2% per week, capped at 10%.",
        source_document="MSA_Acme.docx",
        source_section="Penalty schedule (table)",
        category="value",
        notes="Sourced from a table, not running prose -- checks table-derived clauses too.",
    ),
]


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalExample]:
    with path.open("r", encoding="utf-8") as f:
        return [EvalExample.model_validate_json(line) for line in f if line.strip()]


def write_eval_set(examples: list[EvalExample], path: Path = EVAL_SET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")


def write_seed_template(path: Path = EVAL_SET_PATH) -> None:
    write_eval_set(SEED_EXAMPLES, path)


if __name__ == "__main__":
    write_seed_template()
