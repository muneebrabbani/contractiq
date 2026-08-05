from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from contractiq.config import settings
from contractiq.extraction.db import DB_PATH, ContractRecord

# Secondary, defense-in-depth check only -- the real guarantee is the
# read-only SQLite connection in _execute_readonly(). This regex is a crude
# early-rejection for a clearer error message, not a real SQL parser (it can
# false-positive on a string literal containing one of these words).
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|"
    r"REINDEX|TRIGGER)\b",
    re.IGNORECASE,
)

MAX_ROWS_SHOWN = 20


class SqlQuery(BaseModel):
    sql: str
    explanation: str


def _table_schema_description() -> str:
    lines = [f"  - {col.name} ({col.type})" for col in ContractRecord.__table__.columns]
    return "Table: contracts\nColumns:\n" + "\n".join(lines)


def _system_prompt() -> str:
    return (
        "You translate natural-language analytics questions into a single read-only "
        "SQLite SELECT query against the following table.\n\n"
        f"{_table_schema_description()}\n\n"
        "Notes on specific columns:\n"
        "- related_doc_id: if set, this row is an amendment/addendum/renewal of the "
        "contract whose doc_id equals this value. A contract is currently superseded "
        "(no longer the active/current version) if its doc_id appears as some other "
        "row's related_doc_id -- e.g. 'which contracts are current' means "
        "`doc_id NOT IN (SELECT related_doc_id FROM contracts WHERE related_doc_id IS "
        "NOT NULL)`, and 'which contracts have been renewed/amended' means "
        "`related_doc_id IS NOT NULL` on the amending row, or the inverse membership "
        "test on the original row.\n\n"
        "Rules:\n"
        "- Generate exactly one SELECT statement. Never INSERT, UPDATE, DELETE, DROP, "
        "ALTER, CREATE, or any other write/DDL statement.\n"
        "- Never chain multiple statements with semicolons.\n"
        "- Use only the columns listed above.\n"
        "- If the question cannot be answered from this table, explain why in "
        "`explanation` and leave `sql` as an empty string rather than guessing."
    )


def _validate_select_only(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if not stripped:
        raise ValueError("No query was generated.")
    if ";" in stripped:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT queries are supported.")
    if _FORBIDDEN_RE.search(stripped):
        raise ValueError("Query contains a disallowed keyword.")


def _generate_sql(question: str, client: OpenAI) -> SqlQuery:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": question},
        ],
        response_format=SqlQuery,
    )
    return completion.choices[0].message.parsed


def _execute_readonly(sql: str, db_path: Path) -> tuple[list[str], list[tuple]]:
    """Connection opened in SQLite's URI read-only mode -- this is the real
    safety guarantee: the connection is structurally incapable of writing,
    independent of what SQL text reaches it."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def _escape_cell(value: object) -> str:
    """Neutralize characters that would break Markdown table syntax."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_result(columns: list[str], rows: list[tuple]) -> str:
    """Markdown table formatting, deliberately not a second LLM call -- the
    computed numbers must reach the user exactly as SQL returned them."""
    if not rows:
        return "No matching records were found."
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}"

    header = "| " + " | ".join(_escape_cell(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    lines.extend(
        "| " + " | ".join(_escape_cell(v) for v in row) + " |" for row in rows[:MAX_ROWS_SHOWN]
    )
    if len(rows) > MAX_ROWS_SHOWN:
        lines.append(f"*... and {len(rows) - MAX_ROWS_SHOWN} more rows*")
    return "\n".join(lines)


def sql_agent(question: str, client: OpenAI, db_path: Path = DB_PATH) -> tuple[str, str]:
    """Returns (answer_text, sql_used). The LLM only ever produces the query
    string; execution and result formatting are deterministic."""
    query = _generate_sql(question, client)

    try:
        _validate_select_only(query.sql)
    except ValueError as exc:
        return f"Could not answer safely: {exc}", query.sql

    try:
        columns, rows = _execute_readonly(query.sql, db_path)
    except sqlite3.Error as exc:
        return f"The generated query failed to execute: {exc}", query.sql

    return _format_result(columns, rows), query.sql
