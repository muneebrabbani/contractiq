# ContractIQ

A retrieval-augmented contract intelligence assistant for a telecom procurement team.

## Status

Working system, not a skeleton: ingestion, redaction, metadata extraction,
hybrid retrieval, a LangGraph agent layer (narrative / analytics / drafting /
decline), a Streamlit UI (Chat, Alerts, Draft, Upload), and OpenShift
deployment manifests are all built and wired together. For the "why" behind
the design decisions and a running troubleshooting log, see `HANDOVER.md`;
for per-component technical rationale (useful when packaging/deploying), see
`ARCHITECTURE.md`.

## Setup

```
pip install -e .
cp .env.example .env  # then fill in OPENAI_API_KEY
```

`pyproject.toml` is the source of truth for dependencies — `requirements.txt`
is stale (missing `streamlit` and others); don't install from it.

Ingestion's OCR fallback (for scanned PDFs) needs the Tesseract OCR engine
installed separately and on PATH — it is not a pip package. See
https://github.com/tesseract-ocr/tesseract for install instructions.

Extraction's address redaction needs spaCy's English model, downloaded
separately after `pip install`:

```
python -m spacy download en_core_web_sm
```

Then, to run the app locally:

```
streamlit run src/contractiq/ui/app.py --server.port 8501
```

## Ingestion

Supports PDF (native text, with automatic OCR fallback via Tesseract for
scanned pages) and DOCX. Documents are split into fixed-size token chunks
(500 tokens, 50 overlap, `cl100k_base` encoding) and written as JSONL to
`data/processed/`. This path is independent of the clause-aware chunking used
by retrieval (below) — nothing downstream reads its output, so it's safe to
omit from a minimal deployment.

```python
from contractiq.ingestion import ingest_directory

ingest_directory()  # data/raw/ -> data/processed/*.jsonl
```

## Extraction

Two passes over `data/raw/`, run before any chunking/retrieval:

- **Recon** (`contractiq.extraction.run_recon`): scans every PDF/DOCX and
  reports, per file, native-vs-scanned classification (per-page character
  count against the same OCR threshold ingestion uses), page count, char
  count, and a table-presence flag (PyMuPDF `find_tables()` / python-docx
  `doc.tables` — not a heuristic). Writes `data/processed/recon_report.csv`
  and a console summary with overall native/scanned/mixed counts.

- **Redaction** (`contractiq.extraction.redact_directory`): strips phone
  numbers, emails, IP addresses (regex, IP validated via stdlib
  `ipaddress`), and physical addresses (regex-anchored on street-number +
  street-type structure, extended through trailing city/state/zip via spaCy
  GPE/LOC only — never triggered by a bare place name, and never extended
  across a party name) from every file, native or scanned. Scanned pages
  (~85% of this corpus by page count) are OCR'd via Tesseract at 300 DPI
  first — the same fallback ingestion uses — and the same redaction regex/NER
  then runs over the OCR'd text; OCR noise means regex-based redaction is
  marginally less reliable on scanned pages than on clean native text, a
  known, stated trade-off rather than an oversight. Vendor/party/signatory
  names and all commercial terms (values, dates, payment terms, penalties)
  are never touched. Writes cleaned, page-structured JSON per document plus
  a single `data/processed/redaction_log.csv` audit trail (file, category,
  original snippet, replacement, page).

```python
from contractiq.extraction import run_recon, redact_directory

run_recon()          # data/raw/ -> data/processed/recon_report.csv
redact_directory()   # data/raw/ -> data/processed/*.redacted.json + redaction_log.csv
```

Two more stages build on the redacted output in `data/processed/*.redacted.json`:

- **Clause-aware chunking** (`contractiq.extraction.chunk_document_by_clause`):
  splits on detected section/clause numbering (`4.1`, `Section 4`, `ARTICLE
  IV`) instead of fixed token windows, keeping each clause intact as one
  chunk. Headings are folded into `section_title` metadata rather than
  emitted as their own near-empty chunks; a clause that runs unusually long
  falls back to fixed-token-window splitting (reusing the ingestion
  chunker's splitter) so no chunk blows past ~800 tokens. Each chunk carries
  `doc_id`, `source_file`, `clause_number`, `section_title`, `page`.

- **Metadata extraction** (`contractiq.extraction.extract_all`): calls the
  OpenAI API with Structured Outputs (strict JSON schema, not tool-calling)
  against the `ContractMetadata` schema (vendor, contract type, segment,
  status, effective/expiry date, value, currency, internal `department`
  responsible, signatory names, payment terms, notice period, and an
  optional `related_doc_id` link — never LLM-extracted, set only via the
  Upload UI page when a document is marked as a renewal/addendum of an
  existing contract), one record per contract, persisted via SQLAlchemy to
  `data/processed/contractiq.sqlite3`. Re-running upserts by `doc_id` rather
  than duplicating rows. Requires `OPENAI_API_KEY` in `.env`.
  `segment`/`status` feed retrieval's metadata pre-filtering below — a
  document indexed before metadata extraction has run just won't match
  segment/status filters yet.

- **Spot-check** (`contractiq.extraction.print_spot_check`): prints every
  extracted record's fields next to the source document's head and tail
  text (title/parties usually front-load; signatories/notice terms usually
  sit in the signature block) for manual review.

```python
from contractiq.extraction import chunk_document_by_clause, extract_all, print_spot_check
from contractiq.extraction.redaction import load_redacted_document

document = load_redacted_document(Path("data/processed/<doc_id>.redacted.json"))
chunks = chunk_document_by_clause(document)

extract_all()        # data/processed/*.redacted.json -> contractiq.sqlite3
print_spot_check()   # console review of extracted fields vs. source text
```

Adding a single new document (e.g. a renewal) without re-running the whole
corpus pipeline: use the Upload page (see UI, below), which runs this same
redact → chunk → extract → index sequence for one file in-process.

## Eval

A RAGAS-based harness, built before retrieval exists so retrieval quality
can be measured from day one.

- **Eval set** (`eval_data/eval_set.jsonl`, tracked in git — unlike
  `data/`, this is curated test-suite content, not disposable derived
  data): one JSON object per line — `id`, `question`, `gold_answer`,
  `source_document`, `source_section`, plus optional `category`/`notes`.
  Seeded with 5 examples; hand-label toward ~80 using the same format.
  Note: since gold answers echo real contract content, reconsider tracking
  it in git if that's a concern for your corpus.

- **Retriever interface** (`contractiq.eval.RagPipeline`): any function
  `(question: str) -> RagResult` (`answer: str`, `contexts: list[str]`) —
  no base class, so the real retrieval-plus-generation pipeline drops in
  unchanged when it exists. `make_stub_pipeline()` is a trivial
  placeholder: naive keyword-overlap search over the already-chunked
  redacted corpus, no embeddings or LLM call, so it runs free and
  instantly. Expect mediocre scores from it — it exists to prove the
  harness works end-to-end, not to retrieve well.

- **Runner** (`contractiq.eval.run_eval`): scores every example on
  faithfulness, answer relevancy, and context precision (the
  reference-based variant, using `gold_answer`) via RAGAS, judged by
  OpenAI (`settings.chat_model` for the judge, `settings.embedding_model`
  for answer-relevancy's similarity check). `print_report` prints mean
  scores followed by a per-question breakdown sorted worst-first, with a
  threshold flag, so failures are the first thing you see. Requires
  `OPENAI_API_KEY` in `.env`. Run eval questions through the agent-routed
  pipeline (`make_eval_pipeline`, below), not `retrieval.answer` directly —
  and exclude analytics questions from RAGAS scoring, since the SQL agent
  produces no `retrieved_contexts` for RAGAS's context-grounding metrics to
  score in the first place. RAGAS scores also aren't perfectly
  deterministic run-to-run (the judge is itself an LLM call) — a ~10-point
  swing on one metric between identical runs is plausible noise, not
  necessarily a regression.

```python
from contractiq.eval import load_eval_set, make_stub_pipeline, run_eval, print_report

report = run_eval(load_eval_set(), make_stub_pipeline())
print_report(report)
```

## Retrieval

Hybrid retrieval over clause chunks, built to match the eval harness's
interface so it can be scored the same way the stub was.

```
query ─┬─► dense (Chroma, text-embedding-3-large)  ─┐
        └─► BM25 (rank_bm25, in-memory)             ─┼─► RRF fusion ─► cross-encoder rerank ─► top-k
```

- **Indexing** (`contractiq.retrieval.build_index`): embeds every clause
  chunk (`text-embedding-3-large`, batched) into a persistent ChromaDB
  collection at `data/processed/chroma/`. Each chunk's metadata (for
  pre-filtering) is joined from the metadata-extraction stage's SQLite
  table by `doc_id` — `segment`, `status`, plus `clause_number`,
  `section_title`, `page` from the chunker, and `clause_type` from the
  classifier below. Clause-type classification (the slow part) runs in a
  small thread pool over just the LLM-fallback subset — about 7.5x faster
  than classifying one chunk at a time serially on a full corpus re-index.

- **Dense + BM25 + fusion**: both legs pull a ~20-candidate pool per query.
  Fusion is Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)` across
  whichever lists a chunk appears in) rather than a weighted score blend —
  dense cosine similarity and BM25 scores aren't on comparable scales, so
  only rank position is used, not raw confidence.

- **Reranking** (`contractiq.retrieval.reranker`): a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers` —
  pulls in `torch`) rescores the fused shortlist by jointly encoding
  (query, passage) pairs, then the top-k survive. Loaded once, not per call.

- **`retrieve(query, k, segment=None, status=None, doc_ids=None)`**: the
  building block above, returning `RetrievedChunk`s (text + score +
  citation metadata). `segment`/`status`/`doc_ids` all filter both the
  dense leg (Chroma `where`) and the BM25 leg (doc_id allowlist)
  identically, so fusion never mixes a filtered leg with an unfiltered one.
  `doc_ids` is how the RAG agent's vendor pre-filtering (below) narrows a
  search to one vendor's documents.

- **`answer(query, k, segment=None, status=None, doc_ids=None)`**: calls
  `retrieve()`, then asks the OpenAI chat model to answer strictly from the
  retrieved passages, each labeled `[document, section, page]` in the
  prompt; the system prompt requires every claim to cite one of those
  labels and to say so explicitly rather than guess when context is
  insufficient. Returns `AnswerResult` (`answer`, `citations`, `contexts`).

- **`make_eval_pipeline()`**: adapts `answer()` to the eval harness's
  `Callable[[str], RagResult]` shape — `retrieve`/`answer` stay unchanged,
  only this thin wrapper knows about the eval harness's narrower interface.

```python
from contractiq.retrieval import build_index, retrieve, answer, make_eval_pipeline
from contractiq.eval import load_eval_set, run_eval, print_report

build_index()  # data/processed/*.redacted.json -> data/processed/chroma/

chunks = retrieve("What is the notice period?", k=5)
result = answer("What is the notice period?")

report = run_eval(load_eval_set(), make_eval_pipeline())
print_report(report)
```

## Agents

A LangGraph supervisor rewrites conversational follow-ups into standalone
questions, classifies each one, and routes it to one of three content agents
or a fourth, non-content decline route. Does not modify anything under
`retrieval/` — the RAG agent is a thin wrapper around `retrieval.answer`.

```
START ──► contextualize (LLM rewrite of follow-ups using recent chat
              │            history — see "Conversational memory" below;
              │            no-op if there's no history)
              ▼
           classify (OpenAI structured-output intent + logged RouteDecision)
              │
              ├─ narrative    ──► rag agent      (wraps retrieval.answer + vendor pre-filter) ──► END
              ├─ analytics    ──► sql agent      (generate → validate → execute → format)      ──► END
              ├─ drafting     ──► drafting agent (precedent retrieval → assembly → DOCX)        ──► END
              └─ out_of_scope ──► decline node   (fixed refusal message, no LLM/DB call)        ──► END
```

- **Classification**: OpenAI Structured Outputs against a small
  `{intent, reasoning}` schema — always commits to exactly one of
  `narrative` / `analytics` / `drafting` / `out_of_scope`, no "unsure"
  branch. `out_of_scope` is a deliberately narrow guardrail, biased hard
  against over-using it — anything contract-adjacent, even oddly-phrased or
  corpus-wide questions, still routes to one of the three content agents;
  it's reserved for genuinely unrelated small talk/general knowledge.
  Falls back to `narrative` if the classification call itself fails. Every
  routing decision is logged (`contractiq.agents.graph` logger: question,
  contextualized question, intent, reasoning) and returned as a structured
  `RouteTrace` on the response, so routing failures are debuggable from
  either logs or the response object directly.

- **Vendor resolution** (`contractiq.agents.vendor_resolution`): shared by
  the RAG and drafting agents. Extracts a vendor/company name explicitly
  named in a question or drafting brief (LLM call), then resolves it to that
  vendor's `doc_id`s in SQLite for use as a hard retrieval pre-filter — this
  corpus has ~17 near-identical boilerplate contracts (same template,
  different vendor), and a cross-encoder rerank or single-best-match dense
  search genuinely can't tell them apart by text content alone once a
  question names one specifically. Vendor names are normalized before
  matching (`&`/standalone `"n"`/punctuation → a common form) so spelling
  variants for the same real vendor across DB records and filenames (e.g.
  "Asif & Co" / "ASIF AND CO" / "Asif n Co") resolve as one.

- **Text-to-SQL agent** (`contractiq.agents.sql_agent`): three independent
  safety layers, not one.
  1. *Connection-level (the real guarantee)*: the SQLite connection is
     opened in URI read-only mode (`file:...?mode=ro`) — structurally
     incapable of writing regardless of what SQL text reaches it. Verified
     directly: `DELETE FROM contracts` against this connection raises
     `sqlite3.OperationalError: attempt to write a readonly database`.
  2. *Statement-shape validation before execution*: rejects anything that
     isn't a single statement starting with `SELECT` (no chained
     statements, no write/DDL keywords) — a clearer error message than
     letting SQLite's own rejection surface, and an earlier one.
  3. *Schema-scoped generation*: the model only ever sees the `contracts`
     table's schema, introspected from the `ContractRecord` SQLAlchemy
     model so it can't drift out of sync with the real schema.

  The LLM's role stops at generating the SQL string — execution is real
  SQLite, and turning the result into text is **plain Python string
  formatting, not a second LLM call**, so there's no path for the answer
  to drift from the number SQL actually returned. The SQL agent sees the
  same contextualized question the classifier routed on, not the raw one —
  a follow-up like "when was the addendum on the main contract" needs "the
  main contract" resolved to a specific vendor before it reaches SQL
  generation, or there's nothing to filter on.

- **Clause-type classifier** (`contractiq.extraction.classify_clause_type`
  / `classify_clause_types_batch`): a controlled `ClauseType` vocabulary
  (termination, governing_law, payment_terms, confidentiality,
  indemnification, scope_of_work, ...) that `section_title` alone doesn't
  provide — two contracts can title the same clause type differently.
  Keyword-matches `section_title` first (free, deterministic, covers the
  common case since real contracts mostly use conventional headings); falls
  back to an LLM classification (`temperature=0`) of the clause text only
  when the title is missing or matches nothing. `retrieval/indexing.py`
  joins this into each chunk's Chroma metadata at index time, alongside the
  existing `segment`/`status` join — nothing else under `retrieval/`
  changed.

- **Drafting agent** (`contractiq.agents.drafting_agent`) — the highest-risk
  feature, built conservatively: it assembles a draft from retrieved
  precedent clauses rather than generating clause text freely.
  1. *Precedent retrieval*: for each clause type in the agreement type's
     checklist (MSA/NDA/SOW, each now including a `scope_of_work` clause),
     calls `retrieval.vector_store.dense_search()` directly (a stage-4
     function, reused as-is) with a hard `clause_type` metadata filter plus
     semantic ranking against the business brief. This is deliberately
     *not* routed through the full `retrieve()` hybrid pipeline — precedent
     lookup is "find the best exemplar of a known type," a different
     problem from open-ended retrieval, and a hard type filter already does
     the precision work. If the business brief names a specific vendor,
     search is scoped to that vendor's own documents first (via vendor
     resolution, above), falling back to the best corpus-wide match only
     for clause types that vendor's contract doesn't have — and the
     generated DOCX labels that fallback explicitly rather than silently
     mixing a different vendor's boilerplate in.
  2. *Assembly*: clause selection is deterministic (top match within the
     type filter). Party names are placeholder-substituted deterministically
     (`ContractMetadata.vendor` → `[VENDOR NAME]`) — no LLM involved. The
     **only** LLM step is per-clause language smoothing, under a prompt
     that forbids changing any number, date, or substantive term, backed
     by a post-hoc check: numeric tokens (regex, normalized for cosmetic
     formatting differences like leading zeros or comma-grouping so OCR
     noise doesn't force an unnecessary revert) are extracted from the text
     before and after smoothing, and any genuine mismatch reverts to the
     raw precedent text. This isn't theoretical — real runs against
     GPT-4o-mini triggered this fallback (smoothing reworded "5 years" in a
     way that changed the extracted tokens), and the check caught it.
  3. *Checklist* (`contractiq.agents.checklists`): a hardcoded
     per-agreement-type (MSA / NDA / SOW, generic fallback for unrecognized
     types) list of required `ClauseType`s — a business/legal judgment
     call, not something left to the model. Missing clauses aren't
     dropped: they get an explicit placeholder under their own heading in
     the exported DOCX *and* a summary completeness section.
  4. *Export*: the DOCX includes a real Word Table of Contents field (keyed
     off Heading styles — shows placeholder text until updated in Word,
     which is expected native behavior) and best-effort table extraction
     (re-reads the source PDF for a native table on the precedent's exact
     page; only ever succeeds for the non-scanned fraction of the corpus).
  5. *Human-review gate*: not a workflow block — nothing here enforces
     approval — but a structurally redundant label: a bold, dark-red
     banner as the DOCX's first paragraph, the same banner repeated in the
     footer (visible from any page), and the same banner text on the
     returned response object, so a caller that never opens the file still
     sees it.

  ```python
  from contractiq.agents import drafting_agent

  draft = drafting_agent("MSA", "Master service agreement for a network equipment vendor.")
  # draft.clauses -- each with .status ("drafted"/"missing"), .text, .source_document/.source_section/.source_page
  # draft.completeness -- required/present/missing ClauseTypes
  # draft.docx_path -- the exported, banner-carrying DOCX
  ```

- **Routing tests** (`contractiq.agents.run_routing_tests`): a clear
  narrative question, a clear counting question, a genuinely ambiguous one
  (observed and logged, not asserted — there's no single correct route for
  it by construction), and a drafting request. Called with no chat history,
  so it exercises the single-turn baseline, same as the eval harness below.

```python
from contractiq.agents import run_supervisor, make_eval_pipeline, run_routing_tests
from contractiq.eval import load_eval_set, run_eval, print_report

response = run_supervisor("How many contracts are active in Network Equipment?")
print(response.answer, response.trace.agent, response.trace.reasoning)

run_routing_tests()

# eval-set questions are all narrative, so scores should match the
# retrieval-only run -- this re-run is a regression check on the
# classification hop itself.
report = run_eval(load_eval_set(), make_eval_pipeline())
print_report(report)
```

## Conversational memory

The Chat page (below) is the only caller that passes chat history into
`run_supervisor()`. A `contextualize` node (`contractiq.agents.contextualize`)
rewrites a follow-up like "what about the price?" into a standalone question
("What is the price in the Asif & Co agreement?") using the last 6 messages
of history, before classification/retrieval ever see it. A second,
independent mechanism backs this up for narrative questions specifically:
the RAG document scope a prior turn resolved to is carried forward as a
sticky fallback, used only when a follow-up's own vendor-hint extraction
comes up empty. Both are session-scoped only (`st.session_state` in the
Streamlit UI) — nothing is persisted once the browser session ends, and
`eval`/`run_routing_tests` callers pass no history, so this doesn't affect
the single-turn RAGAS baseline.

```python
from contractiq.agents import run_supervisor

response1 = run_supervisor("What is the notice period in the Asif & Co agreement?")
response2 = run_supervisor(
    "What about the price?",
    history=[{"role": "user", "content": "What is the notice period in the Asif & Co agreement?"},
             {"role": "assistant", "content": response1.answer}],
    active_doc_ids=response1.active_doc_ids,
)
```

## UI

Streamlit app (`streamlit run src/contractiq/ui/app.py`), four pages via
`st.navigation`/`st.Page`, each a thin wrapper over the modules above — no
business logic lives in the UI layer itself:

- **Chat** (`ui/chat_view.py`) → `agents.run_supervisor`, threading session
  chat history and the sticky RAG document scope across turns.
- **Alerts** (`ui/alerts_view.py`) → `alerts.compute_expiry_alerts` /
  `generate_digest` — pure computation over the `contracts` SQLite table,
  zero LLM calls, callable independently of the UI (e.g. from a cron job).
- **Draft** (`ui/draft_view.py`) → `agents.drafting_agent`.
- **Upload** (`ui/upload_view.py`) → `extraction.pipeline.process_uploaded_document`
  — add a new contract, optionally linked as a renewal/addendum/variation
  order of an existing one (sets `related_doc_id`, which the SQL agent and
  `extraction.db.get_superseded_doc_ids()` use to exclude superseded rows).

## Deployment

Packaged as a container image (`Dockerfile`, CPU-only, models baked in at
build time) and a set of OpenShift manifests (`openshift/`), with a GitLab
CI pipeline (`.gitlab-ci.yml`) that validates, builds, and (manually)
deploys. See `openshift/README.md` for the operational runbook and
`ARCHITECTURE.md` / `HANDOVER.md` for the full rationale — cluster details,
what's confirmed vs. still pending from infra, and a security-review flag on
the network policy's broad HTTPS egress.

## Layout

```
src/contractiq/
├── ingestion/    # document loading and parsing
├── extraction/   # metadata / structured field extraction
├── retrieval/    # hybrid retrieval over contract corpus
├── agents/       # agent orchestration (contextualize / classify / rag / sql / drafting / decline)
├── alerts/       # contract-expiry computation + digest text generation
├── eval/         # evaluation harness
└── ui/           # Streamlit UI (chat / alerts / draft / upload)
```
