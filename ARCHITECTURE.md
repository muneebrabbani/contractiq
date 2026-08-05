# ContractIQ — Architecture & Coding Logic

Reference document describing what this codebase does and how it's built, for
use as context when packaging/deploying it (e.g., containerizing for
OpenShift). This describes the system **as implemented today**, not an
aspirational design — flag anything below that no longer matches the code
before acting on it.

## What it is

A retrieval-augmented contract-intelligence assistant for a telecom
procurement team. Ingests contract PDFs/DOCX, redacts PII, extracts
structured metadata via LLM, builds a hybrid (dense + BM25) retrieval index,
and exposes three capabilities through a LangGraph-routed agent layer and a
Streamlit UI: narrative Q&A over contracts (RAG), analytics (text-to-SQL over
extracted metadata), and template-assisted contract drafting.

## Repo layout

```
src/contractiq/
├── ingestion/    # raw file loading, OCR fallback, fixed-token chunking
├── extraction/   # recon, PII redaction, clause chunking, LLM metadata extraction, DB
├── retrieval/    # hybrid (dense+BM25+RRF+rerank) retrieval and answer generation
├── agents/       # LangGraph supervisor: rag / sql / drafting agents
├── alerts/       # contract-expiry computation + digest text generation
├── eval/         # RAGAS-based retrieval evaluation harness
└── ui/           # Streamlit app (chat / alerts / draft pages)
```

No `pyproject.toml`/`setup.py` exists — the package is **not pip-installed**.
Every entry point requires `src/` on `PYTHONPATH` (or run from a working
directory where that's already set). This matters for containerization: the
Dockerfile either needs to add a minimal `pyproject.toml` for an editable
install, or set `ENV PYTHONPATH=/app/src`.

## Data flow (batch pipeline, run in order)

```
data/raw/*.pdf|docx
   │
   ├─► recon (contractiq.extraction.run_recon)
   │     native-vs-scanned classification, page/char counts, table detection
   │     → data/processed/recon_report.csv
   │
   ├─► redaction (contractiq.extraction.redact_directory)
   │     loads every file via contractiq.ingestion.loaders.load_document
   │     (PDF pages under ~20 chars of native text are OCR'd via Tesseract
   │     at 300 DPI, transparently — same fallback ingestion uses)
   │     strips phone/email/IP (regex) and addresses (regex anchor +
   │     spaCy GPE/LOC extension) from every page
   │     → data/processed/<doc_id>.redacted.json  (one per source file)
   │     → data/processed/redaction_log.csv         (audit trail)
   │
   ├─► clause-aware chunking (contractiq.extraction.chunk_document_by_clause)
   │     splits redacted JSON on detected clause numbering (not fixed
   │     token windows); computed on the fly by retrieval indexing, not
   │     persisted as a separate artifact
   │
   ├─► metadata extraction (contractiq.extraction.extract_all)
   │     OpenAI Structured Outputs → ContractMetadata (vendor, type,
   │     segment, status, dates, value, currency, owner, signatories,
   │     payment terms, notice period)
   │     → data/processed/contractiq.sqlite3 (table: contracts; upserts by doc_id)
   │     requires OPENAI_API_KEY
   │
   └─► indexing (contractiq.retrieval.build_index)
         embeds every clause chunk (text-embedding-3-large), joins
         segment/status from SQLite + clause_type (keyword match,
         LLM fallback) as Chroma metadata
         → data/processed/chroma/ (persistent ChromaDB collection)
         requires OPENAI_API_KEY
```

`ingestion.ingest_directory()` (fixed 500-token/50-overlap chunks →
`data/processed/*.jsonl`) is a parallel, independent artifact — nothing
downstream reads it. Safe to omit from a minimal deployment.

## Serving-time flow (agents + UI)

```
question ─► classify (OpenAI structured intent: narrative/analytics/drafting)
                │
                ├─ narrative ─► rag agent  ─► retrieval.answer()
                │                              retrieve(): dense (Chroma) + BM25
                │                              (rank_bm25, in-memory, rebuilt from
                │                              Chroma corpus) → RRF fusion →
                │                              cross-encoder rerank
                │                              (sentence-transformers,
                │                              cross-encoder/ms-marco-MiniLM-L-6-v2
                │                              — pulls in torch) → top-k
                │                              → OpenAI chat completion, answer
                │                              constrained to cite retrieved passages
                │
                ├─ analytics ─► sql agent  ─► LLM generates SELECT-only SQL against
                │                              introspected `contracts` schema →
                │                              validated (single statement, SELECT-only)
                │                              → executed on a read-only SQLite
                │                              connection (file:...?mode=ro) →
                │                              plain Python string formatting to
                │                              text (no second LLM call)
                │
                └─ drafting  ─► drafting agent ─► for each required ClauseType
                                                    (per hardcoded MSA/NDA/SOW
                                                    checklist): dense_search()
                                                    filtered by clause_type →
                                                    top precedent chunk →
                                                    placeholder party substitution
                                                    → LLM language-smoothing (numeric
                                                    tokens checked before/after;
                                                    mismatch reverts to raw text)
                                                    → DOCX export with a mandatory
                                                    human-review banner
                                                    → data/processed/drafts/*.docx
```

Every routing decision is logged (`contractiq.agents.graph` logger) and
returned as a `RouteTrace` on the response.

The Streamlit app (`contractiq.ui.app`, `streamlit run src/contractiq/ui/app.py`)
exposes three pages, each a thin wrapper around the above:
- **Chat** (`ui/chat_view.py`) → `agents.run_supervisor`
- **Alerts** (`ui/alerts_view.py`) → `alerts.compute_expiry_alerts` / `generate_digest`
  (pure computation over the `contracts` SQLite table, zero LLM calls, callable
  independent of the UI process — e.g. from a cron sidecar)
- **Draft** (`ui/draft_view.py`) → `agents.drafting_agent`

## Persistent state (must survive container restarts)

All of it lives under `data/processed/`:

| Path | What | Written by |
|---|---|---|
| `contractiq.sqlite3` | extracted contract metadata (`contracts` table) | metadata extraction; read by SQL agent, alerts, indexing join |
| `chroma/` | persistent vector store (dense embeddings + metadata) | `build_index`; read by every RAG/drafting retrieval call |
| `*.redacted.json` | per-document redacted page text | redaction; read by chunking/indexing |
| `redaction_log.csv`, `recon_report.csv` | audit trail | redaction, recon |
| `drafts/*.docx` | generated draft contracts | drafting agent |

`data/raw/` is the input corpus (source PDFs/DOCX) — read-only at pipeline
time, never written by the app.

**Deployment implication**: this is single-node, file-based state (SQLite +
local ChromaDB), not a networked database. A PersistentVolumeClaim mounted at
`data/processed/` (and `data/raw/` if raw docs are supplied by mounting rather
than an upload step) is required. SQLite is single-writer — running more than
one replica of anything that writes (batch pipeline, or a UI instance that
somehow triggers writes) against the same volume risks lock contention/corruption.
The Streamlit UI itself only reads this state at request time, so it can
scale UI replicas read-only as long as the batch pipeline is a separate,
single-writer job.

## Configuration / secrets

`contractiq.config.Settings` (pydantic-settings) reads a local `.env`:

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | required for extraction, indexing, RAG/SQL/drafting agents |
| `CHAT_MODEL` | `gpt-4o-mini` | judge/generation model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | (currently unused by retrieval, which hardcodes `text-embedding-3-large` via `retrieval_embedding_model`) |

For OpenShift: `OPENAI_API_KEY` should become a mounted Secret /
environment variable injected by the platform, not a checked-in `.env`
(current repo keeps a real key in `.env`, gitignored — **do not commit it**).

**Network egress**: every extraction/indexing/agent call reaches
`api.openai.com` over HTTPS. The cluster's egress policy must allow this;
there is no local/offline model path currently.

## System-level (non-pip) dependencies

- **Tesseract OCR** — required for any scanned PDF (this corpus is
  ~85% scanned pages by page count). Not a pip package; container image must
  `apt-get install tesseract-ocr` (or equivalent) and ensure it's on `PATH`.
  `pytesseract` just shells out to the `tesseract` binary.
- **spaCy `en_core_web_sm`** — downloaded via `python -m spacy download
  en_core_web_sm` after `pip install`, not resolved by `requirements.txt` alone.
- **`sentence-transformers` / `torch`** — pulled in for cross-encoder
  reranking; meaningfully increases image size vs. a CPU-only slim base.

## Known gaps relevant to packaging

- `requirements.txt` does not list `streamlit`, though `ui/app.py` requires it
  to run the web UI — add it before building a UI-serving image.
- No `Dockerfile` exists yet in this repo.
- No `pyproject.toml`/`setup.py` — packaging must handle `PYTHONPATH=src`
  explicitly (see "Repo layout" above).
- No health-check/readiness endpoint beyond whatever Streamlit exposes by default.
- The redaction stage's OCR path was added locally (previously OCR was
  explicitly out-of-scope there); OCR'd text is noisier than native
  extraction, which can reduce regex-based PII-redaction recall on scanned
  documents — worth knowing before treating redacted output as a compliance
  guarantee on OCR'd files specifically.
