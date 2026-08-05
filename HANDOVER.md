# ContractIQ — Project Handover & Concept Guide

_Last updated: 2026-08-04 — refreshed for handover to the troubleshooter
supporting the testing team (using Claude Desktop). Section 9 (Deployment)
and Section 12 (Troubleshooting log) are the parts that changed since the
original write-up; everything else (Sections 1–8, 10–11) still describes the
shipped application logic accurately._

Prepared as study/reference material for a supervisor review, and now doing
double duty as the handover doc for whoever picks up testing/troubleshooting.
Covers what the system does, how each piece works technically, and the "why"
behind the design decisions — including the databases, agents, and the
different senses of "memory" that show up in a RAG/agent system like this
one — plus (Section 12) a running log of real issues hit and how they were
fixed, since that's what a troubleshooter actually needs on day one.

---

## 1. What this project is

**ContractIQ** is a retrieval-augmented contract-intelligence assistant built
for a telecom procurement team (Trans World Enterprise Services / Transworld
Associates — TES/TWA). It ingests real contract PDFs/DOCX (Civil Works
agreements, addenda, variation orders with ~46 documents in the working
corpus), and answers three kinds of questions through one interface:

1. **Narrative** — "What is the notice period in the SMK agreement?" (needs
   reading contract text)
2. **Analytics** — "How many contracts are active in Civil Works Services?"
   (needs counting/filtering structured data)
3. **Drafting** — "Draft an MSA for a new vendor." (needs assembling a new
   document from precedent clauses)

A single LangGraph **supervisor agent** classifies each question and routes
it to the agent built for that job (plus a fourth, non-content branch that
declines anything unrelated to contracts — Section 5), rather than using one
generic do-everything prompt. New contracts, including renewals/addenda of
ones already in the system, are added through an Upload page (Section 5a),
and the whole app is packaged to run on the enterprise's private-cloud
OpenShift cluster (Section 9) rather than only locally.

---

## 2. End-to-end pipeline (batch, run once per new document)

```
data/raw/*.pdf|docx
   │
   ▼
① RECON            classify native vs. scanned, page/char counts, table detection
   ▼
② REDACTION        strip PII (phone/email/IP/address); OCR (Tesseract) any
   │                scanned page first, then redact the OCR'd text
   ▼
③ CLAUSE CHUNKING  split on detected clause numbering (4.1, Section 4, ARTICLE
   │                IV) — not fixed-size windows — so each clause stays intact
   ▼
④ METADATA         LLM (Structured Outputs) extracts ~20 structured fields
   EXTRACTION       per contract → SQLite
   ▼
⑤ INDEXING         embed every clause chunk → persistent vector store (Chroma),
                    joined with SQLite metadata for filtering
```

Each stage writes to `data/processed/` and is **idempotent / upsert-based** —
re-running the whole pipeline after adding new documents is safe, matched by
a content-derived `doc_id`.

### Why redaction happens before anything else
Vendor/party names and commercial terms (values, dates, penalties) are
**never** touched — only phone numbers, emails, IPs, and physical addresses.
This is a deliberate compliance boundary: strip contact-level PII, keep every
commercially meaningful fact intact for the business logic downstream.

### Why OCR sits inside redaction, not as a separate optional step
~85% of this corpus (page-count-weighted) is **scanned, image-only PDFs** —
no extractable native text. Originally, OCR was scoped *out* of redaction
(it only handled cleanly-parsed native PDFs), which meant the majority of real
contracts never entered the system at all. The fix: route every page through
the same OCR fallback ingestion already used (`pytesseract`, 300 DPI), *then*
run the same regex/NER redaction over the OCR'd text. Trade-off worth
knowing: OCR text is noisier than native extraction, so regex-based PII
redaction is theoretically slightly less reliable on scanned pages than on
clean text — a garbled digit could let something slip past a phone-number
regex. Framed correctly to a supervisor: **this is a stated, understood
trade-off, not an oversight.**

---

## 3. The databases — three different stores, three different jobs

This is a good place to be precise, because the project deliberately uses
**three separate persistence mechanisms**, each suited to a different kind of
question:

| Store | Technology | What's in it | Which agent reads it | Why this store |
|---|---|---|---|---|
| **Structured metadata** | SQLite (`contractiq.sqlite3`, table `contracts`) | ~20 extracted fields per contract: vendor, business unit, department, dates, value, currency, payment terms, etc. | **SQL agent** (analytics), Alerts page | Relational/tabular questions ("how many," "which contracts expire in 30 days") are what SQL is *for* — no LLM needed to compute an aggregate correctly |
| **Vector store** | ChromaDB (persistent, `data/processed/chroma/`) | Dense embeddings (`text-embedding-3-large`) of every clause chunk, plus metadata (doc_id, section, page, segment, status, clause_type) for filtering | **RAG agent** (narrative), Drafting agent (precedent lookup) | Semantic similarity search — "find clauses that mean something like this question," not exact keyword match |
| **Lexical index** | BM25 (`rank_bm25`, in-memory, rebuilt from the same chunks) | Same clause chunks, scored by term-frequency statistics | RAG agent, fused with the vector store's results | Catches exact-term/keyword matches (contract numbers, specific defined terms) that pure semantic embedding similarity can sometimes miss |

**Why not just one database?** Different question types need fundamentally
different retrieval mechanics:
- "How many contracts are in the Network Equipment segment?" → this is a
  `COUNT(*) WHERE segment = ...` — computing this via embedding similarity
  would be unreliable and slower than SQL, which gets it *exactly* right,
  every time.
- "What does the termination clause say?" → there's no SQL query for "find
  text that means this" — that's what embeddings + keyword search are for.

This is a standard pattern in production RAG+agent systems: **structured
store for facts/aggregates, vector store for unstructured text search.**

### How the two retrieval legs combine: Reciprocal Rank Fusion (RRF)
```
query ─┬─► dense search (Chroma, ~20 candidates)   ─┐
        └─► BM25 search (~20 candidates)            ─┼─► RRF fusion ─► cross-encoder rerank ─► top-k
```
RRF combines the two ranked lists using `score = Σ 1/(60 + rank)` — **only
rank position is used, not raw similarity scores**, because cosine similarity
(dense) and BM25 term-frequency scores aren't on comparable numeric scales.
Fusing raw scores would silently favor whichever leg happens to produce
larger numbers. Fusing by rank sidesteps that.

A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) then rescores
the fused shortlist by jointly encoding (query, passage) pairs — this is more
accurate than either single-vector search but too slow to run over the whole
corpus, so it only reranks the already-narrowed candidate pool.

---

## 4. "Memory" — the concept your supervisor will likely probe

This term gets used loosely in AI coursework, so it's worth being precise
about which kind of memory this system has, and which kind it deliberately
does **not** have.

### a) Context window / short-term memory (per single call, not persisted)
Every LLM call in this system — classification, RAG answer generation, SQL
generation, drafting — is a **single, stateless request**. The "memory" for
that one call is just whatever is placed in the prompt: system instructions +
retrieved passages + the user's question. Nothing about that call is
remembered afterward. Call it again with the same question and it re-derives
everything from scratch (retrieval runs again, generation runs again).

### b) Long-term / knowledge memory (persistent, built once, reused every call)
This is the actual "memory" of the system in a meaningful sense — the three
databases in Section 3. They are built once (the batch pipeline) and queried
repeatedly at answer time. This is the standard RAG framing: **the LLM's own
parametric knowledge is frozen and generic; the vector store + SQL table are
the system's actual, current, correct knowledge about these 46 specific
contracts.** The LLM never "knows" contract facts — it's shown the relevant
facts as retrieved context and asked to answer only from that.

### c) Conversational memory (does NOT exist in this system — by design, currently)
There is **no session/chat history** carried between questions. Ask "What's
the notice period in the SMK agreement?" and then "What about the payment
terms?" — the second question is answered with **zero awareness** that the
first question existed or that "the SMK agreement" was just discussed. Each
call to `run_supervisor(question)` is fully independent. If your supervisor
asks "does it remember the conversation?" — the honest, correct answer is
**no**, and that's a legitimate, common scope boundary for a v1 system (adding
it would mean threading conversation history into the classifier + RAG
prompt, and deciding how much history to keep in context — a real design
decision, not yet made here).

### Summary table for the exam-prep angle

| Type of memory | Exists here? | Where |
|---|---|---|
| Per-call context (prompt) | Yes | Assembled fresh every call from retrieved passages |
| Long-term knowledge store | Yes | SQLite + Chroma + BM25 (Section 3) |
| Cross-question conversational memory | **No** | Not implemented — each question is stateless |
| Model's own trained/parametric memory | Yes, but untrusted | The system prompt explicitly forbids answering from the model's general knowledge — answers must cite retrieved passages |

---

## 5. The agents — LangGraph supervisor pattern

```
question ──► classify (OpenAI Structured Outputs: {intent, reasoning})
                 │
                 ├─ narrative    ─► RAG agent      ─► retrieval.answer()
                 ├─ analytics    ─► SQL agent       ─► generate → validate → execute → format
                 ├─ drafting     ─► Drafting agent  ─► precedent retrieval → assembly → DOCX
                 └─ out_of_scope ─► decline node    ─► fixed refusal message, no LLM/DB call
```

### Classifier
A small structured-output call that must commit to exactly one of four
intents — `narrative` / `analytics` / `drafting` / `out_of_scope` (no
"unsure" branch — ambiguous-but-on-topic questions still get a best-effort
narrative/analytics/drafting routing decision, logged with its reasoning for
auditability). Falls back to `narrative` if the classification call itself
fails.

**`out_of_scope` is a deliberately narrow guardrail**, added after noticing
the original three-way classifier had no way to say "this isn't a contracts
question" — it would force something like "what's the weather today?" into
whichever of the three intents looked least wrong, and a downstream agent
would then either hallucinate an answer or fail confusingly. The prompt is
written to bias hard against over-using this branch: anything that mentions
contracts/clauses/vendors/agreements — even corpus-wide or oddly-phrased
questions like "what clauses are common across all our contracts?" — must
still resolve to narrative/analytics/drafting. `out_of_scope` is reserved for
genuinely unrelated small talk/general knowledge. Routed to a `decline` graph
node that returns a fixed message and makes **no LLM or database call at
all** — cheaper and more predictable than asking a model to write its own
refusal.

### RAG agent (narrative)
Thin wrapper around the retrieval pipeline in Section 3, plus one addition we
built during this project: **vendor-name pre-filtering**. This corpus has 17
near-identical boilerplate "Civil Works" contracts (same template, different
vendor). A cross-encoder reranker scores passages by *text content* — and
when 17 documents share nearly word-for-word clause text, the reranker
genuinely cannot tell them apart, so a question naming a specific vendor
("Under the SMK Agreement...") could retrieve some *other* vendor's
near-identical clause instead. The fix: extract a vendor name from the
question (if one is explicitly named) via a small LLM call, resolve it to
that vendor's `doc_id`s in SQLite, and pass that as a hard pre-filter into the
retrieval call — the same mechanism already used for `segment`/`status`
filtering. This is a good concrete "how we found and fixed a real retrieval
bug" story for a supervisor discussion.

### SQL agent (analytics) — three independent safety layers
1. **Connection-level (the real guarantee):** SQLite opened in URI read-only
   mode (`file:...?mode=ro`) — structurally incapable of writing, verified
   directly (a `DELETE` against this connection raises
   `sqlite3.OperationalError: attempt to write a readonly database`).
2. **Statement-shape validation:** rejects anything that isn't a single
   `SELECT` statement before it ever reaches SQLite.
3. **Schema-scoped generation:** the model only ever sees the `contracts`
   table's schema (introspected from the SQLAlchemy model, so it can't drift
   out of sync).

Critically: **the LLM's job stops at generating the SQL string.** Execution
is real SQLite, and turning the result into text is a plain Python
Markdown-table formatter (with `|`/newline escaping so a cell value can't
break the table syntax) — **not** a second LLM call. This means there is no
path for the final answer's numbers to drift from what SQL actually
computed.

The SQL agent's system prompt also documents how to reason about
`related_doc_id` (Section 5a below) in query form — e.g. "which contracts are
currently active" needs to exclude superseded rows via a `NOT IN (SELECT
related_doc_id ...)` subquery, not just `status = 'active'`, since a renewed
contract's original row may still carry a stale status value.

### 5a. Upload page & lightweight contract versioning
Originally, the only way to get a document into the system was the offline
batch pipeline (Section 2) run over the whole `data/raw/` corpus. A
**Streamlit Upload page** now runs the same stages — redact → clause-chunk →
extract metadata → index — for a **single newly-added file**, in-process, no
separate job:
- `extraction/redaction.py` was refactored to expose `redact_document()` (one
  file, returns the redacted doc + records) as the shared primitive under
  both `redact_directory()` (the batch corpus sweep) and the new
  `extraction/pipeline.py:process_uploaded_document()` (single-file path) —
  so redaction logic isn't duplicated between the two entry points.
- The redaction audit log gets an `append_redaction_log()` variant so a
  single upload doesn't clobber the corpus-wide `redaction_log.csv` the way
  `redact_directory()`'s write does.
- Re-uploading a file with identical content is idempotent (hash-compared);
  re-uploading the same filename with *different* content is rejected rather
  than silently overwritten.

**Versioning/renewal linking.** When uploading, the user can optionally mark
the new document as a renewal/addendum/variation order of an existing
contract via a dropdown (`ui/upload_view.py`). That sets a new
`related_doc_id` column on the `contracts` table (SQLite `ALTER TABLE`,
applied idempotently on startup since the existing corpus's table predates
this column) pointing at the doc_id it supersedes. This is deliberately
**never LLM-extracted** — it's a fact only the uploader can assert, not
something inferable from the document text alone. Downstream, this one
column answers "is this contract still current?" without any new table:
`extraction/db.py:get_superseded_doc_ids()` and the SQL agent's prompt
(Section 5) both treat "doc_id appears as some other row's `related_doc_id`"
as *this row is no longer the active version*. This addresses what was
previously listed as a known gap ("no document-versioning system") — it's
intentionally minimal (one nullable FK-like column, not a full version-graph
or diff view) but closes the "which contract is the current one" question
that matters most for procurement.

### Drafting agent — the highest-risk feature, built conservatively
Does not generate contract text freely. Instead:
1. For each required clause type (per a hardcoded MSA/NDA/SOW checklist),
   retrieves the best matching **precedent clause** from an existing contract
   (hard `clause_type` filter + semantic ranking — deliberately bypasses the
   full hybrid pipeline, since "find the best exemplar of a known type" is a
   different, simpler problem than open-ended retrieval).
2. Party names are substituted deterministically (no LLM). The **only** LLM
   step is language smoothing — under a prompt that forbids changing any
   number/date/term — backed by a **post-hoc numeric check**: extract all
   numbers from the text before and after smoothing; any mismatch reverts to
   the raw precedent text. This isn't theoretical — it was observed
   triggering for real against GPT-4o-mini (smoothing reworded "5 years" in a
   way that changed the numbers), and the check caught it.
3. Missing clauses aren't silently dropped — they get an explicit placeholder
   plus a completeness summary.
4. A mandatory human-review banner appears in the DOCX (first paragraph +
   footer) and on the response object itself — a structurally redundant
   safety label, not a workflow gate (nothing here *enforces* human sign-off).

### Clause-type classification
A controlled vocabulary (`termination`, `governing_law`, `payment_terms`,
etc.) needed because `section_title` alone isn't reliable — two contracts
title the same clause differently. Keyword-matches the section title first
(free, deterministic, covers the common case); falls back to an LLM call only
when the title is missing or unrecognized.

---

## 6. Metadata extraction schema (Core Metadata tier)

Aligned to a procurement-team requirements document. Extracted per contract
via OpenAI Structured Outputs (strict JSON schema, not tool-calling) against
a Pydantic model, persisted to SQLite, **upserted by `doc_id`** (re-running
never duplicates rows):

`contract_number`, `agreement_title`, `contract_type`, `agreement_category`,
`vendor`, `project_name`, `business_unit` (TES/TWA — which internal entity is
the contracting party), `department` (internal department responsible —
*not* a person), `segment`, `contract_description`, `status`,
`execution_date`, `effective_date`, `expiry_date`, `renewal_date`,
`contract_duration`, `notice_period`, `auto_renewal`, `value`, `currency`,
`payment_terms`, `payment_milestones`, `signatory_names`.

**A worked example of a real extraction-quality problem and its fix:** the
schema originally had an `owner` field ("internal business owner ... if
stated"). In practice, the model kept extracting **TES/Transworld Associates
itself** or a **signatory's name** as the "owner" — because those are the
most prominent names in a contract, even though neither is actually an
internal department. The fix wasn't a code bug fix — it was tightening the
**field description** in the Structured Outputs schema to explicitly exclude
the counterparty name and signatory names, and renaming the concept from
"owner" (person) to "department" (e.g. "Procurement & Contracts", "Legal").
Re-running extraction with the corrected description dropped false-positive
"owners" from 10 documents to 0, while correctly keeping the 9 genuine
department mentions. **This is a good talking point**: LLM extraction quality
is often a prompt/schema-description problem, not a retrieval or model
problem.

---

## 7. Evaluation harness (RAGAS)

Built *before* real retrieval existed, so quality could be measured from day
one against a trivial keyword-overlap stub pipeline, then re-measured as each
real component landed.

**Metrics** (judged by an LLM, `gpt-4o-mini`):
- **Faithfulness** — is every claim in the answer actually supported by the
  retrieved passages? (catches hallucination)
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — did retrieval surface the passages relevant to the
  reference (gold) answer?

**A worked example of an eval methodology mistake and its fix:** the eval set
mixes narrative and analytics questions. Initially, all questions were run
through the **RAG-only** pipeline — which meant analytics questions
("How many contracts are active...") were being scored on retrieval quality
for a question type that should have gone to the **SQL agent** instead,
guaranteeing failure. Worse: the SQL agent doesn't produce
`retrieved_contexts` at all (it queries SQLite directly) — RAGAS's
faithfulness/context-precision metrics *require* non-empty retrieved context
and hard-error without it. **This is a structural mismatch, not a bug**:
those two metrics measure "is the answer grounded in retrieved passages,"
which doesn't apply to a SQL-grounded answer. The fix: route eval questions
through the **agent-routed** pipeline (matching production routing) and
exclude analytics questions from RAGAS scoring specifically, since they need
a different correctness check (comparing the SQL result against the gold
answer directly, not a RAGAS metric).

**Also worth knowing:** RAGAS scores are **not perfectly deterministic**
across runs of the exact same question/pipeline — the judge itself is an LLM
call. A ~10-point swing in one metric between two identical runs is
plausible measurement noise, not necessarily a regression. Don't over-read a
single run.

---

## 8. UI (Streamlit)

Four pages, each a thin wrapper over the logic above — no business logic
lives in the UI layer itself:
- **Chat** → `agents.run_supervisor()` — narrative/analytics/drafting/decline, routed
- **Alerts** → `alerts.compute_expiry_alerts()` / `generate_digest()` — pure
  computation over the SQLite metadata table, **zero LLM calls**, callable
  independently of the UI (e.g. from a cron job) for exactly that reason.
  Buckets contracts into expired/30/60/90-day windows, mutually exclusive
  (tightest threshold wins), plus a separate list for dates that don't parse
  as ISO 8601 rather than silently dropping them.
- **Draft** → `agents.drafting_agent()`
- **Upload** → `extraction.pipeline.process_uploaded_document()` (Section
  5a) — add a new contract, or link one as a renewal/addendum/variation
  order of an existing contract; runs redact/extract/index synchronously in
  the request and shows the extracted metadata back to the uploader for a
  quick sanity check.

Uses `st.navigation`/`st.Page` (multi-page app API) rather than Streamlit's
older `pages/` directory convention, with explicit `url_path`s so each page
is directly linkable.

---

## 9. Deployment — private cloud (OpenShift)

The system packages as a real container image and a set of OpenShift
manifests, for running inside the enterprise's **private cloud OpenShift
cluster** rather than a laptop, plus a GitLab CI pipeline that automates the
validate → build → deploy sequence. Users reach it only from inside the
enterprise network; the pod's only required outbound path at *runtime* is
`api.openai.com`.

### Cluster is now confirmed (this changed since the original write-up)
Infra has come back with concrete values — a troubleshooter should treat
these as settled, not TBD:

| Item | Value |
|---|---|
| API server URL | `https://api.rhcloud-south.tes.com.pk:6443` |
| Namespace | `contract-iq` |
| Route/ingress hostname | `contract-iq.apps.rhcloud-south.tes.com.pk` |
| Storage class | `powerstore-block` |
| Internet access | **Direct** via the cluster's egress IP — no forward proxy, at build or runtime |
| Internal image registry route | `https://default-route-openshift-image-registry.apps.rhcloud-south.tes.com.pk` (only for pulling/pushing from outside the cluster; in-cluster manifests use the internal `image-registry.openshift-image-registry.svc:5000` service DNS) |
| Server version | OpenShift 4.21.8 / Kubernetes 1.34.5 |

**"No forward proxy" simplified several things** that the original version of
this doc assumed would need proxy configuration: the Dockerfile's model-baking
`RUN` steps and `oc start-build` no longer need `--build-arg` proxy values,
and `openshift/configmap.yaml` doesn't need proxy env vars either. The
trade-off called out in `openshift/networkpolicy.yaml`: since a
`NetworkPolicy` can't match on a DNS name like `api.openai.com`, only
IPs/CIDRs, the egress rule allows outbound HTTPS broadly (`0.0.0.0/0:443`)
rather than scoping to a proxy address. That file has an explicit **FLAG FOR
SECURITY REVIEW** comment: this workload touches PII-adjacent contract data
and there's no proxy layer inspecting/logging egress — get security sign-off
before treating this as production-ready, and narrow it to OpenAI's published
IP ranges if/when available.

**Still pending from infra — do not guess these** (unchanged from before,
still open, each has a `# TODO(infra): ...` in the relevant manifest):
1. `oc login` auth method (password / token / kubeconfig) — not yet told to us.
2. Network path to the API server (VPN / jump host / direct) — not yet confirmed.
3. RBAC role granted in `contract-iq` (edit vs admin) — not yet confirmed.
4. SCC assigned to the namespace — if stricter than `restricted-v2`, the
   `securityContext` blocks in `deployment.yaml`, `pipeline-job.yaml`,
   `data-loader-pod.yaml`, and `digest-cronjob.yaml` may need adjusting.
5. TLS termination on the Route (edge vs passthrough vs custom certs).
6. Route reachability scope (internal-only/VPN vs broader) — unconfirmed.

### GitLab CI (`.gitlab-ci.yml`) — new since the original write-up
Three stages, gated on merge requests and pushes to `main`:
- **`validate`** — `pip install .` + `compileall` + `import contractiq`
  sanity check. Cheap, runs on everything.
- **`docker-build`** — builds the image with **Kaniko**
  (`gcr.io/kaniko-project/executor`), not Docker-in-Docker, because GitLab
  runners here don't have a Docker daemon available; `--no-push
  --single-snapshot` means this stage only *proves the image still builds*,
  it doesn't publish anywhere. `allow_failure: true` so a build hiccup
  doesn't block the pipeline while this is still being hardened.
- **`oc-start-build`** — the real deploy: `oc login` (via `OC_LOGIN_TOKEN` CI
  variable, masked/protected) → `oc start-build --from-dir=. --follow --wait`
  → `oc set image` → `oc rollout status`. `when: manual` and
  `main`-branch-only on purpose — this shouldn't fire unattended until the
  "still pending from infra" items above are resolved. Runs under a
  `resource_group: production` (so two deploys can't race each other) with a
  10-minute rollout timeout, and `after_script` wipes the temporary
  `.kube/config` so the token doesn't linger on the runner's filesystem.
- Jobs are pinned to runners tagged `contract-iq` (`default: tags:`) rather
  than any shared runner.

### What changed in the app packaging to make this possible
- **`pyproject.toml`** — the codebase previously had no packaging manifest at
  all (`PYTHONPATH=src` hacks only); it's now a real installable package
  (`pip install .`), which is what the Dockerfile installs from — not the
  older, incomplete `requirements.txt` (which is missing `streamlit` and a
  few others).
- **`Dockerfile`** — `python:3.11-slim` base. Installs `tesseract-ocr` via
  `apt` (system dependency, not pip — required for the ~85%-scanned corpus's
  OCR fallback). Installs **CPU-only `torch`** first, from PyTorch's own CPU
  wheel index, *before* `pip install .` — left to default PyPI resolution,
  `sentence-transformers` pulls the CUDA build of torch plus ~2GB of
  `nvidia-*` wheels this cluster has no GPU to use. Also pre-warms
  `tiktoken`'s `cl100k_base` encoding at build time (`TIKTOKEN_CACHE_DIR=/app/.tiktoken`)
  so the running pod never fetches it over the network either. Runs Streamlit
  as the container's entrypoint on `0.0.0.0:8501` with
  `--server.enableCORS=false --server.enableXsrfProtection=false` (the
  standard fix for Streamlit's WebSocket handshake behind a TLS-terminating
  reverse proxy, i.e. the OpenShift Route).
- **Two models baked in at *build* time, not fetched at runtime**: spaCy's
  `en_core_web_sm` (redaction's address-NER step) and the cross-encoder
  reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (from huggingface.co). This
  matters because the running pod never needs to reach PyPI, spaCy's model
  host, or huggingface.co after startup — it only needs `api.openai.com`.
- **Non-root by construction, not by a hardcoded `USER`**: OpenShift's
  `restricted-v2` SCC runs containers as an arbitrary UID from the
  namespace's allocated range (always in group 0), so the Dockerfile never
  sets a fixed `USER` — instead it `chgrp -R 0 /app && chmod -R g=u,g+w /app`
  so whatever UID OpenShift assigns can still write where needed (e.g. a
  mounted PVC subpath).
- **`.dockerignore`** deliberately excludes `data/` (the PII-bearing contract
  corpus must come from a mounted PVC, never baked into an image layer that
  could land in a registry) and `.env` (secrets come from an OpenShift
  `Secret`, never a checked-in file).

### `openshift/` manifests
`buildconfig.yaml` (in-cluster image build via `oc start-build --from-dir=.`
— no external registry or git remote needed), `pvc.yaml` (persistent storage
for `data/processed/` + `data/raw/`, using the confirmed `powerstore-block`
storage class), `configmap.yaml`, `secret.yaml` (placeholder template only —
the real secret is created via `oc create secret generic contractiq-secrets
--from-literal=OPENAI_API_KEY=...`, never applied from the template file),
`deployment.yaml` + `service.yaml` + `route.yaml` (the running app),
`networkpolicy.yaml` (default-deny except the Route on ingress; DNS + broad
HTTPS egress, see the security-review flag above), `pipeline-job.yaml`
(one-shot batch pipeline as an OpenShift `Job`), `data-loader-pod.yaml` (a
throwaway pod for `oc rsync`-ing the existing local corpus onto the PVC
before first run), and `digest-cronjob.yaml` (optional — runs the expiry
digest on a schedule, but only writes to pod logs today since there's no
email/Slack delivery in the codebase yet).

**Why `replicas: 1` is load-bearing, not a default left unconsidered**: both
SQLite and the local Chroma store are single-writer, file-based state
(Section 3). Scaling the Deployment beyond 1 replica risks two processes
writing the same PVC-backed SQLite file concurrently. The full runbook —
build, load data, run the pipeline, deploy, and ongoing operations like
scaling down before a bulk re-sync — is in `openshift/README.md`; the
per-component technical rationale (why each persistent path needs a PVC,
what's single-writer, what env vars matter) is in `ARCHITECTURE.md`.

### What's verified vs. not
`docker build .` was run locally to validate the Dockerfile end-to-end (see
Section 12 for a bug this actually caught and the fix applied). The
OpenShift manifests themselves have **not** been applied against a live
cluster — no cluster access from this environment. The `oc-start-build` CI
job is `when: manual` specifically because of this: don't let it fire
unattended until the six "still pending from infra" items above are
resolved, especially SCC strictness vs. `restricted-v2`.

---

## 10. Known limitations (say these proactively — it reads as maturity, not weakness)

1. **No conversational memory** across questions (Section 4c).
2. **Single-node, file-based storage** — SQLite + local Chroma, not a
   networked database. Fine for one machine; the OpenShift deployment
   (Section 9) works around this with a PVC and a hard `replicas: 1`
   constraint rather than solving it — a real multi-writer story would mean
   swapping SQLite/Chroma for networked equivalents (e.g. Postgres +
   pgvector), not yet done.
3. **Boilerplate/near-duplicate contracts** are a structural challenge for
   any purely text-similarity-based reranker — the vendor-filter fix
   (Section 5) addresses the case where the question names a vendor
   explicitly, but doesn't solve retrieval ambiguity in general.
4. **OCR quality vs. redaction confidence** — scanned pages go through OCR
   before redaction; regex-based PII stripping is marginally less reliable on
   OCR'd text than on clean native text.
5. **RAGAS can't score every agent type** — SQL-agent answers need a
   different, not-yet-built correctness check (comparing SQL output to gold
   answer) rather than RAGAS's context-grounding metrics.
6. **Versioning is a single link, not a history** — `related_doc_id`
   (Section 5a) records that one contract supersedes another, asserted by
   whoever uploads the renewal, but there's no chain/diff view beyond that
   one pointer and no way to correct a wrong link except re-uploading.
7. **No delivery channel for the expiry digest** — `alerts.generate_digest()`
   and its optional CronJob only produce text; there's no email/Slack/Teams
   integration to actually send it anywhere yet.

---

## 11. Likely supervisor questions and how to answer them

- **"Why two databases (SQL + vector) instead of one?"** → Section 3:
  different question types need fundamentally different retrieval mechanics;
  aggregation belongs in SQL, semantic text search belongs in a vector store.
- **"Does the system remember previous questions?"** → No, by current
  design (Section 4c) — each call is stateless; explain what it *would* take
  to add it.
- **"How do you stop the LLM from hallucinating a wrong SQL answer?"** →
  Section 5, SQL agent: the LLM only writes the query string; execution and
  formatting are deterministic Python/SQLite, not a second LLM call.
- **"How do you know retrieval actually works?"** → Section 7: the RAGAS
  harness, plus a worked example of finding and fixing a real retrieval bug
  (vendor confusion across boilerplate contracts).
- **"What happens if the LLM messes up during drafting?"** → Section 5,
  drafting agent: the post-hoc numeric-token check that reverts to raw
  precedent text — with a real observed case of it firing.
- **"Is PII actually protected?"** → Section 2: redaction strips contact PII
  before anything else touches the raw text, with the OCR caveat stated
  honestly rather than glossed over.
- **"What stops someone asking it something totally unrelated to contracts?"**
  → Section 5: a fourth `out_of_scope` classifier intent routes to a fixed
  decline message with no LLM/DB call, deliberately biased so any
  contract-adjacent question still gets routed properly rather than declined.
- **"How do you know which version of a contract is current?"** → Section
  5a: an explicit, human-asserted `related_doc_id` link set at upload time
  (never LLM-inferred), which both the SQL agent and a helper query can use
  to exclude superseded rows.
- **"How would this actually get deployed for the procurement team to use?"**
  → Section 9: containerized (Dockerfile + pyproject.toml) for the
  enterprise's private-cloud OpenShift cluster, with models baked in at build
  time so the running pod only needs OpenAI egress, a PVC for the
  single-writer SQLite/Chroma state, and a deliberate `replicas: 1`
  constraint — full manifests in `openshift/`, runbook in
  `openshift/README.md`.

---

## 12. Troubleshooting log (for whoever picks this up next)

This section is written for a troubleshooter supporting the testing team,
not a supervisor review — real issues hit, root cause, and the fix, plus how
to get the app running locally on Windows to reproduce anything the testing
team reports. Add to this list rather than replacing it as new issues turn up.

### Running it locally (Windows, no Docker)
The testing team doesn't need a container to try the app — the project's own
`.venv` runs it directly:

```powershell
cd E:\Personal\atomcamp\final_project\code
.\.venv\Scripts\python.exe -m streamlit run src\contractiq\ui\app.py --server.port 8501
```

Then open `http://localhost:8501`. Requirements: `.env` must exist (copy from
`.env.example`) with a real `OPENAI_API_KEY` — the Chat/Draft/Upload pages all
call OpenAI and will error without one. If `.venv` is missing or stale,
recreate it with `pip install -e .` from `pyproject.toml` (not
`requirements.txt` — see Section 9, it's missing `streamlit`).

### Issue: `pydantic_core.ValidationError` — `oc_login_token: Extra inputs are not permitted`
- **Symptom**: app fails to start (or crashes on first import of
  `contractiq.config`), traceback ending in `Settings()` at
  `src/contractiq/config.py`.
- **Root cause**: `Settings` (`config.py`) only declares four fields
  (`openai_api_key`, `chat_model`, `embedding_model`,
  `retrieval_embedding_model`), but pydantic-settings reads *every* variable
  present in `.env`. `OC_LOGIN_TOKEN` was added to `.env`/`.env.example` for
  the `oc-start-build` CI/deploy step (Section 9) and for local `oc` CLI use
  — it was never meant to be consumed by the app itself — but with no
  `extra=` policy set, pydantic-settings' default is to **forbid** unknown
  fields, so any `.env` var the model doesn't declare crashes the whole app.
- **Fix applied**: `model_config = SettingsConfigDict(..., extra="ignore")`
  in `config.py` — the app now silently ignores `.env` vars it doesn't model
  (like the deploy-only OC token) instead of crashing on them. If a *future*
  env var genuinely needs to reach the app's own settings, add it as a real
  field on `Settings` rather than relying on `extra="ignore"` to paper over
  it.

### Issue: Dockerfile's dependency-install `RUN` step was silently broken
- **Symptom**: would have failed the `docker-build` CI stage (or a manual
  `docker build .`) with pip's "You must give at least one requirement to
  install" error — not yet hit in CI because `docker-build` currently has
  `allow_failure: true` (Section 9), so this could have shipped unnoticed for
  a while.
- **Root cause**: when the CPU-only `torch` pre-install step was added ahead
  of installing the app itself (to avoid pulling ~2GB of CUDA wheels via
  `sentence-transformers`, Section 9), the trailing `.` (the actual "install
  this package" argument) was dropped from the second `pip install
  --no-cache-dir` call, leaving it with no target at all.
- **Fix applied**: restored the `.` — `pip install --no-cache-dir torch
  --index-url https://download.pytorch.org/whl/cpu && pip install
  --no-cache-dir .`. Worth a fresh `docker build .` (or letting the
  `docker-build` CI job run) to confirm the full image still builds
  end-to-end after this.

### Issue: `oc` not recognized in PowerShell
- **Symptom**: `oc version --client` → `CommandNotFoundException`, even
  though `oc.exe` is sitting right there in the project folder.
- **Root cause**: PowerShell doesn't execute from the current directory by
  default (unlike old `cmd.exe`), and the folder wasn't on `PATH`.
- **Fix applied / caveats**: full diagnosis and fix (adding the project
  folder to the **User** `PATH`) is in `oc-cli-troubleshooting.md` — read
  that file directly if a tester hits this. Flagged there as a fragile
  workaround (`oc.exe` shouldn't really live inside the git working tree
  long-term; the recommended follow-up is moving it to something like
  `%USERPROFILE%\bin` instead and gitignoring it).

### Known open item (not yet fixed): `README.md` has unresolved merge-conflict markers
`README.md` currently has literal `<<<<<<< HEAD` / `=======` / `>>>>>>>`
markers starting at line 1 — a merge conflict was committed without being
resolved. Doesn't affect the running app, but it's the first file anyone new
opens, so it should get cleaned up (pick the correct side, or hand-merge)
before this doc set goes out to the testing team.

### Where to look next for anything not covered here
- App logic questions → Sections 1–8 of this document.
- Deployment/cluster questions → Section 9 above, `openshift/README.md`
  (runbook), `ARCHITECTURE.md` (per-component rationale).
- Local `oc` CLI issues → `oc-cli-troubleshooting.md`.
- CI pipeline behavior → `.gitlab-ci.yml` directly; the comments were
  trimmed down from an earlier, more heavily-annotated version, so if a CI
  job's *intent* isn't obvious from the YAML alone, check this doc's Section
  9 CI write-up first.
