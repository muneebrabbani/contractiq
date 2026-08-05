<<<<<<< HEAD
# ContractIQ

A retrieval-augmented contract intelligence assistant for a telecom procurement team.

## Status

Project skeleton. Stages to be built incrementally: ingestion, metadata extraction,
evaluation harness, hybrid retrieval, agent orchestration, drafting, and a Streamlit UI.

## Setup

```
pip install -r requirements.txt
cp .env.example .env  # then fill in OPENAI_API_KEY
```

Ingestion's OCR fallback (for scanned PDFs) needs the Tesseract OCR engine
installed separately and on PATH — it is not a pip package. See
https://github.com/tesseract-ocr/tesseract for install instructions.

Extraction's address redaction needs spaCy's English model, downloaded
separately after `pip install`:

```
python -m spacy download en_core_web_sm
```

## Ingestion

Supports PDF (native text, with automatic OCR fallback via Tesseract for
scanned pages) and DOCX. Documents are split into fixed-size token chunks
(500 tokens, 50 overlap, `cl100k_base` encoding) and written as JSONL to
`data/processed/`.

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

- **Redaction** (`contractiq.extraction.redact_directory`): for files recon
  would classify as fully native (OCR is out of scope for this pass — scanned
  files are skipped and logged), strips phone numbers, emails, IP addresses
  (regex, IP validated via stdlib `ipaddress`), and physical addresses
  (regex-anchored on street-number + street-type structure, extended through
  trailing city/state/zip via spaCy GPE/LOC only — never triggered by a bare
  place name, and never extended across a party name). Vendor/party/signatory
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
  status, effective/expiry date, value, currency, owner, signatory names,
  payment terms, notice period), one record per contract, persisted via
  SQLAlchemy to `data/processed/contractiq.sqlite3`. Re-running upserts by
  `doc_id` rather than duplicating rows. Requires `OPENAI_API_KEY` in
  `.env`. `segment`/`status` feed retrieval's metadata pre-filtering below —
  a document indexed before metadata extraction has run just won't match
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
  `OPENAI_API_KEY` in `.env`.

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
  `section_title`, `page` from the chunker.

- **Dense + BM25 + fusion**: both legs pull a ~20-candidate pool per query.
  Fusion is Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)` across
  whichever lists a chunk appears in) rather than a weighted score blend —
  dense cosine similarity and BM25 scores aren't on comparable scales, so
  only rank position is used, not raw confidence.

- **Reranking** (`contractiq.retrieval.reranker`): a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers` —
  pulls in `torch`) rescores the fused shortlist by jointly encoding
  (query, passage) pairs, then the top-k survive. Loaded once, not per call.

- **`retrieve(query, k, segment=None, status=None)`**: the building block
  above, returning `RetrievedChunk`s (text + score + citation metadata).
  `segment`/`status` filter both the dense leg (Chroma `where`) and the
  BM25 leg (doc_id allowlist) identically, so fusion never mixes a
  filtered leg with an unfiltered one.

- **`answer(query, k, segment=None, status=None)`**: calls `retrieve()`,
  then asks the OpenAI chat model to answer strictly from the retrieved
  passages, each labeled `[document, section, page]` in the prompt; the
  system prompt requires every claim to cite one of those labels and to
  say so explicitly rather than guess when context is insufficient.
  Returns `AnswerResult` (`answer`, `citations`, `contexts`).

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

A LangGraph supervisor routes each question to one of three agents. Does
not modify anything under `retrieval/` — the RAG agent is a thin wrapper
around `retrieval.answer`.

```
START ──► classify (OpenAI structured-output intent + logged RouteDecision)
              │
              ├─ narrative ──► rag agent      (wraps retrieval.answer, unchanged)  ──► END
              ├─ analytics ──► sql agent      (generate → validate → execute → format) ──► END
              └─ drafting  ──► drafting agent (precedent retrieval → assembly → checklist → DOCX) ──► END
```

- **Classification**: OpenAI Structured Outputs against a small
  `{intent, reasoning}` schema — always commits to exactly one of
  `narrative` / `analytics` / `drafting`, no "unsure" branch. Falls back to
  `narrative` if the classification call itself fails. Every routing
  decision is logged (`contractiq.agents.graph` logger: question, intent,
  reasoning) and returned as a structured `RouteTrace` on the response, so
  routing failures are debuggable from either logs or the response object
  directly.

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
  to drift from the number SQL actually returned.

- **Clause-type classifier** (`contractiq.extraction.classify_clause_type`): a
  controlled `ClauseType` vocabulary (termination, governing_law,
  payment_terms, confidentiality, indemnification, ...) that
  `section_title` alone doesn't provide — two contracts can title the same
  clause type differently. Keyword-matches `section_title` first (free,
  deterministic, covers the common case since real contracts mostly use
  conventional headings); falls back to an LLM classification of the
  clause text only when the title is missing or matches nothing.
  `retrieval/indexing.py` joins this into each chunk's Chroma metadata at
  index time, alongside the existing `segment`/`status` join — nothing
  else under `retrieval/` changed.

- **Drafting agent** (`contractiq.agents.drafting_agent`) — the highest-risk
  feature, built conservatively: it assembles a draft from retrieved
  precedent clauses rather than generating clause text freely.
  1. *Precedent retrieval*: for each clause type in the agreement type's
     checklist, calls `retrieval.vector_store.dense_search()` directly
     (a stage-4 function, reused as-is) with a hard `clause_type` metadata
     filter plus semantic ranking against the business brief. This is
     deliberately *not* routed through the full `retrieve()` hybrid
     pipeline — precedent lookup is "find the best exemplar of a known
     type," a different problem from open-ended retrieval, and a hard
     type filter already does the precision work.
  2. *Assembly*: clause selection is deterministic (top match within the
     type filter). Party names are placeholder-substituted deterministically
     (`ContractMetadata.vendor` → `[VENDOR NAME]`) — no LLM involved. The
     **only** LLM step is per-clause language smoothing, under a prompt
     that forbids changing any number, date, or substantive term, backed
     by a post-hoc check: numeric tokens (regex) are extracted from the
     text before and after smoothing, and any mismatch reverts to the raw
     precedent text. This isn't theoretical — real runs against GPT-4o-mini
     triggered this fallback (smoothing reworded "5 years" in a way that
     changed the extracted tokens), and the check caught it every time.
  3. *Checklist* (`contractiq.agents.checklists`): a hardcoded
     per-agreement-type (MSA / NDA / SOW, generic fallback for unrecognized
     types) list of required `ClauseType`s — a business/legal judgment
     call, not something left to the model. Missing clauses aren't
     dropped: they get an explicit placeholder under their own heading in
     the exported DOCX *and* a summary completeness section, verified
     directly on generated files (paragraph count, banner presence, style
     names).
  4. *Human-review gate*: not a workflow block — nothing here enforces
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
  it by construction), and a drafting request.

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

## Layout

```
src/contractiq/
├── ingestion/    # document loading and parsing
├── extraction/   # metadata / structured field extraction
├── retrieval/    # hybrid retrieval over contract corpus
├── agents/       # agent orchestration
├── eval/         # evaluation harness
└── ui/           # Streamlit UI
```
=======
# contract-iq



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin http://172.29.125.8:4444/contract-iq/contract-iq.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](http://172.29.125.8:4444/contract-iq/contract-iq/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
>>>>>>> 31b0b483e50327240051588438a9db7065aaf1cc
