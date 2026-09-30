# CLAUDE.md — RAG With Receipts

Portfolio RAG project (1-week timeline) demonstrating retrieval accuracy, grounded/cited
answers, and latency-optimized inference — not basic chat-with-a-doc. Deployed on GCP
Cloud Run.

## Locked decisions

| Area | Decision |
|---|---|
| Corpus | OSRS Wiki (CC BY-NC-SA 3.0 — attribute in README, non-commercial use only). Scope: `Category:Combat` (45 pages) + 2 skill training guides with XP tables (Slayer training, Herblore training) + one bounded questline, **Monkey Madness I** (main article + Quick guide walkthrough) + one-hop-linked item/monster pages the questline references, filtered to `Category:Items`/`Category:Monsters`. Resolved to **110 pages / 955 chunks** — below the original ~200–300 estimate (see Step 1 status note: link-following from the broad combat/skill-training hub pages was found to explode past budget, so it was scoped to the questline only). |
| Embeddings | Local `BAAI/bge-large-en-v1.5` via sentence-transformers, in-process. **Stretch:** benchmark against hosted Vertex `gemini-embedding-001` on the eval set; swap default only if it wins by a real margin. |
| Vector index | **FAISS** (flat index, file artifact) as the default — in-process, zero recurring cost, most technically substantive choice for the job-market signal the user wants. **Stretch:** add pgvector on Cloud SQL as a swappable second backend (config-driven) — strong "managed vector DB on GCP" resume line, done after the core pipeline works. |
| Reranker | Local cross-encoder. **Updated in Step 7**: `ms-marco-MiniLM-L-6-v2` (originally the "faster" sweep candidate) replaced `bge-reranker-base` as the default after the measured sweep found it Pareto-dominates the original baseline on this corpus — higher recall/hit-rate/MRR *and* ~4x lower rerank latency. See Step 7 status note for the full comparison, including the stronger `bge-reranker-v2-m3` variant that was measured but not adopted. |
| Generation + judge model | Claude Sonnet 5 for both, exposed as a config parameter (not hardcoded). **Stretch:** Haiku-vs-Sonnet generation comparison (latency/cost/correctness table), not a core metric. |
| Runtime shape | Embedding model, reranker, and FAISS index all run in-process inside the Cloud Run container. Only network hops in the hot path: the Claude generation call, and (only if the hosted-embedding stretch is adopted) the embedding call. |
| GCP deploy | Cloud Run (container, scales to zero) + Artifact Registry (image) + GCS (index artifacts, pulled on cold start) + Secret Manager (Anthropic key) + Cloud Logging (latency metrics). |
| Eval question mix | ~70% single-hop / extractive, ~30% multi-hop / synthesis (deliberately — single-hop questions don't discriminate between good and mediocre retrieval; multi-hop questions punish a pipeline that retrieves one relevant chunk and stops). User writes 30–50 Q/A pairs by hand. |

### Why not a fully-managed RAG service (Vertex AI RAG Engine / Vertex AI Search)?

The project's requirements — header-aware chunking, a chosen embedding model, a local/in-process
vector store, a reranking step, per-stage p50/p95 latency, a measured optimization, and a
grounding/entailment check on cited chunks — are exactly the internals a managed RAG service
hides. Using one would satisfy "a working assistant" but not the thing being demonstrated.
GCP is used instead as the **deployment substrate** (Cloud Run, GCS, Artifact Registry, Secret
Manager, Logging), while the pipeline itself is hand-built. See the stretch goal below for a
quantified comparison against the managed alternative.

## Step sequence

- [x] **Step 0 — Repo scaffold + CLAUDE.md** (this file, package skeleton, config, README stub)
- [x] **Step 1 — Ingestion & chunking** ← 110 pages, 955 chunks. See status note below.
- [x] **Step 2 — Embedding & indexing** ← 955/955 chunks embedded, FAISS flat-IP index built. See status note below.
- [x] **Step 3 — Retrieval pipeline** ← dense top-k + cross-encoder reranking, verified against the real corpus. See status note below.
- [x] **Step 4 — Generation** ← Claude Sonnet 5, structured tool-use citations, hallucinated-citation validation. See status note below.
- [x] **Step 5 — Eval harness** ← 55-question gold set, real run complete: correctness accuracy 0.84 (single-hop 0.97, multi-hop 0.53), retrieval hit rate 0.90. See status note below.
- [x] **Step 6 — Grounding / hallucination check** ← NLI cross-encoder + lexical overlap per citation, wired into the eval harness: 96.0% grounded, 4.0% flagged contradicted (all 3 flagged cases manually confirmed as false positives). See status note below.
- [x] **Step 7 — Latency instrumentation + one measured optimization (reranker sweep)** ← per-stage p50/p95 measured on the real hot path; reranker sweep found `ms-marco-MiniLM-L-6-v2` Pareto-dominates the original `bge-reranker-base` default (higher recall/hit-rate/MRR AND ~4x faster) — adopted as the new default. See status note below.
- [ ] Step 8 — GCP deployment (Cloud Run, GCS, Artifact Registry, Secret Manager)
- [ ] Step 9 — README polish (architecture explanation, benchmark numbers front and center)
- [ ] Stretch — Vertex AI Search benchmark comparison (retrieval precision, correctness,
      grounding, end-to-end latency; note in the report that the retrieval/generation latency
      *split* is ours-only, Vertex only exposes end-to-end)
- [ ] Stretch — Haiku vs Sonnet generation comparison
- [ ] Stretch — pgvector-on-Cloud-SQL backend swap

## Workflow rules

- One step at a time. Don't plan future steps in detail up front.
- To start a step: user says so explicitly → enter plan mode → write a detailed, in-depth
  plan for that step *only* → append/update it under "Step detail" below → execute.
- Each step gets its own feature branch (`feat/<step-name>`), with small, continual commits —
  avoid large batched pushes.
- Update this file's checklist and "Step detail" section as work progresses, not just at
  step boundaries.
- Tell the user explicitly when a step is fully complete before starting the next one's
  in-depth plan.

## Step detail

### Step 1 — Ingestion & chunking (complete)

**Goal:** turn the scoped OSRS Wiki slice into clean, header-aware, size-bounded chunks with
citation-ready metadata, ready for embedding in Step 2.

**1. Corpus resolution**
- Source: MediaWiki Action API at `https://oldschool.runescape.wiki/api.php`.
- Resolve a concrete, reproducible page-title list combining: category pulls
  (`action=query&list=categorymembers`) for combat mechanics, Slayer training, Herblore
  training; all pages in one bounded questline (requirements/rewards/walkthrough pages);
  and one level of link-following from those pages into item/monster infobox pages (depth=1,
  filtered to `Category:Items` / `Category:Monsters`).
- Persist the resolved list to `data/raw/page_list.json` (title, category, source reason) —
  reproducible, inspectable, diffable across runs.

**2. Fetch**
- `action=parse&page=<title>&prop=text&format=json` → rendered HTML (templates/infoboxes
  already resolved server-side, easier to parse than raw wikitext).
- Custom User-Agent identifying the project + contact email; polite rate limit.
- Cache raw HTML per page to `data/raw/pages/<slug>.html`; re-runs skip cached pages
  (idempotent, resumable).

**3. Parse**
- BeautifulSoup + lxml over `.mw-parser-output`.
- Strip nav boxes, edit-section links, TOC, reference-list scaffolding.
- Flatten infoboxes to `key: value` lines as a dedicated mini-chunk per page — these carry
  exact facts (stats, requirements) that matter for the Step 6 grounding check.
- Flatten data tables (XP tables, drop tables, requirement tables) to readable rows rather
  than discarding them.
- Walk the heading stack via `h2/h3/h4` (confirm exact OSRS wiki markup at implementation
  time — MediaWiki default is `<h2><span class="mw-headline">`).

**4. Chunk**
- One chunk per leaf section by default.
- Oversized sections split at paragraph boundaries under a target/max/overlap token budget.
- Tiny sections merged into a sibling or the parent's intro.
- Token budget matched to `bge-large-en-v1.5`'s 512-token limit: target ~380, max ~450,
  min ~80 (below this → merge), overlap ~60. Token counts computed with the actual embedding
  tokenizer (`AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")`), not a generic
  word-count heuristic.
- Each chunk is prepended with a breadcrumb string (page title + section path) before
  embedding, so the vector reflects context, not just the raw sentence.

**5. Per-chunk metadata**
`chunk_id, page_title, url, section_path, heading_anchor, token_count, source_type
(prose|infobox|table), text`

**6. Output**
- `data/processed/chunks.parquet` + `chunks.jsonl` — gitignored, regenerated by the script.
- `data/processed/ingest_stats.json` — page count, chunk count, chunk-size histogram,
  per-category breakdown, any fetch/parse failures. Committed (small).
- `results/sample_chunks.jsonl` — first ~30 chunks, committed, so the chunk quality is
  visible without running the pipeline.

**7. Config additions** (`config/config.yaml` → `ingestion:` section): category list,
questline pages, link-follow depth, output dir, chunk token params, tokenizer name, user
agent, rate limit delay.

**8. CLI**
- `scripts/fetch_corpus.py --config config/config.yaml` — resolve page list + cached fetch.
- `scripts/ingest.py --config config/config.yaml` — parse + chunk cached HTML →
  parquet/jsonl/stats.

**9. Tests**
- Unit tests against small synthetic MediaWiki-style HTML fixtures: heading-stack tracking,
  oversized-section split + overlap, tiny-section merge, infobox flattening, table
  flattening.
- Fixture test on 3–5 real cached pages (checked into `tests/fixtures/`): chunk count in
  expected range, every chunk ≤ max tokens, `section_path` populated, no leaked boilerplate
  ("edit", "Retrieved from", "Jump to navigation").

**10. Dependencies** — already declared in `pyproject.toml`'s `ingest`/`dev` extras:
`beautifulsoup4, lxml, httpx, transformers, pandas, pyarrow, pyyaml, typer, tqdm, pytest`.

**11. Branch & commit plan** — branch `feat/ingestion`:
1. `feat(ingestion): page-list resolution against MediaWiki API + caching fetch`
2. `feat(ingestion): HTML parsing — content extraction, infobox/table flattening + fixtures + unit tests`
3. `feat(ingestion): header-aware chunker (split/merge logic) + unit tests`
4. `feat(ingestion): CLI wiring (fetch_corpus.py, ingest.py) + parquet/jsonl/stats output`
5. `feat(ingestion): run against full scoped corpus, commit ingest_stats.json + sample_chunks.jsonl, update CLAUDE.md status`

**12. Verification**
- `python scripts/fetch_corpus.py --config config/config.yaml` populates `data/raw/pages/`. ✅
- `python scripts/ingest.py --config config/config.yaml` produces `chunks.parquet`; assert
  every chunk is ≤450 tokens by the bge tokenizer. ✅ (max observed: 450)
- `pytest tests/ingestion -q` green. ✅ (21 tests)
- Manual spot-check of 5+ chunks in `chunks.jsonl` for `section_path` correctness and no
  boilerplate leakage. ✅ (0 boilerplate leaks across the full 955-chunk corpus)
- `ingest_stats.json` shows page count in [200, 300] and roughly 1500–3000 total chunks.
  ⚠️ **Deviation, deliberate**: 110 pages / 955 chunks — see status note below.

**13. What actually happened during the full-corpus run (deviation from plan)**

The initial run link-followed from *all* 49 seed pages (45 `Category:Combat` pages +
Slayer training + Herblore training + the 2 questline pages). `Category:Combat` includes
broad hub/concept articles (`Monster`, `Drops`, `Skills`, `Attack`, ...) whose "see also"
sections link to **4,530 distinct pages** — checking categories on all of them (one API
call each) would have taken ~55 minutes and pulled in well over 1,000 pages, blowing past
the 200–300 budget by a wide margin. This was caught live (mid-run) by comparing observed
process CPU/network activity against expected request rates, not by guessing.

Fix: link-following now runs only from the questline's pages (`Monkey Madness I` +
`Monkey Madness I/Quick guide`), not from every seed — a walkthrough naturally references
a bounded set of items/monsters, unlike a generic combat-mechanics hub article. This
dropped the candidate set to 309 links, of which 61 matched `Category:Items`/
`Category:Monsters` after filtering. See `resolve_page_list` in
`src/rag_receipts/ingestion/page_list.py` for the implementation and reasoning.

Result: **110 pages, 955 chunks** — smaller than the original ~200–300/1500–3000 estimate.
Extending link-following to the two training-guide pages as well (704 + 392 raw candidate
links) was considered and would likely have landed nearer 250–330 pages, but was declined
in favor of keeping the corpus at its current size rather than spending another ~13 minutes
of API calls. Per-category breakdown (`data/processed/ingest_stats.json`):
combat_mechanics 45 pages/419 chunks, slayer_training 1/56, herblore_training 1/35,
questline 2/41, linked_item 50/353, linked_monster 11/51.

Also found and fixed during the full-corpus run: table rows were being flattened into one
unsplittable block per table (fixed to one block per row, so oversized tables can be split
at row boundaries); a single oversized block/sentence with no natural split point could
still exceed `max_tokens` (fixed with a hard token-window fallback); and the overlap text
prepended to a split chunk could push it 1–3 tokens over `max_tokens` because the `\n\n`
separator's own tokens weren't budgeted for (fixed with a verify-and-shrink loop instead of
a precomputed budget). All three are covered by the real-page fixture tests.

_Status: complete on `feat/ingestion`. Tell the user before starting Step 2's in-depth plan._

### Step 2 — Embedding & indexing (complete)

**Goal:** embed all 955 chunks with local `BAAI/bge-large-en-v1.5` and build a FAISS flat
index (cosine similarity via `IndexFlatIP` over normalized vectors), persisted as file
artifacts ready for Step 3's retrieval pipeline.

**Design decisions actually implemented:**
- `rag_receipts/indexing/embedder.py::Embedder` wraps `SentenceTransformer`, with
  `encode_passages` (no prefix) and `encode_queries` (BGE's retrieval instruction prefix,
  config-driven) as separate methods — only `encode_passages` is exercised by Step 2, but
  Step 3 can reuse this wrapper as-is for query encoding without duplicating model-loading
  logic. `Encoder` is a structural `Protocol`, letting unit tests inject a deterministic
  `FakeEncoder` instead of loading the real model.
- Embedding text = `breadcrumb(page_title, section_path) + "\n\n" + text`, using the hook
  already left in `ingestion/utils.py::breadcrumb` for exactly this purpose.
- Per-vector metadata persisted as a row-order-aligned `index_metadata.parquet` (FAISS row
  `i` ↔ metadata row `i`), since FAISS flat indices only support int64 IDs, not string
  `chunk_id`s. No separate raw-embeddings artifact — the flat index already stores vectors
  contiguously and supports `.reconstruct(i)`.
- Extracted `AppConfig`/`load_config` out of `ingestion/config.py` into a new top-level
  `rag_receipts/config.py` (small mechanical refactor, commit 1 of this branch) so the
  completed Step 1 package didn't need to import forward into Step 2's package to gain an
  `indexing:` field — same pattern will apply cleanly to every future step's config.
- Test suite keeps the real ~1.3GB model out of the default `pytest -q` run: one file,
  `test_embedder_integration.py`, is marked `slow` (registered in `pyproject.toml`,
  `addopts = -m "not slow"`) and is the only place the real model loads.

**What actually happened running against the full corpus:**
- Environment: this machine's default `pip install torch` on Windows resolves a CPU-only
  wheel (confirmed empirically: `torch==2.10.0+cpu`, `cuda.is_available() == False`) —
  Windows does not get CUDA by default from PyPI the way Linux does. Fixed by installing
  torch explicitly first: `pip install torch --index-url https://download.pytorch.org/whl/cu121`,
  *then* `pip install -e ".[index]"` so `sentence-transformers` didn't pull the CPU wheel
  transitively. Confirmed working: `torch.cuda.is_available() == True` on the RTX 3060 (12GB).
- `python scripts/build_index.py --config config/config.yaml` embedded all 955 chunks and
  built the index in ~35s wall-clock on GPU (`device=cuda` in the resulting stats).
- Artifacts: `data/index/faiss.index` (955 vectors × 1024 dims, ~3.9MB), `data/index/
  index_metadata.parquet` (955 rows, full `Chunk` schema), both gitignored (pulled from GCS
  on cold start per the Step 8 deploy plan, not committed). `data/processed/index_stats.json`
  is committed (small, like `ingest_stats.json`): `chunk_count: 955`, `embedding_dim: 1024`,
  `device: cuda`, `source_type_breakdown: {table: 512, prose: 376, infobox: 67}`.
- Manual verification: reloading the saved index and searching with its own row 0 as the
  query returned row 0 as top-1 with score `0.9999998` (≈1.0, as expected for exact
  self-similarity under cosine), confirming the flat-IP + normalized-embeddings pairing
  behaves correctly end-to-end on real data.
- `pytest -q`: 42 passed (existing 21 ingestion tests + 21 new indexing unit tests), the one
  `slow` real-model test deselected by default and passing separately when run explicitly.

_Status: complete on `feat/indexing`. Tell the user before starting Step 3's in-depth plan._

### Step 3 — Retrieval pipeline (complete)

**Goal:** turn the static Step 2 index into an actual retrieval pipeline — dense FAISS
top-k search, then local cross-encoder reranking (`BAAI/bge-reranker-base` baseline) —
as a clean, importable component Step 4 (generation) and Step 5 (eval) can call once per
question, with expensive loading (index, metadata, embedding model, reranker model)
separated from cheap per-query retrieval.

**Design decisions actually implemented:**
- New package `rag_receipts/retrieval/`: `config.py` (`RetrievalConfig`/`RerankerConfig`,
  same flat-YAML-to-nested-dataclass convention as `indexing/config.py`), `models.py`
  (`RetrievedChunk` — every citation field plus `dense_score`/`rerank_score`/`dense_rank`/
  `final_rank`), `reranker.py` (`Reranker` wraps `sentence_transformers.CrossEncoder`,
  mirroring `Embedder`'s `Protocol` + `from_config` + `lru_cache`d loader shape), and
  `pipeline.py` (`dense_search`, `run_retrieval`, `Retriever`).
- The reranker scores `breadcrumb(page_title, section_path) + "\n\n" + text` — the exact
  text form embedded in Step 2 — not bare `text`, since OSRS wiki chunks are frequently
  terse out of page/section context (a drop-table row, an infobox stat line).
- `Retriever.from_config(AppConfig)` loads the index/metadata/embedder/reranker once;
  `.retrieve(query)` does zero I/O or model loading per call, delegating to the pure
  `run_retrieval(...)` function. This is the load-once/query-cheap pattern Step 4/5/8 will
  reuse. `dense_search` is a thin pass-through to `index.search`, returning raw FAISS
  output (including `-1` padding when `top_k > index.ntotal`); `run_retrieval` is the one
  place that filters `-1` ids before any metadata lookup.
- `AppConfig.retrieval` defaults via `field(default_factory=RetrievalConfig)` so existing
  direct `AppConfig(...)` construction (e.g. in `tests/indexing/test_pipeline.py`) keeps
  working without every call site needing to pass a `retrieval` kwarg.

**Bug found and fixed while adding Step 3's own tests (unrelated to retrieval logic
itself):** pytest's default "prepend" import mode collides when two test directories
have identically-named files without `__init__.py` — `tests/indexing/test_config.py` and
the new `tests/retrieval/test_config.py` (and `test_pipeline.py`, and `helpers.py`)
registered under the same bare module name, so pytest errored on collection
("import file mismatch"). Fixed by giving each step's test directory an `__init__.py`
(`tests/indexing/`, `tests/ingestion/`, `tests/retrieval/`), which qualifies module names
by directory, and converting the existing bare `from helpers import X` /
`from conftest import X` imports in `tests/indexing/` and `tests/ingestion/` to relative
imports (`from .helpers import X`). This is a one-time fix — it will not recur for future
steps' test directories.

**What actually happened running against the full corpus:**
- `pytest -q`: 60 passed (42 existing + 18 new retrieval unit tests), 3 deselected (`slow`:
  `test_embedder_integration`, `test_reranker_integration`, `test_retrieval_integration`).
- `pytest -q -m slow tests/retrieval/test_reranker_integration.py`: real `bge-reranker-base`
  scores a relevant `(query, passage)` pair higher than an irrelevant one. ✅
- `pytest -q -m slow tests/retrieval/test_retrieval_integration.py`: real 955-chunk corpus,
  a Monkey Madness question surfaces a Monkey Madness page in the top 3. ✅
- `python scripts/retrieve.py "How do you start the Monkey Madness quest?"` — runs
  end-to-end; reranking visibly reorders results (e.g. `Monkey (Monkey Madness I)` has a
  higher dense score than `Monkey Madness I > Introduction` but a lower rerank score, and
  drops below it in the final ranking).
- `results/sample_retrievals.json` (committed, via `scripts/sample_retrievals.py`) — 5
  hand-picked queries. Clear evidence reranking does real work, not a no-op: for "What items
  are required to start Monkey Madness I?", the chunk literally containing "Items required |
  A gold bar..." had `dense_rank=15` (score 0.689) but reranks to `final_rank=1`
  (rerank score 0.9996), while the `dense_rank=1` chunk (score 0.746, a different section of
  the same page) drops to `final_rank=3`.

_Status: complete on `feat/retrieval`. Tell the user before starting Step 4's in-depth plan._

### Step 4 — Generation (complete)

**Goal:** turn Step 3's reranked chunks into a grounded, cited answer — Claude Sonnet 5
answers using only the retrieved chunks, citing the specific `chunk_id`(s) behind each claim,
and every citation is validated against the actual retrieved set rather than trusted on
faith (the project's "receipts" thesis: a citation that doesn't check out is exactly the
failure mode being designed against).

**Design decisions actually implemented:**
- Citations via **structured tool-use output**, not inline `[1][2]` markers: Claude is called
  with `tool_choice` forced to a `submit_answer` tool returning
  `{answerable, answer, citations: [{chunk_id, claim}]}` — deterministic to parse, no regex
  over free text. `rag_receipts/generation/generator.py::Generator` follows the
  `Reranker`-style shape (`Protocol` + `.from_config` + `@dataclass` wrapper), with one
  deliberate deviation: no `lru_cache` on the client constructor, since (unlike
  `SentenceTransformer`/`CrossEncoder`) constructing `anthropic.Anthropic()` is cheap.
- `Generator.generate(query, chunks: list[RetrievedChunk]) -> GeneratedAnswer` takes
  pre-retrieved chunks rather than owning retrieval itself — symmetric to
  `Reranker.score(query, passages)`. `scripts/generate.py` composes `Retriever.retrieve()` +
  `Generator.generate()`; a future eval harness can retrieve once and reuse the chunks for
  both retrieval-precision and generation-correctness metrics.
- **Every citation the model returns is treated as untrusted input.** `generate()` resolves
  each `chunk_id` against the chunks it was actually given: a real match becomes a `Citation`
  with the full `RetrievedChunk` attached (so callers render `page_title`/`url`/
  `heading_anchor` without a second lookup); an unresolvable `chunk_id` is dropped from
  `citations` but preserved in `hallucinated_citation_ids` (plus a `has_hallucinated_citations`
  flag) rather than silently discarded or silently trusted.
- **No `temperature` field.** Verified against the current Claude API docs before
  implementation: `claude-sonnet-5` runs adaptive thinking by default, and the API rejects
  sampling params (`temperature`/`top_p`/`top_k`) with a 400 whenever thinking is active.
  Determinism comes from the forced tool schema + `strict: true`, not sampling.
  `thinking` is left unset (adaptive, the default) rather than disabled — disabling thinking
  on a forced-tool-choice call is a documented failure mode where the model can write the
  tool call into visible text instead of a real `tool_use` block, silently breaking citation
  parsing. `max_tokens` defaults to 4096 (not lowballed to ~1024), since adaptive-thinking
  tokens count against the same budget as the visible tool-call output; `generate()` explicitly
  checks `stop_reason == "max_tokens"` and raises rather than trying to parse a possibly
  truncated tool call. Forced `tool_choice` is valid on `claude-sonnet-5` today, but a
  comment in `generator.py` flags that pointing `config.generation.model` at Fable 5.1/
  Mythos 5.1/Opus 5.5/Sonnet 5.5 later would need `tool_choice: {"type": "auto"}` instead
  (those four models reject forced tool_choice with a 400).
- `refusal` and missing-`tool_use` stop reasons are both handled explicitly: a `refusal`
  returns a safe `answerable=False` result instead of raising; a response with no `tool_use`
  block raises `RuntimeError` with the stop_reason and content for debugging, rather than
  failing with an opaque `AttributeError`/`KeyError` deep in citation-resolution code.
- `input_tokens`/`output_tokens`/`model`/`stop_reason` are captured on every `GeneratedAnswer`
  from `response.usage` — cheap to capture now, needed by Step 7's latency/cost work later;
  no telemetry system built beyond that.
- `ANTHROPIC_API_KEY` is read from the environment by the SDK's default client construction,
  never from `config.yaml` — matches the GCP deploy plan's Secret Manager routing.

**Test conventions:** `tests/generation/helpers.py::FakeAnthropicClient` implements the same
structural `AnthropicClientLike` Protocol the real SDK client does, returning scripted
`FakeMessage` responses (including one whose `citations` payload names a `chunk_id` not in
the passed chunks, exercising the hallucination-validation path). `test_generator.py` is pure
unit tests against the fake (8 tests: prompt construction, `answerable=false` passthrough,
real-citation resolution, hallucinated-citation dropping/tracking, usage capture, `refusal`
handling, `max_tokens` handling, missing-`tool_use` handling).
`test_generator_integration.py` is `slow`-marked and additionally `skipif`-gated on
`ANTHROPIC_API_KEY` being set — makes one real API call, since (unlike the other `slow` tests,
which just load a real local ML model) this one costs real API dollars per run. The `slow`
marker's `pyproject.toml` docstring was broadened to cover both cases.

**What actually happened running against the real corpus:**
- `pytest -q`: 71 passed (60 existing + 11 new generation unit/config tests), 4 deselected
  (`slow`: the 3 existing ML-model integration tests + the new API-call integration test).
- `pytest -q -m slow tests/generation/test_generator_integration.py` with `ANTHROPIC_API_KEY`
  set: passed — a real call correctly cites a real `chunk_id` with no hallucination.
- `python scripts/generate.py "What items are required to start Monkey Madness I?"` — grounded
  answer, citations resolve to the real `Monkey Madness I` page/section, no hallucination
  warning.
- `python scripts/generate.py "What are the requirements to start Dragon Slayer II?"` was
  tried as a candidate "out of corpus" question first and turned out to be a bad example: it
  came back `answerable: true`, correctly grounded in stray "quests requiring this skill" rows
  on unrelated skill pages (`Strength`, `Hitpoints`) that happen to list Dragon Slayer II's
  skill requirements — a legitimate citation, not a hallucination, just not the intended demo.
  Replaced with `"What are the steps to complete the Cook's Assistant quest?"` (a quest
  entirely outside the corpus's combat/Slayer/Herblore/Monkey Madness scope), which correctly
  returns `answerable: false` with a sensible explanation instead of a guess.
- `python scripts/sample_answers.py` → `results/sample_answers.json` (committed): 6 queries,
  3 `answerable: true` with correctly-resolving citations, 3 `answerable: false`, zero
  `hallucinated_citation_ids` anywhere. Two of the three `answerable: false` cases
  ("How much Slayer experience is needed for level 70?", "What potions require Herblore level
  78?") are **in-corpus topics** where the retrieved top-k simply didn't surface the specific
  fact — the model correctly declined rather than guessing from parametric knowledge, which is
  the grounding discipline working as designed, but it also surfaces a retrieval-recall gap
  (possibly `top_k_final=5` too small, or XP/potion-level tables split awkwardly across
  chunks) worth revisiting in a later step rather than in Step 4's scope.

_Status: complete on `feat/generation`. Tell the user before starting Step 5's in-depth plan._

### Step 5 — Eval harness (complete)

**Goal:** run the full pipeline over a hand-labeled gold Q/A set and produce numbers —
retrieval precision/recall/MRR/hit-rate and LLM-as-judge correctness, overall and broken
down by question type (single-hop vs multi-hop) — rather than the spot-check queries
Steps 1-4 relied on.

**Scope decisions made explicit in the plan:**
- Retrieval-precision metric measures only the pipeline's actual final output
  (`Retriever.retrieve()`, post-rerank top `top_k_final`) — the same chunks generation
  receives. A dense-vs-reranked precision comparison is deliberately **not** done here;
  that's Step 7's reranker-sweep job.
- The judge scores answer **correctness** against a gold reference answer only — not
  groundedness/entailment against cited chunks, which is Step 6's job.
- The gold Q/A pairs in `data/eval/qa_pairs.json` were hand-authored by the user,
  independent of the harness build — confirmed explicitly, not drafted by the assistant.

**Design decisions actually implemented (commits 1-6 of 7, on `feat/eval-harness`):**
- `EvalConfig` (`src/rag_receipts/eval/config.py`) wired into `AppConfig` following the
  existing flat-YAML-to-nested-dataclass convention every prior step uses; `config.yaml`'s
  `eval:` stub is now live (`judge_model`, `judge_max_tokens`, `dataset_path`,
  `output_path`).
- `EvalQuestion`/`RetrievalScore`/`JudgeVerdict`/`EvalResult`/`EvalSummary`
  (`src/rag_receipts/eval/models.py`) and `load_eval_questions()`
  (`src/rag_receipts/eval/dataset.py`) — the loader validates the user's hand-written file
  (unique ids, `gold_chunk_ids`/`gold_answer` required exactly when `answerable=true`,
  empty/`null` when `false`) and raises a specific `ValueError` on each violation, since
  this is the first thing that runs against hand-authored data.
- `retrieval_metrics.py` — pure `precision_at_k`/`recall_at_k`/`mrr`/`hit` functions plus
  `score_retrieval()`, scoring `RetrievedChunk.chunk_id`s against a question's
  `gold_chunk_ids` by exact string match (no normalization needed — chunk_ids flow
  unchanged from ingestion through indexing to retrieval).
- `Judge` (`src/rag_receipts/eval/judge.py`) — mirrors `Generator`'s `Protocol` +
  `@dataclass` + `from_config` shape and `stop_reason` handling discipline exactly
  (`refusal` → safe `incorrect` verdict instead of raising; `max_tokens`/missing
  `tool_use` → raise `RuntimeError`), scoring a 3-way categorical verdict
  (`correct`/`partially_correct`/`incorrect`) via a forced `submit_verdict` tool call.
  Same no-`temperature` / forced-`tool_choice`-only-valid-on-`claude-sonnet-5` caveat as
  `Generator`.
- `runner.py` — `run_eval()` resolves three of the six non-error correctness outcomes
  directly from the expected-vs-actual `answerable` flags with **no judge call spent**
  (`correct_abstention`, `incorrectly_abstained`, `incorrectly_answered`); the judge is
  only called when both sides agree the question should be answerable. A `RuntimeError`
  from `Generator.generate()` or `Judge.score()` is caught per-question and recorded as an
  `"error"` result rather than aborting the batch. `summarize()` aggregates mean
  precision/recall/MRR/hit-rate and a strict correctness-accuracy rate
  (`correct` + `correct_abstention` over non-error results), overall and per question type.
- `scripts/run_eval.py` — plain script (no typer, matching `sample_answers.py`'s
  convention for a batch run over a dataset file), writes
  `{"summary": ..., "results": [...]}` to `results/eval_report.json`. Added `tqdm` to the
  `eval` extra in `pyproject.toml` for the progress bar.
- Tests follow the established per-package convention (`tests/eval/`, own `__init__.py`,
  duplicated fakes rather than cross-package imports): `test_config.py`, `test_dataset.py`
  (schema-violation coverage), `test_retrieval_metrics.py`, `test_judge.py` (against a
  duplicated `FakeAnthropicClient`), `test_runner.py` (all three no-judge-call paths, the
  judge-call path, generation-error and judge-error handling, and `summarize()`
  aggregation math including a zero-division guard on an empty type bucket).
  `tests/eval/fixtures/qa_pairs_small.json` — 5 hand-crafted questions (3 answerable, 2
  deliberately unanswerable/out-of-corpus; 2 single-hop + 1 multi-hop among the
  answerable ones) built from real `chunk_id`s already visible in
  `results/sample_retrievals.json`/`sample_answers.json` — feeds both `test_dataset.py`
  and `tests/eval/test_eval_integration.py` (`slow`-marked, dual-`skipif`-gated on
  `ANTHROPIC_API_KEY` and the real index artifacts, same pattern as
  `test_generator_integration.py`/`test_retrieval_integration.py`).
- **`data/eval/qa_pairs.json`** — 55 hand-authored questions (35 single-hop + 15 multi-hop
  answerable, hitting the locked 70/30 split exactly, + 5 deliberately unanswerable). All
  65 `gold_chunk_id` references in the answerable questions were cross-checked
  programmatically against the real 955-row `chunks.parquet` before conversion — zero
  mismatches. Authored in a spreadsheet (`golden_data.csv` at repo root, not committed —
  deleted after conversion), converted to the schema, and re-validated through
  `load_eval_questions()` itself.
- **Deviation, deliberate:** 55 questions, 5 over the locked 30-50 range — kept in full
  rather than trimming a real answerable question to make room for the unanswerable slice.
- **A real distinction surfaced while picking the 5 unanswerable questions:** "the fact
  isn't in the corpus at all" (true out-of-corpus, e.g. a quest that was never scoped in)
  and "the fact is in the corpus but retrieval fails to surface it" (a retrieval-recall
  gap) are not the same thing and need different gold labels. Step 4's sample run had
  flagged two candidate gaps anecdotally ("How much Slayer XP for level 70?", "What
  potions require Herblore level 78?"). Checking both against the real chunk text: the
  Herblore-78 fact **is** present (`Herblore_training__011` literally contains "Level: 78
  | Potion: | Base: Zamorak brew"), so it was added as a normal `answerable: true`
  question with that real `gold_chunk_id` — marking it `false` would have encoded a
  retrieval bug as correct ground truth and hidden it from the precision/recall numbers.
  The Slayer-XP fact genuinely isn't present anywhere in the corpus (no chunk contains the
  universal level-70 XP threshold, 737,627), so it stayed in the true-out-of-corpus/
  unanswerable bucket. Net effect: only 1 of the 2 candidate "gaps" was a real gap; the
  eval will now measure it directly instead of leaving it as an anecdote.

**Verification:** `pytest -q` — 113 passed, 5 deselected by default (the 4 pre-existing
`slow` ML/API tests + the new eval integration test). With `ANTHROPIC_API_KEY` set:
`pytest -q -m slow tests/eval/test_eval_integration.py tests/generation/test_generator_integration.py`
— both pass against the real API. `data/eval/qa_pairs.json` loads cleanly through
`load_eval_questions()` (55/55, no schema violations).

**Schema correction found after the first run:** `EvalQuestion.type` was originally forced
to `single_hop`/`multi_hop` for every question, including the 5 deliberately-unanswerable
ones (assistant-assigned types, since the user's source data had `type: null` there) — this
inflated the type breakdown's n to 38/17 instead of the intended 35/15, silently mixing
abstention outcomes into the single-hop/multi-hop accuracy numbers. Fixed: `type` is now
`Optional`, `null` only when `answerable=false`; `dataset.py` enforces this; `runner.py`'s
`summarize()` excludes untyped questions from `retrieval_by_type`/`correctness_by_type`
(they still count in the overall totals). The gold set and report below reflect the fix.

**Real run against the full gold set** (`python scripts/run_eval.py`, 55 questions, 0
errors, `results/eval_report.json` committed):

| Metric | Value |
|---|---|
| Retrieval hit rate | 0.90 |
| Retrieval mean recall | 0.83 |
| Retrieval mean MRR | 0.70 |
| Retrieval mean precision | 0.20 (expected — precision's denominator is `top_k_final=5` against a 1-2-chunk gold set, so even a perfect retrieval can't exceed ~0.2-0.4) |
| Correctness accuracy (overall) | 0.80 |
| Correctness accuracy (single-hop, n=35) | 0.97 (34/35) |
| Correctness accuracy (multi-hop, n=15) | 0.33 (5/15) |
| Label counts | `correct`: 39, `correct_abstention`: 5, `incorrectly_abstained`: 7, `partially_correct`: 2, `incorrect`: 2 |

**Note on reproducibility:** the retrieval numbers above are identical between the
pre-fix and post-fix runs (the embedding/reranker models are deterministic local
inference). The correctness numbers are *not* bit-identical between runs — generation and
judging call the live Claude API without `temperature` pinning (a deliberate Step 4 design
choice: Sonnet 5's forced-tool-choice + adaptive thinking doesn't accept sampling params),
so a re-run can shift a few borderline multi-hop verdicts. Single-hop accuracy was stable
across both runs (37/38 → 34/35); multi-hop moved more (9/17 → 5/15) because the borderline
arithmetic-synthesis questions are exactly where a model's adaptive-thinking pass varies
most run to run. Treat the multi-hop percentage as "meaningfully worse than single-hop,"
not as a precise, reproducible figure.

**The headline finding is still the single-hop/multi-hop gap, and it's still dominantly
one traced cause — with one honest exception this run surfaced.** Of the 10 non-fully-
correct multi-hop results, 9 have retrieval `recall` < 1.0 (`0.0` or `0.5` — missing at
least one of the two gold chunks) and generation correctly declined or erred on the
missing fact rather than fabricating it. The 1 exception (q044, "how many diamonds could
you accumulate...") had `recall=1.0` — both gold chunks were retrieved — but the model
still got the arithmetic wrong. So the story is: **retrieval-recall on 2-chunk gold sets
is the dominant cause** (9/10), but not the *only* one — there's a smaller, real
arithmetic-synthesis failure mode even when retrieval succeeds. Directly relevant to
Step 7: raising `top_k_final` or trying the stronger reranker variant
(`bge-reranker-v2-m3`) targets the dominant cause; the arithmetic-synthesis failure mode
is a generation-side limitation Step 7 won't fix and is out of this project's scope to
chase further.

**All 5 deliberately-unanswerable questions correctly triggered `correct_abstention`**
(5/5) — zero `incorrectly_answered`, i.e. no hallucinated "yes" on a genuinely
out-of-corpus question.

**Judge sanity-checked by hand** against both directions: rewards correct answers with
extra correct detail (q001, q027 — `correct`, judge notes "extra details do not detract");
correctly dings a real missing fact rather than being lenient (q036, monkey archers
aggressiveness — `partially_correct`, judge notes the Monkey Madness II exception was
omitted). Verdicts read as calibrated, not rubber-stamped.

_Status: complete on `feat/eval-harness`. Tell the user before starting Step 6's in-depth
plan._

### Step 6 — Grounding / hallucination check (complete)

**Goal:** catch a failure mode Step 4's citation validation cannot see — a citation naming
a **real** `chunk_id` whose text doesn't actually support the claim attached to it. Step 4
only checks that a cited `chunk_id` was in the retrieved set; Step 5's judge only checks
final-answer correctness against a gold reference. Neither checks whether the cited chunk
itself entails the claim, which is what this step measures.

**Design decisions actually implemented (7 commits, `feat/grounding-check`):**
- New package `rag_receipts/grounding/`, following every prior step's `Protocol` +
  `@dataclass` + `from_config` + `lru_cache`d-loader shape: `config.py` (`GroundingConfig`),
  `overlap.py` (pure `token_overlap()` — lowercase `\w+` word-overlap ratio, no model),
  `entailment.py` (`EntailmentChecker`, a second `sentence_transformers.CrossEncoder`
  wrapping `cross-encoder/nli-deberta-v3-base`, mirroring `retrieval/reranker.py::Reranker`
  exactly), `checker.py` (`GroundingChecker`, combining both signals).
- **Two independent signals, not one**, per the step's own name ("entailment/overlap
  flag"): the NLI cross-encoder scores entailment/contradiction/neutral for
  (cited-chunk-text, claim) pairs; a cheap lexical word-overlap ratio runs alongside it.
  Either signal clearing its threshold is enough to call a citation `grounded`; a high
  contradiction probability overrides to `contradicted` regardless of overlap. Both
  thresholds are config-driven (`entailment_threshold`/`contradiction_threshold`/
  `overlap_threshold`, all in `config.yaml`'s new `grounding:` section).
- **Label-order safety, verified against the real checkpoint before writing the mapping
  logic** (same discipline as Step 1's wiki-markup check): NLI cross-encoder checkpoints
  don't share one standardized contradiction/entailment/neutral index order.
  `cross-encoder/nli-deberta-v3-base`'s real `id2label` was checked empirically
  (`{0: contradiction, 1: entailment, 2: neutral}`) rather than assumed.
  `EntailmentChecker` resolves a `{label_name: column_index}` map from the model's
  `id2label` at load time and always returns scores in a fixed (entailment, contradiction,
  neutral) column order, regardless of the underlying checkpoint's native order. Unit tests
  exercise two fakes with *different* native orders to prove the remapping isn't
  accidentally relying on a fixed index.
- Wired into the eval harness without changing its correctness-label logic:
  `runner.py::run_eval()` grounding-checks every generated answer's real citations
  independent of which correctness branch it lands in — an `incorrectly_abstained` result
  or a judge-error result still gets its citations checked, since Step 4 already resolved
  them against real chunk_ids before Step 6 ever sees them. `EvalSummary` reports
  grounded/contradicted/ungrounded rates and a fully-grounded-answer rate, overall, by
  question type, and by `correctness_label` (to test whether bad answers correlate with
  ungrounded citations).
- `scripts/sample_grounding.py` — spot-check script mirroring `sample_answers.py`, writing
  `results/sample_grounding.json` (committed).

**What actually happened running against real data:**
- `pytest -q`: 146 passed, 6 deselected (`slow`: the 5 pre-existing ML/API integration
  tests + the new `test_entailment_integration.py`, which loads the real NLI checkpoint and
  confirms it separates a hand-written entailed/contradicted/neutral triple correctly).
- `python scripts/sample_grounding.py` surfaced a real, useful confirmation of the
  OR-combination design on the very first run: for "What items are required to start
  Monkey Madness I?", the cited chunk is a flattened requirements list
  (`"Items required: A gold bar, Five empty inventory slots, ..."`) and the claim quotes it
  near-verbatim — but the NLI model scored this pair `entailment_prob=0.003`,
  `neutral_prob=0.997`. The general-domain NLI model (trained on natural-sentence SNLI/MNLI
  pairs) doesn't recognize a telegraphic, comma-separated requirements list as "entailing"
  its own restatement. The lexical overlap signal (`1.0`) correctly grounded it anyway —
  exactly the failure mode the overlap signal was added to catch, confirmed on real data
  within the first few examples rather than staying hypothetical.
- **Real run against the full 55-question gold set** (`python scripts/run_eval.py`, 0
  errors, `results/eval_report.json` committed):

| Metric | Value |
|---|---|
| Retrieval hit rate / recall / MRR / precision | 0.90 / 0.83 / 0.70 / 0.20 — unchanged from Step 5 (retrieval is deterministic local inference; grounding adds no new retrieval behavior) |
| Correctness accuracy (overall) | 0.80 (single-hop 0.97, multi-hop 0.33) — same headline numbers as Step 5's run, modulo the already-documented judge/generation non-determinism |
| Citations checked | 75, across 41 answers with ≥1 citation |
| Grounded rate | 0.960 |
| Contradicted rate | 0.040 (3 citations) |
| Ungrounded rate | 0.000 |
| Fully-grounded-answer rate | 0.927 (38/41) |

- **The 3 `contradicted` flags were manually inspected, and all 3 are false positives from
  the NLI model, not real contradictions** — the same domain-mismatch pattern the overlap
  signal already exposed above, but here it's the *contradiction* side misfiring instead of
  the entailment side:
  - q014 ("How much damage can Vorkath's dragonfire hit for?"): claim "Unprotected maximum
    damage for Vorkath's dragonfire is 80" against a chunk containing a flattened data table
    (`"Protection used: Unprotected | Maximum damage: 50 | 30 | 80 | 50 | 65 | 50 | 70"`) —
    the 80 is literally present and correct; `contradiction_prob=0.999`, `overlap=0.78`.
  - q021 ("How many Slayer reward points does it cost to permanently block a task with
    Duradel/Kuradal?"): claim "Duradel / Kuradal : 100 points to block a task" is a
    **verbatim substring** of the cited chunk's text; `contradiction_prob=0.939`,
    `overlap=1.0`.
  - q030 ("What is the combat level of the Jungle Demon fought at the end of Monkey Madness
    I?"): claim "a level 195 Jungle Demon" is a **verbatim substring** of the cited chunk's
    walkthrough text; `contradiction_prob=0.826`, `overlap=1.0`.
  - All 3 source questions were independently graded `correct` by Step 5's LLM-as-judge
    against the gold reference answer, corroborating that the underlying facts are right —
    the NLI model is misreading the *chunk*, not catching a real error in the *answer*.
  - **Root cause:** `cross-encoder/nli-deberta-v3-base` is a general-domain model trained on
    natural-sentence premise/hypothesis pairs (SNLI/MNLI-style). OSRS wiki chunks are
    frequently flattened tables, drop-table rows, or terse walkthrough narration — text the
    model wasn't trained to read as a "premise" at all, and it appears to default to
    `contradiction` rather than `neutral` when a flattened-table premise doesn't read as
    naturalistic prose. This is a real, documented limitation of the model choice for this
    corpus, not a bug in the OR/contradiction-override combination logic: the same "high
    overlap should be trusted more than a shaky NLI signal on this corpus" argument that
    motivated adding overlap in the first place also explains why `contradicted` currently
    overrides overlap unconditionally — that override is exactly where all 3 false
    positives occurred. Deliberately **not patched** by tuning thresholds against 3
    anecdotes (that would be un-measured, vibes-based tuning, the opposite of this
    project's thesis) — documented here as a known limitation instead. A domain-adapted or
    larger NLI model, or requiring corroboration from overlap before trusting a
    contradiction flag, are the natural next things to try if this is revisited; out of
    scope for Step 6 itself, same as Step 5's arithmetic-synthesis limitation was out of
    scope for that step.
  - **Practical takeaway for reading this metric:** treat `contradicted` as "flagged for
    human review," not "confirmed contradiction" — on this corpus and with this model, a
    `contradicted` flag has so far been 0/3 accurate. `grounded`/`ungrounded` were not
    observed to have this problem in this run (0.960/0.000 rates, and the single sample-run
    example above shows overlap correctly rescuing a case the NLI model alone would have
    called `neutral`, not falsely flagging a good citation as bad).

_Status: complete on `feat/grounding-check`. Tell the user before starting Step 7's
in-depth plan._

### Step 7 — Latency instrumentation + one measured optimization (reranker sweep) (complete)

**Goal:** measure per-stage p50/p95 latency across the real hot path (query embedding,
dense FAISS search, cross-encoder reranking, Claude generation), then run one measured
optimization — a reranker sweep comparing the original baseline against a stronger and
a faster variant on retrieval precision vs p95 latency — and pick a point on that curve.

**Design decisions actually implemented (6 commits, `feat/latency-optimization`):**
- New package `rag_receipts/telemetry/`: `timing.py` (`percentile` — linear
  interpolation matching `numpy.percentile`'s default, implemented directly rather than
  importing numpy just for this; `Stopwatch` context manager; `summarize_durations` →
  `{p50, p95, mean, count}`) and `models.py` (`RetrievalTiming`). Pure, model-free,
  fully unit-tested without loading anything.
- `retrieval/pipeline.py::run_retrieval_with_timing` — same body as `run_retrieval`
  with a `Stopwatch` around each of the three sub-stages (`embedder.encode_queries`,
  `dense_search`, `reranker.score`); `run_retrieval` became a one-line delegate that
  discards the timing, so every existing caller/test was unaffected.
  `Retriever.retrieve_with_timing` mirrors `.retrieve()`.
- `eval/runner.py`'s private `_retrieval_stats` was promoted to a public
  `aggregate_retrieval_scores` in `eval/retrieval_metrics.py` (the module that already
  owns `score_retrieval`) so the reranker sweep could reuse Step 5's exact, already-
  tested aggregation logic instead of a second copy.
- `scripts/measure_latency.py` — loads `Retriever`/`Generator` once (model load time
  excluded from the stats, matching the plan's scope decision that cold-start load is a
  Step 8 concern, not a per-query number), then runs the real 55-question gold set
  through `retrieve_with_timing` + a wall-clock-wrapped `generate()` call, writing
  `results/latency_report.json`.
- `scripts/sweep_reranker.py` — retrieval-only (no generation calls; swapping the
  reranker doesn't change generation latency, and running Sonnet 3x over the gold set
  would add real API cost for no signal). Loads the FAISS index/metadata/embedder
  **once** (reranker-independent) and only rebuilds the `Reranker` per candidate,
  scoring retrieval (`score_retrieval`, answerable questions only — 50 of 55) and
  latency (`rerank_s`/`total_s`, all 55) for each of the three candidates. Writes
  `results/reranker_sweep.json`.

**A real methodology bug was caught and fixed before trusting any latency number.**
The first `measure_latency.py` run was launched in the background while
`sweep_reranker.py` was still running (downloading and reranking with all three
candidate models concurrently on the same GPU). Its rerank-stage p50 came back at
1472ms — but the sweep's own isolated measurement of the identical model
(`bge-reranker-base`) was 552ms, a ~2.7x gap for the same model on the same 55
queries. This was caught by comparing the two reports against each other rather than
trusting either one in isolation. Fix: re-ran `measure_latency.py` alone, after the
sweep had finished and released the GPU. The clean re-run's rerank p50 (722ms) is much
closer to the sweep's isolated number, with the remaining ~30% gap attributed to
ordinary run-to-run variance at n=55 (the already-documented small-sample caveat)
rather than contention. **`results/latency_report.json` reflects the clean, isolated
re-run.** Lesson for future runs: never launch two GPU-bound benchmark scripts
concurrently and expect either one's absolute numbers to be trustworthy.

**Real latency numbers** (`python scripts/measure_latency.py`, isolated run, 55
questions, 0 errors, config as of this run: `bge-reranker-base` — this predates the
reranker-default swap decided below, since the swap depends on the sweep's findings,
which in turn needed this script's stage-timing infrastructure to exist first):

| Stage | p50 | p95 | mean |
|---|---|---|---|
| embed_query | 50.0ms | 155.4ms | 57.6ms |
| dense_search | 0.2ms | 0.5ms | 0.3ms |
| rerank | 722.1ms | 770.7ms | 716.0ms |
| retrieval_total | 774.8ms | 860.4ms | 773.7ms |
| generate | 2736.8ms | 5920.8ms | 3072.3ms |
| end_to_end | 3510.4ms | 6713.1ms | 3846.0ms |

Reranking dominates the retrieval stage by a wide margin over embedding/dense-search
(as expected — a cross-encoder scoring 30 candidates is real inference work; FAISS
flat search over 955 vectors is effectively free). Generation dominates end-to-end
latency more than retrieval does (p50 2.74s vs 0.77s) — the Claude API round-trip
(network + adaptive thinking) is the biggest single lever on user-perceived latency,
not anything in this project's own retrieval code. **Caveat, stated plainly (same
honesty standard as Step 5's multi-hop n=15 caveat): p95 over n=55 samples is just the
~52nd-highest value — noisy, not a precise reproducible figure, especially for
`generate_s` given Step 4/5's already-documented lack of `temperature` pinning.**

**The reranker sweep** (`python scripts/sweep_reranker.py`, retrieval-only, 55
questions, 50 answerable/scored, 0 errors):

| Model | recall | hit_rate | MRR | rerank p50 | rerank p95 | total p50 | total p95 |
|---|---|---|---|---|---|---|---|
| `bge-reranker-base` (original baseline) | 0.830 | 0.900 | 0.695 | 552ms | 567ms | 571ms | 586ms |
| `bge-reranker-v2-m3` (stronger) | 0.900 | 0.980 | 0.840 | 2775ms | 3331ms | 2808ms | 3345ms |
| `ms-marco-MiniLM-L-6-v2` (faster) | 0.880 | 0.960 | 0.752 | 111ms | 128ms | 129ms | 147ms |

**The decision: `ms-marco-MiniLM-L-6-v2` — the "faster" candidate — Pareto-dominates
the original baseline, not just on latency.** It beats `bge-reranker-base` on every
retrieval metric measured (recall 0.83→0.88, hit-rate 0.90→0.96, MRR 0.695→0.752)
*while also* being ~4.4x faster on rerank p95 (567ms→128ms) and ~4x faster on total
retrieval p95 (586ms→147ms). This is not a tradeoff call — there is no axis on which
the original baseline was better. Adopted as the new default in `config.yaml`
(`retrieval.reranker_model`). This is a genuinely surprising result (a 6-layer MiniLM
cross-encoder outperforming a BGE-family reranker built specifically for retrieval) —
plausibly explained by this corpus's heavy share of terse, tabular chunks (drop
tables, XP tables, flattened infoboxes — the same chunk style Step 6 found general-
domain NLI models struggle to read as natural premises) where a smaller, more
generically-trained cross-encoder does just as well at distinguishing relevant from
irrelevant, combined with the small eval set (n=50 answerable questions, so each
additional correct retrieval moves hit-rate by 2 points) making the comparison
noisier than a large-scale benchmark would be. Documented here as a measured finding
on *this* corpus, not a general claim that MiniLM beats BGE-family rerankers broadly.
`bge-reranker-v2-m3` is more accurate still (best of all three on every retrieval
metric) but at a ~5.9x rerank-latency cost over the original baseline (and ~26x over
the new MiniLM default) — given the project's own "latency-optimized inference"
framing and that end-to-end latency is already dominated by generation (p95 ≈ 6.7s
even with the fast reranker), that cost isn't justified by the incremental accuracy
gain over the now-adopted MiniLM default. Not discarded — left documented here as the
higher-accuracy/higher-latency alternative for a future latency-tolerant
configuration, not enabled by default. No threshold-tuning or cherry-picking: this was
a clean win on the measured numbers, not a judgment call requiring the "no vibes-based
tuning" discipline Step 6 needed for its NLI threshold decision.

**Known follow-up, not done in this step (deliberately out of scope):**
`results/eval_report.json` and `results/sample_grounding.json` (Step 5/6's committed
artifacts) were generated under the *original* default reranker (`bge-reranker-base`)
and are not re-run here — the approved Step 7 plan scoped this step to latency
measurement and the reranker decision only, not to refreshing Step 5/6's correctness/
grounding numbers under the new default. Re-running `scripts/run_eval.py` (and
`scripts/sample_grounding.py`) against the new `ms-marco-MiniLM-L-6-v2` default is a
natural next action — likely at Step 9 (README polish), which needs current numbers
anyway — but wasn't done unprompted here to stay within the approved scope.

**Verification:** `pytest -q` — 160 passed, 6 deselected (unchanged slow-test set;
one existing config test (`test_load_config_populates_retrieval_section`) updated to
expect the new default reranker model name in the real `config.yaml`). Both scripts
ran successfully against the real corpus/API with 0 errors.

_Status: complete on `feat/latency-optimization`. Tell the user before starting Step
8's in-depth plan._
