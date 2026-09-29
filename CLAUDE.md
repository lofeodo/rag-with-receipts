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
| Reranker | Local cross-encoder, `BAAI/bge-reranker-base` baseline. Latency-optimization step compares a stronger (`bge-reranker-v2-m3`) and a faster (`ms-marco-MiniLM-L-6-v2`) variant on precision vs p95, and picks a point on that curve. |
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
- [ ] Step 3 — Retrieval pipeline (dense top-k + cross-encoder reranking)
- [ ] Step 4 — Generation (grounded answers with chunk citations)
- [ ] Step 5 — Eval harness (retrieval precision + LLM-as-judge correctness, results report)
- [ ] Step 6 — Grounding / hallucination check (entailment/overlap flag on cited chunks)
- [ ] Step 7 — Latency instrumentation + one measured optimization (reranker sweep)
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
