# AttogradDB

A lightweight, document-oriented vector store for semantic retrieval over plaintext, PDF, and JSON documents. Pure-Python library, published to PyPI as `attogradDB`.

There is one public surface: **`VectorStore`** in `attogradDB/attodb.py`. Add text, search it, scope by project/session/kind, delete what you no longer want.

## Layout

```
attogradDB/
  attodb.py      VectorStore. SQLite for durability, numpy arrays as the search index.
  embedding.py   QwenEmbedding — Qwen3-Embedding-0.6B via ONNX, last-token pooled (1024-d).
  io.py          TextSplitter — fixed-size character chunking with overlap.
  utils.py       read_pdf() via pypdf.
  tests/         pytest suites; conftest.py provides a stub embedder for the fast ones.
examples/        quickstart.py — the de-facto integration test.
sample_data/     PDFs used by the example.
```

`attogradDB/__init__.py` exports `VectorStore` and `read_pdf`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest attogradDB/tests -v
```

## State

Rewritten on 2026-08-18. Two things will look wrong if you don't know why they are the way
they are:

- **There is no ANN index, deliberately.** Search scans every candidate. hnswlib was removed
  after benchmarking: it wins unfiltered by ~90x but loses filtered queries by ~275x, and
  nearly every query here is scoped. See **Measured decisions** below for the numbers and
  the conditions under which this flips.
- **SQLite is the durable store; the numpy arrays are the index.** `_ids`, `_vectors` and
  `_scopes` are all rebuilt by `_load()`. Scope columns live in memory too — round-tripping
  90k ids through SQL cost 20ms against 0.2ms for a numpy comparison. `_texts()` is the one
  hot-path SQL call, and it only ever fetches the top-k rows.

- **Queries and documents are embedded differently, on purpose.** Qwen3 is trained
  asymmetrically: `embed_query()` prepends an instruction, `embed_document()` does not.
  Collapsing them back into one method silently costs retrieval quality.
- **Embedding is one text at a time, on purpose.** Batching measured flat at ~20
  chunks/sec from batch 1 to 32, and padding changes the output (0.946 cosine against
  the unpadded vector). Don't add batching without re-measuring both.
- **int8 is the fastest build on CPU, measured.** The 4-bit exports (q4, q4f16, bnb4)
  are ~2x slower because CPUs dequantise them per op, and they only agree 0.88-0.90 with
  int8 so they change results too. Don't "optimise" by switching quantisation without
  re-running the comparison under **Measured decisions**.
- **Execution provider is auto-detected, with a guaranteed CPU fallback.**
  `available_providers()` puts CUDA and DirectML ahead of CPU; `_open_session()` catches
  a failed accelerator and reopens on CPU. Never hardcode `["CPUExecutionProvider"]`
  again -- that silently ignored a GPU. And do not add CoreML to the auto-preferred list:
  it ships in the default macOS wheel, so preferring it would change behaviour for every
  Mac user unasked, and it can be slower on a heavily partitioned model like this one.
  `AzureExecutionProvider` is also in the default build and is *remote inference* -- the
  allowlist keeps it out, and a test locks that in.
- **Full 1024-d vectors go to SQLite; the in-memory index is truncated to `dim`.** That
  is what lets `dim` change without re-embedding. Truncation must renormalise.
- **Scopes accept a value or a list.** `Scope = str | list[str] | None`. `search()`
  matches in memory via `np.isin`; `delete()` builds a SQL `IN`. An empty list matches
  nothing rather than everything -- that distinction is tested, don't "simplify" it away.

`keyValueStore` is gone. Don't reintroduce a second store. `torch`, `transformers` and
`tiktoken` are gone too — `tokenizers` loads `tokenizer.json` directly. Don't pull them
back in without a reason that survives "this adds 2.5GB to every install".

## Hard rules

Before touching any file, ask: is the task unambiguous, and is it small (< 3 steps, no architectural decision, single module)? If both, proceed. If either is false, write the plan down first and get agreement.

- Bug with clear, reproducible symptoms → fix autonomously: reproduce it, find the root cause, fix it, verify.
- "Add X" / "change Y" without an exact spec → read the existing code first, propose an approach, wait for go-ahead.
- Vague intent ("make retrieval better", "clean this up") → ask, don't guess and edit.
- Never claim what code does without reading it. Never speculate about an API shape, a model's output dimensionality, or a library parameter — check the source or the docs.
- Research before design, not from memory. `onnxruntime`, `tokenizers` and `numpy` APIs shift between versions; read the current docs rather than relying on training-data priors. Lay out the real alternatives with actual trade-offs and recommend one with a reason.

## Execution

- Turn tasks into verifiable goals before starting: "fix the loader" → write the failing round-trip test first, then make it pass.
- Never mark a task done without proving it: run the tests, show the output. If a change touches retrieval quality, show before/after results on a fixed query set — not a claim that it "feels better".
- Match existing style in a file you're editing even if you'd choose differently. Don't refactor adjacent code that isn't part of the task. Every changed line should trace back to the task at hand.
- No temporary patches for root causes you understand. If a proper fix is out of scope, say so explicitly and record it — a rejected approach with numbers goes under **Measured decisions**, a wishlist item goes in the local `to-do.txt` (which is gitignored and does not travel with the repo).

## Code standards

- Simplest correct solution over the extensible one. A function earns its existence by being reused or by making the code clearer — not by anticipating future need. This is a ~250-line library; it does not need a plugin architecture.
- No comments that restate what the code does. Comment only the non-obvious: why an `ef_construction` value is what it is, a workaround for a library bug, an invariant not visible locally.
- No commented-out code left behind.
- Files under ~300 lines, functions under ~30 lines. Every module is currently well inside both.
- Type-hint every public signature. Add hints to code you touch rather than in a sweeping pass.
- Duplicated logic at 3+ occurrences gets extracted; 2 usually don't.
- Every IO operation that can fail (missing collection, malformed JSON, unreadable PDF) gets explicit handling where the failure is actionable — not a blanket `try/except` at the call site.
- Naming follows Python convention: `snake_case` methods.
- Clean imports at module top level. The model download and ONNX session load are deferred into `QwenEmbedding._load()` so that opening a store costs nothing — that laziness is deliberate, not an oversight.
- No build artifacts committed. Compiled output, `.egg-info/`, `build/` and `dist/` stay untracked — add them to `.gitignore` rather than committing them.
- No `TODO` comments scattered through source. Anything worth keeping goes in **Measured decisions** if it is a settled finding, or the local `to-do.txt` if it is just a wish.

## Tests

- `pytest` is the runner, plain functions not `unittest` classes. `test_integration.py` runs real BERT and stays small; everything else uses the `stub_embedding` fixture. New tests go in `attogradDB/tests/`, named `test_*.py`.
- Every bug fix lands with a test that fails before it and passes after. Show the failure, not just the pass.
- Test observable behaviour, not internals. Assert on what `get_similar` returns, not on the shape of `self.index`.
- Embedding-dependent tests are slow (they download and run BERT). Use the `stub_embedding` fixture from `conftest.py`, which maps text to a deterministic 768-d vector, unless the test genuinely needs the real model.
- Don't assert hardcoded floats to 15 decimal places against model output — `test_similarity` gets away with it only because it operates on literal vectors.
- The full suite runs in ~10s. Keep it that way.

## Branching and PRs

- `main` is the only long-lived branch. Everything else is short-lived and deleted after merge.
- Branch names: `fix/`, `feat/`, `refactor/`, `chore/` plus a short slug (`fix/vectorstore-init-attributeerror`).
- Commits: conventional style (`feat:`, `fix:`, `refactor:`, `chore:`), one logical change per commit. The message says why, not just what.
- One PR per logical change. A PR that fixes the constructor and also restructures a module is two PRs.
- PR description states: what changed, why, how it was verified (the actual command and its output), and anything deliberately left out of scope.
- Never force-push to `main`, and no direct commits to `main` for anything non-trivial.
- Rebase on `main` before opening the PR; keep history linear.

## Code review

Every non-trivial PR gets a review pass before merge — self-review counts on a solo project, but it has to be an actual read of the diff, not a glance.

Reviewer checks, in priority order:

1. **Correctness** — does it do what the description claims? Are the edge cases (empty store, `top_n` larger than the store, duplicate vectors, a missing collection file) handled or explicitly out of scope?
2. **Verification** — is there a test, and does it actually fail without the fix? "Tested manually" with no output is not review-ready.
3. **Scope** — every changed line traces to the stated purpose. Unrelated reformatting gets pulled out.
4. **Simplification** — is there a shorter correct version? A new abstraction with one caller gets flagged.
5. **Performance** — only where it's measured. `update_index` is O(n) per insert and brute-force `get_similar` is O(n) per query; that's known and acceptable at current scale. Don't optimise either without a benchmark showing it matters.

Run `/simplify` on the diff before requesting review. Don't merge with unresolved review comments, and don't weaken a test to make CI pass — fix what it's catching, or fix the test if the test is the thing that's wrong.

## CI

No workflow exists yet, but both commands below are real and green today, so a workflow can land whenever you want one. It triggers on push and PR to `main`:

- **lint** — `ruff check`
- **test** — `python -m pytest attogradDB/tests`

A red run blocks merge. Don't add stub jobs with nothing to run.

## Measured decisions

Numbers taken on a 16-core Windows laptop with an RTX 4060. They exist so nobody redoes
this work or "optimises" one of these back into the code. `to-do.txt` is gitignored and
local-only, so anything durable belongs here instead.

**No ANN index.** At 100k chunks x 768d, end to end:

| | unfiltered | scoped 2% | scoped 90% |
|---|---|---|---|
| exhaustive scan | 8.4 ms | 3.3 ms | 10.0 ms |
| hnswlib | 0.09 ms | 8.25 ms | — |

hnswlib wins unfiltered by ~90x and loses filtered by ~275x: a filter disconnects its
graph, while it only shortens an exhaustive scan. Almost every query here is scoped, and
8 ms is invisible next to a ~2000 ms LLM call. Revisit past ~1M vectors, where a scan
reaches ~230 ms.

**`SUBSET_SCAN_THRESHOLD = 0.10`.** `self._vectors[rows]` copies. Below ~10% selectivity
the copy is cheaper than scanning everything; above it the copy dominates (at 40%: 29 ms
copy vs 8 ms masked scan). Re-measure if the dimension or dtype changes.

**Scopes match in memory, not in SQL.** Pulling 90k ids back out of SQLite cost 19.5 ms
against ~0.2 ms for a numpy comparison. SQLite is durable storage plus the top-k text
lookup; the numpy arrays are the index.

**Normalise on insert.** Identical to per-query cosine within 4.8e-08 and 18x faster
(129 ms -> 7.2 ms at 100k).

**Qwen3-Embedding-0.6B over EmbeddingGemma-300M.** Gemma's official repo is *gated*, so
every user would need an HF token — fatal for zero-config install regardless of licence.
Qwen is Apache-2.0, ungated, and MRL-trained to 32 dims against Gemma's 128 floor. On the
quickstart query "why did we not use redis": BERT scored the correct answer 0.656 and an
irrelevant note 0.581 (gap 0.075); Qwen scores 0.699 vs 0.398 (gap 0.301, irrelevant
ranks last). Dropping BERT also dropped torch, transformers and tiktoken — about 2.5 GB.

**int8 is the fastest build on CPU.** Lower-bit exports are slower *and* change results:

| build | KV dtype | query | 65 tok | 520 tok | cosine vs int8 |
|---|---|---|---|---|---|
| int8 | float32 | 22 ms | 57 ms | 478 ms | 1.0000 |
| q4f16 | float16 | 21 ms | 119 ms | 990 ms | 0.8827 |
| q4 | float32 | 21 ms | 113 ms | 921 ms | 0.8850 |
| bnb4 | float32 | 115 ms | 232 ms | 1332 ms | 0.8956 |

CPUs dequantise 4-bit per op with no native SIMD path. Thread pinning did not help
either. `model_fp16` needs its `.onnx_data` companion and will not load without it.

**No batching.** Throughput is flat at ~20 chunks/sec from batch 1 to 32, and padding
perturbs the output — a padded text scores 0.946 against its unpadded self, only partly
improved by deriving `position_ids` from the attention mask. Unbatched needs no padding
and is exactly correct.

**GPU is supported but untried, and safe to leave uninstalled.** `pip install
onnxruntime-directml` (Windows, no CUDA setup) or `onnxruntime-gpu` is picked up with no
code change; both *replace* the `onnxruntime` package rather than coexisting. The payoff
is ingest only: a 200-chunk PDF is ~95 s on CPU, while query embedding at 22 ms is
already invisible.

The fallback is not assumed. onnxruntime's own degradation was verified against a
provider missing from the build (with and without CPU listed), an unknown provider name,
and an empty list -- all four ended on CPU, the bogus name printing "Falling back to
['CPUExecutionProvider'] and retrying". The one path that cannot be tested from a machine
without the hardware is a provider that is in the build but fails at runtime init, so
`_open_session()` wraps session creation and reopens on CPU itself rather than trusting
onnxruntime's Python layer, which has changed before. `QwenEmbedding.provider` reports
what is actually running.

Mac support is the reason CoreML is opt-in rather than automatic. Macs get the same
CPU path as everyone else unless someone measures a CoreML win and passes it explicitly.

**Caveats on the above.** Every timing used synthetic vectors; correctness tests use the
real model. MRL truncation was validated on 8 documents and 4 queries — top-1 survived to
64 dims and broke at 32 — so `DEFAULT_DIM = 256` is chosen for headroom, not because 128
was proven. `onnx-community` declares no licence in its repo metadata; the Apache-2.0
grant is upstream Qwen's, and the revision is unpinned.

## Roadmap

The live thread is the **agent-memory layer**: chunk session traces and expose them over
MCP so an agent can query mid-task. That is the product; this store is plumbing. The
README roadmap and the local `to-do.txt` hold the rest of the wishlist.
