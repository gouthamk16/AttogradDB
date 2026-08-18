# AttogradDB

A lightweight, document-oriented vector store for semantic retrieval over plaintext, PDF, and JSON documents. Pure-Python library (published to PyPI as `attogradDB`), with an in-progress C/CUDA backend for brute-force similarity search.

Two public surfaces, both in `attogradDB/attodb.py`:

- **`VectorStore`** — embed text, index it (HNSW or brute-force), query by cosine similarity, decode results back to source text.
- **`keyValueStore`** — a JSON-file-backed NoSQL store organised as master collection → collection → documents, with `toVector()` to promote a collection into a `VectorStore`.

## Layout

```
attogradDB/
  attodb.py      VectorStore + keyValueStore. The only module with real logic.
  embedding.py   BertEmbedding — HF AutoModel, mean-pooled last hidden state (768-d).
  indexing.py    HNSW wrapper over hnswlib (cosine, dim=768, max_elements=1000).
  tokenizer.py   tokenize()/decode() — tiktoken for gpt-4, AutoTokenizer for BERT.
  io.py          TextSplitter — fixed-size character chunking with overlap.
  utils.py       read_pdf() via PyPDF2.
  tests/         unittest suites for VectorStore (brute-force) and HNSW.
examples/        Runnable usage examples. These are the de-facto integration tests.
sample_data/     PDFs used by the examples.
```

`attogradDB/__init__.py` re-exports only `VectorStore` and `read_pdf`; `keyValueStore` is imported from `attogradDB.attodb` directly.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest attogradDB/tests -v
```

CUDA/C backend (standalone, built from `attogradDB/cuda/`):

```bash
nvcc search.cu -o search
```

## Known-broken state — read before touching `attodb.py`

This is a revamp, not a greenfield build. The following are confirmed defects on `main`, not stylistic complaints. Fix them at the root; do not work around them.

- **`VectorStore.__init__` cannot construct.** It reads `self.save_path` before ever assigning it → `AttributeError` on every instantiation, including a bare `VectorStore()`. Both test suites therefore fail at `setUp`. The package is currently non-functional.
- **`load_index(self)` takes no path argument** but `__init__` calls `self.load_index(load_path)` → `TypeError`. It also reads `self.save_path`, and runs before `save_path` is set.
- **`save_index()` serialises numpy arrays with `json.dump`** → `TypeError`. Vectors need an explicit `.tolist()` or a dtype-preserving format.
- **`reverse_embedding` is an in-memory dict keyed by the embedding tuple.** It does not survive a save/load cycle, so `decode_results=True` returns `None` for any reloaded index.
- **`HNSW` hard-caps at `max_elements=1000`** and never resizes; adding the 1001st vector raises.
- **README documents `save_path()` and `load_path()` as methods.** They don't exist — they're constructor kwargs. Fix the docs alongside the code.
- **`pyproject.toml` is empty**, `setup.py` says version `0.4`, README says `0.4.2`, and `requirements.txt` omits deps `setup.py` requires (`scipy`, `accelerate`, `huggingface_hub`). Pick one source of truth — prefer `pyproject.toml`.

## Hard rules

Before touching any file, ask: is the task unambiguous, and is it small (< 3 steps, no architectural decision, single module)? If both, proceed. If either is false, write the plan down first and get agreement.

- Bug with clear, reproducible symptoms → fix autonomously: reproduce it, find the root cause, fix it, verify.
- "Add X" / "change Y" without an exact spec → read the existing code first, propose an approach, wait for go-ahead.
- Vague intent ("make retrieval better", "clean this up") → ask, don't guess and edit.
- Never claim what code does without reading it. Never speculate about an API shape, a model's output dimensionality, or an `hnswlib` parameter — check the source or the docs.
- Research before design, not from memory. `transformers`, `hnswlib`, and `tiktoken` APIs shift between versions; read the current docs rather than relying on training-data priors. Lay out the real alternatives with actual trade-offs and recommend one with a reason.

## Execution

- Turn tasks into verifiable goals before starting: "fix the loader" → write the failing round-trip test first, then make it pass.
- Never mark a task done without proving it: run the tests, show the output. If a change touches retrieval quality, show before/after results on a fixed query set — not a claim that it "feels better".
- Match existing style in a file you're editing even if you'd choose differently. Don't refactor adjacent code that isn't part of the task. Every changed line should trace back to the task at hand.
- No temporary patches for root causes you understand. If a proper fix is out of scope, say so explicitly and record it in `to-do.txt` — don't paper over it.

## Code standards

- Simplest correct solution over the extensible one. A function earns its existence by being reused or by making the code clearer — not by anticipating future need. This is a ~500-line library; it does not need a plugin architecture.
- No comments that restate what the code does. Comment only the non-obvious: why an `ef_construction` value is what it is, a workaround for a library bug, an invariant not visible locally. Delete the existing `## Auto generated by Cursor` / `## Don't know if this implementation is correct` markers as you fix the code they sit on.
- No commented-out code left behind (see the dead demo block at the bottom of `embedding.py`, the unused `GPT2Tokenizer` import in `tokenizer.py`).
- Files under ~300 lines, functions under ~30 lines. `attodb.py` is at the limit and holds two unrelated stores — splitting `keyValueStore` into its own module is the right call when it next needs work.
- Type-hint every public signature. `io.py` already does; the rest doesn't. Add hints to code you touch rather than in a sweeping pass.
- Duplicated logic at 3+ occurrences gets extracted; 2 usually don't. The four hand-rolled `os.path.join(self.base_path, self.current_master, ...)` paths in `keyValueStore` are past that line.
- Every IO operation that can fail (missing collection, malformed JSON, unreadable PDF) gets explicit handling where the failure is actionable — not a blanket `try/except` at the call site.
- Naming follows Python convention: `snake_case` methods. `toVector` is the outlier; rename it (keeping a deprecated alias) when that class is next touched.
- Clean imports at module top level. Exception: heavy imports (`transformers`, `torch`) may be deferred into the function that uses them if it measurably improves import time — note why inline if you do.
- No build artifacts committed. `attogradDB/cuda/*.exe`, `*.exp`, `*.lib` must stay untracked — add them to `.gitignore` rather than committing them. If `cuda/` is ever brought back into git, commit the sources only.
- No `TODO` comments scattered through source. Items belong in `to-do.txt` with enough context to act on.

## Tests

- `pytest` is the runner (the suites use `unittest` classes, which pytest collects fine). New tests go in `attogradDB/tests/`, named `test_*.py`.
- Every bug fix lands with a test that fails before it and passes after. The broken-state list above is the immediate backlog: each item is a test waiting to be written.
- Test observable behaviour, not internals. Assert on what `get_similar` returns, not on the shape of `self.index`.
- Embedding-dependent tests are slow (they download and run BERT). Keep them few and deterministic; prefer covering `similarity`, `TextSplitter`, `save_index`/`load_index` round-trips, and `keyValueStore` with fixtures over live model calls.
- Don't assert hardcoded floats to 15 decimal places against model output — `test_similarity` gets away with it only because it operates on literal vectors.

## Branching and PRs

- `main` is the only long-lived branch. Everything else is short-lived and deleted after merge.
- Branch names: `fix/`, `feat/`, `refactor/`, `chore/` plus a short slug (`fix/vectorstore-init-attributeerror`).
- Commits: conventional style (`feat:`, `fix:`, `refactor:`, `chore:`), one logical change per commit. The message says why, not just what.
- One PR per logical change. A PR that fixes the constructor and also rewrites the CUDA kernel is two PRs.
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

No workflow exists yet. When one lands (`.github/workflows/`), it triggers on push and PR to `main`, and each job is added in the same change that gives it something real to check:

- **lint** — `ruff check`
- **test** — `python -m pytest attogradDB/tests`

A red run blocks merge. Don't add stub jobs with nothing to run.

## Roadmap context

`to-do.txt` and the README roadmap hold the standing backlog. The two live threads:

- **GPU/native acceleration** — `attogradDB/cuda/` is an exploration, not a shipped feature, and as of the 2026-08-18 branch cleanup it is **untracked local-only work**: the `cuda` branch that held `search.cu` was deleted, so nothing in that directory is in git history. It has no Python binding, and the CPU baseline in `search.c` exists to answer whether GPU offload is worth it at all. Answer that question with numbers before wiring it into `VectorStore` — and before re-committing it, fix the known bugs in `findSimilar`: it passes `&arr[row * n + col]` (should be the row base `&arr[row * n]`) and `m` as the dimension (should be `n`), the 2-D grid recomputes the same row once per column, and `float h_arr[1000000][100]` is a 400 MB stack allocation that will overflow.
- **More embedding models and index types** — both are currently selected by string comparison in `VectorStore.__init__`. Adding a third of either is the point at which a small registry earns its keep; not before.
