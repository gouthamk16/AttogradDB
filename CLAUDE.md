# AttogradDB

A lightweight, document-oriented vector store for semantic retrieval over plaintext, PDF, and JSON documents. Pure-Python library, published to PyPI as `attogradDB`.

Two public surfaces, both in `attogradDB/attodb.py`:

- **`VectorStore`** — embed text, index it (HNSW or brute-force), query by cosine similarity, decode results back to source text.
- **`keyValueStore`** — a JSON-file-backed NoSQL store organised as master collection → collection → documents, with `toVector()` to promote a collection into a `VectorStore`.

## Layout

```
attogradDB/
  attodb.py      VectorStore only. Owns {id: vector} and {id: text}.
  kvstore.py     keyValueStore. All filesystem access goes through _path/_read/_write.
  embedding.py   BertEmbedding — HF AutoModel, mean-pooled last hidden state (768-d).
  indexing.py    HNSW wrapper over hnswlib (cosine, dim=768, doubles capacity when full).
  tokenizer.py   tokenize()/decode() — tiktoken, falling back to AutoTokenizer.
  io.py          TextSplitter — fixed-size character chunking with overlap.
  utils.py       read_pdf() via pypdf.
  tests/         pytest suites; conftest.py provides a stub embedder for the fast ones.
examples/        Runnable usage examples. These are the de-facto integration tests.
sample_data/     PDFs used by the examples.
```

`attogradDB/__init__.py` exports `VectorStore`, `keyValueStore`, and `read_pdf`. `attogradDB.attodb` still re-exports `keyValueStore` for backwards compatibility, but new code should import it from `attogradDB.kvstore`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest attogradDB/tests -v
```

## State

The package was non-functional before the 2026-08-18 revamp — `VectorStore` could not be
constructed at all — and a code review found 15 defects. All are fixed and covered by tests;
`git log` has the details. Two things worth knowing before making changes:

- **`VectorStore` owns the id→text map, not `BertEmbedding`.** Decoding used to work by hashing
  a 768-float tuple, which never survived a save/load and collapsed duplicate texts onto one id.
  Don't reintroduce a reverse lookup keyed on embedding values.
- **`keyValueStore` is scheduled for replacement by Postgres/pgvector.** Its JSON backend is
  deliberately minimal. Fix bugs in it, but don't invest in features there — reimplement
  `_path`/`_read`/`_write` when the migration happens (`to-do.txt` item 10).

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

- Simplest correct solution over the extensible one. A function earns its existence by being reused or by making the code clearer — not by anticipating future need. This is a ~400-line library; it does not need a plugin architecture.
- No comments that restate what the code does. Comment only the non-obvious: why an `ef_construction` value is what it is, a workaround for a library bug, an invariant not visible locally.
- No commented-out code left behind.
- Files under ~300 lines, functions under ~30 lines. Every module is currently well inside both.
- Type-hint every public signature. Add hints to code you touch rather than in a sweeping pass.
- Duplicated logic at 3+ occurrences gets extracted; 2 usually don't.
- Every IO operation that can fail (missing collection, malformed JSON, unreadable PDF) gets explicit handling where the failure is actionable — not a blanket `try/except` at the call site.
- Naming follows Python convention: `snake_case` methods. The camelCase `keyValueStore` class name is kept because it is published; `toVector` survives only as a deprecated alias of `to_vector`.
- Clean imports at module top level. Exception: heavy imports (`transformers`, `torch`) may be deferred into the function that uses them if it measurably improves import time — note why inline if you do.
- No build artifacts committed. Compiled output, `.egg-info/`, `build/` and `dist/` stay untracked — add them to `.gitignore` rather than committing them.
- No `TODO` comments scattered through source. Items belong in `to-do.txt` with enough context to act on.

## Tests

- `pytest` is the runner. `test_core.py`/`test_hnsw.py` are the original `unittest` classes and run real BERT; everything newer is plain pytest functions. New tests go in `attogradDB/tests/`, named `test_*.py`.
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

## Roadmap context

`to-do.txt` and the README roadmap hold the standing backlog. The two live threads:

- **More embedding models and index types** — both are currently selected by string comparison in `VectorStore.__init__`. Adding a third of either is the point at which a small registry earns its keep; not before.
