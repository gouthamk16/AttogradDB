"""AttogradDB quickstart: add, search, scope, forget."""

from pathlib import Path

from attogradDB import VectorStore, read_pdf
from attogradDB.io import TextSplitter

HERE = Path(__file__).parent

store = VectorStore(path=str(HERE / "memory.db"))

# Ids are generated for you -- you never invent them.
store.add(
    [
        "The retry logic lives in client.py and backs off exponentially.",
        "We ruled out Redis for the queue: no durability guarantee we could rely on.",
        "The flaky test was a timezone issue, not a race condition.",
    ],
    project="payments",
    session="2026-08-18-morning",
)
store.add(["Unrelated note about the marketing site."], project="website")

for doc_id, score, text in store.search("why did we not use redis"):
    print(f"  {score:.3f}  {text}")

# Scoping is the point: search one project without the others bleeding in.
print("\nScoped to 'website':")
for _, score, text in store.search("redis", project="website"):
    print(f"  {score:.3f}  {text}")

# Chunk a PDF into the same store.
splitter = TextSplitter(chunk_size=300, chunk_overlap=20)
splitter.split_text(read_pdf(str(HERE.parent / "sample_data" / "dpo_paper.pdf")))
store.add(splitter.get_docs()[:20], project="papers", kind="pdf")
print(f"\nStore now holds {len(store)} chunks")

best = store.search("preference optimization", top_n=1, kind="pdf")[0]
print("Best match in papers:", best[2][:70])

# Forgetting is a first-class operation.
removed = store.delete(session="2026-08-18-morning")
print(f"\nForgot {removed} chunks from that session; {len(store)} remain")

store.close()
Path(HERE / "memory.db").unlink()
