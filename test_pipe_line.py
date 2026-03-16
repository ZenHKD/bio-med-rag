# test_pipeline.py  (run from bio-med-rag/)
import sys
sys.path.insert(0, ".")

from src.vectorstore.store import VectorStore
from src.pipeline.rag_chain import get_baseline_chain, query

print("Loading VectorStore (index + chunks)...")
store = VectorStore()
print(f"  Index size: {store.index.ntotal} vectors")
print(f"  Chunks loaded: {len(store.chunks)}")

print("\nBuilding baseline chain...")
chain, retriever = get_baseline_chain(store, k=5)

question = "What is hypertension?"
print(f"\nQuestion: {question}")
print("Generating answer...\n")

result = query(chain, retriever, question)

print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(result["result"])

print(f"\n{'=' * 60}")
print(f"SOURCES ({len(result['source_documents'])} chunks):")
print("=" * 60)
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n--- Chunk {i} [{doc.metadata.get('source', '')}] ---")
    print(doc.page_content[:200] + "...")
