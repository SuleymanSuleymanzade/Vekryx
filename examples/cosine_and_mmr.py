"""Run with: python examples/cosine_and_mmr.py"""
from vector_database import VectorStore

store = VectorStore(dimensions=2)
store.insert("chunk-a", [1.0, 0.0])
store.insert("chunk-b", [1.0, 0.0])  # Redundant content with another ID.
store.insert("chunk-c", [0.6, 0.8])  # A different direction/topic.
query = [1.0, 0.0]

print("Cosine nearest:", store.search(query, k=2, metric="cosine"))
# [('chunk-a', 0.0), ('chunk-b', 0.0)]

print("MMR diverse:", store.mmr_search(query, k=2, fetch_k=3, lambda_mult=0.2))
# [('chunk-a', 0.0), ('chunk-c', approximately 0.4)]

# lambda_mult=1 uses relevance only, yielding the normal nearest ordering.
print("MMR relevance only:", store.mmr_search(query, k=2, lambda_mult=1.0))
