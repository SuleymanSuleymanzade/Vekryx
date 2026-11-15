"""Run with: python examples/hierarchical_chunking.py

Toy keyword features demonstrate the pipeline without an embedding dependency.
Use the same real embedding model for documents and queries in an application.
"""
from vekryx import FixedSizeChunker, HierarchicalChunker, Vekryx

text = (
    "Rust manages vector storage. Rust traits separate search strategies. "
    "Python calls the vector store. Python applications retrieve matching chunks. "
) * 3

flat = FixedSizeChunker(chunk_size=60, overlap=10).chunk(text, "flat-guide")
print("Flat chunks:", len(flat))

chunks = HierarchicalChunker(chunk_sizes=[140, 60], overlap=10).chunk(text, "guide")
lookup = {chunk.id: chunk for chunk in chunks}


def embed(value):
    """Illustrative features, not a semantic embedding model."""
    value = value.lower()
    return [1.0, float(value.count("rust")), float(value.count("python"))]


store = Vekryx(dimensions=3)
for chunk in chunks:
    if chunk.is_leaf:
        store.insert(chunk.id, embed(chunk.text))

query = embed("Rust")
print("Cosine matches:", store.search(query, k=3, metric="cosine"))
results = store.mmr_search(query, k=3, fetch_k=20, lambda_mult=0.5)

# Search small leaves, then retrieve their larger parents for answering.
seen = set()
for chunk_id, distance in results:
    leaf = lookup[chunk_id]
    parent = lookup[leaf.parent_id] if leaf.parent_id else leaf
    if parent.id not in seen:
        seen.add(parent.id)
        print(f"Parent {parent.id}, matched leaf distance {distance:.3f}:")
        print(parent.text)
