"""Run: python examples/python_usage.py [snapshot-path]

Unlike the temporary persistence example, this leaves a snapshot on disk.
"""
from pathlib import Path
import sys

from vector_database import HierarchicalChunker, VectorStore

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/vectors.json")
path.parent.mkdir(parents=True, exist_ok=True)
store = VectorStore(3)
store.insert("item1", [1.0, 2.0, 3.0])
store.insert("item2", [3.0, 2.0, 1.0])
query = [1.0, 2.0, 3.0]
print("Cosine:", store.search(query, k=2, metric="cosine"))
print("MMR:", store.mmr_search(query, k=2, fetch_k=2))
chunks = HierarchicalChunker([32, 12], overlap=2).chunk(
    "Rust stores vectors. Python retrieves chunks.", "guide"
)
print("Leaf chunks:", sum(chunk.is_leaf for chunk in chunks))
store.save(path)
restored = VectorStore.load(path)
assert restored.get("item1") == [1.0, 2.0, 3.0]
print("Saved", len(restored), "vectors to", path.resolve())
