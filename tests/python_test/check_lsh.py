"""Basic smoke example for the supported Python API."""
from vector_database import VectorStore

store = VectorStore(dimensions=3)
store.insert("example", [1.0, 2.0, 3.0])
print(store.search([1.1, 2.0, 3.0], k=1))
