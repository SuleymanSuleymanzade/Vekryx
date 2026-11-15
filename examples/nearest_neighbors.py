"""Run with: python examples/nearest_neighbors.py"""
from vekryx import Vekryx

# Coordinates are supplied directly; the store does not generate embeddings.
store = Vekryx(dimensions=2)
for record_id, coordinates in {
    "origin": [0.0, 0.0],
    "east": [1.0, 0.0],
    "north": [0.0, 1.0],
    "far": [5.0, 5.0],
}.items():
    store.insert(record_id, coordinates)

query = [0.0, 0.0]
print("Three nearest records to", query)
for record_id, distance in store.search(query, k=3):
    print(f"{record_id}: distance={distance:.3f}, vector={store.get(record_id)}")

# Distances are ascending; equal distances are ordered by ID (east, then north).
# Identical vectors under different IDs remain separate records.
store.insert("origin-copy", [0.0, 0.0])
print("Exact matches:", store.search(query, k=2))
