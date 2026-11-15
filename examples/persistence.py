"""Run with: python examples/persistence.py

Uses a temporary directory so running the example leaves no snapshot behind.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from vekryx import Vekryx

with TemporaryDirectory() as directory:
    path = Path(directory) / "vectors.json"
    store = Vekryx(dimensions=3)
    store.insert("item1", [1.0, 2.0, 3.0])
    store.save(path)

    restored = Vekryx.load(path)
    print("Restored vector:", restored.get("item1"))
    print("Search:", restored.search([1.0, 2.0, 3.0], k=1))

    # Loaded stores are independent. Save again to persist subsequent changes.
    restored.insert("item2", [4.0, 5.0, 6.0])
    restored.save(path)
    print("Saved record count:", len(Vekryx.load(path)))
    print("Original in-memory count:", len(store))

# For a lasting snapshot, use a path such as Path("vectors.json") instead.
