import json
from pathlib import Path
import tempfile
import unittest

from vector_database import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_crud_and_ranked_search(self):
        store = VectorStore(dimensions=3)
        store.insert("b", [1.0, 2.0, 3.0])
        store.insert("a", [1.0, 2.0, 3.0])
        store.insert("far", [9.0, 2.0, 3.0])
        self.assertEqual(len(store), 3)
        self.assertEqual(store.dimensions, 3)
        self.assertEqual(store.search([1, 2, 3], k=2), [("a", 0.0), ("b", 0.0)])
        self.assertEqual(store.search([1, 2, 3], k=0), [])
        store.insert("a", [2, 2, 3])
        self.assertEqual(store.get("a"), [2.0, 2.0, 3.0])
        self.assertEqual(len(store), 3)
        self.assertTrue(store.delete("a"))
        self.assertFalse(store.delete("missing"))
        with self.assertRaises(KeyError):
            store.get("a")

    def test_validation(self):
        with self.assertRaises(ValueError):
            VectorStore(0)
        store = VectorStore(2)
        for vector in ([1], [float("nan"), 0], [float("inf"), 0]):
            with self.assertRaises(ValueError):
                store.insert("bad", vector)
        with self.assertRaises(ValueError):
            store.insert("", [0, 0])
        with self.assertRaises(ValueError):
            store.search([0])
        self.assertEqual(len(store), 0)

    def test_snapshots_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.json"
            store = VectorStore(2)
            store.insert("item", [1, 2])
            store.save(path)
            store.insert("item", [3, 4])
            store.save(path)
            restored = VectorStore.load(path)
            self.assertEqual(restored.get("item"), [3.0, 4.0])
            self.assertEqual(restored.search([3, 4]), [("item", 0.0)])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            path.write_text(json.dumps({"dimensions": 2, "vectors": {"x": [1]}}))
            with self.assertRaises(ValueError):
                VectorStore.load(path)
            with self.assertRaises(OSError):
                store.save(Path(directory) / "missing" / "vectors.json")
            with self.assertRaises(OSError):
                VectorStore.load(Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()
