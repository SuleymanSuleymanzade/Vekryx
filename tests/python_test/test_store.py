import json
from pathlib import Path
import tempfile
import unittest

from vekryx import Vekryx


class VekryxTests(unittest.TestCase):
    def test_crud_and_ranked_search(self):
        store = Vekryx(dimensions=3)
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

    def test_cosine_and_diverse_search(self):
        store = Vekryx(2)
        store.insert("a", [1, 0])
        store.insert("b", [1, 0])
        store.insert("c", [0.6, 0.8])
        query = [1, 0]
        nearest = store.search(query, k=3, metric="cosine")
        self.assertEqual([key for key, _ in nearest], ["a", "b", "c"])
        self.assertAlmostEqual(nearest[2][1], 0.4)
        diverse = store.mmr_search(query, k=2, fetch_k=3, lambda_mult=0.2)
        self.assertEqual([key for key, _ in diverse], ["a", "c"])
        self.assertAlmostEqual(diverse[1][1], 0.4)
        self.assertEqual(store.mmr_search(query, k=3, lambda_mult=1.0), nearest)
        self.assertEqual(len(store.mmr_search(query, k=10)), 3)
        self.assertEqual(store.mmr_search(query, k=0), [])
        self.assertEqual(store.mmr_search(query, k=2, fetch_k=3,
                                         lambda_mult=0.2, metric="euclidean")[1][0], "c")
        for metric in ("unknown", "mmr"):
            with self.assertRaises(ValueError):
                store.search(query, metric=metric)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                store.mmr_search(query, lambda_mult=value)
        with self.assertRaises(ValueError):
            store.mmr_search(query, k=3, fetch_k=2)
        with self.assertRaises(ValueError):
            store.search([0, 0], metric="cosine")
        store.insert("zero", [0, 0])
        with self.assertRaises(ValueError):
            store.mmr_search(query)
        # Metric choice is per search; snapshots remain compatible.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.json"
            store.delete("zero")
            store.save(path)
            self.assertEqual(Vekryx.load(path).search(query, k=3, metric="cosine"), nearest)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Vekryx(0)
        store = Vekryx(2)
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
            store = Vekryx(2)
            store.insert("item", [1, 2])
            store.save(path)
            store.insert("item", [3, 4])
            store.save(path)
            restored = Vekryx.load(path)
            self.assertEqual(restored.get("item"), [3.0, 4.0])
            self.assertEqual(restored.search([3, 4]), [("item", 0.0)])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            path.write_text(json.dumps({"dimensions": 2, "vectors": {"x": [1]}}))
            with self.assertRaises(ValueError):
                Vekryx.load(path)
            with self.assertRaises(OSError):
                store.save(Path(directory) / "missing" / "vectors.json")
            with self.assertRaises(OSError):
                Vekryx.load(Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()
