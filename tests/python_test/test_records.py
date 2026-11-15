import json
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

from vekryx import Vekryx


class Model:
    def embed_document(self, text):
        return [1, 0] if "cat" in text else [0, 1]

    def embed_query(self, text):
        return [1, 0]


class RecordTests(unittest.TestCase):
    def test_optional_ids_and_legacy_call_shapes(self):
        store = Vekryx(embedding_model=Model())
        a = store.insert("cat")
        b = store.insert("cat")
        self.assertNotEqual(a, b)
        UUID(a)
        self.assertEqual(store.insert("cat", id="explicit"), "explicit")
        self.assertEqual(store.insert("legacy", "cat"), "legacy")
        self.assertEqual(store.insert("legacy-kw", vector=[1, 0]), "legacy-kw")
        self.assertEqual(store.insert(id="all-kw", vector=[1, 0]), "all-kw")
        self.assertEqual(store.get(store.insert(vector=[1, 0])), [1, 0])
        self.assertEqual(store.get(store.insert([1, 0])), [1, 0])
        with self.assertRaises(TypeError):
            store.insert()
        with self.assertRaises(TypeError):
            store.insert("a", "cat", id="duplicate")

    def test_records_are_owned_and_replacement_removes_old_metadata(self):
        store = Vekryx(embedding_model=Model())
        record = {"text": "cat", "metadata": {"domain": "animals", "tags": ["pet"]}}
        id = store.insert(record)
        record["metadata"]["tags"].append("changed")
        result = store.get_record(id)
        self.assertEqual(result["text"], "cat")
        self.assertEqual(result["metadata"]["tags"], ["pet"])
        result["metadata"]["tags"].append("changed")
        self.assertEqual(store.get_record(id)["metadata"]["tags"], ["pet"])
        store.insert("dog", id=id)
        self.assertEqual(store.get_record(id)["metadata"], {})
        self.assertEqual(store.get_record(id)["text"], "dog")
        self.assertTrue(store.delete(id))
        with self.assertRaises(KeyError):
            store.get_record(id)

    def test_prefilter_fills_k_and_excluded_zero_vectors_never_reach_cosine(self):
        store = Vekryx(embedding_model=Model())
        for i in range(4):
            store.insert("cat", id="outside" + str(i), metadata={"tenant": "other"})
        store.insert([0, 0], id="zero", metadata={"tenant": "other"})
        store.insert({"id": "inside", "text": "dog", "metadata": {"tenant": "mine"}})
        options = {"where": {"tenant": "mine"}}
        self.assertEqual(store.search("cat", k=1, metric="cosine", **options), [("inside", 1)])
        self.assertEqual(store.mmr_search("cat", k=1, fetch_k=1, **options), [("inside", 1)])
        self.assertEqual(store.hybrid_search("dog", k=1, **options)[0][0], "inside")
        for method in (store.search, store.mmr_search, store.hybrid_search):
            self.assertEqual(method("cat", where={"tenant": "absent"}), [])
        with self.assertRaises(ValueError):
            store.search("cat", metric="cosine")

    def test_exact_and_filters_missing_null_nested_and_types(self):
        store = Vekryx(embedding_model=Model())
        store.insert("cat", id="a", metadata={"domain": "pets", "active": True, "extra": None,
                                              "nested": {"x": 1}})
        store.insert("cat", id="b", metadata={"domain": "pets", "active": False})
        self.assertEqual(store.search("cat", where={"domain": "pets", "active": True}), [("a", 0)])
        self.assertEqual(store.search("cat", where={"extra": None}), [("a", 0)])
        self.assertEqual(store.search("cat", where={"active": 1}), [])
        self.assertEqual(store.search("cat", where={"nested": {"x": 1}}), [("a", 0)])
        self.assertEqual(len(store.search("cat", where={})), 2)

    def test_hybrid_fusion_and_parameters(self):
        store = Vekryx(embedding_model=Model())
        store.insert("cat", id="semantic")
        store.insert("dog", id="lexical")
        mixed = store.hybrid_search("DOG!", k=2)
        self.assertEqual(mixed[0][0], "lexical")
        self.assertAlmostEqual(mixed[0][1], 0.5/61 + 0.5/62)
        self.assertEqual(store.hybrid_search("dog", vector_weight=0), [("lexical", 1/61)])
        self.assertEqual(store.hybrid_search("dog", k=1, vector_weight=1)[0][0], "semantic")
        self.assertEqual(store.hybrid_search("unknown", vector_weight=0), [])
        self.assertEqual(store.hybrid_search("dog", k=0), [])
        for weight in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                store.hybrid_search("dog", vector_weight=weight)
        with self.assertRaises(ValueError):
            store.hybrid_search("dog", rrf_k=0)
        with self.assertRaises(TypeError):
            store.hybrid_search([1, 0])

    def test_validation_does_not_replace_existing_record(self):
        store = Vekryx(embedding_model=Model())
        store.insert("cat", id="a", metadata={"keep": True})
        for metadata in ([], {1: "bad"}, {"bad": {1: 2}}, {"bad": float("nan")}, {"bad": object()}):
            with self.assertRaises(ValueError):
                store.insert("dog", id="a", metadata=metadata)
            with self.assertRaises(ValueError):
                store.search("cat", where=metadata)
        for value in ({"text": 3}, {"text": "cat", "unknown": 2}):
            with self.assertRaises(ValueError):
                store.insert(value)
        with self.assertRaises(ValueError):
            store.insert({"id": "b", "text": "cat"}, id="a")
        self.assertEqual(store.get_record("a")["metadata"], {"keep": True})
        self.assertEqual(len(store), 1)

    def test_disk_round_trip_records_and_old_vector_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            store = Vekryx(embedding_model=Model())
            id = store.insert({"text": "cat", "metadata": {"domain": "pets"}})
            store.save(path)
            restored = Vekryx.load(path, embedding_model=Model())
            self.assertEqual(restored.get_record(id), store.get_record(id))
            self.assertEqual(restored.hybrid_search("cat", where={"domain": "pets"})[0][0], id)
            path.write_text(json.dumps({"dimensions": 2, "vectors": {"old": [1, 0]}}))
            legacy = Vekryx.load(path)
            self.assertEqual(legacy.get_record("old"), {"id": "old", "vector": [1, 0], "text": None, "metadata": {}})
            self.assertEqual(legacy.search([1, 0], where={"domain": "pets"}), [])
