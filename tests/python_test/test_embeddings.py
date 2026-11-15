"""Offline integration tests: no model downloads, tokens, or provider charges."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vekryx import Vekryx, OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings


class ToyModel:
    def __init__(self):
        self.calls = []

    def embed_document(self, text):
        self.calls.append(("document", text))
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [1.0, 0.0]


class EmbeddingTests(unittest.TestCase):
    def test_inferred_dimensions_text_crud_search_and_mmr(self):
        model = ToyModel()
        store = Vekryx(embedding_model=model)
        self.assertEqual(store.dimensions, 2)
        self.assertIs(store.embedding_model, model)
        store.insert("cat", "a cat")
        store.insert("dog", "a dog")
        self.assertEqual(store.get("cat"), [1, 0])
        self.assertEqual(store.search("feline", k=1, metric="cosine"), [("cat", 0)])
        self.assertEqual(store.mmr_search("feline", k=1), [("cat", 0)])
        self.assertIn(("document", "a cat"), model.calls)
        self.assertIn(("query", "feline"), model.calls)
        store.insert("cat", "a dog")
        self.assertEqual(store.get("cat"), [0, 1])
        self.assertTrue(store.delete("cat"))

    def test_model_string_selects_local_adapter(self):
        with patch("vekryx.SentenceTransformerEmbeddings", return_value=ToyModel()) as factory:
            store = Vekryx(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(store.dimensions, 2)
            factory.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    def test_validation_and_failed_embeddings_do_not_mutate_store(self):
        with self.assertRaises(ValueError):
            Vekryx()
        with self.assertRaises(ValueError):
            Vekryx(2, embedding_model=ToyModel())
        with self.assertRaises(TypeError):
            Vekryx(embedding_model=object())
        with self.assertRaises(ValueError):
            Vekryx(2).insert("a", "text")
        model = ToyModel()
        store = Vekryx(embedding_model=model)
        store.insert("a", "cat")
        for invalid in ([], [1], [float("nan"), 0], [[1, 2]]):
            model.embed_document = lambda text: invalid
            with self.assertRaises(ValueError):
                store.insert("a", "replacement")
            self.assertEqual(store.get("a"), [1, 0])
        model.embed_document = lambda text: (_ for _ in ()).throw(RuntimeError("provider failed"))
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            store.insert("a", "replacement")
        self.assertEqual(len(store), 1)

    def test_snapshot_reattaches_model_and_checks_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            store = Vekryx(embedding_model=ToyModel())
            store.insert("cat", "a cat")
            store.save(path)
            restored = Vekryx.load(path, embedding_model=ToyModel())
            self.assertEqual(restored.search("cat", k=1), [("cat", 0)])
            self.assertEqual(Vekryx.load(path).get("cat"), [1, 0])
            model = ToyModel()
            model.embed_document = lambda text: [1, 2, 3]
            with self.assertRaisesRegex(ValueError, "dimensions"):
                Vekryx.load(path, embedding_model=model)

    def test_openai_request_and_errors(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1, 2, 3])])
        adapter = OpenAIEmbeddings("custom-model", client=SimpleNamespace(
            embeddings=SimpleNamespace(create=create)))
        store = Vekryx(embedding_model=adapter)
        self.assertEqual(store.dimensions, 3)
        store.insert("a", "hello")
        store.search("query")
        self.assertEqual(calls[-1], {"model": "custom-model", "input": "query"})

    def test_huggingface_sentence_vectors_and_rejects_token_matrix(self):
        calls = []
        def extract(text, **kwargs):
            calls.append((text, kwargs))
            return [1, 0]
        adapter = HuggingFaceEmbeddings("org/model", client=SimpleNamespace(feature_extraction=extract))
        self.assertEqual(adapter.embed_query("hello"), [1, 0])
        self.assertEqual(calls, [("hello", {"model": "org/model"})])
        adapter._client.feature_extraction = lambda *args, **kwargs: [[1, 0], [0, 1]]
        with self.assertRaisesRegex(ValueError, "pooling"):
            adapter.embed_document("hello")

    def test_local_model_uses_document_and_query_methods(self):
        encoder = SimpleNamespace(encode=lambda text: [9, 9],
                                  encode_document=lambda text: [1, 0],
                                  encode_query=lambda text: [0, 1])
        factory = unittest.mock.Mock(return_value=encoder)
        with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
            adapter = SentenceTransformerEmbeddings("org/model", device="cpu", revision="commit")
        self.assertEqual(adapter.embed_document("doc"), [1, 0])
        self.assertEqual(adapter.embed_query("query"), [0, 1])
        factory.assert_called_once_with("org/model", device="cpu", revision="commit")
        del encoder.encode_document
        del encoder.encode_query
        self.assertEqual(adapter.embed_document("doc"), [9, 9])
        self.assertEqual(adapter.embed_query("query"), [9, 9])
