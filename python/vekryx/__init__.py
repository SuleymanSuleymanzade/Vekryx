"""Rust vector search with optional Python embedding adapters."""
import json
from uuid import uuid4

_MISSING = object()

from ._native import Vekryx as _VectorStore, Chunk, FixedSizeChunker, HierarchicalChunker
from .embeddings import (
    EmbeddingModel, SentenceTransformerEmbeddings, HuggingFaceEmbeddings,
    OpenAIEmbeddings, _vector,
)

__all__ = [
    "Vekryx", "Chunk", "FixedSizeChunker", "HierarchicalChunker",
    "EmbeddingModel", "SentenceTransformerEmbeddings", "HuggingFaceEmbeddings",
    "OpenAIEmbeddings",
]


class Vekryx:
    """Select embedding_model for text input, or dimensions for numeric vectors.

    A model name string selects a local Sentence Transformer. Model construction
    probes one document embedding to infer dimensions; hosted adapters make a
    billable request. Snapshots contain records and vectors: reattach the same model on
    load. Credentials and model objects are never serialized.
    """

    def __init__(self, dimensions=None, *, embedding_model=None):
        if embedding_model is not None:
            if dimensions is not None:
                raise ValueError("dimensions are inferred when embedding_model is provided")
            if isinstance(embedding_model, str):
                embedding_model = SentenceTransformerEmbeddings(embedding_model)
            if not all(callable(getattr(embedding_model, method, None))
                       for method in ("embed_document", "embed_query")):
                raise TypeError("embedding_model must implement embed_document and embed_query")
            dimensions = len(_vector(embedding_model.embed_document("dimension probe")))
        elif dimensions is None:
            raise ValueError("provide embedding_model, or dimensions for numeric vectors")
        self._store = _VectorStore(dimensions)
        self._embedding_model = embedding_model

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def dimensions(self):
        return self._store.dimensions

    def __len__(self):
        return len(self._store)

    def _encode(self, value, *, query=False):
        if isinstance(value, str):
            if self._embedding_model is None:
                raise ValueError("text input requires embedding_model in the constructor")
            method = (self._embedding_model.embed_query if query
                      else self._embedding_model.embed_document)
            return _vector(method(value))
        return value

    def insert(self, *args, id=None, vector=_MISSING, metadata=None):
        """Insert text/vector/record and return its ID.

        insert(text), insert(record), insert(vector=[...]) generate UUIDs.
        insert("my-id", text_or_vector) preserves explicit-ID usage.
        insert({"text": "...", "metadata": {...}, "id": "optional"})
        embeds only text and retains the record for filtering/keyword search.
        """
        if len(args) > 2:
            raise TypeError("insert accepts text/record/vector, or an ID and value")
        if len(args) == 2:
            if id is not None or vector is not _MISSING:
                raise TypeError("ID or vector supplied more than once")
            explicit_id, value = args
        elif len(args) == 1:
            if vector is not _MISSING:
                if id is not None:
                    raise TypeError("ID supplied more than once")
                explicit_id, value = args[0], vector
            else:
                explicit_id, value = id, args[0]
        else:
            explicit_id, value = id, vector
            if value is _MISSING:
                raise TypeError("insert requires text, a record, or a vector")
        text = None
        if isinstance(value, dict):
            if set(value) - {"text", "metadata", "id"}:
                raise ValueError("record accepts only text, metadata, and id")
            if metadata is not None:
                raise ValueError("put metadata in the record or the keyword argument, not both")
            record_id = value.get("id")
            if explicit_id is not None and record_id is not None and explicit_id != record_id:
                raise ValueError("conflicting record IDs")
            explicit_id = explicit_id if explicit_id is not None else record_id
            metadata = value.get("metadata", {})
            value = value.get("text")
            if not isinstance(value, str):
                raise ValueError("record text must be a string")
        if isinstance(value, str):
            text = value
        if explicit_id is None:
            explicit_id = str(uuid4())
        if not isinstance(explicit_id, str) or not explicit_id:
            raise ValueError("id must be a nonempty string")
        metadata_json = self._json_object({} if metadata is None else metadata, "metadata")
        self._store.insert_record(explicit_id, self._encode(value), text, metadata_json)
        return explicit_id

    @staticmethod
    def _json_object(value, name):
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError(name + " must be a JSON object with string keys")
        # Reject non-JSON values and lossy key coercion, including nested values.
        def validate(item):
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError(name + " keys must be strings")
                for nested in item.values():
                    validate(nested)
            elif isinstance(item, list):
                for nested in item:
                    validate(nested)
            elif item is not None and type(item) not in (str, int, float, bool):
                raise ValueError(name + " must contain JSON values")
        validate(value)
        try:
            return json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(name + " must contain finite JSON values") from error

    @classmethod
    def _filter(cls, where):
        return None if where is None else cls._json_object(where, "where")

    def get_record(self, id):
        """Return an owned copy of ID, original text, metadata, and vector."""
        record = json.loads(self._store.record_json(id))
        return dict(id=id, vector=self.get(id), **record)

    def get(self, id):
        return self._store.get(id)

    def delete(self, id):
        return self._store.delete(id)

    def search(self, query, k=5, *, metric="euclidean", where=None):
        filter_json = self._filter(where)
        return self._store.search(self._encode(query, query=True), k, metric=metric,
                                  filter_json=filter_json)

    def mmr_search(self, query, k=5, *, fetch_k=20, lambda_mult=0.5, metric="cosine", where=None):
        filter_json = self._filter(where)
        return self._store.mmr_search(self._encode(query, query=True), k,
                                      fetch_k=fetch_k, lambda_mult=lambda_mult, metric=metric,
                                      filter_json=filter_json)

    def hybrid_search(self, query, k=5, *, where=None, metric="cosine",
                      vector_weight=0.5, rrf_k=60):
        """Return (ID, fusion score), largest first, after metadata filtering.

        query is text; requires an embedding model. BM25 scores stored text.
        vector_weight=0 selects keyword results, 1 selects vector rankings.
        """
        if not isinstance(query, str):
            raise TypeError("hybrid_search query must be text")
        filter_json = self._filter(where)
        return self._store.hybrid_search(
            query, self._encode(query, query=True), k, metric=metric,
            vector_weight=vector_weight, rrf_k=rrf_k, filter_json=filter_json)

    def save(self, path):
        """Save vectors, text, metadata, and dimensions; model configuration is not serialized."""
        self._store.save(path)

    @classmethod
    def load(cls, path, *, embedding_model=None):
        store = _VectorStore.load(path)
        if embedding_model is None:
            instance = cls(store.dimensions)
        else:
            instance = cls(embedding_model=embedding_model)
            if instance.dimensions != store.dimensions:
                raise ValueError("embedding model dimensions do not match the snapshot")
        instance._store = store
        return instance
