"""Optional embedding adapters. Provider SDKs are imported only when selected."""
from math import isfinite
from typing import Protocol, Sequence


class EmbeddingModel(Protocol):
    """Custom adapters implement both methods, returning one sentence vector."""

    def embed_document(self, text: str) -> Sequence[float]: ...
    def embed_query(self, text: str) -> Sequence[float]: ...


def _vector(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("embedding must be a one-dimensional numeric vector; token embeddings require pooling") from error
    if not result or not all(isfinite(value) for value in result):
        raise ValueError("embedding must be nonempty and contain only finite values")
    return result


class SentenceTransformerEmbeddings:
    """Run a Hugging Face sentence embedding model locally."""

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", *, device=None, revision=None):
        from sentence_transformers import SentenceTransformer
        self.model = model
        self._encoder = SentenceTransformer(model, device=device, revision=revision)

    def embed_document(self, text):
        encode = getattr(self._encoder, "encode_document", self._encoder.encode)
        return _vector(encode(text))

    def embed_query(self, text):
        encode = getattr(self._encoder, "encode_query", self._encoder.encode)
        return _vector(encode(text))


class HuggingFaceEmbeddings:
    """Use a Hugging Face feature-extraction endpoint returning sentence vectors."""

    def __init__(self, model, *, token=None, provider=None, client=None):
        self.model = model
        if client is None:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=model, token=token, provider=provider)
        self._client = client

    def embed_document(self, text):
        return _vector(self._client.feature_extraction(text, model=self.model))

    def embed_query(self, text):
        return self.embed_document(text)


class OpenAIEmbeddings:
    """Use OpenAI or an OpenAI-compatible embedding endpoint."""

    def __init__(self, model="text-embedding-3-small", *, api_key=None, base_url=None, client=None):
        self.model = model
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    def embed_document(self, text):
        response = self._client.embeddings.create(model=self.model, input=text)
        return _vector(response.data[0].embedding)

    def embed_query(self, text):
        return self.embed_document(text)
