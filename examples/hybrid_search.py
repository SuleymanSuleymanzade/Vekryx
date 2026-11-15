"""Offline runnable example; replace ExampleEmbeddings with a provider adapter."""
from vekryx import Vekryx


class ExampleEmbeddings:
    def embed_document(self, text):
        return [1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0]

    def embed_query(self, text):
        return self.embed_document(text)


store = Vekryx(embedding_model=ExampleEmbeddings())
record_id = store.insert({
    "text": "Cats are domestic animals.",
    "metadata": {"domain": "animals", "language": "en"},
})
store.insert("Rust prevents memory errors.", metadata={"domain": "programming"})
print("Generated ID:", record_id)
print("Filtered vector search:", store.search("cats", where={"domain": "animals"}))
for id, score in store.hybrid_search("cats", where={"domain": "animals"}):
    print("Hybrid:", score, store.get_record(id)["text"])
