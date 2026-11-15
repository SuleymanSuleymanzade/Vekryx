"""Install with: python -m pip install '.[sentence-transformers]'"""
from vekryx import Vekryx

store = Vekryx(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
store.insert("Cats are small domestic animals.")
store.insert("Rust is a systems programming language.")
print("Model:", store.embedding_model.model)
print("Inferred dimensions:", store.dimensions)
print(store.search("Tell me about pets", k=1, metric="cosine"))
print(store.mmr_search("Tell me about pets", k=2))
