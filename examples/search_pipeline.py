"""Run against the installed package: python examples/search_pipeline.py."""
from vekryx import HybridOptions, MmrOptions, SearchOptions, Vekryx


class Model:
    """Offline contract demonstration; not a semantic embedding model."""

    def embed_document(self, text):
        return [0, 1] if "birds" in text else [1, 0]

    def embed_query(self, text):
        return [1, 0]


store = Vekryx(embedding_model=Model())
for id, text in (("a", "cats"), ("b", "dogs"), ("c", "birds")):
    store.insert(text, id=id, metadata={"group": "public"})

options = SearchOptions(k=2, candidates=3)
where = {"group": "public"}
hits = store.search_with_options("cats", options, metric="cosine", where=where)
assert hits[0].distance_kind == "exact"
print("Vector:", hits)
diverse = store.mmr_search_with_options("cats", options, mmr=MmrOptions(0), where=where)
assert diverse[1].id == "c"
print("MMR:", diverse)
fused = store.hybrid_search_with_options("birds", options, hybrid=HybridOptions(), where=where)
assert fused[0].id == "c"
print("Hybrid:", fused)
