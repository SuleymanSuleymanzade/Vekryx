"""Run against the installed package: python examples/ann_search.py."""
import math
from vekryx import (
    HnswConfig, HnswSearchOptions, IvfConfig, IvfPqConfig, IvfSearchOptions,
    PqConfig, SearchOptions, Vekryx,
)


class Model:
    """Offline feature mapping, not a semantic embedding model."""
    def embed_document(self, text):
        i = int(text.split()[1]) if text.startswith("doc ") else 0
        return [math.sin(i * (d + 1) + 0.3) + i / 20 for d in range(4)]

    def embed_query(self, text):
        return self.embed_document(text)


store = Vekryx(embedding_model=Model())
for i in range(64):
    store.insert("doc " + str(i), id=str(i), metadata={"group": i % 2})

ivf = IvfConfig(nlist=4)
pq = PqConfig(subquantizers=2, bits=2)
for config, backend in (
    (HnswConfig(), HnswSearchOptions(64)),
    (ivf, IvfSearchOptions(4)),
    (pq, None),
    (IvfPqConfig(ivf, pq), IvfSearchOptions(4)),
):
    index = store.build_index(config, where={"group": 0})
    options = SearchOptions(k=3, candidates=32, backend=backend)
    hits = index.search("doc 6", options)
    assert hits[0].id == "6" and hits[0].distance == 0
    assert all(hit.distance_kind == "exact" for hit in hits)
    print(type(config).__name__, hits)
    print("MMR:", index.mmr_search("doc 6", options))
    print("Hybrid:", index.hybrid_search("doc 6", options))
