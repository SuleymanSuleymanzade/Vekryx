import math
import unittest

from vekryx import (
    HnswConfig, HnswSearchOptions, HybridOptions, IvfConfig, IvfPqConfig,
    IvfSearchOptions, MmrOptions, PqConfig, SearchOptions, Vekryx,
)


def vector(i):
    return [math.sin(i * (d + 1) + 0.3) + i / 20 for d in range(4)]


class Model:
    def embed_document(self, text):
        return vector(int(text.split()[1]) if text.startswith("doc ") else 0)

    def embed_query(self, text):
        return self.embed_document(text)


class IndexTests(unittest.TestCase):
    def configs(self):
        ivf = IvfConfig(nlist=4, iterations=8)
        pq = PqConfig(subquantizers=2, bits=2, iterations=8)
        return [HnswConfig(m=8, ef_construction=32), ivf, pq, IvfPqConfig(ivf, pq)]

    def options(self, config, *, k=5, candidates=32, rerank_exact=True):
        backend = (HnswSearchOptions(64) if isinstance(config, HnswConfig) else
                   IvfSearchOptions(4) if isinstance(config, (IvfConfig, IvfPqConfig)) else None)
        return SearchOptions(k=k, candidates=candidates, rerank_exact=rerank_exact, backend=backend)

    def store(self):
        store = Vekryx(embedding_model=Model())
        for i in range(48):
            store.insert("doc " + str(i), id=str(i), metadata={"group": i % 2})
        return store

    def test_all_backends_search_filter_mmr_and_hybrid_with_named_results(self):
        store = self.store()
        for config in self.configs():
            with self.subTest(config=config):
                index = store.build_index(config, where={"group": 0})
                self.assertEqual(len(index), 24)
                self.assertEqual(index.guarantee, "approximate")
                options = self.options(config)
                hits = index.search("doc 6", options)
                self.assertEqual(hits[0].id, "6")
                self.assertEqual(hits[0].distance, 0)
                self.assertTrue(all(h.distance_kind == "exact" for h in hits))
                self.assertTrue(all(int(h.id) % 2 == 0 for h in hits))
                diverse = index.mmr_search("doc 6", options, mmr=MmrOptions(0.2))
                self.assertEqual(len(diverse), 5)
                self.assertEqual(len({h.id for h in diverse}), 5)
                fused = index.hybrid_search("doc 6", options, hybrid=HybridOptions())
                self.assertEqual(len(fused), 5)
                self.assertTrue(all(int(h.id) % 2 == 0 for h in fused))
                self.assertGreater(fused[0].fusion_score, 0)

    def test_cosine_hnsw_and_all_ivf_lists_agree_with_exact(self):
        store = self.store()
        for config in self.configs()[:2]:
            index = store.build_index(config, metric="cosine")
            hits = index.search("doc 6", self.options(config))
            expected = store.search("doc 6", 5, metric="cosine")
            self.assertEqual([(h.id, h.distance) for h in hits], expected)

    def test_pq_estimates_and_exact_reranking_are_distinct(self):
        store = self.store()
        for config in self.configs()[2:]:
            index = store.build_index(config)
            approximate = index.search("doc 6", self.options(config, rerank_exact=False))
            self.assertTrue(all(h.distance_kind == "approximate" for h in approximate))
            exact = index.search("doc 6", self.options(config, candidates=48))
            self.assertEqual([(h.id, h.distance) for h in exact], store.search("doc 6", 5))
            self.assertEqual(index.guarantee, "approximate")

    def test_snapshot_stays_consistent_after_source_replace_delete_and_insert(self):
        store = self.store()
        for config in self.configs():
            index = store.build_index(config)
            options = self.options(config)
            before = index.search("doc 6", options)
            store.insert([99, 99, 99, 99], id="6")
            store.delete("7")
            store.insert([100, 100, 100, 100], id="new")
            self.assertEqual(index.search("doc 6", options), before)
            store.insert("doc 6", id="6")
            store.insert("doc 7", id="7")
            store.delete("new")

    def test_invalid_settings_metrics_and_backend_option_mismatches(self):
        store = self.store()
        for config in self.configs()[2:]:
            with self.assertRaises(ValueError):
                store.build_index(config, metric="cosine")
        for config in (IvfConfig(nlist=100), PqConfig(subquantizers=3), PqConfig(bits=8)):
            with self.assertRaises(ValueError):
                store.build_index(config)
        index = store.build_index(IvfConfig(nlist=4))
        with self.assertRaises(ValueError):
            index.search("doc 1", SearchOptions(backend=HnswSearchOptions()))
        with self.assertRaises(ValueError):
            index.search("doc 1", SearchOptions(backend=IvfSearchOptions(5)))
        with self.assertRaises(ValueError):
            store.search_with_options("doc 1", SearchOptions(backend=IvfSearchOptions()))
        for make in (lambda: HnswConfig(m=1), lambda: HnswConfig(ef_construction=1),
                     lambda: HnswSearchOptions(0), lambda: IvfSearchOptions(0),
                     lambda: IvfConfig(nlist=-1), lambda: PqConfig(bits=9)):
            with self.assertRaises(ValueError):
                make()
        for config in self.configs():
            empty = store.build_index(config, where={"group": "absent"})
            self.assertEqual(empty.search("doc 1", self.options(config)), [])
        numeric = Vekryx(4)
        numeric.insert("zero", [0, 0, 0, 0])
        for config in self.configs()[:2]:
            with self.assertRaises(ValueError):
                numeric.build_index(config, metric="cosine")
        numeric_index = numeric.build_index(HnswConfig())
        for query in ([1], [float("nan")] * 4):
            with self.assertRaises(ValueError):
                numeric_index.search(query)
        with self.assertRaises(ValueError):
            numeric_index.search("text")


if __name__ == "__main__":
    unittest.main()
