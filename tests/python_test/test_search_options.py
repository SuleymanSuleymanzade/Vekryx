import unittest

from vekryx import FusionHit, HybridOptions, MmrOptions, SearchHit, SearchOptions, Vekryx


class Model:
    def embed_document(self, text):
        return [1, 0] if "cat" in text else [0, 1]

    def embed_query(self, text):
        return [1, 0]


class SearchInterfaceTests(unittest.TestCase):
    def test_named_results_match_legacy_search_and_mmr_after_filtering(self):
        store = Vekryx(2)
        for id, vector in (("a", [1, 0]), ("b", [1, 0]), ("c", [0, 1])):
            store.insert(vector, id=id, metadata={"group": "public"})
        store.insert([0, 0], id="outside", metadata={"group": "private"})
        where = {"group": "public"}
        options = SearchOptions(k=2, candidates=3)
        hits = store.search_with_options([1, 0], options, metric="cosine", where=where)
        self.assertEqual(hits, [SearchHit("a", 0, "exact"), SearchHit("b", 0, "exact")])
        self.assertEqual([(h.id, h.distance) for h in hits],
                         store.search([1, 0], k=2, metric="cosine", where=where))
        diverse = store.mmr_search_with_options(
            [1, 0], options, mmr=MmrOptions(lambda_mult=0), where=where)
        self.assertEqual([h.id for h in diverse], ["a", "c"])
        self.assertEqual([(h.id, h.distance) for h in diverse],
                         store.mmr_search([1, 0], k=2, fetch_k=3, lambda_mult=0, where=where))
        self.assertEqual(store.search_with_options([1, 0], SearchOptions(k=0), where=where), [])
        self.assertEqual(store.search_with_options([1, 0], where={"group": "absent"}), [])
        # Disabling reranking does not make exact backend distances approximate.
        hits = store.search_with_options([1, 0], SearchOptions(k=1, rerank_exact=False), where=where)
        self.assertEqual(hits[0].distance_kind, "exact")

    def test_bounded_hybrid_pool_changes_fusion_and_full_pool_preserves_legacy(self):
        store = Vekryx(embedding_model=Model())
        store.insert("cat", id="a")
        store.insert("dog", id="b")
        full = store.hybrid_search_with_options("dog", SearchOptions(k=1, candidates=2))
        self.assertIsInstance(full[0], FusionHit)
        self.assertEqual([(h.id, h.fusion_score) for h in full], store.hybrid_search("dog", k=1))
        bounded = store.hybrid_search_with_options("dog", SearchOptions(k=1))
        self.assertEqual(bounded[0], FusionHit("a", 0.5 / 61))
        keywords = store.hybrid_search_with_options(
            "dog", SearchOptions(k=1), hybrid=HybridOptions(vector_weight=0))
        self.assertEqual(keywords[0], FusionHit("b", 1 / 61))
        no_keywords = store.hybrid_search_with_options(
            "dog", SearchOptions(k=1), hybrid=HybridOptions(keyword_candidates=0))
        self.assertEqual(no_keywords[0].id, "a")

    def test_options_and_queries_reject_invalid_values(self):
        for kwargs in ({"k": -1}, {"k": True}, {"candidates": 0},
                       {"candidates": 1.5}, {"rerank_exact": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchOptions(**kwargs)
        for value in (-1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MmrOptions(value)
            with self.assertRaises(ValueError):
                HybridOptions(vector_weight=value)
        for kwargs in ({"rrf_k": 0}, {"keyword_candidates": -1}):
            with self.assertRaises(ValueError):
                HybridOptions(**kwargs)
        store = Vekryx(2)
        for query in ([1], [float("nan"), 0], [0, 0]):
            with self.assertRaises(ValueError):
                store.search_with_options(query, SearchOptions(k=0), metric="cosine")
        with self.assertRaises(ValueError):
            store.search_with_options([1, 0], metric="unsupported")
        with self.assertRaises(TypeError):
            store.hybrid_search_with_options([1, 0])


if __name__ == "__main__":
    unittest.main()
