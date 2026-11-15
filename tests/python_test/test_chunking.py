import unittest

from vector_database import FixedSizeChunker, HierarchicalChunker, VectorStore


class ChunkingTests(unittest.TestCase):
    def test_fixed_size_unicode_overlap_and_metadata(self):
        text = "aé🙂\n世界 xyz"
        chunks = FixedSizeChunker(chunk_size=4, overlap=1).chunk(text, document_id="unicode")
        self.assertEqual([(c.start, c.end) for c in chunks], [(0, 4), (3, 7), (6, 10)])
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
            self.assertIsNone(chunk.parent_id)
            self.assertTrue(chunk.is_leaf)
            self.assertEqual(chunk.level, 0)
            with self.assertRaises(AttributeError):
                chunk.text = "changed"
        self.assertEqual(FixedSizeChunker().chunk(""), [])
        self.assertEqual(FixedSizeChunker(3).chunk(" \n ")[0].text, " \n ")

    def test_hierarchy_parent_expansion_and_search(self):
        text = "Rust stores vectors. Python retrieves chunks. " * 4
        chunks = HierarchicalChunker(chunk_sizes=[60, 20], overlap=4).chunk(text, "guide")
        lookup = {c.id: c for c in chunks}
        self.assertEqual(len(lookup), len(chunks))
        store = VectorStore(2)
        for c in chunks:
            self.assertEqual(c.text, text[c.start:c.end])
            if c.parent_id is not None:
                parent = lookup[c.parent_id]
                self.assertEqual(c.level, parent.level + 1)
                self.assertGreaterEqual(c.start, parent.start)
                self.assertLessEqual(c.end, parent.end)
                self.assertFalse(parent.is_leaf)
            if c.is_leaf:
                store.insert(c.id, [1.0, float(len(c.text))])
        for key, _ in store.mmr_search([1, 20], k=3, fetch_k=10):
            leaf = lookup[key]
            self.assertTrue(leaf.is_leaf)
            self.assertIn(leaf.text, lookup[leaf.parent_id].text)
        self.assertEqual(HierarchicalChunker().chunk(""), [])
        self.assertEqual(HierarchicalChunker([10]).chunk("hi")[0].text, "hi")
        self.assertNotEqual(chunks[0].id, HierarchicalChunker([60, 20], 4).chunk(text, "other")[0].id)

    def test_invalid_configuration(self):
        for sizes in ([], [0], [2, 3], [3, 3], [3, 0]):
            with self.assertRaises(ValueError):
                HierarchicalChunker(sizes)
        for size, overlap in ((0, 0), (3, 3), (3, 4)):
            with self.assertRaises(ValueError):
                FixedSizeChunker(size, overlap)
        with self.assertRaises(ValueError):
            HierarchicalChunker([6, 3], overlap=3)
        for chunker in (FixedSizeChunker(), HierarchicalChunker()):
            with self.assertRaises(ValueError):
                chunker.chunk("text", document_id="")


if __name__ == "__main__":
    unittest.main()
