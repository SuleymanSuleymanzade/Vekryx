use super::*;
use std::collections::{HashMap, HashSet};

#[test]
fn fixed_chunks_preserve_unicode_whitespace_and_overlap() {
    let text = "aé🙂\n世界 xyz";
    let splitter = FixedSizeChunker::new(4, 1).unwrap();
    let chunks = splitter.chunk(text, "doc").unwrap();
    assert_eq!(
        chunks.iter().map(|c| c.start..c.end).collect::<Vec<_>>(),
        vec![0..4, 3..7, 6..10]
    );
    for c in chunks {
        assert_eq!(
            c.text,
            text.chars()
                .skip(c.start)
                .take(c.end - c.start)
                .collect::<String>()
        );
        assert!(c.parent_id.is_none() && c.is_leaf && c.level == 0);
    }
    assert!(splitter.chunk("", "doc").unwrap().is_empty());
    assert!(splitter.chunk("hello", "").is_err());
    assert!(FixedSizeChunker::new(0, 0).is_err());
    assert!(FixedSizeChunker::new(3, 3).is_err());
    assert_eq!(
        FixedSizeChunker::new(usize::MAX, 0)
            .unwrap()
            .chunk("abc", "d")
            .unwrap()[0]
            .text,
        "abc"
    );
}

#[test]
fn hierarchy_has_unique_ids_containment_and_complete_leaf_coverage() {
    let text = "aé🙂\n世界 xyz";
    let strategy = HierarchicalChunker::new(vec![6, 3, 2], 1).unwrap();
    let chunks = strategy.chunk(text, "doc").unwrap();
    assert_eq!(chunks, strategy.chunk(text, "doc").unwrap());
    let lookup: HashMap<_, _> = chunks.iter().map(|c| (c.id.as_str(), c)).collect();
    assert_eq!(lookup.len(), chunks.len());
    let mut covered = HashSet::new();
    for c in &chunks {
        assert_eq!(
            c.text,
            text.chars()
                .skip(c.start)
                .take(c.end - c.start)
                .collect::<String>()
        );
        assert!(c.end - c.start <= [6, 3, 2][c.level]);
        if let Some(parent) = c.parent_id.as_deref() {
            let parent = lookup[parent];
            assert_eq!(c.level, parent.level + 1);
            assert!(parent.start <= c.start && c.end <= parent.end);
            assert!(!parent.is_leaf);
        } else {
            assert_eq!(c.level, 0);
        }
        if c.is_leaf {
            covered.extend(c.start..c.end);
        }
    }
    assert_eq!(covered.len(), text.chars().count());
    assert!(strategy.chunk("", "doc").unwrap().is_empty());
    assert!(strategy.chunk("x", "").is_err());
    let single = HierarchicalChunker::new(vec![10], 0)
        .unwrap()
        .chunk("x", "doc")
        .unwrap();
    assert!(single[0].is_leaf);
}

#[test]
fn hierarchy_validates_sizes_and_supports_custom_splitters() {
    for sizes in [vec![], vec![0], vec![2, 3], vec![3, 3], vec![3, 0]] {
        assert!(HierarchicalChunker::new(sizes, 0).is_err());
    }
    assert!(HierarchicalChunker::new(vec![6, 3], 3).is_err());
    assert!(HierarchicalChunker::from_splitters(vec![]).is_err());
    struct WholeText;
    impl TextSplitter for WholeText {
        fn split(&self, text: &str) -> Result<Vec<std::ops::Range<usize>>, String> {
            Ok(if text.is_empty() {
                vec![]
            } else {
                std::iter::once(0..text.chars().count()).collect()
            })
        }
    }
    let custom = HierarchicalChunker::from_splitters(vec![
        Box::new(WholeText),
        Box::new(FixedSizeChunker::new(2, 0).unwrap()),
    ])
    .unwrap();
    let chunks = custom.chunk("abcd", "d").unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1].parent_id.as_deref(), Some(chunks[0].id.as_str()));
    struct Invalid;
    impl TextSplitter for Invalid {
        fn split(&self, _: &str) -> Result<Vec<std::ops::Range<usize>>, String> {
            Ok(std::iter::once(0..usize::MAX).collect())
        }
    }
    assert!(HierarchicalChunker::from_splitters(vec![Box::new(Invalid)])
        .unwrap()
        .chunk("x", "d")
        .is_err());
}
