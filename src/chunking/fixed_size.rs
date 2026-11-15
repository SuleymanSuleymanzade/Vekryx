use super::{boundaries, chunk_id, validate_document_id, Chunk, ChunkingStrategy, TextSplitter};
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self, String> {
        if chunk_size == 0 || overlap >= chunk_size {
            return Err(
                "chunk_size must be positive and overlap must be smaller than chunk_size".into(),
            );
        }
        Ok(Self {
            chunk_size,
            overlap,
        })
    }
}

impl TextSplitter for FixedSizeChunker {
    fn split(&self, text: &str) -> Result<Vec<Range<usize>>, String> {
        let length = text.chars().count();
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < length {
            let end = start.saturating_add(self.chunk_size).min(length);
            ranges.push(start..end);
            if end == length {
                break;
            }
            start = end - self.overlap;
        }
        Ok(ranges)
    }
}

impl ChunkingStrategy for FixedSizeChunker {
    fn chunk(&self, text: &str, document_id: &str) -> Result<Vec<Chunk>, String> {
        validate_document_id(document_id)?;
        let offsets = boundaries(text);
        Ok(self
            .split(text)?
            .into_iter()
            .enumerate()
            .map(|(i, range)| Chunk {
                id: chunk_id(document_id, i),
                parent_id: None,
                level: 0,
                start: range.start,
                end: range.end,
                text: text[offsets[range.start]..offsets[range.end]].to_owned(),
                is_leaf: true,
            })
            .collect())
    }
}
