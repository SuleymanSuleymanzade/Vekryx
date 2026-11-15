//! Text splitting is independent of embeddings, vector storage, and retrieval.
use crate::Error;
mod fixed_size;
mod hierarchical;
#[cfg(feature = "python")]
mod python;

pub use fixed_size::FixedSizeChunker;
pub use hierarchical::HierarchicalChunker;
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub id: String,
    pub parent_id: Option<String>,
    pub level: usize,
    /// Unicode scalar-value offsets into the original document, end exclusive.
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub is_leaf: bool,
}

/// A replaceable, flat splitter. Return nonempty character ranges within text.
/// Preserve document order; offsets count Unicode scalar values, not UTF-8 bytes.
pub trait TextSplitter: Send + Sync {
    fn split(&self, text: &str) -> Result<Vec<Range<usize>>, Error>;
}

/// Produces document-scoped records, independently of how embeddings are made.
pub trait ChunkingStrategy: Send + Sync {
    fn chunk(&self, text: &str, document_id: &str) -> Result<Vec<Chunk>, Error>;
}

pub(crate) fn boundaries(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect()
}

pub(crate) fn validate_document_id(document_id: &str) -> Result<(), Error> {
    if document_id.is_empty() {
        return Err("document_id must not be empty".into());
    }
    Ok(())
}

pub(crate) fn validate_range(range: &Range<usize>, length: usize) -> Result<(), Error> {
    if range.start >= range.end || range.end > length {
        return Err("splitter returned an empty or out-of-bounds character range".into());
    }
    Ok(())
}

pub(crate) fn chunk_id(document_id: &str, index: usize) -> String {
    format!("{document_id}:chunk:{index}")
}

#[cfg(feature = "python")]
pub(crate) use python::register;

#[cfg(test)]
mod tests;
