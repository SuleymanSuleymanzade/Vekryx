use super::{
    boundaries, chunk_id, validate_document_id, validate_range, Chunk, ChunkingStrategy,
    FixedSizeChunker, TextSplitter,
};

pub struct HierarchicalChunker {
    levels: Vec<Box<dyn TextSplitter>>,
}

impl HierarchicalChunker {
    /// Sizes must strictly decrease; overlap applies independently at each level.
    pub fn new(chunk_sizes: Vec<usize>, overlap: usize) -> Result<Self, String> {
        if chunk_sizes.is_empty() || chunk_sizes.windows(2).any(|pair| pair[0] <= pair[1]) {
            return Err("chunk_sizes must be nonempty and strictly decreasing".into());
        }
        let levels = chunk_sizes
            .into_iter()
            .map(
                |size| Ok(Box::new(FixedSizeChunker::new(size, overlap)?) as Box<dyn TextSplitter>),
            )
            .collect::<Result<Vec<_>, String>>()?;
        Self::from_splitters(levels)
    }

    /// Compose custom flat splitters, e.g. sections at one level and sentences next.
    pub fn from_splitters(levels: Vec<Box<dyn TextSplitter>>) -> Result<Self, String> {
        if levels.is_empty() {
            return Err("at least one splitter is required".into());
        }
        Ok(Self { levels })
    }
}

impl ChunkingStrategy for HierarchicalChunker {
    fn chunk(&self, text: &str, document_id: &str) -> Result<Vec<Chunk>, String> {
        validate_document_id(document_id)?;
        let offsets = boundaries(text);
        let mut chunks: Vec<Chunk> = Vec::new();
        // Breadth-first output, with local child offsets translated to document offsets.
        let mut parents: Vec<(Option<usize>, usize, usize)> = vec![(None, 0, text.chars().count())];
        for (level, splitter) in self.levels.iter().enumerate() {
            let mut next = Vec::new();
            for (parent_index, start, end) in parents {
                let parent_text = &text[offsets[start]..offsets[end]];
                let ranges = splitter.split(parent_text)?;
                if let Some(index) = parent_index {
                    if !ranges.is_empty() {
                        chunks[index].is_leaf = false;
                    }
                }
                for range in ranges {
                    validate_range(&range, end - start)?;
                    let child_start = start + range.start;
                    let child_end = start + range.end;
                    let index = chunks.len();
                    chunks.push(Chunk {
                        id: chunk_id(document_id, index),
                        parent_id: parent_index.map(|i| chunks[i].id.clone()),
                        level,
                        start: child_start,
                        end: child_end,
                        text: text[offsets[child_start]..offsets[child_end]].to_owned(),
                        is_leaf: true,
                    });
                    next.push((Some(index), child_start, child_end));
                }
            }
            parents = next;
        }
        Ok(chunks)
    }
}
