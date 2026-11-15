use super::{Chunk, ChunkingStrategy, FixedSizeChunker, HierarchicalChunker};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "Chunk", module = "vector_database", frozen, get_all)]
struct PyChunk {
    id: String,
    parent_id: Option<String>,
    level: usize,
    start: usize,
    end: usize,
    text: String,
    is_leaf: bool,
}

impl From<Chunk> for PyChunk {
    fn from(c: Chunk) -> Self {
        Self {
            id: c.id,
            parent_id: c.parent_id,
            level: c.level,
            start: c.start,
            end: c.end,
            text: c.text,
            is_leaf: c.is_leaf,
        }
    }
}

fn chunk(strategy: &dyn ChunkingStrategy, text: &str, document_id: &str) -> PyResult<Vec<PyChunk>> {
    Ok(strategy
        .chunk(text, document_id)
        .map_err(PyValueError::new_err)?
        .into_iter()
        .map(Into::into)
        .collect())
}

#[pyclass(name = "FixedSizeChunker", module = "vector_database")]
struct PyFixedSizeChunker {
    strategy: FixedSizeChunker,
}

#[pymethods]
impl PyFixedSizeChunker {
    #[new]
    #[pyo3(signature = (chunk_size=512, overlap=0))]
    fn new(chunk_size: usize, overlap: usize) -> PyResult<Self> {
        Ok(Self {
            strategy: FixedSizeChunker::new(chunk_size, overlap).map_err(PyValueError::new_err)?,
        })
    }

    #[pyo3(signature = (text, document_id="document"))]
    fn chunk(&self, text: &str, document_id: &str) -> PyResult<Vec<PyChunk>> {
        chunk(&self.strategy, text, document_id)
    }
}

#[pyclass(name = "HierarchicalChunker", module = "vector_database")]
struct PyHierarchicalChunker {
    strategy: HierarchicalChunker,
}

#[pymethods]
impl PyHierarchicalChunker {
    #[new]
    #[pyo3(signature = (chunk_sizes=None, overlap=0))]
    fn new(chunk_sizes: Option<Vec<usize>>, overlap: usize) -> PyResult<Self> {
        Ok(Self {
            strategy: HierarchicalChunker::new(
                chunk_sizes.unwrap_or_else(|| vec![1024, 256]),
                overlap,
            )
            .map_err(PyValueError::new_err)?,
        })
    }

    #[pyo3(signature = (text, document_id="document"))]
    fn chunk(&self, text: &str, document_id: &str) -> PyResult<Vec<PyChunk>> {
        chunk(&self.strategy, text, document_id)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChunk>()?;
    m.add_class::<PyFixedSizeChunker>()?;
    m.add_class::<PyHierarchicalChunker>()?;
    Ok(())
}
