use crate::{chunking, distance, store::Store};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// An in-memory vector store with configurable exact search and JSON snapshots.
#[pyclass(module = "vector_database")]
struct VectorStore {
    store: Store,
}

#[pymethods]
impl VectorStore {
    #[new]
    fn new(dimensions: usize) -> PyResult<Self> {
        Ok(Self {
            store: Store::new(dimensions).map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.store.dimensions()
    }

    fn __len__(&self) -> usize {
        self.store.len()
    }

    /// Insert a vector, replacing the existing vector when the ID already exists.
    fn insert(&mut self, id: String, vector: Vec<f64>) -> PyResult<()> {
        self.store.insert(id, vector).map_err(PyValueError::new_err)
    }

    fn get(&self, id: &str) -> PyResult<Vec<f64>> {
        self.store
            .get(id)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(id.to_owned()))
    }

    fn delete(&mut self, id: &str) -> bool {
        self.store.delete(id)
    }

    /// Return (ID, distance) pairs, nearest first; ties sort by ID.
    #[pyo3(signature = (query, k=5, *, metric="euclidean"))]
    fn search(&self, query: Vec<f64>, k: usize, metric: &str) -> PyResult<Vec<(String, f64)>> {
        let metric = distance::named_metric(metric).map_err(PyValueError::new_err)?;
        self.store
            .search_with_metric(&query, k, metric.as_ref())
            .map_err(PyValueError::new_err)
    }

    /// Select relevant, diverse candidates; return query distances in MMR order.
    #[pyo3(signature = (query, k=5, *, fetch_k=20, lambda_mult=0.5, metric="cosine"))]
    fn mmr_search(
        &self,
        query: Vec<f64>,
        k: usize,
        fetch_k: usize,
        lambda_mult: f64,
        metric: &str,
    ) -> PyResult<Vec<(String, f64)>> {
        let metric = distance::named_metric(metric).map_err(PyValueError::new_err)?;
        self.store
            .mmr_search(&query, k, fetch_k, lambda_mult, metric.as_ref())
            .map_err(PyValueError::new_err)
    }

    /// Atomically replace a JSON snapshot on the same filesystem.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.store.save(path).map_err(Into::into)
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        match Store::load(path) {
            Ok(store) => Ok(Self { store }),
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                Err(PyValueError::new_err(error.to_string()))
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[pymodule]
fn vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorStore>()?;
    chunking::register(m)?;
    Ok(())
}
