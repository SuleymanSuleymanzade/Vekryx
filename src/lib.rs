mod store;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::fs;
use std::path::PathBuf;
#[path = "state_manager/snapshot.rs"]
mod snapshot;
use store::Store;

/// An in-memory vector store with exact Euclidean search and JSON snapshots.
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

    /// Return (ID, Euclidean distance) pairs, nearest first; ties sort by ID.
    #[pyo3(signature = (query, k=5))]
    fn search(&self, query: Vec<f64>, k: usize) -> PyResult<Vec<(String, f64)>> {
        self.store.search(&query, k).map_err(PyValueError::new_err)
    }

    /// Atomically replace a JSON snapshot on the same filesystem.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let json = self
            .store
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        snapshot::save(&path, json.as_bytes())?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let json = fs::read_to_string(path)?;
        Ok(Self {
            store: Store::from_json(&json).map_err(PyValueError::new_err)?,
        })
    }
}

#[pymodule]
fn vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorStore>()?;
    Ok(())
}
