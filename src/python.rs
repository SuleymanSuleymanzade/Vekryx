use crate::{
    chunking, distance,
    store::{Record, Vekryx},
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::path::PathBuf;

/// An in-memory vector store with configurable exact search and JSON snapshots.
#[pyclass(name = "Vekryx", module = "vekryx")]
struct PyVekryx {
    store: Vekryx,
}

impl PyVekryx {
    fn corpus(&self, filter_json: Option<&str>) -> PyResult<Cow<'_, Vekryx>> {
        match filter_json {
            None => Ok(Cow::Borrowed(&self.store)),
            Some(json) => {
                let filter = serde_json::from_str(json)
                    .map_err(|e| PyValueError::new_err(format!("invalid filter: {e}")))?;
                Ok(Cow::Owned(self.store.filtered(&filter)))
            }
        }
    }
}

#[pymethods]
impl PyVekryx {
    #[new]
    fn new(dimensions: usize) -> PyResult<Self> {
        Ok(Self {
            store: Vekryx::new(dimensions).map_err(PyValueError::new_err)?,
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

    #[pyo3(signature = (id, vector, text, metadata_json))]
    fn insert_record(
        &mut self,
        id: String,
        vector: Vec<f64>,
        text: Option<String>,
        metadata_json: &str,
    ) -> PyResult<()> {
        let metadata = serde_json::from_str(metadata_json)
            .map_err(|e| PyValueError::new_err(format!("invalid metadata: {e}")))?;
        self.store
            .insert_record(id, vector, Record { text, metadata })
            .map_err(PyValueError::new_err)
    }

    fn record_json(&self, id: &str) -> PyResult<String> {
        if self.store.get(id).is_none() {
            return Err(PyKeyError::new_err(id.to_owned()));
        }
        let record = self.store.record(id).cloned().unwrap_or(Record {
            text: None,
            metadata: Default::default(),
        });
        serde_json::to_string(&record).map_err(|e| PyValueError::new_err(e.to_string()))
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
    #[pyo3(signature = (query, k=5, *, metric="euclidean", filter_json=None))]
    fn search(
        &self,
        query: Vec<f64>,
        k: usize,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64)>> {
        let metric = distance::named_metric(metric).map_err(PyValueError::new_err)?;
        self.corpus(filter_json)?
            .search_with_metric(&query, k, metric.as_ref())
            .map_err(PyValueError::new_err)
    }

    /// Select relevant, diverse candidates; return query distances in MMR order.
    #[pyo3(signature = (query, k=5, *, fetch_k=20, lambda_mult=0.5, metric="cosine", filter_json=None))]
    fn mmr_search(
        &self,
        query: Vec<f64>,
        k: usize,
        fetch_k: usize,
        lambda_mult: f64,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64)>> {
        let metric = distance::named_metric(metric).map_err(PyValueError::new_err)?;
        self.corpus(filter_json)?
            .mmr_search(&query, k, fetch_k, lambda_mult, metric.as_ref())
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (text, query, k=5, *, metric="cosine", vector_weight=0.5, rrf_k=60, filter_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_search(
        &self,
        text: &str,
        query: Vec<f64>,
        k: usize,
        metric: &str,
        vector_weight: f64,
        rrf_k: usize,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64)>> {
        let metric = distance::named_metric(metric).map_err(PyValueError::new_err)?;
        self.corpus(filter_json)?
            .hybrid_search(text, &query, k, metric.as_ref(), vector_weight, rrf_k)
            .map_err(PyValueError::new_err)
    }

    /// Atomically replace a JSON snapshot on the same filesystem.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.store.save(path).map_err(Into::into)
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        match Vekryx::load(path) {
            Ok(store) => Ok(Self { store }),
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                Err(PyValueError::new_err(error.to_string()))
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVekryx>()?;
    chunking::register(m)?;
    Ok(())
}
