use crate::{
    chunking, distance,
    index::DistanceKind,
    store::{Corpus, Record, Vekryx},
    HybridOptions, MmrOptions, SearchHit, SearchOptions, SnapshotFormat,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;
mod index;

/// An in-memory vector store with configurable exact search and JSON snapshots.
#[pyclass(name = "Vekryx", module = "vekryx")]
struct PyVekryx {
    store: Vekryx,
}

fn parse_filter(
    json: Option<&str>,
) -> PyResult<Option<serde_json::Map<String, serde_json::Value>>> {
    json.map(|json| {
        serde_json::from_str(json)
            .map_err(|error| PyValueError::new_err(format!("invalid filter: {error}")))
    })
    .transpose()
}

impl PyVekryx {
    fn corpus<'a>(
        &'a self,
        filter: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Corpus<'a> {
        match filter {
            None => self.store.view(),
            Some(filter) => self.store.filtered_view(filter),
        }
    }
}

#[pymethods]
impl PyVekryx {
    #[new]
    fn new(dimensions: usize) -> PyResult<Self> {
        Ok(Self {
            store: Vekryx::new(dimensions)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.store.dimensions()
    }

    fn __len__(&self) -> usize {
        self.store.len()
    }

    #[pyo3(signature = (config_json, *, metric="euclidean", filter_json=None))]
    fn build_index(
        &self,
        config_json: &str,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<index::PySearchIndex> {
        index::build_index(self, config_json, metric, filter_json)
    }

    /// Insert a vector, replacing the existing vector when the ID already exists.
    fn insert(&mut self, id: String, vector: Vec<f64>) -> PyResult<()> {
        self.store
            .insert(id, vector)
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
            .map(<[f64]>::to_vec)
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
        let metric = distance::named_metric(metric)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let filter = parse_filter(filter_json)?;
        self.corpus(filter.as_ref())
            .search_with_metric(&query, k, metric.as_ref())
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
        let metric = distance::named_metric(metric)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let filter = parse_filter(filter_json)?;
        self.corpus(filter.as_ref())
            .mmr_search(&query, k, fetch_k, lambda_mult, metric.as_ref())
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
        let metric = distance::named_metric(metric)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let filter = parse_filter(filter_json)?;
        self.corpus(filter.as_ref())
            .hybrid_search(text, &query, k, metric.as_ref(), vector_weight, rrf_k)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyo3(signature = (query, k, candidates, rerank_exact, *, metric="euclidean", filter_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn search_with_options(
        &self,
        query: Vec<f64>,
        k: usize,
        candidates: usize,
        rerank_exact: bool,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64, bool)>> {
        let metric =
            distance::named_metric(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let filter = parse_filter(filter_json)?;
        let options = SearchOptions {
            k,
            candidates,
            rerank_exact,
            backend: (),
        };
        self.corpus(filter.as_ref())
            .search_with_options(&query, metric.as_ref(), &options)
            .map(named_distances)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (query, k, candidates, rerank_exact, lambda_mult, *, metric="cosine", filter_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn mmr_search_with_options(
        &self,
        query: Vec<f64>,
        k: usize,
        candidates: usize,
        rerank_exact: bool,
        lambda_mult: f64,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64, bool)>> {
        let metric =
            distance::named_metric(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let filter = parse_filter(filter_json)?;
        let options = SearchOptions {
            k,
            candidates,
            rerank_exact,
            backend: (),
        };
        self.corpus(filter.as_ref())
            .mmr_search_with_options(
                &query,
                metric.as_ref(),
                &options,
                &MmrOptions { lambda_mult },
            )
            .map(named_distances)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (text, query, k, candidates, rerank_exact, vector_weight, rrf_k, keyword_candidates=None, *, metric="cosine", filter_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_search_with_options(
        &self,
        text: &str,
        query: Vec<f64>,
        k: usize,
        candidates: usize,
        rerank_exact: bool,
        vector_weight: f64,
        rrf_k: usize,
        keyword_candidates: Option<usize>,
        metric: &str,
        filter_json: Option<&str>,
    ) -> PyResult<Vec<(String, f64)>> {
        let metric =
            distance::named_metric(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let filter = parse_filter(filter_json)?;
        let options = SearchOptions {
            k,
            candidates,
            rerank_exact,
            backend: (),
        };
        let hybrid = HybridOptions {
            vector_weight,
            rrf_k,
            keyword_candidates,
        };
        self.corpus(filter.as_ref())
            .hybrid_search_with_options(text, &query, metric.as_ref(), &options, &hybrid)
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| (hit.id, hit.fusion_score))
                    .collect()
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Atomically replace a JSON snapshot on the same filesystem.
    #[pyo3(signature = (path, *, format="json"))]
    fn save(&self, path: PathBuf, format: &str) -> PyResult<()> {
        self.store
            .save_with_format(path, snapshot_format(format)?)
            .map_err(Into::into)
    }

    fn save_mapped(&self, path: PathBuf) -> PyResult<()> {
        self.store.save_mapped(path).map_err(Into::into)
    }

    #[staticmethod]
    #[pyo3(signature = (path, *, format="json"))]
    fn load(path: PathBuf, format: &str) -> PyResult<Self> {
        match Vekryx::load_with_format(path, snapshot_format(format)?) {
            Ok(store) => Ok(Self { store }),
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                Err(PyValueError::new_err(error.to_string()))
            }
            Err(error) => Err(error.into()),
        }
    }
}

fn snapshot_format(format: &str) -> PyResult<SnapshotFormat> {
    match format {
        "json" => Ok(SnapshotFormat::Json),
        "compressed_json" => Ok(SnapshotFormat::CompressedJson),
        _ => Err(PyValueError::new_err(
            "format must be 'json' or 'compressed_json'",
        )),
    }
}

fn named_distances(hits: Vec<SearchHit>) -> Vec<(String, f64, bool)> {
    hits.into_iter()
        .map(|hit| {
            (
                hit.id,
                hit.distance,
                hit.distance_kind == DistanceKind::Exact,
            )
        })
        .collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVekryx>()?;
    m.add_class::<index::PySearchIndex>()?;
    chunking::register(m)?;
    Ok(())
}
