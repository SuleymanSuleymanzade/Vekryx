//! Owned Python index snapshots avoid self-referential Rust borrows and staleness.
use super::{named_distances, parse_filter, PyVekryx};
use crate::{
    distance::{named_metric, DistanceMetric},
    index::*,
    search::Corpus,
    HybridOptions, MmrOptions, SearchOptions, Vekryx,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{de::DeserializeOwned, Deserialize};

#[derive(Deserialize)]
#[serde(
    tag = "algorithm",
    content = "config",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Config {
    Hnsw(HnswConfig),
    Ivf(IvfConfig),
    Pq(PqConfig),
    IvfPq(IvfPqConfig),
}
enum Backend {
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
    Pq(PqIndex),
    IvfPq(IvfPqIndex),
}

#[pyclass(name = "SearchIndex", module = "vekryx", frozen)]
pub(super) struct PySearchIndex {
    store: Vekryx,
    metric: Box<dyn DistanceMetric>,
    index: Backend,
}

pub(super) fn build_index(
    owner: &PyVekryx,
    config_json: &str,
    metric: &str,
    filter_json: Option<&str>,
) -> PyResult<PySearchIndex> {
    let config: Config = serde_json::from_str(config_json).map_err(value_error)?;
    let metric = named_metric(metric).map_err(value_error)?;
    let filter = parse_filter(filter_json)?;
    let store = match filter.as_ref() {
        Some(filter) => owner.store.filtered(filter),
        None => owner.store.clone(),
    };
    let corpus = store.view();
    let index = match config {
        Config::Hnsw(config) => {
            Backend::Hnsw(config.build(corpus, metric.as_ref()).map_err(value_error)?)
        }
        Config::Ivf(config) => {
            Backend::Ivf(config.build(corpus, metric.as_ref()).map_err(value_error)?)
        }
        Config::Pq(config) => {
            Backend::Pq(config.build(corpus, metric.as_ref()).map_err(value_error)?)
        }
        Config::IvfPq(config) => {
            Backend::IvfPq(config.build(corpus, metric.as_ref()).map_err(value_error)?)
        }
    };
    Ok(PySearchIndex {
        store,
        metric,
        index,
    })
}

#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum Method {
    Vector,
    Mmr {
        lambda_mult: f64,
    },
    Hybrid {
        text: String,
        vector_weight: f64,
        rrf_k: usize,
        keyword_candidates: Option<usize>,
    },
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
fn execute<I: CandidateIndex>(
    index: &I,
    corpus: Corpus<'_>,
    metric: &dyn DistanceMetric,
    query: &[f64],
    options: &str,
    method: &Method,
) -> PyResult<Vec<(String, f64, bool)>>
where
    I::SearchOptions: DeserializeOwned,
{
    let options: SearchOptions<I::SearchOptions> =
        serde_json::from_str(options).map_err(value_error)?;
    let session = SearchSession {
        corpus,
        metric,
        index,
    };
    match method {
        Method::Vector => session
            .search(query, &options)
            .map(named_distances)
            .map_err(value_error),
        Method::Mmr { lambda_mult } => session
            .mmr_search(
                query,
                &options,
                &MmrOptions {
                    lambda_mult: *lambda_mult,
                },
            )
            .map(named_distances)
            .map_err(value_error),
        Method::Hybrid {
            text,
            vector_weight,
            rrf_k,
            keyword_candidates,
        } => {
            let hybrid = HybridOptions {
                vector_weight: *vector_weight,
                rrf_k: *rrf_k,
                keyword_candidates: *keyword_candidates,
            };
            session
                .hybrid_search(text, query, &options, &hybrid)
                .map(|hits| {
                    hits.into_iter()
                        .map(|h| (h.id, h.fusion_score, false))
                        .collect()
                })
                .map_err(value_error)
        }
    }
}

#[pymethods]
impl PySearchIndex {
    fn __len__(&self) -> usize {
        self.store.len()
    }
    #[getter]
    fn guarantee(&self) -> &'static str {
        "approximate"
    }
    /// The public Python facade supplies typed options, then this thin adapter
    /// dispatches to the same generic Rust session pipelines.
    fn query(
        &self,
        query: Vec<f64>,
        options_json: &str,
        method_json: &str,
    ) -> PyResult<Vec<(String, f64, bool)>> {
        let method: Method = serde_json::from_str(method_json).map_err(value_error)?;
        let corpus = self.store.view();
        match &self.index {
            Backend::Hnsw(index) => execute(
                index,
                corpus,
                self.metric.as_ref(),
                &query,
                options_json,
                &method,
            ),
            Backend::Ivf(index) => execute(
                index,
                corpus,
                self.metric.as_ref(),
                &query,
                options_json,
                &method,
            ),
            Backend::Pq(index) => execute(
                index,
                corpus,
                self.metric.as_ref(),
                &query,
                options_json,
                &method,
            ),
            Backend::IvfPq(index) => execute(
                index,
                corpus,
                self.metric.as_ref(),
                &query,
                options_json,
                &method,
            ),
        }
    }
}
