//! Candidate retrieval independently of storage, compression, and final ranking.
//!
//! Build through [`Corpus::build_index`]. A [`SearchSession`] borrows its corpus
//! and metric, preventing store mutations for the session's lifetime. Builders
//! own construction/training settings; each index chooses typed query options.
use crate::{distance::DistanceMetric, search::Corpus, Error};
mod hnsw;
mod ivf;
mod pq;
#[cfg(test)]
mod tests;
mod training;
pub use hnsw::{HnswConfig, HnswIndex, HnswSearchOptions};
pub use ivf::{IvfConfig, IvfIndex, IvfSearchOptions};
pub use pq::{IvfPqConfig, IvfPqIndex, PqConfig, PqIndex};

/// Whether candidate selection guarantees the nearest eligible records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalGuarantee {
    Exact,
    Approximate,
}

/// Precision of a reported distance, independently of candidate recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceKind {
    Exact,
    Approximate,
}

/// A backend's candidate. Distances use the bound metric's units, smaller first.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub distance: f64,
    pub distance_kind: DistanceKind,
}

/// Construction/training settings for a search backend.
///
/// Validate supported dimensions and metrics here. An implementation may train
/// codebooks or build graph/list structures, retaining stable IDs rather than
/// taking ownership of canonical records. The returned index is session-local.
pub trait IndexBuilder {
    type Index: CandidateIndex;

    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<Self::Index, Error>;
}

/// Replaceable candidate retrieval, not a complete search/reranking pipeline.
pub trait CandidateIndex {
    type SearchOptions: Default;

    /// Default conservatively; exact reranking never upgrades this guarantee.
    fn guarantee(&self) -> RetrievalGuarantee {
        RetrievalGuarantee::Approximate
    }

    /// Return at most `limit` unique, eligible IDs, applying the corpus filter
    /// before the limit. Distances must be finite and nonnegative. Exact indexes
    /// return the true top `limit`, breaking ties by ID. Approximate backends may
    /// underfill; the pipeline never silently fills from an exact scan.
    ///
    /// The session supplies the same corpus and metric used during construction.
    /// Implementations may internally compose an encoder (for example PQ) with
    /// their retrieval structure. Do not expose compressed codes as record IDs.
    fn candidates(
        &self,
        corpus: Corpus<'_>,
        query: &[f64],
        metric: &dyn DistanceMetric,
        limit: usize,
        options: &Self::SearchOptions,
    ) -> Result<Vec<Candidate>, Error>;
}

impl<I: CandidateIndex> CandidateIndex for &I {
    type SearchOptions = I::SearchOptions;
    fn guarantee(&self) -> RetrievalGuarantee {
        (**self).guarantee()
    }
    fn candidates(
        &self,
        corpus: Corpus<'_>,
        query: &[f64],
        metric: &dyn DistanceMetric,
        limit: usize,
        options: &Self::SearchOptions,
    ) -> Result<Vec<Candidate>, Error> {
        (**self).candidates(corpus, query, metric, limit, options)
    }
}

fn rank_candidates(candidates: &mut Vec<Candidate>, limit: usize) {
    candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    candidates.truncate(limit);
}

/// Exact scan builder/backend. No training and no backend-specific query knobs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactSearch;

impl IndexBuilder for ExactSearch {
    type Index = Self;

    fn build(&self, _corpus: Corpus<'_>, _metric: &dyn DistanceMetric) -> Result<Self, Error> {
        Ok(*self)
    }
}

impl CandidateIndex for ExactSearch {
    type SearchOptions = ();

    fn guarantee(&self) -> RetrievalGuarantee {
        RetrievalGuarantee::Exact
    }

    fn candidates(
        &self,
        corpus: Corpus<'_>,
        query: &[f64],
        metric: &dyn DistanceMetric,
        limit: usize,
        _options: &(),
    ) -> Result<Vec<Candidate>, Error> {
        exact_candidates(corpus.vectors().map(Ok), query, metric, limit)
    }
}

pub(crate) fn exact_candidates<'a>(
    vectors: impl Iterator<Item = Result<(&'a str, &'a [f64]), Error>>,
    query: &[f64],
    metric: &dyn DistanceMetric,
    limit: usize,
) -> Result<Vec<Candidate>, Error> {
    let mut candidates = vectors
        .map(|entry| {
            let (id, vector) = entry?;
            let distance = metric.distance(query, vector)?;
            validate_distance(distance)?;
            Ok((id, distance))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));
    candidates.truncate(limit);
    Ok(candidates
        .into_iter()
        .map(|(id, distance)| Candidate {
            id: id.to_owned(),
            distance,
            distance_kind: DistanceKind::Exact,
        })
        .collect())
}

/// Immutable index bound to one corpus and metric; not part of JSON snapshots.
///
/// Drop the session before mutating the store and build a new session afterward.
/// This deliberately uses rebuilds rather than promising incremental updates for
/// every future backend. No stale index fallback or metric switching is implicit.
///
/// ```compile_fail
/// use vekryx::{distance::Euclidean, index::ExactSearch, SearchOptions, Vekryx};
/// let mut store = Vekryx::new(2).unwrap();
/// let session = store.view().build_index(&ExactSearch, &Euclidean).unwrap();
/// store.insert("new".into(), vec![1.0, 0.0]).unwrap();
/// session.search(&[1.0, 0.0], &SearchOptions::new(1)).unwrap();
/// ```
pub struct SearchSession<'a, I: CandidateIndex> {
    pub(crate) corpus: Corpus<'a>,
    pub(crate) metric: &'a dyn DistanceMetric,
    pub(crate) index: I,
}

impl<I: CandidateIndex> SearchSession<'_, I> {
    pub fn guarantee(&self) -> RetrievalGuarantee {
        self.index.guarantee()
    }
}

pub(crate) fn validate_distance(distance: f64) -> Result<(), Error> {
    if !distance.is_finite() || distance < 0.0 {
        return Err("distance must be finite and nonnegative".into());
    }
    Ok(())
}
