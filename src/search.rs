//! Borrowed corpora and backend-independent vector/MMR search pipelines.
use crate::{
    distance::{DistanceMetric, Euclidean},
    index::{
        validate_distance, CandidateIndex, DistanceKind, ExactSearch, IndexBuilder, SearchSession,
    },
    store::{Record, Vekryx},
    Error,
};
use std::collections::BTreeMap;

/// Named vector result. MMR returns these in selection order.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: String,
    pub distance: f64,
    pub distance_kind: DistanceKind,
}

/// Shared retrieval settings. Backend-specific query knobs have their own type.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchOptions<O = ()> {
    pub k: usize,
    /// Candidate pool before reranking, MMR selection, or hybrid fusion.
    pub candidates: usize,
    /// Recompute approximate distances using canonical vectors. Does not
    /// guarantee recall. Already exact backend distances need no recomputation.
    pub rerank_exact: bool,
    pub backend: O,
}

impl<O: Default> SearchOptions<O> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            candidates: k,
            rerank_exact: true,
            backend: O::default(),
        }
    }
}

impl<O> SearchOptions<O> {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.candidates < self.k {
            return Err("candidates must be greater than or equal to k".into());
        }
        Ok(())
    }
}

/// MMR selection settings, independent of how candidates are retrieved.
#[derive(Debug, Clone, Copy)]
pub struct MmrOptions {
    pub lambda_mult: f64,
}

impl Default for MmrOptions {
    fn default() -> Self {
        Self { lambda_mult: 0.5 }
    }
}

pub(crate) fn distance_pairs(hits: Vec<SearchHit>) -> Vec<(String, f64)> {
    hits.into_iter().map(|hit| (hit.id, hit.distance)).collect()
}

/// A read-only search corpus borrowing its vectors and records from a store.
#[derive(Clone, Copy)]
pub struct Corpus<'a> {
    store: &'a Vekryx,
    filter: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

impl<'a> Corpus<'a> {
    pub(crate) fn new(
        store: &'a Vekryx,
        filter: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Self {
        Self { store, filter }
    }

    fn matches(&self, id: &str) -> bool {
        self.filter.is_none_or(|filter| {
            filter.iter().all(|(key, value)| {
                self.store
                    .record(id)
                    .and_then(|record| record.metadata.get(key))
                    == Some(value)
            })
        })
    }

    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Iterate eligible IDs and borrowed vectors without exposing storage layout.
    pub fn vectors(&self) -> impl Iterator<Item = (&str, &[f64])> {
        self.store
            .vectors()
            .filter(|(id, _)| self.matches(id))
            .map(|(id, vector)| (id.as_str(), vector.as_slice()))
    }

    pub fn dimensions(&self) -> usize {
        self.store.dimensions()
    }

    /// Lookup within this corpus, respecting its metadata filter.
    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.matches(id).then(|| self.store.get(id)).flatten()
    }

    /// Build/train once and bind to this immutable corpus and metric.
    pub fn build_index<B: IndexBuilder>(
        self,
        builder: &B,
        metric: &'a dyn DistanceMetric,
    ) -> Result<SearchSession<'a, B::Index>, Error> {
        let index = builder.build(self, metric)?;
        Ok(SearchSession {
            corpus: self,
            metric,
            index,
        })
    }

    pub fn records(&self) -> impl Iterator<Item = (&String, &Record)> {
        self.store.records().filter(|(id, _)| self.matches(id))
    }

    pub fn len(&self) -> usize {
        self.vectors().count()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors().next().is_none()
    }

    pub(crate) fn validate(&self, query: &[f64]) -> Result<(), Error> {
        self.store.validate(query)
    }

    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(String, f64)>, Error> {
        self.search_with_metric(query, k, &Euclidean)
    }

    pub fn search_with_metric(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, Error> {
        self.search_with_options(query, metric, &SearchOptions::new(k))
            .map(distance_pairs)
    }

    /// Exact backend with named results; build a session to select another backend.
    pub fn search_with_options(
        &self,
        query: &[f64],
        metric: &dyn DistanceMetric,
        options: &SearchOptions,
    ) -> Result<Vec<SearchHit>, Error> {
        self.build_index(&ExactSearch, metric)?
            .search(query, options)
    }

    /// Greedy MMR reranking of the nearest `fetch_k` candidates.
    /// Returns original query distances in selection order, not MMR scores.
    pub fn mmr_search(
        &self,
        query: &[f64],
        k: usize,
        fetch_k: usize,
        lambda_mult: f64,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, Error> {
        self.mmr_search_with_options(
            query,
            metric,
            &SearchOptions {
                candidates: fetch_k,
                ..SearchOptions::new(k)
            },
            &MmrOptions { lambda_mult },
        )
        .map(distance_pairs)
    }

    pub fn mmr_search_with_options(
        &self,
        query: &[f64],
        metric: &dyn DistanceMetric,
        options: &SearchOptions,
        mmr: &MmrOptions,
    ) -> Result<Vec<SearchHit>, Error> {
        self.build_index(&ExactSearch, metric)?
            .mmr_search(query, options, mmr)
    }
}

impl<I: CandidateIndex> SearchSession<'_, I> {
    pub(crate) fn validate_query(&self, query: &[f64]) -> Result<(), Error> {
        self.corpus.validate(query)?;
        // Validate even for an empty corpus or zero k (cosine zero queries).
        validate_distance(self.metric.distance(query, query)?)
    }

    pub(crate) fn retrieve(
        &self,
        query: &[f64],
        options: &SearchOptions<I::SearchOptions>,
    ) -> Result<Vec<SearchHit>, Error> {
        options.validate()?;
        self.validate_query(query)?;
        let candidates = self.index.candidates(
            self.corpus,
            query,
            self.metric,
            options.candidates,
            &options.backend,
        )?;
        if candidates.len() > options.candidates {
            return Err("backend exceeded the requested candidate limit".into());
        }
        let mut unique: BTreeMap<String, SearchHit> = BTreeMap::new();
        for candidate in candidates {
            validate_distance(candidate.distance)?;
            // Defense in depth: never return missing or ineligible records.
            let vector = self
                .corpus
                .get(&candidate.id)
                .ok_or("backend returned an unknown or filtered-out ID")?;
            let (distance, distance_kind) =
                if options.rerank_exact && candidate.distance_kind == DistanceKind::Approximate {
                    let distance = self.metric.distance(query, vector)?;
                    validate_distance(distance)?;
                    (distance, DistanceKind::Exact)
                } else {
                    (candidate.distance, candidate.distance_kind)
                };
            let hit = SearchHit {
                id: candidate.id,
                distance,
                distance_kind,
            };
            // Duplicate IDs cannot occupy multiple result slots. Prefer the
            // lowest reported distance, and exact distances on equal estimates.
            unique
                .entry(hit.id.clone())
                .and_modify(|current| {
                    if hit.distance < current.distance
                        || (hit.distance == current.distance
                            && hit.distance_kind == DistanceKind::Exact)
                    {
                        *current = hit.clone();
                    }
                })
                .or_insert(hit);
        }
        let mut hits: Vec<_> = unique.into_values().collect();
        hits.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(hits)
    }

    pub fn search(
        &self,
        query: &[f64],
        options: &SearchOptions<I::SearchOptions>,
    ) -> Result<Vec<SearchHit>, Error> {
        let mut hits = self.retrieve(query, options)?;
        hits.truncate(options.k);
        Ok(hits)
    }

    /// Diversify retrieved candidates. Output distances keep their precision tag.
    pub fn mmr_search(
        &self,
        query: &[f64],
        options: &SearchOptions<I::SearchOptions>,
        mmr: &MmrOptions,
    ) -> Result<Vec<SearchHit>, Error> {
        let lambda_mult = mmr.lambda_mult;
        if !lambda_mult.is_finite() || !(0.0..=1.0).contains(&lambda_mult) {
            return Err("lambda_mult must be finite and between 0 and 1".into());
        }
        let k = options.k;
        let candidates = self.retrieve(query, options)?;
        if k == 0 || candidates.is_empty() {
            return Ok(Vec::new());
        }
        let vectors: Vec<_> = candidates
            .iter()
            .map(|hit| {
                self.corpus
                    .get(&hit.id)
                    .expect("candidate belongs to the borrowed store")
            })
            .collect();
        let relevance: Vec<_> = vectors
            .iter()
            .map(|v| self.metric.similarity(query, v))
            .collect::<Result<_, _>>()?;
        let mut selected = vec![false; candidates.len()];
        let mut redundancy = vec![f64::NEG_INFINITY; candidates.len()];
        let mut results = Vec::new();
        // Seed with the nearest record, including when lambda is zero.
        let mut best = 0;
        for _ in 0..k.min(candidates.len()) {
            selected[best] = true;
            results.push(candidates[best].clone());
            let mut next = None;
            for i in 0..candidates.len() {
                if selected[i] {
                    continue;
                }
                redundancy[i] =
                    redundancy[i].max(self.metric.similarity(vectors[i], vectors[best])?);
                let score = lambda_mult * relevance[i] - (1.0 - lambda_mult) * redundancy[i];
                // Candidate order gives deterministic distance/ID tie breaking.
                if next.is_none_or(|(_, best_score)| score > best_score) {
                    next = Some((i, score));
                }
            }
            match next {
                Some((i, _)) => best = i,
                None => break,
            }
        }
        Ok(results)
    }
}
