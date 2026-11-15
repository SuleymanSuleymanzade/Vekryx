//! Borrowed corpora for exact ranking and MMR selection.
use crate::{
    distance::{DistanceMetric, Euclidean},
    store::{Record, Vekryx},
    Error,
};

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

    fn vectors(&self) -> impl Iterator<Item = (&String, &Vec<f64>)> {
        self.store.vectors().filter(|(id, _)| self.matches(id))
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
        self.store.validate(query)?;
        // Validate even on an empty store (notably cosine's nonzero constraint).
        metric.distance(query, query)?;
        let mut results = self
            .vectors()
            .map(|(id, vector)| Ok((id.clone(), metric.distance(query, vector)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        results.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        Ok(results)
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
        if !lambda_mult.is_finite() || !(0.0..=1.0).contains(&lambda_mult) {
            return Err("lambda_mult must be finite and between 0 and 1".into());
        }
        if fetch_k < k {
            return Err("fetch_k must be greater than or equal to k".into());
        }
        let candidates = self.search_with_metric(query, fetch_k, metric)?;
        if k == 0 || candidates.is_empty() {
            return Ok(Vec::new());
        }
        let vectors: Vec<_> = candidates
            .iter()
            .map(|(id, _)| {
                self.store
                    .get(id)
                    .expect("candidate belongs to the borrowed store")
            })
            .collect();
        let relevance: Vec<_> = vectors
            .iter()
            .map(|v| metric.similarity(query, v))
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
                redundancy[i] = redundancy[i].max(metric.similarity(vectors[i], vectors[best])?);
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
