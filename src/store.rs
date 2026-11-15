use crate::distance::{DistanceMetric, Euclidean};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Store {
    dimensions: usize,
    vectors: BTreeMap<String, Vec<f64>>,
}

impl Store {
    pub fn new(dimensions: usize) -> Result<Self, String> {
        if dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
        }
        Ok(Self {
            dimensions,
            vectors: BTreeMap::new(),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn get(&self, id: &str) -> Option<&Vec<f64>> {
        self.vectors.get(id)
    }
    pub fn delete(&mut self, id: &str) -> bool {
        self.vectors.remove(id).is_some()
    }

    fn validate(&self, vector: &[f64]) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err("vector values must be finite".into());
        }
        Ok(())
    }

    pub fn insert(&mut self, id: String, vector: Vec<f64>) -> Result<(), String> {
        if id.is_empty() {
            return Err("id must not be empty".into());
        }
        self.validate(&vector)?;
        self.vectors.insert(id, vector);
        Ok(())
    }

    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(String, f64)>, String> {
        self.search_with_metric(query, k, &Euclidean)
    }

    pub fn search_with_metric(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, String> {
        self.validate(query)?;
        // Validate even on an empty store (notably cosine's nonzero constraint).
        metric.distance(query, query)?;
        let mut results = self
            .vectors
            .iter()
            .map(|(id, vector)| Ok((id.clone(), metric.distance(query, vector)?)))
            .collect::<Result<Vec<_>, String>>()?;
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
    ) -> Result<Vec<(String, f64)>, String> {
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
            .map(|(id, _)| self.vectors[id].as_slice())
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

    /// Save a JSON snapshot at the given path; its parent directory must exist.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        crate::snapshot::save(path.as_ref(), json.as_bytes())
    }

    /// Load a snapshot. Invalid contents produce an InvalidData I/O error.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let store: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        if store.dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
        }
        for (id, vector) in &store.vectors {
            if id.is_empty() {
                return Err("id must not be empty".into());
            }
            store.validate(vector)?;
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::Store;

    #[test]
    fn disk_snapshots_replace_round_trip_and_report_invalid_data() {
        use std::{
            fs,
            io::ErrorKind,
            time::{SystemTime, UNIX_EPOCH},
        };
        let directory = std::env::temp_dir().join(format!(
            "vector-store-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("vectors.json");
        let mut store = Store::new(2).unwrap();
        store.insert("item".into(), vec![1.0, 2.0]).unwrap();
        store.save(&path).unwrap();
        store.insert("item".into(), vec![3.0, 4.0]).unwrap();
        store.save(&path).unwrap();
        assert_eq!(
            Store::load(&path).unwrap().get("item").unwrap(),
            &vec![3.0, 4.0]
        );
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::write(&path, "invalid JSON").unwrap();
        assert_eq!(
            Store::load(&path).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            Store::load(directory.join("absent")).unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert!(store.save(directory.join("absent/vectors.json")).is_err());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn cosine_and_mmr_rank_diverse_results_and_validate_options() {
        use crate::distance::{Cosine, Euclidean};
        let mut store = Store::new(2).unwrap();
        store.insert("a".into(), vec![1.0, 0.0]).unwrap();
        store.insert("b".into(), vec![1.0, 0.0]).unwrap();
        store.insert("c".into(), vec![0.6, 0.8]).unwrap();
        let query = [1.0, 0.0];
        let nearest = store.search_with_metric(&query, 3, &Cosine).unwrap();
        assert_eq!(nearest[0], ("a".into(), 0.0));
        assert_eq!(
            store.mmr_search(&query, 3, 3, 1.0, &Cosine).unwrap(),
            nearest
        );
        assert_eq!(
            store.mmr_search(&query, 2, 3, 0.2, &Cosine).unwrap()[1].0,
            "c"
        );
        assert_eq!(
            store.mmr_search(&query, 2, 3, 0.0, &Cosine).unwrap()[1].0,
            "c"
        );
        assert_eq!(
            store.mmr_search(&query, 2, 2, 0.2, &Cosine).unwrap()[1].0,
            "b"
        );
        assert_eq!(
            store.mmr_search(&query, 2, 3, 0.2, &Euclidean).unwrap()[1].0,
            "c"
        );
        assert!(store
            .mmr_search(&query, 0, 0, 0.5, &Cosine)
            .unwrap()
            .is_empty());
        for lambda in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(store.mmr_search(&query, 2, 3, lambda, &Cosine).is_err());
        }
        assert!(store.mmr_search(&query, 3, 2, 0.5, &Cosine).is_err());
        assert!(Store::new(2)
            .unwrap()
            .search_with_metric(&[0.0, 0.0], 5, &Cosine)
            .is_err());
        store.insert("zero".into(), vec![0.0, 0.0]).unwrap();
        assert!(store.search_with_metric(&query, 3, &Cosine).is_err());
    }

    #[test]
    fn mmr_handles_negative_similarity_and_returns_selection_order() {
        use crate::distance::Cosine;
        let mut store = Store::new(2).unwrap();
        store.insert("a".into(), vec![1.0, 0.0]).unwrap();
        store.insert("b".into(), vec![-0.6, 0.8]).unwrap();
        store.insert("c".into(), vec![-1.0, 0.0]).unwrap();
        let results = store.mmr_search(&[1.0, 0.0], 3, 3, 0.0, &Cosine).unwrap();
        assert_eq!(
            results
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "c", "b"]
        );
        assert!(results[1].1 > results[2].1);
        assert!(Store::new(2)
            .unwrap()
            .mmr_search(&[1.0, 0.0], 5, 20, 0.5, &Cosine)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn custom_metrics_work_without_modifying_the_store() {
        use crate::distance::DistanceMetric;
        struct Manhattan;
        impl DistanceMetric for Manhattan {
            fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
                Ok(left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum())
            }
        }
        let mut store = Store::new(2).unwrap();
        store.insert("a".into(), vec![3.0, 0.0]).unwrap();
        store.insert("b".into(), vec![2.0, 2.0]).unwrap();
        assert_eq!(
            store
                .search_with_metric(&[0.0, 0.0], 1, &Manhattan)
                .unwrap(),
            vec![("a".into(), 3.0)]
        );
        assert_eq!(
            store
                .mmr_search(&[0.0, 0.0], 1, 2, 0.5, &Manhattan)
                .unwrap(),
            vec![("a".into(), 3.0)]
        );
    }

    #[test]
    fn ranks_preserves_collisions_and_breaks_ties_by_id() {
        let mut store = Store::new(2).unwrap();
        store.insert("b".into(), vec![1.0, 0.0]).unwrap();
        store.insert("a".into(), vec![1.0, 0.0]).unwrap();
        store.insert("c".into(), vec![3.0, 0.0]).unwrap();
        assert_eq!(
            store.search(&[0.0, 0.0], 2).unwrap(),
            vec![("a".into(), 1.0), ("b".into(), 1.0)]
        );
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn validates_updates_and_round_trips() {
        assert!(Store::new(0).is_err());
        let mut store = Store::new(2).unwrap();
        assert!(store.insert("a".into(), vec![1.0]).is_err());
        assert!(store.insert("a".into(), vec![f64::NAN, 0.0]).is_err());
        assert!(store.insert("".into(), vec![0.0, 0.0]).is_err());
        store.insert("a".into(), vec![1.0, 2.0]).unwrap();
        store.insert("a".into(), vec![3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 1);
        let mut restored = Store::from_json(&store.to_json().unwrap()).unwrap();
        assert_eq!(restored.get("a").unwrap(), &vec![3.0, 4.0]);
        assert!(restored.delete("a"));
        assert!(!restored.delete("a"));
        assert!(restored.search(&[0.0, 0.0], 5).unwrap().is_empty());
        assert!(restored.search(&[0.0], 5).is_err());
        assert!(Store::from_json(r#"{"dimensions":2,"vectors":{"a":[1.0]}}"#).is_err());
    }
}
