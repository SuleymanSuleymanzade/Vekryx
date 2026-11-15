//! Exact BM25 and vector retrieval combined with weighted reciprocal rank fusion.
use crate::{distance::DistanceMetric, store::Vekryx};
use std::collections::{BTreeMap, BTreeSet};

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

impl Vekryx {
    /// Search this corpus (call `filtered` first for metadata restrictions).
    /// Larger scores are better. BM25 uses k1=1.2, b=0.75 and positive IDF.
    pub fn hybrid_search(
        &self,
        text: &str,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
        vector_weight: f64,
        rrf_k: usize,
    ) -> Result<Vec<(String, f64)>, String> {
        if !vector_weight.is_finite() || !(0.0..=1.0).contains(&vector_weight) {
            return Err("vector_weight must be finite and between 0 and 1".into());
        }
        if rrf_k == 0 {
            return Err("rrf_k must be greater than zero".into());
        }
        // Validate query shape even in keyword-only mode, but do not score vectors.
        self.validate(query)?;
        metric.distance(query, query)?;
        let vector_results = if vector_weight > 0.0 {
            self.search_with_metric(query, self.len(), metric)?
        } else {
            Vec::new()
        };
        let docs: Vec<_> = self
            .records()
            .filter_map(|(id, record)| {
                record
                    .text
                    .as_deref()
                    .map(|text| (id.clone(), tokens(text)))
            })
            .collect();
        let query_terms: BTreeSet<_> = tokens(text).into_iter().collect();
        let n = docs.len() as f64;
        let average = docs.iter().map(|(_, t)| t.len()).sum::<usize>() as f64 / n.max(1.0);
        let mut lexical = Vec::new();
        if average > 0.0 && vector_weight < 1.0 {
            let idfs: BTreeMap<_, _> = query_terms
                .iter()
                .map(|term| {
                    let df = docs.iter().filter(|(_, t)| t.contains(term)).count() as f64;
                    (term, (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
                })
                .collect();
            for (id, terms) in &docs {
                let mut score = 0.0;
                for term in &query_terms {
                    let tf = terms.iter().filter(|t| *t == term).count() as f64;
                    if tf == 0.0 {
                        continue;
                    }
                    let idf = idfs[term];
                    score +=
                        idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * terms.len() as f64 / average));
                }
                if score > 0.0 {
                    lexical.push((id.clone(), score));
                }
            }
        }
        lexical.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut scores = BTreeMap::new();
        for (results, weight) in [
            (&vector_results, vector_weight),
            (&lexical, 1.0 - vector_weight),
        ] {
            if weight == 0.0 {
                continue;
            }
            for (rank, (id, _)) in results.iter().enumerate() {
                *scores.entry(id.clone()).or_insert(0.0) +=
                    weight / (rrf_k as f64 + rank as f64 + 1.0);
            }
        }
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        distance::Cosine,
        store::{Record, Vekryx},
    };
    use serde_json::json;

    fn insert(store: &mut Vekryx, id: &str, vector: Vec<f64>, text: &str, group: &str) {
        store
            .insert_record(
                id.into(),
                vector,
                Record {
                    text: Some(text.into()),
                    metadata: json!({"group": group}).as_object().unwrap().clone(),
                },
            )
            .unwrap();
    }

    #[test]
    fn filtering_precedes_vector_validation_ranking_and_mmr_limits() {
        let mut store = Vekryx::new(2).unwrap();
        insert(&mut store, "outside", vec![0.0, 0.0], "cat", "private");
        insert(&mut store, "inside", vec![0.0, 1.0], "dog", "public");
        let filtered = store.filtered(json!({"group":"public"}).as_object().unwrap());
        assert_eq!(
            filtered
                .search_with_metric(&[1.0, 0.0], 1, &Cosine)
                .unwrap()[0]
                .0,
            "inside"
        );
        assert_eq!(
            filtered
                .mmr_search(&[1.0, 0.0], 1, 1, 0.5, &Cosine)
                .unwrap()[0]
                .0,
            "inside"
        );
        assert_eq!(
            filtered
                .hybrid_search("dog", &[1.0, 0.0], 1, &Cosine, 0.5, 60)
                .unwrap()[0]
                .0,
            "inside"
        );
        assert!(store.search_with_metric(&[1.0, 0.0], 1, &Cosine).is_err());
    }

    #[test]
    fn weighted_rrf_scores_endpoints_and_ties_are_deterministic() {
        let mut store = Vekryx::new(2).unwrap();
        insert(&mut store, "a", vec![1.0, 0.0], "dog", "public");
        insert(&mut store, "b", vec![0.0, 1.0], "cat", "public");
        let mixed = store
            .hybrid_search("CAT!", &[1.0, 0.0], 5, &Cosine, 0.5, 60)
            .unwrap();
        assert_eq!(mixed[0].0, "b");
        assert!((mixed[0].1 - (0.5 / 62.0 + 0.5 / 61.0)).abs() < 1e-12);
        assert_eq!(
            store
                .hybrid_search("cat", &[1.0, 0.0], 5, &Cosine, 0.0, 60)
                .unwrap(),
            vec![("b".into(), 1.0 / 61.0)]
        );
        assert_eq!(
            store
                .hybrid_search("cat", &[1.0, 0.0], 1, &Cosine, 1.0, 60)
                .unwrap()[0]
                .0,
            "a"
        );
        assert!(store
            .hybrid_search("missing", &[1.0, 0.0], 5, &Cosine, 0.0, 60)
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .hybrid_search("!!!", &[1.0, 0.0], 1, &Cosine, 0.5, 60)
                .unwrap()[0]
                .0,
            "a"
        );
        assert!(store
            .hybrid_search("cat", &[1.0, 0.0], 0, &Cosine, 0.5, 60)
            .unwrap()
            .is_empty());
        insert(&mut store, "c", vec![0.0, 1.0], "cat", "public");
        let ties = store
            .hybrid_search("cat", &[1.0, 0.0], 5, &Cosine, 0.0, 60)
            .unwrap();
        assert_eq!(
            ties.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        for weight in [-0.1, 1.1, f64::NAN] {
            assert!(store
                .hybrid_search("cat", &[1.0, 0.0], 1, &Cosine, weight, 60)
                .is_err());
        }
        assert!(store
            .hybrid_search("cat", &[1.0, 0.0], 1, &Cosine, 0.5, 0)
            .is_err());
    }

    #[test]
    fn bm25_rewards_term_frequency_and_penalizes_document_length() {
        let mut store = Vekryx::new(2).unwrap();
        insert(
            &mut store,
            "a-long",
            vec![1.0, 0.0],
            "cat dog bird fish horse mouse sheep cow",
            "public",
        );
        insert(&mut store, "b-short", vec![1.0, 0.0], "cat", "public");
        insert(
            &mut store,
            "c-repeated",
            vec![1.0, 0.0],
            "cat cat cat",
            "public",
        );
        let results = store
            .hybrid_search("cat cat", &[1.0, 0.0], 5, &Cosine, 0.0, 60)
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["c-repeated", "b-short", "a-long"]
        );
        assert_eq!(
            results,
            store
                .hybrid_search("CAT!", &[1.0, 0.0], 5, &Cosine, 0.0, 60)
                .unwrap()
        );
    }

    #[test]
    fn records_round_trip_replace_delete_and_validate_references() {
        let mut store = Vekryx::new(2).unwrap();
        insert(&mut store, "a", vec![1.0, 0.0], "cat", "public");
        let mut restored = Vekryx::from_json(&store.to_json().unwrap()).unwrap();
        assert_eq!(restored.record("a").unwrap().text.as_deref(), Some("cat"));
        assert_eq!(restored.record("a").unwrap().metadata["group"], "public");
        assert!(restored
            .insert_record(
                "a".into(),
                vec![1.0],
                Record {
                    text: None,
                    metadata: Default::default()
                }
            )
            .is_err());
        assert!(restored.record("a").is_some());
        restored.insert("a".into(), vec![0.0, 1.0]).unwrap();
        assert!(restored.record("a").is_none());
        restored.delete("a");
        assert!(restored.records().next().is_none());
        assert!(Vekryx::from_json(
            r#"{"dimensions":2,"vectors":{},"records":{"a":{"text":"cat","metadata":{}}}}"#
        )
        .is_err());
        assert!(Vekryx::from_json(r#"{"dimensions":2,"vectors":{"a":[1,0]}}"#).is_ok());
    }
}
