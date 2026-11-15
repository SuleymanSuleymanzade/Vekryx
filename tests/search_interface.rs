//! Exercise extension points as a downstream crate, without private access.
use serde_json::json;
use vekryx::{
    distance::{Cosine, DistanceMetric, Euclidean, MetricKind},
    index::{
        Candidate, CandidateIndex, DistanceKind, ExactSearch, IndexBuilder, RetrievalGuarantee,
    },
    search::Corpus,
    store::Record,
    Error, HybridOptions, MmrOptions, SearchOptions, Vekryx,
};

#[derive(Default)]
struct QueryOptions {
    skip: usize,
}

struct SubsetBuilder;
struct SubsetIndex(Vec<String>);

impl IndexBuilder for SubsetBuilder {
    type Index = SubsetIndex;

    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<Self::Index, Error> {
        if metric.kind() != Some(MetricKind::Cosine) {
            return Err("this example backend supports only cosine".into());
        }
        Ok(SubsetIndex(
            corpus.vectors().map(|(id, _)| id.to_owned()).collect(),
        ))
    }
}

impl CandidateIndex for SubsetIndex {
    type SearchOptions = QueryOptions;

    fn candidates(
        &self,
        _corpus: Corpus<'_>,
        _query: &[f64],
        _metric: &dyn DistanceMetric,
        limit: usize,
        options: &QueryOptions,
    ) -> Result<Vec<Candidate>, Error> {
        // Deliberately incomplete retrieval with inaccurate distances. This is
        // a contract fixture, not an ANN implementation or performance claim.
        Ok(self
            .0
            .iter()
            .skip(options.skip)
            .take(limit)
            .rev()
            .enumerate()
            .map(|(i, id)| Candidate {
                id: id.clone(),
                distance: i as f64 + 10.0,
                distance_kind: DistanceKind::Approximate,
            })
            .collect())
    }
}

fn store() -> Vekryx {
    let mut store = Vekryx::new(2).unwrap();
    for (id, vector, text, group) in [
        ("a", vec![1.0, 0.0], "dog", "public"),
        ("b", vec![1.0, 0.0], "dog", "public"),
        ("c", vec![0.0, 1.0], "cat", "public"),
        ("private", vec![0.0, 0.0], "cat", "private"),
    ] {
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
    store
}

#[test]
fn exact_session_preserves_legacy_search_mmr_and_full_hybrid_rankings() {
    let store = store();
    let filter = json!({"group": "public"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    assert_eq!(corpus.dimensions(), 2);
    assert!(corpus.get("private").is_none());
    assert!(!corpus.contains("missing"));
    let session = corpus.build_index(&ExactSearch, &Cosine).unwrap();
    assert_eq!(session.guarantee(), RetrievalGuarantee::Exact);
    let options = SearchOptions {
        candidates: 3,
        ..SearchOptions::new(2)
    };
    let hits = session.search(&[1.0, 0.0], &options).unwrap();
    assert_eq!(
        hits.iter()
            .map(|h| (h.id.clone(), h.distance))
            .collect::<Vec<_>>(),
        corpus.search_with_metric(&[1.0, 0.0], 2, &Cosine).unwrap()
    );
    assert!(hits.iter().all(|h| h.distance_kind == DistanceKind::Exact));
    let diverse = session
        .mmr_search(&[1.0, 0.0], &options, &MmrOptions { lambda_mult: 0.0 })
        .unwrap();
    assert_eq!(
        diverse.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["a", "c"]
    );
    let hybrid = session
        .hybrid_search("cat", &[1.0, 0.0], &options, &HybridOptions::default())
        .unwrap();
    assert_eq!(
        hybrid
            .iter()
            .map(|h| (h.id.clone(), h.fusion_score))
            .collect::<Vec<_>>(),
        corpus
            .hybrid_search("cat", &[1.0, 0.0], 2, &Cosine, 0.5, 60)
            .unwrap()
    );
}

#[test]
fn custom_backend_uses_typed_options_and_reranking_does_not_restore_missing_neighbors() {
    let store = store();
    let filter = json!({"group": "public"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    assert!(corpus.build_index(&SubsetBuilder, &Euclidean).is_err());
    let session = corpus.build_index(&SubsetBuilder, &Cosine).unwrap();
    let mut options = SearchOptions {
        k: 2,
        candidates: 2,
        rerank_exact: false,
        backend: QueryOptions { skip: 1 },
    };
    let estimated = session.search(&[1.0, 0.0], &options).unwrap();
    assert_eq!(estimated[0].id, "c");
    assert_eq!(estimated[0].distance_kind, DistanceKind::Approximate);
    options.rerank_exact = true;
    let exact_distances = session.search(&[1.0, 0.0], &options).unwrap();
    assert_eq!(
        exact_distances
            .iter()
            .map(|h| h.id.as_str())
            .collect::<Vec<_>>(),
        ["b", "c"]
    );
    assert_eq!(exact_distances[0].distance, 0.0);
    assert!(exact_distances
        .iter()
        .all(|h| h.distance_kind == DistanceKind::Exact));
    assert_eq!(session.guarantee(), RetrievalGuarantee::Approximate);
    assert_eq!(
        corpus.search_with_metric(&[1.0, 0.0], 1, &Cosine).unwrap()[0].0,
        "a"
    );
    let mmr = session
        .mmr_search(&[1.0, 0.0], &options, &MmrOptions::default())
        .unwrap();
    assert_eq!(mmr[0].id, "b");
    let fused = session
        .hybrid_search(
            "dog",
            &[1.0, 0.0],
            &options,
            &HybridOptions {
                vector_weight: 1.0,
                ..HybridOptions::default()
            },
        )
        .unwrap();
    assert_eq!(fused[0].id, "b");
    options.backend.skip = 2;
    assert_eq!(session.search(&[1.0, 0.0], &options).unwrap().len(), 1);
}

#[test]
fn bounded_hybrid_rankings_have_explicitly_different_fusion_scores() {
    let store = store();
    let filter = json!({"group": "public"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    let options = SearchOptions::new(1);
    let bounded = corpus
        .hybrid_search_with_options(
            "cat",
            &[1.0, 0.0],
            &Cosine,
            &options,
            &HybridOptions::default(),
        )
        .unwrap();
    // c appears only in the keyword list; a only in the bounded vector list.
    assert_eq!(bounded[0].id, "a");
    assert_eq!(bounded[0].fusion_score, 0.5 / 61.0);
    let full = corpus
        .hybrid_search("cat", &[1.0, 0.0], 1, &Cosine, 0.5, 60)
        .unwrap();
    assert_eq!(full[0].0, "c");
    let no_keywords = corpus
        .hybrid_search_with_options(
            "cat",
            &[1.0, 0.0],
            &Cosine,
            &options,
            &HybridOptions {
                keyword_candidates: Some(0),
                ..HybridOptions::default()
            },
        )
        .unwrap();
    assert_eq!(no_keywords[0].id, "a");
}

struct FixedCandidates(Vec<Candidate>);
impl IndexBuilder for FixedCandidates {
    type Index = Self;
    fn build(&self, _: Corpus<'_>, _: &dyn DistanceMetric) -> Result<Self, Error> {
        Ok(Self(self.0.clone()))
    }
}
impl CandidateIndex for FixedCandidates {
    type SearchOptions = ();
    fn candidates(
        &self,
        _: Corpus<'_>,
        _: &[f64],
        _: &dyn DistanceMetric,
        _: usize,
        _: &(),
    ) -> Result<Vec<Candidate>, Error> {
        Ok(self.0.clone())
    }
}

#[test]
fn backend_boundary_rejects_invalid_output_and_deduplicates_ids() {
    let store = store();
    let filter = json!({"group": "public"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    let candidate = |id: &str, distance| Candidate {
        id: id.into(),
        distance,
        distance_kind: DistanceKind::Approximate,
    };
    for invalid in [
        candidate("missing", 1.0),
        candidate("private", 1.0),
        candidate("a", f64::NAN),
        candidate("a", f64::INFINITY),
        candidate("a", -1.0),
    ] {
        let session = corpus
            .build_index(&FixedCandidates(vec![invalid]), &Cosine)
            .unwrap();
        assert!(session.search(&[1.0, 0.0], &SearchOptions::new(1)).is_err());
    }
    let session = corpus
        .build_index(
            &FixedCandidates(vec![candidate("a", 4.0), candidate("a", 3.0)]),
            &Cosine,
        )
        .unwrap();
    assert!(session.search(&[1.0, 0.0], &SearchOptions::new(1)).is_err());
    let hits = session
        .search(
            &[1.0, 0.0],
            &SearchOptions {
                rerank_exact: false,
                ..SearchOptions::new(2)
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].distance, 3.0);
}

#[test]
fn options_validate_empty_queries_and_session_rebuilds_after_mutation() {
    let mut store = Vekryx::new(2).unwrap();
    {
        let session = store.view().build_index(&ExactSearch, &Cosine).unwrap();
        assert!(session.search(&[0.0, 0.0], &SearchOptions::new(0)).is_err());
        assert!(session.search(&[1.0], &SearchOptions::new(0)).is_err());
        assert!(session
            .search(&[f64::NAN, 0.0], &SearchOptions::new(0))
            .is_err());
        assert!(session
            .search(
                &[1.0, 0.0],
                &SearchOptions {
                    candidates: 0,
                    ..SearchOptions::new(1)
                }
            )
            .is_err());
        assert!(session
            .search(&[1.0, 0.0], &SearchOptions::new(0))
            .unwrap()
            .is_empty());
    }
    store.insert("a".into(), vec![1.0, 0.0]).unwrap();
    let json = store.to_json().unwrap();
    let session = store.view().build_index(&ExactSearch, &Cosine).unwrap();
    assert_eq!(
        session.search(&[1.0, 0.0], &SearchOptions::new(1)).unwrap()[0].id,
        "a"
    );
    assert_eq!(store.to_json().unwrap(), json);
    let restored = Vekryx::from_json(&json).unwrap();
    assert_eq!(restored.search(&[1.0, 0.0], 1).unwrap()[0].0, "a");
}
