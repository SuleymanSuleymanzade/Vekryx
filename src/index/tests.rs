use super::*;
use crate::{
    distance::{Cosine, Euclidean, MetricKind},
    store::Record,
    HybridOptions, MmrOptions, SearchOptions, Vekryx,
};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

fn corpus(n: usize, dimensions: usize) -> Vekryx {
    let mut random = training::Random(73);
    let mut store = Vekryx::new(dimensions).unwrap();
    for i in 0..n {
        let vector = (0..dimensions)
            .map(|_| (random.next() % 100_000) as f64 / 10_000.0 - 5.0)
            .collect();
        store
            .insert_record(
                format!("{i:04}"),
                vector,
                Record {
                    text: Some(if i % 2 == 0 { "cat" } else { "dog" }.into()),
                    metadata: serde_json::from_value(serde_json::json!({"group": i % 2})).unwrap(),
                },
            )
            .unwrap();
    }
    store
}
fn pq() -> PqConfig {
    PqConfig {
        subquantizers: 4,
        bits: 3,
        iterations: 12,
        seed: 42,
    }
}
fn ivf() -> IvfConfig {
    IvfConfig {
        nlist: 8,
        iterations: 12,
        seed: 42,
    }
}
fn recall(actual: &[crate::SearchHit], expected: &[(String, f64)]) -> f64 {
    let ids: BTreeSet<_> = actual.iter().map(|h| h.id.as_str()).collect();
    expected
        .iter()
        .filter(|(id, _)| ids.contains(id.as_str()))
        .count() as f64
        / expected.len() as f64
}

#[test]
fn hnsw_has_high_recall_against_exact_for_both_geometries() {
    let store = corpus(256, 8);
    for metric in [&Euclidean as &dyn DistanceMetric, &Cosine] {
        let config = HnswConfig {
            m: 12,
            ef_construction: 80,
            seed: 7,
        };
        let session = store.view().build_index(&config, metric).unwrap();
        let options = SearchOptions {
            k: 10,
            candidates: 10,
            rerank_exact: true,
            backend: HnswSearchOptions { ef_search: 64 },
        };
        let mut total = 0.0;
        for i in 0..16 {
            let query: Vec<_> = store
                .get(&format!("{:04}", i * 13))
                .unwrap()
                .iter()
                .map(|x| x + 0.07)
                .collect();
            let found = session.search(&query, &options).unwrap();
            total += recall(
                &found,
                &store.search_with_metric(&query, 10, metric).unwrap(),
            );
            assert!(found.iter().all(|h| h.distance_kind == DistanceKind::Exact));
        }
        assert!(total / 16.0 >= 0.95, "HNSW recall = {}", total / 16.0);
        assert_eq!(session.guarantee(), RetrievalGuarantee::Approximate);
        let second = store.view().build_index(&config, metric).unwrap();
        assert_eq!(
            session.search(&[1.0; 8], &options).unwrap(),
            second.search(&[1.0; 8], &options).unwrap()
        );
    }
}

struct CountingMetric(AtomicUsize);
impl DistanceMetric for CountingMetric {
    fn kind(&self) -> Option<MetricKind> {
        Some(MetricKind::Euclidean)
    }
    fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Euclidean.distance(a, b)
    }
}
#[test]
fn graph_and_inverted_lists_prune_distance_evaluations() {
    let store = corpus(512, 8);
    let metric = CountingMetric(AtomicUsize::new(0));
    let graph = store
        .view()
        .build_index(&HnswConfig::default(), &metric)
        .unwrap();
    metric.0.store(0, Ordering::Relaxed);
    let options = SearchOptions {
        backend: HnswSearchOptions { ef_search: 8 },
        ..SearchOptions::new(3)
    };
    assert_eq!(
        graph.search(store.get("0000").unwrap(), &options).unwrap()[0].id,
        "0000"
    );
    assert!(metric.0.load(Ordering::Relaxed) < store.len());
    let lists = store.view().build_index(&ivf(), &metric).unwrap();
    metric.0.store(0, Ordering::Relaxed);
    let options = SearchOptions {
        backend: IvfSearchOptions { nprobe: 1 },
        ..SearchOptions::new(3)
    };
    assert!(!lists.search(&[1.0; 8], &options).unwrap().is_empty());
    assert!(metric.0.load(Ordering::Relaxed) < store.len());
}

#[test]
fn all_ivf_lists_match_exact_and_single_list_really_limits_candidates() {
    let store = corpus(128, 8);
    for metric in [&Euclidean as &dyn DistanceMetric, &Cosine] {
        let index = store.view().build_index(&ivf(), metric).unwrap();
        let options = SearchOptions {
            backend: IvfSearchOptions { nprobe: 8 },
            ..SearchOptions::new(10)
        };
        for query in [[1.0; 8], [-1.0; 8], [0.25; 8]] {
            let hits = index.search(&query, &options).unwrap();
            assert_eq!(
                hits.iter()
                    .map(|h| (h.id.clone(), h.distance))
                    .collect::<Vec<_>>(),
                store.search_with_metric(&query, 10, metric).unwrap()
            );
        }
        let options = SearchOptions {
            backend: IvfSearchOptions { nprobe: 1 },
            ..SearchOptions::new(128)
        };
        assert!(index.search(&[1.0; 8], &options).unwrap().len() < 128);
    }
}

#[test]
fn pq_and_residual_ivfpq_rerank_candidates_against_exact_oracle() {
    let store = corpus(256, 8);
    let pq_config = pq();
    let index = pq_config.build(store.view(), &Euclidean).unwrap();
    assert_eq!(index.code_bytes_per_vector(), 4);
    let session = store.view().build_index(&pq_config, &Euclidean).unwrap();
    let combined = store
        .view()
        .build_index(
            &IvfPqConfig {
                ivf: ivf(),
                pq: pq_config,
            },
            &Euclidean,
        )
        .unwrap();
    let estimates = session
        .search(
            &[1.0; 8],
            &SearchOptions {
                rerank_exact: false,
                ..SearchOptions::new(10)
            },
        )
        .unwrap();
    assert!(estimates
        .iter()
        .all(|h| h.distance_kind == DistanceKind::Approximate));
    assert!(estimates.iter().any(|h| (h.distance
        - Euclidean
            .distance(&[1.0; 8], store.get(&h.id).unwrap())
            .unwrap())
    .abs()
        > 1e-6));
    let mut pq_recall = 0.0;
    let mut combined_recall = 0.0;
    for i in 0..12 {
        let query = store.get(&format!("{:04}", i * 17)).unwrap();
        let exact = store.search(query, 10).unwrap();
        let reranked = session
            .search(
                query,
                &SearchOptions {
                    candidates: 80,
                    ..SearchOptions::new(10)
                },
            )
            .unwrap();
        let ivfpq = combined
            .search(
                query,
                &SearchOptions {
                    k: 10,
                    candidates: 80,
                    rerank_exact: true,
                    backend: IvfSearchOptions { nprobe: 8 },
                },
            )
            .unwrap();
        for hit in reranked.iter().chain(&ivfpq) {
            assert_eq!(hit.distance_kind, DistanceKind::Exact);
            assert_eq!(
                hit.distance,
                Euclidean
                    .distance(query, store.get(&hit.id).unwrap())
                    .unwrap()
            );
        }
        pq_recall += recall(&reranked, &exact);
        combined_recall += recall(&ivfpq, &exact);
    }
    assert!(
        pq_recall / 12.0 >= 0.9,
        "PQ reranked recall {}",
        pq_recall / 12.0
    );
    assert!(
        combined_recall / 12.0 >= 0.9,
        "IVF-PQ reranked recall {}",
        combined_recall / 12.0
    );
    assert_eq!(combined.guarantee(), RetrievalGuarantee::Approximate);
    // Retrieving every code then exact reranking restores the exact oracle.
    let all = session
        .search(
            &[1.0; 8],
            &SearchOptions {
                candidates: 256,
                ..SearchOptions::new(10)
            },
        )
        .unwrap();
    assert_eq!(recall(&all, &store.search(&[1.0; 8], 10).unwrap()), 1.0);
}

#[test]
fn every_backend_supports_filtered_mmr_and_hybrid_and_preserves_ids() {
    fn check<B: IndexBuilder>(
        store: &Vekryx,
        builder: &B,
        options: SearchOptions<<B::Index as CandidateIndex>::SearchOptions>,
    ) {
        let filter = serde_json::json!({"group": 0});
        let view = store.filtered_view(filter.as_object().unwrap());
        let session = view.build_index(builder, &Euclidean).unwrap();
        let hits = session
            .mmr_search(&[1.0; 8], &options, &MmrOptions::default())
            .unwrap();
        assert_eq!(hits.len(), 3);
        assert!(hits.iter().all(|h| view.contains(&h.id)));
        assert_eq!(
            hits.iter().map(|h| &h.id).collect::<BTreeSet<_>>().len(),
            hits.len()
        );
        let fused = session
            .hybrid_search("cat", &[1.0; 8], &options, &HybridOptions::default())
            .unwrap();
        assert_eq!(fused.len(), 3);
        assert!(fused.iter().all(|h| view.contains(&h.id)));
    }
    let store = corpus(64, 8);
    check(
        &store,
        &HnswConfig::default(),
        SearchOptions {
            candidates: 16,
            ..SearchOptions::new(3)
        },
    );
    check(
        &store,
        &ivf(),
        SearchOptions {
            candidates: 16,
            backend: IvfSearchOptions { nprobe: 8 },
            ..SearchOptions::new(3)
        },
    );
    check(
        &store,
        &pq(),
        SearchOptions {
            candidates: 16,
            ..SearchOptions::new(3)
        },
    );
    check(
        &store,
        &IvfPqConfig {
            ivf: ivf(),
            pq: pq(),
        },
        SearchOptions {
            candidates: 16,
            backend: IvfSearchOptions { nprobe: 8 },
            ..SearchOptions::new(3)
        },
    );
    let mut duplicates = Vekryx::new(8).unwrap();
    for id in ["a", "b", "c", "d", "e", "f", "g", "h"] {
        duplicates.insert(id.into(), vec![1.0; 8]).unwrap();
    }
    let session = duplicates.view().build_index(&pq(), &Euclidean).unwrap();
    assert_eq!(
        session
            .search(&[1.0; 8], &SearchOptions::new(8))
            .unwrap()
            .len(),
        8
    );
}

#[test]
fn empty_indexes_and_invalid_training_and_query_options_are_explicit() {
    let empty = Vekryx::new(8).unwrap();
    let hnsw = empty
        .view()
        .build_index(&HnswConfig::default(), &Euclidean)
        .unwrap();
    assert!(hnsw
        .search(&[1.0; 8], &SearchOptions::new(5))
        .unwrap()
        .is_empty());
    assert!(hnsw
        .search(
            &[1.0; 8],
            &SearchOptions {
                backend: HnswSearchOptions { ef_search: 0 },
                ..SearchOptions::new(0)
            }
        )
        .is_err());
    let ivf_session = empty.view().build_index(&ivf(), &Euclidean).unwrap();
    assert!(ivf_session
        .search(&[1.0; 8], &SearchOptions::new(5))
        .unwrap()
        .is_empty());
    for nprobe in [0, 9] {
        assert!(ivf_session
            .search(
                &[1.0; 8],
                &SearchOptions {
                    backend: IvfSearchOptions { nprobe },
                    ..SearchOptions::new(0)
                }
            )
            .is_err());
    }
    let pq_session = empty.view().build_index(&pq(), &Euclidean).unwrap();
    assert!(pq_session
        .search(&[1.0; 8], &SearchOptions::new(5))
        .unwrap()
        .is_empty());
    let combined = empty
        .view()
        .build_index(
            &IvfPqConfig {
                ivf: ivf(),
                pq: pq(),
            },
            &Euclidean,
        )
        .unwrap();
    assert!(combined
        .search(&[1.0; 8], &SearchOptions::new(5))
        .unwrap()
        .is_empty());
    let store = corpus(4, 8);
    assert!(store.view().build_index(&ivf(), &Euclidean).is_err());
    assert!(store.view().build_index(&pq(), &Euclidean).is_err());
    assert!(store.view().build_index(&pq(), &Cosine).is_err());
    assert!(store
        .view()
        .build_index(&IvfPqConfig::default(), &Cosine)
        .is_err());
    for config in [
        HnswConfig {
            m: 1,
            ..HnswConfig::default()
        },
        HnswConfig {
            ef_construction: 1,
            ..HnswConfig::default()
        },
    ] {
        assert!(config.build(store.view(), &Euclidean).is_err());
    }
    for config in [
        PqConfig { bits: 9, ..pq() },
        PqConfig {
            subquantizers: 3,
            ..pq()
        },
        PqConfig {
            iterations: 0,
            ..pq()
        },
    ] {
        assert!(config.build(empty.view(), &Euclidean).is_err());
    }
    let mut zero = Vekryx::new(8).unwrap();
    zero.insert("zero".into(), vec![0.0; 8]).unwrap();
    assert!(HnswConfig::default().build(zero.view(), &Cosine).is_err());
    assert!(IvfConfig { nlist: 1, ..ivf() }
        .build(zero.view(), &Cosine)
        .is_err());
}
