//! Run: cargo run --release --example ann_search
//! Deterministic recall/timing demonstration, not a representative benchmark.
use std::time::Instant;
use vekryx::{
    distance::Euclidean,
    index::{
        CandidateIndex, HnswConfig, HnswSearchOptions, IndexBuilder, IvfConfig, IvfPqConfig,
        IvfSearchOptions, PqConfig,
    },
    SearchOptions, Vekryx,
};

fn report<B: IndexBuilder>(
    name: &str,
    store: &Vekryx,
    builder: B,
    options: SearchOptions<<B::Index as CandidateIndex>::SearchOptions>,
) -> Result<(), vekryx::Error> {
    let start = Instant::now();
    let session = store.view().build_index(&builder, &Euclidean)?;
    let build = start.elapsed();
    let mut recall = 0.0;
    let mut query_time = std::time::Duration::ZERO;
    for i in 0..16 {
        let query: Vec<_> = store
            .get(&format!("{:04}", i * 13))
            .unwrap()
            .iter()
            .map(|x| x + 0.05)
            .collect();
        let exact = store.search(&query, 10)?;
        let start = Instant::now();
        let hits = session.search(&query, &options)?;
        query_time += start.elapsed();
        recall += exact
            .iter()
            .filter(|(id, _)| hits.iter().any(|hit| hit.id == *id))
            .count() as f64
            / 10.0;
    }
    println!(
        "{name:10} recall@10={:.3}, build={build:?}, 16 queries={query_time:?}",
        recall / 16.0
    );
    Ok(())
}
fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(8)?;
    for i in 0..256 {
        let vector = (0..8)
            .map(|d| ((i * 37 + d * 17) % 251) as f64 / 25.0 + (i as f64 * 0.1 + d as f64).sin())
            .collect();
        store.insert(format!("{i:04}"), vector)?;
    }
    let pq = PqConfig {
        subquantizers: 4,
        bits: 3,
        ..PqConfig::default()
    };
    let ivf = IvfConfig {
        nlist: 8,
        ..IvfConfig::default()
    };
    report(
        "HNSW",
        &store,
        HnswConfig::default(),
        SearchOptions {
            candidates: 40,
            backend: HnswSearchOptions { ef_search: 64 },
            ..SearchOptions::new(10)
        },
    )?;
    report(
        "IVF-Flat",
        &store,
        ivf,
        SearchOptions {
            candidates: 40,
            backend: IvfSearchOptions { nprobe: 4 },
            ..SearchOptions::new(10)
        },
    )?;
    report(
        "PQ",
        &store,
        pq,
        SearchOptions {
            candidates: 80,
            ..SearchOptions::new(10)
        },
    )?;
    report(
        "IVF-PQ",
        &store,
        IvfPqConfig { ivf, pq },
        SearchOptions {
            candidates: 80,
            backend: IvfSearchOptions { nprobe: 4 },
            ..SearchOptions::new(10)
        },
    )?;
    Ok(())
}
