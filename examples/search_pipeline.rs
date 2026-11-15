//! Run: cargo run --example search_pipeline
use serde_json::json;
use vekryx::{
    distance::Cosine,
    index::{DistanceKind, ExactSearch, RetrievalGuarantee},
    store::Record,
    HybridOptions, MmrOptions, SearchOptions, Vekryx,
};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    for (id, text, vector) in [
        ("a", "cats", vec![1.0, 0.0]),
        ("b", "dogs", vec![1.0, 0.0]),
        ("c", "birds", vec![0.0, 1.0]),
    ] {
        store.insert_record(
            id.into(),
            vector,
            Record {
                text: Some(text.into()),
                metadata: json!({"group": "public"}).as_object().unwrap().clone(),
            },
        )?;
    }
    let filter = json!({"group": "public"});
    let session = store
        .filtered_view(filter.as_object().unwrap())
        .build_index(&ExactSearch, &Cosine)?;
    let options = SearchOptions {
        candidates: 3,
        ..SearchOptions::new(2)
    };
    assert_eq!(session.guarantee(), RetrievalGuarantee::Exact);
    let hits = session.search(&[1.0, 0.0], &options)?;
    assert_eq!(hits[0].distance_kind, DistanceKind::Exact);
    println!("Vector: {hits:?}");
    let diverse = session.mmr_search(&[1.0, 0.0], &options, &MmrOptions { lambda_mult: 0.0 })?;
    assert_eq!(diverse[1].id, "c");
    println!("MMR: {diverse:?}");
    let fused = session.hybrid_search("birds", &[1.0, 0.0], &options, &HybridOptions::default())?;
    assert_eq!(fused[0].id, "c");
    println!("Hybrid: {fused:?}");
    Ok(())
}
