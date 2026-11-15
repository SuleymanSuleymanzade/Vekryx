//! Run: cargo run --example hybrid_search
use serde_json::json;
use vekryx::{distance::Cosine, store::Record, Vekryx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = Vekryx::new(2)?;
    for (id, text, vector, domain) in [
        (
            "cats",
            "Cats are domestic animals.",
            vec![1.0, 0.0],
            "animals",
        ),
        (
            "rust",
            "Rust prevents memory errors.",
            vec![0.0, 1.0],
            "programming",
        ),
    ] {
        store.insert_record(
            id.into(),
            vector,
            Record {
                text: Some(text.into()),
                metadata: json!({"domain": domain}).as_object().unwrap().clone(),
            },
        )?;
    }
    // Rust callers supply model-generated vectors. Filter BEFORE ranking.
    let corpus = store.filtered(json!({"domain": "animals"}).as_object().unwrap());
    let results = corpus.hybrid_search("cats", &[1.0, 0.0], 5, &Cosine, 0.5, 60)?;
    assert_eq!(results[0].0, "cats");
    println!("Hybrid (ID, fusion score): {results:?}");
    Ok(())
}
