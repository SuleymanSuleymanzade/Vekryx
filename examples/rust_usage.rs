//! Run: cargo run --example rust_usage -- [snapshot-path]
use std::path::PathBuf;
use vector_database::{
    chunking::{ChunkingStrategy, HierarchicalChunker},
    distance::Cosine,
    store::Store,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/vectors.json"));
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut store = Store::new(3)?;
    store.insert("item1".into(), vec![1.0, 2.0, 3.0])?;
    store.insert("item2".into(), vec![3.0, 2.0, 1.0])?;
    let query = [1.0, 2.0, 3.0];
    println!(
        "Cosine: {:?}",
        store.search_with_metric(&query, 2, &Cosine)?
    );
    println!("MMR: {:?}", store.mmr_search(&query, 2, 2, 0.5, &Cosine)?);
    let chunks = HierarchicalChunker::new(vec![32, 12], 2)?
        .chunk("Rust stores vectors. Python retrieves chunks.", "guide")?;
    println!(
        "Leaf chunks: {}",
        chunks.iter().filter(|c| c.is_leaf).count()
    );
    store.save(&path)?;
    let restored = Store::load(&path)?;
    assert_eq!(restored.get("item1").unwrap(), &vec![1.0, 2.0, 3.0]);
    println!(
        "Saved {} vectors to {}",
        restored.len(),
        path.canonicalize()?.display()
    );
    Ok(())
}
