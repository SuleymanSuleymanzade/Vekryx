use vekryx::Vekryx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = Vekryx::new(3)?;
    store.insert("example".into(), vec![1.0, 2.0, 3.0])?;
    println!(
        "Nearest neighbors: {:?}",
        store.search(&[1.1, 2.0, 3.0], 1)?
    );
    Ok(())
}
