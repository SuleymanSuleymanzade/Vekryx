use crate::distance::DistanceMetric;
pub use crate::search::Corpus;
use crate::state_manager::snapshot::{invalid_data, write_atomic};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

/// Encoding for snapshots that load into a mutable in-memory store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SnapshotFormat {
    #[default]
    Json,
    /// The same JSON document wrapped in gzip.
    CompressedJson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub text: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "StoreSnapshot")]
pub struct Vekryx {
    dimensions: usize,
    vectors: BTreeMap<String, Vec<f64>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    records: BTreeMap<String, Record>,
}

impl Vekryx {
    pub fn new(dimensions: usize) -> Result<Self, Error> {
        if dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
        }
        Ok(Self {
            dimensions,
            vectors: BTreeMap::new(),
            records: BTreeMap::new(),
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
    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.vectors.get(id).map(Vec::as_slice)
    }
    pub(crate) fn vectors(&self) -> impl Iterator<Item = (&String, &Vec<f64>)> {
        self.vectors.iter()
    }

    pub fn delete(&mut self, id: &str) -> bool {
        self.records.remove(id);
        self.vectors.remove(id).is_some()
    }

    pub(crate) fn validate(&self, vector: &[f64]) -> Result<(), Error> {
        validate_vector(self.dimensions, vector)
    }

    pub fn insert(&mut self, id: String, vector: Vec<f64>) -> Result<(), Error> {
        if id.is_empty() {
            return Err("id must not be empty".into());
        }
        self.validate(&vector)?;
        self.records.remove(&id);
        self.vectors.insert(id, vector);
        Ok(())
    }

    pub fn insert_record(
        &mut self,
        id: String,
        vector: Vec<f64>,
        record: Record,
    ) -> Result<(), Error> {
        self.insert(id.clone(), vector)?;
        self.records.insert(id, record);
        Ok(())
    }

    pub fn record(&self, id: &str) -> Option<&Record> {
        self.records.get(id)
    }

    pub fn records(&self) -> impl Iterator<Item = (&String, &Record)> {
        self.records.iter()
    }

    /// Copy matching entries into an independent store. Use `filtered_view` for search.
    /// All filter properties must match; missing properties never match null.
    pub fn filtered(&self, filter: &serde_json::Map<String, serde_json::Value>) -> Self {
        let matches = |id: &String| {
            filter.iter().all(|(key, value)| {
                self.records.get(id).and_then(|r| r.metadata.get(key)) == Some(value)
            })
        };
        Self {
            dimensions: self.dimensions,
            vectors: self
                .vectors
                .iter()
                .filter(|(id, _)| matches(id))
                .map(|(id, v)| (id.clone(), v.clone()))
                .collect(),
            records: self
                .records
                .iter()
                .filter(|(id, _)| matches(id))
                .map(|(id, r)| (id.clone(), r.clone()))
                .collect(),
        }
    }

    /// Borrow all records without copying vectors or metadata.
    pub fn view(&self) -> Corpus<'_> {
        Corpus::new(self, None)
    }

    /// Borrow a metadata-filtered corpus. Both store and filter must outlive it.
    pub fn filtered_view<'a>(
        &'a self,
        filter: &'a serde_json::Map<String, serde_json::Value>,
    ) -> Corpus<'a> {
        Corpus::new(self, Some(filter))
    }

    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(String, f64)>, Error> {
        self.view().search(query, k)
    }

    pub fn search_with_metric(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, Error> {
        self.view().search_with_metric(query, k, metric)
    }

    /// Exact search with explicit retrieval options and named distance results.
    pub fn search_with_options(
        &self,
        query: &[f64],
        metric: &dyn DistanceMetric,
        options: &crate::SearchOptions,
    ) -> Result<Vec<crate::SearchHit>, Error> {
        self.view().search_with_options(query, metric, options)
    }

    pub fn mmr_search_with_options(
        &self,
        query: &[f64],
        metric: &dyn DistanceMetric,
        options: &crate::SearchOptions,
        mmr: &crate::MmrOptions,
    ) -> Result<Vec<crate::SearchHit>, Error> {
        self.view()
            .mmr_search_with_options(query, metric, options, mmr)
    }

    pub fn mmr_search(
        &self,
        query: &[f64],
        k: usize,
        fetch_k: usize,
        lambda_mult: f64,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, Error> {
        self.view()
            .mmr_search(query, k, fetch_k, lambda_mult, metric)
    }

    /// Save a JSON snapshot at the given path; its parent directory must exist.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_with_format(path, SnapshotFormat::Json)
    }

    /// Load a snapshot. Invalid contents produce an InvalidData I/O error.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::load_with_format(path, SnapshotFormat::Json)
    }

    /// Atomically replace a snapshot using the selected encoding.
    pub fn save_with_format(
        &self,
        path: impl AsRef<Path>,
        format: SnapshotFormat,
    ) -> io::Result<()> {
        write_atomic(path.as_ref(), |file| {
            let mut writer = BufWriter::new(file);
            match format {
                SnapshotFormat::Json => serde_json::to_writer(&mut writer, self)?,
                SnapshotFormat::CompressedJson => {
                    let mut encoder =
                        flate2::write::GzEncoder::new(&mut writer, Default::default());
                    serde_json::to_writer(&mut encoder, self)?;
                    encoder.finish()?;
                }
            }
            writer.flush()
        })
    }

    /// Load a snapshot using the selected encoding and validate its contents.
    pub fn load_with_format(path: impl AsRef<Path>, format: SnapshotFormat) -> io::Result<Self> {
        match format {
            SnapshotFormat::Json => {
                let json = std::fs::read_to_string(path)?;
                Self::from_json(&json).map_err(invalid_data)
            }
            SnapshotFormat::CompressedJson => {
                let reader = BufReader::new(File::open(path)?);
                serde_json::from_reader(flate2::bufread::MultiGzDecoder::new(reader))
                    .map_err(invalid_data)
            }
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, Error> {
        serde_json::from_str(json).map_err(Error::from)
    }
}

pub(crate) fn validate_vector(dimensions: usize, vector: &[f64]) -> Result<(), Error> {
    if vector.len() != dimensions {
        return Err(format!("expected {dimensions} dimensions, got {}", vector.len()).into());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("vector values must be finite".into());
    }
    Ok(())
}

/// The wire representation is private so every deserializer validates invariants.
#[derive(Deserialize)]
struct StoreSnapshot {
    dimensions: usize,
    vectors: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    records: BTreeMap<String, Record>,
}

impl TryFrom<StoreSnapshot> for Vekryx {
    type Error = Error;

    fn try_from(snapshot: StoreSnapshot) -> Result<Self, Self::Error> {
        let store = Self {
            dimensions: snapshot.dimensions,
            vectors: snapshot.vectors,
            records: snapshot.records,
        };
        if store.dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
        }
        if store
            .records
            .keys()
            .any(|id| !store.vectors.contains_key(id))
        {
            return Err(Error::InvalidSnapshot(
                "record references a missing vector".into(),
            ));
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
    use super::Vekryx;
    use crate::Error;

    #[test]
    fn direct_deserialization_rejects_invalid_snapshots() {
        for json in [
            r#"{"dimensions":0,"vectors":{}}"#,
            r#"{"dimensions":2,"vectors":{"a":[1.0]}}"#,
            r#"{"dimensions":2,"vectors":{" ":[1.0,2.0]},"records":{"missing":{"text":null,"metadata":{}}}}"#,
            r#"{"dimensions":2,"vectors":{" ":[1.0,2.0],"":[1.0,2.0]}}"#,
        ] {
            assert!(serde_json::from_str::<Vekryx>(json).is_err(), "{json}");
            assert!(Vekryx::from_json(json).is_err(), "{json}");
        }
        let json = r#"{"dimensions":2,"vectors":{"a":[1.0,2.0]}}"#;
        let restored: Vekryx = serde_json::from_str(json).unwrap();
        assert_eq!(restored.get("a"), Some([1.0, 2.0].as_slice()));
        assert_eq!(
            serde_json::to_value(restored).unwrap(),
            serde_json::from_str::<serde_json::Value>(json).unwrap()
        );
    }

    #[test]
    fn filtered_views_borrow_records_and_match_owned_search() {
        use super::Record;
        use crate::distance::Cosine;
        let mut store = Vekryx::new(2).unwrap();
        store
            .insert_record(
                "a".into(),
                vec![1.0, 0.0],
                Record {
                    text: Some("cats".into()),
                    metadata: serde_json::from_value(serde_json::json!({"group": "public"}))
                        .unwrap(),
                },
            )
            .unwrap();
        store.insert("b".into(), vec![0.0, 1.0]).unwrap();
        let filter = serde_json::from_value(serde_json::json!({"group": "public"})).unwrap();
        let view = store.filtered_view(&filter);
        assert_eq!(view.len(), 1);
        assert!(std::ptr::eq(
            view.records().next().unwrap().1,
            store.record("a").unwrap()
        ));
        let owned = store.filtered(&filter);
        assert_eq!(
            view.search(&[1.0, 0.0], 2).unwrap(),
            owned.search(&[1.0, 0.0], 2).unwrap()
        );
        assert_eq!(
            view.mmr_search(&[1.0, 0.0], 1, 2, 0.5, &Cosine).unwrap(),
            owned.mmr_search(&[1.0, 0.0], 1, 2, 0.5, &Cosine).unwrap()
        );
        assert_eq!(
            view.hybrid_search("cats", &[1.0, 0.0], 2, &Cosine, 0.5, 60)
                .unwrap(),
            owned
                .hybrid_search("cats", &[1.0, 0.0], 2, &Cosine, 0.5, 60)
                .unwrap()
        );
    }

    #[test]
    fn disk_snapshots_replace_round_trip_and_report_invalid_data() {
        use std::{
            fs,
            io::ErrorKind,
            time::{SystemTime, UNIX_EPOCH},
        };
        let directory = std::env::temp_dir().join(format!(
            "vekryx-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("vectors.json");
        let mut store = Vekryx::new(2).unwrap();
        store.insert("item".into(), vec![1.0, 2.0]).unwrap();
        store.save(&path).unwrap();
        store.insert("item".into(), vec![3.0, 4.0]).unwrap();
        store.save(&path).unwrap();
        assert_eq!(
            Vekryx::load(&path).unwrap().get("item").unwrap(),
            &vec![3.0, 4.0]
        );
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::write(&path, "invalid JSON").unwrap();
        assert_eq!(
            Vekryx::load(&path).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            Vekryx::load(directory.join("absent")).unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert!(store.save(directory.join("absent/vectors.json")).is_err());
        let gzip = directory.join("vectors.json.gz");
        let format = super::SnapshotFormat::CompressedJson;
        store.save_with_format(&gzip, format).unwrap();
        let mut compressed = fs::read(&gzip).unwrap();
        assert!(compressed.starts_with(&[0x1f, 0x8b]));
        assert_eq!(
            Vekryx::load_with_format(&gzip, format).unwrap().get("item"),
            store.get("item")
        );
        compressed.pop();
        fs::write(&gzip, compressed).unwrap();
        assert_eq!(
            Vekryx::load_with_format(&gzip, format).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn cosine_and_mmr_rank_diverse_results_and_validate_options() {
        use crate::distance::{Cosine, Euclidean};
        let mut store = Vekryx::new(2).unwrap();
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
        assert!(Vekryx::new(2)
            .unwrap()
            .search_with_metric(&[0.0, 0.0], 5, &Cosine)
            .is_err());
        store.insert("zero".into(), vec![0.0, 0.0]).unwrap();
        assert!(store.search_with_metric(&query, 3, &Cosine).is_err());
    }

    #[test]
    fn mmr_handles_negative_similarity_and_returns_selection_order() {
        use crate::distance::Cosine;
        let mut store = Vekryx::new(2).unwrap();
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
        assert!(Vekryx::new(2)
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
            fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, Error> {
                Ok(left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum())
            }
        }
        let mut store = Vekryx::new(2).unwrap();
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
        let mut store = Vekryx::new(2).unwrap();
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
        assert!(Vekryx::new(0).is_err());
        let mut store = Vekryx::new(2).unwrap();
        assert!(store.insert("a".into(), vec![1.0]).is_err());
        assert!(store.insert("a".into(), vec![f64::NAN, 0.0]).is_err());
        assert!(store.insert("".into(), vec![0.0, 0.0]).is_err());
        store.insert("a".into(), vec![1.0, 2.0]).unwrap();
        store.insert("a".into(), vec![3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 1);
        let mut restored = Vekryx::from_json(&store.to_json().unwrap()).unwrap();
        assert_eq!(restored.get("a").unwrap(), &vec![3.0, 4.0]);
        assert!(restored.delete("a"));
        assert!(!restored.delete("a"));
        assert!(restored.search(&[0.0, 0.0], 5).unwrap().is_empty());
        assert!(restored.search(&[0.0], 5).is_err());
        assert!(Vekryx::from_json(r#"{"dimensions":2,"vectors":{"a":[1.0]}}"#).is_err());
    }
}
