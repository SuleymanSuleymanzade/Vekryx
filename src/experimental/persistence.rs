//! JSON persistence for experimental indexes, independent of retrieval traits.
use serde::{de::DeserializeOwned, Serialize};
use std::{fs, io, path::Path};

pub struct LocalSaver;

impl LocalSaver {
    pub fn save_state<T: Serialize>(value: &T, path: impl AsRef<Path>) -> io::Result<()> {
        let json = serde_json::to_vec(value)?;
        crate::state_manager::snapshot::save(path.as_ref(), &json)
    }

    pub fn load_state<T: DeserializeOwned>(
        value: &mut T,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let restored = serde_json::from_slice(&fs::read(path)?)?;
        *value = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vekryx;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn serialization_is_independent_of_search_and_failed_load_preserves_state() {
        let path = std::env::temp_dir().join(format!(
            "vekryx-experimental-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut store = Vekryx::new(2).unwrap();
        store.insert("a".into(), vec![1.0, 2.0]).unwrap();
        LocalSaver::save_state(&store, &path).unwrap();
        let mut restored = Vekryx::new(3).unwrap();
        LocalSaver::load_state(&mut restored, &path).unwrap();
        assert_eq!(restored.get("a"), store.get("a"));
        fs::write(&path, r#"{"dimensions":0,"vectors":{}}"#).unwrap();
        assert!(LocalSaver::load_state(&mut restored, &path).is_err());
        assert_eq!(restored.dimensions(), 2);
        assert_eq!(restored.get("a"), store.get("a"));
        fs::remove_file(path).unwrap();
    }
}
