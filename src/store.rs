use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Store {
    dimensions: usize,
    vectors: BTreeMap<String, Vec<f64>>,
}

impl Store {
    pub fn new(dimensions: usize) -> Result<Self, String> {
        if dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
        }
        Ok(Self {
            dimensions,
            vectors: BTreeMap::new(),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn get(&self, id: &str) -> Option<&Vec<f64>> {
        self.vectors.get(id)
    }
    pub fn delete(&mut self, id: &str) -> bool {
        self.vectors.remove(id).is_some()
    }

    fn validate(&self, vector: &[f64]) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err("vector values must be finite".into());
        }
        Ok(())
    }

    pub fn insert(&mut self, id: String, vector: Vec<f64>) -> Result<(), String> {
        if id.is_empty() {
            return Err("id must not be empty".into());
        }
        self.validate(&vector)?;
        self.vectors.insert(id, vector);
        Ok(())
    }

    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(String, f64)>, String> {
        self.validate(query)?;
        let mut results: Vec<_> = self
            .vectors
            .iter()
            .map(|(id, vector)| {
                // hypot avoids unnecessary overflow when squaring large coordinates.
                let distance = vector
                    .iter()
                    .zip(query)
                    .fold(0.0_f64, |distance, (a, b)| distance.hypot(a - b));
                (id.clone(), distance)
            })
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        Ok(results)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let store: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        if store.dimensions == 0 {
            return Err("dimensions must be greater than zero".into());
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
    use super::Store;

    #[test]
    fn ranks_preserves_collisions_and_breaks_ties_by_id() {
        let mut store = Store::new(2).unwrap();
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
        assert!(Store::new(0).is_err());
        let mut store = Store::new(2).unwrap();
        assert!(store.insert("a".into(), vec![1.0]).is_err());
        assert!(store.insert("a".into(), vec![f64::NAN, 0.0]).is_err());
        assert!(store.insert("".into(), vec![0.0, 0.0]).is_err());
        store.insert("a".into(), vec![1.0, 2.0]).unwrap();
        store.insert("a".into(), vec![3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 1);
        let mut restored = Store::from_json(&store.to_json().unwrap()).unwrap();
        assert_eq!(restored.get("a").unwrap(), &vec![3.0, 4.0]);
        assert!(restored.delete("a"));
        assert!(!restored.delete("a"));
        assert!(restored.search(&[0.0, 0.0], 5).unwrap().is_empty());
        assert!(restored.search(&[0.0], 5).is_err());
        assert!(Store::from_json(r#"{"dimensions":2,"vectors":{"a":[1.0]}}"#).is_err());
    }
}
