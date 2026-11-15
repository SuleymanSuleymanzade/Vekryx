use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HashGenerator {
    hash_size: usize,
    dimension_size: usize,
    projections: Array2<f64>,
    hash_map: HashMap<String, Array1<f64>>,
}

impl Serialize for HashGenerator {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HashGenerator", 4)?;
        state.serialize_field("hash_size", &self.hash_size)?;
        state.serialize_field("dimention_size", &self.dimension_size)?;
        state.serialize_field("projections", &self.projections.as_slice().unwrap())?;
        let hash_map_serialized: HashMap<String, Vec<f64>> = self
            .hash_map
            .iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect();
        state.serialize_field("hash_map", &hash_map_serialized)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HashGenerator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct HashGeneratorHelper {
            hash_size: usize,
            #[serde(rename = "dimention_size", alias = "dimension_size")]
            dimension_size: usize,
            projections: Vec<f64>,
            hash_map: HashMap<String, Vec<f64>>,
        }

        let helper = HashGeneratorHelper::deserialize(deserializer)?;
        let projections = Array2::from_shape_vec(
            (helper.hash_size, helper.dimension_size),
            helper.projections,
        )
        .map_err(serde::de::Error::custom)?;

        let hash_map: HashMap<String, Array1<f64>> = helper
            .hash_map
            .into_iter()
            .map(|(k, v)| (k, Array1::from(v)))
            .collect();

        Ok(HashGenerator {
            hash_size: helper.hash_size,
            dimension_size: helper.dimension_size,
            projections,
            hash_map,
        })
    }
}

impl HashGenerator {
    pub fn new(hash_size: usize, dimension_size: usize) -> Self {
        let mut rng = thread_rng();
        let projections =
            Array2::from_shape_fn((hash_size, dimension_size), |_| rng.gen_range(-1.0..1.0));

        HashGenerator {
            hash_size,
            dimension_size,
            projections,
            hash_map: HashMap::new(),
        }
    }

    pub fn generate_hash(&self, vector: &Array1<f64>) -> String {
        let dot_product = vector.dot(&self.projections.t());
        dot_product
            .iter()
            .map(|&x| if x > 0.0 { '1' } else { '0' })
            .collect()
    }

    pub fn insert(&mut self, vector: &Array1<f64>) {
        let hash_value = self.generate_hash(vector);
        self.hash_map.insert(hash_value, vector.clone());
    }

    pub fn get(&self, vector: &Array1<f64>) -> Option<Array1<f64>> {
        let hash_value = self.generate_hash(vector);
        self.hash_map.get(&hash_value).cloned()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LSH {
    num_tables: usize,
    hash_size: usize,
    dimension_size: usize,
    generators: Vec<HashGenerator>,
}

impl LSH {
    pub fn new(num_tables: usize, hash_size: usize, dimension_size: usize) -> Self {
        let mut generators = vec![];
        for _ in 0..num_tables {
            let hash_generator = HashGenerator::new(hash_size, dimension_size);
            generators.push(hash_generator)
        }

        Self {
            num_tables,
            hash_size,
            dimension_size,
            generators,
        }
    }

    pub fn insert(&mut self, vector: &Array1<f64>) {
        for item in self.generators.iter_mut() {
            item.insert(vector);
        }
    }

    pub fn nearest(&self, vector: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut candidates = Vec::new();
        for item in self.generators.iter() {
            let candidate = item.get(vector);
            if let Some(data) = candidate {
                candidates.push(data);
            }
        }
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn identical_vectors_are_retrieved_from_every_table() {
        let mut lsh = LSH::new(5, 6, 5);
        let vector = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        lsh.insert(&vector);
        let candidates = lsh.nearest(&vector);
        assert_eq!(candidates.len(), 5);
        assert!(candidates.iter().all(|candidate| candidate == vector));
    }
}
