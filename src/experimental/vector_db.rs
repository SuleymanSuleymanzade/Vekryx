pub type Vector = [f32; 3];

#[derive(Debug, Default)]
pub struct LinearIndex {
    vectors: Vec<Vector>,
}

impl LinearIndex {
    pub fn new() -> Self {
        LinearIndex {
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }

    pub fn get_vector(&self, index: usize) -> Option<&Vector> {
        self.vectors.get(index)
    }

    pub fn find_closest(&self, query: Vector) -> Option<&Vector> {
        self.vectors.iter().min_by(|&a, &b| {
            let distance_a = LinearIndex::euclidean_distance(&query, a);
            let distance_b = LinearIndex::euclidean_distance(&query, b);
            distance_a.total_cmp(&distance_b)
        })
    }

    pub fn euclidean_distance(left: &Vector, right: &Vector) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl super::fast_search::FastSearch for LinearIndex {
    type Vector = Vector;

    fn nearest_neighbor(&self, query: &Vector) -> Option<&Vector> {
        self.find_closest(*query)
    }
}
