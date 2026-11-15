//! A fixed, three-dimensional experimental KD-tree.
use super::{
    fast_search::FastSearch,
    vector_db::{LinearIndex, Vector},
};
use crate::Error;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Leaf(Vector),
    Branch {
        left: Box<Node>,
        right: Box<Node>,
        split_value: f32,
        split_dimension: usize,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct KDTreeConfig {
    pub initial_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KDTree {
    root: Option<Node>,
    config: KDTreeConfig,
}

impl KDTree {
    pub fn new(points: Vec<Vector>, config: KDTreeConfig) -> Result<Self, Error> {
        for point in &points {
            validate(point)?;
        }
        Ok(Self {
            root: build(points, config.initial_depth % 3),
            config,
        })
    }

    pub fn insert(&mut self, point: Vector) -> Result<(), Error> {
        validate(&point)?;
        match &mut self.root {
            Some(root) => insert(root, point, self.config.initial_depth % 3),
            None => self.root = Some(Node::Leaf(point)),
        }
        Ok(())
    }
}

fn validate(point: &Vector) -> Result<(), Error> {
    if point.iter().any(|value| !value.is_finite()) {
        return Err("vector values must be finite".into());
    }
    Ok(())
}

fn build(mut points: Vec<Vector>, dimension: usize) -> Option<Node> {
    match points.len() {
        0 => None,
        1 => Some(Node::Leaf(points[0])),
        _ => {
            points.sort_by(|a, b| a[dimension].total_cmp(&b[dimension]));
            let right = points.split_off(points.len() / 2);
            let split_value = right[0][dimension];
            let next_dimension = (dimension + 1) % 3;
            Some(Node::Branch {
                // Both halves are nonempty for any input of at least two points.
                left: Box::new(build(points, next_dimension).unwrap()),
                right: Box::new(build(right, next_dimension).unwrap()),
                split_value,
                split_dimension: dimension,
            })
        }
    }
}

fn insert(node: &mut Node, point: Vector, dimension: usize) {
    match node {
        Node::Leaf(existing) => *node = build(vec![*existing, point], dimension).unwrap(),
        Node::Branch {
            left,
            right,
            split_value,
            split_dimension,
        } => {
            let next = if point[*split_dimension] < *split_value {
                left
            } else {
                right
            };
            insert(next, point, (*split_dimension + 1) % 3);
        }
    }
}

fn nearest<'a>(node: &'a Node, query: &Vector, best: &mut Option<(&'a Vector, f32)>) {
    match node {
        Node::Leaf(point) => {
            let distance = LinearIndex::euclidean_distance(query, point);
            if best.is_none_or(|(_, current)| distance < current) {
                *best = Some((point, distance));
            }
        }
        Node::Branch {
            left,
            right,
            split_value,
            split_dimension,
        } => {
            let (near, far) = if query[*split_dimension] < *split_value {
                (left, right)
            } else {
                (right, left)
            };
            nearest(near, query, best);
            if best.is_none_or(|(_, distance)| {
                (query[*split_dimension] - split_value).abs() <= distance
            }) {
                nearest(far, query, best);
            }
        }
    }
}

impl FastSearch for KDTree {
    type Vector = Vector;

    fn nearest_neighbor(&self, query: &Vector) -> Option<&Vector> {
        validate(query).ok()?;
        let mut best = None;
        nearest(self.root.as_ref()?, query, &mut best);
        best.map(|(point, _)| point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branching_insertion_and_round_trip_match_linear_search() {
        let points = vec![[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]];
        for initial_depth in [0, 1, 2, usize::MAX] {
            let mut linear = LinearIndex::new();
            for point in &points {
                linear.add_vector(*point);
            }
            let mut tree = KDTree::new(points.clone(), KDTreeConfig { initial_depth }).unwrap();
            tree.insert([2.0, 1.0, -3.0]).unwrap();
            linear.add_vector([2.0, 1.0, -3.0]);
            let restored: KDTree =
                serde_json::from_str(&serde_json::to_string(&tree).unwrap()).unwrap();
            for query in [[0.0; 3], [10.0; 3], [-5.0; 3], [2.0, 1.0, -3.0]] {
                // Different indexes may choose different points at equal distance.
                let nearest = tree.nearest_neighbor(&query).unwrap();
                let expected = linear.nearest_neighbor(&query).unwrap();
                assert_eq!(
                    LinearIndex::euclidean_distance(&query, nearest),
                    LinearIndex::euclidean_distance(&query, expected)
                );
                assert_eq!(restored.nearest_neighbor(&query), Some(nearest));
            }
        }
    }

    #[test]
    fn empty_tree_insertion_and_invalid_input() {
        let mut tree = KDTree::new(vec![], KDTreeConfig::default()).unwrap();
        assert!(tree.nearest_neighbor(&[0.0; 3]).is_none());
        tree.insert([1.0; 3]).unwrap();
        tree.insert([2.0; 3]).unwrap();
        assert!(tree.insert([f32::NAN; 3]).is_err());
        assert_eq!(tree.nearest_neighbor(&[1.1; 3]), Some(&[1.0; 3]));
        assert!(KDTree::new(vec![[f32::NAN; 3]], KDTreeConfig::default()).is_err());
    }
}
