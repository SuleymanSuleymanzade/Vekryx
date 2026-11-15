//! Multilayer HNSW with bounded best-first exploration and diversity pruning.
use super::{
    training::{distance, geometry, validate_corpus, Random},
    Candidate, CandidateIndex, DistanceKind, IndexBuilder,
};
use crate::{distance::DistanceMetric, search::Corpus, Error};
use serde::Deserialize;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub seed: u64,
}
impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            seed: 42,
        }
    }
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HnswSearchOptions {
    pub ef_search: usize,
}
impl Default for HnswSearchOptions {
    fn default() -> Self {
        Self { ef_search: 64 }
    }
}

#[derive(Debug)]
struct Node {
    links: Vec<Vec<usize>>,
}
#[derive(Debug)]
pub struct HnswIndex {
    ids: Vec<String>,
    nodes: Vec<Node>,
    entry: Option<usize>,
    m: usize,
}

#[derive(Clone, Copy, Debug)]
struct Scored {
    distance: f64,
    node: usize,
}
impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.node.cmp(&other.node))
    }
}

impl HnswIndex {
    fn layer_search(
        &self,
        entry: usize,
        layer: usize,
        ef: usize,
        score: impl Fn(usize) -> Result<f64, Error>,
    ) -> Result<Vec<Scored>, Error> {
        let first = Scored {
            node: entry,
            distance: score(entry)?,
        };
        let mut pending = BinaryHeap::from([Reverse(first)]);
        let mut best = BinaryHeap::from([first]);
        let mut visited = HashSet::from([entry]);
        while let Some(Reverse(current)) = pending.pop() {
            if best.len() >= ef && current > *best.peek().unwrap() {
                break;
            }
            for &neighbor in &self.nodes[current.node].links[layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Scored {
                    node: neighbor,
                    distance: score(neighbor)?,
                };
                if best.len() < ef || candidate < *best.peek().unwrap() {
                    pending.push(Reverse(candidate));
                    best.push(candidate);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        Ok(best.into_sorted_vec())
    }

    fn select(
        &self,
        candidates: Vec<Scored>,
        limit: usize,
        between: impl Fn(usize, usize) -> Result<f64, Error>,
    ) -> Result<Vec<usize>, Error> {
        let mut selected = Vec::new();
        let mut discarded = Vec::new();
        for candidate in candidates {
            let mut diverse = true;
            for &neighbor in &selected {
                if between(candidate.node, neighbor)? < candidate.distance {
                    diverse = false;
                    break;
                }
            }
            if diverse && selected.len() < limit {
                selected.push(candidate.node);
            } else {
                discarded.push(candidate.node);
            }
        }
        // Keep pruned connections when needed (also preserves useful duplicate-ID nodes).
        selected.extend(
            discarded
                .into_iter()
                .take(limit.saturating_sub(selected.len())),
        );
        Ok(selected)
    }
}

impl IndexBuilder for HnswConfig {
    type Index = HnswIndex;
    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<HnswIndex, Error> {
        geometry(metric, false)?;
        if self.m < 2 || self.m > usize::MAX / 2 || self.ef_construction < self.m {
            return Err("HNSW requires m >= 2 and ef_construction >= m".into());
        }
        validate_corpus(corpus, metric)?;
        let (ids, vectors): (Vec<_>, Vec<_>) =
            corpus.vectors().map(|(id, v)| (id.to_owned(), v)).unzip();
        let mut index = HnswIndex {
            ids,
            nodes: Vec::new(),
            entry: None,
            m: self.m,
        };
        let mut random = Random(self.seed);
        for point in 0..vectors.len() {
            // P(level >= l) = M^-l, bounded to avoid unbounded allocations.
            let mut level = 0;
            while level < 32 && random.next() % self.m as u64 == 0 {
                level += 1;
            }
            index.nodes.push(Node {
                links: vec![Vec::new(); level + 1],
            });
            let Some(mut entry) = index.entry else {
                index.entry = Some(point);
                continue;
            };
            let top = index.nodes[entry].links.len() - 1;
            let score = |i| distance(metric, vectors[point], vectors[i]);
            for layer in ((level + 1)..=top).rev() {
                entry = index.layer_search(entry, layer, 1, score)?[0].node;
            }
            for layer in (0..=level.min(top)).rev() {
                let candidates = index.layer_search(entry, layer, self.ef_construction, score)?;
                entry = candidates[0].node;
                let neighbors = index.select(candidates, index.m, |a, b| {
                    distance(metric, vectors[a], vectors[b])
                })?;
                index.nodes[point].links[layer] = neighbors.clone();
                for neighbor in neighbors {
                    index.nodes[neighbor].links[layer].push(point);
                    let cap = if layer == 0 { index.m * 2 } else { index.m };
                    if index.nodes[neighbor].links[layer].len() > cap {
                        let mut candidates = index.nodes[neighbor].links[layer]
                            .iter()
                            .map(|&node| {
                                Ok(Scored {
                                    node,
                                    distance: distance(metric, vectors[neighbor], vectors[node])?,
                                })
                            })
                            .collect::<Result<Vec<_>, Error>>()?;
                        candidates.sort();
                        index.nodes[neighbor].links[layer] =
                            index.select(candidates, cap, |a, b| {
                                distance(metric, vectors[a], vectors[b])
                            })?;
                    }
                }
            }
            if level > top {
                index.entry = Some(point);
            }
        }
        Ok(index)
    }
}

impl CandidateIndex for HnswIndex {
    type SearchOptions = HnswSearchOptions;
    fn candidates(
        &self,
        corpus: Corpus<'_>,
        query: &[f64],
        metric: &dyn DistanceMetric,
        limit: usize,
        options: &HnswSearchOptions,
    ) -> Result<Vec<Candidate>, Error> {
        if options.ef_search == 0 {
            return Err("ef_search must be positive".into());
        }
        let Some(mut entry) = self.entry else {
            return Ok(Vec::new());
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let score = |i: usize| {
            distance(
                metric,
                query,
                corpus
                    .get(&self.ids[i])
                    .ok_or("HNSW corpus no longer contains an indexed ID")?,
            )
        };
        for layer in (1..self.nodes[entry].links.len()).rev() {
            entry = self.layer_search(entry, layer, 1, score)?[0].node;
        }
        let ef = options.ef_search.max(limit.min(self.ids.len()));
        Ok(self
            .layer_search(entry, 0, ef, score)?
            .into_iter()
            .take(limit)
            .map(|hit| Candidate {
                id: self.ids[hit.node].clone(),
                distance: hit.distance,
                distance_kind: DistanceKind::Exact,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{distance::Euclidean, Vekryx};
    #[test]
    fn seeded_graph_has_multiple_layers_bounded_degrees_and_no_self_links() {
        let mut store = Vekryx::new(3).unwrap();
        for i in 0..256 {
            store
                .insert(
                    format!("{i:03}"),
                    vec![i as f64, (i % 17) as f64, (i % 11) as f64],
                )
                .unwrap();
        }
        let config = HnswConfig {
            m: 4,
            ef_construction: 30,
            seed: 2,
        };
        let index = config.build(store.view(), &Euclidean).unwrap();
        assert!(index.nodes.iter().any(|n| n.links.len() > 1));
        for (i, node) in index.nodes.iter().enumerate() {
            for (layer, neighbors) in node.links.iter().enumerate() {
                assert!(neighbors.len() <= if layer == 0 { config.m * 2 } else { config.m });
                assert!(!neighbors.contains(&i));
                let unique: HashSet<_> = neighbors.iter().collect();
                assert_eq!(unique.len(), neighbors.len());
                assert!(neighbors
                    .iter()
                    .all(|&n| index.nodes[n].links.len() > layer));
            }
        }
    }
}
