//! Shared deterministic training and checked numeric operations.
use crate::{
    distance::{Cosine, DistanceMetric, Euclidean, MetricKind},
    search::Corpus,
    Error,
};

pub(super) struct Random(pub u64);
impl Random {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

pub(super) fn geometry(
    metric: &dyn DistanceMetric,
    euclidean_only: bool,
) -> Result<MetricKind, Error> {
    match metric.kind() {
        Some(MetricKind::Euclidean) => Ok(MetricKind::Euclidean),
        Some(MetricKind::Cosine) if !euclidean_only => Ok(MetricKind::Cosine),
        _ => Err(if euclidean_only {
            "this backend requires Euclidean distance"
        } else {
            "this backend requires built-in Euclidean or cosine geometry"
        }
        .into()),
    }
}

pub(super) fn distance(metric: &dyn DistanceMetric, a: &[f64], b: &[f64]) -> Result<f64, Error> {
    let value = metric.distance(a, b)?;
    super::validate_distance(value)?;
    Ok(value)
}

pub(super) fn validate_corpus(
    corpus: Corpus<'_>,
    metric: &dyn DistanceMetric,
) -> Result<(), Error> {
    for (_, vector) in corpus.vectors() {
        distance(metric, vector, vector)?;
    }
    Ok(())
}

pub(super) fn normalize(vector: &[f64]) -> Result<Vec<f64>, Error> {
    let scale = vector.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    if scale == 0.0 {
        return Err("cosine distance is undefined for zero vectors".into());
    }
    let norm = vector
        .iter()
        .map(|x| (x / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(vector.iter().map(|x| (x / scale) / norm).collect())
}

pub(super) fn nearest(
    vector: &[f64],
    centers: &[Vec<f64>],
    kind: MetricKind,
) -> Result<usize, Error> {
    let mut best = None;
    for (i, center) in centers.iter().enumerate() {
        let d = match kind {
            MetricKind::Euclidean => distance(&Euclidean, vector, center)?,
            MetricKind::Cosine => distance(&Cosine, vector, center)?,
        };
        if best.is_none_or(|(_, current)| d < current) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| "no trained centroids".into())
}

/// Farthest-first seeded initialization and Lloyd updates; spherical for cosine.
/// Means use per-coordinate scaling to avoid overflowing sums of finite values.
/// Empty clusters keep their previous centroid, retaining duplicate records.
pub(super) fn kmeans(
    data: &[Vec<f64>],
    clusters: usize,
    iterations: usize,
    seed: u64,
    kind: MetricKind,
) -> Result<Vec<Vec<f64>>, Error> {
    if clusters == 0 || iterations == 0 {
        return Err("clusters and iterations must be positive".into());
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if clusters > data.len() {
        return Err("training requires at least as many vectors as centroids".into());
    }
    let metric: &dyn DistanceMetric = match kind {
        MetricKind::Euclidean => &Euclidean,
        MetricKind::Cosine => &Cosine,
    };
    let mut rng = Random(seed);
    let first = (rng.next() % data.len() as u64) as usize;
    let mut centers = vec![data[first].clone()];
    let mut chosen = vec![false; data.len()];
    chosen[first] = true;
    let mut closest = vec![f64::INFINITY; data.len()];
    while centers.len() < clusters {
        let last = centers.last().unwrap();
        let mut farthest = None;
        for (i, vector) in data.iter().enumerate() {
            closest[i] = closest[i].min(distance(metric, vector, last)?);
            if !chosen[i] && farthest.is_none_or(|(_, d)| closest[i] > d) {
                farthest = Some((i, closest[i]));
            }
        }
        let next = farthest.unwrap().0;
        chosen[next] = true;
        centers.push(data[next].clone());
    }
    let mut previous = vec![usize::MAX; data.len()];
    for _ in 0..iterations {
        let assignments = data
            .iter()
            .map(|v| nearest(v, &centers, kind))
            .collect::<Result<Vec<_>, _>>()?;
        if assignments == previous {
            break;
        }
        let mut members = vec![Vec::new(); clusters];
        for (i, &cluster) in assignments.iter().enumerate() {
            members[cluster].push(i);
        }
        for (cluster, indices) in members.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }
            let mut mean = vec![0.0; data[0].len()];
            for (dimension, value) in mean.iter_mut().enumerate() {
                let scale = indices
                    .iter()
                    .fold(0.0_f64, |s, &i| s.max(data[i][dimension].abs()));
                if scale != 0.0 {
                    let avg = indices
                        .iter()
                        .map(|&i| data[i][dimension] / scale / indices.len() as f64)
                        .sum::<f64>();
                    *value = avg.clamp(-1.0, 1.0) * scale;
                }
            }
            // Antipodal vectors can have a zero spherical mean. Keep a valid
            // member direction instead of inventing a cosine for zero.
            if kind == MetricKind::Cosine {
                mean = if mean.iter().all(|x| *x == 0.0) {
                    data[indices[0]].clone()
                } else {
                    normalize(&mean)?
                };
            }
            centers[cluster] = mean;
        }
        previous = assignments;
    }
    Ok(centers)
}

pub(super) fn residual(vector: &[f64], center: &[f64]) -> Result<Vec<f64>, Error> {
    let values: Vec<_> = vector.iter().zip(center).map(|(a, b)| a - b).collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("residual exceeds the finite f64 range".into());
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn means_handle_extreme_scales_duplicates_and_antipodal_cosine() {
        let large = vec![vec![1e308, 1e-320], vec![1e308, 2e-320]];
        let centers = kmeans(&large, 1, 3, 42, MetricKind::Euclidean).unwrap();
        assert_eq!(centers[0][0], 1e308);
        assert!(centers[0][1] > 0.0);
        let opposite = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let centers = kmeans(&opposite, 1, 3, 42, MetricKind::Cosine).unwrap();
        assert!(Cosine.distance(&centers[0], &centers[0]).is_ok());
        assert!(kmeans(
            &[vec![-1e308], vec![1e308]],
            2,
            3,
            42,
            MetricKind::Euclidean
        )
        .is_err());
        assert!(residual(&[1e308], &[-1e308]).is_err());
        assert!(normalize(&[1e-320, 1e-320])
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }
}
