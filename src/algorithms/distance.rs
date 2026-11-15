//! Extensible vector metrics. Implement `DistanceMetric` to add a metric without
//! changing storage or ranking. Distances must be finite and nonnegative; similarity is finite and larger
//! for more related vectors and should have a comparable scale across pairs.

pub trait DistanceMetric: Send + Sync {
    fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, String>;

    /// Similarity used by MMR. Override when the metric has a natural similarity.
    fn similarity(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
        Ok(1.0 / (1.0 + self.distance(left, right)?))
    }
}

fn validate_pair(left: &[f64], right: &[f64]) -> Result<(), String> {
    if left.is_empty() || left.len() != right.len() {
        return Err("vectors must have the same positive dimension".into());
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err("vector values must be finite".into());
    }
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
        validate_pair(left, right)?;
        let distance = left
            .iter()
            .zip(right)
            .fold(0.0_f64, |d, (a, b)| d.hypot(a - b));
        if !distance.is_finite() {
            return Err("Euclidean distance exceeds the finite f64 range".into());
        }
        Ok(distance)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Cosine;

impl DistanceMetric for Cosine {
    fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
        Ok(1.0 - self.similarity(left, right)?)
    }

    fn similarity(&self, left: &[f64], right: &[f64]) -> Result<f64, String> {
        validate_pair(left, right)?;
        // Scale before computing norms to handle both huge and subnormal values.
        let left_scale = left.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        let right_scale = right.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        if left_scale == 0.0 || right_scale == 0.0 {
            return Err("cosine distance is undefined for zero vectors".into());
        }
        let left_norm = left
            .iter()
            .map(|x| (x / left_scale).powi(2))
            .sum::<f64>()
            .sqrt();
        let right_norm = right
            .iter()
            .map(|x| (x / right_scale).powi(2))
            .sum::<f64>()
            .sqrt();
        let dot: f64 = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a / left_scale / left_norm) * (b / right_scale / right_norm))
            .sum();
        Ok(dot.clamp(-1.0, 1.0))
    }
}

/// Python's named metric registry. Rust callers can supply any implementation.
pub fn named_metric(name: &str) -> Result<Box<dyn DistanceMetric>, String> {
    match name {
        "euclidean" => Ok(Box::new(Euclidean)),
        "cosine" => Ok(Box::new(Cosine)),
        _ => Err(format!(
            "unknown metric {name:?}; expected 'euclidean' or 'cosine'"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_distances_and_cosine_extreme_magnitudes() {
        assert_eq!(Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap(), 5.0);
        assert_eq!(Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 1.0);
        assert_eq!(Cosine.distance(&[1.0, 0.0], &[-2.0, 0.0]).unwrap(), 2.0);
        for scale in [1e308, 1e-320] {
            assert!(Cosine.distance(&[scale, scale], &[scale, scale]).unwrap() < 1e-14);
        }
        assert!(Cosine.distance(&[0.0], &[1.0]).is_err());
        assert!(Euclidean.distance(&[f64::MAX], &[-f64::MAX]).is_err());
        assert!(Cosine.distance(&[f64::NAN], &[1.0]).is_err());
        assert!(Euclidean.distance(&[1.0], &[1.0, 2.0]).is_err());
        assert!(named_metric("unknown").is_err());
    }
}
