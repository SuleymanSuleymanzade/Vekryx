//! Product quantization with asymmetric lookup-table distance estimates.
//! Standalone PQ scans all codes; IVF-PQ scans selected residual-code lists.
use super::{
    ivf::CoarseIndex,
    training::{geometry, kmeans, nearest, residual},
    validate_distance, Candidate, CandidateIndex, DistanceKind, IndexBuilder, IvfConfig,
    IvfSearchOptions,
};
use crate::{
    distance::{DistanceMetric, Euclidean, MetricKind},
    search::Corpus,
    Error,
};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PqConfig {
    pub subquantizers: usize,
    /// One byte per subquantizer is stored, with 2^bits trained centroids.
    /// Codes below 8 bits are not bit-packed.
    pub bits: u8,
    pub iterations: usize,
    pub seed: u64,
}
impl Default for PqConfig {
    fn default() -> Self {
        Self {
            subquantizers: 4,
            bits: 4,
            iterations: 20,
            seed: 42,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IvfPqConfig {
    pub ivf: IvfConfig,
    pub pq: PqConfig,
}

#[derive(Debug)]
struct ProductQuantizer {
    dimensions: usize,
    width: usize,
    books: Vec<Vec<Vec<f64>>>,
}
impl ProductQuantizer {
    fn train(data: &[Vec<f64>], dimensions: usize, config: &PqConfig) -> Result<Self, Error> {
        if config.subquantizers == 0 || dimensions % config.subquantizers != 0 {
            return Err(
                "dimensions must be divisible by a positive number of subquantizers".into(),
            );
        }
        if !(1..=8).contains(&config.bits) || config.iterations == 0 {
            return Err("PQ requires bits in 1..=8 and positive iterations".into());
        }
        let width = dimensions / config.subquantizers;
        let mut books = Vec::with_capacity(config.subquantizers);
        for sub in 0..config.subquantizers {
            let training: Vec<_> = data
                .iter()
                .map(|v| v[sub * width..(sub + 1) * width].to_vec())
                .collect();
            books.push(kmeans(
                &training,
                1usize << config.bits,
                config.iterations,
                config.seed.wrapping_add(sub as u64),
                MetricKind::Euclidean,
            )?);
        }
        Ok(Self {
            dimensions,
            width,
            books,
        })
    }
    fn encode(&self, vector: &[f64]) -> Result<Vec<u8>, Error> {
        vector
            .chunks_exact(self.width)
            .zip(&self.books)
            .map(|(part, book)| Ok(nearest(part, book, MetricKind::Euclidean)? as u8))
            .collect()
    }
    // Tables contain L2 norms per subspace. Combining with hypot is equivalent
    // to sqrt(sum of squared lookup distances), without overflowing squares.
    fn table(&self, query: &[f64]) -> Result<Vec<Vec<f64>>, Error> {
        if query.len() != self.dimensions {
            return Err("PQ query dimensions do not match training".into());
        }
        query
            .chunks_exact(self.width)
            .zip(&self.books)
            .map(|(part, book)| {
                book.iter()
                    .map(|center| super::training::distance(&Euclidean, part, center))
                    .collect()
            })
            .collect()
    }
    fn estimate(&self, table: &[Vec<f64>], code: &[u8]) -> Result<f64, Error> {
        let distance = code
            .iter()
            .zip(table)
            .fold(0.0_f64, |acc, (&code, row)| acc.hypot(row[code as usize]));
        validate_distance(distance)?;
        Ok(distance)
    }
}

#[derive(Debug)]
pub struct PqIndex {
    ids: Vec<String>,
    quantizer: ProductQuantizer,
    codes: Vec<u8>,
}
impl PqIndex {
    /// Code payload only; excludes IDs, codebooks, allocation overhead and store.
    pub fn code_bytes_per_vector(&self) -> usize {
        self.quantizer.books.len()
    }
}
impl IndexBuilder for PqConfig {
    type Index = PqIndex;
    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<PqIndex, Error> {
        geometry(metric, true)?;
        let (ids, data): (Vec<_>, Vec<_>) = corpus
            .vectors()
            .map(|(id, v)| (id.to_owned(), v.to_vec()))
            .unzip();
        let quantizer = ProductQuantizer::train(&data, corpus.dimensions(), self)?;
        let codes = data
            .iter()
            .map(|v| quantizer.encode(v))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(PqIndex {
            ids,
            quantizer,
            codes,
        })
    }
}
impl CandidateIndex for PqIndex {
    type SearchOptions = ();
    fn candidates(
        &self,
        _corpus: Corpus<'_>,
        query: &[f64],
        _metric: &dyn DistanceMetric,
        limit: usize,
        _: &(),
    ) -> Result<Vec<Candidate>, Error> {
        if self.ids.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let table = self.quantizer.table(query)?;
        let mut candidates = self
            .ids
            .iter()
            .zip(self.codes.chunks_exact(self.code_bytes_per_vector()))
            .map(|(id, code)| {
                Ok(Candidate {
                    id: id.clone(),
                    distance: self.quantizer.estimate(&table, code)?,
                    distance_kind: DistanceKind::Approximate,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        super::rank_candidates(&mut candidates, limit);
        Ok(candidates)
    }
}

#[derive(Debug)]
pub struct IvfPqIndex {
    coarse: CoarseIndex,
    quantizer: ProductQuantizer,
    codes: Vec<u8>,
}
impl IvfPqIndex {
    /// Code payload only; canonical vectors remain available for exact reranking.
    pub fn code_bytes_per_vector(&self) -> usize {
        self.quantizer.books.len()
    }
}
impl IndexBuilder for IvfPqConfig {
    type Index = IvfPqIndex;
    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<IvfPqIndex, Error> {
        geometry(metric, true)?;
        let coarse = CoarseIndex::train(corpus, metric, &self.ivf)?;
        let mut residuals = vec![Vec::new(); coarse.ids.len()];
        for (list, members) in coarse.lists.iter().enumerate() {
            for &i in members {
                residuals[i] =
                    residual(corpus.get(&coarse.ids[i]).unwrap(), &coarse.centers[list])?;
            }
        }
        let quantizer = ProductQuantizer::train(&residuals, corpus.dimensions(), &self.pq)?;
        let codes = residuals
            .iter()
            .map(|v| quantizer.encode(v))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(IvfPqIndex {
            coarse,
            quantizer,
            codes,
        })
    }
}
impl CandidateIndex for IvfPqIndex {
    type SearchOptions = IvfSearchOptions;
    fn candidates(
        &self,
        _corpus: Corpus<'_>,
        query: &[f64],
        metric: &dyn DistanceMetric,
        limit: usize,
        options: &IvfSearchOptions,
    ) -> Result<Vec<Candidate>, Error> {
        let probes = self.coarse.probes(query, metric, options.nprobe)?;
        let mut candidates = Vec::new();
        if limit == 0 {
            return Ok(candidates);
        }
        let m = self.code_bytes_per_vector();
        for list in probes {
            let query_residual = residual(query, &self.coarse.centers[list])?;
            let table = self.quantizer.table(&query_residual)?;
            for &i in &self.coarse.lists[list] {
                candidates.push(Candidate {
                    id: self.coarse.ids[i].clone(),
                    distance: self
                        .quantizer
                        .estimate(&table, &self.codes[i * m..(i + 1) * m])?,
                    distance_kind: DistanceKind::Approximate,
                });
            }
        }
        super::rank_candidates(&mut candidates, limit);
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vekryx;

    #[test]
    fn asymmetric_tables_match_independently_reconstructed_vectors() {
        let mut store = Vekryx::new(4).unwrap();
        for i in 0..24 {
            store
                .insert(
                    format!("{i:02}"),
                    vec![i as f64, (i % 7) as f64, (i % 5) as f64, -(i as f64)],
                )
                .unwrap();
        }
        let config = PqConfig {
            subquantizers: 2,
            bits: 2,
            iterations: 10,
            seed: 3,
        };
        let index = config.build(store.view(), &Euclidean).unwrap();
        let combined = IvfPqConfig {
            ivf: IvfConfig {
                nlist: 3,
                iterations: 10,
                seed: 7,
            },
            pq: config,
        }
        .build(store.view(), &Euclidean)
        .unwrap();
        let query = [7.5, 2.0, 3.0, -7.0];
        let table = index.quantizer.table(&query).unwrap();
        for code in index.codes.chunks_exact(2) {
            let reconstructed: Vec<_> = code
                .iter()
                .enumerate()
                .flat_map(|(sub, &code)| index.quantizer.books[sub][code as usize].clone())
                .collect();
            let expected = Euclidean.distance(&query, &reconstructed).unwrap();
            assert!((index.quantizer.estimate(&table, code).unwrap() - expected).abs() < 1e-12);
        }
        for (list, members) in combined.coarse.lists.iter().enumerate() {
            let center = &combined.coarse.centers[list];
            let table = combined
                .quantizer
                .table(&residual(&query, center).unwrap())
                .unwrap();
            for &i in members {
                let code = &combined.codes[i * 2..i * 2 + 2];
                let decoded_residual: Vec<_> = code
                    .iter()
                    .enumerate()
                    .flat_map(|(sub, &code)| combined.quantizer.books[sub][code as usize].clone())
                    .collect();
                let reconstructed: Vec<_> = center
                    .iter()
                    .zip(decoded_residual)
                    .map(|(a, b)| a + b)
                    .collect();
                let expected = Euclidean.distance(&query, &reconstructed).unwrap();
                assert!(
                    (combined.quantizer.estimate(&table, code).unwrap() - expected).abs() < 1e-12
                );
            }
        }
    }
}
