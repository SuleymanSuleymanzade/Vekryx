//! Trained inverted lists with exact distances inside the selected lists.
use super::{
    training::{distance, geometry, kmeans, nearest, normalize, validate_corpus},
    Candidate, CandidateIndex, DistanceKind, IndexBuilder,
};
use crate::{
    distance::{DistanceMetric, MetricKind},
    search::Corpus,
    Error,
};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IvfConfig {
    pub nlist: usize,
    pub iterations: usize,
    pub seed: u64,
}
impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            nlist: 16,
            iterations: 20,
            seed: 42,
        }
    }
}
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IvfSearchOptions {
    pub nprobe: usize,
}
impl Default for IvfSearchOptions {
    fn default() -> Self {
        Self { nprobe: 1 }
    }
}

#[derive(Debug)]
pub(super) struct CoarseIndex {
    pub ids: Vec<String>,
    pub centers: Vec<Vec<f64>>,
    pub lists: Vec<Vec<usize>>,
    pub nlist: usize,
}
impl CoarseIndex {
    pub fn train(
        corpus: Corpus<'_>,
        metric: &dyn DistanceMetric,
        config: &IvfConfig,
    ) -> Result<Self, Error> {
        let kind = geometry(metric, false)?;
        if config.nlist == 0 || config.iterations == 0 {
            return Err("nlist and iterations must be positive".into());
        }
        validate_corpus(corpus, metric)?;
        let (ids, data): (Vec<_>, Vec<_>) = corpus
            .vectors()
            .map(|(id, vector)| {
                Ok((
                    id.to_owned(),
                    if kind == MetricKind::Cosine {
                        normalize(vector)?
                    } else {
                        vector.to_vec()
                    },
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .unzip();
        let centers = kmeans(&data, config.nlist, config.iterations, config.seed, kind)?;
        let mut lists = vec![Vec::new(); centers.len()];
        for (i, vector) in data.iter().enumerate() {
            lists[nearest(vector, &centers, kind)?].push(i);
        }
        Ok(Self {
            ids,
            centers,
            lists,
            nlist: config.nlist,
        })
    }

    pub fn probes(
        &self,
        query: &[f64],
        metric: &dyn DistanceMetric,
        nprobe: usize,
    ) -> Result<Vec<usize>, Error> {
        if nprobe == 0 || nprobe > self.nlist {
            return Err("nprobe must be between 1 and nlist".into());
        }
        let mut ranked = self
            .centers
            .iter()
            .enumerate()
            .map(|(i, center)| Ok((i, distance(metric, query, center)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        Ok(ranked.into_iter().take(nprobe).map(|(i, _)| i).collect())
    }
}

#[derive(Debug)]
pub struct IvfIndex {
    pub(super) coarse: CoarseIndex,
}
impl IndexBuilder for IvfConfig {
    type Index = IvfIndex;
    fn build(&self, corpus: Corpus<'_>, metric: &dyn DistanceMetric) -> Result<IvfIndex, Error> {
        Ok(IvfIndex {
            coarse: CoarseIndex::train(corpus, metric, self)?,
        })
    }
}
impl CandidateIndex for IvfIndex {
    type SearchOptions = IvfSearchOptions;
    fn candidates(
        &self,
        corpus: Corpus<'_>,
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
        for list in probes {
            for &i in &self.coarse.lists[list] {
                let id = &self.coarse.ids[i];
                let vector = corpus
                    .get(id)
                    .ok_or("IVF corpus no longer contains an indexed ID")?;
                candidates.push(Candidate {
                    id: id.clone(),
                    distance: distance(metric, query, vector)?,
                    distance_kind: DistanceKind::Exact,
                });
            }
        }
        super::rank_candidates(&mut candidates, limit);
        Ok(candidates)
    }
}
