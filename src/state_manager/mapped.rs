use super::snapshot::{invalid_data, write_atomic};
use crate::distance::{DistanceMetric, Euclidean};
use crate::index::{exact_candidates, validate_distance};
use crate::store::{validate_vector, Record};
use crate::{Error, Vekryx};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

// Version 1 stores magic, u64 dimensions, and a u64 metadata offset,
// followed by aligned f64 rows and a JSON array of (ID, optional record) pairs.
// All binary fields are little-endian and rows are ordered by ID.
const MAGIC: &[u8; 8] = b"VEKMAP01";
const HEADER_LEN: usize = 24;
const VALUE_LEN: usize = size_of::<f64>();

/// A read-only snapshot with on-demand access to file-backed vectors.
/// IDs and records are loaded into memory when opening the snapshot.
#[derive(Debug)]
pub struct MappedVekryx {
    map: Mmap,
    dimensions: usize,
    vector_end: usize,
    entries: Vec<(String, Option<Record>)>,
}

impl Vekryx {
    /// Atomically replace a versioned memory-mappable snapshot.
    pub fn save_mapped(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let vector_end = self
            .len()
            .checked_mul(self.dimensions())
            .and_then(|count| count.checked_mul(VALUE_LEN))
            .and_then(|bytes| bytes.checked_add(HEADER_LEN))
            .ok_or_else(|| invalid_data("mapped snapshot size overflow"))?;

        write_atomic(path.as_ref(), |file| {
            let mut writer = BufWriter::new(file);
            writer.write_all(MAGIC)?;
            writer.write_all(&(self.dimensions() as u64).to_le_bytes())?;
            writer.write_all(&(vector_end as u64).to_le_bytes())?;

            let mut entries = Vec::with_capacity(self.len());
            for (id, vector) in self.vectors() {
                for value in vector {
                    writer.write_all(&value.to_le_bytes())?;
                }
                entries.push((id, self.record(id)));
            }
            serde_json::to_writer(&mut writer, &entries)?;
            writer.flush()
        })
    }
}

impl MappedVekryx {
    /// Open a version-1 snapshot on a little-endian host.
    ///
    /// # Safety
    ///
    /// The underlying file must not be modified or truncated while this mapping
    /// is being opened or remains alive, including via another process or hard
    /// link. Replacing its directory entry atomically with a new snapshot is
    /// compatible with existing mappings on filesystems that support it.
    pub unsafe fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "mapped snapshots require a little-endian host",
            ));
        }
        let file = File::open(path)?;
        if file.metadata()?.len() < HEADER_LEN as u64 {
            return Err(invalid_data("missing mapped snapshot header"));
        }
        // SAFETY: The caller guarantees the mapped file remains immutable.
        let map = unsafe { Mmap::map(&file)? };
        if map.get(..8) != Some(MAGIC.as_slice()) {
            return Err(invalid_data("unsupported mapped snapshot header"));
        }
        let dimensions = read_usize(&map[8..16])?;
        let vector_end = read_usize(&map[16..HEADER_LEN])?;
        let row_bytes = dimensions
            .checked_mul(VALUE_LEN)
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| invalid_data("invalid mapped snapshot dimensions"))?;
        if !(HEADER_LEN..map.len()).contains(&vector_end)
            || (vector_end - HEADER_LEN) % row_bytes != 0
        {
            return Err(invalid_data("invalid mapped vector region"));
        }

        let entries: Vec<(String, Option<Record>)> =
            serde_json::from_slice(&map[vector_end..]).map_err(invalid_data)?;
        if entries.len() != (vector_end - HEADER_LEN) / row_bytes
            || entries.iter().any(|(id, _)| id.is_empty())
            || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(invalid_data("invalid mapped snapshot entries"));
        }
        Ok(Self {
            map,
            dimensions,
            vector_end,
            entries,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow a vector, validating its values on access.
    pub fn get(&self, id: &str) -> Result<Option<&[f64]>, Error> {
        self.position(id)
            .map(|index| {
                self.checked_vector(&self.rows()[index * self.dimensions..][..self.dimensions])
            })
            .transpose()
    }

    pub fn record(&self, id: &str) -> Option<&Record> {
        self.position(id)
            .and_then(|index| self.entries[index].1.as_ref())
    }

    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<(String, f64)>, Error> {
        self.search_with_metric(query, k, &Euclidean)
    }

    pub fn search_with_metric(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> Result<Vec<(String, f64)>, Error> {
        validate_vector(self.dimensions, query)?;
        validate_distance(metric.distance(query, query)?)?;
        let vectors = self
            .entries
            .iter()
            .zip(self.rows().chunks_exact(self.dimensions))
            .map(|((id, _), vector)| Ok((id.as_str(), self.checked_vector(vector)?)));
        Ok(exact_candidates(vectors, query, metric, k)?
            .into_iter()
            .map(|candidate| (candidate.id, candidate.distance))
            .collect())
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.entries
            .binary_search_by(|entry| entry.0.as_str().cmp(id))
            .ok()
    }

    fn rows(&self) -> &[f64] {
        let bytes = &self.map[HEADER_LEN..self.vector_end];
        // SAFETY: The mapping starts at a page boundary, the header is aligned
        // for f64, all f64 bit patterns are valid, and the file is immutable.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / VALUE_LEN) }
    }

    fn checked_vector<'a>(&self, vector: &'a [f64]) -> Result<&'a [f64], Error> {
        validate_vector(self.dimensions, vector)
            .map_err(|error| Error::InvalidSnapshot(error.to_string()))?;
        Ok(vector)
    }
}

fn read_usize(bytes: &[u8]) -> io::Result<usize> {
    usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap()))
        .map_err(|_| invalid_data("mapped snapshot header value exceeds platform limits"))
}

#[cfg(all(test, target_endian = "little"))]
mod tests {
    use super::*;
    use crate::distance::Cosine;

    #[test]
    fn mapped_snapshot_preserves_records_and_search_order() {
        let directory = std::env::temp_dir().join(format!(
            "vekryx-mapped-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("store.vmap");

        let mut store = Vekryx::new(2).unwrap();
        store.save_mapped(&path).unwrap();
        // SAFETY: The file stays immutable until this mapping is dropped.
        assert!(unsafe { MappedVekryx::open(&path) }.unwrap().is_empty());
        store.insert("b".into(), vec![1.0, 0.0]).unwrap();
        store.insert("c".into(), vec![0.0, 1.0]).unwrap();
        store
            .insert_record(
                "a".into(),
                vec![1.0, 0.0],
                Record {
                    text: Some("first".into()),
                    metadata: serde_json::from_value(serde_json::json!({"group": "test"})).unwrap(),
                },
            )
            .unwrap();
        store.save_mapped(&path).unwrap();
        // SAFETY: All in-place writes occur after the mapping is dropped.
        let mapped = unsafe { MappedVekryx::open(&path) }.unwrap();
        assert_eq!(mapped.dimensions(), 2);
        assert_eq!(mapped.len(), 3);
        assert_eq!(mapped.get("a").unwrap(), store.get("a"));
        assert!(mapped.get("missing").unwrap().is_none());
        assert_eq!(mapped.record("a").unwrap().metadata["group"], "test");
        assert_eq!(mapped.record("a").unwrap().text.as_deref(), Some("first"));
        let query = [1.0, 0.0];
        assert_eq!(
            mapped.search(&query, 2).unwrap(),
            store.search(&query, 2).unwrap()
        );
        assert_eq!(
            mapped.search_with_metric(&query, 2, &Cosine).unwrap(),
            store.search_with_metric(&query, 2, &Cosine).unwrap()
        );
        assert!(mapped.search(&[1.0], 2).is_err());
        #[cfg(unix)]
        {
            store.delete("a");
            store.save_mapped(&path).unwrap();
            // SAFETY: Replacement changed the directory entry and left the
            // original mapping's file intact. The new file remains immutable.
            let replacement = unsafe { MappedVekryx::open(&path) }.unwrap();
            assert!(replacement.get("a").unwrap().is_none());
            assert_eq!(mapped.get("a").unwrap(), Some([1.0, 0.0].as_slice()));
            drop(replacement);
            store.insert("a".into(), vec![1.0, 0.0]).unwrap();
            store.save_mapped(&path).unwrap();
        }
        drop(mapped);

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_LEN..HEADER_LEN + VALUE_LEN].copy_from_slice(&f64::NAN.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        // SAFETY: The file stays immutable until this mapping is dropped.
        let mapped = unsafe { MappedVekryx::open(&path) }.unwrap();
        assert!(matches!(mapped.get("a"), Err(Error::InvalidSnapshot(_))));
        drop(mapped);
        std::fs::write(&path, &bytes[..HEADER_LEN + 1]).unwrap();
        // SAFETY: The file is immutable during the call.
        assert_eq!(
            unsafe { MappedVekryx::open(&path) }.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
