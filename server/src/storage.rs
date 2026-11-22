use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use vekryx::Vekryx;

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    #[default]
    Cosine,
    Euclidean,
}

impl Metric {
    pub fn engine(self) -> &'static dyn vekryx::distance::DistanceMetric {
        match self {
            Self::Cosine => &vekryx::distance::Cosine,
            Self::Euclidean => &vekryx::distance::Euclidean,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Collection {
    pub model: String,
    pub metric: Metric,
    pub store: Vekryx,
}

pub type Entry = Arc<RwLock<Collection>>;

pub struct Storage {
    pub collections: RwLock<BTreeMap<String, Entry>>,
    directory: PathBuf,
    _lock: File,
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

impl Storage {
    pub fn open(directory: &Path) -> io::Result<Self> {
        fs::create_dir_all(directory)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join(".lock"))?;
        FileExt::try_lock_exclusive(&lock)?;
        let mut collections = BTreeMap::new();
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|name| name.to_str())
                .filter(|name| valid_name(name))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid collection filename")
                })?;
            let collection =
                serde_json::from_reader(BufReader::new(File::open(&path)?)).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}: {error}", path.display()),
                    )
                })?;
            collections.insert(name.to_owned(), Arc::new(RwLock::new(collection)));
        }
        Ok(Self {
            collections: RwLock::new(collections),
            directory: directory.to_owned(),
            _lock: lock,
        })
    }

    pub fn save(&self, name: &str, collection: &Collection) -> io::Result<()> {
        let mut file = tempfile::NamedTempFile::new_in(&self.directory)?;
        serde_json::to_writer(file.as_file_mut(), collection)?;
        file.flush()?;
        file.as_file().sync_all()?;
        file.persist(self.path(name))?;
        Ok(())
    }

    pub fn delete(&self, name: &str) -> io::Result<()> {
        fs::remove_file(self.path(name))
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.json"))
    }
}
