use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temporary, mut file) = loop {
        let candidate = parent.join(format!(
            ".vekryx-{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let result = (|| {
        write(&mut file)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(feature = "experimental")]
pub fn save(path: &Path, contents: &[u8]) -> io::Result<()> {
    use std::io::Write;
    write_atomic(path, |file| file.write_all(contents))
}

pub(crate) fn invalid_data(
    error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
