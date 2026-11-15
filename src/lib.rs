//! Exact vector storage, configurable metrics, text chunking, and JSON snapshots.
pub mod chunking;
pub mod distance;
pub mod error;
pub use error::Error;
mod hybrid;
pub mod index;
pub mod search;
pub use hybrid::{FusionHit, HybridOptions};
pub use search::{MmrOptions, SearchHit, SearchOptions};
mod state_manager;
pub mod store;
pub use store::Vekryx;

/// Older, fixed-dimension search experiments, separate from the supported store.
#[cfg(feature = "experimental")]
pub mod experimental;

#[cfg(feature = "python")]
mod python;
