<p align="center">
  <img src="assets/vector_db.png" width="150" alt="Vekryx pixelated V logo">
</p>

<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/wordmark.svg">
    <img src="assets/wordmark.svg" width="260" alt="VEKRYX">
  </picture>
</h1>

<p align="center">
  <strong>Exact vector search. Composable Rust. Optional Python.</strong>
</p>

<p align="center">
  <a href="#installation"><img src="assets/compatibility.svg" width="264" alt="Rust 1.82+ · Python 3.8–3.12"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#repository-structure">File structure</a> ·
  <a href="#traits-and-object-relationships">Traits</a> ·
  <a href="#usage">Usage</a> ·
  <a href="contribution.md">Contribute</a>
</p>

## Overview

Vekryx is an early-stage, in-memory vector search library with native
Rust and optional Python APIs. Store numeric vectors by string ID, retrieve
nearest neighbors, diversify results with maximal marginal relevance (MMR), and
split text into flat or hierarchical chunks before embedding.

The default search methods perform an exact scan and sort matches. Build an
index to use **HNSW, IVF-Flat, PQ, or residual IVF-PQ** through the same vector,
MMR, and hybrid pipelines. These are initial in-memory implementations; the
repository does not publish representative performance benchmarks. Python can generate embeddings through optional
model adapters; Rust accepts vectors from your application.

The Rust crate and Python package are named `vekryx`; the public store type is
`Vekryx` in both languages. This is a breaking package/API rename: update imports
and rebuild the extension. Existing vector-only JSON snapshots remain readable; new snapshots also retain
original text and metadata when supplied.

The standalone [Actix HTTP server](server/README.md) ingests documents, embeds
their chunks through Ollama, and returns ranked chunks for plain-text queries.
The [React frontend](UI/README.md) provides a browser workspace for the same API.

## Features

- **Record storage:** optional UUIDs in Python, original text, JSON metadata, and finite `f64` vectors.
- **Filter before search:** exact metadata matching before ranking and candidate limits.
- **Hybrid retrieval:** BM25 keyword search fused with vector rankings using weighted RRF.
- **Exact retrieval:** Euclidean and cosine distances with deterministic ties.
- **Approximate retrieval:** seeded HNSW graphs, trained IVF lists, product quantization, and residual IVF-PQ.
- **Diverse results:** MMR selection over a configurable nearest-candidate pool.
- **Text preparation:** fixed-size, overlapping, and hierarchical parent–child chunks.
- **Explicit persistence:** plain and gzip-compressed JSON in Rust and Python, plus read-only memory-mapped vector search in Rust.
- **Composable retrieval:** index builders, typed backend query options, and shared vector/MMR/hybrid pipelines.
- **Explicit result semantics:** named distances with precision tags, separate hybrid fusion scores.
- **Trait-based extension:** independent metrics, candidate indexes, flat splitters, and chunking strategies.

The Rust core uses ownership and typed interfaces to separate storage from
search behavior. Python bindings call the same implementation through PyO3.

## Installation

### Rust

Requires **Rust 1.82+**, Cargo, and a native linker/build toolchain. Python is not
required for the default Rust build.

```bash
git clone https://github.com/SuleymanSuleymanzade/Vekryx.git
cd Vekryx
cargo build --lib
cargo run --example rust_usage
```

For another local Rust project, add the repository as a path dependency:

```toml
[dependencies]
vekryx = { path = "/path/to/Vekryx" }
```

### Python

Requires the Rust toolchain above and **Python 3.8–3.12** with `venv` and pip.
Python 3.11 is tested; the pinned PyO3 version does not support Python 3.13/3.14.
From the repository root:

```bash
bash install.sh
source .venv/bin/activate
python examples/python_usage.py
```

The [installer](install.sh) selects a compatible interpreter, creates or reuses
an isolated environment, builds the extension, and checks its import. No sudo,
global pip install, or direct maturin command is needed. Maturin remains the
internal build backend; downloading build dependencies requires network access
unless cached. NumPy is not required.

To select the interpreter or environment explicitly:

```bash
bash install.sh --python python3.11 --venv .venv
```

Run `bash install.sh --help` for options. The script targets Unix-style virtual
environments. Rerun it after changing Rust code. The optional `python` Cargo
feature and extension-module mode are enabled by Python packaging automatically.

## Quick Start

### Rust

```rust
use vekryx::{distance::Cosine, Vekryx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = Vekryx::new(3)?;
    store.insert("chunk-a".into(), vec![1.0, 0.0, 0.0])?;
    store.insert("chunk-b".into(), vec![0.0, 1.0, 0.0])?;

    let matches = store.search_with_metric(&[1.0, 0.0, 0.0], 2, &Cosine)?;
    assert_eq!(matches, vec![("chunk-a".into(), 0.0), ("chunk-b".into(), 1.0)]);

    std::fs::create_dir_all("data")?;
    store.save("data/vekryx.json")?;
    let restored = Vekryx::load("data/vekryx.json")?;
    assert_eq!(restored.len(), 2);
    Ok(())
}
```

### Python

To embed text automatically, install a provider extra in your virtual environment:

```bash
python -m pip install '.[sentence-transformers]'
# Other choices: '.[huggingface]' for hosted inference, '.[openai]' for OpenAI.
```

```python
from vekryx import Vekryx

store = Vekryx(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
cats_id = store.insert("Cats are small domestic animals.")
store.insert("Rust is a systems programming language.")
# Optional explicit ID: store.insert("Some text", id="my-chunk")
print(store.dimensions)  # Inferred from the model's output, never hardcoded.
print(store.search("Tell me about pets", k=1, metric="cosine"))
print(store.mmr_search("Tell me about pets", k=2))
```

An unqualified model string loads a local Sentence Transformers model from
Hugging Face (or a local model directory). This downloads model weights on first
use. To select a hosted provider, pass an adapter instead:

```python
from vekryx import Vekryx, OpenAIEmbeddings, HuggingFaceEmbeddings

# Uses OPENAI_API_KEY; also accepts api_key, base_url, or an existing client.
store = Vekryx(embedding_model=OpenAIEmbeddings("text-embedding-3-small"))

# Select a sentence embedding model deployed by your HF provider/endpoint.
store = Vekryx(embedding_model=HuggingFaceEmbeddings(
    "sentence-transformers/all-MiniLM-L6-v2", token="your-hf-token"
))
```

Hosted model availability depends on your provider. Hugging Face endpoints must
return a single pooled sentence vector; token matrices are rejected rather than
silently pooled. For local models, `SentenceTransformerEmbeddings(model, device="cpu",
revision="commit-or-tag")` allows device and revision selection. Document/query
encoding uses the model's respective methods when available.

The constructor embeds a probe string to infer dimensions. With hosted adapters
this makes an API request and may incur a charge. Subsequent insertions and
queries use the same adapter and validate vector shape and finite values.
Custom adapters implement `embed_document(text)` and `embed_query(text)`, each
returning one numeric vector. Provider errors propagate without replacing stored
vectors. The model can be inspected through `store.embedding_model`.

Snapshots save vectors, dimensions, original text, and metadata. Reattach the **same model and
configuration** with `Vekryx.load(path, embedding_model=model)` to search by text.
Dimensions are checked on load; different models with the same dimensions cannot
be detected. Credentials and model weights are never stored in snapshots. Original
text is retained for text records; `get(id)` returns the stored vector, while
`get_record(id)` returns ID, text, metadata, and vector.

For applications that already generate embeddings, the numeric API remains:

```python
from pathlib import Path
from vekryx import Vekryx

store = Vekryx(dimensions=3)
store.insert("chunk-a", [1.0, 0.0, 0.0])
store.insert("chunk-b", [0.0, 1.0, 0.0])

matches = store.search([1.0, 0.0, 0.0], k=2, metric="cosine")
assert matches == [("chunk-a", 0.0), ("chunk-b", 1.0)]

path = Path("data/vekryx.json")
path.parent.mkdir(parents=True, exist_ok=True)
store.save(path)
restored = Vekryx.load(path)
assert len(restored) == 2
print("Snapshot:", path.resolve())
```

Python `insert` returns the supplied or generated UUID string. Inserting an
existing ID replaces its vector, text, and metadata together; omitted metadata
becomes an empty object. Legacy `insert("id", vector_or_text)` calls still work. Rust `get` returns an `Option`;
Python `get` returns a copy or raises `KeyError`. `delete` returns whether the
record existed in both APIs. IDs must be nonempty, and every vector/query must
match the configured positive dimension and contain only finite numbers.

## Architecture

```mermaid
flowchart TD
    D[Documents] --> C[Optional chunking and embedding adapters]
    C --> S[(Vekryx: stable IDs, vectors, text, metadata)]
    S <--> P[Atomic snapshots: JSON, gzip, or mapped vectors]
    S --> F[Corpus: optional metadata filter]
    F --> B[IndexBuilder: construction and training settings]
    B --> I[SearchSession: fixed corpus, metric, backend]
    Q[Query vector and typed SearchOptions] --> I
    I --> R[CandidateIndex: retrieve eligible IDs and distances]
    EX[ExactSearch / HNSW / IVF-Flat / PQ / IVF-PQ] --> R
    PQ[PQ: compressed subvector codes] --> IP[IVF-PQ: centroid residual encoding]
    IP --> R
    R --> V[Validate IDs, distances and limits; deduplicate]
    V --> E[Optional exact distance reranking]
    E --> K[Top k SearchHit results]
    E --> M[MMR selection: SearchHit in selection order]
    E --> H[Weighted reciprocal rank fusion]
    F --> BM[BM25: eligible text corpus]
    T[Query text] --> BM
    BM --> L[Optional keyword candidate limit]
    L --> H
    H --> FH[Top k FusionHit results]
```

[Standalone SVG schematic](assets/architecture_vec_db.svg).
All backends shown are implemented. HNSW and IVF-Flat return exact distances
for approximate candidate sets; PQ variants can report approximate distances
or use the shared exact reranker.

- [Vekryx](src/store.rs) owns canonical records and vectors, validates mutations, and saves snapshots.
- [Corpus](src/search.rs) borrows an eligible corpus and exposes filter-aware iteration/lookup to backends.
- [IndexBuilder and CandidateIndex](src/index.rs) separate construction/training from query-time candidate retrieval. `ExactSearch` implements both traits; HNSW, IVF, PQ, and IVF-PQ use separate
  configuration/builders and trained index types in [src/index/](src/index).
- `SearchSession` binds the built backend to one immutable corpus and one `DistanceMetric`.
- [Search pipeline](src/search.rs) validates candidates, deduplicates IDs, optionally recomputes approximate distances, and applies top-k or MMR selection.
- [Hybrid ranking](src/hybrid.rs) combines a bounded vector ranking with BM25 using weighted RRF.
- [DistanceMetric](src/distance.rs) supplies distances, MMR similarities, and optional built-in geometry identification for specialized builders.
- [TextSplitter and ChunkingStrategy](src/chunking/mod.rs) separate flat splitting from hierarchy construction.
- [Snapshot writer](src/state_manager/snapshot.rs) atomically replaces snapshot files on the same filesystem.
- [Mapped snapshots](src/state_manager/mapped.rs) keep vector values file-backed for read-only exact search.
- [Python bindings](src/python.rs) adapt the shared pipelines; [Python options](python/vekryx/search.py) expose immutable options and named results.

Filtering precedes candidate limits and BM25 statistics. Backends must respect the
bound corpus; the shared pipeline rejects unknown or filtered-out IDs rather than
returning them. A backend can return fewer candidates than requested. The pipeline
does not silently fill missing results with an exact scan.

A session borrows its corpus and metric. Rust prevents mutation of the store or
filter while that session remains in use. End the session, mutate canonical data,
and build a new session to rebuild/retrain the backend. This first interface uses
immutable sessions rather than claiming universal incremental-update support.
Indexes, training state, and session options are not serialized into store snapshots.
Python `build_index()` instead owns an independent snapshot of eligible records
and its trained backend; source store mutations leave that snapshot unchanged.
Build a new Python index to include edits. This ownership avoids self-referential
borrows across the Python boundary, at the cost of copying the eligible corpus.
Filtered sessions are built for their particular eligible corpus; reusing one global
index across changing filters is not implemented.

The experimental KD-tree/LSH prototypes remain separate from this pipeline and do
not accelerate `Vekryx`. There is no persistent CLI store or automatic parent
expansion. Python's optional [embedding adapters](python/vekryx/embeddings.py)
prepare vectors before calling Rust.


### Repository Structure

```text
Vekryx/
├── README.md / .gitignore        # Usage guide and generated-artifact exclusions
├── Cargo.toml / Cargo.lock       # Rust features, dependencies, resolved versions
├── pyproject.toml               # Python packaging and optional provider extras
├── install.sh                   # Isolated Python installation
├── src/
│   ├── lib.rs                   # Public Rust API and feature gates
│   ├── main.rs                  # Small supported-store demo: cargo run
│   ├── store.rs                 # Vekryx, Record, CRUD, validated snapshot decoding
│   ├── index.rs                 # IndexBuilder, CandidateIndex, ExactSearch, SearchSession
│   ├── index/
│   │   ├── hnsw.rs              # Multilayer graph, heaps, diversity pruning
│   │   ├── ivf.rs               # Coarse quantizer and inverted lists
│   │   ├── pq.rs                # Subvector codebooks, ADC, residual IVF-PQ
│   │   ├── training.rs          # Deterministic k-means and numeric helpers
│   │   └── tests.rs             # Recall, filtering, validation, shared pipelines
│   ├── search.rs                # Corpus, typed options/results, shared ranking and MMR
│   ├── hybrid.rs                # BM25 and weighted reciprocal rank fusion
│   ├── distance.rs              # DistanceMetric, Euclidean, Cosine
│   ├── error.rs                 # Shared Error enum
│   ├── python.rs                # PyO3 bridge to the Rust store
│   ├── python/index.rs          # Owned Python index snapshots and dispatch
│   ├── chunking/
│   │   ├── mod.rs               # Chunk, TextSplitter, ChunkingStrategy
│   │   ├── fixed_size.rs        # FixedSizeChunker: splitter and chunk producer
│   │   ├── hierarchical.rs      # HierarchicalChunker: composes flat splitters
│   │   ├── python.rs            # Python chunk and chunker wrappers
│   │   └── tests.rs             # Unicode, overlap, hierarchy contracts
│   ├── state_manager.rs         # Private persistence module declaration
│   ├── state_manager/
│   │   ├── mapped.rs           # Binary snapshots and file-backed exact search
│   │   └── snapshot.rs          # Temporary-file write and atomic replacement
│   └── experimental/            # Optional: --features experimental
│       ├── mod.rs
│       ├── fast_search.rs       # Retrieval-only FastSearch trait
│       ├── kdtree.rs            # Three-dimensional KDTree and typed config
│       ├── vector_db.rs         # Fixed-dimension LinearIndex
│       ├── locality_sensitive_hashing.rs
│       └── persistence.rs       # Serde-based LocalSaver
├── python/vekryx/
│   ├── __init__.py              # Public Python facade, IDs, records, filters
│   ├── embeddings.py            # EmbeddingModel protocol and provider adapters
│   ├── search.py                # Immutable options and named search/fusion results
│   └── index.py                 # Typed backend settings and built index facade
├── examples/
│   ├── rust_usage.rs / python_usage.py
│   ├── ann_search.rs / ann_search.py
│   ├── search_pipeline.rs / search_pipeline.py
│   ├── hybrid_search.rs / hybrid_search.py
│   ├── basic_usage.py / nearest_neighbors.py / cosine_and_mmr.py
│   ├── hierarchical_chunking.py / persistence.py / embedding_models.py
│   ├── store_smoke.py
│   └── experimental/lsh_prototype.py
├── tests/
│   ├── search_interface.rs      # Downstream backend contract tests
│   ├── python_test/test_*.py     # Tests against the installed Python package
│   └── fixtures/legacy_kdtree.json  # Historical experimental snapshot
├── assets/                      # README artwork
├── AGENTS.md                    # Repository conventions
└── contribution.md              # Contributor workflow
```

Rust unit tests also live beside their implementations. Build output (`target/`),
virtual environments, Python caches, wheels, and root-level `data/` snapshots are
Git-ignored. `Cargo.lock` is tracked. The historical KD-tree fixture is separate
from the supported `Vekryx` snapshot format.

### Traits and Object Relationships

Rust structs play the role of objects with state; `impl` blocks provide methods,
and traits describe replaceable behavior. This design uses composition and
interface-style polymorphism. There is no base-store inheritance hierarchy.

```mermaid
classDiagram
    class Vekryx {
        -dimensions
        -vectors
        -records
        +view() Corpus
        +filtered_view(filter) Corpus
        +search_with_options(query, metric, options)
        +save(path)
    }
    class Corpus {
        +vectors()
        +get(id)
        +contains(id)
        +build_index(builder, metric) SearchSession
    }
    class IndexBuilder {
        <<trait>>
        +build(corpus, metric) Index
    }
    class CandidateIndex {
        <<trait>>
        +SearchOptions associated_type
        +guarantee() RetrievalGuarantee
        +candidates(corpus, query, metric, limit, options)
    }
    class ExactSearch
    class HnswConfig
    class HnswIndex
    class IvfConfig
    class IvfIndex
    class PqConfig
    class PqIndex
    class IvfPqConfig
    class IvfPqIndex
    class SearchSession {
        -corpus
        -metric
        -index
        +search(query, options) SearchHit[]
        +mmr_search(query, options, mmr) SearchHit[]
        +hybrid_search(text, query, options, hybrid) FusionHit[]
        +guarantee() RetrievalGuarantee
    }
    class DistanceMetric {
        <<trait>>
        +kind() optional_builtin_geometry
        +distance(left, right)
        +similarity(left, right)
    }
    class SearchOptions {
        +k
        +candidates
        +rerank_exact
        +backend typed_options
    }
    class SearchHit {
        +id
        +distance
        +distance_kind
    }
    class FusionHit {
        +id
        +fusion_score
    }
    class Euclidean
    class Cosine
    class TextSplitter {
        <<trait>>
        +split(text) character_ranges
    }
    class ChunkingStrategy {
        <<trait>>
        +chunk(text, document_id) chunks
    }
    class FixedSizeChunker
    class HierarchicalChunker
    Corpus --> Vekryx : borrows
    Corpus ..> IndexBuilder : builds
    IndexBuilder ..> CandidateIndex : produces
    IndexBuilder <|.. ExactSearch
    CandidateIndex <|.. ExactSearch
    IndexBuilder <|.. HnswConfig
    IndexBuilder <|.. IvfConfig
    IndexBuilder <|.. PqConfig
    IndexBuilder <|.. IvfPqConfig
    CandidateIndex <|.. HnswIndex
    CandidateIndex <|.. IvfIndex
    CandidateIndex <|.. PqIndex
    CandidateIndex <|.. IvfPqIndex
    SearchSession --> Corpus : borrows through view
    SearchSession *-- CandidateIndex : owns backend
    SearchSession --> DistanceMetric : borrows fixed metric
    SearchSession ..> SearchOptions : accepts
    SearchSession ..> SearchHit : vector and MMR
    SearchSession ..> FusionHit : hybrid
    DistanceMetric <|.. Euclidean
    DistanceMetric <|.. Cosine
    TextSplitter <|.. FixedSizeChunker
    ChunkingStrategy <|.. FixedSizeChunker
    ChunkingStrategy <|.. HierarchicalChunker
    HierarchicalChunker *-- TextSplitter : composes
```

`Corpus` is a view, not a subclass of `Vekryx`. It borrows the store and an
optional metadata filter. Rust lifetimes prevent either from being dropped or
mutated while a view is still used. `filtered_view()` copies no stored vectors,
text, or metadata; `filtered()` deliberately builds an independent owned store.
Python's `where=` searches use the borrowed view internally.

| Interface | Implementations | Role and contract |
| --- | --- | --- |
| `IndexBuilder` | `ExactSearch`, `HnswConfig`, `IvfConfig`, `PqConfig`, `IvfPqConfig`, downstream builders | Own build/training settings; validate supported geometry/dimensions; produce a session-local candidate index. |
| `CandidateIndex` | `ExactSearch`, `HnswIndex`, `IvfIndex`, `PqIndex`, `IvfPqIndex`, downstream backends | Associated typed query options; retrieve eligible stable IDs and finite, nonnegative distances. Declare exact or approximate candidate selection. |
| `DistanceMetric: Send + Sync` | `Euclidean`, `Cosine`, user-defined Rust metrics | Strategy for distance and MMR similarity. Distances must be finite and nonnegative; similarities must be finite and larger for more related pairs. Default similarity is `1 / (1 + distance)`. |
| `TextSplitter: Send + Sync` | `FixedSizeChunker`, user-defined Rust splitters | Return nonempty, in-bounds character ranges in document order. Offsets count Unicode scalar values. |
| `ChunkingStrategy: Send + Sync` | `FixedSizeChunker`, `HierarchicalChunker` | Produce document-scoped `Chunk` records. Hierarchy composes `Box<dyn TextSplitter>`. |
| Python `EmbeddingModel` protocol | `SentenceTransformerEmbeddings`, `HuggingFaceEmbeddings`, `OpenAIEmbeddings`, custom adapters | Implement `embed_document(text)` and `embed_query(text)`; return one finite, nonempty numeric vector. No inheritance is required. |
| Experimental `FastSearch` | `KDTree`, `LinearIndex` | Associated `Vector` type and `nearest_neighbor`; persistence is independent. Experimental LSH has its own candidate API and does not implement this trait. |

Compatibility methods accept `&dyn DistanceMetric` per call. A built session
fixes its metric at construction; queries cannot substitute another metric.
`DistanceMetric::kind()` identifies Euclidean/cosine geometry; custom metrics default
to `None` and still work with `ExactSearch`. Specialized builders must reject
unsupported metrics explicitly. Python resolves the names `"euclidean"`
and `"cosine"` to those implementations. Arbitrary Python metrics and custom
Python splitters are not exposed by the current bindings; custom embedding
adapters are supported through structural typing.

### Architectural Decisions

| Decision | OOP-style interpretation | Reason and tradeoff |
| --- | --- | --- |
| Keep vectors and records private in `Vekryx` | Encapsulation | Mutations validate IDs, dimensions, and finite values before changing state. Public `Deserialize` also passes through a private, validated snapshot representation. |
| Separate `Corpus` from storage | Read-only view object | Reuse vector, MMR, and hybrid retrieval on the same eligible corpus without cloning stored data. Filter matching and exact scoring still scan entries. |
| Bind backend, corpus, and metric in a borrowed session | Composition and lifecycle safety | Construction is separate from querying. End the session before mutations, then rebuild; no stale index state is hidden in the store. |
| Use associated backend query options | Typed extension point | New backends introduce their own settings without adding irrelevant HNSW/IVF fields to every request. |
| Separate candidate recall from distance precision | Explicit result contract | Exact reranking improves distances within retrieved candidates; it does not recover missing neighbors. |
| Inject metrics through a trait | Strategy pattern and dependency inversion | Retrieval depends on the metric interface; adding a Rust metric does not change storage. Trait objects allow runtime selection. |
| Compose hierarchy from flat splitters | Composition over inheritance | Each level can use a different `TextSplitter`; the hierarchy owns parent/child construction and offset translation. |
| Keep Python orchestration outside Rust search | Facade and adapter patterns | Python handles embedding providers and convenient inputs; PyO3 converts arguments and errors; Rust owns validation and ranking. |
| Use explicit snapshots | Separate persistence responsibility | In-memory storage does not assume a filesystem path. Saving writes a temporary file and renames it; parent directories must exist and writers are not coordinated. |
| Use a shared typed `Error` | Explicit failure contract | Callers can distinguish invalid arguments, invalid snapshot references, and JSON errors. Filesystem APIs return `io::Result`; Python maps invalid input to `ValueError`, missing IDs to `KeyError`, and I/O errors to `OSError`. |
| Gate legacy indexes with `experimental` | Separate experimental subsystem | Default builds do not need `ndarray` or `rand`. These indexes do not accelerate the supported exact store. |
| Keep model objects outside snapshots | Application-owned dependency | Rust and Python exchange the same vectors and records. Python must reattach the same embedding model/configuration for text queries after loading. |

Exact search scores all eligible vectors and sorts results: for `n` eligible
vectors of dimension `d`, ordinary retrieval costs `O(n*d + n*log(n))` and keeps
`O(n)` scored candidates before truncation. The shared pipeline also keeps the
returned candidate pool for validation and ranking. A borrowed view avoids corpus copies;
it does not turn the scan into an index or eliminate ranking allocations.

### Rust API Compatibility

Retrieval, metric, and chunking methods return `vekryx::Error`; custom trait
implementations use that error type. `Vekryx::get` returns `Option<&[f64]>`.
`save`/`load` return `std::io::Result`, and `to_json` returns a Serde JSON result.
Existing tuple-returning search/MMR/hybrid methods retain their defaults and
result shapes. New `*_with_options` methods return named records.
`Corpus::build_index()` is the Rust backend extension entry point.
The supported snapshot fields are shared with Python. Experimental index
representations are separate and should not be treated as supported snapshots.


## Usage

The examples below pair the same workflows in Rust and Python. Each code block
is self-contained unless stated otherwise. Record examples in a separate Rust
application also need `serde_json = "1"` in `[dependencies]`.


### Composable Search Interface

Use an index session to select a Rust backend once, then reuse the same vector,
MMR, and hybrid methods. Built-ins are `ExactSearch`, `HnswConfig`, `IvfConfig`,
`PqConfig`, and `IvfPqConfig`.

```rust
use vekryx::{
    distance::Cosine,
    index::{ExactSearch, RetrievalGuarantee},
    HybridOptions, MmrOptions, SearchOptions, Vekryx,
};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    store.insert("a".into(), vec![1.0, 0.0])?;
    store.insert("b".into(), vec![0.0, 1.0])?;
    let session = store.view().build_index(&ExactSearch, &Cosine)?;
    let options = SearchOptions { candidates: 20, ..SearchOptions::new(5) };
    let hits = session.search(&[1.0, 0.0], &options)?;
    assert_eq!(hits[0].id, "a");
    assert_eq!(session.guarantee(), RetrievalGuarantee::Exact);
    let diverse = session.mmr_search(&[1.0, 0.0], &options, &MmrOptions::default())?;
    let fused = session.hybrid_search("example", &[1.0, 0.0], &options, &HybridOptions::default())?;
    assert_eq!(diverse.len(), 2);
    assert!(fused[0].fusion_score > 0.0);
    Ok(())
}
```

For exact search without a reusable session, `Vekryx` and `Corpus` both expose
`search_with_options(query, metric, options)`,
`mmr_search_with_options(query, metric, options, mmr)`, and
`hybrid_search_with_options(text, query, metric, options, hybrid)`.

Python exposes exact options-driven pipelines on `Vekryx`.
`Vekryx.build_index(config, metric=..., where=...)` creates an immutable
`SearchIndex` with `search`, `mmr_search`, and `hybrid_search` methods; see below.
Custom Python backend registration is not exposed.

```python
from vekryx import MmrOptions, SearchOptions, Vekryx

store = Vekryx(2)
store.insert("a", [1, 0])
store.insert("b", [0, 1])
options = SearchOptions(k=1, candidates=10)
hits = store.search_with_options([1, 0], options, metric="cosine")
assert (hits[0].id, hits[0].distance, hits[0].distance_kind) == ("a", 0, "exact")
diverse = store.mmr_search_with_options([1, 0], options, mmr=MmrOptions(lambda_mult=0.2))
assert diverse[0].id == "a"
```

For text queries with an embedding model, call
`store.hybrid_search_with_options(text, options, hybrid=HybridOptions(...), where=...)`.
The [Rust](examples/search_pipeline.rs) and [Python](examples/search_pipeline.py)
examples demonstrate filtering and all three result paths offline.

| Setting / result | Contract |
| --- | --- |
| `SearchOptions.k` | Maximum final result count. Rust `SearchOptions::new(k)` defaults `candidates` to `k`; Python defaults `k=5` and `candidates=None` resolves to `k`. |
| `SearchOptions.candidates` | Backend candidate limit before reranking, MMR, or fusion; must be at least `k`. This replaces `fetch_k` in the new MMR interface. |
| `SearchOptions.rerank_exact` | Defaults to true. Recompute approximate distances from canonical vectors, then sort; already exact distances are retained. Does not upgrade candidate recall. |
| Rust `SearchOptions.backend` | Backend-specific associated query-options type: `HnswSearchOptions`, `IvfSearchOptions`, or `()` for exact/PQ. Python uses those option dataclasses or `None`. |
| `MmrOptions.lambda_mult` | Finite value in `[0,1]`, default `0.5`; selection operates only on the retrieved pool. |
| `HybridOptions` | `vector_weight=0.5`, `rrf_k=60`, `keyword_candidates=None` (complete eligible BM25 ranking). Vector ranking is limited by `SearchOptions.candidates`. |
| `SearchHit` | `id`, `distance`, `distance_kind`. Smaller distance is better; MMR outputs selection order. Rust uses `DistanceKind`, Python uses `"exact"` / `"approximate"`. |
| `FusionHit` | `id`, `fusion_score`; larger scores first. |
| `SearchSession.guarantee()` | Exact or approximate candidate selection, independent of each hit's distance precision. An exact session with a truncated fusion pool does not promise full-ranking hybrid equivalence. |

Distances must use the bound metric's units: an L2 backend must convert squared
L2 scores before returning candidates. The pipeline validates finite, nonnegative
distances, checks eligible IDs and the raw candidate limit, deduplicates by ID,
and breaks distance ties by ID. Invalid backend output returns an error. Duplicate
IDs never consume multiple final slots; identical vectors with distinct IDs remain
separate records. Approximate backends may underfill and miss true neighbors even
when all returned distances are exact.

The existing `hybrid_search` preserves complete-ranking fusion. Its new options
variant deliberately bounds the vector list; limiting either ranking can change
fusion scores and final results even with an exact candidate backend. Metadata
filters still precede limits and BM25 statistics. `keyword_candidates` limits the
ranked keyword list, not the cost of scanning/scoring text.

#### Adding a Backend or Encoding

1. Define a builder containing construction/training settings and implement
   `IndexBuilder`. Use the supplied filtered `Corpus` to train/build, and reject
   unsupported `DistanceMetric::kind()` values or dimensions.
2. Return a `CandidateIndex` with an associated `SearchOptions` type for query
   controls. Return eligible stable IDs, distances, and `DistanceKind`; declare
   `RetrievalGuarantee::Exact` only if true top-candidate retrieval is guaranteed.
3. Build through `corpus.build_index(&builder, &metric)`. Shared validation,
   reranking, MMR, and fusion work without editing the store or duplicating those
   pipelines. The session always supplies the corpus and metric used at build time.
4. Keep compression internal to the backend: IVF with full vectors and IVF-PQ can
   implement the same contract. PQ is an encoding technique, not a mutually
   exclusive alternative to a retrieval structure. No public encoder trait is
   imposed before an encoding implementation establishes its requirements.
5. Compare against `ExactSearch` for recall and ranking, and measure build time,
   latency, memory, filtering, and rebuild cost. See
   [downstream contract tests](tests/search_interface.rs) for a custom backend
   with typed options, approximate distances, and deliberately missing candidates.

Incremental index mutation and index persistence remain future work. All four
approximate backends below use immutable build/train sessions.

### HNSW, IVF, and Product Quantization

| Builder | Candidate retrieval | Supported metrics | Query options |
| --- | --- | --- | --- |
| `HnswConfig` | Seeded multilayer proximity graph; greedy upper-layer descent and bounded best-first bottom-layer search, with diversity-based link pruning. | Euclidean, cosine | `HnswSearchOptions { ef_search }` |
| `IvfConfig` | Train centroids, assign inverted lists, probe nearest lists, and score their full vectors. | Euclidean, cosine (spherical training on normalized vectors) | `IvfSearchOptions { nprobe }` |
| `PqConfig` | Train independent subvector codebooks and scan all compressed codes using asymmetric distance lookup tables. | Euclidean | Rust `()` / Python `None` |
| `IvfPqConfig` | IVF coarse centroids plus PQ-trained residuals; score codes only in probed lists. | Euclidean | `IvfSearchOptions { nprobe }` |

These implementations follow the [HNSW paper](https://arxiv.org/abs/1603.09320)
and its [parameter definitions](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md),
with IVF/PQ organization described in the
[Faiss index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).
They are native Rust implementations, not wrappers around those libraries.
Construction uses a local seeded random generator and deterministic tie-breaking;
no new runtime dependencies or `experimental` feature are required.

#### Rust: build once, query repeatedly

```rust
use vekryx::{
    distance::Euclidean,
    index::{HnswConfig, HnswSearchOptions, IvfConfig, IvfPqConfig, IvfSearchOptions, PqConfig},
    SearchOptions, Vekryx,
};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(4)?;
    for i in 0..32 {
        store.insert(format!("{i:02}"), vec![i as f64, 1.0, (i % 5) as f64, 2.0])?;
    }
    let query = [6.0, 1.0, 1.0, 2.0];
    let graph = store.view().build_index(&HnswConfig::default(), &Euclidean)?;
    let hits = graph.search(&query, &SearchOptions {
        candidates: 16,
        backend: HnswSearchOptions { ef_search: 64 },
        ..SearchOptions::new(3)
    })?;
    assert_eq!(hits[0].id, "06");

    let ivf = IvfConfig { nlist: 4, ..IvfConfig::default() };
    let pq = PqConfig { subquantizers: 2, bits: 2, ..PqConfig::default() };
    let lists = store.view().build_index(&ivf, &Euclidean)?;
    let quantized = store.view().build_index(&pq, &Euclidean)?;
    let combined = store.view().build_index(&IvfPqConfig { ivf, pq }, &Euclidean)?;
    let list_options = SearchOptions {
        candidates: 32,
        backend: IvfSearchOptions { nprobe: 4 },
        ..SearchOptions::new(3)
    };
    assert_eq!(lists.search(&query, &list_options)?[0].id, "06");
    assert_eq!(combined.search(&query, &list_options)?[0].id, "06");
    let pq_options = SearchOptions { candidates: 32, ..SearchOptions::new(3) };
    assert_eq!(quantized.search(&query, &pq_options)?[0].id, "06");
    Ok(())
}
```

Pass the same options to `session.mmr_search(query, options, mmr)` or
`session.hybrid_search(text, query, options, hybrid)`. Use a `filtered_view()`
before building to restrict construction and retrieval to eligible records.
The builder is not borrowed after construction, so temporary configs are valid.

#### Python: immutable index snapshots

```python
from vekryx import (
    HnswConfig, HnswSearchOptions, IvfConfig, IvfPqConfig, IvfSearchOptions,
    PqConfig, SearchOptions, Vekryx,
)

store = Vekryx(4)
for i in range(32):
    store.insert(f"{i:02}", [i, 1, i % 5, 2])

index = store.build_index(HnswConfig(), metric="euclidean")
options = SearchOptions(k=3, candidates=16, backend=HnswSearchOptions(ef_search=64))
assert index.search([6, 1, 1, 2], options)[0].id == "06"

ivf = IvfConfig(nlist=4)
pq = PqConfig(subquantizers=2, bits=2)
for config in (ivf, pq, IvfPqConfig(ivf=ivf, pq=pq)):
    index = store.build_index(config)
    backend = None if isinstance(config, PqConfig) else IvfSearchOptions(nprobe=4)
    hits = index.search([6, 1, 1, 2], SearchOptions(k=3, candidates=32, backend=backend))
    assert hits[0].id == "06"
```

For text, the index retains the store's embedding adapter and exposes
`index.search(text, options)`, `index.mmr_search(text, options, mmr=...)`, and
`index.hybrid_search(text, options, hybrid=...)`. Filters are fixed at
`build_index(config, where=...)`; queries cannot replace the filter or metric.
The built index owns copies of eligible vectors and records. Source insertions,
replacements, and deletions do not affect it: explicitly rebuild to include them.
The existing `store.search` and `*_with_options` methods continue to use exact
search. Backend query options on those store methods raise `ValueError`.

#### Training, tuning, and result guarantees

- **HNSW:** defaults `m=16`, `ef_construction=100`, `seed=42`, `ef_search=64`.
  Require `m >= 2`, `ef_construction >= m`, and positive `ef_search`. The effective
  search breadth is at least the requested candidate pool (bounded by corpus size).
  Layer zero permits up to `2*m` neighbors; upper layers permit `m`. Larger graph
  and exploration settings generally trade construction/memory/query cost for recall.
- **IVF:** defaults `nlist=16`, `iterations=20`, `seed=42`, `nprobe=1`. Nonempty
  training requires at least `nlist` eligible vectors. Require `1 <= nprobe <= nlist`.
  Training uses seeded farthest-first initialization and Lloyd updates; cosine
  uses spherical centroids. Probing all lists in IVF-Flat reproduces exact vector
  ranking. Probing fewer lists may underfill or miss neighbors.
- **PQ:** defaults `subquantizers=4`, `bits=4`, `iterations=20`, `seed=42`.
  Dimensions must be divisible by the positive subquantizer count; `bits` is
  between 1 and 8. Nonempty training requires at least `2**bits` eligible vectors
  for each codebook. Codes use **one byte per subquantizer**, including when
  `bits < 8`; sub-byte codes are not bit-packed. Codebooks and training use `f64`.
  Standalone PQ still scans every code, so it is exhaustive over compressed
  representations but approximate in distance/ranking.
- **IVF-PQ:** configure nested `ivf` and `pq` settings. Train one global set of
  residual subvector codebooks after coarse clustering. Both training size
  constraints apply. Per probed list, subtract its centroid from the query and
  use residual-code lookup tables; this differs from applying PQ to raw vectors.

Valid configurations can build empty indexes, which return no results. Invalid
settings are rejected even on empty corpora. Zero cosine vectors and unsupported
metrics are rejected during construction. Non-finite/unrepresentable computed
distances return errors rather than entering ranking.

All four ANN indexes conservatively report `RetrievalGuarantee::Approximate`
(Python `index.guarantee == "approximate"`). HNSW and IVF-Flat report exact
**distances** for retrieved records. PQ variants initially report estimated L2
(not squared L2); `rerank_exact=True` recomputes canonical distances for the
retrieved pool. Increasing the pool can improve reranked recall. Reranking cannot
recover neighbors missing from that pool. To compare with an exact oracle, use
the same metric, filter, and stable IDs.

PQ compresses the **index code payload**, not canonical storage: full `f64`
vectors remain in `Vekryx` for lookup and reranking. Python additionally copies
its eligible corpus into the built snapshot. Do not interpret code bytes per
vector as total process memory. There is no index serialization, incremental
mutation, concurrent build machinery, SIMD/GPU acceleration, or production-scale
performance claim in this initial implementation.

Run the [Rust comparison example](examples/ann_search.rs) for recall@10 against
exact search and illustrative build/query timings, or the
[Python example](examples/ann_search.py) for all four filtered pipelines:

```bash
cargo run --release --example ann_search
python examples/ann_search.py
```

The comparison uses 256 synthetic eight-dimensional vectors and 16 deterministic
queries. Those timings are a demonstration, not a representative ANN benchmark.
Rust tests also cover independent PQ reconstruction, graph topology, bounded
list/graph exploration, filtering, duplicates, invalid settings, and extreme values.

### Insert, Update, Retrieve, and Delete

These examples run without an embedding model. Reusing an ID replaces its vector
and clears prior text/metadata unless a replacement record supplies them.

#### Rust

```rust
use vekryx::Vekryx;

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    store.insert("note".into(), vec![1.0, 0.0])?;
    store.insert("note".into(), vec![0.0, 1.0])?;
    assert_eq!(store.len(), 1);
    assert_eq!(store.get("note"), Some([0.0, 1.0].as_slice()));
    assert_eq!(store.search(&[0.0, 1.0], 1)?, vec![("note".into(), 0.0)]);
    assert!(store.insert("bad".into(), vec![1.0]).is_err());
    assert!(store.get("missing").is_none());
    assert!(store.delete("note"));
    assert!(!store.delete("note"));
    Ok(())
}
```

#### Python

```python
from vekryx import Vekryx

store = Vekryx(dimensions=2)
store.insert("note", [1.0, 0.0])
store.insert("note", [0.0, 1.0])
assert len(store) == 1
assert store.get("note") == [0.0, 1.0]
assert store.search([0.0, 1.0], k=1) == [("note", 0.0)]
try:
    store.insert("bad", [1.0])
except ValueError:
    pass
try:
    store.get("missing")
except KeyError:
    pass
assert store.delete("note")
assert not store.delete("note")
```


### Dictionary Records and Filters

#### Rust

```rust
use serde_json::json;
use vekryx::{distance::Cosine, store::Record, Vekryx};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    for (id, text, vector, domain) in [
        ("cats", "Cats are domestic animals.", vec![1.0, 0.0], "animals"),
        ("rust", "Rust prevents memory errors.", vec![0.0, 1.0], "programming"),
    ] {
        store.insert_record(id.into(), vector, Record {
            text: Some(text.into()),
            metadata: json!({"domain": domain, "language": "en"})
                .as_object().unwrap().clone(),
        })?;
    }
    let filter = json!({"domain": "animals", "language": "en"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    let matches = corpus.search_with_metric(&[1.0, 0.0], 5, &Cosine)?;
    assert_eq!(matches, vec![("cats".into(), 0.0)]);
    assert_eq!(store.record("cats").unwrap().text.as_deref(), Some("Cats are domestic animals."));
    Ok(())
}
```

#### Python

This adapter produces deterministic toy vectors for an offline demonstration.
Replace it with a real embedding model for semantic retrieval.

```python
from vekryx import Vekryx

class ToyEmbeddings:
    def embed_document(self, text):
        return [1.0, 0.0] if "cat" in text.lower() or "pet" in text.lower() else [0.0, 1.0]

    def embed_query(self, text):
        return self.embed_document(text)


store = Vekryx(embedding_model=ToyEmbeddings())
record_id = store.insert({
    "text": "Cats are domestic animals.",
    "metadata": {"domain": "animals", "language": "en", "published": True},
})
store.insert({
    "text": "Rust prevents many memory errors.",
    "metadata": {"domain": "programming", "language": "en"},
})

# All properties must match BEFORE ranking; excluded records cannot consume k.
matches = store.search("pets", k=5, metric="cosine",
                       where={"domain": "animals", "language": "en"})
diverse = store.mmr_search("pets", k=2, fetch_k=10,
                           where={"domain": "animals"})
record = store.get_record(record_id)  # {id, text, metadata, vector}; owned copy
```

A dictionary record accepts `text`, `metadata`, and optional `id`; `text` must be a
string. For plain text or vectors, use `metadata={...}` on `insert` instead.
`where` matches top-level metadata properties with exact JSON equality and AND
semantics. Nested objects/arrays match as complete values; there are no comparison
operators, dotted paths, or implicit coercions. A missing property does not match
`None`; an explicitly stored JSON null does. An empty filter matches every record,
and no matching records produces `[]`. Metadata keys must be strings and values
must be finite JSON values. Filters apply to vector, MMR, and hybrid search alike.

### Hybrid Keyword and Vector Search

#### Rust

Rust supplies query text for BM25 and an application-generated vector for vector
ranking. The following vectors are toy fixtures, not model embeddings.

```rust
use serde_json::json;
use vekryx::{distance::Cosine, store::Record, Vekryx};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    for (id, text, vector) in [
        ("cats", "Cats are domestic animals.", vec![1.0, 0.0]),
        ("dogs", "Dogs are domestic animals.", vec![0.8, 0.2]),
    ] {
        store.insert_record(id.into(), vector, Record {
            text: Some(text.into()),
            metadata: json!({"domain": "animals"}).as_object().unwrap().clone(),
        })?;
    }
    let filter = json!({"domain": "animals"});
    let corpus = store.filtered_view(filter.as_object().unwrap());
    let results = corpus.hybrid_search("domestic cats", &[1.0, 0.0], 5, &Cosine, 0.5, 60)?;
    assert_eq!(results[0].0, "cats");
    for (id, score) in results {
        println!("{score}: {}", store.record(&id).unwrap().text.as_deref().unwrap());
    }
    Ok(())
}
```

#### Python

```python
from vekryx import Vekryx


class ToyEmbeddings:
    def embed_document(self, text):
        return [1.0, 0.0] if "cat" in text.lower() else [0.8, 0.2]

    def embed_query(self, text):
        return self.embed_document(text)


store = Vekryx(embedding_model=ToyEmbeddings())
store.insert("cats", "Cats are domestic animals.", metadata={"domain": "animals"})
store.insert("dogs", "Dogs are domestic animals.", metadata={"domain": "animals"})
results = store.hybrid_search(
    "domestic cats", k=5, where={"domain": "animals"}, vector_weight=0.5, rrf_k=60,
)
assert results[0][0] == "cats"
for record_id, score in results:
    print(score, store.get_record(record_id)["text"])
```

`hybrid_search` requires a text query and an embedding model. It applies `where`
first, ranks all eligible vectors by the selected distance (default `cosine`), and
ranks eligible stored text using BM25 (`k1=1.2`, `b=0.75`, positive IDF). Tokenization
splits on non-alphanumeric characters and lowercases Unicode tokens. There is no
stemming or stop-word removal; repeated query tokens count once. BM25 statistics
are computed only from eligible records with text, including empty texts. Records
without original text can contribute to vector results but not keyword results.

The compatibility `hybrid_search` method combines the two full rankings using
[reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion):
`vector_weight / (rrf_k + vector_rank) + (1 - vector_weight) / (rrf_k + keyword_rank)`.
Ranks start at 1; absent ranks contribute zero. `rrf_k` defaults to 60 and must be
positive. `vector_weight` must be finite and between 0 and 1: 0 returns keyword
matches only; 1 uses vector ranks only. A query with no keyword matches falls back
to vector ranking unless its vector weight is zero. Keyword-only mode still embeds
and validates the query. This compatibility method uses the exact scan backend;
the options/session interface also accepts bounded vector candidate rankings.
Python metadata filters and Rust `filtered_view()` borrow the matching corpus;
they avoid copying stored vectors and records.

Hybrid results are `(ID, fusion_score)` with **larger scores first**, unlike the
smaller-is-better distances from `search` and `mmr_search`. Ties sort by ID; only
after fusion are results truncated to `k`. `k=0` returns `[]` after validation.

Rust callers use `insert_record(id, vector, Record { text, metadata })`,
`filtered_view(&metadata_filter)`, and then `hybrid_search(text, query_vector, k,
&metric, vector_weight, rrf_k)` on the borrowed corpus. See the runnable
[Python example](examples/hybrid_search.py) and [Rust example](examples/hybrid_search.rs).

### Distance Selection

| Choice | Use when | Behavior |
| --- | --- | --- |
| Euclidean | Your embedding model expects L2 distance, or magnitude matters. | Default for `search`; smaller is closer. Zero vectors are valid. |
| Cosine | Your embedding model compares direction rather than magnitude. | `1 − cosine similarity`, from 0 to 2. Zero query/stored vectors raise an error. |
| MMR + either metric | You want relevant matches with less repeated context. | Reranks candidates for relevance and diversity; returns original query distances in selection order. |

Choose the metric expected by your embedding model. For unit-normalized vectors,
Euclidean and cosine give the same nearest-neighbor ranking. Ordinary search
breaks distance ties by ID. `k=0` returns no matches after validation; `k` above
the record count returns all records. Euclidean overflow is reported as an error.

#### Rust

```rust
use vekryx::{distance::{Cosine, Euclidean}, Vekryx};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    store.insert("a".into(), vec![1.0, 0.0])?;
    store.insert("b".into(), vec![0.0, 1.0])?;
    assert_eq!(store.search(&[1.0, 0.0], 2)?,
               store.search_with_metric(&[1.0, 0.0], 2, &Euclidean)?);
    assert_eq!(store.search_with_metric(&[1.0, 0.0], 2, &Cosine)?[0], ("a".into(), 0.0));
    Ok(())
}
```

#### Python

```python
from vekryx import Vekryx

store = Vekryx(2)
store.insert("a", [1.0, 0.0])
store.insert("b", [0.0, 1.0])
assert store.search([1.0, 0.0], k=2) == store.search([1.0, 0.0], k=2, metric="euclidean")
assert store.search([1.0, 0.0], k=2, metric="cosine")[0] == ("a", 0.0)
```

### Diverse Retrieval

MMR is a selection strategy, not a separate distance. Python defaults to cosine
for MMR; Rust accepts a `DistanceMetric` implementation explicitly.

#### Rust

```rust
use vekryx::{distance::Cosine, Vekryx};

fn main() -> Result<(), vekryx::Error> {
    let mut store = Vekryx::new(2)?;
    for (id, vector) in [("a", vec![1.0, 0.0]), ("b", vec![1.0, 0.0]), ("c", vec![0.6, 0.8])] {
        store.insert(id.into(), vector)?;
    }
    let results = store.mmr_search(&[1.0, 0.0], 2, 3, 0.2, &Cosine)?;
    assert_eq!(results.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["a", "c"]);
    Ok(())
}
```

#### Python

```python
from vekryx import Vekryx

store = Vekryx(2)
store.insert("a", [1.0, 0.0])
store.insert("b", [1.0, 0.0])
store.insert("c", [0.6, 0.8])

results = store.mmr_search([1.0, 0.0], k=2, fetch_k=3, lambda_mult=0.2)
assert [record_id for record_id, _ in results] == ["a", "c"]
```

`fetch_k` must be at least `k`. `lambda_mult` must be finite and between 0 and 1:
higher values favor relevance; lower values favor diversity. The nearest result
is selected first; `1` reproduces nearest-neighbor order. Defaults are `k=5`,
`fetch_k=20`, and `lambda_mult=0.5` in Python.

Cosine MMR uses cosine similarity; Euclidean MMR uses `1 / (1 + distance)`.
Returned numbers are query distances, not MMR scores, and need not be sorted.

### Hierarchical Chunking

| Strategy | Python constructor | Tradeoff |
| --- | --- | --- |
| Fixed size | `FixedSizeChunker(chunk_size=512)` | Predictable bounds; may split words or ideas. |
| Overlapping windows | `FixedSizeChunker(chunk_size=512, overlap=64)` | Retains boundary context but repeats text. |
| Parent–child hierarchy | `HierarchicalChunker(chunk_sizes=[1024, 256])` | Search small leaves, then resolve larger parents through a separate lookup. |

#### Rust

```rust
use std::collections::BTreeMap;
use vekryx::chunking::{ChunkingStrategy, FixedSizeChunker, HierarchicalChunker};

fn main() -> Result<(), vekryx::Error> {
    let text = "Rust stores vectors. Python retrieves chunks.";
    let flat = FixedSizeChunker::new(12, 2)?.chunk(text, "flat-guide")?;
    assert!(flat.iter().all(|chunk| chunk.is_leaf));
    let chunks = HierarchicalChunker::new(vec![32, 12], 2)?.chunk(text, "guide")?;
    let lookup: BTreeMap<_, _> = chunks.iter().map(|chunk| (chunk.id.as_str(), chunk)).collect();
    for leaf in chunks.iter().filter(|chunk| chunk.is_leaf) {
        let parent = leaf.parent_id.as_deref().map(|id| lookup[id]).unwrap_or(leaf);
        let original: String = text.chars().skip(leaf.start).take(leaf.end - leaf.start).collect();
        assert_eq!(original, leaf.text);
        println!("{} -> {}", leaf.id, parent.text);
    }
    Ok(())
}
```

#### Python

```python
from vekryx import FixedSizeChunker, HierarchicalChunker

text = "Rust stores vectors. Python retrieves chunks."
flat = FixedSizeChunker(chunk_size=12, overlap=2).chunk(text, "flat-guide")
assert all(chunk.is_leaf for chunk in flat)
chunks = HierarchicalChunker(chunk_sizes=[32, 12], overlap=2).chunk(text, "guide")
lookup = {chunk.id: chunk for chunk in chunks}

for leaf in (chunk for chunk in chunks if chunk.is_leaf):
    parent = lookup[leaf.parent_id] if leaf.parent_id else leaf
    assert text[leaf.start:leaf.end] == leaf.text
    print(leaf.id, "→", parent.text)
```

Each chunk contains `id`, `parent_id`, `level`, `start`, `end`, `text`, and
`is_leaf`. Sizes and offsets count **Unicode scalar values, not tokens or bytes**;
whitespace is preserved. Hierarchical sizes must be positive and strictly
decreasing; overlap must be nonnegative and smaller than every configured size.
Empty text returns no chunks. Output is breadth-first, with level-zero roots.

Use unique document IDs. Chunk IDs are deterministic positional identifiers;
re-chunking a document requires removing obsolete IDs and replacing its metadata.
Embed leaves with your own model, store their IDs and vectors, and retain the
chunk lookup for parent expansion. See the [complete example](examples/hierarchical_chunking.py).
Semantic, sentence-aware, and code-aware splitters are not implemented.

### Snapshots

Persistence is explicit in both APIs: `save(path)` writes a JSON snapshot and
`load(path)` creates an independent in-memory store. Relative paths resolve
against the process's current working directory. There is no default database
location or automatic saving.

The quick starts and full usage examples write **`data/vekryx.json`**, containing:

```json
{"dimensions":3,"vectors":{"chunk-a":[1.0,0.0,0.0],"chunk-b":[0.0,1.0,0.0]}}
```

The parent directory must exist. Saving replaces the snapshot; concurrent writers
are not coordinated. Chunking results are not saved automatically: persist the
chunk lookup separately, or insert chunk text and hierarchy fields as record text
and JSON metadata. Those record fields are included in snapshots in both APIs.
Root-level `data/` is Git-ignored.

Python reports invalid values/snapshots as `ValueError` and filesystem failures
as `OSError` subclasses. Rust snapshot methods return `std::io::Result`, with
`ErrorKind::InvalidData` for invalid contents. Existing snapshots work across
both languages; metric choice is per search and is not part of the file format.

#### Rust

```rust
use vekryx::Vekryx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("vekryx-readme-{}.json", std::process::id()));
    let mut store = Vekryx::new(2)?;
    store.insert("a".into(), vec![1.0, 0.0])?;
    store.save(&path)?;
    let restored = Vekryx::load(&path)?;
    assert_eq!(restored.get("a"), store.get("a"));
    let json = store.to_json()?;
    assert_eq!(Vekryx::from_json(&json)?.dimensions(), 2);
    std::fs::remove_file(path)?;
    Ok(())
}
```

#### Python

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from vekryx import Vekryx

with TemporaryDirectory() as directory:
    path = Path(directory) / "store.json"
    store = Vekryx(2)
    store.insert("a", [1.0, 0.0], metadata={"source": "guide"})
    store.save(path)
    restored = Vekryx.load(path)
    assert restored.get("a") == store.get("a")
    assert restored.get_record("a")["metadata"] == {"source": "guide"}
```

Either language can load the other's supported snapshot. Loading a Python text
store with `embedding_model=` reattaches the adapter; loading it without one
still permits numeric-vector searches.

#### Compressed JSON

| Format | Rust | Python |
| --- | --- | --- |
| Plain JSON | `save` / `load` | `save` / `load` |
| Gzip JSON | `save_with_format` / `load_with_format` with `SnapshotFormat::CompressedJson` | `save` / `load` with `format="compressed_json"` |
| Memory-mapped | `save_mapped` / `MappedVekryx::open` | Export with `save_mapped` |

Gzip snapshots store the same JSON document and load into a mutable store. Select
the format explicitly when saving and loading.

```rust
use vekryx::{SnapshotFormat, Vekryx};

let store = Vekryx::new(2)?;
let format = SnapshotFormat::CompressedJson;
store.save_with_format("store.json.gz", format)?;
let restored = Vekryx::load_with_format("store.json.gz", format)?;
```

```python
store.save("store.json.gz", format="compressed_json")
restored = Vekryx.load("store.json.gz", format="compressed_json")
```

#### Memory-mapped vectors

Rust and Python can export a versioned binary snapshot with `save_mapped(path)`.
Rust's `MappedVekryx` opens that file read-only on little-endian hosts. It keeps
IDs and records in memory and borrows vector values from the mapped file. Exact
search scans the mapped vectors and can page them in as needed.

```rust
use vekryx::{MappedVekryx, Vekryx};

let mut store = Vekryx::new(2)?;
store.insert("a".into(), vec![1.0, 0.0])?;
store.save_mapped("store.vmap")?;

// SAFETY: This example owns the file and keeps it immutable while mapped.
let mapped = unsafe { MappedVekryx::open("store.vmap")? };
assert_eq!(mapped.search(&[1.0, 0.0], 1)?, vec![("a".into(), 0.0)]);
```

Keep the underlying file immutable for the lifetime of every open mapping.
Publish updates by saving a new snapshot with `save_mapped`, which writes a
temporary file and replaces the destination entry on success.

### Extending Through Interfaces

The extension points differ by language: Rust accepts custom metrics and flat
splitters through traits; Python accepts custom embedding adapters through its
`EmbeddingModel` protocol. Python currently exposes built-in metrics and chunkers,
so a Python subclass does not add a native metric or splitter implementation.

#### Rust: Inject a Metric and Compose Splitters

This Manhattan metric validates its public inputs and uses the trait's default
MMR similarity. The custom first-level splitter keeps the complete document;
the fixed-size second level produces smaller children. Neither extension requires
changing `Vekryx` or inheriting from another concrete type.

```rust
use std::ops::Range;
use vekryx::{
    chunking::{ChunkingStrategy, FixedSizeChunker, HierarchicalChunker, TextSplitter},
    distance::DistanceMetric,
    Error, Vekryx,
};

struct Manhattan;

impl DistanceMetric for Manhattan {
    fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, Error> {
        if left.is_empty() || left.len() != right.len() {
            return Err("vectors must have the same positive dimension".into());
        }
        if left.iter().chain(right).any(|value| !value.is_finite()) {
            return Err("vector values must be finite".into());
        }
        let distance: f64 = left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum();
        if !distance.is_finite() {
            return Err("Manhattan distance exceeds the finite f64 range".into());
        }
        Ok(distance)
    }
}

struct WholeDocument;

impl TextSplitter for WholeDocument {
    fn split(&self, text: &str) -> Result<Vec<Range<usize>>, Error> {
        let length = text.chars().count();
        Ok(if length == 0 { Vec::new() } else { std::iter::once(0..length).collect() })
    }
}

fn main() -> Result<(), Error> {
    let mut store = Vekryx::new(2)?;
    store.insert("a".into(), vec![1.0, 2.0])?;
    assert_eq!(store.search_with_metric(&[0.0, 0.0], 1, &Manhattan)?, vec![("a".into(), 3.0)]);
    let hierarchy = HierarchicalChunker::from_splitters(vec![
        Box::new(WholeDocument),
        Box::new(FixedSizeChunker::new(12, 2)?),
    ])?;
    let chunks = hierarchy.chunk("One document, several child chunks.", "guide")?;
    assert!(chunks.iter().any(|chunk| chunk.parent_id.is_some()));
    Ok(())
}
```

#### Python: Supply an Embedding Adapter

The adapter implements the protocol structurally; it does not inherit from
`EmbeddingModel`. This offline toy feature mapping only demonstrates the contract,
not semantic embedding quality. A real adapter can wrap a local model or service.

```python
from vekryx import EmbeddingModel, Vekryx


class TopicEmbeddings:
    def embed_document(self, text):
        return [1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0]

    def embed_query(self, text):
        return self.embed_document(text)


model: EmbeddingModel = TopicEmbeddings()
store = Vekryx(embedding_model=model)
store.insert("cats", "Cats are domestic animals.")
store.insert("rust", "Rust is a programming language.")
assert store.dimensions == 2
assert store.search("cats", k=1, metric="cosine")[0][0] == "cats"
```

### More Examples

| Example | Covers |
| --- | --- |
| [Rust usage](examples/rust_usage.rs) / [Python usage](examples/python_usage.py) | Matching storage, retrieval, chunking, and disk workflows. |
| [Basic operations](examples/basic_usage.py) | Updates, deletion, missing IDs, and validation. |
| [Nearest neighbors](examples/nearest_neighbors.py) | Distances, ties, and identical vectors under different IDs. |
| [ANN algorithms: Rust](examples/ann_search.rs) / [Python](examples/ann_search.py) | HNSW, IVF-Flat, PQ, residual IVF-PQ, and exact-oracle recall comparison. |
| [Search pipeline: Rust](examples/search_pipeline.rs) / [Python](examples/search_pipeline.py) | Typed options, filtered sessions, named results, MMR, and hybrid fusion. |
| [Hybrid retrieval: Rust](examples/hybrid_search.rs) / [Python](examples/hybrid_search.py) | Filtered BM25 and vector fusion. |
| [Cosine and MMR](examples/cosine_and_mmr.py) | Relevance versus diversity. |
| [Hierarchical chunks](examples/hierarchical_chunking.py) | Leaf retrieval and parent expansion with clearly labeled toy features. |
| [Persistence](examples/persistence.py) | Snapshot replacement in a temporary directory. |

```bash
cargo run --example rust_usage -- /tmp/rust-vectors.json
python examples/python_usage.py /tmp/python-vectors.json
python examples/hierarchical_chunking.py
```

The full usage examples replace their selected snapshot with sample records.

## Development

```bash
# Rust core; no Python required.
cargo test --all-targets --features experimental
cargo test --doc
cargo fmt --check
cargo clippy --all-targets --features experimental -- -D warnings
bash -n install.sh

# Rebuild and test the Python extension.
bash install.sh --python python3.11
source .venv/bin/activate
python -m unittest discover -s tests/python_test -p 'test_*.py'
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" cargo test --lib --features python
```

Tests cover numeric validation, metrics, MMR, hierarchy invariants, disk
persistence, downstream backend contracts, session mutation safety, bounded hybrid
fusion, HNSW recall/topology, IVF all-list equivalence, PQ reconstruction and
reranking, owned Python index snapshots, and the installed Python API. Dependency and build configuration
live in [Cargo.toml](Cargo.toml) and [pyproject.toml](pyproject.toml).

See [Contributing](contribution.md) for setup, focused validation, and pull
request guidelines. Public API changes should update both language examples.

## License

This repository currently has no `LICENSE` file or license declaration in
`Cargo.toml`. Licensing terms have not yet been specified.
