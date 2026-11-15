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
  <a href="#usage">Usage</a> ·
  <a href="contribution.md">Contribute</a>
</p>

## Overview

Vekryx is an early-stage, in-memory vector search library with native
Rust and optional Python APIs. Store numeric vectors by string ID, retrieve
nearest neighbors, diversify results with maximal marginal relevance (MMR), and
split text into flat or hierarchical chunks before embedding.

The supported store performs an exact scan and sorts matches. It is intended
for small collections; this repository does not publish benchmark results or
claim indexed-search performance. Python can generate embeddings through optional
model adapters; Rust accepts vectors from your application.

The Rust crate and Python package are named `vekryx`; the public store type is
`Vekryx` in both languages. This is a breaking package/API rename: update imports
and rebuild the extension. Existing vector-only JSON snapshots remain readable; new snapshots also retain
original text and metadata when supplied.

## Features

- **Record storage:** optional UUIDs in Python, original text, JSON metadata, and finite `f64` vectors.
- **Filter before search:** exact metadata matching before ranking and candidate limits.
- **Hybrid retrieval:** BM25 keyword search fused with vector rankings using weighted RRF.
- **Exact retrieval:** Euclidean and cosine distances with deterministic ties.
- **Diverse results:** MMR selection over a configurable nearest-candidate pool.
- **Text preparation:** fixed-size, overlapping, and hierarchical parent–child chunks.
- **Explicit persistence:** interoperable JSON snapshots for Rust and Python.
- **Trait-based extension:** independent metrics, flat splitters, and chunking strategies.

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
    D[Document] --> C[Optional chunking]
    C --> T[Text or dictionary record]
    T --> ID[Optional ID or generated UUID]
    ID --> E[Embedding adapter]
    E --> V[Validate vector and inferred dimensions]
    V --> S[(Rust store: IDs, vectors, text, metadata)]
    S <--> P[Atomic JSON snapshots]
    Q[Query text] --> QE[Same embedding adapter]
    W[Metadata where filter] --> F[Filter corpus first]
    S --> F
    QE --> VS[Vector scoring on filtered corpus]
    F --> VS
    F --> BM[BM25 on filtered text corpus]
    Q --> BM
    VS --> NN[Nearest vector results]
    VS --> M[MMR candidate selection]
    VS --> R[Weighted reciprocal rank fusion]
    BM --> R
    R --> H[Top k hybrid results: ID and fusion score]
    NN --> L[Record lookup by ID]
    M --> L
    H --> L
```

The diagram uses a fenced `mermaid` block that GitHub renders directly. Metadata
filtering precedes scoring, BM25 corpus statistics, MMR `fetch_k`, and final `k`.
The query embedding is computed independently; only eligible records are ranked.

- [Vekryx](src/store.rs) owns records, validates vectors, ranks matches, and selects MMR results.
- [Hybrid ranking](src/hybrid.rs) combines BM25 and vector ranks after filtering.
- [DistanceMetric](src/algorithms/distance.rs) separates Euclidean/cosine behavior from storage.
- [TextSplitter and ChunkingStrategy](src/chunking/mod.rs) separate flat splitting from hierarchy construction. `HierarchicalChunker::from_splitters` composes custom splitters.
- [Snapshot writer](src/state_manager/snapshot.rs) replaces JSON files through a temporary file in the same directory.
- [Python bindings](src/python.rs) adapt the store; [chunk bindings](src/chunking/python.rs) expose read-only chunk records.

The legacy KD-tree/LSH prototypes and `src/main.rs` demonstration are separate
from the supported store. They do not accelerate `Vekryx`.
There is no persistent CLI store or automatic parent expansion. Python's optional
[embedding adapters](python/vekryx/embeddings.py) prepare vectors before calling
the Rust store.

## Usage

### Dictionary Records and Filters

```python
from vekryx import Vekryx

store = Vekryx(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
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

```python
results = store.hybrid_search(
    "domestic cats", k=5,
    where={"domain": "animals"},
    vector_weight=0.5,
)
for id, score in results:
    print(score, store.get_record(id)["text"])
```

`hybrid_search` requires a text query and an embedding model. It applies `where`
first, ranks all eligible vectors by the selected distance (default `cosine`), and
ranks eligible stored text using BM25 (`k1=1.2`, `b=0.75`, positive IDF). Tokenization
splits on non-alphanumeric characters and lowercases Unicode tokens. There is no
stemming or stop-word removal; repeated query tokens count once. BM25 statistics
are computed only from eligible records with text, including empty texts. Records
without original text can contribute to vector results but not keyword results.

The two full rankings are combined using
[reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion):
`vector_weight / (rrf_k + vector_rank) + (1 - vector_weight) / (rrf_k + keyword_rank)`.
Ranks start at 1; absent ranks contribute zero. `rrf_k` defaults to 60 and must be
positive. `vector_weight` must be finite and between 0 and 1: 0 returns keyword
matches only; 1 uses vector ranks only. A query with no keyword matches falls back
to vector ranking unless its vector weight is zero. Keyword-only mode still embeds
and validates the query. This is exact in-memory retrieval, with no search index.
Metadata filters currently materialize a matching corpus, so they add copying cost.

Hybrid results are `(ID, fusion_score)` with **larger scores first**, unlike the
smaller-is-better distances from `search` and `mmr_search`. Ties sort by ID; only
after fusion are results truncated to `k`. `k=0` returns `[]` after validation.

Rust callers use `insert_record(id, vector, Record { text, metadata })`,
`filtered(&metadata_filter)`, and then `hybrid_search(text, query_vector, k,
&metric, vector_weight, rrf_k)` on the filtered store. See the runnable
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

### Diverse Retrieval

MMR is a selection strategy, not a separate distance. Python defaults to cosine
for MMR; Rust accepts a `DistanceMetric` implementation explicitly.

```python
from vekryx import Vekryx

store = Vekryx(2)
store.insert("a", [1.0, 0.0])
store.insert("b", [1.0, 0.0])
store.insert("c", [0.6, 0.8])

results = store.mmr_search([1.0, 0.0], k=2, fetch_k=3, lambda_mult=0.2)
assert [record_id for record_id, _ in results] == ["a", "c"]
```

The Rust equivalent is `store.mmr_search(&query, 2, 3, 0.2, &Cosine)?`.
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

```python
from vekryx import HierarchicalChunker

text = "Rust stores vectors. Python retrieves chunks."
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
are not coordinated. Chunk text and hierarchy metadata are **not included** and
must be persisted separately. `data/` is Git-ignored.

Python reports invalid values/snapshots as `ValueError` and filesystem failures
as `OSError` subclasses. Rust snapshot methods return `std::io::Result`, with
`ErrorKind::InvalidData` for invalid contents. Existing snapshots work across
both languages; metric choice is per search and is not part of the file format.

### More Examples

| Example | Covers |
| --- | --- |
| [Rust usage](examples/rust_usage.rs) / [Python usage](examples/python_usage.py) | Matching storage, retrieval, chunking, and disk workflows. |
| [Basic operations](examples/basic_usage.py) | Updates, deletion, missing IDs, and validation. |
| [Nearest neighbors](examples/nearest_neighbors.py) | Distances, ties, and identical vectors under different IDs. |
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
cargo test --lib
cargo clippy --lib --examples -- -D warnings
bash -n install.sh

# Rebuild and test the Python extension.
bash install.sh --python python3.11
source .venv/bin/activate
python -m unittest discover -s tests/python_test -p 'test_*.py'
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" cargo test --lib --features python
```

Tests cover numeric validation, metrics, MMR, hierarchy invariants, disk
persistence, and the installed Python API. Dependency and build configuration
live in [Cargo.toml](Cargo.toml) and [pyproject.toml](pyproject.toml).

See [Contributing](contribution.md) for setup, focused validation, and pull
request guidelines. Public API changes should update both language examples.

## License

This repository currently has no `LICENSE` file or license declaration in
`Cargo.toml`. Licensing terms have not yet been specified.
