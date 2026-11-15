# Vector Database

![Vector Database logo](./assets/vector_db.png)

A Rust vector store with a Python object API. Store vectors by string ID, retrieve
or replace them, and find the closest records using exact Euclidean search.
The Python API is intended for small, in-memory collections and supports JSON
snapshots for persistence.

## Installation

Requirements: a Rust toolchain with Cargo, and Python 3.8–3.12. Python 3.11 is
tested. The current PyO3 dependency does not support Python 3.13 or 3.14.
Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

NumPy is not required to use the API. Reinstall after changing the Rust source,
or use `maturin develop` as described below.

## Quick Start

```python
from vector_database import VectorStore

store = VectorStore(dimensions=3)
store.insert("apple", [1.0, 2.0, 3.0])
store.insert("orange", [4.0, 5.0, 6.0])

print(store.get("apple"))                   # [1.0, 2.0, 3.0]
print(store.search([1.0, 2.0, 3.0], k=1))   # [('apple', 0.0)]
print(len(store))                          # 2
print(store.dimensions)                    # 3

# Inserting an existing ID replaces its vector.
store.insert("apple", [2.0, 2.0, 3.0])
store.save("vectors.json")

restored = VectorStore.load("vectors.json")
print(restored.get("apple"))               # [2.0, 2.0, 3.0]
print(restored.delete("orange"))           # True
print(restored.delete("missing"))          # False
```

## Runnable Examples

After installing the package, run these scripts from the repository root:

```bash
python examples/basic_usage.py
python examples/nearest_neighbors.py
python examples/persistence.py
```

- [Basic usage](examples/basic_usage.py): insert, retrieve, update, delete, and
  handle missing IDs or invalid dimensions.
- [Nearest neighbors](examples/nearest_neighbors.py): ranked search, distance
  values, ties, and identical vectors stored under different IDs.
- [Persistence](examples/persistence.py): save, load, and replace snapshots in a
  temporary directory without leaving files behind.

## Python API

Import the class with `from vector_database import VectorStore`.

| API | Returns | Behavior |
| --- | --- | --- |
| `VectorStore(dimensions)` | `VectorStore` | Create an empty store with a positive, fixed dimension. |
| `store.dimensions` | `int` | Read the configured dimension; this property is read-only. |
| `len(store)` | `int` | Count stored records. |
| `store.insert(id, vector)` | `None` | Add a record or replace the vector for an existing ID. |
| `store.get(id)` | `list[float]` | Return a copy of the vector; raise `KeyError` if absent. |
| `store.delete(id)` | `bool` | Remove a record; return `False` if the ID is absent. |
| `store.search(query, k=5)` | `list[tuple[str, float]]` | Return up to `k` `(ID, distance)` pairs, nearest first. |
| `store.save(path)` | `None` | Write all records and the dimension to a JSON snapshot. |
| `VectorStore.load(path)` | `VectorStore` | Construct a new store from a validated JSON snapshot. |

### Inputs and Search Results

- IDs must be nonempty strings. Different IDs can contain identical vectors.
- Vectors and queries accept Python numeric sequences, including lists and
  tuples, and must contain exactly `dimensions` finite numbers.
- Invalid dimensions, empty IDs, NaN, and infinity raise `ValueError`.
  Incompatible Python argument types raise conversion errors.
- Search uses Euclidean distance, not a similarity score: smaller is closer and
  zero means identical coordinates. Ties are sorted by ID.
- `k` must be a nonnegative integer. `k=0` returns `[]`; a larger `k` than the
  record count returns all records. An empty store returns `[]` for a valid query.
- Search scans every stored vector and sorts the results. It does not use the
  experimental LSH or KD-tree implementations.

### Persistence and Errors

`save` and `load` accept strings or `pathlib.Path` objects. Saving replaces an
existing snapshot using a temporary file in the same directory. The parent
directory must already exist. Changes remain in memory until explicitly saved;
loading creates an independent store.

Malformed JSON or invalid snapshot contents raise `ValueError`. Filesystem
failures, such as missing files or directories, raise `OSError` subclasses.

```python
try:
    vector = store.get("unknown")
except KeyError:
    print("Record does not exist")

try:
    store.insert("bad", [1.0])
except ValueError as error:
    print(error)  # expected 3 dimensions, got 1
```

Snapshots contain raw vector data and IDs. The format is a JSON object with
`dimensions` and `vectors` fields; it is separate from the legacy index state
in `local_file.json`.

## Development and Testing

Inside the activated Python environment:

```bash
python -m pip install maturin
maturin develop
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
cargo test --lib
cargo clippy --lib -- -D warnings
python -m unittest discover -s tests/python_test -p 'test_*.py'
```

`maturin develop` rebuilds and installs the extension into the active environment.
The Rust tests exercise storage and ranking; the Python tests exercise the
installed API, validation, and persistence. Dependency versions are defined in
[Cargo.toml](Cargo.toml), and Python packaging in [pyproject.toml](pyproject.toml).

## Repository Layout and Status

- `src/store.rs`: ID-based storage, validation, and exact search.
- `src/lib.rs`: Python bindings for `VectorStore`.
- `src/state_manager/snapshot.rs`: snapshot file writes.
- `src/algorithms/`: experimental KD-tree, LSH, and earlier vector store code.
- `src/main.rs`: hardcoded LSH demonstration, runnable with `cargo run` after
  selecting a compatible Python interpreter; it is not a persistent CLI store.
- `tests/python_test/`: API tests and exploratory examples.

The supported Python API replaces the earlier experimental `VectorDB` wrapper.
Java/Mojo APIs, HDFS/S3 integration, PyTorch integration, and a web UI are not
implemented. The legacy algorithms have known limitations and are not used by
`VectorStore`.

## Contributing

See [contribution.md](contribution.md) for the contribution workflow. Include a
clear description of behavior changes and the relevant test results in pull
requests.
