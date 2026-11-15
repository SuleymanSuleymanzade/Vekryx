# Contributing

Fork the repository, clone your fork, and create a focused branch:

```bash
git clone git@github.com:YOUR_NAME/Vekryx.git
cd Vekryx
git switch -c describe-your-change
```

## Development Setup

| Workflow | Requirements | Setup |
| --- | --- | --- |
| Rust core | Rust 1.82+, Cargo, native linker/build tools | `cargo build --lib`; Python is not required. |
| Python bindings | Rust toolchain and Python 3.8–3.12 with venv/pip | `bash install.sh --python python3.11`, then `source .venv/bin/activate`. |
| Custom Python environment | Same Python requirements | `bash install.sh --python /path/to/python --venv /path/to/environment`. |

The installer builds and verifies the Python package without requiring direct
maturin commands or global pip installs. Rerun it after changing Rust code;
editing the source does not update an already-installed extension. Packaging
uses maturin internally, as configured in `pyproject.toml`. Use `Cargo.toml` as
the source of truth for dependencies rather than copying version lists.

## Architecture and Style

Keep storage/search logic in Rust and argument/error conversion in Python
bindings. Python support is behind the optional `python` Cargo feature; the core
must continue to build without it.

| Area | Location / extension point | Expected responsibility |
| --- | --- | --- |
| Hybrid retrieval | `src/hybrid.rs` | BM25 and weighted reciprocal rank fusion on the filtered corpus. |
| Storage and MMR | `src/store.rs` | IDs, validation, ranking, selection, and snapshot API. |
| Metrics | `src/algorithms/distance.rs`, `DistanceMetric` | Distance and similarity behavior independent of storage. |
| Chunking | `src/chunking/`, `TextSplitter` and `ChunkingStrategy` | Flat splitting and hierarchy composition independent of embeddings. |
| Persistence | `src/state_manager/snapshot.rs` | Atomic snapshot replacement. |
| Python | `src/python.rs`, `src/chunking/python.rs` | Expose Rust behavior with stable arguments and Python exceptions. |

Use rustfmt, four-space indentation, `snake_case` functions/modules, and
`PascalCase` types/traits. Prefer composition through existing traits over
inheritance-like wrappers. Avoid unrelated formatting of legacy algorithms.

When adding a metric, test dimensions, nonfinite values, ties, zero-vector
behavior, and numeric stability. When adding chunking, test Unicode offsets,
text coverage, overlap, unique IDs, and parent containment. New functionality
should preserve existing defaults or explicitly document compatibility changes.

## Verification

Run checks appropriate to the affected components:

```bash
cargo test --lib
cargo clippy --lib -- -D warnings
cargo run --example rust_usage -- /tmp/vector-contributor-rust.json
bash -n install.sh
```

For bindings, rebuild through the installer and test the actual installation:

```bash
bash install.sh --python python3.11
source .venv/bin/activate
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" cargo test --lib --features python
python -m unittest discover -s tests/python_test -p 'test_*.py'
python examples/python_usage.py /tmp/vector-contributor-python.json
python examples/hierarchical_chunking.py
```

Use `rustfmt --edition 2021 path/to/changed.rs` for changed modules, or
`cargo fmt --check` to inspect repository formatting. Legacy modules may have
pre-existing warnings or formatting differences; report which targets you checked.

Persistence changes need disk round trips, replacement/error checks, and
compatibility tests for the JSON format. Python and Rust must read each other's
vector snapshots. Save paths are caller-selected; the API never saves implicitly.
Record text and JSON metadata are saved with vectors; old vector-only snapshots
remain readable. Chunk hierarchy relationships must be supplied as record metadata
by the application if they should be persisted.
Do not commit local environments or generated databases (`data/` is ignored).

## Pull Requests

Use short imperative commit subjects and keep each commit focused. Push your
branch and open a pull request against `main`:

```bash
git add path/to/changed/files
git commit -m "Describe the behavior change"
git push -u origin describe-your-change
```

Describe the problem, resulting behavior, linked issues where applicable, and
the commands/results used for validation. Update README API tables and both
Python and Rust examples for public functionality. Explain any file-format,
installation, or backwards-compatibility changes and disclose untested paths.
