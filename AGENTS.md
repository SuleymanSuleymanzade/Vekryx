# Repository Guidelines

## Project Structure & Module Organization

This Rust 2021 project implements vector storage and nearest-neighbor search with an experimental Python interface.

- `src/store.rs` contains the supported `Vekryx` store; `src/search.rs` defines borrowed `Corpus` search views.
- `src/distance.rs` defines metrics; `src/hybrid.rs` implements hybrid retrieval.
- `src/state_manager/snapshot.rs` handles atomic JSON persistence.
- `src/experimental/` contains optional legacy indexes, enabled with the `experimental` feature.
- `src/lib.rs` exports the Rust API; `src/python.rs` defines optional PyO3 bindings.
- `src/main.rs` runs a supported-store demonstration; `examples/` contains usage examples.
- Rust unit tests live beside implementations. `tests/python_test/` tests the installed Python API.
- `assets/` holds README illustrations; `tests/fixtures/` holds sample snapshots.

## Build, Test, and Development Commands

Run commands from the repository root:

- `cargo build` — compile the Rust binary and shared library.
- `cargo run` — execute the supported-store demonstration.
- `cargo test --all-targets --features experimental` — test the core, experimental indexes, and examples.
- `cargo fmt --check` — check standard Rust formatting; use `cargo fmt` to apply it.
- `cargo clippy --all-targets` — inspect compiler and lint diagnostics.
- `maturin develop` — build/install the Python extension inside an activated virtual environment, as described in `README.md`. Install Python and maturin first.

The official project name is Vekryx. Use `vekryx` for the Rust crate and Python module, and `Vekryx` for the public store type. Keep naming consistent across bindings, examples, and docs. Use `Cargo.toml` as the dependency reference.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for types and traits. Keep supported retrieval separate from `experimental` indexes and persistence logic in `state_manager`. Avoid unrelated formatting changes when editing legacy code.

## Testing Guidelines

Use Rust's built-in `#[test]` framework in local `#[cfg(test)]` modules. Name new tests after the behavior being verified. Cover insertion, retrieval, dimensional assumptions, and serialization round trips when relevant. LSH uses random projections; prefer reproducible fixtures and assertions that tolerate valid approximate results. No coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `unit-test for LSH` and `file structure updated`; no strict prefix convention is established. Write focused, imperative subjects and keep commits scoped. Follow the fork-and-feature-branch workflow in `contribution.md`. In pull requests, describe the changed behavior, link relevant issues, and report validation commands and any failures or Python compatibility limitations.
