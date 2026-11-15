# Repository Guidelines

## Project Structure & Module Organization

This Rust 2021 project implements vector storage and nearest-neighbor search with an experimental Python interface.

- `src/algorithms/` contains KD-tree, locality-sensitive hashing (LSH), vector storage, and the `FastSearch` trait; `src/algorithms.rs` declares these modules.
- `src/state_manager/local_saver.rs` handles JSON persistence.
- `src/lib.rs` defines the PyO3 bindings and Python module; `src/main.rs` runs a standalone LSH demonstration.
- Rust unit tests live beside their implementation. `tests/python_test/` contains exploratory Python scripts, rather than an automated Python test suite.
- `assets/` holds README illustrations; `local_file.json` is sample persisted state.

## Build, Test, and Development Commands

Run commands from the repository root:

- `cargo build` — compile the Rust binary and shared library.
- `cargo run` — execute the LSH demonstration.
- `cargo test --bin vector_database` — run algorithm unit tests through the binary target.
- `cargo fmt --check` — check standard Rust formatting; use `cargo fmt` to apply it.
- `cargo clippy --all-targets` — inspect compiler and lint diagnostics.
- `maturin develop` — build/install the Python extension inside an activated virtual environment, as described in `README.md`. Install Python and maturin first.

The Python interface is experimental: the Cargo library name, PyO3 module name, and example imports currently differ. Check those names before relying on the Python scripts. Use `Cargo.toml` as the dependency reference; `contribution.md` includes older versions.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for types and traits. Keep search algorithms in their existing modules and persistence logic in `state_manager`. Avoid unrelated formatting changes when editing legacy code.

## Testing Guidelines

Use Rust's built-in `#[test]` framework in local `#[cfg(test)]` modules. Name new tests after the behavior being verified. Cover insertion, retrieval, dimensional assumptions, and serialization round trips when relevant. LSH uses random projections; prefer reproducible fixtures and assertions that tolerate valid approximate results. No coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `unit-test for LSH` and `file structure updated`; no strict prefix convention is established. Write focused, imperative subjects and keep commits scoped. Follow the fork-and-feature-branch workflow in `contribution.md`. In pull requests, describe the changed behavior, link relevant issues, and report validation commands and any failures or Python compatibility limitations.
