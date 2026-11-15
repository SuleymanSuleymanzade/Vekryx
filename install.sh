#!/usr/bin/env bash
# Install the Python API in an isolated virtual environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--python EXECUTABLE] [--venv DIRECTORY]

Build and install vector_database in a virtual environment (default: .venv next
to this script). Requires Cargo/Rust, a native build toolchain, and Python
3.8–3.12 with venv support. Build dependencies are installed automatically.

Options:
  --python EXECUTABLE  Select a Python interpreter, e.g. python3.11
  --venv DIRECTORY    Select the installation environment
  -h, --help          Show this help

Existing compatible environments are reused. No sudo or global pip install is
used. Run this script again after changing the Rust source.
HELP
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
compatible() {
    "$1" -c 'import sys; sys.exit(not ((3, 8) <= sys.version_info[:2] < (3, 13)))' >/dev/null 2>&1
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$repo_dir/.venv"
chosen_python=""
while (($#)); do
    case "$1" in
        --python|--venv)
            (($# >= 2)) || die "Missing value for $1"
            [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            if [[ "$1" == --python ]]; then chosen_python="$2"; else venv_dir="$2"; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

# Rustup installations may not yet be on a newly started process's PATH.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
command -v cargo >/dev/null 2>&1 || die "Install Rust/Cargo first, then rerun this script."
cargo --version >/dev/null 2>&1 || die "Cargo is present but its toolchain is not working."
command -v rustc >/dev/null 2>&1 || die "rustc is missing from the Rust toolchain."
rustc --version >/dev/null 2>&1 || die "The Rust compiler is not working."

if [[ -n "$chosen_python" ]]; then
    compatible "$chosen_python" || die "--python must select a working Python 3.8–3.12 interpreter."
elif [[ -x "$venv_dir/bin/python" ]] && compatible "$venv_dir/bin/python"; then
    chosen_python="$venv_dir/bin/python"
else
    for candidate in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && compatible "$candidate"; then
            chosen_python="$candidate"
            break
        fi
    done
    [[ -n "$chosen_python" ]] || die "Python 3.8–3.12 was not found; install one or pass --python /path/to/python."
fi

if [[ -e "$venv_dir" ]]; then
    [[ -f "$venv_dir/pyvenv.cfg" && -x "$venv_dir/bin/python" ]] || die "$venv_dir already exists and is not a virtual environment."
    compatible "$venv_dir/bin/python" || die "$venv_dir uses an unsupported Python; choose a different --venv directory."
    requested_version="$("$chosen_python" -c 'import sys; print(sys.version_info[:2])')"
    environment_version="$("$venv_dir/bin/python" -c 'import sys; print(sys.version_info[:2])')"
    [[ "$requested_version" == "$environment_version" ]] || die "The existing environment uses another Python version; choose a different --venv directory."
else
    "$chosen_python" -m venv "$venv_dir" || die "Could not create the environment; check Python's venv/ensurepip support."
fi

venv_dir="$(cd -- "$venv_dir" && pwd)"
venv_python="$venv_dir/bin/python"
"$venv_python" -m pip --version >/dev/null 2>&1 || die "pip is missing from the selected environment."
printf 'Installing with %s into %s\n' "$("$venv_python" --version)" "$venv_dir"
PYO3_PYTHON="$venv_python" "$venv_python" -m pip install "$repo_dir"
"$venv_python" -c 'from vector_database import VectorStore, FixedSizeChunker, HierarchicalChunker; assert len(VectorStore(3)) == 0; print("Import check passed")'
printf '\nInstalled successfully. Activate with:\n  source %q\n' "$venv_dir/bin/activate"
printf 'Then run:\n  python %q\n' "$repo_dir/examples/python_usage.py"
