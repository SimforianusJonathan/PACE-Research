#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${PACE_RUNTIME_DIR:-/tmp/pace-kaggle}"
python_version=3.11.15
uv_version=0.13.5
dry_run=()
if [[ ${1:-} == --dry-run && $# -eq 1 ]]; then
    dry_run=(--dry-run)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/setup_kaggle.sh [--dry-run]" >&2
    exit 2
fi

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "The Kaggle CUDA lock targets Linux x86-64." >&2
    exit 1
fi

mkdir -p "$runtime"
export UV_PYTHON_INSTALL_DIR="$runtime/python"
export UV_CACHE_DIR="$runtime/cache"
if [[ ! -x "$runtime/bootstrap/bin/uv" ]] || [[ "$("$runtime/bootstrap/bin/uv" --version)" != "uv $uv_version"* ]]; then
    if command -v uv >/dev/null; then
        uv venv --allow-existing --python "$(command -v python3)" "$runtime/bootstrap"
        uv pip install --python "$runtime/bootstrap/bin/python" "uv==$uv_version"
    else
        python3 -m venv "$runtime/bootstrap"
        "$runtime/bootstrap/bin/python" -m pip install --disable-pip-version-check "uv==$uv_version"
    fi
fi
uv="$runtime/bootstrap/bin/uv"
if [[ ! -x "$runtime/venv/bin/python" ]]; then
    "$uv" venv --managed-python --python "$python_version" "$runtime/venv"
fi
pace_python="$runtime/venv/bin/python"
if [[ "$("$pace_python" -c 'import platform; print(platform.python_version())')" != "$python_version" ]]; then
    echo "Use a fresh PACE_RUNTIME_DIR or a venv with CPython $python_version." >&2
    exit 1
fi

build_constraints="$repo/requirements/kaggle-build-constraints.txt"
"$uv" pip sync --python "$pace_python" --build-constraint "$build_constraints" "${dry_run[@]}" "$repo/requirements/kaggle-linux-cuda12-py311.txt"
if [[ ${#dry_run[@]} -gt 0 ]]; then
    exit 0
fi
"$uv" pip install --python "$pace_python" --build-constraint "$build_constraints" --no-deps --editable "$repo"
"$uv" pip check --python "$pace_python"
printf 'PACE Python: %s\n' "$pace_python"
