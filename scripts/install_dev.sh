#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip
python -m pip install "build>=1" "pyarrow>=14" "pytest>=7" "ruff>=0.4" "conan>=2"
python scripts/build_milvus_storage.py

wheel="$(ls -t .deps/wheelhouse/milvus_storage-*.whl | head -n 1)"
python -m pip install "$wheel"
python -m pip install -e . --no-deps
