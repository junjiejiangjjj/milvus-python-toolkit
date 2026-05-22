# Milvus Python Toolkit

Milvus Python Toolkit is an early Offline SDK for Milvus and Milvus Lite snapshot inspection and offline reads.

The current MVP focuses on this local, testable path:

```text
snapshot metadata -> read plan -> unified StorageReader -> PyArrow Table -> inspect CLI
```

## Install for development

`milvus-storage` is tracked as a Git submodule rather than consumed from PyPI. Use the project install script so the submodule is initialized, the native library is compiled, and `milvus_storage` is bundled inside the `milvus_toolkit._vendor` package:

```bash
scripts/install_dev.sh
```

Linux prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libaio-dev patchelf
python -m pip install "conan>=2" "build>=1"
```

A Rust toolchain is also required by the upstream native build. To pin a specific upstream revision, checkout the desired commit inside `third_party/milvus-storage`, then commit the parent repository's submodule pointer.

A bare `pip install "git+https://github.com/milvus-io/milvus-storage.git#subdirectory=python"` is not sufficient because it does not run `make -C cpp python-lib` or bundle the native library into the `milvus-toolkit` wheel.

To build the project wheel directly:

```bash
git submodule update --init --recursive
python -m build --wheel
```

## Run checks

```bash
python -m pytest
python -m ruff check .
```

## Inspect a snapshot

```bash
milvus-toolkit inspect \
  --snapshot tests/fixtures/snapshot_storage_v3.json \
  --s3-endpoint localhost:9000 \
  --s3-bucket bucket \
  --json
```

## Storage abstraction

The upper API, planner, dataset, and execution engines should not branch on Milvus vs Milvus Lite. Storage differences are isolated behind a unified `StorageReader` protocol with two implementations:

- `MilvusStorageReader`: wraps `milvus-storage` for Milvus StorageV3 data.
- `MilvusLiteStorageReader`: will read Milvus Lite local data through the lite-storage adapter.

## Python API

```python
import milvus_toolkit as mt

storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")
info = mt.inspect_snapshot("tests/fixtures/snapshot_storage_v3.json", storage=storage)
print(info.segment_count)
```

## Current non-goals

The MVP does not implement online Milvus reads, StorageV3 writes, backfill, Ray, Daft, Pandas, Polars, DuckDB, import/export, transform, validate, vector KNN, old packed parquet, or filter pushdown. The Milvus Lite storage adapter is still pending implementation.
