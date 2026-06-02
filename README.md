# ray-milvus

`ray-milvus` is the Python counterpart to `spark-milvus`. It focuses on offline Milvus data access first: snapshot inspection, StorageV3 reads/writes, backfill helpers, and a future lightweight Ray execution path.

The current MVP focuses on this local, testable path:

```text
snapshot metadata -> read plan -> unified StorageReader -> PyArrow RecordBatch/Table -> CLI / RayMilvus API
```

The Ray execution design keeps `read_snapshot(...)` engine-neutral. Users read locally with `MilvusDataset.iter_batches(batch_size=...)`; future Ray Core and Ray Data entry points submit worker-side read tasks and assemble Ray blocks internally:

```python
dataset = ray_milvus.read_snapshot(...)

for batch in dataset.iter_batches(batch_size=10_000):
    ...

blocks = dataset.to_ray_blocks(target_block_size="128MiB")      # Ray Core
ray_ds = dataset.to_ray_dataset(target_block_size="128MiB")     # Ray Data
```

Block assembly and storage read batch sizing are framework concerns. `target_block_size` and `parallelism` may be exposed as optional Ray tuning parameters, but users should not need to implement block construction themselves. See `docs/python-offline-toolkit-design-cn.md` for the detailed Ray Core / Ray Data strategy.

## Install for development

`milvus-storage` is tracked as a Git submodule rather than consumed from PyPI. Use the project install script so the submodule is initialized, the native library is compiled, and `milvus_storage` is bundled inside the `ray_milvus._vendor` package.

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

A bare `pip install "git+https://github.com/milvus-io/milvus-storage.git#subdirectory=python"` is not sufficient because it does not run `make -C cpp python-lib` or bundle the native library into the `ray-milvus` wheel.

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
ray-milvus inspect \
  --snapshot tests/fixtures/snapshot_storage_v3.json \
  --s3-endpoint localhost:9000 \
  --s3-bucket bucket \
  --json
```

## Storage abstraction

The upper API, planner, dataset, and execution path should not branch on Milvus vs Milvus Lite. Storage differences are isolated behind a unified `StorageReader` / `StorageWriter` protocol with two implementations:

- `MilvusStorageReader` / `MilvusStorageWriter`: wrap `milvus-storage` for Milvus StorageV3 data.
- `MilvusLiteStorageReader` / `MilvusLiteStorageWriter`: pending Milvus Lite local storage adapter.

## Python API

```python
import ray_milvus as rm

ray = rm.RayMilvus(storage=rm.StorageConfig(endpoint="localhost:9000", bucket="bucket"))
info = ray.inspect_snapshot("tests/fixtures/snapshot_storage_v3.json")
print(info.segment_count)
```

The functional API remains available:

```python
import ray_milvus as rm

storage = rm.StorageConfig(endpoint="localhost:9000", bucket="bucket")
info = rm.inspect_snapshot("tests/fixtures/snapshot_storage_v3.json", storage=storage)
```

## Current scope

The MVP supports snapshot inspection, snapshot JSON creation/import, StorageV3 manifest-backed local reads/writes through bundled `milvus-storage`, and local backfill semantics compatible with `spark-milvus` merge modes (`replace`, `coalesce`, `overwrite`). Ray execution, Milvus Lite storage, vector functions, old packed parquet, and filter pushdown are still pending.
