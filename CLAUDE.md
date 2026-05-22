# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository contains an initialized Python package skeleton for the Milvus Python Toolkit / Offline SDK MVP. The package currently provides public API, core planning/inspection models, a local execution skeleton, CLI wiring, fixtures, tests, and a `milvus-storage` backed `MilvusStorageReader`. Milvus Lite storage reads are not wired yet.

The current design direction is a Milvus Python Toolkit / Offline SDK rather than a Ray-specific connector. Use `docs/python-offline-toolkit-design-cn.md` as the primary architecture reference; `docs/python-connector-design-cn.md` is an earlier, more connector-focused design that was folded into the broader toolkit direction. Use `docs/mvp-plan-cn.md` for the immediate MVP scope.

## Commands

```bash
# Install package with internal bundled milvus_storage from the submodule
scripts/install_dev.sh

# Build the bundled project wheel
python -m build --wheel

# Fast install for unit tests without building native milvus-storage
python -m pip install "pyarrow>=17.0.0" "numpy>=1.20.0" "cffi>=1.17.1" "pandas>=2.3.3" "pytest>=7" "ruff>=0.4"
python -m pip install -e . --no-deps

# Run tests
python -m pytest

# Run lint
python -m ruff check .

# Check CLI help
milvus-toolkit --help
milvus-toolkit inspect --help

# Inspect the local StorageV3 fixture
milvus-toolkit inspect --snapshot tests/fixtures/snapshot_storage_v3.json --s3-endpoint localhost:9000 --s3-bucket bucket --json

# Inspect repository status
git status --short

# List tracked/untracked files
find . -maxdepth 3 -type f | sort
```

## High-level architecture direction

The intended package is a Milvus-focused Python toolkit for offline data access and processing. It should support snapshot inspection, StorageV3 manifest-backed segment reads/writes, Milvus Lite local storage access, schema and metadata handling, backfill, validation, import/export workflows, and execution adapters such as local, Ray, and Daft.

Recommended package shape from the current design:

```text
milvus_toolkit/
  api.py
  types.py
  errors.py
  core/
  io/
  ops/
  engines/
  cli/
  testing/
```

Primary layer responsibilities:

- `api.py`: stable public facade; normalize user parameters and dispatch to planners/execution engines.
- `types.py` / `errors.py`: public lightweight option/result types and unified exception hierarchy.
- `core/`: Milvus domain semantics, schema conversion, metadata, snapshot parsing, manifest parsing, dataset abstractions, logical plans, planner, and inspection models.
- `io/`: external system access, including Milvus client wrappers, object storage, unified StorageReader/StorageWriter protocols, `milvus-storage` and lite-storage adapters, readers, writers, and import/export helpers.
- `ops/`: higher-level offline operations such as transform, backfill, validate, migrate, and repair.
- `engines/`: execution backends such as local, Ray, and Daft. These execute plans and convert Arrow batches into backend-native datasets/dataframes.
- `cli/`: thin command-line wrapper around public API or ops; it should not bypass the core abstractions.
- `testing/`: reusable fixtures, assertions, and parity helpers for unit, integration, and Spark/Python parity tests.

Core principle:

```text
core defines Milvus semantics; io performs data access; ops defines offline task semantics; engines execute; api exposes stable user entry points.
```

## Dependency boundaries

Keep Milvus semantics engine-neutral:

```text
api -> core / io / ops / engines
ops -> core / io / engines
io -> core
engines -> core / io / ops
cli -> api
```

Avoid these dependencies:

```text
core    -/-> io
core    -/-> engines
core    -/-> ray / daft
io      -/-> engines
ops     -/-> engine-specific SDK unless isolated behind an optional engine path
cli     -/-> storage internals
```

`core/` should not depend on Ray, Daft, Pandas, Polars, DuckDB, or Spark. StorageV3 access, `milvus-storage` API details, and Milvus Lite local storage details should be isolated behind the `io` storage adapter layer. Upper layers should depend on a unified StorageReader/StorageWriter protocol rather than branching on Milvus vs Milvus Lite. `milvus-storage` must be consumed from the `third_party/milvus-storage` Git submodule and bundled into the `milvus-toolkit` wheel under `milvus_toolkit._vendor`; do not change it back to a plain PyPI dependency, direct Git dependency, top-level package, or separately installed `milvus_storage` wheel.

## Important domain constraints

- The Python toolkit/connector scope includes Milvus StorageV3 manifest-backed data paths and Milvus Lite local data paths behind the same storage reader abstraction. Old packed parquet storage is not in scope for Python connector support.
- Snapshot and collection reads should be planned as engine-neutral tasks before any execution backend is selected.
- Backfill merge modes must match the Spark implementation:
  - `replace`: parquet/backfill data is the full source of truth for target fields; unmatched source rows get null target values.
  - `coalesce`: keep non-null source values, fill source nulls from parquet values, keep unmatched source rows unchanged.
  - `overwrite`: matched parquet rows overwrite source values, including nulls; unmatched source rows keep source values.
- Before writing backfilled StorageV3 data, rows must be ordered by ascending `row_offset` within each `segment_id`.
- Engine adapters should not re-parse snapshots, redefine StorageV3/Lite storage semantics, redefine backfill merge modes, directly assemble StorageV3 write metadata, or call `milvus-storage` / lite-storage implementations directly.

## Suggested MVP order from the design docs

Start with a compact first milestone rather than implementing every adapter at once:

1. Establish package structure and public API facade.
2. Define config/options, schema models, metadata models, dataset abstractions, and plan/task dataclasses.
3. Implement snapshot JSON parsing and StorageV3 manifest-backed read planning.
4. Define the unified StorageReader protocol and factory.
5. Wrap `milvus-storage` Python APIs behind a Milvus storage adapter.
6. Add a Milvus Lite storage adapter behind the same protocol.
7. Implement local sequential read and `to_arrow` for baseline behavior.
8. Add snapshot inspection.
9. Add Ray read support.
10. Add backfill planning/execution with Spark-compatible `replace`, `coalesce`, and `overwrite` semantics.
11. Add Daft and other dataframe/query adapters after core semantics are stable.

## Documentation references

- `docs/python-offline-toolkit-design-cn.md`: current broader toolkit/offline SDK design and recommended architecture.
- `docs/python-connector-design-cn.md`: earlier connector-core design with detailed notes on StorageV3, engine adapters, backfill, inspection, and test strategy.
