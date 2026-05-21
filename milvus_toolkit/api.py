from __future__ import annotations

from collections.abc import Sequence

from milvus_toolkit.core.dataset import MilvusDataset
from milvus_toolkit.core.inspection import inspect_snapshot_metadata
from milvus_toolkit.core.planner import plan_snapshot_read
from milvus_toolkit.engines.local import execute_read_plan
from milvus_toolkit.io.object_store import load_snapshot_json
from milvus_toolkit.io.storage import create_storage_reader
from milvus_toolkit.types import InspectionResult, ReadOptions, StorageConfig


def read_snapshot(
    snapshot_path: str,
    storage: StorageConfig,
    columns: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
) -> MilvusDataset:
    options = _read_options(columns=columns, include=include)
    snapshot = load_snapshot_json(snapshot_path)
    plan = plan_snapshot_read(
        snapshot,
        storage=storage,
        columns=options.columns,
        include=options.include,
    )
    reader = create_storage_reader(storage)
    return MilvusDataset(read_plan=plan, _to_arrow=lambda: execute_read_plan(plan, reader))


def inspect_snapshot(snapshot_path: str, storage: StorageConfig) -> InspectionResult:
    del storage
    snapshot = load_snapshot_json(snapshot_path)
    return inspect_snapshot_metadata(snapshot)


def _read_options(
    columns: Sequence[str] | None,
    include: Sequence[str] | None,
) -> ReadOptions:
    return ReadOptions(
        columns=None if columns is None else tuple(columns),
        include=() if include is None else tuple(include),
    )
