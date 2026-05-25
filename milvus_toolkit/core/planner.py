from __future__ import annotations

from typing import Any

from milvus_toolkit.errors import UnsupportedFeatureError
from milvus_toolkit.types import StorageConfig

from .plans import ReadPlan, SegmentReadTask
from .schema import project_fields
from .snapshot import SnapshotMetadata, parse_snapshot

_ALLOWED_INCLUDE = {"segment_id", "row_offset"}


def plan_snapshot_read(
    snapshot: SnapshotMetadata | dict[str, Any],
    storage: StorageConfig,
    columns: tuple[str, ...] | None = None,
    include: tuple[str, ...] = (),
) -> ReadPlan:
    snapshot_metadata = parse_snapshot(snapshot) if isinstance(snapshot, dict) else snapshot
    unsupported_include = sorted(set(include) - _ALLOWED_INCLUDE)
    if unsupported_include:
        raise UnsupportedFeatureError(
            f"Unsupported metadata column(s): {', '.join(unsupported_include)}"
        )

    schema = snapshot_metadata.schema
    projected_fields = project_fields(schema, columns)
    tasks = []
    for segment in snapshot_metadata.segments:
        tasks.append(
            SegmentReadTask(
                segment=segment,
                schema=schema,
                projected_fields=projected_fields,
                include=include,
                storage=storage,
            )
        )

    return ReadPlan(
        schema=schema,
        tasks=tuple(tasks),
        projected_fields=projected_fields,
        include=include,
        storage=storage,
    )
