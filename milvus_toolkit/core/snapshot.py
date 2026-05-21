from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from milvus_toolkit.errors import SnapshotError
from milvus_toolkit.types import SegmentMetadata

from .schema import parse_schema


@dataclass(frozen=True)
class SnapshotMetadata:
    collection_name: str | None
    schema_data: dict[str, Any]
    segments: tuple[SegmentMetadata, ...]
    raw: dict[str, Any]

    @property
    def schema(self):
        return parse_schema(
            {"collection_schema": self.schema_data, "collection_name": self.collection_name}
        )


def parse_snapshot(data: dict[str, Any]) -> SnapshotMetadata:
    if not isinstance(data, dict):
        raise SnapshotError("Snapshot JSON must be an object")

    collection = data.get("collection", {})
    if collection and not isinstance(collection, dict):
        raise SnapshotError("Snapshot collection must be an object")

    schema_data = data.get("collection_schema") or collection.get("schema")
    if not isinstance(schema_data, dict):
        raise SnapshotError("Snapshot must include collection_schema")

    collection_name = (
        data.get("collection_name") or collection.get("name") or schema_data.get("name")
    )
    segments = tuple(_parse_segments(data))
    return SnapshotMetadata(
        collection_name=collection_name,
        schema_data=schema_data,
        segments=segments,
        raw=data,
    )


def _parse_segments(data: dict[str, Any]) -> list[SegmentMetadata]:
    raw_segments = data.get("segments")
    if raw_segments is None:
        raw_segments = []
        for partition in data.get("partitions", []):
            if isinstance(partition, dict):
                partition_id = partition.get("partition_id")
                for segment in partition.get("segments", []):
                    if isinstance(segment, dict) and "partition_id" not in segment:
                        segment = {**segment, "partition_id": partition_id}
                    raw_segments.append(segment)

    if not isinstance(raw_segments, list):
        raise SnapshotError("Snapshot segments must be a list")

    segments = []
    for segment_data in raw_segments:
        if not isinstance(segment_data, dict):
            raise SnapshotError("Snapshot segment entries must be objects")
        segment_id = segment_data.get("segment_id", segment_data.get("id"))
        if segment_id is None:
            raise SnapshotError("Snapshot segment entries must include segment_id")

        manifest = segment_data.get("manifest", {})
        if manifest is not None and not isinstance(manifest, dict):
            raise SnapshotError("Snapshot segment manifest must be an object")

        segments.append(
            SegmentMetadata(
                segment_id=int(segment_id),
                partition_id=_optional_int(segment_data.get("partition_id")),
                row_count=_optional_int(segment_data.get("row_count")),
                storage_version=_optional_str(
                    segment_data.get("storage_version") or segment_data.get("storageVersion")
                ),
                manifest_path=_optional_str(
                    segment_data.get("manifest_path") or manifest.get("path")
                ),
                manifest_version=_optional_str(
                    segment_data.get("manifest_version") or manifest.get("version")
                ),
                raw=segment_data,
            )
        )
    return segments


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
