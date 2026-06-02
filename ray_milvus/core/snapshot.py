from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ray_milvus.errors import SnapshotError
from ray_milvus.types import SegmentMetadata

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


def build_snapshot_payload(
    schema: dict[str, Any],
    segments: dict[str, Any] | list[dict[str, Any]],
    collection_name: str | None = None,
) -> dict[str, Any]:
    schema_data = _canonical_schema_data(schema, collection_name)
    canonical_segments = _canonical_segments_data(segments)
    payload = {
        "collection_name": collection_name or schema_data.get("name"),
        "collection_schema": schema_data,
        "segments": canonical_segments,
    }
    parse_snapshot(payload)
    return payload


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
        if not raw_segments:
            _reject_storage_v2_manifest_segments(data)
            legacy_manifest_segments = _parse_legacy_manifest_segments(data)
            if legacy_manifest_segments:
                return legacy_manifest_segments

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
                raw={
                    **segment_data,
                    **(
                        {"packed_parquet_manifest": True}
                        if segment_data.get("packed_parquet_manifest")
                        else {}
                    ),
                },
            )
        )
    return segments


def _reject_storage_v2_manifest_segments(data: dict[str, Any]) -> None:
    raw_items = data.get("storagev2_manifest_list", data.get("storagev2-manifest-list"))
    if raw_items is not None:
        raise SnapshotError("Milvus StorageV2 snapshot manifests are not supported")


def _parse_legacy_manifest_segments(data: dict[str, Any]) -> list[SegmentMetadata]:
    manifest_paths = data.get("manifest_list")
    segment_ids = data.get("segment_ids")
    if manifest_paths is None:
        return []
    if not isinstance(manifest_paths, list):
        raise SnapshotError("Snapshot manifest_list must be a list")
    if segment_ids is not None and not isinstance(segment_ids, list):
        raise SnapshotError("Snapshot segment_ids must be a list")

    segments = []
    collection_id = _optional_str(data.get("snapshot_info", {}).get("collection_id"))
    partition_id = _first_string(data.get("snapshot_info", {}).get("partition_ids"))
    for index, manifest_path in enumerate(manifest_paths):
        segment_id = None if segment_ids is None else segment_ids[index]
        if segment_id is None:
            segment_id = _segment_id_from_manifest_path(str(manifest_path))
        if segment_id is None:
            raise SnapshotError("Snapshot manifest_list entries require matching segment_ids")
        segments.append(
            SegmentMetadata(
                segment_id=int(segment_id),
                partition_id=_optional_int(partition_id),
                row_count=None,
                storage_version="StorageV3",
                manifest_path=_transaction_path_from_manifest_path(
                    str(manifest_path),
                    collection_id=collection_id,
                    partition_id=partition_id,
                    segment_id=str(segment_id),
                ),
                manifest_version=None,
                raw={
                    "manifest_path": manifest_path,
                    "packed_parquet_manifest": True,
                },
            )
        )
    return segments



def _segment_id_from_manifest_path(manifest_path: str) -> str | None:
    stem = manifest_path.rsplit("/", 1)[-1].removesuffix(".avro")
    return stem or None



def _transaction_path_from_manifest_path(
    manifest_path: str,
    collection_id: str | None,
    partition_id: str | None,
    segment_id: str,
) -> str:
    parts = manifest_path.split("/")
    if collection_id is not None and partition_id is not None:
        try:
            snapshots_index = parts.index("snapshots")
            prefix = parts[:snapshots_index]
        except ValueError:
            prefix = []
        return "/".join([*prefix, "insert_log", collection_id, partition_id, segment_id])
    try:
        manifests_index = parts.index("manifests")
    except ValueError:
        return manifest_path
    if manifests_index < 2 or manifests_index + 2 >= len(parts):
        return manifest_path
    collection_id = parts[manifests_index - 1]
    return "/".join([*parts[: manifests_index - 2], "insert_log", collection_id, segment_id])



def _first_string(values: Any) -> str | None:
    if isinstance(values, list) and values:
        return str(values[0])
    if values is None:
        return None
    return str(values)



def _parse_manifest_content(value: Any, segment_id: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SnapshotError(
                f"Snapshot manifest for segment {segment_id} is not valid JSON"
            ) from exc
        if isinstance(decoded, dict):
            return decoded
    raise SnapshotError(f"Snapshot manifest for segment {segment_id} must be an object")


def _partition_id_from_base_path(base_path: str) -> int | None:
    parts = base_path.split("/")
    try:
        insert_log_index = parts.index("insert_log")
    except ValueError:
        return None
    if insert_log_index + 2 >= len(parts):
        return None
    return int(parts[insert_log_index + 2])



def _canonical_schema_data(
    schema: dict[str, Any],
    collection_name: str | None,
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise SnapshotError("Snapshot schema input must be an object")
    schema_data = dict(schema.get("collection_schema", schema))
    parsed = parse_schema(
        {"collection_schema": schema_data, "collection_name": collection_name}
    )
    return {
        "name": collection_name or parsed.collection_name,
        "fields": [
            {
                "name": field.name,
                "field_id": field.field_id,
                "data_type": field.data_type,
                "is_primary": field.is_primary,
                "nullable": field.nullable,
                "params": field.params,
            }
            for field in parsed.fields
        ],
    }


def _canonical_segments_data(
    segments: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(segments, dict):
        segments_data = segments.get("segments")
    else:
        segments_data = segments
    if not isinstance(segments_data, list):
        raise SnapshotError("Snapshot segments input must be a list or an object with segments")

    parsed_segments = _parse_segments({"segments": segments_data})
    return [
        {
            "segment_id": segment.segment_id,
            "partition_id": segment.partition_id,
            "row_count": segment.row_count,
            "storage_version": segment.storage_version,
            "manifest_path": segment.manifest_path,
            "manifest_version": segment.manifest_version,
            **(
                {"packed_parquet_manifest": True}
                if segment.raw.get("packed_parquet_manifest")
                else {}
            ),
        }
        for segment in parsed_segments
    ]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
