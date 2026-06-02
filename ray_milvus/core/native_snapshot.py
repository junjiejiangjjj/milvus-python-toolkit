from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ray_milvus.core.snapshot import build_snapshot_payload
from ray_milvus.errors import ConfigError, SnapshotError

_SEGMENT_MANIFEST_AVRO_SCHEMA = {
    "type": "record",
    "name": "ManifestEntry",
    "fields": [
        {"name": "segment_id", "type": "long"},
        {"name": "partition_id", "type": "long"},
        {"name": "segment_level", "type": "long"},
        {"name": "channel_name", "type": "string"},
        {"name": "num_of_rows", "type": "long"},
        {
            "name": "start_position",
            "type": {
                "type": "record",
                "name": "AvroMsgPosition",
                "fields": [
                    {"name": "channel_name", "type": "string"},
                    {"name": "msg_id", "type": "bytes"},
                    {"name": "msg_group", "type": "string"},
                    {"name": "timestamp", "type": "long"},
                ],
            },
        },
        {"name": "dml_position", "type": "AvroMsgPosition"},
        {"name": "storage_version", "type": "long"},
        {"name": "is_sorted", "type": "boolean"},
        {
            "name": "binlog_files",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "AvroFieldBinlog",
                    "fields": [
                        {"name": "field_id", "type": "long"},
                        {
                            "name": "binlogs",
                            "type": {
                                "type": "array",
                                "items": {
                                    "type": "record",
                                    "name": "AvroBinlog",
                                    "fields": [
                                        {"name": "entries_num", "type": "long"},
                                        {"name": "timestamp_from", "type": "long"},
                                        {"name": "timestamp_to", "type": "long"},
                                        {"name": "log_path", "type": "string"},
                                        {"name": "log_size", "type": "long"},
                                        {"name": "log_id", "type": "long"},
                                        {"name": "memory_size", "type": "long"},
                                    ],
                                },
                            },
                        },
                    ],
                },
            },
        },
    ],
}


def build_snapshot_payload_from_native_snapshot(
    metadata_path: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    collection_id: str | int | None = None,
    snapshot_id: str | int | None = None,
) -> dict[str, Any]:
    metadata, resolved_manifest_dir = _resolve_native_snapshot_paths(
        metadata_path=metadata_path,
        manifest_dir=manifest_dir,
        snapshot_root=snapshot_root,
        collection_id=collection_id,
        snapshot_id=snapshot_id,
    )
    raw = _load_json(metadata)
    schema = _native_schema(raw)
    collection_name = _native_collection_name(raw, schema)
    segments = _native_segments(raw, resolved_manifest_dir)
    return build_snapshot_payload(schema, segments, collection_name=collection_name)


def _resolve_native_snapshot_paths(
    metadata_path: str | Path | None,
    manifest_dir: str | Path | None,
    snapshot_root: str | Path | None,
    collection_id: str | int | None,
    snapshot_id: str | int | None,
) -> tuple[Path, Path | None]:
    if metadata_path is not None:
        metadata = Path(metadata_path)
        resolved_manifest_dir = (
            Path(manifest_dir) if manifest_dir is not None else _infer_manifest_dir(metadata)
        )
        return metadata, resolved_manifest_dir

    if snapshot_root is None or collection_id is None or snapshot_id is None:
        raise SnapshotError(
            "Native snapshot import requires metadata_path or snapshot_root, "
            "collection_id, and snapshot_id"
        )

    root = Path(snapshot_root)
    metadata = root / str(collection_id) / "metadata" / f"{snapshot_id}.json"
    resolved_manifest_dir = (
        Path(manifest_dir)
        if manifest_dir is not None
        else root / str(collection_id) / "manifests" / str(snapshot_id)
    )
    return metadata, resolved_manifest_dir


def _infer_manifest_dir(metadata_path: Path) -> Path | None:
    if metadata_path.parent.name != "metadata":
        return None
    collection_dir = metadata_path.parent.parent
    return collection_dir / "manifests" / metadata_path.stem


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError as exc:
        raise SnapshotError(f"Native snapshot metadata file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Native snapshot metadata file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise SnapshotError("Native snapshot metadata must be an object")
    return data


def _native_schema(metadata: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (
        metadata.get("collection_schema"),
        metadata.get("collectionSchema"),
        metadata.get("schema"),
        _nested(metadata, "collection", "schema"),
        _nested(metadata, "collection", "collection_schema"),
        _nested(metadata, "collection_description", "schema"),
        _nested(metadata, "collectionDescription", "schema"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)
    raise SnapshotError("Native snapshot metadata must include collection schema")


def _native_collection_name(metadata: Mapping[str, Any], schema: Mapping[str, Any]) -> str | None:
    candidates = (
        metadata.get("collection_name"),
        metadata.get("collectionName"),
        _nested(metadata, "collection", "name"),
        _nested(metadata, "collection_description", "name"),
        _nested(metadata, "collectionDescription", "name"),
        schema.get("name"),
    )
    for candidate in candidates:
        if candidate is not None:
            return str(candidate)
    return None


def _native_segments(
    metadata: Mapping[str, Any],
    manifest_dir: Path | None,
) -> list[dict[str, Any]]:
    segments = _segments_from_metadata(metadata)
    if segments:
        return [
            _canonical_segment_data(segment, manifest_dir=manifest_dir)
            for segment in segments
        ]

    if manifest_dir is None:
        raise SnapshotError(
            "Native snapshot metadata must include segments or a manifest directory"
        )
    manifest_paths = sorted(manifest_dir.glob("*.avro")) if manifest_dir.exists() else []
    if not manifest_paths:
        raise SnapshotError(
            f"Native snapshot manifest directory contains no Avro files: {manifest_dir}"
        )
    return [
        _canonical_segment_data({"manifest_path": path}, manifest_dir=manifest_dir)
        for path in manifest_paths
    ]


def _segments_from_metadata(metadata: Mapping[str, Any]) -> list[Any]:
    candidates = (
        metadata.get("segments"),
        metadata.get("segment_manifests"),
        metadata.get("segmentManifests"),
        _nested(metadata, "segment_data", "segments"),
        _nested(metadata, "segmentData", "segments"),
    )
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
    return []


def _canonical_segment_data(segment: Any, manifest_dir: Path | None) -> dict[str, Any]:
    if isinstance(segment, (str, Path)):
        segment = {"manifest_path": segment}
    if not isinstance(segment, Mapping):
        raise SnapshotError("Native snapshot segment entries must be objects or manifest paths")

    manifest_path = _resolve_manifest_path(segment, manifest_dir)
    manifest_record = _read_manifest_record(manifest_path) if manifest_path is not None else {}
    combined = {**manifest_record, **segment}
    segment_id = _first(combined, "segment_id", "segmentID", "segmentId", "id")
    if segment_id is None and manifest_path is not None:
        segment_id = manifest_path.stem
    if segment_id is None:
        raise SnapshotError(
            "Native snapshot segment entries must include segment_id or manifest path"
        )

    storage_version = _first(combined, "storage_version", "storageVersion") or "StorageV3"
    canonical = {
        "segment_id": int(segment_id),
        "partition_id": _optional_int(
            _first(combined, "partition_id", "partitionID", "partitionId")
        ),
        "row_count": _optional_int(
            _first(combined, "row_count", "rowCount", "num_rows", "numRows", "num_of_rows")
        ),
        "storage_version": _storage_version_name(storage_version),
        "manifest_path": str(
            _first(combined, "transaction_path", "transactionPath") or manifest_path
        ),
        "manifest_version": _optional_str(
            _first(combined, "manifest_version", "manifestVersion", "version")
        ),
    }
    return canonical


def _resolve_manifest_path(segment: Mapping[str, Any], manifest_dir: Path | None) -> Path | None:
    value = _first(
        segment,
        "manifest_path",
        "manifestPath",
        "path",
        "file",
        "file_path",
        "filePath",
    )
    if value is None:
        segment_id = _first(segment, "segment_id", "segmentID", "segmentId", "id")
        if segment_id is not None and manifest_dir is not None:
            return manifest_dir / f"{segment_id}.avro"
        return None
    path = Path(value)
    if not path.is_absolute() and manifest_dir is not None and not path.exists():
        return manifest_dir / path
    return path


def _read_manifest_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        import fastavro
    except ImportError as exc:
        raise ConfigError(
            "fastavro is required to import Milvus native snapshot manifests. "
            "Install it with `pip install ray-milvus[native-snapshot]`."
        ) from exc

    with path.open("rb") as manifest_file:
        schemaless_error = None
        if hasattr(fastavro, "schemaless_reader"):
            try:
                record = fastavro.schemaless_reader(
                    manifest_file,
                    _SEGMENT_MANIFEST_AVRO_SCHEMA,
                )
                if not isinstance(record, Mapping):
                    raise SnapshotError(
                        f"Native snapshot schemaless manifest record must be an object: {path}"
                    )
                return _normalize_manifest_record(record, path)
            except Exception as exc:
                schemaless_error = exc
                manifest_file.seek(0)

        try:
            records = list(fastavro.reader(manifest_file))
        except Exception as exc:
            if schemaless_error is None:
                raise SnapshotError(
                    f"Failed to decode native snapshot Avro manifest {path}: {exc}"
                ) from exc
            raise SnapshotError(
                f"Failed to decode native snapshot Avro manifest {path} "
                f"as schemaless or OCF Avro: schemaless={schemaless_error}; ocf={exc}"
            ) from exc
    if not records:
        return {}
    if not isinstance(records[0], Mapping):
        raise SnapshotError(f"Native snapshot manifest record must be an object: {path}")
    return _normalize_manifest_record(records[0], path)



def _normalize_manifest_record(record: Mapping[str, Any], path: Path) -> dict[str, Any]:
    normalized = dict(record)
    _validate_binlog_files(normalized.get("binlog_files"), path)
    return normalized



def _validate_binlog_files(value: Any, path: Path) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise SnapshotError(f"Native snapshot manifest binlog_files must be a list: {path}")
    for field_binlog in value:
        if not isinstance(field_binlog, Mapping):
            raise SnapshotError(
                f"Native snapshot manifest binlog_files entries must be objects: {path}"
            )
        if _first(field_binlog, "field_id", "fieldID", "fieldId") is None:
            raise SnapshotError(
                f"Native snapshot manifest binlog_files entries must include field_id: {path}"
            )
        binlogs = field_binlog.get("binlogs")
        if binlogs is None:
            continue
        if not isinstance(binlogs, list):
            raise SnapshotError(
                f"Native snapshot manifest binlog_files binlogs must be a list: {path}"
            )
        for binlog in binlogs:
            if not isinstance(binlog, Mapping):
                raise SnapshotError(
                    f"Native snapshot manifest binlog entries must be objects: {path}"
                )
            if _first(binlog, "log_path", "logPath", "path") is None:
                raise SnapshotError(
                    f"Native snapshot manifest binlog entries must include log_path: {path}"
                )



def _storage_version_name(value: Any) -> str:
    versions = {
        3: "StorageV3",
        "3": "StorageV3",
    }
    return versions.get(value, str(value))


def _nested(data: Mapping[str, Any], first: str, second: str) -> Any:
    value = data.get(first)
    if isinstance(value, Mapping):
        return value.get(second)
    return None


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
