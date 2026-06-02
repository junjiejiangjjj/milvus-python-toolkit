from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from milvus_toolkit.core.dataset import MilvusDataset
from milvus_toolkit.core.inspection import inspect_snapshot_metadata
from milvus_toolkit.core.native_snapshot import build_snapshot_payload_from_native_snapshot
from milvus_toolkit.core.planner import plan_snapshot_read
from milvus_toolkit.core.schema import parse_schema
from milvus_toolkit.core.snapshot import build_snapshot_payload, parse_snapshot
from milvus_toolkit.engines.local import execute_read_plan
from milvus_toolkit.errors import ConfigError
from milvus_toolkit.io.milvus_client import create_snapshot_for_read, load_collection_schema
from milvus_toolkit.io.object_store import load_snapshot_json, load_snapshot_json_from_storage
from milvus_toolkit.io.storage import create_storage_reader, create_storage_writer
from milvus_toolkit.types import FieldSchema, InspectionResult, ReadOptions, StorageConfig


def read_snapshot(
    snapshot_path: str,
    storage: StorageConfig,
    columns: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    manifest_version: str | int | None = None,
) -> MilvusDataset:
    options = _read_options(
        columns=columns,
        include=include,
        manifest_version=manifest_version,
    )
    snapshot = load_snapshot_json(snapshot_path)
    plan = plan_snapshot_read(
        snapshot,
        storage=storage,
        columns=options.columns,
        include=options.include,
        manifest_version=options.manifest_version,
    )
    reader = create_storage_reader(storage)
    return MilvusDataset(read_plan=plan, _to_arrow=lambda: execute_read_plan(plan, reader))


def inspect_snapshot(snapshot_path: str, storage: StorageConfig) -> InspectionResult:
    del storage
    snapshot = load_snapshot_json(snapshot_path)
    return inspect_snapshot_metadata(snapshot)


def create_snapshot(
    schema: dict[str, Any] | str | Path,
    segments: dict[str, Any] | list[dict[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    collection_name: str | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    payload = build_snapshot_payload(
        _load_json_if_path(schema),
        _load_json_if_path(segments),
        collection_name=collection_name,
    )
    if output_path is not None:
        _write_snapshot_payload(payload, Path(output_path), overwrite=overwrite, pretty=pretty)
    return payload


def create_snapshot_from_milvus(
    uri: str,
    collection_name: str,
    segments: dict[str, Any] | list[dict[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    schema = load_collection_schema(
        uri=uri,
        collection_name=collection_name,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    )
    return create_snapshot(
        schema,
        segments,
        output_path=output_path,
        collection_name=collection_name,
        overwrite=overwrite,
        pretty=pretty,
    )


def create_snapshot_from_milvus_snapshot(
    uri: str,
    collection_name: str,
    snapshot_name: str,
    output_path: str | Path | None = None,
    storage: StorageConfig | None = None,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    description: str | None = None,
    compaction_protection_seconds: int | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    snapshot_location = create_snapshot_for_read(
        uri=uri,
        collection_name=collection_name,
        snapshot_name=snapshot_name,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
        description=description,
        compaction_protection_seconds=compaction_protection_seconds,
    )
    snapshot_data = _load_milvus_snapshot_json(snapshot_location.location, storage)
    snapshot = parse_snapshot(snapshot_data)
    payload = build_snapshot_payload(
        {"collection_schema": snapshot.schema_data, "collection_name": snapshot.collection_name},
        [
            {
                "segment_id": segment.segment_id,
                "partition_id": segment.partition_id,
                "row_count": segment.row_count,
                "storage_version": segment.storage_version,
                "manifest_path": segment.manifest_path,
                "manifest_version": segment.manifest_version,
                **(
                    {"legacy_binlog_manifest": True}
                    if segment.raw.get("legacy_binlog_manifest")
                    else {}
                ),
                **(
                    {"field_path_aliases": segment.raw.get("field_path_aliases")}
                    if segment.raw.get("field_path_aliases")
                    else {}
                ),
            }
            for segment in snapshot.segments
        ],
        collection_name=collection_name or snapshot.collection_name,
    )
    if output_path is not None:
        _write_snapshot_payload(payload, Path(output_path), overwrite=overwrite, pretty=pretty)
    return payload



def import_milvus_snapshot(
    metadata_path: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    collection_id: str | int | None = None,
    snapshot_id: str | int | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    payload = build_snapshot_payload_from_native_snapshot(
        metadata_path=metadata_path,
        manifest_dir=manifest_dir,
        snapshot_root=snapshot_root,
        collection_id=collection_id,
        snapshot_id=snapshot_id,
    )
    if output_path is not None:
        _write_snapshot_payload(payload, Path(output_path), overwrite=overwrite, pretty=pretty)
    return payload


def write_segment(
    table: pa.Table,
    schema: dict[str, Any] | Sequence[FieldSchema],
    storage: StorageConfig,
    segment_path: str,
    segment_id: int,
    partition_id: int | None = None,
    manifest_version: str | None = None,
    mode: str = "append",
) -> dict[str, Any]:
    fields = _schema_fields(schema)
    writer = create_storage_writer(storage)
    result = writer.write_segment_table(table, fields, segment_path, mode=mode)
    return {
        "segment_id": segment_id,
        "partition_id": partition_id,
        "row_count": table.num_rows,
        "storage_version": "StorageV3",
        "manifest_path": segment_path,
        "manifest_version": manifest_version or result.manifest_version,
    }


def write_snapshot(
    table: pa.Table,
    schema: dict[str, Any],
    storage: StorageConfig,
    segment_path: str,
    segment_id: int,
    output_path: str | Path | None = None,
    collection_name: str | None = None,
    partition_id: int | None = None,
    manifest_version: str | None = None,
    mode: str = "append",
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    return write_snapshot_segments(
        [
            {
                "table": table,
                "segment_path": segment_path,
                "segment_id": segment_id,
                "partition_id": partition_id,
                "manifest_version": manifest_version,
                "mode": mode,
            }
        ],
        schema,
        storage,
        output_path=output_path,
        collection_name=collection_name,
        overwrite=overwrite,
        pretty=pretty,
    )



def write_snapshot_segments(
    segments: Sequence[dict[str, Any]],
    schema: dict[str, Any],
    storage: StorageConfig,
    output_path: str | Path | None = None,
    collection_name: str | None = None,
    mode: str = "append",
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    segment_payloads = [
        write_segment(
            _required_segment_spec(segment, "table"),
            schema,
            storage,
            segment_path=_required_segment_spec(segment, "segment_path"),
            segment_id=int(_required_segment_spec(segment, "segment_id")),
            partition_id=_optional_segment_int(segment.get("partition_id")),
            manifest_version=_optional_segment_str(segment.get("manifest_version")),
            mode=_optional_segment_str(segment.get("mode")) or mode,
        )
        for segment in segments
    ]
    return create_snapshot(
        schema,
        segment_payloads,
        output_path=output_path,
        collection_name=collection_name,
        overwrite=overwrite,
        pretty=pretty,
    )



def backfill_snapshot(
    snapshot_path: str,
    storage: StorageConfig,
    backfill_table: pa.Table,
    schema: dict[str, Any],
    primary_key: str,
    fields: Sequence[str],
    output_path: str | Path | None = None,
    mode: str = "coalesce",
    segment_path_template: str = "{manifest_path}",
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    _validate_backfill_mode(mode)
    source = _source_with_segment_metadata(
        snapshot_path,
        read_snapshot(
            snapshot_path,
            storage=storage,
            include=("segment_id", "row_offset"),
        ).to_arrow(),
    )
    target_schema = _target_schema(schema, fields)
    segments = _backfill_segments(
        source=source,
        backfill_table=backfill_table,
        target_schema=target_schema,
        primary_key=primary_key,
        fields=fields,
        mode=mode,
        segment_path_template=segment_path_template,
    )
    return write_snapshot_segments(
        segments,
        {"name": "backfill", "fields": [_field_to_dict(field) for field in target_schema]},
        storage,
        output_path=output_path,
        collection_name="backfill",
        mode="addfield",
        overwrite=overwrite,
        pretty=pretty,
    )



def _load_milvus_snapshot_json(
    snapshot_location: str,
    storage: StorageConfig | None,
) -> dict[str, Any]:
    if storage is None or snapshot_location.startswith(("s3://", "gs://", "az://")):
        return load_snapshot_json(snapshot_location)
    return load_snapshot_json_from_storage(
        snapshot_location,
        storage_type=storage.storage_type,
        endpoint=storage.endpoint,
        bucket=storage.bucket,
        access_key=storage.access_key,
        secret_key=storage.secret_key,
        use_ssl=storage.use_ssl,
        region=storage.region,
    )



def _schema_fields(schema: dict[str, Any] | Sequence[FieldSchema]) -> tuple[FieldSchema, ...]:
    if isinstance(schema, dict):
        return parse_schema(schema).fields
    return tuple(schema)



def _target_schema(schema: dict[str, Any], fields: Sequence[str]) -> tuple[FieldSchema, ...]:
    parsed = parse_schema(schema)
    target_fields = []
    for field_name in fields:
        field = parsed.field_by_name(field_name)
        if field is None:
            raise ConfigError(f"Backfill field not found in schema: {field_name}")
        target_fields.append(field)
    return tuple(target_fields)



def _validate_backfill_mode(mode: str) -> None:
    if mode not in {"replace", "coalesce", "overwrite"}:
        raise ConfigError(f"Unsupported backfill mode: {mode}")



def _source_with_segment_metadata(snapshot_path: str, source: pa.Table) -> pa.Table:
    snapshot = load_snapshot_json(snapshot_path)
    metadata_by_segment = {
        task.segment.segment_id: task.segment
        for task in plan_snapshot_read(snapshot, storage=StorageConfig()).tasks
    }
    manifest_paths = []
    partition_ids = []
    for segment_id in source["segment_id"].to_pylist():
        segment = metadata_by_segment.get(int(segment_id))
        manifest_paths.append(None if segment is None else segment.manifest_path)
        partition_ids.append(None if segment is None else segment.partition_id)
    if "manifest_path" not in source.column_names:
        source = source.append_column("manifest_path", pa.array(manifest_paths, type=pa.string()))
    if "partition_id" not in source.column_names:
        source = source.append_column("partition_id", pa.array(partition_ids, type=pa.int64()))
    return source



def _backfill_segments(
    source: pa.Table,
    backfill_table: pa.Table,
    target_schema: tuple[FieldSchema, ...],
    primary_key: str,
    fields: Sequence[str],
    mode: str,
    segment_path_template: str,
) -> list[dict[str, Any]]:
    _validate_backfill_inputs(source, backfill_table, primary_key, fields)
    backfill_rows = {
        row[primary_key]: row
        for row in backfill_table.select([primary_key, *fields]).to_pylist()
    }
    rows_by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in source.to_pylist():
        segment_id = int(row["segment_id"])
        rows_by_segment.setdefault(segment_id, []).append(row)

    segments = []
    for segment_id, rows in rows_by_segment.items():
        ordered_rows = sorted(rows, key=lambda row: row["row_offset"])
        payload = {
            field.name: [
                _backfill_value(row, backfill_rows.get(row[primary_key]), field.name, mode)
                for row in ordered_rows
            ]
            for field in target_schema
        }
        segments.append(
            {
                "table": pa.table(payload),
                "segment_path": segment_path_template.format(
                    segment_id=segment_id,
                    manifest_path=ordered_rows[0].get("manifest_path", f"segments/{segment_id}"),
                ),
                "segment_id": segment_id,
                "partition_id": ordered_rows[0].get("partition_id"),
                "mode": "addfield",
            }
        )
    return segments



def _validate_backfill_inputs(
    source: pa.Table,
    backfill_table: pa.Table,
    primary_key: str,
    fields: Sequence[str],
) -> None:
    required_source = {primary_key, "segment_id", "row_offset"}
    missing_source = sorted(required_source - set(source.column_names))
    if missing_source:
        raise ConfigError(f"Source table missing column(s): {', '.join(missing_source)}")
    required_backfill = {primary_key, *fields}
    missing_backfill = sorted(required_backfill - set(backfill_table.column_names))
    if missing_backfill:
        raise ConfigError(f"Backfill table missing column(s): {', '.join(missing_backfill)}")



def _backfill_value(
    source_row: dict[str, Any],
    backfill_row: dict[str, Any] | None,
    field: str,
    mode: str,
):
    if mode == "replace":
        return None if backfill_row is None else backfill_row[field]
    if mode == "coalesce":
        source_value = source_row.get(field)
        if source_value is not None:
            return source_value
        return None if backfill_row is None else backfill_row[field]
    if mode == "overwrite":
        if backfill_row is not None:
            return backfill_row[field]
        return source_row.get(field)
    raise ConfigError(f"Unsupported backfill mode: {mode}")



def _field_to_dict(field: FieldSchema) -> dict[str, Any]:
    return {
        "name": field.name,
        "field_id": field.field_id,
        "data_type": field.data_type,
        "is_primary": field.is_primary,
        "nullable": field.nullable,
        "params": field.params,
    }



def _required_segment_spec(segment: dict[str, Any], key: str):
    if key not in segment:
        raise ConfigError(f"Segment spec must include {key}")
    return segment[key]



def _optional_segment_int(value: Any) -> int | None:
    return None if value is None else int(value)



def _optional_segment_str(value: Any) -> str | None:
    return None if value is None else str(value)



def _read_options(
    columns: Sequence[str] | None,
    include: Sequence[str] | None,
    manifest_version: str | int | None,
) -> ReadOptions:
    return ReadOptions(
        columns=None if columns is None else tuple(columns),
        include=() if include is None else tuple(include),
        manifest_version=None if manifest_version is None else str(manifest_version),
    )


def _load_json_if_path(value):
    if isinstance(value, str | Path):
        path = Path(value)
        try:
            with path.open(encoding="utf-8") as json_file:
                return json.load(json_file)
        except FileNotFoundError as exc:
            from milvus_toolkit.errors import SnapshotError

            raise SnapshotError(f"Snapshot input file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            from milvus_toolkit.errors import SnapshotError

            raise SnapshotError(f"Snapshot input file is not valid JSON: {path}") from exc
    return value


def _write_snapshot_payload(
    payload: dict[str, Any],
    output_path: Path,
    overwrite: bool,
    pretty: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise ConfigError(f"Snapshot output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    output_path.write_text(f"{text}\n", encoding="utf-8")
