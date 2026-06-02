from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from ray_milvus.core.dataset import MilvusDataset
from ray_milvus.core.inspection import inspect_snapshot_metadata
from ray_milvus.core.native_snapshot import build_snapshot_payload_from_native_snapshot
from ray_milvus.core.planner import plan_snapshot_read
from ray_milvus.core.schema import parse_schema
from ray_milvus.core.snapshot import build_snapshot_payload, parse_snapshot
from ray_milvus.engines.local import execute_read_plan_batches
from ray_milvus.errors import ConfigError
from ray_milvus.io.milvus_service import MilvusService
from ray_milvus.io.object_store import load_snapshot_json, load_snapshot_json_from_storage
from ray_milvus.io.storage import create_storage_reader, create_storage_writer
from ray_milvus.types import FieldSchema, InspectionResult, MilvusConfig, ReadOptions, StorageConfig


class RayMilvus:
    """High-level client for Milvus snapshot workflows.

    RayMilvus keeps a default storage configuration and exposes convenience methods
    for reading snapshots, creating snapshot payloads, writing StorageV3 segments,
    importing native Milvus snapshots, and backfilling fields by primary key.

    Example:
        >>> from ray_milvus import RayMilvus
        >>> client = RayMilvus()
        >>> table = client.read_snapshot("snapshot.json").to_arrow()
    """

    def __init__(
        self,
        storage: StorageConfig | None = None,
        milvus: MilvusConfig | None = None,
    ):
        """Initialize the client.

        Args:
            storage: Default storage configuration used by methods that read or
                write segment data. When omitted, a local/default StorageConfig is used.
            milvus: Optional Milvus service connection used by methods that create
                or import snapshots from a running Milvus instance.

        Example:
            >>> from ray_milvus import MilvusConfig, RayMilvus
            >>> client = RayMilvus(milvus=MilvusConfig(uri="http://localhost:19530"))
        """
        self.storage = storage or StorageConfig()
        self.milvus = milvus

    def read_snapshot(
        self,
        snapshot_path: str,
        columns: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
        manifest_version: str | int | None = None,
    ) -> MilvusDataset:
        """Create a lazy dataset for reading rows from a snapshot.

        Args:
            snapshot_path: Path to a canonical snapshot JSON file.
            columns: Optional field names to project. When omitted, all fields are read.
            include: Optional metadata columns to add to the result, such as
                "segment_id" or "row_offset".
            manifest_version: Optional manifest version override for StorageV3 reads.

        Returns:
            A MilvusDataset for local batch iteration, debug materialization,
            or Ray Core block reads.

        Example:
            >>> client = RayMilvus()
            >>> dataset = client.read_snapshot(
            ...     "snapshot.json",
            ...     columns=["id", "vector"],
            ...     include=["segment_id"],
            ... )
            >>> table = dataset.to_arrow()
        """
        return read_snapshot(
            snapshot_path,
            storage=self.storage,
            columns=columns,
            include=include,
            manifest_version=manifest_version,
        )

    def inspect_snapshot(
        self,
        snapshot_path: str,
    ) -> InspectionResult:
        """Inspect whether snapshot metadata is readable by RayMilvus.

        Args:
            snapshot_path: Path to a canonical snapshot JSON file.

        Returns:
            An InspectionResult containing collection metadata, segment metadata,
            and diagnostics for unsupported segments or malformed manifests.

        Example:
            >>> client = RayMilvus()
            >>> result = client.inspect_snapshot("snapshot.json")
            >>> result.diagnostics
            ()
        """
        return inspect_snapshot(snapshot_path, storage=self.storage)

    def create_snapshot(
        self,
        schema: dict[str, Any] | str | Path,
        segments: dict[str, Any] | list[dict[str, Any]] | str | Path,
        output_path: str | Path | None = None,
        collection_name: str | None = None,
        overwrite: bool = False,
        pretty: bool = True,
    ) -> dict[str, Any]:
        """Build a canonical snapshot payload from schema and segment metadata.

        Args:
            schema: Collection schema as a dictionary or a path to a JSON file.
            segments: Segment metadata as a list, an object containing segments, or
                a path to a JSON file.
            output_path: Optional path where the generated snapshot JSON is written.
            collection_name: Optional collection name override.
            overwrite: Whether to replace output_path when it already exists.
            pretty: Whether to format output JSON with indentation.

        Returns:
            The canonical snapshot payload as a dictionary.

        Example:
            >>> client = RayMilvus()
            >>> snapshot = client.create_snapshot(
            ...     schema="schema.json",
            ...     segments="segments.json",
            ...     output_path="snapshot.json",
            ... )
        """
        return create_snapshot(
            schema,
            segments,
            output_path=output_path,
            collection_name=collection_name,
            overwrite=overwrite,
            pretty=pretty,
        )

    def create_snapshot_from_milvus(
        self,
        collection_name: str,
        snapshot_name: str | None = None,
        output_path: str | Path | None = None,
        auto_snapshot_name: bool = False,
        description: str | None = None,
        compaction_protection_seconds: int | None = None,
        overwrite: bool = False,
        pretty: bool = True,
    ) -> dict[str, Any]:
        """Create a Milvus snapshot and convert its exported metadata into a payload.

        Args:
            collection_name: Name of the source collection.
            snapshot_name: Name assigned to the Milvus snapshot. Required unless
                auto_snapshot_name is True.
            output_path: Optional path where the generated snapshot JSON is written.
            auto_snapshot_name: Whether to generate a unique snapshot name. Cannot be
                True when snapshot_name is set.
            description: Optional description passed to Milvus when creating the snapshot.
            compaction_protection_seconds: Optional compaction protection duration for
                the Milvus snapshot.
            overwrite: Whether to replace output_path when it already exists.
            pretty: Whether to format output JSON with indentation.

        Returns:
            The canonical snapshot payload as a dictionary.

        Example:
            >>> client = RayMilvus(milvus=MilvusConfig(uri="http://localhost:19530"))
            >>> snapshot = client.create_snapshot_from_milvus(
            ...     collection_name="demo",
            ...     snapshot_name="demo_snapshot",
            ...     output_path="snapshot.json",
            ... )
        """
        milvus = self._require_milvus_config()
        return create_snapshot_from_milvus(
            milvus.uri,
            collection_name,
            snapshot_name=snapshot_name,
            output_path=output_path,
            storage=self.storage,
            auto_snapshot_name=auto_snapshot_name,
            token=milvus.token,
            user=milvus.user,
            password=milvus.password,
            db_name=milvus.db_name,
            description=description,
            compaction_protection_seconds=compaction_protection_seconds,
            overwrite=overwrite,
            pretty=pretty,
        )

    def import_milvus_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        output_path: str | Path | None = None,
        overwrite: bool = False,
        pretty: bool = True,
    ) -> dict[str, Any]:
        """Import an existing Milvus snapshot by collection and snapshot name.

        Args:
            collection_name: Name of the source collection.
            snapshot_name: Name of an existing Milvus snapshot.
            output_path: Optional path where the generated snapshot JSON is written.
            overwrite: Whether to replace output_path when it already exists.
            pretty: Whether to format output JSON with indentation.

        Returns:
            The canonical snapshot payload as a dictionary.

        Example:
            >>> client = RayMilvus(milvus=MilvusConfig(uri="http://localhost:19530"))
            >>> snapshot = client.import_milvus_snapshot(
            ...     collection_name="demo",
            ...     snapshot_name="demo_snapshot",
            ... )
        """
        milvus = self._require_milvus_config()
        return import_milvus_snapshot(
            uri=milvus.uri,
            collection_name=collection_name,
            snapshot_name=snapshot_name,
            output_path=output_path,
            storage=self.storage,
            token=milvus.token,
            user=milvus.user,
            password=milvus.password,
            db_name=milvus.db_name,
            overwrite=overwrite,
            pretty=pretty,
        )

    def write_snapshot(
        self,
        table: pa.Table,
        schema: dict[str, Any],
        segment_path: str,
        segment_id: int,
        collection_name: str | None = None,
        partition_id: int | None = None,
        manifest_version: str | None = None,
        mode: str = "append",
    ) -> dict[str, Any]:
        """Write one segment and build a snapshot payload for it.

        Args:
            table: Arrow table containing rows to write.
            schema: Collection schema dictionary used for writing and snapshot creation.
            segment_path: Destination StorageV3 segment path.
            segment_id: Segment ID recorded in the snapshot.
            collection_name: Optional collection name override.
            partition_id: Optional partition ID recorded in the snapshot.
            manifest_version: Optional manifest version override recorded in the snapshot.
            mode: Write mode passed to the storage writer, such as "append".

        Returns:
            The canonical snapshot payload for the written segment.

        Example:
            >>> client = RayMilvus()
            >>> snapshot = client.write_snapshot(
            ...     table,
            ...     schema=schema,
            ...     segment_path="insert_log/1/2/3",
            ...     segment_id=3,
            ... )
        """
        return write_snapshot(
            table,
            schema,
            storage=self.storage,
            segment_path=segment_path,
            segment_id=segment_id,
            collection_name=collection_name,
            partition_id=partition_id,
            manifest_version=manifest_version,
            mode=mode,
        )

    def write_snapshot_segments(
        self,
        segments: Sequence[dict[str, Any]],
        schema: dict[str, Any],
        collection_name: str | None = None,
        mode: str = "append",
    ) -> dict[str, Any]:
        """Write multiple segment tables and build a snapshot payload for them.

        Args:
            segments: Segment write specs. Each item must include table, segment_path,
                and segment_id, and may include partition_id, manifest_version, or mode.
            schema: Collection schema dictionary used for writing and snapshot creation.
            collection_name: Optional collection name override.
            mode: Default write mode for segment specs that do not provide one.

        Returns:
            The canonical snapshot payload for the written segments.

        Example:
            >>> client = RayMilvus()
            >>> snapshot = client.write_snapshot_segments(
            ...     [
            ...         {"table": table_a, "segment_path": "insert_log/1/2/3", "segment_id": 3},
            ...         {"table": table_b, "segment_path": "insert_log/1/2/4", "segment_id": 4},
            ...     ],
            ...     schema=schema,
            ... )
        """
        return write_snapshot_segments(
            segments,
            schema,
            storage=self.storage,
            collection_name=collection_name,
            mode=mode,
        )

    def backfill_snapshot(
        self,
        snapshot_path: str,
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
        """Backfill selected fields into an existing snapshot by primary key.

        Args:
            snapshot_path: Path to the source canonical snapshot JSON file.
            backfill_table: Arrow table containing primary keys and replacement field values.
            schema: Collection schema dictionary for the output snapshot.
            primary_key: Field name used to match existing rows with backfill rows.
            fields: Field names to update from backfill_table.
            output_path: Optional path where the generated snapshot JSON is written.
            mode: Backfill write mode, such as "coalesce".
            segment_path_template: Template used to derive output segment paths from source
                segment metadata.
            overwrite: Whether to replace output_path when it already exists.
            pretty: Whether to format output JSON with indentation.

        Returns:
            The canonical snapshot payload for the backfilled data.

        Example:
            >>> client = RayMilvus()
            >>> snapshot = client.backfill_snapshot(
            ...     "snapshot.json",
            ...     backfill_table=updates,
            ...     schema=schema,
            ...     primary_key="id",
            ...     fields=["score"],
            ...     output_path="snapshot_backfilled.json",
            ... )
        """
        return backfill_snapshot(
            snapshot_path,
            storage=self.storage,
            backfill_table=backfill_table,
            schema=schema,
            primary_key=primary_key,
            fields=fields,
            output_path=output_path,
            mode=mode,
            segment_path_template=segment_path_template,
            overwrite=overwrite,
            pretty=pretty,
        )

    def _require_milvus_config(self) -> MilvusConfig:
        if self.milvus is None:
            raise ConfigError(
                "RayMilvus was not configured with MilvusConfig; "
                "pass milvus=MilvusConfig(uri=...) when constructing RayMilvus"
            )
        return self.milvus


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
    return MilvusDataset(
        read_plan=plan,
        _iter_batches=lambda batch_size: execute_read_plan_batches(
            plan,
            reader,
            batch_size=batch_size,
        ),
        _to_ray_blocks=lambda target_block_size, parallelism: _execute_ray_read_plan_blocks(
            plan,
            target_block_size=target_block_size,
            parallelism=parallelism,
        ),
    )


def _execute_ray_read_plan_blocks(
    plan,
    target_block_size: int | str | None = None,
    parallelism: int | None = None,
):
    from ray_milvus.engines.ray import execute_read_plan_blocks

    return execute_read_plan_blocks(
        plan,
        target_block_size=target_block_size,
        parallelism=parallelism,
    )


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
    snapshot_name: str | None = None,
    output_path: str | Path | None = None,
    storage: StorageConfig | None = None,
    auto_snapshot_name: bool = False,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    description: str | None = None,
    compaction_protection_seconds: int | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    resolved_snapshot_name = _resolve_milvus_snapshot_name(
        collection_name,
        snapshot_name=snapshot_name,
        auto_snapshot_name=auto_snapshot_name,
    )
    snapshot_location = MilvusService(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    ).create_snapshot_for_read(
        collection_name=collection_name,
        snapshot_name=resolved_snapshot_name,
        description=description,
        compaction_protection_seconds=compaction_protection_seconds,
    )
    return _build_snapshot_payload_from_milvus_location(
        snapshot_location,
        collection_name=collection_name,
        storage=storage,
        output_path=output_path,
        overwrite=overwrite,
        pretty=pretty,
    )



def _build_snapshot_payload_from_milvus_location(
    snapshot_location,
    collection_name: str,
    storage: StorageConfig | None,
    output_path: str | Path | None,
    overwrite: bool,
    pretty: bool,
) -> dict[str, Any]:
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
                    {"packed_parquet_manifest": True}
                    if segment.raw.get("packed_parquet_manifest")
                    else {}
                ),
            }
            for segment in snapshot.segments
        ],
        collection_name=collection_name or snapshot.collection_name,
    )
    payload["snapshot_name"] = snapshot_location.name
    if output_path is not None:
        _write_snapshot_payload(payload, Path(output_path), overwrite=overwrite, pretty=pretty)
    return payload



def _resolve_milvus_snapshot_name(
    collection_name: str,
    snapshot_name: str | None,
    auto_snapshot_name: bool,
) -> str:
    if snapshot_name is not None and auto_snapshot_name:
        raise ConfigError("snapshot_name cannot be set when auto_snapshot_name is True")
    if snapshot_name is not None:
        return snapshot_name
    if auto_snapshot_name:
        normalized_collection_name = "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in collection_name
        )
        return f"ray_milvus_{normalized_collection_name}_{uuid.uuid4().hex}"
    raise ConfigError("snapshot_name is required unless auto_snapshot_name is True")



def import_milvus_snapshot(
    uri: str,
    collection_name: str,
    snapshot_name: str,
    output_path: str | Path | None = None,
    storage: StorageConfig | None = None,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    overwrite: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    snapshot_location = MilvusService(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    ).describe_snapshot_for_read(
        collection_name=collection_name,
        snapshot_name=snapshot_name,
    )
    return _build_snapshot_payload_from_milvus_location(
        snapshot_location,
        collection_name=collection_name,
        storage=storage,
        output_path=output_path,
        overwrite=overwrite,
        pretty=pretty,
    )



def import_native_milvus_snapshot(
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


def _write_segment(
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
    collection_name: str | None = None,
    partition_id: int | None = None,
    manifest_version: str | None = None,
    mode: str = "append",
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
        collection_name=collection_name,
    )



def write_snapshot_segments(
    segments: Sequence[dict[str, Any]],
    schema: dict[str, Any],
    storage: StorageConfig,
    collection_name: str | None = None,
    mode: str = "append",
) -> dict[str, Any]:
    segment_payloads = [
        _write_segment(
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
        collection_name=collection_name,
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
    payload = write_snapshot_segments(
        segments,
        {"name": "backfill", "fields": [_field_to_dict(field) for field in target_schema]},
        storage,
        collection_name="backfill",
        mode="addfield",
    )
    if output_path is not None:
        _write_snapshot_payload(payload, Path(output_path), overwrite=overwrite, pretty=pretty)
    return payload



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
            from ray_milvus.errors import SnapshotError

            raise SnapshotError(f"Snapshot input file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            from ray_milvus.errors import SnapshotError

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
