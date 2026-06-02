from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa

from milvus_toolkit.core.manifest import validate_storage_v3_manifest
from milvus_toolkit.core.plans import SegmentReadTask
from milvus_toolkit.errors import StorageError, UnsupportedFeatureError
from milvus_toolkit.types import FieldSchema, StorageConfig


@dataclass(frozen=True)
class SegmentWriteResult:
    column_groups: list
    manifest_version: str | None


class StorageReader(Protocol):
    def read_segment_table(self, task: SegmentReadTask) -> pa.Table: ...


class StorageWriter(Protocol):
    def write_segment_table(
        self,
        table: pa.Table,
        schema: tuple[FieldSchema, ...],
        segment_path: str,
        mode: str = "append",
    ) -> SegmentWriteResult:
        ...


def create_storage_writer(storage: StorageConfig) -> StorageWriter:
    if storage.backend == "milvus_storage":
        return MilvusStorageWriter(storage)
    if storage.backend == "milvus_lite":
        return MilvusLiteStorageWriter(storage)
    raise UnsupportedFeatureError(
        "Unsupported storage backend "
        f"{storage.backend!r}; expected 'milvus_storage' or 'milvus_lite'"
    )


def create_storage_reader(storage: StorageConfig) -> StorageReader:
    if storage.backend == "milvus_storage":
        return MilvusStorageReader(storage)
    if storage.backend == "milvus_lite":
        return MilvusLiteStorageReader(storage)
    raise UnsupportedFeatureError(
        "Unsupported storage backend "
        f"{storage.backend!r}; expected 'milvus_storage' or 'milvus_lite'"
    )


class MilvusStorageReader:
    def __init__(self, storage: StorageConfig):
        self.storage = storage

    def read_segment_table(self, task: SegmentReadTask) -> pa.Table:
        validate_storage_v3_manifest(task.segment)
        milvus_storage = _load_milvus_storage()
        properties = _storage_properties(task.storage)
        schema = _to_pyarrow_schema(task.projected_fields)
        columns = [field.name for field in task.projected_fields]
        transaction_path = _transaction_path(task.manifest_path)

        try:
            if task.segment.raw.get("legacy_binlog_manifest"):
                return _read_legacy_binlog_segment(task, properties)
            else:
                with milvus_storage.Transaction(
                    transaction_path,
                    properties=properties,
                ) as transaction:
                    manifest = _get_manifest(
                        transaction,
                        task.manifest_version,
                    )
                column_groups = milvus_storage.ColumnGroups.from_list(manifest.column_groups)
            with milvus_storage.Reader(
                column_groups,
                schema,
                columns=columns,
                properties=properties,
            ) as reader:
                table = _scan_result_to_table(reader.scan(), schema=schema)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to read StorageV3 segment "
                f"{task.segment.segment_id} from {task.manifest_path}: {exc}"
            ) from exc

        return table.select(columns) if columns else table


class MilvusStorageWriter:
    def __init__(self, storage: StorageConfig):
        self.storage = storage

    def write_segment_table(
        self,
        table: pa.Table,
        schema: tuple[FieldSchema, ...],
        segment_path: str,
        mode: str = "append",
    ) -> SegmentWriteResult:
        milvus_storage = _load_milvus_storage()
        properties = _storage_properties(self.storage)
        arrow_schema = _to_pyarrow_schema(schema)
        table = _coerce_table_schema(table, arrow_schema)
        transaction_path = _transaction_path(segment_path)

        try:
            writer = milvus_storage.Writer(transaction_path, arrow_schema, properties)
            for batch in table.to_batches():
                writer.write(batch)
            column_groups = writer.close()
            transaction = milvus_storage.Transaction(transaction_path, properties)
            _write_transaction_changes(transaction, column_groups, schema, mode)
            manifest_version = _commit_manifest_version(transaction.commit())
            transaction.close()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to write StorageV3 segment to {segment_path}: {exc}"
            ) from exc
        return SegmentWriteResult(
            column_groups=list(column_groups),
            manifest_version=manifest_version,
        )


class MilvusLiteStorageReader:
    def __init__(self, storage: StorageConfig):
        self.storage = storage

    def read_segment_table(self, task: SegmentReadTask) -> pa.Table:
        if task.storage.root_path is None and "lite.path" not in task.storage.extra:
            raise StorageError(
                "Milvus Lite storage requires StorageConfig.root_path or extra['lite.path']"
            )
        raise UnsupportedFeatureError(
            "Milvus Lite storage reader is not implemented yet; wire the lite-storage "
            "API inside MilvusLiteStorageReader"
        )


class MilvusLiteStorageWriter:
    def __init__(self, storage: StorageConfig):
        self.storage = storage

    def write_segment_table(
        self,
        table: pa.Table,
        schema: tuple[FieldSchema, ...],
        segment_path: str,
        mode: str = "append",
    ) -> SegmentWriteResult:
        del table, schema, segment_path, mode
        if self.storage.root_path is None and "lite.path" not in self.storage.extra:
            raise StorageError(
                "Milvus Lite storage requires StorageConfig.root_path or extra['lite.path']"
            )
        raise UnsupportedFeatureError(
            "Milvus Lite storage writer is not implemented yet; wire the lite-storage "
            "API inside MilvusLiteStorageWriter"
        )


MilvusStorageAdapter = MilvusStorageReader


def _read_legacy_binlog_segment(task: SegmentReadTask, properties: dict[str, str]) -> pa.Table:
    if task.segment.manifest_path is None:
        raise StorageError("Legacy binlog snapshot segment must include manifest_path")
    arrays = {}
    row_count = 0
    for field in task.projected_fields:
        field_path = task.segment.raw.get("field_path_aliases", {}).get(
            str(field.field_id),
            str(field.field_id),
        )
        table = _read_legacy_binlog_field(
            task.segment.manifest_path,
            field_path,
            field,
            properties,
        )
        row_count = max(row_count, table.num_rows)
        arrays[field.name] = table[field.name]
    if not arrays:
        return pa.table({})
    return pa.table(arrays)



def _read_legacy_binlog_field(
    manifest_path: str,
    field_path: str,
    field: FieldSchema,
    properties: dict[str, str],
) -> pa.Table:
    import pyarrow.parquet as pq

    filesystem, path_prefix = _legacy_binlog_filesystem(properties)
    selector = pa.fs.FileSelector(
        f"{path_prefix}/{manifest_path}/{field_path}",
        recursive=False,
    )
    file_infos = [info for info in filesystem.get_file_info(selector) if info.is_file]
    if not file_infos:
        raise StorageError(
            f"Legacy binlog field {field.name} has no files at {manifest_path}/{field_path}"
        )
    tables = []
    for file_info in sorted(file_infos, key=lambda info: info.path):
        with filesystem.open_input_file(file_info.path) as file_obj:
            table = pq.read_table(file_obj)
        tables.append(_coerce_legacy_binlog_field_table(table, field))
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]



def _legacy_binlog_filesystem(properties: dict[str, str]):
    storage_type = properties.get("fs.storage_type")
    if storage_type == "local":
        return pa.fs.LocalFileSystem(), properties.get("fs.root_path", "")
    if storage_type == "remote":
        options = {
            "endpoint_override": properties.get("fs.address"),
            "scheme": "https" if properties.get("fs.use_ssl", "true") == "true" else "http",
        }
        if properties.get("fs.access_key_id") is not None:
            options["access_key"] = properties["fs.access_key_id"]
        if properties.get("fs.access_key_value") is not None:
            options["secret_key"] = properties["fs.access_key_value"]
        if properties.get("fs.region") is not None:
            options["region"] = properties["fs.region"]
        bucket = properties.get("fs.bucket_name")
        if bucket is None:
            raise StorageError("Remote legacy binlog storage requires fs.bucket_name")
        return pa.fs.S3FileSystem(**options), bucket
    raise StorageError(f"Unsupported legacy binlog storage type: {storage_type}")



def _coerce_legacy_binlog_field_table(table: pa.Table, field: FieldSchema) -> pa.Table:
    if field.name not in table.column_names:
        raise StorageError(f"Legacy binlog file missing column {field.name}")
    column = table[field.name]
    if field.data_type.replace("_", "").lower() == "floatvector":
        column = _fixed_size_binary_to_float_vector(column, field)
    else:
        column = column.cast(_to_pyarrow_type(field))
    return pa.table({field.name: column})



def _fixed_size_binary_to_float_vector(column: pa.ChunkedArray, field: FieldSchema) -> pa.Array:
    dim = int(field.params.get("dim", 0))
    if dim <= 0:
        raise StorageError(f"FloatVector field {field.name} requires dim param")
    values = []
    for value in column.to_pylist():
        if value is None:
            values.append(None)
        else:
            values.append(list(struct.unpack(f"<{dim}f", value)))
    return pa.array(values, type=pa.list_(pa.float32()))



def _legacy_binlog_column_groups(
    milvus_storage,
    manifest_path: str | None,
    fields,
    field_path_aliases: dict[str, str],
):
    if manifest_path is None:
        raise StorageError("Legacy binlog snapshot segment must include manifest_path")
    column_groups = []
    for field in fields:
        field_path = field_path_aliases.get(str(field.field_id), str(field.field_id))
        column_groups.append(
            milvus_storage.ColumnGroup(
                columns=[field.name],
                format="parquet",
                files=[
                    milvus_storage.ColumnGroupFile(
                        f"{manifest_path}/{field_path}/",
                        0,
                        2**63 - 1,
                    )
                ],
            )
        )
    return milvus_storage.ColumnGroups.from_list(column_groups)



def _load_milvus_storage():
    try:
        from milvus_toolkit._vendor import milvus_storage
    except ImportError as exc:
        raise StorageError(
            "Bundled milvus_storage is unavailable; run scripts/install_dev.sh "
            "or install a milvus-toolkit wheel built with python -m build --wheel"
        ) from exc
    return milvus_storage


def _storage_properties(storage: StorageConfig) -> dict[str, str]:
    properties = {"fs.storage_type": storage.storage_type}
    if storage.root_path is not None:
        properties["fs.root_path"] = storage.root_path
    if storage.endpoint is not None:
        properties["fs.address"] = storage.endpoint
    if storage.bucket is not None:
        properties["fs.bucket_name"] = storage.bucket
    if storage.access_key is not None:
        properties["fs.access_key_id"] = storage.access_key
    if storage.secret_key is not None:
        properties["fs.access_key_value"] = storage.secret_key
    if storage.region is not None:
        properties["fs.region"] = storage.region
    properties["fs.use_ssl"] = str(storage.use_ssl).lower()
    properties.update(storage.extra)
    return _normalize_storage_properties(properties)



def _normalize_storage_properties(properties: dict[str, str]) -> dict[str, str]:
    aliases = {
        "fs.endpoint": "fs.address",
        "fs.secret_access_key": "fs.access_key_value",
        "fs.secret_key": "fs.access_key_value",
    }
    normalized = dict(properties)
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized[alias]
    return {key: str(value) for key, value in normalized.items()}


def _coerce_table_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    if table.schema == schema:
        return table
    missing = [name for name in schema.names if name not in table.column_names]
    if missing:
        raise StorageError(f"Cannot write table missing column(s): {', '.join(missing)}")
    return table.select(schema.names).cast(schema)



def _write_transaction_changes(
    transaction,
    column_groups,
    schema: tuple[FieldSchema, ...],
    mode: str,
) -> None:
    if mode == "append":
        transaction.append_files(column_groups)
        return
    if mode == "addfield":
        for field in schema:
            transaction.drop_column(str(field.field_id))
        if hasattr(transaction, "add_column_groups"):
            transaction.add_column_groups(column_groups)
        else:
            column_groups_cls = type(column_groups)
            for column_group in _column_group_items(column_groups):
                _add_column_group(transaction, column_group, column_groups_cls)
        return
    raise StorageError(f"Unsupported StorageV3 write mode: {mode}")



def _column_group_items(column_groups):
    if hasattr(column_groups, "to_list"):
        return column_groups.to_list()
    return column_groups



def _add_column_group(transaction, column_group, column_groups_cls) -> None:
    try:
        transaction.add_column_group(column_group)
    except AttributeError as exc:
        if "metadata" not in str(exc):
            raise
        column_groups = column_groups_cls.from_list([column_group])
        transaction._lib.loon_transaction_add_column_group(
            transaction._handle,
            column_groups._get_c_ptr().column_group_array,
        )



def _commit_manifest_version(value) -> str | None:
    if value is None:
        return None
    version = value.version if hasattr(value, "version") else value
    if version is None:
        return None
    return str(version)



def _get_manifest(transaction, manifest_version: str | None):
    if manifest_version is None:
        return transaction.get_manifest()
    if hasattr(transaction, "get_manifest_at_version"):
        return transaction.get_manifest_at_version(_manifest_version_int(manifest_version))
    if hasattr(transaction, "get_manifest"):
        try:
            return transaction.get_manifest(version=_manifest_version_int(manifest_version))
        except TypeError:
            pass
    raise StorageError(
        "milvus-storage Transaction does not support reading a specific "
        f"manifest version: {manifest_version}"
    )



def _manifest_version_int(manifest_version: str) -> int:
    version = manifest_version[1:] if manifest_version.startswith("v") else manifest_version
    try:
        return int(version)
    except ValueError as exc:
        raise StorageError(
            f"Manifest version must be an integer, got {manifest_version!r}"
        ) from exc



def _scan_result_to_table(scan_result, schema: pa.Schema | None = None) -> pa.Table:
    if isinstance(scan_result, pa.Table):
        return scan_result

    if hasattr(scan_result, "read_all"):
        table = scan_result.read_all()
        if isinstance(table, pa.Table):
            return table
        raise StorageError(
            "milvus-storage Reader.scan().read_all() must return a pyarrow.Table, "
            f"got {type(table).__name__}"
        )

    if isinstance(scan_result, Iterable):
        batches = list(scan_result)
        if all(isinstance(batch, pa.RecordBatch) for batch in batches):
            return pa.Table.from_batches(batches, schema=schema)

    raise StorageError(
        "milvus-storage Reader.scan() must return a pyarrow.Table, an object with "
        f"read_all(), or an iterable of pyarrow.RecordBatch; got {type(scan_result).__name__}"
    )


def _to_pyarrow_schema(fields: tuple[FieldSchema, ...]) -> pa.Schema:
    return pa.schema([_to_pyarrow_field(field) for field in fields])


def _to_pyarrow_field(field: FieldSchema) -> pa.Field:
    metadata = {b"PARQUET:field_id": str(field.field_id).encode("utf-8")}
    return pa.field(
        field.name,
        _to_pyarrow_type(field),
        nullable=field.nullable,
        metadata=metadata,
    )


def _to_pyarrow_type(field: FieldSchema) -> pa.DataType:
    data_type = field.data_type.replace("_", "").lower()
    if data_type == "bool":
        return pa.bool_()
    if data_type == "int8":
        return pa.int8()
    if data_type == "int16":
        return pa.int16()
    if data_type == "int32":
        return pa.int32()
    if data_type == "int64":
        return pa.int64()
    if data_type == "float":
        return pa.float32()
    if data_type == "double":
        return pa.float64()
    if data_type in {"string", "varchar"}:
        return pa.string()
    if data_type == "floatvector":
        return _float_vector_type(field)
    raise UnsupportedFeatureError(
        f"Unsupported Milvus field type for field {field.name}: {field.data_type}"
    )


def _float_vector_type(field: FieldSchema) -> pa.DataType:
    return pa.list_(pa.float32())


def _transaction_path(manifest_path: str) -> str:
    return manifest_path
