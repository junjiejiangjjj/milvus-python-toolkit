from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pyarrow as pa

from milvus_toolkit.core.plans import SegmentReadTask
from milvus_toolkit.errors import StorageError, UnsupportedFeatureError
from milvus_toolkit.types import FieldSchema, StorageConfig


class StorageReader(Protocol):
    def read_segment_table(self, task: SegmentReadTask) -> pa.Table: ...


class StorageWriter(Protocol):
    pass


def create_storage_reader(storage: StorageConfig) -> StorageReader:
    return MilvusStorageReader(storage)


class MilvusStorageReader:
    def __init__(self, storage: StorageConfig):
        self.storage = storage

    def read_segment_table(self, task: SegmentReadTask) -> pa.Table:
        milvus_storage = _load_milvus_storage()
        properties = _storage_properties(task.storage)
        schema = _to_pyarrow_schema(task.projected_fields)
        columns = [field.name for field in task.projected_fields]
        transaction_path = _transaction_path(task.manifest_path)

        try:
            with milvus_storage.Transaction(transaction_path, properties=properties) as transaction:
                manifest = transaction.get_manifest()
            column_groups = milvus_storage.ColumnGroups.from_list(manifest.column_groups)
            with milvus_storage.Reader(
                column_groups,
                schema,
                columns=columns,
                properties=properties,
            ) as reader:
                table = _scan_result_to_table(reader.scan())
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to read StorageV3 segment "
                f"{task.segment.segment_id} from {task.manifest_path}: {exc}"
            ) from exc

        return table.select(columns) if columns else table


MilvusStorageAdapter = MilvusStorageReader


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
        properties["fs.endpoint"] = storage.endpoint
    if storage.bucket is not None:
        properties["fs.bucket_name"] = storage.bucket
    if storage.access_key is not None:
        properties["fs.access_key_id"] = storage.access_key
    if storage.secret_key is not None:
        properties["fs.secret_access_key"] = storage.secret_key
    if storage.region is not None:
        properties["fs.region"] = storage.region
    properties["fs.use_ssl"] = str(storage.use_ssl).lower()
    properties.update(storage.extra)
    return properties


def _scan_result_to_table(scan_result) -> pa.Table:
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
            return pa.Table.from_batches(batches)

    raise StorageError(
        "milvus-storage Reader.scan() must return a pyarrow.Table, an object with "
        f"read_all(), or an iterable of pyarrow.RecordBatch; got {type(scan_result).__name__}"
    )


def _to_pyarrow_schema(fields: tuple[FieldSchema, ...]) -> pa.Schema:
    return pa.schema([_to_pyarrow_field(field) for field in fields])


def _to_pyarrow_field(field: FieldSchema) -> pa.Field:
    return pa.field(field.name, _to_pyarrow_type(field), nullable=field.nullable)


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
    dim = field.params.get("dim")
    if dim is None:
        return pa.list_(pa.float32())
    return pa.list_(pa.float32(), int(dim))


def _transaction_path(manifest_path: str) -> str:
    return manifest_path
