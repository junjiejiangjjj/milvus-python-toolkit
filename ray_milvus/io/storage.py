from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa

from ray_milvus.core.manifest import validate_storage_v3_manifest
from ray_milvus.core.plans import SegmentReadTask
from ray_milvus.errors import StorageError, UnsupportedFeatureError
from ray_milvus.types import FieldSchema, StorageConfig


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
        batches = list(self.read_segment_batches(task))
        return pa.Table.from_batches(batches) if batches else pa.table({})

    def read_segment_batches(
        self,
        task: SegmentReadTask,
        batch_size: int | None = None,
    ) -> Iterable[pa.RecordBatch]:
        validate_storage_v3_manifest(task.segment)
        properties = _storage_properties(task.storage)

        try:
            if task.segment.raw.get("packed_parquet_manifest"):
                yield from _read_packed_parquet_segment_batches(
                    task,
                    properties,
                    batch_size=batch_size,
                )
                return

            milvus_storage = _load_milvus_storage()
            schema = _to_pyarrow_schema(task.projected_fields)
            columns = [field.name for field in task.projected_fields]
            transaction_path = _transaction_path(task.manifest_path)
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
                yield from _scan_result_to_batches(
                    reader.scan(),
                    schema=schema,
                    columns=columns,
                    batch_size=batch_size,
                )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to read StorageV3 segment "
                f"{task.segment.segment_id} from {task.manifest_path}: {exc}"
            ) from exc


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


def _read_packed_parquet_segment_batches(
    task: SegmentReadTask,
    properties: dict[str, str],
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    if task.segment.manifest_path is None:
        raise StorageError("Packed parquet snapshot segment must include manifest_path")
    filesystem, path_prefix = _packed_parquet_filesystem(properties)
    selector = pa.fs.FileSelector(
        _packed_parquet_segment_path(path_prefix, task.segment.manifest_path),
        recursive=True,
    )
    file_infos = sorted(
        (info for info in filesystem.get_file_info(selector) if info.is_file),
        key=lambda info: info.path,
    )
    if not file_infos:
        raise StorageError(
            f"Packed parquet segment {task.segment.segment_id} has no files at "
            f"{task.segment.manifest_path}"
        )

    if not task.projected_fields:
        return

    file_index = _packed_parquet_file_index(file_infos, filesystem)
    field_iterators = []
    for field in task.projected_fields:
        field_files = _packed_parquet_files_for_field(file_index, field)
        if not field_files:
            raise StorageError(
                f"Packed parquet segment {task.segment.segment_id} has no files for "
                f"field {field.name} ({field.field_id})"
            )
        field_iterators.append(
            (
                field,
                iter(
                    _read_packed_parquet_field_batches(
                        field_files,
                        filesystem,
                        field,
                        batch_size=batch_size,
                    )
                ),
            )
        )

    while True:
        batches = []
        for index, (field, iterator) in enumerate(field_iterators):
            try:
                batch = next(iterator)
            except StopIteration:
                if index == 0:
                    _ensure_packed_parquet_iterators_exhausted(field_iterators[1:])
                    return
                raise StorageError(
                    f"Packed parquet field {field.name} ended before other fields"
                ) from None
            batches.append(batch)
        yield _merge_packed_parquet_field_batches(batches)



def _ensure_packed_parquet_iterators_exhausted(field_iterators) -> None:
    for field, iterator in field_iterators:
        try:
            next(iterator)
        except StopIteration:
            continue
        raise StorageError(f"Packed parquet field {field.name} has extra rows")



def _merge_packed_parquet_field_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    row_count = batches[0].num_rows
    arrays = []
    fields = []
    for batch in batches:
        if batch.num_rows != row_count:
            raise StorageError("Packed parquet field batch row counts do not match")
        arrays.extend(batch.columns)
        fields.extend(batch.schema)
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))



def _packed_parquet_segment_path(path_prefix: str, manifest_path: str) -> str:
    if not path_prefix:
        return manifest_path
    return f"{path_prefix.rstrip('/')}/{manifest_path.lstrip('/')}"



def _packed_parquet_file_index(file_infos, filesystem):
    import pyarrow.parquet as pq

    field_id_files = {}
    field_name_files = {}
    for file_info in file_infos:
        with filesystem.open_input_file(file_info.path) as file_obj:
            schema = pq.ParquetFile(file_obj).schema_arrow
        for arrow_field in schema:
            field_name_files.setdefault(arrow_field.name, []).append(file_info)
            field_id = _parquet_field_id(arrow_field)
            if field_id is not None:
                field_id_files.setdefault(field_id, []).append(file_info)
    return field_id_files, field_name_files



def _packed_parquet_files_for_field(file_index, field: FieldSchema):
    field_id_files, field_name_files = file_index
    return field_id_files.get(str(field.field_id), field_name_files.get(field.name, []))



def _read_packed_parquet_field_batches(
    file_infos,
    filesystem,
    field: FieldSchema,
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    import pyarrow.parquet as pq

    for file_info in file_infos:
        with filesystem.open_input_file(file_info.path) as file_obj:
            parquet_file = pq.ParquetFile(file_obj)
            source_name = _packed_parquet_source_column(parquet_file.schema_arrow, field)
            if source_name is None:
                raise StorageError(
                    f"Packed parquet file missing field {field.name} ({field.field_id})"
                )
            iter_batches_kwargs = {"columns": [source_name]}
            if batch_size is not None:
                iter_batches_kwargs["batch_size"] = batch_size
            for batch in parquet_file.iter_batches(**iter_batches_kwargs):
                yield _coerce_packed_parquet_field_batch(batch, field)



def _parquet_field_id(field: pa.Field) -> str | None:
    metadata = field.metadata or {}
    field_id = metadata.get(b"PARQUET:field_id")
    return None if field_id is None else field_id.decode("utf-8")



def _packed_parquet_filesystem(properties: dict[str, str]):
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
            raise StorageError("Remote packed parquet storage requires fs.bucket_name")
        return pa.fs.S3FileSystem(**options), bucket
    raise StorageError(f"Unsupported packed parquet storage type: {storage_type}")



def _coerce_packed_parquet_field_batch(
    batch: pa.RecordBatch,
    field: FieldSchema,
) -> pa.RecordBatch:
    source_name = _packed_parquet_source_column(batch.schema, field)
    if source_name is None:
        raise StorageError(
            f"Packed parquet file missing field {field.name} ({field.field_id})"
        )
    column = batch.column(source_name)
    if field.data_type.replace("_", "").lower() == "floatvector":
        column = _fixed_size_binary_to_float_vector(column, field)
    else:
        column = column.cast(_to_pyarrow_type(field))
    return pa.RecordBatch.from_arrays([column], names=[field.name])



def _packed_parquet_source_column(schema: pa.Schema, field: FieldSchema) -> str | None:
    for arrow_field in schema:
        if _parquet_field_id(arrow_field) == str(field.field_id):
            return arrow_field.name
    return field.name if field.name in schema.names else None



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




def _load_milvus_storage():
    try:
        from ray_milvus._vendor import milvus_storage
    except ImportError as exc:
        raise StorageError(
            "Bundled milvus_storage is unavailable; run scripts/install_dev.sh "
            "or install a ray-milvus wheel built with python -m build --wheel"
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

    batches = list(_scan_result_to_batches(scan_result, schema=schema))
    return pa.Table.from_batches(batches, schema=schema) if batches else pa.table({})



def _scan_result_to_batches(
    scan_result,
    schema: pa.Schema | None = None,
    columns: list[str] | None = None,
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    if isinstance(scan_result, pa.Table):
        table = scan_result.select(columns) if columns else scan_result
        yield from table.to_batches(max_chunksize=batch_size)
        return

    if hasattr(scan_result, "read_all"):
        table = scan_result.read_all()
        if isinstance(table, pa.Table):
            table = table.select(columns) if columns else table
            yield from table.to_batches(max_chunksize=batch_size)
            return
        raise StorageError(
            "milvus-storage Reader.scan().read_all() must return a pyarrow.Table, "
            f"got {type(table).__name__}"
        )

    if isinstance(scan_result, Iterable):
        for batch in scan_result:
            if not isinstance(batch, pa.RecordBatch):
                raise StorageError(
                    "milvus-storage Reader.scan() iterable must yield pyarrow.RecordBatch, "
                    f"got {type(batch).__name__}"
                )
            table = pa.Table.from_batches([batch])
            if columns:
                table = table.select(columns)
            if schema is not None:
                table = _coerce_table_schema(table, schema)
            yield from table.to_batches(max_chunksize=batch_size)
        return

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
