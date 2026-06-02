from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ray_milvus.core.plans import SegmentReadTask
from ray_milvus.errors import (
    ManifestError,
    StorageError,
    UnsupportedFeatureError,
    UnsupportedSegmentError,
)
from ray_milvus.io.storage import (
    MilvusLiteStorageReader,
    MilvusStorageReader,
    MilvusStorageWriter,
    SegmentWriteResult,
    _coerce_table_schema,
    _scan_result_to_table,
    _storage_properties,
    _to_pyarrow_schema,
    create_storage_reader,
    create_storage_writer,
)
from ray_milvus.types import FieldSchema, MilvusSchema, SegmentMetadata, StorageConfig


def _segment_read_task(
    storage: StorageConfig | None = None,
    storage_version: str | None = "StorageV3",
    manifest_path: str | None = "segments/10/manifest.json",
) -> SegmentReadTask:
    return SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=10,
            partition_id=1,
            row_count=2,
            storage_version=storage_version,
            manifest_path=manifest_path,
            manifest_version="v1",
        ),
        schema=MilvusSchema(
            collection_name="demo",
            fields=(FieldSchema("id", 100, "Int64"),),
        ),
        projected_fields=(FieldSchema("id", 100, "Int64"),),
        include=(),
        storage=storage or StorageConfig(storage_type="local", root_path="/data/root"),
    )


def test_create_storage_reader_dispatches_milvus_storage():
    reader = create_storage_reader(StorageConfig(backend="milvus_storage"))

    assert isinstance(reader, MilvusStorageReader)


def test_create_storage_reader_dispatches_milvus_lite():
    reader = create_storage_reader(StorageConfig(backend="milvus_lite"))

    assert isinstance(reader, MilvusLiteStorageReader)


def test_create_storage_reader_rejects_unknown_backend():
    with pytest.raises(UnsupportedFeatureError, match="unknown"):
        create_storage_reader(StorageConfig(backend="unknown"))


def test_create_storage_writer_dispatches_milvus_storage():
    writer = create_storage_writer(StorageConfig(backend="milvus_storage"))

    assert isinstance(writer, MilvusStorageWriter)


def test_create_storage_writer_rejects_unknown_backend():
    with pytest.raises(UnsupportedFeatureError, match="unknown"):
        create_storage_writer(StorageConfig(backend="unknown"))


def test_storage_properties_for_local_storage():
    properties = _storage_properties(
        StorageConfig(storage_type="local", root_path="/tmp/milvus-storage")
    )

    assert properties["fs.storage_type"] == "local"
    assert properties["fs.root_path"] == "/tmp/milvus-storage"
    assert properties["fs.use_ssl"] == "true"


def test_storage_properties_for_object_storage_and_extra_overrides():
    properties = _storage_properties(
        StorageConfig(
            storage_type="s3",
            endpoint="localhost:9000",
            bucket="bucket",
            access_key="ak",
            secret_key="sk",
            use_ssl=False,
            region="us-east-1",
            root_path="wrong",
            extra={"fs.root_path": "right", "custom": "value"},
        )
    )

    assert properties == {
        "fs.storage_type": "s3",
        "fs.address": "localhost:9000",
        "fs.bucket_name": "bucket",
        "fs.access_key_id": "ak",
        "fs.access_key_value": "sk",
        "fs.use_ssl": "false",
        "fs.region": "us-east-1",
        "fs.root_path": "right",
        "custom": "value",
    }


def test_storage_properties_accept_spark_milvus_extra_keys():
    properties = _storage_properties(
        StorageConfig(
            storage_type="remote",
            endpoint="s3.us-west-2.amazonaws.com",
            bucket="bucket",
            use_ssl=True,
            extra={
                "fs.use_iam": "true",
                "fs.use_virtual_host": "true",
                "fs.cloud_provider": "aws",
                "fs.iam_endpoint": "iam.amazonaws.com",
                "fs.request_timeout_ms": "30000",
                "fs.gcp_native_without_auth": "false",
                "fs.gcp_credential_json": "{}",
                "fs.use_custom_part_upload": "true",
            },
        )
    )

    assert properties["fs.address"] == "s3.us-west-2.amazonaws.com"
    assert properties["fs.use_iam"] == "true"
    assert properties["fs.use_virtual_host"] == "true"
    assert properties["fs.cloud_provider"] == "aws"
    assert properties["fs.iam_endpoint"] == "iam.amazonaws.com"
    assert properties["fs.request_timeout_ms"] == "30000"
    assert properties["fs.gcp_native_without_auth"] == "false"
    assert properties["fs.gcp_credential_json"] == "{}"
    assert properties["fs.use_custom_part_upload"] == "true"



def test_storage_properties_normalizes_legacy_aliases():
    properties = _storage_properties(
        StorageConfig(
            extra={
                "fs.endpoint": "localhost:9000",
                "fs.secret_access_key": "sk",
            }
        )
    )

    assert properties["fs.address"] == "localhost:9000"
    assert properties["fs.secret_access_key"] == "sk"
    assert properties["fs.access_key_value"] == "sk"



def test_to_pyarrow_schema_maps_mvp_field_types():
    schema = _to_pyarrow_schema(
        (
            FieldSchema("id", 100, "Int64", nullable=False),
            FieldSchema("name", 101, "VarChar"),
            FieldSchema("vector", 102, "FloatVector", params={"dim": "2"}),
        )
    )

    assert schema.field("id").type == pa.int64()
    assert schema.field("id").nullable is False
    assert schema.field("id").metadata == {b"PARQUET:field_id": b"100"}
    assert schema.field("name").type == pa.string()
    assert schema.field("name").metadata == {b"PARQUET:field_id": b"101"}
    assert schema.field("vector").type == pa.list_(pa.float32())
    assert schema.field("vector").metadata == {b"PARQUET:field_id": b"102"}


def test_coerce_table_schema_selects_and_casts_columns():
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False, metadata={b"PARQUET:field_id": b"100"}),
            pa.field("name", pa.string(), metadata={b"PARQUET:field_id": b"101"}),
        ]
    )
    table = pa.table({"name": ["a"], "id": pa.array([1], type=pa.int32()), "extra": ["x"]})

    result = _coerce_table_schema(table, schema)

    assert result.schema == schema
    assert result.to_pydict() == {"id": [1], "name": ["a"]}


def test_coerce_table_schema_reports_missing_columns():
    schema = pa.schema([pa.field("id", pa.int64())])
    table = pa.table({"name": ["a"]})

    with pytest.raises(StorageError, match="missing column"):
        _coerce_table_schema(table, schema)


def test_scan_result_to_table_accepts_table():
    table = pa.table({"id": [1]})

    assert _scan_result_to_table(table) is table


def test_scan_result_to_table_accepts_read_all_object():
    table = pa.table({"id": [1]})

    class FakeBatchReader:
        def read_all(self):
            return table

    assert _scan_result_to_table(FakeBatchReader()) is table


def test_scan_result_to_table_accepts_record_batch_iterable():
    schema = pa.schema([pa.field("id", pa.int64())])
    batches = [pa.RecordBatch.from_pydict({"id": [1, 2]}, schema=schema)]

    table = _scan_result_to_table(batches, schema=schema)

    assert table.schema == schema
    assert table.to_pydict() == {"id": [1, 2]}


def test_scan_result_to_table_rejects_unexpected_read_all_result():
    class FakeBatchReader:
        def read_all(self):
            return "not a table"

    with pytest.raises(StorageError, match="got str"):
        _scan_result_to_table(FakeBatchReader())


def test_to_pyarrow_schema_reports_unsupported_field_name():
    with pytest.raises(UnsupportedFeatureError, match="payload: JSON"):
        _to_pyarrow_schema((FieldSchema("payload", 200, "JSON"),))


def test_milvus_storage_reader_uses_transaction_manifest_and_reader(monkeypatch):
    calls = {}

    class FakeManifest:
        column_groups = ["group"]

    class FakeTransaction:
        def __init__(self, path, properties):
            calls["transaction"] = {"path": path, "properties": properties}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get_manifest(self):
            calls["manifest_version"] = None
            return FakeManifest()

    class FakeColumnGroups:
        @staticmethod
        def from_list(column_groups):
            calls["column_groups"] = column_groups
            return "ffi-column-groups"

    class FakeBatchReader:
        def read_all(self):
            return pa.table({"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]})

    class FakeReader:
        def __init__(self, column_groups, schema, columns, properties):
            calls["reader"] = {
                "column_groups": column_groups,
                "schema": schema,
                "columns": columns,
                "properties": properties,
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def scan(self):
            return FakeBatchReader()

    fake_milvus_storage = SimpleNamespace(
        Transaction=FakeTransaction,
        ColumnGroups=FakeColumnGroups,
        Reader=FakeReader,
    )
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    storage = StorageConfig(storage_type="local", root_path="/data/root")
    task = SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=10,
            partition_id=1,
            row_count=2,
            storage_version="StorageV3",
            manifest_path="segments/10/manifest.json",
            manifest_version="v1",
        ),
        schema=MilvusSchema(
            collection_name="demo",
            fields=(
                FieldSchema("id", 100, "Int64"),
                FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
            ),
        ),
        projected_fields=(
            FieldSchema("id", 100, "Int64"),
            FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
        ),
        include=(),
        storage=storage,
    )

    table = MilvusStorageReader(storage).read_segment_table(task)

    assert calls["transaction"] == {
        "path": "segments/10/manifest.json",
        "properties": {
            "fs.storage_type": "local",
            "fs.root_path": "/data/root",
            "fs.use_ssl": "true",
        },
    }
    assert calls["manifest_version"] is None
    assert calls["column_groups"] == ["group"]
    assert calls["reader"]["column_groups"] == "ffi-column-groups"
    assert calls["reader"]["columns"] == ["id", "vector"]
    assert calls["reader"]["schema"].field("id").type == pa.int64()
    assert calls["reader"]["schema"].field("id").metadata == {
        b"PARQUET:field_id": b"100"
    }
    assert table.to_pydict() == {"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]}


def test_milvus_storage_reader_streams_record_batches(monkeypatch):
    calls = {}
    batches = [
        pa.RecordBatch.from_pydict({"id": [1], "vector": [[0.1, 0.2]]}),
        pa.RecordBatch.from_pydict({"id": [2], "vector": [[0.3, 0.4]]}),
    ]

    class FakeManifest:
        column_groups = ["group"]

    class FakeTransaction:
        def __init__(self, path, properties):
            calls["transaction"] = {"path": path, "properties": properties}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get_manifest(self):
            return FakeManifest()

    class FakeColumnGroups:
        @staticmethod
        def from_list(column_groups):
            calls["column_groups"] = column_groups
            return "ffi-column-groups"

    class FakeReader:
        def __init__(self, column_groups, schema, columns, properties):
            calls["reader"] = {"columns": columns}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def scan(self):
            return iter(batches)

    fake_milvus_storage = SimpleNamespace(
        Transaction=FakeTransaction,
        ColumnGroups=FakeColumnGroups,
        Reader=FakeReader,
    )
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    storage = StorageConfig(storage_type="local", root_path="/data/root")
    task = SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=10,
            partition_id=1,
            row_count=2,
            storage_version="StorageV3",
            manifest_path="segments/10/manifest.json",
            manifest_version="v1",
        ),
        schema=MilvusSchema(
            collection_name="demo",
            fields=(
                FieldSchema("id", 100, "Int64"),
                FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
            ),
        ),
        projected_fields=(
            FieldSchema("id", 100, "Int64"),
            FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
        ),
        include=(),
        storage=storage,
    )

    result = list(MilvusStorageReader(storage).read_segment_batches(task, batch_size=1))

    assert calls["column_groups"] == ["group"]
    assert calls["reader"] == {"columns": ["id", "vector"]}
    assert [batch.to_pydict() for batch in result] == [
        {"id": [1], "vector": [[0.10000000149011612, 0.20000000298023224]]},
        {"id": [2], "vector": [[0.30000001192092896, 0.4000000059604645]]},
    ]



def test_milvus_storage_writer_writes_batches_and_commits(monkeypatch):
    calls = {}

    class FakeWriter:
        def __init__(self, path, schema, properties):
            calls["writer"] = {"path": path, "schema": schema, "properties": properties}
            calls["batches"] = []

        def write(self, batch):
            calls["batches"].append(batch)

        def close(self):
            calls["writer_closed"] = True
            return ["group-a"]

    class FakeTransaction:
        def __init__(self, path, properties):
            calls["transaction"] = {"path": path, "properties": properties}

        def append_files(self, column_groups):
            calls["append_files"] = column_groups
            calls["column_groups"] = column_groups

        def commit(self):
            calls["committed"] = True
            return 7

        def close(self):
            calls["transaction_closed"] = True

    fake_milvus_storage = SimpleNamespace(Writer=FakeWriter, Transaction=FakeTransaction)
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    storage = StorageConfig(storage_type="local", root_path="/data/root")
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    schema = (
        FieldSchema("id", 100, "Int64", nullable=False),
        FieldSchema("name", 101, "VarChar"),
    )

    result = MilvusStorageWriter(storage).write_segment_table(
        table,
        schema,
        "segments/10",
    )

    assert result == SegmentWriteResult(["group-a"], "7")
    assert calls["writer"]["path"] == "segments/10"
    assert calls["writer"]["schema"].field("id").metadata == {b"PARQUET:field_id": b"100"}
    assert len(calls["batches"]) == 1
    assert calls["transaction"] == {
        "path": "segments/10",
        "properties": {
            "fs.storage_type": "local",
            "fs.root_path": "/data/root",
            "fs.use_ssl": "true",
        },
    }
    assert calls["column_groups"] == ["group-a"]
    assert calls["append_files"] == ["group-a"]
    assert calls["committed"] is True
    assert calls["transaction_closed"] is True


def test_milvus_storage_writer_addfield_drops_fields_and_adds_column_groups(monkeypatch):
    calls = {"dropped": []}

    class FakeWriter:
        def __init__(self, path, schema, properties):
            calls["writer"] = {"path": path, "schema": schema, "properties": properties}

        def write(self, batch):
            calls["batch"] = batch

        def close(self):
            return ["group-a"]

    class FakeTransaction:
        def __init__(self, path, properties):
            calls["transaction"] = {"path": path, "properties": properties}

        def drop_column(self, field_id):
            calls["dropped"].append(field_id)

        def add_column_groups(self, column_groups):
            calls["added"] = column_groups

        def commit(self):
            return 8

        def close(self):
            calls["transaction_closed"] = True

    fake_milvus_storage = SimpleNamespace(Writer=FakeWriter, Transaction=FakeTransaction)
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    result = MilvusStorageWriter(StorageConfig(storage_type="local")).write_segment_table(
        pa.table({"name": ["a"]}),
        (FieldSchema("name", 101, "VarChar"),),
        "segments/10",
        mode="addfield",
    )

    assert result == SegmentWriteResult(["group-a"], "8")
    assert calls["dropped"] == ["101"]
    assert calls["added"] == ["group-a"]
    assert calls["transaction_closed"] is True


def test_milvus_storage_writer_rejects_unknown_write_mode(monkeypatch):
    class FakeWriter:
        def __init__(self, path, schema, properties):
            pass

        def write(self, batch):
            pass

        def close(self):
            return ["group-a"]

    class FakeTransaction:
        def __init__(self, path, properties):
            pass

    fake_milvus_storage = SimpleNamespace(Writer=FakeWriter, Transaction=FakeTransaction)
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    with pytest.raises(StorageError, match="Unsupported StorageV3 write mode"):
        MilvusStorageWriter(StorageConfig()).write_segment_table(
            pa.table({"id": [1]}),
            (FieldSchema("id", 100, "Int64"),),
            "segments/10",
            mode="replace",
        )


def test_milvus_storage_writer_wraps_backend_errors(monkeypatch):
    class FakeWriter:
        def __init__(self, path, schema, properties):
            raise RuntimeError("backend failed")

    fake_milvus_storage = SimpleNamespace(Writer=FakeWriter)
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    with pytest.raises(StorageError, match="Failed to write StorageV3 segment"):
        MilvusStorageWriter(StorageConfig()).write_segment_table(
            pa.table({"id": [1]}),
            (FieldSchema("id", 100, "Int64"),),
            "segments/10",
        )


def test_milvus_storage_reader_reads_manifest_list_packed_parquet(tmp_path):
    segment_root = tmp_path / "files" / "insert_log" / "100" / "200" / "300"
    id_dir = segment_root / "0"
    vector_dir = segment_root / "101"
    id_dir.mkdir(parents=True)
    vector_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "RowID": pa.array([10, 11], type=pa.int64()),
                "Timestamp": pa.array([1000, 1001], type=pa.int64()),
            },
            schema=pa.schema(
                [
                    pa.field(
                        "id",
                        pa.int64(),
                        nullable=False,
                        metadata={b"PARQUET:field_id": b"100"},
                    ),
                    pa.field(
                        "RowID",
                        pa.int64(),
                        nullable=False,
                        metadata={b"PARQUET:field_id": b"0"},
                    ),
                    pa.field(
                        "Timestamp",
                        pa.int64(),
                        nullable=False,
                        metadata={b"PARQUET:field_id": b"1"},
                    ),
                ],
                metadata={
                    b"storage_version": b"1.0.0",
                    b"group_field_id_list": b"100,0,1;101",
                },
            ),
        ),
        id_dir / "log-1",
    )
    pq.write_table(
        pa.table(
            {
                "vector": pa.array(
                    [b"\x00\x00\x80?\x00\x00\x00@", b"\x00\x00@@\x00\x00\x80@"],
                    type=pa.binary(8),
                )
            },
            schema=pa.schema(
                [
                    pa.field(
                        "vector",
                        pa.binary(8),
                        nullable=False,
                        metadata={b"PARQUET:field_id": b"101"},
                    )
                ],
                metadata={
                    b"storage_version": b"1.0.0",
                    b"group_field_id_list": b"100,0,1;101",
                },
            ),
        ),
        vector_dir / "log-2",
    )

    task = SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=300,
            partition_id=200,
            row_count=2,
            storage_version="PackedParquet",
            manifest_path="files/insert_log/100/200/300",
            manifest_version=None,
            raw={"packed_parquet_manifest": True},
        ),
        schema=MilvusSchema(
            collection_name="demo",
            fields=(
                FieldSchema("id", 100, "Int64"),
                FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
            ),
        ),
        projected_fields=(
            FieldSchema("id", 100, "Int64"),
            FieldSchema("vector", 101, "FloatVector", params={"dim": "2"}),
        ),
        include=(),
        storage=StorageConfig(storage_type="local", root_path=str(tmp_path)),
    )

    reader = MilvusStorageReader(task.storage)
    table = reader.read_segment_table(task)
    batches = list(reader.read_segment_batches(task, batch_size=1))

    assert table.to_pydict() == {
        "id": [1, 2],
        "vector": [[1.0, 2.0], [3.0, 4.0]],
    }
    assert [batch.to_pydict() for batch in batches] == [
        {"id": [1], "vector": [[1.0, 2.0]]},
        {"id": [2], "vector": [[3.0, 4.0]]},
    ]



def test_milvus_storage_reader_rejects_non_storage_v3_before_backend(monkeypatch):
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: pytest.fail("milvus-storage backend should not be loaded"),
    )

    with pytest.raises(UnsupportedSegmentError):
        MilvusStorageReader(StorageConfig()).read_segment_table(
            _segment_read_task(storage_version="StorageV1")
        )


def test_milvus_storage_reader_rejects_missing_manifest_before_backend(monkeypatch):
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: pytest.fail("milvus-storage backend should not be loaded"),
    )

    with pytest.raises(ManifestError, match="manifest_path"):
        MilvusStorageReader(StorageConfig()).read_segment_table(
            _segment_read_task(manifest_path=None)
        )


def test_milvus_lite_reader_does_not_require_storage_v3_manifest():
    task = _segment_read_task(
        storage=StorageConfig(backend="milvus_lite", root_path="/lite/db"),
        storage_version="PackedParquet",
        manifest_path=None,
    )

    with pytest.raises(UnsupportedFeatureError, match="Milvus Lite storage reader"):
        MilvusLiteStorageReader(task.storage).read_segment_table(task)


def test_milvus_lite_reader_requires_local_path():
    task = _segment_read_task(
        storage=StorageConfig(backend="milvus_lite"),
        storage_version="PackedParquet",
        manifest_path=None,
    )

    with pytest.raises(StorageError, match="root_path"):
        MilvusLiteStorageReader(task.storage).read_segment_table(task)


def test_milvus_storage_reader_wraps_backend_errors(monkeypatch):
    class FakeTransaction:
        def __init__(self, path, properties):
            pass

        def __enter__(self):
            raise RuntimeError("backend failed")

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_milvus_storage = SimpleNamespace(Transaction=FakeTransaction)
    monkeypatch.setattr(
        "ray_milvus.io.storage._load_milvus_storage",
        lambda: fake_milvus_storage,
    )

    task = SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=10,
            partition_id=1,
            row_count=2,
            storage_version="StorageV3",
            manifest_path="segments/10/manifest.json",
            manifest_version="v1",
        ),
        schema=MilvusSchema(
            collection_name="demo",
            fields=(FieldSchema("id", 100, "Int64"),),
        ),
        projected_fields=(FieldSchema("id", 100, "Int64"),),
        include=(),
        storage=StorageConfig(storage_type="local", root_path="/data/root"),
    )

    with pytest.raises(StorageError, match="segment 10 from segments/10/manifest.json"):
        MilvusStorageReader(task.storage).read_segment_table(task)
