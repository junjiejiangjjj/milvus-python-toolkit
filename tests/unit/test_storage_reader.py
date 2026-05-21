from types import SimpleNamespace

import pyarrow as pa

from milvus_toolkit.core.plans import SegmentReadTask
from milvus_toolkit.io.storage import (
    MilvusStorageReader,
    _storage_properties,
    _to_pyarrow_schema,
)
from milvus_toolkit.types import FieldSchema, MilvusSchema, SegmentMetadata, StorageConfig


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
        "fs.endpoint": "localhost:9000",
        "fs.bucket_name": "bucket",
        "fs.access_key_id": "ak",
        "fs.secret_access_key": "sk",
        "fs.use_ssl": "false",
        "fs.region": "us-east-1",
        "fs.root_path": "right",
        "custom": "value",
    }


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
    assert schema.field("name").type == pa.string()
    assert schema.field("vector").type == pa.list_(pa.float32(), 2)


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
        "milvus_toolkit.io.storage._load_milvus_storage",
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
    assert calls["column_groups"] == ["group"]
    assert calls["reader"]["column_groups"] == "ffi-column-groups"
    assert calls["reader"]["columns"] == ["id", "vector"]
    assert calls["reader"]["schema"].field("id").type == pa.int64()
    assert table.to_pydict() == {"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]}
