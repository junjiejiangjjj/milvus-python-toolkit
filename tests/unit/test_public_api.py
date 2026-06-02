import json

import pyarrow as pa
import pytest

import ray_milvus as mt
from ray_milvus.io.storage import SegmentWriteResult


def test_public_api_exports_mvp_symbols():
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")

    assert storage.endpoint == "localhost:9000"
    assert mt.MilvusConfig is not None
    assert mt.RayMilvus is not None
    assert mt.backfill_snapshot is not None
    assert mt.read_snapshot is not None
    assert mt.create_snapshot is not None
    assert mt.create_snapshot_from_milvus is not None
    assert mt.import_milvus_snapshot is not None
    assert mt.import_native_milvus_snapshot is not None
    assert mt.inspect_snapshot is not None
    assert not hasattr(mt, "write_segment")
    assert mt.write_snapshot is not None
    assert mt.write_snapshot_segments is not None
    assert issubclass(mt.UnsupportedSegmentError, mt.MilvusToolkitError)



def test_ray_milvus_facade_uses_default_storage(monkeypatch):
    calls = []
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")

    def read_snapshot(snapshot_path, **kwargs):
        calls.append((snapshot_path, kwargs))
        return "dataset"

    monkeypatch.setattr("ray_milvus.api.read_snapshot", read_snapshot)

    ray = mt.RayMilvus(storage=storage)
    dataset = ray.read_snapshot("snapshot.json", columns=["id"], include=["segment_id"])

    assert dataset == "dataset"
    assert calls == [
        (
            "snapshot.json",
            {
                "storage": storage,
                "columns": ["id"],
                "include": ["segment_id"],
                "manifest_version": None,
            },
        )
    ]



def test_ray_milvus_facade_uses_milvus_config(monkeypatch):
    calls = []
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")
    milvus = mt.MilvusConfig(
        uri="http://localhost:19530",
        token="secret",
        db_name="default",
    )

    def create_snapshot_from_milvus(*args, **kwargs):
        calls.append((args, kwargs))
        return {"snapshot_name": kwargs["snapshot_name"], "segments": []}

    monkeypatch.setattr(
        "ray_milvus.api.create_snapshot_from_milvus",
        create_snapshot_from_milvus,
    )

    ray = mt.RayMilvus(storage=storage, milvus=milvus)
    payload = ray.create_snapshot_from_milvus(
        collection_name="demo_collection",
        snapshot_name="snapshot-1",
    )

    assert payload["snapshot_name"] == "snapshot-1"
    assert calls == [
        (
            ("http://localhost:19530", "demo_collection"),
            {
                "snapshot_name": "snapshot-1",
                "output_path": None,
                "storage": storage,
                "auto_snapshot_name": False,
                "token": "secret",
                "user": None,
                "password": None,
                "db_name": "default",
                "description": None,
                "compaction_protection_seconds": None,
                "overwrite": False,
                "pretty": True,
            },
        )
    ]



def test_ray_milvus_facade_requires_milvus_config():
    ray = mt.RayMilvus()

    with pytest.raises(mt.ConfigError, match="MilvusConfig"):
        ray.import_milvus_snapshot("demo_collection", "snapshot-1")


def test_create_snapshot_from_memory():
    payload = mt.create_snapshot(
        {"name": "demo", "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}]},
        [{"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "segment-10"}],
        collection_name="override",
    )

    assert payload["collection_name"] == "override"
    assert payload["collection_schema"]["name"] == "override"
    assert payload["segments"][0]["segment_id"] == 10


def test_create_snapshot_from_files_and_overwrite_guard(tmp_path):
    schema_path = tmp_path / "schema.json"
    segments_path = tmp_path / "segments.json"
    output_path = tmp_path / "snapshot.json"
    schema_path.write_text(
        json.dumps(
            {"name": "demo", "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}]}
        ),
        encoding="utf-8",
    )
    segments_path.write_text(
        json.dumps(
            [{"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "segment-10"}]
        ),
        encoding="utf-8",
    )

    mt.create_snapshot(schema_path, segments_path, output_path=output_path)

    assert json.loads(output_path.read_text(encoding="utf-8"))["segments"][0]["segment_id"] == 10
    with pytest.raises(mt.ConfigError, match="already exists"):
        mt.create_snapshot(schema_path, segments_path, output_path=output_path)
    mt.create_snapshot(schema_path, segments_path, output_path=output_path, overwrite=True)


def test_create_snapshot_from_milvus_uses_snapshot_location(monkeypatch, tmp_path):
    output_path = tmp_path / "snapshot.json"

    class FakeMilvusService:
        def __init__(self, **kwargs):
            assert kwargs["uri"] == "http://localhost:19530"

        def create_snapshot_for_read(self, **kwargs):
            assert kwargs["collection_name"] == "demo_collection"
            assert kwargs["snapshot_name"] == "snapshot-1"
            return type(
                "SnapshotLocation",
                (),
                {"name": "snapshot-1", "location": "s3://bucket/snapshot.json"},
            )()

    def load_snapshot_json(path):
        assert path == "s3://bucket/snapshot.json"
        return {
            "collection": {
                "name": "demo_collection",
                "schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
                },
            },
            "snapshot_info": {"collection_id": "1", "partition_ids": ["2"]},
            "manifest_list": ["files/snapshots/1/manifests/99/10.avro"],
            "segment_ids": ["10"],
        }

    monkeypatch.setattr("ray_milvus.api.MilvusService", FakeMilvusService)
    monkeypatch.setattr("ray_milvus.api.load_snapshot_json", load_snapshot_json)

    payload = mt.create_snapshot_from_milvus(
        "http://localhost:19530",
        "demo_collection",
        "snapshot-1",
        output_path=output_path,
    )

    assert payload["segments"][0]["manifest_path"] == "files/insert_log/1/2/10"
    assert payload["snapshot_name"] == "snapshot-1"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["collection_name"] == "demo_collection"
    assert written["snapshot_name"] == "snapshot-1"



def test_create_snapshot_from_milvus_uses_storage_for_relative_location(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "snapshot.json"

    class FakeMilvusService:
        def __init__(self, **kwargs):
            assert kwargs["uri"] == "http://localhost:19530"

        def create_snapshot_for_read(self, **kwargs):
            return type(
                "SnapshotLocation",
                (),
                {"name": kwargs["snapshot_name"], "location": "files/snapshot.json"},
            )()

    def load_snapshot_json_from_storage(path, **kwargs):
        assert path == "files/snapshot.json"
        assert kwargs["storage_type"] == "s3"
        assert kwargs["endpoint"] == "localhost:9000"
        assert kwargs["bucket"] == "a-bucket"
        return {
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10, "manifest_path": "files/insert_log/1/2/10"}],
        }

    monkeypatch.setattr("ray_milvus.api.MilvusService", FakeMilvusService)
    monkeypatch.setattr(
        "ray_milvus.api.load_snapshot_json_from_storage",
        load_snapshot_json_from_storage,
    )

    payload = mt.create_snapshot_from_milvus(
        "http://localhost:19530",
        "demo_collection",
        "snapshot-1",
        output_path=output_path,
        storage=mt.StorageConfig(
            storage_type="s3",
            endpoint="localhost:9000",
            bucket="a-bucket",
        ),
    )

    assert payload["segments"][0]["manifest_path"] == "files/insert_log/1/2/10"
    assert payload["snapshot_name"] == "snapshot-1"



def test_create_snapshot_from_milvus_generates_snapshot_name(monkeypatch):
    class FakeMilvusService:
        def __init__(self, **kwargs):
            assert kwargs["uri"] == "http://localhost:19530"

        def create_snapshot_for_read(self, **kwargs):
            assert kwargs["collection_name"] == "demo-collection"
            assert kwargs["snapshot_name"].startswith("ray_milvus_demo_collection_")
            return type(
                "SnapshotLocation",
                (),
                {"name": kwargs["snapshot_name"], "location": "s3://bucket/snapshot.json"},
            )()

    def load_snapshot_json(path):
        assert path == "s3://bucket/snapshot.json"
        return {
            "collection_schema": {
                "name": "demo-collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10, "manifest_path": "files/insert_log/1/2/10"}],
        }

    monkeypatch.setattr("ray_milvus.api.MilvusService", FakeMilvusService)
    monkeypatch.setattr("ray_milvus.api.load_snapshot_json", load_snapshot_json)

    payload = mt.create_snapshot_from_milvus(
        "http://localhost:19530",
        "demo-collection",
        auto_snapshot_name=True,
    )

    assert payload["snapshot_name"].startswith("ray_milvus_demo_collection_")



def test_create_snapshot_from_milvus_validates_snapshot_name_options():
    with pytest.raises(mt.ConfigError, match="snapshot_name cannot be set"):
        mt.create_snapshot_from_milvus(
            "http://localhost:19530",
            "demo_collection",
            snapshot_name="snapshot-1",
            auto_snapshot_name=True,
        )

    with pytest.raises(mt.ConfigError, match="snapshot_name is required"):
        mt.create_snapshot_from_milvus("http://localhost:19530", "demo_collection")



def test_import_milvus_snapshot_uses_existing_snapshot_location(monkeypatch, tmp_path):
    output_path = tmp_path / "snapshot.json"

    class FakeMilvusService:
        def __init__(self, **kwargs):
            assert kwargs["uri"] == "http://localhost:19530"
            assert kwargs["token"] == "secret"

        def describe_snapshot_for_read(self, **kwargs):
            assert kwargs == {
                "collection_name": "demo_collection",
                "snapshot_name": "snapshot-1",
            }
            return type(
                "SnapshotLocation",
                (),
                {"name": "snapshot-1", "location": "s3://bucket/snapshot.json"},
            )()

    def load_snapshot_json(path):
        assert path == "s3://bucket/snapshot.json"
        return {
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10, "manifest_path": "files/insert_log/1/2/10"}],
        }

    monkeypatch.setattr("ray_milvus.api.MilvusService", FakeMilvusService)
    monkeypatch.setattr("ray_milvus.api.load_snapshot_json", load_snapshot_json)

    payload = mt.import_milvus_snapshot(
        "http://localhost:19530",
        "demo_collection",
        "snapshot-1",
        output_path=output_path,
        token="secret",
    )

    assert payload["snapshot_name"] == "snapshot-1"
    assert payload["segments"][0]["segment_id"] == 10
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["collection_name"] == "demo_collection"
    with pytest.raises(mt.ConfigError, match="already exists"):
        mt.import_milvus_snapshot(
            "http://localhost:19530",
            "demo_collection",
            "snapshot-1",
            output_path=output_path,
            token="secret",
        )



def test_import_native_milvus_snapshot_writes_output(monkeypatch, tmp_path):
    output_path = tmp_path / "snapshot.json"

    def build_payload(**kwargs):
        assert kwargs["metadata_path"] == "metadata.json"
        return {
            "collection_name": "demo_collection",
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [
                {"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "10.avro"}
            ],
        }

    monkeypatch.setattr(
        "ray_milvus.api.build_snapshot_payload_from_native_snapshot",
        build_payload,
    )

    payload = mt.import_native_milvus_snapshot("metadata.json", output_path=output_path)

    assert payload["segments"][0]["segment_id"] == 10
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["collection_name"] == "demo_collection"
    with pytest.raises(mt.ConfigError, match="already exists"):
        mt.import_native_milvus_snapshot("metadata.json", output_path=output_path)


def test_write_snapshot_writes_segment_and_returns_snapshot(monkeypatch):
    calls = {}

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls["table"] = table
            calls["schema"] = schema
            calls["segment_path"] = segment_path
            calls["mode"] = mode
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())
    table = pa.table({"id": [1, 2]})

    snapshot = mt.write_snapshot(
        table,
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        segment_path="segments/10",
        segment_id=10,
        collection_name="demo_collection",
        partition_id=1,
        manifest_version="v1",
    )

    assert calls["table"] is table
    assert calls["schema"][0].name == "id"
    assert calls["segment_path"] == "segments/10"
    assert calls["mode"] == "append"
    assert snapshot["collection_name"] == "demo_collection"
    assert snapshot["segments"][0]["row_count"] == 2
    assert snapshot["segments"][0]["manifest_version"] == "v1"
    assert snapshot["segments"][0]["manifest_path"] == "segments/10"



def test_backfill_snapshot_merges_and_writes_addfield_segments(monkeypatch, tmp_path):
    written_segments = []

    class FakeDataset:
        def to_arrow(self):
            return pa.table(
                {
                    "id": [1, 2, 3],
                    "new_value": [None, "keep", "old"],
                    "segment_id": [10, 10, 11],
                    "row_offset": [1, 0, 0],
                }
            )

    def read_snapshot(*args, **kwargs):
        assert kwargs["include"] == ("segment_id", "row_offset")
        return FakeDataset()

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            written_segments.append((table, schema, segment_path, mode))
            return SegmentWriteResult(["group-a"], str(len(written_segments)))

    snapshot_path = tmp_path / "source.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "collection_name": "demo",
                "collection_schema": {
                    "name": "demo",
                    "fields": [
                        {"name": "id", "field_id": 100, "data_type": "Int64"},
                        {"name": "new_value", "field_id": 101, "data_type": "VarChar"},
                    ],
                },
                "segments": [
                    {
                        "segment_id": 10,
                        "partition_id": 1,
                        "storage_version": "StorageV3",
                        "manifest_path": "segments/10",
                    },
                    {
                        "segment_id": 11,
                        "partition_id": 2,
                        "storage_version": "StorageV3",
                        "manifest_path": "segments/11",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ray_milvus.api.read_snapshot", read_snapshot)
    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())
    output_path = tmp_path / "snapshot.json"

    snapshot = mt.backfill_snapshot(
        str(snapshot_path),
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        pa.table({"id": [1, 3], "new_value": ["one", "three"]}),
        {
            "name": "demo",
            "fields": [
                {"name": "id", "field_id": 100, "data_type": "Int64"},
                {"name": "new_value", "field_id": 101, "data_type": "VarChar"},
            ],
        },
        primary_key="id",
        fields=["new_value"],
        output_path=output_path,
        mode="coalesce",
        segment_path_template="backfill/{segment_id}",
    )

    assert [item[2] for item in written_segments] == ["backfill/10", "backfill/11"]
    assert [item[3] for item in written_segments] == ["addfield", "addfield"]
    assert written_segments[0][0].to_pydict() == {"new_value": ["keep", "one"]}
    assert written_segments[1][0].to_pydict() == {"new_value": ["old"]}
    assert written_segments[0][1][0].field_id == 101
    assert snapshot["segments"][0]["manifest_version"] == "1"
    assert json.loads(output_path.read_text(encoding="utf-8"))["segments"][1]["segment_id"] == 11



def test_backfill_snapshot_replace_and_overwrite_modes(monkeypatch, tmp_path):
    class FakeDataset:
        def to_arrow(self):
            return pa.table(
                {
                    "id": [1, 2],
                    "target": ["old-1", "old-2"],
                    "segment_id": [10, 10],
                    "row_offset": [0, 1],
                }
            )

    snapshot_path = tmp_path / "source.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "collection_name": "demo",
                "collection_schema": {
                    "name": "demo",
                    "fields": [
                        {"name": "id", "field_id": 100, "data_type": "Int64"},
                        {"name": "target", "field_id": 101, "data_type": "VarChar"},
                    ],
                },
                "segments": [
                    {
                        "segment_id": 10,
                        "storage_version": "StorageV3",
                        "manifest_path": "segments/10",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ray_milvus.api.read_snapshot", lambda *args, **kwargs: FakeDataset())
    captured = []

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            captured.append(table.to_pydict())
            return SegmentWriteResult(["group-a"], "1")

    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())
    schema = {
        "name": "demo",
        "fields": [
            {"name": "id", "field_id": 100, "data_type": "Int64"},
            {"name": "target", "field_id": 101, "data_type": "VarChar"},
        ],
    }
    backfill = pa.table({"id": [1], "target": ["new-1"]})

    mt.backfill_snapshot(
        str(snapshot_path),
        mt.StorageConfig(),
        backfill,
        schema,
        "id",
        ["target"],
        mode="replace",
    )
    mt.backfill_snapshot(
        str(snapshot_path),
        mt.StorageConfig(),
        backfill,
        schema,
        "id",
        ["target"],
        mode="overwrite",
    )

    assert captured[0] == {"target": ["new-1", None]}
    assert captured[1] == {"target": ["new-1", "old-2"]}



def test_backfill_snapshot_rejects_unknown_mode(monkeypatch):
    class FakeDataset:
        def to_arrow(self):
            return pa.table({"id": [1], "segment_id": [10], "row_offset": [0]})

    monkeypatch.setattr("ray_milvus.api.read_snapshot", lambda *args, **kwargs: FakeDataset())

    with pytest.raises(mt.ConfigError, match="Unsupported backfill mode"):
        mt.backfill_snapshot(
            "snapshot.json",
            mt.StorageConfig(),
            pa.table({"id": [1], "target": ["x"]}),
            {
                "name": "demo",
                "fields": [
                    {"name": "id", "field_id": 100, "data_type": "Int64"},
                    {"name": "target", "field_id": 101, "data_type": "VarChar"},
                ],
            },
            "id",
            ["target"],
            mode="bad",
        )



def test_write_snapshot_segments_writes_multiple_segments(monkeypatch, tmp_path):
    calls = []

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls.append((table, schema, segment_path, mode))
            return SegmentWriteResult([f"group-{segment_path}"], str(len(calls)))

    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())

    snapshot = mt.write_snapshot_segments(
        [
            {
                "table": pa.table({"id": [1, 2]}),
                "segment_path": "segments/10",
                "segment_id": 10,
                "partition_id": "1",
            },
            {
                "table": pa.table({"id": [3]}),
                "segment_path": "segments/11",
                "segment_id": "11",
                "manifest_version": "v3",
            },
        ],
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        collection_name="demo_collection",
    )

    assert [call[2] for call in calls] == ["segments/10", "segments/11"]
    assert [call[3] for call in calls] == ["append", "append"]
    assert snapshot["collection_name"] == "demo_collection"
    assert snapshot["segments"] == [
        {
            "segment_id": 10,
            "partition_id": 1,
            "row_count": 2,
            "storage_version": "StorageV3",
            "manifest_path": "segments/10",
            "manifest_version": "1",
        },
        {
            "segment_id": 11,
            "partition_id": None,
            "row_count": 1,
            "storage_version": "StorageV3",
            "manifest_path": "segments/11",
            "manifest_version": "v3",
        },
    ]



def test_write_snapshot_segments_passes_addfield_mode(monkeypatch):
    calls = []

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls.append((segment_path, mode))
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())

    mt.write_snapshot_segments(
        [
            {"table": pa.table({"id": [1]}), "segment_path": "segments/10", "segment_id": 10},
            {
                "table": pa.table({"id": [2]}),
                "segment_path": "segments/11",
                "segment_id": 11,
                "mode": "append",
            },
        ],
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        mode="addfield",
    )

    assert calls == [("segments/10", "addfield"), ("segments/11", "append")]



def test_write_snapshot_segments_requires_segment_fields(monkeypatch):
    monkeypatch.setattr(
        "ray_milvus.api.create_storage_writer",
        lambda storage: pytest.fail("writer should not be created"),
    )

    with pytest.raises(mt.ConfigError, match="segment_path"):
        mt.write_snapshot_segments(
            [{"table": pa.table({"id": [1]}), "segment_id": 10}],
            {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
            mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        )



def test_internal_write_segment_returns_snapshot_segment_metadata(monkeypatch):
    calls = {}

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls["table"] = table
            calls["schema"] = schema
            calls["segment_path"] = segment_path
            calls["mode"] = mode
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("ray_milvus.api.create_storage_writer", lambda storage: FakeWriter())
    table = pa.table({"id": [1, 2]})

    from ray_milvus.api import _write_segment

    segment = _write_segment(
        table,
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        segment_path="segments/10",
        segment_id=10,
        partition_id=1,
        manifest_version="v1",
    )

    assert calls["table"] is table
    assert calls["schema"][0].name == "id"
    assert calls["segment_path"] == "segments/10"
    assert segment == {
        "segment_id": 10,
        "partition_id": 1,
        "row_count": 2,
        "storage_version": "StorageV3",
        "manifest_path": "segments/10",
        "manifest_version": "v1",
    }

    segment = _write_segment(
        table,
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        segment_path="segments/10",
        segment_id=10,
    )

    assert segment["manifest_version"] == "7"
