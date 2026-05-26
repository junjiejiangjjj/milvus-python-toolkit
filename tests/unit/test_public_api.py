import json

import pyarrow as pa
import pytest

import milvus_toolkit as mt
from milvus_toolkit.io.storage import SegmentWriteResult


def test_public_api_exports_mvp_symbols():
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")

    assert storage.endpoint == "localhost:9000"
    assert mt.backfill_snapshot is not None
    assert mt.read_snapshot is not None
    assert mt.create_snapshot is not None
    assert mt.create_snapshot_from_milvus is not None
    assert mt.import_milvus_snapshot is not None
    assert mt.inspect_snapshot is not None
    assert mt.write_segment is not None
    assert mt.write_snapshot is not None
    assert mt.write_snapshot_segments is not None
    assert issubclass(mt.UnsupportedSegmentError, mt.MilvusToolkitError)


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


def test_create_snapshot_from_milvus_uses_loaded_schema(monkeypatch, tmp_path):
    output_path = tmp_path / "snapshot.json"

    def load_schema(**kwargs):
        assert kwargs["uri"] == "http://localhost:19530"
        assert kwargs["collection_name"] == "demo_collection"
        assert kwargs["token"] == "secret"
        return {"name": "demo", "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}]}

    monkeypatch.setattr("milvus_toolkit.api.load_collection_schema", load_schema)

    payload = mt.create_snapshot_from_milvus(
        "http://localhost:19530",
        "demo_collection",
        [{"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "segment-10"}],
        output_path=output_path,
        token="secret",
    )

    assert payload["collection_name"] == "demo_collection"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["collection_name"] == "demo_collection"


def test_import_milvus_snapshot_writes_output(monkeypatch, tmp_path):
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
        "milvus_toolkit.api.build_snapshot_payload_from_native_snapshot",
        build_payload,
    )

    payload = mt.import_milvus_snapshot("metadata.json", output_path=output_path)

    assert payload["segments"][0]["segment_id"] == 10
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["collection_name"] == "demo_collection"
    with pytest.raises(mt.ConfigError, match="already exists"):
        mt.import_milvus_snapshot("metadata.json", output_path=output_path)


def test_write_snapshot_writes_segment_and_snapshot(monkeypatch, tmp_path):
    calls = {}

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls["table"] = table
            calls["schema"] = schema
            calls["segment_path"] = segment_path
            calls["mode"] = mode
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())
    table = pa.table({"id": [1, 2]})
    snapshot_path = tmp_path / "snapshot.json"

    snapshot = mt.write_snapshot(
        table,
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        segment_path="segments/10",
        segment_id=10,
        output_path=snapshot_path,
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
    written = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert written["segments"][0]["manifest_path"] == "segments/10"



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
    monkeypatch.setattr("milvus_toolkit.api.read_snapshot", read_snapshot)
    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())
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
    monkeypatch.setattr("milvus_toolkit.api.read_snapshot", lambda *args, **kwargs: FakeDataset())
    captured = []

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            captured.append(table.to_pydict())
            return SegmentWriteResult(["group-a"], "1")

    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())
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

    monkeypatch.setattr("milvus_toolkit.api.read_snapshot", lambda *args, **kwargs: FakeDataset())

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

    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())
    snapshot_path = tmp_path / "snapshot.json"

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
        output_path=snapshot_path,
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
    written = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert written["segments"][1]["segment_id"] == 11



def test_write_snapshot_segments_passes_addfield_mode(monkeypatch):
    calls = []

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls.append((segment_path, mode))
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())

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
        "milvus_toolkit.api.create_storage_writer",
        lambda storage: pytest.fail("writer should not be created"),
    )

    with pytest.raises(mt.ConfigError, match="segment_path"):
        mt.write_snapshot_segments(
            [{"table": pa.table({"id": [1]}), "segment_id": 10}],
            {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
            mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        )



def test_write_segment_returns_snapshot_segment_metadata(monkeypatch):
    calls = {}

    class FakeWriter:
        def write_segment_table(self, table, schema, segment_path, mode="append"):
            calls["table"] = table
            calls["schema"] = schema
            calls["segment_path"] = segment_path
            calls["mode"] = mode
            return SegmentWriteResult(["group-a"], "7")

    monkeypatch.setattr("milvus_toolkit.api.create_storage_writer", lambda storage: FakeWriter())
    table = pa.table({"id": [1, 2]})

    segment = mt.write_segment(
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

    segment = mt.write_segment(
        table,
        {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]},
        mt.StorageConfig(storage_type="local", root_path="/tmp/storage"),
        segment_path="segments/10",
        segment_id=10,
    )

    assert segment["manifest_version"] == "7"
