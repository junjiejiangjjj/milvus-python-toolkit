import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray_milvus as mt
from ray_milvus.core.planner import plan_snapshot_read
from ray_milvus.engines.local import execute_read_plan
from ray_milvus.errors import StorageError
from ray_milvus.io.object_store import load_snapshot_json
from ray_milvus.types import StorageConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


class FakeStorageAdapter:
    def read_segment_table(self, task):
        assert task.segment.segment_id == 10
        return pa.table({"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]})



class MultiSegmentFakeStorageAdapter:
    def read_segment_table(self, task):
        return pa.table({"id": [task.segment.segment_id]})


def test_local_engine_reads_with_fake_storage_adapter():
    plan = plan_snapshot_read(
        load_snapshot_json(str(FIXTURE)),
        storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=("id", "vector"),
        include=("segment_id", "row_offset"),
    )

    table = execute_read_plan(plan, FakeStorageAdapter())

    assert table.column_names == ["id", "vector", "segment_id", "row_offset"]
    assert table["segment_id"].to_pylist() == [10, 10]
    assert table["row_offset"].to_pylist() == [0, 1]



def test_local_engine_validates_segment_row_count():
    snapshot = load_snapshot_json(str(FIXTURE))
    snapshot["segments"][0]["row_count"] = 3
    plan = plan_snapshot_read(
        snapshot,
        storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=("id", "vector"),
    )

    with pytest.raises(StorageError, match="Segment 10 row count mismatch"):
        execute_read_plan(plan, FakeStorageAdapter())


def test_local_engine_concats_metadata_for_multiple_segments():
    snapshot = load_snapshot_json(str(FIXTURE))
    snapshot["segments"] = [
        {**snapshot["segments"][0], "segment_id": 10, "row_count": 1},
        {**snapshot["segments"][0], "segment_id": 11, "row_count": 1},
    ]
    plan = plan_snapshot_read(
        snapshot,
        storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=("id",),
        include=("segment_id", "row_offset"),
    )

    table = execute_read_plan(plan, MultiSegmentFakeStorageAdapter())

    assert table.to_pydict() == {
        "id": [10, 11],
        "segment_id": [10, 11],
        "row_offset": [0, 0],
    }


def test_write_and_read_snapshot_with_real_milvus_storage_smoke(tmp_path):
    if os.environ.get("MILVUS_STORAGE_READ_SMOKE") != "1":
        pytest.skip("set MILVUS_STORAGE_READ_SMOKE=1 after scripts/install_dev.sh")

    pytest.importorskip("ray_milvus._vendor.milvus_storage")
    storage_path = tmp_path / "segment-10"
    toolkit_schema = {
        "name": "demo_collection",
        "fields": [
            {
                "name": "id",
                "field_id": 100,
                "data_type": "Int64",
                "is_primary": True,
                "nullable": False,
            },
            {"name": "name", "field_id": 101, "data_type": "VarChar"},
            {
                "name": "vector",
                "field_id": 102,
                "data_type": "FloatVector",
                "params": {"dim": "2"},
            },
        ],
    }
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": ["a", "b"],
            "vector": [[0.1, 0.2], [0.3, 0.4]],
        }
    )
    storage = mt.StorageConfig(storage_type="local", root_path=str(tmp_path))

    snapshot_path = tmp_path / "snapshot.json"
    snapshot = mt.write_snapshot(
        table,
        toolkit_schema,
        storage,
        segment_path=str(storage_path),
        segment_id=10,
        partition_id=1,
        manifest_version="v1",
    )
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    result = mt.read_snapshot(
        str(snapshot_path),
        storage=storage,
        columns=["id", "name", "vector"],
        include=["segment_id", "row_offset"],
    ).to_arrow()

    assert result.column_names == ["id", "name", "vector", "segment_id", "row_offset"]
    assert result["id"].to_pylist() == [1, 2]
    assert result["name"].to_pylist() == ["a", "b"]
    assert result["segment_id"].to_pylist() == [10, 10]
    assert result["row_offset"].to_pylist() == [0, 1]



def test_cli_write_native_segment_with_real_milvus_storage_smoke(tmp_path, capsys):
    if os.environ.get("MILVUS_STORAGE_READ_SMOKE") != "1":
        pytest.skip("set MILVUS_STORAGE_READ_SMOKE=1 after scripts/install_dev.sh")

    pytest.importorskip("ray_milvus._vendor.milvus_storage")
    from ray_milvus.cli.main import main

    input_path = tmp_path / "input.parquet"
    schema_path = tmp_path / "schema.json"
    snapshot_path = tmp_path / "snapshot.json"
    storage_path = tmp_path / "segment-10"
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": ["a", "b"],
            "vector": [[0.1, 0.2], [0.3, 0.4]],
        }
    )
    pq.write_table(table, input_path)
    schema_path.write_text(
        json.dumps(
            {
                "name": "demo_collection",
                "fields": [
                    {
                        "name": "id",
                        "field_id": 100,
                        "data_type": "Int64",
                        "is_primary": True,
                        "nullable": False,
                    },
                    {"name": "name", "field_id": 101, "data_type": "VarChar"},
                    {
                        "name": "vector",
                        "field_id": 102,
                        "data_type": "FloatVector",
                        "params": {"dim": "2"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "write-native-segment",
            "--input",
            str(input_path),
            "--schema-file",
            str(schema_path),
            "--segment-path",
            str(storage_path),
            "--segment-id",
            "10",
            "--partition-id",
            "1",
            "--manifest-version",
            "v1",
            "--storage-root",
            str(tmp_path),
            "--snapshot-output",
            str(snapshot_path),
            "--collection-name",
            "demo_collection",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Wrote segment 10 and snapshot" in output

    result = mt.read_snapshot(
        str(snapshot_path),
        storage=mt.StorageConfig(storage_type="local", root_path=str(tmp_path)),
        columns=["id", "name", "vector"],
        include=["segment_id", "row_offset"],
    ).to_arrow()

    assert result.column_names == ["id", "name", "vector", "segment_id", "row_offset"]
    assert result["id"].to_pylist() == [1, 2]
    assert result["name"].to_pylist() == ["a", "b"]
    assert result["segment_id"].to_pylist() == [10, 10]
    assert result["row_offset"].to_pylist() == [0, 1]




def test_backfill_snapshot_with_real_milvus_storage_smoke(tmp_path):
    if os.environ.get("MILVUS_STORAGE_READ_SMOKE") != "1":
        pytest.skip("set MILVUS_STORAGE_READ_SMOKE=1 after scripts/install_dev.sh")

    pytest.importorskip("ray_milvus._vendor.milvus_storage")
    storage_path = tmp_path / "segment-10"
    source_schema = {
        "name": "demo_collection",
        "fields": [
            {
                "name": "id",
                "field_id": 100,
                "data_type": "Int64",
                "is_primary": True,
                "nullable": False,
            },
            {"name": "name", "field_id": 101, "data_type": "VarChar"},
        ],
    }
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": ["a", "b"],
        }
    )
    storage = mt.StorageConfig(storage_type="local", root_path=str(tmp_path))
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = mt.write_snapshot(
        source,
        source_schema,
        storage,
        segment_path=str(storage_path),
        segment_id=10,
        collection_name="demo_collection",
        partition_id=1,
    )
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    backfill_schema = {
        "name": "demo_collection",
        "fields": [
            {
                "name": "id",
                "field_id": 100,
                "data_type": "Int64",
                "is_primary": True,
                "nullable": False,
            },
            {"name": "age", "field_id": 102, "data_type": "Int64"},
        ],
    }
    backfilled = mt.backfill_snapshot(
        str(snapshot_path),
        storage,
        pa.table({"id": pa.array([1, 2], type=pa.int64()), "age": [10, 20]}),
        backfill_schema,
        primary_key="id",
        fields=["age"],
        mode="replace",
        output_path=tmp_path / "backfilled.json",
        overwrite=True,
    )

    result = mt.read_snapshot(
        str(tmp_path / "backfilled.json"),
        storage=storage,
        columns=["age"],
        include=["segment_id", "row_offset"],
    ).to_arrow()

    assert backfilled["segments"][0]["manifest_path"] == str(storage_path)
    assert result.to_pydict() == {
        "age": [10, 20],
        "segment_id": [10, 10],
        "row_offset": [0, 1],
    }



def test_read_snapshot_with_real_milvus_storage_smoke(tmp_path):
    if os.environ.get("MILVUS_STORAGE_READ_SMOKE") != "1":
        pytest.skip("set MILVUS_STORAGE_READ_SMOKE=1 after scripts/install_dev.sh")

    milvus_storage = pytest.importorskip("ray_milvus._vendor.milvus_storage")
    storage_path = tmp_path / "segment-10"
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
                nullable=False,
                metadata={b"PARQUET:field_id": b"100"},
            ),
            pa.field("name", pa.string(), metadata={b"PARQUET:field_id": b"101"}),
            pa.field(
                "vector",
                pa.list_(pa.float32()),
                metadata={b"PARQUET:field_id": b"102"},
            ),
        ]
    )
    batch = pa.RecordBatch.from_pydict(
        {
            "id": [1, 2],
            "name": ["a", "b"],
            "vector": [[0.1, 0.2], [0.3, 0.4]],
        },
        schema=schema,
    )
    properties = {"fs.storage_type": "local", "fs.root_path": str(tmp_path)}

    writer = milvus_storage.Writer(str(storage_path), schema, properties)
    writer.write(batch)
    column_groups = writer.close()
    transaction = milvus_storage.Transaction(str(storage_path), properties)
    transaction.append_files(column_groups)
    transaction.commit()
    transaction.close()

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "collection_name": "demo_collection",
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [
                        {
                            "name": "id",
                            "field_id": 100,
                            "data_type": "Int64",
                            "is_primary": True,
                            "nullable": False,
                        },
                        {"name": "name", "field_id": 101, "data_type": "VarChar"},
                        {
                            "name": "vector",
                            "field_id": 102,
                            "data_type": "FloatVector",
                            "params": {"dim": "2"},
                        },
                    ],
                },
                "segments": [
                    {
                        "segment_id": 10,
                        "partition_id": 1,
                        "row_count": 2,
                        "storage_version": "StorageV3",
                        "manifest_path": str(storage_path),
                        "manifest_version": "v1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    table = mt.read_snapshot(
        str(snapshot_path),
        storage=mt.StorageConfig(storage_type="local", root_path=str(tmp_path)),
        columns=["id", "name", "vector"],
        include=["segment_id", "row_offset"],
    ).to_arrow()

    assert table.column_names == ["id", "name", "vector", "segment_id", "row_offset"]
    assert table["id"].to_pylist() == [1, 2]
    assert table["name"].to_pylist() == ["a", "b"]
    assert table["segment_id"].to_pylist() == [10, 10]
    assert table["row_offset"].to_pylist() == [0, 1]
