from pathlib import Path

import pyarrow as pa

import milvus_toolkit as mt
import milvus_toolkit.api as api

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


class FakeStorageReader:
    def read_segment_table(self, task):
        assert task.segment.segment_id == 10
        return pa.table({"id": [1, 2]})


def test_read_snapshot_returns_dataset_with_plan():
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=["id"],
        include=["segment_id"],
    )

    assert dataset.read_plan.tasks[0].segment.segment_id == 10
    assert [field.name for field in dataset.read_plan.projected_fields] == ["id"]


def test_read_snapshot_to_arrow_uses_storage_reader_factory(monkeypatch):
    monkeypatch.setattr(api, "create_storage_reader", lambda storage: FakeStorageReader())

    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=["id"],
        include=["segment_id"],
    )

    table = dataset.to_arrow()

    assert table.column_names == ["id", "segment_id"]
    assert table["id"].to_pylist() == [1, 2]
    assert table["segment_id"].to_pylist() == [10, 10]
