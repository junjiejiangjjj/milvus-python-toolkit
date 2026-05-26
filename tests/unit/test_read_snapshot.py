from pathlib import Path

import pyarrow as pa
import pytest

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
    assert dataset.read_plan.tasks[0].segment.manifest_version == "v1"
    assert dataset.read_plan.tasks[0].manifest_version is None
    assert dataset.read_plan.tasks[0].predicate is None
    assert [field.name for field in dataset.read_plan.projected_fields] == ["id"]


def test_read_snapshot_can_override_manifest_version():
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        manifest_version=7,
    )

    assert dataset.read_plan.manifest_version == "7"
    assert dataset.read_plan.tasks[0].manifest_version == "7"



def test_read_snapshot_can_pass_predicate_to_plan():
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        predicate="id > 100",
    )

    assert dataset.read_plan.predicate == "id > 100"
    assert dataset.read_plan.tasks[0].predicate == "id > 100"



def test_read_snapshot_rejects_invalid_predicate():
    with pytest.raises(mt.ConfigError, match="predicate must be a string"):
        mt.read_snapshot(
            str(FIXTURE),
            storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
            predicate=123,
        )

    with pytest.raises(mt.ConfigError, match="predicate cannot be empty"):
        mt.read_snapshot(
            str(FIXTURE),
            storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
            predicate="  ",
        )



def test_read_snapshot_to_arrow_uses_storage_reader_factory(monkeypatch):
    factory_calls = []

    def create_fake_reader(storage):
        factory_calls.append(storage)
        return FakeStorageReader()

    monkeypatch.setattr(api, "create_storage_reader", create_fake_reader)
    storage = mt.StorageConfig(
        backend="milvus_lite", endpoint="localhost:9000", bucket="bucket"
    )

    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=storage,
        columns=["id"],
        include=["segment_id"],
    )

    table = dataset.to_arrow()

    assert factory_calls == [storage]
    assert table.column_names == ["id", "segment_id"]
    assert table["id"].to_pylist() == [1, 2]
    assert table["segment_id"].to_pylist() == [10, 10]
