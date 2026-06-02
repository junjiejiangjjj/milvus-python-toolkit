from pathlib import Path

import pyarrow as pa

import ray_milvus as mt
import ray_milvus.api as api

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


class FakeStorageReader:
    def read_segment_table(self, task):
        assert task.segment.segment_id == 10
        return pa.table({"id": [1, 2]})


class FakeBatchStorageReader:
    def read_segment_batches(self, task, batch_size=None):
        assert task.segment.segment_id == 10
        assert batch_size == 1
        yield pa.RecordBatch.from_pydict({"id": [1]})
        yield pa.RecordBatch.from_pydict({"id": [2]})


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
    assert [field.name for field in dataset.read_plan.projected_fields] == ["id"]


def test_read_snapshot_can_override_manifest_version():
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        manifest_version=7,
    )

    assert dataset.read_plan.manifest_version == "7"
    assert dataset.read_plan.tasks[0].manifest_version == "7"



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



def test_read_snapshot_iter_batches_streams_batches_and_metadata(monkeypatch):
    factory_calls = []

    def create_fake_reader(storage):
        factory_calls.append(storage)
        return FakeBatchStorageReader()

    monkeypatch.setattr(api, "create_storage_reader", create_fake_reader)
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")

    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=storage,
        columns=["id"],
        include=["segment_id", "row_offset"],
    )

    batches = list(dataset.iter_batches(batch_size=1))

    assert factory_calls == [storage]
    assert [batch.to_pydict() for batch in batches] == [
        {"id": [1], "segment_id": [10], "row_offset": [0]},
        {"id": [2], "segment_id": [10], "row_offset": [1]},
    ]



def test_read_snapshot_iter_batches_rejects_invalid_batch_size():
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=["id"],
    )

    try:
        next(iter(dataset.iter_batches(batch_size=0)))
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected invalid batch_size to be rejected")



def test_read_snapshot_to_ray_blocks_delegates_to_ray_engine(monkeypatch):
    calls = []

    def execute_ray_read_plan_blocks(plan, target_block_size=None, parallelism=None):
        calls.append((plan, target_block_size, parallelism))
        return ["block-ref"]

    monkeypatch.setattr(api, "_execute_ray_read_plan_blocks", execute_ray_read_plan_blocks)
    dataset = mt.read_snapshot(
        str(FIXTURE),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=["id"],
    )

    blocks = dataset.to_ray_blocks(target_block_size="64MiB", parallelism=2)

    assert blocks == ["block-ref"]
    assert calls == [(dataset.read_plan, "64MiB", 2)]
