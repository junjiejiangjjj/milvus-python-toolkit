import pyarrow as pa
import pytest

from ray_milvus.core.plans import SegmentReadTask
from ray_milvus.engines import ray as ray_engine
from ray_milvus.errors import ConfigError, UnsupportedFeatureError
from ray_milvus.types import FieldSchema, MilvusSchema, SegmentMetadata, StorageConfig


def make_task(segment_id):
    schema = MilvusSchema(
        collection_name="demo",
        fields=(FieldSchema(name="id", field_id=1, data_type="Int64"),),
    )
    return SegmentReadTask(
        segment=SegmentMetadata(
            segment_id=segment_id,
            partition_id=None,
            row_count=2,
            storage_version="StorageV3",
            manifest_path=f"segment-{segment_id}",
            manifest_version=None,
        ),
        schema=schema,
        projected_fields=schema.fields,
        include=("segment_id",),
        storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
    )


def test_group_segment_read_tasks_defaults_to_one_segment_per_group():
    tasks = tuple(make_task(segment_id) for segment_id in [1, 2, 3])

    groups = ray_engine.group_segment_read_tasks(tasks)

    assert groups == ((tasks[0],), (tasks[1],), (tasks[2],))


def test_group_segment_read_tasks_splits_into_at_most_parallelism_groups():
    tasks = tuple(make_task(segment_id) for segment_id in [1, 2, 3, 4, 5])

    groups = ray_engine.group_segment_read_tasks(tasks, parallelism=2)

    assert groups == (tasks[:3], tasks[3:])


def test_group_segment_read_tasks_rejects_invalid_parallelism():
    with pytest.raises(ConfigError, match="parallelism"):
        ray_engine.group_segment_read_tasks((make_task(1),), parallelism=0)


def test_read_segment_task_group_blocks_reads_on_worker_path(monkeypatch):
    class FakeReader:
        def read_segment_batches(self, task, batch_size=None):
            assert batch_size == 10
            yield pa.RecordBatch.from_pydict({"id": [task.segment.segment_id * 10]})
            yield pa.RecordBatch.from_pydict({"id": [task.segment.segment_id * 10 + 1]})

    def create_storage_reader(storage):
        return FakeReader()

    monkeypatch.setattr(ray_engine, "create_storage_reader", create_storage_reader)
    tasks = (make_task(1), make_task(2))

    blocks = list(
        ray_engine.read_segment_task_group_blocks(
            tasks,
            target_block_size=1_000_000,
            read_batch_size=10,
        )
    )

    assert len(blocks) == 1
    assert blocks[0].to_pydict() == {
        "id": [10, 11, 20, 21],
        "segment_id": [1, 1, 2, 2],
    }


def test_execute_read_plan_blocks_requires_ray(monkeypatch):
    def require_ray():
        raise UnsupportedFeatureError("Ray Core execution requires Ray")

    monkeypatch.setattr(ray_engine, "_require_ray", require_ray)
    plan = type("Plan", (), {"tasks": (make_task(1),)})()

    with pytest.raises(UnsupportedFeatureError, match="Ray Core"):
        ray_engine.execute_read_plan_blocks(plan)
