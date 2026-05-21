from pathlib import Path

import pyarrow as pa

from milvus_toolkit.core.planner import plan_snapshot_read
from milvus_toolkit.engines.local import execute_read_plan
from milvus_toolkit.io.object_store import load_snapshot_json
from milvus_toolkit.types import StorageConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


class FakeStorageAdapter:
    def read_segment_table(self, task):
        assert task.segment.segment_id == 10
        return pa.table({"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]})


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
