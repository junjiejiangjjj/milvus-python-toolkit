from pathlib import Path

import pytest

from milvus_toolkit.core.planner import plan_snapshot_read
from milvus_toolkit.errors import UnsupportedFeatureError
from milvus_toolkit.io.object_store import load_snapshot_json
from milvus_toolkit.types import StorageConfig

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_plan_snapshot_read_storage_v3():
    plan = plan_snapshot_read(
        load_snapshot_json(str(FIXTURES / "snapshot_storage_v3.json")),
        storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
        columns=("id",),
        include=("segment_id", "row_offset"),
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].manifest_path == "segments/10/manifest.json"
    assert [field.name for field in plan.projected_fields] == ["id"]
    assert plan.include == ("segment_id", "row_offset")


def test_plan_snapshot_read_keeps_non_storage_v3_engine_neutral():
    plan = plan_snapshot_read(
        load_snapshot_json(str(FIXTURES / "snapshot_non_storage_v3.json")),
        storage=StorageConfig(backend="milvus_lite", root_path="/lite/db"),
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].segment.segment_id == 20
    assert plan.tasks[0].segment.storage_version == "PackedParquet"


def test_plan_snapshot_read_rejects_unknown_include():
    with pytest.raises(UnsupportedFeatureError, match="partition_id"):
        plan_snapshot_read(
            load_snapshot_json(str(FIXTURES / "snapshot_storage_v3.json")),
            storage=StorageConfig(endpoint="localhost:9000", bucket="bucket"),
            include=("partition_id",),
        )
