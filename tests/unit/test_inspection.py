from pathlib import Path

import ray_milvus as mt

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_inspect_snapshot_returns_result_for_storage_v3_fixture():
    result = mt.inspect_snapshot(
        str(FIXTURES / "snapshot_storage_v3.json"),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
    )

    assert result.collection_name == "demo_collection"
    assert result.segment_count == 1
    assert result.diagnostics == ()


def test_inspect_snapshot_reports_unsupported_segment_diagnostic():
    result = mt.inspect_snapshot(
        str(FIXTURES / "snapshot_non_storage_v3.json"),
        storage=mt.StorageConfig(endpoint="localhost:9000", bucket="bucket"),
    )

    assert result.segment_count == 1
    assert result.diagnostics[0].segment_id == 20
    assert "not StorageV3" in result.diagnostics[0].message
