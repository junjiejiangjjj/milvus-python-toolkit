import os

import pytest

import milvus_toolkit as mt


def test_import_real_native_snapshot_artifact_smoke(tmp_path):
    if os.environ.get("MILVUS_NATIVE_SNAPSHOT_SMOKE") != "1":
        pytest.skip("set MILVUS_NATIVE_SNAPSHOT_SMOKE=1 with native snapshot artifact paths")

    metadata_path = os.environ.get("MILVUS_NATIVE_SNAPSHOT_METADATA")
    if metadata_path is None:
        pytest.skip("MILVUS_NATIVE_SNAPSHOT_METADATA is required")

    output_path = tmp_path / "toolkit-snapshot.json"
    payload = mt.import_milvus_snapshot(
        metadata_path=metadata_path,
        manifest_dir=os.environ.get("MILVUS_NATIVE_SNAPSHOT_MANIFEST_DIR"),
        output_path=output_path,
    )

    assert payload["collection_schema"]["fields"]
    assert payload["segments"]
    assert output_path.exists()

    inspected = mt.inspect_snapshot(str(output_path), storage=mt.StorageConfig())
    assert inspected.segment_count == len(payload["segments"])
