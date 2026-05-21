from pathlib import Path

from milvus_toolkit.core.snapshot import parse_snapshot
from milvus_toolkit.io.object_store import load_snapshot_json

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


def test_parse_snapshot_fixture():
    snapshot = parse_snapshot(load_snapshot_json(str(FIXTURE)))

    assert snapshot.collection_name == "demo_collection"
    assert snapshot.schema.fields[0].name == "id"
    assert snapshot.segments[0].segment_id == 10
    assert snapshot.segments[0].manifest_path == "segments/10/manifest.json"
