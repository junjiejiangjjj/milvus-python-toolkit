import tempfile
from pathlib import Path

import ray_milvus as rm

SCHEMA = {
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
        {"name": "vector", "field_id": 102, "data_type": "FloatVector", "params": {"dim": "2"}},
    ],
}

SEGMENTS = [
    {
        "segment_id": 10,
        "partition_id": 1,
        "row_count": 2,
        "storage_version": "StorageV3",
        "manifest_path": "segments/10",
        "manifest_version": "1",
    }
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_path = Path(tmp) / "snapshot.json"
        snapshot = rm.create_snapshot(
            SCHEMA,
            SEGMENTS,
            output_path=snapshot_path,
            collection_name="demo_collection",
        )
        info = rm.inspect_snapshot(str(snapshot_path), storage=rm.StorageConfig())

        print(f"snapshot: {snapshot_path}")
        print(f"collection: {snapshot['collection_name']}")
        print(f"segments: {info.segment_count}")


if __name__ == "__main__":
    main()
