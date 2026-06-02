import json
import tempfile
from pathlib import Path

import pyarrow as pa

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


def main() -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": ["alice", "bob"],
            "vector": [[0.1, 0.2], [0.3, 0.4]],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshot_path = root / "snapshot.json"
        segment_path = root / "segments" / "10"
        storage = rm.StorageConfig(storage_type="local", root_path=str(root))

        try:
            snapshot = rm.write_snapshot(
                table,
                SCHEMA,
                storage,
                segment_path=str(segment_path),
                segment_id=10,
                collection_name="demo_collection",
                partition_id=1,
            )
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            result = rm.read_snapshot(
                str(snapshot_path),
                storage=storage,
                columns=["id", "name", "vector"],
                include=["segment_id", "row_offset"],
            ).to_arrow()
        except rm.StorageError as exc:
            print(f"storage demo skipped: {exc}")
            print("run scripts/install_dev.sh to build the bundled milvus-storage native library")
            return

        print(result.to_pydict())


if __name__ == "__main__":
    main()
