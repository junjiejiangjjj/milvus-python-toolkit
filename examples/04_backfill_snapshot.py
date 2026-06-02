import json
import tempfile
from pathlib import Path

import pyarrow as pa

import ray_milvus as rm

SOURCE_SCHEMA = {
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
    ],
}

BACKFILL_SCHEMA = {
    "name": "demo_collection",
    "fields": [
        {
            "name": "id",
            "field_id": 100,
            "data_type": "Int64",
            "is_primary": True,
            "nullable": False,
        },
        {"name": "age", "field_id": 102, "data_type": "Int64"},
    ],
}


def main() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": ["alice", "bob", "carol"],
        }
    )
    backfill = pa.table(
        {
            "id": pa.array([1, 3], type=pa.int64()),
            "age": pa.array([30, 28], type=pa.int64()),
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = rm.StorageConfig(storage_type="local", root_path=str(root))
        source_snapshot_path = root / "source_snapshot.json"
        backfilled_snapshot_path = root / "backfilled_snapshot.json"
        segment_path = root / "segments" / "10"

        try:
            source_snapshot = rm.write_snapshot(
                source,
                SOURCE_SCHEMA,
                storage,
                segment_path=str(segment_path),
                segment_id=10,
                collection_name="demo_collection",
                partition_id=1,
            )
            source_snapshot_path.write_text(json.dumps(source_snapshot), encoding="utf-8")
            rm.backfill_snapshot(
                str(source_snapshot_path),
                storage,
                backfill,
                BACKFILL_SCHEMA,
                primary_key="id",
                fields=["age"],
                output_path=backfilled_snapshot_path,
                mode="replace",
                overwrite=True,
            )
            result = rm.read_snapshot(
                str(backfilled_snapshot_path),
                storage=storage,
                columns=["age"],
                include=["segment_id", "row_offset"],
            ).to_arrow()
        except rm.StorageError as exc:
            print(f"backfill demo skipped: {exc}")
            print("run scripts/install_dev.sh to build the bundled milvus-storage native library")
            return

        print(result.to_pydict())


if __name__ == "__main__":
    main()
