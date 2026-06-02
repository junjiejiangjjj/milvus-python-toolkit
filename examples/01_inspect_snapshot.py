from pathlib import Path

import ray_milvus as rm

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "fixtures" / "snapshot_storage_v3.json"


def main() -> None:
    ray = rm.RayMilvus(storage=rm.StorageConfig(endpoint="localhost:9000", bucket="bucket"))
    info = ray.inspect_snapshot(str(SNAPSHOT))

    print(f"collection: {info.collection_name}")
    print(f"segments: {info.segment_count}")
    print("fields:")
    for field in info.schema.fields:
        print(f"  - {field.name} ({field.data_type}, id={field.field_id})")
    if info.diagnostics:
        print("diagnostics:")
        for diagnostic in info.diagnostics:
            print(f"  - {diagnostic.level}: {diagnostic.message}")


if __name__ == "__main__":
    main()
