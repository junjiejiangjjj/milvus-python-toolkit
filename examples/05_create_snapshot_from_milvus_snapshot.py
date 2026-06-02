import os
import tempfile
from pathlib import Path

import ray_milvus as rm

REQUIRED_ENV = ("MILVUS_URI", "MILVUS_COLLECTION", "MILVUS_SNAPSHOT")


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print("live Milvus snapshot demo skipped")
        print("set " + ", ".join(missing) + " to run it")
        return

    storage = rm.StorageConfig(
        storage_type=os.environ.get("MILVUS_STORAGE_TYPE", "s3"),
        endpoint=os.environ.get("MILVUS_S3_ENDPOINT"),
        bucket=os.environ.get("MILVUS_S3_BUCKET"),
        access_key=os.environ.get("MILVUS_S3_ACCESS_KEY"),
        secret_key=os.environ.get("MILVUS_S3_SECRET_KEY"),
        region=os.environ.get("MILVUS_S3_REGION"),
        use_ssl=os.environ.get("MILVUS_S3_USE_SSL", "true").lower() == "true",
        root_path=os.environ.get("MILVUS_STORAGE_ROOT"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "snapshot.json"
        payload = rm.create_snapshot_from_milvus(
            uri=os.environ["MILVUS_URI"],
            collection_name=os.environ["MILVUS_COLLECTION"],
            snapshot_name=os.environ["MILVUS_SNAPSHOT"],
            output_path=output_path,
            storage=storage,
            token=os.environ.get("MILVUS_TOKEN"),
            db_name=os.environ.get("MILVUS_DB_NAME"),
            overwrite=True,
        )
        print(f"snapshot: {output_path}")
        print(f"collection: {payload['collection_name']}")
        print(f"segments: {len(payload['segments'])}")


if __name__ == "__main__":
    main()
