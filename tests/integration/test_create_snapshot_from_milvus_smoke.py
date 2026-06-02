import os
import uuid

import pytest

import ray_milvus as mt


def test_create_snapshot_from_live_milvus_and_read_data_e2e(tmp_path):
    if os.environ.get("MILVUS_SNAPSHOT_E2E") != "1":
        pytest.skip("set MILVUS_SNAPSHOT_E2E=1 with live Milvus and storage config")

    pymilvus = pytest.importorskip("pymilvus")
    pytest.importorskip("ray_milvus._vendor.milvus_storage")
    uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    collection_name = f"toolkit_snapshot_e2e_{uuid.uuid4().hex[:8]}"
    snapshot_name = f"toolkit_snapshot_{uuid.uuid4().hex[:8]}"
    output_path = tmp_path / "snapshot.json"
    client = _pymilvus_connection(pymilvus, uri)

    try:
        client.create_collection(
            collection_name=collection_name,
            dimension=2,
            primary_field_name="id",
            vector_field_name="vector",
            metric_type="L2",
            auto_id=False,
        )
        client.insert(
            collection_name=collection_name,
            data=[
                {"id": 1, "vector": [0.1, 0.2]},
                {"id": 2, "vector": [0.3, 0.4]},
            ],
        )
        client.flush(collection_name=collection_name)

        storage = _storage_config_from_env()
        payload = mt.create_snapshot_from_milvus(
            uri=uri,
            collection_name=collection_name,
            snapshot_name=snapshot_name,
            output_path=output_path,
            storage=storage,
            token=os.environ.get("MILVUS_TOKEN"),
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            db_name=os.environ.get("MILVUS_DB_NAME"),
        )

        assert payload["collection_name"] == collection_name
        assert payload["segments"]
        assert output_path.exists()

        result = mt.read_snapshot(
            str(output_path),
            storage=storage,
            columns=["id", "vector"],
            include=["segment_id", "row_offset"],
        ).to_arrow()

        rows = sorted(result.to_pylist(), key=lambda row: row["id"])
        assert [row["id"] for row in rows] == [1, 2]
        assert [pytest.approx(row["vector"]) for row in rows] == [[0.1, 0.2], [0.3, 0.4]]
        assert all(row["segment_id"] is not None for row in rows)
        assert all(row["row_offset"] is not None for row in rows)
    finally:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)



def _pymilvus_connection(pymilvus, uri: str):
    kwargs = {"uri": uri}
    token = os.environ.get("MILVUS_TOKEN")
    user = os.environ.get("MILVUS_USER")
    password = os.environ.get("MILVUS_PASSWORD")
    db_name = os.environ.get("MILVUS_DB_NAME")
    if token is not None:
        kwargs["token"] = token
    if user is not None:
        kwargs["user"] = user
    if password is not None:
        kwargs["password"] = password
    if db_name is not None:
        kwargs["db_name"] = db_name
    return pymilvus.MilvusClient(**kwargs)



def _storage_config_from_env() -> mt.StorageConfig:
    extra = _storage_extra_from_env()
    root_path = os.environ.get("MILVUS_STORAGE_ROOT")
    if root_path is not None:
        extra.setdefault("fs.root_path", root_path)
    return mt.StorageConfig(
        storage_type=os.environ.get("MILVUS_STORAGE_TYPE", "s3"),
        endpoint=os.environ.get("MILVUS_STORAGE_ENDPOINT"),
        bucket=os.environ.get("MILVUS_STORAGE_BUCKET"),
        access_key=os.environ.get("MILVUS_STORAGE_ACCESS_KEY"),
        secret_key=os.environ.get("MILVUS_STORAGE_SECRET_KEY"),
        region=os.environ.get("MILVUS_STORAGE_REGION"),
        root_path=root_path,
        use_ssl=_env_bool("MILVUS_STORAGE_USE_SSL", True),
        extra=extra,
    )



def _storage_extra_from_env() -> dict[str, str]:
    values = {}
    for item in os.environ.get("MILVUS_STORAGE_EXTRA", "").split(","):
        if not item:
            continue
        key, separator, value = item.partition("=")
        if not separator:
            pytest.skip("MILVUS_STORAGE_EXTRA must be comma-separated KEY=VALUE entries")
        values[key] = value
    return values



def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


