import json
from types import SimpleNamespace

import pytest

from ray_milvus.errors import SnapshotError
from ray_milvus.io.object_store import load_snapshot_json


def test_load_snapshot_json_from_s3(monkeypatch):
    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"collection_schema": {}, "segments": []}).encode("utf-8")

    class FakeFileSystem:
        def open_input_file(self, path):
            assert path == "snapshots/demo.json"
            return FakeFile()

    monkeypatch.setattr(
        "ray_milvus.io.object_store._filesystem_from_uri",
        lambda pafs, uri, **kwargs: (FakeFileSystem(), "snapshots/demo.json"),
    )

    assert load_snapshot_json("s3://bucket/snapshots/demo.json") == {
        "collection_schema": {},
        "segments": [],
    }



def test_load_snapshot_json_from_storage_builds_s3_filesystem(monkeypatch):
    from ray_milvus.io.object_store import load_snapshot_json_from_storage

    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"collection_schema": {}, "segments": []}).encode("utf-8")

    class FakeS3FileSystem:
        def __init__(self, **kwargs):
            assert kwargs == {
                "endpoint_override": "localhost:9000",
                "scheme": "http",
                "access_key": "ak",
                "secret_key": "sk",
                "region": "us-east-1",
            }

        def open_input_file(self, path):
            assert path == "bucket/files/snapshot.json"
            return FakeFile()

    monkeypatch.setattr(
        "ray_milvus.io.object_store._filesystem_from_uri",
        lambda pafs, uri, **kwargs: (
            FakeS3FileSystem(
                endpoint_override=kwargs["endpoint_override"],
                scheme="https" if kwargs["use_ssl"] else "http",
                access_key=kwargs["access_key"],
                secret_key=kwargs["secret_key"],
                region=kwargs["region"],
            ),
            "bucket/files/snapshot.json",
        ),
    )

    assert load_snapshot_json_from_storage(
        "files/snapshot.json",
        storage_type="s3",
        endpoint="localhost:9000",
        bucket="bucket",
        access_key="ak",
        secret_key="sk",
        use_ssl=False,
        region="us-east-1",
    ) == {"collection_schema": {}, "segments": []}


def test_load_snapshot_json_from_s3_rejects_malformed_json(monkeypatch):
    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"not-json"

    fake_filesystem = SimpleNamespace(open_input_file=lambda path: FakeFile())
    monkeypatch.setattr(
        "ray_milvus.io.object_store._filesystem_from_uri",
        lambda pafs, uri, **kwargs: (fake_filesystem, "snapshot.json"),
    )

    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot_json("s3://bucket/snapshot.json")
