import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from milvus_toolkit.core.inspection import inspect_snapshot_metadata
from milvus_toolkit.core.native_snapshot import build_snapshot_payload_from_native_snapshot
from milvus_toolkit.errors import ConfigError, SnapshotError

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "native_snapshot"
METADATA = FIXTURE_ROOT / "123" / "metadata" / "456.json"
MANIFEST_DIR = FIXTURE_ROOT / "123" / "manifests" / "456"


def test_import_native_snapshot_from_metadata_references():
    payload = build_snapshot_payload_from_native_snapshot(METADATA, manifest_dir=MANIFEST_DIR)

    assert payload == {
        "collection_name": "demo_collection",
        "collection_schema": {
            "name": "demo_collection",
            "fields": [
                {
                    "name": "id",
                    "field_id": 100,
                    "data_type": "Int64",
                    "is_primary": True,
                    "nullable": False,
                    "params": {},
                },
                {
                    "name": "vector",
                    "field_id": 101,
                    "data_type": "FloatVector",
                    "is_primary": False,
                    "nullable": True,
                    "params": {"dim": "2"},
                },
            ],
        },
        "segments": [
            {
                "segment_id": 10,
                "partition_id": 1,
                "row_count": 2,
                "storage_version": "StorageV3",
                "manifest_path": str(MANIFEST_DIR / "10.avro"),
                "manifest_version": "v1",
            }
        ],
    }
    inspected = inspect_snapshot_metadata(payload)
    assert inspected.collection_name == "demo_collection"
    assert inspected.segment_count == 1


def test_import_native_snapshot_from_structured_paths():
    payload = build_snapshot_payload_from_native_snapshot(
        snapshot_root=FIXTURE_ROOT,
        collection_id=123,
        snapshot_id=456,
    )

    assert payload["segments"][0]["segment_id"] == 10


def test_import_native_snapshot_scans_manifest_dir(tmp_path):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"not-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="fastavro is required"):
        build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)


def test_import_native_snapshot_reads_avro_manifest_record(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"fake-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )

    fastavro = ModuleType("fastavro")

    def reader(_file):
        return [
            {
                "segment_id": 10,
                "partition_id": 1,
                "num_of_rows": 2,
                "storage_version": 3,
                "transaction_path": "segments/10",
                "manifest_version": "v1",
                "binlog_files": [
                    {
                        "field_id": 100,
                        "binlogs": [
                            {
                                "log_id": 1,
                                "log_path": "segments/10/100/1",
                                "entries_num": 2,
                            }
                        ],
                    }
                ],
            }
        ]

    fastavro.reader = reader
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    payload = build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)

    assert payload["segments"] == [
        {
            "segment_id": 10,
            "partition_id": 1,
            "row_count": 2,
            "storage_version": "StorageV3",
            "manifest_path": "segments/10",
            "manifest_version": "v1",
        }
    ]


def test_import_native_snapshot_metadata_overrides_avro_manifest(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"fake-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                },
                "segments": [
                    {
                        "segment_id": 10,
                        "row_count": 5,
                        "storage_version": "StorageV3",
                        "manifest_path": "10.avro",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fastavro = ModuleType("fastavro")
    fastavro.reader = lambda _file: [
        {
            "segment_id": 10,
            "partition_id": 1,
            "num_of_rows": 2,
            "storage_version": 3,
        }
    ]
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    payload = build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)

    assert payload["segments"] == [
        {
            "segment_id": 10,
            "partition_id": 1,
            "row_count": 5,
            "storage_version": "StorageV3",
            "manifest_path": str(manifest_dir / "10.avro"),
            "manifest_version": None,
        }
    ]



def test_import_native_snapshot_rejects_invalid_binlog_files(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"fake-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )
    fastavro = ModuleType("fastavro")
    fastavro.reader = lambda _file: [
        {
            "segment_id": 10,
            "partition_id": 1,
            "num_of_rows": 2,
            "storage_version": 3,
            "binlog_files": "broken",
        }
    ]
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    with pytest.raises(SnapshotError, match="binlog_files must be a list"):
        build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)



def test_import_native_snapshot_reads_schemaless_avro_manifest_record(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"schemaless-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )
    fastavro = ModuleType("fastavro")

    def schemaless_reader(_file, schema):
        assert schema["name"] == "ManifestEntry"
        assert schema["fields"][-1]["name"] == "binlog_files"
        return {
            "segment_id": 10,
            "partition_id": 1,
            "num_of_rows": 2,
            "storage_version": 3,
            "binlog_files": [
                {
                    "field_id": 100,
                    "binlogs": [
                        {
                            "entries_num": 2,
                            "timestamp_from": 0,
                            "timestamp_to": 1,
                            "log_path": "segments/10/100/1",
                            "log_size": 128,
                            "log_id": 1,
                            "memory_size": 256,
                        }
                    ],
                }
            ],
        }

    def reader(_file):
        raise AssertionError("OCF reader should not be used after schemaless success")

    fastavro.schemaless_reader = schemaless_reader
    fastavro.reader = reader
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    payload = build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)

    assert payload["segments"] == [
        {
            "segment_id": 10,
            "partition_id": 1,
            "row_count": 2,
            "storage_version": "StorageV3",
            "manifest_path": str(manifest_dir / "10.avro"),
            "manifest_version": None,
        }
    ]



def test_import_native_snapshot_falls_back_to_ocf_avro_reader(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "10.avro").write_bytes(b"ocf-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )
    fastavro = ModuleType("fastavro")
    fastavro.schemaless_reader = lambda _file, _schema: (_ for _ in ()).throw(
        ValueError("not schemaless")
    )
    fastavro.reader = lambda _file: [
        {
            "segment_id": 10,
            "partition_id": 1,
            "num_of_rows": 2,
            "storage_version": 3,
        }
    ]
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    payload = build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)

    assert payload["segments"][0]["row_count"] == 2
    assert payload["segments"][0]["storage_version"] == "StorageV3"



def test_import_native_snapshot_reports_avro_decode_failures(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "10.avro"
    manifest_path.write_bytes(b"broken-avro")
    metadata.write_text(
        json.dumps(
            {
                "collection_schema": {
                    "name": "demo_collection",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                }
            }
        ),
        encoding="utf-8",
    )
    fastavro = ModuleType("fastavro")
    fastavro.schemaless_reader = lambda _file, _schema: (_ for _ in ()).throw(
        ValueError("schemaless failed")
    )
    fastavro.reader = lambda _file: (_ for _ in ()).throw(ValueError("ocf failed"))
    monkeypatch.setitem(sys.modules, "fastavro", fastavro)

    with pytest.raises(SnapshotError, match=f"{manifest_path}.*schemaless.*ocf"):
        build_snapshot_payload_from_native_snapshot(metadata, manifest_dir=manifest_dir)



def test_import_native_snapshot_requires_enough_paths():
    with pytest.raises(SnapshotError, match="requires metadata_path"):
        build_snapshot_payload_from_native_snapshot()
