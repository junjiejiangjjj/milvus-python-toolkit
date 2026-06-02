import json
from pathlib import Path

import pytest

from milvus_toolkit.core.snapshot import build_snapshot_payload, parse_snapshot
from milvus_toolkit.errors import SnapshotError
from milvus_toolkit.io.object_store import load_snapshot_json

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


def test_parse_snapshot_fixture():
    snapshot = parse_snapshot(load_snapshot_json(str(FIXTURE)))

    assert snapshot.collection_name == "demo_collection"
    assert snapshot.schema.fields[0].name == "id"
    assert snapshot.segments[0].segment_id == 10
    assert snapshot.segments[0].manifest_path == "segments/10/manifest.json"


def test_build_snapshot_payload_emits_canonical_shape():
    payload = build_snapshot_payload(
        {
            "name": "demo_collection",
            "fields": [
                {
                    "name": "id",
                    "fieldID": 100,
                    "dataType": "Int64",
                    "isPrimary": True,
                    "nullable": False,
                }
            ],
        },
        {
            "segments": [
                {
                    "id": "10",
                    "partition_id": "1",
                    "row_count": "2",
                    "storageVersion": "StorageV3",
                    "manifest": {"path": "segments/10", "version": "v1"},
                }
            ]
        },
    )

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
                }
            ],
        },
        "segments": [
            {
                "segment_id": 10,
                "partition_id": 1,
                "row_count": 2,
                "storage_version": "StorageV3",
                "manifest_path": "segments/10",
                "manifest_version": "v1",
            }
        ],
    }
    assert parse_snapshot(payload).segments[0].segment_id == 10


def test_build_snapshot_payload_rejects_invalid_segments():
    with pytest.raises(SnapshotError, match="segments input"):
        build_snapshot_payload(
            {"name": "demo", "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}]},
            {"not_segments": []},
        )



def test_parse_snapshot_storage_v3_manifest_list():
    snapshot = parse_snapshot(
        {
            "snapshot_info": {"collection_id": 100},
            "collection": {
                "name": "demo",
                "schema": {
                    "name": "demo",
                    "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
                },
            },
            "storagev2_manifest_list": [
                {
                    "segmentID": "300",
                    "rowCount": "5",
                    "manifest": json.dumps(
                        {"ver": 7, "base_path": "files/insert_log/100/200/300"}
                    ),
                }
            ],
        }
    )

    assert snapshot.collection_name == "demo"
    assert snapshot.segments[0].segment_id == 300
    assert snapshot.segments[0].partition_id == 200
    assert snapshot.segments[0].row_count == 5
    assert snapshot.segments[0].storage_version == "StorageV3"
    assert snapshot.segments[0].manifest_path == "files/insert_log/100/200/300"
    assert snapshot.segments[0].manifest_version == "7"



def test_parse_snapshot_legacy_manifest_list_derives_transaction_path():
    snapshot = parse_snapshot(
        {
            "collection_schema": {
                "name": "demo",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "snapshot_info": {"collection_id": "100", "partition_ids": ["200"]},
            "manifest_list": ["files/snapshots/100/manifests/999/300.avro"],
            "segment_ids": ["300"],
        }
    )

    assert snapshot.segments[0].segment_id == 300
    assert snapshot.segments[0].partition_id == 200
    assert snapshot.segments[0].storage_version == "StorageV3"
    assert snapshot.segments[0].manifest_path == "files/insert_log/100/200/300"




def test_parse_snapshot_storage_v3_manifest_rejects_invalid_json():
    with pytest.raises(SnapshotError, match="not valid JSON"):
        parse_snapshot(
            {
                "collection_schema": {
                    "name": "demo",
                    "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
                },
                "storagev2_manifest_list": [
                    {"segmentID": 300, "manifest": "not-json"}
                ],
            }
        )
