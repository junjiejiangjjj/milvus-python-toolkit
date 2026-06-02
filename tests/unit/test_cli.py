import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from milvus_toolkit.cli.main import main

FIXTURE = Path(__file__).parents[1] / "fixtures" / "snapshot_storage_v3.json"


def test_cli_help(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "milvus-toolkit" in output


def test_cli_inspect_json(capsys):
    exit_code = main(
        [
            "inspect",
            "--snapshot",
            str(FIXTURE),
            "--s3-endpoint",
            "localhost:9000",
            "--s3-bucket",
            "bucket",
            "--json",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"collection_name": "demo_collection"' in output
    assert '"segment_count"' not in output


def test_cli_create_snapshot_and_inspect_round_trip(tmp_path, capsys):
    schema_path = tmp_path / "schema.json"
    segments_path = tmp_path / "segments.json"
    snapshot_path = tmp_path / "snapshot.json"
    schema_path.write_text(
        json.dumps(
            {"name": "demo", "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}]}
        ),
        encoding="utf-8",
    )
    segments_path.write_text(
        json.dumps(
            [{"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "segment-10"}]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-snapshot",
            "--schema-file",
            str(schema_path),
            "--segments-file",
            str(segments_path),
            "--output",
            str(snapshot_path),
            "--collection-name",
            "demo_collection",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Created snapshot" in output
    assert snapshot_path.exists()

    exit_code = main(["inspect", "--snapshot", str(snapshot_path), "--json"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"collection_name": "demo_collection"' in output


def test_cli_create_snapshot_reports_errors(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    exit_code = main(
        [
            "create-snapshot",
            "--schema-file",
            str(missing),
            "--segments-file",
            str(missing),
            "--output",
            str(tmp_path / "snapshot.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error:" in captured.err


def test_cli_create_snapshot_from_milvus(tmp_path, capsys, monkeypatch):
    segments_path = tmp_path / "segments.json"
    snapshot_path = tmp_path / "snapshot.json"
    segments_path.write_text(
        json.dumps(
            [{"segment_id": 10, "storage_version": "StorageV3", "manifest_path": "segment-10"}]
        ),
        encoding="utf-8",
    )

    def create_snapshot_from_milvus(**kwargs):
        assert kwargs["uri"] == "http://localhost:19530"
        assert kwargs["collection_name"] == "demo_collection"
        assert kwargs["token"] == "secret"
        return {
            "collection_name": "demo_collection",
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10}],
        }

    monkeypatch.setattr("milvus_toolkit.create_snapshot_from_milvus", create_snapshot_from_milvus)

    exit_code = main(
        [
            "create-snapshot",
            "--schema-from-milvus",
            "--uri",
            "http://localhost:19530",
            "--collection-name",
            "demo_collection",
            "--segments-file",
            str(segments_path),
            "--output",
            str(snapshot_path),
            "--token",
            "secret",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Created snapshot" in output


def test_cli_create_snapshot_from_milvus_requires_uri(tmp_path, capsys):
    exit_code = main(
        [
            "create-snapshot",
            "--schema-from-milvus",
            "--collection-name",
            "demo_collection",
            "--segments-file",
            str(tmp_path / "segments.json"),
            "--output",
            str(tmp_path / "snapshot.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--uri is required" in captured.err


def test_cli_import_milvus_snapshot(tmp_path, capsys, monkeypatch):
    output_path = tmp_path / "snapshot.json"

    def import_milvus_snapshot(**kwargs):
        assert kwargs["metadata_path"] == "metadata.json"
        assert kwargs["manifest_dir"] == "manifests"
        assert kwargs["output_path"] == str(output_path)
        return {
            "collection_name": "demo_collection",
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10}],
        }

    monkeypatch.setattr("milvus_toolkit.import_milvus_snapshot", import_milvus_snapshot)

    exit_code = main(
        [
            "import-milvus-snapshot",
            "--metadata",
            "metadata.json",
            "--manifest-dir",
            "manifests",
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Imported Milvus snapshot" in output


def test_cli_create_milvus_snapshot(tmp_path, capsys, monkeypatch):
    output_path = tmp_path / "snapshot.json"

    def create_snapshot_from_milvus_snapshot(**kwargs):
        assert kwargs["uri"] == "http://localhost:19530"
        assert kwargs["collection_name"] == "demo_collection"
        assert kwargs["snapshot_name"] == "snapshot-1"
        assert kwargs["output_path"] == str(output_path)
        assert kwargs["token"] == "secret"
        return {
            "collection_name": "demo_collection",
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
            },
            "segments": [{"segment_id": 10}],
        }

    monkeypatch.setattr(
        "milvus_toolkit.create_snapshot_from_milvus_snapshot",
        create_snapshot_from_milvus_snapshot,
    )

    exit_code = main(
        [
            "create-milvus-snapshot",
            "--uri",
            "http://localhost:19530",
            "--collection-name",
            "demo_collection",
            "--snapshot-name",
            "snapshot-1",
            "--output",
            str(output_path),
            "--token",
            "secret",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Created Milvus snapshot snapshot-1" in output



def test_cli_import_milvus_snapshot_requires_structured_ids(tmp_path, capsys):
    exit_code = main(
        [
            "import-milvus-snapshot",
            "--snapshot-root",
            "snapshots",
            "--output",
            str(tmp_path / "snapshot.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--collection-id and --snapshot-id" in captured.err



def test_cli_write_segment_outputs_metadata(tmp_path, capsys, monkeypatch):
    input_path = tmp_path / "input.parquet"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "segment.json"
    pq.write_table(pa.table({"id": [1, 2]}), input_path)
    schema_path.write_text(
        json.dumps(
            {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]}
        ),
        encoding="utf-8",
    )

    def write_segment(
        table,
        schema,
        storage,
        segment_path,
        segment_id,
        partition_id,
        manifest_version,
    ):
        assert table.num_rows == 2
        assert schema["name"] == "demo"
        assert storage.root_path == str(tmp_path)
        assert storage.endpoint == "localhost:9000"
        assert storage.bucket == "bucket"
        assert storage.access_key == "ak"
        assert storage.secret_key == "sk"
        assert storage.region == "us-east-1"
        assert storage.use_ssl is True
        assert storage.extra == {"fs.use_iam": "true"}
        assert segment_path == "segments/10"
        assert segment_id == 10
        assert partition_id == 1
        assert manifest_version == "v1"
        return {
            "segment_id": 10,
            "partition_id": 1,
            "row_count": 2,
            "storage_version": "StorageV3",
            "manifest_path": "segments/10",
            "manifest_version": "v1",
        }

    monkeypatch.setattr("milvus_toolkit.write_segment", write_segment)

    exit_code = main(
        [
            "write-segment",
            "--input",
            str(input_path),
            "--schema-file",
            str(schema_path),
            "--segment-path",
            "segments/10",
            "--segment-id",
            "10",
            "--partition-id",
            "1",
            "--manifest-version",
            "v1",
            "--storage-root",
            str(tmp_path),
            "--s3-endpoint",
            "localhost:9000",
            "--s3-bucket",
            "bucket",
            "--s3-access-key",
            "ak",
            "--s3-secret-key",
            "sk",
            "--s3-region",
            "us-east-1",
            "--s3-use-ssl",
            "--storage-extra",
            "fs.use_iam=true",
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Wrote segment 10 metadata" in output
    assert json.loads(output_path.read_text(encoding="utf-8"))["row_count"] == 2


def test_cli_backfill_snapshot(tmp_path, capsys, monkeypatch):
    snapshot_path = tmp_path / "source.json"
    backfill_path = tmp_path / "backfill.parquet"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "backfilled.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "collection_name": "demo",
                "collection_schema": {
                    "name": "demo",
                    "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
                },
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.table({"id": [1], "target": ["x"]}), backfill_path)
    schema_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "fields": [
                    {"name": "id", "field_id": 100, "data_type": "Int64"},
                    {"name": "target", "field_id": 101, "data_type": "VarChar"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def backfill_snapshot(
        snapshot_path_arg,
        storage,
        backfill_table,
        schema,
        primary_key,
        fields,
        output_path,
        mode,
        segment_path_template,
        overwrite,
        pretty,
    ):
        assert snapshot_path_arg == str(snapshot_path)
        assert storage.storage_type == "local"
        assert storage.root_path == str(tmp_path)
        assert storage.endpoint == "localhost:9000"
        assert storage.bucket == "bucket"
        assert storage.use_ssl is True
        assert storage.extra == {"fs.use_virtual_host": "true"}
        assert backfill_table.to_pydict() == {"id": [1], "target": ["x"]}
        assert schema["name"] == "demo"
        assert primary_key == "id"
        assert fields == ("target",)
        assert output_path == str(output_path)
        assert mode == "overwrite"
        assert segment_path_template == "backfill/{segment_id}"
        assert overwrite is True
        assert pretty is False
        payload = {
            "collection_name": "backfill",
            "collection_schema": {"name": "backfill", "fields": []},
            "segments": [{"segment_id": 10}],
        }
        Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("milvus_toolkit.backfill_snapshot", backfill_snapshot)

    exit_code = main(
        [
            "backfill-snapshot",
            "--snapshot",
            str(snapshot_path),
            "--backfill",
            str(backfill_path),
            "--schema-file",
            str(schema_path),
            "--primary-key",
            "id",
            "--fields",
            "target",
            "--output",
            str(output_path),
            "--mode",
            "overwrite",
            "--segment-path-template",
            "backfill/{segment_id}",
            "--storage-root",
            str(tmp_path),
            "--s3-endpoint",
            "localhost:9000",
            "--s3-bucket",
            "bucket",
            "--s3-use-ssl",
            "--storage-extra",
            "fs.use_virtual_host=true",
            "--overwrite",
            "--compact",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Backfilled snapshot" in output
    assert json.loads(output_path.read_text(encoding="utf-8"))["segments"][0]["segment_id"] == 10



def test_cli_storage_extra_requires_key_value(capsys):
    exit_code = main(
        [
            "backfill-snapshot",
            "--snapshot",
            "snapshot.json",
            "--backfill",
            "backfill.parquet",
            "--schema-file",
            "schema.json",
            "--primary-key",
            "id",
            "--fields",
            "target",
            "--output",
            "out.json",
            "--storage-extra",
            "broken",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--storage-extra must be KEY=VALUE" in captured.err



def test_cli_write_segment_outputs_snapshot(tmp_path, capsys, monkeypatch):
    input_path = tmp_path / "input.parquet"
    schema_path = tmp_path / "schema.json"
    snapshot_path = tmp_path / "snapshot.json"
    pq.write_table(pa.table({"id": [1, 2]}), input_path)
    schema_path.write_text(
        json.dumps(
            {"name": "demo", "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}]}
        ),
        encoding="utf-8",
    )

    def write_snapshot(*args, **kwargs):
        payload = {
            "collection_name": "demo_collection",
            "collection_schema": {
                "name": "demo_collection",
                "fields": [{"name": "id", "field_id": 100, "data_type": "Int64"}],
            },
            "segments": [
                {
                    "segment_id": 10,
                    "partition_id": None,
                    "row_count": 2,
                    "storage_version": "StorageV3",
                    "manifest_path": "segments/10",
                    "manifest_version": None,
                }
            ],
        }
        snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("milvus_toolkit.write_snapshot", write_snapshot)

    exit_code = main(
        [
            "write-segment",
            "--input",
            str(input_path),
            "--schema-file",
            str(schema_path),
            "--segment-path",
            "segments/10",
            "--segment-id",
            "10",
            "--snapshot-output",
            str(snapshot_path),
            "--collection-name",
            "demo_collection",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Wrote segment 10 and snapshot" in output
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["collection_name"] == "demo_collection"
