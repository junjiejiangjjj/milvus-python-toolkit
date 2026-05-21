from pathlib import Path

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
