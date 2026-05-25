from pathlib import Path

ROOT = Path(__file__).parents[2]
CORE = ROOT / "milvus_toolkit" / "core"
CLI = ROOT / "milvus_toolkit" / "cli"
IO = ROOT / "milvus_toolkit" / "io"

CORE_FORBIDDEN = (
    "milvus_toolkit.io",
    "milvus_toolkit.engines",
    "import ray",
    "import daft",
    "import pandas",
    "import polars",
    "import duckdb",
    "milvus_storage",
)


def test_core_does_not_import_io_engines_or_optional_frameworks():
    for path in CORE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in CORE_FORBIDDEN:
            assert forbidden not in source, f"{path} imports forbidden dependency {forbidden}"


def test_cli_uses_public_api_not_storage_internals():
    for path in CLI.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "milvus_toolkit.io" not in source


def test_io_does_not_import_engines():
    for path in IO.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "milvus_toolkit.engines" not in source


def test_milvus_storage_import_is_isolated_to_io():
    for path in (ROOT / "milvus_toolkit").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "import milvus_storage" in source or "_vendor import milvus_storage" in source:
            assert IO in path.parents, f"{path} references milvus_storage outside io/"
