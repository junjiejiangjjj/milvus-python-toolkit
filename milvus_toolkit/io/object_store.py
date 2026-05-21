from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from milvus_toolkit.errors import SnapshotError, UnsupportedFeatureError


def load_snapshot_json(snapshot_path: str) -> dict[str, Any]:
    if snapshot_path.startswith(("s3://", "gs://", "az://")):
        raise UnsupportedFeatureError(
            "Remote snapshot loading is not implemented yet; use a local fixture path"
        )

    path = Path(snapshot_path)
    try:
        with path.open(encoding="utf-8") as snapshot_file:
            data = json.load(snapshot_file)
    except FileNotFoundError as exc:
        raise SnapshotError(f"Snapshot file not found: {snapshot_path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Snapshot file is not valid JSON: {snapshot_path}") from exc

    if not isinstance(data, dict):
        raise SnapshotError("Snapshot JSON must be an object")
    return data
