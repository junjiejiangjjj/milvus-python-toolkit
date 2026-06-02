from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ray_milvus.errors import SnapshotError, UnsupportedFeatureError


def load_snapshot_json(snapshot_path: str) -> dict[str, Any]:
    data = _load_snapshot_json_data(snapshot_path)
    if not isinstance(data, dict):
        raise SnapshotError("Snapshot JSON must be an object")
    return data



def load_snapshot_json_from_storage(
    snapshot_path: str,
    storage_type: str,
    endpoint: str | None = None,
    bucket: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    use_ssl: bool = True,
    region: str | None = None,
) -> dict[str, Any]:
    data = _load_remote_json(
        _storage_uri(snapshot_path, storage_type=storage_type, bucket=bucket),
        endpoint_override=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        use_ssl=use_ssl,
        region=region,
    )
    if not isinstance(data, dict):
        raise SnapshotError("Snapshot JSON must be an object")
    return data



def _load_snapshot_json_data(snapshot_path: str) -> Any:
    if snapshot_path.startswith("s3://"):
        return _load_remote_json(snapshot_path)
    if snapshot_path.startswith(("gs://", "az://")):
        raise UnsupportedFeatureError(
            f"Remote snapshot loading is not implemented for {snapshot_path!r}"
        )
    return _load_local_json(snapshot_path)



def _storage_uri(snapshot_path: str, storage_type: str, bucket: str | None) -> str:
    if snapshot_path.startswith(("s3://", "gs://", "az://")):
        return snapshot_path
    if storage_type not in {"s3", "remote"}:
        raise UnsupportedFeatureError(
            f"Remote snapshot loading is not implemented for storage type {storage_type!r}"
        )
    if bucket is None:
        raise SnapshotError("Storage bucket is required to load relative snapshot location")
    return f"s3://{bucket}/{snapshot_path.lstrip('/')}"


def _load_local_json(snapshot_path: str) -> Any:
    path = Path(snapshot_path)
    try:
        with path.open(encoding="utf-8") as snapshot_file:
            return json.load(snapshot_file)
    except FileNotFoundError as exc:
        raise SnapshotError(f"Snapshot file not found: {snapshot_path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Snapshot file is not valid JSON: {snapshot_path}") from exc


def _load_remote_json(
    snapshot_path: str,
    endpoint_override: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    use_ssl: bool = True,
    region: str | None = None,
) -> Any:
    try:
        import pyarrow.fs as pafs
    except ImportError as exc:
        raise UnsupportedFeatureError(
            "pyarrow.fs is required to load remote snapshot JSON files"
        ) from exc

    try:
        filesystem, path = _filesystem_from_uri(
            pafs,
            snapshot_path,
            endpoint_override=endpoint_override,
            access_key=access_key,
            secret_key=secret_key,
            use_ssl=use_ssl,
            region=region,
        )
        with filesystem.open_input_file(path) as snapshot_file:
            payload = snapshot_file.read().decode("utf-8")
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Snapshot file is not valid JSON: {snapshot_path}") from exc
    except Exception as exc:
        raise SnapshotError(f"Failed to load remote snapshot file {snapshot_path}: {exc}") from exc


def _filesystem_from_uri(
    pafs,
    snapshot_path: str,
    endpoint_override: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    use_ssl: bool = True,
    region: str | None = None,
):
    if endpoint_override is None:
        return pafs.FileSystem.from_uri(snapshot_path)

    bucket, path = _split_s3_uri(snapshot_path)
    del bucket
    options = {
        "endpoint_override": endpoint_override,
        "scheme": "https" if use_ssl else "http",
    }
    if access_key is not None:
        options["access_key"] = access_key
    if secret_key is not None:
        options["secret_key"] = secret_key
    if region is not None:
        options["region"] = region
    return pafs.S3FileSystem(**options), path



def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SnapshotError(f"Snapshot path is not an S3 URI: {uri}")
    bucket_and_path = uri.removeprefix("s3://")
    bucket, separator, path = bucket_and_path.partition("/")
    if not bucket or not separator or not path:
        raise SnapshotError(f"Snapshot S3 URI must include bucket and path: {uri}")
    return bucket, f"{bucket}/{path}"
