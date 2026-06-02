from __future__ import annotations

from ray_milvus.errors import ManifestError, UnsupportedSegmentError
from ray_milvus.types import SegmentMetadata

_STORAGE_V3_VALUES = {"storagev3", "storage_v3", "v3", "3", "packedparquet", "packed_parquet"}


def validate_storage_v3_manifest(segment: SegmentMetadata) -> None:
    storage_version = (segment.storage_version or "").replace("-", "_").lower()
    if storage_version not in _STORAGE_V3_VALUES:
        raise UnsupportedSegmentError(
            f"Segment {segment.segment_id} is not StorageV3 manifest-backed"
        )
    if not segment.manifest_path:
        raise ManifestError(f"Segment {segment.segment_id} is missing manifest_path")
