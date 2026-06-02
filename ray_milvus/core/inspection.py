from __future__ import annotations

from typing import Any

from ray_milvus.errors import ManifestError, UnsupportedSegmentError
from ray_milvus.types import InspectionDiagnostic, InspectionResult

from .manifest import validate_storage_v3_manifest
from .snapshot import SnapshotMetadata, parse_snapshot


def inspect_snapshot_metadata(snapshot: SnapshotMetadata | dict[str, Any]) -> InspectionResult:
    snapshot_metadata = parse_snapshot(snapshot) if isinstance(snapshot, dict) else snapshot
    diagnostics = []
    for segment in snapshot_metadata.segments:
        try:
            validate_storage_v3_manifest(segment)
        except (ManifestError, UnsupportedSegmentError) as exc:
            diagnostics.append(
                InspectionDiagnostic(
                    level="error",
                    message=str(exc),
                    segment_id=segment.segment_id,
                )
            )

    return InspectionResult(
        collection_name=snapshot_metadata.collection_name,
        schema=snapshot_metadata.schema,
        segments=snapshot_metadata.segments,
        diagnostics=tuple(diagnostics),
    )
