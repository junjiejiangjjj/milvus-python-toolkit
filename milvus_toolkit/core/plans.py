from __future__ import annotations

from dataclasses import dataclass

from milvus_toolkit.types import FieldSchema, MilvusSchema, SegmentMetadata, StorageConfig


@dataclass(frozen=True)
class SegmentReadTask:
    segment: SegmentMetadata
    schema: MilvusSchema
    projected_fields: tuple[FieldSchema, ...]
    include: tuple[str, ...]
    storage: StorageConfig
    manifest_version: str | None = None

    @property
    def manifest_path(self) -> str:
        assert self.segment.manifest_path is not None
        return self.segment.manifest_path


@dataclass(frozen=True)
class ReadPlan:
    schema: MilvusSchema
    tasks: tuple[SegmentReadTask, ...]
    projected_fields: tuple[FieldSchema, ...]
    include: tuple[str, ...]
    storage: StorageConfig
    manifest_version: str | None = None
