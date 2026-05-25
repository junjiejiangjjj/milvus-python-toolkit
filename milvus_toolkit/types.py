from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "milvus_storage"
    storage_type: str = "local"
    endpoint: str | None = None
    bucket: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    use_ssl: bool = True
    region: str | None = None
    root_path: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadOptions:
    columns: tuple[str, ...] | None = None
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldSchema:
    name: str
    field_id: int
    data_type: str
    is_primary: bool = False
    nullable: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MilvusSchema:
    collection_name: str | None
    fields: tuple[FieldSchema, ...]

    def field_by_name(self, name: str) -> FieldSchema | None:
        for field_schema in self.fields:
            if field_schema.name == name:
                return field_schema
        return None


@dataclass(frozen=True)
class SegmentMetadata:
    segment_id: int
    partition_id: int | None
    row_count: int | None
    storage_version: str | None
    manifest_path: str | None
    manifest_version: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionDiagnostic:
    level: str
    message: str
    segment_id: int | None = None


@dataclass(frozen=True)
class InspectionResult:
    collection_name: str | None
    schema: MilvusSchema
    segments: tuple[SegmentMetadata, ...]
    diagnostics: tuple[InspectionDiagnostic, ...] = ()

    @property
    def segment_count(self) -> int:
        return len(self.segments)
