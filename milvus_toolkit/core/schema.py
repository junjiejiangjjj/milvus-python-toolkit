from __future__ import annotations

from typing import Any

from milvus_toolkit.errors import SchemaError
from milvus_toolkit.types import FieldSchema, MilvusSchema


def parse_schema(data: dict[str, Any]) -> MilvusSchema:
    schema_data = data.get("collection_schema", data)
    collection_name = schema_data.get("name") or data.get("collection_name")
    fields_data = schema_data.get("fields")
    if not isinstance(fields_data, list) or not fields_data:
        raise SchemaError("Snapshot collection schema must contain a non-empty fields list")

    fields = []
    for field_data in fields_data:
        if not isinstance(field_data, dict):
            raise SchemaError("Schema field entries must be objects")
        name = field_data.get("name")
        field_id = field_data.get("field_id", field_data.get("fieldID"))
        data_type = field_data.get("data_type", field_data.get("dataType"))
        if not name or field_id is None or data_type is None:
            raise SchemaError("Schema fields must include name, field_id, and data_type")
        fields.append(
            FieldSchema(
                name=str(name),
                field_id=int(field_id),
                data_type=str(data_type),
                is_primary=bool(field_data.get("is_primary", field_data.get("isPrimary", False))),
                nullable=bool(field_data.get("nullable", True)),
                params=dict(field_data.get("params", {})),
            )
        )

    return MilvusSchema(collection_name=collection_name, fields=tuple(fields))


def project_fields(
    schema: MilvusSchema,
    columns: tuple[str, ...] | None,
) -> tuple[FieldSchema, ...]:
    if columns is None:
        return schema.fields

    projected = []
    missing = []
    for column in columns:
        field_schema = schema.field_by_name(column)
        if field_schema is None:
            missing.append(column)
        else:
            projected.append(field_schema)

    if missing:
        raise SchemaError(f"Unknown projected field(s): {', '.join(missing)}")

    return tuple(projected)
