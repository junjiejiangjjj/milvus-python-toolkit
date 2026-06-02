from __future__ import annotations

from typing import Any

from ray_milvus.errors import SchemaError
from ray_milvus.types import FieldSchema, MilvusSchema

_SYSTEM_FIELDS = {"RowID", "Timestamp"}


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
        data_type = field_data.get("data_type", field_data.get("dataType", field_data.get("type")))
        if name in _SYSTEM_FIELDS and field_id is None:
            continue
        if not name or field_id is None or data_type is None:
            raise SchemaError("Schema fields must include name, field_id, and data_type")
        fields.append(
            FieldSchema(
                name=str(name),
                field_id=int(field_id),
                data_type=str(data_type),
                is_primary=bool(field_data.get("is_primary", field_data.get("isPrimary", False))),
                nullable=bool(field_data.get("nullable", True)),
                params=_field_params(field_data),
            )
        )

    return MilvusSchema(collection_name=collection_name, fields=tuple(fields))



def _field_params(field_data: dict[str, Any]) -> dict[str, Any]:
    params = dict(field_data.get("params", {}) or {})
    type_params = field_data.get("type_params")
    if isinstance(type_params, dict):
        params.update(type_params)
    elif isinstance(type_params, list):
        for item in type_params:
            if isinstance(item, dict) and "key" in item and "value" in item:
                params[str(item["key"])] = item["value"]
    return params



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
