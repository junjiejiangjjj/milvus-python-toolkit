from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from milvus_toolkit.core.schema import parse_schema
from milvus_toolkit.errors import ConfigError, SchemaError


def load_collection_schema(
    uri: str,
    collection_name: str,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
) -> dict[str, Any]:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise ConfigError(
            "PyMilvus is required for --schema-from-milvus. "
            "Install it with `pip install pymilvus` or `pip install milvus-toolkit[pymilvus]`."
        ) from exc

    kwargs: dict[str, str] = {}
    if token is not None:
        kwargs["token"] = token
    if user is not None:
        kwargs["user"] = user
    if password is not None:
        kwargs["password"] = password
    if db_name is not None:
        kwargs["db_name"] = db_name

    try:
        client = MilvusClient(uri=uri, **kwargs)
        schema = client.describe_collection(collection_name=collection_name)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to load Milvus collection schema: {exc}") from exc
    return normalize_collection_schema(schema, collection_name=collection_name)


def normalize_collection_schema(schema: Any, collection_name: str | None = None) -> dict[str, Any]:
    schema_mapping = _as_mapping(schema)
    schema_data = _schema_data(schema_mapping, schema)
    schema_name = (
        collection_name or _get_value(schema_data, "name") or _get_value(schema_mapping, "name")
    )
    fields_data = _get_value(schema_data, "fields") or _get_value(schema_mapping, "fields")
    if not isinstance(fields_data, list) or not fields_data:
        raise SchemaError("Milvus collection schema must contain a non-empty fields list")

    normalized = {
        "name": schema_name,
        "fields": [_normalize_field(field) for field in fields_data],
    }
    parse_schema({"collection_name": schema_name, "collection_schema": normalized})
    return normalized


def _schema_data(schema_mapping: Mapping[str, Any], schema: Any) -> Any:
    schema_data = schema_mapping.get("schema") or schema_mapping.get("collection_schema")
    if schema_data is not None:
        return schema_data
    schema_attr = getattr(schema, "schema", None)
    if schema_attr is not None:
        return schema_attr
    return schema


def _normalize_field(field: Any) -> dict[str, Any]:
    field_mapping = _as_mapping(field)
    name = _get_value(field, "name")
    field_id = _first_value(field, ("field_id", "fieldID", "id"))
    data_type = _normalize_data_type(
        _first_value(field, ("data_type", "dataType", "type", "dtype"))
    )
    if name is None or field_id is None or data_type is None:
        raise SchemaError("Milvus schema fields must include name, field_id, and data_type")

    params = _field_params(field, field_mapping)
    normalized = {
        "name": str(name),
        "field_id": int(field_id),
        "data_type": data_type,
        "is_primary": bool(_first_value(field, ("is_primary", "isPrimary", "primary_key"), False)),
        "nullable": bool(_get_value(field, "nullable", True)),
        "params": params,
    }
    return normalized


def _field_params(field: Any, field_mapping: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(_get_value(field, "params", {}) or {})
    for key in ("dim", "max_length", "max_capacity", "element_type"):
        value = _get_value(field, key)
        if value is not None:
            params[key] = value
    type_params = field_mapping.get("type_params")
    if isinstance(type_params, Mapping):
        params.update(type_params)
    return params


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _first_value(value: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        result = _get_value(value, name)
        if result is not None:
            return result
    return default


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    return getattr(value, name, default)


def _normalize_data_type(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is not None:
        return _DATA_TYPE_NAMES.get(str(name).upper(), str(name))
    text = str(value)
    token = text.rsplit(".", maxsplit=1)[-1].upper()
    return _DATA_TYPE_NAMES.get(token, text)


_DATA_TYPE_NAMES = {
    "BOOL": "Bool",
    "INT8": "Int8",
    "INT16": "Int16",
    "INT32": "Int32",
    "INT64": "Int64",
    "FLOAT": "Float",
    "DOUBLE": "Double",
    "VARCHAR": "VarChar",
    "JSON": "JSON",
    "ARRAY": "Array",
    "BINARY_VECTOR": "BinaryVector",
    "FLOAT_VECTOR": "FloatVector",
    "FLOAT16_VECTOR": "Float16Vector",
    "BFLOAT16_VECTOR": "BFloat16Vector",
    "SPARSE_FLOAT_VECTOR": "SparseFloatVector",
}
