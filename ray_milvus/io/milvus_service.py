from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ray_milvus.core.schema import parse_schema
from ray_milvus.errors import ConfigError, SchemaError, UnsupportedFeatureError


@dataclass(frozen=True)
class MilvusSnapshotLocation:
    name: str
    location: str


class MilvusService:
    def __init__(
        self,
        uri: str,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        db_name: str | None = None,
    ):
        self.uri = uri
        self.token = token
        self.user = user
        self.password = password
        self.db_name = db_name
        self._client = None

    def load_collection_schema(self, collection_name: str) -> dict[str, Any]:
        try:
            schema = self.client.describe_collection(collection_name=collection_name)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to load Milvus collection schema: {exc}") from exc
        return normalize_collection_schema(schema, collection_name=collection_name)

    def create_snapshot_for_read(
        self,
        collection_name: str,
        snapshot_name: str,
        description: str | None = None,
        compaction_protection_seconds: int | None = None,
    ) -> MilvusSnapshotLocation:
        create_method = _required_method(self.client, ("create_snapshot", "createSnapshot"))
        create_kwargs = {
            "snapshot_name": snapshot_name,
            "collection_name": collection_name,
        }
        if self.db_name is not None:
            create_kwargs["db_name"] = self.db_name
        if description is not None:
            create_kwargs["description"] = description
        if compaction_protection_seconds is not None:
            create_kwargs["compaction_protection_seconds"] = compaction_protection_seconds
        try:
            _call_with_fallback(create_method, create_kwargs)
        except UnsupportedFeatureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to create Milvus snapshot for read: {exc}") from exc
        return self.describe_snapshot_for_read(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
        )

    def describe_snapshot_for_read(
        self,
        collection_name: str,
        snapshot_name: str,
    ) -> MilvusSnapshotLocation:
        describe_method = _required_method(self.client, ("describe_snapshot", "describeSnapshot"))
        try:
            response = _call_with_fallback(
                describe_method,
                {
                    "snapshot_name": snapshot_name,
                    "collection_name": collection_name,
                    **({"db_name": self.db_name} if self.db_name is not None else {}),
                },
            )
        except UnsupportedFeatureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to describe Milvus snapshot for read: {exc}") from exc
        location = _first_value(
            response,
            ("s3Location", "s3_location", "location", "path", "uri"),
        )
        if location is None:
            raise ConfigError("Milvus describe snapshot response did not include s3Location")
        return MilvusSnapshotLocation(name=snapshot_name, location=str(location))

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise ConfigError(
                "PyMilvus is required for Milvus service access. "
                "Install it with `pip install pymilvus` or `pip install ray-milvus[pymilvus]`."
            ) from exc

        kwargs: dict[str, str] = {}
        if self.token is not None:
            kwargs["token"] = self.token
        if self.user is not None:
            kwargs["user"] = self.user
        if self.password is not None:
            kwargs["password"] = self.password
        if self.db_name is not None:
            kwargs["db_name"] = self.db_name
        return MilvusClient(uri=self.uri, **kwargs)


def load_collection_schema(
    uri: str,
    collection_name: str,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
) -> dict[str, Any]:
    return MilvusService(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    ).load_collection_schema(collection_name)


def create_snapshot_for_read(
    uri: str,
    collection_name: str,
    snapshot_name: str,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    description: str | None = None,
    compaction_protection_seconds: int | None = None,
) -> MilvusSnapshotLocation:
    return MilvusService(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    ).create_snapshot_for_read(
        collection_name=collection_name,
        snapshot_name=snapshot_name,
        description=description,
        compaction_protection_seconds=compaction_protection_seconds,
    )


def describe_snapshot_for_read(
    uri: str,
    collection_name: str,
    snapshot_name: str,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
) -> MilvusSnapshotLocation:
    return MilvusService(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
    ).describe_snapshot_for_read(
        collection_name=collection_name,
        snapshot_name=snapshot_name,
    )


def _required_method(client: Any, names: tuple[str, ...]):
    for name in names:
        method = getattr(client, name, None)
        if callable(method):
            return method
    raise UnsupportedFeatureError(
        "PyMilvus client does not expose Milvus snapshot APIs: "
        f"expected one of {', '.join(names)}"
    )


def _call_with_fallback(method, kwargs: dict[str, Any]):
    try:
        return method(**kwargs)
    except TypeError:
        fallback = dict(kwargs)
        if "collection_name" in fallback:
            fallback["collectionName"] = fallback.pop("collection_name")
        if "db_name" in fallback:
            fallback["dbName"] = fallback.pop("db_name")
        if "compaction_protection_seconds" in fallback:
            fallback["compactionProtectionSeconds"] = fallback.pop(
                "compaction_protection_seconds"
            )
        return method(**fallback)


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
