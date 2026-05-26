import sys
from types import SimpleNamespace

import pytest

from milvus_toolkit.errors import ConfigError
from milvus_toolkit.io.milvus_client import load_collection_schema, normalize_collection_schema


def test_normalize_collection_schema_from_dict_shape():
    schema = normalize_collection_schema(
        {
            "schema": {
                "fields": [
                    {
                        "name": "id",
                        "fieldID": 100,
                        "type": "DataType.INT64",
                        "is_primary": True,
                    },
                    {
                        "name": "vector",
                        "field_id": 101,
                        "data_type": "FLOAT_VECTOR",
                        "params": {"dim": "2"},
                    },
                ]
            }
        },
        collection_name="demo_collection",
    )

    assert schema == {
        "name": "demo_collection",
        "fields": [
            {
                "name": "id",
                "field_id": 100,
                "data_type": "Int64",
                "is_primary": True,
                "nullable": True,
                "params": {},
            },
            {
                "name": "vector",
                "field_id": 101,
                "data_type": "FloatVector",
                "is_primary": False,
                "nullable": True,
                "params": {"dim": "2"},
            },
        ],
    }


def test_normalize_collection_schema_from_object_shape():
    schema = normalize_collection_schema(
        SimpleNamespace(
            name="demo_collection",
            fields=[
                SimpleNamespace(
                    name="id",
                    field_id=100,
                    dtype=SimpleNamespace(name="INT64"),
                    is_primary=True,
                    nullable=False,
                    params={},
                ),
                SimpleNamespace(
                    name="text",
                    field_id=101,
                    dtype=SimpleNamespace(name="VARCHAR"),
                    nullable=True,
                    params={},
                    max_length=64,
                ),
            ],
        )
    )

    assert schema["name"] == "demo_collection"
    assert schema["fields"][0]["data_type"] == "Int64"
    assert schema["fields"][0]["nullable"] is False
    assert schema["fields"][1]["data_type"] == "VarChar"
    assert schema["fields"][1]["params"] == {"max_length": 64}


def test_load_collection_schema_reports_missing_pymilvus(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)

    with pytest.raises(ConfigError, match="PyMilvus is required"):
        load_collection_schema("http://localhost:19530", "demo_collection")
