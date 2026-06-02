import pytest

from ray_milvus.core.schema import parse_schema, project_fields
from ray_milvus.errors import SchemaError


def test_parse_schema_and_project_fields():
    schema = parse_schema(
        {
            "name": "demo",
            "fields": [
                {"name": "id", "field_id": 1, "data_type": "Int64"},
                {"name": "vector", "field_id": 2, "data_type": "FloatVector"},
            ],
        }
    )

    projected = project_fields(schema, ("vector",))

    assert schema.collection_name == "demo"
    assert [field.name for field in projected] == ["vector"]
    assert [field.field_id for field in projected] == [2]


def test_project_fields_rejects_unknown_column():
    schema = parse_schema(
        {
            "name": "demo",
            "fields": [{"name": "id", "field_id": 1, "data_type": "Int64"}],
        }
    )

    with pytest.raises(SchemaError, match="missing"):
        project_fields(schema, ("missing",))
