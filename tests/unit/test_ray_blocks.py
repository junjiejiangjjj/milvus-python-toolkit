import pyarrow as pa
import pytest

from ray_milvus.engines.blocks import (
    DEFAULT_TARGET_BLOCK_SIZE,
    iter_arrow_blocks,
    parse_target_block_size,
)
from ray_milvus.errors import ConfigError


def test_parse_target_block_size_accepts_default_int_and_strings():
    assert parse_target_block_size(None) == DEFAULT_TARGET_BLOCK_SIZE
    assert parse_target_block_size(1024) == 1024
    assert parse_target_block_size("128MiB") == 128 * 1024 * 1024
    assert parse_target_block_size("128MB") == 128 * 1000 * 1000
    assert parse_target_block_size("1GiB") == 1024 * 1024 * 1024
    assert parse_target_block_size("4096") == 4096


@pytest.mark.parametrize("value", [0, -1, "0MiB", "abc", "1TiB", object()])
def test_parse_target_block_size_rejects_invalid_values(value):
    with pytest.raises(ConfigError):
        parse_target_block_size(value)


def test_iter_arrow_blocks_flushes_by_target_size():
    batches = [
        pa.RecordBatch.from_pydict({"id": [1]}),
        pa.RecordBatch.from_pydict({"id": [2]}),
        pa.RecordBatch.from_pydict({"id": [3]}),
    ]
    target_size = batches[0].nbytes + batches[1].nbytes

    blocks = list(iter_arrow_blocks(batches, target_block_size=target_size))

    assert [block["id"].to_pylist() for block in blocks] == [[1, 2], [3]]


def test_iter_arrow_blocks_rejects_invalid_batch_input():
    with pytest.raises(ConfigError, match="RecordBatch"):
        list(iter_arrow_blocks([pa.table({"id": [1]})], target_block_size=1024))
