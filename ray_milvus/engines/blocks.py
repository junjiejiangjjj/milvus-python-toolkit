from __future__ import annotations

import re
from collections.abc import Iterable

import pyarrow as pa

from ray_milvus.errors import ConfigError

DEFAULT_TARGET_BLOCK_SIZE = 128 * 1024 * 1024

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}
_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*([a-zA-Z]*)\s*$")


def parse_target_block_size(value: int | str | None) -> int:
    if value is None:
        return DEFAULT_TARGET_BLOCK_SIZE

    if isinstance(value, int):
        if value <= 0:
            raise ConfigError("target_block_size must be greater than 0")
        return value

    if isinstance(value, str):
        match = _SIZE_PATTERN.match(value)
        if match is None:
            raise ConfigError(f"Invalid target_block_size {value!r}")
        amount = int(match.group(1))
        unit = match.group(2).lower()
        multiplier = _SIZE_UNITS.get(unit)
        if amount <= 0 or multiplier is None:
            raise ConfigError(f"Invalid target_block_size {value!r}")
        return amount * multiplier

    raise ConfigError("target_block_size must be an int, string, or None")


def iter_arrow_blocks(
    batches: Iterable[pa.RecordBatch],
    target_block_size: int,
) -> Iterable[pa.Table]:
    if target_block_size <= 0:
        raise ConfigError("target_block_size must be greater than 0")

    pending: list[pa.RecordBatch] = []
    pending_bytes = 0

    for batch in batches:
        if not isinstance(batch, pa.RecordBatch):
            raise ConfigError("iter_arrow_blocks expects pyarrow.RecordBatch inputs")
        pending.append(batch)
        pending_bytes += batch.nbytes
        if pending_bytes >= target_block_size:
            yield pa.Table.from_batches(pending)
            pending = []
            pending_bytes = 0

    if pending:
        yield pa.Table.from_batches(pending)
