from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from ray_milvus.errors import UnsupportedFeatureError

from .plans import ReadPlan


@dataclass(frozen=True)
class MilvusDataset:
    read_plan: ReadPlan
    _iter_batches: Callable[[int | None], Iterable[pa.RecordBatch]]
    _to_ray_blocks: Callable[[int | str | None, int | None], list[Any]] | None = None

    def iter_batches(self, batch_size: int | None = None) -> Iterable[pa.RecordBatch]:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        yield from self._iter_batches(batch_size)

    def to_arrow(self) -> pa.Table:
        batches = list(self.iter_batches())
        return pa.Table.from_batches(batches) if batches else pa.table({})

    def to_ray_blocks(
        self,
        target_block_size: int | str | None = None,
        parallelism: int | None = None,
    ) -> list[Any]:
        if self._to_ray_blocks is None:
            raise UnsupportedFeatureError("Ray Core execution is not configured for this dataset")
        return self._to_ray_blocks(target_block_size, parallelism)
