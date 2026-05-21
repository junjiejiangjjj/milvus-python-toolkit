from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pyarrow as pa

from .plans import ReadPlan


@dataclass(frozen=True)
class MilvusDataset:
    read_plan: ReadPlan
    _to_arrow: Callable[[], pa.Table]

    def to_arrow(self) -> pa.Table:
        return self._to_arrow()
