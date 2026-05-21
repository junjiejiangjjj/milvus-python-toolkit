from __future__ import annotations

from typing import Protocol

import pyarrow as pa

from milvus_toolkit.core.plans import SegmentReadTask
from milvus_toolkit.errors import StorageError


class SegmentTableAdapter(Protocol):
    def read_segment_table(self, task: SegmentReadTask) -> pa.Table: ...


def read_segment_as_table(task: SegmentReadTask, adapter: SegmentTableAdapter) -> pa.Table:
    table = adapter.read_segment_table(task)
    if not isinstance(table, pa.Table):
        raise StorageError("Storage adapter must return a pyarrow.Table")

    for metadata_column in task.include:
        if metadata_column == "segment_id" and "segment_id" not in table.column_names:
            table = table.append_column(
                "segment_id",
                pa.array([task.segment.segment_id] * table.num_rows, type=pa.int64()),
            )
        elif metadata_column == "row_offset" and "row_offset" not in table.column_names:
            table = table.append_column(
                "row_offset",
                pa.array(range(table.num_rows), type=pa.int64()),
            )
    return table
