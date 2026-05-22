from __future__ import annotations

import pyarrow as pa

from milvus_toolkit.core.plans import ReadPlan, SegmentReadTask
from milvus_toolkit.errors import StorageError
from milvus_toolkit.io.reader import SegmentTableAdapter, read_segment_as_table


def execute_read_plan(plan: ReadPlan, adapter: SegmentTableAdapter) -> pa.Table:
    tables = [_read_task_table(task, adapter) for task in plan.tasks]
    if not tables:
        return pa.table({})
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def _read_task_table(task: SegmentReadTask, adapter: SegmentTableAdapter) -> pa.Table:
    table = read_segment_as_table(task, adapter)
    expected_row_count = task.segment.row_count
    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise StorageError(
            f"Segment {task.segment.segment_id} row count mismatch: "
            f"expected {expected_row_count}, got {table.num_rows}"
        )
    return table
