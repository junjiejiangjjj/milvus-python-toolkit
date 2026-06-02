from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

from ray_milvus.core.plans import ReadPlan, SegmentReadTask
from ray_milvus.errors import StorageError
from ray_milvus.io.reader import SegmentTableAdapter, read_segment_as_batches


def execute_read_plan(
    plan: ReadPlan,
    adapter: SegmentTableAdapter,
    batch_size: int | None = None,
) -> pa.Table:
    batches = list(execute_read_plan_batches(plan, adapter, batch_size=batch_size))
    return pa.Table.from_batches(batches) if batches else pa.table({})


def execute_read_plan_batches(
    plan: ReadPlan,
    adapter: SegmentTableAdapter,
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    for task in plan.tasks:
        yield from execute_segment_read_task_batches(task, adapter, batch_size=batch_size)


def execute_segment_read_task_batches(
    task: SegmentReadTask,
    adapter: SegmentTableAdapter,
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    row_count = 0
    for batch in read_segment_as_batches(task, adapter, batch_size=batch_size):
        row_count += batch.num_rows
        yield batch

    expected_row_count = task.segment.row_count
    if expected_row_count is not None and row_count != expected_row_count:
        raise StorageError(
            f"Segment {task.segment.segment_id} row count mismatch: "
            f"expected {expected_row_count}, got {row_count}"
        )
