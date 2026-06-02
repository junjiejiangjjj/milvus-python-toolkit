from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import pyarrow as pa

from ray_milvus.core.plans import SegmentReadTask
from ray_milvus.errors import StorageError


class SegmentTableAdapter(Protocol):
    def read_segment_table(self, task: SegmentReadTask) -> pa.Table: ...


@runtime_checkable
class SegmentBatchAdapter(Protocol):
    def read_segment_batches(
        self,
        task: SegmentReadTask,
        batch_size: int | None = None,
    ) -> Iterable[pa.RecordBatch]:
        ...


def read_segment_as_batches(
    task: SegmentReadTask,
    adapter: SegmentTableAdapter,
    batch_size: int | None = None,
) -> Iterable[pa.RecordBatch]:
    row_offset = 0
    for batch in _read_raw_segment_batches(task, adapter, batch_size=batch_size):
        if not isinstance(batch, pa.RecordBatch):
            raise StorageError("Storage adapter must yield pyarrow.RecordBatch")
        batch = _append_metadata_columns(batch, task, row_offset)
        row_offset += batch.num_rows
        yield batch


def read_segment_as_table(task: SegmentReadTask, adapter: SegmentTableAdapter) -> pa.Table:
    batches = list(read_segment_as_batches(task, adapter))
    return pa.Table.from_batches(batches) if batches else pa.table({})


def _read_raw_segment_batches(
    task: SegmentReadTask,
    adapter: SegmentTableAdapter,
    batch_size: int | None,
) -> Iterable[pa.RecordBatch]:
    if isinstance(adapter, SegmentBatchAdapter):
        yield from adapter.read_segment_batches(task, batch_size=batch_size)
        return

    table = adapter.read_segment_table(task)
    if not isinstance(table, pa.Table):
        raise StorageError("Storage adapter must return a pyarrow.Table")
    yield from table.to_batches(max_chunksize=batch_size)


def _append_metadata_columns(
    batch: pa.RecordBatch,
    task: SegmentReadTask,
    row_offset: int,
) -> pa.RecordBatch:
    for metadata_column in task.include:
        if metadata_column == "segment_id" and "segment_id" not in batch.schema.names:
            batch = batch.append_column(
                "segment_id",
                pa.array([task.segment.segment_id] * batch.num_rows, type=pa.int64()),
            )
        elif metadata_column == "row_offset" and "row_offset" not in batch.schema.names:
            batch = batch.append_column(
                "row_offset",
                pa.array(range(row_offset, row_offset + batch.num_rows), type=pa.int64()),
            )
    return batch
