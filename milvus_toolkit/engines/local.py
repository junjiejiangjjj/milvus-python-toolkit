from __future__ import annotations

import pyarrow as pa

from milvus_toolkit.core.plans import ReadPlan
from milvus_toolkit.io.reader import SegmentTableAdapter, read_segment_as_table


def execute_read_plan(plan: ReadPlan, adapter: SegmentTableAdapter) -> pa.Table:
    tables = [read_segment_as_table(task, adapter) for task in plan.tasks]
    if not tables:
        return pa.table({})
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")
