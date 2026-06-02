from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from ray_milvus.core.plans import ReadPlan, SegmentReadTask
from ray_milvus.engines.blocks import iter_arrow_blocks, parse_target_block_size
from ray_milvus.engines.local import execute_segment_read_task_batches
from ray_milvus.errors import ConfigError, UnsupportedFeatureError
from ray_milvus.io.storage import create_storage_reader

DEFAULT_READ_BATCH_SIZE = 65_536


def execute_read_plan_blocks(
    plan: ReadPlan,
    target_block_size: int | str | None = None,
    parallelism: int | None = None,
) -> list[Any]:
    target_block_bytes = parse_target_block_size(target_block_size)
    groups = group_segment_read_tasks(plan.tasks, parallelism=parallelism)
    if not groups:
        return []

    ray = _require_ray()
    remote_reader = ray.remote(num_returns="streaming")(_read_segment_task_group_blocks)
    generators = [
        remote_reader.remote(group, target_block_bytes, DEFAULT_READ_BATCH_SIZE)
        for group in groups
    ]
    return [block_ref for generator in generators for block_ref in generator]


def group_segment_read_tasks(
    tasks: tuple[SegmentReadTask, ...],
    parallelism: int | None = None,
) -> tuple[tuple[SegmentReadTask, ...], ...]:
    if parallelism is None:
        return tuple((task,) for task in tasks)
    if parallelism <= 0:
        raise ConfigError("parallelism must be greater than 0")
    if not tasks:
        return ()

    group_count = min(parallelism, len(tasks))
    base_size, remainder = divmod(len(tasks), group_count)
    groups = []
    start = 0
    for index in range(group_count):
        group_size = base_size + (1 if index < remainder else 0)
        end = start + group_size
        groups.append(tasks[start:end])
        start = end
    return tuple(groups)


def read_segment_task_group_blocks(
    segment_task_group: tuple[SegmentReadTask, ...],
    target_block_size: int,
    read_batch_size: int = DEFAULT_READ_BATCH_SIZE,
) -> Iterable[pa.Table]:
    yield from iter_arrow_blocks(
        _read_segment_task_group_batches(segment_task_group, read_batch_size=read_batch_size),
        target_block_size=target_block_size,
    )


def _read_segment_task_group_blocks(
    segment_task_group: tuple[SegmentReadTask, ...],
    target_block_size: int,
    read_batch_size: int,
) -> Iterable[pa.Table]:
    yield from read_segment_task_group_blocks(
        segment_task_group,
        target_block_size=target_block_size,
        read_batch_size=read_batch_size,
    )


def _read_segment_task_group_batches(
    segment_task_group: tuple[SegmentReadTask, ...],
    read_batch_size: int,
) -> Iterable[pa.RecordBatch]:
    readers_by_storage = {}
    for task in segment_task_group:
        storage_key = _storage_cache_key(task.storage)
        reader = readers_by_storage.get(storage_key)
        if reader is None:
            reader = create_storage_reader(task.storage)
            readers_by_storage[storage_key] = reader
        yield from execute_segment_read_task_batches(
            task,
            reader,
            batch_size=read_batch_size,
        )


def _storage_cache_key(storage):
    return (
        storage.backend,
        storage.storage_type,
        storage.endpoint,
        storage.bucket,
        storage.access_key,
        storage.secret_key,
        storage.use_ssl,
        storage.region,
        storage.root_path,
        tuple(sorted(storage.extra.items())),
    )


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise UnsupportedFeatureError(
            "Ray Core execution requires Ray; install it with `pip install ray-milvus[ray]`."
        ) from exc
    return ray
