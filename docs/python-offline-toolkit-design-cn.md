# ray-milvus 设计方案

本文档记录 Python 项目的更高层设计方向：项目不再定位为单纯的 Python connector，而是定位为一个面向 Milvus 的 Python 功能 SDK / 离线处理 SDK。

`read` / `write` 只是其中一部分能力。项目后续可能集成更多 Milvus 相关功能，包括 metadata、snapshot、schema、storage、backfill、import/export、数据转换、数据校验、迁移、inspection、workflow 编排，以及 Ray / Daft 等执行引擎集成。

## 1. 设计定位

推荐项目定位：

```text
ray-milvus
```

它封装 Milvus / Milvus Lite 的离线数据访问、元数据解析、snapshot 处理、StorageV3 / Lite 本地数据读取、数据转换、Backfill、导入导出、校验和 inspection 等能力。

Ray、Daft 或其他 Python 数据处理项目通过 `import` 该项目，直接调用其公开 API。

```text
Ray / Daft / 普通 Python 项目
        |
        v
ray-milvus
        |
        +-- Milvus / Milvus Lite metadata / schema / snapshot
        +-- Unified storage reader / writer
        +-- Dataset / transform
        +-- Backfill / import / export / validate
        +-- Ray / Daft execution adapter
```

历史设计曾考虑使用更通用的 `milvus-offline` 命名，但当前项目已确定为 `ray-milvus`，与 `spark-milvus` 对应。后续设计仍应保持 Milvus 语义与执行引擎解耦，避免把 core 逻辑写死在 Ray adapter 中。

## 2. 总体架构

目录不宜拆得过细。第一版建议控制在少量顶层包中，把相关能力放在同一层内部模块里，后续只有当单个模块明显膨胀时再二次拆分。

推荐结构：

```text
ray_milvus/
  __init__.py
  api.py
  types.py
  errors.py

  core/
    __init__.py
    config.py
    models.py
    schema.py
    metadata.py
    snapshot.py
    manifest.py
    dataset.py
    plans.py
    planner.py
    inspection.py

  io/
    __init__.py
    client.py
    storage.py
    milvus_storage.py
    lite_storage.py
    object_store.py
    reader.py
    writer.py
    import_export.py

  ops/
    __init__.py
    transform.py
    backfill.py
    validate.py
    migrate.py
    repair.py

  engines/
    __init__.py
    base.py
    local.py
    ray.py
    daft.py

  cli/
    __init__.py
    main.py

  testing/
    __init__.py
    fixtures.py
    assertions.py
    parity.py
```

这个版本把原先较分散的目录合并为 5 个主要层次：

| 目录 | 职责 |
|---|---|
| `core/` | Milvus 语义、schema、metadata、snapshot、plan、inspection model。 |
| `io/` | Milvus client、object store、统一 StorageReader、milvus-storage / lite-storage 实现、read/write/import/export。 |
| `ops/` | transform、backfill、validate、migrate、repair 等离线任务语义。 |
| `engines/` | local、Ray、Daft 等执行后端和外部集成。 |
| `cli/` | 命令行入口，尽量薄。 |

核心原则：

```text
core 定义语义；io 负责数据接入；ops 组织离线任务；engines 负责执行；api 对外暴露稳定接口。
```

## 3. 模块职责

### 3.1 `api.py`

高层公开 API facade。

职责：

- 暴露最常用的用户入口。
- 归一化用户参数。
- 创建 read / write / transform / workflow plan。
- 按 `engine` 参数选择 local、Ray、Daft 等执行后端。
- 保持稳定 API，避免用户直接依赖内部模块。

示例 API：

```python
import ray_milvus as rm

mt.read_snapshot(...)
mt.inspect_snapshot(...)
mt.backfill(...)
mt.export_collection(...)
mt.import_data(...)
mt.validate_snapshot(...)
```

`api.py` 不应实现 snapshot 解析、StorageV3 manifest 解析、Backfill merge mode、Ray task 调度等细节。

### 3.2 `types.py` / `errors.py`

`types.py` 放公共轻量类型，适合在用户 API 中暴露：

- `StorageConfig`
- `MilvusConnectionConfig`
- `ReadOptions`
- `WriteOptions`
- `BackfillOptions`
- `ReadResult`
- `WriteResult`
- `BackfillResult`
- `InspectionResult`

`errors.py` 定义统一异常层级：

```text
MilvusToolkitError
  ConfigError
  MilvusClientError
  SnapshotError
  SchemaError
  StorageError
  UnsupportedSegmentError
  TransformError
  BackfillError
  EngineError
```

### 3.3 `core/`

`core/` 是 Milvus 语义中心，不依赖 Ray、Daft、Pandas、Polars、DuckDB 等执行引擎。

```text
core/
  config.py      # 内部配置归一化；把用户参数转成稳定 option model
  models.py      # Collection、Partition、Segment、Field、Storage 等领域模型
  schema.py      # Milvus schema <-> Arrow schema/type 转换
  metadata.py    # metadata 聚合模型；CollectionMetadata、SnapshotMetadata 等
  snapshot.py    # snapshot JSON parser、snapshot schema/segment 提取
  manifest.py    # StorageV3 manifest entry/path/version/row count 解析
  dataset.py     # MilvusDataset；engine-neutral 数据集抽象
  plans.py       # ReadPlan、WritePlan、BackfillPlan、TransformPlan、ValidatePlan
  planner.py     # client/snapshot read plan、write plan、backfill plan 生成
  inspection.py  # snapshot / segment inspection model 和诊断逻辑
```

职责：

- 定义 Milvus 离线处理的领域模型，而不是只放工具函数。
- 定义 engine-neutral 的 `MilvusDataset`、`ReadPlan`、`SegmentReadTask`、`SegmentWriteTask`、`BackfillPlan` 等中间表示。
- 解析 collection schema、snapshot metadata、segment metadata。
- 识别 StorageV3 manifest-backed segment，并保留 manifest path、version、row count、field layout 等信息。
- 根据 columns / extra columns / storage config 生成读取计划。
- 为 backfill、export、validate、transform 等 workflow 生成 logical plan。
- 定义 inspection 诊断结果的数据结构。

`core/` 不应该直接访问对象存储，也不应该调用 `milvus-storage` 或 lite-storage native API；这些由 `io/` 完成。`core/` 可以比 `io/` 和 `ops/` 更“厚”，因为它承载项目的稳定领域语义。

### 3.4 `io/`

`io/` 负责所有外部系统读写，包括 Milvus 服务、对象存储、`milvus-storage` Python API 以及 Milvus Lite 本地数据访问。上层只依赖统一的 storage reader / writer 协议，不感知底层是 Milvus 还是 Milvus Lite。

```text
io/
  client.py          # pymilvus / Milvus API wrapper；collection、partition、import job、insert
  storage.py         # StorageReader / StorageWriter protocol 和 factory
  milvus_storage.py  # milvus-storage adapter；StorageV3 segment read/write
  lite_storage.py    # Milvus Lite 本地 storage adapter
  object_store.py    # S3 / MinIO / OSS / GCS 普通对象读写
  reader.py          # 根据 SegmentReadTask 调用统一 StorageReader，返回 Arrow batches/tables
  writer.py          # write segment / insert / write collection
  import_export.py   # export collection、import_data、parquet/arrow 格式转换
```

职责：

- 访问 Milvus 服务。
- 获取在线 collection / partition / segment metadata。
- 在线 insert。
- 触发和查询 import job。
- 读取 snapshot JSON 和 result JSON。
- 定义统一 `StorageReader` / `StorageWriter` 协议。
- 封装 `milvus-storage` Python API 作为 `MilvusStorageReader` / `MilvusStorageWriter`。
- 封装 Milvus Lite 本地文件访问作为 `MilvusLiteStorageReader` / `MilvusLiteStorageWriter`。
- 根据同一个 `SegmentReadTask` 读取 segment 并返回 PyArrow batch/table。
- 处理 S3、MinIO、OSS、GCS、IAM、本地路径等 storage 配置。

上层不应直接调用 `milvus-storage` 或 lite-storage 实现，也不应基于 Milvus / Milvus Lite 分支处理读取逻辑；差异必须收敛在 `io.storage` factory 和具体 adapter 中。

建议协议：

```python
class StorageReader(Protocol):
    def read_segment_table(self, task: SegmentReadTask) -> pa.Table: ...

class StorageWriter(Protocol):
    def write_segment_table(self, task: SegmentWriteTask, table: pa.Table) -> SegmentMetadata: ...
```

建议配置类型保持显式，而不是把互斥字段塞进一个大配置：

```python
StorageConfig = MilvusStorageConfig | MilvusLiteStorageConfig
```

### 3.5 `MilvusDataset`

`MilvusDataset` 放在 `core/dataset.py`，作为 engine-neutral 数据集描述。它属于核心领域模型，不属于 `io/reader.py` 的实现细节。

当前读取语义分三层：

```text
本地 / 通用流式读取：MilvusDataset.iter_batches(...)
Ray Core 执行：MilvusDataset.to_ray_blocks(...)
Ray Data 执行：MilvusDataset.to_ray_dataset(...)
```

建议对象形态：

```python
class MilvusDataset:
    read_plan: ReadPlan

    def iter_batches(self, batch_size: int | None = None) -> Iterable[pa.RecordBatch]: ...
    def to_arrow(self) -> pa.Table: ...

    def to_ray_blocks(
        self,
        target_block_size: int | str | None = None,
        parallelism: int | None = None,
    ) -> list["ray.ObjectRef[pa.Table]"]: ...

    def to_ray_dataset(
        self,
        target_block_size: int | str | None = None,
        parallelism: int | None = None,
    ) -> "ray.data.Dataset": ...
```

`read_snapshot(...)` 只负责生成逻辑 dataset 和 `ReadPlan`，不强制选择执行引擎。用户根据场景选择：

```python
dataset = ray_milvus.read_snapshot(...)

# 本地流式，适合 debug、小数据、普通 Python job
for batch in dataset.iter_batches(batch_size=10_000):
    ...

# Ray Core，适合用户自己组合 ray.wait / ray.get / remote task
blocks = dataset.to_ray_blocks(target_block_size="128MiB")

# Ray Data，适合进入 Ray Data pipeline
ray_ds = dataset.to_ray_dataset(target_block_size="128MiB")
```

`to_arrow()` 仍保留作为小数据 / 测试 / debug 的便利接口，但它会 materialize 全量数据，不应作为大 collection 的默认读取方式。

#### Batch 与 Block 的职责边界

本地流式读取中，用户直接拿到 `pyarrow.RecordBatch`，因此 `iter_batches(batch_size=...)` 的 `batch_size` 是用户可感知的处理粒度。

Ray Core / Ray Data 中，下游用户拿到的是 `pyarrow.Table` block 或 `ray.data.Dataset` block，不是 storage reader 的中间 `RecordBatch`。因此 Ray API 不应把 `batch_size` 作为主参数；读取 batch size 属于 worker 内部实现细节，可以由框架默认决定。

Ray block 是框架内部执行细节：

```text
RecordBatch -> internal block builder -> pyarrow.Table block -> Ray ObjectRef / Ray Dataset block
```

因此，组装 block 的逻辑应由 ray-milvus 内部提供，而不是让用户在 Ray job 中手写 `iter_arrow_blocks`。内部可以提供类似：

```text
ray_milvus/engines/blocks.py
  parse_target_block_size(...)
  iter_arrow_blocks(...)
```

但这些属于内部工具，用户 API 不直接依赖它们。

`target_block_size` 是可选高级调优项，不应是必选参数。未设置时使用框架默认值，例如：

```python
DEFAULT_TARGET_BLOCK_SIZE = "128MiB"
```

第一版 block builder 策略：

```text
读取 RecordBatch
  -> 用 batch.nbytes 估算大小
  -> 累积到 target_block_size
  -> flush 成 pyarrow.Table block
```

如果单个内部 batch 已超过 `target_block_size`，第一版允许生成超大 block。内部 read batch size 可以先使用框架默认值，后续如确有必要再作为高级调优项暴露，例如 `read_batch_size`，但不应成为 Ray API 的主要参数。

`MilvusDataset` 本身不直接实现复杂 Ray / Daft 逻辑；`to_ray_blocks()` / `to_ray_dataset()` 是薄入口，内部委托给 `engines.ray`。这样可以在不改变用户 API 的前提下，把内部实现从 segment-level Ray task 演进到 file / row-group-level task 或 Ray Data custom datasource。

### 3.6 `ops/`

`ops/` 放离线处理任务和数据转换能力。它比 `io/` 更高层，负责定义“要做什么”，而不是“如何读取或写入”。

```text
ops/
  transform.py  # TransformOp、字段映射、类型转换、vector normalize、UDF wrapper
  backfill.py   # Backfill plan、merge mode、provenance、result 统计
  validate.py   # schema、row count、PK、segment order、一致性校验
  migrate.py    # schema/data migration workflow
  repair.py     # offline repair / rewrite 任务
```

Backfill workflow 示例：

```text
读取 Milvus 原始数据
  -> 读取外部 parquet 新字段
  -> 按 PK join
  -> 应用 merge mode
  -> 按 segment_id 分组
  -> 按 row_offset 排序
  -> 写回 StorageV3
  -> 输出 result / provenance / validation report
```

Backfill 必须保持与 Spark 版本一致的 merge mode 语义：

| Mode | 语义 | 是否读取源端目标字段 |
|------|------|----------------------|
| `replace` | parquet 是目标字段完整事实来源；未匹配源行目标字段为 null | 否 |
| `coalesce` | 源值非空保留源值，源值为空使用 parquet 值；未匹配源行保留源值 | 是 |
| `overwrite` | parquet 命中的 PK 覆盖源值，null 也覆盖；未命中源行保留源值 | 是 |

写入前必须保证：

```text
同一个 segment 内按 row_offset 升序写入
```

后续可在 `ops/` 中继续扩展：

- 数据脱敏。
- 字段类型迁移。
- vector 重新归一化。
- embedding 字段重算。
- partition 级 export。
- snapshot 对比。
- schema compatibility check。
- offline rewrite / repair。

### 3.7 `engines/`

`engines/` 负责“用什么方式执行”。第一版只保留一个目录，不额外拆 `integrations/`。

```text
engines/
  base.py   # ExecutionEngine protocol
  local.py  # 本地顺序执行；debug、小数据、单元测试
  ray.py    # Ray Dataset / Ray task adapter；同时提供 Ray 用户直接 import 的 API
  daft.py   # Daft scan source / DataFrame adapter；同时提供 Daft 用户直接 import 的 API
```

职责：

- 执行 read plan。
- 执行 transform plan。
- 执行 backfill / export / import / validate workflow。
- 将 Arrow batches 转为 Ray Dataset、Daft DataFrame 等对象。
- 为 Ray / Daft 项目提供直接 import 的集成函数。

建议接口：

```python
class ExecutionEngine:
    def execute_read(self, plan): ...
    def execute_transform(self, plan): ...
    def execute_backfill(self, plan): ...
    def execute_export(self, plan): ...
```

Ray 项目中优先通过 `MilvusDataset` 使用：

```python
dataset = ray_milvus.read_snapshot(...)

# Ray Core
blocks = dataset.to_ray_blocks(target_block_size="128MiB")

# Ray Data
ray_ds = dataset.to_ray_dataset(target_block_size="128MiB")
```

`engines.ray` 也可以提供直接 import 的低层函数，但不作为普通用户首选入口：

```python
from ray_milvus.engines.ray import execute_read_plan_blocks, execute_read_plan_dataset
```

Daft 项目中可以直接使用：

```python
from ray_milvus.engines.daft import scan_snapshot

df = scan_snapshot(...)
```

如果未来 Ray / Daft 代码变得很大，再从单文件升级为子包：

```text
engines/ray/
  __init__.py
  dataset.py
  backfill.py
```

第一版不需要提前引入 `integrations/` 目录。

Ray / Daft 应作为可选依赖：

```bash
pip install ray-milvus
pip install ray-milvus[ray]
pip install ray-milvus[daft]
pip install ray-milvus[all]
```

#### Ray Core 读取策略

Ray Core 面向希望自己控制 `ray.remote`、`ray.wait`、`ray.get` 和后续 task 编排的用户。对外接口建议是：

```python
blocks = dataset.to_ray_blocks(
    target_block_size="128MiB",
    parallelism=None,
)
```

返回值是：

```python
list[ray.ObjectRef[pyarrow.Table]]
```

这里的 `pyarrow.Table` 是 Ray block，而不是完整 collection。Ray Core 用户可以继续把这些 block refs 传给自己的 remote task：

```python
@ray.remote
def process_block(block: pa.Table):
    ...

refs = [process_block.remote(block_ref) for block_ref in blocks]
```

第一版内部执行策略：

```text
driver:
  read_snapshot(...) 生成 ReadPlan
  -> 根据 parallelism 把 ReadPlan.tasks 分成 segment_task_group
  -> 提交 read_segment_task_group.remote(...)
  -> 收集 pyarrow.Table block 的 ObjectRef

worker:
  segment_task_group: tuple[SegmentReadTask, ...]
  -> worker 内部 create_storage_reader(task.storage)
  -> read_segment_as_batches(task, reader, batch_size=内部默认值)
  -> internal iter_arrow_blocks(..., target_block_size=默认值或用户传入值)
  -> 返回 / yield pyarrow.Table block refs
```

数据读取必须发生在 Ray worker 上，而不是 driver 上。driver 不应执行 `dataset.iter_batches(...)` 后再 `ray.put(...)`；这种写法会让 driver 成为 IO 和内存瓶颈。Ray API 的语义是 driver 生成计划并提交任务，worker 访问 MinIO / S3 / 本地文件 / `milvus-storage` 并生成 block。

`parallelism` 控制 ray-milvus 内部提交的读取 task / task group 数量，类似 Ray Core 示例中的：

```python
futures = [read_segment_task_group.remote(group) for group in groups]
```

第一版中 group 类型可以是：

```python
segment_task_group: tuple[SegmentReadTask, ...]
```

用户不会直接接触 group。`parallelism=None` 时可以默认一个 segment 一个 task；指定 `parallelism=N` 时，框架把 `ReadPlan.tasks` 分成最多 N 个 `segment_task_group`，每个 Ray task 顺序读取自己 group 内的 segment。Ray 实际同时运行多少 task 仍由集群资源和调度器决定。

Ray Core 路径应优先使用 Ray generator task：一个 segment group reader task 边读边 yield 多个 block，避免一个 task 最后返回巨大的 `list[pa.Table]`。如果 Ray 版本限制导致 generator task 不可用，应重新评估实现方案，而不是把 eager block list 作为发布级语义。

并行粒度演进顺序：

1. **Segment-level parallelism**：一个 `SegmentReadTask` 一个 Ray task，简单可靠，适合 segment 数量足够多的 collection。
2. **Segment grouping**：小 segment 合并到一个 Ray task，降低 task 数量和调度开销。
3. **File / row-group-level parallelism**：对 packed parquet 大 segment，按 parquet file 或 row group 进一步拆 task，提高少量大 segment 的并行度。

无论内部粒度如何变化，对外仍保持：

```python
dataset.to_ray_blocks(target_block_size=..., parallelism=...)
```

#### Ray Data 读取策略

Ray Data 面向希望进入 Ray Dataset pipeline 的用户。对外接口建议是：

```python
ray_ds = dataset.to_ray_dataset(
    target_block_size="128MiB",
    parallelism=None,
)
```

Ray Data 路径应直接实现 custom datasource，不把 `to_ray_blocks() + ray.data.from_arrow_refs(...)` 作为发布级实现。项目应在 Ray Data 原生读取路径完成后再发布，因此没有必要引入一个 eager submit 的中间版本。

`to_ray_dataset()` 内部应使用 Ray Data datasource 接入：

```text
dataset.to_ray_dataset(...)
  -> ray.data.read_datasource(MilvusSnapshotDatasource(...))
  -> Ray Data 调用 datasource 创建 ReadTask
  -> ReadTask 在 Ray worker 上读取 segment_task_group
  -> worker 内部 StorageReader -> RecordBatch stream -> internal block builder
  -> pyarrow.Table blocks -> ray.data.Dataset
```

这样 `to_ray_dataset()` 更接近 Ray Data 原生 lazy execution：构造 Dataset 时不应由 ray-milvus 先 eager 提交全部读取任务；真正执行由 Ray Data 在 action / pipeline 执行阶段调度。

用户可以继续使用 Ray Data：

```python
ray_ds = dataset.to_ray_dataset(target_block_size="128MiB")
ray_ds = ray_ds.map_batches(fn, batch_size=10_000, batch_format="pyarrow")
ray_ds.write_parquet("s3://bucket/output")
```

`MilvusSnapshotDatasource` 负责把 `ReadPlan` 转成 Ray Data read tasks：

```python
class MilvusSnapshotDatasource(Datasource):
    read_plan: ReadPlan
    target_block_size: int

    def get_read_tasks(self, parallelism: int) -> list[ReadTask]: ...
```

每个 `ReadTask` 对应一个内部 group：

```python
segment_task_group: tuple[SegmentReadTask, ...]
```

ReadTask metadata 应尽量提供：

- Arrow schema：由 `ReadPlan.projected_fields` 和 `include` metadata columns 生成。
- 预计 rows：由 group 内 segment `row_count` 累加，缺失时允许 unknown。
- 预计 bytes：第一版可以 unknown，后续可按字段类型 / vector dim / parquet metadata 估算。

custom datasource 让 Ray Data 管理 read tasks、block metadata、parallelism、lazy execution、backpressure 和调度 locality，是 Ray Data 集成的目标实现。

#### Backfill 与 shuffle 约束

Ray backfill 不能把外部 backfill 表全量 broadcast 到每个 segment reader。读取层应产生可 partition / shuffle 的 Ray Dataset 或 block 流，后续按 primary key hash partition 对齐：

```text
source snapshot blocks
  -> hash partition by primary key
external backfill blocks
  -> hash partition by primary key
same partition join / merge
  -> write StorageV3 addfield output
```

因此 Ray 读取 API 应服务于 partitioned data flow，而不是 driver 端 `concat` 或 full-table materialization。

### 3.8 `cli/`

命令行入口保持薄层。第一版可以只有：

```text
cli/
  main.py
```

在 `main.py` 中注册子命令：

```text
ray-milvus inspect ...
ray-milvus backfill ...
ray-milvus export ...
ray-milvus import ...
ray-milvus validate ...
```

如果 CLI 变复杂，再拆：

```text
cli/inspect.py
cli/backfill.py
cli/export.py
```

CLI 应该调用 `api.py` 或 `ops/`，不应绕过核心抽象直接访问 storage internals。

### 3.9 `testing/`

放测试和 parity job 复用的辅助函数。

```text
testing/
  fixtures.py    # 构造小型 snapshot、schema、backfill 输入
  assertions.py  # Arrow table、schema、result 比较
  parity.py      # Spark / Python result normalization
```

## 4. 依赖方向

推荐依赖方向：

```text
api
 ├── core
 ├── io
 ├── ops
 └── engines

ops
 ├── core
 ├── io
 └── engines

io
 └── core

engines
 ├── core
 ├── io
 └── ops

cli
 └── api
```

禁止依赖方向：

```text
core    -/-> io
core    -/-> engines
core    -/-> ray / daft
io      -/-> engines
ops     -/-> engine-specific SDK unless behind optional path
```

核心原则：

```text
Milvus 语义不能依赖执行引擎；执行引擎只负责调度和 DataFrame/Dataset 表达。
```

## 5. 推荐 API 分层

### 5.1 Level 1：高层快捷 API

适合普通用户。

```python
import ray_milvus as rm

mt.read_snapshot(...)
mt.backfill(...)
mt.inspect_snapshot(...)
mt.export_collection(...)
mt.import_data(...)
mt.validate_snapshot(...)
```

### 5.2 Level 2：对象式 API

适合 Ray / Daft / 数据处理项目集成。

```python
dataset = mt.MilvusDataset.from_snapshot(...)
dataset = dataset.select(["id", "vector"])
dataset = dataset.transform([...])
ray_ds = dataset.to_ray()
df = dataset.to_daft()
```

### 5.3 Level 3：底层 plan API

适合高级用户、debug 和内部 workflow。

```python
plan = mt.plan_snapshot_read(...)

for task in plan.tasks:
    for batch in mt.read_segment(task):
        consume(batch)
```

## 6. 使用示例

### 6.1 读取 snapshot

```python
import ray_milvus as rm

storage = mt.StorageConfig(
    endpoint="s3.us-west-2.amazonaws.com",
    bucket="milvus-bucket",
    region="us-west-2",
)

dataset = mt.read_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
    columns=["id", "vector", "text"],
    include=["segment_id", "row_offset"],
)
```

默认可以返回 engine-neutral 的 `MilvusDataset`。

```python
table = dataset.to_arrow()
pdf = dataset.to_pandas()
```

### 6.2 在 Ray 项目中使用

```python
import ray_milvus as rm

ray_ds = mt.read_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
    engine="ray",
)
```

或者：

```python
from ray_milvus.engines.ray import read_snapshot_as_dataset

ray_ds = read_snapshot_as_dataset(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
)
```

### 6.3 在 Daft 项目中使用

```python
import ray_milvus as rm

df = mt.read_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
    engine="daft",
)
```

或者：

```python
from ray_milvus.engines.daft import scan_snapshot

df = scan_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
)
```

### 6.4 Backfill

```python
result = mt.backfill(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    backfill_data="s3://source-bucket/new_fields.parquet",
    storage=storage,
    mode="coalesce",
    column_mapping={
        "id": "id",
        "new_text": "text",
    },
    engine="ray",
)
```

### 6.5 Inspection

```python
info = mt.inspect_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    storage=storage,
)

print(info.segments)
```

### 6.6 Import job

```python
job = mt.import_data(
    collection="my_collection",
    source_path="s3://milvus-bucket/export/foo.parquet",
    uri="http://localhost:19530",
    storage=storage,
)
```

### 6.7 数据转换

```python
dataset = mt.read_snapshot(...)

new_dataset = dataset.transform([
    mt.transforms.rename_field("old_name", "new_name"),
    mt.transforms.cast("age", "int64"),
    mt.transforms.normalize_vector("embedding"),
])
```

也可以提供 workflow 风格：

```python
result = mt.transform_snapshot(
    snapshot_path="s3://milvus-bucket/snapshots/foo.json",
    output_path="s3://milvus-bucket/output/",
    transforms=[
        mt.transforms.field_mapping({...}),
        mt.transforms.vector_normalize("embedding"),
    ],
    engine="ray",
)
```

## 7. MVP 建议

第一阶段不建议一次实现全部模块。推荐先完成最小闭环：

```text
ray_milvus/
  api.py
  types.py
  errors.py

  core/
    models.py
    schema.py
    metadata.py
    snapshot.py
    manifest.py
    dataset.py
    plans.py
    planner.py
    inspection.py

  io/
    storage.py
    object_store.py
    reader.py

  ops/
    backfill.py
    validate.py

  engines/
    base.py
    local.py
    ray.py

  cli/
    main.py
```

第一阶段目标：

1. `read_snapshot`。
2. `inspect_snapshot`。
3. `MilvusDataset`。
4. `to_arrow`。
5. `to_ray`。
6. `backfill(..., engine="ray")`。
7. StorageV3 manifest-backed segment read。
8. `segment_id` / `row_offset` 保序。

第二阶段再加：

1. Daft scan source。
2. transform API。
3. export / import workflow。
4. validate workflow 增强。
5. vector transform。
6. online Milvus client operations。
7. Polars / DuckDB 支持。

## 8. 与原 Python Connector 设计的关系

原 Python Connector 设计可以作为本方案中的 `core`、`io`、`ops` 和 `engines` 的一部分。

对应关系：

| 原 connector 设计 | 新 toolkit 设计 |
|---|---|
| `core/config.py` | `types.py` / `core/config.py` |
| `core/schema.py` | `core/schema.py` |
| `core/snapshot.py` | `core/snapshot.py` |
| `core/planner.py` | `core/planner.py` + `core/plans.py` |
| `core/read.py` | `core/plans.py` + `io/reader.py` + `io/storage.py` |
| `core/write.py` | `io/writer.py` + `io/storage.py` |
| `core/backfill.py` | `ops/backfill.py` |
| `core/inspection.py` | `core/inspection.py` |
| `engines/ray.py` | `engines/ray.py` |
| `engines/daft.py` | `engines/daft.py` |
| `cli/*` | `cli/main.py`，复杂后再拆子命令文件 |

也就是说，connector 能力仍然存在，但不再是项目的架构中心，而是 toolkit 的 `io` 能力和部分 `core` 语义。

## 9. 结论

如果项目只做 Milvus 读写，Python connector 架构已经足够。

但如果项目会集成更多 Milvus 功能，并被 Ray、Daft 或其他 Python 项目直接 import 使用，则更推荐设计为：

```text
ray-milvus
```

核心结构应保持紧凑：

```text
api + core + io + ops + engines + cli
```

其中：

- `core`：Milvus 语义、schema、snapshot、metadata、plan、inspection。
- `io`：Milvus client、object store、StorageV3 read/write、import/export。
- `ops`：transform、backfill、validate、migrate、repair。
- `engines`：local、Ray、Daft 等执行和生态集成。
- `cli`：命令行薄封装。

这种拆分既避免目录过散，也能避免项目被某个执行引擎或 connector 视角绑定，便于后续持续扩展 Milvus 离线处理能力。
