# ray-milvus 多执行引擎设计方案

本文档记录一个比“Ray 版本”更抽象的设计：构建一个 **Milvus Python Connector Core**，再在其上适配 Ray、Daft 以及后续其他 Python 数据处理项目。

新的范围约束：**Python Connector 只支持 Milvus StorageV3 manifest-backed 数据路径**。旧 packed parquet 存储格式不纳入 Python Connector 支持范围，也不在设计、测试和 API 中暴露。

## 1. 背景和目标

当前 Spark 项目已经实现了较完整的 Milvus connector 能力。Python Connector 只选取后续需要维护的功能范围：

- 读取 Milvus collection。
- 读取 Milvus snapshot，不依赖运行中的 Milvus 服务。
- 读取 StorageV3 manifest-backed segment。
- 通过 `milvus-storage` Python API 读取/写入 StorageV3 segment。
- 在线 insert 写入 Milvus。
- Backfill 新字段，支持 `replace`、`coalesce`、`overwrite`。
- 按 `segment_id` 和 `row_offset` 保持 segment 内原始顺序。
- 提供 segment inspection 工具。
- 提供向量距离和 KNN 类辅助能力。

最初可以把这些能力迁移到 Ray。项目当前已确定命名为 `ray-milvus`，但仍应避免把 Milvus 语义写死在 Ray adapter 中。因此目标是：

```text
Milvus / Snapshot / Object Storage
        |
        v
Engine-neutral Python Core
        |
        v
Ray / Daft / Pandas / Polars / DuckDB / 普通 Python job
```

## 2. 总体架构

核心设计分为两层：

1. **Milvus Connector Core**：引擎无关层，只处理 Milvus 相关的语义、元数据、StorageV3 存储读写和计划生成。
2. **Execution Engine Adapter**：执行引擎适配层，把 Core 产出的 task/Arrow batch 映射到 Ray、Daft、Pandas、Polars、DuckDB 等不同框架。

项目当前命名为 `ray-milvus`，与 `spark-milvus` 对应；包内仍保持 engine-neutral core，并通过 extra 选择执行引擎依赖。

安装时通过 extra 选择引擎依赖：

```bash
pip install ray-milvus[ray]
pip install ray-milvus[daft]
pip install ray-milvus[polars]
pip install ray-milvus[duckdb]
```

## 3. 建议项目架构

Python Connector 建议按“公共 API / Core / Storage Adapter / Engine Adapter / CLI / Tests”分层。核心目标是：所有 Milvus 语义只在 Core 中实现一次，Ray、Daft、Pandas 等执行引擎只负责任务调度和 DataFrame/Dataset 表达。

### 3.1 顶层目录

如果放在当前仓库内，可以先采用 monorepo 子目录：

```text
python-connector/
  pyproject.toml
  README.md
  milvus_connector/
    __init__.py
    api.py
    types.py
    core/
    storage/
    engines/
    cli/
    testing/
  tests/
    fixtures/
    unit/
    storage/
    engines/
    integration/
    parity/
```

如果未来独立成单仓库，则顶层结构保持不变，只把 `python-connector/` 提升为 repo root。

### 3.2 package 结构

```text
milvus_connector/
  __init__.py
  api.py
  types.py
  errors.py

  core/
    __init__.py
    config.py
    schema.py
    metadata.py
    client.py
    snapshot.py
    planner.py
    read.py
    write.py
    backfill.py
    inspection.py
    vector.py

  storage/
    __init__.py
    base.py
    milvus_storage.py
    properties.py
    object_store.py

  engines/
    __init__.py
    base.py
    local.py
    pandas.py
    ray.py
    daft.py
    polars.py
    duckdb.py

  cli/
    __init__.py
    main.py
    inspect.py
    read.py
    backfill.py
    insert.py
    write_segment.py

  testing/
    __init__.py
    fixtures.py
    assertions.py
    parity.py
```

### 3.3 模块职责

#### `milvus_connector/__init__.py`

面向普通用户暴露最常用 API，不暴露内部 task、planner 和 storage 细节。

建议导出：

```python
from milvus_connector.api import (
    read_milvus,
    read_snapshot,
    insert,
    backfill,
    inspect_snapshot,
)
```

#### `api.py`

高层 facade，负责把用户参数转换为 Core options，并按 `engine` 参数分发到对应 engine adapter。

职责：

- 提供稳定用户入口。
- 做轻量参数归一化。
- 选择默认 engine，例如 `local` 或 `pandas`。
- 调用 Core planner。
- 调用 Engine adapter 执行。

不应在这里实现 snapshot 解析、manifest 解析、Backfill merge mode 等细节。

#### `types.py`

放跨层共享的轻量公共类型，例如：

- `ReadResult`
- `BackfillResult`
- `WriteResult`
- `InspectionResult`
- `VectorMetric`

复杂内部 task 建议放在 `core/metadata.py` 或 `core/planner.py`，避免公共 API 过早暴露内部结构。

#### `errors.py`

定义 connector 统一异常层级，便于 CLI、Ray task、Daft scan source 统一处理错误。

建议异常：

```text
MilvusConnectorError
  ConfigError
  SnapshotError
  UnsupportedSegmentError
  SchemaError
  StorageError
  BackfillError
  EngineError
```

### 3.4 Core 子包

Core 子包是项目的语义中心，不依赖任何执行引擎。

```text
core/
  config.py      # StorageConfig、ReadOptions、BackfillOptions、InsertOptions
  schema.py      # Milvus schema <-> PyArrow schema/type 转换
  metadata.py    # SnapshotMetadata、SegmentMetadata、ReadPlan、SegmentReadTask
  client.py      # pymilvus / Milvus API metadata wrapper
  snapshot.py    # snapshot JSON 读取、解析、校验
  planner.py     # client/snapshot read plan 和 backfill plan 生成
  read.py        # 顺序读取 task，输出 Arrow batches/table
  write.py       # 顺序写 StorageV3 segment
  backfill.py    # merge mode、column mapping、provenance、local baseline
  inspection.py  # snapshot/segment inspection model
  vector.py      # engine-neutral vector functions
```

Core 的依赖方向：

```text
config/schema/metadata
        |
        v
snapshot/client
        |
        v
planner
        |
        v
read/write/backfill/inspection/vector
```

Core 允许依赖：

- `pyarrow`
- `numpy`
- `pymilvus` 或 Milvus Python client
- connector 自己的 `storage` 抽象

Core 不允许依赖：

- `ray`
- `daft`
- `pandas`
- `polars`
- `duckdb`
- Spark/JVM 相关包

### 3.5 Storage 子包

Storage 子包隔离 `milvus-storage` Python API 和对象存储配置细节。

```text
storage/
  base.py            # StorageAdapter protocol / abstract base class
  milvus_storage.py  # milvus-storage Python API 实现
  properties.py      # StorageConfig -> milvus-storage properties 映射
  object_store.py    # snapshot JSON、result JSON 等普通对象读写辅助
```

职责边界：

- `core.planner` 只生成 `SegmentReadTask` / `SegmentWriteTask`。
- `storage.milvus_storage` 才知道如何把 task 翻译成 `milvus-storage` Python API 调用。
- `engines.*` 不直接调用 `milvus-storage`，必须通过 Core read/write 或 StorageAdapter。

这样可以把 native dependency、Arrow batch 生命周期、S3/IAM/MinIO properties 全部收敛到一层。

### 3.6 Engine 子包

Engine 子包只负责执行，不定义 Milvus 存储语义。

```text
engines/
  base.py     # ExecutionEngine protocol
  local.py    # 纯 Python / PyArrow 顺序执行
  pandas.py   # 小数据 DataFrame baseline
  ray.py      # Ray Dataset / Ray task implementation
  daft.py     # Daft DataFrame / scan source implementation
  polars.py   # Polars DataFrame integration
  duckdb.py   # DuckDB Arrow relation integration
```

每个 engine adapter 的职责：

- 接收 Core `ReadPlan` / `BackfillPlan`。
- 调用 Core read/write 函数执行 segment task。
- 将 Arrow batches 转为对应 engine 的 Dataset/DataFrame/Relation。
- 实现 join、group by segment、sort by row_offset 等执行细节。
- 把执行结果转换回 Core 定义的 `BackfillResult` / `WriteResult`。

engine adapter 不应：

- 重新解析 snapshot。
- 重新判断 StorageV3 manifest 语义。
- 重新定义 Backfill merge mode。
- 直接拼接 StorageV3 manifest/write metadata。

### 3.7 CLI 子包

CLI 只是高层 API 的命令行包装，不应绕过 `api.py` 或 Core。

```text
cli/
  main.py           # argparse/typer/click entrypoint
  inspect.py        # ray-milvus inspect
  read.py           # ray-milvus read
  backfill.py       # ray-milvus backfill
  insert.py         # ray-milvus insert
  write_segment.py  # ray-milvus write-segment
```

CLI 职责：

- 参数解析。
- storage / Milvus 连接参数归一化。
- 调用 `ray_milvus.api`。
- 统一错误展示和 exit code。
- 支持 JSON 输出，便于脚本和 CI 使用。

### 3.8 testing 子包

`testing/` 放给测试和外部 parity job 复用的辅助函数，不参与生产路径。

```text
testing/
  fixtures.py    # 构造小型 snapshot/schema/backfill 输入
  assertions.py  # Arrow table、schema、BackfillResult 比较
  parity.py      # Spark/Python parity result normalization
```

这样 parity tests、engine tests 和 integration tests 可以共享断言逻辑，避免每个测试重复写 schema/table 对比。

### 3.9 依赖规则

推荐强制遵守以下依赖方向：

```text
api
 |-> core
 |-> engines

engines
 |-> core

core
 |-> storage

storage
 |-> milvus-storage python package
```

禁止反向依赖：

```text
core     -/-> engines
storage  -/-> engines
storage  -/-> api
cli      -/-> storage internals
```

这条规则可以后续用 import-linter 或简单单测约束。

### 3.10 读取数据流

```text
read_snapshot/read_milvus API
  -> normalize options
  -> core.planner creates ReadPlan
  -> engine adapter receives ReadPlan
  -> each SegmentReadTask calls core.read.read_segment
  -> storage adapter calls milvus-storage Python API
  -> returns PyArrow RecordBatch/Table
  -> engine adapter builds Dataset/DataFrame/Relation
```

### 3.11 Backfill 数据流

```text
backfill API
  -> normalize options
  -> core.planner creates BackfillPlan
  -> engine adapter reads source plan
  -> engine adapter reads backfill parquet
  -> core.backfill validates mapping/types/mode
  -> engine adapter joins by PK
  -> core.backfill applies merge expressions or provides expression spec
  -> engine adapter groups by segment_id and sorts by row_offset
  -> core.write writes StorageV3 segment through storage adapter
  -> engine adapter aggregates BackfillResult
```

### 3.12 最小可交付架构

第一阶段不需要一次实现所有 adapter。建议最小闭环是：

```text
api.py
core/config.py
core/schema.py
core/snapshot.py
core/planner.py
core/read.py
storage/base.py
storage/milvus_storage.py
engines/local.py
engines/ray.py
cli/inspect.py
cli/read.py
```

这个闭环可以先完成：

- snapshot 解析。
- StorageV3 read planning。
- local sequential read。
- Ray read snapshot。
- inspect CLI。

Backfill、Daft、Polars、DuckDB 可以在这个架构稳定后继续补齐。

## 4. Core 层职责

Core 层不依赖 Ray、Daft、Pandas、Polars、DuckDB 或 Spark。它只处理 Milvus 和 StorageV3 数据格式相关的确定性逻辑。

Core 层负责：

- 解析 Milvus snapshot metadata JSON。
- 识别 snapshot 中的 StorageV3 manifest entries。
- 规划 StorageV3 segment read task。
- 封装 `milvus-storage` Python API。
- 读取 StorageV3 segment，输出 PyArrow `RecordBatch` 或 `Table`。
- 写入 StorageV3 manifest-backed segment。
- Milvus schema 与 PyArrow schema 互转。
- 生成 Backfill logical plan。
- 定义 Backfill merge mode 语义。
- 生成 Backfill result / provenance 统计。
- 生成 snapshot / segment inspection 结果。
- 提供引擎无关的向量函数实现。

Core 层不应该出现：

- `ray`
- `daft`
- `pandas`
- `polars`
- `duckdb`
- `spark`
- 分布式 shuffle 的具体实现
- DataFrame engine 专属表达式

## 5. Storage Adapter

`milvus-storage` 已经提供 Python 接口，因此 Python Connector 应优先直接使用该接口，不再引入 JVM bridge。

建议在 Core 中封装一个 storage adapter，隔离 `milvus-storage` Python API 的具体形态：

```python
class MilvusStorageAdapter:
    def read_segment(self, task: SegmentReadTask) -> Iterable[pa.RecordBatch]:
        ...

    def write_segment(
        self,
        task: SegmentWriteTask,
        batches: Iterable[pa.RecordBatch],
    ) -> WriteResult:
        ...
```

这样上层只依赖 connector 自己的稳定接口。如果未来 `milvus-storage` Python API 调整，只需要改 `storage.py`。

只有当 `milvus-storage` Python 接口缺失关键能力时，才考虑临时补充其他 bridge，例如：

- 无法读取 StorageV3 manifest-backed segment。
- 无法写 StorageV3 manifest / transaction。
- 无法以 PyArrow/Arrow batch 高效返回数据。
- 无法配置 S3/IAM/MinIO/OSS storage properties。

## 6. 通用数据模型

### 6.1 StorageConfig

```python
@dataclass
class StorageConfig:
    endpoint: str
    bucket: str
    root_path: str = "files"
    access_key: str | None = None
    secret_key: str | None = None
    use_ssl: bool = False
    region: str = "us-east-1"
    use_iam: bool = False
    path_style_access: bool | None = None
    extra: dict[str, str] = field(default_factory=dict)
```

该配置需要能映射到 `milvus-storage` Python API 所需的 filesystem properties。

### 6.2 MilvusReadOptions

```python
@dataclass
class MilvusReadOptions:
    uri: str | None = None
    token: str = ""
    database: str = "default"
    collection: str | None = None
    partition: str | None = None
    snapshot_path: str | None = None
    columns: list[str] | None = None
    extra_columns: list[str] = field(default_factory=list)
    storage: StorageConfig | None = None
```

`uri` 用于 client mode，`snapshot_path` 用于 snapshot mode。二者可以分开建更明确的 API，但内部可以统一为 read options。

### 6.3 SegmentReadTask

Planner 输出通用 task 描述，不绑定执行引擎：

```python
@dataclass
class SegmentReadTask:
    segment_id: int
    partition_id: int
    schema: MilvusSchema
    projected_fields: list[str]
    projected_field_ids: list[int]
    extra_columns: list[str]
    storage: StorageConfig
    manifest_path: str
    read_version: int | None = None
```

### 6.4 SegmentWriteTask

```python
@dataclass
class SegmentWriteTask:
    segment_id: int
    partition_id: int
    collection_id: int
    schema: MilvusSchema
    field_name_to_id: dict[str, int]
    storage: StorageConfig
    base_path: str
    batch_size: int = 1024
```

### 6.5 BackfillPlan

```python
@dataclass
class BackfillPlan:
    source_read_plan: ReadPlan
    backfill_data_path: str
    pk_field: str
    target_fields: list[str]
    mode: Literal["replace", "coalesce", "overwrite"]
    column_mapping: dict[str, str] | None
    storage: StorageConfig
    source_storage: StorageConfig | None
    segment_write_tasks: dict[int, SegmentWriteTask]
```

Backfill plan 只描述要做什么，不绑定 Ray 或 Daft 如何执行。

## 7. Planner 层

Planner 负责把 Milvus metadata 或 snapshot metadata 转为通用 task。

### 7.1 Client mode planner

流程：

1. 连接 Milvus，获取 collection schema。
2. 获取 collection ID、partition ID、segment list。
3. 只选择 StorageV3 manifest-backed segments。
4. 根据 `columns` 做 projection。
5. 生成 `SegmentReadTask`。

如果 collection 中存在不在支持范围内的 segment，planner 应明确报错或在显式配置下跳过，不应静默降级。

### 7.2 Snapshot mode planner

流程：

1. 读取 snapshot metadata JSON。
2. 解析 collection schema、collection ID、partition IDs。
3. 解析 snapshot 中的 StorageV3 manifest entries。
4. 根据 `columns` 做 projection。
5. 生成 `ReadPlan`。

必须保留的约束：

- StorageV3 reader 使用 fieldID-as-string。
- `segment_id`、`row_offset` 等元数据列由上层补充。
- 非 StorageV3 segment 不在 Python Connector 支持范围内。

## 8. Core 低层 API

Core API 应允许任何 Python 项目直接使用，不依赖执行引擎。

### 8.1 规划读取

```python
from milvus_connector.core import plan_snapshot_read, plan_collection_read

plan = plan_snapshot_read(
    snapshot_path="s3://bucket/files/snapshots/.../metadata.json",
    storage=storage,
    columns=["id", "vector"],
    extra_columns=["segment_id", "row_offset"],
)
```

### 8.2 顺序读取 segment

```python
from milvus_connector.core import read_segment

for task in plan.tasks:
    for batch in read_segment(task):
        consume(batch)
```

### 8.3 转 PyArrow Table

```python
table = read_segment_as_table(task)
```

### 8.4 规划 Backfill

```python
plan = plan_backfill(
    snapshot_path="s3://bucket/snapshots/foo.json",
    backfill_data="s3://source/new_fields.parquet",
    storage=storage,
    source_storage=source_storage,
    mode="coalesce",
    column_mapping={"pk": "id", "new_col": "new_field"},
)
```

### 8.5 顺序执行 Backfill

用于小数据测试或 debug：

```python
result = execute_backfill_local(plan)
```

## 9. 高层 Engine API

用户侧 API 通过 `engine` 参数选择执行后端：

```python
import milvus_connector as mc

ray_ds = mc.read_snapshot(..., engine="ray")
daft_df = mc.read_snapshot(..., engine="daft")
pd_df = mc.read_snapshot(..., engine="pandas")
```

或者显式使用 engine module：

```python
from milvus_connector.engines.ray import read_snapshot
from milvus_connector.engines.daft import read_snapshot
```

## 10. Execution Engine Adapter

定义基础接口：

```python
class ExecutionEngine:
    def read_segments(self, tasks: list[SegmentReadTask]) -> Any:
        ...

    def read_parquet(self, path: str, storage: StorageConfig) -> Any:
        ...

    def join_on_pk(self, source: Any, backfill: Any, pk: str) -> Any:
        ...

    def group_by_segment_sort(self, data: Any) -> Iterable[tuple[int, pa.Table]]:
        ...

    def to_arrow_batches(self, data: Any) -> Iterable[pa.RecordBatch]:
        ...

    def execute_backfill(self, plan: BackfillPlan) -> BackfillResult:
        ...
```

不同 engine adapter 只负责执行模型，不重复实现 Milvus 存储语义。

## 11. Ray Adapter

Ray 是第一个分布式后端，适合大规模读取和 Backfill。

### 11.1 读取

```text
ReadPlan.tasks
  -> Ray task per segment
  -> MilvusStorageAdapter.read_segment
  -> PyArrow Table
  -> ray.data.Dataset
```

API：

```python
ds = mc.read_snapshot(..., engine="ray")
```

返回：

```python
ray.data.Dataset
```

### 11.2 Backfill

Ray Backfill pipeline：

1. Core 生成 `BackfillPlan`。
2. Ray 读取 source collection 为 Dataset。
3. Ray 读取 backfill parquet 为 Dataset。
4. 按 PK join。
5. 应用 merge mode。
6. 按 `segment_id` group/repartition。
7. 每个 segment 内按 `row_offset` 排序。
8. Ray task 调用 Core writer 写 StorageV3 segment。
9. 汇总 result。

Ray adapter 可以提供两种 join backend：

- Ray Data join。
- 手工 hash bucket join，用于大规模稳定性优化。

## 12. Daft Adapter

Daft 更适合 DataFrame / query engine 场景，尤其是 lazy execution、projection/filter pushdown 和与其他 lakehouse 数据源混合分析。

### 12.1 读取

Daft adapter 应尽量把 Milvus snapshot 暴露为自定义 scan source：

```text
MilvusSnapshotScan
  -> list[SegmentReadTask]
  -> each scan task reads Arrow batches
  -> Daft DataFrame
```

API：

```python
df = mc.read_snapshot(..., engine="daft")
```

返回：

```python
daft.DataFrame
```

### 12.2 Projection / Filter

Daft planner 可以把 projection 传给 Core planner，避免读取无关字段。

Filter pushdown 分阶段实现：

1. 第一版：由 Daft 在读取后过滤。
2. 后续：将可下推的简单 filter 转为 `SegmentReadTask.filter`。
3. StorageV3 可尝试复用 native reader 的 filter 能力。

### 12.3 Backfill

Daft 适合表达 dataframe join，但 direct storage write 仍应回到 Core writer。

流程与 Ray 类似：

```text
Daft source df
  join backfill df by PK
  merge mode expression
  group by segment_id
  sort by row_offset
  Core writer writes segment
```

第一版可以优先做 Daft read adapter，Backfill 仍以 Ray/local 为主。

## 13. Pandas / Polars / DuckDB Adapter

这些 adapter 主要用于本地开发、测试、小数据验证和生态集成。

### 13.1 Pandas

- 适合小数据 debug。
- Core batches 转 Pandas DataFrame。
- Backfill 可用 Pandas merge 实现，作为语义基准。

### 13.2 Polars

- 适合本地高性能 DataFrame。
- Core 输出 PyArrow Table，Polars 可直接 ingest。
- 可实现 lazy scan 或 eager read。

### 13.3 DuckDB

- 适合 SQL 分析。
- Core 输出 Arrow Table 后注册为 DuckDB relation。
- Backfill join 语义可通过 SQL 验证。

## 14. Backfill 抽象设计

Backfill 不应写死在 Ray pipeline 中，而应拆为：

1. `plan_backfill(...)`：Core 生成 logical plan。
2. `execute_backfill(plan, engine=...)`：engine adapter 负责执行。

### 14.1 Core 负责的语义

- 识别 primary key。
- 应用 column mapping。
- 校验目标字段存在。
- 校验 PK 类型。
- 校验 `coalesce/overwrite` 下字段类型精确匹配。
- 定义 merge mode 语义。
- 规划每个 segment 的 write task。
- 定义 result JSON schema。

### 14.2 Engine 负责的执行

- 读取 source collection。
- 读取 backfill parquet。
- 检查 backfill PK 重复。
- 按 PK join。
- 计算 merge mode 表达式。
- 生成 provenance flags。
- 按 segment 分组。
- 按 `row_offset` 排序。
- 调用 Core writer。
- 汇总结果。

### 14.3 Merge mode

必须与 Spark 版本一致：

| Mode | 语义 | 是否读取源端目标字段 |
|------|------|----------------------|
| `replace` | parquet 是目标字段完整事实来源；未匹配源行目标字段为 null | 否 |
| `coalesce` | 源值非空保留源值，源值为空使用 parquet 值；未匹配源行保留源值 | 是 |
| `overwrite` | parquet 命中的 PK 覆盖源值，null 也覆盖；未命中源行保留源值 | 是 |

### 14.4 顺序保证

Backfill 写入前必须满足：

```text
同一个 segment 内按 row_offset 升序写入
```

不能依赖任何 engine 的 block 原始顺序。

## 15. Inspection 抽象设计

Inspection 应该是 Core 能力，不依赖 Ray 或 Daft。

API：

```python
info = inspect_snapshot(snapshot_path, storage=storage)
```

CLI：

```bash
ray-milvus inspect \
  --snapshot s3://bucket/files/snapshots/.../metadata.json \
  --s3-endpoint localhost:9000 \
  --s3-bucket bucket
```

输出内容：

- snapshot id / name。
- collection id / schema。
- partition ids。
- StorageV3 manifest segment 列表。
- manifest path、segment id、partition id、row count、manifest version。
- 缺失 manifest、路径无法解析、不支持 storage version 等诊断。

## 16. Vector 抽象设计

Vector functions 也应分 Core 和 Engine adapter：

- Core 提供 NumPy/PyArrow 级别函数。
- Engine adapter 将函数映射到 Ray map_batches、Daft expressions、Polars expressions 或 Pandas apply。

函数包括：

- `cosine`
- `l2`
- `inner_product`
- `hamming`
- `jaccard`
- `vector_knn`

`vector_knn` 建议使用两阶段 topK：

1. 每个 block / partition / segment 计算 local topK。
2. reduce 阶段合并 global topK。

## 17. CLI 设计

统一 CLI 名称为：

```bash
ray-milvus
```

子命令：

```bash
ray-milvus inspect ...
ray-milvus read ...
ray-milvus backfill ...
ray-milvus insert ...
ray-milvus write-segment ...
```

示例：

```bash
ray-milvus backfill \
  --engine ray \
  --snapshot s3://milvus-bucket/snapshots/foo.json \
  --backfill-data s3://source-bucket/new_fields.parquet \
  --mode coalesce \
  --column-mapping pk:id,new_col:new_field \
  --s3-endpoint s3.us-west-2.amazonaws.com \
  --s3-bucket milvus-bucket \
  --s3-region us-west-2 \
  --output-result s3://milvus-bucket/backfill/result.json
```

## 18. 开发阶段建议

### Phase 1：Core MVP

- 建立 Python package 结构。
- 定义 config、schema、task dataclass。
- 实现 snapshot JSON parser。
- 实现 StorageV3 read planner。
- 封装 milvus-storage Python read API。
- 提供 local sequential read。

### Phase 2：Ray Read Adapter

- 实现 Ray task per segment。
- 返回 Ray Dataset。
- 支持 projection。
- 支持 `segment_id`、`row_offset`。
- 支持 StorageV3 snapshot。

### Phase 3：Inspection CLI

- 实现 engine-independent inspect。
- 支持 JSON 和 human-readable 输出。
- 对 StorageV3 manifest 和元数据缺失给出诊断。

### Phase 4：Backfill Core + Ray Execution

- 实现 `plan_backfill`。
- 实现 Ray Backfill execution。
- 支持 `replace/coalesce/overwrite`。
- 支持 provenance stats。
- 支持 StorageV3 writer。

### Phase 5：Daft Read Adapter

- 实现 Daft read snapshot。
- 优先支持 projection。
- 后续支持 filter pushdown。
- 暂缓 Daft Backfill，等 read adapter 稳定后再做。

### Phase 6：本地 DataFrame Adapters

- Pandas adapter。
- Polars adapter。
- DuckDB adapter。
- 用作小数据测试和语义基准。

### Phase 7：Packaging / CI

- `pyproject.toml` extras。
- Docker runtime image。
- native dependency smoke check。
- Unit tests。
- MinIO/Milvus integration tests。
- Spark/Python connector parity tests。

## 19. 测试策略

### 19.1 Core unit tests

- snapshot JSON 解析。
- StorageV3 manifest list 解析。
- schema 转换。
- storage version dispatch。
- fieldID-as-string Arrow column naming。

### 19.2 Storage adapter tests

- milvus-storage Python API load。
- StorageV3 read。
- StorageV3 write。
- S3/IAM/MinIO 配置。
- PyArrow batch 输出。

### 19.3 Engine adapter tests

- Ray read snapshot。
- Ray Backfill merge mode。
- Daft read snapshot。
- Pandas/Polars/DuckDB 小数据读取。

### 19.4 Parity tests

与当前 Spark 版本对齐：

- 同一 StorageV3 snapshot 的 schema。
- 同一 StorageV3 snapshot 的 row count。
- 同一 projection 的字段结果。
- 同一 Backfill 输入的 result JSON 关键统计。
- 同一 vector 函数的计算结果。

## 20. 关键风险

### 20.1 milvus-storage Python API 覆盖不足

如果 Python API 不支持某些 StorageV3 现有能力，需要在 Storage Adapter 层补齐或临时 bridge。

### 20.2 多执行引擎语义不一致

Backfill join、null 处理、排序、类型转换在不同 DataFrame engine 中可能存在差异。Core 需要定义明确语义，adapter 必须通过 parity tests。

### 20.3 不支持范围内 segment 的处理

Python Connector 只支持 StorageV3。遇到其他存储格式时，planner 必须明确报错或在显式配置下跳过，不能静默产生不完整结果。

### 20.4 Backfill 大规模 shuffle

Ray 和 Daft 的 join/shuffle 能力不同。Backfill execution 需要支持不同 backend，并提供参数控制分区数、spill、parallelism。

### 20.5 Native dependency 分发

`milvus-storage` native 依赖、Arrow、glibc、平台架构和对象存储 SDK 需要清晰的 packaging 策略。生产使用建议提供官方 Docker runtime image。

## 21. 结论

后续不应只开发一个绑定 Ray 的 connector，而应开发一个 **Milvus Python Connector Core**。

核心抽象是：

```text
Core = snapshot + planner + storage + schema + backfill semantics + inspection
Adapters = Ray / Daft / Pandas / Polars / DuckDB / Local
```

这样可以确保：

- Ray 是第一个执行后端，但不是架构中心。
- Daft 或其他 Python 项目可以复用同一套 Milvus 逻辑。
- StorageV3、Backfill、schema 和 manifest 语义只实现一次。
- 后续新增执行引擎时只需要实现 adapter，不需要重写 Milvus connector core。
