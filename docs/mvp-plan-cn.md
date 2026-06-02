# ray-milvus MVP 规划

本文档定义 ray-milvus 的第一版 MVP 范围。MVP 目标是先完成一个可验证、可测试、可扩展的数据读取闭环，而不是一次性实现完整工具集。

## 1. MVP 目标

MVP 只聚焦 Milvus / Milvus Lite 离线 snapshot 读取和 inspection 能力。读取路径通过统一 storage adapter 暴露，上层不感知底层是 `milvus-storage` 还是 lite-storage。

核心闭环：

```text
snapshot metadata -> read plan -> unified StorageReader -> PyArrow Table -> inspect CLI
```

MVP 完成后，应能够：

1. 解析 Milvus snapshot metadata。
2. 识别 Milvus StorageV3 manifest-backed segment 和 Milvus Lite 可读取 segment。
3. 生成 engine-neutral `ReadPlan`。
4. 通过 local engine 和统一 `StorageReader` 顺序读取 segment。
5. 输出 PyArrow `Table`。
6. 提供 snapshot inspection API 和 CLI。
7. 对非支持范围内的数据格式明确报错。

## 2. MVP 功能范围

### 2.1 必须支持

- Python package 基础结构。
- Storage 配置模型。
- Snapshot JSON 解析。
- Milvus collection schema 解析。
- StorageV3 manifest metadata 解析。
- Milvus Lite 本地 metadata / segment 定位解析。
- Projection：按字段名选择读取列。
- Metadata columns：支持 `segment_id` 和 `row_offset`。
- Engine-neutral read plan。
- Local sequential read engine。
- 统一 `StorageReader` protocol / factory。
- `milvus-storage` Python API adapter。
- Milvus Lite lite-storage adapter。
- `MilvusDataset.to_arrow()`。
- `inspect_snapshot(...)` API。
- `ray-milvus inspect` CLI。
- 单元测试和基础集成测试。

### 2.2 明确不支持

MVP 不做以下能力：

1. 在线读取 Milvus collection。
2. 写入 StorageV3 segment。
3. Backfill。
4. Ray adapter。
5. Daft adapter。
6. Pandas / Polars / DuckDB adapter。
7. import / export workflow。
8. transform API。
9. validate workflow。
10. vector KNN 或距离计算。
11. 非 StorageV3 segment。
12. 旧 packed parquet 格式。
13. filter pushdown。
14. 多 cloud storage 的完整适配矩阵。

遇到不支持的 segment、storage version 或格式时，planner 应抛出明确错误，不应静默跳过或降级。

## 3. 推荐目录结构

MVP 阶段保持结构紧凑：

```text
ray_milvus/
  __init__.py
  api.py
  types.py
  errors.py

  core/
    __init__.py
    schema.py
    snapshot.py
    manifest.py
    plans.py
    planner.py
    dataset.py
    inspection.py

  io/
    __init__.py
    object_store.py
    storage.py
    milvus_storage.py
    lite_storage.py
    reader.py

  engines/
    __init__.py
    local.py

  cli/
    __init__.py
    main.py

tests/
  fixtures/
  unit/
  integration/
```

MVP 阶段暂不创建 `ops/`。Backfill、transform、validate、migrate、repair 等离线任务语义等到读取闭环稳定后再加入。

## 4. Public API

### 4.1 读取 snapshot

```python
import ray_milvus as rm

storage = rm.StorageConfig(
    endpoint="localhost:9000",
    bucket="bucket",
    access_key="minioadmin",
    secret_key="minioadmin",
    use_ssl=False,
)

# Milvus Lite 使用同一个上层 API，只替换 storage config：
# storage = rm.StorageConfig(path="/path/to/milvus_lite.db")

dataset = rm.read_snapshot(
    snapshot_path="s3://bucket/snapshots/foo.json",
    storage=storage,
    columns=["id", "vector"],
    include=["segment_id", "row_offset"],
)

table = dataset.to_arrow()
```

### 4.2 Inspection API

```python
import ray_milvus as rm

info = rm.inspect_snapshot(
    snapshot_path="s3://bucket/snapshots/foo.json",
    storage=storage,
)

print(info.segments)
```

### 4.3 CLI

MVP 必须支持：

```bash
ray-milvus inspect \
  --snapshot s3://bucket/snapshots/foo.json \
  --s3-endpoint localhost:9000 \
  --s3-bucket bucket \
  --s3-access-key minioadmin \
  --s3-secret-key minioadmin
```

可选支持：

```bash
ray-milvus read \
  --snapshot s3://bucket/snapshots/foo.json \
  --columns id,vector \
  --output /tmp/out.arrow
```

如果 `read` CLI 会拖慢 MVP，可以先只实现 `inspect` CLI，把读取能力保留在 Python API 中。

## 5. 内部数据流

### 5.1 `read_snapshot(...)`

```text
read_snapshot(...)
  -> normalize ReadOptions
  -> load snapshot JSON
  -> parse collection schema
  -> parse segment metadata
  -> build ReadPlan
  -> local engine executes SegmentReadTask sequentially
  -> io.storage creates StorageReader from StorageConfig
  -> MilvusStorageReader calls milvus-storage, or MilvusLiteStorageReader calls lite-storage
  -> returns PyArrow RecordBatch/Table
  -> MilvusDataset.to_arrow()
```

### 5.2 `inspect_snapshot(...)`

```text
inspect_snapshot(...)
  -> load snapshot JSON
  -> parse collection schema
  -> parse partitions and segments
  -> parse StorageV3 manifest metadata
  -> build InspectionResult
  -> return API object or CLI output
```

Inspection 不需要执行 engine，也不需要读取完整 segment 数据。

## 6. 核心对象

### 6.1 `StorageConfig`

```python
@dataclass
class StorageConfig:
    # Public API accepts MilvusStorageConfig | MilvusLiteStorageConfig.
    # Core/planner/engine should treat both as opaque StorageConfig.

@dataclass
class MilvusStorageConfig:
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

@dataclass
class MilvusLiteStorageConfig:
    path: str
    collection: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

StorageConfig = MilvusStorageConfig | MilvusLiteStorageConfig
```

### 6.2 `ReadOptions`

```python
@dataclass
class ReadOptions:
    snapshot_path: str
    storage: StorageConfig
    columns: list[str] | None = None
    include: list[str] = field(default_factory=list)
```

### 6.3 `SegmentReadTask`

```python
@dataclass
class SegmentReadTask:
    segment_id: int
    partition_id: int
    schema: MilvusSchema
    projected_fields: list[str]
    projected_field_ids: list[int]
    include: list[str]
    storage: StorageConfig
    manifest_path: str
    row_count: int | None = None
    manifest_version: int | None = None
```

### 6.4 `ReadPlan`

```python
@dataclass
class ReadPlan:
    schema: MilvusSchema
    tasks: list[SegmentReadTask]
```

### 6.5 `MilvusDataset`

```python
@dataclass
class MilvusDataset:
    schema: MilvusSchema
    read_plan: ReadPlan

    def to_arrow(self) -> pa.Table:
        ...
```

### 6.6 `InspectionResult`

```python
@dataclass
class InspectionResult:
    collection: CollectionMetadata
    schema: MilvusSchema
    segments: list[SegmentMetadata]
    diagnostics: list[InspectionDiagnostic]
```

## 7. 模块职责

### 7.1 `api.py`

负责提供用户入口：

- `read_snapshot(...)`
- `inspect_snapshot(...)`

`api.py` 只做参数归一化、调用 planner / inspection / engine，不实现 snapshot 解析、StorageV3 manifest 解析或 storage 读取细节。

### 7.2 `core/`

`core/` 是 Milvus 语义中心，负责：

- schema model。
- snapshot parser。
- manifest parser。
- read plan / segment task。
- planner。
- dataset abstraction。
- inspection model 和诊断逻辑。

`core/` 不依赖 Ray、Daft、Pandas、Polars、DuckDB，也不直接依赖 `milvus-storage` 或 lite-storage native API。

### 7.3 `io/`

`io/` 负责外部系统访问：

- 读取对象存储或本地路径中的 snapshot JSON。
- 定义统一 `StorageReader` protocol 和 factory。
- 封装 `milvus-storage` Python API 为 `MilvusStorageReader`。
- 封装 Milvus Lite 本地数据访问为 `MilvusLiteStorageReader`。
- 根据 `SegmentReadTask` 读取 segment。
- 管理 PyArrow batch/table 输出。

上层不应直接调用 `milvus-storage` 或 lite-storage，也不应分支判断 Milvus / Milvus Lite；差异应由 `io.storage` 根据 `StorageConfig` 创建的 reader 实现处理。

### 7.4 `engines/local.py`

Local engine 是 MVP 的唯一执行引擎，负责：

- 顺序执行 `ReadPlan.tasks`。
- 调用 `io.reader` 和统一 `StorageReader` 读取 segment。
- 合并 PyArrow batches / tables。

Local engine 是后续 Ray、Daft adapter 的语义基准。

### 7.5 `cli/main.py`

CLI 保持薄层：

- 解析命令行参数。
- 构造 `StorageConfig`。
- 调用 `api.inspect_snapshot(...)`。
- 输出 JSON 或 human-readable 结果。

CLI 不应绕过 `api.py` 直接访问 storage internals。

## 8. 依赖边界

MVP 阶段依赖方向：

```text
api -> core / engines
engines -> core / io
io -> core
cli -> api
```

禁止方向：

```text
core    -/-> io
core    -/-> engines
core    -/-> milvus-storage / lite-storage
core    -/-> ray / daft / pandas / polars / duckdb
io      -/-> engines
cli     -/-> storage internals
```

核心原则：

```text
core 定义 Milvus 语义；io 负责数据访问；engines 执行计划；api 暴露稳定入口。
```

## 9. 测试范围

### 9.1 Unit tests

必须覆盖：

- `StorageConfig` 参数映射。
- snapshot JSON parser。
- Milvus schema parser。
- StorageV3 manifest metadata parser。
- projection 字段选择。
- 不存在字段报错。
- 非支持的 Milvus / Milvus Lite segment 报错。
- `ReadPlan` 生成。
- `InspectionResult` 生成。

### 9.2 Integration tests

MVP 至少需要一个基础集成测试：

```text
fixture snapshot JSON
  + fake/minimal StorageV3 or Milvus Lite metadata
  + mocked StorageReader adapter
  -> read_snapshot(...)
  -> PyArrow Table
```

如果 `milvus-storage` Python API、本地对象存储环境或 Milvus Lite 本地 fixture 可用，再增加真实 smoke test：

```text
MinIO + StorageV3 fixture
  -> read_snapshot(...)
  -> PyArrow Table
```

真实 StorageV3 smoke test 可以先标记为 integration，不阻塞普通 unit test。

## 10. 成功标准

MVP 完成时必须满足：

1. `pip install -e .` 成功。
2. `pytest` 成功。
3. `ray-milvus inspect ...` 能输出 snapshot / schema / segment / manifest 信息。
4. `rm.inspect_snapshot(...)` 可用。
5. `rm.read_snapshot(...).to_arrow()` 可用。
6. 能通过统一 `StorageReader` 读取 Milvus StorageV3 manifest-backed segment。
7. 能通过同一上层读取路径接入 Milvus Lite storage reader。
8. 遇到非支持数据明确报错。
9. `core/` 没有 Ray / Daft / Pandas / Polars / DuckDB 依赖。
10. `milvus-storage` 和 lite-storage API 细节只出现在 `io/` 层。
11. Local engine 可以作为后续 Ray / Daft adapter 的行为基准。

## 11. 建议实现顺序

推荐按以下顺序实现：

1. 项目初始化：`pyproject.toml`、package skeleton、test skeleton。
2. 基础类型和异常：`types.py`、`errors.py`。
3. Schema model 和 schema parser。
4. Snapshot parser。
5. Manifest parser。
6. ReadPlan / SegmentReadTask。
7. Snapshot read planner。
8. Inspection API。
9. CLI inspect。
10. `StorageReader` protocol / factory。
11. `MilvusStorageReader` adapter。
12. `MilvusLiteStorageReader` adapter。
13. Local read engine。
14. `MilvusDataset.to_arrow()`。
15. 基础集成测试。
16. 更新 `CLAUDE.md` 中实际开发命令。

## 12. MVP 之后的扩展顺序

MVP 稳定后再扩展：

1. Ray read adapter。
2. Backfill core semantics。
3. Ray backfill execution。
4. Daft read adapter。
5. Validate workflow。
6. Transform API。
7. Import / export workflow。
8. Pandas / Polars / DuckDB adapters。
9. Vector functions。

这些能力都应复用 MVP 中已经稳定的 snapshot parser、manifest parser、read planner、统一 StorageReader adapter 和 local semantic baseline。
