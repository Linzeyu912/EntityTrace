# EntityTrace 宿主 Agent 调用说明（`entitytrace.v1`）

> **边界声明**：本文档**提供通用 CLI 调用说明，平台封装待核实**。当前交付形态为本地 Python CLI（Python 3.12+，标准库 SQLite），尚未完成 特定平台的专用安装包、宿主适配或集成验证。

---

## 1. 何时调用

当用户需要对长短篇小说／叙事文本执行以下任务时，由具备本地命令执行能力的宿主 agent 调用 `python -X utf8 -m entitytrace`：

1. **实体提及发现**：定位全文具体人物、地点、组织、器物等提及的 Unicode 码点地址 `[start, end)`。
2. **身份判断与名册归组**：将跨章全名、简称、别名与指代链接为实体名册（默认算法 `entity-roster-v1`），并按影响范围（`impact`）列出待审疑点。
3. **按需取证与确定性纠错**：无需将整份数兆字节的 `result.json` 读入上下文窗口，直接通过只读查询分页检视实体、回溯原文上下文，并在获得用户明确确认后叠加可撤回的更正。

---

## 2. 统一响应外壳（`protocol: "entitytrace.v1"`）

所有开启 `--json` 的命令以及默认输出 JSON 的子命令（`status`、`entities`、`entity`、`review`、`correct`）遵守同一约定：

- **`stdout` 只输出恰好一行 UTF-8 JSON**；进度与人类可读提示一律写入 `stderr`。
- **成功响应**：
  ```json
  {"protocol":"entitytrace.v1","command":"<命令名>","run_id":"...","result_revision":"...","error":null}
  ```
  其后拼接各命令自身载荷字段。每次查询与更正回执都携带当前最新的 `result_revision`，可直接传给下一步 `correct --expect-revision`。
- **失败响应**（同样输出至 `stdout` 单行 JSON）：
  ```json
  {"protocol":"entitytrace.v1","command":"<命令名>","run_id":"<已知则填，否则 null>","error":{"category":"...","message":"...","retryable":false}}
  ```
  注：`run --json` 在运行尚未建立（`run_id` 为 `null`）即失败时，`status` 固定为 `"failed"`，退出码为 `1`。

---

## 3. 标准调用顺序

宿主 agent 应按以下 5 步编排，全程通过分页查询与码点取证控制上下文体积：

### 第 1 步：启动或续跑全文处理（`run --json`）

```powershell
python -X utf8 -m entitytrace run "<输入TXT路径>" --db "<数据库路径>" --output "<输出根目录>" --config "<凭证配置文件>" --max-requests 1000 --max-seconds 2400 --json
```

> **预算估算参考**：CLI 默认不限制 `--max-requests` 与 `--max-seconds`；宿主按需设置单次运行预算时，可参考实测规模约 **每万字 10—17 次请求、14—18 秒** 估算。

读取 `stdout` 单行回执中的核心字段：
- `run_id`、`result_revision`、`status`（`completed` | `completed_with_review` | `incomplete` | `failed`）
- `stages`：`chapters`、`discovery`、`identity` 各自的 `{"complete","seconds","resumed"}`
- `counts`：`{"mentions","entities","review_items"}`
- `review_top`：按 `impact` 降序前 10 条未决待审摘要 `[{"id","stage","reason","impact"}]`
- `output_dir`、`files`（`result`、`review`、`manifest` 路径）与 `error`

### 第 2 步：按需只读查询（零模型调用，内存实时构建）

全部只读查询均通过内存实时重放有效用户更正（不依赖磁盘导出文件，仅支持 `entity-roster-v1` 运行；分页参数 `--limit` 默认 `50`、上限 `200`，有下一页时 `next_offset` 为整数，否则为 `null`）：

1. **查运行与断点状态**（未完成运行亦可查）：
   ```powershell
   python -X utf8 -m entitytrace status <run_id> --db "<数据库路径>" --output "<输出根目录>"
   ```
   返回：`status`、`stages`、`last_error`、`counts`、`output_dir`、`files`、`result_revision`。
2. **查待审清单**（默认 `--status pending`，按 `impact` 降序、`id` 升序）：
   ```powershell
   python -X utf8 -m entitytrace review <run_id> --status pending --limit 50 --offset 0 --db "<数据库路径>"
   ```
   可选：`--status pending|resolved|all`、`--stage <阶段名>`。返回：`counts`、`total`、`limit`、`offset`、`next_offset`、`items`。
3. **查实体名册列表或按精确称呼检索**：
   ```powershell
   python -X utf8 -m entitytrace entities <run_id> --kind person --min-mentions 5 --limit 50 --offset 0 --db "<数据库路径>"
   python -X utf8 -m entitytrace entities <run_id> --surface "<精确称呼>" --db "<数据库路径>"
   ```
   返回 `items`（按 `mention_count` 降序、`id` 升序），每项含 `id`、`name`、`kind`、`mention_count`、`first_chapter_id`、`identity_status`；指定 `--surface` 时仅返回包含该精确提及文本的实体，并附加 `surface_count`。
4. **查单个实体详情与提及片段**（只接受实体 ID）：
   ```powershell
   python -X utf8 -m entitytrace entity <run_id> <entity_id> --limit 50 --offset 0 --context 80 --db "<数据库路径>"
   ```
   返回：`entity_id`、`name`、`names`、`aliases`、`kind`、`identity_status`、`mention_count`、`first_chapter_id`、`link_methods`、`review_ids`（相关未决待审 ID），以及分页 `mentions`（每项含 `id`、`chapter_id`、`start`、`end`、`text`、`link_method`、`snippet`；`--context` 范围 `0—400` 码点，默认 `80`，`snippet` 绝不跨章节边界）。

### 第 3 步：按码点地址回取原文核实证据（`quote --json`）

对待审项 `evidence` 或提及 `[start, end)` 需要更大上下文时调用：

```powershell
python -X utf8 -m entitytrace quote <run_id> <start> <end> --json --context 160 --db "<数据库路径>"
```

返回：`start`、`end`、`text`、`context_before`、`context_after`、`chapter_id`、`result_revision`。

### 第 4 步：与用户确认后写入更正（`correct`，零模型调用）

- 若宿主仅做自身推断、**尚未获用户明确确认**，只能写 `--source agent`（仅存入 `review.json` 的 `agent_suggestions`，不改变 `result.json` 与 `result_revision`）。
- **获得用户明确确认后**，使用最近一次查询得到的 `result_revision` 执行 `--source user`（若同时消解某条待审项则附 `--resolves <review_id>`）：

```powershell
python -X utf8 -m entitytrace correct <run_id> merge <保留目标实体ID> <被并实体ID> --source user --expect-revision <result_revision> [--resolves <review_id>] [--note "<依据>"] --db "<数据库路径>" --output "<输出根目录>"
python -X utf8 -m entitytrace correct <run_id> split-surface <实体ID> "<称呼>" --source user --expect-revision <result_revision> --db "<数据库路径>" --output "<输出根目录>"
python -X utf8 -m entitytrace correct <run_id> assign <提及ID> <目标实体ID或new> --source user --expect-revision <result_revision> [--resolves <review_id>] --db "<数据库路径>" --output "<输出根目录>"
python -X utf8 -m entitytrace correct <run_id> not-entity <提及ID> --source user --expect-revision <result_revision> [--resolves <review_id>] --db "<数据库路径>" --output "<输出根目录>"
python -X utf8 -m entitytrace correct <run_id> revoke <更正ID> [--cascade] --source user --expect-revision <result_revision> --db "<数据库路径>" --output "<输出根目录>"
```

返回：`correction_id`、`status`（`applied` | `suggested` | `duplicate` | `revoked`）、`result_revision_before`、`result_revision`、`affected_entity_ids`、`output_dir`。

### 第 5 步：复查或移交下游

更正生效后 `result_revision` 更新；可再次调用 `review`、`entities` 或 `entity` 确认待审项已转为 `resolved` 且名册已更新。

---

## 4. 退出码与 `error.retryable` 决策表

**核心准则**：退出码表示**数据库数据状态**（`0` 完成、`2` 有可续跑断点、`1` 无可续跑或命令被拒）；**是否允许自动重试只看 `error.retryable`**。`error.category` 严格限定为以下 8 类：

| 退出码 | `error.category` | `error.retryable` | 典型场景 | 宿主 Agent 标准动作 |
| --- | --- | --- | --- | --- |
| `0` | `null` | — | `run` 完成（含 `completed` 与 `completed_with_review`）、查询成功、`correct` 成功 | 正常读取 JSON 结果；若 `status="completed_with_review"`，按 `review_top` 或 `review` 查看高影响待审项 |
| `2` | `network` | `true` | 网络连接中断、超时、服务端 `5xx` 或连续断网熔断，数据库已保存断点 | 可退避等待后使用**同一 `--db` 与输入参数**自动重跑 `run --json` 断点续跑 |
| `2` | `budget` | `true` | 触达 `--max-requests` 或 `--max-seconds` 预算上限停止，已完成任务保存在库 | 在宿主被授予的总预算范围内可扩大预算参数同库续跑；超出用户授权预算时询问用户 |
| `2` | `model_output` | `true` | 个别任务模型输出连续格式错误，阶段未完成但留有断点 | 允许同库自动续跑 **1 次**（已完成任务全部复用不重发）；若仍失败则停止并向用户报告 |
| `2` | `quota` | **`false`** | API 余额不足、硬额度耗尽（`hard_quota`）或 `402/429` 配额限制，库内保留断点 | **严禁自动重跑**；必须立即向用户报告额度耗尽，待用户充值或更换凭证后再同库恢复 |
| `2` | `null` | — | 显式指定 `--stop-after chapters` 或 `--stop-after discovery` 到达阶段断点 | 按编排计划决定是否继续推进后续阶段 |
| `1` | `conflict` | `false` | `correct --expect-revision` 与当前实际 `result_revision` 不一致（结果版本已过期） | **不得直接用旧参数强推**；必须重新调用 `status` / `entities` / `entity` / `review` 读取最新状态与实体 ID，核对后再提交 |
| `1` | `config` | `false` | 缺少 `DEEPSEEK_API_KEY`、配置文件路径错误、服务地址非法或 `401/403` 鉴权失败 | **禁止自动重试**；提示用户检查本地配置文件或环境变量凭证配置 |
| `1` | `input` | `false` | 输入文件不存在／编码错误、章节结构阻断（区间内存在缺失章号的疑似标题或重复章节，`blocking=true`）、未知 `run_id`／`entity_id`、`--limit > 200`、`groups-v2` 运行调用只读查询、未完成运行调用 `correct` | **禁止自动重跑**（章节阻断重跑命中缓存结果相同，可通过 `review <run_id> --stage chapters` 核查；原文章号跳号或回退后按新章题连续递增记为 `blocking=false` 提示，不阻断运行）；检查参数或询问用户 |
| `1` | `local` | `false` | 保存的原文 SHA-256 指纹不匹配或本地覆盖不变量校验异常 | **立即停止**并向用户报告数据库／产物完整性异常，不得覆盖原库 |

---

## 5. 硬性约束（宿主 Agent 必须遵守）

1. **没有用户的明确确认，严禁传 `--source user`**：宿主 agent 自行阅读上下文做出的消歧判断只能使用 `--source agent` 记录为建议，绝不能冒充人工确认污染 `result.json`。
2. **实体 ID 不是跨版本永久编号**：名册实体的 `id` 由成员首处地址与强称呼集合摘要计算，用户更正（如 `split-surface`、`assign ... new`、`not-entity`）也会派生新实体 ID 或移除被并入的旧实体 ID。宿主不得跨运行或跨版本硬编码缓存实体 ID。
3. **不把模型自动归组或计算完成说成人工标准答案**：退出码 `0` 仅代表管线计算与地址校验完成，不代表语义 100% 准确；`identity_status="unresolved_singleton"` 仅代表当前证据不足，不代表该提及与全书其他实体必然不同。
4. **每次写操作必须绑定最新 `result_revision`**：执行任何 `correct` 前必须传入最近一次回执中的 `--expect-revision`；收到 `conflict`（退出码 `1`）后必须重新读取最新数据。
