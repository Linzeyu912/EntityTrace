# EntityTrace · 实体溯源

EntityTrace 是面向 agent 的插件，提供小说实体发现、身份判断与归组，以及原文证据回溯能力。项目定位分为工具形态和功能定位两层。

## 工具形态

**面向 agent 调用的插件（plugin）**，回答“以什么形式被使用”。目标调用方包括各类宿主 agent；宿主负责理解用户需求、组织调用、解释结果及衔接后续工作。由此确定开发边界：不建设独立前端、实体编辑页面或人工操作工作台，需要人参与的确认由宿主在对话中承接。

当前交付形态是可供 agent 执行的本地 CLI 与 JSON 文件，使用 Python 3.12+ 和标准库 SQLite，无额外运行依赖，不需要 Docker 或数据库服务。**尚未提供 特定平台的专用安装包、宿主适配或接入验证。** 插件定位已确定，具体分发与协议封装待落实。默认流程仅调用在线 DeepSeek（仅在显式切换 `--identity-algorithm groups-v2` 时才需要 TypeSafe/Jev）；宿主 agent 的模型与插件内部模型分工独立，本机运行不等于模型离线运行。

## 功能定位

**实体发现、身份判断与归组、原文证据回溯**，回答“具体解决什么问题”。

- **实体发现**：从小说原文发现具体对象的提及，保留称呼与出现位置。
- **身份判断与归组**：依据原文判断哪些提及属于同一个对象，保留不同身份约束和未决项。
- **原文证据回溯**：通过固定文本版本与地址找回原文，追查判断依据和归组过程。

首版输出带原文地址的实体分组 JSON；描述、外貌、服饰和经历提取留到下一阶段。插件形态决定交付与调用方式，功能定位决定处理能力与结果：接入不同宿主可复用同一核心能力，功能扩展按独立需求和质量验证推进。

## Agent 工作流与交付边界

目标工作流是：用户向宿主 agent 提出任务 → agent 调用 EntityTrace → 读取运行状态、结果和待审项 → 按需回取原文、补充确认 → 将带证据的结果交给后续工具。插件的重点是输入输出清楚、状态可判断、失败可恢复、证据可追溯，以及更正后能继续流转。

| 环节 | 当前可用 | 后续补齐 |
| --- | --- | --- |
| 启动与恢复 | `run` 执行全文处理，相同输入与配置复用断点；支持 `--max-requests`、`--max-seconds` 预算与连续断网停止 | 统一机器可读调用回执、细粒度错误回执与独立状态查询 |
| 读取与取证 | JSON／JSONL 文件、`export` 重导出、`quote` 取回原文 | 按需查询，减少 agent 读取整份大型产物 |
| 待审与更正 | `review.json` 按 `impact` 降序提供问题编号与证据；已完成名册运行支持 `correct` 叠加式更正、待审消解（`--resolves`）、版本校验与撤回（零模型调用） | 与按需查询回执联动简化宿主编排 |
| 宿主接入 | 有命令执行和文件访问能力的 agent 可按本文调用 CLI | 核实目标宿主能力后封装适配，验证安装与完整调用流程 |

当前 `run` 的终端输出包含进度文本，不能当作统一 JSON 响应；退出码、产物状态和待审项须结合读取。计算完成不代表全部身份已经确认，宿主 agent 的推断也不能记作用户已确认的答案。

## 当前流程

默认身份流程为 **`entity-roster-v1`（实体名册链接，`--identity-algorithm roster`）**；旧版配对组消歧流程保留为 `--identity-algorithm groups-v2` 供对照与兼容导出。

| 步骤 | 执行者 | 实际行为 |
| --- | --- | --- |
| 导入 TXT | 本地程序 | 保存原始字节、冻结文本及指纹，建立 Unicode 码点地址 |
| 章节识别 | 本地程序候选筛选 + DeepSeek 按 ID 标注 + 本地结构核验 | 提取整行短候选并发标注（`temperature=0`），经“标题后须有正文”与“章号连续”两条确定性结构核验确认 |
| 地址核验 | 本地程序 | 检查章节范围完整覆盖原文，再建立发现任务 |
| 提及发现 | DeepSeek + 本地并发任务池 | 各章独立发现称呼与引用，长章拆任务；逐字核对提及地址 |
| 章内归并（C1） | DeepSeek + 本地结构与边界校验 | 按发现片段整段上下文将称呼归入章内对象，区分专名／称谓／描述词，支持逐提及例外指派与非实体建议 |
| 跨章名册与复核（C2） | 本地并查集 + DeepSeek 分区复核／补合并核验 | 仅 `proper_name` 参与跨章候选聚合；多名称、粗类冲突或弱专名送分区复核；按粗类与跨粗类提议补合并，经双侧 ≥600 码点原文核验确认，不可合并约束全程硬拦截 |
| 未决与非专名链接（C3） | 本地唯一称呼直连 + DeepSeek 候选链接 | 章内对象确定性落入名册；只含非专名称呼的对象与章内未决提及在候选名册与原文证据下链接，受专名一致性硬约束保护 |
| 待审排序与统一导出（C4） | 本地程序 | 汇总身份与定位待审并按受影响提及数（`impact`）降序排列，校验全部码点地址与覆盖完整性，导出 `schema_version=3` 产物 |

默认模型配置仅使用 `deepseek-flash`（章节标注、提及发现与名册身份阶段），无需配置 `TYPESAFE_API_KEY`；仅在切换 `--identity-algorithm groups-v2` 时加载并使用固定版本 `jev-1.13.0` 执行配对身份消歧。`entity-roster-v1` 不使用未校准概率分数作主路由，而是显式记录每处提及的链接来源（`chapter_local`、`c3_model`、`surface_unique`、`unresolved`、`c3_rejected_name_mismatch`、`not_entity_suggested`）与关联请求 ID。

## 运行

以下命令既是宿主 agent 当前的调用入口，也可用于开发调试。以仓库目录为工作目录执行，或显式指定输入、配置、数据库和输出路径。默认 `roster` 流程只需配置 `DEEPSEEK_API_KEY`（即使环境或配置文件中存在 TypeSafe 配置亦忽略，不影响 `run_id`）；仅 `--identity-algorithm groups-v2` 需要额外配置 `TYPESAFE_API_KEY`。新环境可从 `.env.example` 创建配置文件并填写自己的密钥；系统环境变量优先于文件。配置、数据库和运行产物均被 Git 忽略。

```powershell
python -X utf8 -m entitytrace run "test/text-library/texts/03_中文长篇_九命奇冤_36回.txt"
python -X utf8 -m entitytrace run "test/text-library/texts/04_英文长篇_Pride_and_Prejudice_61章.txt"
```

也可处理自己的 UTF-8 TXT，或按需附加请求／时间预算与身份算法开关：

```powershell
python -X utf8 -m entitytrace run "D:\小说\样本.txt" --discovery-concurrency 6 --max-requests 1000 --max-seconds 2400
python -X utf8 -m entitytrace run "D:\小说\样本.txt" --identity-algorithm groups-v2 --identity-rounds 12 --group-candidates 3
```

默认章节标注同时最多 3 个生成请求、发现与名册阶段同时最多 6 个生成请求（切换 `groups-v2` 时身份消歧同时最多 3 个 Decision 请求），发现任务最多约 9,000 个码点，均为**每个进程**的上限。多开进程会叠加并发；多 key 不代表账户额度相乘。模型调用、重试和等待均在 SQLite 事务外，任务结果与完成标记用短事务一起保存。

重复同一命令会复用完成阶段和任务；失败任务可重试。输入、业务源码、模型或核心流程参数（如 `--identity-algorithm`、`--task-chars`）变化会产生新运行编号，内容完全相同的成功 HTTP 请求仍可命中缓存。**预算参数（`--max-requests`、`--max-seconds`）不写入 `run_id` 配置摘要**：当运行因预算耗尽或连续网络传输失败（默认连续 9 次）停止时，扩大预算或待网络恢复后重跑同一命令，将在同一 `run_id` 上断点续跑，已完成任务不会重复发起网络请求。需要独立无缓存验证时，使用新的 `--db` 路径；不要把缓存复用计时当成首次全文处理速度。

服务明确返回余额或硬额度不足时，本次运行停止该服务的新请求，仍可读取成功缓存；未完成任务保留。额度恢复后可重新执行命令。服务错误、预算停止与断网不会转换为身份裁决。

`run` 主要参数（详见 `python -X utf8 -m entitytrace run --help`）：
- `--identity-algorithm roster|groups-v2`：身份归组算法，默认 `roster`（`entity-roster-v1`）；`groups-v2` 为旧版实体组消歧。
- `--max-requests N`、`--max-seconds S`：单次运行的新网络请求上限与运行秒数上限，默认不限；另内置连续 9 次传输层网络失败自动熔断停止。
- `--encoding`、`--task-chars`、`--discovery-concurrency`、`--decision-concurrency`：输入编码、发现任务长度预算与各服务单进程并发上限。
- `--stop-after chapters|discovery|identity`：在指定阶段完成后停止。
- `--json`：将进度日志全部写往 stderr，stdout 仅在结束时输出一行 `entitytrace.v1` JSON 回执（不加 `--json` 时 `run` 的 stdout 文本保持原样）。
- `--identity-rounds`（1—24，默认 12）、`--group-candidates`（1—3，默认 3）：**仅对 `--identity-algorithm groups-v2` 生效**。

退出码约定（适用于 `run`、只读查询命令与 `correct`；与旧版“所有失败或未完成一律返回 `2`”的行为不同，现按数据状态区分 `1` 与 `2`，是否自动重试以 JSON 回执的 `error.retryable` 为准）：
- `0`：计算或命令完成（含 `completed` 与带待审的 `completed_with_review`、查询成功、`correct` 成功）。
- `2`：数据库中留有**可续跑的断点**（包括网络中断 `network`、预算耗尽 `budget`、模型输出连续格式错误 `model_output`、额度耗尽 `quota`，以及 `--stop-after` 阶段断点）。注意：`quota` 退出码虽为 `2`（保留已完成缓存与断点），但 `error.retryable=false`，宿主不得自动重跑。
- `1`：**无可续跑的断点**，或命令被拒绝（包括输入文件缺失／解码失败 `input`、配置缺失或鉴权失败 `config`、未知 `run_id`、非法参数、`groups-v2` 运行不支持查询、结果版本过期 `conflict` 及本地校验异常 `local`，`retryable` 均为 `false`）。

## 输出与回溯

终端打印完整运行编号和 `outputs/<编号前16位>/` 目录。

| 文件 | 内容 |
| --- | --- |
| `result.json` | 来源、章节、提及、原始发现记录、实体名册分组及汇总信息；不含描述提取 |
| `review.json` | 身份证据不足、分歧未决与定位失败等待审问题，含稳定编号、影响范围（`impact`）及证据引用 |
| `decisions.jsonl` | 章节候选标注追溯记录（`chapter-label`，含采纳或无正文丢弃结果），以及 `groups-v2` 模式下 Decision 每次判断的问题、证据、概率与置信度 |
| `requests.jsonl` | 本运行使用的成功模型请求和响应，不含 Authorization 或模型内部推理字段 |
| `manifest.json` | 输入和源码版本、配置、阶段状态、每次调用预算与请求／缓存／失败统计、耗时和数量 |
| `source.txt`、`source.original` | 用于地址定位的冻结文本、原始输入字节 |

`result.json` 的 `status` 为 `completed`、`completed_with_review` 或 `incomplete`。计算流程完成不等于实体名单全部确认；只有部分结果时不会标成全文完成。`identity_status=unresolved_singleton` 表示暂定的单独提及，不能据此断言它和全书其他对象都不同。

默认 `roster`（`entity-roster-v1`）运行输出 **`schema_version=3`**：
- `entities[]`：包含 `id`、`name`、`names`、`aliases`（强称呼集合）、`mention_ids`、`mention_count`、`start`、`first_chapter_id`、`kind`、`identity_status`（`linked_by_roster`、`unresolved_singleton`、`not_entity_suggested`）和 `link_methods`（各链接方法计数）。**注意：名册条目 `id` 由成员首处地址与强称呼集合摘要生成，不是跨版本永久编号。**
- `mentions[]`：在章节与原文地址、定位证据基础上，增加 `entity_id`、`link_method`、`link_refs`（以及非实体建议项的 `identity_status=not_entity_suggested`）。
- 顶层附加 `roster_summary`（条目数、活跃条目数、单项数、各链接方法计数、`c3_rejected_name_mismatch`、身份待审数、歧义称呼数）与 `ambiguous_surfaces`（映射到多个名册条目的称呼及分布，仅作信息展示，不列为待审）。
- `review.json`：合并身份待审与发现／定位待审，按 `impact` 降序排列，并在 `counts` 中分列 `identity_reviews` 与 `grounding_discovery_reviews`。
- 使用 `--identity-algorithm groups-v2` 或重新导出历史旧运行时，`export` 仍按原 `schema_version=2`（或 `1`）导出，既有字段与结构保持不变。

清单 `manifest.json` 记录每次调用的各阶段耗时、`wall_seconds`（从管线入口、数据库初始化至主输出导出完成，不含最终回执回写与进程启停）、`export_seconds`、`identity_seconds` 及 `budget` 执行摘要。严格比较时另测整个命令的外部墙钟时间，并注明冷／热缓存口径。

地址为 `text_version_id + offset_unit + [start,end)`，从 0 开始、包含起点不含终点，单位是 Unicode 码点。原文换行和空格保留；UTF-8 BOM 仅从定位文本中去除，原始文件另存。章内偏移从整个章节范围起点计算，章节包含标题；`ordinal` 是含篇首／卷等分段的顺序，不等于作品印刷章号。

从数据库按需只读查询、取回原文或重新导出，不调用模型（将 `运行编号` 替换为完整值；只读查询默认输出单行 `entitytrace.v1` JSON，实时通过内存 `build_export_bundle` 构建并带最新 `result_revision`，仅支持 `entity-roster-v1` 运行）：

```powershell
python -X utf8 -m entitytrace status 运行编号
python -X utf8 -m entitytrace entities 运行编号 --kind person --min-mentions 5 --limit 50 --offset 0
python -X utf8 -m entitytrace entities 运行编号 --surface "称呼"
python -X utf8 -m entitytrace entity 运行编号 实体ID --limit 50 --offset 0 --context 80
python -X utf8 -m entitytrace review 运行编号 --status pending --limit 50 --offset 0
python -X utf8 -m entitytrace quote 运行编号 100 120 --json --context 80
python -X utf8 -m entitytrace quote 运行编号 100 120
python -X utf8 -m entitytrace export 运行编号
```

分页查询（`entities`、`entity`、`review`）的 `--limit` 默认 `50`、上限 `200`（超限按 `input` 拒绝，退出码 `1`），响应含 `total` 与 `next_offset`（无下一页时为 `null`）；`entity` 的 `snippet` 上下文上限 `400` 码点且不跨章节边界；`quote` 不加 `--json` 时纯文本输出保持原样。

对已完成的默认 `entity-roster-v1` 运行，可通过 `correct` 叠加确定性更正或撤回（零模型调用，`groups-v2` 与未完成运行不支持）：

```powershell
python -X utf8 -m entitytrace correct 运行编号 merge 目标实体ID 被并实体ID --source user --expect-revision 当前result_revision
python -X utf8 -m entitytrace correct 运行编号 split-surface 实体ID "称呼" --source user --expect-revision 当前result_revision
python -X utf8 -m entitytrace correct 运行编号 assign 提及ID 实体ID或new --source user --expect-revision 当前result_revision --resolves 待审ID
python -X utf8 -m entitytrace correct 运行编号 not-entity 提及ID --source user --expect-revision 当前result_revision
python -X utf8 -m entitytrace correct 运行编号 revoke 更正ID --cascade --source user --expect-revision 当前result_revision
```

`correct` 生效规则：
- `--expect-revision` 必须与当前 `result.json` 的 `result_revision` 一致，否则以 `conflict` 错误（退出码 `1`）拒绝写入；`--resolves` 须指向当前 `review.json` 中存在的待审项，生效后标为 `resolved` 并记录 `resolved_by`。
- **只有 `--source user` 且未撤回的更正参与重放并改变 `result.json`**；`--source agent` 仅作为建议写入 `review.json` 的 `agent_suggestions`，不改变 `result.json` 与 `result_revision`。`user` 可撤回任意更正，`agent` 仅可撤回 `agent` 建议；有后续更正依赖目标更正时须显式加 `--cascade` 一并撤回。
- 同一生效操作重复提交幂等返回 `status="duplicate"`；`split-surface` 后原实体剩余的 `c3_model` 提及会生成 `correction-c3-affected` 待审项提示复核而不自动搬移；零生效用户更正或全部撤回后，`result.json`、`review.json`（无未撤回建议时）与 `manifest.json` 逐字节复原。
- 成功时 stdout 输出单行 `entitytrace.v1` JSON：`{"protocol":"entitytrace.v1","command":"correct","run_id","error":null,"correction_id","status":"applied|suggested|duplicate|revoked","result_revision_before","result_revision","affected_entity_ids","output_dir"}`；失败时 stdout 输出单行 JSON 错误（含 `error.category` / `message` / `retryable`）并以退出码 `1` 退出，数据库零写入。

## 验证与当前限制

```powershell
python -X utf8 -m unittest discover -s tests -v
```

离线验证覆盖原文字节／换行／emoji 地址、重复提及、错误引用、身份冲突、请求大小与拆包、事务边界、并发补位、竞争候选、类别纠错、组快照恢复及缓存证据隔离。

后续同时比较耗时与准确性，包括纠错 loop 的累计时间、自动纠正及新错误。较快首轮与较慢高质量方案按最终质量及总时间比较。

已有[32 对身份诊断样本](test/identity-calibration/README.md)，含原文地址、待独立人工复核的标签和实验工具。额度恢复后，两次单问题在线实验的原始选择符合标签从短证据 28/32、29/32 提升至长证据均 31/32；完整单候选复核下，0.95／0.90 门槛自动处理 3／12 对，错误均 0、待审 29／20，耗时接近。生产门槛未改；样本小且非独立留出，不能外推全书。

候选召回暂未使用语义向量，可能漏掉没有字面联系的化名；发现只覆盖模型实际返回并成功定位的提及，处理了全文并不保证找全实体。章节样本已经过资料整理，复杂排版、目录混排及多卷小说需要另外评测。实体／身份没有完整人工标准答案，不能用地址校验代替语义准确率。

## 文档与测试资料

- [测试文本库](test/text-library/README.md)：六份中英文完整小说；正文、独立章节参考及来源许可分别保存，参考答案不进入模型输入。
- [GitHub 仓库](https://github.com/Linzeyu912/EntityTrace)。
