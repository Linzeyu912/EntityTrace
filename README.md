# EntityTrace · 实体溯源

EntityTrace 是面向 agent 的插件，提供小说实体发现、身份判断与归组，以及原文证据回溯能力。项目定位分为工具形态和功能定位两层。

## 工具形态

**面向 agent 调用的插件（plugin）**，回答“以什么形式被使用”。目标调用方包括各类宿主 agent；宿主负责理解用户需求、组织调用、解释结果及衔接后续工作。由此确定开发边界：不建设独立前端、实体编辑页面或人工操作工作台，需要人参与的确认由宿主在对话中承接。

当前交付形态是可供 agent 执行的本地 CLI 与 JSON 文件，使用 Python 3.12+ 和标准库 SQLite，无额外运行依赖，不需要 Docker 或数据库服务。**尚未提供特定平台的专用安装包、宿主适配或接入验证。** 插件定位已确定，具体分发与协议封装待落实。默认流程调用在线 DeepSeek API，也支持通过 `--generator openai`（或配置 `GENERATOR_PROVIDER=openai`）接入通用 OpenAI 兼容服务地址；同时支持通过可选的 TypeSafe/Jev Decision 封闭集服务（`--decision auto|on|off` / `ENTITYTRACE_DECISION_MODE`）在有凭证时加速名册复核与描述复核、减少生成调用次数（无 `TYPESAFE_API_KEY` 时默认 `auto` 模式静默走纯生成链路；显式切换 `--identity-algorithm groups-v2` 时则固定需要 TypeSafe/Jev）；宿主 agent 的模型与插件内部模型分工独立，本机运行不等于模型离线运行。

## 功能定位

**实体发现、身份判断与归组、原文证据回溯**，回答“具体解决什么问题”。

- **实体发现**：从小说原文发现具体对象的提及，保留称呼与出现位置。
- **身份判断与归组**：依据原文判断哪些提及属于同一个对象，保留不同身份约束和未决项。
- **原文证据回溯**：通过固定文本版本与地址找回原文，追查判断依据和归组过程。
- **优先级与分阶段描述抽取**：按全书或指定正文章节范围计算实体优先级（`rank`），并对头部实体抽取带逐字原文证据的描述事实、状态变化事件、非空阶段切分（`stages`）与维度空缺清单（`describe`）。

插件形态决定交付与调用方式，功能定位决定处理能力与结果：接入不同宿主可复用同一核心能力，功能扩展按独立需求和质量验证推进。

## Agent 工作流与交付边界

目标工作流是：用户向宿主 agent 提出任务 → agent 调用 EntityTrace → 读取运行状态、结果和待审项 → 按需回取原文、补充确认或抽取分阶段描述 → 将带证据的结果交给后续工具。插件的重点是输入输出清楚、状态可判断、失败可恢复、证据可追溯，以及更正后能继续流转。

| 环节 | 当前可用 | 后续补齐 |
| --- | --- | --- |
| 启动与恢复 | `run` 执行全文处理（`--json` 输出单行 `entitytrace.v1` 回执），相同输入与配置复用断点；支持 `--max-requests`、`--max-seconds` 预算与连续断网停止 | 后台异步任务提交、轮询与取消接口 |
| 读取与取证 | `status`、`entities`（含 `--surface`）、`entity`、`review`、`rank`、`quote` 按需只读查询（实时重放有效更正并带 `result_revision`），以及 `export` 重导出完整产物 | 跨进程并发读写同库的锁隔离 |
| 待审与更正 | `review.json` 按 `impact` 降序提供问题编号与证据；已完成名册运行支持 `correct` 叠加式更正、待审消解（`--resolves`）、`--expect-revision` 版本校验与撤回（零模型调用） | 章节识别阻断项的人工交互修正入口 |
| 分阶段描述 | `describe` 按全书或章节范围 top-K 实体执行多遍轮换抽取、过半表决、持有者引文补全、事实与事件逐条复核及确定性非空阶段切分（只给原文事实与空缺清单，不推断补全） | 扩展组织标志与场景环境音等附加维度 |
| 宿主接入 | 有命令执行和文件访问能力的 agent 可按本文与 [宿主调用说明](plugin/README.md) 调用 CLI | 按需核实特定宿主平台协议封装 |

不加 `--json` 时 `run` 的终端输出包含进度文本；宿主编程调用应传入 `--json`（进度写往 stderr，stdout 仅输出一行 `entitytrace.v1` JSON 回执），并结合退出码与 `error.retryable` 决策。计算完成不代表全部身份已经确认，宿主 agent 的推断也不能记作用户已确认的答案。

## 当前流程

默认身份流程为 **`entity-roster-v1`（实体名册链接，`--identity-algorithm roster`）**；旧版配对组消歧流程保留为 `--identity-algorithm groups-v2` 供对照与兼容导出。

| 步骤 | 执行者 | 实际行为 |
| --- | --- | --- |
| 导入 TXT | 本地程序 | 保存原始字节、冻结文本及指纹，建立 Unicode 码点地址 |
| 章节识别 | 本地程序候选筛选 + DeepSeek 按 ID 标注 + 本地结构核验 | 提取整行短候选并经非标题句法确定性预筛后并发标注（`temperature=0`），再经“标题后须有正文（含无正文卷／部标题改判）”与“章号连续（含按原文原始候选补齐唯一未标注章题缺口）”两条确定性结构核验确认 |
| 地址核验 | 本地程序 | 检查章节范围完整覆盖原文，再建立发现任务 |
| 提及发现 | DeepSeek + 本地并发任务池 | 各章独立发现称呼与引用，长章拆任务；逐字核对提及地址 |
| 章内归并（C1） | DeepSeek + 本地结构与边界校验 | 按发现片段整段上下文将称呼归入章内对象，区分专名／称谓／描述词，支持逐提及例外指派与非实体建议 |
| 跨章名册与复核（C2） | 本地并查集 + DeepSeek 分区复核／补合并核验 | 仅 `proper_name` 参与跨章候选聚合；多名称、粗类冲突或弱专名送分区复核；按粗类与跨粗类提议补合并，启用 Decision 时先对提议每批最多 5 对执行高置信复核（门槛 0.50），否则经生成模型双侧 ≥600 码点原文核验确认，不可合并约束全程硬拦截 |
| 未决与非专名链接（C3） | 本地唯一称呼直连 + DeepSeek/Decision 候选链接 | 章内对象确定性落入名册；只含非专名称呼的对象与章内未决提及在候选名册与原文证据下链接（启用 Decision 时每子批最多 15 项做封闭集预选，最高概率 ≥0.90 直接采纳，其余缩编候选池后回退生成模型），受专名一致性硬约束保护；单任务若连续返回非法格式则自动将该任务待链项转入待审而不阻断全书完成 |
| 待审排序与统一导出（C4） | 本地程序 | 同任务无歧义 C1 专名补全（`same_task_proper`）并自动消解已完整覆盖的同任务定位待审（`auto_same_task_covered`）；汇总身份与定位待审并按受影响提及数（`impact`）降序排列，校验全部码点地址与覆盖完整性，导出 `schema_version=3` 产物 |

默认生成服务商为 `deepseek`（模型默认 `deepseek-flash`，并发默认 6、单请求超时 120 秒；章节标注、提及发现与名册身份全部阶段默认固定 `temperature=0`，若单次尝试因模型输出截断或格式校验失败则下一次重试自动恢复服务端默认温度）。Decision 加速开关支持 `--decision {auto,on,off}`（或 `ENTITYTRACE_DECISION_MODE`，默认 `auto`）：在默认 `roster` 与 `describe` 流程中，若配置了 `TYPESAFE_API_KEY`（模型默认 `jev-latest`），`auto` 与 `on` 会启用 TypeSafe 封闭集判断加速复核与链接；未配置 `TYPESAFE_API_KEY` 时，`auto` 自动静默退回纯生成链路（且不写入 Decision 配置，保持 `run_id` 与 `describe_id` 与 `--decision off` 逐字节一致），`on` 则直接以配置错误拒绝运行，`off` 显式关闭。运行时若 `auto` 模式遇到非预算类的 Decision 异常（如服务出错或输出异常），会在当前运行内立即软熔断回退至生成模型并维持原有连续失败计数，而预算上限错误仍正常传播；切换 `--identity-algorithm groups-v2` 时固定加载 `TYPESAFE_API_KEY` 与版本 `jev-1.13.0` 执行配对身份消歧。也可传 `--generator openai`（或设 `GENERATOR_PROVIDER=openai`）接入通用 OpenAI 兼容地址，通过 `ENTITYTRACE_OPENAI_BASE_URL`、`ENTITYTRACE_OPENAI_MODEL`、`ENTITYTRACE_OPENAI_API_KEY`、`ENTITYTRACE_OPENAI_EXTRA_BODY`（可选 JSON 对象，顶层合并至请求载荷，`null` 值删除对应键）与 `ENTITYTRACE_OPENAI_TIMEOUT_SECONDS`（默认 420 秒，并发默认 8）配置。`entity-roster-v1` 不使用未校准概率分数作主路由，而是显式记录每处提及的链接来源（`chapter_local`、`c3_model`、`surface_unique`、`same_task_proper`、`unresolved`、`c3_rejected_name_mismatch`、`not_entity_suggested`）与关联请求 ID。

## 运行

以下命令既是宿主 agent 当前的调用入口，也可用于开发调试。以仓库目录为工作目录执行，或显式指定输入、配置、数据库和输出路径。默认 `roster` 与 `describe` 流程在 `deepseek` 服务商下最少只需配置 `DEEPSEEK_API_KEY`（未配置 `TYPESAFE_API_KEY` 时 `--decision auto` 静默走纯生成且与 `--decision off` 共用同一 `run_id`/`describe_id`）；若同时配置 `TYPESAFE_API_KEY` 则 `auto` 自动启用封闭集提速（可用 `--decision off` 关闭或 `--decision on` 强制校验）；切换 `--generator openai` 时配置 `ENTITYTRACE_OPENAI_BASE_URL`、`ENTITYTRACE_OPENAI_MODEL` 与 `ENTITYTRACE_OPENAI_API_KEY`；仅 `--identity-algorithm groups-v2` 固定要求配置 `TYPESAFE_API_KEY`。新环境可从 `.env.example` 创建配置文件并填写自己的密钥；系统环境变量优先于文件。配置、数据库和运行产物均被 Git 忽略。

```powershell
python -X utf8 -m entitytrace run "test/text-library/texts/03_中文长篇_九命奇冤_36回.txt"
python -X utf8 -m entitytrace run "test/text-library/texts/04_英文长篇_Pride_and_Prejudice_61章.txt"
```

也可处理自己的 UTF-8 TXT，或按需附加请求／时间预算与身份算法开关：

```powershell
python -X utf8 -m entitytrace run "D:\小说\样本.txt" --discovery-concurrency 6 --max-requests 1000 --max-seconds 2400
python -X utf8 -m entitytrace run "D:\小说\样本.txt" --identity-algorithm groups-v2 --identity-rounds 12 --group-candidates 3
```

默认章节标注同时最多 3 个生成请求、发现与名册阶段同时最多 6 个生成请求（启用 TypeSafe Decision 时，常规阶段同时最多 3 个 Decision 请求、默认 `roster` 阶段同时最多 8 个 Decision 请求，切换 `groups-v2` 时同时最多 3 个 Decision 请求），发现任务最多约 9,000 个码点，均为**每个进程**的上限。多开进程会叠加并发；多 key 不代表账户额度相乘。模型调用、重试和等待均在 SQLite 事务外，任务结果与完成标记用短事务一起保存。

重复同一命令会复用完成阶段和任务；失败任务可重试。输入、业务源码、模型或核心流程参数（如 `--identity-algorithm`、`--task-chars`）变化会产生新运行编号，内容完全相同的成功 HTTP 请求仍可命中缓存。**预算参数（`--max-requests`、`--max-seconds`）不写入 `run_id` 配置摘要**：当运行因预算耗尽或连续网络传输失败（默认连续 9 次）停止时，扩大预算或待网络恢复后重跑同一命令，将在同一 `run_id` 上断点续跑，已完成任务不会重复发起网络请求。需要独立无缓存验证时，使用新的 `--db` 路径；不要把缓存复用计时当成首次全文处理速度。

服务明确返回 429、额度限制或余额不足（`quota`）时，本次运行停止该服务的新请求，仍可读取成功缓存；数据库保留未完成断点。若为 429 或额度窗口限制，稍后用同一命令同库续跑；若为 DeepSeek 余额不足，充值后同库续跑。服务错误、预算停止与断网不会转换为身份裁决。

`run` 主要参数（详见 `python -X utf8 -m entitytrace run --help`）：
- `--generator deepseek|openai`：生成服务商，默认读取 `GENERATOR_PROVIDER`（缺省 `deepseek`）。
- `--identity-algorithm roster|groups-v2`：身份归组算法，默认 `roster`（`entity-roster-v1`）；`groups-v2` 为旧版实体组消歧。
- `--max-requests N`、`--max-seconds S`：单次运行的新网络请求上限与运行秒数上限，默认不限（宿主需要设置单次预算时，可按实测约每万字 10—17 次请求、14—18 秒估算）；另内置连续 9 次传输层网络失败自动熔断停止。
- `--encoding`、`--task-chars`、`--discovery-concurrency`、`--decision-concurrency`：输入编码、发现任务长度预算与各生成／Decision 服务单进程并发上限。
- `--decision auto|on|off`：TypeSafe Decision 封闭集提速开关，默认读取 `ENTITYTRACE_DECISION_MODE`（缺省 `auto`：有 `TYPESAFE_API_KEY` 自动启用并在非预算异常时软熔断回退生成链路，无 key 静默纯生成）；`on` 强制要求凭证，`off` 显式关闭。
- `--stop-after chapters|discovery|identity`：在指定阶段完成后停止。
- `--json`：将进度日志全部写往 stderr，stdout 仅在结束时输出一行 `entitytrace.v1` JSON 回执（不加 `--json` 时 `run` 的 stdout 文本保持原样）。
- `--identity-rounds`（1—24，默认 12）、`--group-candidates`（1—3，默认 3）：**仅对 `--identity-algorithm groups-v2` 生效**。

退出码约定（适用于 `run`、只读查询命令与 `correct`；与旧版“所有失败或未完成一律返回 `2`”的行为不同，现按数据状态区分 `1` 与 `2`，是否自动重试以 JSON 回执的 `error.retryable` 为准）：
- `0`：计算或命令完成（含 `completed` 与带待审的 `completed_with_review`、查询成功、`correct` 成功）。
- `2`：数据库中留有**可续跑的断点**（包括网络中断 `network`、预算耗尽 `budget`、阶段内模型输出连续格式错误 `model_output`（注：C3 链接任务若连续返回非法输出则自动将该任务待链项转待审，不阻断阶段完成）、额度耗尽 `quota`，以及 `--stop-after` 阶段断点）。注意：`quota` 退出码虽为 `2`（保留已完成缓存与断点），但 `error.retryable=false`，宿主不得自动重跑。
- `1`：**无可续跑的断点**，或命令被拒绝（包括输入文件缺失／解码失败或章节结构阻断 `input`、配置缺失或鉴权失败 `config`、未知 `run_id`、非法参数、`groups-v2` 运行不支持查询、结果版本过期 `conflict` 及本地校验异常 `local`，`retryable` 均为 `false`）。章节阶段先对跳号缺口按原文唯一未标注章题候选确定性补齐（`result="gap_filled"`，并对后接章／附章的无正文卷／部标题改判为 `volume`、`result="adopted_as_volume"`），再按原文证据分级处理章号异常：原文跳号（区间内找不到缺失章号）或回退后按新章题连续递增记为非阻断提示（`blocking=false`，`chapters` 仍标为 `complete=true`）；无法唯一补齐且区间内存在缺失章号的疑似标题或重复章节记为阻断待审（`blocking=true`），回执报 `error.category="input"`、`retryable=false`（退出码 `1`，重跑命中缓存结果相同，不可自动重试，可通过 `review --stage chapters` 核查）。

## 输出与回溯

终端打印完整运行编号和 `outputs/<编号前16位>/` 目录。

| 文件 | 内容 |
| --- | --- |
| `result.json` | 来源、章节、提及、原始发现记录、实体名册分组及汇总信息；不含描述提取 |
| `review.json` | 身份证据不足、分歧未决与定位失败等待审问题，含稳定编号、影响范围（`impact`）及证据引用 |
| `decisions.jsonl` | 章节候选标注追溯记录（`chapter-label`，含采纳、缺口补齐、卷题改判或无正文丢弃结果），以及 `groups-v2` 模式下 Decision 每次判断的问题、证据、概率与置信度 |
| `requests.jsonl` | 本运行使用的成功模型请求和响应，不含 Authorization 或模型内部推理字段 |
| `manifest.json` | 输入和源码版本、配置、阶段状态、每次调用预算与请求／缓存／失败统计、耗时和数量 |
| `source.txt`、`source.original` | 用于地址定位的冻结文本、原始输入字节 |

`result.json` 的 `status` 为 `completed`、`completed_with_review` 或 `incomplete`。计算流程完成不等于实体名单全部确认；只有部分结果时不会标成全文完成。`identity_status=unresolved_singleton` 表示暂定的单独提及，不能据此断言它和全书其他对象都不同。

默认 `roster`（`entity-roster-v1`）运行输出 **`schema_version=3`**：
- `entities[]`：包含 `id`、`name`、`names`、`aliases`（强称呼集合）、`mention_ids`、`mention_count`、`start`、`first_chapter_id`、`kind`、`identity_status`（`linked_by_roster`、`unresolved_singleton`、`not_entity_suggested`）和 `link_methods`（各链接方法计数）。**注意：名册条目 `id` 由成员首处地址与强称呼集合摘要生成，不是跨版本永久编号。**
- `mentions[]`：在章节与原文地址、定位证据基础上，增加 `entity_id`、`link_method`、`link_refs`（以及非实体建议项的 `identity_status=not_entity_suggested`）。
- 顶层附加 `roster_summary`（条目数、活跃条目数、单项数、各链接方法计数、`c3_rejected_name_mismatch`、身份待审数、歧义称呼数）与 `ambiguous_surfaces`（映射到多个名册条目的称呼及分布，仅作信息展示，不列为待审）。
- `review.json`：合并身份待审与发现／定位待审，按 `impact` 降序排列，并在 `counts` 中分列 `identity_reviews` 与 `grounding_discovery_reviews`（含自动消解或用户更正消解项时另附 `pending` 与 `resolved`）。
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
python -X utf8 -m entitytrace rank 运行编号 --kind person --from-chapter 1 --to-chapter 10 --limit 50 --offset 0
python -X utf8 -m entitytrace quote 运行编号 100 120 --json --context 80
python -X utf8 -m entitytrace quote 运行编号 100 120
python -X utf8 -m entitytrace export 运行编号
```

分页查询（`entities`、`entity`、`review`、`rank`）的 `--limit` 默认 `50`、上限 `200`（超限按 `input` 拒绝，退出码 `1`），响应含 `total` 与 `next_offset`（无下一页时为 `null`）；`rank` 按 `sqrt(范围内提及数 × 范围内章节数)` 对范围内至少出现一次的实体排序，`--from-chapter`／`--to-chapter` 为正文章序（只计 `kind=="chapter"` 条目，两端含，缺省全书含卷题区间提及），`--kind` 限定粗类，每项返回 `kind_rank`、`score`、`mention_count_in_range`、`chapter_count_in_range`、`first_chapter_id`、`last_chapter_id`、`span_chapters`；`entity` 的 `snippet` 上下文上限 `400` 码点且不跨章节边界；`quote` 不加 `--json` 时纯文本输出保持原样。

### 描述事实抽取（`describe`）

在已完成的 `entity-roster-v1` 运行上，按 top-K 实体抽取原文描述事实与状态变化事件（只给原文事实与空缺清单，不推断、不补全）。只读上游原文与查询快照，不改动已有名册、提及、待审与导出：

```powershell
python -X utf8 -m entitytrace describe 运行编号 --db 数据库路径 --top-k 20 --kinds person,place,object,ability --concurrency 8 --out 描述导出.json
python -X utf8 -m entitytrace describe 运行编号 --db 数据库路径 --from-chapter 1 --to-chapter 15 --out 描述导出.json
python -X utf8 -m entitytrace describe 运行编号 --db 数据库路径 --views 1 --entities-per-request 0 --verify 0 --out 描述导出.json
```

- 按发现任务块逐块抽取，跳过不含 top-K 实体的块。默认启用 `--views 3 --entities-per-request 4 --verify 1`（`--top-k` 默认 `20`）：以《封神演义》百回四类 top-20 实测为准，默认三遍且每请求最多 `4` 个实体的抽取任务数为 `1,545` 次，约为 `--views 1 --entities-per-request 0` 单遍整块抽取（`114` 次）的 `13.6` 倍（单遍按 `4` 实体分组为 `515` 次，三遍为其 `3` 倍）；事实复核每 `12` 条表决后事实约 `1` 个请求（单遍实测 `84`—`89` 次，`--verify 3` 三遍实测 `269`—`280` 次），状态变化事件复核每 `12` 条表决后事件约 `1` 个请求（单遍约 `10` 次，三遍约 `30` 次）。显式传入 `--views 1 --entities-per-request 0 --verify 0` 可回到最省请求的单遍模式。
- `--from-chapter N` / `--to-chapter M`（同义别名 `--start-chapter` / `--end-chapter`）：按正文章序（只计 `kind=="chapter"` 条目，从 `1` 起、两端含；只给一端时另一端补全为首／末章，都不传则为全书）限定描述抽取范围。未传 `--topk-file` 时按该章节范围内的 `rank` 选出 top-K 实体；传了 `--topk-file` 时沿用指定名单并只扫描所选正文章节分块，实体称呼名册仍保留全书本名。越界或 `N > M` 按 `input` 错误拒绝（退出码 `1`）。
- `--views N`（`1`—`5`，默认 `3`）：多遍轮换分组请求并按跨遍地址重叠做过半表决。第 `v` 遍把块内实体按优先级排序后循环左移 `v − 1` 位再分组，输入带 `view` 字段，各遍负载因此不同；导出只给支持数达到 `⌊N/2⌋ + 1` 的条目并附 `support`。原始各遍结果照常落库，`--views` 以约 `N` 倍请求数换取更高精确率。
- `--entities-per-request N` 默认 `4`（传 `0` 表示整块一组）；`N > 0` 时块内实体按优先级（`score` 降序、`entity_id` 升序）排序，每 `N` 个一组单独请求，块正文相同、任务编号含遍号与组序号。
- `--verify {0,1,3}` 默认 `1`：`1` 在全部抽取任务完成后先做过半表决，再把表决后的描述事实与状态变化事件分别按固定键排序、每 `12` 条一批逐条复核（keep / drop），只导出判为 keep 的事实与事件，同时对保留事件判定 `stage_split`（是否造成跨场景或跨章节持久不同的外貌衣饰、肉身形态、建筑存毁或器物归属）；`3` 把同一批复核跑三遍并按票数表决（`drop` 票至少两票判 `drop`，事实原因按 `not_description → wrong_entity → wrong_facet → unsupported_value` 破平，事件原因按 `not_state_event → wrong_entity → wrong_event_type → unsupported_value` 破平；有效票至少两票判 `keep`，事件 `stage_split` 在 `3 keep` 时取至少两票、在 `2 keep + 1 drop` 时要求两张 `keep` 都为 `true`；不足两票判 `unverified`）。三遍的第一批分别为 `12`、`8`、`4` 条，其后每批 `12` 条，第 0 遍的批次与负载与 `1` 完全相同。复核要求模型逐字回抄输入的 `value`，不一致会重试并在 `verification.echo_mismatch_retries` / `event_verification.echo_mismatch_retries` 计数。复核批次按任务编号续跑，已完成批次不重发，网络或额度停止仍保留断点并以退出码 `2` 结束。开启 `--decision auto|on` 且配置 `TYPESAFE_API_KEY` 时，事实复核每批最多 `8` 条、事件复核每批最多 `6` 条走封闭集混合筛（最高概率 ≥0.80 直接判票，低于 0.80 仅将低置信项回退原生成模型，负载使用短编号不泄露内部 64 位 ID），均强制执行 `disguise` 事件禁止切分及无有效锚点空侧时间线整数守卫。
- `--decision auto|on|off`、`--decision-concurrency N` 同样作用于 `describe`（或通过 `ENTITYTRACE_DECISION_MODE` 设置）：未配置 `TYPESAFE_API_KEY` 时 `auto` 与 `off` 均不写入 Decision 配置，保持 `describe_id` 一致。
- 相同上游运行、top-K 名单、章节范围、`--views`、`--entities-per-request`、`--verify` 与实际生效的模型／Decision 配置得到相同 `describe_id`，自动续跑已完成任务；更换其中任一项或生效配置会得到新编号。需要独立冷跑时使用新的 `--db` 副本。
- 成功时 stdout 输出单行 JSON：`{"ok":true,"describe_id":...,"run_id":...,"views":...,"entities_per_request":...,"verify":...,"chapter_range":{...},"counts":{"tasks","facts","events","stages","multi_stage_entities","reviews"},"out":...}`。`--out` 导出 `{describe_id, run_id, views, entities_per_request, top_k, kinds, chapter_range, entities, reviews, verification, event_verification, stats}`；每个实体含 `facts`、`state_events`、按确定性非空阶段规则切分的 `stages`（每项含 `stage_index`、`start`、`end`、`from_chapter`、`to_chapter`、`first_fact_chapter_id`、`last_fact_chapter_id`、`trigger_events`、`facts`、`gaps`；有事实实体绝不产生零事实空壳阶段）与实体整体 `gaps`，`stats` 含请求数、token 与墙钟。
- facet 枚举：person 为 `age/gender/build/face/hair/feature/costume/voice`，place 为 `terrain/architecture/scale/atmosphere`，object 为 `shape/size/color_material/visual_effect/holder`，ability 为 `visual_effect/sound_effect`；每个实体输入都带 `allowed_facets`（按类别取自该枚举）。
- 事件类型为九类 `costume_change/injury/disguise/transform/age_jump/upgrade/damage/owner_change/death`；模型输出其它事件类型（含 `other`）会按 `bad-enum` 转入描述待审，不进入事件导出。
- 模型输出逐条核验：`quote` 必须在请求输入原文内逐字出现且不超过 `120` 码点，多次出现时取距该实体提及最近的一处；未知实体、facet 与类别不符（含已删除的 `demeanor`）、枚举非法、引用找不到或超长会写入该次运行的描述待审（`describe_reviews`），不中断整轮。`holder` 事实的引文不含称呼时，会在同一片段内向左扩展到最近的持有者称呼；候选称呼取事实 `value`、`holder_entity_id` 的全书称呼（不要求该实体出现在本请求中）以及 `value` 中可解析出的人物实体的全书称呼，后者按最长匹配解析、并列属不同实体时不解析；扩展不到则按 `holder-not-in-quote` 转入描述待审。401/402/429 等硬额度与预算停止保留断点，按原命令同库续跑。

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

- [宿主 Agent 调用说明](plugin/README.md)：提供通用 CLI 调用说明（`entitytrace.v1` 回执、按需查询、退出码决策表与更正硬性约束），平台封装待核实。
- [测试文本库](test/text-library/README.md)：六份中英文完整小说；正文、独立章节参考及来源许可分别保存，参考答案不进入模型输入。
- [GitHub 仓库](https://github.com/Linzeyu912/EntityTrace)。
