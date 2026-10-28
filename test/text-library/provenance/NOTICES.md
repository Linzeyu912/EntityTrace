# 来源、署名与迁移范围

迁移日期：2026-10-28。用户明确要求将旧仓库最近整理的中英文完整小说测试文本转入 EntityTrace。

来源仓库为 [YingHe-entity](https://github.com/Linzeyu912/YingHe-entity)，来源提交为 `1512f7b32ee2c40aa7bb5a833af266f84be1e07d`。本次复制测试资料，不转移或改写旧 Git 提交，不复制旧业务工程、数据库、模型输出、环境配置或凭证。新迁移提交不改变作品原作者和来源归属。

## 资料映射

| 来源路径（相对于来源仓库） | 本库路径（相对于测试文本库） | 处理 |
| --- | --- | --- |
| `test/text-library/texts/` 下清单指定的六份 TXT | `texts/` 中同名文件 | 原样复制，SHA-256 与原清单核对 |
| `test/text-library/manifest.json` | `reference/source-manifest.json` | 原样复制作为章节参考及历史元数据 |
| `test/text-library/metadata/正文复核报告.json` | `provenance/metadata/正文复核报告.json` | 保留原四篇的历史复核报告 |
| `test/text-library/metadata/2026-10-21-百万字长篇复核报告.json` | `provenance/metadata/2026-10-21-百万字长篇复核报告.json` | 保留后两篇的历史复核报告 |
| `novel-samples/2026-10-20/来源与统计说明.txt` | `provenance/2026-10-20/来源与统计说明.txt` | 原样复制 |
| `novel-samples/2026-10-21/来源与统计说明.txt` | `provenance/2026-10-21/来源与统计说明.txt` | 原样复制 |
| `LICENSE` | `provenance/YingHe-entity-LICENSE.txt` | 保留来源仓库完整许可 |
| 两批 `sources/` 中选定的六份下载文件 | `provenance/source-downloads/日期/` | 原样复制，保留来源站点的署名、许可及制作人员信息 |

新建的 `manifest.json` 从原清单整理六份输入的必要元数据，把逐章参考移到独立文件；没有修改小说正文。新增本说明、使用说明和 Git 字节保留约定。

## 小说原作者与站点资料

六部作品的原作者、来源链接和整理操作见新清单与原清单；迁移者不替代作品署名。原作者依次为鲁迅、Arthur Conan Doyle、吴趼人、Jane Austen、施耐庵、Herman Melville。

本地来源下载包括：

- `source-downloads/2026-10-20/shangshi-luxun.html`：《伤逝》原来源页。
- `source-downloads/2026-10-20/jiuming-index.html`：《九命奇冤》目录来源页。
- `source-downloads/2026-10-20/holmes.txt`：Project Gutenberg《The Return of Sherlock Holmes》完整下载；模型输入只取其中完整独立短篇 The Adventure of the Second Stain。
- `source-downloads/2026-10-20/pride.txt`：Project Gutenberg《Pride and Prejudice》完整下载。
- `source-downloads/2026-10-21/shuihu-引首.html`：《水浒传》引首来源页。
- `source-downloads/2026-10-21/moby_dick.txt`：Project Gutenberg《Moby-Dick》完整下载。

上述 HTML 与电子书保留原来源中的声明。整理资料沿用所附 [YingHe-entity 原许可](YingHe-entity-LICENSE.txt)，不据此重新授权小说作品或替换来源站点的原条款；新项目业务代码的许可不在本次迁移中决定。

全部原始下载页与整理脚本仍可在来源提交的 [2026-10-20 归档](https://github.com/Linzeyu912/YingHe-entity/tree/1512f7b32ee2c40aa7bb5a833af266f84be1e07d/novel-samples/2026-10-20) 和 [2026-10-21 归档](https://github.com/Linzeyu912/YingHe-entity/tree/1512f7b32ee2c40aa7bb5a833af266f84be1e07d/novel-samples/2026-10-21) 查到。旧来源说明中的 `sources/`、脚本和相对路径保持原文，按原仓库对应日期目录理解；它们不是新库可直接运行的命令。
