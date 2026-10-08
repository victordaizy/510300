# 510300 研究资料库

[![仓库完整性检查](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml)

本版上传已盘点的完整510300关联研究资料：**363,126个原文件、51.308 GiB原始材料**。研究代码、文档及主要结果保留在Git中；完整数据、详细结果、历史版本与研究审阅包通过Release附件保存。文件路径、字节数、SHA-256及附件位置均见[累计索引](catalog/README.md)。

[项目状态](docs/PROJECT_STATE.md) · [研究决策](docs/RESEARCH_DECISIONS.md) · [日周线状态](docs/PROJECT_STATE_TECHNICAL_LINE.md) · [日周线决策](docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md) · [完整资料更新](docs/UPDATE_20261007_FULL.md)

## 完整数据与代码

本轮在10月6日的代码、文档和主要结果版本上补齐125,123个原文件，新增113个附件、20.308 GiB压缩或分片数据，复用63个既有附件。范围包括行情、量价点位、宏观与全因素、成分及财报来源、训练与账户明细、解析失败及研究拒绝记录、历史原始包。没有复制整套来源目录：直接从原件逐批生成、上传、核对、清理。本轮临时附件硬上限512 MiB，全部临时目录与状态文件上限1 GiB。

克隆得到Git文件和完整索引；GitHub的Code ZIP只包含Git文件。完整原始数据在[本版Release](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-10-07-full)及索引所引用的历史Release中，可用[还原工具](tools/repository/snapshot.py)恢复原始相对路径，并验证每个源文件的SHA-256。建议先按研究目录或数据前缀还原：[使用说明](docs/REPOSITORY_GUIDE.md)。

## 研究状态

日周线正式目标为20万元完整账户扣费后净年化≥10%、净夏普≥1.5、最大回撤≤10%，另须净pB、净期望和独立验证。**目前目标尚未实现，原研究状态仍为research_blocked。** 最新TECH.R268固定双文本×量价阶段账户四场景均拒绝；上传与材料完整性检查不改变研究结论，也未重跑金融研究。

来源盘点于2026-10-06固定，随后按文件稳定读取保存原始版本；本轮完整发布补齐这份固定资料范围，不宣称整个目录在同一瞬间冻结。可用来源缺失和未计算状态按原件保留，详见[来源引用记录](catalog/STATUS_REFERENCES_VERIFICATION.json)。

[上传范围](docs/UPLOAD_SCOPE.md) · [研究目录](catalog/studies.csv) · [完整文件变化](catalog/changes-20261007-full.csv) · [贡献规范](CONTRIBUTING.md) · [协作约定](AGENTS.md) · [变更记录](CHANGELOG.md)
