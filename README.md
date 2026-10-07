# 510300 研究资料库

[![仓库完整性检查](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml)

最新研究材料截至2026年10月6日，按用户在10月7日明确选择的范围上传：**新研究代码、文档和主要结果，排除历史大包**。本轮纳入 5,579 个代码、文档和结果文件，原文件合计 157.13 MiB；没有新增数据压缩包或 Release 附件。逐文件变化见 [变化索引](catalog/changes-20261006.csv)。

[项目状态](docs/PROJECT_STATE.md) · [研究决策](docs/RESEARCH_DECISIONS.md) · [日周线状态](docs/PROJECT_STATE_TECHNICAL_LINE.md) · [日周线决策](docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md) · [本轮更新](docs/UPDATE_20261006.md)

## 本轮内容

日周线量价点位、具体上涨与失败案例、宏观与全因素、盈利联合评分、央行经济描述与政策指引双文本阶段实验。包括研究实现、测试代码、冻结协议、结论、主要比较结果、图和模型记录；来源未知、研究失败及被拒绝方向按原文保留。

主要结果 CSV 新增 390 个；当前状态见 [原始研究状态](reports/research/510300_daily_weekly_goal_continuation_20261001/state.json)，最新完整结论见 [TECH.R268研究结论](reports/research/510300_pbc_report_phase_account_v1/央行双文本与量价阶段_完整研究结论.md)。

## 当前研究结论

正式目标为20万元完整账户、扣费后净年化≥10%、净夏普≥1.5、最大回撤≤10%，另须净pB、净期望和独立验证。**目标尚未实现，当前research_blocked。** 最新TECH.R268固定央行双文本×量价阶段账户四场景均拒绝：较早压力年化−1.2850%、夏普−0.527740；近期压力年化−0.0444%、夏普−0.014740。原A近期年化3.9908%、夏普1.216910也未满足目标。上传不改变这些结论。

## 上传范围与使用

本轮没有上传新的历史数据全量副本、财报/PDF原件全集、详细训练和逐日账本全集、完整审阅ZIP及重复状态备份；这些原件仍保存在本地。研究代码复算仍需要其冻结协议指定的本地输入，不能宣称仅克隆仓库即可完整重跑所有研究。具体遗漏引用在 [引用与范围记录](catalog/STATUS_REFERENCES_VERIFICATION.json) 中明示。

已有9月17日及更早版本保持原内容，累计索引继续引用其 63 个历史附件；本轮没有为它们新增本地副本。GitHub的Code ZIP包含本轮提交的代码、文档和主要结果。已有历史Release材料可以按原还原工具与索引另行获取；`restore --all`只覆盖索引中的已发布历史材料和本轮Git文件，不包含本轮明确省略的原始数据。

本次仅归档已有研究，不重跑拟合、回测、行情下载或交易。必需检查只验证Git文件字节、累计索引、忽略规则和还原工具，不构成独立金融验证。

[目录导航](docs/REPOSITORY_GUIDE.md) · [文件索引](catalog/README.md) · [上传范围](docs/UPLOAD_SCOPE.md) · [贡献规范](CONTRIBUTING.md) · [协作约定](AGENTS.md) · [变更记录](CHANGELOG.md)
