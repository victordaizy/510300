# 仓库目录与阅读导航

当前版本为 `snapshot-2026-10-06`，增量基线为 `snapshot-2026-09-17`。具体文件、源字节摘要及附件位置由累计索引记录。

## 当前研究入口

- [项目状态](PROJECT_STATE.md)：项目长期事实、已完成研究、来源限制、原冻结口径及当前受阻条件。
- [研究决策](RESEARCH_DECISIONS.md)：重要方向的假设、方法、结果、接受或拒绝原因和再验证条件。
- [日周线项目状态](PROJECT_STATE_TECHNICAL_LINE.md)与[日周线研究决策](RESEARCH_DECISIONS_TECHNICAL_LINE.md)：当前点位与仓位研究的具体事实。
- [本轮更新导航](UPDATE_20261006.md)：本次有新增或修改的研究目录及报告入口。
- [当前状态原件](../reports/research/510300_daily_weekly_goal_continuation_20261001/state.json)：保存时刻与实际金融裁决。
- [正式目标](../config/510300_high_return_sharpe_goal_v1.json)：20万元完整账户、扣费后10%净年化、1.5净夏普、10%回撤上限及其余验收条件。

当前研究目标未实现，最新实际金融裁决是 TECH.R268 固定央行双文本与量价阶段账户拒绝；独立验证未成立，受阻状态按原件保存。本次上传只保存已有工作。

## 文件与来源

研究代码、测试、配置、可读报告和导航在普通 Git 文件中。行情、财报、央行与券商报告原件、详细账本、逐日结果及审阅包根据索引通过 Release 附件管理。

[文件索引](../catalog/README.md)、[研究目录](../catalog/studies.csv)、[逐文件变化](../catalog/changes-20261006.csv)和[纳入范围](UPLOAD_SCOPE.md)说明实际范围。多个历史 Release 由累计索引统一选择，旧标签和附件保留原版本。

## 下载和核对

[首页](../README.md)提供 Windows PowerShell 命令。使用 `tools/repository/snapshot.py restore --only <相对路径>` 按需还原，或者 `restore --all` 还原完整累计资料；已还原文件可用 `verify --all` 核对。GitHub 自动生成的代码 ZIP 只包含普通 Git 文件。

[新附件回读回执](../catalog/SAVED_ASSETS_VERIFICATION.json)覆盖本轮新附件全部成员；[当前状态引用回执](../catalog/STATUS_REFERENCES_VERIFICATION.json)区分已纳入路径和原状态中尚未实际存在的前瞻路径。旧发布回执保存在 `catalog/history/`，不能替代本轮的验证范围。

冻结研究不为上传而重新拟合、回测或改参数。归档核对不构成独立策略验证，具体研究结论以各自的原协议和结果为准。
