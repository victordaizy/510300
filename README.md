# 510300 研究资料库

[![仓库完整性检查](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml)

510300（沪深300 ETF）的研究代码、冻结协议、来源资料、历史结果和审阅交付物。最新版本为 **snapshot-2026-10-06**，在 9 月 17 日快照上新增 **130,698** 个原文件、更新 **4** 个原文件；包含来源工作区的相关未提交内容。

**阅读入口：** [本轮更新导航](docs/UPDATE_20261006.md) · [目录导航](docs/REPOSITORY_GUIDE.md) · [项目状态](docs/PROJECT_STATE.md) · [研究决策](docs/RESEARCH_DECISIONS.md) · [日周线项目状态](docs/PROJECT_STATE_TECHNICAL_LINE.md) · [当前研究状态原件](reports/research/510300_daily_weekly_goal_continuation_20261001/state.json) · [研究目录索引](catalog/studies.csv) · [文件索引](catalog/README.md)

## 最新累计快照

| 内容 | 数量或大小 |
|---|---:|
| 原始文件 | 363,126，共 51.308 GiB |
| Git 原始文件 | 21,537 |
| 通过 Release 还原的原始文件 | 341,589 |
| 本次新增附件 | 57，共 20.232 GiB |
| 继续复用的历史附件 | 63 |
| 累计索引引用附件 | 120，共 38.805 GiB |
| 研究目录索引 | 1018 个目录 |

本次附件见 [2026-10-06 Release](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-10-06)，复用附件仍在 [2026-09-17 Release](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-09-17)。文件索引将全部历史版本组合为完整累计快照；全部原文件均有路径、大小、SHA-256 和附件映射。

## 下载与还原

GitHub 的 Code → Download ZIP 只包含主仓库。需要完整资料时，使用下面的工具自动从对应 Release 下载并还原。新用户无需手工选择新旧附件。

📁 `tools/repository/snapshot.py`（先克隆仓库，再通过 PowerShell 执行）

```powershell
git clone https://github.com/victordaizy/510300.git
Set-Location 510300
python tools/repository/snapshot.py verify
python tools/repository/snapshot.py restore --all
python tools/repository/snapshot.py verify --all
```

支持 `restore --only <相对文件或目录路径>` 按需还原；只使用 Python 标准库。附件缓存保存在 `.release-downloads/`，应同时为缓存、展开内容和临时文件预留空间。原包 `.bin` 分片由工具按序重组，不能独立解压。

若已还原旧版本，先保留本地修改，再切换到新版本。工具遇到不同内容的本地文件默认停止；明确需要用索引版本替换时才使用 `--overwrite`。旧版本始终可以按固定标签单独克隆还原。

## 研究状态与验证范围

当前日周线研究的正式目标是 **20万元完整账户，扣费后净年化≥10%、净夏普≥1.5、最大回撤≤10%**，同时保留净pB、净期望及独立验证要求。**目标尚未实现，当前状态为research_blocked。** 最新金融裁决为 TECH.R268：固定央行双文本×量价阶段账户四场景均拒绝；较早压力年化-1.2850%/夏普-0.527740，近期压力年化-0.0444%/夏普-0.014740。原A近期3.9908%/1.216910也未满足目标。不存在已完成独立验证的达标策略。

本次包含12类83项全因素讨论、具体上涨及失败案例解释、盈利与宏观联合实验、来源公布时钟、央行59季度来源序列及58份实际取得报告、未知项、逐笔交易与全日历账户结果。旧解析/对齐失败版本和后续技术修复版本均按原件保留。日周线方法入口见 [本轮导航](docs/UPDATE_20261006.md)，不能用历史分钟分支的口径替代。

各研究的结论和停止条件以原始协议、结果和时间为准。近期汇总保存时间为 `2026-10-06T16:35:38.263628+08:00`，其中原始状态为 `research_blocked`；该状态不因本次上传改变。根目录 `RESEARCH_STATUS.md` 是较早的历史导航，不能替代最新版本的原始记录。

本次只归档已有材料，未重跑研究、拟合、回测、行情下载或交易。CI 验证累计索引、Git 原文件、忽略规则和还原工具；本地回读验证本轮全部新增附件，历史附件沿用原发布验证并核对远端摘要。未进行全面安全审计。

原始路径、编码、换行和文件字节保持不变。历史完整审阅包中的其他研究上下文按原包保存；范围见 [上传说明](docs/UPLOAD_SCOPE.md)。环境声明见 `requirements.txt` 与 [requirements-snapshot.txt](requirements-snapshot.txt)，不宣称所有历史研究已在全新环境重跑。

## 维护

[贡献规范](CONTRIBUTING.md) · [协作约定](AGENTS.md) · [来源与权属](RIGHTS_AND_SOURCES.md) · [变更记录](CHANGELOG.md)

主分支通过 Pull Request 和 `repository-integrity` 检查维护，禁止强推和删除。后续继续以新标签和新附件版本发布；完整累计索引控制原文件还原位置，旧 Release 不被覆盖。
