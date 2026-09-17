# 510300 研究资料库

[![仓库完整性检查](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml)

510300（沪深300 ETF）的研究代码、冻结协议、来源资料、历史结果和审阅交付物。最新版本为 **snapshot-2026-09-17**，在 9 月 9 日快照上新增 **35,851** 个原文件、更新 **7** 个原文件；包含来源工作区的相关未提交内容。

**阅读入口：** [本轮更新导航](docs/UPDATE_20260917.md) · [目录导航](docs/REPOSITORY_GUIDE.md) · [近期研究状态原文](reports/research/510300_sharpe_1_2_latest_research.json) · [研究目录索引](catalog/studies.csv) · [文件索引](catalog/README.md)

## 最新累计快照

| 内容 | 数量或大小 |
|---|---:|
| 原始文件 | 232,428，共 23.735 GiB |
| Git 原始文件 | 10,033 |
| 通过 Release 还原的原始文件 | 222,395 |
| 本次新增附件 | 5，共 1.422 GiB |
| 继续复用的历史附件 | 58 |
| 累计索引引用附件 | 63，共 18.573 GiB |
| 研究目录索引 | 391 个目录 |

本次附件见 [2026-09-17 Release](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-09-17)，复用附件仍在 [2026-09-09 Release](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-09-09)。文件索引将两版组合为完整累计快照；全部原文件均有路径、大小、SHA-256 和附件映射。

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

各研究的结论和停止条件以原始协议、结果和时间为准。近期汇总保存时间为 `2026-09-17T11:27:00+08:00`，其中原始状态为 `TERMINATED_BY_USER_HIGH_OVERFITTING_RISK`；该状态不因本次上传改变。根目录 `RESEARCH_STATUS.md` 是较早的历史导航，不能替代最新版本的原始记录。

本次只归档已有材料，未重跑研究、拟合、回测、行情下载或交易。CI 验证累计索引、Git 原文件、忽略规则和还原工具；本地回读验证本轮全部新增附件，历史附件沿用原发布验证并核对远端摘要。未进行全面安全审计。

原始路径、编码、换行和文件字节保持不变。历史完整审阅包中的其他研究上下文按原包保存；范围见 [上传说明](docs/UPLOAD_SCOPE.md)。环境声明见 `requirements.txt` 与 [requirements-snapshot.txt](requirements-snapshot.txt)，不宣称所有历史研究已在全新环境重跑。

## 维护

[贡献规范](CONTRIBUTING.md) · [协作约定](AGENTS.md) · [来源与权属](RIGHTS_AND_SOURCES.md) · [变更记录](CHANGELOG.md)

主分支通过 Pull Request 和 `repository-integrity` 检查维护，禁止强推和删除。后续继续以新标签和新附件版本发布；完整累计索引控制原文件还原位置，旧 Release 不被覆盖。
