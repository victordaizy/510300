# 510300 研究资料库

[![仓库完整性检查](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml/badge.svg)](https://github.com/victordaizy/510300/actions/workflows/repository-checks.yml)

510300（沪深300 ETF）的研究代码、冻结协议、来源资料、历史结果和审阅交付物。本仓库来自 `New project 8` 的一次文件快照，包含原工作区中尚未提交到 Git 的相关材料。

**从这里开始：** [目录导航](docs/REPOSITORY_GUIDE.md) · [研究状态原文](RESEARCH_STATUS.md) · [研究报告索引](catalog/studies.csv) · [全部附件下载](https://github.com/victordaizy/510300/releases/tag/snapshot-2026-09-09) · [文件索引](catalog/README.md)

## 本次快照

| 项目 | 数量或范围 |
|---|---:|
| 快照版本 | `snapshot-2026-09-09` |
| 原始文件总数 | 196,577 |
| 原始文件总大小 | 21.461 GiB |
| 主仓库中的原始文件 | 7,583 |
| 通过 Release 还原的原始文件 | 188,994 |
| Release 附件 | 58 个，共 17.151 GiB |
| 来源 Git 提交 | `9617d925c8c3ea3e9cb179e59967c270f7e49379` |

所有原始文件均有路径、字节数、SHA-256 和存储位置记录。主仓库还增加了本 README、维护规范、文件索引和还原工具，这些新文件不计入上述原始文件数。

## 获取材料

代码、协议和可读报告可以直接在 GitHub 浏览或克隆。大型原始资料、研究证据、历史交付 ZIP 和需要无损分片的原包存放在本仓库的 Releases。GitHub 页面自带的 **Download ZIP** 只包含主仓库，完整材料需要另行下载附件。

📁 `tools/repository/snapshot.py`（先克隆仓库，再在 PowerShell 中执行）

```powershell
git clone https://github.com/victordaizy/510300.git
Set-Location 510300
python tools/repository/snapshot.py verify
python tools/repository/snapshot.py restore --all
python tools/repository/snapshot.py verify --all
```

也可以按路径下载某一研究或历史审阅包。

📁 `tools/repository/snapshot.py`（在仓库根目录的 PowerShell 执行）

```powershell
python tools/repository/snapshot.py restore --only data/raw/510300_forward_eps_soochow_originals_v1/
python tools/repository/snapshot.py restore --only deliverables/510300_CSP_V1_STATIC_VS_TIMING_ATTRIBUTION_V1_GPT_REVIEW_20260906.zip
```

还原工具只使用 Python 标准库；保留原始路径、编码和换行，验证附件及还原文件的摘要，并支持重复执行。附件缓存位于 `.release-downloads/`；完整下载时应为附件缓存、展开文件和临时文件预留足够空间。存在不同内容的本地文件时默认停止，防止覆盖本地修改。

## 阅读与使用边界

各轮研究的结论、停止条件和权限以对应冻结文件为准。目录中同时保留成功、拒绝、阻断和未计算等历史状态；已有 `RESEARCH_STATUS.md` 是来源项目原文，阅读时需结合其所指向的具体研究版本。

本次仅整理并上传已有材料，没有重新下载数据、拟合、回测或生成仓位和订单。既有历史审阅包可能同时保留其他研究的对照或上下文；这些原包按原字节保存，来源范围见 [上传范围说明](docs/UPLOAD_SCOPE.md)。

研究环境声明见 `requirements.txt`，已安装包的版本快照见 [requirements-snapshot.txt](requirements-snapshot.txt)，完整本机环境记录见 [runtime.json](catalog/runtime.json)。全部历史研究未在本次上传中重新运行；CI 只验证仓库索引、主仓库文件字节和还原工具。

## 维护

[贡献规范](CONTRIBUTING.md) · [协作约定](AGENTS.md) · [权属与第三方来源](RIGHTS_AND_SOURCES.md) · [变更记录](CHANGELOG.md)

主分支通过 Pull Request 和 `repository-integrity` 检查维护，禁止强制推送和删除。原始冻结文件通过 `.gitattributes` 保留原字节；大文件与附件映射通过 `catalog/` 管理。
