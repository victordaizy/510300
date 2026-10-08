# 仓库阅读与资料还原

[项目状态](PROJECT_STATE.md)、[研究决策](RESEARCH_DECISIONS.md)、[日周线状态](PROJECT_STATE_TECHNICAL_LINE.md)和[日周线决策](RESEARCH_DECISIONS_TECHNICAL_LINE.md)保存长期事实。新增完整数据范围见[完整资料更新](UPDATE_20261007_FULL.md)，研究目录入口见 `catalog/studies.csv`。

Git克隆包含代码、文档、主要结果与累计索引；Release附件包含纳入范围的完整数据、详细账本、来源及历史原始包。GitHub的Code ZIP不包含Release附件。还原工具依据累计索引同时读取新旧Release，恢复原路径并核对文件摘要。

在克隆后的仓库根目录，用PowerShell执行下列命令。还原会写入选择的原文件，需要为这些原件保留足够空间；默认一次下载一个附件，使用后删除下载缓存。

📁 克隆目录内的 `tools\repository\snapshot.py`

```powershell
python tools/repository/snapshot.py restore --only reports/research/510300_pbc_report_phase_account_v1/
python tools/repository/snapshot.py restore --only data/nbs_monthly/
python tools/repository/snapshot.py verify
```

每条 `--only` 是原始路径前缀，可以重复指定。`restore --all` 还原完整Release材料；还原原件合计超过50 GiB，须预留对应空间。`verify --all` 在完整还原后验证全部源文件。`--keep-downloads` 明确保留下载缓存，默认无需该参数。

索引保留来源原字节和历史状态，未保证所有历史入口都能在新环境立即复跑。本轮未重跑研究；原目标尚未实现，拒绝、未知及缺失按原件保留。
