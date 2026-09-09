# 贡献与版本管理

## 修改流程

1. 从 `main` 创建 `codex/研究编号-改动说明` 分支。
2. 在 Pull Request 中说明问题、变更结果、来源文件和实际验证。
3. 通过 `repository-integrity` 检查后合并。主分支采用线性历史，禁止强制推送和删除；合并后删除工作分支。

推荐提交类型为 `docs:`、`chore:`、`fix:`、`feat:`、`research:`。标题说明具体结果，不把资料整理写成模型有效性结论。

## 冻结文件与新增版本

`catalog/files-*.csv` 中的每一行对应本次迁移的一个原始文件。已有文件按原始字节保存；修改换行、编码或小数格式同样会改变摘要。研究结论和冻结协议需要修订时，创建明确的新研究版本，保留旧版本及旧 Release。

新增文档和工具通过普通 Git 跟踪。后续增加大数据快照时创建新的 Release 和文件索引，不用同名附件覆盖已经发布的冻结材料。索引应说明源路径、大小、SHA-256、存储位置和包含理由，并验证每个附件与源文件的映射。

发布索引更新后，运行 `python tools/repository/update_ignore_rules.py` 生成相应的还原路径忽略规则。它会使用Git逐条验证Release路径与Git源文件路径，避免还原后的批量数据被再次纳入普通Git提交。

## 本地验证

📁 `tools/repository/snapshot.py`（在仓库根目录的 PowerShell 执行）

```powershell
python tools/repository/snapshot.py verify
python -m unittest discover -s tests/repository -v
```

第一条命令仅检查主仓库中的原始文件与完整索引；第二条验证还原工具的路径边界、无损分片和失败时保留本地修改。完整材料还原后，可以使用 `verify --all` 检查全部源文件。

这些检查不调用行情服务，不重新计算研究收益，不构成研究结论验证。历史 `tests/` 中还有冻结前状态断言、数据依赖与可能触发研究流程的测试；执行前应先确认该研究的权限与入口。

## 依赖与环境

`requirements.txt` 保留原项目声明，`catalog/runtime.json` 记录本次归档时的 Python 和已安装包版本。新增的下载还原工具只使用 Python 标准库。环境记录不是所有历史研究已在全新环境重跑通过的证明。

## 权属与第三方来源

见 `RIGHTS_AND_SOURCES.md`。新增来源保留原链接、来源日期和数据定义；不要把第三方资料重新标为本项目原创，也不要替源材料添加未经授权的开放许可。
