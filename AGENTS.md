# 协作约定

- 对话、新增代码注释和运行日志默认使用简体中文；开发命令须兼容 Windows PowerShell。
- 本仓库服务于 510300 研究。已有跨市场及其他证券材料可能作为既有研究的输入、对照或交付上下文保存，不能据此扩展研究或交易权限。
- 新分支使用 `codex/` 前缀，通过 Pull Request 合并；提交说明写清改动目的和实际验证结果。
- `catalog/` 记录原始文件与 Release 附件的映射。冻结协议、清单、原始数据、回执及历史结果不做格式化或换行转换；修订建立新版本并说明来源。
- `NO_VIEW`、`ABSTAIN`、`BLOCKED_*`、`NOT_COMPUTED` 等状态按原始语义保留。归档上传不授权重新下载、拟合、回测、Paper/Shadow、券商连接、下单或实盘。
- 默认检查命令为 `python tools/repository/snapshot.py verify` 和 `python -m unittest discover -s tests/repository -v`。不为验证仓库上传而直接运行整个历史研究测试目录。
- 不提交 `.env`、本机凭据、虚拟环境和可再生缓存。不增加与具体改动无关的审计工作。
- 参考 `CONTRIBUTING.md`、`docs/REPOSITORY_GUIDE.md` 和各轮原始研究文档。
