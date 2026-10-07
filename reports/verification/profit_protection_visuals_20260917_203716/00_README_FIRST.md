# 510300 先涨后回撤退出：图形补充包

请从 [图形说明与结论](reports/research/510300_profit_protection_visuals_v1/00_README_FIRST.md) 开始，再看该目录内两张 PNG。

- [人工路径机制图](reports/research/510300_profit_protection_visuals_v1/01_先上涨后回撤退出_机制示意.png)
- [真实历史净值、回撤、仓位图](reports/research/510300_profit_protection_visuals_v1/02_实际历史_完整日度净值回撤仓位.png)
- [方法与规则](reports/research/510300_profit_protection_visuals_v1/方法与规则说明.md)
- [用户要求](reports/research/510300_profit_protection_visuals_v1/USER_REQUEST.md)
- [可复制审阅提示词](GPT审阅提示词.md)
- [覆盖范围与排除项](reports/research/510300_profit_protection_visuals_v1/包内容与排除项.md)

示例采用 10% 启动、5% 回撤，但没有将它们冻结为研究参数，也没有运行该退出策略的账户。历史图中的 A、C、买入持有未加入新退出规则。此前终止策略维持终止。

全部直接绘图账本、保存指标、绘图脚本与日度导出数据均已包含。使用 Python 和目录内运行环境说明中的依赖，执行 research/visualize_profit_protection_and_monthly_baselines_v1.py 可重画，加 --verify-only 可仅复核日度数据。宏观上下文只保留协议、报告和状态快照；原宏观原始库和账户引擎不属于本包的复现范围。

FILE_INDEX.csv 是本包索引；不索引其自身。CRC、索引、大小/哈希以及解压件数值复核记录在包外交付回执。本包未上传，未经过外部模型评阅。
