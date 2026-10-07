# 510300 M1/M2月频增量与风险复核

截至2026-09-11的有限历史研究；打包时间2026-09-17T20:18:17.314201+08:00。

本轮旧口径宏观增量未通过，新口径样本不足；B/D账户未运行。已有风险预测门通过，但C账户收益和夏普低于A。85/15保持终止。没有独立达标证据，没有订单。

建议阅读顺序：

1. reports/research/510300_m1_m2_monthly_increment_v1/研究报告.md
2. reports/research/510300_m1_m2_monthly_increment_v1/USER_REQUEST.md及GPT审阅提示词.md
3. reports/research/510300_m1_m2_monthly_increment_v1/protocol.json、source_admission.json、prediction_result.json、status.json
4. reports/research/510300_m1_m2_monthly_increment_v1/证据地图与排除项.md及去重与选择历史.md
5. reports/research/510300_m1_m2_monthly_increment_v1/verification中的复核回执与risk_reused_account_comparison.csv
6. FILE_INDEX.csv及history/configuration_snapshot中的上游登记配置

直接来源、全部预测、保存系数、区块索引、已复核风险研究的全部20份账户及本轮代码均包含。原始PBC目录及失败回执保留。完整旧实验配置是上下文，不代表附带整个旧项目的运行数据。

离线复核使用Python、numpy、pandas、scipy、requests、beautifulsoup4等现有运行环境。在本包根目录运行 `python -m research.verify_m1_m2_monthly_increment_v1 --output 本次离线复核`，会写入新的核对回执，不重新拟合、不重新抽样、不下载、不生成账户。输出目录必须没有同名回执；不需运行admit/freeze/predict。

读取HTML和CSV即可人工审阅。全部新统计结果属于已用历史的顺序回放；原官网页面为现时重建，严格前向月份为0。只有本地算术核对及包结构校验，不代表外部GPT已经评审。包不扩展研究或交易权限。
