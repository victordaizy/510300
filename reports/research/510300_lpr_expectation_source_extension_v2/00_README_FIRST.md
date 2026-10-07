# 阅读导航

1. `研究结论.md`：新增七个月、完整缺口、继承样本要求与本轮停止条件。
2. `用户需求.md`、`mandate.json`、`source_plan.json`：用户20万元/1.5/10%约束与有限检索计划。
3. `新增七个月调查事实.csv`、`new_facts.json`、`new_sources.json`：新增数值、日期、时钟类型、链接、哈希和保留的歧义。
4. `inputs/`、`全部84月事前预期识别.csv`、`ledger.json`：原49月快照、合并56月资料和完整84月母集。
5. `protocol.json`、`source_freeze_receipt.json`、`summary.json`、`prediction_feasibility.json`：冻结身份、结果与36训练/24评价下的乐观计数上限。
6. `receipts/`、`evidence/`：来源回执、继承和新执行查询、三个尚未准入的原文线索。
7. `GPT审阅提示词.md`、`EXCLUSIONS.md`：反驳重点与交付范围。

离线核对命令：使用Python执行 `code/lpr_expectation_source_extension_v2.py verify --root <解压目录>`。仅需标准库，复算时钟关系、数值区间、全84月和汇总；不联网、不读取证券行情。来源下载脚本另外依赖requests、beautifulsoup4。全文缓存只留本地，网页首版认证不在离线复核范围内。
