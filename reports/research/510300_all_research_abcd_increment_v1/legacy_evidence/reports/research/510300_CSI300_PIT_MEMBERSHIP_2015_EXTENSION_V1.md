# 沪深300点时成分 2015 扩展验收报告

## 结论

- 成分状态：`PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION`。
- 合并面板：2015-01-05 至 2026-08-14，共 2823 个开市日、846900 行，每日恰好 300 只。
- 2015 官方调样：5 轮、共 42 进 42 出。
- 2015-03-02 官方锚点集合差异为 0；2015-07-01、2015-09-01 两个独立官方快照差异均为 0。
- 与 2016H1 官方主链在 2016-06-13 的集合差异为 0；第二次确定性重放哈希一致。
- 权重状态仍为 `BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS`。2015 权重文件在这里只作为成分集合锚点，不能据此把历史权重值判为合格。

## 研究边界

本轮没有读取市场价格、未来收益或标签，没有运行预测模型、组合回测、Paper/Shadow 或实盘交易。被冻结候选仍保持 `BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS`；当前剩余数据障碍已缩窄为历史权重的逐期版本或 as-of 来源凭证。

## 主要交付物

- 完整日频成分面板：`data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet`
- 2015 扩展来源清单：`data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/official_2015_extension_manifest.json`
- 2015 扩展验收清单：`data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/official_2015_extension_admission_manifest.json`
- 重放审计：`data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/official_2015_extension_replay_audit.json`
- 官方快照比较：`data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/official_2015_extension_snapshot_comparison.parquet`
