# A股财报公布后漂移（PEAD）数据可行性裁决

- 候选：基于已公布财报盈利变化的横截面 `POST_EARNINGS_ANNOUNCEMENT_DRIFT`。
- 状态：`NO_VIEW_BLOCKED_NO_STRICT_POINT_IN_TIME_FINANCIAL_VINTAGE`。
- 决定：`STOP_BEFORE_SIGNAL_OR_RETURN_CALCULATION`。

## 为什么不运行

本地财务档案有 45,198 条事件、927 只证券，但它们是 2026 年统一回取后按供应商公告日期重建 `available_at`。既有分层审计抽查 240 条，`revision_possible=true` 为 240 条，能够证明原公告日版本的 `vintage_verified=true` 为 0 条。因此，这些文件可以做当前档案重建，不能证明历史预测原点实际可见的数值与今天回取值一致。

历史沪深300成员与权重也未形成同期官方闭环：15 个成员变更事件中完整官方事件为 0，120 个月历史权重中具有同期官方快照归档的月份为 0。当前官网快照一致不能外推历史。

证据文件：

- `reports/data_quality/csi300_financial_vintage_audit_20260819.md`
- `reports/data_quality/csi300_point_in_time_membership_evidence_20260819.md`
- `reports/data_quality/000300_normalized_earnings_financial_extension_v1.md`

## 研究边界

本裁决没有计算盈利信号、未来收益、IC、组合、成本、仓位或订单。只有获得原公告日同期财务版本归档，并补齐点时成员证据后，才允许在新协议下重新评估；不得把 2026 年回取值当成严格 PIT 数据完成历史 Alpha 验证。
