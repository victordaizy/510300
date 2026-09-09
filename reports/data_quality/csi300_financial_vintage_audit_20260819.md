# 沪深300财务事件历史版本审计

状态：`BLOCKED_VENDOR_REVISION_RISK`

## 总体与抽样

- 总体：45,198条、927只证券、15个报告年份。
- 确定性分层样本：240条，覆盖15个年份、32个行业层和4种季度类型。
- 样本中 `revision_possible=true`：240条。
- 样本中 `vintage_verified=true`：0条。
- 三类checkpoint哈希齐全：240条。

## 阻断缺口

- 45,198条事件由2026年统一回取的供应商响应构造，缺少各原公告日同期原始响应。
- available_at重建供应商声明的历史可得日，但不能证明该日看到的数值与2026回取值一致。
- 供应商数据没有revision_id、前值、修订时间或历史vintage快照链。
- checkpoint文件证明本次回取输入，不是原公告日同期归档；抽样记录全部revision_possible=true。
- original_publication_at来自供应商ann_date字段合并，不是交易所公告原文逐条核验结果。

## 证据表

- `reports/audit/csi300_financial_vintage_sample_20260819.csv`
- `reports/audit/csi300_financial_checkpoint_inventory_20260819.csv`

本审计不修改原始数据，不把2026年统一回取值称为严格PIT。
