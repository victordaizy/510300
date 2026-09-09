# 510300 PIT 盈余信息扩散 10 日风险 V1：数据可行性

- 权威状态：`BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS`
- 研究阶段：`DATA_FEASIBILITY_ONLY`
- 模型动作：`ABSTAIN`
- 模型仓位目标：`UNSET`
- 市场价格读取：`0`
- 未来数据读取：`0`
- 预测模型/未来10日标签/组合回测：均未创建

## 顺序启动

期权链前序状态为 `BLOCKED_NO_COMPLETE_POINT_IN_TIME_OPTION_CHAIN`；顺序启动门为 `PASS`。

## 公告与成员覆盖

- 全 A 股目标公告：88,677
- 历史沪深300成员公告：3,970
- 历史成员发行人：511
- 不同公告日期：1,194
- 日期级保守时钟占比：87.63%
- 首个拥有严格前序合格权重的成员公告日：`2016-10-10 00:00:00`
- 该日起严格前序权重覆盖率：99.97%

现有权重从 2016-08 才开始，因此不能制造 2015 年加权历史。同日权重不用于同日公告，所有公告最早在公告日之后首个交易日使用。

## 事实抽取状态

- 检查点状态：`RUNNING_MULTIPROCESS_FACT_CHECKPOINT_EXECUTION`
- 本轮完成：65000/79471
- 显式失败：0
- 自动汇总状态：`NOT_YET_AVAILABLE`
- 双人独立人工复核完成：`false`

## 门槛结论

- 已失败硬门：['historical_membership_and_weights']
- 尚不可评价硬门：['fact_checkpoint_execution', 'fact_automation', 'first_public_weighted_fact_prevalence', 'formal_dual_human_review']

当前状态若为 `RUNNING` 或 `PENDING`，均不等于数据通过；在正式 `PASS_*_DATA_FEASIBILITY_ONLY` 前不得冻结预测模型、读取未来10日标签或运行组合回测。

## 来源与限制

- 盈利公告元数据和 PDF 来自巨潮资讯官方源；首次事实仍需冻结抽取和人工复核。
- 历史成分是第三方调样公告重建，不得称作官方历史成分库。
- 月度权重来自 Tushare 历史快照，供应商历史修订版本不可证明。
- 不使用分析师一致预期，不使用供应商财务历史，不以 IF、宏观或价格指标补缺。
