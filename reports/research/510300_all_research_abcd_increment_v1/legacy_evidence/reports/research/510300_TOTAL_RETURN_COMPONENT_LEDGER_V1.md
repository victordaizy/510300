# 510300_TOTAL_RETURN_COMPONENT_LEDGER_V1

## 裁决

60D/120D 总回报成分账本已完成。它是会计与归因产物，不是收益预测、策略回测或仓位映射。

- 协议哈希：`f32e6a66c0bcdd3f73331c3fe3cef38d829463e6f4367dffc47710bfaff0e277`
- 账本：`807` 行；覆盖率：`269` 行；恒等式审计：`807` 行。
- 财务覆盖状态：`{"PARTIAL_FIXED_FINANCIAL_COHORT_COVERAGE": 63, "PASS_FIXED_FINANCIAL_COHORT_COVERAGE": 206}`
- 恒等式状态：`{"NO_VIEW_OR_FAIL_FIXED_MARKET_IDENTITY": 44, "PASS_EXACT_IDENTITY_WITH_UNRESOLVED_EARNINGS_MULTIPLE": 225, "PASS_EXACT_MULTIPLICATIVE_IDENTITY": 269, "PASS_EXACT_MULTIPLICATIVE_IDENTITY_ON_FINANCIAL_COHORT": 269}`
- `RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`。

## 三种口径

| 口径 | horizon | 行数 | 恒等式通过 | 市场覆盖中位数 | 市场覆盖最低 | 财务覆盖中位数 | 财务覆盖最低 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTUAL_INDEX_CHAINED_SCOPE | 60D | 136 | 136 | 100.00% | 100.00% | 100.00% | 100.00% |
| ACTUAL_INDEX_CHAINED_SCOPE | 120D | 133 | 133 | 100.00% | 100.00% | 100.00% | 100.00% |
| ORIGIN_FIXED_COMPONENTS_AND_WEIGHTS_SCOPE | 60D | 136 | 115 | 99.80% | 90.33% | NA | NA |
| ORIGIN_FIXED_COMPONENTS_AND_WEIGHTS_SCOPE | 120D | 133 | 110 | 99.80% | 89.51% | NA | NA |
| ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE | 60D | 136 | 136 | 99.80% | 90.33% | 99.01% | 90.22% |
| ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE | 120D | 133 | 133 | 99.80% | 89.51% | 98.22% | 88.68% |

## 成分定义与限制

1. 实际指数链式口径：`H00300/000300` 给出隐含股息再投资因子；`000300/PE_TTM` 给出链式盈利代理，`PE_TTM` 变化给出倍数重估。恒等式精确，但历史 PE 并非首次发布版本档案，因此盈利/倍数状态为 `PARTIAL`。
2. origin 固定成分与权重口径：以冻结 origin 的市值代理权重买入并持有。总回报与原始价格之比只能识别“分配及公司行动”合并项；缺少成分股逐笔分配档案，纯股息项为 `NO_VIEW`。
3. 固定财务覆盖 cohort：只在同一 origin 固定持股、起止 PIT 财务事实、有效股本和起止价格均可用的覆盖 cohort 上，将价格精确分解为盈利增长×倍数重估。低于 80% 权重覆盖时不填数。
4. `H00300/固定 cohort 总回报` 被登记为“成分/权重＋指数方法＋代理权重残差”，不能冒充纯粹的换仓收益。历史官方权重版本仍未验证。
5. 跟踪残差使用含官方现金分红的 510300 总回报与 H00300 总回报之比。

## 精确恒等式

- 实际指数：`指数隐含股息 × 链式盈利 × 链式倍数 × ETF跟踪 = ETF总回报`。
- 固定市场 cohort：`分配及公司行动 × 未拆分价格 × 成分权重方法残差 × ETF跟踪 = ETF总回报`。
- 固定财务 cohort：`分配及公司行动 × 盈利增长 × 倍数重估 × 成分权重方法残差 × ETF跟踪 = ETF总回报`。

恒等式通过只证明账本算术闭合，不证明任何成分可预测。

## 下一步

分别检验 CF→盈利/现金流/breadth、DR→倍数/ERP gap、RC→波动/下行半方差/相关性/流动性冲击的测量有效性。任何模块都不得直接改回总回报目标。当前仍为 `ABSTAIN / POSITION_UNSET`。
