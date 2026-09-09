# 历史行情情境补充 V1

- 情境截止：`2026-08-18`
- 状态：`PARTIAL_STALE_INDUSTRY_CONTEXT`
- 原预测：`NO_VIEW`（未改变）
- 价格风险情境：`PRICE_RISK_MIXED`
- 趋势：`SHORT_RECOVERY_NOT_CONFIRMED`
- 估值历史位置：`HIGH`
- 成交参与位置：`MIDDLE`
- 波动历史位置：`MIDDLE`

## 当前关键指标

- `close`：`7077.880000`
- `pe_ttm`：`14.570000`
- `amount`：`6198.290000`
- `amount_percentile_242d`：`0.450413`
- `close_vs_ma20`：`0.015942`
- `close_vs_ma60`：`-0.006328`
- `drawdown_60d`：`-0.058461`
- `pe_ttm_percentile_1210d`：`0.930579`
- `return_120d`：`0.017737`
- `return_20d`：`-0.001066`
- `return_60d`：`-0.028580`
- `rv_20d`：`0.173775`
- `rv_60d`：`0.233092`
- `rv_20d_history_percentile`：`0.663765`

## 历史相似行情

选出 `8` 个间隔去重的相似日。

| 排名 | 相似日 | 距离 | 后20日 | 20日最差路径 | 后60日 | 60日最差路径 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2021-08-25 | 0.6486 | -0.97% | -1.97% | -0.66% | -1.97% |
| 2 | 2025-12-02 | 0.6543 | 2.32% | -1.19% | 2.72% | -1.19% |
| 3 | 2021-10-11 | 0.6753 | -1.71% | -2.25% | -0.29% | -2.25% |
| 4 | 2024-12-31 | 0.8246 | -2.12% | -5.32% | -1.64% | -5.32% |
| 5 | 2025-12-31 | 0.9969 | 1.77% | 1.68% | -3.90% | -4.39% |
| 6 | 2026-02-13 | 1.0462 | -5.20% | -5.20% | 4.18% | -5.20% |
| 7 | 2025-02-06 | 1.0951 | 2.95% | 1.05% | -0.14% | -6.58% |
| 8 | 2025-10-23 | 1.0965 | -0.73% | -0.73% | 3.25% | -3.27% |

相似样本分布：

- 20日：中位数 `-0.85%`，四分位区间 `-1.81%` 至 `1.91%`，正收益比例 `37.5%`，最差路径样本 `-5.32%`。
- 60日：中位数 `-0.21%`，四分位区间 `-0.91%` 至 `2.85%`，正收益比例 `37.5%`，最差路径样本 `-6.58%`。

该分布是描述性条件样本，不是样本外胜率。

## 行业价格位置

行业行情截止 `2026-08-14`，落后市场截止 `2` 个交易日，状态为 `STALE_HISTORICAL_CONTEXT`。

| 行业 | 权重 | 原预期差 | 20日收益 | 60日收益 | 60日历史分位 | 价格位置 | 旁注 |
|---|---:|---|---:|---:|---:|---|---|
| 电子 | 17.49% | BALANCED | -0.26% | -1.27% | 39.7% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 银行 | 11.33% | UNOBSERVED | 1.25% | 4.51% | 62.8% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 非银行金融 | 8.74% | BALANCED | -0.94% | 1.86% | 64.9% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 通信 | 8.55% | BALANCED | -2.66% | -2.47% | 34.6% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 电力设备及新能源 | 8.13% | UNOBSERVED | 10.04% | -12.34% | 18.7% | EXTENDED_DOWN | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 有色金属 | 5.92% | POSITIVE | 11.36% | -9.47% | 19.2% | EXTENDED_DOWN | POSITIVE_GAP_NOT_PRICE_EXTENDED |
| 食品饮料 | 5.78% | UNOBSERVED | 5.10% | 0.86% | 55.8% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 医药 | 4.35% | UNOBSERVED | 8.63% | 12.64% | 85.8% | EXTENDED_UP | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 计算机 | 3.76% | BALANCED | 8.49% | -8.97% | 26.4% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 汽车 | 3.14% | NEGATIVE | 2.69% | -10.72% | 17.3% | EXTENDED_DOWN | NEGATIVE_GAP_BUT_PRICE_ALREADY_WEAK |
| 家电 | 3.02% | BALANCED | -0.19% | 0.63% | 45.5% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 交通运输 | 2.91% | BALANCED | 4.04% | -4.29% | 25.2% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 电力及公用事业 | 2.88% | BALANCED | 0.05% | 2.15% | 51.8% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 机械 | 2.76% | BALANCED | 4.47% | -10.97% | 15.0% | EXTENDED_DOWN | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 基础化工 | 2.22% | POSITIVE | 7.34% | -8.95% | 27.1% | MIDDLE_RANGE | POSITIVE_GAP_NOT_PRICE_EXTENDED |
| 石油石化 | 1.54% | BALANCED | 6.83% | -2.24% | 37.0% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 建筑 | 1.33% | NEGATIVE | -0.67% | -5.77% | 25.0% | MIDDLE_RANGE | NEGATIVE_GAP_NOT_FULLY_VISIBLE_IN_PRICE |
| 国防军工 | 1.27% | BALANCED | 9.05% | -15.20% | 8.2% | EXTENDED_DOWN | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 煤炭 | 1.09% | UNOBSERVED | 5.94% | 5.22% | 47.8% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 农林牧渔 | 0.92% | UNOBSERVED | -3.02% | -1.67% | 48.1% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 传媒 | 0.79% | BALANCED | 4.21% | -9.14% | 31.7% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |
| 建材 | 0.52% | NEGATIVE | -2.01% | 0.27% | 48.5% | MIDDLE_RANGE | NEGATIVE_GAP_NOT_FULLY_VISIBLE_IN_PRICE |
| 钢铁 | 0.46% | NEGATIVE | 2.56% | -5.56% | 29.4% | MIDDLE_RANGE | NEGATIVE_GAP_NOT_FULLY_VISIBLE_IN_PRICE |
| 房地产 | 0.33% | NEGATIVE | 8.28% | -10.09% | 33.7% | MIDDLE_RANGE | NEGATIVE_GAP_NOT_FULLY_VISIBLE_IN_PRICE |
| 商贸零售 | 0.14% | UNOBSERVED | 10.10% | -9.92% | 40.2% | MIDDLE_RANGE | PRICE_CONTEXT_ONLY_NO_GAP_CHANGE |

## 仍未补齐

- `CURRENT_FUNDING_LIQUIDITY`
- `CURRENT_EQUITY_FLOW`
- `CURRENT_NATIONAL_TEAM_HOLDINGS`
- `CURRENT_NATIONAL_TEAM_ACTIVITY`
- `CURRENT_INDUSTRY_PRICE_CONTEXT_AFTER_2026-08-14`

历史行情补充了价格情境，但不能把原NO_VIEW升级为方向观点；相似行情结果是描述性条件样本，不是样本外Edge。
