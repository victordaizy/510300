# 510300估值仓位与技术指标统一重筛

> 本轮一次性比较15条技术轨道。估值模型决定仓位，技术指标只优化普通加减仓时点；任何历史赢家都不自动授权交易。

## 数据纠正

- 正式输入是2021-08-12至2026-08-12的五年Tushare代理1分钟K线聚合15分钟OHLCVA，共1211个完整交易日。
- 日OHLCVA与独立日线1211日逐日完全一致；来源仍有第三方代理和时间戳语义警告。
- 旧的“两年High/Low限制”只属于通达信原生15分钟主文件，不代表仓库缺少五年可审计分钟OHLCV。

## 估值基线

- 5bp总收益3.93%，CAGR 0.77%，最大回撤-30.68%，期末权益20785.74元。

## 候选排名

| 排名 | 技术轨道 | 5bp总收益 | 相对基线权益 | 最大回撤 | 15bp总收益 | 买/卖确认 | 分段 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | MACD_12_26_9_CROSS | 4.71% | +155.91元 | -30.84% | 3.47% | 26/22 | -0.19%/+1.42% | `REJECTED_RETROSPECTIVE` |
| 2 | VWAP_REVERSION_V1 | 4.17% | +48.94元 | -30.44% | 3.16% | 8/14 | +0.74%/-1.06% | `INSUFFICIENT_EVIDENCE` |
| 3 | ORB_30M_RVOL_BREAKOUT | 4.00% | +14.95元 | -30.03% | 2.24% | 25/20 | +0.70%/-1.21% | `NEAR_MISS_FORWARD_OBSERVATION_ONLY` |
| 4 | BOLLINGER_20_2_REENTRY | 3.25% | -136.25元 | -30.96% | 1.56% | 26/20 | +0.09%/-1.09% | `REJECTED_RETROSPECTIVE` |
| 5 | EMA_8_21_CROSS | 2.92% | -201.84元 | -31.23% | 1.15% | 21/19 | -0.65%/-0.13% | `INSUFFICIENT_EVIDENCE` |
| 6 | LC_FVG_PV_V1 | 2.52% | -281.13元 | -30.64% | 1.22% | 7/4 | -0.08%/-1.75% | `INSUFFICIENT_EVIDENCE` |
| 7 | COMPOSITE_RVOL_VWAP_EMA_ADX_RSI_ORB | 2.25% | -335.26元 | -30.49% | 0.66% | 19/14 | +0.19%/-2.62% | `INSUFFICIENT_EVIDENCE` |
| 8 | CLOSE_PRESSURE_30M_MODEL | 1.59% | -468.62元 | -30.85% | 0.41% | 17/16 | -0.39%/-2.44% | `INSUFFICIENT_EVIDENCE` |
| 9 | RSI_7_REVERSAL_30_70 | 1.54% | -477.95元 | -30.75% | 0.10% | 26/20 | +0.00%/-3.23% | `REJECTED_RETROSPECTIVE` |
| 10 | SESSION_VWAP_CROSS | 1.30% | -525.59元 | -31.22% | 0.11% | 28/23 | -0.56%/-2.50% | `REJECTED_RETROSPECTIVE` |
| 11 | MOMENTUM_16BAR_ZERO_CROSS | 0.84% | -616.97元 | -30.77% | -0.13% | 26/21 | -0.24%/-3.73% | `REJECTED_RETROSPECTIVE` |
| 12 | FVG_15M_FORMATION | 0.56% | -674.40元 | -30.86% | -0.79% | 28/23 | -0.26%/-4.08% | `REJECTED_RETROSPECTIVE` |
| 13 | PREVIOUS_CLOSE_ANCHOR_V2 | 0.55% | -675.19元 | -31.53% | -0.39% | 11/16 | -0.70%/-3.26% | `INSUFFICIENT_EVIDENCE` |
| 14 | MOMENTUM_4BAR_ZERO_CROSS | 0.44% | -697.58元 | -31.99% | -0.94% | 29/23 | -1.32%/-2.24% | `REJECTED_RETROSPECTIVE` |
| 15 | ADX_DI_14_CROSS | 0.27% | -731.34元 | -31.86% | -0.67% | 18/16 | -1.38%/-2.37% | `INSUFFICIENT_EVIDENCE` |

## 治理结论

- 历史筛选通过：0条；近似可用：1条。
- 推荐纸面前向观察：`ORB_30M_RVOL_BREAKOUT`。
- 本轮同时比较多条候选，存在多重检验与赢家诅咒；即便历史门槛全过，也只能冻结后向前观察。
- 两万元正式仓位生成器保持不变。
