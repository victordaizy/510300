# 510300图形状态马丁-海龟V2冻结历史检验

- 数据截止：2026-08-14
- 证据标签：HISTORICALLY_CONTAMINATED
- 背离事件：底背离 8 个，顶背离 7 个
- 注册候选通过数：0 / 5
- 安全边界：仅研究/纸面账本；真实仓位映射、订单生成、券商连接均关闭。

| 轨道 | 决策 | 闭合交易 | CAGR | 最大回撤 | Sharpe | 平均仓位 | 时机贡献 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0_DONCHIAN_20_10 | FAIL | 55 | 4.36% | -24.67% | 0.275 | 39.02% | 22.99% |
| T1_TURTLE_GRAPH_EXIT | FAIL | 56 | 5.04% | -23.15% | 0.330 | 37.56% | 42.06% |
| M0_CAPPED_MARTINGALE_ONLY | INSUFFICIENT_EVIDENCE | 8 | 1.52% | -0.16% | 0.113 | 0.03% | 0.26% |
| P0_BIAS28_SWING_ONLY | INSUFFICIENT_EVIDENCE | 0 | 1.51% | 0.00% | N/A | 0.00% | 0.00% |
| S1_MARTINGALE_TURTLE_SWITCH | FAIL | 63 | 5.06% | -23.11% | 0.331 | 37.57% | 42.51% |
| S2_BIAS28_TURTLE_SWITCH | FAIL | 56 | 5.04% | -23.15% | 0.330 | 37.56% | 42.06% |

## 基准

- 510300含分红买入持有：CAGR 5.64%，最大回撤 -44.24%，Sharpe 0.298。
- H00300全收益指数：CAGR 6.63%，Sharpe 0.339。

## 判定纪律

任何失败轨道都按 `REJECT_WITHOUT_PARAMETER_RESCUE` 处理；交易不足30次则为 `INSUFFICIENT_EVIDENCE`。本报告不构成真实交易指令。
