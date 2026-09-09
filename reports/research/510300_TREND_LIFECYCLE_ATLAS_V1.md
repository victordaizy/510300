# 510300 趋势生命周期图谱 V1

- 最终状态：`ATLAS_COMPLETE_NO_DUAL_LABEL_LEADING_STRUCTURE_NO_ROUTER_PROMOTION_NO_RESCUE`
- 数据合同：`PASS`
- 证据等级：回顾性、历史污染的双标签事件图谱；不是策略回测。
- 历史权重来源门：`BLOCKED_NO_VERSION_PROVEN_STRICT_PIT_OFFICIAL_ARCHIVE`。
- 收益评价：`NOT_ALLOWED_PHASE_1_EVENT_LABEL_DIAGNOSTICS_ONLY`。

## 双标签事件样本

| 标签 | 转移 | 全部事件 | 完整[-120,+60]事件 |
|---|---|---:|---:|
| `LABEL_A_DIRECTIONAL_CHANGE` | `DOWN_END` | 12 | 12 |
| `LABEL_A_DIRECTIONAL_CHANGE` | `DOWN_START` | 13 | 12 |
| `LABEL_A_DIRECTIONAL_CHANGE` | `UP_END` | 13 | 12 |
| `LABEL_A_DIRECTIONAL_CHANGE` | `UP_START` | 12 | 12 |
| `LABEL_B_FUTURE_PATH` | `DOWN_END` | 5 | 5 |
| `LABEL_B_FUTURE_PATH` | `DOWN_START` | 5 | 5 |
| `LABEL_B_FUTURE_PATH` | `RANGE_END` | 27 | 26 |
| `LABEL_B_FUTURE_PATH` | `RANGE_START` | 27 | 27 |
| `LABEL_B_FUTURE_PATH` | `UP_END` | 4 | 4 |
| `LABEL_B_FUTURE_PATH` | `UP_START` | 4 | 4 |

## 双标签领先结构门

| 转移 | 双标签变量数 | 机制模块数 | 通过 |
|---|---:|---:|---:|
| `UP_START` | 0 | 0 | 否 |
| `UP_END` | 0 | 0 | 否 |
| `DOWN_START` | 0 | 0 | 否 |

## 通过单标签门的主要变量

| 标签 | 转移 | 变量 | 模块 | 事件数 | 方向一致率 | 事前中位Z | 中位领先日 |
|---|---|---|---|---:|---:|---:|---:|
| 无 | 无 | 无变量通过全部冻结条件 | 无 | 0 | NA | NA | NA |

## 七状态事后参考图

| 参考状态 | 交易日 |
|---|---:|
| `X_TRANSITION_NO_VIEW` | 675 |
| `R_LOW_STRESS_RANGE` | 268 |
| `U2_FRAGILE_LATE_UP` | 241 |
| `U0_EARLY_UP` | 206 |
| `D0_EARLY_DOWN` | 142 |
| `D1_MARKDOWN_PANIC` | 53 |
| `U1_BROAD_UPTREND` | 24 |

## 裁决边界

- 双标签结构门：`FAIL`。
- 历史成员/权重严格点时来源门：`FAIL`；当前结果不得直接进入路由器冻结。
- H00300仅用于标签；未来若有策略，510300未复权执行、分红现金和T+1时钟仍需在新协议中另行冻结。
- 未计算策略净值、净夏普、UP_CAPTURE、DOWN_CAPTURE、最大回撤或RANGE_T净边际。
- 未生成当前状态概率、当前因子、Paper、Shadow、仓位、订单、券商连接或实盘授权。
