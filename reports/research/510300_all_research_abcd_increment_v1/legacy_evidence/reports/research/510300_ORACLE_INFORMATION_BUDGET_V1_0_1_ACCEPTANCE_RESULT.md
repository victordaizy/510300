# 510300 Oracle 信息预算 V1.0.1 保守验收

- 最终状态：`ORACLE_VALUE_NOT_REALISTICALLY_ACCESSIBLE_STOP_UP20_FORECAST_BRANCH`
- 证据等级：已知原V1结果后的保守合同修正；不是盲测、不是现实预测器回测。
- 完美Oracle普通日频夏普：`1.740`。
- 完美Oracle保守夏普（三种调整口径最小值）：`1.278`。

## 八个冻结预算字段

| 字段 | 随机95%稳健 | 对抗/保守 |
|---|---:|---:|
| `MIN_UP_RECALL_FOR_SHARPE_1_2` | 95.7% | 73.9% |
| `MIN_UP_PRECISION_FOR_SHARPE_1_2` | 100.0% | 100.0% |
| `MAX_DOWN_FALSE_LONG_RATE` | 0.0% | 0.0% |
| `MAX_RANGE_FALSE_LONG_RATE` | 3.1% | 1.0% |
| `MAX_ENTRY_DELAY_DAYS` | — | 1 |
| `MAX_EXIT_DELAY_DAYS` | — | >=10 |
| `MIN_ORACLE_RETURN_CAPTURE` | — | 54.9% |
| `SHARPE_1_2_FEASIBLE_REGION` | 1/216格 | 交集1/216格 |

精确率必须与同一单元的召回率、震荡误入率和大跌误入率共同读取。退出延迟若显示`>=10`，含义是冻结网格内右删失，不是已识别精确上限。

## 裁决

- 非退化保守可行区域：`FALSE`。
- UP20协议资格：`STOPPED_ORACLE_VALUE_NOT_REALISTICALLY_ACCESSIBLE`。
- 现实UP20预测器：`NOT_EVALUATED`。
- 仓位输出：`NONE`；实盘授权：`FALSE`。
