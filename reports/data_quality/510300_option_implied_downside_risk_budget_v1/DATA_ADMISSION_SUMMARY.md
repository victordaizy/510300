# 510300 期权隐含下行风险预算 V1：数据准入结论

生成时间：2026-09-04 21:00（Asia/Shanghai）

## 结论

```text
FORMAL_STATE=BLOCKED_G0_DATA_ADMISSION
G0_PERMISSION_PROBE=PASS_CORE
ETF_MINS=FAILED_40203_NO_PERMISSION
G0_CLEAN_NEW_PROCESS_REPLAY=PASS
G0_REMAINING_DATA_FAILURE=FUND_DIV_DUPLICATE_PRIMARY_KEY_ROWS_16
G1_OBSERVED_RESULT=FAIL
VALID_SURFACE_DAYS=347/1626
OVERALL_VALID_COVERAGE=21.3407%
RETURN_EVALUATION=NOT_ALLOWED
MODEL_TRAINING=NOT_RUN
PORTFOLIO_EVALUATION=NOT_RUN
POSITION_IMPACT=0
NEXT_ACTION=STOP_CURRENT_FROZEN_BRANCH
```

本轮只完成协议阶段 A、原始数据阶段 B 和数据准入阶段 C。没有生成未来五日方差或尾部标签，没有训练 B0/B1/B2/C1/C2，没有生成仓位、净值、Sharpe、订单或券商动作。

## 权限探针

| 接口 | 结果 | 样本行数 | 备注 |
| --- | ---: | ---: | --- |
| `trade_cal` | 通过 | 9 | 核心 |
| `fund_daily` | 通过 | 7 | 核心 |
| `fund_div` | 通过 | 22 | 接口权限通过，后续全量主键质量未通过 |
| `opt_basic` | 通过 | 12,000 | 精确 `OP510300.SH` 共 2,894 行 |
| `opt_daily` | 通过 | 260 | 2019-12-23 全 SSE 样本 |
| `shibor` | 通过 | 7 | 核心 |
| `etf_mins` | 失败 | 0 | 错误码 40203，无接口权限 |

核心接口权限允许继续预测研究的数据采集，但分钟权限失败意味着最终 09:35—09:39 VWAP 执行回测从一开始就保持阻断。未使用 `stk_mins`、日开盘价或其他来源替代。

## 全量不可变采集

历史范围为 2019-12-23 至 2026-09-04，共 1,626 个上交所开放交易日。

| 数据 | 分区数 | 行数 |
| --- | ---: | ---: |
| `trade_cal` | 1 | 2,448 |
| `fund_daily` | 1 | 1,626 |
| `fund_div` | 1 | 22 |
| `opt_basic` | 1 | 12,000 |
| 每日全 SSE `opt_daily` | 1,626 | 706,806 |
| 分年 `shibor` | 8 | 1,657 |

采集摘要绑定 1,638 个成功分区收据。逐收据重新检查原始响应路径与 SHA-256 后，1,638/1,638 全部匹配；采集摘要自身规范指纹也匹配。任务原始目录共 4,934 个文件、约 0.133 GiB；其中 1,645 份响应和收据包含 7 份权限探针，失败的 `etf_mins` 探针没有派生 Parquet。

## G0 裁决

通过项：

- 1,626 个开放交易日均有且只有对应日期的全 SSE `opt_daily` 分区；
- 没有意外日期、损坏分区或响应哈希漂移；
- `trade_cal`、`fund_daily`、`opt_basic`、全 SSE `opt_daily` 和 `shibor` 的冻结主键没有重复；
- 精确 `opt_code == OP510300.SH` 映射非空，共 2,894 个合约，其中标准乘数 10,000 的合约 2,284 个；
- 第二个干净进程得到同一决策指纹 `ff8d2eb92a99adbe0d8cae9a06813efcadae1a5584ea7ff5ecfb64990c1c796f`。

未通过项：

- `fund_div` 的冻结复合主键有 16 行参与重复，其中 10 行是所有返回字段完全相同的供应商重复记录；其余成对记录在 `net_ex_date` 或 `base_unit` 等字段上存在版本差异。V1 没有预注册修订选择规则，因此不能在看到数据后自行选行或去重；
- `etf_mins` 无权限，最终执行数据覆盖为 0/1,626；该项不阻止期权风险预测数据研究，但阻止最终组合回测。

冻结的自动上交所公告检查曾把 HTTP 200 的 UTF-8 字节按错误编码解释，因而对两个中文字符串产生假阴性。不可变原始字节实际上同时包含 `510300`、`2019年12月23日` 和华泰柏瑞沪深 300 ETF 标的名称；追加更正收据已经记录这一事实，原始文件和冻结裁决均未修改。更正后仍有上述 `fund_div` 主键失败，所以 G0 正式状态不提升。

## G1 曲面裁决

精确映射后共有 195,548 行 510300 期权日数据进入合约有效期与标准乘数检查。最终只有 347/1,626 个交易日形成同时具备 30 日 ATM、30 日 5% 虚值认沽和 60 日 ATM 的合法曲面，覆盖率 21.3407%。

### 日级瀑布

| 日状态 | 天数 |
| --- | ---: |
| `PASS_VALID_SURFACE` | 347 |
| `NO_VIEW_NO_MATURITY_BRACKET` | 1,159 |
| `NO_VIEW_NO_ELIGIBLE_CONTRACTS` | 105 |
| `NO_VIEW_NO_VALID_EXPIRY` | 15 |

### 到期月份瀑布

| 到期状态 | 数量 |
| --- | ---: |
| `PASS_EXPIRY_SURFACE` | 2,983 |
| `NO_VIEW_RATE_TENOR_OUT_OF_RANGE` | 578 |
| `NO_VIEW_NO_PUT5_STRIKE_BRACKET` | 4 |
| `NO_VIEW_INSUFFICIENT_VALID_STRIKES` | 2 |
| `NO_VIEW_NO_ATM_TWO_SIDED_BRACKET` | 2 |

主要失败不是 Black-76 求根、平价离散或行权价插值，而是冻结规则要求 30 日和 60 日都必须由合法到期月份包围，同时禁止期限外推；1,159 个交易日缺少这套完整期限几何。

### 年度覆盖

| 年份 | 完整年份 | 有效日/开放日 | 覆盖率 |
| ---: | :---: | ---: | ---: |
| 2019 | 否 | 7/7 | 100.00% |
| 2020 | 是 | 51/243 | 20.99% |
| 2021 | 是 | 56/243 | 23.05% |
| 2022 | 是 | 56/242 | 23.14% |
| 2023 | 是 | 60/242 | 24.79% |
| 2024 | 是 | 47/242 | 19.42% |
| 2025 | 是 | 36/243 | 14.81% |
| 2026 | 否 | 34/164 | 20.73% |

### 冻结门比较

| G1 门 | 冻结要求 | 实际 | 结果 |
| --- | ---: | ---: | ---: |
| 有效曲面日 | 至少 1,000 | 347 | 失败 |
| 总体覆盖 | 至少 80% | 21.34% | 失败 |
| 每个完整年份覆盖 | 至少 65% | 14.81%—24.79% | 失败 |
| 已准入曲面 30/60 日合法内插 | 必须 | 通过 | 通过 |
| 已准入曲面无套利与平价 | 必须 | 通过 | 通过 |
| 缺失保持 `NO_VIEW` | 必须 | 通过 | 通过 |

G1 的三个覆盖门均大幅失败。这个结果不证明期权风险信息无用；它证明按本次冻结的数据、利率期限和严格 30/60 日期限包围契约，无法获得足够连续的历史曲面去合法检验该模型。

## 停止边界

本分支不允许进入阶段 D，因此：

- 不构造 `V5` 或 `TAIL5`；
- 不读取未来标签分布或组合收益；
- 不拟合风险模型；
- 不计算 Sharpe 或尝试改变期限、利率外推、结算价、行权价目标、阈值、成本或仓位规则；
- 不用相似数据补缺，不用结果后参数调整营救。

准确状态是 `BLOCKED_G0_DATA_ADMISSION`，并附带一个可重放的 `G1_OBSERVED_FAIL`。两者都足以阻止任何后续预测、组合和交易解释。

## 主要证据文件

- `permission_probe.json`：逐接口权限与原始响应哈希；
- `acquisition_summary.json`：1,638 个成功采集分区的绑定清单；
- `G0_G1_ADJUDICATION.json`：冻结实现的正式裁决与重放指纹；
- `sse_underlying_cross_check_encoding_correction.json`：公告 UTF-8 假阴性的追加更正；
- `510300_option_contract_map.parquet`：精确合约映射；
- `510300_option_surface_quality_ledger.parquet`：1,626 日 `PASS/NO_VIEW` 账本；
- `510300_option_surface_daily.parquet`：347 个合法曲面日。

