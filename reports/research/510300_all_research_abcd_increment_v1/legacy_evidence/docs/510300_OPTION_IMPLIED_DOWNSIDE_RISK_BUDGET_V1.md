# 510300 期权隐含下行风险预算策略 V1

## 1. 冻结声明

```text
MODEL_ID=510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1
VERSION=1.0.0
FREEZE_STATE=PROTOCOL_FROZEN_BEFORE_FIRST_API_PROBE
RESEARCH_STAGE=DATA_ADMISSION_ONLY
HISTORICAL_LABEL=RESEARCH_OBSERVED_HISTORICAL
VALIDATED_ALPHA=FALSE
LIVE_TRADING_AUTHORIZED=FALSE
POSITION_IMPACT=0
TRADABLE_UNIVERSE=510300.SH+CASH_CNY
OPTIONS=SIGNAL_ONLY_NOT_TRADABLE
```

本文件和同名 YAML 在第一次带凭据的 API 探针、任何新期权曲面统计、未来标签以及组合收益读取之前冻结。YAML 是机器可读的唯一参数真源；本文件解释研究边界和停止规则。任何冻结哈希漂移都必须停止当前版本，不得原地修补后继续。

本研究不能承诺高夏普。它只检验一个可证伪命题：510300 期权结算价曲面是否能够在严格前序条件下，对未来五个交易日实现风险提供价格历史之外的稳定增量，并且该增量在真实 ETF 执行约束下是否足以改善风险预算组合。

## 2. 与旧分支的边界

本版本不是旧期权成交量、持仓量、买卖方向、订单簿、衍生品压力或 BAD10 分支的救援。旧冻结文件、旧结果和旧阈值不修改。V1：

- 不交易期权；
- 不用期权成交量或持仓量预测方向；
- 不使用 PCR；
- 不预测 510300 收益正负；
- 不把稀有危机事件设为唯一目标；
- 只用期权结算价曲面预测未来五日实现方差，尾部概率只是预注册的可选仓位上限；
- 唯一可交易资产是 `510300.SH` 和 `CASH_CNY`；
- 当前仅有研究与数据采集权限，没有 Paper、Shadow、券商连接、持仓修改、订单或实盘权限。

## 3. 数据与时间契约

历史起点固定为 2019-12-23。终点按所有强制日频数据共同可得的最后一个已完成上交所交易日确定。日内数据、当日 `opt_daily`、当日 `opt_basic` 和当日 SHIBOR 必须在 `20:15 Asia/Shanghai` 之后才可共同形成该日信号。

强制接口与用途：

| 接口 | 主要字段 | 用途 |
| --- | --- | --- |
| `trade_cal` | `cal_date,is_open,pretrade_date` | 交易日、成熟日、执行日 |
| `fund_daily` | `trade_date,open,high,low,close,pre_close,vol,amount` | 510300 风险特征和总财富底表 |
| `fund_div` | 全部默认字段 | 现金分红和未复权总财富 |
| `opt_basic` | 合约、标的、认购认沽、行权价、乘数、挂牌和到期字段 | 合约映射与期限结构 |
| `opt_daily` | `close,settle,vol,amount,oi` 等默认字段 | 期权曲面 |
| `shibor` | `date,1w,2w,1m,3m` | 贴现率代理 |
| `etf_mins` | `trade_time,open,high,low,close,vol,amount` | 次日 09:35—09:39 最终执行价 |

不得用近似接口替代缺失强制输入。特别是 `etf_mins` 失败时，不得把 `stk_mins`、日开盘价或其他分钟源冒充最终执行数据；此时仅允许继续预测研究，最终执行回测保持阻断。

同日 SHIBOR 缺失即 `NO_VIEW`，不得前值填充。1 周、2 周、1 个月和 3 个月按日历天线性插值，内部年化利率单位为小数。7—90 日节点外不外推，因此相关到期月份记为 `NO_VIEW_RATE_TENOR_OUT_OF_RANGE`。

## 4. 临时凭据与代理契约

代理教程已于 2026-09-04 读取。允许的 HTTPS 根地址只有：

- `https://fast.xiaodefa.cn`
- `https://tt.xiaodefa.cn`

只读根路径预检中 `tt.xiaodefa.cn` 延迟更低，因此 V1 冻结使用该地址。API 采用根路径 POST、`x-api-key` 请求头、`Accept-Encoding: gzip`。请求起点之间至少间隔 0.65 秒。凭据只允许来自隐藏交互输入或进程环境，不落盘，不进入日志、原始响应、收据、报告、清单和 Git。

## 5. G0 权限探针

全量下载前逐项执行附件指定的最小请求，并记录：接口名、脱敏参数、请求和响应时间、是否成功、行数、返回列、错误代码、脱敏错误、原始响应 SHA-256 和端点来源。

裁决：

```text
trade_cal、fund_daily、fund_div、opt_basic、opt_daily、shibor 任一失败
=> CORE_STRATEGY_BLOCKED

etf_mins 失败
=> PREDICTION_RESEARCH_ALLOWED
   FINAL_EXECUTION_BACKTEST_BLOCKED
```

探针成功只证明当前凭据和参数能够返回样本，不证明全历史连续，不证明合约映射正确，不证明 G1，不证明预测能力，更不证明可交易。

## 6. 不可变原始采集

下载顺序固定为：

1. `trade_cal`；
2. `fund_daily`；
3. `fund_div`；
4. `opt_basic`；
5. 对每个开放交易日下载整个 SSE `opt_daily`；
6. 分年下载 `shibor`；
7. 分自然月下载 510300 `etf_mins`。

每天的 `opt_daily` 必须先保存上交所全量原始响应，之后才允许依据 `opt_basic` 筛选 510300。不得先按名称或代码把供应商响应缩成 510300 子集。Tushare 文档注明 `opt_daily` 单次上限 15,000 行；任何响应达到或超过该上限都按可能截断阻断，不得当作完整日链。

原始响应以 API、请求参数和检索时间组织，只追加、不覆盖。每个响应都有原始字节哈希和独立收据。标准化 Parquet 是派生缓存，不能替代原始响应证据。中断后只可跳过已经存在、哈希仍匹配且收据完整的请求。

## 7. 510300 合约映射

上交所公告确认：沪深 300 ETF 期权于 2019-12-23 上市，合约标的是华泰柏瑞沪深 300 ETF，证券代码 510300。

Tushare V1 映射必须同时满足：

```text
exchange == SSE
opt_code == OP510300.SH
list_date <= trade_date <= delist_date
call_put in {C, P}
exercise_price > 0
maturity_date 有效
```

禁止名称模糊匹配。所有调整合约保留在原始数据和质量账本中；主曲面只使用 `opt_multiplier == 10000` 的标准合约。Tushare 映射与上交所标的公告必须生成单独交叉核对收据。

## 8. 曲面算法

单合约日先要求：`settle > 0`、`vol > 0`、`oi > 0`、`5 <= DTE <= 120`、主键唯一且价格通过冻结的无套利边界。主价格永远是 `settle`；`close` 只作质量对照。

认购认沽按 `trade_date,maturity_date,exercise_price,opt_multiplier` 配对。每个到期月份选取距离 510300 当日收盘价最近的最多五组，至少三组。由平价关系计算每个行权价的远期：

\[
F_{K,T}=K+e^{r(T)T}(C_{K,T}-P_{K,T})
\]

到期月份远期取中位数。若 `MAD(F) / median(F) > 0.5%`，到期月份为 `NO_VIEW_PARITY_DISPERSION`。

对数价内外程度为 `k = ln(K/F)`。`k < 0` 使用认沽，`k >= 0` 使用认购，以 Black-76 和有界 Brent 求根反解隐含波动率。只接受：

- `-0.12 <= k <= 0.12`；
- `1% <= IV <= 200%`；
- 至少五个有效行权价；
- `k=0` 两侧都有行权价；
- `k=-0.05` 位于有效相邻行权价内部。

先按行权价对总方差 `w(k,T)=IV(k,T)^2*T` 线性内插，得到每个到期月份的 ATM 和 5% 虚值认沽 IV。再用合法包围目标期限的两个到期月份对总方差线性内插，构造 `IV30_ATM`、`IV30_PUT5` 和 `IV60_ATM`。所有外推禁止；缺少 30 日或 60 日期限包围时为 `NO_VIEW_NO_MATURITY_BRACKET`。

## 9. 冻结特征与标签

B1 只有五项价格风险特征：

```text
LOG_RV5
LOG_RV20
NEG_RETURN5
DRAWDOWN20_RISK
DOWNSIDE_SHARE20
```

B2 只增加：

```text
LOG_IV30
PUT_SKEW30
TERM_INVERSION30_60
IV_SHOCK5
SKEW_SHOCK5
```

总财富收益只使用未复权 510300 价格加实际现金分红。禁止技术指标、估值择时、宏观水平、新闻、资金流、成分股、期货、510050 期权、PCF/IOPV，以及根据收益筛选特征。

信号日 `t` 的执行价固定为下一交易日 09:35:00—09:39:59 一分钟成交额加权 VWAP。主标签 `V5` 是从该执行价开始的未来五交易日年化实现方差，并仅在第五个未来交易日收盘后成熟。尾部标签 `TAIL5` 固定为五日路径相对执行价最低跌幅不高于 -3%；连续重叠正标签合并为事件。独立事件少于 40 或四个时代任一少于 5 个事件时，尾部模块为 `NO_VIEW`，不改变主方差模型。

## 10. 模型与仓位

B0 是仅用已成熟训练样本的无条件 `V5` 均值。B1、B2 都使用指数链接、所有斜率非负、`L2=1.0` 的 QLIKE Ridge。标准化只能使用当时训练样本。训练窗口扩展；至少 504 个有效成熟日；每月第一个交易日重估；月内系数固定；标签至少滞后五个交易日。

禁止 lambda 网格、特征筛选、符号翻转、随机森林、XGBoost、神经网络和按组合 Sharpe 选模型。

波动仓位为 `min(1, sigma_ref / sigma_hat)`，其中参考风险是当时已成熟样本 `sqrt(V5)` 的中位数。通过的尾部模型可给出 `min(1, p_base / p_hat)` 上限；未通过则尾部权重固定为 1。最终原始权重取两者最小值并向下映射为 `0/25/50/75/100%`。目标下降下一交易日一次到位，目标上升每天最多增加 25 个百分点。

## 11. 执行与成本

账户、成本和交易机制固定：

```text
INITIAL_CAPITAL=200000 CNY
COMMISSION=max(notional*0.0002,5)
LOT_SIZE=100 shares
T_PLUS_1=TRUE
CASH_RETURN=0
BASE_SLIPPAGE=5 bp/leg
STRESS_SLIPPAGE=10 bp/leg
EXECUTION=NEXT_DAY_09_35_TO_09_39_VWAP
```

不允许融资、做空、期权交易、日内 T、止损止盈、同日收盘成交、理论中间价、非整数手或忽略 T+1 库存批次。

## 12. 门槛与停止条件

G0 要求核心接口可用、510300 标的映射经官方核对、原始文件哈希完整、主键无重复、干净新进程可重放。最终组合还要求 `etf_mins` 可用且 09:35—09:39 覆盖率至少 98%。

G1 要求至少 1,000 个有效曲面日、总体覆盖至少 80%、每个完整年份至少 65%、30 日和 60 日都合法内插、无套利和平价门通过且缺失保持 `NO_VIEW`。G1 失败立即停止。

G2 要求 B1 相对 B0 前序 QLIKE 改善为正、至少 3/4 时代为正、块 Bootstrap 单侧 90% 下界为正。

G3 要求 B2 相对 B1：QLIKE 改善至少 5%、对数方差 MSE 改善为正、至少 3/4 时代 QLIKE 改善为正、块 Bootstrap 单侧 90% 下界为正、至少 750 个前序预测日、校准斜率 0.75—1.25、平均预测/实际方差 0.85—1.15。G3 失败即 `MODEL_STATE_REJECTED_FROZEN_NO_RESCUE`，组合评价不允许。

G4 尾部模块要求 C2 相对 C1 的 Brier 和 Log Loss 均至少改善 2%，至少 3/4 时代同时改善，事件块 Bootstrap 单侧 90% 下界为正，校准斜率 0.70—1.30。失败只禁用尾部上限，不改主模型。

G5 只能在 G0—G3 全部通过后一次性运行。比较买入持有、同平均暴露常数基准、B1 和 B2。全部正式门精确记录在 YAML，包括基础成本净 Sharpe 至少 1.20、B2 比 B1 至少高 0.20、回撤降低至少 30%、净 CAGR 至少 4%、上涨/下跌捕获、平均仓位、换手、四时代稳定性、压力过程集中度和 10bp/边压力滑点。Deflated Sharpe 的试验次数必须覆盖整个 510300 项目累计试验，而非只计本次。

## 13. 阶段状态机

```text
A 冻结协议，不读新曲面统计、未来标签或组合收益
B 下载原始数据，不计算收益
C 只做 G0/G1 数据准入；失败即停止
D 仅在 G0/G1 通过后构造标签，并先冻结时代边界
E 只做 B0/B1/B2/C1/C2 前序预测，不生成净值或 Sharpe
F 仅按预测门冻结最终模型
G 仅在 G0—G3 通过后执行一次组合裁决
H 历史通过后也只能进入 SHADOW_ONLY_FORWARD，252/504 个全新交易日
```

当前授权只覆盖 A、B 和 C。任何阶段完成都不会自动提升下一阶段权限。

## 14. 永久禁止的结果后救援

读取结果后，不得改变五日目标、-3% 尾部阈值、30 日 IV、5% 虚值认沽、0.5% 平价离散门、结算价字段、09:35—09:39 执行窗口、25% 仓位档位、恢复速度、成本或 lambda；不得加入 PCR/OI/成交量方向、技术/估值过滤、优选年份/方向、510050 历史或其他资产。核心门失败时状态为 `HISTORICAL_REJECTED_FROZEN_NO_RESCUE`。

## 15. 当前准确状态

```text
CURRENT_ALLOWED_ACTIONS=FREEZE_PROTOCOL,PERMISSION_PROBE,RAW_DOWNLOAD,G0_G1_DATA_ADMISSION
RETURN_EVALUATION=NOT_ALLOWED
MODEL_TRAINING=NOT_ALLOWED_BEFORE_G0_G1_PASS_AND_SEPARATE_PHASE_ENTRY
PORTFOLIO_EVALUATION=NOT_ALLOWED
POSITION=ABSTAIN_UNSET
ORDER=NOT_ALLOWED
BROKER_CONNECTION=NOT_ALLOWED
LIVE_TRADING=NOT_AUTHORIZED
```
