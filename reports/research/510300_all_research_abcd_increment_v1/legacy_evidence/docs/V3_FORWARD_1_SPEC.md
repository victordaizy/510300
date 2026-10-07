# V3_FORWARD_1前瞻验证规范

## 目的与边界

`V3_FORWARD_1`自2026-08-13起进行真正样本外验证。它验证估值Edge是否能预测沪深300未来超额总收益，不授权真实订单，也不替代现有510300实盘范围。

历史V3结果仅用于形成本次预注册。任何冻结文件、参数、状态定义或缺失值规则发生变化，都必须创建新版本，禁止覆盖`V3_FORWARD_1`。

## 冻结对象

- 正常化EPS算法；
- FairPE算法；
- 年化均值回归速度`λ=0.3374309932944419`；
- Dividend、Cash与Edge公式；
- 20、60、120、242交易日四个期限；
- 盈利、市场趋势、波动三个状态的定义；
- 数据可得时间、权重、缺失值和跳过规则；
- 60日Edge到Shadow仓位的固定映射。

机器可读参数位于`config/v3_forward_1.yaml`，冻结文件SHA-256位于`config/v3_forward_1_manifest.json`。每日运行前必须验证全部指纹。

## 预注册状态假设

仅允许检验以下三个方向：

- H1：`IC(盈利上行) > IC(盈利下行)`；
- H2：`IC(熊市) > IC(牛市)`；
- H3：`IC(低波动) > IC(高波动)`。

第一轮结束前禁止新增利率、北向、信用、PMI、ERP、市场宽度或流动性等状态分类，也不使用`λ(State)`。

## 不可变前瞻记录

每个交易日收盘后，数据刷新成功且核心行情共同覆盖当日时，程序向以下文件追加一次：

- `data/forward/v3_forward_1/valuation_forward_signal_log.parquet`；
- `data/forward/v3_forward_1/valuation_forward_outcome_log.parquet`；
- `data/forward/v3_forward_1/valuation_shadow_position_log.parquet`。

信号日志按日期递增，并使用前序哈希和逐行内容哈希形成链。相同日期只接受完全相同的幂等重试，禁止回填、修改和覆盖。成分股当日收盘快照独立保存在`data/raw/forward/v3_forward_1_constituent_close.parquet`。

未来结果只有在目标交易日真实出现后才能填入；一旦解析，后续供应商修订不会覆盖已锁定结果。信号日没有共同收盘数据、当日成分覆盖不足或财务/宏观刷新失败时，程序跳过信号并记录状态。

## 监控与Gate

每20个新交易日信号生成一次报告。20日优先级高，60日最高，120日中，242日仅作长期经济逻辑辅助。

在至少有120个已兑现60日样本、且精确非重叠队列至少有4个观测前，状态只能是`INSUFFICIENT_DATA`。60日主Gate使用冻结配置中的Rank IC、HAC、区块Bootstrap、Top-Bottom、滚动窗口、前后半段和剔除单个20日区块后的稳定性门槛。

允许的报告状态为`MONITORING`、`PASS`、`FAIL`和`INSUFFICIENT_DATA`。任何单次监控结果都不得反向修改模型。

## Shadow仓位

60日Edge固定映射为：

| 60日Edge | Shadow仓位 |
|---|---:|
| ≤ -10% | 0% |
| (-10%, -5%] | 25% |
| (-5%, +5%] | 50% |
| (+5%, +10%] | 75% |
| > +10% | 100% |

Shadow目标在信号日后下一交易日开盘视为生效，只用于计算假想持仓与未来PnL。程序中没有真实下单路径。

## 日运行

Windows日终入口为：

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\run_v3_forward_daily.py'
```

程序先刷新必要输入，再生成信号、Shadow记录、到期结果和按期监控报告。最新运行状态写入`paper/v3_forward_1_latest_status.json`，输入刷新状态写入`paper/v3_forward_1_input_refresh_status.json`。

权重和点时财务刷新需要在被Git忽略的`.env`中配置`TUSHARE_TOKEN`。仅当现有权重与财务文件的`retrieved_at`已经等于目标日时，程序才允许在当前进程无凭据的情况下复用；否则日运行失败并禁止生成信号，避免静默使用陈旧财务。
