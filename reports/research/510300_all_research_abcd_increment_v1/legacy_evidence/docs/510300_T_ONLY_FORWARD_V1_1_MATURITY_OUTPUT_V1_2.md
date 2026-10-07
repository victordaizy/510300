# 510300 T-only V1.1 成熟度输出 V1.2

## 裁决

T-only V1.1 继续按原冻结协议自动采集。达到 252 个新交易日且闭合 3 个周期前，人工可见交付物只包含：

- 新交易日数量；
- 闭合周期数量；
- 数据、账本与本次运行完整性；
- `NO_VIEW_UNTIL_FORWARD_MATURITY`。

不发布中途收益、信号、目标仓位、门槛布尔值或“暂时有效/无效”判断。

## 版本边界

V1.2 是运维输出修复，不是研究版本升级。以下内容保持 V1.1 冻结哈希与定义：

- 策略参数、成本模型与仓位目标；
- 252 个新交易日、3 个闭合周期的成熟门槛；
- 交易日历、周线因果修复和前瞻起点；
- 行情刷新、影子账本、执行事件与闭合周期计算。

冻结 V1.1 计算仍先生成内部计算结果；V1.2 阶段在同一任务内将人工可见状态和每日操作卡原子替换为成熟度口径。下一次日更开始时，冻结计算会重新生成所需内部结果，因此不会用旧状态替代本次计算。

## 日更入口

计划任务 `Codex-510300-T-Only-Forward-V1` 固定在工作日 16:30 运行：

```text
.venv\Scripts\python.exe scripts\run_t_only_forward_v1_daily_v1_2.py \
  --manifest config\t_only_forward_v1_1_maturity_only_v1_2_manifest.json \
  --expected-manifest-sha256 <清单文件SHA-256>
```

入口先校验清单及受控文件哈希，再依次运行行情刷新和成熟度口径前瞻阶段。计划任务禁止错过时间后补跑，禁止自动重试，且忽略并发重复实例。

## 权威产物

- `reports/forward/t_only_forward_v1_1_status.json`
- `reports/forward/t_only_forward_v1_1_status.md`
- `reports/forward/t_only_forward_v1_1_daily_guide.json`
- `reports/forward/t_only_forward_v1_1_daily_guide.md`
- `paper/t_only_forward_v1_1/daily_run_status_v1_2.json`
- `paper/t_only_forward_v1_1/daily_run_receipts_v1_2/<run_id>.json`

单次运行失败时，权威公开状态改为 `FAILED` 与 `NO_VIEW_OPERATIONAL_FAILURE`，并明确区分 `PROGRAM_FAILED`、`DATA_GATE_FAILED` 和 `EXTERNAL_FREE_SOURCE_FAILED`；不会把上一日状态登记成本次成功。
