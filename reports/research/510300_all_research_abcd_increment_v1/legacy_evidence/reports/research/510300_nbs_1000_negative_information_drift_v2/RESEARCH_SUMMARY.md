# 510300 历史研究关闭及严格前向交接

裁决日期：2026-09-05。DSV5 仓位政策已归档；B1 保留为只读预测基准。NBS V2 在 G0 分钟 VWAP 硬门失败后终止，510300 后续主动研究切换为 `STRICT_FORWARD_ONLY`。

| 对象 | 当前状态 |
| --- | --- |
| DSV5 政策 V1 | `ARCHIVED_NO_VIEW_INSUFFICIENT_POLICY_VARIATION` |
| B1 价格风险预测 | `RETAIN_AS_FORECAST_BENCHMARK_ONLY` |
| B1 仓位映射 | `DISABLED` |
| 原 dsv5-g1 自动任务 | `PAUSED` |
| NBS V1 | `ARCHIVED_BLOCKED_BEFORE_RETURN_READ` |
| NBS V2 | `ARCHIVED_BLOCKED_G0_BEFORE_RETURN_READ` |
| 后续 510300 历史 Alpha 实验 | `NOT_ALLOWED` |
| 当前已验证高 Sharpe 策略 | `NONE` |
| 仓位影响 / 实盘授权 | `0 / false` |

## NBS V2 来源与无收益账本

| 检查 | 实测 | 冻结要求 | 结果 |
| --- | ---: | ---: | --- |
| 标准 241 根开放日 | 2,350 / 2,350 | ≥99.9% | 通过 |
| 合格事件四窗口完整 | 88 / 88 | 100% | 通过 |
| 重复主键 / 非法时刻 | 0 / 0 | 0 / 0 | 通过 |
| 日成交量最大相对误差 | 5.0602×10⁻⁹ | ≤0.001 | 通过 |
| 日成交额最大相对误差 | 9.4360×10⁻⁹ | ≤0.001 | 通过 |
| 独立来源四窗口 VWAP | 4,844 / 4,844 | ≥99.9% | 通过 |
| BAR_END 文档及既有证据 | 完整 | 必须具备 | 通过 |
| 逐根 VWAP 超出 low/high ±0.001 元 | **263 根** | **0 根** | **失败** |
| 零量、零额导致 VWAP 无法定义 | 232 根 | 不填补为价格 | 记录缺失 |
| 训练及评价时代切分 | 36 / 17 / 17 / 16 | 36 / 17 / 17 / 16 | 通过 |

复核使用已有 566,350 根分钟，没有重新下载、更换来源、扩大误差容限或删除异常柱。263 根越界均为有成交量记录；232 根无法定义的记录全部为零量且零额。最大越出 high/low 边界约 0.016947 元。四类关键时钟窗口中合计 15 根越界，88 个合格事件当天四窗口越界为 0；冻结合同要求逐根全部满足，因此保留全量来源失败裁决。该结论不证明数据一定错误，也不构成 Alpha 统计失败结论。

日线 high、low、close 与分钟聚合的差异仅保留为诊断，分别有 47、52、54 个交易日超过 0.001 元，未用于 V2 硬门。13 个无法证明事前日程版本的事件继续排除。合格事件 88 个，其中 2 个缺少足够的事前基线，86 个模型候选事件的无收益账本已经冻结。

用户明确授权补齐的安慰剂 B 已写入 V2：四个真实区间为 09:40—09:45、09:45—09:50、09:51—09:56、14:35—14:40；主实验窗口不变。来源门失败后未计算任一安慰剂的收益关系。

## 停止回执

G0 失败；G1 无收益切分通过；G2、G3、G4 均为 `NOT_RUN_BLOCKED_BY_G0`。事件目标 Y 没有构造，NBS B0/B1 没有训练，42bp 信号密度没有计算，20 万元账户没有运行，Sharpe 没有计算。

来源合同和代码先提交为 `326e345c9673246d5c6dbbab7912de1cb5dbcade`，随后执行测量。完整 V2 协议、账本和 B1 观察器冻结提交为 `0e571f73b5d00164d69e6cb1ca15c32a8ca63966`，其后创建终止尝试 claim。claim 状态为 `ONE_SHOT_ATTEMPT_TERMINATED_BEFORE_RETURN_READ`：登记了终止尝试，收益读取权从未消耗，重新启动为 false。

V1 文件均保留，归档决定采用新增文件。DSV5 旧政策保存清单为 `reports/research/510300_dsv5_policy_archive_20260905_preservation.json`；NBS 来源全部原始与规范化文件身份保存于 `config/510300_stk_mins_source_admission_v2_manifest.json`。Git 冻结采用原始字节保存；重建目录时应使用 `core.autocrlf=false`，防止 Windows 自动换行改变冻结哈希。

## B1 观察器与定时任务

独立自动任务 `510300_B1_DSV5_FORECAST_OBSERVATORY_V1` 已创建并启用，ID 为 `510300-b1-dsv5-forecast-observatory-v1`，在当前任务内于工作日北京时间 19:30 跟进。原 `dsv5-g1` 保持暂停，提示词已改为归档后禁止再调用政策记录入口。

观察器仅在固定五交易日原点记录冻结的 13 个预测字段，到期后再写实际 DSV5 与 QLIKE。首次新原点为 2026-09-11。今天实际运行得到 `NOT_ORIGIN_NOOP`，严格前向预测为 0 条。历史漏跑不得补填，只在 52、104 个新原点评价；失败则归档观察器。

当前输入准备仍有明确边界：官方分红覆盖证明仅到 2026-08-14；官方日历覆盖到 2026-12-31。合法原点若没有更新后的官方覆盖证明，返回 `NO_VIEW_STALE_OFFICIAL_DIVIDEND_COVERAGE`；跨年度若日历没有更新，则返回 `NO_VIEW_OFFICIAL_CALENDAR_EXTENSION_REQUIRED`。不会自动延长证明日期，或把登记、任务启用解释为已经产生有效前向预测。

## 验证与入口

14 项针对性测试通过。来源测试覆盖成交总量、逐根 VWAP、重复分钟和固定时代；观察器使用合成数据完整执行风险拟合、首次预测、五日标签成熟、损失评分和重复触发，验证原始预测不被到期结果覆盖。真实入口在当前非原点日通过。测试未使用 NBS 事件收益。

- 最终状态：`reports/research/510300_nbs_1000_negative_information_drift_v2/status.json`
- 来源数值回执：`reports/data_quality/510300_stk_mins_source_admission_v2/source_adjudication.json`
- 逐根异常：`reports/data_quality/510300_stk_mins_source_admission_v2/bar_vwap_defects.csv`
- 无收益事件账本：`data/curated/510300_nbs_1000_negative_information_drift_v2/event_ledger_pre_return.csv`
- 一次性终止 claim：`reports/research/510300_nbs_1000_negative_information_drift_v2/one_shot_claim.json`
- 后续研究停止线：`config/510300_historical_alpha_stop_gate_20260905.json`
- B1 运行说明：`docs/510300_B1_DSV5_FORECAST_OBSERVATORY_V1.md`

后续仅保留 PCF/IOPV、A50 盘后信息时差、宏观首次发布 vintage、B1 风险观察及实际券商执行成本等授权范围内的严格前向研究。PCF/IOPV 既有任务仍为 ACTIVE；该状态不证明任何新增采集成功。
