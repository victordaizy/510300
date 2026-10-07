# 项目研究状态

> 状态快照：2026-10-02，Asia/Shanghai。依据当前工作区、Git 历史、代码、冻结协议和已保存结果编写。
> 本文件与 [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md) 是模型切换后的首读入口。原始结果与冻结协议仍是数值和实验定义的直接证据。
> 本次只整理事实和后续实验设计，没有启动新策略试验、更新市场数据、修改原策略、提交 Git 或执行交易。

## 1. 当前结论

1. 当前主线是 **510300 日线及此前已完成周线、多头优先、提高完整账户扣费收益和夏普，同时控制过拟合**。增加有效交易次数是软目标，不再要求每年必须五次。
2. **目标尚未完成；没有可以宣布已经去除过拟合、经过独立验证的高夏普策略。** 当前可比较的 A、B 是历史开发候选，不是实盘建议。
3. 近期主候选 A 保留仓位强弱后，在共同压力成本账户中净年化约 **3.99%**、净夏普 **1.217**、最大回撤 **2.75%**、实际净胜率乘盈亏比 **1.073**。但 2015—2019 年对应结果为 **1.84%、0.438、6.75%、0.611**，跨期质量仍不足。
4. 简化为统一正目标、加入 NR7 增次、第二候选 B 替代 A，均未通过各自固定比较；保留失败，不据结果换参数救回。
5. 最新完成工作是 A 的真实加减仓路径归因：四份已保存账户、5,710 个账户日、1,860 行周期路径；没有产生新的收益曲线。退出决定后的次日成交不是普遍损失来源，持有期间优势如何衰减仍缺少事前辨别证据。
6. 唯一已经冻结的新期间账户比较是 **A 的 SAVED_WEIGHT 对 POINT_BINARY**。冻结后真实账户日及完整周期均为 0；不能把历史补齐、合成测试或登记意向计作前瞻结果。

主要证据：[当前任务状态](../reports/research/510300_daily_weekly_goal_continuation_20261001/state.json)、[有效研究要求](../reports/research/510300_daily_weekly_goal_continuation_20261001/active_goal_effective_requirements.json)、[总报告](../reports/research/510300_trade_quality_and_sharpe_summary_20261001/全部判断与收益夏普进展.md)、[最新路径解释](../reports/research/510300_point_weight_path_bottleneck_v1/研究解释与下一步.md)。

## 2. 范围、目标与容易混淆的旧口径

| 对象 | 有效含义 | 不应混用的内容 |
|---|---|---|
| 本任务 | 510300 标的；日线及已完成周线；多头优先；20万元共同账户比较；提高净收益和净夏普 | 不是分钟交易任务，也不是恢复所有历史策略 |
| 交易质量 | 实际扣费胜率 p × 实际平均盈亏比 B > 1，并要求平均净收益为正 | 计划止盈/止损比不等于实际 B；pB 本身不是标准数学期望 |
| 标准期望 | 以平均亏损为单位：pB−q，q 为亏损率；无平手时 q=1−p | 不把 pB、利润因子 pB/q 和账户年化混为一谈 |
| 频率 | 按空仓到完全清仓的自然周期统计，列出逐年次数和零交易年份 | 加减仓订单数不等于完整交易次数；不强迫每年五笔 |
| 空头/期权 | 空头后置；用户允许未来用期权表达方向，但当前只研究标的点位 | 当前没有期权合约收益、融券可执行收益或订单授权 |
| 历史 Sharpe-1.2 研究 | 各自冻结的净夏普≥1.2、年化≥10%等门槛继续约束原研究 | 不能把旧试验未达标改写为通过，也不能自动把旧门槛当本任务新要求 |
| 稀疏机会委托 | 另有20万元、净夏普≥1.5、回撤≤10%的冻结 mandate，年化门槛为 null，年化用242日 | 不是当前252日共同账户比较的自动替代验收口径 |
| 当前实际持仓 | UNKNOWN；研究状态不推断用户仓位 | NO_VIEW、ABSTAIN、空仓研究目标不自动等于用户应卖出 |

证据：[稀疏机会 mandate](../config/510300_sparse_opportunity_mandate_v1.json)、[旧抗过拟合要求](../config/510300_anti_overfitting_requirements_v1.json)、[本任务范围修订目录](../reports/research/510300_daily_weekly_goal_continuation_20261001/)。最新明确用户指令可以改变后续工作范围，但不会追溯改变旧实验的失败结果。

## 3. 仓库与运行环境

- 工作区：`C:\Users\戴周阳\Documents\New project 8`；Windows / PowerShell。
- Python 入口：`.venv\Scripts\python.exe`。不要默认使用系统 Python。
- `data` 是到 `E:\ResearchData\New project 8\data` 的 Junction。
- `reports` 是到 `E:\ResearchData\New project 8\reports` 的 Junction。
- `.venv` 是到 `D:\CDriveData\戴周阳\project_venv` 的 Junction；`deliverables` 指向 `D:\ResearchArchive\New project 8\deliverables`。
- Git checkout 本身不能重建这些盘符上的数据、结果、运行环境和未跟踪文件。
- [requirements.txt](../requirements.txt) 列有 pandas、numpy、pyarrow、scipy、scikit-learn、matplotlib、duckdb、akshare、tushare 等；多数未锁版本。精确环境应读取各实验自己的 runtime/freeze 记录。
- 根目录没有实际的 `AGENTS.md` 文件；本任务使用用户提供的中文、Windows 和完整代码要求。未因此创建或修改代理指令文件。

| 目录 | 实际职责 |
|---|---|
| `research/` | 主要研究实现，通常一个方向对应一个或一组版本化脚本；当前直接包含1519个 `.py`，不存在一个代表全部研究的单体模型 |
| `config/` | YAML/JSON 实验参数、来源合同、manifest、权限与部分研究注册表 |
| `scripts/` | 采集、运行、复算、交付与状态记录入口；不要仅凭脚本存在认定任务正在运行 |
| `market_data/` | ETF日线交叉核对、PCF/IOPV、严格交易所传输、分钟/TDX等不同来源适配 |
| `backtest/` | 较早的估值、横截面、小账户、日内T等执行引擎；不是本次点位账户的唯一执行入口 |
| `src/` | 另有 A 股集中低风险趋势、极端压缩等分支实现，不属于本任务自动执行范围 |
| `tests/` | 大量方向专用测试；测试通过主要证明实现，不证明预测或交易优势 |
| `reports/research/` | 各研究的协议、输入快照、结果与报告；还需结合 `reports/backtest`、`data_quality`、`forward` 等目录 |
| `docs/` | 协议、旧判断、交付入口；文件名或最近修改时间不能代替逐项裁决 |

本次清点读取 `reports/research` 根层及研究一级目录中 **1281个常见结果/状态/协议 JSON**，所在目录快照为769个研究子目录。该数含本次状态整理目录；是文件清点，不是1281次独立实验，也不是对每个脚本重新验证。机器可读入口：[清点摘要](../reports/research/project_state_snapshot_20261002/inventory_summary.json)、[研究文件索引](../reports/research/project_state_snapshot_20261002/research_artifact_inventory.json)。特殊名称的裁决文件及其他报告目录另按下文直接引用。

## 4. Git 历史、diff 与工作区事实

清点时刻：2026-10-02约01:15，写入这两份文档之前。

- 分支：`codex/510300-nbs-negative-information-drift-v2`。
- HEAD：`9617d925c8c3ea3e9cb179e59967c270f7e49379`，2026-09-05 13:23:40 +08:00。
- 本地 `git log --all` 共84个提交；未联网检查远端，也未拉取或推送。
- **21个已跟踪文件存在工作区修改，暂存区为空。**
- 在 `research/scripts/tests/config/docs/src/market_data/backtest` 范围内，清点到5609个未跟踪文件：research 1431、scripts 1595、tests 671、config 1093、docs 813、src 3、market_data 3。此计数不含报告/数据、也不含稍后新增的两份文档。
- 普通 `git diff` 不显示这些未跟踪研究。不能用 HEAD、空暂存区或少量 diff 推断当前研究已经完整版本化。
- 全仓状态扫描曾遇到旧 `.pytest-tmp-*` 和迁移残留目录的权限警告；没有删除、修改权限或清理这些目录。清点重点是授权研究代码和已保存结果。

| 提交 | 已提交事实 |
|---|---|
| `bab9147`，2026-08-16 | 冻结R5与V3基线 |
| `ffcb02c` / `aff0c5a` | 修复纸面执行链、冻结R6及其失败归因协议 |
| `6be33d3` | 正式拒绝R6历史模型族 |
| `3081d58`，2026-08-30 | IF强制资金流拒绝及方向切换研究完成 |
| `51272d2` / `5b1a72b` / `cbae61b` | 期权与PIT盈利可行性阻断、官方历史成分接纳、盈利代理诊断完成；成分接纳不等于全部权重和财报PIT通过 |
| `aa3caa7`，2026-09-05 | DSV5政策变化不足停止裁决 |
| `564d02f` / `4fe013f` | NBS固定五分钟用途授权冻结、收益读取前的来源准入 |
| `9617d92` | NBS首次G2失败后关闭家族 |

当前 tracked diff 主要涉及：

1. `primary_market_forward.yaml`、`primary_market_forward_readiness.py`：从简单快照数量改为同日PCF、时区、去重、时钟延迟等质量合同，区分20/40/80/120完整质量日的不同阶段；20日不是收益评价准入。
2. `download_510300_daily.py`、`quality_check_daily.py`：预热期、来源分段、实际日期/行数/hash元数据、原子写入与多项一致性检查。
3. `collect_510300_primary_market.py`、刷新脚本和PowerShell运行入口：采集、刷新、运行状态与失败处理的工作区修改。
4. `RESEARCH_STATUS.md`、旧 `docs/DECISIONS.md`、部分质量/前向状态JSON：累计研究与运行记录变化。
5. `test_primary_market_forward.py`、`test_v3_forward_2_inputs.py`：相应回归检查变化。本次文档整理没有重新运行整个测试库。

证据：[Git快照及逐文件diff行数](../reports/research/project_state_snapshot_20261002/git_snapshot.json)、[完整本地提交索引](../reports/research/project_state_snapshot_20261002/git_history.tsv)。最近的 `point_*` 研究代码和报告属于当前工作区事实，不能声称已包含在9月5日的提交中。

## 5. 当前模型究竟是什么

当前不是“MACD加一条量能条件”的简单模型。MACD、量价、波动率及周线既用于解释，也存在于若干已独立拒绝的候选中，不能把它们统称为当前已验证因子。

当前两条点位来源：

- **A：`CORE_AUXILIARY_DRAWDOWN_GATE`**，回撤限制的核心/辅助组合。
- **B：`LAG_CONFIRMED_RUNS_AUXILIARY`**，带滞后方向确认的辅助组合。

它们依赖既有九组机械状态和两类月度退出模型。核心链含 `D60_INTRA` 日内相对隔夜状态、`V6_PANIC_RECOVERY`、普通岭退出、周期内模型、入场时固定版本退出、参考风险预算、最小方差/下行预算、模型支持路由、核心与辅助合成。代码事实见 [核心链](../research/point_core_observation_inputs_v1.py)、[来源重建](../research/point_state_reconstruction_v1.py)。

| 层 | 主要入口 | 已实现的约束 |
|---|---|---|
| 自然参考持仓与成熟标签 | [point_monthly_model_inputs_v1.py](../research/point_monthly_model_inputs_v1.py) | 未结束周期不生成训练标签；标签成熟后才进入拟合；不使用人工终点退出 |
| 两类原月度模型 | 同上及 [within_cycle_exit_inputs_v1.py](../research/within_cycle_exit_inputs_v1.py) | 最近20个成熟周期，至少10个周期/100行；按周期等权，原岭惩罚1.0；保持原月度制度 |
| 核心来源与连续预算 | [point_core_observation_inputs_v1.py](../research/point_core_observation_inputs_v1.py) | 已完成收盘状态、参考账户、预算与路由分别保留 |
| 完整收盘观察及点位 | [point_close_observation_v1.py](../research/point_close_observation_v1.py)、[point_forward_observer_v1.py](../research/point_forward_observer_v1.py) | 真实收盘接纳，未来标签不反写；未知状态保留 |
| 原二值点位账户 | [point_account_nr7_inputs_v1.py](../research/point_account_nr7_inputs_v1.py) | 正目标只表示可持有，含相同资金/风险/成本约束 |
| 保留目标大小的账户 | [point_weight_information_inputs_v1.py](../research/point_weight_information_inputs_v1.py) | 目标大小与10个百分点调整带；实际账户份额仍受风险预算约束 |
| B来源字段适配 | [point_second_weight_inputs_v1.py](../research/point_second_weight_inputs_v1.py) | 只把B原目标交给共同执行器；逐表保留真实来源身份；没有将B与A混合 |
| 真实仓位路径归因 | [point_weight_path_inputs_v1.py](../research/point_weight_path_inputs_v1.py) | 现金流、库存、股息、日内/隔夜损益逐日还原；峰值仅为事后标签 |
| 官方日历及前瞻账户 | [point_forward_calendar_check_v1.py](../research/point_forward_calendar_check_v1.py)、[point_weight_forward_v1.py](../research/point_weight_forward_v1.py) | 现金日也核对覆盖；不补成合格的迟到登记 |

最近保存拟合为2026-09-01：790行来自20个自然成熟周期，每周期总权重1。**790行不是790个独立样本；20个周期也未证明相互独立。** 正则化、周期等权、自然成熟标签已经存在，不能重复把它们写成新改进。[训练核对](../reports/research/510300_point_binary_rebalance_diagnostic_v1/model_training_review.json)。

## 6. 本任务已完成的研究链

下表按研究问题组织，不将重复日期、相同信号的不同费用或复用对照相加成独立证据。详细接受/拒绝理由在 [研究决策记录](RESEARCH_DECISIONS.md)。

| 阶段 | 已完成内容 | 当前含义/直接证据 |
|---|---|---|
| 油管订单流翻译 | 区分逐笔/盘口/Footprint与SMC结构语言；研究后转为日周线量价与价格结构 | 日线无法还原真实订单簿或主动买卖方向；相关历史筛查见 [SMC结果](../reports/research/510300_smc_sweep_fvg_historical_v1/result.json) |
| SMC扫流动性/FVG | 固定逐层条件及成本比较，主条件未通过 | `REJECTED_FROZEN_NO_PARAMETER_RESCUE` |
| 日线供给测试 | 直接收复、价格回踩、量能供给测试比较 | 压力主方案夏普约-0.078，12个周期；[结果](../reports/research/510300_daily_supply_test_v1/result.json) |
| 上涨段反推解释 | 已完成上涨段、量价、MACD、波动与周线解剖 | [解释图谱](../reports/research/510300_upward_episode_anatomy_v1/result.json)不是事前策略验证 |
| 更密集技术入场/周线多空 | MACD柱、EMA20、20日区间、周支撑/阻力及镜像比较 | [点位瓶颈](../reports/research/510300_point_payoff_bottleneck_v1/研究结论.md)：各固定规则实际pB均未达1 |
| RSI失败摆动 | 量化日线RSI失败摆动并统计真实点位 | [研究目录](../reports/research/510300_rsi_failure_swing_points_v1/)；没有合格新增点位策略 |
| 历史来源转点位 | 将既有来源重建为可解释、连续的日线点位 | [状态重建](../research/point_state_reconstruction_v1.py)，不复活已终止的旧混合账户 |
| 入场/训练退出分解 | 固定入场配对、独立连续规则、无训练空头镜像 | [贡献报告](../reports/research/510300_point_entry_exit_contribution_v1/研究结论.md)：局部改善不等于完整策略通过 |
| 核心/辅助路径归因 | 检查额外再进入来自哪一层、何时真实持有 | [来源路径报告](../reports/research/510300_point_component_path_attribution_v1/研究结论.md)：辅助增次跨期表现反转 |
| 更新到9月30日 | 补齐价格、月度训练、完整核心链与既有候选观察 | [任务状态](../reports/research/510300_daily_weekly_goal_continuation_20261001/state.json)；历史补齐不是真实前瞻 |
| 空头与等待诊断 | 空头仅7个完整历史点位；多头长空仓期逐日解释 | [空头诊断](../reports/research/510300_point_short_signal_diagnostic_v1/summary.json)、[等待诊断](../reports/research/510300_point_long_wait_diagnostic_v1/summary.json)；空头后置 |
| NR7补充频率 | 一个固定补充机制与共同账户对照 | [结果](../reports/research/510300_point_account_nr7_complement_v1/summary.json)：次数上升，收益与夏普下降，拒绝 |
| 保留原目标大小 | A的完整权重与二值账户比较 | [结果目录](../reports/research/510300_point_weight_information_diagnostic_v1/)；历史机制改善，非独立验证 |
| 抗过拟合否证 | 利润集中、去最大盈利诊断、区块区间、试验历史与准入核对 | [结果](../reports/research/510300_anti_overfit_falsification_v1/summary.json)：独立晋升拒绝 |
| 前瞻实现 | 官方日历覆盖与唯一权重比较的实际接续程序 | [冻结安排](../reports/research/510300_point_weight_forward_v1/研究安排与当前结果.md)；真实新增结果为0 |
| 控制再平衡后的简化 | 相同执行器下，所有正目标统一为50%上限 | [结果](../reports/research/510300_point_binary_rebalance_diagnostic_v1/summary.json)：四场景收益和夏普都下降 |
| B完整权重补测 | 新增4个B账户，复用4个A对照 | [结果](../reports/research/510300_point_second_weight_comparison_v1/summary.json)：早期改善、近期变差，拒绝替代 |
| 实际持仓路径分解 | 四个保存A账户的现金流/库存/股息和阶段损益 | [最新解释](../reports/research/510300_point_weight_path_bottleneck_v1/研究解释与下一步.md)：没有新账户，研究优先级转向持有优势辨别 |

## 7. 当前可比较的完整账户结果

下面只比较同一执行器、同一资金风险口径的保存结果。较早区间从2015-01-05至2019-12-31，近期从2020-01-02至2026-09-30，各自独立启动20万元；不拼接净值。

| 时期 | 成本 | 版本 | 净年化 | 净夏普 | 最大回撤 | 实际净pB | 完整周期 |
|---|---|---|---:|---:|---:|---:|---:|
| 2015—2019 | 基础 | A完整权重 | 2.14% | 0.501 | 6.51% | 0.656 | 22 |
| 2015—2019 | 基础 | B完整权重 | 3.09% | 0.746 | 4.04% | 0.616 | 15 |
| 2015—2019 | 压力 | A完整权重 | 1.84% | 0.438 | 6.75% | 0.611 | 22 |
| 2015—2019 | 压力 | B完整权重 | 2.86% | 0.696 | 4.18% | 0.704 | 15 |
| 2020—2026-09-30 | 基础 | A完整权重 | 4.24% | 1.290 | 2.63% | 1.178 | 32 |
| 2020—2026-09-30 | 基础 | B完整权重 | 4.02% | 1.251 | 2.63% | 1.084 | 31 |
| 2020—2026-09-30 | 压力 | A完整权重 | 3.99% | 1.217 | 2.75% | 1.073 | 32 |
| 2020—2026-09-30 | 压力 | B完整权重 | 3.77% | 1.175 | 2.75% | 0.989 | 31 |

证据：[完整指标表](../reports/research/510300_point_second_weight_comparison_v1/results/完整权重共同账户比较.csv)。

- A近期旧二值账户压力结果约3.12%/0.880；保留目标大小后为3.99%/1.217。原比较同时改变了再平衡方式，不能把全部增量只归因于信号强弱。
- 后续在相同再平衡下统一正目标，近期变为3.47%/0.887，增加订单但完整周期仍32；该对照补强了权重信息的历史作用。
- 当前固定名义点位A/B的pB为1.095/1.029；上述完整账户为1.073/0.989。**前者不是同资金现金账户收益，不能替换表中pB。**
- A近期完整年度2020—2025年均4.17个周期，B为4.00；2026不完整，不计年度均值。B的2023年为零交易年。
- A近期前五笔盈利占完整周期净利润约76.2%；仅作诊断移除最大盈利周期，pB由1.073降至0.965。不得把删除交易做成可执行策略。
- A较早末端有开放周期；权益包含市值，胜率仅统计自然完成周期。

## 8. 数据源、点时可得性与限制

| 数据 | 当前直接事实 | 限制及入口 |
|---|---|---|
| 当前点位日线 | 3488行，2012-05-28至2026-09-30；OHLCV及成交额 | [当前冻结价格](../reports/research/510300_point_second_weight_comparison_v1/inputs/prices.parquet)；用于历史开发，不是当年不可变版本认证 |
| 日线来源组成 | 新浪AKShare3453行；独立来源共识纠正3行；已存新浪增量20行；最新新浪增量12行 | `source/source_original/retrieved_at/correction_*` 保留；2015-12-07、2017-02-21、2018-09-04更正不能抹去来源差异 |
| 原价与总收益 | 不复权OHLC用于实际成交；股息单列；研究特征另用含息链 | 不能把复权高低价直接当真实成交价，也不能将除息跌幅全算隔夜损失 |
| 分红 | 登记、除息、支付分开；账户应收与现金分开 | [冻结分红表](../reports/research/510300_point_second_weight_comparison_v1/inputs/dividends.csv)；新分红到来时现有前瞻接纳器须扩展中性数据接续，不能忽略 |
| 官方交易日历 | 当前前瞻实现支持已公布2026日历；最近保存检查下一交易日为10月8日 | 未来年度日历尚非全量准备完成；不能用普通工作日替代 |
| 旧主日线质量报告 | `510300_daily_quality.json` 的PASS仅覆盖2431行、截至2026-08-18 | 不能为9月30日快照背书；读取各实验自己的输入/freeze/接纳记录 |
| PCF/IOPV | 交易所PCF及时间同步快照，按真实接入日起保存 | 旧readiness截至8月18日仅2个完整质量日；是旧快照，不是当前在线状态。必须有同日原子回执，不能补造历史IOPV |
| NBS分钟历史 | 原566350行在获批八个五分钟窗口用途下通过准入，随后G2失败 | 仍有263根正成交VWAP越界和232根零量零额记录；用途PASS不等于通用分钟数据PASS。本任务不使用分钟线 |
| 中证成分与权重 | 官方成分重建从2015年起有接纳记录 | 成分覆盖、历史权重版本、财报首公开时钟分别验收；当前权重不能回填历史 |
| 财务/估值 | 交易所/巨潮原公告、供应商整理及部分PIT层共存 | 公布时间、预告、快报、修订和首次公开应区分；公司改善不自动传导成ETF收益优势 |
| 住房资料 | 2021-01至2026-08共68份原月报、4760个70城二手房观察 | 已公布下跌城市比例不是房产总市值损失/贷款损失/股市资金流；基期与调查方法变化保留 |
| 制造业价格传导 | 2017-01至2026-08共116份报告 | 出厂减购进扩散指数不是利润率或共识意外；2015—2016缺失不补；行业分类、样本规模及季调修订有影响 |
| 政策/资金/海外 | 央行、统计局、FOMC等原文与部分市场响应数据 | 统计期末、首次公告、网页版本与本地取得时间不同；额度、成交、融资、购股、账面增值及净敞口不可混用 |
| 期权 | 有独立的可行性/观察/历史研究分支 | 完整期权链、历史bid/ask与报价时间、合约资格和成交可得性仍需逐研究认证；标的收益不能替代期权收益 |

[来源质量旧快照](../reports/data_quality/510300_daily_quality.json)、[PCF旧readiness](../reports/data_quality/510300_primary_market_readiness.json)、[住房研究](../reports/research/510300_housing_collateral_daily_v1/本轮研究结论.md)、[制造业研究](../reports/research/510300_manufacturing_price_transmission_daily_v1/本轮研究结论.md)。

## 9. 已冻结的实验口径

### 9.1 当前共同账户

以各研究 `protocol.json`、`freeze.json` 和冻结代码为准，不通过本文件修改参数。

- 20万元初始资金，仅510300与人民币现金；现金收益和无风险率均设0。
- 全交易日净值包含空仓日；当前共同账户按252交易日年化。
- 原目标最多50%；10个百分点调仓带；5日条件ES预算2.5%；10%跳空预算5%并受回撤余量限制；10%收盘回撤触发后续可执行退出及停止新开仓。预算不是实盘回撤必不超10%的保证。
- 已知收盘状态决定下一交易日开盘；开盘只可按原计划、现金和风险缩量；T+1、100份整数、0.001元不利刻度、最低5元佣金。
- BASE：单边佣金0.0002、滑点0.0005；STRESS：佣金0.0004、滑点0.001。
- 周期从空仓到清仓，累计买入支出为周期回报分母；未知目标不自动当0；明确退出意图锁定至可执行；末端不强平。
- 两时期和两成本完整报告；收益/夏普同时改善、pB>1、净均值>0、回撤要求分别检查。

证据：[A权重协议](../reports/research/510300_point_weight_information_diagnostic_v1/protocol.json)、[B共同账户协议](../reports/research/510300_point_second_weight_comparison_v1/protocol.json)、[执行器](../research/point_weight_information_inputs_v1.py)。

### 9.2 唯一冻结的新期间权重比较

- 冻结时间：2026-10-01 23:39:24 +08:00。
- 旧设计资料截至2026-09-30；首个新收盘原点2026-10-08 15:05以后，最早账户执行2026-10-09开盘。
- A `SAVED_WEIGHT` 对 `POINT_BINARY`，两费用、四个分别20万元账户，旧净值不并入。
- 只在第1008个实际交易日收盘正式评价；至少30个完整周期、5赢5亏。信息不足不延长到达线，中途季度只描述。
- 两成本均要求净年化和净夏普严格超过对照且为正，pB>1、净均值>0、最大回撤≤10%；同时保留预设20/252日区块与年份组不确定性筛查。
- 这个设计只验证已登记的仓位信息增量；30个周期和约四年不保证统计功效，也不自动证明整个复杂来源有效。
- 初始化已执行，不重新 `initialize`。当前真实账户日、登记和完整周期为0；没有据此创建后台服务。

直接事实：[协议](../reports/research/510300_point_weight_forward_v1/protocol.json)、[冻结文件](../reports/research/510300_point_weight_forward_v1/freeze.json)、[安排与限制](../reports/research/510300_point_weight_forward_v1/研究安排与当前结果.md)。

### 9.3 旧研究的不同门槛继续有效

NBS的来源G0/G1、机制G2和未运行G3/G4，DSV5的政策变化门，VAL04预测门，R6动态择时门，PCF/IOPV的20/40/80/120日门，不能被当前账户统计替代。旧研究常用242日、不同资金/样本/末端处理；比较前先对齐，不能跨表挑最大夏普。

## 10. 已否定方向和其他项目分支

完整决策见 [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md)。这里保留跨分支边界：

- R6趋势/波动/估值族25个候选正式拒绝；R5只作旧基准与工程回归。
- NBS在限定五分钟来源准入后仍失败；DSV5仓位政策因变化不足停止，不代表独立风险观察器同步取得策略资格。
- VAL04、MACRO-02、TECH-03及多个结构/市场压力/资金代理研究，按其原门槛分别拒绝或在来源/机制门停止；“未计算账户夏普”不能改写为“夏普为0”。
- `SELECTED_MIX_BAND10_SIMPLE2` 已由用户终止。不得因这里引用其历史诊断而恢复日更、D60、混合权重或参数优化。
- 96因子/18策略库是研究假设目录；其旧批次49账户与245个反事实没有达到目标。后续项目状态另记累计800个admitted账户场景和952个executed场景，**这些是该注册表口径，包含相关/重复场景，不是全项目独立试验数**。
- A/B/C/D综合、住房抵押品、制造业成本传导均完成过独立版本的有限实验，未建立合格策略；不可把最新点位研究的局部结果拿去改判这些分支。
- 财报现金质量、公司到指数、披露新增性、FOMC、政策约束、股票融资兑现、SFISF净敞口等已形成来源和机制诊断。它们不是已验证收益因子，但能约束后续新假设的经济含义。
- 仓库还含A股横截面、ORJ公告事件、现金要约、极端压缩、可转债、多ETF、QDII、美股与数字资产研究。它们有各自对象、授权和终态；**没有纳入当前510300执行范围**。多资产40%超额目标的保存终态为 `GOAL_NOT_ACHIEVED_NO_CANDIDATE_ADVANCES`，不能从中挑一个可见分区高分直接移植。

## 11. 尚未解决的问题

1. **可泛化优势不足。** A较早pB不足1，近期利润集中；B、增次和简化不能同时解决。
2. **已使用历史没有独立性。** 重新切分相同历史、换名称或把最近一段单列，都不能恢复未见样本资格。
3. **试验选择范围不完整。** 既有否证仅清点到部分216轮、583设置、599个已评价来源版本、另5个未运行；这些不是独立试验数。正式DSR/PBO保持NOT_COMPUTED，不能用目录总数填公式。
4. **持有优势的事前识别缺口。** 压力场景较早亏损11笔中5笔曾有净浮盈；近期亏损12笔中6笔曾有净浮盈。还有另一半从未净盈利；单纯止盈无法处理全部问题。
5. **频率与净优势冲突。** NR7增加周期却降低收益与夏普；A曾连续203交易日空仓、B350日，主要不是未知输入或执行故障。
6. **复杂来源的增量可解释性仍不足。** 模型已有周期等权和正则化，但从预测到合成目标、风险预算及实际持仓存在多层映射；需要分清预测、映射和收益三种问题。
7. **新期间结果尚未出现。** 最近保存检查为2026-10-02 00:22，最近完整日线9月30日。不是当前日期的买卖观点；本次文档整理未刷新行情。
8. **未来接口维护未完成。** 2027以后官方日历、新分红接纳尚需中性扩展；不能改变已冻结交易算法或补成迟到登记。
9. **版本化不完整。** 大量未跟踪文件和外部盘Junction使仅恢复Git提交不足。当前清点可定位文件，但不等于已经建立可独立重建的完整发布包。
10. **没有检查所有旧自动任务是否实时运行。** 保存的ACTIVE/PAUSED/状态JSON是对应时点证据；此文不以其推断当前进程、自动化或数据服务仍在线。

## 12. 当前最值得继续的方向与具体实验

本节是工作排序。除E03已有正式协议外，E01/E02是明确标记的后续研究提案，**NOT_RUN，未冻结，未产生结果**；不是新接受策略。

### E01：退出预测到实际目标的传递诊断（下一项开发工作）

- 问题：在持有中间阶段，原退出模型何时提出减持/退出，合成目标何时归零，实际风险预算与整数份额如何响应？当前路径损失来自原预测、父规则覆盖，还是目标到库存的映射？
- 数据：固定A的两个时期、两成本账户；对应原月度模型与逐日来源状态。先核对 [入场退出贡献](../reports/research/510300_point_entry_exit_contribution_v1/研究结论.md)、[组件路径归因](../reports/research/510300_point_component_path_attribution_v1/研究结论.md) 和 [剩余价值迁移](../reports/research/510300_remaining_holding_value_transfer_v1/result.json)，复用已完成部分，不重复旧筛选。
- 方法：逐日联结原点、所用fit版本、原模型预测及退出意图、核心/辅助贡献、目标权重、风险裁剪、下一开盘实际份额。固定分类为“无成熟预测”“预测仍支持持有”“预测退出但其他已知父规则仍持有”“父目标退出并实际清仓”“成交受阻/未知”；全体状态和周期全部报告。
- 输出：`origin → prediction → parent target → desired shares → actual fill` 对照表；逐类覆盖、第一次分歧、对应实际日损益与自然周期结果。拟定的新脚本/协议在执行前创建，不声称已存在。
- 边界：不拟合、不改变系数/窗口/退出，不生成峰值最优账户。未来损益仅作为标签，不能参与当时分类。
- 继续条件：只有找到原模型尚未表达、在决策时可观察的明确机制差异，才提出E02。若只是已知的亏损、回撤、模型偏差或少数高峰，结束诊断，不自动调参。

### E02：一个新信息的有限增量试验（条件性提案）

- 启动条件：E01或可核验的新来源给出具体机制，并与本决策记录去重；先完成字段定义、发布时间/取得时间、历史覆盖、缺失含义及失败出口。
- 当前尚未选定新的合格字段，故状态为 `NOT_REGISTERED_NOT_RUN`；不能把“尝试全部指标”当实验设计。
- 设计：只改变一项事前信息及其预定作用，保留A作共同资金/风险/成本对照。原已拒绝的B替换、统一仓位、NR7、旧混合、旧学习退出参数不重开。
- 在读新增结果前固定所有候选、阈值、时期、成本及统计方法；两时期、两费用全部报告。至少比较净年化、全账户净夏普、pB、净均值、回撤、自然周期数、毛损益与费用。
- 判定：不以单时期高分通过；不以增加订单冒充增次；没有同时改善和足够质量则拒绝该固定版本，不改窗口或方向救回。即便历史开发改善，也不得称独立验证。

### E03：继续已有唯一前瞻比较（已冻结、等待合法新原点）

- 接续原冻结程序，不重新初始化、重新选择A/B或重设开始时间。
- PowerShell按顺序执行 `.venv\Scripts\python.exe research\point_forward_calendar_check_v1.py --advance`，再执行 `.venv\Scripts\python.exe research\point_weight_forward_v1.py record`；仅在有合格新完整日线及原点登记窗口时产生新观察。
- 只读检查可用 `.venv\Scripts\python.exe research\point_weight_forward_v1.py check`。不得把反复检查旧资料作为新验证样本。
- 2026-10-08 15:05以后至下一开盘前是协议规定的首个新登记窗口；实际数据不合格时保留NO_VIEW/缺失，不填补成当时已登记。
- 到预定终点按原协议判定，不能首次夏普达线就提前验收。

### E04：中性数据与交接维护（不改变交易规则）

在确有新官方日历或新分红时扩展接纳器，保留旧来源身份、日期和首次登记。数据接口维护单独验证，不趁接口升级改策略。当前文档已完成入口与证据索引；代码/结果完整版本化仍是仓库维护问题，本次未擅自提交几千个未跟踪文件。

## 13. 后续模型如何使用和维护这份事实来源

1. 先读本文件，再读 [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md)，然后读取相关研究自己的协议、状态与结果；不要从聊天摘要猜测当前有效版本。
2. 一切数值注明时期、资金、费用、分母和年化口径；一切当前状态注明实际检查时间。
3. 新结果更新这两份文档时，写入证据路径与状态日期；旧拒绝、来源失败、未运行和暂停记录不覆盖。
4. `PASS`必须带宾语：来源、实现、复算、历史经济条件、独立验证、交付完整性分别记录。
5. 旧 [RESEARCH_STATUS.md](../RESEARCH_STATUS.md)、[DECISIONS.md](DECISIONS.md)、[authoritative状态](../reports/research/510300_authoritative_research_status_v1.json)及 [research_registry.jsonl](../config/research_registry.jsonl)是重要历史入口，但其对象/日期各异；本文件不会把旧全局禁令误扩展成所有后续授权研究都未获准。
6. 当前授权没有实盘订单、券商连接或对外发布。本次长期事实文件的完成不等于收益研究目标完成。
