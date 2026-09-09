# 市场状态感知策略资格与风险配置器 V1 冻结规范

## 1. 研究身份与目标

- 候选 ID：`REGIME_AWARE_STRATEGY_ALLOCATOR_V1`
- 版本：`1.0.0`
- 研究状态：`PREREGISTERED_BEFORE_FIRST_FORWARD_EVIDENCE`
- 证据标签：`META_FORWARD_ONLY_HISTORICAL_COMPONENT_RESULTS_ALREADY_VIEWED`
- 目标：仅在点时环境与真正前向条件证据同时通过冻结门槛时，为对应的冻结策略轨道分配研究风险权重；否则把未使用风险预算保留为现金。

本系统是独立的元策略研究候选。它不修改任何组件策略的信号、参数、成本、资产范围、历史结果或终态。组件的 `REJECTED_FROZEN`、`NO_VIEW`、`FORWARD_COLLECTING` 和未授权状态继续有效。某个已拒绝组件未来即使在特定环境中取得条件证据，也只是新元策略的冻结输入轨道通过条件门，不得把原组件改写为历史上已经有效。

## 2. 不可回避的污染边界

截至 2026-08-14 的数字资产组件结果已经被查看。该区间以及更早的可见期只能用于：

1. 固定组件身份和策略族；
2. 记录既有拒绝状态；
3. 开发输入格式、状态机和合成测试；
4. 证明简单的“追逐上一期赢家”不是合格方法。

该区间不得用于取得 V1 资格、选择环境阈值、调整权重上限、选择成本、替换基准或计算所谓独立样本外成功。V1 的真正前向起点固定为清单 `frozen_at` 之后的首个点时观察。任何资格证据的 `evaluation_window_start` 必须严格晚于清单 `frozen_at`，不得回填。

冻结脚本在生成清单前必须确认环境证据、策略证据、前向观察账本、评价报告、运行状态和回执均不存在或为空。任何一类前向数据先出现、后补清单的做法都视为事后冻结并拒绝。

## 3. 固定组件策略池

V1 只包含以下五条冻结研究轨道：

1. `BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3`
2. `DIGITAL_ASSET_DUAL_LONG_HORIZON_TREND_V4`
3. `DIGITAL_ASSET_RELATIVE_STRENGTH_ROTATION_V5`
4. `DIGITAL_ASSET_MULTI_HORIZON_TREND_VOTE_V6`
5. `DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V15`

策略池、策略族、状态来源、组件清单、适用环境和成本项均在 `config/regime_aware_strategy_allocator_v1.yaml` 冻结。V1 失败后不得增加第六条轨道，不得删除亏损轨道，不得把策略 ID 换成事后更有利的版本。

## 4. 点时输入契约

### 4.1 环境证据

环境证据为追加式 JSONL。每条记录至少包含：

- 唯一 `evidence_id`；
- `observation_at` 和带时区的 `available_at`；
- `data_status=PASS|NO_VIEW`；
- 项目相对 `source_path` 和对应 SHA-256；
- 当且仅当 `data_status=PASS` 时存在六个冻结特征：慢趋势、快趋势、上涨广度、风险、流动性、Carry 机会。

决策时只能选择 `available_at <= decision_at` 的最新记录。`observation_at` 不得晚于 `available_at`，且环境观察距离决策时点不得超过 24 小时。未来记录、哈希不符、重复 ID、同一可得时点冲突、缺字段、陈旧记录或越界特征均不得形成可用环境。

缺少环境文件、没有当时可得记录或最新记录为 `NO_VIEW` 时，唯一结果为 `NO_VIEW_CASH_ONLY`。旧环境不得作为今日证据沿用。

### 4.2 策略条件证据

策略条件证据同样是追加式 JSONL。资格记录必须明确绑定单一 `strategy_id` 和单一 `environment_id`，并包含：

- `evidence_type=TRUE_FORWARD_CONDITIONAL`；
- `meta_evidence_status=CONDITIONAL_FORWARD_PASSED`；
- 真正前向评价窗口起止时间；
- 最新前向观察时点及其原始状态；最新状态不是 `PASS` 时必须立即撤销统计通过；
- 累计合格前向日数、最近 504 日资格窗口、最近 90 日健康窗口和独立环境片段数；
- 连续进入确认次数；
- 基础、压力成本后的条件优势下界；
- 压力成本后相对冻结静态等风险组合的增量优势下界；
- Deflated Sharpe 概率、回测过拟合概率和环境利润集中度；
- 预测年化波动率；
- 完整成本项和完整基准声明；
- 来源路径、来源哈希、记录时间和可得时间。

系统按“最新可得记录”而不是“历史最佳记录”选择证据。策略证据距离决策时点不得超过 192 小时，必须按周刷新；更晚的未成熟、失败或 `NO_VIEW` 记录会覆盖旧的通过记录。`REJECTED_FROZEN` 是同一策略、同一环境、同一 V1 版本内的终态，之后不得恢复。缺失、陈旧、未成熟或失败证据不能被前值替代。

### 4.3 真正前向观察账本与固定评价器

资格指标不得由人工填写。`forward_observation_ledger.jsonl` 必须逐条保存清单冻结后产生的点时观察，包括策略、环境、信号时点、结果时点、结果可得时点、环境证据 ID、环境片段 ID、策略毛收益、基础/压力逐项成本、基础/压力成本后策略收益、现金收益、冻结静态等风险组合压力收益、完整基准及原始来源哈希。每条收益观察周期固定为精确 24 小时，结果最迟在结果时点后 24 小时可得；按 `Asia/Shanghai` 日期，同一策略与环境每天最多计一条，不能用日内多条或短周期记录凑足 504 日。`PASS`、`NO_VIEW`、`RUNNING`、`PARTIAL_SUCCESS`、`FAILED`、`PROGRAM_FAILED`、`SKIPPED` 和 `CENSORED_NO_OUTCOME` 均为保留状态；只有 `PASS` 可以携带收益，其他状态的收益必须为 `null`、成本明细必须为空，不得以昨日值或后续回填替代。

每条观察必须绑定 `signal_at` 当时最新可得且不超过 24 小时的环境证据。环境标签必须由冻结分类器从该证据重算，环境片段 ID 必须由完整追加式环境序列派生：环境类别真实变化，或同一环境的合格记录间隔超过 168 小时时，才开启新片段。短暂 `NO_VIEW`、失败或缺失记录不会把同一环境人工切成新的独立片段。人工改环境标签、手工拆分盈利片段、复用旧环境证据或引用冻结前环境记录均直接拒绝。

基础和压力净收益必须由评价器按“策略毛收益减逐项成本比例之和”复算，数值容差固定为 `1e-12`。成本明细键必须与该策略冻结成本项完全一致；压力情景的每一项成本不得低于基础情景；发生元配置切换时，`allocator_switching` 必须分别精确使用策略池冻结的 1bp 和 10bp，未发生切换时必须为 0。只声明“成本已计入”但无法逐项复算的记录不得进入评价。

固定评价器 `REGIME_AWARE_STRATEGY_FORWARD_EVIDENCE_V1` 只使用决策时点已经可得且质量完整的观察，重新计算：

- 基础策略相对现金的年化净优势区块 Bootstrap 5% 下界；
- 压力策略相对现金的年化净优势区块 Bootstrap 5% 下界；
- 压力策略相对静态等风险组合的年化净增量区块 Bootstrap 5% 下界；
- 最近 90 个完整观察日内，压力成本后策略相对现金、相对冻结静态等风险组合的年化净优势点估计；
- 基于收益偏度、峰度和 35 次冻结试验数修正的 Deflated Sharpe 概率；
- 8 个连续分区、训练/验证各 4 区的组合对称交叉验证 PBO；候选集合固定为该环境全部允许策略加 `STATIC_EQUAL_RISK_FROZEN_POOL` 零增量基线，缺少任一冻结策略的同期观察则 PBO 不成立；
- 按独立环境片段计算的最大正利润集中度。

主资格统计、独立环境片段计数和 PBO 只使用最新 504 个完整观察日，不使用不断扩张的全历史窗口；更早观察只保留在不可改写账本中，不能继续支撑当前资格。最近 90 个完整观察日另构成衰减闸门：压力成本后相对现金以及相对冻结静态等风险组合的年化净优势都必须严格大于 0，任一项不再为正就立即撤销统计通过并归零确认次数。该近期闸门是冻结的绝对基准检验，不是策略间最近赢家排名。

区块长度固定 20 个观察日，Bootstrap 次数固定 2000，随机种子固定 20260827。评价报告先以不可变 JSON 写入，再由资格记录绑定报告路径、SHA-256、评价器 ID、受控实现总哈希、冻结清单哈希、前向账本前缀哈希和环境证据前缀哈希。资格引擎会逐字段复核报告中的 `evidence_core`、`evaluation`、安全字段、追加式前缀字节数、实际 JSONL 记录数和哈希；后续只允许追加，新数据加入后旧报告仍可按前缀复核，截断或改写历史字节立即失败。自报指标、任意来源 JSON 或不同实现生成的报告均不得取得资格。

以下字段被明确禁止进入资格输入：`trailing_return`、`trailing_sharpe`、`recent_rank`、`winner_rank`、`best_strategy`、`oracle_weight`。系统不得根据最近收益或事后赢家排序。

## 5. 固定环境分类

环境分类优先级和阈值全部在配置中冻结。分类只使用当时可得的六个连续特征，按以下优先级输出一个环境：

1. `DEFENSIVE_HIGH_RISK`
2. `CARRY_DISPERSION`
3. `TREND_PERSISTENT_UP`
4. `TREND_PERSISTENT_DOWN`
5. `TREND_TRANSITION`
6. `MEAN_REVERTING_LIQUID`
7. `UNKNOWN`

`NO_VIEW` 不是市场环境，不代表看空、卖出或风险上升。环境标签只是资格匹配条件，本身不构成收益预测。

## 6. 资格硬门

一条策略轨道只有在当前环境完全匹配且以下硬门全部通过时才取得研究配置资格：

1. 清单、协议、实现、组件依赖和输入来源哈希全部通过；
2. 证据为清单冻结后真正前向产生，且没有历史污染；
3. 累计至少 504 个合格前向观察日，主资格窗口固定为最新 504 个完整观察日；
4. 最近 90 个完整观察日的健康窗口完整；
5. 最近 90 日压力成本后相对现金的年化净优势严格大于 0；
6. 最近 90 日压力成本后相对冻结静态等风险组合的年化净优势严格大于 0；
7. 最新一条前向观察的原始状态为 `PASS`；
8. 最新 504 日内至少 12 个独立环境片段；
9. 至少连续 2 个决策时点确认；
10. 最新 504 日基础成本后条件优势 95% 下界严格大于 0；
11. 最新 504 日压力成本后条件优势 95% 下界严格大于 0；
12. 最新 504 日压力成本后相对静态等风险组合的增量优势 95% 下界严格大于 0；
13. Deflated Sharpe 概率不低于 0.95；
14. 回测过拟合概率不高于 0.10；
15. 单一环境片段利润集中度不高于 0.50；
16. 预测年化波动率大于 0 且不高于 3.0；
17. 冻结基准集合完整；
18. 该策略要求的全部成本项已计入；
19. `paper_position_generation`、`shadow_signal_generation`、`order_generation`、账户连接和实盘授权全部为 false。

任何一项失败，策略权重为 0。`INSUFFICIENT_EVIDENCE`、`FORWARD_COLLECTING` 和 `NO_VIEW` 必须保留，不能当成失败参数待优化，也不能当成成功。

统计门首次全部通过时仅记为 `CONDITIONALLY_PASSING_CONFIRMATION_PENDING`。第二次确认必须与前一次有效确认至少相隔 168 小时、增加至少 7 个完整前向观察日、且评价窗口终点严格推进；在此之前重复运行或只改变决策时间不得增加确认次数。只有全部统计门再次通过且连续确认达到 2 次，才写入 `CONDITIONAL_FORWARD_PASSED`。统计门失败立即把确认次数归零；没有新增完整观察的运行不生成新资格记录，不能靠刷新时间延长旧资格。如果后续评价转弱、未成熟或 `NO_VIEW`，新记录立即覆盖旧通过记录；若超过 192 小时没有新记录，旧资格自动失效。

## 7. 成本后风险配置

合格策略的未归一化分数固定为：

`max(0, 压力成本后相对静态等风险组合的增量优势下界) / 预测年化波动率`

先在策略族之间、再在策略族内部按分数分配。固定约束为：

- 策略总权重不高于 0.80；
- 现金权重不低于 0.20；
- 单策略权重不高于 0.35；
- 单策略族权重不高于 0.50；
- 单次决策新增风险权重合计不高于 0.25；
- 风险降低不受新增风险上限阻止；
- 禁止负权重、卖空和杠杆；
- 现金是正式、合格且无需证明 Alpha 的剩余状态。

前次权重不得来自可编辑的“最新状态”或任意 JSON。只有位于冻结回执目录、绑定同一候选与同一清单、`run_status=SUCCESS`、安全开关全关、决策和生成时点早于当前决策、文件名与 `receipt_id` 一致的不可变回执，才可作为风险增加上限和切换成本的前态。回执必须保存完整策略与现金权重且合计精确为 1；否则本次运行失败，不得伪造已有权重绕过 0.25 新增风险上限。

基础和压力分配切换成本按各策略冻结的单边基点与绝对权重变化计算并在报告中单列。资格证据中的增量优势必须已经包含组件内部交易成本及元配置切换成本，运行时估算不得被重复解释为新的优势。

## 8. 固定比较基准

条件证据必须同时保留：

1. `CASH_CNY`；
2. `STATIC_EQUAL_RISK_FROZEN_POOL`；
3. `H00300_TOTAL_RETURN_CAPITAL_ALTERNATIVE`；
4. 每条冻结组件的静态轨道。

事后最优轮换只允许作为不可执行的上界诊断，不得成为资格证据。基准不得在看完结果后改成更容易击败的对象。

## 9. 状态机与停止规则

系统输出以下互斥主状态：

- `PROTOCOL_FROZEN_FORWARD_NOT_STARTED`
- `NO_VIEW_CASH_ONLY`
- `NO_ELIGIBLE_STRATEGY_CASH_ONLY`
- `RESEARCH_RISK_ALLOCATION_ELIGIBLE`
- `INPUT_OR_PROTOCOL_FAILURE`

策略级原因至少区分：组件终态、环境不匹配、无条件证据、前向未成熟、统计门失败、成本门失败、来源哈希失败和条件证据通过。

出现以下任一情况，V1 必须停止并保留失败：

- 受控文件或组件依赖哈希漂移；
- 使用冻结前历史区间取得资格；
- 读取未来可得证据；
- 通过最近赢家排名选择策略；
- 删除失败样本、改环境标签、改成本、改基准、改门槛或增加策略轨道救回结果；
- 把研究权重转成 Paper、Shadow、仓位、订单、账户连接或实盘动作。

## 10. 输出与安全边界

每次运行生成：

- 前向评价器最新构建状态和不可变构建回执；缺失账本、缺失环境证据、无新增完整观察和构建失败均必须落盘；
- 最新机器可读状态 JSON；
- 最新中文审计报告；
- 不可变运行回执，包含清单、环境证据、策略证据、前向观察账本、前次回执、状态和报告哈希，以及完整研究风险与现金权重；
- 明确的组件状态、环境状态、逐门判断、研究权重、现金权重和切换成本。

所有输出均为研究层资格与风险预算，不是证券数量、目标仓位、交易信号或订单。V1 永久固定：

- `paper_position_generation=false`
- `shadow_signal_generation=false`
- `position_mapping_enabled=false`
- `order_generation_enabled=false`
- `broker_or_exchange_connection_enabled=false`
- `live_trading_authorized=false`

## 11. Windows 固定入口

1. 首次且仅首次冻结：`.\.venv\Scripts\python.exe scripts\freeze_regime_aware_strategy_allocator_v1.py --mode freeze`。
2. 每次运行前验证清单：`.\.venv\Scripts\python.exe scripts\freeze_regime_aware_strategy_allocator_v1.py --mode verify`。
3. 从真实追加式账本构建条件证据：`.\.venv\Scripts\python.exe scripts\build_regime_aware_strategy_forward_evidence_v1.py`。缺失账本或环境证据时只写 `NO_VIEW` 状态与回执，不生成资格。
4. 运行研究资格与风险配置：`.\.venv\Scripts\python.exe scripts\run_regime_aware_strategy_allocator_v1.py`。需要延续风险前态时只可追加 `--previous-receipt <不可变成功回执路径>`。

以上入口均不得产生 Paper、Shadow、仓位、订单、账户连接或实盘动作。
