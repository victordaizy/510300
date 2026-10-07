# 项目研究状态：510300 分钟过程增量、日周线点位与宏观联合评分

> 状态日期：2026-10-02。本文是切换模型时的项目事实入口，不是新的实验结果或交易指令。
> 工作区：`C:\Users\戴周阳\Documents\New project 8`。配套决策账本：[RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md)。
> 项目结论：**尚未建立经过独立验证、满足所属完整目标的策略。** 各任务的active/paused/blocked状态分别读取，不从任何一条任务推断整个项目的运行状态。初次整理后，用户授权继续；2026-10-02 02:42完成盘口PR-E01/02来源实验，没有新增模型拟合、账户回测或市场采集。
> 本分支最新进度：2026-10-02用户明确允许现有数据上的有限历史再开发，并允许入场、持有、退出共同检验，已原样运行其零联网启动包。TREND/REPAIR/MIX50加BUY_HOLD/STATIC50/VOL10，共24账户、每份2,823交易日，全部结果保留；参数未改，0联网、0实盘。全部12个候选情景均未达到净夏普1.2及年化10%；20万元压力TREND年化0.651%/夏普0.123，REPAIR 0.129%/0.234，MIX50 0.546%/0.158；VOL10 4.567%/0.449。REPAIR仅4周期/10个收盘持仓日；MIX50与TREND日收益相关约0.991，平均仓位14.529%。本机合成自检和保存指标复算通过；本固定批结束，候选晋升0，不调参救回。当前入口第15D及[首轮结果](../reports/research/510300_offline_daily_policy_batch_v1/analysis/首轮研究结果与结论.md)。第15A—15C与其48任务状态保留为原实验及历史快照，不再把“必须等新字段/先有单独盈利入场”作为本批或所有历史研发的通用门槛；其他分支权限不变。

> 宏观联合评分最新进展：2026-10-02完成E53—E58。E53两原月生产同49.8但生产变化−0.3/−2.8、服务建筑订单不同，原数据公告无具体原因；原贸易Q2研究实际2019Q2，旧提案2025Q2引用补正。E54两官方原解释明确不同整体共同原因：2024-08高温多雨/生产淡季，2025-04前期较高基数/外部环境急变，均有需求不足等并存因素；只两后验月份，非单生产降幅或股票因果/完整预期，显示钟原决定前但首件版本未知。E55服务/建筑原订单两列全1212/1711源支持，E56一次2424原参数拟合；主期13机会净均−0.063346%、胜率46.1538%、树MSE恶化1.849081%，固定表达拒绝。E57主期零费界+0.239266%/费用差0.302613百分点，较早33净−1.108423%/零费−0.808623%，入场年2024=3、2025=10；新增四贡献集中于9/26，但全13保留。E58新共享指针仍2026-09-02旧weight源，主表及三原响应无amount，0网络关闭本线索。夏普NOT_COMPUTED/目标未达，0新账户/标签/既有源码；两公开GET/两官方查询仅E54。下一E59全原月份共同原因来源仅提案，不重训这些失败树；其他分支/DR暂无访问保持。

> 技术线本轮最新TECH.R127：A02突破后1—3完整日保留率的单系数可选周期内残差完成并双期拒绝。原R84现金前向OHLC/前20日高点/前序ATR20/3日/0.5ATR/最新严格过去突破全部不改；3488日线576字段，1507状态371已知1136NO_VIEW保持，94真实零仍已知，旧完整第九项门仍失败。唯一单列函数，原八项/142月115可用27未知保持，25辅助估计/90月复用、核心0重拟合；1010完整配对/497原未知、776精确回退、2符号变化非交易。早期MSE增0.049499%，改进95%[-0.000004805474,0]；近期增0.014160%，[-0.000009399845,+0.000004845883]跨零，整体FAIL；经济SKIPPED、账户/新标签0、收益夏普NOT_COMPUTED。12必要测试/50冻结对象，115月设计信息预检未读目标，3488字段与R84完全相同/142版本/25方程/1507预测/24周期误差复算通过、预测差0。TECH.R126协议/R127结果，原R84、T01、旧突破回踩及E03保持，0联网/行情。下一C03固定十日量价效率差的单系数另用途提案NOT_RUN；24旧源/3488字段/1507身份元数据预检一致，早期262/262、近期742/748原可用预测有字段；原R88及HIGH/LOW失败不改。独立验证、去过拟合和完整目标未达。

## 0. 共享项目中三条研究线的口径

同一工作区同时保存多项任务。**不能让同名状态文档的最后一次写入，自动把某项任务的权限变成整个项目的权限。** 本文件合并三条研究线；第1—5、7—10节保留宏观原因/联合评分的详细事实，第6节及其技术线快照说明点位研究，第12—15节保留盘口历史，第15A节保存分钟过程原实验，第15B节保存第一批基准诊断，第15C节记录后续全部路线图的授权与执行，第15D节记录用户新授权下已完成的纯日线完整政策比较。先按研究线选协议，再读“下一步”。

| 研究线 | 范围与目标来源 | 接续边界 |
|---|---|---|
| 分钟过程 / 完整日线政策 / 盘口储备 | [用户原目标](../reports/research/510300_intraday_process_increment_v1/goal_objective.md)、[分钟配置](../config/510300_intraday_process_increment_v1.json)、[D-native配置](../config/510300_daily_native_baseline_v1.json)、[路线图配置](../config/510300_roadmap_execution_v1.json)、[日线政策固定配置](../reports/research/510300_offline_daily_policy_batch_v1/launch/rules.json)；20万元主账户/2万元敏感性，净夏普1.2及年化10%，242主口径，不要求每天交易 | 当前第15D已按用户新授权完成24账户；已有数据可做有限历史开发，完整入场/持有/退出可联合检验，不以新字段或孤立盈利入场为通用前提。第15A—15C旧结果与观察任务保留；真实参数未确认、实盘仍无具体授权，其他分支不变 |
| 机制、预期差与联合评分 | 任务 ID `01a0e7c2-ae0c-7233-9b6e-e246c6647732`；净夏普1.2、年化10%、回撤10%、完整年五次；指数历史发现 | 本文第3、5、8—10节；该分支不做新前瞻、不扩个股 |
| 共享 repo：技术点位与仓位权重 | [其有效要求](../reports/research/510300_daily_weekly_goal_continuation_20261001/active_goal_effective_requirements.json)；频率软目标、实际净pB、提高收益和夏普；有独立新期间比较 | 本文第6节概述；完整代码、结果及其后续方案见[技术点位现行状态](PROJECT_STATE_TECHNICAL_LINE.md) |

技术线的[完整决策账本](RESEARCH_DECISIONS_TECHNICAL_LINE.md)可直接阅读；[原62项快照](RESEARCH_DECISIONS_TECHNICAL_LINE_20261002_310bf09f.md)也逐字保留，并在当前决策账本末尾提供 TECH 编号导航。原技术线整理还保存了[研究文件索引](../reports/research/project_state_snapshot_20261002/research_artifact_inventory.json)，可补充查找其他项目分支；这是文件定位索引，不是全部条目已经复核的证明。

技术线快照保留原文，其中“本任务”“当前共同账户”“E01/E03”只指技术线；它是带日期的分支快照，后续授权修订以本文件第16节及直接回执为准。技术线的252日年化、软频率、期权讨论或前瞻安排，不替代宏观分支的242日月度账户、完整年五次和历史范围，也不扩大盘口分支的执行资产。决策账本采用R编号记录宏观原因与历史研究，TECH.R编号引用技术线现行完整决策及原始快照，PR编号记录盘口研究，避免同一R编号被不同任务复用。E1为已完成宏观归因，E2为已完成并拒绝的财政原因增量，E3为已完成但来源门未通过的宏观研究，E4为已完成Bill探针及固定月中长期券来源，E5为已完成而期限不匹配的机构预期探针，E6为已完成并拒绝的单项季度供给增量，E7为已结束且完整时钟源门未过的持仓来源，E8为已完成单表准备金构成事实，E9为已完成并拒绝的实际周度三原因增量评分，E10为已完成同季度借款修订构成事实，E11为已完成来源但原完整门未过的16季度原因面板，E12为已完成并拒绝的现金及扣现金两渠道评分，E13为已完成上游QT政策先后事实，E14为已完成的5个非零SOMA季度上游政策先后事实，E15为已完成并拒绝的已公布国债兑付路径状态增量，E16为已完成但连续历史/公布时钟源门未过的国股银票来源研究，E17为已完成但当前数值版本到原公布时钟未建立的企业贷款来源，E18为已完成的原月度账户目标缺口诊断，E19为已完成但实际起息/到期来源门未过的逆回购研究，E20为已完成但历史数值版本未准入的结售汇分项来源，E21为已完成且未通过原高分准入的日频融资输入纠错，E22为已完成且未支持简单先涨后无收益叙述的价格时段事实，E23为已完成的保存叶标签来源与重叠解释，E24为已完成的原两训练大涨标签政策先后解释（首次公开/预告与严格开盘准入仍未建立），E25为已结束的SFISF原因输入旧用途核对，E26为已完成并封存的固定外需日频增量，E27为已完成的该原机会费用/贡献/入场年及有限端点路径目标缺口，E28为已完成的全部保存模型四格响应及成熟叶成员信息变化解释，E29为已完成但完整历史面板未准入的银行监管来源初探，E30为已完成的同版本前三季度两比率分子/分母恒等事实，E31为已完成的两季度四个舍入公开点值及息差方法口径否定，完整原版本未准入，E32为已完成的真实DR007成交额字段/单位及旧日空响应来源核对，E33为到期停止且0请求/历史行的实际采集尝试，E35为已完成但月度R身份/短菜单未通过原DR源门的公开目录核对，E36为已完成并封存的采购量固定联合增量及失败归因，E37为已完成的24月采购事实/评分时点及原数据页原因未知核对，E38为已完成的固定三月官方解读事实/原因输入未准入，E39为已完成的全国实际用电周期/跨公布衔接残差及固定三单月源门未过；E40仅不同原发布机构补单月资格提案；E01—E04为技术线提案，PR-E01起为盘口实验；初次整理时均未启动，后续已授权并完成的PR-E01/02见第15节，不能将该执行状态移植到另外两线。

## 1. 宏观联合评分分支：接手时先知道什么

1. 本节至第10节中，未另行注明的“主任务/本任务”专指 **510300指数层面的历史原因发现与多维非线性联合评分**，并非整个仓库或盘口研究分支。该分支要求快速看到结果，减少重复检验，允许阶段差异；同时保留当时可知、费用与真实账户约束。
2. 要回答的是“因子为什么变化、当时哪些约束发生改变、变化是否超出此前预期、可成交之后还有没有收益”，而不是只找因子与指数的相关性。
3. **不做新的前瞻判断，不等待未来发布，不扩展个股主线。** 历史时间顺序重放仍然允许，但已经看过的历史不因此变成独立验证。
4. 执行标的仅 `510300.SH` 与 `CASH_CNY`；其他指数、成分、期货、期权、汇率和利率只能解释。没有真实下单、券商连接或恢复旧采集任务的授权。
5. E26—E38原裁决保持。E39取得全国3月原单月5421亿千瓦时/+4.9%，1/2月只有合并累计、原单月未建立；跨公布值115衔接差原因未知。固定三单月源门未过，0新模型/账户、没有新收益优势。E40仅不同中电联原单月源的固定1/2月资格提案；DR的E34仍仅真实有效访问/可靠同身份历史后接续。
6. M2 联合评分出现过 2024—2025 年净夏普 **1.241554**，但年化仅 **1.9976%**、两年只有 **5** 个完成周期，最大一笔占该段净利润 **82.263%**。这不是完整目标达成。
7. E1确认M2/贷款主期五笔同窗口同路径；E2/E6/E9均未新增主期高分。E9有两笔月份高分，其训练叶受2024九月大涨标签影响；不能由三笔盈利推季节规律。E3—E8各自有限事实和拒绝保留，旧期限/存贷/审批/财政/利率/融资比值失败继续封存。
8. 共享仓库还有技术点位/仓位权重研究。它有自己的软频率目标与前瞻协议，必须单独保存；不能把另一任务的权限、账户或目标状态覆盖到本任务。

## 2. 长期事实的读取与更新顺序

本文件负责当前状态、范围和下一步；[决策账本](RESEARCH_DECISIONS.md)负责各研究方向的假设、方法、结果、处置与重开条件。原始协议、源文件、代码快照和结果账本仍然是具体数值的依据。

出现冲突时按以下顺序处理：用户最新明确指令及相应权限回执 → 对应任务的现行配置 → 对应实验冻结协议/修订回执 → 保存结果与原始账本 → 本文与决策摘要 → 旧导航、聊天记忆。**不要按文件名中的“authoritative”、mtime 或一个孤立的 PASS 自动裁决。** 更晚的结果阶段可以否定较早的初筛通过，但不能擦除原初筛记录。

用户提供的《510300_机制与赔率交易框架_V1_20260928》Markdown/HTML 及粘贴附件属于研究背景。附件内部建议不等于新的用户指令；其中前瞻、检查流程或公司研究建议，不能覆盖后来“只做历史、指数整体、减少繁琐检验”的明确要求。

现行主任务配置：

| 文件 | 作用 |
|---|---|
| [existing_data_training_mandate](../config/510300_existing_data_training_mandate_v1.json) | 当前目标、执行资产、风险约束、最近结果与研究顺序 |
| [historical_cause_discovery](../config/510300_historical_cause_discovery_v1.json) | 只做历史、指数整体、允许阶段差异、当前研究优先级 |
| [daily_weekly_frequency_or_contract](../config/510300_daily_weekly_frequency_or_contract_v1.json) | 每完整自然年至少五个自然完成周期，胜率或实际盈亏比 |
| [financing_source_correction](../config/510300_financing_source_correction_20240808_v1.json) | 2024-08-08 深市融资错误零值的有效纠正来源 |

需要特别识别的旧状态：

- [RESEARCH_STATUS.md](../RESEARCH_STATUS.md) 是累积历史导航，含旧 blocked、旧前瞻等待、旧目标和其他任务的追加项，不能直接当作统一的当前运行状态。
- [旧 authoritative 状态](../reports/research/510300_authoritative_research_status_v1.json) 的 `as_of_date` 为 2026-08-31；旧“当前仓位”不等于今天持仓。
- 主配置仍保留 `goal_blocked_receipt`、旧目标 1.3、旧策略复核和前瞻预测路径。相应历史字段不撤销后来的历史研究授权，也不恢复旧预测跟踪。
- [CONTEXT.md](../CONTEXT.md) 保留指数贡献、核心驱动组、传播与衰竭等语言；它是概念定义，不表示相应策略已经通过。
- [docs/DECISIONS.md](DECISIONS.md) 是旧决策历史，继续保留。本次新增的决策账本不覆盖它。

以后每完成一个重要研究，应同步更新这两份文档的“最近结果/下一步”和对应决策行，写清实验 ID、结果路径、适用期间、失败边界；旧拒绝记录只能追加修订理由，不能改成未发生。

## 3. 宏观联合评分分支的目标与冻结账户口径

| 项目 | 当前主任务口径 |
|---|---|
| 主账户 | 200,000 元完整账户；20,000 元是独立的小额费用比较账户 |
| 目标 | 费用后净夏普 ≥1.2、净复合年化 ≥10%、最大回撤 ≤10% |
| 频率 | 最终完整策略每个完整自然年至少 5 个自然完成周期；不要求每个研究节点各自达到，不为凑数强行交易 |
| 单次优势 | 正成本后期望；高胜率 **或** 高实际盈亏比。55%/2 是既有研究比较线，并非用户指定数值 |
| 周期计数 | 从现金入场至全部退出；分批订单合并；按入场年计一次；终点强平不制造自然完成周期；残年另列 |
| 可用频率 | 日线、已经完成的周线；本任务不读取新的分钟数据 |
| 最大仓位 | 50%；不得为达成年化目标扩大仓位或加杠杆 |
| 尾部约束 | 条件五日 ES95 预算 2.5% NAV；假设 -10% 缺口损失不超过 5% NAV，并不超过距既有净值峰值 90% 剩余空间的一半 |
| 终止规则 | 收盘回撤触发 10% 后，在下一可卖开盘退出，当前账户不恢复；T+1、限价和停牌使其不是绝对回撤保证 |
| 当前动作 | `NO_VIEW`；仓位目标 `UNSET`；不把缺失信息解释为自动买入、卖出或现金建议 |

最近月度完整账户使用的具体合同见 [M2 account_protocol](../reports/research/510300_multidim_money_surprise_score_v1/account_protocol.json)：单边佣金 4bp、滑点 10bp、最低佣金 5 元，100 份一手，价格 tick=0.001；242 日年化，现金利率与夏普扣除的无风险利率均设为 0。全现金日进入夏普与年化，零波动现金账户夏普不定义。不同旧实验的 242/252 日、无风险利率、费用、仓位和区间必须各自保留，不能直接拼表选优。

最新账户以开盘前可知的参考价与价格上限确定整数数量，开盘只判成交；持有中只减不加，风险信息取前一交易日。登记日确定分红权益，除息日记应收，支付日转现金；应收不能支付买单。样本末持仓估值并计退出准备，不用终点清仓提高交易次数。两年主段是原连续账户的切片，不能解释成重置资金后的独立账户。

## 4. Repo、Git 与代码现状

### 4.1 本次实查的 Git 快照

| 项目 | 2026-10-02 整理开始时的结果 |
|---|---|
| 分支 | `codex/510300-nbs-negative-information-drift-v2` |
| HEAD | `9617d925c8c3ea3e9cb179e59967c270f7e49379` |
| 最后提交 | 2026-09-05 13:23:40 +08:00；NBS 固定五分钟用途通过来源后，一次性 G2 失败封存 |
| 已跟踪修改 | 21 个文件；diff 为 1,945 行增加、278 行删除；无暂存 diff |
| 未跟踪 | `git status --porcelain -z -uall` 返回 282,030 个路径；包括大量报告、原始资料、交付副本和临时目录，**不是实验次数** |
| 未跟踪分布举例 | research 1,431；scripts 1,594；config 1,093；docs 813；tests 671；reports 182,446 |
| 研究目录 | `reports/research` 的一级研究子目录 769 个；另有直接放在该目录下的 JSON/MD 结果 |

上述数字是写入本文前的快照。Git 枚举对若干历史 pytest/迁移残留目录给出权限警告，因此不是磁盘全部文件的完整清单。本次没有修改 Git 索引、提交、分支或这些旧修改。

重要提交链：`564d02f` 固定 NBS 五分钟用途；`4fe013f` 在读取收益前完成准入；`9617d92` 记录 G2 失败。此前 `367d2c8` 拒绝 DSV5 成分脆弱性增量，`aa3caa7` 归档 DSV5 政策变化不足。**Git 历史主要覆盖早期工作，当前最新研究主要存在于未跟踪工作树；只 checkout HEAD 无法还原当前项目。**

完整本地历史清点为84提交。较早里程碑还有：`bab9147`冻结R5/V3基线，`ffcb02c`/`aff0c5a`建立R6执行/失败归因，`6be33d3`正式拒绝R6家族，`3081d58`完成IF强制资金流拒绝，`51272d2`记录期权/PIT可行性阻断，`5b1a72b`接纳2015年起官方历史成分，`cbae61b`完成盈利代理诊断。成分准入不代表历史权重或财报首次版本已经准入；上述提交均在[完整快照](evidence/project_state_20261002_snapshot.json)中留存，未检查远端是否另有提交。

已跟踪 diff 的实际内容：

- `research/primary_market_forward_readiness.py`、`scripts/collect_510300_primary_market.py` 等增加 PCF/IOPV 来源回执、时钟、失败状态与准备度判断；`config/primary_market_forward.yaml` 增加有限重试、覆盖日与来源合同。
- `scripts/quality_check_daily.py`、`scripts/download_510300_daily.py`、`scripts/refresh_v3_forward_inputs.py` 将行情日期、行数、来源分段、文件哈希与元数据一起核对/写入。
- `scripts/refresh_r5_daily_signal.py` 增加 510300、000300、H00300 与估值联合日期检查；失败时写不可消费状态，防止把旧信号当成今日信号。
- 相应测试、`paper` 状态、数据质量报告和旧导航也有修改。这些是已有工作树变更，本次没有重新运行旧采集或全量测试，也不将代码改动本身称为金融有效性证据。

### 4.2 代码地图

| 路径/模块 | 当前用途与注意事项 |
|---|---|
| [research/multidim_nonlinear_score_v1.py](../research/multidim_nonlinear_score_v1.py) | 八变量数据拼接、五日费用标签、日频树/线性/均值对照；源码保留原冻结输入路径，新研究须走纠正副本 |
| [multidim_policy_transmission_score_v1.py](../research/multidim_policy_transmission_score_v1.py) | 日频八变量加政策利率/资金利差变化，已失败封存 |
| [multidim_money_surprise_score_v1.py](../research/multidim_money_surprise_score_v1.py) | 月度事件、成熟训练池、深度 2 树、评分与树对象保存 |
| [multidim_money_surprise_account_v1.py](../research/multidim_money_surprise_account_v1.py) | 将固定高分机会转成完整现金/份额账户；被贷款账户复用 |
| [multidim_loan_surprise_score_v1.py](../research/multidim_loan_surprise_score_v1.py) / [account](../research/multidim_loan_surprise_account_v1.py) | 贷款事前预期解析、十变量增量和四个账户；显式引用融资纠正入口 |
| [multidim_public_fund_share_score_v1.py](../research/multidim_public_fund_share_score_v1.py) | AMAC 目录、官方 PDF、同表月变、十变量月度评分；阶段为 prepare/catalog/sources/run |
| [finalize_multidim_public_fund_share_v1.py](../research/finalize_multidim_public_fund_share_v1.py) | 已保存标签/树路径的必要核对、来源解释、图和报告；不代表重新训练 |
| [historical_msci_inclusion_demand_v1.py](../research/historical_msci_inclusion_demand_v1.py) | 固定五次纳入事件与联合状态；已完成、未进入账户 |
| [冻结 account_engine.py](../reports/research/510300_factor96_t11_date_proxy_account_v1/code/account_engine.py) | 最新月度账户实际加载的冻结引擎，不能只保留 research 目录而漏掉它 |
| [冻结风险输入](../reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/price_features.parquet) | 月度账户复用的成熟五日 ES 等风险输入 |
| [point_weight_path_bottleneck_v1.py](../research/point_weight_path_bottleneck_v1.py) | 共享仓库另一研究线的持仓损益归因；结果与本任务独立 |
| `research/`、`scripts/`、`backtest/` | 研究逻辑、采集/执行入口及旧账户组件；并非一个统一的当前生产策略 |
| `config/`、`reports/research/` | 权限、冻结协议、源资料、输入快照、结果账本、图表与验证回执 |
| `data/`、`paper/`、`deliverables/`、`review_packages/` | 原始/参考资料、旧观察状态、交付与审阅快照；交付包不等于外部审阅 |

Windows/PowerShell 下使用项目 `.venv\Scripts\python.exe -X utf8`。核心代码依赖 pandas、numpy、scikit-learn，PDF 链用 pdfplumber/requests，图使用 matplotlib。冻结目录常同时保存原代码和输入，重现前应读具体协议与阶段入口；不要直接对已完成研究再次执行 `prepare`/`run`，一些程序有覆盖保护。

## 5. 宏观与历史原因分支已经完成的研究脉络

下表是方向导航；完整“假设→方法→结果→处置”见 [决策账本](RESEARCH_DECISIONS.md)。不同期间、费用、资金、风险和源版本的结果不是可直接择优的比赛。

| 阶段 | 已完成的工作 | 有效结论与当前处置 |
|---|---|---|
| 早期估值、分钟传播、NBS、风险预测 | 正常化估值、PB/ROE、价格发现、DSV5 风险预测及政策转换 | 存在部分预测/排序关系，但未取得合格可交易策略；NBS、成分脆弱性、旧政策等终止裁决保留 |
| 大量技术及账户改造 | 趋势/回撤/反转/量价/状态路由/退出/仓位组合；旧 SELECTED_MIX 原貌复核 | 某些历史点值达线，选择膨胀、局部稳定性和独立性问题未解决；旧混合策略不恢复 |
| 96 因子 / 18 策略库 | 候选登记、原件与时钟、固定定义账户、实现修复、失败归档 | 库是研究目录，不是 96 个有效因子。项目登记截至 9 月 28 日累计执行 952 个账户场景，800 个准入、152 个实现无效；这些是旧范围计数，不能当全项目独立试验数或最新累计数 |
| 夏普瓶颈拆分 | 49 份原账本、245 个成本/约束反事实 | 最高夏普 0.69417；无一达标。冻结信号与持有退出的毛收益优势不足是主要问题，不能只靠降费/放仓修复 |
| 宏观需求与抵押品 | 住房、制造业价格传导、每日条件尾部模型 | 住房 12 个账户、制造业 8 个账户均未达完整目标；校准偏差不自动构成反向交易理由 |
| 公司经营原因 | 现金质量、回款、财务披露增量、配对预测、权重桥接 | 得到经营构成与信息新旧的解释，未证明指数剩余优势；按用户纠正已收束个股主线 |
| 历史指数原因链 | 信用、融资用途、国内资金、美国利率/FOMC/通胀就业、开放政策、盈利总体、行业抵消、汇率配置 | 区分多种机制与阶段，保留所有反例；相同因子方向经常对应不同原因，已检验的简化买入规则未成立 |
| 政策融资与实际购买 | SFISF、回购贷款、机构披露、抵押约束、融资保证金、MSCI 纳入 | 额度、授信、股票账面量、份额与实际净股票需求不等价；没有识别到可直接交易的被迫买盘金额 |
| 多维非线性评分 | 八变量日频、政策十变量、M2 九变量、贷款十变量、公募份额十变量 | 已真实训练与评分；未得到符合完整目标且信息增量明确的策略。共同收益来源归因已完成，下一步只接纳不同的原因信息，不增加模型深度营救 |

### 5.1 最近联合评分的实际结果

以下八/十变量日频数值属于原输入版本；2026-10-02 E21已按相同参数纠正融资源，最新配对数值见第5.21/R79。原月度各版本及原失败记录继续保留。

| 固定表达 | 样本与比较 | 保存结果 | 处置 |
|---|---|---|---|
| 八变量日频树 | 1,212 个评分日；深度 3、每叶至少 60；均值/岭回归对照 | 2024—2025 年 10 个不重叠高分机会，平均净收益 -0.8115%，胜率 40%；全期 36 次 | 固定表达失败，不进入账户 |
| 加政策传导两变量 | 同样 1,212 日；新增已观察逆回购利率与资金利差变化 | 主期 MSE 比八变量增加 0.4352%；高分负期望未改善；利率变化未被树使用 | 无主要期增量，不进入账户 |
| M2 事前预期 + 八状态 | 68 个准入月度事件，47 次评分；主期 20 次 | 主期 M2 未进入树；五笔高分均由既有指数状态给出。主账户夏普 1.241554、年化 1.9976%、回撤 0.5151%；全期夏普 0.699294、年化 0.7394% | 仅局部夏普达线；年化、逐年次数、增量证据不满足 |
| 已存事件树转到普通日期 | 1,082 次评分，普通日期 1,028 个；不再拟合 | 主期 35 个普通日不重叠高分，胜率 37.14%，平均净收益约 -0.23% | 不支持把月度高分条件扩展为普通日买入 |
| 融资收缩分解 | 主期 48 个不重叠观察、28 段收缩 | 平均净收益 -0.1087%，胜率 45.83%；47/48 次买入与偿还同向 | 余额下降不能直接理解成卖压释放 |
| 融资金额规模/强度分解 | 2024—2025 年 24 月、485 日 | 24/24 月买入与偿还变化同向；月度成交分母有范围差异 | 接受构成解释；未检验新预测，也不能回填为五日输入 |
| 贷款预期偏差增量 | 52 个共同月，40 次评分；同池九变量对照 | 主期 MSE 增加 5.9803%；五笔高分日期与对照相同。主账户夏普 1.225970、年化 1.9228%、回撤 0.4934%；全期夏普 0.518563 | 四个账户完成，当前增量拒绝 |
| 公募份额增量 | 26 份官方月报；12 次预热后 14 次评分 | 13 次预测改变，MSE 增加 1.7072%；三次联合高分净收益 -1.3348%、-1.9793%、-2.6689%，均值 -1.9943% | 三项预定账户入口只满足“新增高分日期”，不进入账户；已封存 |
| MSCI 纳入需求 | 2018—2019 年五次实施，两条路线；实施前固定五开盘间隔 | 五次仅一正；平均毛收益 -0.4850%、平均净收益 -0.7829%；实施日北向净买均正 | 固定表达失败；不能将北向买入直接映射为指数剩余收益 |

最近主要结果入口：[M2](../reports/research/510300_multidim_money_surprise_score_v1/result.json)、[贷款](../reports/research/510300_multidim_loan_surprise_score_v1/result.json)、[公募份额](../reports/research/510300_multidim_public_fund_share_score_v1/result.json)、[MSCI](../reports/research/510300_historical_msci_inclusion_demand_v1/result.json)。普通日期研究的精确修正版在 [纠正目录](../reports/research/510300_multidim_financing_composition_v1/data_correction/event_state_control/result.json)，旧版本不覆盖。

最近追加结果：[三评分统一归因](../reports/research/510300_multidim_shared_evidence_attribution_v1/result.json)。101条评分对应61个不同窗口，主期13条高分对应8窗口；M2/贷款五笔同路径、同叶、同预测，公募另三笔均亏损。零新拟合、零新账户。

### 5.2 已发现的集中性，及尚不能说的结论

- M2 主期最大交易为 2024-09-18 至 09-25，净利润 6,634.68 元，占主期净利润 82.263%；决定日为 09-13，早于 09-24 新政策。持有期间遇到后续好消息，不能倒说模型提前使用该政策。
- 公募份额三个高分叶的**训练标签带符号收益合计**，约 91.10%、94.18%、99.07% 来自同一条 2024-09-25 至 10-09 的 +22.43% 标签。它解释树为什么给高分，不是政策的因果贡献比例，也不是三个独立成功案例。
- 两段九月标签相邻，分别为 09-18→09-25 和 09-25→10-09；不是同一笔价格观察。要区分训练叶依赖、实际持仓重复、广义行情共同暴露。
- 2024年7/8/9月三次M2/贷款高分已统一核对：最大正训练标签为2024-02-19→02-26的+2.7005%，占叶内正收益合计27.30%—34.57%；九月标签尚未成熟。2025年1/4月才使用09-18→09-25标签，占叶内正收益合计51.00%/54.31%。原“全部高分由九月训练造成”的猜测不成立。
- 公募三个高分的最大正标签占叶内**正收益合计**均84.47%；与91.10%—99.07%的带符号分母比例分别保存，不混称。主期M2/贷款五笔实际窗口完全相同；公募三笔实际窗口与其分离，训练集中与实际重复是不同结论。

证据：[M2 收益集中度](../reports/research/510300_multidim_money_surprise_score_v1/主要期收益集中度.json)、[公募份额与高分来源](../reports/research/510300_multidim_public_fund_share_score_v1/份额规模与高分来源解释.json)。

### 5.3 继续寻找原因信息时确认的三项旧裁决

2026-10-02只读取旧结果，没有重跑。三项在2026-09-26已经完成，各12次原账户计算，共36次计算中有4个重复匹配对照，原报告记32条不同账户路径；不能作为本轮的新原因字段重复拟合。

| 固定原因表达 | 主压力净夏普 | 年化 | 最大回撤 | 原处置 |
|---|---:|---:|---:|---|
| 贷款期限构成 | 0.299661 | 0.7918% | 3.8039% | 无目标通过，构成增量门未过 |
| 住户存贷净增加差额 | -0.362314 | -0.9083% | 7.9252% | 无目标通过，两个固定时期增量均负 |
| 同需求判断下的银行审批条件 | 0.135432 | 0.3111% | 6.7375% | 无目标通过，审批增量区间跨零 |

银行来源有32季、29个保守公布日，季度日填充不产生独立宏观事实。中长期贷款不等于实际投资，住户存贷差额不等于股市资金，审批扩散指数不等于批准率或因果供给冲击。直接结果与hash见[E2旧来源复用核对](../reports/research/510300_multidim_shared_evidence_attribution_v1/E2_旧来源复用核对.json)，R56—R58保留各自假设和重验条件。

### 5.4 E2财政性存款原因增量：已完成、拒绝

独立实验[510300_multidim_fiscal_deposit_cause_score_v1](../reports/research/510300_multidim_fiscal_deposit_cause_score_v1/)于2026-10-02 03:25完成。104份央行原文财政字段均可提取，56次直接累计优先/48次单月；按同年连续已知报告转成年初累计状态，未制造单月财政流量。字段为累计净增/当期人民币存款余额的百分比。与旧财政预算支出同比、住户存贷差额的实质差异在原协议中登记，旧失败未改。

匹配对照与联合表达均含原M2九变量、月份sin/cos；仅联合表达加财政字段。两层树/叶最少6、504交易日成熟事件、费用与高分线不变。68原事件、47次评分，2树和1线性参考各47次，共141拟合，无参数网格，无源码改动。

| 指标 | 2021—2023 | 2024—2025主期 |
|---|---:|---:|
| 匹配对照MSE | 0.0005772249 | 0.0004952065 |
| 财政联合MSE | 0.0005331305 | 0.0004952065 |
| 同池均值MSE | 0.0003889368 | 0.0006027387 |
| 对照/联合高分次数 | 3 / 3 | 5 / 5 |
| 对照/联合高分平均净5日收益 | −0.734559% / +0.097741% | +1.903720% / +1.903720% |
| 财政字段实际路径使用 | 6次 | 0次 |
| 不同于对照和原M2的主期高分 | 不按此段营救主期 | 0次 |

主期20次财政预测全部与同池删除字段对照相同，原五个高分窗口不变。较早期MSE相对对照改善7.639%，仍劣于同池均值；三笔高分仅一笔正收益，中位数负。预先固定的两个独立机会条件失败，**REJECTED_FROZEN_FISCAL_DEPOSIT_INCREMENT_NOT_ESTABLISHED**，账户NOT_RUN_OWN_INCREMENT_GATE、净夏普NOT_COMPUTED。不得把旧M2账户夏普转给本字段。

新增月份控制令本次匹配对照与原M2并非完全同一表达：相对原M2主期14/20次预测相同，六次非高分差异不能归给财政；判断财政增量使用本次20/20相同的匹配对照。金融机构财政性存款不等于央行政府存款、财政实际支出、净投放或实际股票需求。历史首版未认证，2023统计范围扩展仍有影响，因果识别和独立有效性未建立。

直接依据：[冻结协议](../reports/research/510300_multidim_fiscal_deposit_cause_score_v1/protocol.json)、[结果](../reports/research/510300_multidim_fiscal_deposit_cause_score_v1/result.json)、[中文结论](../reports/research/510300_multidim_fiscal_deposit_cause_score_v1/财政性存款原因增量_研究结论.md)、[保存输出复核](../reports/research/510300_multidim_fiscal_deposit_cause_score_v1/保存输出复核.json)。完整标准输入程序保存在实验JSON回执，不能覆盖重跑。


### 5.5 E3来源核对完成，E4公告探针完成

E3于2026-10-02 03:57结束固定来源路线：11份官方页面，2背景、9有日期的规则页面、8不同原文日期、3政策根，核实12项范围/支付事实。IPO和再融资收紧是选择性约束；在审项目、部分发行方式和行业例外须分开，不能写成全市场停止融资。制度取消申购预缴不等于获配无需缴款，申购、获配缴款、结算和发行人收款不是同一个时点。9份规则页面也不等于9次独立供给冲击。

旧目录为历史沪深300成员并集范围：658登记企业、642有文档发行人，987检索任务中34不完整；90,500出现记录去重为74,129文档。4,513是IPO标题文档数，不是IPO事件数。缺连续发行金额、项目适用/受理状态和认证的缴款日期，不能建立全市场供给字段。原68个月仅按决定元数据连接，31个月能连接此前规则原文日期、37个月无此路线先前页面，连续字段合格数仍0；没有加载新收益或对这些31个月计算策略。

E3状态为 **SOURCE_GATE_NOT_PASSED_NO_CONTINUOUS_EQUITY_SUPPLY_FIELD**；评分 **NOT_RUN_OWN_SOURCE_GATE**、账户 **NOT_RUN**、夏普 **NOT_COMPUTED**。这一来源路线结束，不补零拟合，不把缺输入说成回测亏损，也不扩大为逐公司采集工程。部分证监会迁站元数据为2026-08-01，与2015/2023/2024正文日期不一致；正文日期只作暂定历史连接，首次版本仍未认证。

E3的后续先去重旧美债研究：V20已覆盖原104个月，名义/实质/通胀补偿关系及FOMC、2018—2019原因解释已有结果。E4改问上游的美国国债计划供给公告，未重新拟合旧利率字段。官方API一次请求取得2024年1月拍卖样例5条、114字段；这5条全为Bill，只投影11个公告候选字段，排除其余拍卖后结果。取得同一公告XML与一页PDF，金额为700亿美元，打印公告时刻为2023-12-28 11:00美东，即上海2023-12-29 00:00；拍卖为2024-01-02，发行2024-01-04。投标截止时刻不能代替公告公布时刻。

原件中该笔是42天现金管理Bill，不代表Note/Bond；单券单位不能推广到全部券类。打印时刻不是同时保存的HTTP收件记录，历史首版及全期间覆盖未认证。XML到期金额/日期没有在这份PDF同样显示，未进入候选白名单；公告gross规模不能称净流动性或市场意外。E4是 **PROBE_STRUCTURE_AVAILABLE_NOT_HISTORICAL_ADMISSION**；本轮0新拟合、0账户，不宣称因果识别或完整目标达成。

直接依据：[E3协议](../reports/research/510300_equity_supply_constraint_source_qualification_v1/protocol.json)、[E3结果](../reports/research/510300_equity_supply_constraint_source_qualification_v1/result.json)、[E3结论](../reports/research/510300_equity_supply_constraint_source_qualification_v1/IPO再融资供给约束_来源结论.md)、[旧利率复用核对](../reports/research/510300_equity_supply_constraint_source_qualification_v1/旧利率分解复用核对.json)、[E4探针结果](../reports/research/510300_treasury_auction_announcement_source_probe_v1/result.json)、[原件单位/时刻核对](../reports/research/510300_treasury_auction_announcement_source_probe_v1/announcement_pdf_unit_clock_check.json)、[E4结论](../reports/research/510300_treasury_auction_announcement_source_probe_v1/美国国债供给公告_探针结论.md)。

### 5.6 E4固定月来源及E5机构预期匹配：已完成

E4续接固定2024年1月范围，直接只请求15个公告/券类字段，完整38条中排除Bill/TIPS/FRN等31条，留下7条常规固定利率券。取得并逐页读取7PDF和7XML；4新发/3续发，七券gross合计2850亿美元。美东1月4、11、18日11:00只有3个公告根，均为上海次日00:00。金额和范围通过有限历史重建，首次版本、全部期间输入和金融有效性未认证。

| 原期限 | 2年 | 3年 | 5年 | 7年 | 10年 | 20年 | 30年 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 公告规模，十亿美元 | 60 | 52 | 61 | 41 | 37 | 13 | 21 |
| 2023-11-01提前计划 | 60 | 52 | 61 | 41 | 37 | 13 | 21 |
| 新规模偏差 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**七券全部由此前季度计划覆盖，逐次公告不能当新增规模意外。** 这不证明价格已完全吸收已知计划，也不否定供给影响。按原表与10月比较，1月总规模增加300亿美元，90%来自2—7年券，长端只增加30亿。到期桶在七份PDF中重复：1月15日919.10亿出现3次，1月31日1792.56亿出现4次；去重2711.66亿美元，逐行相加会虚增7215.88亿。到期、SOMA/FIMA与gross范围不一致，不计算伪净流动性。E4状态 **COMPLETED_FIXED_MONTH_ALL_COUPON_AMOUNTS_PREANNOUNCED**，逐次规模意外表达NOT_RUN。

E5固定读取2023目录Q4唯一一页官方调查。24家机构每期限去掉最高/最低，保留22份计算截尾均值及标准差；真实表头是FY24/FY25/FY26年末规模及机构认为不意外的范围。该范围不是置信区间，年度预测不是下一季度或1月共识，打印调查月不是首次公布时刻；目录Q2/Q4不能变成四季完整源。状态 **COMPLETED_AGGREGATED_EXPECTATIONS_PRESENT_SHORT_HORIZON_NOT_MATCHED**，当前短期预期差NOT_RUN，停止下载同类文件营救。

两项实际来源问题已回答，0新拟合、0账户、0源码改动，净夏普NOT_COMPUTED。本轮19项官方原对象仅提供原因/时钟事实，不证明完整目标或因果效应。

直接依据：[E4协议](../reports/research/510300_treasury_coupon_jan2024_source_qualification_v1/protocol.json)、[E4结果](../reports/research/510300_treasury_coupon_jan2024_source_qualification_v1/result.json)、[七券逐项对齐](../reports/research/510300_treasury_coupon_jan2024_source_qualification_v1/七券公告与提前计划对齐.csv)、[E4结论](../reports/research/510300_treasury_coupon_jan2024_source_qualification_v1/一月国债公告与提前计划_来源结论.md)、[E5结果](../reports/research/510300_treasury_dealer_expectation_source_probe_v1/result.json)、[E5结论](../reports/research/510300_treasury_dealer_expectation_source_probe_v1/机构预期时段匹配_来源结论.md)。

### 5.7 E6季度长端供给原因增量：已完成、拒绝

2026-10-02 05:19按来源前登记完成一次固定比较。16季实际财政部声明、20原件（1目录、16HTML、3图，其中一组复用E4）；13表解析、3图查看，共96月次、288个10/20/30年规模。只有一个原因字段：未来三月已公布gross相对此前三月同范围gross的百分比增幅。2022前三季缩减，2023Q3/Q4增加9.2308%/4.2254%，2024Q1增加4.0541%，2024Q2至2025Q4均为0；16季中6次非零。

原68事件全部保存，40有已知登记季度源，28在首份源之前未准入；40中12不满足12次成熟标签，实际28次评分（较早8、主期20），总84拟合。各模型共同池、原标签、504日、深2叶6、原80高分线不变，参数搜索0，源码改动0。2022无评分，不以阶段标题掩盖。来源可得时点保守用美国正文日末转上海，历史首版仍未认证。

| 主期固定比较 | 匹配11变量树 | 单字段12变量联合 |
|---|---:|---:|
| 评分 | 20 | 20 |
| MSE | 0.000491807 | 0.000506061（+2.8983%） |
| 固定高分窗口 | 5 | 5 |
| 平均原费用标签 | +1.903720% | +1.903720% |
| 不同于原M2/匹配对照的新高分 | — | 0 |

主期18次预测相同，2次供给走到实际路径：2024-12-13误差改善，2025-03-14误差变坏；都在融资净变化节点后按供给分组，分数47.73/50，未进入高分。另一次树内有供给节点但实际未走到。五个高分全部仍只用融资五日净变化；两笔亏损也完整保留。主期20事件只有9季度根、3种数值，其中16次增幅0。

状态 **REJECTED_FROZEN_LONG_SUPPLY_INCREMENT_NOT_ESTABLISHED**：MSE不恶化、新窗口、新窗口使用供给三个登记条件失败。0新账户，账户夏普NOT_COMPUTED；旧M2的1.241554不属于E6。本结果接受有限供给事实，拒绝该固定表达，不证明供给机制永远无效，也不认证融资变量因果作用。不得改期限、幅度、符号、模型或选子期救援。

直接依据：[协议](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/protocol.json)、[结果](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/result.json)、[16季度来源](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/16季度供给增幅与来源.csv)、[两次实际路径及反例](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/两次原因字段实际使用与反例.csv)、[中文结论](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/季度长端供给增量_历史结论.md)。E7仅源提案，未下载年度持仓数值、未注册评分。

### 5.8 E7持仓真实迟报与E8准备金原因构成：已完成

**E7有限来源门未过，不是策略失败。** 固定2023 TFF Futures Only年档（2,809行/87列），原10年美债043602唯一定位，52周字段完整；51个周二和一个7月3日周一。5件官方原件、1解压原文本。ION事件使7次报告按顺序迟报：1月31日至3月14日观察对应2月24日至3月21日实际公布，最迟比原定日多21天；7报告是同一服务事件根，不是7个经济冲击。

只读原M2统计月与decision_at，未读股票标签。2023-02-10若套普通周五会误用1月31日持仓（实际2月24日才公布）；2023-03-10会误用2月28日（实际3月14日公布）。原M2不含CFTC，故不改原模型结果、不声称已发现它的该未来函数。七迟报周有官方日末上界，另外45周仅通常日历、未逐周定位实际日。状态 **SOURCE_GATE_NOT_PASSED_FULL_WEEKLY_PUBLICATION_CLOCK_NOT_LOCATED**；比例未算、评分/账户NOT_RUN，有限路线结束。净空与业务类别也不能认定现金债卖出、基差套利或被迫平仓。

**E8接受单例会计事实，未建立指数优势。** 原2023-09-14 H.4.1表1，统一9月13日止周均及相对9月6日周均变化，1原件/10字段。持券−59.38亿美元，供给总项−32.02亿；逆回购减少释放650.94亿、TGA增加吸收506.26亿；其他因素净贡献169.02亿，准备金合计＋281.68亿。供给减吸收及七项完整桥原单位误差均0。持券已含在供给总项内，不重复相加；逆回购总项含外国官方与其他账户，不能称纯ON RRP。周三时点/同比列未用。

E8状态 **COMPLETED_SINGLE_RELEASE_RESERVE_CAUSE_IDENTITY_ACCEPTED_FACT_ONLY**。表例在注册前已浏览，已披露选择，不是独立验证；首次版本/当年收取未认证。两次程序在空白解析断言/字符串语法处停止，只修实现、不改源或经济口径，失败与成功程序分别保留。E7/E8合计0新评分、0新账户、0源码改动，净夏普NOT_COMPUTED，不能以会计解释代替股票因果效应。

直接依据：[E7结果](../reports/research/510300_cftc_treasury_position_source_qualification_v1/result.json)、[7次迟报对齐](../reports/research/510300_cftc_treasury_position_source_qualification_v1/7次实际迟报公布对齐.csv)、[原决定日时点错配](../reports/research/510300_cftc_treasury_position_source_qualification_v1/原月度决策受迟报错配影响.csv)、[E7结论](../reports/research/510300_cftc_treasury_position_source_qualification_v1/美债持仓与真实迟报_来源结论.md)、[E8协议](../reports/research/510300_fed_reserve_causes_single_release_v1/protocol.json)、[E8结果](../reports/research/510300_fed_reserve_causes_single_release_v1/result.json)、[完整准备金构成](../reports/research/510300_fed_reserve_causes_single_release_v1/准备金周变化完整原因构成.csv)、[E8结论](../reports/research/510300_fed_reserve_causes_single_release_v1/准备金变化的上游原因_同表结论.md)。

### 5.9 E9实际周度准备金三原因联合评分：已完成、拒绝

2026-10-02 06:23完成。来源前协议及来源后模型分别登记；实际日历208次、每年52次，8次非周四；208原表全部取得，其中一件复用E8，1040原字段。统一周均及对前周变化，供给减吸收变化误差最大2百万美元（容差3）。25份额外公告表导致首次DOM定位失败，保留原尝试，只按表头重定位，无源/经济口径或参数变化。

实际公开日纽约日末转上海后首个中国交易日16点决定，下一开盘至原五持有间隔；历史首版/当年收件未认证。208覆盖保留、2较早同中国决定日不重复供模型，206合格独特价格事件；13预热，193评分（早期90/主期103），两树及岭共579拟合，深2叶6、504日/成熟12、原费用与80线不变，0调参/源码改动。2025末固定标签可延伸至2026初，不等于2025末完整账户。

| 主期比较 | 同池10字段对照 | 13字段三原因联合 |
|---|---:|---:|
| MSE | 0.001289257384 | 0.001325846407（+2.8380%） |
| 正预测不重叠高分 | 3 | 3 |
| 高分平均原净收益 | +1.003615% | +1.003615% |
| 不同于对照的新高分 | — | 0 |
| 新原因实际路径使用 | — | 63/103 |

供给28、逆回购35、TGA1次实际使用可重合；主期40预测相同。三个高分全部不使用原因：2025-09-08→15及09-22→29只用公布月份正弦，11-24→12-01用趋势/融资。37个已保存高分叶归因：三主期联合叶最大正训练标签均2024-09-23→30的+22.1754%，占叶内正收益合计72.9480%/70.1972%/53.1338%；带符号合计89.2837%/85.1975%/54.8955%，均不是因果贡献。成熟时间合格，不删该标签重训。

**REJECTED_FROZEN_FED_RESERVE_CAUSE_INCREMENT_NOT_ESTABLISHED**：误差不恶化、新机会、新机会用原因三个门失败；早期MSE改善约2.58%但高分净均值−0.1146%，不能营救。账户NOT_RUN_OWN_INCREMENT_GATE，净夏普NOT_COMPUTED；不将三笔盈利、未定义实际盈亏比或原M2账户夏普提升为本分支策略。低分档均值高于高分不作反向交易。下一E10只来源提案，不改E9窗口/符号/树/分数线。

直接依据：[协议](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/protocol.json)、[源门](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/source_gate.json)、[模型登记](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/model_registration.json)、[结果](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/result.json)、[全部高分叶归因](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/全部高分叶来源归因.csv)、[必要复核](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/必要复核.json)、[中文结论](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/周度准备金三原因联合评分_历史结论.md)、[E10提案](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/next_experiment.json)。

### 5.10 E10同季度借款预估修订的原因：事实已完成

2026-10-02 06:51完成两份固定官方原文（2023-05-01/07-31）及7月31日公告所链一页资金来源与用途PDF，全部原页渲染查看。只问同2023Q3计划更新，不比较不同季度gross；来源前登记，不读取股票标签，0新拟合/账户/源码改动。

| 同季度原量，十亿美元 | 5月预估 | 7月预估/已知实际 | 对借款修订的构成 |
|---|---:|---:|---:|
| 净私人市场化借款 | 733 | 1007 | +274总变化 |
| 起始现金 | 550预估 | 402实际 | +148 |
| 期末现金目标 | 600 | 650 | +50 |
| Financing Need | 438 | 521 | +83 |
| All Other Sources | −244 | −238 | −6 |
| SOMA兑付memo | −158 | −158 | 0修订 |

现金两项198占headline修订72.26%；扣现金的借款残余76不能直接等同原文融资需要83。PDF其他融资改善6抵消，加原显示残差−1，274=148+50+83−6−1。未取得未取整值，不声称精确未取整恒等式认证。旧行733−244为489而原Total为488，该显示差已在原表存在，不填成新经济因子。SOMA仅memo，不能重复计；9月30日22十亿美元到期在10月2日现金结算，不含这里Q3借款。

5月现金目标以债限暂停/提高为条件；官方前次计划不等于市场共识，现金缺口与新收支预估的时间不同。7月PDF对旧预估的复述仅用于7月公布的修订构成，不能回填为5月已认证Financing Need；其他历史行不变成新增历史准入。两日期仅保守日末代理，历史首版及实际收件未认证。

状态 **COMPLETED_MATCHED_QUARTER_BORROWING_REVISION_CAUSES_ACCEPTED_FACT_ONLY**。首次1.007 trillion二进制缩放精确断言停止，未写结果；改Decimal，原定义/源值不变，失败谱系保存。借款上调不能全部解释为新预算需要、流动性收紧或市场意外。本题结束，净夏普NOT_COMPUTED；当时提出E11全部16季度的同期限四原因来源。后续E11已结束、原完整源门未过，见第5.11节；不从单例设股市方向。

直接依据：[协议](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/protocol.json)、[三源资格](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/result.json)、[两公告字段](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/两份公告同Q3字段.csv)、[原PDF三行](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/原PDF同Q3两预估及修订行.csv)、[完整构成](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/借款修订完整构成含显示残差.csv)、[中文结论](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/同季度借款修订的现金与收支原因_来源结论.md)、[E11提案](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/next_experiment.json)。

### 5.11 E11全部16季度借款修订原因：面板事实完成，完整源门未过

2026-10-02 07:26完成2022Q1—2025Q4全部固定季度、必要2021Q4边界。34原对象（1目录、17公告、15当前PDF及1当前PNG；31新/3复用E10）。源前固定全部16季度，15有直接同季度两预估/修订行，14当前与旧日期标签均和实际原公告一致；未读新增股市标签，0拟合/账户/源码改动。

| 来源事实，十亿美元 | 公告报告借款修订 | 现金安排贡献 | 融资需要修订 | 其他融资对借款贡献 | 显示残差 |
|---|---:|---:|---:|---:|---:|
| 2025Q2 | +391 | +444 | +3 | −57 | +1 |
| 2024Q3 | −106 | −28 | +15 | −94 | +1 |
| 2023Q2 | +449 | +322 | 缺当前原表 | 缺当前原表 | 未完整拆分 |

16中4季度headline与扣现金后的借款修订方向相反（2024Q4、2025Q1/Q2/Q4）；有直接融资字段的15中4季度headline与融资需要修订方向相反（2024Q3/Q4、2025Q1/Q4）。这不是股市信号准确率。2025Q2扣现金后反为−53；2024Q3融资需要上调15却被其他融资改善94及现金28抵消，SOMA memo修订86不重复加。原报告修订与显示两量相减可能差1，原值/差额分别保留，不认证未取整精确恒等式。

2023Q2当前PNG642×109只有现金表（起始500→178、期末550→550、现金影响增加322）；正文低收入/高支出变化117单独保留，不补缺失原表。较晚7月表复述5月112/−242不回填。2024Q1原PDF旧标签October31、实际前次正文October30，金额816/750一致但日期不一致，实际正文与原标签均保留；不是随意改历史时钟。当前表中的旧预估只用于当前公布时的修订，历史首版/当年接收未认证。

状态 **COMPLETED_SOURCE_PANEL_INCOMPLETE_FIXED_MODEL_NOT_RUN**，source_gate=false。接受有限连续原因构成，拒绝把不完整四分量直接准入模型；没有新的金融失败回测。首次提取同日两季日期命中2条而停止，仅按紧邻修订行修定位，源/样本/数值不变。PDF全部页文字读取/渲染，关键PNG/5页本轮目视，其他页不伪称全目视。

原E11结束，完整目标未达、净夏普NOT_COMPUTED；完成时提出E12两观察渠道，后续已分别来源登记/通过及模型登记/运行，固定增量拒绝，见第5.12节。两观察渠道不等同预算缺口或纯流动性；季度根复用不增加独立信息，原E6/E9拒绝保留。

直接依据：[协议](../reports/research/510300_treasury_borrowing_revision_panel_v1/protocol.json)、[源资格](../reports/research/510300_treasury_borrowing_revision_panel_v1/source_gate.json)、[结果](../reports/research/510300_treasury_borrowing_revision_panel_v1/result.json)、[16季度面板](../reports/research/510300_treasury_borrowing_revision_panel_v1/16季度借款修订原因面板_缺项保留.csv)、[45条原PDF行](../reports/research/510300_treasury_borrowing_revision_panel_v1/15份原PDF的同季度两预估及修订行.csv)、[覆盖与缺口](../reports/research/510300_treasury_borrowing_revision_panel_v1/16季度来源覆盖与缺口.csv)、[中文结论](../reports/research/510300_treasury_borrowing_revision_panel_v1/16季度借款修订原因_来源结论.md)、[提取修正](../reports/research/510300_treasury_borrowing_revision_panel_v1/来源提取定位修正回执.json)、[E12提案](../reports/research/510300_treasury_borrowing_revision_panel_v1/next_experiment.json)。

### 5.12 E12现金与扣现金后借款修订：已完成并拒绝增量

2026-10-02 07:48。先直接重建全16季度17原公告字段，两渠道源门通过（不补E11缺失拆分），后另登记11项控制/13项联合比较。沿原纠正后68M2事件、八国内状态/M2偏差/两月份控制，原5持有间隔成本标签、504交易日/12成熟事件/两层每叶6、固定80分线与alpha10参考不改。

40源准入、28未准入；12成熟数不足、28实际评分（较早8/主期20），84拟合、0参数搜索。主期MSE控制0.0004918070037224331、联合0.000493352604612737，点值增加0.31427%，非统计显著性声明。20中19预测相同，仅统计月2023-12/决定2024-01-12因扣现金部分改变低分预测；现金贡献主期路径0。原五笔高分平均净标签+1.903720%，同窗同预测同融资路径、新机会0。固定五项增量条件3未过，完整账户未准入。

全部五档、实际路径、叶成员、反例及根复用保存。28评分共享11季度根，主期20共享9，不当独立宏观事件。高分前三叶最大正成员为2024-01，后两为成熟2024-08，其正标签集中占51.00%/54.31%；未删大涨重训，收益不是美国新变量贡献。

状态 **REJECTED_FROZEN_CASH_REVISION_CAUSE_INCREMENT_NOT_ESTABLISHED**。0新账户/源码改动，净夏普NOT_COMPUTED。原两原因评分表达封存，不改方向/阈值/期窗/深度营救。E11完整四分量来源失败不变；上游首次公开问题E13为另题来源事实，不能恢复E12金融准入。

直接依据：[来源协议](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/protocol.json)、[源资格](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/source_gate.json)、[模型协议](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/model_protocol.json)、[模型冻结](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/model_freeze.json)、[结果](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/result.json)、[评分](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/逐事件借款两渠道联合评分.csv)、[高分](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/全部固定高分机会.csv)、[实际路径](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/全部实际路径.json)、[根复用](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/季度原因根被国内事件复用次数.csv)、[中文结论](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/借款两渠道联合评分_研究结论.md)。

### 5.13 E13借款修订的上游政策何时公开：事实已完成

2026-10-02 08:01。固定2024Q3源构成例，不按A股收益选；最多1新Fed页，复用4/7月Treasury两公告/两原PDF。实际5对象（1新/4复用）。4月原表Q3行SOMA−177/借款847，与7月复述旧行一致；7月SOMA−91/借款740，报告借款修订−106、其他融资+94。

Fed2024-05-01 14:00 EDT（上海05-02 02:00）正式宣布自6月国债月度减持上限60→25十亿美元、MBS35不改、政策利率5.25%—5.50%维持。政策在原4月预测之后、7月29借款更新前89个日历日公开。因此7月的上游QT政策背景已经公开，不能自动把SOMA86变化整体命名为7月当日新QT政策意外。

仅接受政策先后，不证明86完整市场预期、价格已吸收或全部现金/收支估计无新信息。35上限差乘3为105、不是86，差19不填成新外生因子；实际到期、票据和跨季现金结算等仍影响季度量。不重复加SOMA，报告修订/显示量差各保留。旧32次FOMC已有May1价格窗口，本轮只补原政策与借款计划连线，没有重跑旧5/20日收益，历史首版/接收未认证。

状态 **COMPLETED_SOMA_UPSTREAM_POLICY_PREANNOUNCED_FACT_ONLY**。0新增股市标签/拟合/账户/源码改动，目标未达，E6/E9/E12拒绝保留。本题结束；E14只全部原可见非零SOMA修订5季度来源提案，2023Q2未知保留，不按价格挑样本，不用未来May1解释4月已公布的12。

直接依据：[协议](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/protocol.json)、[源资格](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/source_gate.json)、[结果](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/result.json)、[正式时点](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/Fed原声明关键段及正式时点.json)、[三次公开顺序](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/上游政策与两季度计划公开顺序.csv)、[两表交叉](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/两当前原表的同Q3预估及旧行交叉.csv)、[中文结论](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/政策先于借款修订_来源结论.md)、[E14提案](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/next_experiment.json)。

### 5.14 E14全部非零SOMA修订的上游政策：事实完成

2026-10-02 08:38。按E11全部15可见字段机械选5非零修订，2023Q2未知保留。3新Fed页、复用2旧Fed及18Treasury公告/原表，共23原对象；27复用输入hash不变，5当前旧行与前次原件均一致。新声明使用正式美东14:00，Treasury仍日末代理，历史首版/收件未认证。没有新股票标签、拟合、账户或源码修改。

| 固定季度 | 原SOMA兑付预估，十亿美元 | 上游政策公开日 | 比当前借款更新早 | 简单路径与量差 |
|---|---:|---|---:|---|
| 2022Q3 | 0→120 | 2022-05-04 | 89日 | 7/8月30、9月60，合计120；仅数值相容 |
| 2024Q2 | 197→185 | 固定来源未定位 | 未识别 | 12仍未知；不能用晚于4月29日的5月1日政策回填 |
| 2024Q3 | 177→91 | 2024-05-01 | 89日 | 上限差105与量差86相差19，未强制解释 |
| 2025Q2 | 75→15 | 2025-03-19 | 40日 | 25→5的三月差60；仅数值相容 |
| 2025Q4 | 15→10 | 2025-10-29 | 5日 | 12月1日结束组合缩减；少一个月5相容，声明未打印国债0上限 |

**COMPLETED_NONZERO_SOMA_UPSTREAM_POLICY_CHRONOLOGY_ACCEPTED_FACT_ONLY**。接受4/5政策先公开及3简单路径相容；不接受完整金额共识、价格吸收、因果股市作用或交易优势。2022同日声明/计划只一政策根；2025Oct29同时降息25bp，不能作纯QT效应。2024Jan/Apr跨季说明显示到期/结算与政策月不同，不凭它反推12。SOMA memo不重复加；源事实通过不等于模型或账户准入。E6/E9/E12拒绝及E11原失败保留，净夏普NOT_COMPUTED、完整目标未达。

直接依据：[协议](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/protocol.json)、[源资格](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/source_gate.json)、[结果](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/result.json)、[五季度表](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/五季度SOMA修订的上游信息及未知量.csv)、[原表交叉](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/同季度原表三行及前次原件交叉.json)、[中文结论](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/五季度上游政策与未知金额_来源结论.md)、[E15提案](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/next_experiment.json)。

### 5.15 E15已公布国债兑付上限路径：来源通过，固定增量拒绝

2026-10-02 09:26。沿原32会议声明，加开始计划及结束实施说明，34原对象（26新/8复用），25次明确续行、4政策根。原68事件中38准入、30在首份登记计划前未知；12成熟数不足，实际26评分（较早6/主20）及78拟合。源后模型另登记，11控制/12联合，同504日/成熟12/深2叶6/原费用五日与80线，0调参/源码改动。

| 主期2024—2025 | 匹配对照 | 单字段联合 |
|---|---:|---:|
| 评分 | 20 | 20 |
| MSE | 0.0005463861696556954 | 0.0005463861696556954 |
| 高分不重叠窗口 | 5 | 5 |
| 高分费用后标签均值 | +1.903720% | +1.903720% |
| 新字段进入树 | — | 0 |
| 新高分窗口 | — | 0 |

全部26预测相同，五主期高分同原融资五日净变化路径、含两亏损。固定五项条件前3项通过、最后2项（新机会及新机会使用上限）失败。**REJECTED_FROZEN_ANNOUNCED_QT_CAP_PATH_INCREMENT_NOT_ESTABLISHED**，账户NOT_RUN_OWN_INCREMENT_GATE/夏普NOT_COMPUTED；不转移旧M2夏普，不重训降低叶数或择子期营救。较早6评分无高分；2024/2025分别3/2高分，不满足完整年五次。

只将未来三个整月已知政策上限作为约束状态，不称实际现金流、净流动性或市场意外。四根复用不增加独立冲击；早期训练只一根，2025结束根仅一评分且尚无该根成熟标签。结束实施说明无独立正式时刻，按纽约正文日末保守上界。全26保存叶/秩、原标签/时点/成熟池及45输入来源核对通过，0新拟合；首版、共识、因果与独立有效性未建立。本轮共同池不同于E12，跨实验MSE差不归给新字段。

直接依据：[来源协议](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/protocol.json)、[源资格](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/source_gate.json)、[模型协议](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/model_protocol.json)、[模型冻结](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/model_freeze.json)、[结果](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/result.json)、[评分](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/逐事件已公布QT路径联合评分.csv)、[全部高分](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/全部固定高分机会.csv)、[保存结果复核](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/保存结果必要复核.json)、[中文结论](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/已公布QT路径联合评分_研究结论.md)、[下一来源提案](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/next_experiment.json)。


### 5.16 E16国股银票六个月转贴现：官方样例通过，完整来源未过

2026-10-02 10:28完成来源前登记的有限路线。4入口、12追加官方请求、4目录检索，直接HTTP共16次；14原响应、9冻结父输入保存核对一致。人民银行首页“票据市场”原链接指向shcpe.com.cn，与票交所原主体介绍互证；.com域名出售页面排除。HTTPS失败与HTTP实际成功分别保存，不继续沿用“官网全部不可达”。

| 固定统计日请求 | 原返回统计日 | 六个月利率(%) | 六个月收益率(%) | 原业务响应 |
|---|---|---:|---:|---|
| 2022-01-04 | 2022.01.04 | 2.3900 | 2.4196 | 000，请求成功 |
| 2025-12-31 | 无 | 未取得 | 未取得 | 200，无相关数据 |

原页面利率/收益率两列和001/BAEX-1/6M已定位，差2.96bp不称信用溢价；完整年化/计息、电子承兑范围、剩余期限与编制/成交性质未取得。日期点号规范为同日，只补识别记录，不覆盖原程序或源值。2025年末该次空响应不推广成全年没有数据。

公开下载函数只展示单日staticsTime，不断言服务端没有其他正式范围服务。连续2022—2025输入、正式发布规则和合法历史可知上界未建立；统计日、今天HTTP时间和2026更新的主体介绍不替代曲线历史首次公开，默认全零序列未使用。

**SOURCE_GATE_NOT_PASSED_CONTINUOUS_HISTORY_AND_PUBLICATION_CLOCK_NOT_ESTABLISHED**，来源阶段结束、模型未注册，评分/账户NOT_RUN、净夏普NOT_COMPUTED。0新股票标签/拟合/账户/源码改动；不把缺输入写成策略亏损或经济机制无效，不换AAA债券/商票/期限、事后年图或扩大同一路线救援。E15及其他旧拒绝保留。

直接依据：[来源协议](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/protocol.json)、[字段合同](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/公开接口与字段合同.json)、[源门](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/result.json)、[中文结论](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/国股银票六个月转贴现_来源结论.md)、[必要核对](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/next_experiment.json)。

### 5.17 E17企业贷款加权价格：当前原表完整，历史数值版本时钟未过

2026-10-02 11:16完成固定2022Q1—2025Q4全部16报告。35次直接官方请求、35原对象、4目录检索，10冻结父输入及35原对象未变。16封面季度一致；同季度末月份表3的企业贷款同名子项、百分数单位、直接水平值和页面显示正式公布时刻均保留，四拼页逐表目视核对。原06在目录表题断言处失败，06r1只跳过无实际数值行的目录，仍要求每季度唯一实际表，原失败保留。

当前原表首末为4.36%和3.10%，仅当前历史表值的降幅1.26个百分点，不是股市效果。2023Q3/Q4、2024Q1/Q2/Q4、2025Q1/Q2七份当前PDF的文件名日期与生成/修改日期均晚于发布页；例如2024Q2页2024-08-09、当前文件/生成2025-08-12，2025Q1页2025-05-09、当前文件2025-08-15/生成08-14。元数据时区异常原字符串保留，不把它当公开时刻。

**SOURCE_GATE_NOT_PASSED_CURRENT_PDF_VINTAGE_TO_HISTORICAL_RELEASE_NOT_ESTABLISHED**。报告原发布日真实不代表当前各数值当时可知；晚生成也不证明每个数值必改。未认证目标值版本映射，不按原页更早时钟训练，不缩季度、换一般/总贷款或用后表回填。2025Q4直接3.10/变化−0.05与Q3直接3.14有1bp显示差，保留原值不反算。加权价格不是固定借款人冲击或包含全部非利息费用的综合成本。

来源问题结束，模型未注册，评分/账户NOT_RUN、净夏普NOT_COMPUTED；0新股市标签/拟合/账户/源码改动。只是来源未过，不是该机制收益失败。两追加目录检索为空不证明所有路线不存在，剩余5方法请求不凑数使用。E16及E6/E9/E12/E15原失败保留。

直接依据：[协议](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/protocol.json)、[原值16季度](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/16季度企业贷款表3原值.csv)、[版本时钟](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/当前PDF与历史公布时钟对齐.json)、[源门](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/result.json)、[中文结论](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/企业贷款加权价格_来源结论.md)、[必要核对](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/保存结果必要核对.json)。

### 5.18 E18原月度完整账户目标缺口：费用/原窗口内头寸不能填满缺口

2026-10-02 11:32。先核旧49压力反事实与E1共享收益归因的范围，另登记四原state/joint×20万/2万元、三固定日历段。只复用日账本及原自然周期：4,848日行、12指标行、30完成周期，16来源前后不变；净收益/242日年化/夏普/回撤、周期乘积、原金额与费用均和保存结果一致。

| 原M2联合账户主期2024—2025 | 20万元 | 2万元 |
|---|---:|---:|
| 原费用后年化 | 1.997638% | 1.910991% |
| 同原持仓逐日返费解释年化 | 2.324601% | 2.289328% |
| 事后完美半仓同原退出比例路径限定上限 | 2.772435% | 2.769160% |
| 自然完成周期 | 5（2024/25为3/2） | 5（2024/25为3/2） |

20万主期485日中25持有收盘，平均市值仓位2.2096%、有持仓收盘平均42.8671%，入场初始市值平均43.3007% NAV；每原完成周期NAV几何净收益0.7960%。段内净利润8,065.21元、费用1,354.49元，10%年化目标对应段内利润缺口33,914.75元。最大盈利贡献82.263%保留，不删样本或账户营救。

**COMPLETED_FIXED_MONTHLY_ACCOUNT_GAP_SIGNAL_OPPORTUNITY_DENSITY_AND_AMPLITUDE_INSUFFICIENT**。返费路径固定原所有动作、不回投费用、不改变风险预算；半仓上限故意事后跳过非正毛周期、无费用、首仓≤50%、保持原退出份额比例。它只是该有限交易路径比例族的解释上限，不是可执行策略、其他退出或新机会的上界，没有新账户引擎运行。

每年五笔、半仓、无费用且等幅的理想情景约需每笔资产毛收益3.8490%；按原两年五笔则约需7.7884%。这是情景算术，不是预测或无条件门。旧月度五窗口的密度和幅度不足，新原因若只复现同五笔也不能填补目标；下一先审能形成不同自然事件的实质原因信息。0新价格标签/拟合/引擎账户/源码改动，目标未达、旧冻结失败保留。

直接依据：[旧用途复核](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/prior_definition_review.json)、[协议](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/protocol.json)、[解释表达事前补充](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/diagnostic_expression_addendum.json)、[12行缺口](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/01_四原账户三时期目标缺口.csv)、[30周期](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/02_全部原自然周期收益与头寸分解.csv)、[结果](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/result.json)、[中文结论](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/月度账户目标缺口_研究结论.md)、[保存复核](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/保存结果必要复核.json)、[下一提案](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/next_experiment.json)。

E18原提案后续：E19已完成有限来源，实际结算字段未准入，见第5.19/R77；原E18账本和限定上限不变。

### 5.19 E19常规逆回购已知到期：原公告完整复用，实际结算来源未过

- **假设与范围**：旧公告量不是实际到期约束；固定2022—2025所有已保存常规明示期限，先验证实际资金起日、到期和非营业日规则，不预设指数方向。买断式、MLF、央票、正回购不替代。
- **实际来源**：19父对象冻结；1,041原行来自997原公告，1,040实际常规操作/1显式不操作；7天972行、14天68行。全部原身份及历史显示时钟匹配，正文/表头未检出明示起息/到期/假日结算项。旧覆盖不自动证明全部历史公告无漏项；首版收件仍未认证。
- **官方方法与裁决**：4目录检索、8直接请求，7成功原件共291,407字节；1清算所TLS EOF失败保留、未重试。一般交易规则有T+0/T+1和下一营业日定义，央行概述有逆回购到期收回机制，但未证明本批常规操作的清算速度/适用推导和完整结算日历。**SOURCE_GATE_NOT_PASSED_ACTUAL_START_AND_APPLICABLE_MATURITY_CALENDAR_NOT_ESTABLISHED**，源阶段结束；评分/账户NOT_RUN、夏普NOT_COMPUTED，0新标签/拟合/账户/源码修改。
- **处置**：接受公告原金额、期限、历史显示公开及有限机制事实；不将公告日加期限/股票日历/买断明示日改名实际到期，未来续作不补零。有限路线未通过不证明经济机制无效或所有其他原规则不存在；旧七天/全期限数量失败不重跑。
- **下一问题**：E20仅银行代客结售汇原因分项的旧用途/来源提案，未登记。旧六个月汇率研究核过一个月总额，未建立连续分项优势；先去重、核同范围原公开及时点，再决定是否另登记评分。不得将结售汇/涉外收付款或证券投资直接相加为沪深300资金。
- **证据**：[协议](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/protocol.json)、[原公告合同字段](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/固定区间逐原公告合同字段.json)、[原操作行](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/固定区间原操作行与实际到期状态.csv)、[方法适用范围](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/官方方法适用范围裁决.json)、[源门](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/source_gate.json)、[结果](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/result.json)、[中文结论](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/逆回购合同到期_来源结论.md)、[必要核对](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/next_experiment.json)。

E19原E20提案后续：E20已登记并结束来源阶段，E21纠错已完成，见第5.20/5.21；原E19源门与NOT_RUN不变。

### 5.20 E20银行代客结售汇原因分项：当前月表完整，历史版本未准入

- **假设与方法**：同一个结售汇总差额，可能混合经常项目和证券投资等不同原因。先核旧汇率/北向/RMB/跨境的实际定义与拒绝，再登记2021—2025固定60月、银行代客、美元同范围原月值。20父文件冻结；未选择模型、阈值、窗口或指数方向。
- **来源结果**：4目录检索、3实际原件请求，保存官方入口/栏目/连续原XLSX共297,858字节。当前60月、18原字段、1,080单元格完整有限，原结汇/售汇/差额保持显示单位亿美元；17月代客总差额正而证券投资差额负、5月相反、29月经常项目与证券投资差额异号。这只证明总量掩盖统计分项，证券投资不直接等于沪深300流入。
- **版本裁决**：官方入口URL含2023，实际页面公布2026-09-15，工作簿修改2026-09-08；当前数值未绑定原2021—2025各月首次公布内容/时刻。**SOURCE_GATE_NOT_PASSED_CURRENT_LONG_TABLE_TO_HISTORICAL_MONTHLY_RELEASE_VALUES_NOT_ESTABLISHED**。有限路线提前结束，不把当前完整长表回填旧公布日；不证明全部历史档案不存在或已发生修订。
- **计数与下一发现**：0新标签/拟合/账户/源码修改，夏普NOT_COMPUTED。旧融资纠错副本已存在，而原八/十变量两列仍为旧值：62单元格/60日期。这是真实输入错误，随后按原协议登记E21纠错，不能作为新经济候选或调参救援。
- **证据**：[协议](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/protocol.json)、[当前字段与版本](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/当前原表字段与版本合同.json)、[原单元格绑定](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/60月当前原单元格绑定.json)、[范围差异](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/当前内容范围差异_仅来源事实.json)、[结果](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/result.json)、[中文结论](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/银行代客结售汇原因分项_来源结论.md)。

### 5.21 E21原日频八/十变量融资纠错：分数变化，失败结论保持

- **实际纠错范围**：19父文件冻结，只沿用深交所2024-08-08全零融资行的官方已存纠正副本；两融资字段62单元格/60日期改变，其余全部字段、标签、时钟和原1,212评分日不变。原504日/成熟252、树深3/叶60、岭alpha10、五日成本、高分80、自然不重叠规则完全相同；未新增经济候选。
- **主期2024—2025**：原两模型各10笔、平均成本后−0.811464%、胜率40%；纠错后各9笔、平均−0.928734%、胜率33.3333%，两套选出的机会同一组。2024/25分别2/7笔；八树86评分日、十树62评分日改变（全2021—2025）。十树对八树主期MSE仍高0.318720%；两树均未胜同成熟池均值。
- **准入裁决**：**COMPLETED_EXACT_INPUT_CORRECTION_NO_PRIMARY_HIGH_SCORE_EDGE**。原高分机会均值非正，完整账户**NOT_RUN_PRIMARY_FIXED_HIGH_SCORE_MEAN_NONPOSITIVE**；夏普NOT_COMPUTED。纠正真实错误得到的是单独保存的新输入/结果谱系，旧结果和拒绝仍留存；不能从事件均值推满账户绩效，也不能反向交易负收益。
- **实际计数与核对**：4数值模型家族×1,212=4,848同参数拟合，0新标签/来源/完整账户/参数搜索/源码改动。1,212共同成熟池、2,424保存树的当前/训练预测及排名复算；首个输入变化前保存预测匹配。官方后来取得历史重建、first_vintage=false，不称独立验证；只覆盖两日频模型，不称整个factor96已纠正。
- **下一问题**：E22仅固定历史价格反应时段解释：先登记E21纠错输入、原保存分数和机会，再按原20日趋势/信号日/信号收盘至次日开盘/可成交五日净标签分段，保留全部485主期日、五档和9笔自然机会；无新源/标签/拟合/账户，不改窗、阈值、入场或持有，不将先后关系当因果或反向策略。尚未登记/运行。
- **证据**：[协议](../reports/research/510300_daily_score_financing_input_correction_v1/protocol.json)、[62个字段绑定](../reports/research/510300_daily_score_financing_input_correction_v1/62个融资字段纠错绑定.csv)、[机会配对](../reports/research/510300_daily_score_financing_input_correction_v1/原保存与纠错机会比较.csv)、[误差配对](../reports/research/510300_daily_score_financing_input_correction_v1/原保存与同参数纠错误差.csv)、[逐年自然机会](../reports/research/510300_daily_score_financing_input_correction_v1/逐年自然机会.csv)、[结果](../reports/research/510300_daily_score_financing_input_correction_v1/result.json)、[中文结论](../reports/research/510300_daily_score_financing_input_correction_v1/融资输入纠错_原口径结果.md)、[必要核对](../reports/research/510300_daily_score_financing_input_correction_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_daily_score_financing_input_correction_v1/next_experiment.json)。

E21原E22提案后续：E22价格时段和E23保存叶来源均已登记完成，见第5.22/5.23；原E21高分准入失败及融资纠错谱系不变。

### 5.22 E22固定高分价格时段：不是简单的上涨后入场

- **冻结与范围**：12父文件；原两评分各1,212日、主期485/较早727，全部五分档及各9主期/26较早自然事件保留。20日取原趋势分子，信号日含在其中；19日仅分解复利恒等式，未选新窗口。
- **主期事实**：同9次机会，信号前20日平均总回报−1.519519%、只有2次此前为正；信号日平均−0.728833%，除息对齐次日开盘缺口−0.483528%，原可成交五日净标签平均−0.928734%。**COMPLETED_FIXED_TIMING_SIMPLE_PRE_SIGNAL_RALLY_STORY_NOT_SUPPORTED**；不接受“此前普遍已涨，追涨后才失败”的简单叙述。
- **边界与实现**：只有时段事实，不能证明经济因果、完全信息吸收或提前买入优势；次日缺口为事后诊断，不能作为原16点输入，除息对齐不代表新买者股息权益。CSV/Parquet日期微秒/毫秒dtype检查首次报错，实际1,212日期逐值一致；只修相等比较到纳秒，失败记录保留，未改输入/口径。
- **计数**：2,424评分段绑定、20分档摘要、70原事件；收益标签只复用，0新源/标签/拟合/账户/源码修改，夏普NOT_COMPUTED。两个原复利/隔夜恒等式及原分数/标签匹配。
- **后续**：E23已登记并完成原保存树正预测来源解释；未通过调整入场或持有去救旧评分。
- **证据**：[协议](../reports/research/510300_fixed_score_price_response_timing_v1/protocol.json)、[全部五档](../reports/research/510300_fixed_score_price_response_timing_v1/原五分档全部价格时段.csv)、[全部事件](../reports/research/510300_fixed_score_price_response_timing_v1/全部固定自然机会价格时段.csv)、[结果](../reports/research/510300_fixed_score_price_response_timing_v1/result.json)、[中文结论](../reports/research/510300_fixed_score_price_response_timing_v1/固定高分价格时段_研究结论.md)、[必要核对](../reports/research/510300_fixed_score_price_response_timing_v1/保存结果必要核对.json)、[日期比较修正](../reports/research/510300_fixed_score_price_response_timing_v1/日期精度比较的实现修正.json)。

### 5.23 E23保存高分叶：正预期集中于少数历史月份

- **假设与方法**：既然原高分多发生在下跌状态，读取原保存树/共同成熟训练池，查看正预测的具体历史来源。12父文件；全部70自然事件，原float32分支、叶成员、节点样本数与均值逐例复算，所有标签退出不晚于各原决策日；不重新拟合、不删成员。
- **主期9例**：八/十变量对应叶成员完全相同。7个2025高分叶的最大正贡献月均2024年9月，占该叶正标签总和58.6546%—69.7788%；另外两2024高分例最大月为2024年2月、占24.30%/24.67%，不能把全部9例都归给9月。所有9条实际决策路径只用订单水平/融资五日净变化；十变量新增政策/资金变化均未出现在这些路径。
- **预测与实际**：主期叶均值预测平均+1.498983%，对应原五日净标签均值−0.928734%。叶有64—96个训练行，但原五日标签区间仅2—17个相交连通组；连通组数不是独立有效样本估计。2例叶正均值而中位数非正，不能泛称所有叶中位数负或只由单个样本决定。
- **主要来源**：原主期全部9例最大正单标签机械映射到两个日期：2024-02-02的6.777289%及2024-09-23的40.062709%。2025七例该单标签占正标签和约19.67%—23.78%，整个9月占比更大；这些是训练均值算术贡献，不是账户利润或已证实政策因果。
- **处置与计数**：接受原模型正预测来源和集中性事实；不接受这两个订单/融资状态已经识别上游原因或可重复收益。5,352原叶成员、全部逐月贡献/路径保存；0新源/标签/拟合/账户/源码修改。原高分和失败不变，不删9月、不改中位数目标、不重加权或重训。夏普NOT_COMPUTED、完整目标未达。
- **下一问题**：E24先登记E23主期全部9例机械映射的两个原最大正标签：2024-02-02（入2/5、出2/20）和2024-09-23（入9/24、出10/8）；复用原政策/托底/信用原文，再核信息在原16点决策前、次日开盘前还是持有期公开。仅竞争解释和时钟，不将两个大涨标签当盈利事件池/新候选，不删样本或重训；尚未登记/取得新源/运行。
- **证据**：[协议](../reports/research/510300_saved_high_score_leaf_contribution_v1/protocol.json)、[全部70例](../reports/research/510300_saved_high_score_leaf_contribution_v1/全部70事件保存叶分布.csv)、[全部训练成员](../reports/research/510300_saved_high_score_leaf_contribution_v1/全部原叶训练成员.csv)、[全部逐月贡献](../reports/research/510300_saved_high_score_leaf_contribution_v1/全部原叶逐月标签贡献.csv)、[决策路径](../reports/research/510300_saved_high_score_leaf_contribution_v1/全部保存决策路径.json)、[结果](../reports/research/510300_saved_high_score_leaf_contribution_v1/result.json)、[中文结论](../reports/research/510300_saved_high_score_leaf_contribution_v1/高分为何预期反弹_保存叶结论.md)、[必要核对](../reports/research/510300_saved_high_score_leaf_contribution_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_saved_high_score_leaf_contribution_v1/next_experiment.json)。

E23原E24提案后续：已登记完成限定时钟解释，见第5.24/R82；首版、预告与严格开盘门没有因此通过。原E23提案作为冻结谱系保留，接续记录单独保存。

### 5.24 E24原训练大涨标签：已有背景、后来新信息与时钟缺口

- **假设与范围**：E23全部主期9例最大正成员机械映射得到2024-02-02和2024-09-23两个原标签，只解释其窗口。冻结22父文件，复用5原件、新取得2原件；不是新盈利事件池，不删成员或重训。
- **2月窗口**：原2/2 16点决策、2/5入—2/20出、净标签+6.777289%。2023-10-23已公告ETF支持，2024-02-06更新近期扩大范围和未来力度；公告处于持有期，“近日”不是实际买入日，也不是逐日510300净买金额。
- **9月窗口**：原9/23 16点决策、9/24入—10/8出、净标签+40.062709%。原次日政策宣布、9/26后续经济/住房/资本市场部署与9/27已预告利率实施都在标签窗口；不能全归给此前订单/融资状态。实施日不是同一已公告内容的第二次首次意外。
- **逐项时钟**：官方归档段落中工具名称/一般用途位于09:10:58—09:19:36，股票工具5000/3000亿元额度/条件位于11:39:26—11:42:50附近；这些是归档段落标记，未认证首次公开、市场接收或发言秒，不能把全文所有内容一起倒填开盘。股票额度不同于5000万户/保障房3000亿元，也不同于实际指数净买入。
- **未解决**：CSRC可见2024-09-24、PubDate元数据2026-08-01；英文页9/24 9时为会议时间、全文链接为10/18路径。原9/23预告是否16点前可知、首版/严格开盘准入均NOT_ESTABLISHED；预期差与政策因果收益份额NOT_IDENTIFIED。未找到不等于不存在/零预期，有限来源不覆盖全部市场信息。
- **结果与计数**：COMPLETED_TRAINING_TAIL_POLICY_CLOCK_MIXTURE_WITH_PUBLICATION_GAPS_PRESERVED；8时钟记录含1未知预告、7原件共353,346字节，4目录检索/3直接请求/2成功新原件136,993字节。0新标签/拟合/账户/源码修改，原净标签和失败保持，夏普NOT_COMPUTED、完整目标未达。只接受限定先后/混合事实，不证明全部上涨政策造成或原状态完全无法预测反应概率。
- **下一问题**：E25先核SFISF股票融资工具的旧用途与官方公开操作来源：固定2024-09-24—2025-12-31，区分宣布的潜在额度、正式开放、已公布的实际互换操作及后续融资约束。先复用两机构专属披露和旧政策/数量研究；只有实质不同信息才另登记来源。操作量不是510300净买入，无共识不称已识别意外；同用途覆盖或完整时钟/版本不足即结束。当前只提出来源问题，未登记新字段、模型或账户。
- **证据**：[冻结协议](../reports/research/510300_training_tail_policy_clock_explanation_v1/protocol.json)、[原标签与完整时钟](../reports/research/510300_training_tail_policy_clock_explanation_v1/两个原训练标签_政策公开先后.csv)、[逐项原文/标记](../reports/research/510300_training_tail_policy_clock_explanation_v1/逐项原文与记录标记绑定.json)、[原件索引](../reports/research/510300_training_tail_policy_clock_explanation_v1/source_manifest.json)、[源门](../reports/research/510300_training_tail_policy_clock_explanation_v1/source_gate.json)、[结果](../reports/research/510300_training_tail_policy_clock_explanation_v1/result.json)、[中文结论](../reports/research/510300_training_tail_policy_clock_explanation_v1/训练大涨标签_政策信息先后结论.md)、[必要核对](../reports/research/510300_training_tail_policy_clock_explanation_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_training_tail_policy_clock_explanation_v1/next_experiment.json)。

E24原E25提案后续：旧用途核对已完成、原因环节没有区别；随后E26固定外需增量及E27目标缺口完成，见第5.25—5.27/R83—R85。原E24冻结提案与首次公开缺口不改。

### 5.25 E25拟议SFISF原因环节：已有用途覆盖，来源前结束

- **假设与核对**：对E24提案的额度/开放/实际操作/使用约束，先绑定16父文件，直接读旧十节点传导、两机构四报告、官方保存目录和旧事件/数量协议。不是把旧事实换名当新因子。
- **结果**：旧链已覆盖开放申请、两次互换、部分融资实际使用/增持及公布滞后；保存官方目录两操作均已纳入。没有建立实质不同输入及条件用途，按提案停止条件结束。目录不是完整历史全集证明，也不否定整个工具机制。
- **区别与封存**：旧数量代码实际为七日逆回购，确实不同于SFISF，但只区别于这个模型不足以忽略已有SFISF传导用途。原固定20日账户失败保持。COMPLETED_PRIOR_USE_REVIEW_PROPOSED_SFISF_CAUSE_INPUT_NOT_DISTINCT；0新源/标签/拟合/账户/源码改动。
- **证据**：[协议](../reports/research/510300_sfisf_prior_use_review_v1/protocol.json)、[直接旧用途](../reports/research/510300_sfisf_prior_use_review_v1/prior_definition_review.json)、[结果](../reports/research/510300_sfisf_prior_use_review_v1/result.json)、[中文裁决](../reports/research/510300_sfisf_prior_use_review_v1/旧用途裁决_不重复SFISF原因输入.md)。

### 5.26 E26外需条件联合评分：高分均值转正，固定增量门未过

- **不同用途与来源**：固定新增新出口订单水平/连续月度变化。旧15个月CPI背景描述和库存32账户用途/失败保留，不从总订单减出口推算国内订单，不称新物理来源。132份当月原文及复制表一致、原订单同源/同钟、旧15月值一致，原全部评分/成熟成员完整；历史首版/首次收件未认证，探索性重构不是独立验证。
- **冻结比较**：原1,212日（主485/早727），504/252成熟池、深度3/叶60/seed20261001、岭alpha10/原clip、五日费用/登记权益标签和80分自然不重叠规则全保持。只一新经济候选，2,424新拟合（树/岭各1,212），原八变量预测复用、控制不重训。
- **主期**：原9机会净均值−0.928734%，外需10机会+0.284198%、胜率40%；4条自然高分实际路径使用外需。树MSE比原树下降1.095102%，但仍大于同池简单均值。五项预定门4过1败，REJECTED_FROZEN_FIXED_EXPORT_ORDER_CONDITION_NO_PRIMARY_INCREMENT，不进入账户。正均值改善保留，不将点值改善称目标完成。
- **较早**：原26机会+0.028983%，外需22机会+0.218691%、胜率45.4545%。全部五档、均值/岭/树比较及自然机会均保存；不按时期选优或删除外需列。
- **边界与计数**：0新来源/训练标签/完整账户/源码改动，夏普NOT_COMPUTED；股市外需因果与市场预期差未识别。原评分主期按信号日，标签跨2026允许原成熟规则；原逐年文件按信号年，不直接用于按入场年合同频率。E27已另做完整目标缺口诊断，未改本裁决/预测。
- **证据**：[来源协议](../reports/research/510300_export_order_cause_increment_v1/protocol_source.json)、[原文与字段](../reports/research/510300_export_order_cause_increment_v1/全部132月外需原文绑定.json)、[来源/成员支持](../reports/research/510300_export_order_cause_increment_v1/source_gate.json)、[唯一模型协议](../reports/research/510300_export_order_cause_increment_v1/protocol.json)、[全部误差](../reports/research/510300_export_order_cause_increment_v1/同日同池五项误差比较.csv)、[全部机会](../reports/research/510300_export_order_cause_increment_v1/两模型全部自然高分机会.csv)、[结果](../reports/research/510300_export_order_cause_increment_v1/result.json)、[中文结论](../reports/research/510300_export_order_cause_increment_v1/外需条件联合评分_固定结果.md)、[年份/诊断补充](../reports/research/510300_export_order_cause_increment_v1/diagnostic_followup.json)。

### 5.27 E27固定外需机会目标缺口：正贡献集中，费用不足以解释缺口

- **范围与方法**：只复用E26全部32机会（主10/早22）、原报价/10000份/佣金/登记权益，原净标签精确复算；无新候选/训练/账户。主期原信号定义不变，入场年另列；目标口径使用独立保存快照，动态进度文件不是冻结模型输入。
- **收益贡献**：主期净均值+0.284198%，最大正标签2024-09-26为+9.307767%，占正标签和70.355945%。其余九行平均−0.718420%是事后描述，不能当删样本过滤器或真实账户利润；不证明单一政策造成全部上涨。
- **年份与费用**：原信号年2024/25为3/7，实际入场年2024/25/2026为3/6/1；2025-12-31信号入2026-01-05，残年另列。2024仍少于五自然机会，尚无风险账户完成周期。返佣金/取消原端点滑点分别保存，不能通过费用变化获得独立信号。
- **限定宽松上界**：只限原10窗口的完整持有/原端点退出，事后完美跳过非正原始收益、每次起点半仓、零费用/可分份额，忽略现实预算和权益现金限制；从2024首交易日至原最后退出2026-01-12全现金日计491日，242日年化上界3.600862%。这不是可执行账户/其他退出或任意策略上界，仍低于10%。
- **处置**：COMPLETED_FIXED_EXPORT_OPPORTUNITY_GAP_CONCENTRATED_AND_INSUFFICIENT。接受固定路径目标缺口诊断，不恢复E26、拼接旧收益或改出口字段/高分/持有营救。0新训练标签/拟合/完整账户/源码，夏普NOT_COMPUTED，完整目标未达。
- **接续**：原E28提案已执行，见第5.28/R86；原冻结提案不覆盖，执行谱系另存。下一E29见第10节，仅旧用途/来源定义初探提案。
- **证据**：[协议](../reports/research/510300_export_score_target_gap_diagnostic_v1/protocol.json)、[全部32标签/费用/年份](../reports/research/510300_export_score_target_gap_diagnostic_v1/全部32原机会_费用与入场年归属.csv)、[两种年份](../reports/research/510300_export_score_target_gap_diagnostic_v1/主期固定十机会_信号年与入场年.csv)、[完整诊断](../reports/research/510300_export_score_target_gap_diagnostic_v1/diagnostic_summary.json)、[结果](../reports/research/510300_export_score_target_gap_diagnostic_v1/result.json)、[中文结论](../reports/research/510300_export_score_target_gap_diagnostic_v1/固定十机会_目标缺口结论.md)、[必要核对](../reports/research/510300_export_score_target_gap_diagnostic_v1/保存结果必要核对.json)、[下一提案](../reports/research/510300_export_score_target_gap_diagnostic_v1/next_experiment.json)。

### 5.28 E28保存高分的信息来源：输入变化与模型更新已经分开

- **方法与范围**：17父文件、两模型各1,212评分日，全部2,422相邻保存对、67自然机会（外需32/原八变量35）。复用原树、原float32分支、成熟成员；9,688函数值保留旧/新输入×旧/新模型与交互，不重拟合。35控制叶复用E23实际成员缓存并核对。
- **9月26日实际来源**：出口仍为8月31日09:30已公布的8月原值、26日未换文/值；旧树对9/25及9/26输入均预测−0.747599%，当前树对两输入均+1.174437%。旧分数38.7776、当前93.0862；输入作用0、模型作用+1.922037个百分点、交互为浮点0。其他六日频字段虽改变，但两树在本例都对其变化不响应；不能称当天新增外需消息。
- **成熟叶变化**：当前叶69成员，最大正贡献月为2024二月、32.738676%，不是九月；最大单行已知标签为9/13信号/9/25退出、+7.775089%。一行新成熟标签是9/18信号、9/19入场、9/26退出、+6.597780%，其均值算术贡献+0.095620个百分点；保留成员分组/权重变化+1.826417个百分点，滚出直接贡献0。这是叶均值恒等分解，不是某行标签引发树重分组的因果证明。
- **全样本反例**：外需主期485对中406预测变化、384没有出口文更新；其间其他日频输入也可变化，不能一律归模型。主10自然机会只有3次出口更新；2025七例最大正贡献月均为2024九月、份额60.7254%—77.5449%，2024三例均为二月。静态外需状态仍可作为条件，变量出现在分支上不等于收到新消息。
- **接受与边界**：接受保存函数与成熟成员的有限信息来源解释，未识别外需/政策的经济因果、市场共识或独立有效策略；当前模型输入旧状态是事后反事实，不可回填旧决策。E26高分均值改善及五门失败、E27集中收益/机会密度/3.600862%有限端点上界全部保持。0新源/拟合/训练标签/完整账户/源码，夏普NOT_COMPUTED、目标未达、NO_VIEW/UNSET。
- **证据**：[协议](../reports/research/510300_saved_export_information_update_v1/protocol.json)、[全部相邻四格](../reports/research/510300_saved_export_information_update_v1/两模型全部2422对_输入模型四格分解.csv)、[全部67机会](../reports/research/510300_saved_export_information_update_v1/全部67自然机会_信息更新与叶来源.csv)、[完整成员算术贡献](../reports/research/510300_saved_export_information_update_v1/全部机会模型变化_成员算术贡献.csv)、[主期十例与最大正标签](../reports/research/510300_saved_export_information_update_v1/主期十机会与最大正标签_解释.json)、[分期完整摘要](../reports/research/510300_saved_export_information_update_v1/分期全部更新摘要.json)、[结果](../reports/research/510300_saved_export_information_update_v1/result.json)、[中文结论](../reports/research/510300_saved_export_information_update_v1/高分变化来自哪些已知信息_研究结论.md)。
- **接续**：原E29提案已实际初探并结束，E30同版本比率构成已完成，见第5.29—5.30/R87—R88；原提案保留，执行谱系另存。

### 5.29 E29银行业整体约束来源：实质不同用途，完整原历史输入未准入

- **旧用途与实际来源**：当前research/config/docs的3,269个py/json/md定位及原公司/融资桥协议，区分个股经营传导与行业监管约束；不是全历史repo无旧用途证明。2查询/8唯一直链，6原对象281042字节（模板/两脚本/两正文/xls）及2个经济根文件；两web403原样保留不重试，公开正文接口取得原件。
- **已证实断点**：2019起邮储纳入商业银行汇总；2024起资本按新办法、官方明确不直接可比。商业银行整体不能等同沪深300金融权重；发布正文资本口径不含外国银行分行。
- **实际版本缺口**：2025Q3发布正文当前显示2025-11-14 17:25:47，其链接的2025年度表当前显示2026-02-12 18:42:00，xls有Q4。当前附件及模板Last-Modified不倒填原季度；历史首版值/完整31季度支持未建立。九拟议单元格的原值/格式已绑定但不准入训练；Q4仅保存原件、不用于拟议字段。
- **净息差未知**：原表净利润明确本年累计，净息差没有期间定义；保持NOT_ESTABLISHED，不把另一行说明转移到息差或差分为单季冲击。原比率是比例/百分比格式，不混为百分点。
- **处置**：COMPLETED_INITIAL_BANK_REGULATORY_SOURCE_PROBE_FULL_PANEL_NOT_ADMITTED；接受有限源/定义事实，不准入全历史字段/模型。0新拟合/训练标签/账户/源码，夏普NOT_COMPUTED、目标未达；E26/E27/E28保持。
- **证据**：[协议](../reports/research/510300_bank_industry_constraint_source_probe_v1/protocol.json)、[旧用途](../reports/research/510300_bank_industry_constraint_source_probe_v1/prior_definition_review.json)、[已证实定义/断点](../reports/research/510300_bank_industry_constraint_source_probe_v1/字段定义与已证实断点.json)、[时钟](../reports/research/510300_bank_industry_constraint_source_probe_v1/发布正文与年度滚动表时钟.json)、[单元格/格式](../reports/research/510300_bank_industry_constraint_source_probe_v1/原表九个拟议字段单元格与格式.json)、[源门](../reports/research/510300_bank_industry_constraint_source_probe_v1/source_gate.json)、[结果](../reports/research/510300_bank_industry_constraint_source_probe_v1/result.json)、[中文结论](../reports/research/510300_bank_industry_constraint_source_probe_v1/银行业约束来源初探_研究结论.md)。

### 5.30 E30比率调整的第一层原因：两区间、同版本分子与分母

- **固定范围**：原同一2026-02-12版本2025Q1→Q2及Q2→Q3，资本净额/应用底线后的风险加权资产、不良余额/(正常+关注+不良)。先原显示0.00%精度核金额派生比率/原值残差，再按分子先、分母后作精确恒等分解；全部三季度输入与四组结果保留，0新源。
- **资本率**：Q1→Q2派生+0.295679个百分点（资本+0.547220、RWA−0.251541）；Q2→Q3派生−0.218722（资本+0.053617、RWA−0.272340）。后一区间资本净额仍增0.344185%、RWA增1.773121%，资本率下降不是资本净额下降。
- **不良率**：Q1→Q2派生−0.021671个百分点（不良余额−0.001058、贷款总额−0.020613）；Q2→Q3派生+0.024727（不良余额+0.038338、贷款总额−0.013611）。后一区间不良余额增2.570192%、贷款总额增0.897594%，分母扩张抵消部分比率上升，不是贷款总量缩水。
- **接受边界**：同表金额与原比率残差均在冻结显示精度内；恒等项不是唯一因果份额。资本发行/利润留存/扣除、贷款核销/转让/重分类/借款人恶化仍未识别。不能由一个区间推广指数规律；原季度版本/时钟未准入，净息差不差分，Q4不计算。
- **处置**：COMPLETED_SAME_VINTAGE_BANK_RATIO_NUMERATOR_DENOMINATOR_IDENTITIES；0新源/拟合/标签/完整账户/源码，夏普NOT_COMPUTED、目标未达。原下一E31已执行并结束，见第5.31/R89；原提案保留。
- **证据**：[协议](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/protocol.json)、[原金额与残差](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/全部三个季度_同表比率金额及舍入残差.json)、[全部四组恒等式](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/全部四组_分子分母固定顺序贡献.json)、[结果](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/result.json)、[中文结论](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/资本率与不良率为什么变化_同版本研究结论.md)。

### 5.31 E31原季度公开点值与净息差方法：有限问题已结束

- **固定来源**：2查询/2新唯一直链/2原件1,088,434字节；Q2正文doc1221429新取得，Q3复用E29。已知失败URL未重试，未扩大31季度搜集。
- **全部四点**：Q2正文显示2025-08-15 17:00:07公开不良率1.49%/资本率15.58%；Q3显示2025-11-14 17:25:47公开1.52%/15.36%。四点均与当前2026-02-12年度表的两位百分比显示一致。统计期末至正文日期46/45自然日，仅日期差，不认证精确首次公开/首收件，也不能认证当前附件高精度当时已公开。
- **版本边界**：两正文均链接同一后来滚动年度表；Q2没有独立当期附件。两固定查询内Q1独立正文未定位，不等于来源不存在，Q2环比不能回造Q1原版。2019总体与2024资本制度断点保持。
- **方法拒绝**：PBC2016/4全文24页、定义页13/14渲染核对；16家上市银行2007Q1—2014Q3/Wind报表贷款/负债利率代理，不是行业监管净息差。不得转移季度年化口径；监管净息差期间继续NOT_ESTABLISHED。HTTP修改日不等于论文原发布日期。
- **处置**：COMPLETED_BOUNDED_BANK_RELEASE_ROUNDED_VALUES_NIM_PERIOD_AND_FULL_VINTAGE_UNESTABLISHED；接受有限公开点值和方法不同的事实，完整原版本/31季度/全部原成熟池仍未准入。0新拟合/股票标签/账户/源码，夏普NOT_COMPUTED、完整目标未达，E26—E30原裁决保持。
- **接续**：原E32提案已实际核对，E33采集因现有访问到期停止，用户答复暂时没有；见第5.32—5.33/R90—R91。原提案保留，不把来源可读或原件存在当完整模型准入。
- **证据**：[协议](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/protocol.json)、[全部四点](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/全部四个公开点值_与当前年度表显示比较.json)、[时钟](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/两个季度正文时钟与年度表链接.json)、[方法裁决](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/方法论文口径裁决.json)、[来源门](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/source_gate.json)、[结果](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/result.json)、[中文结论](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/原季度公开值与净息差定义_研究结论.md)。

### 5.32 E32同DR007成交额来源：字段真实，完整原历史仍未准入

- **真实字段与单位**：提供方repo_daily文档doc256有amount万元，20200804样例DR007.IB/DR007成交8,751,142万元=875.1142亿元、weight2.1144%，与旧原响应同日利率一致。官方已存20260928 DR007有trdVol823.8757亿元、实际平均期限6.80日；不同日期不比较数值一致，名义7日不等于每笔恰好7日。
- **实质用途与边界**：旧单日日报含数量原件，旧用途为利率/原因观察；新拟议为每日数量对订单/融资/价格联合条件，不是FR-FDR价差、全市场R007/非银或央行操作量。当前代码/配置/文档有界定位不是全历史不存在证明。
- **固定历史请求反例**：官方接口查询文档唯一20200804示例日，HTTP200且lastDate回显，但records为空，窗口2026近期；不当历史数值、不记0、不换日重试。旧12提供方响应只有四字段/weight，没有amount。
- **处置**：COMPLETED_DR007_TRANSACTION_AMOUNT_IDENTITY_UNIT_AND_LIMITED_SAMPLES_FULL_PANEL_NOT_ADMITTED。2查询/2新直链/37,392字节；字段身份/单位接受，原完整成熟训练支持、决定日、金额时钟/首版未建立，0带凭据API/拟合/标签/账户/源码，夏普NOT_COMPUTED、完整目标未达。
- **证据**：[协议](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/protocol.json)、[字段/两样例](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/字段身份_单位_全部两个有限样例.json)、[用途](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/旧用途与本候选实质区别.json)、[请求](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/request_ledger.json)、[源门](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/source_gate.json)、[结果](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/result.json)、[中文结论](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/DR007成交额身份与历史支持_研究结论.md)。

### 5.33 E33成交额历史采集：现有访问到期，经济假设未检验

- **事前采集合同**：同API、DR007.IB/DR007原12年度分块20150105—20260814，旧四字段增加amount；仅复用既有有效运输、保留旧源码/响应。不扩支付/绕过到期，源门未过不拟合。
- **实际停止原因**：当前旧到期守卫在请求前拒绝临时访问，没有有效标准运输被选择；0提供方请求、0下载、0金额历史原行。STOPPED_HISTORY_ACQUISITION_EXISTING_TRANSPORT_EXPIRED_NO_PROVIDER_REQUESTS。不是成交额机制/预测失败，没有新增夏普。
- **用户及本地事实**：用户回复“暂时没有”。data/curated有界检查5个利率流动性表，2,905行DR007与2,904行发布表都没有amount；唯一量字段是2,104行央行七日逆回购操作量，不能代替市场成交额。这个字段清单不是全仓数据不存在证明。
- **接续状态**：E34仅真实有效访问更新后的收件提案，未注册；无活进程/会话、不把聊天当作业等待，不自动重跑旧到期配置。完整训练支持/金额公开时钟/版本仍NOT_ESTABLISHED。目标active，本目标轮E32/E33提供新证据；该访问阻碍首次实证，不据此立即把整个目标标blocked。
- **证据**：[协议](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/protocol.json)、[无凭据运输结果](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/现有运输可用性_无凭据.json)、[实际采集](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/acquisition_summary.json)、[用户答复](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/用户访问答复_事实.json)、[本地五表](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/本地其他利率流动性数据_有限字段清单.json)、[源门](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/source_gate.json)、[结果](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/result.json)、[中文结论](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/成交额历史采集停止原因_研究结论.md)。

### 5.34 补充旧用途：日历条件不是未用过的新输入（未注册新评分）

- **考虑的假设**：缺数据时能否用当时日期的月内/季内位置区分同资金状态。只作已有用途核对，没有登记新的E35或模型。
- **实际旧覆盖**：calendar_liquidity_timing_v1已有剩余自然天数、月/年相位、月末/季末条件，以及CALENDAR/PRICE/FULL的RIDGE和ET非线性模型；旧主方案状态COMPLETED_PRIMARY_TARGET_NOT_MET，原后选点值/独立性限制保留。
- **本轮裁决**：没有证明改为连续距月末/季末或重新联合使用具有实质不同。不能因访问不足重做旧日历窗口/参数。不是宣称日历永久无效；未新拟合或取得独立策略结论。
- **证据**：[有界定位/原字段/原结果绑定](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/本地日历候选_旧用途核对与不重开裁决.json)。

### 5.35 补充旧用途：制造业价格分项与订单原因已被历史研究观察

- **考虑的问题**：无DR访问时，是否可用现有采购/出厂价格分项配合订单作为不同条件输入。本轮只核已有用途，没有登记E35或新模型。
- **直接旧证据**：原manufacturing_price_transmission_daily仅加入出厂减购进价差，其固定8账户未获合格策略；historical_price_gap_causes又在51历史事件/31主期事件拆解售价走强、成本回落或两者同降，交叉订单确认，未建立独立优势。macro_transmission_context已联接原分项作解释。
- **本轮裁决**：尚未建立足够不同的原因信息，当前不因访问缺口把旧分项/订单解释换成日频或非线性模型重开；不声称所有价格分项组合已穷尽，也不永久否定制造业信息。单独共同价格水平未登记、未准入或拟合，不将想法写成已验证结果。
- **接续边界**：DR成交额仍是未检验假设；新拟合/标签/账户/源码修改0，目标未达。只有实质不同原因/用途或独立新证据可另立研究，原失败不调参营救。
- **证据**：[有界原用途/协议/结果绑定](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/本地制造业价格候选_旧用途核对与不重开裁决.json)、[用户答复及本地候选补充结论](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/访问缺口与本地候选_补充结论.md)。

### 5.36 E35官方成交概览：月度R总体及短菜单，不能补齐原DR日度输入

- **实际来源**：沿已存官方导航另登记1查询/4直链的公开目录核对，4原对象/40,146字节均HTTP200；旧PrDlyBltn和到期运输0重试。
- **直接事实**：成交概览为质押式回购月报，原数据有R007等十品种及合计，没有DR007，金额为亿元；展示2026-09，所列菜单2023-10—2026-09，共36月。固定2020-08-04不在菜单，未发历史POST。菜单不是全档案不存在证明。
- **统计与时钟**：备注为每月月初更新上月数据；机构统计是买卖两方向交易总和，不能叫净融资/净流入。备注及当次head.ts不能建立历史精确首次公开/首版。
- **裁决**：作为原DR007日度输入未准入，不换总体/频率、缩原训练或挑日期；DR经济假设仍未检验。新增拟合/标签/账户/源码0，夏普NOT_COMPUTED。该访问阻碍是第二目标轮遭遇，但另有E36实际独立行动，不据此判整个目标blocked。
- **证据**：[协议](../reports/research/510300_dr007_official_history_catalog_probe_v1/protocol.json)、[四请求](../reports/research/510300_dr007_official_history_catalog_probe_v1/request_ledger.json)、[实际身份/菜单/备注](../reports/research/510300_dr007_official_history_catalog_probe_v1/实际品种_月度菜单_机构双边口径.json)、[源门](../reports/research/510300_dr007_official_history_catalog_probe_v1/source_gate.json)、[结果](../reports/research/510300_dr007_official_history_catalog_probe_v1/result.json)、[中文结论](../reports/research/510300_dr007_official_history_catalog_probe_v1/官方成交概览为何不能补齐DR历史_研究结论.md)。

### 5.37 E36采购量联合评分：源支持完整，固定增量拒绝并完成失败归因

- **不同信息及准入**：采购量描述企业报告的投入采购涨跌普遍程度，区别于订单、价格差、产成品库存；此前有限定位只有15个月CPI背景解释，未证明全历史绝无同用途。复用132当月原报告，全部1,212原评分日及原成熟成员完整、同订单时钟、旧15月采购值一致；原八变量/标签/成员不变，first_vintage=false。
- **固定表达**：原八变量+采购量水平/月变；原504日/成熟252、深3叶60/seed20261001、岭alpha10/clip5、五日费用标签、80线、自然不重叠、两期及五门保持。源门通过后单独登记模型；2,424新拟合，基准0重拟合，无新网络/标签/账户/参数搜索/源码修改。
- **主期结果**：485日树MSE比原纠错八变量增加0.608269%；也差于同池均值。8自然机会平均净五日−1.023325%、胜率37.5%，原9笔为−0.928734%；0新增自然日期，1笔实际路径用采购。五门仅路径使用通过，REJECTED_FROZEN_FIXED_PURCHASE_QUANTITY_CONDITION_NO_PRIMARY_INCREMENT。
- **较早期完整保留**：727日，27自然机会平均净五日−0.049841%，原26笔+0.028983%；采购胜率48.15%，4笔实际路径使用字段。两期岭/树/均值及全部原五档全部保存，不选较好模型或时期。
- **失败归因**：主期预测改变41日、分数改变70日，其中29日预测未变但训练排名改变。保留全部8原端点和登记日权益，零费用算术界平均仍−0.729214%，费用不是唯一原因；它不是新策略/账户。按原自然机会入场年为2024=1、2025=7，不能称已运行账户周期；仓位/风险限制效果NOT_COMPUTED，不能通过放仓或改费用营救。
- **接续**：评分失败封存，不反号、删列、改窗口/期限/标签/时期营救；完整账户NOT_RUN、夏普NOT_COMPUTED、目标未达，采购动机与指数收益因果未识别。E37只是全部24主期统计月的原文可知原因核对提案，未登记、0新网络/模型/账户；原因不足保留MIXED/UNKNOWN，结束有限问题后不变成参数搜索。DR的E34仍仅有效访问/可靠同身份历史出现后接续。
- **证据**：[来源协议](../reports/research/510300_purchase_quantity_cause_increment_v1/protocol_source.json)、[源门](../reports/research/510300_purchase_quantity_cause_increment_v1/source_gate.json)、[模型协议](../reports/research/510300_purchase_quantity_cause_increment_v1/protocol.json)、[误差](../reports/research/510300_purchase_quantity_cause_increment_v1/同日同池五项误差比较.csv)、[全部机会](../reports/research/510300_purchase_quantity_cause_increment_v1/两模型全部自然高分机会.csv)、[固定结果](../reports/research/510300_purchase_quantity_cause_increment_v1/result.json)、[中文结果](../reports/research/510300_purchase_quantity_cause_increment_v1/采购条件联合评分_固定结果.md)、[失败归因](../reports/research/510300_purchase_quantity_cause_increment_v1/固定失败归因_信号费用与账户边界.json)、[E37提案](../reports/research/510300_purchase_quantity_cause_increment_v1/next_experiment.json)。

### 5.38 E37采购调整原因：24月原报告已核，直接原因未知

- **固定问题及来源**：2024-01—2025-12全部24当月原报告，当前采购量及7同期问项；原源、E36模型/标签/评分和成熟成员不改。人工逐月核制造业正文及排除发布者/方法附注后的采购主题，没有因解释缺失删月。
- **完整事实**：采购指数13月上升/11月下降；低于50时上升5月、下降9月，高于50时上升8月、下降2月，其余五格0。上升仍低于50的5月为2024-01/07/10、2025-05/11，改善与扩张分开，不当采购实际吨数/金额。
- **原因裁决**：24月直接采购原因全部UNKNOWN；同期需求/生产/库存/价格/交付只保留竞争解释线索。2025-02官方解释春节后生产恢复，其被解释对象是生产，不能改写成采购动机。配送时间高于50对应加快、合成PMI反向。
- **评分时点**：24个新统计月首次进入E36保存评分时仅2024-08统计月（决定日2024-09-02）实际树路径使用采购。2024-07采购升而订单/生产降，2024-09采购降而订单/生产升；跨日其他输入/成熟池/排名共同变化，采购对分数差因果效应NOT_IDENTIFIED。不能把此1/24计数替代E36全主期41日路径使用。
- **限制与运行**：原发布日期到可用日为后来保存重构，first_vintage=false。2024-01其他7问项月变UNKNOWN，不补第25份原报告。分数CSV未读未来净收益列；前一步原输入parquet曾全列载入但未使用未来标签值，不按收益选月/解释。0新网络/标签/模型/拟合/账户/源码；E36拒绝不变，夏普NOT_COMPUTED、完整目标未达、NO_VIEW/UNSET。
- **接续**：本24月问题结束。E38已按不同官方解读原文的固定前三月问题完成，3月补采购直接官方叙述、1月有产能下界，但原因输入未准入，详见第5.39。不因未知原因改组合或重试E36。DR访问缺口第三目标轮仍存在，但当前完成E37实质进展，根目标连续阻塞/无进展轮数0，不据单条源缺口伪称全项目阻塞。
- **证据**：[协议](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/protocol.json)、[24月人工裁决](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/人工裁决_全部24月采购原因.json)、[完整事实](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/全部24月采购事实与竞争解释线索.csv)、[全部九格](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/采购水平与方向_全部九格.csv)、[原制造业正文](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/全部24月制造业正文_逐月核对.md)、[结果](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/result.json)、[中文结论](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/采购为何调整_24月历史核对结论.md)、[E38来源提案](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/next_experiment.json)。

### 5.39 E38同期官方解读：不同来源事实成立，量化原因输入未准入

- **固定来源与实际执行**：2024-01/02/03三月，登记后有限旧用途定位与2官方域名检索、3原页一次GET；全部HTTP200、原响应219,621字节。原正文/日期/当前收件与三月身份绑定，无换月、无带凭据请求或重试旧访问。
- **接受的不同事实**：1月大型制造业调查中产能利用率≥80%的企业比例>七成，精确比例/样本数UNKNOWN；3月官方把采购增加与生产、需求回升共同联系，属于直接官方解释而非已识别因果作用；竞争/需求困难比重仍较高仅定性。2月春节/返乡解释对象为制造业活跃度，并非采购直接原因。
- **来源门和边界**：未形成三月同定义独立原因精确原行及完整原训练支持；不将>七成填成70%，不将较高编码成精确比重。1月只有成文日期2024-01-31，日内发布时间UNKNOWN；2/3月页面显示3/1及3/31 09:30，URL日期分别3/18及4/11，日期差异分开记录，首版未认证。模型原因字段准入0。
- **冻结与目标**：E37原数据发布页24UNKNOWN的范围结论保持，新解读补充不覆盖旧原件；E36失败、成本归因不重开。0新标签/模型/拟合/账户/源码；夏普NOT_COMPUTED、目标未达、NO_VIEW/UNSET。前三月来源问题结束，不自动补全年、调查字段或模型。
- **下一方向选择**：工业利润原65报告方向已冻结毛筛拒绝，成本/利润率和回款已有明确源码用途，不当新信息。当前限定research/config用电原字段0命中，未证明全历史绝无同用途。E39已按实际第二产业用电量的固定三月原问题完成，3月原单月成立、1/2月未建立及跨公布值115衔接差未知；详见第5.40。不能当DR金额或CSI300利润。
- **证据**：[协议](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/protocol.json)、[原收件](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/原官方解读收件.json)、[原正文](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/全部三月制造业解读_逐段原文.md)、[人工裁决](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/全部三月来源信息_人工裁决.json)、[来源门](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/result.json)、[中文结论](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/官方解读能补什么原因信息_固定三月结论.md)、[E39提案](../reports/research/510300_official_pmi_interpretation_cause_source_qualification_v1/next_experiment.json)。

### 5.40 E39实际用电量：原周期及衔接差已核，三单月源门未过

- **真实来源执行**：固定2024-01/02/03，2官方域名检索/3一次GET（全国1—2月通报、全国3月通报、同3月必要目录），全部HTTP200/24,260字节；首次GET前保存一次目录导航补充，原字段/全国总体/月份/评判和总请求上限不改。
- **接受原行**：3月全社会7942、第二产业5421亿千瓦时，原同比7.4%/4.9%，4月17日发布。3月20日原1—2月合并量为全社会15316、第二产业9520，原同比11.0%/9.7%；4月17日原1—3月累计23373/15056，原同比9.8%/8.0%。累计与单月分别存，不当两份原单月。
- **新测量问题**：四组之和每原周期均等于全社会，但当前季度累计减去前次两月累计与当前3月原单月之和，全社会/第二产业各剩115亿千瓦时。原文未说明差异原因，UNKNOWN；不判某一修订原因，不用累计差额拼1/2月单月。
- **源门和原因**：固定能源局路线只建立3月原单月，1/2月原单月UNKNOWN，不证明所有公开源都没有。两页只有显示日期、日内时间UNKNOWN、首版未认证；产出/气温/工作日/能效/行业结构调整贡献未识别。完整原训练支持/模型准入未建立。
- **停止与目标**：固定三月本来源问题结束，0预测字段准入/新标签/模型/拟合/账户/源码，夏普NOT_COMPUTED、目标未达，NO_VIEW/UNSET。E36模型拒绝、E37原数据页24未知、E38不同解读事实/源门保持；不把省级、发电量或年度量替代。
- **接续**：E40已按不同原发布对象的固定2024-01/02两检索执行，未定位原文、0GET/原单月，结束该路线且不证明全源没有。E41另问实际原区间公告状态，完整原同池支持未过；详见第5.41/5.42。原E39三月/全国/单月定义、115差异未知及DR访问条件保持。
- **证据**：[协议](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/protocol.json)、[运输导航补充](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/02_同三月原文目录导航路由补充登记.json)、[原收件](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/原公开收件_全部.json)、[原周期字段](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/全部实际原周期与用电字段.json)、[三单月资格](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/固定三月原单月资格.json)、[衔接残差](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/不同公布值衔接残差_不重构缺失单月.csv)、[源门](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/result.json)、[中文结论](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/用电量原周期与公布差异_固定三月结论.md)、[E40提案](../reports/research/510300_electricity_physical_quantity_source_qualification_v1/next_experiment.json)。


### 5.41 E40中电联固定索引路线：两查询空，原单月未建立

- **固定实际执行**：2024-01/02同全国原单月/同比，先登记父件后2个中电联域名查询，均返回空索引；未获得可请求原文地址，实际原GET/字节/原行均0，不猜地址或重发到期访问。
- **裁决**：COMPLETED_FIXED_CEC_INDEXED_QUERY_ROUTE_NO_NEW_ORIGINAL_MONTH_ROW；这是该两次索引路线结束，不能证明中电联或所有原通报不存在。E39两月单月UNKNOWN、三月原事实及115衔接差原因未知不改，经济有效性未检验。
- **进展分类**：E40本身没有新增数值输入，不把空检索/提案记成改善收益。后续E41是不同实际公告区间的信息问题，并非把累计行改名为单月或重开E36评分。
- **边界及证据**：0新标签/模型/拟合/账户/源码，夏普NOT_COMPUTED、NO_VIEW/UNSET；[协议](../reports/research/510300_cec_national_single_month_electricity_source_qualification_v1/protocol.json)、[完整检索](../reports/research/510300_cec_national_single_month_electricity_source_qualification_v1/官方域名两次查询_完整结果.json)、[源门](../reports/research/510300_cec_national_single_month_electricity_source_qualification_v1/source_gate.json)、[结果](../reports/research/510300_cec_national_single_month_electricity_source_qualification_v1/result.json)。

### 5.42 E41实际用电公告状态：原字段建立，完整原同池支持未过

- **不同信息及冻结**：每次公告优先原最新单月、否则原最新累计，保留原区间同比/月份数/公告年龄；累计不冒充单月。表达选择已见E39/E40源结构，明确属于开发选择、非独立验证；未看股票未来收益。只取固定2018—2025原统计期，原八字段/标签/1,212日/成熟成员不改，超过90自然日未知。
- **实际来源**：旧首页实际用动态列表；只沿实际DOM和脚本规则追加同域目录JSON导航，先登记路由且原15导航/100报告预算不变。新5导航GET+52原报告GET，51报告HTTP200、共882,669字节，99原周期提取记录、51次公告状态（E43核明4条嵌套段落完全重复，唯一95原周期，原记录/源门未改）：44单月/1合并累计/6年度，原发布日期2021-01-20—2026-01-17。末次2025年年度在2026年公布，不进入截至2025-12-31原评分日。
- **完整支持**：原需要2018-12-06—2025-12-31的1,711独立日期；已知1,155/未知556。512日此前未有本路线原对象；2022-11原单月只有第二产业量4789亿千瓦时、未给其同比，23日未知；已列2024-10新原文HTTPS失败、21日阻断旧值回退。早期727评分日当前691已知、主期485日当前464已知，但全部1,212评分日同一成熟池均含未知，0完整模型日。不丢原日/成员、不把同比未知填0或转累计营救。
- **时钟与原因**：网页日内缺失UNKNOWN，保守研究假设公布日23:59:59+08才可用；目录日/URL日/原显示日/当前收件分开、首版未认证。能耗代理与物理事实接受，天气/工作日/能效/行业结构原因、指数回报方向及因果作用均未识别。
- **实现事实保留**：旧收件键http_status未被采集器status_code识别，2已有原对象冗余各GET一次，计入52，不伪称零请求复用。年度段首发布动作日期与实际年度、缺单月同比候选被误略后回退累计两处按原正文/原优先规则纠正；初稿和纠正回执均保存，未改源门或模型参数。所有新对象每URL仅一次，不重试失败、未关闭证书/登录。
- **停止及目标**：COMPLETED_ORIGINAL_ELECTRICITY_PUBLICATION_STATES_SOURCE_SUPPORT_NOT_COMPLETE。完整来源门未过，联合模型/账户NOT_RUN；0新标签/拟合/账户/源码，夏普NOT_COMPUTED、目标未达。E36模型拒绝、E37原因范围、E39/E40终态与DR用户暂无访问保持。E42已核明索引转载直接同比−1.0%但原根/原钟未准入，E43完成全部原构成描述，详见第5.43/5.44；均非新模型或原同池准入。
- **证据**：[协议](../reports/research/510300_electricity_original_publication_state_v1/protocol.json)、[实际目录绑定](../reports/research/510300_electricity_original_publication_state_v1/03_实际动态列表地址绑定_请求前.json)、[全原报告](../reports/research/510300_electricity_original_publication_state_v1/全部原报告收件与复用.json)、[99原区间](../reports/research/510300_electricity_original_publication_state_v1/全部实际原统计区间.json)、[逐公告状态](../reports/research/510300_electricity_original_publication_state_v1/每次原公告实际状态_不拼单月.json)、[1,711日期输入](../reports/research/510300_electricity_original_publication_state_v1/原1711日期实际公告状态与未知.csv)、[完整原池支持](../reports/research/510300_electricity_original_publication_state_v1/全部1212评分日原同池输入支持.json)、[源门](../reports/research/510300_electricity_original_publication_state_v1/source_gate.json)、[结果](../reports/research/510300_electricity_original_publication_state_v1/result.json)、[中文结论](../reports/research/510300_electricity_original_publication_state_v1/用电原公告状态_完整同池资格结论.md)、[E42提案](../reports/research/510300_electricity_original_publication_state_v1/next_experiment.json)。


### 5.43 E42原2022-11直接同比：政府索引陈述成立，原模型来源未准入

- **固定执行**：登记一个月/全国二产原量4789锚点后2官方查询；同政府对象2公开读取（自行GET一次200/492字节仅应用外壳，web.open一次给70行、标注last year抓取的索引全文）。没有追加原URL重试/脚本/API或新根搜索。
- **直接历史陈述**：门户明确来源中电联，原11月第二产业4789亿千瓦时、同比−1.0%，非自行比去年量/累计差额产生；接受带层级标记的描述事实，不冒称当前HTTP完整原件。转载显示2022-12-19 17:31:54，CEC原根及原钟UNKNOWN、首版未认证；若以所显示转载钟作研究假设，最早12/20决定，不能倒填NEA12/15。
- **版本及结构**：同二产/全社会月量一致，第一产业89与NEA88差1亿千瓦时，原因UNKNOWN，不拼源。同转载工业−0.9%、制造业−0.2%、高技术及装备+3.4%、消费品−6.4%，不同统计口径/原权重和上游原因未知，不算制造子组贡献或指数收益。
- **裁决**：COMPLETED_FIXED_NOV2022_INDEXED_REPORTED_YOY_FACT_ORIGINAL_MODEL_SOURCE_NOT_ADMITTED。精确描述事实接受，0新模型源准入，E41原556未知/源门终态不改。固定源问题结束，0新标签/拟合/账户/源码、夏普NOT_COMPUTED、目标未达、NO_VIEW/UNSET。
- **证据**：[协议](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/protocol.json)、[原GET与壳](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/request_ledger.json)、[同原对象可读全文](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/同原对象工具可读全文_完整结果.json)、[原同比层级事实](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/同全国单月原同比_索引转载事实.json)、[版本/时钟差](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/不同原发布量与时钟_不拼接.json)、[源门](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/source_gate.json)、[结果](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/result.json)、[中文结论](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/固定原同比与来源层级_研究结论.md)。

### 5.44 E43全部原电量构成：测量反例成立，指数收益未测试

- **完整原范围**：只本地E41原99提取记录和E42另根索引案例，0新网络/模型；所有原期、缺值、版本及合计残差保留。4完全重复来自HTML外层p/嵌套p，原显示字段/日期一致；不删原99，补唯一95原周期（单月44/累计45/年度6），原51公告状态及E41源門不变。此前“99区间”应理解为99提取记录，非99唯一周期。
- **全部原单月频数**：44期、43可比；40期全社会/二产同正，2期同负，1期总正二产负，1期二产同比未知。原反例2022-07为全社会+6.3%、二产−0.1%、三产+11.5%、居民+26.8%。原累计45/年度6皆同正；不能选累计掩盖单月反例或以总体频数作收益概率，来源覆盖不全/版本不独立。
- **不同根案例不混入**：E42索引转载2022-11全社会+0.4%、二产−1.0%、三产+3.5%、居民+4.2%，另列而不填NEA原未知。制造业子组异向仅描述，第二产业/工业/制造业及高技术/消费品各自口径不等同，权重未知。
- **原显示构成演示**：只同原量与同比代数隐含基数，非取得去年原量或认证舍入。额外转载案例二产约−0.712百分点、三产+0.546、居民+0.506、第一+0.112，推算合计+0.453%，原报+0.4%；当前四组和比原总量多1亿千瓦时，差异保留、不冒称精确因果贡献/模型字段。全部99原显示量/同比近似构成与误差保存。
- **接受/未测试**：否定“全社会同比正足以同周期二产同比正”的测量充分条件，接受来源内构成可异向。股票收益方向/增量、天气/工作日/能效/真实结构原因未识别；已见案例后开展历史开发描述，非独立验证，不表示所有时期都相反或该因子已无交易价值。
- **停止与目标**：COMPLETED_ORIGINAL_COMPONENT_DIVERGENCE_AND_MEASUREMENT_SUFFICIENCY_COUNTEREXAMPLES。原1212日/556未知源门不重开；0新标签/模型/拟合/账户/源码，夏普NOT_COMPUTED、目标未达。E43结束时E44仅不同同原份额/原期长的自然公告源合同提案，未登记/执行；其后实际E44—E46见第5.45—5.47；不把份额替代原同比，不缩旧训练成员/目标。
- **证据**：[协议](../reports/research/510300_electricity_component_state_description_v1/protocol.json)、[全部99记录](../reports/research/510300_electricity_component_state_description_v1/全部99原周期_四组状态与显示值近似构成.csv)、[95唯一期](../reports/research/510300_electricity_component_state_description_v1/同原95唯一周期_原提取99行全部保留.csv)、[重复澄清](../reports/research/510300_electricity_component_state_description_v1/同原周期去重与嵌套段落澄清.json)、[全部反例](../reports/research/510300_electricity_component_state_description_v1/全部原反例_完整同周期字段.json)、[显示构成/因果边界](../reports/research/510300_electricity_component_state_description_v1/显示数值分解与因果边界.json)、[结果](../reports/research/510300_electricity_component_state_description_v1/result.json)、[中文发现](../reports/research/510300_electricity_component_state_description_v1/原用电四组异向与总量代理边界_历史发现.md)、[E44提案](../reports/research/510300_electricity_component_state_description_v1/next_experiment.json)。


## 5.45 E44原构成与自然公告源合同：已见完整支持成立

- **假设/登记**：同原第二产业量/全社会量、居民量/全社会量和原期長是不同结构信息；不是E41精确同比替代，不改旧日频样本。源/时钟合同先登记，不读股票未来值。
- **实际来源**：2官方查询/1新原转载GET，200/15177字节；国家能源局福建监管办公室全国稿给2024-10总量7742、二产5337、居民932亿千瓦时。同稿1—10月累计另存、发电量不用作分母。稿内原根11/20与转载11/21 15:14分开；取11/21EOD研究可用钟，11/22收盘决定；未重试旧失败URL、未倒填。
- **完整已见事件**：52原根全部保留，其中51决定日2021—2025、1公告2026年在范围外；全部51当前及原成熟池完整，12预热保留/39可评分（早17/主22）。份额来自同段原显示量，合计残差保留，原全年/累计/单月身份不变。
- **边界/接受**：PASS_KNOWN_ORIGINAL_EVENT_COMPOSITION_SOURCE_AND_CLOCK_CONTRACT；仅已见目录，未证明所有月份档案完整/历史首版，EOD为开发假设，上游原因UNKNOWN。源门通过后另登记E45，不表示金融目标达标。E41原556未知与原门保持，0标签/拟合/账户/源码。
- **依据**：[协议](../reports/research/510300_electricity_composition_event_source_contract_v1/protocol.json)、[源门](../reports/research/510300_electricity_composition_event_source_contract_v1/source_gate.json)、[52全部事件](../reports/research/510300_electricity_composition_event_source_contract_v1/全部52原根公告_同原构成与原八状态_无收益标签.csv)、[51同池支持](../reports/research/510300_electricity_composition_event_source_contract_v1/全部51评估公告_相同成熟池支持.json)。

## 5.46 E45十一变量同自然事件评分：固定增量拒绝

- **固定方法**：E44源门后另登记原8/联合11树及两岭/均值，过去504交易日/成熟12事件，树深2叶6/岭10/原10000份5日费用标签/80线/连续不重叠，39次×4=156拟合；同决定日及训练IDs，无参数搜索/新标签/源码或账户。
- **实际主期2024—2025**：22评分；原八8高分机会净均−0.414290%，联合7次−0.760923%/胜率28.57%；入场年2024/25为4/3，新日期0，新构成进入实际高分路径0/7。较早17评分，两树各1相同机会+1.485227%，全部保留，不择早期救主要期。
- **误差/裁决**：联合树MSE0.009129135510，原八0.009211409184，同池均值0.008143812229；五门true/false/false/false/false。REJECTED_FROZEN_FIXED_COMPOSITION_JOINT_SCORE_NO_PRIMARY_INCREMENT，账户NOT_RUN、夏普NOT_COMPUTED、目标未达。更小MSE点值不等于交易增量；不换岭/阈值/窗口/费用营救。
- **执行事实**：完整模型/评分/分层/机会/result先写成功，最后终端摘要NumPy布尔序列化失败；未重新拟合，E46从保存结果核对156预测身份通过。当前版本/覆盖/开发独立性限制都保留。
- **依据**：[事前协议](../reports/research/510300_electricity_composition_joint_score_v1/protocol.json)、[结果](../reports/research/510300_electricity_composition_joint_score_v1/result.json)、[全部自然机会](../reports/research/510300_electricity_composition_joint_score_v1/全部固定高分自然机会.csv)、[保存路径](../reports/research/510300_electricity_composition_joint_score_v1/全部39次保存模型与实际决策路径.json)、[完整结论](../reports/research/510300_electricity_composition_joint_score_v1/用电构成联合评分与失败原因_研究结论.md)。

## 5.47 E46原失败归因：信号失效、费用与集中程度分开

- **固定范围/方法**：只E45全部39保存模型与51含预热事件、17两树模型机会记录（9不同自然入场日）、22主要期；无拟合/删极端/改预测。保存树/岭重走、原滑点/tick/佣金/登记权益身份核对；零费用端点只解释界，不是新策略。
- **实际结果**：156保存当前预测和17费用标签身份通过。联合主期7次平均预测+4.908497%/实际净−0.760923%；零费毛界仍−0.460097%，平均费用影响0.300826百分点。7路径仅资金价差、新构成0；原信号失败，费用不是唯一原因，账户未跑不能归因仓位/ES。
- **完整叶组证据**：5次高分当前6成员叶均含E44_037，2024-09-24开盘至10/08开盘5交易日净标签+40.062709%，占各叶绝对标签和87.20%—91.25%，对应预测+6.25%—7.01%。全部成员保留，只说明模型估计极端依赖，不认证政策因果。
- **裁决/接续**：COMPLETED_SAVED_SIGNAL_COST_AND_LEAF_CONCENTRATION_ATTRIBUTION_NO_RETRAINING；E45拒绝保持、账户/夏普未算/目标未达。E46结束时E47仅后选原政策时钟解释提案，现E47/E48实际结果见第5.48—5.49，分别确认不同信息身份并保留预告未知，不据最大收益挑可交易规则。DR无账户状态已答复，不重问/过期访问。
- **依据**：[归因协议](../reports/research/510300_electricity_composition_failure_attribution_v1/protocol.json)、[结果](../reports/research/510300_electricity_composition_failure_attribution_v1/result.json)、[全部原叶成员](../reports/research/510300_electricity_composition_failure_attribution_v1/全部叶组原成熟成员_不删极端.csv)、[全部费用解释](../reports/research/510300_electricity_composition_failure_attribution_v1/全部17原机会_原费用身份与零费解释界.csv)、[E47提案](../reports/research/510300_electricity_composition_failure_attribution_v1/next_experiment.json)。


## 5.48 E47极端成熟标签中的原政策身份与时钟：已完成，因果收益未识别

- **固定问题**：E44_037决定9/23 16:00、入9/24 09:30、出10/08 09:30，原资金价差0.2569百分点且资金源钟9/23 09:30已知。对象在最大标签归因后选择，后选解释，不是独立预测检验。
- **原事实**：复用证监会实录102499字节及9/27实施稿7089字节。10项信息：7项政策幅度/住房/两股票工具方向首次明确在09:19:36段，3项额度与再贷款设计11:42:50段。同稿重复段完整保留、首次明确取值，0新请求/拟合/收益列读取。
- **三钟边界**：现场实录段是重建代理，精确到每句话和网页首次上线钟未知/首版未认证。现场代理下7项盘前，而严格网页证据下开盘可知UNKNOWN；11:42额度不倒填开盘/原决定。9/27实际降到1.5%是9/24已宣布目标的实施，不第二次计20bp新意外。
- **接受/未识别**：COMPLETED_ORIGINAL_POLICY_SEGMENT_CLOCK_BOUND_TO_EXTREME_MATURE_LABEL_NO_CAUSAL_RETURN_IDENTIFIED；原短期资金水平、计划政策率、机构资产流动性、股票融资及住房约束是不同信息，不把计划额度叫股市净流入。完整预期/先前预告/单政策收益因果未识别；旧政策链/股票工具用途不重包装模型。原E45拒绝、目标未达保持。
- **依据**：[协议](../reports/research/510300_extreme_leaf_policy_information_clock_v1/protocol.json)、[十项原段钟](../reports/research/510300_extreme_leaf_policy_information_clock_v1/十项原政策信息_现场段钟与网页钟分开.csv)、[原决定源状态](../reports/research/510300_extreme_leaf_policy_information_clock_v1/原决定时资金与订单源状态_无新未来收益.json)、[结果](../reports/research/510300_extreme_leaf_policy_information_clock_v1/result.json)、[完整结论](../reports/research/510300_extreme_leaf_policy_information_clock_v1/政策信息与会前预告_原决定时钟研究结论.md)。

## 5.49 E48原会前预告：有限实际路线结束，前置公开钟未准入

- **方法/实际**：另登记固定9/24会议、原9/23 16:00决定；6原文件/旧用途后，2官方查询未定位预告。沿旧年目录实际href新GET英文专题200/105312字节，内容为会后实录/报道，没有预告公开钟；其实际中文现场href一次521/732字节。2查询/2源件预算结束，无猜地址、521重试、模型。
- **裁决**：COMPLETED_FIXED_OFFICIAL_PREVIEW_ROUTE_ORIGINAL_PREDECISION_NOTICE_CLOCK_NOT_ADMITTED。原会前预告及具体政策事前共识/精确预期差UNKNOWN，不能将“会议9时”当预告首次上线或由未找到原件断言市场毫无预期/全档案不存在。不把来源未准入叫经济机制失败。
- **接续**：旧国内AAA3年信用差及两腿分解已有实际源码/协议结果，不重包同输入。E49仅提出原非制造业新订单/商务活动/业务活动预期的不同条件用途核对与完整同池源合同；旧15个月已有提取，字段不是未见，模型须另登记。
- **目标/状态**：0新标签/拟合/账户/源码，净夏普NOT_COMPUTED、目标未达、active/根阻碍0；DR特定无访问保持，E45拒绝/原E41源门等保持。无活作业，不是等待后台。
- **依据**：[协议](../reports/research/510300_policy_press_preview_information_clock_v1/protocol.json)、[原专题收件](../reports/research/510300_policy_press_preview_information_clock_v1/专题GET_实际收件.json)、[521收件](../reports/research/510300_policy_press_preview_information_clock_v1/中文现场GET_实际收件.json)、[结果](../reports/research/510300_policy_press_preview_information_clock_v1/result.json)、[旧信用用途](../reports/research/510300_policy_press_preview_information_clock_v1/国内信用利差候选_旧用途不重开.json)、[E49提案](../reports/research/510300_policy_press_preview_information_clock_v1/next_experiment.json)。


## 5.50 E49非制造业需求与经营预期：实际原缺失，固定源门未过

- **固定方法/旧用途**：在原模型前登记同当月非制造业新订单、商务活动、业务活动预期三列；实核旧15月背景、旧14月制造业PMI预期差四账户及有限源码，不冒称新字段或完整历史未试过。当前同八状态三维用途与旧描述/规则有区别，旧四账户拒绝保持。
- **原源/完整成员**：主要价格源116份不能叫132份，补充既有同原订单源132个月原文。95个月三列齐、36早期制造单独公告无非制造业、2022-12只有活动/新订单而未发布经营预期。全部1212原评分日和1711当前/成熟成员并集保持，实际需要2018-11—2025-12；原时钟与制造新订单一致。
- **缺口量化**：2022-12原预期缺失对应15原决定日，原成熟池复用影响519评分日、7500次成员使用；693完整日也不拿来缩期拟合。旧15月44已公布值及1原NULL相同。首次解析把部分已知列全记NULL和时钟项误用整体布尔的实际错误已修正，原记录/输出保持，经济字段/源门/日期未变。
- **裁决**：REJECTED_SOURCE_GATE_UNKNOWN_ORIGINAL_MEMBERS_PRESERVED_MODEL_NOT_RUN。接受用途与源缺口事实，未检验经济收益假设；不删第三列、前后填充或用后来表回溯救当前三维表达。0网络/标签/拟合/账户/既有源码改动。
- **依据**：[协议](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/protocol.json)、[逐字段原文](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/全部132月非制造业原文支持_逐字段保留.json)、[结果](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/result.json)、[时钟分项补正](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/源门分项事实补正.json)、[结论](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/非制造业三字段_旧用途与原训练支持.md)。

## 5.51 E50在手订单与生产：不同业务兑现问项原源通过

- **经济区别**：固定制造业在手订单、生产原扩散值，不新增差分/相减/符号；前者是未完成工作变化普遍程度，非实际订单存量，后者是当期活动普遍程度，非最大产能/实物产量。区别于投入采购、服务/建筑需求或经营预期，不是把E49删列换名。
- **旧用途与源**：旧背景已有两列、旧PMI规则已有生产构成；有限实际核对未见原八同池加两列条件预测，不证明全历史组合未试过。132原月报告、旧15月30值、全部1212评分日及1711原当前/成熟身份与原订单源/钟齐备、0未知。第一次单层表头解析误记生产缺失，按原实际PMI双层表头修正，原失败输出保留，未改问项/时钟/模型计划。
- **限定准入**：PASS_EXISTING_CURRENT_MONTH_FIELDS_ALL_ORIGINAL_SCORE_AND_MATURE_MEMBERS_FIRST_VINTAGE_UNAUTHENTICATED。只接受原当月已存官方网页/公开钟的历史重构比较，首版首次收件未认证，非独立验证/市场预期/因果识别。源后另登记E51唯一模型，源阶段0拟合/标签/账户/网络/源码。
- **依据**：[源协议](../reports/research/510300_backlog_production_cause_increment_v1/protocol_source.json)、[旧用途](../reports/research/510300_backlog_production_cause_increment_v1/实际旧用途与不同问项核对.json)、[原源门](../reports/research/510300_backlog_production_cause_increment_v1/source_gate.json)、[源结果](../reports/research/510300_backlog_production_cause_increment_v1/source_result.json)、[源结论](../reports/research/510300_backlog_production_cause_increment_v1/在手订单生产_原来源与不同用途.md)。

## 5.52 E51在手订单与生产联合评分：一次固定比较拒绝

原504日窗口/252成熟下限、深3叶60/seed20261001、岭alpha10及原标准化、原五日费用分红标签/80高分线/连续自然非重叠规则全保持。原八控制和同池均值只复用；新增1212树和1212岭参考拟合，只有一个经济候选，未调参数或择模型晋升。

| 原固定时期 | 日数 | 原八自然机会 | 新十变量自然机会 | 原平均净五日 | 新平均净五日 | 新胜率 | 新字段高分实际路径 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021—2023较早 | 727 | 26 | 26 | 0.028983% | 0.068358% | 46.153846% | 6 |
| 2024—2025主期 | 485 | 9 | 9 | −0.928734% | −0.452154% | 44.444444% | 4 |

主期树MSE相对原八恶化0.571516%，相对同池均值也更差；固定五项门只过新增日期和实际新增字段路径，误差两门与净机会均值门失败。主期四新自然决定日期为2025-04-30、05-13、05-21、05-29，旧四日期完整保存，不以替换去掉亏损记录。两期、五档、两岭和均值比较均保留。

终态REJECTED_FROZEN_FIXED_BACKLOG_PRODUCTION_CONDITION_NO_PRIMARY_INCREMENT；账户NOT_RUN_FIXED_PRIMARY_INCREMENT_GATE_FAILED、净夏普NOT_COMPUTED。较早均值稍好、比原八负均值接近0或参考岭数字不能替换主期准入；不改持有/分数/成员/字段/方向/模型/成本/仓位营救。0新网络/标签/账户/既有源码修改；完整目标未达。

依据：[模型协议](../reports/research/510300_backlog_production_cause_increment_v1/protocol.json)、[全部同池误差](../reports/research/510300_backlog_production_cause_increment_v1/同日同池五项误差比较.csv)、[全部自然机会](../reports/research/510300_backlog_production_cause_increment_v1/两模型全部自然高分机会.csv)、[必要保存核对](../reports/research/510300_backlog_production_cause_increment_v1/保存结果必要核对.json)、[结果](../reports/research/510300_backlog_production_cause_increment_v1/result.json)、[固定结论](../reports/research/510300_backlog_production_cause_increment_v1/在手订单生产联合评分_固定结果.md)。

## 5.53 E52原失败诊断：同值状态、费用与九月重叠标签

- **完整原机会**：保留九主期机会，原佣金/滑点/取整/登记日权益标签逐笔一致。平均净−0.452154%，原开盘同端点零费用算术界−0.157862%，费用影响差0.294292百分点；去掉费用也负，因此费用不是唯一原因。该界不是可执行策略或完整账户。
- **实际模型行为**：485主期日中预测变84、分数变136，52日只排名变而预测未变；四高分路径均使用生产，在手订单用于这些高分0次。九原叶/707次成员使用/195不同原成员的均值和成熟钟均复核，没有重新拟合或删除标签。
- **同值与集中**：四生产高分全是2025-04生产49.8/在手订单43.2；每个原成熟叶前五大正标签都是2024-09-19—25决定的重叠五日窗口，彼时生产也是2024-08的49.8。这五标签占各叶带符号收益合计85.185170%—89.711337%，占比不是因果概率。此处确认同值分组受到重叠行情影响，不能推断两次生产调整原因相同或低生产稳定看涨；九月政策的单因果回报未识别、E47/E48未知保持。
- **频率/账户边界**：候选九自然机会按入场年全部2025，2024为0；这是信号机会、非完整账户周期。完整账户、风险/权重约束影响及夏普均未计算，不通过放仓/降费救回。E51原拒绝保持。
- **下一**：该原E53提案后已实际登记并完成，E54补两原官方共同解释，见5.54—5.55及实际完成接续；旧提案作为谱系保留，不再称当前未运行。原E51拒绝不改。
- **依据**：[归因协议](../reports/research/510300_backlog_production_cause_increment_v1/固定失败归因协议.json)、[归因结果](../reports/research/510300_backlog_production_cause_increment_v1/固定失败归因_信号费用与原叶.json)、[九机会费用界](../reports/research/510300_backlog_production_cause_increment_v1/全部9主期机会_原端点费用与叶解释.csv)、[同值与贡献](../reports/research/510300_backlog_production_cause_increment_v1/四生产路径同值与原重叠标签贡献.json)、[完整诊断](../reports/research/510300_backlog_production_cause_increment_v1/固定失败归因_信号费用与原叶.md)、[E53提案](../reports/research/510300_backlog_production_cause_increment_v1/next_experiment.json)。


## 5.54 E53同生产49.8的原联合信息：不同背景成立，具体原因原公告未知

固定2024-08与2025-04两原月、全部四原生产路径和原九机会背景，先登记再读原信息，0网络/股票标签/拟合/账户。生产水平同49.8，原生产月变分别−0.3与−2.8，制造业新订单48.9/49.2，服务业新订单46.8/45.9、建筑业新订单43.5/39.6；供货配送49.6/50.2的高值意味着更快交货，不能解释为时长增加。原当前月与同次公告上一月完整保存，不用后面修订回填。

原统计数据公告只有活动变快/变慢的描述，没有明确天气、贸易等具体原因，因此此阶段仍UNKNOWN。2025-04-30、05-13、05-21、05-29决定前有原四月联合值，不等于掌握完整预期差。旧E38仅2024-01/02/03官方解释，未覆盖这两原月；原指数贸易季度研究实际2019Q2，旧E53提案误写2025Q2已保存明确补正，不再作为相同原因研究证据。原E47/E48显示钟、预览未知与旧拒绝不变。

终态COMPLETED_EQUAL_PRODUCTION_VALUE_DIFFERENT_JOINT_CONTEXT_SPECIFIC_CAUSE_NOT_IDENTIFIED；接受同值异背景事实，单因果股票收益与历史首件版本未识别。

依据：[实际结果](../reports/research/510300_equal_production_different_cause_information_v1/result.json)、[原两月完整信息](../reports/research/510300_equal_production_different_cause_information_v1/原两月全字段及变化.csv)、[旧用途及年份补正](../reports/research/510300_equal_production_different_cause_information_v1/实际旧用途与年份范围补正.json)、[历史结论](../reports/research/510300_equal_production_different_cause_information_v1/同生产值的不同原信息_历史结论.md)。

## 5.55 E54两月原官方共同解释：原因有区别，不识别单项收益贡献

先登记固定两官方检索、两实际原网页GET、0重试，再按真实返回URL收件；共2个200对象、113314字节。原E38采购三个月有限来源问题仍结束，本研究是E52后验两同值月份的明确原因信息，非重开旧模型。

| 原统计月 | 官方明确作用对象 | 明确共同原因 | 仍有并存原因/限制 |
|---|---|---|---|
| 2024-08 | 制造业整体PMI走弱 | 近期高温多雨、部分行业生产淡季等 | 高耗能行业偏弱；价格解释还有需求不足及原油/煤炭/铁矿石价格波动 |
| 2025-04 | 制造业整体PMI回落 | 前期制造业较快增长形成较高基数、外部环境急剧变化 | 需求不足和部分大宗商品价格下降等并存；关税战评论不等于生产−2.8由关税独自造成 |

两个原页元数据均显示公布日09:30，四月还可见分钟级时刻；原九月标签决定和原四月/五月决定均晚于相应显示钟。但实际收件在2026-10-02，历史首发收件/首版UNKNOWN，不冒称严格实时认证。明确理由针对整体PMI，不能量化生产单项降幅、CSI300暴露或股票收益因果贡献。

接受不同整体共同解释和显示钟事实；不制作互斥“天气/关税”盈利标签、不推完整市场意外。只有两后验月份，原1711成员完整原因面板NOT_ESTABLISHED，没有训练字段准入。0模型/收益标签/账户/源码改动。

官方原页：[2024年8月解读](https://www.stats.gov.cn/xxgk/jd/sjjd2020/202408/t20240831_1956159.html)、[2025年4月解读](https://www.stats.gov.cn/sj/sjjd/202504/t20250430_1959518.html)。本地：[协议](../reports/research/510300_equal_production_official_interpretation_source_v1/protocol.json)、[全部共同原因及未识别项](../reports/research/510300_equal_production_official_interpretation_source_v1/全部两月官方共同原因_明确与未识别分开.json)、[结果](../reports/research/510300_equal_production_official_interpretation_source_v1/result.json)、[解释与可知性](../reports/research/510300_equal_production_official_interpretation_source_v1/相同49点8_原官方共同原因与可知性.md)。

## 5.56 E55服务业和建筑业订单：两不同统计行业的原来源通过

固定两列“服务业新订单原扩散指数”“建筑业新订单原扩散指数”，不同于原制造业订单、E49非制造业总活动/总订单/预期三维或E51在手工作/生产。原值代表相应统计行业需求变化普遍程度；服务业并非纯居民消费，建筑业并非纯房地产，不是沪深300盈利、权重或订单实物金额。

实际有限旧用途核对只见背景、汇总及原公告解析，没有发现本次原八同池两行业需求条件预测；不能由有限搜索断言全历史未试过。复用原132月网页的当月非制造业正文明确数值，96月两列齐、36早期制造单独公告无行业正文；原1212评分日及1711当前/成熟身份需要的月份全部齐、未知0，与原订单源钟完全一致，两后验目标月四值也一致。不得用后月历史表回填；首件版本未认证保留。

源门PASS_EXISTING_CURRENT_MONTH_SECTOR_ORDER_FIELDS_ALL_ORIGINAL_SCORE_AND_MATURE_MEMBERS_FIRST_VINTAGE_UNAUTHENTICATED，只准入这个历史重构比较。源阶段0网络/收益标签/拟合/账户/源码；随后另登记E56唯一模型。

依据：[源协议](../reports/research/510300_service_construction_demand_cause_increment_v1/protocol_source.json)、[原132月来源](../reports/research/510300_service_construction_demand_cause_increment_v1/全部132月原服务建筑订单绑定.json)、[全部成员支持](../reports/research/510300_service_construction_demand_cause_increment_v1/全部1212原日及成熟成员支持.csv)、[源结果](../reports/research/510300_service_construction_demand_cause_increment_v1/source_result.json)、[来源与不同用途](../reports/research/510300_service_construction_demand_cause_increment_v1/服务建筑订单_原来源与不同用途.md)。

## 5.57 E56服务/建筑订单联合评分：固定主期没有合格可交易增量

原504日窗口/最少252成熟标签、深3/叶60/seed20261001、岭alpha10、原标准化/clip、原五日费用分红标签与80高分线均保持。原八控制和同池均值只复用；唯一经济候选新增1212树及1212岭参考拟合，共2424，无网格/新标签/账户/既有代码修改。较早期及主期、五档和五误差对照全部保存。

| 原固定时期 | 配对日数 | 原八机会 | 两行业新机会 | 新平均净五日 | 新胜率 | 新行业高分实际路径 |
|---|---:|---:|---:|---:|---:|---:|
| 2021—2023较早 | 727 | 26 | 33 | −1.108423% | 33.333333% | 19 |
| 2024—2025主期 | 485 | 9 | 13 | −0.063346% | 46.153846% | 6 |

主期树MSE0.001310566273，相比原八树0.001286772804恶化1.849081%，也高于同池均值0.001169689967。五门为F/F/F/T/T：两项误差门、净机会均值为正门均失败，只有新增自然日期及新增字段实际路径通过。参考十变量岭主期MSE0.001167527956略低于均值，较早期仍更差，完整结果保留，不看完改为岭主方案。

终态REJECTED_FROZEN_FIXED_SERVICE_CONSTRUCTION_DEMAND_NO_PRIMARY_INCREMENT；账户NOT_RUN_FIXED_PRIMARY_INCREMENT_GATE_FAILED、净夏普NOT_COMPUTED。有四个新增原自然日期且整体负均值较原八接近0，只支持这次具体模型行为，不建立合格策略或独立行业因果效用。不得改窗口、阈值、年代、方向、字段、费用或仓位救本表达。

依据：[固定模型协议](../reports/research/510300_service_construction_demand_cause_increment_v1/protocol.json)、[全部同池误差](../reports/research/510300_service_construction_demand_cause_increment_v1/同日同池五项误差比较.csv)、[全部自然机会](../reports/research/510300_service_construction_demand_cause_increment_v1/两模型全部自然高分机会.csv)、[实际结果](../reports/research/510300_service_construction_demand_cause_increment_v1/result.json)、[固定结果报告](../reports/research/510300_service_construction_demand_cause_increment_v1/服务建筑订单联合评分_固定结果.md)。

## 5.58 E57拒绝后的必要归因：费用、早期负信号、年频率和集中来源

保留两期全部46个候选自然机会；主期13笔原费用标签、原13保存叶1030次成员使用/292不同原成员的均值和成熟钟全部一致。没有重训或新标签。485主期日预测变69、分数变133，其中64日只排名改变；六条新增字段高分路径分别服务1、建筑5，不把路径频次解释成因果贡献。

| 原期 | 原机会 | 原平均净五日 | 同端点原开盘零费算术界 | 费用/报价差的平均影响 |
|---|---:|---:|---:|---:|
| 2021—2023 | 33 | −1.108423% | −0.808623% | 0.299800百分点 |
| 2024—2025 | 13 | −0.063346% | +0.239266% | 0.302613百分点 |

本候选主期零费转正，与E52在手订单/生产的零费仍负不同，不能把前一失败诊断复制过来。费用影响是主期净均翻负的重要算术解释；较早期即使零费也负、预测误差门仍失败，所以降低成本不足以证明跨期成立。零费只是固定端点解释界，不是可执行策略或完整账户。

原九主期日期全保留、删除0，新增四日期是2024-09-26、2025-01-22、02-12、05-21。四新增净收益算术合计+7.535105百分点，其中9/26决定的单窗口+9.307767%，其他三笔合计−1.772662百分点；该单窗口占带符号增量合计123.525379%，非概率、独立四次或政策因果，不能只运行新增四笔。全部13均值仍为负。

按自然入场年计数2024=3、2025=10，不满足每个完整年五次的必要机会支持；早期2021=10、2022=16、2023=7。信号次数不是受风险/现金/份额约束的完整账户闭合周期；账户权重和约束效果、净夏普均NOT_COMPUTED。E56固定拒绝保持。

依据：[归因协议](../reports/research/510300_service_construction_demand_cause_increment_v1/固定失败归因协议.json)、[全46费用解释界](../reports/research/510300_service_construction_demand_cause_increment_v1/全部46两原期机会_费用解释界.csv)、[实际归因](../reports/research/510300_service_construction_demand_cause_increment_v1/固定失败归因_信号费用与原叶.json)、[全部13保留的增量贡献](../reports/research/510300_service_construction_demand_cause_increment_v1/全部13保留_新增4机会贡献仅描述.json)、[归因报告](../reports/research/510300_service_construction_demand_cause_increment_v1/固定失败归因_信号费用与原叶.md)。

## 5.59 E58共享DR007新指针：实际仍旧利率源，没有金额

先登记有限本地新线索检查，沿TECH.R118/119直接冻结原源指针核对，0网络/认证/模型/标签/账户。技术用途2026-10-02新登记，但源实际2026-09-02收件；2905行主表只是DR007.IB/DR007/weight，加权平均百分比，schema无amount。原12年度块请求均只需ts_code/trade_date/repo_maturity/weight；实际原2015/2024/2026三响应字段也无amount，收件哈希一致。合计四schema检查达到登记上限，停止当前线索。

COMPLETED_BOUNDED_NEW_SHARED_POINTERS_TO_OLD_WEIGHT_ONLY_SOURCE_NO_AMOUNT。接受既有利率源与新用途的区别，不把weight推为金额、不使用不同R007、央行操作量或别日trdVol替代。DR007数量假设仍未被检验/否定；完整金额成员、公开时钟及首版UNKNOWN/NOT_ESTABLISHED，模型NOT_ADMITTED。用户暂无有效访问、E33到期运输停止保持，无新账户问题、轮询或请求。

依据：[协议](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/protocol.json)、[主源schema与原导航](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/实际主源schema与原收件导航.json)、[三响应schema](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/原三响应schema_未读金额值.json)、[结果](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/result.json)、[关闭当前线索的结论](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/共享DR007新线索_仍只有利率无成交额.md)。

## 6. 共享仓库的另一研究线：技术点位与仓位信息

这是此次检查 repo 新确认的重要现状，来源是保存文件，不是当前聊天记忆。主入口：[全部判断与收益夏普进展](../reports/research/510300_trade_quality_and_sharpe_summary_20261001/全部判断与收益夏普进展.md)。该任务把年交易次数作为软目标、增加实际净 `p×B>1` 要求，并独立登记了前瞻比较；**这些口径不替换第 3 节主任务合同。**

| 已保存压力成本账户 | 时期 | 年化 | 夏普 | 回撤 | 完整周期 | 实际净 p×B |
|---|---|---:|---:|---:|---:|---:|
| A：保留原仓位强弱 | 2015—2019 | 1.84% | 0.438 | 6.75% | 22 | 0.611 |
| B：第二既有候选 | 2015—2019 | 2.86% | 0.696 | 4.18% | 15 | 0.704 |
| A：保留原仓位强弱 | 2020—2026-09-30 | 3.99% | 1.217 | 2.75% | 32 | 1.073 |
| B：第二既有候选 | 2020—2026-09-30 | 3.77% | 1.175 | 2.75% | 31 | 0.989 |

保留仓位大小比旧二值点位有历史改善，但该模型较早期仍弱、收益集中、经历过反复选择。统一正目标为 50% 的受控实验在两时期两费用下均降低收益和夏普，已拒绝。B 未形成跨期替代 A 的改进。RSI/NR7 也没有形成已接受的互补机会。

反过拟合诊断读取 8 个旧账户；近期 A 的前五笔盈利占完成周期净利润 76.2%。最近训练的 790 行持仓记录来自 20 个自然成熟周期，不能当成 790 个独立样本。已知 583 设置、599 个已评价来源版本只覆盖部分历史；正式 DSR/PBO 因全局可比试验范围不足而 `NOT_COMPUTED`。

最新持仓路径诊断分解四份 A 账户、54 个完成周期，没有新拟合：真实退出比最后持有收盘的费用后清算标记更有利，且没有普遍延期卖出；亏损更多发生在持有中段。盈利周期也主要靠持有中段挣钱，因此这个事后分解不能变成“统一提早退出”。含股息隔夜损益与原价隔夜损失必须分开。

另一任务的 [前瞻权重比较协议](../reports/research/510300_point_weight_forward_v1/protocol.json) 已登记首决定 2026-10-08、首执行 10-09、唯一终点第 1,008 个交易日；当前真实新账户日和前瞻完成周期均 **0**，合成测试不算市场证据。本次不调用推进入口、不改协议、不启动后台任务；本任务继续历史范围。

## 7. 数据源、已经确认的限制与必须继承的纠错

| 数据/源 | 本地事实入口 | 能回答什么 / 不能回答什么 |
|---|---|---|
| 510300 日线与总回报 | [candidate_features](../reports/research/510300_original_frozen_sse_completion_20260925/candidate_features.parquet)、[分红表](../data/reference/510300_dividends.csv) | 不复权价格用于成交，分红独立记权益；不能把除息跌幅当卖压。各研究截止日不同，不假定全部更新到今天 |
| 指数、估值、成分权重 | [指数盈利总体研究](../reports/research/510300_historical_index_earnings_population_v1/result.json)、[权重来源检查](../reports/research/510300_factor96_weight_float_source_v1/result.json) | 调样、盈利分母和权重日期要分开；当前成分/当前财务不能倒填历史。原权重锚只覆盖 20/1,212 个主期要求日，不等于完整历史权重已准入 |
| 国内订单/PMI | [pmi_new_orders](../reports/research/510300_macro_dynamic_reframe_v1/inputs/pmi_new_orders.parquet)、[PMI 预期研究](../reports/research/510300_historical_index_pmi_expectation_v1/result.json) | 按 available_at 合并；首发文字、统计月与发布日期分开。缺共识时不能用环比变化替代预期差 |
| 银行间资金与政策利率 | [滞后资金输入](../reports/research/510300_factor96_funding_relief_v1/daily_features_lag1.parquet)、[操作记录](../reports/research/510300_macro_dynamic_reframe_v1/inputs/operation_rate_records.parquet) | 资金利率、央行价格、税期/跨季与需求混合；操作观察日不自动等于最早政策公告日 |
| 两市融资 | [有效纠正合同](../config/510300_financing_source_correction_20240808_v1.json) | 总量不是沪深300专属需求；余额变化、买入、偿还、直接还款不等同于净卖股。若范围不一致，恒等式残差保持未解释 |
| M2/贷款预期 | [104 月共识选择](../reports/research/510300_money_consensus_increment_v2/inputs/104个月共识选择.csv)、[贷款来源结果](../reports/research/510300_multidim_loan_surprise_score_v1/source_result.json) | 保留发布前报告、单位、显示精度、月值/YTD 区分；“--”是缺失，不是零。事后取得的历史报告不是当年实时接收回执 |
| 公募份额/净值 | [26 月来源解析](../reports/research/510300_multidim_public_fund_share_score_v1/parsed_sources.json)、[官方目录](https://www.amac.org.cn/sjtj/tjbg/gmjj/) | 同一 PDF 的本月/上月值；份额变化含发行、申赎、拆并、再投与分类；股票基金含 ETF，混合基金非纯股票。不能推导同额二级市场净流入 |
| 公募时钟与断点 | [公募协议](../reports/research/510300_multidim_public_fund_share_score_v1/protocol.json) | 取目录日期与 PDF 文件名日期较晚者，日末可用；这仅保守版本线索。2025-11 起不再列开放/封闭类别，当前样本止于 2025-10；不可直接拼接 |
| 美国利率/政策意外 | [美债阶段比较](../reports/research/510300_historical_index_us_rates_policy_v1/result.json)、[FOMC](../reports/research/510300_historical_fomc_transmission_v1/result.json) | 名义、TIPS、通胀补偿、期限/流动性因素不能混为单一贴现率；下载的 USMPD 历史版本不是当年首版。中国开盘前还有国内/贸易新信息 |
| 季度长端国债供给 | [E6固定16季](../reports/research/510300_multidim_treasury_long_supply_cause_score_v1/source_gate.json) | 10/20/30年同范围gross，公布日末代理、首版未认证；不是市场意外/净流动性，主期16/20次增幅0；固定预测增量拒绝，见R64 |
| 金融期货持仓类别 | [E7固定2023](../reports/research/510300_cftc_treasury_position_source_qualification_v1/result.json) | 原10年043602/52周，7实际迟报已定位、45普通公布日未核；完整源门未过，净空不表示建仓目的 |
| 美联储准备金构成 | [E8同表10字段](../reports/research/510300_fed_reserve_causes_single_release_v1/result.json) | 同周均供给/吸收恒等式，持券减少可与准备金增加并存；仅单例事实，RRP总项非纯ON RRP，未评分 |
| CPI/就业事前预期 | [预期锚](../reports/research/510300_historical_index_expectation_anchor_v1/result.json)、[就业研究](../reports/research/510300_historical_index_employment_surprise_v1/result.json) | 周前调查不是发布前一刻全市场共识；修订、人数、工资和参与率分开；缺失月份保留 |
| 北向/ETF 份额 | [MSCI 结果](../reports/research/510300_historical_msci_inclusion_demand_v1/result.json)、[跨 ETF 原拒绝](../reports/research/510300_cross_etf_forced_flow_binary_screen_v1.json) | 第三方历史净买入无法分离被动/主动及指数归属；份额代理不是强迫成交或未来必买金额；旧周度规则的 T+2 口径不能改 |
| MSCI 官方事件文件 | [来源目录](../reports/research/510300_historical_msci_inclusion_demand_v1/) | 7 份官方网页/PDF内容已取得事实；原 PDF 直接下载受 403 限制，不能谎称七份原始二进制全部落盘；五次实施只来自两条路线 |
| 公司财务/公告/发行回购 | [96 因子项目状态](../reports/research/510300_factor96_program_v1/status.json) | 合并/母公司、归母/扣非、首次披露/更正、授权/实际实施、A/H 股及发行主体要分开；T11 日期代理与严格 first_seen 版本不能混称 |
| 历史自由流通与人民币源 | [当前主配置](../config/510300_existing_data_training_mandate_v1.json) | 自由流通凭据记录为过期，人民币官方 requester-pays 路由未准入；没有新源/访问条件不反复请求同一失败端点 |
| 国股银票六个月曲线 | [E16结果](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/result.json) | 原主体/001/6M和2022-01-04两列通过；2025年末该次无数据，完整历史/方法/公布时钟未建立，评分未运行；不替代旧AAA三年债券、不以统计日作公开日 |
| 银行代客结售汇分项 | [E20结果](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/result.json) | 当前2021—2025的60原月值完整，2026版到各原公布内容/时刻未建立；证券投资不等于沪深300资金，未注册评分/账户 |
| 企业贷款加权价格 | [E17结果](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/result.json) | 当前16同名表值/月份/单位及页面历史时刻完整；7当前PDF晚生成，数值版本到原时钟未建立，不回填/不拟合，不称综合融资成本或纯信用冲击 |
| PCF/IOPV/同步报价的早期状态 | [初始压力修复状态](../reports/research/510300_pressure_recovery_v1/status.json)、[旧暂停回执](../reports/research/510300_forward_host_diagnosis_20260922/user_pause_receipt.json) | 957旧观察/8日期仅为10月1日早期快照，不能代表最新源覆盖；第13节已有181日历史三流。新数据仍未证明同步估值/成交，M1/M2收益未计算；历史定向下载与暂停的计划采集分别管理 |

**融资纠错必须沿用：**2024-08-08 深市融资买入/余额被旧第三方副本写成 0；官方确认融资买入 220.88 亿元、融资余额 6,649.54 亿元。只纠正这一行，派生输入影响 60 个日期、62 个单元。原文件未覆盖；M2 同参数修正后的两类树高分入退出保持一致，13 个改变的评分字段属于线性参考，不需要重算原账户。

- 有效融资副本：[margin_corrected_20240808.parquet](../reports/research/510300_multidim_financing_composition_v1/data_correction/margin_corrected_20240808.parquet)，SHA256 `37b16c8bf60dcc224bac7109d747bbf62c3402527ccfe27196cf3dea9163976b`。
- 有效日频输入：[daily_inputs_corrected_20240808.parquet](../reports/research/510300_multidim_financing_composition_v1/data_correction/daily_inputs_corrected_20240808.parquet)，SHA256 `c6be6dcf35dbff5f162980115de43acc498a15c0bd116b0f003d009c7ec712e7`。
- [原始纠错回执](../reports/research/510300_multidim_financing_composition_v1/data_correction/correction_receipt.json) 与 [同参数重算结果](../reports/research/510300_multidim_financing_composition_v1/data_correction/recalculation_result.json)。旧失败不因数据纠错自动复活。

此前已证实月度M2评分/普通日迁移对照已纠正，后续贷款/公募读取纠正输入；2026-10-02 E21进一步完成原日频八/十变量两模型的同参数融资纠错，见第5.21/R79及[纠错结果](../reports/research/510300_daily_score_financing_input_correction_v1/result.json)。两套主期高分仍为负；旧输入、模型与失败保存不覆盖。**整个factor96家族仍没有全部按纠正源重新运行的证明。** 原件first_vintage=false，局部数值纠正不为所有历史结果背书。

能源局实际公告状态补充：E41保留99原统计区间/51公告的单月、累计、全年身份，原首版未认证。原1,711日期有556未知，完整同池门未过；详见[原状态表](../reports/research/510300_electricity_original_publication_state_v1/原1711日期实际公告状态与未知.csv)，不把当前输入较齐误称完整训练可用。


宏观新增来源限定（E53—E58）：服务/建筑订单只接受原当月行业扩散值与完整原1212/1711支持；两目标月官方解释只是整体共同原因，未覆盖完整原因面板或认证首件。E58新共享来源指针实为既有weight利率源，没有amount，不能当历史资金数量。实际对象及单位/时钟见5.54—5.59，旧各源限制不变。

## 8. 宏观评分实验的冻结口径

八项基础字段为：订单水平、订单月变、资金利率与政策利率差、融资五日净变化、融资买入活跃度、指数趋势强度、短长波动比、成交活跃度。它们通过条件树共同决定预测，**不是八个因子等权相加**。

| 口径 | 日频八/十变量 | 月度 M2/贷款/公募 |
|---|---|---|
| 非线性模型 | DecisionTreeRegressor，深度 3，每叶至少 60，seed=20261001 | 同类树，深度 2，每叶至少 6，seed=20261001 |
| 训练 | 过去 504 个交易日，至少 252 个成熟日标签；逐日更新 | 过去 504 个交易日内的事件，至少 12 个成熟事件；随事件更新 |
| 对照 | 同池均值、同字段标准化岭回归 alpha=10 | 同池状态树、增量树、同字段岭回归、均值；贷款以九变量为对照 |
| 标签 | 决策后下一交易日开盘入，入场后第 5 个交易日开盘出；10,000 份、原压力费用与分红权益 | 同左，按各事件 known_at 映射入场，不复制月值制造日频独立样本 |
| 分数 | 预测净收益在当次训练池拟合值中的中位百分位，0—100 | 同左；**不是上涨概率，也不是跨模型可直接相加的绝对强弱** |
| 展示 | 固定五个 20 分档，空档保留；原高分线 ≥80 且预测净收益 >0 | 同左；保存全部预测与机会，已有机会结束前不重叠入场 |
| 评价 | 2024—2025 主段，2021—2023 较早背景；原历史已被研究 | 公募来源较短，前 12 次预热，不伪造更长来源 |

各实验账户入口不完全相同，应读自己的协议。公募事前要求“联合树 MSE 不更差 + 高分机会均值为正 + 至少新增一个入场”三项同时成立；本轮只满足第三项。不能看完结果再换成较容易的入口。信号诊断的 10,000 份标签不等于 20 万元受风险约束账户。

最低必要检验保留五项即可：信息不晚于决定时点、训练标签已成熟、来源/定义一致、成交费用与权益计算正确、所有事件与反例保留。不要把整个旧审计体系照搬成每次研究的前置工作，也不要为追求速度去掉这五项。

## 9. 宏观联合评分分支尚未解决的问题与优先级

| 未解决问题 | 当前证据 | 下一步应补的内容 |
|---|---|---|
| 因子同值背后的原因 | E53同49.8时原变化与行业订单不同；E54官方整体共同解释也不同 | 对完整原月份建立多原因并存的原信息矩阵；只有两后验月份不能准入模型，整体PMI解释不能当生产/股票单因果 |
| 可交易增量 | E56主期13机会净均−0.063346%，树误差增1.849081%；参考岭主期稍好、较早更差 | 固定表达封存，不后验选岭/近期/新增四笔；新原因信息需要实质区别、来源支持和新协议 |
| 费用与信号 | E57主期零费+0.239266%、净−0.063346%；早期零费−0.808623% | 保留同端点界及全部正负机会；费用不是全时期解释，不把界叫策略，账户约束影响仍未算 |
| 年频率与集中 | 原入场年2024=3、2025=10；新增四贡献主要来自9/26原单窗口 | 不凑年五、不择赢家阶段；信号非完整账户周期，不用一窗口宣称稳定作用 |
| 原来源缺口 | E49原2022-12预期确未公布；E54两原因页首件版本未知 | 原未知与身份保留，不删列/月份、后填或制作互斥纯原因标签 |
| 独立性与标签重叠 | E57原13叶1030次使用仅292不同成员；大量历史已研究 | 不当独立机会，不删大正标签再训；所有结果仍历史开发资料 |
| DR007真实市场量 | 用户暂无访问；E58共享新指针只有旧weight，四schema路线关闭 | 等真正不同可靠原金额或有效访问状态变化；不重复问、不重试到期运输/已穷尽目录 |
| 共享项目状态 | 技术、盘口、宏观协议与授权独立 | 保护最新其他线原文，不能迁入252日/软频率/单独代码授权或用其失败覆盖本线 |

E1—E58各有限问题已结束。本轮E53不同原信息、E54两直接原因源、E55完整行业源、E56固定新比较/E57归因/E58新线索关闭均为实际进展，根目标active、同阻碍计数0，无活作业或后台轮询；金额单项未检验不等于根目标受阻。下一E59仅完整原月份共同原因来源提案，NOT_REGISTERED/NOT_RUN。20万元主/2万元费用对照、净夏普1.2/年化10%/回撤10%/完整年五周期与原仓位/ES/缺口合同全部保留。完整目标和独立验证均未达。

## 10. 宏观联合评分分支的下一步实验

### E1：三项月度联合评分的共同收益来源归因

状态：**COMPLETED_SHARED_EVIDENCE_ATTRIBUTION，2026-10-02完成**。原提案先登记36项输入和hash，再通过项目Python标准输入分析；没有修改源码、重新拟合、重新回测或取得新数据。协议、完整分析程序、输入合同快照与全部结果在[实验目录](../reports/research/510300_multidim_shared_evidence_attribution_v1/)。

- 固定输入：纠正后的M2月度目录、贷款目录、公募目录；各自读取评分CSV、输入parquet、saved_models.json与原高分；四份账户仅读取已有ledger/cycles。
- 范围：保存联合树47+40+14=101次，主期20/20/14；不扩日频模型，不选择最赚钱叶。
- 方法：按保存training_months恢复成员，以float32值和保存阈值重走路径；分别计算最大正标签占带符号收益合计、正收益合计的比例。前者抵消时可超过100%，均不是概率或因果贡献。
- 去重：entry/exit相同为同一价格观察；部分持有间隔重合另算；09-18→09-25与09-25→10-09是相邻窗口。标签成熟、决定、入退出时间逐项对齐。
- 产出：全部101次来源、761条叶成员使用、高分训练贡献、两两窗口关系、四账户周期重合、一张时间线及中文结论。101评分对应61个不同窗口，叶成员对应79个不同历史窗口；原18高分对应12窗口，主期13高分对应8窗口。这些不同窗口数仍不是独立样本数。
- 结果：M2/贷款主期五笔同路径、同叶、同预测且只用融资五日净变化；早期最大正标签为2024年2月，后期才使用成熟九月标签。公募高分路径只用混合份额月变，三次实际窗口均亏损，与前两组分离。
- 保存账户：两资金下主期各五周期、25持有间隔全部重合；20万元实际份额仍不同，不将账本利润当成相同或相加。完整期7/8周期中6同窗。
- 必要复核：101次叶均值最大误差9.89e-17、分数误差7.11e-15；训练成熟、已记录源时点、原高分数量与36项输入均核对。它不认证历史首版、因果作用或独立有效性。
- 处置：归因问题已回答，结束本实验；不删九月样本重训、不拼账户、不改分数线、不把删最大盈利后的账户当新策略。完整目标未完成。

直接入口：[机器结果](../reports/research/510300_multidim_shared_evidence_attribution_v1/result.json)、[归因结论及时间线](../reports/research/510300_multidim_shared_evidence_attribution_v1/共同收益来源归因_研究结论.md)。

### E2：财政性存款“原因状态”增量实验

状态：**已完成并拒绝，2026-10-02 03:25**，直接结果见第5.4节及R59。先完成104原文字段预检和旧财政失败区别，再在新分组收益比较前注册。只加一项财政累计比例，两个共同月份控制在删除字段的匹配对照与联合树中均存在。没有搜索参数或修改源码。

结果：47次评分中较早期6次用到财政，主期0次；20次主期联合预测与匹配对照完全一致，主期五次高分仍来自融资五日净变化、新机会0，独立增量门失败。财政表达封存，新账户未运行；旧预算执行、贷款期限、住户存贷和审批失败保留。不得用较早期误差点值改善重开主期或改累计/单月、季调、方向与模型深度。

### E3：全市场IPO/再融资供给约束来源资格，已结束

状态：**SOURCE_GATE_NOT_PASSED_NO_CONTINUOUS_EQUITY_SUPPLY_FIELD**。2026-10-02按固定11页面路线和旧目录核对完成；规则有解释价值，但缺连续全市场金额、支付时点及适用项目覆盖。未评分，不延长本路线、不沿用标题数或事后融资总额拟合。实际结果见第5.5节及R60。

### E4：美国国债计划供给公告——固定月来源已完成

此前5条Bill/1原件探针保留在R61。2026-10-02新增固定2024-01全月来源，7条常规固定券、7PDF/7XML全部核实；规模均与2023-11-01季度计划一致，3个再公告根并未新增规模。有限来源/时钟通过，短期新增规模表达未评分，该固定月来源路线结束，详见第5.6节和R62。

### E5：官方一级交易商供给预期——匹配问题已结束

真实24机构汇总已取得，但预测FY24/25/26年末，不能当1月或下一季度共识。当前短期预期差路线结束，NOT_RUN_FORECAST_HORIZON_NOT_MATCHED；不继续下载同类文件营救，不否定长期机构预期的解释价值。详见R63。

### E6：单项季度长端供给增量——已完成并拒绝

**REJECTED_FROZEN_LONG_SUPPLY_INCREMENT_NOT_ESTABLISHED**，2026-10-02 05:19。16声明来源完成，28评分/84拟合。主期20次中的18次预测相同，实际使用供给2次但没有新增高分，MSE恶化2.8983%；五个高分仍全部同原M2融资路径。账户准入失败，0新账户/源码改动，原字段、源窗口与模型封存，详见第5.7节/R64。E5目录下原E6提案作为登记前谱系保留，不能继续当当前未运行状态。

### E7：美债持仓类别与真实迟报——有限来源已结束

**SOURCE_GATE_NOT_PASSED_FULL_WEEKLY_PUBLICATION_CLOCK_NOT_LOCATED**，2026-10-02 05:44。2023原10年043602/52周、7实际迟报已定位，其他45周只有常规日历；完整时钟门未过，比例/评分/账户NOT_RUN。两个原M2决定日会受到错误时钟影响，但原M2没有该变量，旧结果不变。有限路线结束，不扩年或换合约营救，详见第5.8节/R65。

### E8：准备金变化的同表上游构成——事实已完成

**COMPLETED_SINGLE_RELEASE_RESERVE_CAUSE_IDENTITY_ACCEPTED_FACT_ONLY**，2026-10-02 05:50。单2023-09-14原表/10字段，两条同周均会计桥误差0；持券减少59.38亿与准备金增加281.68亿美元并存。只是源和构成事实，0股票收益读取/评分/账户，原例表选择已披露，详见第5.8节/R66。

### E9：实际周度公布的三原因联合增量——已完成并拒绝

**REJECTED_FROZEN_FED_RESERVE_CAUSE_INCREMENT_NOT_ESTABLISHED**，2026-10-02 06:23。210官方对象含208原表，193评分/579拟合；主期63次实际路径使用原因，MSE恶化2.8380%，三个高分与对照全部相同，新增0。37高分叶归因已完成，无新拟合。来源和模型登记、全部五档、反例及程序记录保存，账户未运行、源码未改，详见第5.9节及R67。[结果](../reports/research/510300_fed_reserve_three_cause_weekly_score_v1/result.json)。

### E10：同一期净市场化借款预估修订的原因——事实已完成

**COMPLETED_MATCHED_QUARTER_BORROWING_REVISION_CAUSES_ACCEPTED_FACT_ONLY**，2026-10-02 06:51。固定2023Q3及两实际日期，3官方原件；借款修订274=期初现金148+期末目标50+融资需要83−其他融资6−显示残差1（十亿美元）。SOMA同季预估修订0，现金安排合计占72.26%；官员前次计划不是市场共识。没有股票标签/评分/账户/源码改动，不从一例宣称股市优势，详见第5.10节/R68。[结果](../reports/research/510300_treasury_borrowing_revision_source_qualification_v1/result.json)。

### E11：全部16季度四原因面板——来源完成、原完整门未过

**COMPLETED_SOURCE_PANEL_INCOMPLETE_FIXED_MODEL_NOT_RUN**，2026-10-02 07:26。34官方对象（31新/3复用），全部16当前季度，15直接四分量、14当前/旧日期完全一致。2023Q2只有现金PNG，2024Q1旧标签Oct31与前次原文Oct30不同；不补零、不用较晚表回填、不改原标签。4/16 headline与扣现金后方向相反；0新增股票标签/拟合/账户/源码改动，净夏普NOT_COMPUTED。原E11已结束，详见第5.11节/R69。[结果](../reports/research/510300_treasury_borrowing_revision_panel_v1/result.json)。

### E12：现金与扣现金后借款修订两渠道——已完成并拒绝

**REJECTED_FROZEN_CASH_REVISION_CAUSE_INCREMENT_NOT_ESTABLISHED**，2026-10-02 07:48。全16源口径通过后另登记，28评分/84拟合；主期20中19与控制相同，1次扣现金路径，MSE点值+0.31427%。原五高分同融资路径，新机会0、固定增量3条件失败，0账户/源码改动。原评分表达封存，不因E13时点事实重开，详见第5.12节/R70。[结果](../reports/research/510300_multidim_treasury_cash_revision_cause_score_v1/result.json)。

### E13：SOMA修订的上游政策公开顺序——事实已完成

**COMPLETED_SOMA_UPSTREAM_POLICY_PREANNOUNCED_FACT_ONLY**，2026-10-02 08:01。1新Fed页/4复用Treasury原件；2024May1宣布自6月QT减速，早于Jul29借款更新89日。月上限60→25、季度SOMA177→91（86），35×3=105不能强制等86。只政策先后，不证明完整金额预期或价格吸收，0新增股市标签/拟合/账户。旧32FOMC收益及E6/E9/E12拒绝不重跑，见第5.13节/R71。[结果](../reports/research/510300_treasury_soma_revision_prior_policy_clock_v1/result.json)。

### E14：全部非零SOMA修订的上游已知信息——事实完成

**COMPLETED_NONZERO_SOMA_UPSTREAM_POLICY_CHRONOLOGY_ACCEPTED_FACT_ONLY**，2026-10-02 08:38。5固定病例、23原对象（3新/20复用），4例上游政策提前89/89/40/5日，3仅简单上限路径数值相容；2024Q3差19、2024Q2的12原因未识别。源事实门通过，未登记模型/账户，0股票标签/拟合/源码改动，完整目标未达。详见第5.14节/R72，[结果](../reports/research/510300_treasury_soma_all_revision_prior_policy_sources_v1/result.json)。

### E15：上游已公布的国债兑付上限路径状态——已完成并拒绝

**REJECTED_FROZEN_ANNOUNCED_QT_CAP_PATH_INCREMENT_NOT_ESTABLISHED**，2026-10-02 09:26。来源34原对象、38准入/30未知，26评分/78拟合；全26对照联合预测相同，新字段进入树0。主期20次MSE0.0005463861696556954两边相同，五旧融资高分、新机会0，固定5条件中2失败。来源事实接受，固定股票增量拒绝；0新账户/源码改动、夏普NOT_COMPUTED，旧E6/E9/E12及E11裁决保留。详见第5.15/R73，[结果](../reports/research/510300_multidim_announced_treasury_qt_cap_path_v1/result.json)。

### E16：国股银票六个月转贴现价格——有限来源已完成，完整源门未过

**SOURCE_GATE_NOT_PASSED_CONTINUOUS_HISTORY_AND_PUBLICATION_CLOCK_NOT_ESTABLISHED**，2026-10-02 10:28。官方主体与监管链接互证；一个2022-01-04原6M样例利率2.3900%/收益率2.4196%，2025-12-31该次码200无数据，日期格式只补同日识别。连续2022—2025及正式公布时钟未建立；16次HTTP/14原响应，0新股票标签/拟合/账户/源码改动、夏普NOT_COMPUTED。有限路线结束，不证明全年/其他正式路线不存在，不以缺数据称策略失败。实际依据见第5.16/R74及[结果](../reports/research/510300_bank_accepted_bill_discount_source_qualification_v1/result.json)。

### E17：企业贷款加权价格——来源结束，当前版本未接回原时钟

**SOURCE_GATE_NOT_PASSED_CURRENT_PDF_VINTAGE_TO_HISTORICAL_RELEASE_NOT_ESTABLISHED**。固定16原报告的表3同名字段/月/百分数已提取并目视核对；7当前PDF晚生成，原页面历史公布时刻不能直接赋予当前各数值。0拟合/账户/源码改动，夏普NOT_COMPUTED。目录定位只实现修订，原失败保留；不缩季度/换口径营救。详见第5.17/R75及[结果](../reports/research/510300_corporate_loan_weighted_rate_source_qualification_v1/result.json)。

### E18：原M2完整账户目标缺口——已完成

**COMPLETED_FIXED_MONTHLY_ACCOUNT_GAP_SIGNAL_OPPORTUNITY_DENSITY_AND_AMPLITUDE_INSUFFICIENT**。四账本4,848日行、12原指标行及30周期精确核对，0新模型/引擎账户。主期20万元原年化1.997638%、同持仓返费2.324601%、事后完美半仓同原退出路径限定上限2.772435%；2024/25为3/2周期。仅解释，不推广反事实；新原因只复现原五窗口不足。详见第5.18/R76及[结果](../reports/research/510300_monthly_account_target_gap_diagnostic_v1/result.json)。

### E19：国内已公告逆回购合同到期——有限来源完成，实际结算未建立

**SOURCE_GATE_NOT_PASSED_ACTUAL_START_AND_APPLICABLE_MATURITY_CALENDAR_NOT_ESTABLISHED**。固定2022—2025的1,041原操作/997原公告已复用核对；4目录检索/8方法请求/7新原件。没有实际起息/到期适用规则及完整银行间日历，未用公告日加期限或股票日历代替。模型未注册，评分/账户NOT_RUN，夏普NOT_COMPUTED；0新标签/拟合/账户/源码修改。不是经济机制被否定，不重跑旧数量模型。详见第5.19/R77及[结果](../reports/research/510300_domestic_reverse_repo_maturity_prior_source_review_v1/result.json)。

### E20：银行代客结售汇原因分项——来源阶段完成，历史版本未准入

**SOURCE_GATE_NOT_PASSED_CURRENT_LONG_TABLE_TO_HISTORICAL_MONTHLY_RELEASE_VALUES_NOT_ESTABLISHED**。当前60月×18字段完整，3原件；2026版原长表不能回填2021—2025各月公开时钟。0新标签/拟合/账户，评分NOT_RUN；统计分项异号事实不等于指数方向或交易优势。详见第5.20/R78及[结果](../reports/research/510300_safe_client_fx_cause_source_qualification_v1/result.json)。

### E21：原日频融资输入纠错——已完成，原高分准入仍失败

**COMPLETED_EXACT_INPUT_CORRECTION_NO_PRIMARY_HIGH_SCORE_EDGE**。只替换62字段单元格，原两模型各1,212日，共4,848同参数拟合；主期两者同9笔/均值−0.928734%，全账户NOT_RUN。原参数、标签、失败谱系和资金约束不变。详见第5.21/R79及[结果](../reports/research/510300_daily_score_financing_input_correction_v1/result.json)。

### E22：固定历史价格时段——已完成，简单先涨解释未成立

**COMPLETED_FIXED_TIMING_SIMPLE_PRE_SIGNAL_RALLY_STORY_NOT_SUPPORTED**。主期9例前20日均−1.519519%、仅2例此前正，原五日净−0.928734%；未改口径，无新拟合/账户。次日缺口不能输入原信号，不推提前买优势。见第5.22/R80及[结果](../reports/research/510300_fixed_score_price_response_timing_v1/result.json)。

### E23：保存高分叶的正收益来源——已完成，集中性事实成立

**COMPLETED_SAVED_HIGH_SCORE_LEAF_TRAINING_CONTRIBUTIONS_EXPLAINED**。全部70事件/5,352成员复算；主期八/十变量对应叶相同，7/9的正标签贡献过半来自2024年9月，实际路径仅订单/融资变化；不能据此识别政策因果、删标签重训或重定义目标。无新拟合/账户。见第5.23/R81及[结果](../reports/research/510300_saved_high_score_leaf_contribution_v1/result.json)。

### E24：原训练大涨标签的政策公开时钟——限定解释已完成

**COMPLETED_TRAINING_TAIL_POLICY_CLOCK_MIXTURE_WITH_PUBLICATION_GAPS_PRESERVED**。两个原标签包含已有政策背景及原决策后/持有期新公开信息；归档标记、内容日、网页日与实际实施分开。原9/23预告/首次公开/严格开盘输入门仍未建立，预期差和因果收益份额未识别。7原件/8时钟记录、0新标签/拟合/账户/源码改动；不能将事后最大标签当新盈利事件池。见第5.24/R82及[完整结果](../reports/research/510300_training_tail_policy_clock_explanation_v1/result.json)。

### E25：SFISF原因输入旧用途——已结束，未建立不同用途

**COMPLETED_PRIOR_USE_REVIEW_PROPOSED_SFISF_CAUSE_INPUT_NOT_DISTINCT**。额度/开放/两次操作/实际使用已由旧链覆盖，无新来源或模型；旧数量七日逆回购差异不自动使SFISF用途新颖。见第5.25/R83及[裁决](../reports/research/510300_sfisf_prior_use_review_v1/result.json)。

### E26：固定外需条件联合评分——已完成，固定表达封存

**REJECTED_FROZEN_FIXED_EXPORT_ORDER_CONDITION_NO_PRIMARY_INCREMENT**。132当月原文、1,212日、2,424新拟合；主期10机会净均值+0.284198%、预测MSE比原树下降1.095102%，但仍高于均值；五门4过1败，0新账户。正观察不擦除，原参数/输入/标签/失败保持，首次收件和独立验证未建立。见第5.26/R84及[结果](../reports/research/510300_export_order_cause_increment_v1/result.json)。

### E27：原固定机会完整目标缺口——已完成

**COMPLETED_FIXED_EXPORT_OPPORTUNITY_GAP_CONCENTRATED_AND_INSUFFICIENT**。最大正标签占正贡献70.355945%、其余九行−0.718420%；实际入场年3/6/1、2026残年。原十完整持有端点、零费用/半仓/事后跳过的有限年化上界3.600862%，非可执行账户或一般上界；0新拟合/账户。见第5.27/R85及[诊断](../reports/research/510300_export_score_target_gap_diagnostic_v1/result.json)。

### E28：保存评分的信息更新来源——有限解释已完成

**COMPLETED_SAVED_EXPORT_INPUT_MODEL_AND_MATURE_LABEL_UPDATE_EXPLAINED**。2,422相邻对/67机会/9,688函数值，9/26原出口未更新、旧树两输入−0.747599%、新树两输入+1.174437%；叶均值变化已按新成熟成员/保留分组分开。不是经济因果份额、独立验证或新策略；0新拟合/源/账户/源码，原E26/E27保持。见第5.28/R86及[结果](../reports/research/510300_saved_export_information_update_v1/result.json)。

### E29：银行业整体约束来源——有限初探完成，完整历史未准入

**COMPLETED_INITIAL_BANK_REGULATORY_SOURCE_PROBE_FULL_PANEL_NOT_ADMITTED**。6原对象/2根文件、2019总体/2024资本断点和当前2026滚动年度版本确证，净息差期间未知。没有模型/账户，见第5.29/R87及[结果](../reports/research/510300_bank_industry_constraint_source_probe_v1/result.json)。

### E30：同版本比率调整原因——有限构成已完成

**COMPLETED_SAME_VINTAGE_BANK_RATIO_NUMERATOR_DENOMINATOR_IDENTITIES**。三季度/两相邻区间/两指标全部四组；Q2→Q3资本净额增加而RWA增加更快，不良余额/贷款总额均增且分母抵消部分升幅。未识别经营行为因果或原公布时点优势，0新拟合/账户，见第5.30/R88及[结果](../reports/research/510300_bank_current_vintage_ratio_cause_identity_v1/result.json)。

### E31：原季度公开值与净息差期间——有限来源问题已结束

**COMPLETED_BOUNDED_BANK_RELEASE_ROUNDED_VALUES_NIM_PERIOD_AND_FULL_VINTAGE_UNESTABLISHED**。新增Q2独立正文，Q2/Q3四个舍入公开点值与当前表显示一致，不能认证原附件精度/首版。PBC论文口径不支持行业净息差定义；完整历史仍未准入，0新拟合/账户/源码。见第5.31/R89及[结果](../reports/research/510300_bank_release_vintage_and_nim_definition_probe_v1/result.json)。

### E32：同DR007成交额来源——身份/单位已确认，完整历史未准入

**COMPLETED_DR007_TRANSACTION_AMOUNT_IDENTITY_UNIT_AND_LIMITED_SAMPLES_FULL_PANEL_NOT_ADMITTED**。真实amount万元/官方trdVol亿元，固定20200804官方空响应不当历史值。0模型/账户，见第5.32/R90及[结果](../reports/research/510300_dr007_transaction_amount_source_and_prior_use_probe_v1/result.json)。

### E33：原12分块成交额采集——现有访问到期停止

**STOPPED_HISTORY_ACQUISITION_EXISTING_TRANSPORT_EXPIRED_NO_PROVIDER_REQUESTS**。当前无有效标准运输被选择，临时访问已过期；用户答复暂时没有。0提供方请求/金额原行，经济假设未被检验，原完整成员/金额时钟门未过，见第5.33/R91及[结果](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/result.json)。

### E34：真实有效访问更新后的收件——仅条件提案

**PROPOSED_ONLY_AFTER_REAL_VALID_ACCESS_UPDATE_NOT_REGISTERED**。E34仅在真实有效数据访问更新后，按同DR007.IB/DR007/amount和原12分块另存收件接续；用户当前暂时没有访问，未注册、无活进程，不自动轮询旧到期配置或支付服务。完整原成员/决定日和金额公开时钟源门通过后才另注册一次原参数联合评分；目前没有已获准的新策略/账户实验。央行操作量不替代成交额，旧月末/季末日历字段不重新包装。历史目标和既有失败不改。

[下一条件提案](../reports/research/510300_dr007_transaction_amount_history_acquisition_v1/next_experiment.json)。E33当时另外核了旧日历用途，见第5.34/R92；其时尚未注册E35或重训。后续E35公开目录与E36采购评分实际结果见第5.36—5.37节及R94—R96。

### 接手操作清单

1. 先读本文与 [决策账本](RESEARCH_DECISIONS.md)，核对主配置是否有用户新修订；不要从旧 blocked 文件恢复前瞻等待。
2. 宏观分支先读E1—E39及R99/R100：原失败保持，E39用电原周期/衔接差接受、固定三单月门未过；没有新收益优势。E40仅不同原发布机构全国1/2月原单月有限来源提案；E34仍仅有效访问/可靠同身份历史后接续，不移植他任务权限。
3. 新实验先用独立 ID 保存少量固定字段与入口；完成后将结果/原因/下一步写回这两份文档。
4. 无需恢复 PCF 任务、构造交付 ZIP、调用另一任务的 forward advance 或改 Git。没有新增金融代码时不运行全仓测试。
5. 可按需要使用 handoff 技能维护交接；涉及新 PDF 原件时使用 pdf 技能。当前 Markdown 文档不需要外部 Pages、Word、表格或插件服务。

### E37：全部24主期月采购量调整的当时原因证据（已完成）

状态：COMPLETED_24_MONTH_PURCHASE_FACTS_AND_SCORE_CLOCKS_DIRECT_CAUSE_UNKNOWN。24月直接原因全部UNKNOWN；采购上升但仍低于50为5月，新月份首次评分仅1/24路径实际使用采购。原字段/正文与保存时点事实成立，跨日采购因果效应未识别；0新网络/标签/模型/拟合/账户/源码，E36拒绝保持。详见第5.38/R97及[结果](../reports/research/510300_purchase_adjustment_current_report_cause_review_v1/result.json)。

### E38：同期官方解读中的独立原因信息（固定三月已完成）

COMPLETED_FIXED_THREE_OFFICIAL_INTERPRETATIONS_FACTS_ACCEPTED_CAUSE_INPUT_NOT_ADMITTED。3原对象/219,621字节；3月有采购直接官方共同原因叙述、1月有产能调查下界，但可比较独立原因输入未准入，0新模型/账户。本路线结束，详见第5.39/R98。

### E39：实际用电量的不同实物活动信息（固定来源已完成）

COMPLETED_ELECTRICITY_ORIGINAL_PERIOD_FACTS_PRIMARY_THREE_SINGLE_MONTH_GATE_NOT_PASSED。实际2原通报+1目录、24,260字节，3月原单月成立，1/2月UNKNOWN；跨公布115衔接差原因未知、不拼单月。源门未过、0新模型/账户，本运输路线结束，见第5.40/R99。

### E40：中电联固定索引路线——已完成，无新原单月

**COMPLETED_FIXED_CEC_INDEXED_QUERY_ROUTE_NO_NEW_ORIGINAL_MONTH_ROW**。原固定两个月/两查询皆空，0GET/原行，结束此路线、不证明全源没有；原E39及模型失败保持。详见第5.41/R100/[结果](../reports/research/510300_cec_national_single_month_electricity_source_qualification_v1/result.json)。

### E41：保留实际原区间的公告状态——已完成，完整原成员支持未过

**COMPLETED_ORIGINAL_ELECTRICITY_PUBLICATION_STATES_SOURCE_SUPPORT_NOT_COMPLETE**。5导航/52报告GET、51成功，99原区间/51公告。原1,711必要日期1,155已知/556未知，全部1,212原评分日同一成熟池不完整。0模型/标签/账户/源码；不丢训练日、未知单月不回退累计，E36/39/40保持。详见第5.42/R101/[结果](../reports/research/510300_electricity_original_publication_state_v1/result.json)。

### E42：2022-11直接原同比——索引描述已完成，原根/原钟未准入

**COMPLETED_FIXED_NOV2022_INDEXED_REPORTED_YOY_FACT_ORIGINAL_MODEL_SOURCE_NOT_ADMITTED**。2查询/同对象2读取，当前HTTP只是492字节应用壳；工具政府索引全文直接给出二产4789/−1.0%及转载12/19 17:31:54。原根未知、不倒填NEA12/15，异版本第一产业量差1未知。E41556未知与完整原门保持，0新模型/账户。见第5.43/R102/[结果](../reports/research/510300_electricity_nov2022_original_yoy_source_probe_v1/result.json)。

### E43：全部原量构成与测量反例——已完成，收益未测试

**COMPLETED_ORIGINAL_COMPONENT_DIVERGENCE_AND_MEASUREMENT_SUFFICIENCY_COUNTEREXAMPLES**。99提取记录含4完全重复，保留原记录/原源门，唯一95原周期。原44单月43可比中40同正/2同负/1总正二产负/1未知；2022-07是原反例，E42索引2022-11另列。只否定测量充分条件，不推因子/股市方向失败。显示值代数构成不是去年原量/经济因果；0网络/标签/模型/账户/源码。见第5.44/R103/[结果](../reports/research/510300_electricity_component_state_description_v1/result.json)。

### E44：同原构成与自然公告来源——已完成，已见完整支持通过

PASS_KNOWN_ORIGINAL_EVENT_COMPOSITION_SOURCE_AND_CLOCK_CONTRACT。2查询/1新转载GET补2024-10原量，52已见原根保留/51评估源及成熟池完整/12预热/39可评分。晚于转载11/21EOD才用，不倒填旧失败根11/20；公开覆盖/版本限制明确。见第5.45/R104/[结果](../reports/research/510300_electricity_composition_event_source_contract_v1/result.json)。

### E45：原八加三构成同事件评分——已完成，拒绝增量

REJECTED_FROZEN_FIXED_COMPOSITION_JOINT_SCORE_NO_PRIMARY_INCREMENT。39×4=156固定拟合；主22评分，联合7自然机会净均−0.760923%、原八8次−0.414290%。五门仅一项通过，0新入场/0构成高分路径，完整账户NOT_RUN、夏普NOT_COMPUTED。只本次固定表达拒绝，不否定机制永久无效。见第5.46/R105/[完整结论](../reports/research/510300_electricity_composition_joint_score_v1/用电构成联合评分与失败原因_研究结论.md)。

### E46：全部保存信号/费用/叶组归因——已完成，不重训

COMPLETED_SAVED_SIGNAL_COST_AND_LEAF_CONCENTRATION_ATTRIBUTION_NO_RETRAINING。156预测身份/17模型机会费用标签核对通过（9不同日期）；零费界仍−0.460097%，5当前高分叶极端绝对贡献87.20%—91.25%，不是“降成本即可达标”或上游政策已被识别。0拟合/标签/账户/源码，原拒绝保持。见第5.47/R106/[结果](../reports/research/510300_electricity_composition_failure_attribution_v1/result.json)。

### E47：原政策身份与原决定/入场钟——已完成，市场预期和因果未知

COMPLETED_ORIGINAL_POLICY_SEGMENT_CLOCK_BOUND_TO_EXTREME_MATURE_LABEL_NO_CAUSAL_RETURN_IDENTIFIED。复用两原件、10信息：09:19:36段7项方向/幅度，11:42:50段3项额度/设计；原9/23资金状态没有这些明示输入。现场段重建与网页首发/实施分开，不能当完全意外或+40.06%单政策因果。0新拟合/标签/账户/源码，E45拒绝保持。见第5.48/R107/[结论](../reports/research/510300_extreme_leaf_policy_information_clock_v1/政策信息与会前预告_原决定时钟研究结论.md)。

### E48：会前预告原公开钟——已结束，未准入

COMPLETED_FIXED_OFFICIAL_PREVIEW_ROUTE_ORIGINAL_PREDECISION_NOTICE_CLOCK_NOT_ADMITTED。2查询/2实际href GET，英文会后专题200、中文现场521，原会前预告钟/共识仍UNKNOWN；不证明全档案无原预告，不重试/调查询营救。见第5.49/R108/[结果](../reports/research/510300_policy_press_preview_information_clock_v1/result.json)。

### E49：非制造业原需求与经营预期——已完成，原源门未过

原三字段/原1212决定及1711当前成熟身份保存；2022-12原经营预期未发布，15成员缺口影响519评分日，时钟同原订单。模型NOT_RUN，不能删预期、填充或缩完整期营救。见第5.50/R109/[源结论](../reports/research/510300_nonmanufacturing_demand_expectation_source_contract_v1/非制造业三字段_旧用途与原训练支持.md)。

### E50：在手订单与生产不同问项——原源门已过

原132月表、旧30值、全部原评分/成熟身份齐备；只历史重构、首版首次收件未认证。原双层表头真实解析错误已修正，不改变字段和经济候选。见第5.51/R110/[源结果](../reports/research/510300_backlog_production_cause_increment_v1/source_result.json)。

### E51：原八＋在手订单与生产——一次固定评分拒绝封存

2424新拟合、0基准重拟合/新网络/新标签/账户/既有源码改动。主485评分9自然机会净均−0.452154%、胜率44.4444%，MSE比原8增0.571516%，五门仅新日期和新增字段实际路径过。REJECTED_FROZEN_FIXED_BACKLOG_PRODUCTION_CONDITION_NO_PRIMARY_INCREMENT；夏普NOT_COMPUTED。见第5.52/R111/[模型结果](../reports/research/510300_backlog_production_cause_increment_v1/result.json)。

### E52：全部原机会、费用与保存叶——固定失败归因完成

九机会零费界仍−0.157862%，全部入场年2025、2024无自然机会。四新增高分实际仅生产，原当前四月49.8与原九月标签八月49.8同值；五个重叠九月标签贡献叶带符号收益85.19%—89.71%。这是来源/模型事实，不是单因果或独立五机会。原拒绝保持，0新拟合/账户。见第5.53/R112/[完整归因](../reports/research/510300_backlog_production_cause_increment_v1/固定失败归因_信号费用与原叶.md)。

### E53—E54：两同生产值的原信息及官方共同原因——已完成

E53原两月完整数值支持不同联合背景，原公告未明确具体原因；E54两查询/两原GET后确认官方整体共同解释不同，且需求不足等因素并存。显示09:30先于原决定，不认证历史首版或完整市场意外；只两后验目标月，不制造训练原因标签。原贸易季度用途实际2019Q2，旧2025Q2引用已补正。见5.54—5.55与[两官方解释结果](../reports/research/510300_equal_production_official_interpretation_source_v1/result.json)。

### E55—E57：行业需求两字段、固定评分及失败解释——已完成

E55原1212日/1711当前与成熟成员齐备；E56一次原参数2424拟合，13主期自然机会净均−0.063346%、MSE恶化1.849081%，五门仅新日期/新路径通过，固定拒绝。E57主期零费+0.239266%但早期零费−0.808623%，原入场年2024=3、2025=10；新增四贡献集中在9/26窗口，仍保留全部13，0新拟合/账户。见5.56—5.58与[固定归因](../reports/research/510300_service_construction_demand_cause_increment_v1/固定失败归因_信号费用与原叶.json)。

### E58：共享DR007原金额线索——实际检查结束，无amount

新共享指针仍旧2905行weight源，主schema、原12请求字段及已查三原响应均无amount，四schema上限后结束。没有网络/凭据或账户申请，金额经济假设未检验，E33到期停止不改。见5.59与[实际结果](../reports/research/510300_dr007_amount_shared_local_source_probe_v1/result.json)。

### E59：完整原月份共同原因信息——仅提案未登记/执行

[提案](../reports/research/510300_service_construction_demand_cause_increment_v1/next_experiment.json)先由全部原1711当前/成熟成员和1212原评分日反查统计月母集（预计2018-11—2025-12共86月，以原绑定为准），再核E38/E54已有五个月明确解释、作用层级与可知性，建立明确共同原因/同期背景/未知矩阵，登记剩余真实原来源路线。不会自动启动86月网络采集，当前模型/网络/新标签/账户/源码预算均0。

共同原因可并存，未提及不等于不存在，不能用“天气/关税”二分类或指数收益倒贴原因。完整原来源、时钟和字段解释建立后才可另登记不同经济模型，不延续E51/E56删列、改窗口、低费用或只用近期/岭/新增四笔的营救。该来源问题并非盈利结论；完整净夏普1.2目标仍未建立。


## 11. 本次整理的覆盖与限度

本次实读当前配置、Git log/status/diff、近期研究源码、主任务结果与关键较早封存结果，并检查共享 repo 中最新点位/权重研究；没有仅按聊天摘要写作。769 个目录不等于全部逐项重跑或审阅，本次不把目录扫描称为科学复核。96 因子中的 18 个策略逐项状态保留在配套账本，其他重要研究按机制与实验版本记录。

本文引用的数值来自保存结果，除文档一致性与链接检查外，本次没有重算策略绩效；合成测试、来源取得、账本算术正确、压缩包结构正确和独立金融有效性是不同层次。未定义、未运行、未计算、未准入和冻结拒绝都必须原样保留。

补充清点于2026-10-02 01:27:58完成：[机器可读事实快照](evidence/project_state_20261002_snapshot.json)覆盖769个研究目录、2,031份含状态信息的JSON，其中目录内1,748份、根层283份，读取错误0；保存路径、大小、SHA256、状态字段、84条Git提交和业务目录未跟踪文件。它与前述根层/一级目录1,281文件清点采用不同筛选范围，均不是独立实验数或全库科学复核。扩展Git业务范围含paper/tools/CONTEXT，计5,678个未跟踪文件；不要与包含原始数据的282,030个全仓路径混用。

盘点入口：[build_project_state_snapshot.py](../scripts/build_project_state_snapshot.py)。实查Junction为：data→E盘ResearchData下本项目data，reports→同项目reports，.venv→D盘CDriveData下戴周阳/project_venv，deliverables→D盘ResearchArchive下本项目deliverables。仅保存Git提交不能恢复这些目录。此次只做与文档有关的核对，未重跑765个测试文件或重新计算金融结果。

后续增量：2026-10-02 02:42完成PR-E01/02来源实验；543文件hash/行/schema重新准入，保存47,784时点矩阵及会话计数。19项相关边界测试通过；没有运行全仓测试、金融模型或账户。这是初次01:27清点之后的新工作，不倒写进旧清点快照。

后续宏观增量：2026-10-02 05:19完成E6季度长端供给固定比较（28评分/84拟合），来源及评分失败边界写入第5.7节/R64；未修改源码或计算新账户。本更新不是初次Git/目录清点的一部分。

后续宏观增量：2026-10-02 05:50完成E7真实迟报源资格与E8同表准备金构成，新增R65/R66，0股票标签/拟合/账户/源码改动；E9为下一提案。此次来源/构成研究不倒写成初次盘点结果。

后续宏观增量：2026-10-02 09:26完成E15来源后唯一评分（26/78）与保存复核，新增R73；原标签复用、0新账户/源码改动。修正长期文档E15提案状态及主配置过时E13/E14导航；E16仅未登记来源提案，本更新不倒写初次盘点或其他分支。

后续宏观增量：2026-10-02 10:28完成E16有限来源，新增R74；16次HTTP/14原响应，主体及单日6M样例接受、连续历史/公布时钟未建立。0新股票标签/拟合/账户/源码改动；补正长期文档旧E16未注册/官网不可达状态，E17仅来源提案，不倒写初次盘点或其他分支。

后续宏观增量：2026-10-02 11:39完成E17固定16表值/版本时钟来源问题与E18原M2账户目标缺口，新增R75/R76。0新拟合/引擎账户/源码修改；E17源门未过，E18解释路径不推广为策略，完整目标未达。E19仅旧表达/来源提案，不倒写初次盘点或其他分支。

后续宏观增量：2026-10-02 12:15完成E19有限来源，新增R77。997原公告/1,041原行，8直接请求/7新原件；实际资金起日/到期规则及结算日历未建立，源门未过。0新标签/拟合/账户/源码修改；E20仅旧用途/来源提案，不倒写初次盘点或其他分支。

后续宏观增量：2026-10-02 13:05完成E20/E21，新增R78/R79。E20源门未过，E21真实输入纠错仍无主期高分净优势；4,848同参数拟合、0新完整账户/源码修改，E22仅历史解释提案，不倒写其他研究线。

后续宏观增量：2026-10-02 13:23完成E22/E23，新增R80/R81。原9高分的先涨解释未成立，保存叶正预测集中于少数月份；仅解释，0新拟合/账户/源码修改，E24政策公开时钟尚为提案，不改其他研究线。

后续宏观增量：2026-10-02 13:55完成E24限定政策先后解释，新增R82；0新标签/拟合/账户/源码改动，2新原件。首版/预告/开盘门未建立，完整目标未达；E25仅来源提案，不覆盖其他研究线。

后续宏观增量：2026-10-02 14:24完成E25/E26/E27，新增R83—R85。外需固定评分有正均值改善，预定门未过，原有限路径目标缺口已量化；2,424新拟合、0新完整账户/源码改动，E28仅解释提案。不覆盖其他研究线。

宏观接续：2026-10-02 14:47完成E28，新增第5.28/R86及下一E29提案。0新拟合/源/账户/源码；E26/E27及其他分支状态不变。

宏观接续：2026-10-02 15:02完成E29/E30，新增第5.29—5.30/R87—R88及E31提案。2查询/8唯一直链，0新拟合/账户/源码，其他分支原文保留。

宏观接续：2026-10-02 15:28完成E31，新增第5.31/R89及E32仅来源提案；0新拟合/账户/源码。其他分支原文保留。

宏观接续：2026-10-02 15:59完成E32来源身份及E33到期停止诊断，新增第5.32—5.34/R90—R92。用户暂无有效访问，0新拟合/账户/源码；其他分支原文保留。

宏观接续：2026-10-02完成E49—E52，第5.50—5.53/R109—R112保存原未发布字段、原全支持和2424一次新评分、固定拒绝及九原机会/叶/费用解释；E53仅同值异因原源提案。0新网络/标签/账户/既有源码修改，其他分支受保护原文保持。


本轮追加覆盖E53—E58及E59提案；实际E54两官方GET、两查询，E56唯一候选2424辅助/主模型拟合，0原控制重拟合、新标签、完整账户、认证请求或既有源码修改。原同值原源、旧贸易研究年份补正、各实际终态与下一提案区别已写入，不重新审计其他研究线。

## 12. 盘口分支：压力让步与流动性修复研究

### 12.1 问题、权限与实际进展

原问题是：**510300因短时交易压力出现价格让步后，流动性恢复是否还能在实际可取得的成交价开始，覆盖费用、T+1及跨日风险？** 收益定义为全部完成交易上的净期望，含平手；胜率、实际平均盈亏比和损失概率共同决定期望。此处M1/M2是盘口机制编号，**不是货币统计M1/M2，也不是旧月度剪刀差实验**。

原任务已复制到项目内：[用户任务原文](../reports/research/510300_pressure_recovery_v1/input_snapshot/用户任务原文.txt)。后续用户明确只有免费数据、存储有限、拒绝TinyFish扣费；手动完成Hugging Face登录/研究申请，机构填“Independent non-commercial researcher”；允许继续历史下载、按最近波动率选择策略、不必每天交易。已经成功读取免费研究仓库，不能继续把“等待用户申请”作为当前障碍。没有收费、券商下单、自动采集恢复或全市场原始库下载授权。

| 完成阶段 | 实际工作及结果 | 正确解释 |
|---|---|---|
| 原测量协议与旧源复核 | 初始957观察/8日期；正式合格M1/M2事件、订单、已确认成交均为0；6项收益输出未计算 | [初始status](../reports/research/510300_pressure_recovery_v1/status.json)只保留当时覆盖，不是当前源总量 |
| 三日局部探索 | 711测量时点、17个P0价格冲击片段；5分钟价格回升中位约2.11bp，费用量级约6.1bp | 只是价格描述；不是17个正式M2事件、成交样本或独立验证。[探索摘要](../reports/research/510300_pressure_recovery_v1/local_exploration_20261001/summary.json) |
| 时间单位修正 | 微秒/纳秒混用影响6个消息流字段；711点×6字段=4,266项直接窗口核对；价格、价差、17片段及独立分钟策略未改变 | 使用修正版流字段，旧表及影响说明保留。[修正摘要](../reports/research/510300_pressure_recovery_v1/measurement_correction_20261001/summary.json) |
| 逐笔关联与快照核对 | 326,888笔连续交易的被动委托关联完成，无负库存；14,220快照中仅319个十档完全匹配；8,335快照已含名义上更晚成交，影响349/711测量点、12/17片段 | 委托可关联不等于同一时刻盘口可还原；不能调时间平移直到收益变好。[关联结果](../reports/research/510300_pressure_recovery_v1/order_lineage_20261001/summary.json) |
| 盘口条件范围 | 对14,220快照给条件外包范围；17片段价格变化6向上/8向下/3不明，深度6增/3减/8不明，价差17个均不确定 | 包住观察值不等于唯一且联合可实现的盘口；不能认证流动性恢复或成交概率。[范围结果](../reports/research/510300_pressure_recovery_v1/quote_envelope_20261001/summary.json) |
| 旧同制度基线诊断 | 当时三天仅有3/27/44个先前同制度源日期，不足60 | 这是三日样本阶段结论；181日扩展后应重做来源资格矩阵，不能永久沿用旧不足数。[旧基线](../reports/research/510300_pressure_recovery_v1/baseline_feasibility_20261001/summary.json) |
| 固定波动切换价格代理 | 681,307分钟行，2,827日；2016-02-18至2026-08-20评价2,555日，四政策×两资金×两费用=16场景 | 20万元BASE主账户898笔、1,657个无新入场日，夏普-0.7283、年化-8.8854%、回撤67.56%；固定版本失败，不是M1/M2失败。[结果](../reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001/summary.json) |
| 免费历史扩展与覆盖核对 | 181交易日三流全部落盘并逐文件验证；4日有正IOPV字段；153个重叠日收盘匹配 | 取得新研究输入，尚未形成合格估值/执行合同；无新增正式策略收益。[历史摘要](../reports/research/510300_pressure_recovery_v1/historical_expansion_20261002/summary.json) |
| PR-E01字段/盘后消息合同 | 四日正IOPV尾段不能区分稳定/携带/未更新；四条盘后委托全部S，新增/删除0；全181日盘后委托只有10条S，成交1,030条 | 状态消息不能提供排队证据；IOPV字段/单位/经济时点仍未识别。[字段合同](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/01_字段合同.json) |
| PR-E02全181日来源资格 | 47,784时点，42,897连续网格；日历和本地字段候选59/2日期，条件诊断17/0日期；严格M1参考日/M2字段及60日候选均0 | 原来源门槛未通过；正式事件检验NOT_RUN、收益NOT_COMPUTED，不是机制已拒绝或收益等于零。[结果](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/summary.json) |
| PR-E03免费来源可行性 | 4个新原始CSV库的字段文档实际403；免申请ZIP只检查目录/表头；QuantDB免费额度及次方分钟IOPV权限/时间规则已核对；现行STEP合同区分独立IOPV流 | 0个新准入来源，原金融目标未完成；有界检索不证明所有免费来源不存在。[结论](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/免费来源可行性结论.md) |
| 181日四表交付 | 47,784×29同步网格、181日覆盖、543源文件身份；事件/订单登记0行，3版本×2规模的6行结果均未计算；完整Excel约6.41MiB | 33项核对通过，Excel/CSV比较1,392,579个单元格、CSV/上游矩阵比较477,840次数值；只证明交付一致性。正式事件数未知，0登记行不代表无机会。[交付结果](../reports/research/510300_pressure_recovery_v1/four_table_delivery_181day_20261002/summary.json) |

### 12.2 当前代码入口

| 实现 | 用途与边界 |
|---|---|
| [pressure_recovery_v1.py](../research/pressure_recovery_v1.py)、[运行器](../scripts/run_510300_pressure_recovery_v1.py)、[配置](../config/510300_pressure_recovery_v1.json) | M1估值区间、M2事件、时钟/会话、T+1退出请求、佣金、净期望与按交易日聚类区间；没有真实交易网关 |
| [局部探索](../research/pressure_recovery_local_exploration_v1.py)、[消息测量修正](../research/pressure_recovery_measurement_correction_v1_1.py) | 已保存三日描述及修正版消息窗口；不得继续消费旧6个流字段 |
| [l2_order_lineage_v1.py](../research/l2_order_lineage_v1.py)、[quote_state_envelope_v1.py](../research/quote_state_envelope_v1.py) | 逐笔订单生命周期和条件盘口范围；不将无法识别的联合状态输出为确定盘口 |
| [volatility_conditioned_intraday_proxy_v1.py](../research/volatility_conditioned_intraday_proxy_v1.py)、[冻结配置](../config/510300_volatility_conditioned_intraday_proxy_v1.json) | 独立分钟OHLC价格代理试验；已失败封存，未使用完整篮子/盘口成交 |
| [acquire_510300_hf_history_v1.py](../scripts/acquire_510300_hf_history_v1.py)、[子集提取器](../scripts/acquire_510300_hf_l2_subset_v1.py) | 固定HF版本、HTTP Range、行组/证券筛选、断点hash核对和存储预算；不把全市场文件存到本地 |
| [历史覆盖核对](../scripts/profile_510300_hf_history_v1.py)、[IOPV字段核对](../scripts/profile_510300_hf_history_iopv_v1.py) | 流式读取已存543文件、与分钟收盘交叉比较、检查IOPV字段；输出覆盖，不输出策略资格 |
| [来源资格函数](../research/pressure_source_admission_v1.py)、[运行器](../scripts/run_510300_pressure_source_admission_v1.py)、[固定配置](../config/510300_pressure_source_admission_v1.json) | PR-E01/02：逐日源身份、盘后类型、旧时间截断条件诊断、同槽过去日计数；不读取未来收益，不以条件相容性认证真实快照或接收时刻 |
| [免费来源元数据探针](../scripts/probe_510300_pressure_free_source_metadata_v1.py)、[ZIP目录/表头探针](../scripts/probe_510300_zip_source_schema_v1.py)、[本地结论生成器](../scripts/finalize_510300_pressure_free_source_feasibility_v1.py) | PR-E03：有界公开合同/目录和准确HTTP Range；不完整下载ZIP、不执行外部代码、不解析价格行；生成器仅消费保存回执及本地协议 |
| [四表数据准备](../scripts/prepare_510300_pressure_four_table_delivery_v1.py)、[流式工作簿导出](../scripts/export_510300_pressure_four_table_streaming_v1.py)、[保存核对](../scripts/verify_510300_pressure_four_table_delivery_v1.py)、[交付配置](../config/510300_pressure_four_table_delivery_v1.json) | 复用既有资格矩阵、字段合同和源身份，更新全部181日四表；不重扫原始三流、不生成正式事件或收益、不改变原机制合同。核对覆盖全部数据单元格，页面摘录仅用于样式检查 |

原测试入口包括 [机制测试](../tests/test_pressure_recovery_v1.py)、[逐笔关联测试](../tests/test_l2_order_lineage_v1.py)、[条件范围测试](../tests/test_quote_state_envelope_v1.py)及 [历史下载测试](../tests/test_510300_hf_history_v1.py)。既有通过记录只证明各自实现范围；本次文档整理未声称重新通过全库测试。

PR-E01/02新增[来源边界测试](../tests/test_pressure_source_admission_v1.py)，与既有条件范围测试合计19项通过；覆盖未来消息、端点整秒未完全可得、缺行不补零、同百分之一秒次序、跨会话五分钟及过去计数排除当前日/另一制度。[保存日志](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/验证日志.txt)只证明这些实现边界。

## 13. 盘口分支已取得的数据与剩余缺口

### 13.1 免费历史数据已经下载到哪里

| 项目 | 2026-10-02保存事实 |
|---|---|
| 来源及固定版本 | Hugging Face数据仓库venvoo/china-a-share-l2-level2-limit-order-book-tick-data，revision `8d942a74865a67de55ba01b0b982d7ac8743f456`；免费非商业研究访问 |
| 仓库目录范围 | 当次枚举2017-01-03至2026-09-30共2,367源日期；这不是已下载2,367日，也不是全部目标证券日期完整的证明 |
| 本次目标范围 | 2026-01-05至09-30，181个交易日；行情/逐笔委托/逐笔成交各181文件，共543，534新提取、9复用、目标缺文件0 |
| 实际行数 | 行情907,199；逐笔委托61,126,649；逐笔成交17,254,742；总计79,288,590行 |
| 本地与网络 | 含复用源文件870,474,236字节，约830.15MiB；新增目录856,057,843字节，约816.40MiB；含参考证券探查的下载流量10,888,530,790字节，约10.14GiB。小磁盘子集不等于同样小网络流量 |
| 原始目录 | [venvoo_history_20261001](../data/raw/510300_free_channels_v1/20261001/venvoo_history_20261001/)；9个复用文件仍在旧venvoo_selected路径，以[coverage_manifest](../data/raw/510300_free_channels_v1/20261001/venvoo_history_20261001/coverage_manifest.json)为准，不移动/重下 |
| 格式与证券映射 | 行情/委托/成交分别66/10/12列；源证券标签为510300.SZ，保留原字段，研究规范对象是510300.SH；已验证报价/10000的收盘尺度，不能据文件后缀误当深圳ETF |
| 下载预算 | [固定下载合同](../config/510300_hf_history_expansion_v1.json)：新增磁盘≤1.5GiB、网络≤16GiB、单文件≤128MiB、行组缓存96MiB、单次解码限512MiB、2工作线程但串行解码、最低剩余盘2GiB、每文件最多2次尝试 |
| 完成状态 | 7次初始SSL失败均在既定重试内解决；原目标文件全齐；保存的E盘剩余约64.78GiB是完成时快照，不能当永久余量 |

直接证据：[历史summary](../reports/research/510300_pressure_recovery_v1/historical_expansion_20261002/summary.json)、[逐日三流与收盘对照](../reports/research/510300_pressure_recovery_v1/historical_expansion_20261002/01_逐日三流覆盖与收盘对照.csv)、[下载registration](../data/raw/510300_free_channels_v1/20261001/venvoo_history_20261001/registration.json)。

另有[免费一分钟价格](../data/raw/510300_free_channels_v1/20261001/neigezhu/data/etf_1m/SH/510300.parquet)：2015-01-05至2026-08-20、681,307行、2,827交易日，通常241行/日；分钟起止定义尚不能替代成交时点合同。OHLCV没有买卖报价、排队或实际成交证据。旧腾讯4,584条分段记录与957观察的计数对象不同，不拼成连续多年盘口。

### 13.2 新数据补上的缺口与没有补上的缺口

- **来源日期长度扩大，严格资格已经核对。** 7月6日前119日、其后62日。仅按源日期数量，前制度最早4月8日起有59个待评价日具备60个先前日期；后制度最早9月29日起只有2个。本地字段候选也为59/2日期，但没有同槽同步参考/接收合同，严格60日候选均0；不是已有61个正式可评价日。
- **收盘尺度交叉核对成立。** 与分钟源有重叠的153日全部匹配；28日无重叠。它不验证历史接收时钟、消息缺失、排队顺序、参考估值或可得成交。
- **正IOPV仅4日，盘后“委托”已解释为状态消息。** 7月6/7/8/13日分别5,136/5,226/5,254/5,247行，共20,863行。15:05以后分别128/221/246/240行正值，数值恒定；四日各1条该时段委托记录均为S，价格/数量0、方向空、ex_order_id为0；不是新增或删除订单。对应逐笔成交30/73/300/128条不能补出前方队列。值的最后一次观察变化分别15:00:12/21/18/13；不能据此认证15:00经济时点。[四日完整证据](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/02_IOPV与盘后逐日证据.json)
- **正值不等于及时且独立的公允价值。** 还缺字段经济时刻、单位合同、是否为收盘后携带值、参考价首可知时点；四日不能达到原最低30个独立交易日的区间报告条件。
- **旁证证券只完成存在性探查。** 9月30日单日的wind_code探查见510330有4,886行、159919有4,802行，000300/399300均0；尚未下载两只ETF完整报价，不能推广为整个仓库没有指数，也不能用同指数ETF直接替代独立篮子价值。
- **逐消息完整性和历史接收时钟未证明，旧接口指针已纠正。** 固定README明确没有交易所数据接收时间。LDDS2.0.10物理第10页UA3202的DataTimeStamp是最新订单时间（秒），第17页UA3108是最新订单时间（毫秒）；导出没有模板/类别/产品状态和接收字段，不能把整秒time认证为快照在这一秒生成，更不能填成历史接收时间。快照与逐笔没有相对传输先后，通道连续性也无法验证。旧sse_md102_spec.txt是较早接口文本，本条不再据其“最近成交”说明推断当前快照。[正确的已存规范](../reports/research/510300_pressure_recovery_v1/source_followup_20261001/ldds_level2_2_0_10.pdf)、[字段合同及物理页入口](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/01_字段合同.json)。
- **2026 STEP的独立IOPV流规则已补入，但不能跨接口作废导出。** 2026-09-18 STEP0.63物理第2页记录7月3日下线旧竞价行情IOPV；第22/34页将有效IOPV放入SecurityType14、MDE01流。SendingTime是交易所时间，不是客户端接收。当前venvoo完整7117路径没有单列IOPV流文件，固定README认证读取与已存版本相同；它不能证明混合iopv列没有合并MDE01，也不能证明已经合并。必须证明实际接口、版本及字段映射，不能把STEP条款直接套到LDDS或Wind导出、据此宣布四日正值无效。PR-E01/02冻结输出未修改。[接口解释及物理页](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/current_step_iopv_contract.json)。

因此，准确说法是“**181日目标证券三类源文件覆盖完整，来源资格实验已完成但原门槛未通过**”，不是“交易所全消息、逐笔排队、同步估值、真实成交证据完整”。正式M1/M2事件与成交后检验未启动；收益、填单概率和全账户夏普仍未计算。来源门槛失败与经济机制被否定是不同状态。

## 14. 盘口研究已经冻结的实验口径

本节是[原配置](../config/510300_pressure_recovery_v1.json)的文字索引，不能用本文改变参数。

| 内容 | 冻结定义 |
|---|---|
| 时间与基线 | Asia/Shanghai；最大时钟误差1秒；连续报价年龄≤5秒；参考时间偏差≤1秒；固定±10bp深度带；之前60个合格、同制度、精确HH:MM交易日 |
| M1收盘既存让步 | 制度起点2026-07-06；15:06决定；估值经济时点15:00且须在决定前可得；让步分档0/5/10/20/40bp；M1让步与M1让步+价差修复两版本；14:56价差不高于历史同槽中位且小于14:51 |
| M1成交 | 经核对收盘价；仅接受真实成交回执或独立验证过的队列重放。没有队列证据时成交概率为未知，不能填1 |
| M2冲击 | 5分钟ETF自身绝对收益率相对于先前分布处于q05以下、价差至少q90、篮子收益为负；此处“绝对收益”指ETF自身涨跌幅，非对收益取绝对值；相对收益为诊断，不另加必须折价条件 |
| M2修复 | 首次冲击后固定观察5分钟；价差回至历史中位且小于冲击、深度不低于历史中位且增加、相对5分钟收益改善；连续10个合格非冲击分钟才结束片段；缺观察终点为NO_VIEW，不挑后来的最优终点 |
| 合法退出 | 次一交易所交易日09:35发出退出请求，之后首个真实可执行报价或成交；当日新购不得当日卖；部分/缺失退出数量继续持有估值 |
| 费用 | 单边佣金0.0002、最低5元，示例摩擦单边5bp；均为研究假设不是券商报价。真实成交价已含价差/冲击，不重复扣；盘后固定价买入额外价格滑点0 |
| 评价 | 3项主要比较；按交易日聚类5,000次bootstrap、95%区间、seed510300；最低30独立日才报原区间，不代表功效充分。空仓日进入全账户，未成交/未知成交独立列示 |
| 对照 | 价格条件对照匹配同制度/同HH:MM/仅过去数据的收益与波动五分位；不使用未来收益挑对照 |
| M3 | 若优势在合法退出前消失，只能研究既有订单的执行改善；没有既有订单就没有M3执行收益 |
| 停止 | 估值误差无法界定、让步不足费用、填单选择不可识别、修复早于合法退出结束、共同下跌/尾损吞噬优势、冻结检验净期望或全账户目标失败 |

已失败的分钟价格代理另有独立口径：过去20日实现波动、仅以前252日给波动排名，30%/70%/95%切分；低波动和极端波动不新入场，中波动修复、高波动延续；固定5分钟冲击/观察、下一分钟收盘作入场代理、次日09:36收盘退出代理，假设全部成交。它没有篮子、队列或同步IOPV，也没有证明满仓容量。其失败不授权改30/70/95分位、持有期或选子期营救。

## 15. 当前盘口研究最值得继续的工作与具体实验

**PR-E01/02已于2026-10-02 02:42完成，PR-E03于03:46完成，181日四表保存核对于04:41完成，原来源门槛仍未通过。** 用户在读取长期事实后要求“请继续完成”；三轮实质推进分别为181日来源资格、新的免费来源字段可行性和完整四表交付，均没有新增原策略收益。原60日、会话、10bp、M1的30独立日均未放宽。PR-E04保持NOT_RUN_SOURCE_GATE_FAILED；PR-E05未运行，不恢复旧失败。金融目标及独立验证尚未完成，没有待运行的外部申请、下载或工作簿导出作业。

### PR-E01：四个正IOPV日期的字段与盘后覆盖解释

状态：**COMPLETED_FIELD_AND_AFTER_HOURS_CONTRACT；M1_REFERENCE_NOT_ADMITTED**。[字段合同](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/01_字段合同.json)、[逐日证据](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/02_IOPV与盘后逐日证据.json)。

- 输入：四日全部行情、委托、成交；正确的已存LDDS2.0.10、固定revision的源README；保留全部日期，不能按随后收益筛选。
- 方法：检查IOPV数值尺度、有效更新/恒定尾段、参考经济时刻、可用时点与盘后委托/成交类型；区分字段缺失、陈旧携带与真实更新。不读取策略收益来选择解释。
- 输出：一份字段合同、一份逐日/会话覆盖表及每项可识别/不可识别结论；优先复用现有字段核对，不重复下载大文件。
- 门槛：仅证明数值相近不通过；不能认证独立同步估值或盘后队列则M1仍不计算收益；只有4日也不放宽原30日条件。先回答字段可用性，再决定是否存在值得定向补的具体来源。

结果：20,863行正IOPV及四日恒定尾段已核对；四日盘后新增/删除委托0，只有各1条S状态消息。全181日15:05后也只有10条S、0条A/D，而成交1,030条；不能估计个人填单率。官方低/高精度IOPV字段不能自动映射到厂商重标度的iopv。盘后UA3108没有IOPV字段，混合导出尾段的正值究竟携带或真实更新仍未识别。**M1合格参考日0，收益NOT_COMPUTED；机制未拒绝。**

### PR-E02：181日的时钟、会话和60日基线资格矩阵

状态：**COMPLETED_ALL_181_DAY_ADMISSION_MATRIX；ORIGINAL_M2_NOT_ADMITTED**。[矩阵](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/03_日期时间资格矩阵.parquet)、[可直接查看的CSV摘要](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/03_日期时间资格摘要.csv)、[候选日期表](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/06_逐日可评价候选.csv)。

- 输入：已存543文件及manifest；流式逐日处理，只保存小型资格/缺失摘要，不复制全库。
- 方法：沿用旧固定字段与时钟规则，为每个HH:MM保留引用消息、累计成交与快照时刻范围、报价年龄、缺档及制度；区间无法压到1秒时保留不确定，禁止按收益找最佳平移。
- 输出：日期×HH:MM×所需字段的资格矩阵、先前60日可用数、缺失原因、119/62制度拆分、候选可评价日列表。
- 门槛：59/2只是源日期数量的上限线索；若原定义仍不可识别，完成来源结论即停止正式M1/M2收益计算，不修改基线长度造样本。此项可与PR-E01并行做数据层工作，但不提前看策略标签。

结果：47,784时点，含42,897原连续分钟网格及15:00、15:05—15:30探针；引用源行/hash、年龄、固定价带、累计量前缀和条件范围逐点保存。全部857,923条连续报价为整秒，907,199条行情trade_flag均空；839,712条连续报价有精确量前缀，419,592条包含名义上更晚的成交。在旧截断秒假设下838,570条范围相容，这不证明生成/接收时钟或累计量与盘口同次更新。

| 2026-07-06制度分组 | 源日期 | 日历够60过去日的日期 | 本地字段及60日候选：日期/时点 | 旧条件诊断及60日候选：日期/时点 | 原严格候选：日期/时点 |
|---|---:|---:|---:|---:|---:|
| PRE | 119 | 59（首4月8日） | 59 / 13,275 | 17 / 2,777（首5月21日） | 0 / 0 |
| POST | 62 | 2（首9月29日） | 2 / 450 | 0 / 0 | 0 / 0 |

本地字段对采用当前固定10bp完整可见盘口与同会话5分钟前的有效源报价；缺行不作零深度，过去计数不含当前日/另一制度。条件诊断在同一引用源行上额外要求旧假设区间完全在端点前。**引用政策固定为最后一条名义time不晚于端点的源行；条件17/0只描述该选择政策，不是穷尽所有更早源行后的可用性上界。** 没有验证截断假设或真实接收，因此也没有据条件17/0判定市场机会不存在。连续5秒年龄门槛只约束M2；附加盘后探针的年龄仅描述，不用其5秒结果直接否定M1。

543文件内容/行/schema与manifest全部一致，保存矩阵过去计数复算通过；另用独立逐行计数检查238,920项过去数量与47,784项同会话5分钟配对，输出14文件hash一致。[独立来源复核回执](evidence/pressure_source_admission_20261002_verification.json)只验证来源矩阵，不是独立金融验证。19项相关测试通过。新报告8,895,494字节、约8.48MiB，不复制原始库、不产生新下载或费用。[注册回执](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/registration.json)、[保存复算](../reports/research/510300_pressure_recovery_v1/source_admission_20261002/08_保存结果复算.json)。M1/M2正式事件检验**NOT_RUN_SOURCE_GATE_FAILED**，净期望/夏普**NOT_COMPUTED**，不是0收益账户。

### PR-E03：所缺免费历史字段与来源可行性

状态：**COMPLETED_BOUNDED_FREE_SOURCE_FEASIBILITY_NO_NEW_ADMISSIBLE_SOURCE**。[冻结元数据合同](../config/510300_pressure_free_source_feasibility_v1.json)、[结果](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/summary.json)、[逐源合同](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/source_contract_review.json)、[可读结论](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/免费来源可行性结论.md)。

- 输入/方法：PR-E01/02具体缺口；先保存公开目录、定价/字段页与失败响应，再以现有只读认证跟进小型字段文档。公开文档联合预算8MiB、ZIP目录/表头1MiB，零费用；不新建账号、不提交新访问申请、不读未来价格标签。完整目录与首页截断列表严格区分。
- alphat01—04：2023/24/25/26分别列242/240/242/100个7z日档案，版本hash保存；2026日期1月5日至6月5日，含2月24日partial。卡片称99个完整日、partial只缺301381.SZ，未以目标原始样本验证。两个小字段文件的实际认证请求均403；不存在本轮已提交的待审批作业，也未证明与venvoo是独立来源。其历史年份可待有实质证据再研究，不能增加7月6日后的同制度日。
- wind17：2026.zip597,946,428字节，5,486成员完整目录没有510300命名CSV；只抽查另一个证券bj920000的表头，字段为K线结束时间/OHLC/量/额。未证明目标覆盖，不能推断整库同模式或所有编码形式都无目标；没有完整下载570MiB归档。
- QuantDB：官方公布注册免费1GB和免费预览，不能说全部收费。公开Tick字典为成交时刻、买卖五档，未列IOPV、历史接收或盘后队列；没有新账号或合格样本。五档有时可以覆盖10bp，缺口判断依据仍是未认证的必要字段，不是机械拒绝五档。
- 次方：公开历史分钟IOPV存在，但hist_min/hist_intraday为SVIP权限；没有购买或得到零费用合格样本。其分钟时间标开始，不能在09:30用完整09:30—09:31数据；实时updated_at不证明历史IOPV经济时点。该bar规则不能移植给未核对的其他源。
- 官方合同/现有源：认证README200且与固定版本相同；补充2026 STEP的MDE01与接口族边界，不作跨接口推断。有限GitHub搜索未取得目标历史数据；软件仓库和搜索零结果不证明全部免费来源不存在。

结果：**0个新准入来源、新增市场表/解析市场行0、正式事件未运行、期望/夏普未计算。** 直接HTTP回执35次：28文档、1ZIP目录元数据、6准确206分段；响应1,433,269字节约1.367MiB，不含浏览检索通信及TLS开销。47项保存响应、注册输入、预算及目录核对通过，[本地复核](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/verification.json)不是独立金融验证；分段原二进制未保存，hash回执也不是独立远端重验。文件索引与具体访问正文均已保存。

具体下一步是**有条件的小样本准入**：先有实际可免费读取的510300合同，补接口/流、单位、经济时点、历史可得性及误差界；M1还需制度后独立参考和盘后执行证据。仅在该入口实质新增时，另登记少数日期、最少列与预算的小样本，然后重做受影响资格；通过后才进入PR-E04。当前没有可直接启动收益实验的输入，不重跑不变543文件、不盲下2017年以来同类大库、不按收益择源、不缩短60日或30独立日。原目标剩余工作列于[完成条件对照](../reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002/original_goal_remaining_requirements.json)，来源工作完成没有被记为金融目标完成。

### PR-E04：来源通过后的固定M2与价格对照

- 启动条件：原字段、时钟、60合格日基线与同步参考准入；冻结新批次全部输入和比较后再读取未来标签。
- 方法：用原5分钟冲击/观察和10分钟片段合并，报告所有合格及不合格事件；与原同时间/同制度/过去收益与波动五分位的价格对照比较；按交易日聚类。
- 结果层级：只有价格恢复可测时仅报价格诊断；实际成交或独立验证的重放、剩余净空间、T+1退出均成立后才运行20万元主账户与2万元敏感性。记录全部现金日、未知成交、未解决库存，不拿瞬时中价回升作可交易收益。
- 停止：优势不能超过合法退出前后费用和共同篮子风险，或原检验不通过，即冻结拒绝；不挑退出期限、波动档或盈利日期。

### PR-E05：波动状态作为机制分层，而非旧切换策略重试

先按原机制已登记的过去波动分组报告覆盖与效应，并区分源质量、事件频率及成交条件；不再扫描旧RV20切换阈值。只有发现一个决策时可测、在旧机制之外的具体信息差异，才另立一个增量版本及同池删除该信息的对照。新来源取得不自动恢复已经失败的价格代理账户。没有实质增量时，保留“不立项”。

### 最新四表交付与当前受阻点

状态：**COMPLETED_CURRENT_FOUR_TABLE_DELIVERY_FINANCIAL_GOAL_NOT_ACHIEVED**。[交付说明](../reports/research/510300_pressure_recovery_v1/four_table_delivery_181day_20261002/181日四表交付说明.md)、[完整Excel](../outputs/01a0f739-6bb2-7741-8d30-204e31d95c62/four_tables_181day_20261002/510300_压力修复_181日四表.xlsx)、[保存核对](../reports/research/510300_pressure_recovery_v1/four_table_delivery_181day_20261002/verification.json)。这是原四表要求的覆盖更新，不是新金融实验。

- **当前交付**：同步行情47,784行、29列，全181日264格/日；事件/订单只保留完整字段定义，无合格正式事件或订单登记；结果按原3版本×20万元/2万元保存6行未计算状态。另含181日覆盖、543源文件身份、字段说明及公开来源，工作簿共9页，6,726,030字节，约6.41MiB。研究规模是参数，不是账户余额或仓位。
- **未知仍然未知**：名义报价、价差、深度、源行时间和年龄只能描述保存来源，不能认证同步状态。经济/接收时间、参考估值及误差、PCF字段留空；原始IOPV零值不是有效参考，正值单位和流映射仍待证明。网格均NO_VIEW；正式事件总数为未知，0登记行不能解释为市场没有机会；6行收益、胜率、盈亏比、尾部、成交率及夏普均NOT_COMPUTED。
- **验证范围**：33项通过；Excel与CSV逐格比较1,392,579个数据单元格，CSV与资格矩阵数值比较477,840次，5个覆盖公式及缓存一致，九页保存视图已检查。两次原生全量导出因内存终止，改用流式写出仍保留全部网格/列；编排命令退出1，独立保存核对退出0，不能称原生全量导出成功。原957观察Excel哈希未变。文件一致性与页面检查不验证历史行情真值、估值/接收时钟、成交或策略。
- **完整目标对照**：[21项要求核对](../reports/research/510300_pressure_recovery_v1/four_table_delivery_181day_20261002/original_goal_completion_audit.json)区分来源覆盖、表结构、测量及金融结果；实际成交后净期望、T+1退出、完整账户目标、尾损与独立验证仍未满足。[连续受阻依据](../reports/research/510300_pressure_recovery_v1/four_table_delivery_181day_20261002/blocked_audit.json)记录三轮均有进展、但同一金融来源缺口持续存在；当前剩余金融工作需要新合格输入或外部状态变化。该判断仅属于本盘口任务，不能改变宏观或技术线权限/状态，也不证明所有免费来源不存在。

**最值得继续的是补原缺失证据，而不是扩参数搜索。** 实际出现可免费读取的510300接口/独立流映射、经济时点、历史可得性及误差界或执行证据后，先登记少数日期、最少列的字段验证；原来源门槛通过才运行PR-E04固定事件/价格对照，执行与合法退出成立后再运行完整账户。更多相同导出年数、重复交付、重算不变543文件或恢复RV20失败代理，均不能补齐上述条件。本轮新增市场请求、原始三流重扫、费用及订单均0，原机制配置与冲击时已知恢复基线未改变。

### 重新继续后的首轮：新候选字段与现有版本核对

状态：**PR-RESUME-01，COMPLETED_NEW_PUBLIC_CANDIDATE_SCREEN_NO_ADMISSIBLE_SOURCE**。[注册](../reports/research/510300_pressure_recovery_v1/resumed_source_screen_20261002/registration.json)、[逐源字段裁决](../reports/research/510300_pressure_recovery_v1/resumed_source_screen_20261002/source_contract_review.json)、[可读结论](../reports/research/510300_pressure_recovery_v1/resumed_source_screen_20261002/恢复后首轮来源结论.md)。此前四表/文档更新是PROGRESS；本轮新增候选核对也有实质证据增量，但没有形成合格金融输入。

- CNEquity固定版本`1192a51d8ac304572db5000c0ff286c4f8e29488`的`trade_ticks`是聚合分笔，时间只有分钟；只读协议reader和adapter与字段说明一致。当前数据集没有同步报价、IOPV或逐笔委托队列，direction为推断值，fetched_at为抓取时间；不能认证原1秒同步/5秒报价年龄，不把行序号补成秒或将抓取时间回填为历史接收。基金价格除数1000不能移植为venvoo的10000；2024-01-02客户端请求下界不证明目标覆盖。
- LARA论文v4所列2020年ETF样本为159915、512480、512880、515050，未列510300；未取得可读目标数据，不把其他ETF/股票/加密资产的模型结果移入本任务。该裁决仅限已核对材料，不推断作者全部数据不存在。
- 当前venvoo仍是原SHA、7,117路径，没有新版本/字段合同增量；未重扫543原始文件、未重导四表、未重复请求同版字段文档。8个直接公开请求全部200，共856,356响应字节，低于2MiB预算；两个源文件Git blob身份一致。0费用、0市场数据提取、0外部代码执行、0策略代码修改，正式事件仍NOT_RUN_SOURCE_GATE_FAILED，收益/夏普仍NOT_COMPUTED。[20项保存核对](../reports/research/510300_pressure_recovery_v1/resumed_source_screen_20261002/verification.json)只验证文档证据。

目标恢复到active后，[本轮受阻审计](../reports/research/510300_pressure_recovery_v1/resumed_source_screen_20261002/resumed_blocked_audit.json)从第1轮重新记录，尚未满足再次blocked的三轮条件；不能沿用此前计数。没有等待中的活跃下载或申请。下一步仍须有原时点/估值/执行字段的实际新增证据，才登记少数日期的小样本准入；原PR-E04与完整账户未启动，不放宽原协议、不恢复已失败代理。

### 重新继续后的第二轮：历史Tick接口与账号许可

状态：**PR-RESUME-02，COMPLETED_VENDOR_CONTRACT_REVIEW_CONDITIONAL_CANDIDATE_NO_LICENSE_NO_SAMPLE**。[逐接口裁决](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/source_contract_review.json)、[可读结论](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/历史接口与免费许可核对.md)。

- 米筐get_price文档明确提供历史日期、ETF、tick频率与iopv/五档字段，保留为候选；get_ticks/get_live_ticks仅当前日，不能替代历史接口。get_auction_info说明股票盘后申报量，510300新制度ETF覆盖尚未验证；聚合量不建立个人排队或成交事实。交易所datetime不自动建立历史客户端接收、独立IOPV时点或误差界，五档能否覆盖±10基点须实际逐样本判断。
- RQData手册的试用日配额1G不证明实际许可；Quant平台版本权限也不能外推全部RQData账号。用户已回答没有账号，[账号记录](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/user_account_status.json)据此标记NO_ACCOUNT_NO_GRANTED_LICENSE。没有创建账号、申请试用或购买数据。掘金两份公开原页均502，本轮未确认完整字段合同。
- 本轮5个公开请求、3个200/2个502、676,132响应字节，低于4MiB预算；费用、市场样本、订单和策略代码修改均0。原配置及上一轮保存输入保持一致，正式事件仍NOT_RUN_SOURCE_GATE_FAILED、收益仍NOT_COMPUTED。[保存核对](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/verification.json)仅验证材料与状态。

下一步为[单日最小样本计划](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/minimal_sample_plan.json)，PROPOSED_NOT_RUN：实际取得零费用有效许可后，先确认510300合约身份/配额，再取2026-09-30 Tick及有覆盖时的盘后量；验证时钟、IOPV单位误差、深度覆盖与执行数据。单日不满足原60日基线/30独立日要求，不调用time_slice假装流量受限，不修改原恢复基线或退出。恢复后[受阻记录](../reports/research/510300_pressure_recovery_v1/resumed_vendor_contract_20261002/resumed_blocked_audit.json)为第2轮、目标active；没有已确认的活跃申请/下载可等待，完整金融目标仍未完成。

### 恢复后的第三轮：原来源缺口复核与blocked终态

状态：**PR-RESUME-03，COMPLETED_GATE_RECHECK_GOAL_BLOCKED**。上一轮PR-RESUME-02是PROGRESS；本轮NO_PROGRESS_REVALIDATED，未启动新金融实验。[当前门证据](../reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/current_gate_evidence.json)直接复核181日来源准入结果、空事件/订单表及6行NOT_COMPUTED结果；严格M1参考日、M2字段格、60日候选格均0。原配置与原目标身份一致，用户仍无RQData账号/实际免费许可。已知盘口脚本进程匹配0，无确认存活的下载句柄或已提交申请可等待。

[三轮审计](../reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/resumed_blocked_audit.json)仅使用此次恢复后的三轮，不继承旧阶段计数；原历史可得时点、同步估值误差及执行证据的同一缺口持续存在，[已知下一步入口](../reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/next_action_audit.json)需外部新证据才可执行。目标工具已成功返回blocked，见[真实状态更新](../reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/goal_status_update.json)。金融目标未完成，原机制未被否定，此状态不决定其他研究线权限。

[恢复条件](../reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/当前阻塞与恢复条件.md)：实际可读且带原时点/单位合同的510300免费样本；已授予零费用历史许可后先执行现有单日准入；或独立时钟、IOPV误差及执行证据实质新增。账号/单日通过均不替代原60日基线与30独立日。此次9项保存检查不验证金融收益，费用、行情请求、订单及策略源码/配置修改均0。原RV20失败保持冻结，原PR-E04及全账户收益仍未运行。

## 15A. 当前分钟过程主线：固定实验完成，增量未通过

用户已明确调整优先级，完整新目标保存在[目标原文](../reports/research/510300_intraday_process_increment_v1/goal_objective.md)。当前问题是：控制日涨跌、振幅、成交额和近期波动后，冲击修复持续性A、相近活动下推进变化B、固定尾盘变化C是否增加可到达合法跨日成交时点的信息。只使用价格/量额代理，不识别真实订单流、撤补单、机构承接或同步公允价值。旧M1/M2来源缺口与blocked是历史储备状态，不能据此再次阻塞新分钟研究。

### 实际输入、旧重叠与冻结口径

- [实际数据清单](../reports/research/510300_intraday_process_increment_v1/01_实际数据清单.csv)已读取24项文件并记录路径/哈希/覆盖/字段/单位/时钟/用途/限制。主分钟2021-08-12至2026-08-12，共291,851根、1,211日；原字段未改。严格VWAP越界5,489根/1,166日，超过一档0.001元的42根/38日使相关窗口失效。仅45日严格全日金额合格，因此本轮一档容许是代理精度假设，不能声称全面修复或严格金额验证。
- 旧较长NBS分钟实际566,350根、2,350日，2017-01-03至2026-09-04，哈希与旧裁决候选相同；263根异常仍在，只准八个固定五分钟用途。旧广度缓存2,823日的breadth20非缺失2,350日、缺失473日，而综合状态只有1,280日VIEW_ALLOWED；[逐日字段可用性](../reports/research/510300_intraday_process_increment_v1/19_旧广度逐日字段可用性.csv)保留区别。两者均未加入本轮模型或补长样本。
- 已读旧RV20价格代理、午后时钟/退出、EMV和尾盘研究。C复用旧30分钟/60日金额标准化字段，旧单变量REJECTED_GOVERNANCE_GATES_FAILED保持；本轮新增的是对日线基准的条件增量用途，不能称新特征或重开原失败。旧RV20拒绝不变。
- [冻结协议](510300_INTRADAY_PROCESS_INCREMENT_V1.md)、[代码与输入冻结](../reports/research/510300_intraday_process_increment_v1/freeze.json)早于本轮收益标签、拟合与账户计算。固定五分钟触发、30分钟观察及合并事件；A保存未修复/修复/再次失守，B使用固定相邻窗口，C不选尾段参数。D八个日线变量；D+A/B/C分别只加一组。成熟训练至少504日、每20交易日更新、岭alpha=1；60日来源基线与2026-07-06制度分开，后一段仅28日不足训练，保留NO_VIEW。
- 盘后信号，T+1开盘至T+2收盘主合同；第1/2/5日仅并列路径诊断，第1日不能当作新份额合法退出。2万元/20万元×BASE/STRESS×D/A/B/C共16账户，100份整手、最低佣金5元、成交容量/跳空/未成交/库存/分红/现金均保留。分钟价格和竞价容量为条件执行代理，实际成交验证数0；历史曾被多次观察，不是独立样本。

### 固定结果与经济含义

[结果摘要](../reports/research/510300_intraday_process_increment_v1/summary.json)：1,574冲击中618未修复、532修复、424修复后再失守；869共同合格输入日生成364个成熟共同预测日、95次固定拟合。三组相对D的MSE均增加，A/B/C为1.717%/1.668%/0.107%；A持续性相对已有反弹的增量区间也跨0。B的校正后误差改善区间全负。三组没有通过原预测增量门。

|20万元、基础费用|全账户年化收益|净夏普|最大回撤|完成周期|周期净期望|
|---|---:|---:|---:|---:|---:|
|D：仅日线|11.953%|0.975|-9.561%|72|0.325%|
|D+A：修复过程|9.342%|0.770|-13.729%|78|0.243%|
|D+B：推进变化|7.584%|0.654|-10.586%|79|0.198%|
|D+C：尾盘条件|12.811%|1.039|-9.561%|71|0.350%|

完整评价2024-08-30至2026-08-12共471日，预测日止于2026-03-06。后108账户日NO_VIEW并保留完整计价，该段恰为空仓，不能称避险成功。[全部账户](../reports/research/510300_intraday_process_increment_v1/12_全部账户指标.csv)包含2万元及压力费用结果；没有任何A/B/C达到夏普1.2。C相对D的CAGR高0.8580个百分点，但逐日配对年化算术增量仅+0.7605个百分点、校正区间[-0.0683,+2.5083]个百分点跨0；该区间不能误用于CAGR差。

[收益集中度](../reports/research/510300_intraday_process_increment_v1/16_全账户收益集中度.csv)：四组最大贡献都是2024-09-27至09-30同一笔市场机会，D/A/B/C分别占总净利润59.98%/77.22%/95.37%/55.75%。同路径毛报价损益减佣金及滑点等于净损益，A/B毛预测已经变差，不能只归咎交易成本。**2026-10-02封卷修正：总利润集中度不能解释C相对D的边际差额。** C−D累计3,729.8021元，其中2024年9月差额为0，最大正月差为2025年11月2,873.72338元；月差包含资金路径后效，仍不能证明独立alpha。旧报告“C正点值受共同大涨阶段影响”的边际归因不成立，原文件保留并由[纠正说明](../reports/research/510300_daily_native_baseline_v1/封卷修正与设计分类.md)覆盖该解释。没有删除最大交易或据月度贡献改策略。

五项必要测试通过；16账户7,536账本行、2,024模拟请求、1,004合并情景完成周期已保存复算，指标最大误差1.11e-16。合并情景周期数不是独立机会数量。后续解释脚本只补数据盘点和原账本归因，新增拟合/账户0，原冻结代码、输入和核心结果未改。报告目录约6.53MB；本轮新增行情下载、数据费用、真实订单均0。

### 当前处置与下一步

当前状态为`COMPLETED_FIXED_MINUTE_PROCESS_INCREMENT_STUDY / NO_CONFIRMED_INCREMENT_KEEP_FIXED_NEGATIVE_RESULTS`。本轮具体研究交付完成，未建立盈利策略或独立验证；旧盘口原收益保持`NOT_COMPUTED_RESERVE_ONLY`。PR21—PR25记录各方向的接受/拒绝理由。固定A/B/C停止调参，不通过改变符号、窗口、退出、共同样本或费用营救；不直接把广度叠加为第四组。

原A/B/C结束时已完成信号、费用、容量/缺失与集中度解释，当时没有已准入的下一策略实验。随后用户授权的第一批设计诊断见第15B节，并非A/B/C参数营救。后续若提出新过程信息，先说明与旧库的实质不同用途和新增独立证据，冻结单一增量对照后才计算；同用途且同已暴露样本则停止。现有NBS八窗口和广度缓存仅作为已核实的范围清单，不自动扩大用途。原入口为[完整结论](../reports/research/510300_intraday_process_increment_v1/研究结论与下一步.md)及[交付对照](../reports/research/510300_intraday_process_increment_v1/20_本轮交付对照.json)。

2026-10-02按用户要求生成[Pro审阅包](../deliverables/510300_分钟过程增量_V1_Pro审阅包_20261002.zip)：22,533,913字节，101成员/100索引项，SHA256 `cf905b3cba0d24ed2683ef02f79d9b950a8d98a06b9713fc3cdedd356e1abcef`。包含本轮全部结果、24项清单来源、23项冻结文件、原代码/配置/测试、CSV可读账本及审阅提示词。[交付校验](../deliverables/510300_分钟过程增量_V1_Pro审阅包_20261002_交付校验.json)确认CRC/索引/哈希，以及解压副本16账户、364日预测点统计和集中度复算；无新拟合/账户/抽样。原bootstrap逐次索引未保存，本次没有复算区间。用户自行提交，尚未外部审核或上传；包内项目状态为打包时快照，此交付记录写于ZIP生成后，不改变原研究结论。

## 15B. 第一批完成：封卷修正、依赖拆分与D-native基准

用户提交Pro路线图和48项清单后，明确只选择“先完成第一批：封卷修正、依赖拆分与可运行基准”。[授权回执](../reports/research/510300_daily_native_baseline_v1/authority.json)保存范围，附件副本位于同目录`received/`。原ZIP未被替换；第15A节“尚未外部审核”是原打包时状态，现已收到用户转交的Pro文字意见，本批核对其四项具体主张，不等于外部验证本批新结果。

### 四项修正及实现范围

- 原108日共同门阻断拆为80日C当期/60日历史金额资格不足、28日2026-07-06后新制度预热；全部108日的D八项日线输入均有限。这是原预注册共同门的设计限制，不是代码暗中违反原合同。
- 从95组保存参数复算1,820个旧预测值，无须重拟合。当前原点预测未要求自身未来标签成熟；训练只纳入已成熟的历史标签。364预测日与363账户内有效信号日的差异来自2024-08-29首信号和08-30首账户日。
- C−D人民币边际差额已纠正，见15A及[逐月差额](../reports/research/510300_daily_native_baseline_v1/03_C相对D逐月人民币增量.csv)。C区间跨零的旧裁决未变，也未重抽区间。
- 旧来源摘要确认181日/543文件及79,288,590行记录；本批未重读整套三流，严格估值、字段、时钟和成交用途仍未准入。来源存在不等于原M1/M2可回测。

[新冻结协议](510300_DAILY_NATIVE_BASELINE_V1.md)将输入资格、模型资格、预测资格、执行资格分开。D-native直接从日线/分红重建原八项字段，在原1,211个原点逐值一致；D/A/B/C分别有1,211/1,123/1,122/870个输入合格日，旧共同门869日。训练仍从2021-08-12开始，Ridge alpha=1、至少504成熟日、每20日更新、标准化截断±5保持；36次顺序拟合属于一个设计，A/B/C仅拆资格不重新拟合。新的训练成员、首次拟合时钟和模型更新日期也改变，不能将全部收益差归因于一项门控。

D日频定义跨2026-07-06延续是条件迁移假设，不证明分布不变；尾盘字段与执行机制影响另列[制度矩阵](../reports/research/510300_daily_native_baseline_v1/19_制度影响矩阵.csv)。同一T+1开盘至T+2收盘、整手、现金/库存、最低费用、容量、分红、未成交及固定退出合同复用；预测不再被C金额历史阻断，但执行仍核验实际需要的分钟字段。真实成交数0。

### 固定运行结果

累计706个原点可预测，首日2023-09-11，前505日明确为模型训练不足。原评价首信号至末日472个预测、2024-08-30至2026-08-12完整471账户日均有预测，四账户各0日NO_VIEW。主口径242交易日，现金及无风险收益均按0；252只是同净值桥接，不用于选优。研究本金与费用不是实际账户确认。

|20万元、同一471日|年化收益|净夏普|最大回撤|完成周期|
|---|---:|---:|---:|---:|
|原D-common / BASE|11.953%|0.975|−9.561%|72|
|D-native / BASE|1.021%|0.142|−16.978%|93|
|原D-common / STRESS|10.659%|0.921|−9.995%|47|
|D-native / STRESS|−0.603%|0.015|−17.541%|54|

原364共同预测日期MSE从0.0003929839774降至0.0003680022673，点值下降6.357%，没有计算新区间或确立统计优势。20万元BASE的native相对common毛报价损益少38,828.40元、佣金和滑点多6,328.96元、净利润少45,157.36元；毛收益下降与额外摩擦并存，不能只归咎费用。旧80日缺口和新制度28日的native净损益分别−14,928.87元、−11,707.86元；保留全部日期，不能事后恢复C缺失门作为择时。没有达到本分支夏普1.2及年化10%的研究参考，独立验证NOT_ESTABLISHED。

原5项与新增7项测试共12项通过；4账户1,884账本行、592模拟请求、294个合并情景完成周期保存复算通过，指标误差0、账户恒等式最大残差3.48e−11。67项冻结文件与原结果未改；合并周期不是独立机会数。金额诊断没有发现整体100/1000倍单位错配，严格计数5,489与容差金额比较5,487的两条浮点边界已解释，超过一档的42条资格未变；源精度/对齐根因仍UNKNOWN，未插值或以量乘价填补。

### 可运行入口、交付和停止点

日状态命令为项目Python执行`scripts/run_510300_daily_native_baseline_v1.py --stage status --date YYYY-MM-DD`，见[运行合同](../reports/research/510300_daily_native_baseline_v1/依赖与运行合同.md)。2026-10-02实际执行返回`NO_VIEW_NO_CURRENT_LOCAL_RECEIPT`，预测为空、实际仓位UNKNOWN；现有分钟原点末日2026-08-12，历史模拟不能证明当前接收时钟或真实执行可用。`--stage run`已完成且有单次声明，不为重复展示而再拟合。

[第一批报告](../reports/research/510300_daily_native_baseline_v1/第一批研究结论与接续.md)、[48项逐项对照](../reports/research/510300_daily_native_baseline_v1/21_48项清单本批落实对照.csv)、[完成回执](../reports/research/510300_daily_native_baseline_v1/completion_receipt.json)及[文件索引](../reports/research/510300_daily_native_baseline_v1/file_index.json)为接续入口。A01—A06、D02、D04—D07共11项完成本批明确工作；G02/G03/D01/D03共4项分别保留口径范围、家族范围、24项资产盘点和根因未知边界，其余33项未启动。下一批仅待新信息与旧用途差异准入；不根据本批亏损改窗口/参数/费用/月份或复活冻结失败。没有新增行情采集、自动任务、前瞻试验或实际订单。对应PR26—PR29。

第一批后补交付：[第一批Pro审阅包](../deliverables/510300_第一批封卷修正与D_native基准_Pro审阅包_20261002.zip)，24,482,518字节，183成员/182索引，SHA256 `8b818b8ffbe9c99c1dd1c9674893a495d427cd150da81a2b250483a69ab958fb`。复用原包前核对原SHA；新包解压副本通过原16及新4账户、706保存预测、共同364日MSE和C−D月份差复算，0新拟合/账户/抽样。该包在用户追加全部任务之前生成，仅覆盖第一批；新路线图结果见下节。未上传或宣称新结果已由Pro审核。

## 15C. 用户扩大为全部48任务：当前可做事项完成，条件阶段明确保留

2026-10-02用户追加“其余的都进行”，覆盖第15B的第一批后续停止范围，但不抹去各个研究的科学准入条件、旧冻结失败、零费用及实际交易另行授权。见[授权](../reports/research/510300_roadmap_execution_v1/authority.json)、[执行协议](510300_ROADMAP_EXECUTION_V1.md)、[全部报告](../reports/research/510300_roadmap_execution_v1/全部路线图执行报告.md)和[48项台账](../reports/research/510300_roadmap_execution_v1/48项执行台账.csv)。当前21项所列范围完成、11项有限/部分完成、16项待条件；不能称48项科学验证全部完成。

### 机制与源范围核对

- [96因子机制表](../reports/research/510300_roadmap_execution_v1/mechanism_registry.csv)保留16家族、定义、反例、额外观测、旧重叠与原卡来源。787条直接研究摘要构成[试验版本快照](../reports/research/510300_roadmap_execution_v1/study_registry.jsonl)，摘要数不是独立候选数，跨历史选择次数仍未完整识别。单一核准99篇资料索引未找到，不声称逐篇复核。
- R02融资：旧48观察/28收缩段平均五日净收益−0.1087%、胜率45.83%，F03残差及日频纠错结果分别保留；新沪市一日直接偿还汇总也不等于强平或独立行为构成。历史首版和两市完整来源仍未准入。
- R03 ETF：旧五条周度流量规则冻结拒绝；三只沪市300产品迁移HIGH/LOW压力夏普0.3703/0.1394、年化1.1912%/0.3126%，未过原深化门。新920只沪市ETF列表不能代替完整同指数产品范围、深市数据和NAV/份额调整时钟。
- R04事实调整：盈利广度/旧EPS/解禁失败保留；已有回购911候选链/108公司仍0交易字段准入。完整事件全集、方案去重、首次可知及指数暴露尚未联结，不能用事后叙事代替。
- R05只核真实IF持有成本通道，严格点时利率/已公告分红映射未准入，不重跑旧基差/总OI/全球隔夜。R06没有已过经济门的固定入场，不启动退出搜索。R07宏观只作解释账本，R08无新字段不重复导出181日。

各路线的一页假设、旧差异、真实字段、最小对照和停止条件见[机制裁决](../reports/research/510300_roadmap_execution_v1/mechanism_admission.json)及其`mechanism_cards/`。本轮0个新收益实验准入，不能推出所有可能机制永远无效。

### 保存结果的新统计与运行工程

事前固定本次算法、同时承认原结果已见：471完整账户日与364共同预测日分别5,000次圆形20日区块抽样，保存实际[抽样索引](../reports/research/510300_roadmap_execution_v1/bootstrap_indices.npz)。新增拟合/账户0，旧A/B/C区间不替换。20万元BASE的native−common**年化算术收益差**−10.051个百分点，单项95%区间[−20.652,−0.553]个百分点；本次5比较校正区间[−24.342,+2.048]个百分点跨零。它不是CAGR差区间，局部校正不消除全库选择偏差。

共同日MSE改善点值0.0000249817，单项95%区间约[−0.000004,+0.000079]跨零；较小误差未确立可靠优势。[完整区间](../reports/research/510300_roadmap_execution_v1/uncertainty_intervals.csv)保留五项。93个20万元BASE周期胜率49.46%、平均净赢1.3627%、平均净亏1.2530%、等权净期望0.0408%、最差周期−4.8189%，完整账户年化仍仅1.021%。费用已计入，不能再扣一次或用等权周期均值替代全账户。

全部月份、入场集合差、重叠持有事件簇与贡献置零敏感性保存；后者是分解，不是删事件可执行账户。新9项时间隔离、未知回执、区块、净期望、日历及失败路径测试通过。原第一批12项保持。新事件去重/非事件对照、延迟执行和正式组合集成在无合格源或组合时保留NOT_RUN，不虚称全通过。

### 免费公开源观察已运行并已安装任务

2026-10-02 16:15实际获取2026-09-30的沪市融资1行及ETF份额920行，含510300/510310/510330，原响应228,762字节。深市融资本次TLS连接失败，无返回行；无重试、收费或填补。原件及逐请求时钟在[接收目录](../reports/research/510300_roadmap_execution_v1/raw_vintages/2026-10-02/summary.json)，[有限字段表](../reports/research/510300_roadmap_execution_v1/received_public_facts.csv)明确经济日期与今天接收日期不同；不据此回填历史首次可得。

已安装`Codex_510300_Roadmap_PublicVintages_V1`，工作日09:15运行、官方非交易日跳过，pythonw后台、0费用、单次最多3请求、250MiB上限、2026日历过期停止网络访问。需要电脑可用且该用户登录。下次调度10月5日为休市跳过，首个交易日检查10月8日；未来采集成功须有实际回执。见[安装回执](../reports/research/510300_roadmap_execution_v1/scheduled_task_receipt.json)与[运行回执](../reports/research/510300_roadmap_execution_v1/scheduled_observer_last_receipt.json)。这是D08原始公开版本观察，不是F01/F02策略前瞻；0策略信号/模拟订单/真实成交。

### 尚需的条件与下一步

G01/P03已询问真实本金、期限、风险和券商费用，当前未知；研究2万/20万情景可继续独立使用。P04/P05因0合格模块不强行组合或调权重；F01—F03因无合格前瞻策略与未来执行观测未完成；F04仍待明确实际交易授权、F05没有扩容证据。M01月度表保留空金融统计，M02对所有历史盈亏对称登记但机制原因UNKNOWN，M03不重复无新证据的工程，M04不晋升。

下一轮先根据实际新增公开版本及一页卡检查新字段/范围/可知时钟；满足源和旧用途差异门后才注册唯一收益增量。没有实质新证据时，不重跑本轮准入或调参；观察任务独立保留版本。新日期和实际成交证据不能在当前轮次补造。对应PR30—PR34。

2026-10-02 16:29已实际触发公开源Windows任务，休市分支`LastTaskResult=0`，见[触发核对](../reports/research/510300_roadmap_execution_v1/scheduled_trigger_verification.json)；这仍不证明未来交易日联网成功。

本轮交付：[48项执行与未完成条件Pro审阅包](../deliverables/510300_48项路线图_执行与未完成条件_Pro审阅包_20261002.zip)，47,170,493字节，304成员/303索引项，SHA256 `6508cbae4185cb914ccd1377c37e9631a2299ad2960b8ed24861ac953be11059`。包含原实验和第一批证据、本轮66个报告/索引文件、43项冻结依赖、5项附加直接资产及新代码/协议/测试，旧导航保留在历史目录。解压副本复算原16和新4账户、706保存预测、本轮5组保存区间、4组周期分布和全部月份贡献；区间最大误差0、月份差最大误差约3.64e-12元。见[交付校验](../deliverables/510300_48项路线图_执行与未完成条件_Pro审阅包_20261002_交付校验.json)及[审阅提示词](../deliverables/510300_48项路线图_执行与未完成条件_Pro审阅包_20261002_给Pro的审阅提示词.md)。交付验证新增拟合/账户/随机抽样/联网均0；未上传、未获本轮外部审阅、未建立独立金融验证。包内共享文档是打包时快照，本段在ZIP完成后追加，不修改历史包。对应PR35。

## 15D. 已执行：纯日线、低频持有、完整交易政策的固定历史比较

2026-10-02用户明确要求用原审阅ZIP的两个CSV执行现有数据上的完整政策研究，随后提供`510300_零联网可执行研究启动包_20261002.zip`。**授权修订：不要求先有新数据，也不要求先证明孤立入场条件盈利；入场、持有、退出可作为一个事前固定合同联合评价。** 仍属于历史开发，旧失败不改、不称新信息或独立验证；前述第15C的来源及合格入场门不能继续作为所有后续研究的通用门槛。原始观察任务不受本离线脚本控制，本批没有启动或停止它。

入口：[用户启动包原样文件](../reports/research/510300_offline_daily_policy_batch_v1/launch/README_先运行.md)、[本轮授权与输入冻结](../reports/research/510300_offline_daily_policy_batch_v1/authority_and_input_freeze.json)、[固定参数](../reports/research/510300_offline_daily_policy_batch_v1/launch/rules.json)、[实际运行回执](../reports/research/510300_offline_daily_policy_batch_v1/local_execution_receipt.json)、[完整结果报告](../reports/research/510300_offline_daily_policy_batch_v1/analysis/首轮研究结果与结论.md)。启动包SHA256 `81c6f519f067156ec41b5011a811fce9e7394d154269196925315efd71ccd4ba`；原脚本及参数哈希与包内索引一致，未修改。

### 输入、合同与实际执行

只读取原审阅ZIP中`readable/normalized_prices.csv`的3,456条日线（2012-05-28—2026-08-14）与`readable/normalized_dividends.csv`的14次分红。股票/现金限定510300.SH与CNY；不读取分钟金额、盘口、IOPV或新的融资/份额输入。正式账户2015-01-05—2026-08-14，共2,823日；2012—2014预热，全部历史已用于开发。原样合成自检通过后只运行一批真实行情24账户，没有网格搜索。

TREND：总回报财富指数MA60、斜率20日、进入/退出连续两收盘；REPAIR：Z5<-2且财富回升，回均线/固定3sigma信号锚/信号满10日结束；MIX50两状态各半目标预算，单一现金和份额账户。VOL10只按历史波动预算，另列首次买入保持份额与STATIC50。风险预算10%、周五10个百分点调仓带；2万/20万×BASE/STRESS。收盘定订单、次开盘代理成交、限价、整手、现金/T+1、登记/除息/到账与期末持仓按收到代码执行。10%风险预算不保证最大亏损；开盘代理未验证真实成交。

### 20万元完整账户结果

| 政策 | BASE年化 | STRESS年化 | BASE夏普 | STRESS夏普 | STRESS最大回撤幅度 |
|---|---:|---:|---:|---:|---:|
| BUY_HOLD | 3.721% | 3.711% | 0.280 | 0.280 | 44.987% |
| STATIC50 | 2.054% | 2.051% | 0.237 | 0.237 | 27.034% |
| VOL10 | 4.711% | 4.567% | 0.461 | 0.449 | 34.090% |
| TREND | 1.068% | 0.651% | 0.178 | 0.123 | 25.520% |
| REPAIR | 0.162% | 0.129% | 0.297 | 0.234 | 0.962% |
| MIX50 | 0.716% | 0.546% | 0.200 | 0.158 | 13.209% |

全部24份[摘要](../reports/research/510300_offline_daily_policy_batch_v1/results_run1/summary.csv)、288行[年度结果](../reports/research/510300_offline_daily_policy_batch_v1/results_run1/annual.csv)、72行[固定分段](../reports/research/510300_offline_daily_policy_batch_v1/results_run1/segments.csv)、24套账户/订单/决策/周期均保存。三个候选四情景的年化均低于三个基准，夏普均低于VOL10；12候选情景及24全情景均无净夏普≥1.2且年化≥10%的结果。与D-native的471日结果期限不同，不直接用两个年化点值排名。

### 诊断、限制与本轮终点

- TREND每情景34个平仓周期，11胜23负；20万压力最大赢利周期26,641.52元超过全期净利润15,723.51元。其余周期净损益相加−10,918.01元仅为保存贡献分解，不是删除周期后重跑。
- REPAIR每情景4周期/10个收盘持仓日；20万压力平均仓位0.249%、年化0.129%。2015—2019全现金，夏普未定义；最大周期占净利润69.92%。低回撤和3胜1负不能证明稳定优势。
- MIX50的20万压力平均仓位14.529%，日账户收益与TREND相关0.990591；趋势/修复实际同时持仓0日，修复独占10日。时间分离存在但样本极少，仓位和交易路径也不同，未跑暴露匹配对照，不能据此认定互补因果增量。
- 原样合成自检通过；保存24账户67,752行的指标复算误差0，现金+份额市值+应收恒等式误差最大约7e-11元。只读解释阶段新增账户/拟合/随机抽样/联网0。未执行统计区间、全历史多重选择纠偏、前瞻验证或真实成交认证。

本固定批已完成，晋升候选0，设置到此结束，不自动调整均线、修复阈值、费用或权重。不同新版本可以继续使用现有数据，但必须先说明本轮具体诊断对应的有限新假设；历史不恢复为未见样本。后续直接读净值与交易差异，不重做D-native、108日解释或48项盘点。对应PR36—PR39。

已交付[完整results_run1压缩包](../deliverables/510300_纯日线低频政策_results_run1_20261002.zip)：4,246,760字节、142成员/141索引，SHA256 `07056707ea79b6e70d677ba85af7dfa3c7a73ee7957164fc44db14d2ae97b8e1`。原样保留results_run1全部103文件、启动包9成员、两个实际输入CSV、结果解释及净值图；不重复包含47.17MB原审阅ZIP。原脚本/参数、源ZIP和原始结果身份保持，[交付校验](../deliverables/510300_纯日线低频政策_results_run1_20261002_交付校验.json)通过；[给Pro的提示词](../deliverables/510300_纯日线低频政策_results_run1_20261002_给Pro的审阅提示词.md)仅围绕净值、交易差异和最多两项有诊断依据的新假设。已本地打包，尚未上传或获得本轮Pro审核；此交付记录写于压缩后，包内共享文档为此前快照。

## 16. 已过时状态、后续修订与禁止误读

| 容易读错的旧结论 | 当前应继承的事实 | 直接证据 |
|---|---|---|
| “V3冻结未运行” | 10月1日已完成：132模型、66重拟合、6旧合同账户；阶段模型MSE比背景恶化1.9168%；旧压力账户夏普约0.617、年化3.90%、回撤5.64%，没有目标通过；当前半仓风险合同账户仍未运行 | [完成复核](../reports/research/510300_pattern_daily_state_learning_v3/completion_review/result.json) |
| “旧混合被终止，所以连固定复核也未授权” | 9月25日明确授权原貌本地复现，保留原月度/最近20成熟周期规则；22依赖账户、39,006账本行已核对，四历史点值通过；不恢复调参/交易 | [授权修订](../reports/research/510300_selected_mix_reappraisal_v1/authority_update.json)、[复核结果](../reports/research/510300_selected_mix_reappraisal_v1/result.json) |
| “旧混合四点通过就是独立成功” | 主期BASE/STRESS夏普约1.284/1.219、年化10.89%/10.29%；较早期1.229/1.240、10.88%/11.03%；但存在308路径/65目标评价/8外层设置与共同选择历史，早期段也已看过；43周期中最大贡献42.30%、前五85.48%；3新增现金日、0新增独立周期 | [原策略重新评估](../reports/research/510300_selected_mix_reappraisal_v1/原策略重新评估.md)、[重现回执](../reports/research/510300_selected_mix_reappraisal_v1/reproduction_receipt.json) |
| “相邻版本仍盈利等于稳健达标” | D60相邻50/70版本夏普约0.824/0.686，仍可能有正优势但未达原目标；不能称原经济优势完全消失，也不能调权重救回 | 同上 |
| “1.5、1.3、1.2只有一个全局有效值” | 稀疏机会原1.5；既有数据训练后来1.3→1.2且有年化10%；盘口另设1.2；点位另有提高收益/夏普、pB和软频率合同。不同资金/风险/242或252年化各自保留 | [稀疏mandate](../config/510300_sparse_opportunity_mandate_v1.json)、[1.2修订](../reports/research/510300_sequential_patterns_regime_v1/authority_update.json)、第0节范围表 |
| “每年必须五次”或“完全没有次数门槛”可统一全项目 | 两个说法都缺研究范围。宏观完整年五次、点位后来软目标、盘口允许择机不交易、稀疏4/5机会仅参考；不强制为凑数交易 | 各分支配置/有效要求 |
| “免费历史只有几天、尚待申请” | 已成功取得181日；原blocked/local-inputs/初始status仅为阶段记录，不代表最新资料与授权。没有用后来的回执倒造过去已接收 | 第13节摘要/manifest |
| “数据完整，M1/M2收益就能计算” | 三流文件齐备与同步估值、完整消息、队列/实际成交不同；PR-E01/02已完成但原来源门槛未通过，事件检验未启动、收益未计算 | 第13—15节、来源资格summary |
| “原盘口blocked意味着整个任务必须等待完整盘口” | 用户新目标已经改为现有分钟过程增量；本轮A/B/C及16账户实际完成、增量未通过。旧blocked只记录原M1/M2历史，盘口/IOPV退为储备，不阻塞新主线 | 第15A节、PR21—PR25 |
| “C账户点值更高或交付完成，等于夏普/盈利目标通过” | C净夏普1.039低于1.2，配对增量区间跨0，预测无改善，收益集中、108日NO_VIEW；研究交付完成，可靠增量与独立验证未建立 | [分钟结果](../reports/research/510300_intraday_process_increment_v1/summary.json)及16/17表 |
| “降低费用或扩大账户即可救96因子” | 49原账本/245反事实已完成，最高夏普0.69417，最高年化8.5705%，无联合通过；主要缺原固定信号和持有退出的毛优势 | [瓶颈结论](../reports/research/510300_factor96_bottleneck_diagnostic_v1/瓶颈诊断结论.md) |
| “源码存在/测试通过/PASS=有效策略” | PASS必须指明来源、实现、复算或独立验证对象；NBS先来源通过后G2失败，成员分红先毛筛通过后账户失败，股东广度账户未运行 | [NBS最终裁决](510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md)、[分红账户](../reports/research/510300_constituent_dividend_calendar_v1/account_stage/result.json) |

维护时先更新第0节中发生变化的分支范围，再更新该分支的进展、待办及对应决策；技术、宏观、盘口三线各自更新，不把最后写入的分支设为全项目优先。不得用本次两份文档交付成功将任何尚未完成的金融研究目标标为完成。

## 日周线点位主线的接手入口与下一步

本段对应“日周线、多头优先、次数软目标、实际净pB>1、继续提高净收益与夏普”的技术点位任务。项目共同文档不设置唯一的“当前对话”；接手时应以用户正在续接的研究线为准，不能从宏观线或盘口线改写另一线的授权。

- 完整状态、代码地图、8个共同账户、数据限制、冻结合同、E01—E09及最新C05：[PROJECT_STATE_TECHNICAL_LINE.md](PROJECT_STATE_TECHNICAL_LINE.md)。86项完整假设/验证/结果/理由/重验条件：[技术线决策账本](RESEARCH_DECISIONS_TECHNICAL_LINE.md)。原快照保留，决策数非独立试验。此前“下一项/未运行”是当时状态，按后续同对象裁决接续；C05源/成员通过但两期预测拒绝，下一D03只旧用途/OHLC时钟/成员核查未接纳。
- E01及TECH.R64/65已完成并保留：原传递链及4账户现金流还原一致，单字段残差路线较早只有4成熟周期、0月过训练门，拒绝作为唯一改动。随后TECH.R66/67已在用户隔离代码授权下完成：日成交均价表达为旧ETF_CLOSE_VWAP，停止重试；E05训练标签按当时波动缩放并恢复原收益单位，115候选拟合及118旧对照复算，原参数误差0，但较早/近期原收益MSE扩大52.84%/74.90%，预测门失败。经济账户SKIPPED、收益/夏普NOT_COMPUTED，无新策略账户。[E05固定裁决](../reports/research/510300_point_volatility_unit_exit_v1/研究结论与下一步.md)。
- 该线当前仍未完成收益/夏普与抗过拟合目标：A压力近期年化3.99%、夏普1.217、pB1.073，较早为1.84%、0.438、0.611；B替代、NR7增次、统一正目标均未通过。它们不与宏观线242日、不同账户终点的高分混排。
- **E01已完成**，当阶段遵守用户先不修改代码；后来明确允许隔离代码，不追溯改写记录。E02仍未注册、未运行；日成交均价同表达停止，E05失败保留。E06原预测校准及未来标签误差归因完成，不能实时补截距。随后E07成熟原预测误差路线较早60月零支持、最大6周期，拒绝跨期唯一改动。E08不同的随机截距方差方法7项测试、64来源、25组合拟合/75分阶段估计，115旧参数核对；较早/近期原MSE仅降0.0169%/0.0563%，较早区间跨0、整体预测门失败。1,010自然原点符号变化0，经济账户SKIPPED，收益/夏普NOT_COMPUTED，[最新完整报告](../reports/research/510300_point_random_intercept_exit_v1/研究结论与下一步.md)。下一步只先检查F03价格无法解释的融资变化的旧定义、首次可知时钟及252日跨期覆盖；注册G2不等于来源通过，当前未接纳或拟合，不恢复旧融资策略。
- **E03已有冻结协议**：A原权重对POINT_BINARY，两费用、4账户；首新原点2026-10-08，首执行10-09，第1008交易日唯一正式终点。目前真实新账户日和完成周期均0。只对这条线适用，不为宏观或盘口线创建新任务。
- 已核实的旧状态更新：形态V3在2026-10-01已按原协议完成并失败，不能再写“未运行”；新增价量MSE扩大1.92%、压力夏普0.617、约98.34%利润来自一笔。旧货币统计M1/M2月度剪刀差研究是旧定义增量失败、新定义不足，**与第12—15节盘口M1/M2不是同一研究**；波动风险惯性不等于方向预测优势。

[原模型状态](../reports/research/510300_daily_weekly_goal_continuation_20261001/state.json)、[有效要求](../reports/research/510300_daily_weekly_goal_continuation_20261001/active_goal_effective_requirements.json)、[路径结果](../reports/research/510300_point_weight_path_bottleneck_v1/summary.json)、[V3现有结果](../reports/research/510300_pattern_daily_state_learning_v3/summary.json)。

运行环境仍使用项目`.venv\Scripts\python.exe`。`data/reports`分别映射到`E:\ResearchData\New project 8\data`与`reports`，`.venv`映射到D盘；仅恢复Git HEAD不能还原数据、结果及几千个未跟踪研究文件。Git与清点细目另见[技术线清点快照](../reports/research/project_state_snapshot_20261002/git_snapshot.json)。

### 技术点位最新接续：F03准入与首个E02-C06单字段（2026-10-02）

本段只更新技术点位线，宏观和盘口合同不变。用户“允许新增隔离实验代码，保留原冻结策略”的授权继续有效；原E05/E07/E08及旧失败保留。新增TECH.R73/74，完整记录见[技术线状态](PROJECT_STATE_TECHNICAL_LINE.md)和[技术线决策](RESEARCH_DECISIONS_TECHNICAL_LINE.md)。

F03原卡NOT_RUN不代表全项目没试：实际旧十卡初筛已有同类前序融资残差，20万元压力HIGH/LOW旧夏普−0.234/−0.332，未获后续细查。旧输入包含后来官方纠正的2024-08-08深市零值，不能把旧数值当纠正后结果或对整个机制的有效否定。已纠正融资2,674日止于2025-12-31，首版/迟报/改版仍未建立；假设次日时钟的数量支持不能代替历史可知性。本次以旧表达重合和首版未接纳结束，0新拟合/标签/账户，见[F03结果](../reports/research/510300_point_f03_information_intake_v1/研究结论与下一步.md)。

随后E02首个具体C06实现已登记并完成预测检验：完整60日历史收盘成交份额位于当前价上方1—3个当日ATR20两格的占比，作为原八项的第九项。它是日收盘位置的粗成交量代理，非筹码或主动卖单；旧签名量、低点锚定均价、价量相关逐项区别，原标签、原成员与训练参数不改。1,507状态/115成熟月份完整；7必要测试后冻结70来源，25候选与25对照拟合、90月复用，原参数及1,010配对保存预测复算误差0。较早6周期262原点MSE增加1.5397%，近期18/748增加0.6134%，两期门FAIL，经济SKIPPED。自然预测符号改变6次不等于新增交易；收益/夏普NOT_COMPUTED，见[C06完整结果](../reports/research/510300_point_c06_exit_prediction_v1/研究结论与下一步.md)。

下一项[A04计划](../reports/research/510300_point_next_information_intake_20261002/A04_candidate_intake_plan.json)仅检查旧结构表达、右侧2根完整日线确认时钟及原成员数量；尚未接纳、登记模型或拟合。不能只改采样频率或参数营救旧失败。E03原唯一前瞻比较保持，真实新日数及周期仍0。当前收益/夏普及独立验证目标未完成。

## 技术点位最新接续：A04固定预测裁决（2026-10-02）

此更新只接续技术点位主线，不改变其他分支的合同或状态。上节A04“尚未接纳/拟合”是当时准入意向；后续已经完成一个隔离的固定定义及预测实验，详见[A04完整结果](../reports/research/510300_point_a04_exit_prediction_v1/研究结论与下一步.md)及[技术决策TECH.R75](RESEARCH_DECISIONS_TECHNICAL_LINE.md)。

A04采用现金前向平移日收盘、严格左右各2根确认且b+2后可知的最新合法低点a至高点b，输出`(P[b]−P[t])/(P[b]−P[a])`，作为原八项的第九项。原经济标签、全部训练成员、周期等权、20周期/10周期100行、alpha1/clip5及实际入场锁定均保留；原比例可负或大于1，不据结果加最小幅度过滤。

1,507原自然成员及115成熟月份完整；7必要测试后冻结68来源，25候选及25原对照拟合、90月复用，115原参数核对误差0。较早6周期262原点MSE增加0.4418%、年度块改进区间跨0，FAIL；近期18周期748原点MSE减少1.3537%、区间下界正，PASS。整体`REJECTED_FIXED_A04_FIELD_PREDICTION_GATE_FAILED`，不能只取近期、换门槛或按年拼接。全部3,488字段及1,010配对保存预测重建一致，497原未知保留；自然预测符号变化5不是五笔新交易。经济阶段SKIPPED，新的收益/夏普NOT_COMPUTED，真实新期间日数/周期0，金融目标未完成。

下一项[C04回调成交收缩候选](../reports/research/510300_point_next_information_intake_20261002/C04_candidate_intake_plan.json)只提出旧定义、时钟和数量准入。C04原卡及程序均NOT_RUN，但依赖的T01组合已因接近旧研究拒绝重试。先比较旧供给测试/力度/签名量/价量相关及A04/C06，绑定回调e起点及结束、非回调状态、原成交额CNY与前5日基准，检验完整原训练成员支持。不得把未定义状态填零、删原行或组合失败参数强行通过。当前未绑定完整定义、未接纳、未拟合；原冻结策略、旧失败及E03唯一已冻结前瞻比较保留。

## 技术点位最新接续：C04数量裁决及G1旧表达对应（2026-10-02 07:21后）

上节C04只是A04裁决时的提案。此后完成一个隔离的固定数量实验：确认上涨段后，当前价在确认低/高之间才从当日开回调e，禁止追溯未知阶段；阶段平均成交额除以前5日均额，非回调、段失效和初始未知保留NO_VIEW，回撤/ATR仅背景。原T01组合及A04/C06失败保留，不取失败字段或系数。

6项必要测试、24来源冻结后，原1,507自然状态仅502有字段；较早230/658、近期230/748。原34/81成熟月份没有一个全成员完整，每月最少缺274/487原行。全3,488日字段有1,121可用，其他状态仅段失效/非回调/初始未知，没有坏金额缺口。保存的3,488字段、1,507自然及142月支持表精确重建一致，C04/A04/C06的24/68/70来源未变。终态`REJECTED_FIXED_C04_NINTH_FIELD_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT`；仅拒绝该阶段字段直接第九项用途，未检验整个成交收缩机制收益。新增拟合/收益标签/账户/行情0，预测误差及收益夏普NOT_COMPUTED，详见[C04完整报告](../reports/research/510300_point_c04_information_intake_v1/研究结论与下一步.md)和TECH.R76。

另完成[29项G1旧表达对应](../reports/research/510300_point_next_information_intake_20261002/G1_prior_definition_routing_20261002.json)（TECH.R77）：3项本技术线固定拒绝、9项旧初筛已有相关表达、9项程序记录组件/变体、1项相关均价旧拒绝、7项仍需更细核对。C05虽进度表NOT_RUN，旧代码已有同类成交额残差；C01涨跌冲击中位数比也已有实现。原完整卡与有限旧表达分别保留，不反写原表、不重算旧收益、不计作新独立试验或完整机制否定。

下一[C02候选](../reports/research/510300_point_next_information_intake_20261002/C02_candidate_intake_plan.json)只提出相对成交额、波动标准化价格推进两维及方向的旧定义/时钟/用途/数量准入。先与旧Amihud、残差和价量表达去重，绑定经济收益、rv20、中位数窗口及信息块用途；不能事后压分、增加未登记交互或以易通过数量门选择。当前未绑定、未接纳、未拟合。原E03不重置，真实新期间日数/周期0，当前完整收益/夏普和独立验证目标仍未完成。

## 技术点位最新接续：C02固定三列信息块预测裁决（2026-10-02 08:19后）

上节C02“未绑定/接纳/拟合”是数量候选时的状态，此后已完成一个隔离的固定模型实验。仅更新日周线技术主线，其他分支合同、原冻结策略及旧失败保留；用户隔离代码授权继续有效。完整结果见[C02报告](../reports/research/510300_point_c02_exit_prediction_v1/研究结论与下一步.md)及TECH.R78。

本次将当日成交额/此前20完整日金额中位数、abs当日经济对数收益/包含当日20日样本标准差、严格涨跌sign三列加入原八项，共11项；没有压成标量、加入交互或进入规则。原标签、全部成熟成员、周期等权、20周期/10周期100行、alpha1/clip5及实际入场锁定不变。旧初筛已有相对额等组成量；这只是相对原八项的固定线性信息块，不声称全项目新物理数据或完整吸收机制验证。

原1,507自然状态和115成熟月份三列全部支持。7必要测试后冻结70来源，25候选及25原对照拟合、90月复用，115原参数误差0。较早6周期262配对原点MSE增加1.3732%，近期18周期748原点增加0.3893%；两段年度块改进95%区间均跨零、固定门均FAIL，终态`REJECTED_FIXED_C02_INFORMATION_BLOCK_PREDICTION_GATE_FAILED`。3,488字段与1,010配对保存预测重建一致，497原未知保留；10次自然预测符号变化不是十笔交易。首次冻结缺来源元数据在拟合前中止并保留原件，补齐后科学协议未变，不计额外经验候选。经济阶段SKIPPED、新收益/夏普NOT_COMPUTED、新账户0；不改字段、窗口、编码、交互、模型、标签或时期营救本失败。

原[29项G1索引](../reports/research/510300_point_next_information_intake_20261002/G1_prior_definition_routing_20261002.json)已成为冻结来源，不回写；新状态由[C02独立对应补充](../reports/research/510300_point_next_information_intake_20261002/G1_C02_routing_addendum_20261002.json)及TECH.R78接续。下一[E09未接纳计划](../reports/research/510300_point_next_information_intake_20261002/E09_label_objective_intake_plan.json)只复核原变长参考退出标签、经济/费用/分红语义、成熟单位与已有短期目标的重合。不能把未来周期平均误差当实时截距，不能据此认定原标签错误或固定五日更好；未选择新时域、计算新标签或注册模型。原samples仅已完成自然周期，不能冒充完整替代训练集或把重叠日行当独立周期。E03不重置，真实新期间日数/周期仍0，收益/夏普及独立验证目标未完成。

## 技术点位最新接续：E09原目标与成熟时钟复核（2026-10-02 08:57后）

上节E09提案此后已完成一次隔离的只读诊断，未接纳不同的新目标；其他分支合同、原策略和旧失败保留。原标签是下一开盘出售相对持有至自然参考退出的含BASE摩擦/新增权益单位继续收益，分母是参考份额乘早卖原开盘。它不等于完整20万元库存调整、现金再部署和核心/辅助组合账户收益，此区别不证明原目标错误，也不覆盖原现金流增量拒绝。

53来源冻结后排除收益目标及净利润列，只核对元数据。1,507状态/35样本周期的剩余开盘区间1—59、中位25；原36参考周期全自然结束，唯一无继续状态周期2016-01-04至01-05只有一个区间。1,507原无交易请求状态完整对应、T+1年龄与日期索引一致；原142月/115可用成员与最近20完整周期/10周期100行制度一致，未来训练行及新增权益晚于成熟日冲突0。三表与旧合同抽取精确重建，53来源未变；未认证历史首版/部署或重算实际可卖性，见[E09完整报告](../reports/research/510300_point_e09_label_objective_review_v1/研究结论与下一步.md)及TECH.R79。

已确认旧44做过min(自然退出,早卖+5)，自然周期成熟仍保留，且停止相邻期限搜索；79方向、118逐年龄倒推、158概率幅度和95完整入场都有终态。通用5/20Target、20日方向/波动、五日V3及20/60尾部注册分别对应，不把代码或协议存在当作通过。旧242日指标只抽取，不与当前252日共同账户混排。原P04未绑定H且NOT_RUN保留，有限旧表达另存[对应补充](../reports/research/510300_point_next_information_intake_20261002/G1_E09_P04_routing_addendum_20261002.json)，原冻结G1索引不回写。

终态COMPLETED_ORIGINAL_LABEL_CLOCK_AND_PRIOR_REVIEW_NO_DISTINCT_NEW_TARGET_ADMITTED。原目标错误未建立，不据中位25选H、不利用未来共同误差；新标签/拟合/账户/行情0，收益夏普NOT_COMPUTED。下一[A01有方向路径效率计划](../reports/research/510300_point_next_information_intake_20261002/A01_candidate_intake_plan.json)先查旧表达、现金分红价格坐标、20差分/零分母时钟，再查完整原成员数量；实质相同或仅救回旧T01则停止。当前未绑定、接纳或拟合A01，原E03不重置，真实新期间日数/周期0，金融及独立验证目标未完成。

## 技术点位最新接续：A01固定条件增量失败及P03旧用途复核（2026-10-02 09:52后）

E09时的A01提案已由实际结果接续。发现旧日更三家族有同一有向财富ER20公式，旧共同政策失败保留；本次只原八项继续自然持仓收益的第九项条件增量，既有字段的新用途，不是新数据或旧策略营救。原1,507状态/115成熟月完整，8必要测试、73来源先冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6/262及近期18/748的MSE分别增加0.02023%/0.18294%，两段年度块改进区间跨0，两期FAIL，终态REJECTED_FIXED_A01_FIELD_PREDICTION_GATE_FAILED。经济SKIPPED，新账户0、收益夏普NOT_COMPUTED，金融目标未完成。

3,488字段、1,010配对及497未知、24周期误差和区间精确重建，73来源及C02/C04/A04/C06/E09冻结未变；2符号变化不是新增交易。不换窗口/坐标/模型/标签/时期营救。另有限复核P03：原持有年龄/回撤已有，距自身含分红高点时间也已做过原八项第九项退出并关闭；旧压力主夏普0.641136至0.528669，较早账户相同。只抽取旧20万元/242日结果，不与当前252日账户混排；完整P03入场前ATR和联合表达未证明已测，原卡NOT_RUN与G1快照保留。

下一B01仅旧五日均线偏离/标准化和均值反弹等实际用途、价格/波动坐标及原完整成员支持提案，未接纳、未注册、未拟合。原E03不重置、真实新账户日/周期仍0；宏观与盘口任务各自状态保持。见TECH.R80—81、[A01完整报告](../reports/research/510300_point_a01_exit_prediction_v1/研究结论与下一步.md)、[P03有限旧来源](../reports/research/510300_point_next_information_intake_20261002/G1_P03_peak_age_prior_addendum_20261002.json)、[B01未接纳计划](../reports/research/510300_point_next_information_intake_20261002/B01_candidate_intake_plan.json)及[技术点位现行状态](PROJECT_STATE_TECHNICAL_LINE.md)。

## 技术点位最新接续：B01固定两列点值改善但未晋升（2026-10-02）

上一节B01提案已由实际准入和结果接续。原sma5已存在，本次原八项同时加入五点含当日财富均线偏离及20个经济log日收益std乘sqrt5标准化两列，共10项；无新入场阈值或原标签/成员/拟合/锁定版本改变。1,507/115完整，9必要测试、82来源冻结、25候选/25对照拟合、90月复用、115原参数误差0。较早6/262与近期18/748的MSE降低0.09927%/0.56213%，两段年度块改进区间下界仍负，整体REJECTED_FIXED_B01_INFORMATION_BLOCK_PREDICTION_GATE_FAILED。经济SKIPPED、新账户0、收益夏普NOT_COMPUTED，目标未完成。

3,488字段/1,010配对及497未知、24周期误差和区间精确重建，82来源及A01/C02/C04/A04/C06/E09冻结未变；3符号变化不等于新增交易。固定两列有小幅点值改善但稳健证据不足，不能称整个均值回归机制失败或账户表现已提高。不改5/20、sqrt5、维度、方向、模型/标签/成员/时期营救，不事后挑一列，旧R2及反弹辅助/其他旧终态和原G1快照保留。

下一N04只核查原季末末3日与季内涨跌、机构代理交互的旧用途及源资格；不能省略机构重仓代理、用ETF自身涨幅替代或现今权重回填。当时可知交易日历、季度起点及机构代理发布时钟未绑定，尚未接纳/注册/拟合。E03真实新日线/账户日/周期仍0且不重置，其他研究分支状态保持。见TECH.R82、[B01报告](../reports/research/510300_point_b01_exit_prediction_v1/研究结论与下一步.md)、[N04未接纳计划](../reports/research/510300_point_next_information_intake_20261002/N04_prior_source_intake_plan.json)及[技术现行状态](PROJECT_STATE_TECHNICAL_LINE.md)。

## 技术点位最新接续：N04完整来源未准入（2026-10-02）

B01时的N04来源提案已实际复核。原卡末3交易日需与季内涨跌及机构重仓分别交互，旧末5自然日日历/季度申赎只是有限相关用途、旧失败保留。56期55旧声明时钟可复算、1,507原点旧季报声明覆盖完整，仍不提供机构股票持仓。旧日历1,478原点、2026日历114个2026原点，不能用最终回取交易日期认证过去先验末3日；当前指数前十仅2026-08-12一期、历史权重版本/as-of门未过，也不能代机构重仓。完整N04字段/原成员支持NOT_COMPUTED，终态NOT_ADMITTED_N04_COMPLETE_INFORMATION_BLOCK_SOURCE_GATE_FAILED，新拟合/标签/账户0，收益夏普NOT_COMPUTED，金融目标未完成。

原v1日期ns/ms比较实现失败保留，另v1_0_1只统一精度后严格日期数值比较，2回归测试/38来源冻结，56声明时钟和1,507覆盖元数据精确复算；原31及B01/A01/C02/C04/A04/C06/E09冻结未变。来源未准入不是整个季末机制收益失败，不以申赎/ETF收益代代理，不回填现今权重或删未知行；只在合格不同来源及完整时钟/成员建立后另立源版本。原PDF是否另有持仓表和全仓代理完备性均未由本次结构表证明。

下一A02只核对原突破回踩旧用途/T01拒绝、价格和ATR20[e−1]时钟及e后1—3完整日阶段支持；非阶段不填0或延长来凑数量，当前未绑定、接纳或拟合。原E03真实前瞻日/周期仍0且不重置，其他分支状态保持。见TECH.R83、[N04完整报告](../reports/research/510300_point_n04_source_intake_v1_0_1/研究结论与下一步.md)、[A02未接纳计划](../reports/research/510300_point_next_information_intake_20261002/A02_candidate_intake_plan.json)及[技术现行状态](PROJECT_STATE_TECHNICAL_LINE.md)。

## 技术点位最新接续：A02阶段支持不足与旧截距机制复核（2026-10-02）

A02固定突破后1—3完整日比值仅371/1,507原成员支持，115可用月完整0；5必要测试、28来源及3,488/1,507/142保存表一致，停止直接第九项用途，拟合/标签/账户0。原9轮冻结未变。另复核旧第115轮买入前上下文估计周期截距机制及失败，不作为新方法重试；原七项G1细复核均有有限接续，不是七张完整卡验证。下一C05只旧残差同用途/时钟/完整成员提案，未绑定/接纳/拟合。金融目标未完成。

A02在较早/近期支持180/658、156/748，34/81原成熟月均不完整，每月至少缺313/544成员；真实零94原点保留，非阶段/过期未知不填0或删除。仅拒绝这个原完整样本上的直接第九项，不否定整个突破回踩收益，原T01、旧十日回踩及原注册快照不改。旧第115轮入场上下文截距主夏普0.38910/0.34131、年化2.9931%/2.5741%，已劣于当时对照；旧版本每日更新不同于当前入场锁定，不冒充完全等价，也不据此重跑营救。

见TECH.R84—85、[A02报告](../reports/research/510300_point_a02_information_intake_v1/研究结论与下一步.md)、[旧截距有限复核](../reports/research/510300_point_next_information_intake_20261002/entry_context_intercept_prior_review_20261002.json)、[七项有限接续](../reports/research/510300_point_next_information_intake_20261002/G1_fine_prior_followups_20261002.json)和[C05未接纳计划](../reports/research/510300_point_next_information_intake_20261002/C05_candidate_intake_plan.json)。原E03真实新日/周期0且不重置，其他分支状态保持。

## 技术点位最新接续：C05固定成交额残差预测拒绝（2026-10-02）

C05成交额残差固定第九项已拒绝：3,488日线3,216有限，1,507原点/115成熟月全支持；7必要测试、字段74/模型85来源冻结。25候选/25对照收益拟合、90月复用、115原参数误差0；较早6周期262点/近期18周期748点MSE+1.1868%/+3.0510%，预设整年块改进区间均全负。1,507预测/24周期误差和区间精确复算，497未知保留、12符号变化非交易。经济SKIPPED、账户0、金融NOT_COMPUTED。首次字段OLS/保存复算各3,216单列，非独立策略；十轮旧冻结未变。下一D03只旧CLV用途/OHLC时钟/五日成员支持提案，未绑定/接纳/拟合；金融目标未完成。

原八项及标签、成熟成员、周期权重、原拟合/确认/入场锁定均保持。3216逐日OLS是技术字段计算，生产/保存复算分别记账，不冒充零次计算或独立策略。原注册NOT_RUN和G1快照保留，原旧高低政策及全部失败不营救；只拒绝这个固定条件用途，不推导逆向交易。独立验证NOT_ESTABLISHED，正式全搜索DSR/PBO未计算，整体过拟合消除未证明。

见TECH.R86、[C05完整报告](../reports/research/510300_point_c05_exit_prediction_v1/研究结论与下一步.md)、[C05预测结果](../reports/research/510300_point_c05_exit_prediction_v1/prediction_summary.json)和[D03未接纳计划](../reports/research/510300_point_next_information_intake_20261002/D03_candidate_intake_plan.json)。原E03真实新日/周期仍0、安排不重置，其他分支状态保持。

## 技术点位最新接续：D03五日收盘位置跨期预测拒绝（2026-10-02）

D03五日CLV均值/正天数固定两列已拒绝：3,488日线3,484有限，1,507原点/115成熟月全支持，27原未知保持；7必要测试、字段78/模型89来源冻结。25候选/25对照收益拟合、90月复用、115原参数误差0；较早6周期262点MSE−0.8374%、年度块改进下界正，PASS；近期18/748为+0.1991%、区间跨0，FAIL，整体拒绝。3,488/1,507/142支持及1,507预测/24周期误差/两期区间精确复算，497未知、3符号变化非交易。经济SKIPPED/账户0/金融NOT_COMPUTED；字段OLS0。下一C03只旧量价效率用途/十日分组分母及完整成员准入提案，未绑定/接纳/拟合。金融目标未完成。

见TECH.R87、[D03报告](../reports/research/510300_point_d03_exit_prediction_v1/研究结论与下一步.md)、[预测结果](../reports/research/510300_point_d03_exit_prediction_v1/prediction_summary.json)及[C03未接纳计划](../reports/research/510300_point_next_information_intake_20261002/C03_candidate_intake_plan.json)。既有CLV不同条件用途不等于新物理来源；不按年代切模型，不营救原高低政策，原E03不重置，其他研究线内容与状态保持。

## 技术点位最新接续：C03支持失败与D01旧用途裁决（2026-10-02）

C03固定十日量价效率差原成员支持失败：3,488日线3,468有限，1,499/1,507原点、39/115成熟月完整，27原未知保持；8个原状态均无负收益组金额分母。较早658/658原点但0/34月份完整，每月最少缺2旧状态；近期742/748、39/81。7必要测试、24来源冻结，3,488/1,507/142保存表精确复算；新增拟合/标签/账户0、金融NOT_COMPUTED。D01当前强弱第九项旧用途及失败复核完成，仅改60为20/尺度/版本不接纳，完整S20当前用途未验证。下一A03只旧创新高用途/事前高点与原成员提案，未绑定/接纳/拟合；金融目标未完成。

见TECH.R88—89、[C03完整报告](../reports/research/510300_point_c03_information_intake_v1/研究结论与下一步.md)、[D01有限旧用途](../reports/research/510300_point_next_information_intake_20261002/D01_session_strength_prior_review_20261002.json)及[A03未接纳计划](../reports/research/510300_point_next_information_intake_20261002/A03_candidate_intake_plan.json)。只拒绝固定直接增量/重述用途，不否定整个量价或日内隔夜机制。正式DSR/PBO与新收益未计算，原E03不重置，其他研究线内容和状态保持。

## 技术点位最新接续：A03新高频率与加速预测拒绝（2026-10-02）

A03创新高频率/加速固定两列已拒绝：3,488日线3,449有限，1,507原点/115成熟月完整、27原未知保持；7必要测试、字段76/模型87来源冻结。25候选/25对照拟合、90月复用、115原参数误差0；较早6周期262点MSE−5.8434%、年度块改善下界正，PASS；近期18/748 MSE−0.4384%但区间跨0，FAIL，整体预测拒绝。3,488/1,507/142支持与1,507预测/24周期误差/两期区间精确复算，497未知、30符号变化非交易。经济SKIPPED/账户0/金融NOT_COMPUTED、字段回归0。下一B05只旧单位金额下跌推进用途/原卡六负日与旧20日范围及时钟/原成员提案，未绑定/接纳/拟合；金融目标未完成。

见TECH.R90、[A03完整报告](../reports/research/510300_point_a03_exit_prediction_v1/研究结论与下一步.md)、[预测结果](../reports/research/510300_point_a03_exit_prediction_v1/prediction_summary.json)及[B05未接纳计划](../reports/research/510300_point_next_information_intake_20261002/B05_candidate_intake_plan.json)。近期点估计改善但预设稳定性门未过，未计算新账户收益，不按年代切模型、改变门或旧负事件范围营救。全搜索DSR/PBO未计算，原E03不重置，其他研究线内容/状态保持。

## 技术点位最新接续：B05六负事件金额冲击比预测拒绝（2026-10-02）

B05六负事件单位金额推进比固定第九项已拒绝：3,488日线3,454有限、前34日未知；1,507原点/115成熟月完整、27原未知保持。7必要测试、字段79/模型90来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE+0.0115%、年度块改善区间跨0，近期18/748 MSE+0.0917%、区间全负，两期FAIL；1,507预测/24周期误差及区间精确复算，497未知、0符号变化。经济SKIPPED/账户0/金融NOT_COMPUTED、字段回归0；旧20日分位政策不重试。下一A06仅两维旧用途/尺度时钟/重叠与原成员准入提案，未绑定/接纳/拟合；金融目标未完成。

见TECH.R91、[B05完整报告](../reports/research/510300_point_b05_exit_prediction_v1/研究结论与下一步.md)、[预测结果](../reports/research/510300_point_b05_exit_prediction_v1/prediction_summary.json)与[A06未准入计划](../reports/research/510300_point_next_information_intake_20261002/A06_candidate_intake_plan.json)。六真实负事件范围在数量/收益前单一固定，旧二十日分位政策和失败保持。两期误差增加，近期年度块改进区间全负，只拒绝固定条件增量；不重试或推导反向交易。全搜索DSR/PBO与新账户收益未计算，独立验证未建立，E03不重置；其他分支更新与运行状态保持。

## 技术点位最新接续：A06速度与六十日距离预测拒绝（2026-10-02）

A06速度/六十均价距离固定两列已拒绝：3,488日线3,429有限、前59日未知；1,507原点/115成熟月完整、27原未知保持。7必要测试、字段74/模型85来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE−0.8337%但年份块区间跨0，近期18/748 MSE+2.5568%、改进区间全负，两期FAIL；支持表及1,507预测/24周期误差/区间精确复算，497未知、35符号变化非交易。经济SKIPPED/账户0/金融NOT_COMPUTED、字段回归0；旧速度政策保持。下一P01仅下行占比旧用途/时钟/尺度与原成员准入提案，未绑定/接纳/拟合；金融目标未完成。

见TECH.R92、[A06完整报告](../reports/research/510300_point_a06_exit_prediction_v1/研究结论与下一步.md)、[固定预测结果](../reports/research/510300_point_a06_exit_prediction_v1/prediction_summary.json)及[P01未准入计划](../reports/research/510300_point_next_information_intake_20261002/P01_candidate_intake_plan.json)。速度分量已有研究用途，固定两维条件用途不等于新来源。较早点估计改善未过门，近期变差，不按年代/尺度/维度营救。独立验证与全搜索DSR/PBO未建立，新账户收益未计算，原E03不重置，其他分支内容和状态保持。

## 技术点位最新接续：P01全天风险份额预测改善证据不足（2026-10-02）

P01全天净收益下行平方占比/总log波动固定两列已拒绝：3,488日线3,468有限、前20日未知；1,507原点/115成熟月完整、27原未知保持。7必要测试、字段78/模型89来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE−0.2213%、近期18/748 −0.3109%，两期年份块改进区间均跨0、均FAIL。支持表及1,507预测/24周期误差/区间精确复算，497未知、4符号变化非交易。经济SKIPPED/账户0/金融NOT_COMPUTED、字段回归0；旧全天分位与隔夜条件用途保持。下一D02仅负缺口吸收旧用途/时钟/完整原成员准入提案，未绑定/接纳/拟合；金融目标未完成。

见TECH.R93、[P01完整报告](../reports/research/510300_point_p01_exit_prediction_v1/研究结论与下一步.md)、[固定预测](../reports/research/510300_point_p01_exit_prediction_v1/prediction_summary.json)及[D02未准入计划](../reports/research/510300_point_next_information_intake_20261002/D02_candidate_intake_plan.json)。旧隔夜份额已有同类继续价值用途，全天净收益坐标的固定信息不同但不构成新来源。两期点值改善却区间跨零，不能改门、挑时期或删列晋升。全搜索DSR/PBO与新账户收益未计算、独立验证未建立，E03不重置，其他研究线更新与运行状态保持。

## 技术点位最新接续：D02阶段外定义导致直接增量停止（2026-10-02）

D02负缺口吸收/缺口大小固定两列支持失败：3,488日线1,791可用，1,696非负隔夜阶段外、首区间未知1；1,507原状态762可用/745阶段外，115成熟月完整0，27原未知保持。较早324/658及0/34月，近期400/748及0/81月，每月至少缺241/380原成员；真零吸收904日/362原状态保留。7必要测试、26来源冻结、3份支持表精确复算；新增拟合/标签/账户0、金融NOT_COMPUTED。成分股PCA吸收率同名区别已纠正，固定直接用途结束，不否定整个缺口修复机制。下一A05仅事前多日量加权锚的旧用途/来源/完整成员核对提案，未绑定/准入/拟合；收益夏普与独立验证目标未完成。

见TECH.R94及[D02报告](../reports/research/510300_point_d02_information_intake_v1/研究结论与下一步.md)、[用途与现金时钟](../reports/research/510300_point_d02_information_intake_v1/prior_and_source_review.json)、[支持结果](../reports/research/510300_point_d02_information_intake_v1/summary.json)、[保存复算](../reports/research/510300_point_d02_information_intake_v1/saved_output_recomputation_receipt.json)、[A05未准入计划](../reports/research/510300_point_next_information_intake_20261002/A05_candidate_intake_plan.json)。旧rapid负缺口比率政策保持；严格原卡边界/整数表示不是营救旧政策的新信息。成分股协方差吸收率和可用证券训练与本缺口吸收区分，不移植旧结果。下一仅先核原父策略事前入场触发可否作为A05唯一事件锚，不从后验涨跌选锚、事件或样本。历史开发范围、全搜索DSR/PBO未计算及E03新日/周期0保持，其他研究线最新内容/运行状态不由本条改变。

## 技术点位最新接续：A05点值改善但稳定性门失败（2026-10-02）

A05事前D60事件锚定量价距离/首次收复固定两列已拒绝：3,488日线2,961完整、前527日无锚；1,507原状态/115成熟月完整，27原未知保持。7必要测试、字段78/模型89来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE−20.6590%、近期18/748 −1.7380%，两期年份块改进区间均跨0，均FAIL。三份支持表及1,507预测/24周期误差/区间精确复算，497未知、49符号变化非交易。经济SKIPPED、账户0、金融NOT_COMPUTED；旧日均价/滚动量加权/新低锚失败保持。下一B06仅原六日速度差及前日高点确认旧用途/源/完整成员准入提案，未绑定/准入/拟合；收益夏普与独立验证目标未完成。

见TECH.R95及[A05报告](../reports/research/510300_point_a05_exit_prediction_v1/研究结论与下一步.md)、[用途与事件时钟](../reports/research/510300_point_a05_exit_prediction_v1/prior_and_source_review.json)、[完整支持](../reports/research/510300_point_a05_exit_prediction_v1/intake/summary.json)、[固定预测](../reports/research/510300_point_a05_exit_prediction_v1/prediction_summary.json)、[精确复算](../reports/research/510300_point_a05_exit_prediction_v1/saved_output_recomputation_receipt.json)、[B06未准入计划](../reports/research/510300_point_next_information_intake_20261002/B06_candidate_intake_plan.json)。原当前父D60不是旧S1，锚只来自原事前价格条件首次成立，不用实际填单或未来自然退出筛选。量加权wealth值只是收盘份额代理。50锚/48首次收复及49符号变化不计交易。双期点值下降不能降低已冻结的稳定性门或替代收益夏普比较。下一B06先核旧T05/价格对照是否已覆盖当前用途，源/完整成员通过后才能登记，当前尚为提案。历史开发范围、原E03及其他共享研究线最新内容/状态保持。

## 技术线2026-10-02 B06固定条件增量接续（TECH.R96）

B06六日速度差/前高首次确认固定两列已拒绝：3,488日线3,481完整、前7联合区间不足未知；1,507原状态/115成熟月完整、27原未知保持。7必要测试、字段70/模型81来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE+0.02785%、近期18/748 +0.06098%，两期年份块改进区间均跨0、均FAIL。支持三表及1,507预测/24周期误差/区间精确复算，497未知、2符号变化非交易。经济SKIPPED、账户0、金融NOT_COMPUTED；旧T05/价格对照失败保持。下一C01仅原20日涨跌冲击中位数比的旧用途/源/正负各至少5及完整成员准入提案，未绑定/准入/拟合；收益夏普与独立验证目标未完成。

[B06报告](../reports/research/510300_point_b06_exit_prediction_v1/研究结论与下一步.md)、[用途与现金时钟](../reports/research/510300_point_b06_exit_prediction_v1/prior_and_source_review.json)、[完整支持](../reports/research/510300_point_b06_exit_prediction_v1/intake/summary.json)、[固定预测](../reports/research/510300_point_b06_exit_prediction_v1/prediction_summary.json)、[精确复算](../reports/research/510300_point_b06_exit_prediction_v1/saved_output_recomputation_receipt.json)、[路由补充](../reports/research/510300_point_next_information_intake_20261002/G1_B06_routing_addendum_20261002.json)、[C01未准入计划](../reports/research/510300_point_next_information_intake_20261002/C01_candidate_intake_plan.json)保存定义和结果。当前固定3+3经济速度差及前高首个确认不是新物理信息；完整支持不等于预测/金融门通过。下一C01须先核同用途和原20日双组各至少5的完整成员，当前只是提案。旧T05/A05失败、原E03及其他共享研究线内容与状态保持。

## 技术线2026-10-02 C01支持与D06旧用途接续（TECH.R97/98）

C01固定20日涨跌冲击中位数比的原成员支持失败：3,488日线3,441有限，1,489/1,507原状态支持、缺18均为负组少于5；115成熟月完整0，27原未知保持。较早653/658、0/34月，每月至少缺4原行；近期739/748、0/81月，至少缺8。7必要测试、24来源冻结，3,488/1,507/142三表精确复算；新增拟合/标签/账户0、金融NOT_COMPUTED（TECH.R97）。D06隔夜风险份额当前继续价值旧用途确认，仅改20为60/收益尺度/版本不接纳重试；旧107主压力夏普0.30437低于八项0.64114，完整原卡尾部联合当前用途尚未建立（TECH.R98）。旧24保存指标只抽取、不重算；下一P02仅事前EWMA方差惊喜/尾部旧用途、时钟和原成员准入提案，未绑定/准入/拟合；完整收益夏普目标未达。

[C01及D06报告](../reports/research/510300_point_c01_information_intake_v1/研究结论与下一步.md)、[C01用途与时钟](../reports/research/510300_point_c01_information_intake_v1/prior_and_source_review.json)、[支持结果](../reports/research/510300_point_c01_information_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_c01_information_intake_v1/saved_output_recomputation_receipt.json)、[C01路由](../reports/research/510300_point_next_information_intake_20261002/G1_C01_routing_addendum_20261002.json)、[D06有限旧用途](../reports/research/510300_point_next_information_intake_20261002/D06_overnight_share_prior_review_20261002.json)、[D06路由](../reports/research/510300_point_next_information_intake_20261002/G1_D06_prior_routing_addendum_20261002.json)、[P02未准入计划](../reports/research/510300_point_next_information_intake_20261002/P02_candidate_intake_plan.json)保存完整口径和数值。C01原成员的不足组不是可补0的卖压0；D06同风险份额目的已存在，完整尾部用途仍未知。P02须先核同用途和事前预测时钟，当前仅提案。原T05/总体流动性/隔夜份额固定失败、原E03以及其他共享研究线内容与状态保持。

## 技术线2026-10-02 P02事前方差固定增量接续（TECH.R99）

P02事前方差惊喜比/前序90分位超越固定两列已拒绝：3,488日线3,307完整、61种子不足/120历史比值不足；1,507原状态/115成熟月完整、27原未知保持。7必要测试、字段74/模型85来源冻结；25候选/25对照拟合、90月复用、115原参数误差0。较早6周期262点MSE−0.28991%、近期18/748 +0.39192%，两期年份块改进区间均跨0、均FAIL。支持三表及1,507预测/24周期误差/区间精确复算，497未知、5符号变化非交易；固定方差生产1种子/3,426递推、保存复算另相同，不估计衰减。经济SKIPPED、账户0、金融NOT_COMPUTED；旧T16覆盖失败保持。下一B04仅负2倍波动冲击后的rv3/原前序rv20及是否创新低旧用途、事件/低点时钟和完整成员准入提案，未绑定/准入/拟合；完整收益夏普目标未达。

[P02报告](../reports/research/510300_point_p02_exit_prediction_v1/研究结论与下一步.md)、[用途与事前时钟](../reports/research/510300_point_p02_exit_prediction_v1/prior_and_source_review.json)、[完整支持](../reports/research/510300_point_p02_exit_prediction_v1/intake/summary.json)、[固定预测](../reports/research/510300_point_p02_exit_prediction_v1/prediction_summary.json)、[精确复算](../reports/research/510300_point_p02_exit_prediction_v1/saved_output_recomputation_receipt.json)、[P02接续](../reports/research/510300_point_next_information_intake_20261002/G1_P02_routing_addendum_20261002.json)、[B04未准入计划](../reports/research/510300_point_next_information_intake_20261002/B04_candidate_intake_plan.json)保存定义和结果。预测方差及比值阈值严格排除当天信息，完整支持不等于稳定性/账户收益成立；尾部超越和5符号变化不计交易。B04尚为旧用途/源/数量提案，不能加阈值、选择修复成功事件或延长事件营救。原失败、E03及其他共享研究线内容/状态保持。


### 技术线接续 TECH.R100：B04固定两列成员支持

B04固定冲击后波动比/冻结低点两列成员支持失败：3,488日线769完整、78接受冲击/31较近冲击去重；1,507原状态285可用/1,222未知，115成熟月完整0、27原未知保持。较早118/658、0/34月，每月至少缺338原行；近期147/748、0/81月，至少缺621。7必要测试、25来源冻结，3,488/1,507/142三表精确复算；新增拟合/标签/账户0、金融NOT_COMPUTED。旧T03/T14入场/确认与当前逐日继续价值用途区分，固定直接用途结束，不否定所有冲击修复机制。下一B03仅破低收复用时/再跌破旧用途、冻结低点现金时钟和完整成员提案，未绑定/准入/拟合；完整收益夏普目标未达。

直接证据：[B04报告](../reports/research/510300_point_b04_information_intake_v1/研究结论与下一步.md)、[事前用途与事件时钟](../reports/research/510300_point_b04_information_intake_v1/prior_and_source_review.json)、[成员支持](../reports/research/510300_point_b04_information_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_b04_information_intake_v1/saved_output_recomputation_receipt.json)、[B04路由](../reports/research/510300_point_next_information_intake_20261002/G1_B04_routing_addendum_20261002.json)、[B03未准入计划](../reports/research/510300_point_next_information_intake_20261002/B03_candidate_intake_plan.json)。


### 技术线接续 TECH.R101：B03收复与再破位成员支持

B03固定前20日低点首次收复用时/再失守计数两列支持失败：3,488日线3,212完整、256尚未收复/20初始低点不足；1,507原状态1,430支持、77均尚未首次收复，115成熟月完整0、27原未知保持。较早637/658、0/34月，每月至少缺13原行；近期694/748、0/81月，至少缺23。7必要测试、25来源冻结，3,488/1,507/142三表精确复算；新增拟合/标签/账户0、金融NOT_COMPUTED。已更正旧提案归属：二测收复已完成失败为T02，T13供给策略完整NOT_RUN来源门保持。固定直接用途结束，不删除未修复过程营救。下一B02仅两日确认低点/十日二测量价压力对比旧用途、现金与窗口时钟和原成员提案，未绑定/准入/拟合；完整收益夏普目标未达。

直接证据：[B03报告](../reports/research/510300_point_b03_information_intake_v1/研究结论与下一步.md)、[事前定义及归属纠正](../reports/research/510300_point_b03_information_intake_v1/prior_and_source_review.json)、[成员支持](../reports/research/510300_point_b03_information_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_b03_information_intake_v1/saved_output_recomputation_receipt.json)、[B03接续路由](../reports/research/510300_point_next_information_intake_20261002/G1_B03_routing_addendum_20261002.json)、[B02未准入计划](../reports/research/510300_point_next_information_intake_20261002/B02_candidate_intake_plan.json)。


### 技术线接续 TECH.R102：B02二测五项量价原成员支持

B02二次试低五项原量价记录支持失败：3,488日线356破低/138两日确认/43合法二测，43记录源完整；1,507原状态仅15二测日可用、1,492非二测未知，115成熟月完整0、27原未知保持。较早7/658、0/34月，每月至少缺434原行；近期8/748、0/81月，至少缺752。7必要测试、22来源冻结，3,488/1,507/142三表精确复算；新增配置/拟合/标签/账户0、金融NOT_COMPUTED。旧T02五项只记录/未额外筛选、固定失败保持；T13完整NOT_RUN来源门不改。当前直接用途结束，不携带事件/填零/删成员营救。下一P05/P06仅旧T18用途和保存三家族预测/成熟校准/失效时钟及全部原成员提案，未绑定/准入/拟合；完整收益夏普目标未达。

直接证据：[B02报告](../reports/research/510300_point_b02_information_intake_v1/研究结论与下一步.md)、[事前五项与时钟](../reports/research/510300_point_b02_information_intake_v1/prior_and_source_review.json)、[原成员支持](../reports/research/510300_point_b02_information_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_b02_information_intake_v1/saved_output_recomputation_receipt.json)、[B02接续](../reports/research/510300_point_next_information_intake_20261002/G1_B02_routing_addendum_20261002.json)、[P05/P06保存预测未准入计划](../reports/research/510300_point_next_information_intake_20261002/P05_P06_saved_forecast_intake_plan.json)。


### 技术线接续 TECH.R103：P05/P06保存预测来源未准入

P05/P06既有主一日保存四项来源未准入：旧3,307日截至2025-12-31，1,911外层/85,326内层保存范围与成熟上限核对通过，非重拟合/完整公式验证；当前3,488日1,911来源完整、1,396原无模型/181无保存日期。1,507原状态756来源可用、637原无模型/114无日期，115成熟月完整0、27原未知保持。较早131/658、0/34月至少缺438原行；近期625/748、0/81月至少缺39。历史首次发布NOT_ESTABLISHED，不由成熟或shift(1)替代。7必要测试、28来源冻结，3,488/1,507/142三表及保存上限精确复算，元数据复核生产/复算各1,911/85,326；新配置/拟合/标签/账户0、金融NOT_COMPUTED。旧T18日更112账户失败/季度NOT_RUN及旧持仓收益CUSUM无增量保持，不补历史/2026营救。下一G2-J01只中国国债收益率及股价共同状态的旧LPR联合用途/本地官方源与发布时间/完整成员提案，未绑定/准入/采集/拟合；完整收益夏普目标未达。

直接证据：[P05/P06报告](../reports/research/510300_point_p05_p06_saved_forecast_intake_v1/研究结论与下一步.md)、[事前来源口径](../reports/research/510300_point_p05_p06_saved_forecast_intake_v1/prior_and_source_review.json)、[来源支持](../reports/research/510300_point_p05_p06_saved_forecast_intake_v1/summary.json)、[保存复算](../reports/research/510300_point_p05_p06_saved_forecast_intake_v1/saved_output_recomputation_receipt.json)、[P05/P06接续](../reports/research/510300_point_next_information_intake_20261002/G1_P05_P06_saved_source_routing_addendum_20261002.json)、[J01未准入计划](../reports/research/510300_point_next_information_intake_20261002/J01_local_yield_source_intake_plan.json)。


### 技术线接续 TECH.R104：J01本地国债源当前用途未准入

J01固定10年国债源准入未过：来源3,660日/2012-01-04至2026-08-25；当前3,488日3,443算法字段完整、24精确源日期缺失/21初始窗口未知。1,507原状态1,486完整/21缺源（2026-08-27至09-24），115成熟月全部成员算法支持115、27原无模型保持；较早658/658及34/34，近期727/748及81/81。原macro/衍生账簿2,509/2,501公共日10年值精确相同，非发布证明。原官网17:30晚于15:05，只用T−1/T−21的保守日终时钟；历史首版未建立，双来源门未过。7必要测试、30源冻结，3,488/1,507/142三表精确复算；新配置/拟合/标签/账户/采集0、金融NOT_COMPUTED。旧LPR联合收益/风险固定失败、账户NOT_RUN保持，不补源或缩样本营救。下一J02只旧CNH/美元/利差残差用途、保存裁决及本地市场/版本/时钟/原成员提案，未绑定或拟合；完整收益夏普目标未达。

直接证据：[J01报告](../reports/research/510300_point_j01_local_yield_intake_v1/研究结论与下一步.md)、[事前来源及字段](../reports/research/510300_point_j01_local_yield_intake_v1/prior_and_source_review.json)、[来源支持](../reports/research/510300_point_j01_local_yield_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_j01_local_yield_intake_v1/saved_output_recomputation_receipt.json)、[J01接续](../reports/research/510300_point_next_information_intake_20261002/G2_J01_source_routing_addendum_20261002.json)、[J02未准入计划](../reports/research/510300_point_next_information_intake_20261002/J02_rmb_residual_source_intake_plan.json)。


### 技术线接续 TECH.R105：J02保存残差来源门与旧数值通过澄清

J02既有两年保存残差源未准入：旧1,699日/2019-01-02至2025-12-31、1,953共同源，1,396已知保存模型及训练名单范围/15:05算法输入代理核对通过，不重OLS/残差公式或证明16:00旧计算已在15:05发布。当前3,488日1,396来源完整；1,507原状态614完整、792非旧日期/88原训练不足/13陈旧，115成熟月完整0、27原无模型保持。较早0/658、0/34每月最少缺438；近期614/748、0/81最少缺176。旧140有源事件四数值门通过，毛均值0.44639%、0.28%费用代理后0.16839%；旧来源291/359未过、完整账户NOT_RUN，非收益夏普失败。描述性优势差区间跨0且大事件集中，未删主样本。历史首版/同步行情未建立，当前源门未过。7必要测试、34来源冻结、三表3,488/1,507/142精确复算，元数据生产/复算各1,396；新配置/拟合/标签/账户/采集0、金融NOT_COMPUTED。下一J03—J06四项旧用途与本地源有限路由提案，未绑定/准入/拟合，不改旧失败；完整收益夏普目标未达。

直接证据：[J02报告](../reports/research/510300_point_j02_saved_residual_intake_v1/研究结论与下一步.md)、[旧数值与来源区别](../reports/research/510300_point_j02_saved_residual_intake_v1/prior_and_source_review.json)、[来源支持](../reports/research/510300_point_j02_saved_residual_intake_v1/summary.json)、[精确复算](../reports/research/510300_point_j02_saved_residual_intake_v1/saved_output_recomputation_receipt.json)、[J02接续](../reports/research/510300_point_next_information_intake_20261002/G2_J02_saved_source_routing_addendum_20261002.json)、[J03—J06未准入路由](../reports/research/510300_point_next_information_intake_20261002/J03_J06_cross_market_prior_source_routing_plan.json)。

### 技术线接续 TECH.R106：四项跨市场用途边界及下一资金单位诊断

完成J03—J06四项有限旧用途与本地来源路由：J03旧完整首日20万压力主年化−0.173894%、夏普−0.306678且仅1闭周期，原开盘描述非当前验证；J04聚合指数629配对MSE扩大2.612%，压力M1年化/夏普点值高于M0但暴露增加、区间跨0，原历史匹配公司面板未建立；J05月度原版102/108、26主事件毛均值1.278899%，后段数值门失败，账户NOT_RUN，原日频20日卡未绑定；J06三候选32旧账户均未接受，绝对/联合49信号及四账本相同。28所读文件身份和四项/六旧主账户引用一次精确核对，旧账户242口径不排当前252。当前源/字段/模型未准入，原1507/115成员支持NOT_COMPUTED；新配置/拟合/标签/账户/采集0。下一E10有限旧仓位/资金单位/保存方差资格提案，未绑定公式或风险系数；完整收益夏普目标未达。

直接证据：[四项报告](../reports/research/510300_point_j03_j06_prior_source_routing_v1/研究结论与下一步.md)、[完整分项事实](../reports/research/510300_point_j03_j06_prior_source_routing_v1/cases.json)、[旧主账户引用](../reports/research/510300_point_j03_j06_prior_source_routing_v1/saved_old_primary_account_quotes.json)、[保存核对](../reports/research/510300_point_j03_j06_prior_source_routing_v1/saved_output_verification_receipt.json)、[四项路由接续](../reports/research/510300_point_next_information_intake_20261002/G2_J03_J06_prior_source_routing_addendum_20261002.json)、[E10未准入提案](../reports/research/510300_point_next_information_intake_20261002/E10_allocation_units_and_uncertainty_prior_review_plan.json)。

### 技术线接续 TECH.R107：原继续优势和保存方差的资金用途门

E10有限资金单位/旧用途/保存方差核对结束：原Y分母是参考份额×下一开盘原价，区别于cycle_return入场支出和真实累计买入支出，未证明原标签错。142月与原时点/周期数/成熟上限元数据一致，115月方差定义/恒等/缓存通过、27原无模型保持、25旧训练身份；只元数据不重标签/REML/预测。E08残差协方差sigma_e²*n_j*I+sigma_b²*11不是下一日标的方差，未知新周期完整n_j不可借用，没有系数/均值估计协方差或同区间资金映射。旧adaptive效用、A原强弱/二元、B替代、E01/E04/E09实际目的/裁决保留，拒绝现存Y/方差直接次日配置；不否定所有配置/不确定性。25来源、142元数据/六单位/七旧方法摘录保存核对一次通过；新配置/拟合/标签/预测/账户/采集0，当前成员支持及收益夏普NOT_COMPUTED。下一F01—F03融资活动真实旧用途/来源有限路由提案，未绑定/准入/OLS或账户；E03不重置，完整目标未达。

直接证据：[E10报告](../reports/research/510300_point_e10_allocation_units_review_v1/研究结论与下一步.md)、[六类资金单位](../reports/research/510300_point_e10_allocation_units_review_v1/units_and_role_table.json)、[142月元数据](../reports/research/510300_point_e10_allocation_units_review_v1/monthly_variance_role_metadata.json)、[保存核对](../reports/research/510300_point_e10_allocation_units_review_v1/saved_output_verification_receipt.json)、[E10接续](../reports/research/510300_point_next_information_intake_20261002/E10_allocation_units_review_addendum_20261002.json)、[F01—F03未准入提案](../reports/research/510300_point_next_information_intake_20261002/F01_F03_financing_prior_source_routing_plan.json)。

### 技术线接续 TECH.R108：融资旧用途与直接报告源差异

三项当前不准入。F01余额五日变化代理已在日更三家族及拥挤FULL中使用，两旧组合主目标失败不等于F01独立无效。F02旧固定T03主期零交易、CAGR0、夏普null，不放宽事件/融资确认条件。F03旧两种做多初筛CAGR−0.523029%/−0.827645%、净夏普−0.234242/−0.331546；实际源码限定先前两年内最近最多252且最少126个合格日，原卡严格252完整验证未建立，不改最少样本或方向重试。

两保存沪深表均2674日/2015—2025、无直报偿还；官方时钟/滞后不证明逐日首版，完整当前来源未准入。原11深市补缺在收益评估前、2663已有行和参数未改；不是本轮采集或事后营救。原官方沪市2026年1月20行精确核19相邻来源记录，1月22日隐含偿还−直报为−168023619元，与旧必要核对一致，原因NOT_IDENTIFIED；首行缺前源保持未知，不以沪市单月填沪深全段。六旧原主账户行摘录，不重算64/56/112/88原账户或与当前252口径排名。

33来源、三路、六旧行、两表和20源行一次保存核对通过。当前1507原状态/115成熟月支持NOT_COMPUTED，新字段/配置/OLS/模型/标签/预测/账户/采集0，新净CAGR/夏普NOT_COMPUTED。下一F04—F06及G01—G04七项合并有限旧用途/源合同提案，未绑定或准入，缺资格即停。原E03、旧金融终态、其他线权限保持，真实新日/周期0、独立验证NOT_ESTABLISHED，完整目标未达。

直接证据：[研究报告](../reports/research/510300_point_f01_f03_prior_source_routing_v1/研究结论与下一步.md)、[三项假设与路由](../reports/research/510300_point_f01_f03_prior_source_routing_v1/cases.json)、[来源事实](../reports/research/510300_point_f01_f03_prior_source_routing_v1/source_facts.json)、[旧六主账户](../reports/research/510300_point_f01_f03_prior_source_routing_v1/saved_old_primary_account_quotes.json)、[F03实际源码](../reports/research/510300_point_f01_f03_prior_source_routing_v1/implementation_evidence.json)、[官方差额表](../reports/research/510300_point_f01_f03_prior_source_routing_v1/repayment_identity_rows.csv)、[保存核对](../reports/research/510300_point_f01_f03_prior_source_routing_v1/saved_output_verification_receipt.json)、[下一七项未准入提案](../reports/research/510300_point_next_information_intake_20261002/F04_F06_G01_G04_prior_source_routing_plan.json)。

### 技术线接续 TECH.R109：七项融资/ETF份额旧源门

七项当前未准入。F04限于指定来源尚未建立历史有效融资股票集合与明细合同，不声称全项目/外部没有任何面板。F05旧T06拥挤FULL失败，匹配自由流通分母未准入；只引用2026-09-28旧过期/0行凭证，不声称本轮验证当前接口。F06旧T05隐含偿还失败，原卡直报/E02配对未完成，R108差额原因未知保持。

G01/G02相关单基金份额/收盘NAV折溢价八规则原终态拒绝；50万元、现金1.5%、最低佣金0、242年化及H00300超额双20门不等于当前20万元口径。旧份额公布时点局部补充已关闭，本轮仅读其结论，未重扫2220响应或累计成新进展，原五个额外非交易日早已排除不创造新问题。G03/G04仍缺当时动态同指数体系、日频拆分/份额/前日NAV及历史时钟，固定池/今日基金池/PCF开市前规则不代替实际规模查询首版。

旧三沪市产品周度迁移HIGH/LOW均优于匹配价格对照；20万压力CAGR1.191156%/0.312628%、夏普0.370320/0.139443、入场14/16。原继续门至少20入场、CAGR5%、夏普0.8和两段正未过，不能写所有结果为负，也不能选HIGH后段1.0688恢复候选。完整日频NAV原卡仍未验证。七路、五旧主行及八旧规则摘要保存核对一次通过，共26来源。当前1507/115支持NOT_COMPUTED，新字段/配置/拟合/标签/预测/账户/采集0、新净收益夏普NOT_COMPUTED。

本轮合计已完成十项融资/份额有限路由，不是十个新回测。下一96因子卡E01—E06六项成分参与面合并有限旧用途/源合同提案（区别于TECH.E01目标传递方法），未绑定/准入。原E03、旧失败、其他分支保持；独立验证及完整目标未达。

直接证据：[七项报告](../reports/research/510300_point_financing_etf_prior_routes_v1/研究结论与下一步.md)、[分项假设/理由](../reports/research/510300_point_financing_etf_prior_routes_v1/cases.json)、[旧源门及八规则](../reports/research/510300_point_financing_etf_prior_routes_v1/source_and_old_purpose_facts.json)、[五旧主行](../reports/research/510300_point_financing_etf_prior_routes_v1/saved_old_primary_account_quotes.json)、[保存核对](../reports/research/510300_point_financing_etf_prior_routes_v1/saved_output_verification_receipt.json)、[下一六项提案](../reports/research/510300_point_next_information_intake_20261002/E01_E06_breadth_prior_source_routing_plan.json)。

### 技术线接续 TECH.R110：成分参与面四源完整支持上界失败

六项当前不准入。E01原多数上涨与MA20主方案20万压力CAGR−2.627534%、夏普−0.237576，高于删广度价格对照−5.290186%/−0.409612，仍未达标。E02旧T02同成员新低收缩CAGR−0.690841%、夏普−0.616675、11周期，同源增量95%区间跨0；原统计18:00可用假设不能证明本线15:05首版。E03四个固定初筛已做，MA20变化HIGH虽高于两对照但整体CAGR0.022214%、夏普0.021824、后段负，其他三个负、皆无继续资格，收盘极值不等于日内高低。E05旧同成员中位数变化联合削减FULL失败、三个增量区间跨0，ETF减中位差仅描述；不据多组件失败判单字段永远无效。

四原保存缓存只用自身原有效标记核日期覆盖上界：原广度1051/1507、旧参与速度755/1507、旧中位字段775/1507、官方权重广度521/1507；115原可用月完整支持四源均0，最少缺原训练行10/134/122/284。E01/权重同统计日是最乐观日期上界而非15:05信号，E03/E05保留旧lag1且统计日严格前序，不计算新指标或证明实际首版。142月仅读原周期名单/行数/成熟上限，115有模型/27原未知不改，没有读取样本target或重标签/拟合。

E04原T04 PIT权重源门保持，两个指定raw schema没有行业历史/首次发布时间；月权重36000行120统计月2016-08至2026-07、价格484200行1614日2019-12至2026-08，不代当前完整历史合同。旧股票权重HHI、IF合约OI HHI与E06行业正贡献HHI是不同量，不能换名或用等权、今日行业、未来核心组救源门。旧MACD×权重广度和完整广度稀有上涨冻结失败原文保留，不重跑它们的回报账户；旧原242与当前252不排名。

36原来源、六路、11旧主行和3488/1507/568行三表保存核对通过。首次核对因Windows CSV换行字符串差异失败，原代码/表/失败均保留；独立核对器只统一LF后成功，原表数值与规则未变。原月元数据生产568，失败核对568/成功568，不算拟合或独立试验。新字段/配置/OLS/回报拟合/标签/预测/账户/网络采集0、新CAGR/夏普NOT_COMPUTED。下一H01—H05及L01—L06十一项合并有限旧用途/源合同提案未绑定/准入；IF仅信息观察，H06分钟及I期权链不纳本次，仍仅510300点位。原E03及其他分支保持、独立验证未建立，完整目标未达。

直接证据：[六项报告](../reports/research/510300_point_breadth_prior_source_routes_v1/研究结论与下一步.md)、[分项假设/理由](../reports/research/510300_point_breadth_prior_source_routes_v1/cases.json)、[四保存源上界](../reports/research/510300_point_breadth_prior_source_routes_v1/source_upper_bound_summaries.json)、[原月度成员表](../reports/research/510300_point_breadth_prior_source_routes_v1/原月度成员源覆盖上界.csv)、[来源与旧冻结用途](../reports/research/510300_point_breadth_prior_source_routes_v1/source_facts.json)、[保存核对](../reports/research/510300_point_breadth_prior_source_routes_v1/saved_output_verification_receipt.json)、[换行修正](../reports/research/510300_point_breadth_prior_source_routes_v1/verification_repair_v1_0_1.json)、[下一十一项提案](../reports/research/510300_point_next_information_intake_20261002/H01_H05_L01_L06_prior_source_routing_plan.json)。


## 技术线补充：TECH.R111 期货日线与盈利信息十一项核对（2026-10-02）

H01—H05及L01—L06十一项合并核对结束，当前新字段不准入。IF单日OI配对1599成熟预测MSE恶化0.065659%、区间跨零，账户未运行；五日四象限完整卡未验证。旧八条名义基差/曲线已拒绝，扣成本H01/H02/H04分红指数点数、历史权重及首版源门保留。HHI高低固定初筛夏普−0.532676/0.123218，两者无继续资格。盈利现金质量须读取后续修复及最终T11日期代理120账户，主20万元STRESS年化−0.270499%、夏普−0.351761，严格原T11仍NOT_RUN。正常化EY/利差预注册242日预测都拒绝；L06相关分红日历毛筛PASS后16账户已拒绝，主日历年化−0.556401%、夏普−0.291530，不以毛筛代表最终结论。58冻结来源、11假设/6旧主行/7模式保存核对通过，无新配置/字段/标签/拟合/预测/账户/采集，不计算本线1507/115源覆盖。新净收益夏普NOT_COMPUTED。下一P03/N04/P04本地路径/日历/等待目标有限定义核对未准入；原E03与原策略保持，独立验证、整体去过拟合和完整目标未达。

精确定义的失败、部分代理已测、严格原卡NOT_RUN分别保留。L01两份显式叙述复用旧数值，匹配时点/别名差仍未知；EPS增长定义分歧配对MSE恶化5.973152%、没有账户。L02/L04最初七错字段的无效测量不作为合格新源，后续独立修复和最终T11日期代理失败同时记录。L03旧YTD盈利广度STRESS年化0.296111%、夏普0.087205目标失败，但不等于已披露同成员盈利减价格覆盖差的完整检验。L05旧242日用途与原60日卡不同，不能切到另一期限救援。L06日历用途没有分红金额/市值/现金流覆盖的完整合同。

新 source 支持值未计算，七份保存模式只核存在字段及行数；旧结果仍各自242或日历口径，不排序为本线252改善。观察IF不交易期货，金融行业不混入CFO，基金分红不代指数成分分红。下一先核P03路径时间与旧持仓交互重复、N04日历及机构代理资格、P04与当前等待目标重复，成立后才注册一个固定增量比较。

直接证据：[十一项研究报告](../reports/research/510300_point_if_earnings_prior_routes_v1/研究结论与下一步.md)、[分项假设与理由](../reports/research/510300_point_if_earnings_prior_routes_v1/cases.json)、[六主账户原行](../reports/research/510300_point_if_earnings_prior_routes_v1/saved_old_primary_account_quotes.json)、[来源与最终裁决](../reports/research/510300_point_if_earnings_prior_routes_v1/source_and_old_purpose_facts.json)、[保存核对](../reports/research/510300_point_if_earnings_prior_routes_v1/saved_output_verification_receipt.json)、[下一有限定义提案](../reports/research/510300_point_next_information_intake_20261002/G1_P03_N04_P04_prior_definition_plan.json)。


## 技术线补充：TECH.R112 / TECH.R113 入场前ATR回撤实际预测检验（2026-10-02）

TECH.R112完成P03/N04/P04有限查重：旧高点时间九项普通退出已失败（压力夏普0.528669、回撤12.448880%），原N04机构代理/日历首版合同未齐，P04等待目标已由原标签及失败H1—4迁移覆盖；不重跑这些用途。唯一P03入场前ATR尺度回撤组件沿旧raw Wilder ATR14固定，支持原3488日/1507状态/115完整训练月、27原无模型保持。TECH.R113原八项加该一列、25候选拟合及25控制参数重估，115可用月90复用；早期262状态/6周期MSE增加0.231667%，近期748/18增加0.013812%，两改进95%区间跨0、双期拒绝。3预测符号变化非交易、497原未知保持，经济阶段SKIPPED、新账户/标签/采集0、净收益夏普NOT_COMPUTED。7必要反例通过，1010配对预测、25保存模型方程/115原成员/24周期误差核对，最大方程误差8.03e-16；启动导入失败保留，按模块运行，源码不改。下一不同公告/机械供需八项有限源提案未准入，原E03与原策略保持，独立验证、整体去过拟合及完整目标未达。

原先提出高点时间方向后，实际定位发现已有冻结失败，已纠正为已测用途。旧ATR14用于保护/分层距离，本次唯一不同用途为入场固定尺度对当前回撤的条件继续价值：f_t=cycle_drawdown_t/(ATR14_{entry-1}/raw_open_entry)。按原未复权OHLC数学定义计算，除息机械跳空局限保留；实际入场开盘在入场首收盘已发生，ATR来自此前一收盘，不包含未来峰值或标签。原卡持仓时间与高点时间不重拟合，本次不是完整P03认证。

字段完整资格通过后才冻结唯一九项预测模型，原目标/全部成熟成员/最近20周期及10周期100行/周期等权/alpha1/clip5/固定周期截距/入场锁定版本都保持。两期按原收益MSE，入场年份块5000次/seed51030099只是开发敏感性；24自然周期中的1010状态不当独立交易。早期改进区间[-0.000057464,0.000075013]，近期[-0.000033137,0.000031144]，均跨0；不因某点位符号变化而提前运行账户。

字段16冻结源、预测26冻结源；14旧用途引用来源另记、6旧主行只摘录。生产字段元数据142、核对142，25候选及25控制拟合，115保存原成员核对；后续核对没有新拟合，区间复算不算新试验。第一次直接脚本因research导入路径失败，在任何冻结/字段前停止；python -m正确运行，原科学代码与测试不改。失败、原冻结分支及未计算状态保留。

下一M01/M02/M03/M04/M06/N03/N06/O05八项不同公告及机械供需源有限核对，M05旧日历已在R111关闭，不重扫或再次计新进展；只有不同用途与原完整当前样本资格成立才新比较。

直接证据：[固定实验报告](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/研究结论与下一步.md)、[旧用途与公式](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/prior_and_source_review.json)、[完整源成员](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/intake/summary.json)、[双期预测结果](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/prediction_summary.json)、[拟合记账](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/trial_accounting.json)、[保存模型与损失核对](../reports/research/510300_point_p03_entry_atr_exit_prediction_v1/saved_model_and_loss_check.json)、[下一八项提案](../reports/research/510300_point_next_information_intake_20261002/M01_M04_M06_N03_N06_O05_prior_source_plan.json)。

## TECH.R114：八项公告与机械供需有限来源裁决（2026-10-02）

八项公告/机械供需有限源核对完成、全部当前完整卡未准入（45证据对象、三原主压力账户摘录、三表模式）；M01严格非空0，9方案背景不填同文档21未知；M02完整用途/终止链未齐；M03普通股东公告毛线索32次/1.900355%/59.375%，账户仍NOT_RUN、非严格内部人；M04部分延期版本及旧解禁增量失败分开；M06固定36个A股配股不等于完整供给卡；N03旧个股21期425调入事件D4/D5失败停在收益前，非ETF需求；N06只有扩围历史解释、非全成员合同；O05误指PMI定位器已纠正。新增字段/预测/拟合/标签/账户/采集0，本轮收益夏普NOT_COMPUTED。上一实际模型TECH.R113入场ATR回撤双期预测失败保持；原E03、冻结策略和其他分支权限保持，独立验证、整体去过拟合及完整目标未达。下一不同持有人/休市外部信息/新闻预期七项仅有限提案，未准入或拟合。

[完整报告](../reports/research/510300_point_disclosure_mechanical_prior_routes_v1/研究结论与下一步.md)、[八项逐项裁决](../reports/research/510300_point_disclosure_mechanical_prior_routes_v1/cases.json)、[原源与旧用途](../reports/research/510300_point_disclosure_mechanical_prior_routes_v1/source_and_old_purpose_facts.json)、[三原压力账户](../reports/research/510300_point_disclosure_mechanical_prior_routes_v1/saved_old_primary_account_quotes.json)、[保存核对](../reports/research/510300_point_disclosure_mechanical_prior_routes_v1/saved_output_verification_receipt.json)、[下一有限七项提案](../reports/research/510300_point_next_information_intake_20261002/N01_N02_N05_O01_O03_O04_O06_prior_source_plan.json)。

原主压力账户摘录：回购需求年化−0.487764%/夏普−0.231984；固定到期−0.465444%/−0.279268，均是原来源口径而非当前252日模型结果。股东毛筛与其未运行账户不得混称净优势。

## TECH.R115：七项持有人、休市外部信息及文本预期差有限来源裁决（2026-10-02）

七项持有人/休市外部/文本预期差有限旧用途与原源核对完成、完整卡均未准入（34证据对象、两原主压力账户摘录、六表模式/五日期范围）。N01现有56季报/55可用时钟是510300被动ETF，非主动基金池；旧主压力年化−1.624626%/夏普−0.020845。N02每日批次/NAV/赎回分配未绑定，2220首发响应关闭项不重扫。N05旧跨中国休市的美时段区间累积算法已经存在、五海外普通模型旧主压力0.841700%/0.130847；四市场源止于8/18—19、VIX8/25，原状态至9/24，晚于末观测日期的原状态各26/26/26/25/21，重叠不相加且非正式1507/115支持。O01/03六事后机制病例非连续文本母集；旧成员级PIT链七门仅一通过，非所有新闻永久无效。O04旧33对EPS增长定义分歧误差增5.973152%、账户NOT_RUN，非全观点文本；O06原卡仅前瞻，首传闻及失败传闻全集缺。本轮新增字段/预测/拟合/标签/账户/采集0，金融NOT_COMPUTED；上一实际模型R113失败、原E03及其他分支权限保持。下一不同宏观K01—K06仅有限提案、优先K04原文及源范围资格，未准入/拟合；收益夏普、独立验证及整体去过拟合未完成。

[完整报告](../reports/research/510300_point_holder_external_news_prior_routes_v1/研究结论与下一步.md)、[七项逐项裁决](../reports/research/510300_point_holder_external_news_prior_routes_v1/cases.json)、[原源与范围事实](../reports/research/510300_point_holder_external_news_prior_routes_v1/source_and_old_purpose_facts.json)、[两原压力账户](../reports/research/510300_point_holder_external_news_prior_routes_v1/saved_old_primary_account_quotes.json)、[保存核对](../reports/research/510300_point_holder_external_news_prior_routes_v1/saved_output_verification_receipt.json)、[六宏观未准入提案](../reports/research/510300_point_next_information_intake_20261002/K01_K06_macro_prior_source_plan.json)。

本轮只有来源/范围证据；没有新收益或夏普的改善证据。不能将后段缺源补为零，也不将旧成员级精确钟门普遍套用到所有历史研发。


## TECH.R116—R117：订单库存可选残差修正实际检验（2026-10-02）

固定订单库存可选残差修正完成并拒绝：140月原源（复用132、新解析8），原1507状态/142月及115可用、27未知模型保持；可选字段完整1238、未知269保留NULL，完整K04卡不因此准入。只新增25次两系数估计、复用90月，原八项不重新拟合；原1010可用预测完整配对，497未知保持，98缺失分支精确回退，11符号变化非交易。较早原收益MSE增2.996769%、改进95%区间全负；近期减0.155065%、区间跨0，双期门失败。8必要测试、158冻结源及142版本/1507预测/25正规方程/24周期损失复算通过、预测误差0；账户SKIPPED，收益夏普NOT_COMPUTED，0新标签/账户/网络。TECH.R116协议、R117结果，旧失败及E03保持；K05不同连续资金压力残差源合同仅提案/NOT_RUN，独立验证及整体目标未达。

假设为订单库存扩散差水平及同口径三月变化增加原继续价值信息。不同用途在看本次结果前冻结：原八项系数/截距/尺度完全保留，只拟合周期内原预测残差；缺失保持原值NULL，修正不活动时直接返回原预测。所有原成熟训练行及每周期总权重1保留，alpha1、clip5、成熟钟与入场锁定不变；已知设计中心化使全部训练加权均值为0，不增加全局截距。没有将缺失字段视为测量到0，也没有按盈亏选择分支。

| 固定时期 | 配对/周期 | 原MSE | 修正MSE | 误差变化 | 改进95%入场年份块区间 |
|---|---:|---:|---:|---:|---:|
| 2015—2019 | 262/6 | 0.006796098625 | 0.006999761976 | +2.996769% | [−0.000358515142, −0.000030203246] |
| 2020—2026-09-30 | 748/18 | 0.005506526652 | 0.005497987952 | −0.155065% | [−0.000021440663, 0.000059304328] |

原源为历史重建开发材料，公布日23:59:59才可用，物理首发未认证。最新报告不可用时不回退更旧报告；三月差不跨2019分类、2022年4月样本或缺月。统一DI水平缩放是建模假设，不能证明版本完全可比。八月订单50.6与库存48.4仅本地同页重复表一致性，不冒充第二独立来源。首次必要测试发现微秒/纳秒时钟比较错误，在冻结和任何金融预测之前修复，原测试失败记录保留。

停止该固定表达的模型与账户晋升，不调方向/窗口/alpha/缺失掩码或只取近期营救。下一项为K05不同连续压力残差的有限源合同核对：DR007源下一交易日开盘可用，五原交易日政策利差均值及其五日变化，缺应有源观测保持未知，不把8月14日末值拖到9月；当前PROPOSED/NOT_RUN。K01—K06的有限旧用途、失败及完整卡界限在报告中分别记录，不修改原因子库。原E03新账户日/完整周期0、下一已登记收盘2026-10-08 15:05不变。

直接证据：[本轮报告](../reports/research/510300_point_macro_optional_correction_v1/研究结果与下一步.md)、[冻结协议](../reports/research/510300_point_macro_optional_correction_v1/protocol.json)、[真实预测](../reports/research/510300_point_macro_optional_correction_v1/prediction_summary.json)、[保存复算](../reports/research/510300_point_macro_optional_correction_v1/verification.json)、[下一项有限提案](../reports/research/510300_point_macro_optional_correction_v1/next_K05_source_and_model_proposal.json)。这不是独立验证、过拟合已消除或收益夏普提高的证明。


## TECH.R118—R119：资金价格连续状态的可选残差真实比较（2026-10-02）

固定资金利率可选残差修正完成并拒绝：DR007精确加权源2905行全部保留（2823原股票日、82额外银行间日），25操作变更加9/27已核实实施节点共26政策钟；原1507状态可选字段完整1368、139未知保持NULL。原142月/115可用27未知及八项不改，仅25次辅助估计/90月复用；原1010预测完整配对、497未知保持、28缺失精确回退，33符号变化非交易。较早MSE增3.695036%，近期减2.214278%，两改进95%下界均不正，双期门失败。11必要测试、50冻结源及2905源/3488日历/1507状态/142版本/25正规方程/24周期损失复算通过，预测误差0。账户SKIPPED、收益夏普NOT_COMPUTED，0新标签/账户/网络。TECH.R118协议、R119结果；原T14、R117及E03保持。下一H03仅不同用途/完整合约源钟有限提案NOT_RUN，不以换窗口复活旧OI；独立验证、去过拟合及完整目标未达。

原交易日t的两个输入固定为t-5..t-1五个DR日期各减当日已公布且已实施的政策利率均值，以及该均值相对t-5原点的变化，共需t-10..t-1十个原观测。DR为DR007.IB/DR007/weight、加权百分比，各利率日期后下一股票交易日09:30才可用，原点15:05；源末8/14合法延后可用后不再用末值补后续日。9/24宣布目标不当实际利率，9/27实施按23:59:59才可用。原所有成熟训练行、目标和周期总权重1保留；alpha1/clip5、原八项和入场锁定不变，修正无全局截距，未知原值仍NULL且精确保留原预测。

| 固定时期 | 配对/周期 | 原MSE | 修正MSE | 误差变化 | 改进95%年份块区间 |
|---|---:|---:|---:|---:|---:|
| 2015—2019 | 262/6 | 0.006796098625 | 0.007047216893 | +3.695036% | [−0.000534700367,0.000271257931] |
| 2020—2026-09-30 | 748/18 | 0.005506526652 | 0.005384596820 | −2.214278% | [−0.000000342492,0.000273599774] |

较早原可用262行可选输入全知；近期748行中720已知/28未知，全部配对。完整函数不等于完整K05卡字段准入，历史源也没有认证物理首发或独立验证。较早误差变差、近期区间跨0，不能只取近期、改费用/窗口/alpha/掩码或拼接R117营救。停止本固定表达，不运行失败后的经济账户。

下一步有限核对H03：跨全部实际有效IF合约总OI五日变化、现货r5与基差变化分别保存，先检查与旧单日OI及原IF_REL5的实质信息重叠、完整合约母集/到期/换月、2016价格可比界限、日统计发布时间和原全部成熟成员；仅窗口不同不够准入。当前PROPOSED/NOT_RUN，期货只观察，不执行；OI不代表净多资金。原E03真实新日/周期0及2026-10-08 15:05登记时点不变。

直接证据：[本轮报告](../reports/research/510300_point_funding_optional_correction_v1/研究结果与下一步.md)、[冻结协议](../reports/research/510300_point_funding_optional_correction_v1/protocol.json)、[真实预测](../reports/research/510300_point_funding_optional_correction_v1/prediction_summary.json)、[原源覆盖](../reports/research/510300_point_funding_optional_correction_v1/source_summary.json)、[保存复算](../reports/research/510300_point_funding_optional_correction_v1/verification.json)、[下一项有限提案](../reports/research/510300_point_funding_optional_correction_v1/next_H03_prior_and_source_proposal.json)。


## TECH.R120 / TECH.R121 接续：真实IF象限可选条件修正已拒绝

H03现货方向×总持仓方向及同合约基差的唯一可选残差实验完成并拒绝。真实IF源15860行/3965日/199合约、197月包；原1507状态完整输入1216、291未知保持。基差与旧IF_REL5的2572共同源日误差0，不能称新增信息；旧单日OI、旧宏观剩余价值及四组件衍生压力失败保留。原八项与142月115可用27未知不改，仅25次辅助估计/90月复用；1010预测完整配对、497原未知、30缺源精确回退，11符号变化非交易。较早MSE增0.788215%，改进95%区间全负；近期减0.208502%，区间跨零，双期门失败。11必要测试、226冻结对象、3488源日历/1507字段/142版本/25正规方程/24周期误差复算通过，预测误差0。经济SKIPPED、新账户/标签/行情请求0，收益夏普NOT_COMPUTED；定义浏览2次工具调用（2搜索、4打开正文均未读成功），历史首发/源发布钟未认证。TECH.R120协议/R121结果；原E03保持。下一J01股债共同状态仅有限旧用途/固定期限原源提案NOT_RUN；独立验证、去过拟合与完整目标未达。

| 固定时期 | 完整配对状态 / 周期 | 原MSE | 修正MSE | MSE相对变化 | 改进95%入场年块区间 |
|---|---:|---:|---:|---:|---:|
| 2015—2019 | 262 / 6 | 0.006796098625 | 0.006849666469 | +0.788215% | [-0.000245735391, -0.000008494800] |
| 2020—2026-09-30 | 748 / 18 | 0.005506526652 | 0.005495045417 | -0.208502% | [-0.000014820718, 0.000041103469] |


唯一六设计是固定四象限、确知平手类及同合约基差变化；实际平手0，设计列数不等于独立信息数。缺源保留NULL且修正精确原预测回退，不能把零修正当未知源值已测为0。旧普通OI方向交互、旧十宏观字段剩余清算修正及衍生压力家族失败不改；基差本来已用。可用钟沿原下一日收盘假设，实际发布和首版仍未认证。历史结果只是开发证据，MSE门失败没有完整账户收益结论。

下一J01仅有限旧用途和同一期限国债收益率原源合同提案，不能把国债价格收益与收益率变动互换，也不读取未公布终值。原1507/142成员、E03及各共享分支协议保持，未开展下一字段/模型/账户。[完整报告](../reports/research/510300_point_h03_optional_correction_v1/研究结果与下一步.md)、[原预测结果](../reports/research/510300_point_h03_optional_correction_v1/prediction_summary.json)、[保存核对](../reports/research/510300_point_h03_optional_correction_v1/verification.json)、[下一提案](../reports/research/510300_point_h03_optional_correction_v1/next_J01_joint_equity_bond_source_proposal.json)为直接证据。


## TECH.R122 / TECH.R123 接续：股债可选函数的固定双期比较

J01固定十年国债20股票区间变化×原mom20的两项可选周期内残差完成并拒绝。纠正R121下一提案遗漏：TECH.R104已做来源准入，原1486/1507算法值、115训练月齐全但21预测原点缺源且首版未证，旧严格门失败/0拟合保持。本次独立可选函数复用相同3660源日/原字段/23:59:59前序钟，缺源精确原模型回退，不认证旧源门。原八项/142月115可用27未知不改；25辅助估计/90月复用、1010完整配对/497未知、21精确回退、13符号变化非交易。早期MSE减0.450396%且改进95%区间为正，近期增1.032030%且区间跨零，整体FAIL；经济SKIPPED、账户/新标签0、收益夏普NOT_COMPUTED。7必要测试、49冻结对象、3488原字段与R104完全相同/1507预测/142模型/25方程/24周期误差复算通过，预测误差0。1官网定义读取确认当前17:30日终一般说明，页面最新报价不进入模型，未补21缺源或证明历史首版；行情历史下载0。TECH.R122协议/R123结果，原LPR失败、R104和E03保持。下一C04阶段内单系数可选残差仅另用途/函数提案NOT_RUN，R76完整第九项拒绝/T01不重开。独立验证、去过拟合与完整收益夏普目标未达。

| 固定时期 | 完整配对状态 / 自然周期 | 原MSE | 可选修正MSE | 相对变化 | 改进95%入场年块区间 | 门 |
|---|---:|---:|---:|---:|---:|---|
| 2015—2019 | 262 / 6 | 0.006796098625 | 0.006765489244 | -0.450396% | [0.000021472848, 0.000037904218] | PASS |
| 2020—2026-09-30 | 748 / 18 | 0.005506526652 | 0.005563355659 | +1.032030% | [-0.000154076769, 0.000030615504] | FAIL |


R104严格来源门失败与本次完整可选预测函数分别保存。原来源和字段/时钟完全不变，后者只在有值时修正原周期内残差，缺源精确原预测回退，不认证首版或补成零源值。当前官网17:30一般说明只核定义，没有加入网页最新曲线报价。早期过门不能抵消近期失败，经济SKIPPED没有新净收益/夏普；13符号变化非交易。下一C04单系数阶段内可选函数仍NOT_RUN，旧R76直接第九项数量拒绝/T01不重开。

[报告](../reports/research/510300_point_j01_optional_correction_v1/研究结果与下一步.md)、[R104区别及提案纠正](../reports/research/510300_point_j01_optional_correction_v1/prior_definition_and_source_addendum.json)、[原配对真实结果](../reports/research/510300_point_j01_optional_correction_v1/prediction_summary.json)、[保存复算](../reports/research/510300_point_j01_optional_correction_v1/verification.json)、[下一C04提案](../reports/research/510300_point_j01_optional_correction_v1/next_C04_stage_optional_residual_proposal.json)为证据。原E03、其他分支和全部旧金融终态保持，目标未达。


## TECH.R124 / TECH.R125 接续：回调成交额比的单系数可选函数

C04原回调阶段成交额比的固定单系数可选周期内残差实际完成，双期拒绝。原R76确认根数/现金前向价格/实时阶段/前5日固定均额及原字段完全不改，3488日线1121字段，1507原状态502已知1005原NO_VIEW保持；旧完整第九项数量门仍失败。唯一单列函数、原八项/142月115可用27未知保持，25辅助估计/90月复用，核心0重拟合。1010完整配对/497原未知、686精确回退、1符号变化非交易；2015—2019 MSE增0.088718%，2020—2026-09-30增0.051511%，两个改进95%年份块区间均跨零，经济SKIPPED、账户/新标签0、收益夏普NOT_COMPUTED。10必要测试及44冻结对象，与R76原字段完全一致/1507预测/142版本/25方程/24周期误差复算通过，预测差0。TECH.R124协议/R125实际结果，原R76、T01、既有失败及E03前瞻登记保持；0联网/行情请求。下一A02短阶段保留率的单系数另用途提案NOT_RUN；28旧冻结对象/3488字段/1507身份仅元数据预检一致，早期78/262、近期156/748原可用预测有字段，原R84严格拒绝不改。独立验证、去过拟合与完整收益夏普目标未达。

| 固定时期 | 完整配对状态 / 自然周期 | 原MSE | 可选修正MSE | 相对变化 | 改进95%入场年块区间 | 门 |
|---|---:|---:|---:|---:|---:|---|
| 2015_2019 | 262 / 6 | 0.006796098625 | 0.006802127987 | +0.088718% | [-0.000011182661, 0.000000496075] | FAIL |
| 2020_2026 | 748 / 18 | 0.005506526652 | 0.005509363131 | +0.051511% | [-0.000007774184, 0.000001327984] | FAIL |

原R76直接新增完整第九项因全部原成员支持不完整而停止，0收益模型拟合的旧裁决保持。本次另定义可选函数，原字段不改变：当前T确认中心T−2的左右各2根高低点，在已经确认的上涨段中实时开回调阶段e，e..T平均成交额除e之前5个完整日固定均额。非回调/上涨段失效仍原NO_VIEW；不追溯、不前填，不增加深度、期限、第二字段或交互。

新增函数仅一个系数，固定原八项参数、原标签和全部原成熟训练行、周期总权重1。已知行权重标准化并clip±5/再中心化，设计和原目标减固定原预测的残差各在周期内中心化，固定alpha1、新全局截距0；未知字段只令修正不活动并精确返回原预测，未把源值补0。原模型未知时双方未知。双期MSE与原5000次/seed51030099入场年块95%下界门均需过；本次双期失败，停止该版本，不执行账户或调整参数。

当前目标仍是20万元同一完整账户、252日全日历和原BASE/STRESS成本，两期净CAGR和净夏普均超过A、实际净pB>1、标准净期望pB−q>0、最大回撤不超过10%，交易次数为软目标。此处预测误差不能替代完整账户收益。历史均DEVELOPMENT_CALIBRATION，历史物理首版、独立验证与全项目DSR/PBO未建立/未计算；没有晋升或“已去除过拟合”。

下一A02只拟议在原突破后1—3完整日保留率有值时做单系数可选残差。保留原R84严格支持拒绝：371/1507字段、115原可用训练月完整支持0。元数据预检核对28旧冻结对象、3488原字段及1507身份，未读取目标列；原可用预测早期78/262、近期156/748有字段。下一续行先核对设计可识别性及已用函数，完成必要测试后另冻结唯一模型；无可识别设计则0拟合停止，双期门失败则拒绝，不扩大3日/20日/0.5ATR或改原未知。当前A02可选模型NOT_RUN，不创建新入场，不重开旧突破回踩/T01。

原E03 SAVED_WEIGHT/POINT_BINARY登记与首次合法原点2026-10-08 15:05保持，新增前瞻账户日和闭合周期均0。共享宏观E49—E52及盘口日线政策分支的授权、结果与未跑状态保持。

直接证据：[完整报告](../reports/research/510300_point_c04_optional_correction_v1/研究结果与下一步.md)、[协议](../reports/research/510300_point_c04_optional_correction_v1/protocol.json)、[原字段一致](../reports/research/510300_point_c04_optional_correction_v1/source_summary.json)、[实际预测](../reports/research/510300_point_c04_optional_correction_v1/prediction_summary.json)、[保存模型](../reports/research/510300_point_c04_optional_correction_v1/candidate_models.json)、[必要测试](../reports/research/510300_point_c04_optional_correction_v1/tests_receipt.json)、[复算](../reports/research/510300_point_c04_optional_correction_v1/verification.json)、[下一A02提案](../reports/research/510300_point_c04_optional_correction_v1/next_A02_single_optional_residual_proposal.json)。


## TECH.R126 / TECH.R127 接续：突破后保留率的单系数可选函数

A02突破后1—3完整日保留率的单系数可选周期内残差完成并双期拒绝。原R84现金前向OHLC/前20日高点/前序ATR20/3日/0.5ATR/最新严格过去突破全部不改；3488日线576字段，1507状态371已知1136NO_VIEW保持，94真实零仍已知，旧完整第九项门仍失败。唯一单列函数，原八项/142月115可用27未知保持，25辅助估计/90月复用、核心0重拟合；1010完整配对/497原未知、776精确回退、2符号变化非交易。早期MSE增0.049499%，改进95%[-0.000004805474,0]；近期增0.014160%，[-0.000009399845,+0.000004845883]跨零，整体FAIL；经济SKIPPED、账户/新标签0、收益夏普NOT_COMPUTED。12必要测试/50冻结对象，115月设计信息预检未读目标，3488字段与R84完全相同/142版本/25方程/1507预测/24周期误差复算通过、预测差0。TECH.R126协议/R127结果，原R84、T01、旧突破回踩及E03保持，0联网/行情。下一C03固定十日量价效率差的单系数另用途提案NOT_RUN；24旧源/3488字段/1507身份元数据预检一致，早期262/262、近期742/748原可用预测有字段；原R88及HIGH/LOW失败不改。独立验证、去过拟合和完整目标未达。

| 固定时期 | 完整配对状态 / 自然周期 | 原MSE | 可选修正MSE | 相对变化 | 改进95%入场年块区间 | 门 |
|---|---:|---:|---:|---:|---:|---|
| 2015_2019 | 262 / 6 | 0.006796098625 | 0.006799462624 | +0.049499% | [-0.000004805474, 0.000000000000] | FAIL |
| 2020_2026 | 748 / 18 | 0.005506526652 | 0.005507306391 | +0.014160% | [-0.000009399845, 0.000004845883] | FAIL |

原R84直接第九项因全成员字段支持失败而停止，0收益模型拟合保持。此次另定义固定核心之外的可选函数：严格过去最近突破e、前20日经济高点、冻结e−1简单ATR20、e后1—3完整日，最低价>=U−0.5ATR且收盘>=U的日数/已观察完整日数；今日新突破从下一完整日成为锚，原定义不改、不倒回e进入。原值未知仍各原NO_VIEW/NaN，真实零已知。只有一个辅助系数、alpha1，无新全局截距；原八项、全部原成熟成员/目标/周期总权重1、入场锁定保持。设计和原预测残差周期内中心化，缺字段精确回原模型，不认证旧完整字段或历史首版。

12项必要测试首次通过，115原可用月设计信息均正且未读目标，25实际辅助估计/90月复用、核心0；50冻结来源及全部原字段/模型成员/方程/预测/24自然周期误差复算一致。早期区间上界恰为0而非正，近期跨零，双期预测门FAIL，经济SKIPPED，没有新净年化/夏普。2符号变化不计作交易次数；原R84/旧突破回踩/T01/C04及其他冻结失败保持，不调3/20/0.5、方向、未知、成员、时期或门营救，不拼失败分支。

完整目标仍是20万元252日全日历BASE/STRESS，两期净CAGR和净夏普均提高、实际净pB>1、标准期望pB−q>0、最大回撤<=10%，次数软目标。当前历史均DEVELOPMENT_CALIBRATION，独立验证未建立，全项目DSR/PBO未计算，去过拟合和目标未达。

下一C03只拟议固定十日正负经济log收益/各组CNY金额效率差的单系数可选残差，旧R88数量门拒绝/原HIGH-LOW进入失败不重开。24旧源/3488字段/1507身份元数据预检与旧值完全相同、未读取目标，原可用预测早期262/262、近期742/748有字段；8原未知字段仍未知，不填分母0或epsilon。下一须核对已有用途和单列可识别性，完成必要测试后另冻结唯一函数；无识别信息或双期门失败即停止。当前新C03模型未实现、未冻结、NOT_RUN。

原E03 SAVED_WEIGHT/POINT_BINARY前瞻登记与首次合法原点2026-10-08 15:05保持，新增账户日/闭合周期0。共享宏观及盘口各自结果/权限不变。本回合是实际新拟合与固定拒绝的进展，目标active、连续阻塞计数0，不是完成或暂停。

直接证据：[完整报告](../reports/research/510300_point_a02_optional_correction_v1/研究结果与下一步.md)、[定义与区别](../reports/research/510300_point_a02_optional_correction_v1/prior_definition_and_function_addendum.json)、[协议](../reports/research/510300_point_a02_optional_correction_v1/protocol.json)、[字段一致](../reports/research/510300_point_a02_optional_correction_v1/source_summary.json)、[实际预测](../reports/research/510300_point_a02_optional_correction_v1/prediction_summary.json)、[模型](../reports/research/510300_point_a02_optional_correction_v1/candidate_models.json)、[设计预检](../reports/research/510300_point_a02_optional_correction_v1/design_identifiability_preflight.json)、[测试](../reports/research/510300_point_a02_optional_correction_v1/tests_receipt.json)、[复算](../reports/research/510300_point_a02_optional_correction_v1/verification.json)、[下一C03提案](../reports/research/510300_point_a02_optional_correction_v1/next_C03_single_optional_residual_proposal.json)。
