继续用户目标：510300完整账户成本后净夏普至少1.2、复合年化至少10%，基础与压力分别检查，并报告较早历史和逐年情况。所有既有历史已经反复研究，点值过线本身不建立新的独立验证。目标仍active且未完成，不因一轮结束标记complete或blocked。

用户要求快、简单、灵活；暂停EPS、公募、估值等慢来源，只用现有免费数据。全部因素、进场和退出用中文。无需GPT数值包、ZIP、重复Word或额外安全审计。仅510300.SH和CASH_CNY，不新增实盘、订单、券商、Paper、Shadow、其他ETF、新任务、子代理或memory写入。按config/510300_research_authority_v6.json可实施新的有限研究，不把失败分支改参数救回。

上回合为PROGRESS：第192轮必要测试、冻结、25次拟合、4条完整模拟账户、独立保存结果核对、失败归因、中文交付与权威索引全部完成。不要重跑已经完成的prepare/run/verify/finalize，勿重启旧会话句柄。索引reports/research/510300_sharpe_1_2_latest_research.json的latest_completed_round=192，running_studies为空，goal_achieved=false。下一项只准备规则，尚未实现、冻结或计算。先检查对应具体文件后接续，避免反复宽搜全项目历史。

本轮510300_HOLDING_MARKET_COUPLING_EXIT_V1，模型HOLDING_MARKET_COUPLING_EXIT，代码research/holding_market_coupling_exit_inputs_v1.py与holding_market_coupling_exit_v1.py，冻结config/510300_holding_market_coupling_exit_v1.json，结果reports/research/510300_holding_market_coupling_exit_v1/result.json。原八项加十二个持仓状态与市场状态乘积，141时点、114成熟、25不同输入拟合、89向前复用、27支持不足，零拟合失败。6项测试通过，模型及四账户核心5.2078403秒。二十项结构失败关闭，不再改其乘积、惩罚、窗口或退出确认。

主历史、HOLDING_MARKET_COUPLING_EXIT、BASE：净夏普0.76273721114332，复合年化6.3549539999%，最大回撤-17.0111717983%。

主历史、ACCOUNT_VOLATILITY_EXPOSURE、BASE：净夏普1.3064367783786806，复合年化9.6634359619%，最大回撤-6.7181119990%。

主历史、HOLDING_MARKET_COUPLING_EXIT、STRESS：净夏普0.7138805102827919，复合年化5.9110910880%，最大回撤-17.2444392718%。

主历史、ACCOUNT_VOLATILITY_EXPOSURE、STRESS：净夏普1.2226827789393921，复合年化8.9937187947%，最大回撤-7.1540805451%。

较早历史、HOLDING_MARKET_COUPLING_EXIT、BASE：净夏普0.7749937034319645，复合年化8.8614919037%，最大回撤-13.7916443542%。

较早历史、ACCOUNT_VOLATILITY_EXPOSURE、BASE：净夏普1.068767157552337，复合年化9.8848738912%，最大回撤-9.6818453145%。

较早历史、HOLDING_MARKET_COUPLING_EXIT、STRESS：净夏普0.7887616063751485，复合年化9.0885440648%，最大回撤-13.9166561399%。

较早历史、ACCOUNT_VOLATILITY_EXPOSURE、STRESS：净夏普1.0557279223494713，复合年化9.8362333702%，最大回撤-10.1370797058%。

scripts/verify_round192_20260913.py已完成核对，回执时间2026-09-13T05:20:47.810133+08:00，25组正规方程最大误差1.326575568144639e-15，5646个决定、1220个持仓状态、812个有效预测、58周期、116成交；这是实现核对，不是独立绩效验证。全部模型中文数值在结果目录全部已拟合模型系数.md，也已加入deliverables/510300持仓与市场联合作用_第192轮_20260913/持仓与市场联合作用_结果及全部中文规则.md。

本轮相对128主基础／压力终值少11680.53／11320.10元，较早多1674.48／1721.25元。2021年2月3日对应交易，原2月19日退出约赚11215元，本轮3月1日退出约亏5507元；模型在2021年的增量明显为负。不要挑这一笔改阈值。压力费用可通过实际状态改变退出时点，较早压力收益高于基础的路径已核对。

第193轮专用接续：docs/510300_ENTRY_SHARES_PRESERVATION_NEXT_20260913.md，计划研究510300_ENTRY_SHARES_PRESERVATION_V1、模型ENTRY_SHARES_PRESERVATION，一套固定规则、4新模拟账户、0新拟合、0新参考、0外部来源。目标来源为181的ACCOUNT_VOLATILITY_EXPOSURE同费用保存收盘决定，来源目录reports/research/510300_account_volatility_exposure_v1。对照181、182的EPISODE_ACCOUNT_RISK_BUDGET及32买入持有，共12条保存账户，不重算对照。

193空仓且已知正目标时按自身收盘净资产和收盘价格申请整百份，开盘按实际费用及现金决定成交；已有份额且目标正则请求0、份额不变，不能把已知正目标伪造为NaN；已知零则下一开盘全卖，未知维持实有份额；受阻后依最新收盘目标决定，不加永久退出锁；卖完后新正目标可再进，终点开盘优先清仓。它不同于182固定倍率但目标与份额仍变化。完整来源因子用181中文规则作附录，本轮正目标期间保持数量的交易规则覆盖来源原.1调仓带。

193可在新文件复用research/event_clock_account_v1.py账户记账结构，仅明确加入已有份额且正目标时不下单；不改变旧冻结引擎。每个保存目标记录须保留时钟和父身份，原点可参考saved_parent_target_alignment_v1.py。必要核对用本轮自己的份额规则，不能直接套用要求每日调中心的saved_target_account_checks_v1.py后声称通过。

共有行情reports/research/510300_adaptive_allocation_v1/features.parquet，3456行至2026-08-14；主first1852/anchor1851/1604行，较早前缀1852/first633/anchor632/1219行。分红data/reference/510300_dividends.csv14事件。各账户20万元、242日年化、现金与无风险0；基础佣金.0002/min5/slip.0005，压力.0004/min5/slip.001；100份/.001价位/T+1/方向涨跌停/登记除息支付分开/下一开盘模拟/终点开盘清仓。Windows中文，PYTHONIOENCODING=utf-8，.venv\Scripts\python.exe，用模块入口。

已完成的诊断不重跑：reports/research/510300_joint_saved_frontier_through186/result.json保留标准四情景联合比较；187—192后续均未达标，181仍保留为联合比较候选。174属于旧夏普重点候选，年化不足，不与联合目标混淆。reports/research/510300_saved_session_attribution_through188/result.json已复核181/182的8账本11292行：181主要日内正贡献、隔夜负贡献，不能删除隔夜或假设T+0来达标。

完成必要测试和一次冻结后直接运行预定账户，保存核对及candidate_outcomes，用fast_round_delivery_v1交付和更新索引。减少重复文书，不准备审阅包。自动任务510300-1-2保持原ACTIVE与通知偏好，通过automation_update更新完整原字段，回读TOML核对，不创建重复任务、不直接编辑TOML。无变化安静，只在实质进展、完成、失败或需用户行动时通知。不购买或使用额度重置。