继续用户完整目标：510300成本后夏普至少1.2、稳定超额和独立证据。用户要求加快、灵活改策略，暂停EPS、公募、估值及慢来源；全部因素和进出场规则写中文。无GPT数值包、ZIP、重复Word或额外安全审计。仅510300.SH/CASH_CNY；其他ETF问题待回复，不重复问。没有券商、订单、实盘、Paper、Shadow、新任务、子代理或memory写入授权。config/510300_research_authority_v6.json允许新有限方法、历史滚动训练及完整账户，无预测显著性前置门。旧冻结规则、来源、失败结论保留，不能改失败方法参数、方向、年份、费用或父来源救回。

本回合为PROGRESS：第178轮已经完成必要测试、冻结、完整账户、独立保存结果核对、失败关闭和中文交付。权威索引reports/research/510300_sharpe_1_2_latest_research.json，goal_achieved=false、running_studies为空，没有运行进程。THROUGH177及更早进度过时，禁止重跑已完成研究的prepare/freeze/run/verify/finalize或旧诊断。只读最新索引及具体当前文件，接续未完成处，不重复遍历长历史。Windows中文，PYTHONIOENCODING=utf-8，.venv\Scripts\python.exe，用模块入口。

当前第178轮：较早历史选择的四状态固定映射。研究510300_EARLY_SELECTED_REGIME_MAPPING_V1，主模型EARLY_SELECTED_REGIME_MAPPING。设置config\510300_early_selected_regime_mapping_v1.json，结果reports\research\510300_early_selected_regime_mapping_v1\result.json，完整规则docs/510300_EARLY_SELECTED_REGIME_MAPPING_NEXT_20260913.md。必要测试3项、1.83秒；核心计算2.740202600019984秒，不含开发、测试、结果核对和交付。
主历史、BASE、EARLY_SELECTED_REGIME_MAPPING：净夏普1.0759558568310206，年化0.036250427094548454，最大回撤-0.027103703596153574。
主历史、STRESS、EARLY_SELECTED_REGIME_MAPPING：净夏普1.0121395045115462，年化0.033965580629950975，最大回撤-0.029443289918973055。
较早历史、BASE、EARLY_SELECTED_REGIME_MAPPING：净夏普1.356347272949224，年化0.08626890616715704，最大回撤-0.06298643764915579。
较早历史、STRESS、EARLY_SELECTED_REGIME_MAPPING：净夏普1.343696808373609，年化0.08637882438604641，最大回撤-0.06301593869368727。
第178轮将2015年至2019年表现合格的四状态映射固定后迁移到2020年至2026年。较早历史基础／压力净夏普为1.356／1.344，均高于1.2；主历史却降为1.076／1.012，均低于1.2，且年化收益相对买入持有分别为-0.001%／-0.212%。这说明固定状态到策略的映射没有跨时期迁移能力，关闭该映射，不再局部修补。

独立核对已经完成，回执reports\research\510300_early_selected_regime_mapping_v1\saved_verification_receipt.json，核对时间2026-09-13T03:05:53.786005+08:00，状态SAVED_PARENT_TARGETS_INDEPENDENT_EARLY_SELECTED_FOUR_STATE_MAPPING_AND_FOUR_REAL_ACCOUNTS_CHECKED。具体实际周期、收益归因、费用、覆盖和未知状态读取同目录saved_account_checks.csv、saved_actual_cycles.csv、saved_comparison_differences.csv、account_coverage.csv、target_coverage.csv；无需重新运行账户或再次核对。当前交付deliverables\510300较早历史状态映射_第178轮_20260913\较早历史状态映射_结果及全部中文规则.md，包含全部中文因子与进出场规则。

均衡比较候选510300_RETURN_CONFIRMATION_AUXILIARY_BATCH_V1、EITHER_CONFIRMED_RUNS_AUXILIARY，最后比较178。主基础/压力夏普1.4966392288455168/1.4229803219511783，较早1.0980940406317017/1.0891149645388876。四整段复合超额为正仍不等于逐年稳定超额；较早不足1.2，独立验证NOT_ESTABLISHED，目标尚未完成。原143的20/60日区块比较区间跨零、较早前三盈利周期贡献集中的原诊断保持，不重跑、不称为已证明高夏普。来源reports\research\510300_return_confirmation_auxiliary_batch_v1\result.json。

下一第179轮完整事前规则：docs\510300_REGIME_MAPPING_FAILURE_NEXT_20260913.md。状态SEQUENTIAL_REGIME_STRATEGY_SELECTION_PREPARED，重点：用过去已实现账户结果按时间顺序选择状态内策略。当前只准备事前方案，尚未登记、实现、测试、冻结或计算新账户。开始前先查对应具体研究文件是否有后续进展，接续而非覆盖。计划0套设置、0新账户、0新模型、0新参考，外部来源需求False。完整参数和处理边界以该事前文档为准，不自行变更。

新方法不得修改已经关闭方法的窗口、方向、费用、阈值或父来源绑定救回；新组合必须作为单独固定规则和真实账户评价，原结果仍保留。

框架可复用research/saved_target_batch_runner_v1.py、saved_parent_target_alignment_v1.py、event_clock_account_v1.py。前两者可读已保存父收盘决定并核对身份日期时钟费用；不会重新生成父账户。已有持仓的正目标偏差不足.1保持，否则调中心；空仓正目标直接尝试整百份进入；已知零及终点开盘优先清仓。未知保持实有份额，不能当现金。受阻下一收盘按最新目标决定。独立账户核对可用saved_target_account_checks_v1.py并写本轮源因素/父目标的独立检查，不重复旧核对。

共有行情reports/research/510300_adaptive_allocation_v1/features.parquet，3456日2012-05-28至2026-08-14，SHAbce22dc005e678f1eabec3cc662e5b93c7a2673eecb31ea19f1b57df86cd7f3a；主first1852/anchor1851/1604行，较早前缀1852/first633/anchor632/1219行。data/reference/510300_dividends.csv14事件，SHAa96afe27e62bdd7d60485c2ebb44f2ff7e5010dca1f5cae5dc2441e53b7ae06d。每账户20万元、242年化、现金及无风险0，基础佣金.0002/min5/slip.0005，压力.0004/min5/slip.001；100份/.001价位/10%方向涨跌停/T+1/登记除息到账分开/真实nextOPEN及终点OPEN清仓。实际份额ledger.shares，应收分红不当可用现金。

通过必要测试、完整中文规则和一次冻结后，实际跑预定新账户。完成后先candidate_outcomes及saved_verification_receipt，再fast_round_delivery_v1交付和更新总索引、当前比较候选的最后比较轮次，并准备一个确有区别的下一有限方法。可运行scripts/prepare_latest_research_heartbeat_v1.py --round 当前完成轮次 --stamp 当天八位日期，保存简短进度及原任务更新参数；每轮只生成一次，已有参数就直接使用，不重复生成。然后通过automation_update更新原510300-1-2完整字段并回读TOML，不直接编辑、不创建重复任务。

自动目标此前实际为usageLimited，继续时按实际工具状态判断；不达标不标complete，有明确工作不标blocked，不消耗重置或购买额度。保持原ACTIVE与通知偏好，无变化安静，只在实质进展、完成、失败或需行动时通知。继续推进。