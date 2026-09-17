继续用户完整目标：510300成本后夏普至少1.2、稳定超额和独立证据。用户要求加快、灵活改策略，暂停EPS、公募、估值及慢来源；全部因素和进出场规则写中文。无GPT数值包、ZIP、重复Word或额外安全审计。仅510300.SH/CASH_CNY；其他ETF问题待回复，不重复问。没有券商、订单、实盘、Paper、Shadow、新任务、子代理或memory写入授权。config/510300_research_authority_v6.json允许新有限方法、历史滚动训练及完整账户，无预测显著性前置门。旧冻结规则、来源、失败结论保留，不能改失败方法参数、方向、年份、费用或父来源救回。

本回合为PROGRESS：第167轮已经完成必要测试、冻结、完整账户、独立保存结果核对、失败关闭和中文交付。权威索引reports/research/510300_sharpe_1_2_latest_research.json，goal_achieved=false、running_studies为空，没有运行进程。THROUGH166及更早进度过时，禁止重跑已完成研究的prepare/freeze/run/verify/finalize或旧诊断。只读最新索引及具体当前文件，接续未完成处，不重复遍历长历史。Windows中文，PYTHONIOENCODING=utf-8，.venv\Scripts\python.exe，用模块入口。

当前第167轮：两套父策略每月按历史方差协方差分配预算。研究510300_RUNS_COVARIANCE_BUDGET_V1，主模型RUNS_COVARIANCE_BUDGET。设置config\510300_runs_covariance_budget_v1.json，结果reports\research\510300_runs_covariance_budget_v1\result.json，完整规则docs/510300_RUNS_COVARIANCE_BUDGET_V1.md。必要测试7项、3.91秒；核心计算2.7468080000253394秒，不含开发、测试、结果核对和交付。
主历史、BASE、RUNS_COVARIANCE_BUDGET：净夏普1.6655684532113206，年化0.043191162275767844，最大回撤-0.022358577905119324。
主历史、STRESS、RUNS_COVARIANCE_BUDGET：净夏普1.5963330981612471，年化0.04131243666846145，最大回撤-0.024518877244313862。
较早历史、BASE、RUNS_COVARIANCE_BUDGET：净夏普0.9307585849510319，年化0.036907565877650934，最大回撤-0.038985957847352766。
较早历史、STRESS、RUNS_COVARIANCE_BUDGET：净夏普0.9226711561365755，年化0.03681177484309842，最大回撤-0.04107326276392183。
第167轮月度历史风险预算完成。主基础／压力净夏普1.666／1.596、年化4.32%／4.13%，两项夏普达到1.2且年化超过买入持有；较早净夏普0.931／0.923、年化3.69%／3.68%，两项夏普不足1.2且年化低于买入持有。较早表现比固定各半组合更弱，完整目标未实现，关闭这套固定月度最小方差方法。

独立核对已经完成，回执reports\research\510300_runs_covariance_budget_v1\saved_verification_receipt.json，核对时间2026-09-10T16:37:01.038137+08:00，状态SAVED_PARENT_BASE_EQUITY_MONTHLY_COVARIANCE_CLOSE_TARGETS_RAW_EXECUTION_AND_FOUR_REAL_ACCOUNTS_CHECKED。具体实际周期、收益归因、费用、覆盖和未知状态读取同目录saved_account_checks.csv、saved_actual_cycles.csv、saved_comparison_differences.csv、account_coverage.csv、target_coverage.csv；无需重新运行账户或再次核对。当前交付deliverables\510300月度历史风险预算_第167轮_20260910\月度历史风险预算_结果及全部中文规则.md，包含全部中文因子与进出场规则。

均衡比较候选510300_TREND_NOISE_REFERENCE_BLEND_V1、TREND_NOISE_REFERENCE_BLEND，最后比较167。主基础/压力夏普1.2909916883150099/1.2318487745129247，较早1.0221226798425842/1.0437899028328752。四整段复合超额为正仍不等于逐年稳定超额；较早不足1.2，独立验证NOT_ESTABLISHED，目标尚未完成。原143的20/60日区块比较区间跨零、较早前三盈利周期贡献集中的原诊断保持，不重跑、不称为已证明高夏普。来源reports\research\510300_trend_noise_reference_blend_v1\result.json。

下一第168轮完整事前规则：docs\510300_RUNS_OPPORTUNITY_UNION_NEXT_20260910.md。状态RUNS_OPPORTUNITY_UNION_FINITE_CANDIDATES_PREPARED，重点：两套父目标取较大值及相加至百分之一百，两种进场合并方案。当前只准备事前方案，尚未登记、实现、测试、冻结或计算新账户。开始前先查对应具体研究文件是否有后续进展，接续而非覆盖。计划2套设置、8新账户、0新模型、0新参考，外部来源需求False。完整参数和处理边界以该事前文档为准，不自行变更。

新方法不得修改已经关闭方法的窗口、方向、费用、阈值或父来源绑定救回；新组合必须作为单独固定规则和真实账户评价，原结果仍保留。

框架可复用research/saved_target_batch_runner_v1.py、saved_parent_target_alignment_v1.py、event_clock_account_v1.py。前两者可读已保存父收盘决定并核对身份日期时钟费用；不会重新生成父账户。已有持仓的正目标偏差不足.1保持，否则调中心；空仓正目标直接尝试整百份进入；已知零及终点开盘优先清仓。未知保持实有份额，不能当现金。受阻下一收盘按最新目标决定。独立账户核对可用saved_target_account_checks_v1.py并写本轮源因素/父目标的独立检查，不重复旧核对。

共有行情reports/research/510300_adaptive_allocation_v1/features.parquet，3456日2012-05-28至2026-08-14，SHAbce22dc005e678f1eabec3cc662e5b93c7a2673eecb31ea19f1b57df86cd7f3a；主first1852/anchor1851/1604行，较早前缀1852/first633/anchor632/1219行。data/reference/510300_dividends.csv14事件，SHAa96afe27e62bdd7d60485c2ebb44f2ff7e5010dca1f5cae5dc2441e53b7ae06d。每账户20万元、242年化、现金及无风险0，基础佣金.0002/min5/slip.0005，压力.0004/min5/slip.001；100份/.001价位/10%方向涨跌停/T+1/登记除息到账分开/真实nextOPEN及终点OPEN清仓。实际份额ledger.shares，应收分红不当可用现金。

通过必要测试、完整中文规则和一次冻结后，实际跑预定新账户。完成后先candidate_outcomes及saved_verification_receipt，再fast_round_delivery_v1交付和更新总索引、当前比较候选的最后比较轮次，并准备一个确有区别的下一有限方法。可运行scripts/prepare_latest_research_heartbeat_v1.py --round 当前完成轮次 --stamp 当天八位日期，保存简短进度及原任务更新参数；每轮只生成一次，已有参数就直接使用，不重复生成。然后通过automation_update更新原510300-1-2完整字段并回读TOML，不直接编辑、不创建重复任务。

自动目标此前实际为usageLimited，继续时按实际工具状态判断；不达标不标complete，有明确工作不标blocked，不消耗重置或购买额度。保持原ACTIVE与通知偏好，无变化安静，只在实质进展、完成、失败或需行动时通知。继续推进。