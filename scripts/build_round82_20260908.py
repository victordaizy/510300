"""只登记两条既有参考收益的非负最小方差预算，不扫描参数。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    require(not (ROOT / "research/two_policy_min_variance_v1.py").exists(), "第82轮已建立")
    source = (ROOT / "research/two_policy_risk_budget_inputs_v1.py").read_text(encoding="utf-8")
    source = source.replace('"""两条既有参考策略只按已实现风险更新预算，保留缺失与日终目标。"""',
        '"""两条既有参考策略按过去组合方差最小化分预算，完整保留日历。"""')
    source = source.replace("    count = 0\n", "    count = 0\n    covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan\n")
    source = source.replace('"risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD",',
        '"reference_covariance": np.nan, "difference_variance": np.nan, "raw_panic_budget": np.nan,\n                "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD",')
    source = source.replace("            sd = np.full(2, np.nan)\n", "            sd = np.full(2, np.nan)\n            covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan\n")
    old = '''                    # 与标准差倒数归一化相同，避免显式计算极大倒数。
                    weights = np.array([sd[1], sd[0]]) / sd.sum()
                    successful, status = dates[t], "RISK_BUDGET_AVAILABLE"'''
    new = '''                    centered = values-values.mean(axis=0)
                    covariance = float((centered[:, 0]*centered[:, 1]).sum()/(window-1))
                    # 直接计算收益差的方差，避免两项近似相等方差相减造成负值。
                    difference_variance = float(np.var(values[:, 0]-values[:, 1], ddof=1))
                    if not np.isfinite(difference_variance) or difference_variance <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        raw_panic_budget = float((sd[1]**2-covariance)/difference_variance)
                        panic_weight = float(np.clip(raw_panic_budget, 0., 1.))
                        weights = np.array([panic_weight, 1.-panic_weight])
                        successful, status = dates[t], "MIN_VARIANCE_BUDGET_AVAILABLE"'''
    require(source.count(old) == 1, "最小方差替换位置不唯一")
    source = source.replace(old, new)
    source = source.replace('"risk_status": status, "risk_update_scheduled": scheduled,',
        '"reference_covariance": covariance, "difference_variance": difference_variance, "raw_panic_budget": raw_panic_budget,\n            "risk_status": status, "risk_update_scheduled": scheduled,')
    (ROOT / "research/two_policy_min_variance_inputs_v1.py").write_text(source, encoding="utf-8")
    tests = (ROOT / "tests/test_two_policy_risk_budget_v1.py").read_text(encoding="utf-8")
    tests = tests.replace("research.two_policy_risk_budget_inputs_v1", "research.two_policy_min_variance_inputs_v1")
    tests = tests.replace("1/3", "0.")
    tests += '''

def test_covariance_changes_budget_with_same_individual_variances():
    dates, returns, states = fixture()
    same = budget_frame(dates, returns, states, 1, 5)
    opposite = returns.copy()
    opposite[:, 0] *= -1
    hedge = budget_frame(dates, opposite, states, 1, 5)
    month = dates.get_loc(pd.Timestamp("2020-02-03"))
    assert same.panic_budget.iloc[month] == 0.
    assert np.isclose(hedge.panic_budget.iloc[month], 1/3)
    a = hedge.panic_budget.iloc[month]
    assert np.var(opposite[month-4:month+1] @ [a, 1-a], ddof=1) < 1e-25


def test_identical_nonzero_returns_keep_previous_budget_as_unidentified():
    dates, returns, states = fixture()
    march = dates.get_loc(pd.Timestamp("2020-03-02"))
    returns[march-4:march+1, 0] = returns[march-4:march+1, 1]
    result = budget_frame(dates, returns, states, 1, 5)
    assert result.risk_status.iloc[march] == "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
    assert result.panic_budget.iloc[march] == 0.
    assert result.last_successful_risk_origin.iloc[march] < dates[march]


def test_interior_and_boundary_budgets_minimize_the_observed_quadratic():
    dates, returns, states = fixture()
    rng = np.random.default_rng(826)
    returns = rng.normal(0, .01, (len(dates), 2))
    returns[:, 0] *= 2
    result = budget_frame(dates, returns, states, 1, 5)
    for t in np.flatnonzero(result.risk_update_scheduled.to_numpy()):
        if result.risk_status.iloc[t] != "MIN_VARIANCE_BUDGET_AVAILABLE":
            continue
        window = returns[t-4:t+1]
        w = result.panic_budget.iloc[t]
        actual = np.var(window @ [w, 1-w], ddof=1)
        grid = np.linspace(0, 1, 101)
        alternatives = window[:, 0, None]*grid + window[:, 1, None]*(1-grid)
        assert actual <= np.var(alternatives, axis=0, ddof=1).min()+1e-15
'''
    (ROOT / "tests/test_two_policy_min_variance_v1.py").write_text(tests, encoding="utf-8")
    runner = (ROOT / "research/two_policy_risk_budget_v1.py").read_text(encoding="utf-8")
    runner = runner.replace("two_policy_risk_budget", "two_policy_min_variance")
    runner = runner.replace("TWO_POLICY_RISK_BUDGET", "TWO_POLICY_MIN_VARIANCE")
    runner = runner.replace('P46 = ROOT /', 'P76 = ROOT / "reports/research/510300_two_policy_risk_budget_v1"\nP46 = ROOT /', 1)
    runner = runner.replace('NAME = "两条原策略按已实现风险分配预算"', 'NAME = "两条原策略非负最小方差预算"')
    runner = runner.replace('"BUY_HOLD": "买入持有"}', '"BUY_HOLD": "买入持有", "TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算"}')
    runner = runner.replace('round=76,', 'round=82, objective="MINIMIZE_TWO_REFERENCE_FULL_CALENDAR_SAMPLE_VARIANCE",')
    runner = runner.replace('previous_goal_turn_classification="PROGRESS_ROUND75_COMPLETED_AND_DELIVERED"', 'previous_goal_turn_classification="PROGRESS_ROUND81_SOURCE_FIX_COMPLETE"')
    runner = runner.replace('"NO_VIEW_KEEP_PREVIOUS_BUDGET", rules=', '"NO_VIEW_KEEP_PREVIOUS_BUDGET", degenerate_difference="NO_VIEW_KEEP_PREVIOUS_BUDGET", rules=')
    runner = runner.replace('P46 / period / cost / f"{model}_ledger.parquet" for model in CONTROLS', '(P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost / f"{model}_ledger.parquet" for model in CONTROLS')
    runner = runner.replace('pd.read_parquet(P46 / period / cost_id / f"{model}_ledger.parquet")', 'pd.read_parquet((P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost_id / f"{model}_ledger.parquet")')
    runner = runner.replace('"evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8', '"evaluation_accounts": 12, "new_accounts_generated": 2, "reused_control_accounts": 10')
    runner = runner.replace('"earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8', '"earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 10')
    runner = runner.replace('ROOT / cfg["rules"],', 'ROOT / cfg["rules"], ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",', 1)
    runner = runner.replace("第76轮一个242日月度风险预算设置已冻结，尚无新策略收益。", "第82轮单项非负最小方差预算已冻结，尚无新策略收益。")
    runner = runner.replace("新账户和四个保存对照完成", "新账户和五个保存对照完成")
    (ROOT / "research/two_policy_min_variance_v1.py").write_text(runner, encoding="utf-8")
    (ROOT / "docs/510300_TWO_POLICY_MIN_VARIANCE_V1.md").write_text('''# 第82轮：两条既有策略的非负最小方差预算

只登记一个设置。第81轮修正后两段夏普均退步，停止分批进入及尚未登记的组合分批。第76轮倒数波动预算主改善、较早退步，第77轮活动日风险定义失败，全部保留。本次改用组合总方差作为预算目标，显式考虑两个收益同时波动；不是扫描旧窗口、交易门槛或选择历史最高夏普。

有界查重：在研究实现、相关配置、风险与策略协议和最新索引中未找到同一两个参考策略的非负最小方差已运行设置。第76轮前也曾检查这一方向，不能声称整个未索引历史完全不存在类似研究。

## 新增因子与唯一预算公式

每月首个交易日收盘，取两条原基础费用参考账户截至当天的最近242个完整日净收益，包括现金日、分红和费用。分别减去各自窗口平均收益，平方合计除241得到各自样本方差；两者去均值偏差相乘合计除241得到样本协方差。这里均值仅用于去中心化，不作为未来预期收益预测。

急跌策略预算等于“学习策略方差减两策略协方差”，除以“两策略日收益之差的样本方差”；小于零取零，大于一取一。学习预算等于一减急跌预算。分母等价于两策略方差合计减两倍协方差，实现直接计算收益差方差，避免相近数字相减造成负值。这是两预算合计为一、均非负时的样本组合方差最小解；可以选择零或全额预算，不另加上下限或收缩强度。

沿用第76轮两段各自从50%／50%开始、仅月首收盘更新、下一开盘执行、窗口不足或任一参考波动为零或缺失时沿用此前预算。两条非零波动收益如果逐日差值恒定，差值方差为零，无法从方差目标识别唯一预算，本次明确保留无观点及此前预算。月首之间维持预算；终点开盘退出收益不参与新判断。不填缺失，不删除现金日，不借用评价起点前没有对应账户的历史。

原始组合方差含两条方差和两倍交叉协方差项，见William Sharpe在斯坦福的[两资产组合说明](https://web.stanford.edu/~wfsharpe/mia/rr/mia_rr5.htm)。本公式由该方差表达式求导并限制到非负预算区间；该网页后段一般解的系数与其前段方差表达式不一致，未直接抄用后段系数。必要合成测试用组合方差本身验证内部最优和边界最优。最小化过去方差不保证收益或夏普提高。

## 进入、减仓、退出和重新进入

两条原策略的参考进入、持有、退出、学习因子和再次进入规则全部沿用第76轮原协议，随本轮交付附上完整中文原文及已有月度八因子系数。原急跌策略下一开盘进入和退出状态不变；原学习策略继续采用基础费用满仓参考路径，第81轮的分批阶段不参与。

每天收盘股票目标为急跌预算乘原急跌持有状态，加学习预算乘原学习持有状态。两条都退出时股票目标零并请求全卖；只有一条持有就使用其预算；两条都持有则目标全额。即使某参考要求持有，其预算为零时组合也不投入。原状态任一缺失时没有新目标，保持真实股数。

组合在目标为正且实际空仓时，按自身完整净值、收盘价和100份整手计算下一开盘请求；已有持仓而目标与实际比例相差至少10个百分点时重新估算并买入或部分卖出，差距不足维持股数。零目标全卖不受带宽限制。月度预算改变本身可以导致组合退出或重新进入，不额外增加合并冷却。所有原参考内部的退出锁定和重新进入条件保持；合并账户受阻后下一收盘按最新有效目标重算。

基础与压力账户共用原基础参考状态及风险估计，各自真实计算费用和份额，不把参考净收益加权当成合并账户收益。估计对象是两条资金使用规则，并未新增可交易证券；现金保留于各自没有持有要求的预算中。

## 运行口径与快速验收

继续只有510300和人民币现金，20万元、242日年化、现金和无风险收益零。主2020年1月2日至2026年8月14日开盘共1604日，较早2015年1月5日至2019年12月31日开盘共1219日；原基础及压力两档费用、整手、T+1、方向涨跌停、分红登记及到账、全部日期不变。共4个新账户，另复用原各半、原学习、原急跌、买入持有和第76轮预算五类20份旧对照；主和较早各12份评价记录。没有新模型、市场下载或参考策略重建。

10项必要测试包括月首时钟、未来不改过去、完整窗口、零波动和缺失、参考终点、下一开盘与分红、相同边际风险而协方差不同、非零相同收益的不可识别状态、方差最优性。通过后冻结，再立即运行完整账户；只核对预算、实际目标与四个新账户。若失败，不扫描协方差收缩、风险窗口、预算上限或目标函数系数救回；历史点估计并非独立验证。继续暂停慢来源和GPT数值包。
''', encoding="utf-8")
    (ROOT / "reports/research/510300_two_policy_min_variance_v1").mkdir(parents=True, exist_ok=True)
    print("第82轮单项最小方差预算、中文规则与十项必要测试已建立。", flush=True)


if __name__ == "__main__":
    main()
