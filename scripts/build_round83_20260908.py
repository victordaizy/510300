"""复用第76轮参考时钟，建立唯一95%尾部损失预算设置。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    require(not (ROOT / "research/two_policy_tail_loss_v1.py").exists(), "第83轮已建立")
    source = (ROOT / "research/two_policy_risk_budget_inputs_v1.py").read_text(encoding="utf-8")
    source = source.replace("from research.intraday_overnight_increment_v1 import require", "from research.intraday_overnight_increment_v1 import require\nfrom research.two_policy_tail_loss_optimizer_v1 import optimal_tail_budget")
    source = source.replace("first, window=242):", "first, window=242, confidence=.95):")
    source = source.replace("    count = 0\n", "    count = 0\n    details = {key: np.nan for key in ['optimal_tail_loss', 'selected_tail_loss', 'optimal_budget_lower', 'optimal_budget_upper']}\n    details['linear_programs'] = 0\n")
    source = source.replace('"risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD",',
        '"optimal_tail_loss": np.nan, "selected_tail_loss": np.nan, "optimal_budget_lower": np.nan, "optimal_budget_upper": np.nan, "linear_programs": 0,\n                "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD",')
    source = source.replace("            sd = np.full(2, np.nan)\n", "            sd = np.full(2, np.nan)\n            details = {key: np.nan for key in ['optimal_tail_loss', 'selected_tail_loss', 'optimal_budget_lower', 'optimal_budget_upper']}\n            details['linear_programs'] = 0\n")
    old = '''                    # 与标准差倒数归一化相同，避免显式计算极大倒数。
                    weights = np.array([sd[1], sd[0]]) / sd.sum()
                    successful, status = dates[t], "RISK_BUDGET_AVAILABLE"'''
    new = '''                    optimized = optimal_tail_budget(values, weights[0], confidence)
                    status = optimized["status"]
                    details = {key: optimized[key] for key in details}
                    if status == "TAIL_LOSS_BUDGET_AVAILABLE":
                        weights = np.array([optimized["panic_budget"], 1-optimized["panic_budget"]])
                        successful = dates[t]'''
    require(source.count(old) == 1, "尾部预算替换位置不唯一")
    source = source.replace(old, new)
    source = source.replace('"risk_status": status, "risk_update_scheduled": scheduled,', '**details, "risk_status": status, "risk_update_scheduled": scheduled,')
    (ROOT / "research/two_policy_tail_loss_inputs_v1.py").write_text(source, encoding="utf-8")
    tests = (ROOT / "tests/test_two_policy_risk_budget_v1.py").read_text(encoding="utf-8")
    tests = tests.replace("research.two_policy_risk_budget_inputs_v1", "research.two_policy_tail_loss_inputs_v1").replace("1/3", "0.")
    tests = tests.replace('"record_date": [dates[7]], "ex_date": [dates[8]], "payment_date": [dates[9]]', '"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[5]]')
    tests = tests.replace('data.loc[8:,', 'data.loc[4:,').replace('data.loc[8,', 'data.loc[4,')
    tests = tests.replace('ledger.date.eq(dates[7])', 'ledger.date.eq(dates[3])')
    tests = tests.replace('    assert np.isclose(ledger.dividend_recognized.sum(), held_on_record*.1)', '    assert held_on_record > 0\n    assert np.isclose(ledger.dividend_recognized.sum(), held_on_record*.1)')
    tests += '''

def test_tail_uses_fractional_mass_and_keeps_zero_cash_days():
    from research.two_policy_tail_loss_optimizer_v1 import empirical_tail_loss
    losses = np.arange(242, dtype=float)
    expected = (np.arange(230, 242).sum()+.1*229)/12.1
    assert np.isclose(empirical_tail_loss(losses), expected)
    losses = np.r_[.02, np.zeros(241)]
    assert np.isclose(empirical_tail_loss(losses), .02/12.1)


def test_tail_loss_may_be_negative_when_even_worst_days_are_gains():
    from research.two_policy_tail_loss_optimizer_v1 import empirical_tail_loss
    assert np.isclose(empirical_tail_loss(np.full(242, -.01)), -.01)


def test_perfect_opposite_risks_have_the_expected_zero_loss_budget():
    from research.two_policy_tail_loss_optimizer_v1 import optimal_tail_budget
    dates, returns, states = fixture()
    returns[:, 0] *= -1
    answer = optimal_tail_budget(returns, .5)
    assert answer['status'] == 'TAIL_LOSS_BUDGET_AVAILABLE'
    assert np.isclose(answer['panic_budget'], 1/3, atol=1e-7)
    assert abs(answer['selected_tail_loss']) < 1e-8


def test_identical_returns_choose_the_existing_budget_from_the_optimal_interval():
    from research.two_policy_tail_loss_optimizer_v1 import optimal_tail_budget
    dates, returns, states = fixture()
    returns[:, 0] = returns[:, 1]
    answer = optimal_tail_budget(returns, .37)
    assert answer['status'] == 'TAIL_LOSS_BUDGET_AVAILABLE'
    assert answer['panic_budget'] == .37
    assert np.isclose(answer['optimal_budget_lower'], 0.) and np.isclose(answer['optimal_budget_upper'], 1.)


def test_optimizer_failure_is_no_view_and_does_not_force_cash(monkeypatch):
    from types import SimpleNamespace
    from research import two_policy_tail_loss_optimizer_v1 as optimizer
    dates, returns, states = fixture()
    monkeypatch.setattr(optimizer, 'linprog', lambda *args, **kwargs: SimpleNamespace(success=False))
    answer = optimizer.optimal_tail_budget(returns, .37)
    assert answer['status'] == 'NO_VIEW_OPTIMIZER_FAILURE_KEEP_BUDGET'
    assert answer['panic_budget'] == .37 and np.isnan(answer['selected_tail_loss'])
'''
    (ROOT / "tests/test_two_policy_tail_loss_v1.py").write_text(tests, encoding="utf-8")
    runner = (ROOT / "research/two_policy_min_variance_v1.py").read_text(encoding="utf-8")
    runner = runner.replace("two_policy_min_variance", "two_policy_tail_loss").replace("TWO_POLICY_MIN_VARIANCE", "TWO_POLICY_TAIL_LOSS")
    runner = runner.replace("round=82, objective=\"MINIMIZE_TWO_REFERENCE_FULL_CALENDAR_SAMPLE_VARIANCE\",", "round=83, objective=\"MINIMIZE_TWO_REFERENCE_EMPIRICAL_CVAR\", tail_confidence=.95,")
    runner = runner.replace('degenerate_difference="NO_VIEW_KEEP_PREVIOUS_BUDGET",', 'optimizer_failure="NO_VIEW_KEEP_PREVIOUS_BUDGET", optimal_tie="PROJECT_PREVIOUS_BUDGET_ON_OPTIMAL_INTERVAL",')
    runner = runner.replace('"PROGRESS_ROUND81_SOURCE_FIX_COMPLETE"', '"PROGRESS_ROUND82_COMPLETE"')
    runner = runner.replace('P76 = ROOT /', 'P82 = ROOT / "reports/research/510300_two_policy_min_variance_v1"\nP76 = ROOT /', 1)
    runner = runner.replace('NAME = "两条原策略非负最小方差预算"', 'NAME = "两条原策略最差百分之五日损失预算"')
    runner = runner.replace('"TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算"}', '"TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算", "TWO_POLICY_MIN_VARIANCE": "第82轮最小方差预算"}')
    runner = runner.replace('(P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)', '(P82 if model == "TWO_POLICY_MIN_VARIANCE" else P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)')
    runner = runner.replace('ROOT / "research/two_policy_tail_loss_inputs_v1.py",', 'ROOT / "research/two_policy_tail_loss_inputs_v1.py", ROOT / "research/two_policy_tail_loss_optimizer_v1.py",')
    runner = runner.replace('cfg["risk_window"])', 'cfg["risk_window"], cfg["tail_confidence"])')
    runner = runner.replace('"evaluation_accounts": 12, "new_accounts_generated": 2, "reused_control_accounts": 10', '"evaluation_accounts": 14, "new_accounts_generated": 2, "reused_control_accounts": 12')
    runner = runner.replace('"earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 10', '"earlier_diagnostic_accounts": 14, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 12')
    runner = runner.replace("第82轮单项非负最小方差预算已冻结，尚无新策略收益。", "第83轮单项95%尾部损失预算已冻结，尚无新策略收益。")
    runner = runner.replace("新账户和五个保存对照完成", "新账户和六个保存对照完成")
    (ROOT / "research/two_policy_tail_loss_v1.py").write_text(runner, encoding="utf-8")
    (ROOT / "docs/510300_TWO_POLICY_TAIL_LOSS_V1.md").write_text('''# 第83轮：两条既有策略的尾部亏损预算

只登记95%置信度这一项条件风险价值预算，沿用原242日和月首时钟。第82轮主夏普1.137、较早0.520，未达目标；最小方差会同等计入盈利和亏损的大幅变动。本次只按合成参考收益最差5%日的平均损失分配资金，检验能否改善这种取舍，不声称一定提高夏普。

旧《下行风险预算》使用四技术因子预测未来20日不利路径后分仓，旧期权下行风险来自不同来源，其失败保留；有界检索未见这两个已保存原策略的同一尾部预算。没有调整第82轮协方差、窗口、上下限或收缩系数。

## 因子和预算的完整中文规则

每月首交易日完整收盘读取两条原基础费用策略截至当天的242个完整账户日净收益，现金日、已确认分红与真实费用均保留。不使用未来收益，不把较早前缀拼接另一评价时期作暖启动。

两条预算均不小于零且合计为一。任一预算下，每日参考损失是两条日净收益按预算加权后取负数。把242个日损失从大到小排列，最差5%对应12.1个观察：完整取最差12个，再取第13个的十分之一，合计除12.1。零损失和负损失即盈利都按原值保留；不能只留下亏损日，也不能把负尾部损失改成零。

预算选择目标是上述平均尾部损失最小。用线性规划同时求损失分界和超出分界的损失份额，采用[Rockafellar与Uryasev的原始方法](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)中的经验情景形式。本项目把两条既有交易规则的过去完整收益当作情景，并没有新增可交易证券。该方法可减少样本尾部损失，不保证下期损失、收益或夏普。

先求最小尾部损失，再保持该最小值求所有最优急跌预算的最小和最大值；此前预算位于这段区间就不改，位于左边取左端，位于右边取右端。这样并列时选最接近此前预算的解，没有新增换手惩罚、预期收益权重或任意选一端。每次有效窗口最多三个线性优化，分别记录状态、最小值、所选值和最优预算区间。

采用已安装SciPy的HiGHS求解器，原始及对偶可行性容差为十亿分之一，内点最优容差为百亿分之一。最优区间或原始损失复算不符、任一步求解失败时保留无观点及此前预算；损失核对容差为一亿分之一，属于数值检查，不是择时门槛。两条相同收益可以有完整最优区间，保留此前预算。经验尾部质量接近整数至万亿分之一时按整数处理，避免浮点运算误判。

两段各自初始预算为50%／50%。没有242完整日、任一参考窗口缺失或任一日收益波动为零时，与第76轮一致，保留无观点及此前预算。月首之间继续最近预算，不临时按最新市场涨跌重新优化；终点开盘清仓收益不作新收盘判断。

## 进入、持有、退出与重新进入

原急跌与学习策略全部因子和参考进入、持有、退出、冷却及重新进入状态沿用第76轮中文协议；不使用第81轮分批。每日股票目标为急跌预算乘其原持有状态，加学习预算乘其原持有状态。都空仓则目标零、下一开盘全部卖出；只有一条持有则投入其预算；两条都持有为全额。预算为零的分支即使有持有信号也不投入。缺失任一参考状态则没有新目标、保持真实股数。

实际空仓且目标大于零时，以自身完整净值和收盘价计算100份整手目标，下一开盘按可支用现金及真实费用成交。非零目标与实际比例相差至少10个百分点才重新调整；零目标全卖不受此限制。月度预算可以导致额外减仓、退出或重新进入；组合没有额外冷却。原参考内部规则照旧，组合受阻后下一收盘按最新目标重算请求。原模型使用原参考自身状态，合并账户的盈亏不反灌模型。

两费用共用基础参考信号及预算，各自运行真实账户；不将参考日收益按预算加权冒充实际业绩。510300与人民币现金、20万元、242日年化、现金及无风险收益零、两档原费用、100份整手、T+1、方向涨跌停、分红登记及到账和全部日期保持。主2020年1月2日至2026年8月14日开盘1604日，较早2015年1月5日至2019年12月31日开盘1219日。

12项必要测试通过后冻结并跑四个新账户；另复用原两单策略、各半、买入持有、第76及82预算六类24份对照，主和较早各14记录。没有新参考策略、学习模型或下载。只核对月度尾部目标、时钟与四新账户。失败后不扫90%、99%、窗口、尾部质量、预算上限或附加收益项救回；不做GPT数值包，EPS及慢来源继续暂停。
''', encoding="utf-8")
    (ROOT / "reports/research/510300_two_policy_tail_loss_v1").mkdir(parents=True, exist_ok=True)
    print("第83轮单项尾部损失预算、中文规则与十二项必要测试已建立。", flush=True)


if __name__ == "__main__":
    main()
