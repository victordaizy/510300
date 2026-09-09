"""以已验证的账户运行程序建立单项累计净值预算。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / "research/two_policy_wealth_budget_v1.py"
    require(not dest.exists(), "第84轮来源已建立")
    text = (ROOT / "research/two_policy_min_variance_v1.py").read_text(encoding="utf-8")
    text = text.replace("two_policy_min_variance", "two_policy_wealth_budget").replace("TWO_POLICY_MIN_VARIANCE", "TWO_POLICY_WEALTH_BUDGET")
    text = text.replace('P76 = ROOT /', 'P82 = ROOT / "reports/research/510300_two_policy_min_variance_v1"\nP76 = ROOT /', 1)
    text = text.replace('NAME = "两条原策略非负最小方差预算"', 'NAME = "两条原策略累计净值自然分配预算"')
    text = text.replace('"TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算"}', '"TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算", "TWO_POLICY_MIN_VARIANCE": "第82轮最小方差预算"}')
    text = text.replace('(P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)', '(P82 if model == "TWO_POLICY_MIN_VARIANCE" else P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)')
    text = text.replace('round=82, objective="MINIMIZE_TWO_REFERENCE_FULL_CALENDAR_SAMPLE_VARIANCE",', 'round=84, objective="RELATIVE_CUMULATIVE_REFERENCE_EQUITY",')
    text = text.replace('risk_window=242, risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN",', 'budget_clock="EACH_COMPLETE_CLOSE_NEXT_OPEN",')
    text = text.replace('missing_or_zero_risk="NO_VIEW_KEEP_PREVIOUS_BUDGET", degenerate_difference="NO_VIEW_KEEP_PREVIOUS_BUDGET",', 'missing_or_invalid_equity="NO_VIEW_KEEP_PREVIOUS_BUDGET",')
    text = text.replace('"PROGRESS_ROUND81_SOURCE_FIX_COMPLETE"', '"PROGRESS_ROUND83_COMPLETE"')
    text = text.replace("reference_returns", "reference_equities").replace("ref.net_return.to_numpy(float)", "ref.equity.to_numpy(float)")
    text = text.replace('mapped, first, cfg["risk_window"])', 'mapped, first)')
    text = text.replace('"panic_reference_return"', '"panic_reference_equity"').replace('"learned_reference_return"', '"learned_reference_equity"')
    text = text.replace("factors.risk_update_scheduled", "factors.budget_update_scheduled")
    text = text.replace('"risk_update_records.csv"', '"budget_update_records.csv"').replace('"risk_update_count"', '"budget_update_count"')
    text = text.replace('"evaluation_accounts": 12, "new_accounts_generated": 2, "reused_control_accounts": 10', '"evaluation_accounts": 14, "new_accounts_generated": 2, "reused_control_accounts": 12')
    text = text.replace('"earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 10', '"earlier_diagnostic_accounts": 14, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 12')
    text = text.replace("第82轮单项非负最小方差预算已冻结，尚无新策略收益。", "第84轮单项累计净值预算已冻结，尚无新策略收益。")
    text = text.replace("新账户和五个保存对照完成", "新账户和六个保存对照完成")
    dest.write_text(text, encoding="utf-8")
    (ROOT / "reports/research/510300_two_policy_wealth_budget_v1").mkdir(parents=True, exist_ok=True)
    print("第84轮单项累计净值预算运行来源已建立。", flush=True)


if __name__ == "__main__":
    main()
