"""以新来源登记直接承担股票风险日的条件风险，旧第76轮内容保持。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "research/active_risk_budget_inputs_v1.py"
    runner = ROOT / "research/active_risk_budget_v1.py"
    if target.exists() or runner.exists():
        raise ValueError("第77轮来源已经存在，不覆盖")
    text = (ROOT / "research/two_policy_risk_budget_inputs_v1.py").read_text(encoding="utf-8")
    text = text.replace("def budget_frame(dates, reference_returns, expert_states, first, window=242):", "def active_budget_frame(dates, reference_returns, expert_states, first, active_exposure, window=242):")
    text = text.replace('    states = np.asarray(expert_states, dtype=float)', '    states = np.asarray(expert_states, dtype=float)\n    active = np.asarray(active_exposure, dtype=float)\n    require(active.shape == states.shape and (np.isnan(active) | np.isin(active, [0., 1.])).all(), "直接股票风险日标记无效")')
    text = text.replace('    count = 0', '    count = 0\n    selected_counts = [0, 0]', 1)
    text = text.replace('"panic_state": states[t, 0], "learned_state": states[t, 1], "target": np.nan}', '"panic_state": states[t, 0], "learned_state": states[t, 1], "target": np.nan, "panic_active_days": 0, "learned_active_days": 0}')
    text = text.replace('            values = returns[t-count+1:t+1]', '            values = returns[t-count+1:t+1]\n            active_window = active[t-count+1:t+1]\n            selected_counts = [int(np.sum(active_window[:, k] == 1)) for k in range(2)]')
    text = text.replace('            elif not np.isfinite(values).all():', '            elif not np.isfinite(values).all() or not np.isfinite(active_window).all():')
    text = text.replace('            else:\n                sd = np.std(values, axis=0, ddof=1)', '            elif min(selected_counts) < 2:\n                status = "NO_VIEW_TOO_FEW_ACTIVE_DAYS_KEEP_BUDGET"\n            else:\n                sd = np.array([np.std(values[active_window[:, k] == 1, k], ddof=1) for k in range(2)])')
    text = text.replace('"risk_window_start": start, "risk_window_observations": count,', '"risk_window_start": start, "risk_window_observations": count, "panic_active_days": selected_counts[0], "learned_active_days": selected_counts[1],')
    target.write_text(text, encoding="utf-8")
    text = (ROOT / "research/two_policy_risk_budget_v1.py").read_text(encoding="utf-8")
    text = text.replace("from research.two_policy_risk_budget_inputs_v1 import budget_frame", "from research.active_risk_budget_inputs_v1 import active_budget_frame")
    text = text.replace("510300_two_policy_risk_budget_v1", "510300_active_risk_budget_v1")
    text = text.replace('PRIMARY = "TWO_POLICY_RISK_BUDGET"', 'PRIMARY = "ACTIVE_RISK_BUDGET"\nP76 = ROOT / "reports/research/510300_two_policy_risk_budget_v1"')
    text = text.replace('NAME = "两条原策略按已实现风险分配预算"', 'NAME = "按实际承担股票风险日估计两策略风险"')
    text = text.replace('CONTROLS = {', 'CONTROLS = {"TWO_POLICY_RISK_BUDGET": "第76轮完整日风险预算", ', 1)
    text = text.replace('study_id="510300_TWO_POLICY_RISK_BUDGET_V1", round=76', 'study_id="510300_ACTIVE_RISK_BUDGET_V1", round=77')
    text = text.replace('"docs/510300_TWO_POLICY_RISK_BUDGET_V1.md"', '"docs/510300_ACTIVE_RISK_BUDGET_V1.md"')
    text = text.replace('new_model_fits=0, new_reference_accounts=0, previous_goal_turn_classification="PROGRESS_ROUND75_COMPLETED_AND_DELIVERED",', 'new_model_fits=0, new_reference_accounts=0, risk_condition="OPENING_OR_CLOSING_ACTUAL_SHARES_POSITIVE", minimum_active_days=2,\n        previous_goal_turn_classification="PROGRESS_ROUND76_COMPLETED_AND_DELIVERED",')
    text = text.replace('"research/two_policy_risk_budget_inputs_v1.py"', '"research/active_risk_budget_inputs_v1.py"')
    text = text.replace('"tests/test_two_policy_risk_budget_v1.py"', '"tests/test_active_risk_budget_v1.py"')
    text = text.replace('paths.extend(P46 / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)', 'paths.extend((P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)')
    text = text.replace('        reference_returns = np.full((len(frame), 2), np.nan)', '        reference_returns = np.full((len(frame), 2), np.nan)\n        active_exposure = np.full((len(frame), 2), np.nan)')
    text = text.replace('            reference_returns[first:, column] = ref.net_return.to_numpy(float)', '            reference_returns[first:, column] = ref.net_return.to_numpy(float)\n            active_exposure[first:, column] = (ref.shares_before.gt(0) | ref.shares_after.gt(0)).to_numpy(float)')
    text = text.replace('factors = budget_frame(frame.date, reference_returns, mapped, first, cfg["risk_window"])', 'factors = active_budget_frame(frame.date, reference_returns, mapped, first, active_exposure, cfg["risk_window"])\n        factors["panic_active_exposure"], factors["learned_active_exposure"] = active_exposure[:, 0], active_exposure[:, 1]')
    text = text.replace('saved = pd.read_parquet(P46 / period / cost_id / f"{model}_ledger.parquet")', 'saved = pd.read_parquet((P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost_id / f"{model}_ledger.parquet")')
    text = text.replace('"TWO_POLICY_RISK_BUDGET_ACCOUNTS_COMPLETE"', '"ACTIVE_RISK_BUDGET_ACCOUNTS_COMPLETE"')
    text = text.replace('"evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8', '"evaluation_accounts": 12, "new_accounts_generated": 2, "reused_control_accounts": 10')
    text = text.replace('"earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8', '"earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 10')
    text = text.replace('第76轮一个242日月度风险预算设置已冻结', '第77轮一个242日窗口内直接股票风险日条件风险设置已冻结')
    text = text.replace('和四个保存对照完成', '和五个保存对照完成')
    runner.write_text(text, encoding="utf-8")
    print("第77轮条件风险新来源生成，原76文件未修改。")


if __name__ == "__main__":
    main()
