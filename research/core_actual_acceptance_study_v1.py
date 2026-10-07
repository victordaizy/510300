"""唯一固定原A进入后持有实验；先核对冻结，再运行全部完整账户。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import core_actual_acceptance_inputs_v1 as inputs
from research import core_actual_acceptance_account_v1 as execution
from research import support_price_acceptance_study_v1 as previous

parent, controls = previous.parent, previous.controls
ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_core_actual_acceptance_v1"
PRIOR = ROOT / "reports/research/510300_actual_holding_information_v1/implementation_v1_0_1"
DAILY = previous.OUT / "results/全部3488支持信息与价格接受状态.parquet"
RECEIPTS = previous.OUT / "results/全部来源首次原ETF可知_宣布与确认分开.parquet"
OBSERVED = OUT / "results/全部3488原A实际持有用途_明确纳秒钟.parquet"
KEYS = previous.KEYS
REGISTER, RESULT = "TECH.R235", "TECH.R236"
PERIODS, COSTS = previous.PERIODS, previous.COSTS
CONTROL_A, CONTROL_STAGE = previous.CONTROL_A, previous.CONTROL_STAGE
CONTROLS = (CONTROL_A, CONTROL_STAGE, *inputs.POLICIES[1:])
NAMES = {CONTROL_A: "原A", CONTROL_STAGE: "原阶段", **inputs.NAMES}
ACCOUNT_TABLES = previous.ACCOUNT_TABLES
read, write, digest = previous.read, previous.write, previous.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".parquet").exists():
        raise RuntimeError("本轮结果已存在，禁止重复覆盖："+name)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    base, dividends, risks, parents = parent.original.load()
    daily, receipts = pd.read_parquet(DAILY), pd.read_parquet(RECEIPTS)
    if len(daily) != 3488 or len(receipts) != 65 or not daily.symbol.eq("510300.SH").all():
        raise ValueError("固定原观察日或来源节点改变。")
    np.testing.assert_array_equal(pd.to_datetime(base.date).astype("datetime64[ns]"), pd.to_datetime(daily.date).astype("datetime64[ns]"))
    for name in ("open", "high", "low", "close", "volume", "dividend", "cash_shift", "ac", "atr14"):
        np.testing.assert_array_equal(base[name].to_numpy(), daily[name].to_numpy())
    return daily, receipts, dividends, risks, parents


def exact_saved_table(actual, expected):
    """声明日期列逐值一致后统一临时纳秒；其他列完整精确，不改保存表。"""
    a, b = actual[expected.columns].copy(), expected.copy()
    declared = {"date", "origin", "execution_date", "entry_origin", "entry_date", "exit_date", "setup_date"}
    for name in declared.intersection(expected.columns):
        left = pd.to_datetime(a[name]).astype("datetime64[ns]")
        right = pd.to_datetime(b[name]).astype("datetime64[ns]")
        np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())
        a[name], b[name] = left, right
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def preflight():
    if (OUT / "control_and_prefix_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("原A适配器和时序核对已完成，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if tests["passed"] != 11 or tests["exit_code"] != 0:
        raise ValueError("十一项必要时序、时间格式及现金测试未通过。")
    for key, path in (("inputs_sha256", Path(inputs.__file__)), ("account_sha256", Path(execution.__file__)),
                      ("tests_sha256", ROOT / "tests/test_core_actual_acceptance_v1.py")):
        if tests[key] != digest(path):
            raise ValueError("测试版本与当前实现不同。")
    daily, receipts, dividends, risks, parents = load()
    observed = inputs.observations(daily, receipts)
    table(OBSERVED.stem, observed)
    keys = pd.read_parquet(KEYS)
    prefixes = []
    for cut in keys.date:
        truncated = inputs.observations(daily.loc[daily.date.le(cut)], receipts)
        pd.testing.assert_frame_equal(truncated, observed.loc[observed.date.le(cut)].reset_index(drop=True), check_exact=True)
        period = "2015_2019" if pd.Timestamp(cut).year < 2020 else "2020_2026"
        p = parents[period]
        e = inputs.parent_episodes(p)
        ep = inputs.parent_episodes(p.loc[p.origin.le(cut)])
        pd.testing.assert_frame_equal(ep, e.loc[e.origin.le(cut)].reset_index(drop=True), check_exact=True)
        prefixes.append({"cut": str(pd.Timestamp(cut).date()), "all_prior_source_rows_exact": True, "parent_episode_prefix_exact": True})
    checks = []
    for period, (start, end) in PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            actual = execution.account(local, dividends, parents[period], risks, inputs.BASELINE, cost, start)
            saved = controls.saved_account(period, cost, CONTROL_A)
            for name in ACCOUNT_TABLES[:3]:
                exact_saved_table(actual[name], saved[name])
            checks.append({"period": period, "cost": cost, "original_A_daily_orders_trades_exact": True, **parent.verify_account(actual)})
            print(f"{period}/{cost}原A关闭新动作后逐日、订单、周期精确复现。", flush=True)
    if len(keys) != 19 or len(checks) != 4:
        raise ValueError("关键日或原A对照不完整。")
    old = read(previous.OUT / "implementation_v1_0_1/summary.json")
    saved_metrics = []
    for period in PERIODS:
        for cost in COSTS:
            for policy in (CONTROL_A, CONTROL_STAGE):
                stats, _ = controls.statistics(controls.saved_account(period, cost, policy), period, cost, policy)
                reference = next(x for x in old["metrics"] if (x["period"], x["cost"], x["policy"]) == (period, cost, policy))
                for field in ("net_cagr", "net_sharpe", "max_drawdown", "p_times_b", "ending_equity", "total_commission", "total_slippage"):
                    np.testing.assert_allclose(stats[field], reference[field], atol=1e-12, rtol=0, equal_nan=True)
                saved_metrics.append({"period": period, "cost": cost, "policy": policy, "metrics_exact": True})
    write(OUT / "control_and_prefix_preflight.json", {"at": parent.original.now(), "original_A_adapter_replays": 4,
        "adapter_checks": checks, "all_original_prefixes": prefixes, "saved_metrics_checked": saved_metrics,
        "new_candidate_full_accounts": 0, "new_fits": 0, "new_training_labels": 0,
        "comparison_date_resolution_only": "DECLARED_NAIVE_COLUMNS_TO_NS_AFTER_VALUE_EQUALITY"})
    print("原19整段来源/原目标段前缀及8保存对照通过；新增持有金融尚未运行。", flush=True)


def source_paths():
    paths = [Path(__file__), Path(inputs.__file__), Path(execution.__file__), ROOT / "tests/test_core_actual_acceptance_v1.py",
        ROOT / "docs/510300_CORE_ACTUAL_ACCEPTANCE_V1.md", Path(inputs.description.__file__),
        Path(parent.original.__file__), Path(parent.original.learning.__file__), Path(parent.original_account.__file__), Path(parent.measurements.__file__),
        Path(parent.budgets.__file__), ROOT / "research/point_account_nr7_complement_v1.py",
        ROOT / "research/point_directional_confirmation_study_v1.py", Path(controls.__file__), DAILY, RECEIPTS, KEYS,
        PRIOR / "summary.json", PRIOR / "next_core_actual_acceptance_carry_proposal.json",
        previous.OUT / "implementation_v1_0_1/summary.json", previous.OUT / "protocol.json",
        parent.original.CURRENT / "inputs/candidate_prices.parquet", parent.original.WEIGHT / "inputs/dividends.csv", parent.original.WEIGHT / "inputs/risks.parquet",
        parent.original.WEIGHT / "inputs/parent_signals.parquet", parent.original.WEIGHT / "inputs/earlier_signals.parquet",
        OUT / "tests_receipt.json", OUT / "control_and_prefix_preflight.json",
        OBSERVED, OUT / "pre_freeze_first_prefix_failure.json", OUT / "tests_first_pre_freeze_receipt.json"]
    for period in PERIODS:
        for cost in COSTS:
            a = parent.original.CONTROL / period / cost / CONTROL_A
            stage = parent.OUT / "accounts" / period / cost / parent.stage.POLICIES[0]
            paths.extend(a / f"{name}.parquet" for name in ACCOUNT_TABLES[:3])
            paths.extend(stage / f"{name}.parquet" for name in ACCOUNT_TABLES)
            paths.append(stage / "terminal.json")
    return list(dict.fromkeys(paths))


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮已固定，不重新登记。")
    pre = read(OUT / "control_and_prefix_preflight.json")
    if len(pre["adapter_checks"]) != 4 or len(pre["all_original_prefixes"]) != 19 or len(pre["saved_metrics_checked"]) != 8:
        raise ValueError("完整原A适配及来源核对未完成。")
    state = read(ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json")
    if state["latest_technical_decision"] != "TECH.R234" or state["latest_actual_financial_decision"] != "TECH.R232":
        raise ValueError("项目已推进，不能在旧状态登记。")
    protocol = {"at": parent.original.now(), "study": "510300_CORE_ACTUAL_ACCEPTANCE_V1", "registration": REGISTER,
        "decision": RESULT, "primary": inputs.POLICIES[0], "policies": list(inputs.POLICIES), "controls": list(CONTROLS),
        "question": "保持原A进入和权重源，实际进入后的接受保护、新完整周和新增信息是否提高完整净CAGR和Sharpe？",
        "known_before_freeze": "R234全部路径与R232完整失败已知；截至Sep30全部历史开发，后验组不是策略胜率。",
        "rule_card": "docs/510300_CORE_ACTUAL_ACCEPTANCE_V1.md", "rule_card_sha256": digest(ROOT / "docs/510300_CORE_ACTUAL_ACCEPTANCE_V1.md"),
        "entry": "原保存源、0/未知、50%上限、10pp调整带；不新增入场门。不同持有和现金路径可改变后续实际进入。",
        "acceptance": "进入日收盘确定现金平移高低，严格后来的收盘>该高且日柱>0才一次启用该低保护，未接受破低不额外退出。",
        "carry": "当前接受且上一完整周全部交易日在买日起、低>进入日低/收盘>进入日高；主另需已知行业、up_volume_balance5>0、industry_positive5_fraction>.5、leader_mean_return5>0。",
        "trailing": "晋级后保护取原位和后来各已知完整买后周低累计最大，触及次开退；CARRY原目标0/未知时风险减仓不补买，正目标仍原权重。",
        "macro_veto": "原买入09:30后新公布已知PMI订单环比负或买后公布并新观察操作上调，永久撤销本周期CARRY资格，原源动作恢复、保护不降；旧负/未知不撤销。",
        "consumption": "额外结构真实清仓消费第一次退出请求对应的原正目标身份；原明确0后的新正才重启，未知不重启，受阻理由锁定。",
        "account": read(previous.OUT / "protocol.json")["account"], "core_additions": "CORE以及CARRY原正目标时允许原权重补买，沿原现金和风险；不能复用R232不补买权限替代原A。",
        "periods": PERIODS, "costs": read(previous.OUT / "protocol.json")["costs"], "new_candidate_accounts": 12,
        "original_A_adapter_replays": 4, "saved_controls_reused": 8, "necessary_tests": 11, "all_prefix_checks": 19,
        "parameter_grid": False, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "economic_gate": "四场景主净CAGR/Sharpe正并高于四对照，DD<=10%，完成实际净pB>1且标准期望正，次数软目标。",
        "stability": "四场景四对照原20/252循环块各2000、种子510300154，CAGR/Sharpe增量95%下界全部正。",
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "search_selection_correction": "NOT_COMPUTED", "complete_policy_coverage": "NOT_ESTABLISHED",
        "no_rescue": "一次金融，失败不改本动作/符号/窗口/来源/风险/费用/时期，不救R232或拼接赢家。原E03保持。",
        "sources": [{"path": relative(p), "sha256": digest(p)} for p in source_paths()]}
    write(OUT / "protocol.json", protocol)
    print("R235唯一实际接受保护/新增延续用途已固定；12完整新账户尚未运行。", flush=True)


def check_timing(result, local):
    trades, evidence, decisions = result["trades"], result["holding_evidence"], result["decisions"]
    for trade in trades.itertuples():
        if pd.notna(trade.first_price_acceptance) and not trade.first_price_acceptance > trade.entry_date:
            raise AssertionError("接受保护未严格晚于真实进入日。")
        if pd.notna(trade.promotion_date):
            row = local.loc[local.date.eq(trade.promotion_date)].iloc[0]
            if not row.complete_week_first >= trade.entry_date or not row.complete_week_last < trade.promotion_date:
                raise AssertionError("延续用了买前周或未来周。")
    if len(evidence) and not evidence.origin.ge(evidence.entry_date).all():
        raise AssertionError("持有信息早于真实进入。")
    if len(result["orders"]) and not result["orders"].origin.lt(result["orders"].date).all():
        raise AssertionError("实际订单没有先前收盘。")
    blocked = decisions.loc[decisions.reason.eq("FORCED_EXIT_CONSUMED_SAME_PARENT_EPISODE")]
    if len(blocked) and (blocked.desired_shares.ne(0).any() or not blocked.parent_episode_id.eq(blocked.consumed_parent_episode_id).all()):
        raise AssertionError("已消费原目标段重复进入。")
    return {"order_and_actual_entry_week_timing_passed": True, "cycles": len(trades),
        "accepted_cycles": int(trades.first_price_acceptance.notna().sum()) if len(trades) else 0,
        "promoted_cycles": int(trades.promotion_date.notna().sum()) if len(trades) else 0,
        "revoked_cycles": int(trades.extension_revoked.sum()) if len(trades) else 0,
        "consumed_episode_flat_decisions": len(blocked)}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("完整金融已开始，不重复。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定来源改变："+item["path"])
    write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts_planned": 12, "new_fits": 0})
    observed = pd.read_parquet(OBSERVED)
    _, dividends, risks, parents = parent.original.load()
    accounts, records, annual, checks, timings = {}, [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            for policy in (CONTROL_A, CONTROL_STAGE, *inputs.POLICIES):
                if policy in inputs.POLICIES:
                    result = execution.account(local, dividends, parents[period], risks, policy, cost, start)
                    checks.append({"period": period, "cost": cost, "policy": policy, **parent.verify_account(result)})
                    timings.append({"period": period, "cost": cost, "policy": policy, **check_timing(result, local)})
                    folder = OUT / "accounts" / period / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    for name in (*ACCOUNT_TABLES, "holding_evidence"):
                        result[name].to_parquet(folder / f"{name}.parquet", index=False)
                    write(folder / "terminal.json", result["terminal"])
                else:
                    result = controls.saved_account(period, cost, policy)
                accounts[(period, cost, policy)] = result
                stats, yearly = controls.statistics(result, period, cost, policy)
                records.append({"period": period, "cost": cost, "policy": policy, **stats})
                annual.extend(yearly)
                if policy in inputs.POLICIES:
                    print(f"{period}/{cost}/{NAMES[policy]}：净年化{stats['net_cagr']:.4%}、净夏普{stats['net_sharpe']:.6f}、完成{stats['completed_cycles']}、pB{stats['p_times_b']:.6f}。", flush=True)
    frame = pd.DataFrame(records)
    table("二十完整账户_原A实际接受保护与新增延续", frame)
    table("全部逐年实际次数与净收益", pd.DataFrame(annual))
    table("十二完整账户现金复算", pd.DataFrame(checks))
    table("十二完整账户接受周线来源与消费时序", pd.DataFrame(timings))
    write(OUT / "financial_accounts_completed.json", {"at": parent.original.now(), "new_accounts_completed": 12,
        "saved_controls_reused": 8, "metrics": frame.to_dict("records"), "all_accounts_saved_before_intervals": True})
    comparisons, gates, interval_rows = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            group = frame.loc[frame.period.eq(period) & frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[inputs.POLICIES[0]]
            conditions = {"positive_cagr_sharpe": bool(primary.net_cagr > 0 and primary.net_sharpe > 0),
                "drawdown_within_10pct": bool(primary.max_drawdown <= .1), "actual_pB_above_1": bool(primary.p_times_b > 1),
                "actual_standard_EV_positive": bool(primary.standard_expectancy_loss_units > 0)}
            for control in CONTROLS:
                reference = group.loc[control]
                conditions["beats_"+control] = bool(primary.net_cagr > reference.net_cagr and primary.net_sharpe > reference.net_sharpe)
                a = accounts[(period, cost, control)]["daily"]
                b = accounts[(period, cost, inputs.POLICIES[0])]["daily"]
                np.testing.assert_array_equal(pd.to_datetime(a.date).astype("datetime64[ns]").to_numpy(), pd.to_datetime(b.date).astype("datetime64[ns]").to_numpy())
                intervals = parent.intervals(a.net_return.to_numpy(float), b.net_return.to_numpy(float))
                comparison = {"period": period, "cost": cost, "control": control,
                    "cagr_delta": primary.net_cagr-reference.net_cagr, "sharpe_delta": primary.net_sharpe-reference.net_sharpe, "intervals": intervals}
                comparisons.append(comparison)
                for item in intervals:
                    interval_rows.append({"period": period, "cost": cost, "control": control, "block": item["block"], "resamples": item["resamples"],
                        "cagr_delta": comparison["cagr_delta"], "sharpe_delta": comparison["sharpe_delta"],
                        "cagr_delta_lower": item["cagr_delta_95"][0], "cagr_delta_upper": item["cagr_delta_95"][1],
                        "sharpe_delta_lower": item["sharpe_delta_95"][0], "sharpe_delta_upper": item["sharpe_delta_95"][1]})
            gates.append({"period": period, "cost": cost, **conditions, "economic_passed": all(conditions.values())})
            print(f"{period}/{cost}四对照两尺度完整净增量区间完成。", flush=True)
    table("全部四场景四对照两尺度净增量区间", pd.DataFrame(interval_rows))
    table("全部四场景经济门", pd.DataFrame(gates))
    stable = all(item["cagr_delta_95"][0] > 0 and item["sharpe_delta_95"][0] > 0 for comparison in comparisons for item in comparison["intervals"])
    passed = all(x["economic_passed"] for x in gates)
    write(OUT / "summary.json", {"at": parent.original.now(), "registration": REGISTER, "decision": RESULT,
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_CORE_ACTUAL_ACCEPTANCE_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "timing_checks": timings,
        "new_accounts": 12, "saved_controls_reused": 8, "original_A_adapter_replays": 4,
        "saved_account_checks": len(checks), "necessary_tests_passed": 11, "all_prefix_checks": 19,
        "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "search_selection_correction": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False})
    print("R236完整原A接受保护与新增延续终态；不按结果修改本配置。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="原A实际进入后的接受保护与新增信息延续")
    parser.add_argument("action", choices=("preflight", "freeze", "run"))
    action = parser.parse_args().action
    {"preflight": preflight, "freeze": freeze, "run": run}[action]()


if __name__ == "__main__":
    main()
