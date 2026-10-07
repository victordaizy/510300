"""一次完整比较支持信息、固定锚价格接受和确认持有，保存全部账户与失败。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_stage_policy_study_v1 as parent
from research import broker_cohort_failure_study_v1 as controls
from research import support_price_acceptance_inputs_v1 as inputs
from research import support_price_acceptance_account_v1 as execution
from research import original_policy_announcement_trace_v1 as source
from research import original_policy_trace_format_v1_0_1 as source_format

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_support_price_acceptance_v1"
REGISTRATION, RESULT = "TECH.R231", "TECH.R232"
PERIODS, COSTS = parent.PERIODS, parent.COSTS
CONTROL_A, CONTROL_STAGE = controls.CONTROL_A, controls.CONTROL_STAGE
CONTROLS = (CONTROL_A, CONTROL_STAGE, *inputs.POLICIES[1:])
NAMES = {CONTROL_A: "原A", CONTROL_STAGE: "原阶段", **inputs.NAMES}
ACCOUNT_TABLES = controls.ACCOUNT_TABLES
NODES = source_format.OUT / "results/全部来源节点_旧49与原六新源逐值保持.parquet"
RATES = source.OUT / "results/全部25利率操作原文_改变与首发不混同.parquet"
KEYS = source_format.OUT / "results/全部19关键日_原17与两假启动.parquet"
PREVIOUS_FINANCIAL = ROOT / "reports/research/510300_industry_leader_failure_v1/summary.json"
read, write, digest = parent.read, parent.write, parent.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load_initial():
    _, observed, dividends, risks, _ = parent.load()
    daily = pd.read_parquet(source.DAILY)
    daily["date"] = pd.to_datetime(daily.date).astype("datetime64[ns]")
    if len(daily) != 3488 or not daily.symbol.eq("510300.SH").all():
        raise ValueError("原日线数量或研究对象改变。")
    for field in ["date", "open", "high", "low", "close", "volume", "dividend", "cash_shift", "ac", "atr14", "daily_hist", "ema20"]:
        np.testing.assert_array_equal(daily[field].to_numpy(), observed[field].to_numpy())
    nodes, rates = pd.read_parquet(NODES), pd.read_parquet(RATES)
    if len(nodes) != 65 or len(rates) != 25:
        raise ValueError("固定原公告及操作母集改变。")
    records = inputs.messages(nodes, rates)
    return daily, dividends, risks, records


def base_source_paths():
    paths = [Path(__file__), Path(inputs.__file__), Path(execution.__file__),
        ROOT / "tests/test_support_price_acceptance_v1.py", ROOT / "docs/510300_SUPPORT_PRICE_ACCEPTANCE_V1.md",
        Path(parent.__file__), Path(parent.stage.__file__), Path(controls.__file__),
        Path(parent.measurements.__file__), Path(parent.budgets.__file__),
        ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
        source.DAILY, NODES, RATES, KEYS, source_format.OUT / "summary.json", source_format.OUT / "protocol.json",
        source_format.OUT / "next_support_acceptance_state_proposal.json", PREVIOUS_FINANCIAL,
        parent.original.WEIGHT / "inputs/dividends.csv", parent.original.WEIGHT / "inputs/risks.parquet",
        OUT / "tests_receipt.json", OUT / "control_and_prefix_preflight.json",
        OUT / "results/全部3488支持信息与价格接受状态.parquet",
        OUT / "results/全部65来源用途资格_操作前值与非激活保留.parquet",
        OUT / "results/全部来源首次原ETF可知_宣布与确认分开.parquet"]
    for period in PERIODS:
        for cost in COSTS:
            a = parent.original.CONTROL / period / cost / CONTROL_A
            stage = parent.OUT / "accounts" / period / cost / parent.stage.POLICIES[0]
            paths.extend(a / f"{name}.parquet" for name in ACCOUNT_TABLES[:3])
            paths.extend(stage / f"{name}.parquet" for name in ACCOUNT_TABLES)
            paths.append(stage / "terminal.json")
    return list(dict.fromkeys(paths))


def preflight():
    if (OUT / "control_and_prefix_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("支持接受前核对已完成或已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if (tests["passed"] != 8 or tests["exit_code"] != 0 or tests["inputs_sha256"] != digest(Path(inputs.__file__))
            or tests["account_sha256"] != digest(Path(execution.__file__))):
        raise ValueError("八必要时序与执行测试未通过或版本改变。")
    daily, _, _, records = load_initial()
    observed, receipts = inputs.observations(daily, records)
    table("全部65来源用途资格_操作前值与非激活保留", records)
    table("全部来源首次原ETF可知_宣布与确认分开", receipts)
    table("全部3488支持信息与价格接受状态", observed)
    keys = pd.read_parquet(KEYS)
    prefix_checks = []
    for cut in keys.date:
        prefix, _ = inputs.observations(daily.loc[daily.date.le(cut)], records)
        pd.testing.assert_frame_equal(prefix, observed.loc[observed.date.le(cut)].reset_index(drop=True), check_exact=True)
        prefix_checks.append({"cut": str(pd.Timestamp(cut).date()), "all_prior_rows_exact": True})
    old = read(PREVIOUS_FINANCIAL)
    checks = []
    for period in PERIODS:
        for cost in COSTS:
            for policy in [CONTROL_A, CONTROL_STAGE]:
                result = controls.saved_account(period, cost, policy)
                stats, _ = controls.statistics(result, period, cost, policy)
                reference = next(row for row in old["metrics"] if row["period"] == period and row["cost"] == cost and row["policy"] == policy)
                for field in ["net_cagr", "net_sharpe", "max_drawdown", "p_times_b", "ending_equity", "total_commission", "total_slippage"]:
                    np.testing.assert_allclose(stats[field], reference[field], atol=1e-12, rtol=0, equal_nan=True)
                checks.append({"period": period, "cost": cost, "policy": policy, "saved_metrics_match_previous": True,
                               **parent.verify_account(result)})
    if len(keys) != 19 or len(prefix_checks) != 19 or len(checks) != 8:
        raise ValueError("原关键日或保存对照数量不完整。")
    write(OUT / "control_and_prefix_preflight.json", {
        "at": parent.original.now(), "original_saved_controls_checked": 8, "new_original_policy_replays": 0,
        "checks": checks, "prefix_checks": prefix_checks, "new_candidate_accounts": 0,
        "support_activation_days": int(observed.support_activation_allowed.sum()),
        "support_price_accepted_events": int(observed.support_entry_event.sum()),
        "price_only_accepted_events": int(observed.price_entry_event.sum()),
        "all_observed_dates": len(observed), "new_fits": 0, "new_training_labels": 0,
    })
    print("支持信息、价格锚和完整周19整段前缀通过；8保存对照复算一致，候选账户尚未运行。", flush=True)


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("支持接受完整用途已固定，不覆盖。")
    pre, tests = read(OUT / "control_and_prefix_preflight.json"), read(OUT / "tests_receipt.json")
    if pre["original_saved_controls_checked"] != 8 or len(pre["prefix_checks"]) != 19 or tests["passed"] != 8:
        raise ValueError("必要对照和时序核对未完整。")
    for field, path in [("inputs_sha256", Path(inputs.__file__)), ("account_sha256", Path(execution.__file__)),
                        ("tests_sha256", ROOT / "tests/test_support_price_acceptance_v1.py")]:
        if tests[field] != digest(path):
            raise ValueError("必要测试版本在冻结前改变。")
    protocol = {"at": parent.original.now(), "study": "510300_SUPPORT_PRICE_ACCEPTANCE_V1",
        "registration": REGISTRATION, "decision": RESULT, "primary": inputs.POLICIES[0], "policies": list(inputs.POLICIES),
        "controls": list(CONTROLS), "rule_card": "docs/510300_SUPPORT_PRICE_ACCEPTANCE_V1.md",
        "known_before_freeze": "R230全部来源/上涨/反例与R212/R216/R224失败已知；全部历史为开发，无盲样本。",
        "question": "支持信息持续作用与固定价格锚接受、实际持仓完整周确认，能否提高完整净CAGR和Sharpe？",
        "source": "固定65节点/25操作；九明确支持节点及相对保存操作下降激活，上调反向。其他工具/实施/迟发回顾不激活，未知不补；同源不独立加分。",
        "announced_target_confirmation": inputs.ANNOUNCED_RATE_TARGETS,
        "fixed_support_nodes": inputs.SUPPORT_NODES,
        "entry": "支持第一次ETF16点可知日固定现金平移高低。后续收盘严格超过锚高且日柱>0，次开；触及锚低或操作上调失效；新支持不重写活动锚，接受/失效消费，未成交/已有持仓不重试旧点位。无固定消息年龄或等待期限。",
        "holding": "实际进入EARLY，锚低或日柱负且低EMA20退出。真实进入后完成周的低与收盘同时高于此前周，一次晋CARRY，原位和最近完整周低取最大跟踪，只升不降；宏观可滞后。主与固定退出同用反向操作退出。",
        "price_control": "没有活动锚时任何已知价格日可形成新锚，其后接受/失效同价格条件；不使用支持或反向政策信息，持有同主。",
        "fixed_exit_control": "同支持事件及共同锚低/反向信息保护，实际进入经济开盘下1ATR上2ATR和20收盘到期；不晋CARRY。",
        "account": read(parent.OUT / "protocol.json")["account"], "costs": read(parent.OUT / "protocol.json")["costs"],
        "support_stop_difference": "本用途用固定信息/价格观察日低点，替代原阶段进入前周低；三新政策相同账户风险预算，不能把原策略结构止损当账户风险门。",
        "periods": PERIODS, "new_candidate_accounts": 12, "saved_controls_reused": 8,
        "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "parameter_grid": False,
        "necessary_tests": 8, "all_prefix_checks": 19,
        "economic_gate": "四场景主净CAGR/Sharpe正且严格高于原A/原阶段/两归因对照，DD<=10%，实际完成净pB>1且标准期望正；次数软目标。",
        "stability": "四场景四对照20/252循环块各2000、原种子510300154；收益和Sharpe增量95%下界都正。",
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "complete_policy_coverage": "NOT_ESTABLISHED",
        "independent_validation": "NOT_ESTABLISHED", "search_selection_correction": "NOT_COMPUTED",
        "no_rescue": "唯一配置一次完整运行；不按结果改锚、符号、状态、消息年龄、风险、费用或时期，旧失败和13前瞻保持。",
        "sources": [{"path": relative(path), "sha256": digest(path)} for path in base_source_paths()]}
    write(OUT / "protocol.json", protocol)
    print("R231唯一支持—价格接受—延续机制及两对照固定；12新账户未运行。", flush=True)


def check_timing(result, local, policy):
    orders, trades, decisions = result["orders"], result["trades"], result["decisions"]
    if len(orders):
        buys = orders.loc[orders.side.eq("BUY")]
        if not buys.setup_date.lt(buys.origin).all():
            raise AssertionError("实际买入未先等待锚日后的价格接受。")
    for trade in trades.itertuples():
        if not trade.setup_date < trade.entry_origin < trade.entry_date:
            raise AssertionError("实际进入锚点、信号和成交顺序错误。")
        if not pd.isna(trade.promotion_date) and trade.promotion_date < trade.entry_date:
            raise AssertionError("持仓确认被倒填到实际进入之前。")
    active = decisions.loc[decisions.holding_stage.isin(["EARLY", "CARRY"])]
    if active.complete_week_last.notna().any():
        valid = active.complete_week_last.notna()
        if not active.loc[valid,"complete_week_last"].lt(active.loc[valid,"origin"]).all():
            raise AssertionError("持有决定用了未完成周。")
    return {"actual_cycles": len(trades), "order_clock_and_fixed_anchor_passed": True,
        "promoted_cycles": int(trades.promotion_date.notna().sum()) if "promotion_date" in trades else 0,
        "price_control_has_no_policy_exits": not orders.reason.eq("KNOWN_REVERSE_OPERATION").any() if len(orders) and policy == inputs.POLICIES[1] else True}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("支持接受完整金融已经开始，不重复。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定来源或实现改变："+item["path"])
    write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts_planned": 12, "new_fits": 0})
    observed = pd.read_parquet(OUT / "results/全部3488支持信息与价格接受状态.parquet")
    _, dividends, risks, _ = parent.original.load()
    records, annual, checks, timings, accounts = [], [], [], [], {}
    for period, (start, end) in PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            for policy in [CONTROL_A, CONTROL_STAGE, *inputs.POLICIES]:
                if policy in inputs.POLICIES:
                    result = execution.account(local, dividends, risks, policy, cost, start)
                    checks.append({"period": period, "cost": cost, "policy": policy, **parent.verify_account(result)})
                    timings.append({"period": period, "cost": cost, "policy": policy, **check_timing(result, local, policy)})
                    folder = OUT / "accounts" / period / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    for name in ACCOUNT_TABLES:
                        result[name].to_parquet(folder / f"{name}.parquet", index=False)
                    write(folder / "terminal.json", result["terminal"])
                else:
                    result = controls.saved_account(period, cost, policy)
                accounts[(period,cost,policy)] = result
                stats, yearly = controls.statistics(result, period, cost, policy)
                records.append({"period": period, "cost": cost, "policy": policy, **stats})
                annual.extend(yearly)
                if policy in inputs.POLICIES:
                    print(f"{period}/{cost}/{NAMES[policy]}：净年化{stats['net_cagr']:.4%}，夏普{stats['net_sharpe']:.6f}，完成{stats['completed_cycles']}，pB{stats['p_times_b']:.6f}。", flush=True)
    frame = pd.DataFrame(records)
    table("二十完整账户_支持信息价格接受与延续比较", frame)
    table("全部逐年实际次数与净收益", pd.DataFrame(annual))
    table("十二新账户现金订单周期复算", pd.DataFrame(checks))
    table("十二新账户固定锚与实际持仓时序", pd.DataFrame(timings))
    comparisons, gates, interval_rows = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            group = frame.loc[frame.period.eq(period) & frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[inputs.POLICIES[0]]
            conditions = {"positive_cagr_sharpe": bool(primary.net_cagr>0 and primary.net_sharpe>0),
                "drawdown_within_10pct": bool(primary.max_drawdown<=.1), "actual_pB_above_1": bool(primary.p_times_b>1),
                "actual_standard_EV_positive": bool(primary.standard_expectancy_loss_units>0)}
            for control in CONTROLS:
                reference = group.loc[control]
                conditions["beats_"+control] = bool(primary.net_cagr>reference.net_cagr and primary.net_sharpe>reference.net_sharpe)
                a, b = accounts[(period,cost,control)]["daily"], accounts[(period,cost,inputs.POLICIES[0])]["daily"]
                pd.testing.assert_series_equal(a.date, b.date, check_exact=True)
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
            print(f"{period}/{cost}四对照与两尺度成本后增量区间完成。", flush=True)
    table("全部四场景四对照两尺度净增量区间", pd.DataFrame(interval_rows))
    table("四场景完整经济门逐项结果", pd.DataFrame(gates))
    stable = all(item["cagr_delta_95"][0]>0 and item["sharpe_delta_95"][0]>0 for comparison in comparisons for item in comparison["intervals"])
    passed = all(row["economic_passed"] for row in gates)
    write(OUT / "summary.json", {"at": parent.original.now(), "registration": REGISTRATION, "decision": RESULT,
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_SUPPORT_PRICE_ACCEPTANCE_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "timing_checks": timings,
        "new_accounts": 12, "saved_controls_reused": 8, "new_original_policy_replays": 0,
        "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "necessary_tests_passed": 8,
        "all_prefix_checks": 19, "saved_account_checks": len(checks), "history_role": "DEVELOPMENT_CALIBRATION",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "search_selection_correction": "NOT_COMPUTED",
        "overfitting_removed": False, "goal_achieved": False})
    print("R232完整支持信息与价格接受实验终态；所有费用、时期、反例和失败保留，不按结果改配置。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="支持信息与价格接受唯一完整账户用途")
    parser.add_argument("action", choices=["preflight", "freeze", "run"])
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.action]()


if __name__ == "__main__":
    main()
