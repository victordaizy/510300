"""一次检验真实进入后的固定组失效；保存完整账户及价格归因对照。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_cohort_failure_inputs_v1 as candidate
from research import broker_fixed_cohort_inputs_v1 as cohort
from research import broker_fixed_cohort_observation_v1 as observation
from research import broker_stage_policy_study_v1 as parent

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_broker_cohort_failure_v1"
REGISTRATION, RESULT = "TECH.R215", "TECH.R216"
CONTROL_A, CONTROL_STAGE = "A_SAVED_WEIGHT", "R212_CONTROL"
CONTROLS = (CONTROL_A, CONTROL_STAGE, *candidate.POLICIES[1:])
NAMES = {CONTROL_A: "原A", CONTROL_STAGE: "原阶段退出", **candidate.NAMES}
ACCOUNT_TABLES = ("daily", "orders", "trades", "decisions", "rejections")
read, write, digest = parent.read, parent.write, parent.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    base, observed, dividends, risks, _ = parent.load()
    q = pd.read_parquet(observation.SOURCEFILES["classified"])
    m = pd.read_parquet(observation.SOURCEFILES["membership"])
    c = pd.read_parquet(observation.SOURCEFILES["coverage"])
    panel = cohort.make_panel(observed.date, q, m, c)
    return base, observed, dividends, risks, panel


def saved_account(period, cost, policy):
    if policy == CONTROL_A:
        folder = parent.original.CONTROL / period / cost / CONTROL_A
        result = {name: pd.read_parquet(folder / f"{name}.parquet") for name in ACCOUNT_TABLES[:3]}
        result["terminal"] = {"stopped": bool(result["daily"].risk_stopped.iloc[-1]),
            "unpaid_dividend_cny": float(result["daily"].receivable.iloc[-1])}
        return result
    folder = (parent.OUT / "accounts" / period / cost / parent.stage.POLICIES[0]
              if policy == CONTROL_STAGE else OUT / "accounts" / period / cost / policy)
    result = {name: pd.read_parquet(folder / f"{name}.parquet") for name in ACCOUNT_TABLES}
    result["terminal"] = read(folder / "terminal.json")
    if policy in candidate.POLICIES:
        result["failure_checks"] = pd.read_parquet(folder / "failure_checks.parquet")
    return result


def statistics(result, period, cost, policy):
    yearly = parent.annual_rows(result, period, policy, cost)
    stats = {**parent.measurements.metrics(result), **parent.extra_metrics(result, yearly)}
    return stats, yearly


def source_paths():
    paths = [Path(__file__), Path(candidate.__file__), Path(cohort.__file__),
        ROOT / "tests/test_broker_cohort_failure_v1.py", ROOT / "docs/510300_BROKER_COHORT_FAILURE_V1.md",
        parent.OUT / "protocol.json", parent.OUT / "summary.json", parent.OUT / "control_preflight.json",
        observation.OUT / "protocol.json", observation.OUT / "implementation_v1_0_1/summary.json",
        *parent.source_paths(), *[observation.SOURCEFILES[key] for key in
            ("classified", "membership", "coverage", "membership_admission", "membership_2015_admission")]]
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            folder = parent.OUT / "accounts" / period / cost / parent.stage.POLICIES[0]
            paths.extend(folder / f"{name}.parquet" for name in ACCOUNT_TABLES)
            paths.append(folder / "terminal.json")
    return list(dict.fromkeys(paths))


def preflight():
    if (OUT / "control_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("本用途原阶段核验已经执行或已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if (tests["passed"] != 8 or tests["exit_code"] != 0
            or tests["inputs_sha256"] != digest(Path(candidate.__file__))
            or tests["tests_sha256"] != digest(ROOT / "tests/test_broker_cohort_failure_v1.py")):
        raise ValueError("八项必要测试或版本不满足。")
    _, observed, dividends, risks, panel = load()
    checks = []
    for period, (start, end) in parent.PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            actual = candidate.account(local, dividends, risks, panel, CONTROL_STAGE, cost, start)
            saved = saved_account(period, cost, CONTROL_STAGE)
            for name in ACCOUNT_TABLES:
                # 空的无列表保存后列索引类型改变；仅核对时规范，不改变账户。
                comparable = actual[name]
                if comparable.shape == saved[name].shape == (0, 0):
                    comparable = pd.DataFrame(index=comparable.index, columns=saved[name].columns)
                pd.testing.assert_frame_equal(comparable, saved[name], check_exact=True)
            if parent.original.clean(actual["terminal"]) != saved["terminal"] or len(actual["failure_checks"]):
                raise AssertionError("原阶段终态或空适配器不一致。")
            checks.append({"period": period, "cost": cost, "original_R212_all_tables_terminal_exact": True,
                **parent.verify_account(actual)})
            print(f"{period}/{cost}原R212五账表及终态精确复现。", flush=True)
    write(OUT / "control_preflight.json", {"at": parent.original.now(), "original_R212_replays": 4,
        "original_A_replays_this_study": 0, "original_A_saved_read_only": True,
        "checks": checks, "new_policy_accounts": 0, "new_fits": 0,
        "empty_zero_column_table_serialization_normalized_for_comparison_only": True,
        "pre_registration_original_R212_attempts": 5, "pre_registration_failed_attempts": 1})


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("固定组失效用途已固定，不覆盖。")
    preflight_result = read(OUT / "control_preflight.json")
    if preflight_result["original_R212_replays"] != 4:
        raise ValueError("原阶段四账户尚未精确核验。")
    protocol = {"study": "510300_BROKER_COHORT_FAILURE_V1", "at": parent.original.now(),
        "registration": REGISTRATION, "result_decision": RESULT, "primary": candidate.POLICIES[0],
        "policies": list(candidate.POLICIES), "controls": list(CONTROLS), "check_lag": candidate.CHECK_LAG,
        "rule_card": "docs/510300_BROKER_COHORT_FAILURE_V1.md",
        "known_before_freeze": "R212完整失败及R214全部固定组描述已知；五日是结果启发的开发选择，不是独立或盲测。",
        "anchor": "实际成交进入日；成员和20日分组使用进入前一完整日。与R214信号锚点不同。",
        "decision_clock": "仅进入索引加5收盘一次，成分及ETF五个收益到前一交易日；未知不重试。execution_date列是下一计划开盘，实际成交以orders为准。",
        "policy": "原退出优先；两固定组均未过半上涨且ETF同期含股息总回报<=1e-12时额外退出。",
        "unknown": "保持原阶段退出；不填零、不缩评价日历，成分末日2026-08-14。",
        "account": read(parent.OUT / "protocol.json")["account"], "costs": read(parent.OUT / "protocol.json")["costs"],
        "periods": parent.PERIODS, "new_policy_accounts": 12, "original_R212_exact_replays": 4,
        "original_A_new_replays": 0, "necessary_tests": 8, "new_fits": 0, "new_labels": 0,
        "new_market_requests": 0, "parameter_grid": False,
        "economic_gate": "四场景主政策净CAGR/全日历净Sharpe>0且高于四对照；DD<=10%、实际完成pB>1、标准净期望>0。频率软目标。",
        "stability": "四场景相对四对照全部20/252日循环块各2000次，种子510300154；CAGR和Sharpe增量95%下界全部>0。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "first_vintage": "NOT_CERTIFIED", "search_selection_correction": "NOT_COMPUTED",
        "no_rescue": "唯一固定配置一次运行；不按结果改变五日、覆盖、符号、进入退出、费用风险或时期。",
        "sources": [{"path": relative(path), "sha256": digest(path)} for path in
            source_paths() + [OUT / "tests_receipt.json", OUT / "control_preflight.json"]]}
    write(OUT / "protocol.json", protocol)
    print("R215唯一配置及两个价格归因对照已固定；12新账户尚无收益结果。", flush=True)


def check_information(result, local):
    checks = result["failure_checks"]
    if checks.cycle_id.duplicated().any():
        raise AssertionError("单个周期重复进行第五日检查。")
    if len(checks):
        index = {pd.Timestamp(date): i for i, date in enumerate(local.date)}
        for row in checks.itertuples():
            entry, decision = index[pd.Timestamp(row.entry_date)], index[pd.Timestamp(row.decision_date)]
            if decision - entry != candidate.CHECK_LAG or row.relative_session != candidate.CHECK_LAG:
                raise AssertionError("检查不是唯一固定第五日。")
            if (pd.Timestamp(row.cohort_source_date) != pd.Timestamp(local.date.iloc[entry - 1])
                    or pd.Timestamp(row.observation_source_date) != pd.Timestamp(local.date.iloc[decision - 1])):
                raise AssertionError("固定成员或观察源钟违背原日历。")
    extra = result["orders"].loc[result["orders"].reason.fillna("").str.endswith("_FIVE_SOURCE_CLOSES_FAILED")]
    if len(extra):
        requests = checks.loc[checks.extra_exit, ["cycle_id", "decision_date", "extra_reason"]]
        for order in extra.itertuples():
            matched = requests.loc[requests.cycle_id.eq(order.cycle_id) & requests.extra_reason.eq(order.reason)]
            if len(matched) != 1 or not pd.Timestamp(matched.decision_date.iloc[0]) < pd.Timestamp(order.date):
                raise AssertionError("额外退出没有唯一先前请求。")
    return {"checks": len(checks), "known_checks": int(checks.cohort_view_allowed.sum()),
        "unknown_checks": int((~checks.cohort_view_allowed).sum()), "extra_requests": int(checks.extra_exit.sum()),
        "actual_extra_sell_orders": len(extra), "timing_and_single_check_passed": True}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本用途已开始，不重复运行账户。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源已变化：" + item["path"])
    write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_policy_accounts_planned": 12,
        "new_fits": 0, "new_labels": 0})
    _, observed, dividends, risks, panel = load()
    records, annual, saved_checks, timing, combined_checks, accounts = [], [], [], [], [], {}
    for period, (start, end) in parent.PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            for policy in (CONTROL_A, CONTROL_STAGE, *candidate.POLICIES):
                if policy in candidate.POLICIES:
                    result = candidate.account(local, dividends, risks, panel, policy, cost, start)
                    saved_checks.append({"period": period, "cost": cost, "policy": policy,
                        **parent.verify_account(result)})
                    timing.append({"period": period, "cost": cost, "policy": policy,
                        **check_information(result, local)})
                    folder = OUT / "accounts" / period / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    for name in (*ACCOUNT_TABLES, "failure_checks"):
                        result[name].to_parquet(folder / f"{name}.parquet", index=False)
                    write(folder / "terminal.json", result["terminal"])
                    combined_checks.append(result["failure_checks"].assign(period=period, cost=cost))
                else:
                    result = saved_account(period, cost, policy)
                accounts[(period, cost, policy)] = result
                stats, yearly = statistics(result, period, cost, policy)
                records.append({"period": period, "cost": cost, "policy": policy, **stats})
                annual.extend(yearly)
                if policy in candidate.POLICIES:
                    print(f"{period}/{cost}/{NAMES[policy]}：净年化{stats['net_cagr']:.4%}，夏普{stats['net_sharpe']:.5f}，完成{stats['completed_cycles']}，pB{stats['p_times_b']:.5f}。", flush=True)
    frame = pd.DataFrame(records)
    table("二十完整账户_同资金风险成本比较", frame)
    table("全部逐年实际完成周期与账户收益", pd.DataFrame(annual))
    table("十二新账户账本核验", pd.DataFrame(saved_checks))
    table("十二新账户第五日事前源与单次检查", pd.DataFrame(timing))
    table("全部实际周期第五日未知与额外请求", pd.concat(combined_checks, ignore_index=True))
    comparisons, gates, interval_rows = [], [], []
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            group = frame.loc[frame.period.eq(period) & frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[candidate.POLICIES[0]]
            conditions = {"positive_cagr_sharpe": bool(primary.net_cagr > 0 and primary.net_sharpe > 0),
                "drawdown_within_10pct": bool(primary.max_drawdown <= .1),
                "actual_pB_above_1": bool(primary.p_times_b > 1),
                "actual_standard_EV_positive": bool(primary.standard_expectancy_loss_units > 0)}
            for control in CONTROLS:
                reference = group.loc[control]
                conditions[f"beats_{control}"] = bool(primary.net_cagr > reference.net_cagr and primary.net_sharpe > reference.net_sharpe)
                a, b = accounts[(period, cost, control)]["daily"], accounts[(period, cost, candidate.POLICIES[0])]["daily"]
                pd.testing.assert_series_equal(a.date, b.date, check_exact=True)
                intervals = parent.intervals(a.net_return.to_numpy(float), b.net_return.to_numpy(float))
                comparison = {"period": period, "cost": cost, "control": control,
                    "cagr_delta": primary.net_cagr - reference.net_cagr,
                    "sharpe_delta": primary.net_sharpe - reference.net_sharpe, "intervals": intervals}
                comparisons.append(comparison)
                for item in intervals:
                    interval_rows.append({"period": period, "cost": cost, "control": control,
                        "block": item["block"], "resamples": item["resamples"],
                        "cagr_delta": comparison["cagr_delta"], "sharpe_delta": comparison["sharpe_delta"],
                        "cagr_delta_lower": item["cagr_delta_95"][0], "cagr_delta_upper": item["cagr_delta_95"][1],
                        "sharpe_delta_lower": item["sharpe_delta_95"][0], "sharpe_delta_upper": item["sharpe_delta_95"][1]})
            gates.append({"period": period, "cost": cost, **conditions, "economic_passed": all(conditions.values())})
            print(f"{period}/{cost}四个对照、两种固定区块区间已保存于内存。", flush=True)
    table("全部四对照两尺度配对增量区间", pd.DataFrame(interval_rows))
    table("四场景逐项经济门", pd.DataFrame(gates))
    stable = all(item["cagr_delta_95"][0] > 0 and item["sharpe_delta_95"][0] > 0
        for comparison in comparisons for item in comparison["intervals"])
    passed = all(item["economic_passed"] for item in gates)
    write(OUT / "summary.json", {"at": parent.original.now(), "decision": RESULT,
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable
            else "REJECTED_FIXED_COHORT_FAILURE_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "timing_checks": timing,
        "new_accounts": 12, "original_R212_exact_replays": 4, "original_A_new_replays": 0,
        "new_fits": 0, "new_labels": 0, "new_market_requests": 0, "necessary_tests_passed": 8,
        "saved_account_checks": len(saved_checks), "history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False})
    print("R216一次完整金融实验结束；原规则与全部反例保留，不按结果营救配置。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实进入后的固定组失效一次账户实验。")
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.command]()
