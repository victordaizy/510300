"""一次运行固定领先行业失效的完整账户和两个价格归因对照。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_cohort_failure_study_v1 as previous
from research import broker_stage_policy_study_v1 as parent
from research import industry_leader_failure_inputs_v1 as candidate
from research import industry_structure_description_study_v1 as description

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_industry_leader_failure_v1"
REGISTRATION, RESULT = "TECH.R223", "TECH.R224"
CONTROL_A, CONTROL_STAGE = previous.CONTROL_A, previous.CONTROL_STAGE
CONTROLS = (CONTROL_A, CONTROL_STAGE, *candidate.POLICIES[1:])
NAMES = {CONTROL_A: "原A", CONTROL_STAGE: "原阶段账户", **candidate.NAMES}
ACCOUNT_TABLES = previous.ACCOUNT_TABLES
read, write, digest = parent.read, parent.write, parent.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    _, observed, dividends, risks, _ = parent.load()
    _, _, _, saved, panel, members, calendar = description.load()
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), saved.reset_index(drop=True), check_dtype=False, check_exact=True)
    return observed, dividends, risks, candidate.IndustryData(panel, members, calendar)


def saved_account(period, cost, policy):
    return previous.saved_account(period, cost, policy)


def statistics(result, period, cost, policy):
    return previous.statistics(result, period, cost, policy)


def source_paths():
    paths = [Path(__file__), Path(candidate.__file__), ROOT / "tests/test_industry_leader_failure_v1.py",
        ROOT / "docs/510300_INDUSTRY_LEADER_FAILURE_V1.md", Path(previous.__file__),
        *parent.source_paths(), *description.source_paths(),
        description.OUT / "protocol.json", description.OUT / "summary.json", description.OUT / "post_run_diagnosis.json",
        description.OUT / "行业结构_具体上涨与失败反例.md", previous.OUT / "summary.json"]
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            folder = parent.OUT / "accounts" / period / cost / parent.stage.POLICIES[0]
            paths.extend(folder / f"{name}.parquet" for name in ACCOUNT_TABLES)
            paths.append(folder / "terminal.json")
    return list(dict.fromkeys(paths))


def preflight():
    if (OUT / "control_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("原阶段复现已经执行或登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 9 or tests["inputs_sha256"] != digest(Path(candidate.__file__)):
        raise ValueError("九项必要测试未通过或代码版本改变。")
    observed, dividends, risks, source = load()
    checks = []
    for period, (start, end) in parent.PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            actual = candidate.account(local, dividends, risks, source, CONTROL_STAGE, cost, start)
            saved = saved_account(period, cost, CONTROL_STAGE)
            for name in ACCOUNT_TABLES:
                comparable = actual[name]
                if comparable.shape == saved[name].shape == (0, 0):
                    comparable = pd.DataFrame(index=comparable.index, columns=saved[name].columns)
                pd.testing.assert_frame_equal(comparable, saved[name], check_exact=True)
            if parent.original.clean(actual["terminal"]) != saved["terminal"] or len(actual["industry_checks"]):
                raise AssertionError("原阶段终态或空适配器改变。")
            checks.append({"period": period, "cost": cost, "all_five_tables_terminal_exact": True, **parent.verify_account(actual)})
            print(f"{period}/{cost}原阶段五表及终态精确复现。", flush=True)
    write(OUT / "control_preflight.json", {"at": parent.original.now(), "original_R212_exact_replays": 4,
        "checks": checks, "new_policy_accounts": 0, "original_A_new_replays": 0,
        "all_saved_stage_inputs_exact": True, "empty_table_index_normalized_for_comparison_only": True})


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("实际进入行业失效用途已登记，不覆盖。")
    pre = read(OUT / "control_preflight.json")
    tests = read(OUT / "tests_receipt.json")
    if pre["original_R212_exact_replays"] != 4 or tests["passed"] != 9 or tests["exit_code"] != 0:
        raise ValueError("原账户或必要测试不满足。")
    if tests["inputs_sha256"] != digest(Path(candidate.__file__)) or tests["tests_sha256"] != digest(ROOT / "tests/test_industry_leader_failure_v1.py"):
        raise ValueError("事前测试的实现版本改变。")
    write(OUT / "protocol.json", {"at": parent.original.now(), "study": "510300_INDUSTRY_LEADER_FAILURE_V1",
        "registration": REGISTRATION, "decision": RESULT, "primary": candidate.POLICIES[0], "policies": list(candidate.POLICIES),
        "controls": list(CONTROLS), "rule_card": "docs/510300_INDUSTRY_LEADER_FAILURE_V1.md",
        "known_before_freeze": "R212/R216账户失败和R222全部历史描述已知，五日与方向是开发假设；不把原8亏损上下文当盲测。",
        "anchor": "实际进入前一源日成员/分类/20报价，分类公布钟不晚于进入日09:30，固定前三行业不重选。",
        "coverage": "原294/300、至少5成员/98%20报价和4行业形成锚；退出只测三领先行业，各98%后锚完整及当前市场VIEW_ALLOWED/300成员；跟随缺口另存不参与资格。",
        "check_clock": "仅实际进入索引加5收盘一次，用进入日到进入后第4日五收益，未知不重试；额外退出次开，原风险退出优先。",
        "primary_condition": "ETF同期含息累计>1e-12且原三领先行业等权累计均值<=1e-12。",
        "controls_condition": "同领先可观察门仅ETF正退出、全日历仅ETF正退出；原进入/持有/费用/资金风险完全一致。",
        "account": read(parent.OUT / "protocol.json")["account"], "costs": read(parent.OUT / "protocol.json")["costs"],
        "periods": parent.PERIODS, "new_policy_accounts": 12, "original_R212_exact_replays": 4, "original_A_new_replays": 0,
        "necessary_tests": 9, "new_fits": 0, "new_labels": 0, "new_market_requests": 0, "parameter_grid": False,
        "economic_gate": "四场景主政策净CAGR/全日历净Sharpe正且严格高于原A/阶段/同覆盖价格/全价格；DD<=10%，实际完成pB>1且标准净期望正。次数软目标。",
        "stability": "四场景对四对照，全部20/252日循环块各2000次、种子510300154；CAGR/Sharpe增量95%下界均正。",
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "search_selection_correction": "NOT_COMPUTED", "no_rescue": "唯一配置一次完整运行，不按收益改五日/方向/覆盖/源钟/风险/费用/时期。",
        "sources": [{"path": relative(path), "sha256": digest(path)} for path in
            source_paths() + [OUT / "tests_receipt.json", OUT / "control_preflight.json"]]})
    print("R223实际进入锚点和三领先行业失效唯一配置固定，12新账户尚未运行。", flush=True)


def check_information(result, local):
    checks = result["industry_checks"]
    if checks.cycle_id.duplicated().any():
        raise AssertionError("同一实际周期重复行业检查。")
    indices = {pd.Timestamp(date): i for i, date in enumerate(local.date)}
    for row in checks.itertuples():
        entry, decision = indices[pd.Timestamp(row.entry_date)], indices[pd.Timestamp(row.decision_date)]
        if decision - entry != candidate.CHECK_LAG or row.relative_session != candidate.CHECK_LAG:
            raise AssertionError("不是实际进入后的唯一第五日。")
        if pd.Timestamp(row.anchor_source_date) != pd.Timestamp(local.date.iloc[entry - 1]) or pd.Timestamp(row.observation_source_date) != pd.Timestamp(local.date.iloc[decision - 1]):
            raise AssertionError("进入或观察源日错位。")
        if row.leader_view_allowed:
            opening = pd.Timestamp(row.entry_date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
            if pd.Timestamp(row.classification_available_at) > opening or row.known_leader_groups != 3:
                raise AssertionError("领先行业资格或公布钟错误。")
    orders = result["orders"]
    extra = orders.loc[orders.reason.fillna("").str.endswith("_FIVE_SOURCE_CLOSES")]
    for order in extra.itertuples():
        matched = checks.loc[checks.cycle_id.eq(order.cycle_id) & checks.extra_exit & checks.extra_reason.eq(order.reason)]
        if len(matched) != 1 or not pd.Timestamp(matched.decision_date.iloc[0]) < pd.Timestamp(order.date):
            raise AssertionError("额外实际退出没有唯一先前请求。")
    return {"checks": len(checks), "known_leader_checks": int(checks.leader_view_allowed.sum()),
        "unknown_leader_checks": int((~checks.leader_view_allowed).sum()), "extra_requests": int(checks.extra_exit.sum()),
        "actual_extra_sells": len(extra), "timing_and_single_check_passed": True}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("实际进入行业失效账户已经开始，不重复运行。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结文件已改变：" + item["path"])
    write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_policy_accounts_planned": 12, "new_fits": 0, "new_labels": 0})
    observed, dividends, risks, source = load()
    records, annual, checks, timing, combined, accounts = [], [], [], [], [], {}
    for period, (start, end) in parent.PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            for policy in (CONTROL_A, CONTROL_STAGE, *candidate.POLICIES):
                if policy in candidate.POLICIES:
                    result = candidate.account(local, dividends, risks, source, policy, cost, start)
                    checks.append({"period": period, "cost": cost, "policy": policy, **parent.verify_account(result)})
                    timing.append({"period": period, "cost": cost, "policy": policy, **check_information(result, local)})
                    folder = OUT / "accounts" / period / cost / policy
                    folder.mkdir(parents=True, exist_ok=True)
                    for name in (*ACCOUNT_TABLES, "industry_checks", "industry_group_checks"):
                        result[name].to_parquet(folder / f"{name}.parquet", index=False)
                    write(folder / "terminal.json", result["terminal"])
                    combined.append(result["industry_checks"].assign(period=period, cost=cost))
                else:
                    result = saved_account(period, cost, policy)
                accounts[(period, cost, policy)] = result
                stats, yearly = statistics(result, period, cost, policy)
                records.append({"period": period, "cost": cost, "policy": policy, **stats})
                annual.extend(yearly)
                if policy in candidate.POLICIES:
                    print(f"{period}/{cost}/{NAMES[policy]}：净年化{stats['net_cagr']:.4%}，夏普{stats['net_sharpe']:.6f}，完成{stats['completed_cycles']}，pB{stats['p_times_b']:.6f}。", flush=True)
    frame = pd.DataFrame(records)
    table("二十完整账户_行业与价格同资金风险费用比较", frame)
    table("全部逐年实际完成周期与账户收益", pd.DataFrame(annual))
    table("十二账户完整现金订单周期核验", pd.DataFrame(checks))
    table("十二账户实际进入第五日时序与未知", pd.DataFrame(timing))
    table("全部实际周期行业检查_未知与额外退出", pd.concat(combined, ignore_index=True))
    comparisons, gates, interval_rows = [], [], []
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            group = frame.loc[frame.period.eq(period) & frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[candidate.POLICIES[0]]
            conditions = {"positive_cagr_sharpe": bool(primary.net_cagr > 0 and primary.net_sharpe > 0),
                "drawdown_within_10pct": bool(primary.max_drawdown <= .1), "actual_pB_above_1": bool(primary.p_times_b > 1),
                "actual_standard_EV_positive": bool(primary.standard_expectancy_loss_units > 0)}
            for control in CONTROLS:
                reference = group.loc[control]
                conditions[f"beats_{control}"] = bool(primary.net_cagr > reference.net_cagr and primary.net_sharpe > reference.net_sharpe)
                a, b = accounts[(period, cost, control)]["daily"], accounts[(period, cost, candidate.POLICIES[0])]["daily"]
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
            print(f"{period}/{cost}全部四对照和20/252两尺度区间完成。", flush=True)
    table("全部四对照两尺度成本后增量区间", pd.DataFrame(interval_rows))
    table("四场景逐项经济门", pd.DataFrame(gates))
    stable = all(item["cagr_delta_95"][0] > 0 and item["sharpe_delta_95"][0] > 0 for comparison in comparisons for item in comparison["intervals"])
    passed = all(gate["economic_passed"] for gate in gates)
    write(OUT / "summary.json", {"at": parent.original.now(), "decision": RESULT,
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_INDUSTRY_LEADER_FAILURE_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "timing_checks": timing,
        "new_accounts": 12, "original_R212_exact_replays": 4, "original_A_new_replays": 0, "new_fits": 0, "new_labels": 0,
        "new_market_requests": 0, "necessary_tests_passed": 9, "saved_account_checks": len(checks),
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "overfitting_removed": False, "goal_achieved": False})
    print("R224一次完整账户检验结束，保留所有对照和未知，不按结果修改配置。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实际进入后固定领先行业失效一次完整账户实验。")
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.command]()
