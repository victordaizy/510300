"""只接续已保存完整账户的比较；日期精度规范不改变信号、回报或账户。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import support_price_acceptance_study_v1 as study

ROOT, SOURCE = study.ROOT, study.OUT
OUT = SOURCE / "implementation_v1_0_1"
read, write, digest = study.read, study.write, study.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def require_same_dates(a, b):
    """只比较统一纳秒后的原时间值；不同日期、长度、索引或顺序仍失败。"""
    left = pd.to_datetime(a).astype("datetime64[ns]")
    right = pd.to_datetime(b).astype("datetime64[ns]")
    pd.testing.assert_series_equal(left, right, check_exact=True)
    return {"left_original_storage": str(a.dtype), "right_original_storage": str(b.dtype),
            "all_date_values_and_index_exact": True, "days": len(left)}


def saved_account(period, cost, policy):
    if policy not in study.inputs.POLICIES:
        return study.controls.saved_account(period, cost, policy)
    folder = SOURCE / "accounts" / period / cost / policy
    result = {name: pd.read_parquet(folder / f"{name}.parquet") for name in study.ACCOUNT_TABLES}
    result["terminal"] = read(folder / "terminal.json")
    return result


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("账户比较接续已固定，不覆盖。")
    initial = read(SOURCE / "protocol.json")
    failure = read(SOURCE / "acceptance_storage_failure.json")
    tests = read(OUT / "tests_receipt.json")
    frame = pd.read_parquet(SOURCE / "results/二十完整账户_支持信息价格接受与延续比较.parquet")
    if len(frame) != 20 or failure["completed_new_accounts"] != 12 or tests["passed"] != 2 or tests["exit_code"] != 0:
        raise ValueError("原保存金融结果或必要日期检查不完整。")
    if tests["implementation_sha256"] != digest(Path(__file__)):
        raise ValueError("已测试比较实现改变。")
    for item in initial["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原金融冻结输入或实现改变。")
    paths = [Path(__file__), ROOT / "tests/test_support_price_acceptance_acceptance_v1_0_1.py",
        SOURCE / "protocol.json", SOURCE / "acceptance_storage_failure.json",
        SOURCE / "results/二十完整账户_支持信息价格接受与延续比较.parquet",
        SOURCE / "results/全部逐年实际次数与净收益.parquet", SOURCE / "results/十二新账户现金订单周期复算.parquet",
        SOURCE / "results/十二新账户固定锚与实际持仓时序.parquet"]
    for period in study.PERIODS:
        for cost in study.COSTS:
            for policy in study.inputs.POLICIES:
                folder = SOURCE / "accounts" / period / cost / policy
                paths.extend(folder / f"{name}.parquet" for name in study.ACCOUNT_TABLES)
                paths.append(folder / "terminal.json")
    write(OUT / "protocol.json", {"at": study.parent.original.now(), "registration": study.REGISTRATION, "decision": study.RESULT,
        "implementation": "1.0.1_DATE_STORAGE_COMPARISON_ONLY", "completed_new_accounts_reused": 12,
        "original_economic_gate": initial["economic_gate"], "original_stability": initial["stability"],
        "date_handling": "仅比较日期序列时从ms/ns规范为ns，必须逐值和索引精确相同；原保存文件、所有回报、规则、时期和费用不改。",
        "necessary_date_tests": 2, "new_account_replays": 0, "new_fits": 0, "new_requests": 0,
        "original_input_files_unchanged": len(initial["sources"]),
        "files": [{"path": relative(path), "sha256": digest(path)} for path in paths]})
    print("R232日期比较接续固定，复用12新账户；零金融重跑，原所有经济门不变。", flush=True)


def finish():
    if (OUT / "started.json").exists():
        raise RuntimeError("保存账户比较已开始，不重复区间检验。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("保存账户或接续实现改变。")
    write(OUT / "started.json", {"at": study.parent.original.now(), "new_account_replays": 0})
    frame = pd.read_parquet(SOURCE / "results/二十完整账户_支持信息价格接受与延续比较.parquet")
    original_checks = pd.read_parquet(SOURCE / "results/十二新账户现金订单周期复算.parquet")
    timings = pd.read_parquet(SOURCE / "results/十二新账户固定锚与实际持仓时序.parquet")
    comparisons, gates, interval_rows, date_checks = [], [], [], []
    for period in study.PERIODS:
        for cost in study.COSTS:
            group = frame.loc[frame.period.eq(period)&frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[study.inputs.POLICIES[0]]
            conditions = {"positive_cagr_sharpe": bool(primary.net_cagr>0 and primary.net_sharpe>0),
                "drawdown_within_10pct": bool(primary.max_drawdown<=.1), "actual_pB_above_1": bool(primary.p_times_b>1),
                "actual_standard_EV_positive": bool(primary.standard_expectancy_loss_units>0)}
            actual = saved_account(period, cost, study.inputs.POLICIES[0])
            for control in study.CONTROLS:
                reference = group.loc[control]
                conditions["beats_"+control] = bool(primary.net_cagr>reference.net_cagr and primary.net_sharpe>reference.net_sharpe)
                a = saved_account(period, cost, control)["daily"]
                b = actual["daily"]
                date_checks.append({"period": period, "cost": cost, "control": control, **require_same_dates(a.date, b.date)})
                intervals = study.parent.intervals(a.net_return.to_numpy(float), b.net_return.to_numpy(float))
                comparison = {"period": period, "cost": cost, "control": control,
                    "cagr_delta": primary.net_cagr-reference.net_cagr, "sharpe_delta": primary.net_sharpe-reference.net_sharpe, "intervals": intervals}
                comparisons.append(comparison)
                for item in intervals:
                    interval_rows.append({"period": period, "cost": cost, "control": control, "block": item["block"], "resamples": item["resamples"],
                        "cagr_delta": comparison["cagr_delta"], "sharpe_delta": comparison["sharpe_delta"],
                        "cagr_delta_lower": item["cagr_delta_95"][0], "cagr_delta_upper": item["cagr_delta_95"][1],
                        "sharpe_delta_lower": item["sharpe_delta_95"][0], "sharpe_delta_upper": item["sharpe_delta_95"][1]})
            gates.append({"period": period, "cost": cost, **conditions, "economic_passed": all(conditions.values())})
            print(f"{period}/{cost}保存账户全部四对照及两尺度区间完成。", flush=True)
    table("全部四场景四对照两尺度净增量区间", pd.DataFrame(interval_rows))
    table("四场景完整经济门逐项结果", pd.DataFrame(gates))
    table("十六组原日期值与索引精确比较", pd.DataFrame(date_checks))
    stable = all(item["cagr_delta_95"][0]>0 and item["sharpe_delta_95"][0]>0 for comparison in comparisons for item in comparison["intervals"])
    passed = all(item["economic_passed"] for item in gates)
    for item in protocol["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("比较后原账户文件改变。")
    write(OUT / "summary.json", {"at": study.parent.original.now(), "registration": study.REGISTRATION, "decision": study.RESULT,
        "implementation": "1.0.1_DATE_STORAGE_COMPARISON_ONLY",
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_SUPPORT_PRICE_ACCEPTANCE_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "timing_checks": timings.to_dict("records"),
        "original_new_accounts_completed": 12, "original_saved_controls_reused": 8, "new_account_replays_this_completion": 0,
        "original_necessary_tests": 8, "necessary_date_tests": 2, "original_prefix_checks": 19,
        "original_cash_account_checks": len(original_checks), "date_value_comparisons": len(date_checks),
        "all_original_account_files_unchanged": True, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "search_selection_correction": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False})
    print("R232保存12账户的完整比较终态，日期值精确，原收益和配置不改。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="保存账户日期比较与原验收接续")
    parser.add_argument("action", choices=["freeze", "finish"])
    args = parser.parse_args()
    {"freeze": freeze, "finish": finish}[args.action]()


if __name__ == "__main__":
    main()
