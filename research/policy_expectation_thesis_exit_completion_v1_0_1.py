"""原冻结策略的日期/缺失值序列化适配；恢复唯一完整实验而非重复金融。"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research import policy_expectation_thesis_exit_v1 as frozen

ROOT, STATE, parent = frozen.ROOT, frozen.STATE, frozen.parent
OUT = frozen.OUT / "implementation_v1_0_1"


def __getattr__(name):
    return getattr(frozen, name)


def normalize(value):
    """只改变保存表示：未知为null，日期为ISO；不得变成零。"""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    return value


def write(path, obj):
    frozen.write(path, normalize(obj))


def read(path):
    target = path.absolute()
    source_root = (OUT / "sources").absolute()
    if target.is_relative_to(source_root):
        target = frozen.OUT / "sources" / target.relative_to(source_root)
    return frozen.read(target)


def export(frame, name):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".parquet", ".csv"):
        if path.with_suffix(suffix).exists():
            raise FileExistsError("禁止覆盖恢复后的完整结果：" + name)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def prepare():
    protocol = frozen.frozen_exact()
    tests = read(OUT / "serialization_tests_receipt.json")
    if tests["passed"] != 2 or tests["exit_code"] != 0:
        raise ValueError("两项必要序列化测试未通过。")
    failed = frozen.OUT / "accounts/2015_2019/BASE" / frozen.POLICY
    reference = parent.OUT / "accounts/2015_2019/BASE" / frozen.BASELINE
    proofs = []
    for name in parent.ACCOUNT_TABLES:
        actual, expected = pd.read_parquet(failed / f"{name}.parquet"), pd.read_parquet(reference / f"{name}.parquet")
        parent.previous.exact_saved_table(actual, expected)
        proofs.append({"table": name, "saved_candidate_path": frozen.rel(failed / f"{name}.parquet"),
                       "sha256": frozen.digest(failed / f"{name}.parquet"), "all_fields_exact_to_original_no_event_account": True})
    daily = pd.read_parquet(failed / "daily.parquet")
    events = pd.read_parquet(frozen.OUT / "results/全部日线_一次可知事件而非持续禁入.parquet")
    if events.loc[events.date.le(daily.date.max()), "credit_cancellation_event"].any():
        raise ValueError("早期已有新取消事件，不能继承原终态。")
    write(OUT / "implementation_addendum.json", {"at": frozen.now(), "version": "1.0.1", "original_registration": frozen.REGISTER,
        "failure": "原运行在第一个候选账户五表完成后，terminal.json的Timestamp写入中断。", "original_partial_terminal_preserved": frozen.rel(failed / "terminal.json"),
        "original_partial_terminal_sha256": frozen.digest(failed / "terminal.json"),
        "original_financial_run_started": frozen.rel(frozen.OUT / "run_started.json"),
        "economic_change": False, "original_account_code_changed": False, "frozen_original_code_changed": False,
        "complete_saved_first_candidate_reused_without_account_replay": True, "all_five_saved_table_proofs": proofs,
        "terminal_recovery": "候选五表与R240所有字段精确相同且早期没有新事件，因此继承同一原终态，不计算新盈利。",
        "remaining_new_candidate_accounts": 3, "same_old_sources_no_new_http": True,
        "normalization": "Timestamp/date→ISO，NaT/非有限未知→null；不补零，不改收益/窗口/期限/费用/进入退出。",
        "adapter_sha256": frozen.digest(Path(__file__)), "tests_sha256": frozen.digest(ROOT / "tests/test_policy_expectation_thesis_serialization_v1.py")})
    write(OUT / "protocol.json", {**protocol, "original_protocol": frozen.rel(frozen.OUT / "protocol.json"),
        "original_protocol_sha256": frozen.digest(frozen.OUT / "protocol.json"),
        "implementation_format_addendum": frozen.rel(OUT / "implementation_addendum.json")})
    print("1.0.1保存格式适配已登记；首个候选五表精确继承，经济规则0改动、该金融账户0重算。", flush=True)


def frozen_exact():
    original = frozen.frozen_exact()
    addendum = read(OUT / "implementation_addendum.json")
    if frozen.digest(Path(__file__)) != addendum["adapter_sha256"]:
        raise ValueError("已登记保存适配代码改变。")
    for proof in addendum["all_five_saved_table_proofs"]:
        if frozen.digest(ROOT / proof["saved_candidate_path"]) != proof["sha256"]:
            raise ValueError("原已完成候选账本改变。")
    return original


def recover_first():
    failed = frozen.OUT / "accounts/2015_2019/BASE" / frozen.POLICY
    reference = parent.OUT / "accounts/2015_2019/BASE" / frozen.BASELINE
    result = {name: pd.read_parquet(failed / f"{name}.parquet") for name in parent.ACCOUNT_TABLES}
    result["terminal"] = read(reference / "terminal.json")
    parent.previous.parent.verify_account(result)
    return result


def run():
    protocol = frozen_exact()
    write(OUT / "completion_started.json", {"at": frozen.now(), "original_new_candidates_completed_before_failure": 1, "new_candidates_still_to_compute": 3})
    surveys = frozen.classify_surveys(pd.read_csv(frozen.SURVEYS))
    if len(surveys) != 84 or int(surveys.same_instrument_prior_version_clock_valid.sum()) != 56:
        raise ValueError("原母集/调查时钟改变。")
    data, _, dividends, risks, parents = parent.load()
    data = frozen.attach_events(data, surveys)
    export(surveys, "全部84月_双期限兑现与缺失不删")
    export(data[["date", "credit_review_month", "credit_joint_state", "credit_announcement_at", "credit_cancellation_event"]], "全部日线_一次可知事件而非持续禁入")
    checks, metrics, cycle_tables, order_tables, yearly, cancellation = [], [], [], [], [], []
    recovered = recover_first()
    for period, (start, end) in parent.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            folder = parent.OUT / "accounts" / period / cost / frozen.BASELINE
            saved = {name: pd.read_parquet(folder / f"{name}.parquet") for name in parent.ACCOUNT_TABLES}
            saved["terminal"] = read(folder / "terminal.json")
            if period == "2015_2019" and cost == "BASE":
                control = candidate = recovered
                role = "RECOVERED_SAVED_FIRST_NO_FINANCIAL_REPLAY"
            else:
                muted = local.copy()
                muted["credit_cancellation_event"] = False
                control = frozen.account(muted, dividends, parents[period], risks, cost, start)
                candidate = frozen.account(local, dividends, parents[period], risks, cost, start)
                role = "REMAINING_ORIGINAL_FROZEN_ACCOUNT_COMPUTED_ONCE"
            for name in parent.ACCOUNT_TABLES[:3]:
                parent.previous.exact_saved_table(control[name], saved[name])
            checks.append({"period": period, "cost": cost, "R240_muted_event_account_exact": True, "role": role})
            verification = parent.previous.parent.verify_account(candidate)
            destination = OUT / "accounts" / period / cost / frozen.POLICY
            destination.mkdir(parents=True, exist_ok=False)
            for name in parent.ACCOUNT_TABLES:
                candidate[name].to_parquet(destination / f"{name}.parquet", index=False)
            write(destination / "terminal.json", candidate["terminal"])
            write(destination / "verification.json", verification)
            stats, years = parent.previous.controls.statistics(candidate, period, cost, frozen.POLICY)
            metrics.append({"period": period, "cost": cost, "policy": frozen.POLICY, **stats})
            for control_policy, ctrl in ((frozen.BASELINE, saved), (parent.SAVED_CORE, parent.saved_account(period, cost, parent.SAVED_CORE))):
                control_stats, _ = parent.previous.controls.statistics(ctrl, period, cost, control_policy)
                metrics.append({"period": period, "cost": cost, "policy": control_policy, **control_stats})
            cycle_tables.append(candidate["trades"].assign(period=period, cost=cost, policy=frozen.POLICY))
            order_tables.append(candidate["orders"].assign(period=period, cost=cost, policy=frozen.POLICY))
            decisions = candidate["decisions"]
            cancellation.extend({"period": period, "cost": cost, **row} for row in decisions.loc[decisions.reason.eq(frozen.EXIT_REASON)].to_dict("records"))
            yearly.extend(years)
            print(f"{period}/{cost}完整保存：净年化{stats['net_cagr']:.4%}、净夏普{stats['net_sharpe']:.6f}、pB{stats['p_times_b']:.6f}。", flush=True)
    export(pd.DataFrame(metrics), "全部12完整账户_四新与八保存对照")
    export(pd.concat(cycle_tables, ignore_index=True), "全部新账户周期与开放_费用复本非独立")
    export(pd.concat(order_tables, ignore_index=True), "全部真实订单_完整现金反馈")
    export(pd.DataFrame(cancellation), "全部取消请求_真实来源归属与执行钟")
    export(pd.DataFrame(yearly), "逐年完整净收益_缺失年不删")
    write(OUT / "control_verification.json", {"at": frozen.now(), "all_four_muted_accounts_exact": checks, "first_saved_candidate_five_tables_exact": True})
    write(OUT / "financial_raw_summary.json", {"at": frozen.now(), "registration": frozen.REGISTER, "decision": frozen.DECISION,
        "status": "COMPLETED_FIXED_EXPLORATORY_FINANCIAL_EXPERIMENT_NOT_YET_ADJUDICATED", "new_candidate_accounts": 4,
        "new_candidate_account_runs_original_plus_resume": 4, "financial_candidate_replays": 0,
        "muted_adapter_check_accounts": 4, "saved_control_accounts": 8, "new_fits": 0, "new_equity_labels": 0,
        "all_months": len(surveys), "saved_prior_surveys": int(surveys.same_instrument_prior_version_clock_valid.sum()),
        "joint_states": surveys.joint_credit_state.value_counts().to_dict(),
        "cancellation_event_months": surveys.loc[surveys.joint_credit_state.eq("DUAL_CREDIT_UNDERDELIVERY"), "month"].tolist(),
        "primary_metrics": [m for m in metrics if m["policy"] == frozen.POLICY], "archive_requests": 4,
        "archive_receipts": [read(OUT / "sources" / source[0] / "receipt.json") for source in frozen.SOURCES],
        "new_bank_reports_in_numeric_model": 0, "historical_source_first_versions_authenticated": False,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "forward_before": protocol["forward_before"],
        "necessary_economic_tests_passed": 8, "necessary_serialization_tests_passed": 2,
        "original_failure_preserved": True, "implementation_format_version": "1.0.1", "economic_change": False})
    print("原一次主用途四账户全部完成；首个已完成账本精确恢复，新增金融重算0。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="原预期兑现实验日期格式恢复")
    parser.add_argument("action", choices=("prepare", "run"))
    {"prepare": prepare, "run": run}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
