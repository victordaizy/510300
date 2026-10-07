"""接续保存序列的前缀核对，规范全NaT时间列类型，逐时间值和其他特征精确。"""
from __future__ import annotations

import argparse

import pandas as pd

from research import actual_holding_information_study_v1 as parent

ROOT, SOURCE = parent.ROOT, parent.OUT
OUT = SOURCE / "implementation_v1_0_1"
DATES = ("date", "entry_date", "first_post_entry_price_acceptance", "first_actual_entry_low_failure",
         "first_wholly_post_entry_complete_week_acceptance", "complete_week_first", "complete_week_last")
CLOCKS = tuple(parent.inputs.CLOCK_COLUMNS.values())


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def normalized(frame):
    result = frame.copy(deep=True)
    for field in DATES:
        result[field] = pd.to_datetime(result[field]).astype("datetime64[ns]")
    for field in CLOCKS:
        old = result[field]
        if isinstance(old.dtype, pd.DatetimeTZDtype):
            result[field] = old.astype(pd.DatetimeTZDtype(unit="ns", tz=old.dtype.tz))
        elif old.notna().any():
            raise ValueError("源公布钟有有限无时区值，不能当格式差异自动移动。")
        else:
            result[field] = pd.Series(pd.NaT, index=old.index, name=field, dtype="datetime64[ns, Asia/Shanghai]")
    for field in (*DATES, *CLOCKS):
        before = pd.to_datetime(frame[field], utc=True).astype("datetime64[ns, UTC]")
        after = pd.to_datetime(result[field], utc=True).astype("datetime64[ns, UTC]")
        pd.testing.assert_series_equal(before, after, check_exact=True)
    return result


def require_same_prefix(full, partial):
    if full.empty and partial.empty:
        return
    pd.testing.assert_frame_equal(normalized(full), normalized(partial), check_exact=True)


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".parquet").exists():
        raise RuntimeError("接续描述结果已有，不覆盖。")
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("时间类型接续已固定，不重复。")
    original = parent.read(SOURCE / "protocol.json")
    tests = parent.read(OUT / "tests_receipt.json")
    failure = parent.read(SOURCE / "prefix_storage_failure.json")
    if tests["passed"] != 2 or tests["exit_code"] != 0 or failure["original_process_exit_code"] != 1:
        raise ValueError("必要时间检查或原失败缺失。")
    if tests["implementation_sha256"] != parent.digest(ROOT / "research/actual_holding_information_format_v1_0_1.py"):
        raise ValueError("已测试类型接续实现改变。")
    for item in original["sources"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原冻结代码、来源或账户改变。")
    paths = [ROOT / "research/actual_holding_information_format_v1_0_1.py", ROOT / "tests/test_actual_holding_information_format_v1_0_1.py",
        SOURCE / "protocol.json", SOURCE / "RUN_STARTED.json", SOURCE / "prefix_storage_failure.json",
        SOURCE / "results/全部八账户_真实进入后的新增信息与退出后序列.parquet",
        SOURCE / "results/全部实际周期与开放_新增证据和原净结果.parquet"]
    parent.write(OUT / "protocol.json", {"at": parent.previous.parent.original.now(), "registration": parent.REGISTER, "decision": parent.RESULT,
        "implementation": "1.0.1_ALL_NAT_TIME_TYPE_COMPARISON_ONLY", "saved_original_sequence_regenerations": 0,
        "original_accounts_replayed": 0, "time_value_changes": 0, "new_fits": 0, "new_requests": 0,
        "time_handling": "在临时比较副本规范已声明时间列ns和公布钟时区；每时间值/索引精确，有限无时区公布钟仍拒绝，其他全部特征严格。",
        "original_frozen_inputs_exact": len(original["sources"]), "necessary_time_tests": 2,
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in paths]})
    print("R234全NaT时间类型接续固定，原八保存序列不重建。", flush=True)


def finish():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("接续前缀已开始，不重复。")
    protocol = parent.read(OUT / "protocol.json")
    original = parent.read(SOURCE / "protocol.json")
    for item in [*protocol["files"], *original["sources"]]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原输入或保存序列改变。")
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.previous.parent.original.now(), "saved_original_sequence_regenerations": 0})
    observed, receipts = pd.read_parquet(parent.DAILY), pd.read_parquet(parent.RECEIPTS)
    panel = pd.read_parquet(SOURCE / "results/全部八账户_真实进入后的新增信息与退出后序列.parquet")
    cycles = pd.read_parquet(SOURCE / "results/全部实际周期与开放_新增证据和原净结果.parquet")
    accounts = parent.load_accounts()
    checks = []
    for cutoff in pd.read_parquet(parent.KEYS).date:
        small = observed.loc[observed.date.le(cutoff)]
        count, pairs = 0, 0
        for (period, cost, policy), account in accounts.items():
            partial = parent.inputs.sequences(small, account, receipts, period, cost, policy, parent.previous.PERIODS[period][1])
            full = panel.loc[panel.period.eq(period) & panel.cost.eq(cost) & panel.policy.eq(policy) & panel.date.le(cutoff)].reset_index(drop=True)
            if full.empty and partial.empty:
                continue
            require_same_prefix(full, partial.reset_index(drop=True))
            count += len(full)
            pairs += 1
        checks.append({"cutoff": cutoff, "saved_account_pairs_with_existing_rows": pairs,
                       "all_existing_rows_exact": count, "all_states_and_clocks_exact": True})
        print(f"{pd.Timestamp(cutoff):%Y-%m-%d}全部{pairs}保存账户、{count}原已有行和时间值精确。", flush=True)
    table("原十九关键日_八账户全部已有序列前缀精确", pd.DataFrame(checks))
    coverage = []
    for (period, cost, policy, phase), frame in panel.groupby(["period", "cost", "policy", "actual_account_phase"], sort=False):
        row = {"period": period, "cost": cost, "policy": policy, "phase": phase, "daily_rows": len(frame),
            "price_accepted_rows": int(frame.post_entry_price_accepted_now.sum()), "entry_low_failed_rows": int(frame.entry_day_low_failed_now.sum()),
            "whole_new_week_available_rows": int(frame.complete_week_all_observed_days_after_entry.sum()),
            "whole_new_week_accepted_rows": int(frame.post_entry_complete_week_acceptance_now.sum()),
            "industry_known_rows": int(frame.industry_view_allowed.sum()),
            "new_support_publication_events": int(frame.policy_new_support_published_after_entry_ids_today.ne("").sum()),
            "old_support_new_observation_events": int(frame.policy_older_support_first_observed_after_entry_ids_today.ne("").sum()),
            "target_operation_confirmation_events": int(frame.policy_target_confirmation_ids_today.ne("").sum())}
        for name in parent.inputs.INFORMATION:
            row[name + "_known_rows"] = int(frame[name + "_currently_admitted"].sum())
            row[name + "_new_publication_events"] = int(frame[name + "_fresh_publication_event_today_after_entry"].sum())
            row[name + "_old_first_observation_events"] = int(frame[name + "_first_observed_today_without_new_publication"].sum())
        coverage.append(row)
    table("全部持有与退出后_价格信息角色及未知覆盖", pd.DataFrame(coverage))
    table("原十九关键日_全部账户当时信息状态", panel.loc[panel.date.isin(pd.read_parquet(parent.KEYS).date)])
    pressure = cycles.loc[cycles.cost.eq("STRESS")]
    parent.write(OUT / "summary.json", {"at": parent.previous.parent.original.now(), "registration": parent.REGISTER, "decision": parent.RESULT,
        "implementation": protocol["implementation"], "status": "ACTUAL_HOLDING_NEW_INFORMATION_ALL_SAVED_ACCOUNTS_DESCRIPTION_COMPLETED",
        "all_original_observation_rows": len(observed), "all_policy_receipt_nodes": len(receipts),
        "saved_accounts_described": len(accounts), "all_daily_sequence_rows": len(panel),
        "all_actual_saved_cycles_including_open": len(cycles), "all_actual_saved_completed_cycles": int(cycles.status.eq("COMPLETE").sum()),
        "all_actual_saved_open_cycles": int(cycles.status.eq("RIGHT_CENSORED").sum()),
        "pressure_A_cycles": int(pressure.policy.eq(parent.previous.CONTROL_A).sum()),
        "pressure_primary_cycles": int(pressure.policy.eq(parent.previous.inputs.POLICIES[0]).sum()),
        "all_original_key_days": len(checks), "prefix_checks": checks, "necessary_tests_passed": 7,
        "necessary_time_tests_passed": 2, "source_files_exact": len(original["sources"]),
        "saved_sequence_regenerations": 0, "original_account_replays": 0,
        "new_accounts": 0, "new_financial_runs": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "new_strategy_return_sharpe": "NOT_COMPUTED", "condition_group_strategy_win_rate": "NOT_COMPUTED",
        "previous_actual_financial_decision": "TECH.R232", "old_financial_rejection_preserved": True,
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False})
    for item in [*protocol["files"], *original["sources"]]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("接续改变来源或已保存序列。")
    print("R234全部真实持仓与机会描述终态完成，时间类型不改变实际特征或金融。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="只接续已保存真实进入信息的必要时间类型核对")
    parser.add_argument("action", choices=("freeze", "finish"))
    freeze() if parser.parse_args().action == "freeze" else finish()


if __name__ == "__main__":
    main()
