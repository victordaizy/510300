"""登记并一次描述全部保存账户的实际进入、新增信息与退出后机会序列。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research import actual_holding_information_inputs_v1 as inputs
from research import support_price_acceptance_study_v1 as previous

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_actual_holding_information_v1"
SOURCE = previous.OUT
REGISTER, RESULT = "TECH.R233", "TECH.R234"
POLICIES = (previous.CONTROL_A, previous.inputs.POLICIES[0])
DAILY = SOURCE / "results/全部3488支持信息与价格接受状态.parquet"
RECEIPTS = SOURCE / "results/全部来源首次原ETF可知_宣布与确认分开.parquet"
KEYS = previous.KEYS
read, write, digest = previous.read, previous.write, previous.digest


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def account_folder(period: str, cost: str, policy: str) -> Path:
    if policy == previous.CONTROL_A:
        return previous.parent.original.CONTROL / period / cost / policy
    return SOURCE / "accounts" / period / cost / policy


def load_accounts() -> dict:
    accounts = {}
    for period in previous.PERIODS:
        for cost in previous.COSTS:
            for policy in POLICIES:
                path = account_folder(period, cost, policy)
                account = {name: pd.read_parquet(path / f"{name}.parquet") for name in previous.ACCOUNT_TABLES}
                account["terminal"] = read(path / "terminal.json")
                accounts[(period, cost, policy)] = account
    return accounts


def source_paths() -> list[Path]:
    paths = [Path(__file__), Path(inputs.__file__), ROOT / "tests/test_actual_holding_information_v1.py",
        ROOT / "docs/510300_ACTUAL_HOLDING_INFORMATION_V1.md", DAILY, RECEIPTS, KEYS,
        SOURCE / "protocol.json", SOURCE / "implementation_v1_0_1/protocol.json",
        SOURCE / "implementation_v1_0_1/summary.json", SOURCE / "implementation_v1_0_1/post_run_diagnosis.json",
        SOURCE / "implementation_v1_0_1/next_actual_holding_information_proposal.json",
        ROOT / "reports/research/510300_point_weight_information_diagnostic_v1/protocol.json",
        ROOT / "reports/research/510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md"]
    for period in previous.PERIODS:
        for cost in previous.COSTS:
            for policy in POLICIES:
                folder = account_folder(period, cost, policy)
                paths.extend(folder / f"{name}.parquet" for name in previous.ACCOUNT_TABLES)
                paths.append(folder / "terminal.json")
    return list(dict.fromkeys(paths))


def table(name: str, frame: pd.DataFrame) -> None:
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".parquet").exists():
        raise RuntimeError("描述结果已存在，不重复覆盖。")
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("实际进入信息描述已固定，不重复登记。")
    tests = read(OUT / "tests_receipt.json")
    if tests["passed"] != 7 or tests["exit_code"] != 0 or tests["inputs_sha256"] != digest(Path(inputs.__file__)):
        raise ValueError("必要信息时序测试不完整或实现改变。")
    state = read(ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json")
    if state["latest_technical_decision"] != "TECH.R232" or state["latest_actual_financial_decision"] != "TECH.R232":
        raise ValueError("项目已由其他用途推进，停止本次登记。")
    old = read(SOURCE / "implementation_v1_0_1/summary.json")
    if old["all_economic_gates_passed"] or old["historical_stability_passed"]:
        raise ValueError("原固定配置终态不是当前已保存拒绝。")
    accounts = load_accounts()
    counts = [{"period": period, "cost": cost, "policy": policy, "saved_days": len(account["daily"]),
               "saved_cycles": len(account["trades"]), "saved_completed": int(account["trades"].status.eq("COMPLETE").sum()),
               "saved_open": int(account["trades"].status.eq("RIGHT_CENSORED").sum())}
              for (period, cost, policy), account in accounts.items()]
    if len(accounts) != 8:
        raise ValueError("原八账户范围不完整。")
    paths = source_paths()
    write(OUT / "protocol.json", {"at": previous.parent.original.now(), "study": "510300_ACTUAL_HOLDING_INFORMATION_V1",
        "registration": REGISTER, "decision": RESULT, "known_before_freeze": "R232全部金融和原A全部55压力周期已知；全历史开发，不是盲样本。",
        "question": "真实进入后新形成价格、完整周及新公开信息，如何对应原上涨/失败及退出后机会？",
        "rule_card": "docs/510300_ACTUAL_HOLDING_INFORMATION_V1.md", "policies": list(POLICIES),
        "periods": previous.PERIODS, "costs": list(previous.COSTS), "all_original_saved_accounts": counts,
        "entry_anchor": "实际进入日完整高低在收盘后固定，仅用于严格后来观察，不用于当天开盘进入。",
        "weekly_new_information": "上完整周所有实际日>=买入日，周低>进入日低、周收盘>进入日高；原R232判别仅作对照描述，不改其金融。",
        "source_clock": "原16:00观察和原公开上界，真实买入09:30；新公开、旧支持新观察、目标确认、反向和共同源分别保留。",
        "macro": "原known/公布钟/首次观察身份；每日新记录不是独立冲击，未知不填0。",
        "industry": "逐点角色/分类版本/源龄/领先身份保留；不认证进入固定组扩散，不做比例差加分。",
        "after_exit": "原每日份额为零到下一真实进入前/独立时期末；状态不读取未来盈亏或下一进入日期来改过去。",
        "necessary_tests": 7, "prefix_checks": "原19关键日八保存账户全部已有行和特征精确，盈亏标注不参与状态。",
        "new_accounts": 0, "new_financial_runs": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "financial_result": "NOT_COMPUTED", "old_financial_rejection_preserved": "TECH.R232",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "no_rescue": "不改R232周纯度/等待/新闻寿命，不据条件人数宣传策略胜率，不拼接旧账户收益。",
        "sources": [{"path": relative(path), "sha256": digest(path)} for path in paths]})
    print("R233实际进入后新增证据用途固定：八保存账户，零新金融。", flush=True)


def summarize_cycles(panel: pd.DataFrame, accounts: dict) -> pd.DataFrame:
    rows = []
    for (period, cost, policy), account in accounts.items():
        for trade in account["trades"].to_dict("records"):
            selected = panel.loc[panel.period.eq(period) & panel.cost.eq(cost) & panel.policy.eq(policy) & panel.cycle_id.eq(trade["cycle_id"])]
            if selected.empty:
                raise ValueError("实际周期缺少完整进入后的序列。")
            holding = selected.loc[selected.actual_account_phase.eq("ACTUALLY_HOLDING")]
            flat = selected.loc[selected.actual_account_phase.eq("AFTER_ACTUAL_EXIT_BEFORE_NEXT_ENTRY")]
            record = {"period": period, "cost": cost, "policy": policy, **trade,
                "actual_holding_close_rows": len(holding), "after_exit_observed_rows": len(flat),
                "post_entry_price_acceptance_while_holding": bool(holding.post_entry_price_accepted_now.any()),
                "entry_day_low_failure_while_holding": bool(holding.entry_day_low_failed_now.any()),
                "wholly_new_complete_week_acceptance_while_holding": bool(holding.post_entry_complete_week_acceptance_now.any()),
                "original_raised_week_while_holding": bool(holding.previous_week_low_and_close_raised.any()),
                "new_support_publication_while_holding": bool(holding.policy_new_support_published_after_entry_ids_today.ne("").any()),
                "old_support_first_observed_while_holding": bool(holding.policy_older_support_first_observed_after_entry_ids_today.ne("").any()),
                "target_operation_confirmation_while_holding": bool(holding.policy_target_confirmation_ids_today.ne("").any()),
                "new_week_acceptance_after_exit": bool(flat.post_entry_complete_week_acceptance_now.any()),
                "price_reacceptance_after_exit": int(flat.price_reacceptance_after_low_failure_event.sum()),
                "industry_known_holding_rows": int(holding.industry_view_allowed.sum()),
                "industry_unknown_holding_rows": int((~holding.industry_view_allowed).sum()),
                "role": "ALL_SAVED_ACTUAL_OUTCOMES_ANNOTATED_NO_SELECTION_OR_TRAINING_TARGET"}
            for field, name in (("post_entry_price_accepted_now", "first_price_acceptance_while_holding"),
                                ("entry_day_low_failed_now", "first_entry_low_failure_while_holding"),
                                ("post_entry_complete_week_acceptance_now", "first_wholly_new_week_acceptance_while_holding")):
                candidates = holding.loc[holding[field], "date"]
                record[name] = candidates.iloc[0] if len(candidates) else pd.NaT
            for name in inputs.INFORMATION:
                record[name + "_fresh_publication_events_while_holding"] = int(holding[name + "_fresh_publication_event_today_after_entry"].sum())
                record[name + "_old_record_first_observation_while_holding"] = int(holding[name + "_first_observed_today_without_new_publication"].sum())
                record[name + "_unknown_holding_rows"] = int((~holding[name + "_currently_admitted"]).sum())
            if pd.notna(trade["exit_date"]):
                sells = account["orders"].loc[account["orders"].side.eq("SELL") & account["orders"].date.eq(trade["exit_date"])]
                record["actual_exit_decision_origin"] = sells.origin.iloc[-1]
                record["actual_exit_decision_reason"] = sells.reason.iloc[-1]
            else:
                record["actual_exit_decision_origin"], record["actual_exit_decision_reason"] = pd.NaT, "NATURAL_OPEN_ENDPOINT"
            rows.append(record)
    return pd.DataFrame(rows)


def run() -> None:
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("实际进入信息用途已开始，不重复处理。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定来源或规则改变，停止本次描述。")
    write(OUT / "RUN_STARTED.json", {"at": previous.parent.original.now(), "new_financial_runs": 0})
    observed, receipts = pd.read_parquet(DAILY), pd.read_parquet(RECEIPTS)
    accounts = load_accounts()
    frames = {}
    for key, account in accounts.items():
        period, cost, policy = key
        frames[key] = inputs.sequences(observed, account, receipts, period, cost, policy, previous.PERIODS[period][1])
        print(f"{period}/{cost}/{previous.NAMES[policy]}完整实际进入与退出后序列完成：{len(frames[key])}日行。", flush=True)
    panel = pd.concat(frames.values(), ignore_index=True)
    cycles = summarize_cycles(panel, accounts)
    table("全部八账户_真实进入后的新增信息与退出后序列", panel)
    table("全部实际周期与开放_新增证据和原净结果", cycles)
    checks = []
    for cutoff in pd.read_parquet(KEYS).date:
        small = observed.loc[observed.date.le(cutoff)]
        count, pairs = 0, 0
        for key, account in accounts.items():
            period, cost, policy = key
            partial = inputs.sequences(small, account, receipts, period, cost, policy, previous.PERIODS[period][1])
            full = frames[key].loc[frames[key].date.le(cutoff)].reset_index(drop=True)
            if full.empty and partial.empty:
                continue
            pd.testing.assert_frame_equal(full, partial.reset_index(drop=True), check_exact=True)
            count += len(full)
            pairs += 1
        checks.append({"cutoff": cutoff, "saved_account_pairs_with_existing_rows": pairs,
                       "all_existing_rows_exact": count, "all_states_and_clocks_exact": True})
        print(f"{pd.Timestamp(cutoff):%Y-%m-%d}原关键日截断：{pairs}账户、{count}完整日行精确。", flush=True)
    table("原十九关键日_八账户全部已有序列前缀精确", pd.DataFrame(checks))
    coverage_rows = []
    for (period, cost, policy, phase), frame in panel.groupby(["period", "cost", "policy", "actual_account_phase"], sort=False):
        row = {"period": period, "cost": cost, "policy": policy, "phase": phase, "daily_rows": len(frame),
            "price_accepted_rows": int(frame.post_entry_price_accepted_now.sum()),
            "entry_low_failed_rows": int(frame.entry_day_low_failed_now.sum()),
            "whole_new_week_available_rows": int(frame.complete_week_all_observed_days_after_entry.sum()),
            "whole_new_week_accepted_rows": int(frame.post_entry_complete_week_acceptance_now.sum()),
            "industry_known_rows": int(frame.industry_view_allowed.sum()),
            "new_support_publication_events": int(frame.policy_new_support_published_after_entry_ids_today.ne("").sum()),
            "old_support_new_observation_events": int(frame.policy_older_support_first_observed_after_entry_ids_today.ne("").sum()),
            "target_operation_confirmation_events": int(frame.policy_target_confirmation_ids_today.ne("").sum())}
        for name in inputs.INFORMATION:
            row[name + "_known_rows"] = int(frame[name + "_currently_admitted"].sum())
            row[name + "_new_publication_events"] = int(frame[name + "_fresh_publication_event_today_after_entry"].sum())
            row[name + "_old_first_observation_events"] = int(frame[name + "_first_observed_today_without_new_publication"].sum())
        coverage_rows.append(row)
    table("全部持有与退出后_价格信息角色及未知覆盖", pd.DataFrame(coverage_rows))
    key_panel = panel.loc[panel.date.isin(pd.read_parquet(KEYS).date)]
    table("原十九关键日_全部账户当时信息状态", key_panel)
    main = cycles.loc[cycles.cost.eq("STRESS")]
    write(OUT / "summary.json", {"at": previous.parent.original.now(), "registration": REGISTER, "decision": RESULT,
        "status": "ACTUAL_HOLDING_NEW_INFORMATION_ALL_SAVED_ACCOUNTS_DESCRIPTION_COMPLETED",
        "all_original_observation_rows": len(observed), "all_policy_receipt_nodes": len(receipts),
        "saved_accounts_described": len(accounts), "all_daily_sequence_rows": len(panel),
        "all_actual_saved_cycles_including_open": len(cycles), "all_actual_saved_completed_cycles": int(cycles.status.eq("COMPLETE").sum()),
        "all_actual_saved_open_cycles": int(cycles.status.eq("RIGHT_CENSORED").sum()),
        "pressure_A_cycles": int(main.policy.eq(previous.CONTROL_A).sum()),
        "pressure_primary_cycles": int(main.policy.eq(previous.inputs.POLICIES[0]).sum()),
        "all_original_key_days": len(checks), "prefix_checks": checks,
        "necessary_tests_passed": 7, "source_files_exact": len(protocol["sources"]),
        "new_accounts": 0, "new_financial_runs": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "new_strategy_return_sharpe": "NOT_COMPUTED", "condition_group_strategy_win_rate": "NOT_COMPUTED",
        "previous_actual_financial_decision": "TECH.R232", "old_financial_rejection_preserved": True,
        "history_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False})
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("描述处理改变原来源或输入。")
    print("R234全部原A与支持原型的持仓/退出后信息完成；零新策略收益或金融重跑。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="实际持仓后新形成信息的唯一描述用途")
    parser.add_argument("action", choices=("freeze", "run"))
    action = parser.parse_args().action
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
