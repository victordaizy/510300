"""执行用户新增的每完整自然年至少五次持仓周期要求。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import sparse_event_training_core_v3 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_node_joint_payoff_training_v1"


def compute(out):
    previous_accounts = pd.read_csv(out / "inputs/frequency_reference_accounts.csv")
    previous_cycles = pd.read_csv(out / "inputs/frequency_reference_cycles.csv")
    current = pd.read_csv(out / "results/账户联合指标.csv")
    current_cycles = pd.read_csv(out / "results/机会成交账簿.csv")
    new_accounts = pd.DataFrame({
        "account_id": current.apply(lambda r: f"CURRENT_JOINT|{r['model']}|{r['policy']}|{r['scenario']}", axis=1),
        "source_family": "CURRENT_JOINT", "model": current.model, "period": "evaluation", "cost": current.scenario,
        "policy": current.policy, "benchmark": current.policy == "BENCHMARK", "start": current.start, "end": current.end,
        "full_calendar_net_sharpe": current.net_sharpe, "max_drawdown_magnitude": current.max_drawdown,
    })
    new_cycles = pd.DataFrame({"account_id": current_cycles.apply(lambda r: f"CURRENT_JOINT|{r['model']}|{r['policy']}|{r['scenario']}", axis=1),
                               "entry_date": current_cycles.entry_date, "exit_date": current_cycles.exit_date})
    accounts = pd.concat([previous_accounts[new_accounts.columns], new_accounts], ignore_index=True)
    cycles = pd.concat([previous_cycles[new_cycles.columns], new_cycles], ignore_index=True)
    market = pd.read_parquet(out / "inputs/market.parquet")
    calendar = pd.to_datetime(market.date).sort_values().drop_duplicates()
    first_year, last_year = int(calendar.dt.year.min()), int(calendar.dt.year.max())
    year_ranges = calendar.groupby(calendar.dt.year).agg(["min", "max"])
    cycles["entry_year"] = pd.to_datetime(cycles.entry_date).dt.year
    counts = cycles.groupby(["account_id", "entry_year"]).size().to_dict()
    years, reviews = [], []
    for account in accounts.to_dict("records"):
        start, end = pd.Timestamp(account["start"]), pd.Timestamp(account["end"])
        full_counts = []
        for year in range(start.year, end.year + 1):
            complete = bool(first_year < year < last_year and start <= year_ranges.loc[year, "min"] and end >= year_ranges.loc[year, "max"])
            n = int(counts.get((account["account_id"], year), 0))
            years.append({"account_id": account["account_id"], "model": account["model"], "source_family": account["source_family"],
                          "period": account["period"], "policy": account["policy"], "cost": account["cost"], "year": year,
                          "complete_calendar_year": complete, "complete_cycles_entered": n,
                          "at_least_five": n >= 5 if complete else None,
                          "year_status": "PASS" if complete and n >= 5 else "FAIL" if complete else "PARTIAL_YEAR_NOT_ANNUALIZED"})
            if complete:
                full_counts.append(n)
        frequency_pass = bool(full_counts and min(full_counts) >= 5)
        numerical_pass = bool(pd.notna(account["full_calendar_net_sharpe"]) and account["full_calendar_net_sharpe"] >= 1.3 and account["max_drawdown_magnitude"] <= .1)
        reviews.append({**account, "full_calendar_years": len(full_counts), "minimum_annual_complete_cycles": min(full_counts) if full_counts else None,
                        "annual_frequency_requirement": 5, "annual_frequency_pass": frequency_pass if full_counts else None,
                        "annual_frequency_status": "PASS" if frequency_pass else "FAIL" if full_counts else "NO_COMPLETE_YEAR",
                        "sharpe_drawdown_numeric_pass": numerical_pass,
                        "joint_numeric_and_annual_frequency_pass": numerical_pass and frequency_pass})
    yearly, review = pd.DataFrame(years), pd.DataFrame(reviews)
    old_main = review.loc[(review.source_family == "HISTORIC_SELECTED_14") & (review.period == "evaluation")]
    summary = {"minimum_cycles_each_full_calendar_year": 5, "counting_unit": "一次完整持仓周期，按入场年归属，买和卖不拆成两次。",
               "partial_year_rule": "首尾不完整年另列原次数，不年化凑够五次，也不据此断言全年未达标。",
               "reviewed_account_records": len(review), "full_year_record_count": int(yearly.complete_calendar_year.sum()),
               "all_three_requirements_pass_records": int(review.joint_numeric_and_annual_frequency_pass.sum()),
               "old_main_models_reviewed": int(old_main.model.nunique()),
               "old_main_models_with_all_full_years_at_least_five": int(old_main.loc[old_main.annual_frequency_status == "PASS", "model"].nunique()),
               "old_main_minimum_year_count_range": [int(old_main.minimum_annual_complete_cycles.min()), int(old_main.minimum_annual_complete_cycles.max())],
               "applied_after_this_round_training": True, "new_fits": 0, "new_accounts": 0, "new_downloads": 0}
    return yearly, review, summary


def update(out):
    core.require(not (out / "annual_frequency_requirement.json").exists(), "新频率要求已经记录，不覆盖。")
    prior = ROOT / "reports/research/510300_sparse_node_joint_quality_v1"
    sources = {}
    for original, name in [("results/完整账户联合指标.csv", "frequency_reference_accounts.csv"), ("results/统一交易周期.csv", "frequency_reference_cycles.csv")]:
        source, destination = prior / original, out / "inputs" / name
        shutil.copyfile(source, destination)
        sources[destination.relative_to(out).as_posix()] = {"source": str(source.relative_to(ROOT)), "sha256": core.digest(destination)}
    authority = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    contract_path = ROOT / "config/510300_sparse_node_training_contract_v1.json"
    before = core.load(authority)
    old_contract = core.load(contract_path)
    core.save(out / "inputs/mandate_before_annual_frequency.json", before)
    core.save(out / "inputs/contract_before_annual_frequency.json", old_contract)
    changed_at = core.now()
    before.update({"as_of_date": changed_at[:10], "latest_user_instruction": "一年至少五次交易",
                   "opportunities_per_year_not_quota": False,
                   "minimum_complete_cycles_each_full_calendar_year": 5,
                   "annual_frequency_counting": "每个完整自然年按入场年统计完整持仓周期，一买一卖计一次；不使用多年平均替代逐年要求。",
                   "partial_year_frequency_assessment": "另列已发生次数，尚不据不完整区间认定全年通过或失败。",
                   "annual_frequency_revision_at": changed_at,
                   "annual_frequency_revision_receipt": "reports/research/510300_if_node_joint_payoff_training_v1/annual_frequency_requirement.json"})
    old_contract.update({"latest_user_instruction": "一年至少五次交易",
                         "frequency": "每个完整自然年至少五个完整持仓周期，按入场年归属；各年逐一满足。",
                         "minimum_complete_cycles_each_full_calendar_year": 5,
                         "frequency_revision_at": changed_at,
                         "frequency_evaluation": "不满足该频率的策略不达标；高频年份不能抵消低频年份。"})
    core.save(authority, before)
    core.save(contract_path, old_contract)
    core.save(out / "mandate_after_annual_frequency.json", before)
    core.save(out / "node_contract_after_annual_frequency.json", old_contract)
    core.save(out / "annual_frequency_requirement.json", {"recorded_at": changed_at, "user_instruction": "一年至少五次交易",
        "supersedes": "一年四五次可接受但不是交易配额。", "minimum_complete_cycles_each_full_calendar_year": 5,
        "kept": {"capital_cny": 200000, "target_net_sharpe": 1.3, "maximum_drawdown": .1, "assets": ["510300.SH", "CASH_CNY"], "new_collection_enabled": False},
        "training_was_already_completed": True, "counting_does_not_modify_saved_models_or_orders": True})
    for name in ["results/账户联合指标.csv", "results/机会成交账簿.csv", "inputs/market.parquet"]:
        sources[name] = {"source": "本轮既有文件", "sha256": core.digest(out / name)}
    shutil.copyfile(Path(__file__), out / "code/annual_node_frequency_requirement_v1.py")
    sources["code/annual_node_frequency_requirement_v1.py"] = {"source": "research/annual_node_frequency_requirement_v1.py", "sha256": core.digest(Path(__file__))}
    core.save(out / "annual_frequency_sources.json", sources)
    yearly, review, summary = compute(out)
    core.export(out / "results/逐年完整持仓周期.csv", yearly)
    core.export(out / "results/频率夏普回撤联合评价.csv", review)
    core.save(out / "annual_frequency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def verify(out):
    for name, spec in core.load(out / "annual_frequency_sources.json").items():
        core.require(core.digest(out / name) == spec["sha256"], "频率复算来源变化。")
    yearly, review, summary = compute(out)
    for frame, name in [(yearly, "逐年完整持仓周期.csv"), (review, "频率夏普回撤联合评价.csv")]:
        saved = pd.read_csv(out / "results" / name)
        for column in ["at_least_five", "annual_frequency_pass"]:
            if column in frame:
                # 不完整年份的None和CSV空单元格均表示未知，不能填成False。
                frame[column] = frame[column].astype("boolean")
                saved[column] = saved[column].astype("boolean")
        pd.testing.assert_frame_equal(frame, saved, check_dtype=False, rtol=1e-10, atol=1e-12)
    core.require(summary == core.load(out / "annual_frequency_summary.json"), "频率摘要不同。")
    return {"status": "PASS_SAVED_ANNUAL_FREQUENCY_RECOMPUTATION", **summary}


def main():
    parser = argparse.ArgumentParser(description="用户每年至少五次交易要求的逐年测量。")
    parser.add_argument("command", choices=["update", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "update":
        update(args.root)
    else:
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
