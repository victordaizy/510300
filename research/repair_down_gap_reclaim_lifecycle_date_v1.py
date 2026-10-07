"""保留R186首次日期列错误，按原固定口径恢复纯保存归因；不改冻结原代码。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.down_gap_reclaim_lifecycle_attribution_v1 import (
    OUT, FINANCIAL, EXPLANATION, PRIMARY, PERIODS, COSTS, CASE_IDS, LEDGERS,
    KNOWN_TABLE, MAP_TABLE, CASE_TABLE, alignments_and_coverage, saved_cycle, table, groups, charts,
    normalize_dividends, WEIGHT, STATE, read, write_json, digest, now, require)


def freeze():
    require(not (OUT / "repair_protocol.json").exists(), "日期列恢复已经登记。")
    require((OUT / "RUN_STARTED.json").exists() and not (OUT / "summary.json").exists()
            and not (OUT / "results").exists(), "恢复只适用于首次未保存结果的日期列错误。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "原冻结代码或来源改变。")
    source_path = ROOT / "research/down_gap_reclaim_lifecycle_attribution_v1.py"
    source_copy = OUT / "initial_source_at_date_column_failure.py"
    with source_copy.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(source_path.read_text(encoding="utf-8"))
    require(digest(source_copy) == digest(source_path), "首次代码副本不同。")
    failure = {"at": now(), "status": "FAILED_FIRST_DIAGNOSTIC_DATE_MOVED_TO_INDEX",
               "observed_exit_code": 1, "exception": "AttributeError: 'Series' object has no attribute 'date'",
               "location": "saved_cycle: gap_birth_date=parent.date; quotes originally set_index(date) dropped the column",
               "new_accounts": 0, "saved_diagnostic_tables_before_repair": 0,
               "original_frozen_source_unchanged": True, "first_run_marker_preserved": True}
    write_json(OUT / "initial_execution_failure.json", failure, exclusive=True)
    paths = [Path(__file__), source_path, source_copy, OUT / "protocol.json", OUT / "RUN_STARTED.json",
             OUT / "initial_execution_failure.json"]
    write_json(OUT / "repair_protocol.json", {
        "at": now(), "status": "REGISTERED_DATE_COLUMN_REPRESENTATION_REPAIR_ONLY",
        "reason": "日期被设为索引后默认删除列，首次周期出生日期输出失败；保留同值日期列并保留原索引。",
        "only_effective_change": "quotes=known.set_index(date,drop=False); assert date column equals original date index",
        "original_script_and_protocol_unchanged": True, "frozen_groups_population_policy_or_wealth_formulas_changed": False,
        "resume_body": "原run中known读取之后的保存归因正文原样复用，除日期列同值保留及summary修复标记；首个失败标记不覆盖，新恢复标记单独保存。",
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("首次日期列错误、代码和原冻结保存；只登记日期列同值恢复。", flush=True)


def run():
    require(not (OUT / "REPAIR_RUN_STARTED.json").exists() and not (OUT / "summary.json").exists(),
            "日期列恢复已开始，不重启。")
    repair = read(OUT / "repair_protocol.json")
    for source in repair["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "恢复版本或原首次证据改变。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "原冻结来源改变。")
    write_json(OUT / "REPAIR_RUN_STARTED.json", {"at": now(), "new_accounts": 0,
               "first_failed_run_marker_overwritten": False, "repair_protocol_sha256": digest(OUT / "repair_protocol.json")}, exclusive=True)
    known = pd.read_parquet(EXPLANATION / f"results/{KNOWN_TABLE}.parquet")
    quotes = known.set_index("date", drop=False)
    require(np.array_equal(quotes["date"].to_numpy(), quotes.index.to_numpy()), "保留日期列与原索引不一致。")
    require(quotes.index.is_unique, "保存原点日期不唯一。")
    episodes = pd.read_parquet(EXPLANATION / f"results/{MAP_TABLE}.parquet")
    cases = pd.read_parquet(EXPLANATION / f"results/{CASE_TABLE}.parquet")
    dividends = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    aligned, coverage = alignments_and_coverage(known, episodes)
    require(aligned.reclaim_index.nunique() == 60 and len(coverage) == 61 and int(coverage.admitted.sum()) == 49, "原事件或分段总体不同。")
    cycle_rows, holding_rows, wealth_rows, scenarios = [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = FINANCIAL / f"results/accounts/{period}/{cost}/{PRIMARY}"
            account = {name: pd.read_parquet(folder / f"{name}.parquet") for name in LEDGERS}
            for cycle in account["trades"].itertuples(index=False):
                row, holding, wealth = saved_cycle(cycle, account, quotes, dividends, period, cost)
                cycle_rows.append(row)
                holding_rows.extend(holding)
                wealth_rows.extend(wealth)
            scenarios.append({"period": period, "cost": cost, "cycles": len(account["trades"]),
                              "completed": int(account["trades"].status.eq("COMPLETE").sum()),
                              "open": int(account["trades"].status.ne("COMPLETE").sum()), "original_orders": len(account["orders"]),
                              "original_decisions": len(account["decisions"]), "all_original_rejections": len(account["rejections"])})
    cycles, holding, wealth = pd.DataFrame(cycle_rows), pd.DataFrame(holding_rows), pd.DataFrame(wealth_rows)
    require(len(cycles) == 102 and int(cycles.status.eq("COMPLETE").sum()) == 100, "全部保存周期数量不同。")
    require(cycles.loc[cycles.status.ne("COMPLETE"), "actual_net_return"].isna().all(), "开放周期被赋予完成收益。")
    table("全部60首次收复_原上涨位置仅事后对齐", aligned)
    table("原61分段及49上涨_两种原区间覆盖不筛选", coverage)
    table("全部102周期_原锚固定线失效及真实资金时钟", cycles)
    table("全部持有原点_固定线新缺口及原决定核对", holding)
    table("全部周期逐日库存现金股息及收盘财富", wealth)
    grouped = groups(cycles)
    table("全部六组两时期两费用_最高财富与最终结果仅解释", grouped)
    figures = charts(known, cases, cycles, holding, wealth)
    complete = cycles.loc[cycles.status.eq("COMPLETE")]
    pressure = complete.loc[complete.cost.eq("STRESS")]
    diagnostic_rows = []
    for scenario in scenarios:
        selected = complete.loc[complete.period.eq(scenario["period"]) & complete.cost.eq(scenario["cost"])]
        winners = selected.loc[selected.actual_net_pnl.gt(0)]
        largest = winners.loc[winners.actual_net_pnl.idxmax()] if len(winners) else None
        net, positive = float(selected.actual_net_pnl.sum()), float(winners.actual_net_pnl.sum())
        diagnostic_rows.append({**scenario, "completed_positive_close_then_loss": int(selected.loss_with_previously_positive_holding_close.astype(bool).sum()),
                                "risk_reduction_origins": int(selected.risk_reduction_origins.sum()),
                                "final_open_gap_gross_cny": float(selected.final_open_gap_gross_cny.sum()),
                                "post_final_origin_dividend_cny": float(selected.dividend_accrual_after_final_origin_cny.sum()),
                                "final_exit_friction_cny": float((selected.final_sell_commission+selected.final_sell_slippage).sum()),
                                "completed_pnl_cny": net, "close_floor_exits": int(selected.final_exit_reason.eq("DOWN_GAP_RECLAIM_FAILED_KNOWN_CLOSE").sum()),
                                "new_down_gap_exits": int(selected.final_exit_reason.eq("DOWN_GAP_RECLAIM_NEW_DOWN_GAP").sum()),
                                "all_final_exit_reason_counts": selected.final_exit_reason.value_counts().to_dict(),
                                "positive_close_final_win": int(selected.final_result_group.eq("POSITIVE_HOLDING_CLOSE_FINAL_WIN").sum()),
                                "never_positive_close_final_loss": int(selected.final_result_group.eq("NONPOSITIVE_HOLDING_CLOSE_FINAL_LOSS").sum()),
                                "largest_positive_cycle_origin": largest.entry_origin if largest is not None else None,
                                "largest_positive_cycle_pnl_cny": float(largest.actual_net_pnl) if largest is not None else np.nan,
                                "largest_positive_share_of_positive_pnl": float(largest.actual_net_pnl)/positive if largest is not None else np.nan,
                                "largest_positive_share_of_net_pnl": float(largest.actual_net_pnl)/net if largest is not None and net > 0 else np.nan,
                                "exit_identity_max_abs_error": float(selected.exit_clock_identity_error_cny.abs().max())})
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "诊断运行期间来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R186", "implementation_repair": "DATE_COLUMN_RETAINED_WITHOUT_CHANGING_INDEX_OR_QUOTES", "status": "COMPLETED_ALL_SAVED_RECLAIM_COVERAGE_FIXED_FAILURE_AND_WEALTH_IDENTITIES",
        "all_first_reclaims": 60, "episode_alignment_rows": len(aligned), "unmatched_reclaims_preserved": int(aligned.episode_id.eq(-1).sum()),
        "original_episode_rows": len(coverage), "original_admitted_waves": int(coverage.admitted.sum()),
        "admitted_without_bottom_to_peak_reclaim": int((coverage.admitted & coverage.bottom_to_peak_reclaims.eq(0)).sum()),
        "admitted_without_confirmation_to_peak_reclaim": int((coverage.admitted & coverage.confirmation_to_peak_reclaims.eq(0)).sum()),
        "all_actual_cycle_rows": len(cycles), "completed_cycles": len(complete), "open_cycles": int(cycles.status.ne("COMPLETE").sum()),
        "holding_origins_verified": len(holding), "cycle_close_wealth_rows": len(wealth), "complete_exit_identities_verified": len(complete),
        "retrospective_group_cells": len(grouped), "all_scenario_diagnostics": diagnostic_rows, "charts": figures,
        "pressure_complete_cycles": len(pressure), "pressure_positive_holding_close_then_loss": int(pressure.loss_with_previously_positive_holding_close.astype(bool).sum()),
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "necessary_new_tests": 0,
        "frozen_sources": len(protocol["sources"]), "latest_financial_strategy_decision_preserved": "TECH.R185",
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED", "goal_achieved": False,
    }, exclusive=True)
    print(f"R186全60收复、102周期、{len(holding)}持有原点及100退出财富身份完成；0新账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="首次日期列错误的隔离保存归因恢复。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
