"""拆分已保存分批退出路径，不生成新策略账户，不改变已经看过的结果。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import commission, digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_partial_learned_exit_v1"
OLD = ROOT / "reports/research/510300_rearmed_session_exit_v1"
OUT = ROOT / "reports/research/510300_partial_exit_saved_diagnostic_v1"
KEY = "LEARNED_HALF_THEN_NATURAL"


def main():
    cfg = json.loads((ROOT / "config/510300_partial_learned_exit_v1.json").read_text(encoding="utf-8"))
    require((SOURCE / "result.json").is_file(), "分批退出尚未完成，不进行保存路径诊断")
    write_json(OUT / "diagnostic_registration.json", {
        "registered_at": now(), "status": "POST_RESULT_SAVED_PATH_DIAGNOSTIC", "source_result_sha256": digest(SOURCE / "result.json"),
        "questions": ["同一次已发生的减仓，如果同时卖完剩余份额，与继续持有到实际最终退出相比差多少；全部实际部分退出均纳入。",
                      "原全部退出和新分批退出的实际进入日期、全账户价格分红费用差异。",
                      "首次部分退出之前的经济账户是否保持原路径。"],
        "fixed_quantity_comparison_not_new_tradable_strategy": True, "new_accounts": 0, "new_models": 0,
        "independent_validation": False, "position_impact": 0}, exclusive=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    data = pd.read_parquet(ROOT / cfg["features"])
    increments, entry_differences, account_differences, prefixes, summaries = [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date <= cfg["earlier_terminal"]]
        for cost_id, cost in cfg["costs"].items():
            current = pd.read_parquet(SOURCE / period / cost_id / f"{KEY}_ledger.parquet")
            original = pd.read_parquet(OLD / period / cost_id / "REARM_RIDGE_ledger.parquet")
            cycles = pd.read_csv(SOURCE / period / cost_id / f"{KEY}_cycles.csv")
            old_cycles = pd.read_csv(OLD / period / cost_id / "REARM_RIDGE_cycles.csv")
            first_cut = current.loc[current.execution_action_kind.eq("LEARNED_REDUCTION") & current.filled_quantity.lt(0), "date"].min()
            require(pd.notna(first_cut), "没有实际部分退出，不执行该差额比较")
            fields = ["date", "cash", "shares", "equity", "net_return", "commission", "slippage_cost", "filled_quantity", "dividend_receivable", "status"]
            pd.testing.assert_frame_equal(current.loc[current.date < first_cut, fields].reset_index(drop=True),
                                          original.loc[original.date < first_cut, fields].reset_index(drop=True))
            prefixes.append({"period": period, "cost": cost_id, "first_partial_exit": first_cut,
                             "checked_account_days": int((current.date < first_cut).sum()), "all_economic_fields_equal_before_first_change": True})
            original_cut = original.loc[original.date.eq(first_cut)].iloc[0]
            current_cut = current.loc[current.date.eq(first_cut)].iloc[0]
            require(original_cut.filled_quantity == -current_cut.shares_before and original_cut.fill_price == current_cut.fill_price,
                    "首次分批退出不是原全部退出同日同价，不能宣称从这一动作开始变化")
            local = []
            for row in cycles[cycles.partial_exit_fills.gt(0)].itertuples():
                trades = current[current.execution_cycle_id.eq(row.cycle_id) & current.filled_quantity.lt(0)].copy()
                partial = trades[trades.execution_action_kind.eq("LEARNED_REDUCTION")]
                final = trades[~trades.execution_action_kind.eq("LEARNED_REDUCTION")]
                require(len(partial) == 1 and len(final) == 1, "本诊断只处理本次实际一笔减仓及一笔最终退出")
                cut, leave = partial.iloc[0], final.iloc[0]
                left = -int(leave.filled_quantity)
                require(int(row.entry_quantity) == -int(cut.filled_quantity) + left and left == int(cut.shares), "减仓后剩余数量与最终卖出不同")
                eligible = dividends[(dividends.record_date >= cut.date) & (dividends.record_date < leave.date) & (dividends.ex_date <= frame.date.iloc[-1])]
                price_difference = left * (float(leave.fill_price) - float(cut.fill_price))
                dividend_difference = left * float(eligible.cash_dividend_per_share.sum())
                fee_difference = float(cut.commission + leave.commission - commission(int(row.entry_quantity), float(cut.fill_price), cost))
                delta = price_difference + dividend_difference - fee_difference
                item = {"period": period, "cost": cost_id, "cycle_id": int(row.cycle_id), "entry_date": row.entry_date,
                        "partial_exit_date": cut.date, "final_exit_date": leave.date, "remaining_quantity": left,
                        "remaining_holding_intervals": int(np.flatnonzero(frame.date.eq(leave.date))[0] - np.flatnonzero(frame.date.eq(cut.date))[0]),
                        "remaining_execution_price_difference_cny": price_difference, "additional_dividend_rights_cny": dividend_difference,
                        "extra_commission_cny": fee_difference, "fixed_cycle_increment_cny": delta,
                        "increment_over_remaining_value": delta / (left * float(cut.open_price)),
                        "comparison": "实际分批退出减同日立即全部卖出；固定本次进入、份额和实际终点，不是新策略账户"}
                local.append(item)
                increments.append(item)
            same_dates = set(cycles.entry_date) & set(old_cycles.entry_date)
            for label, source, other in [("原进入日期在新账户中未出现", old_cycles, cycles), ("新进入日期在原账户中未出现", cycles, old_cycles)]:
                for row in source[~source.entry_date.isin(other.entry_date)].to_dict("records"):
                    entry_differences.append({"period": period, "cost": cost_id, "difference": label,
                                              **{key: row[key] for key in ["entry_date", "exit_date", "entry_quantity", "net_profit_cny"]}})
            difference = {"period": period, "cost": cost_id,
                          "final_equity_difference": float(current.equity.iloc[-1] - original.equity.iloc[-1]),
                          "price_pnl_difference": float(current.price_pnl.sum() - original.price_pnl.sum()),
                          "dividend_difference": float(current.dividend_recognized.sum() - original.dividend_recognized.sum()),
                          "extra_commission_and_slippage": float(current.commission.sum() + current.slippage_cost.sum() - original.commission.sum() - original.slippage_cost.sum())}
            require(abs(difference["final_equity_difference"] - difference["price_pnl_difference"] - difference["dividend_difference"] + difference["extra_commission_and_slippage"]) < 1e-6,
                    "全账户差额无法由价格、分红、费用核对")
            account_differences.append(difference)
            summaries.append({"period": period, "cost": cost_id, "partial_exit_cycles": len(local),
                              "positive_remaining_increment_cycles": sum(x["fixed_cycle_increment_cny"] > 0 for x in local),
                              "negative_remaining_increment_cycles": sum(x["fixed_cycle_increment_cny"] < 0 for x in local),
                              "sum_fixed_cycle_increment_cny": sum(x["fixed_cycle_increment_cny"] for x in local),
                              "sum_remaining_execution_price_difference_cny": sum(x["remaining_execution_price_difference_cny"] for x in local),
                              "sum_additional_dividend_rights_cny": sum(x["additional_dividend_rights_cny"] for x in local),
                              "sum_extra_commission_cny": sum(x["extra_commission_cny"] for x in local),
                              "new_completed_cycles": len(cycles), "old_completed_cycles": len(old_cycles),
                              "common_entry_dates": len(same_dates),
                              "old_entry_dates_absent_in_new": len(set(old_cycles.entry_date) - set(cycles.entry_date)),
                              "new_entry_dates_absent_in_old": len(set(cycles.entry_date) - set(old_cycles.entry_date))})
    for filename, rows in [("per_partial_cycle_increment.csv", increments), ("entry_date_differences.csv", entry_differences),
                           ("account_differences.csv", account_differences), ("prefix_checks.csv", prefixes), ("summary.csv", summaries)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": "510300_PARTIAL_EXIT_SAVED_DIAGNOSTIC_V1", "completed_at": now(), "status": "COMPLETED_POST_RESULT_FIXED_CYCLE_DIAGNOSTIC",
              "summaries": summaries, "account_differences": account_differences, "prefix_checks": prefixes,
              "saved_partial_cycle_rows": len(increments), "entry_date_difference_rows": len(entry_differences),
              "new_accounts": 0, "new_models": 0, "goal_achieved": False,
              "causal_limit": "固定实际新路径的事后差额不包含释放现金后的再次进入和后续份额变化，不能当另一套可交易账户或完整因果分解。"}
    write_json(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
