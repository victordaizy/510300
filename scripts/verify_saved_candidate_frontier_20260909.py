"""核对四情景前五及主历史最高方案的24条保存账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round96_20260908 import value_equal
from scripts.saved_candidate_frontier_20260909 import ROOT, OUT, FIELDS


def main():
    candidates = json.loads((OUT / "candidate_sources.json").read_text(encoding="utf-8"))
    ranked = pd.read_csv(OUT / "standard_four_scenario_ranking.csv")
    primary = pd.read_csv(OUT / "main_base_only_ranking.csv")
    selected = list(dict.fromkeys(ranked.candidate_id.head(5).tolist()+primary.candidate_id.head(1).tolist()))
    for item in json.loads((OUT / "source_files.json").read_text(encoding="utf-8")):
        require(digest(ROOT / item["path"]) == item["sha256"], "已读取保存指标发生变化")
    cfg = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    checks, cached = [], {}
    for key in selected:
        candidate = next(c for c in candidates if c["candidate_id"] == key)
        source = next(s for s in candidate["sources"] if s["earlier_source"])
        folder = ROOT / source["folder"]
        for period, rows in [("evaluation", candidate["main"]), ("earlier_diagnostic", candidate["earlier"])]:
            start, end = (cfg["evaluation_start"], cfg["data_cutoff"]) if period == "evaluation" else (cfg["earlier_start"], cfg["earlier_terminal"])
            for row in rows:
                path = folder / period / row["cost"] / f"{candidate['model']}_ledger.parquet"
                ledger = pd.read_parquet(path)
                benchmark_path = ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / row["cost"] / "BUY_HOLD_ledger.parquet"
                benchmark = pd.read_parquet(benchmark_path)
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data[data.date.between(start, end)].date)), "前列账户不是同一完整历史区间")
                require(ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == ledger.open.iloc[-1], "前列账户终点未按开盘清仓")
                prior = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
                np.testing.assert_allclose(ledger.equity/prior-1, ledger.net_return, rtol=0, atol=1e-12)
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, rtol=0, atol=1e-6)
                np.testing.assert_allclose(ledger.equity-prior, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, rtol=0, atol=1e-6)
                measured = summarize(ledger, cfg)
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-summarize(benchmark, cfg)["annualized_return"]
                for field in FIELDS:
                    require(value_equal(measured[field], row.get(field)), "前列保存账户与汇总指标不一致")
                checks.append({"candidate_id": key, "model": candidate["model"], "period": period, "cost": row["cost"], "source": str(path.relative_to(ROOT)),
                    "sha256": digest(path), "start": str(ledger.date.iloc[0]), "terminal_open": str(ledger.date.iloc[-1]), **measured})
                cached[(candidate["model"], period, row["cost"])] = ledger
    same = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            a, b = cached[("WITHIN_CYCLE_EXIT", period, cost)], cached[("MARKET_PATH_EXIT", period, cost)]
            pd.testing.assert_frame_equal(a, b)
            same.append({"period": period, "cost": cost, "left": "WITHIN_CYCLE_EXIT", "right": "MARKET_PATH_EXIT", "all_columns_identical": True})
    pd.DataFrame(checks).to_csv(OUT / "top_saved_account_checks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(same).to_csv(OUT / "identical_top_paths.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "TOP_FIVE_FOUR_SCENARIO_PLUS_MAIN_BEST_SAVED_ACCOUNTS_CHECKED",
        "selected_named_candidates": len(selected), "actual_saved_ledgers_checked": len(checks), "identical_114_117_account_pairs": len(same),
        "new_models_or_accounts": 0, "security_audit_performed": False, "reviewer_source_sha256": digest(Path(__file__))}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
