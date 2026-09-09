"""仅比较保存账户，分解费用和持仓损益并估计连续区块区间。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_rearmed_session_exit_v1"
OUT = ROOT / "reports/research/510300_learned_rearm_saved_diagnostic_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    write_json(OUT / "diagnostic_protocol.json", {"recorded_at": now(), "scope": "POST_SELECTED_SAVED_ACCOUNT_DIAGNOSTIC",
               "target": "REARM_RIDGE", "comparators": ["REARM_NONE", "D60_INTRA__CLOSE_NEXT_OPEN"],
               "blocks": [20, 60], "repetitions": 2000, "seed": 20260907, "multiple_selection_adjusted": False,
               "new_models_fit": 0, "new_accounts_generated": 0}, exclusive=True)
    decomposition, intervals, coverage, samples = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            new = pd.read_parquet(SOURCE / period / cost / "REARM_RIDGE_ledger.parquet")
            decisions = pd.read_parquet(SOURCE / period / cost / "REARM_RIDGE_decisions.parquet")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost, "holding_decisions": len(holding),
                             "model_available": int((holding.learning_status == "PREDICTION_AVAILABLE").sum()),
                             "no_model": int((holding.learning_status != "PREDICTION_AVAILABLE").sum())})
            for comparator in ["REARM_NONE", "D60_INTRA__CLOSE_NEXT_OPEN"]:
                base = pd.read_parquet(SOURCE / period / cost / f"{comparator}_ledger.parquet")
                require(pd.DatetimeIndex(new.date).equals(pd.DatetimeIndex(base.date)), "保存账户日期不同")
                row = {"period": period, "cost": cost, "comparator": comparator,
                       "terminal_equity_difference": float(new.equity.iloc[-1] - base.equity.iloc[-1]),
                       "price_pnl_difference": float(new.price_pnl.sum() - base.price_pnl.sum()),
                       "dividend_difference": float(new.dividend_recognized.sum() - base.dividend_recognized.sum()),
                       "additional_commission": float(new.commission.sum() - base.commission.sum()),
                       "additional_slippage": float(new.slippage_cost.sum() - base.slippage_cost.sum())}
                error = row["terminal_equity_difference"] - row["price_pnl_difference"] - row["dividend_difference"] + row["additional_commission"] + row["additional_slippage"]
                require(abs(error) < 1e-6, "保存账户损益分解不平")
                decomposition.append(row)
                difference = new.net_return.to_numpy(float) - base.net_return.to_numpy(float)
                for block in [20, 60]:
                    rng = np.random.default_rng(20260907 + block)
                    n = len(difference)
                    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
                    indexes = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
                    simulated = difference[indexes].mean(axis=1) * cfg["annual_days"]
                    lo, hi = np.quantile(simulated, [.025, .975])
                    intervals.append({"period": period, "cost": cost, "comparator": comparator, "block": block,
                                      "annualized_arithmetic_increment": float(difference.mean() * cfg["annual_days"]),
                                      "interval_95_low": float(lo), "interval_95_high": float(hi),
                                      "interval_crosses_zero": bool(lo <= 0 <= hi), "selection_adjusted": False})
                    samples.append(pd.DataFrame({"period": period, "cost": cost, "comparator": comparator, "block": block,
                                                 "repetition": np.arange(2000), "annualized_arithmetic_increment": simulated}))
    pd.DataFrame(decomposition).to_csv(OUT / "account_difference_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(intervals).to_csv(OUT / "increment_intervals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUT / "model_coverage.csv", index=False, encoding="utf-8-sig")
    pd.concat(samples, ignore_index=True).to_parquet(OUT / "resampled_statistics.parquet", index=False)
    result = {"completed_at": now(), "status": "SAVED_ACCOUNT_DIAGNOSTIC_COMPLETE", "decomposition": decomposition,
              "intervals": intervals, "model_coverage": coverage, "independent_validation": "NOT_ESTABLISHED",
              "new_models_fit": 0, "new_accounts_generated": 0, "goal_achieved": False}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"分解": decomposition, "二十日区间": [x for x in intervals if x["block"] == 20], "模型覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
