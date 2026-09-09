"""仅比较已保存账户的增量和循环区块抽样敏感性。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import block_indices, digest, interval, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_learned_panic_saved_diagnostic_v1"
CONFIG = ROOT / "config/510300_learned_panic_saved_diagnostic_v1.json"


def paths(period, cost):
    return {"COMBINATION": ROOT / "reports/research/510300_panic_learned_equal_blend_v1" / period / cost / "PANIC_LEARNED_HALF_ledger.parquet",
            "ORIGINAL_FULL": ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / cost / "REARM_RIDGE_ledger.parquet",
            "ORIGINAL_HALF": ROOT / "reports/research/510300_entry_volatility_sizing_v1" / period / cost / "ENTRY_HALF_ledger.parquet"}


def freeze():
    inputs = [Path(__file__), ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "docs/510300_LEARNED_PANIC_SAVED_DIAGNOSTIC_V1.md"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            inputs += list(paths(period, cost).values())
    write_json(CONFIG, {"registered_at": now(), "status": "POST_SELECTION_SAVED_DIAGNOSTIC_ONLY", "replications": 2000,
                       "blocks": [20, 60], "random_seed": 20260907, "annual_days": 242,
                       "frozen_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in inputs]}, exclusive=True)
    print("固定组合的保存收益诊断已登记，不产生新账户或训练。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for p in cfg["frozen_files"]:
        require(digest(ROOT / p["path"]) == p["sha256"], "保存收益诊断输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now()}, exclusive=True)
    intervals, differences, draws = [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        accounts = {cost: {name: pd.read_parquet(path) for name, path in paths(period, cost).items()} for cost in ["BASE", "STRESS"]}
        calendar = accounts["BASE"]["COMBINATION"].date
        for cost, ledgers in accounts.items():
            a = ledgers["COMBINATION"]
            for name, b in ledgers.items():
                require(pd.DatetimeIndex(b.date).equals(pd.DatetimeIndex(calendar)), "保存诊断账户日期不一致")
                if name == "COMBINATION":
                    continue
                delta = {"period": period, "cost": cost, "control": name, "final_equity_difference": a.equity.iloc[-1] - b.equity.iloc[-1],
                         "price_pnl_difference": a.price_pnl.sum() - b.price_pnl.sum(), "dividend_difference": a.dividend_recognized.sum() - b.dividend_recognized.sum(),
                         "extra_commission_and_slippage": a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum()}
                require(abs(delta["final_equity_difference"] - delta["price_pnl_difference"] - delta["dividend_difference"] + delta["extra_commission_and_slippage"]) < 1e-6,
                        "保存组合增量与价格分红费用不一致")
                differences.append(delta)
        for block in cfg["blocks"]:
            rng = np.random.default_rng(cfg["random_seed"])
            indices = np.stack([block_indices(rng, len(calendar), block) for _ in range(cfg["replications"])])
            for cost, ledgers in accounts.items():
                boot = {}
                for name, ledger in ledgers.items():
                    samples = ledger.net_return.to_numpy(float)[indices]
                    means = samples.mean(axis=1) * cfg["annual_days"]
                    vol = samples.std(axis=1, ddof=1) * np.sqrt(cfg["annual_days"])
                    sharpe = np.divide(means, vol, out=np.full(len(means), np.nan), where=vol > 1e-15)
                    boot[name] = (means, sharpe)
                for name in ["ORIGINAL_FULL", "ORIGINAL_HALF"]:
                    delta_mean = boot["COMBINATION"][0] - boot[name][0]
                    delta_sharpe = boot["COMBINATION"][1] - boot[name][1]
                    mean_ci, sharpe_ci = interval(delta_mean), interval(delta_sharpe)
                    intervals.append({"period": period, "cost": cost, "control": name, "block_days": block,
                                      "replications": cfg["replications"], "valid_sharpe_draws": int(np.isfinite(delta_sharpe).sum()),
                                      "annual_arithmetic_return_difference_low": mean_ci[0], "annual_arithmetic_return_difference_high": mean_ci[1],
                                      "sharpe_difference_low": sharpe_ci[0], "sharpe_difference_high": sharpe_ci[1],
                                      "sharpe_interval_crosses_zero": sharpe_ci[0] is not None and sharpe_ci[0] <= 0 <= sharpe_ci[1],
                                      "mean_interval_crosses_zero": mean_ci[0] is not None and mean_ci[0] <= 0 <= mean_ci[1]})
                    draws.append(pd.DataFrame({"period": period, "cost": cost, "control": name, "block_days": block,
                                               "draw": np.arange(cfg["replications"]), "annual_arithmetic_return_difference": delta_mean,
                                               "sharpe_difference": delta_sharpe}))
    pd.DataFrame(intervals).to_csv(OUT / "all_increment_intervals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(differences).to_csv(OUT / "saved_account_differences.csv", index=False, encoding="utf-8-sig")
    pd.concat(draws, ignore_index=True).to_parquet(OUT / "all_bootstrap_draws.parquet", index=False)
    result = {"completed_at": now(), "status": "POST_SELECTION_SAVED_DIAGNOSTIC_COMPLETE", "interval_records": len(intervals),
              "sharpe_intervals_crossing_zero": sum(x["sharpe_interval_crosses_zero"] for x in intervals),
              "mean_intervals_crossing_zero": sum(x["mean_interval_crosses_zero"] for x in intervals),
              "intervals": intervals, "differences": differences, "new_accounts_generated": 0, "new_model_fits": 0, "goal_achieved": False}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"区间记录": len(intervals), "夏普差区间跨零": result["sharpe_intervals_crossing_zero"],
                      "年化算术收益差区间跨零": result["mean_intervals_crossing_zero"], "主评价基础": [x for x in intervals if x["period"] == "evaluation" and x["cost"] == "BASE"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
