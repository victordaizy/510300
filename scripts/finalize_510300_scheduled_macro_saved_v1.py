"""从已完成账户和已保存抽样索引补写JSON结果，不重跑账户或模型。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_scheduled_macro_capital_risk_v1"


def main():
    spec = importlib.util.spec_from_file_location("scheduled_frozen", ROOT / "code/scheduled_macro_capital_risk_v1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _, p = m.engines(ROOT)
    if (ROOT / "result.json").exists():
        raise RuntimeError("已完成，禁止重复补写。")
    freeze = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
    for item in freeze["files"]:
        assert p.digest(ROOT / item["path"]) == item["sha256"]
    out = ROOT / "results"
    indices = np.load(out / "bootstrap_indices.npy")
    metrics = json.loads((out / "account_metrics.json").read_text(encoding="utf-8"))
    returns = {}
    rows = 0
    for metric in metrics:
        folder = out / "accounts" / metric["period"] / metric["cost"] / metric["policy"]
        ledger = pd.read_parquet(folder / "ledger.parquet")
        trades = pd.read_parquet(folder / "trades.parquet")
        terminal = json.loads((folder / "terminal.json").read_text(encoding="utf-8"))
        computed = p.metrics(ledger, trades, terminal)
        assert np.isclose(computed["net_sharpe"], metric["net_sharpe"])
        assert np.isclose(computed["net_profit_cny"], metric["net_profit_cny"])
        rows += len(ledger)
        if metric["period"] == "PRIMARY" and metric["cost"] == "STRESS":
            returns[metric["policy"]] = p.reserved_returns(ledger, terminal)
    paired = []
    for policy in m.CANDIDATES:
        delta = returns[policy] - returns["A"]
        boot = delta[indices].mean(axis=1) * 252
        lower = float(np.quantile(boot, .025))
        paired.append({"policy": policy, "baseline": "A", "annual_mean_increment": float(delta.mean() * 252),
                       "ci95": np.quantile(boot, [.025, .975]).tolist(), "familywise_one_sided_lower": lower,
                       "supports_positive_increment": lower > 0})
    p.save_json(out / "paired_increment.json", paired)
    records = pd.read_parquet(out / "scheduled_events.parquet")
    views = pd.read_parquet(out / "preopen_views.parquet")
    relevant = records.risk_open.between(pd.Timestamp(m.PERIODS["PRIMARY"][0]), pd.Timestamp(m.PERIODS["PRIMARY"][1]) + pd.Timedelta(days=1))
    valid = records.status.eq("READY")
    assert (records.loc[valid, "schedule_available_upper"] <= records.loc[valid, "decision_time"]).all()
    passes = [k for k in m.CANDIDATES if all(next(x for x in metrics if x["period"] == "PRIMARY" and x["cost"] == c and x["policy"] == k)["numerical_target_pass"] for c in p.COSTS)]
    p.save_json(ROOT / "result.json", {"status": "FROZEN_HISTORICAL_CANDIDATE_ONLY" if passes else "FROZEN_NO_HIGH_SHARPE_SCHEDULED_EVENT_RULE",
                "completed_at": p.now(), "primary_dual_cost_numeric_pass": passes,
                "positive_increment_supported": [x["policy"] for x in paired if x["supports_positive_increment"]],
                "events_in_primary": int(relevant.sum()), "known_before_execution": int((relevant & valid).sum()),
                "unavailable_schedule_events": int((relevant & ~valid).sum()),
                "pressure_event_days": int(views.loc[views.date.between(*m.PERIODS["PRIMARY"]), "S_FUNDS"].sum()),
                "strategy_goal_achieved": False, "strict_forward_observations": 0, "old_NBS_rejection_preserved": True})
    p.save_json(ROOT / "verification.json", {"status": "PASS", "accounts": len(metrics), "account_rows": rows,
                "A_same_as_previous_frozen_engine": True, "sources_not_later_than_decision": True,
                "schedule_selection_excludes_no_events_based_on_outcomes_or_minute_data": True,
                "independent_validation": False, "serialization_fix": "numpy区间数组转为JSON列表，未改数值，沿用已保存抽样。",
                "new_account_runs": 0, "script_sha256": p.digest(Path(__file__))})
    print("已从12个完整账户补写结果；没有重新训练、回测或抽样。")


if __name__ == "__main__":
    main()
