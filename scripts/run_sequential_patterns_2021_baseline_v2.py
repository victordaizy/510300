"""按用户新范围建立2021—2026原规则基线；不放宽任何原参数。"""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_sequential_patterns_regime_v1"
OUT = ROOT / "reports/research/510300_sequential_patterns_2021_2026_v2"
sys.path.insert(0, str(PARENT / "code"))
from sequential_patterns_regime_v1 import POLICIES, COSTS, account, digest, metrics, now, save_csv, save_json


def main():
    if (OUT / "baseline/comparison.csv").exists():
        raise RuntimeError("原规则基线已运行，禁止重复覆盖。")
    archive = ROOT / "deliverables/510300_连续形态与状态启停_V1_GPT审阅_20260924.zip"
    expected = "909f7d43426b0de12bdade3e79136342573ee4297aa587eda6a8e9fd381a000d"
    assert digest(archive) == expected, "V1交付身份改变"
    mapping = {
        "inputs/features.parquet": PARENT / "results/features.parquet",
        "inputs/signals.parquet": PARENT / "results/signals.parquet",
        "inputs/labels.parquet": PARENT / "results/event_labels.parquet",
        "inputs/controls.parquet": PARENT / "results/controls.parquet",
        "inputs/v1_decisions.parquet": PARENT / "results/daily_decisions.parquet",
        "inputs/dividends.csv": PARENT / "inputs/dividends.csv",
        "inputs/prices.parquet": PARENT / "inputs/prices.parquet",
        "inputs/calendar.csv": PARENT / "inputs/calendar.csv",
        "inputs/historical_calendar.csv": PARENT / "inputs/historical_calendar.csv",
        "inputs/source_manifest_v1.json": PARENT / "source_manifest.json",
        "inputs/parent_summary.json": PARENT / "results/summary.json",
        "inputs/parent_protocol.md": PARENT / "protocol.md",
        "inputs/parent_freeze.json": PARENT / "freeze.json",
        "code/sequential_patterns_regime_v1.py": PARENT / "code/sequential_patterns_regime_v1.py",
        "code/run_sequential_patterns_2021_baseline_v2.py": Path(__file__),
    }
    for target, source in mapping.items():
        path = OUT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    start = int(d.index[d.date >= "2021-01-01"][0])
    end = len(d) - 1
    freeze = {"frozen_at": now(), "user_request": "如果继续放宽，21年到26年",
              "purpose": "先完成2021—2026的原规则同口径基线；放宽部分另定协议，不混入本阶段。",
              "start": d.date.iloc[start], "end": d.date.iloc[end], "initial_capital": 200000,
              "capital_reset_at_start": True, "earlier_mature_history_allowed_for_gate_initialization": True,
              "parent_zip_sha256": expected, "reused_history_not_independent_oos": True,
              "annual_trade_minimum": None, "target_net_cagr": .10, "target_net_sharpe": 1.2,
              "max_drawdown_target": .10,
              "hashes": {rel: digest(OUT / rel) for rel in mapping}}
    save_json(OUT / "baseline_freeze.json", freeze)
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    sig = pd.read_parquet(OUT / "inputs/signals.parquet")
    dec = pd.read_parquet(OUT / "inputs/v1_decisions.parquet")
    rows = []
    for cost in COSTS:
        for policy in (*POLICIES, "BUY_HOLD", "CASH"):
            ledger, trades, rejected, terminal = account(d, div, sig, dec, start, end, policy, cost)
            folder = OUT / "baseline/accounts" / cost / policy
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "daily.parquet", index=False)
            save_csv(folder / "daily.csv", ledger)
            save_csv(folder / "trades.csv", trades if len(trades) else pd.DataFrame(columns=["signal_id", "entry_idx", "exit_idx", "entry_date", "exit_date", "net_pnl", "holding_sessions"]))
            save_csv(folder / "rejected.csv", rejected if len(rejected) else pd.DataFrame(columns=["date", "signal_id", "reason"]))
            m = metrics(ledger, trades, terminal)
            save_json(folder / "metrics.json", m)
            rows.append(dict(variant="V1_UNCHANGED", cost=cost, policy=policy, **m))
    result = pd.DataFrame(rows)
    save_csv(OUT / "baseline/comparison.csv", result)
    save_json(OUT / "baseline/receipt.json", {"completed_at": now(), "new_accounts": 12,
               "new_fits": 0, "new_market_downloads": 0, "new_parameter_searches": 0,
               "period": [d.date.iloc[start], d.date.iloc[end]], "current_view": "NO_VIEW"})
    print(result[["cost", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles"]].to_string(index=False))


if __name__ == "__main__":
    main()
