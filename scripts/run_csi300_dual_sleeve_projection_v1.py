"""评估双袖带父目标向2万元三仓投影的早期与近期表现。"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, build_factor_panel, build_targets, summarize  # noqa: E402
from research.csi300_dual_sleeve_projection_v1 import project_parent_targets  # noqa: E402
from research.small_account_cross_sectional import load_full_component_history, run_small_account_open_backtest  # noqa: E402
from scripts.freeze_csi300_dual_sleeve_projection_v1 import CONFIG_FILE, FROZEN_FILES, MANIFEST_FILE, sha256, tree_sha256  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import _costs, _json_safe, _paired_bootstrap, build_external_breadth  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402


def verify(contract: dict) -> dict:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_DUAL_PERIOD_PROJECTION_READ":
        raise RuntimeError("双时段投影尚未冻结")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    for relative, expected in manifest["input_files"].items():
        if sha256(ROOT / relative) != expected:
            changed.append(relative)
    history = ROOT / contract["inputs"]["recent_component_history_cache"]
    if tree_sha256(history) != manifest["recent_component_history_tree"]:
        changed.append(str(history))
    if changed:
        raise RuntimeError(f"冻结后变化：{changed}")
    return manifest


def run_period(
    panel: pd.DataFrame,
    execution: pd.DataFrame,
    index: pd.DataFrame,
    breadth: pd.DataFrame,
    benchmark_raw: pd.DataFrame,
    period: dict,
    contract: dict,
    parent: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parent_local = copy.deepcopy(parent)
    parent_local["schedule"]["start_date"] = period["start"]
    parent_local["schedule"]["end_date"] = period["end"]
    parent_local["schedule"]["rebalance_every_trading_days"] = int(contract["periods"]["rebalance_every_trading_days"])
    factors = build_factor_panel(panel, index, breadth)
    parent_targets = build_targets(factors, parent_local)
    targets = project_parent_targets(parent_targets, int(contract["projection"]["holdings"]))
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    calendar = pd.DatetimeIndex(sorted(panel.loc[panel["date"].between(start, end), "date"].unique()))
    initial = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "base_slippage_bps_per_leg"), gap)
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "stress_slippage_bps_per_leg"), gap)
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial)
    base, stress = summarize(base_ledger, base_trades, benchmark, initial), summarize(stress_ledger, stress_trades, benchmark, initial)
    boot = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    report = {
        "start": period["start"], "end": period["end"], "parent_signal_dates": int(parent_targets["signal_date"].nunique()),
        "projected_signal_dates": int(targets["signal_date"].nunique()), "base_cost": base, "stress_cost": stress,
        "bootstrap_annualized_excess": boot,
    }
    return report, targets, base_ledger, base_trades, stress_ledger, stress_trades


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parent = yaml.safe_load((ROOT / contract["inputs"]["parent_config"]).read_text(encoding="utf-8"))
    manifest = verify(contract)
    inputs = contract["inputs"]
    recent_panel = pd.read_parquet(ROOT / inputs["recent_member_panel"])
    recent_execution = load_full_component_history([str(path) for path in sorted((ROOT / inputs["recent_component_history_cache"]).glob("*.parquet"))])
    recent = run_period(
        recent_panel, recent_execution, pd.read_parquet(ROOT / inputs["recent_index"]), pd.read_parquet(ROOT / inputs["recent_breadth"]),
        pd.read_parquet(ROOT / inputs["recent_benchmark"]), contract["periods"]["recent"], contract, parent,
    )
    external_panel = pd.read_parquet(ROOT / inputs["external_member_panel"])
    external_breadth = build_external_breadth(external_panel)
    external = run_period(
        external_panel, external_panel, pd.read_parquet(ROOT / inputs["external_index"]), external_breadth,
        pd.read_parquet(ROOT / inputs["external_benchmark"]), contract["periods"]["external"], contract, parent,
    )
    recent_report, recent_targets, recent_base_ledger, recent_base_trades, _, _ = recent
    external_report, external_targets, external_base_ledger, external_base_trades, _, _ = external
    threshold = float(contract["evaluation"]["annualized_excess_minimum"])
    gates = {}
    for label, item in (("recent", recent_report), ("external", external_report)):
        gates[f"{label}_base_excess_20pct"] = item["base_cost"]["annualized_excess"] >= threshold
        gates[f"{label}_stress_excess_20pct"] = item["stress_cost"]["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"])
        gates[f"{label}_bootstrap_lower_positive"] = item["bootstrap_annualized_excess"]["interval_95pct"][0] > 0.0
    all_trades = pd.concat([recent_base_trades, external_base_trades], ignore_index=True)
    gates["all_trades_at_least_5000"] = bool(all_trades.empty or all_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all())
    status = "DUAL_SLEEVE_PROJECTION_RETROSPECTIVE_PASS" if all(gates.values()) else "DUAL_SLEEVE_PROJECTION_REJECTED_FROZEN"
    report = _json_safe({
        "project_id": contract["protocol"]["project_id"], "status": status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "manifest_sha256": sha256(MANIFEST_FILE),
        "frozen_at": manifest["frozen_at"], "factor_count": 8, "projection_holdings": 3,
        "recent": recent_report, "external": external_report, "gates": gates,
        "execution_audit": {"minimum_trade_notional_cny": float(all_trades["notional"].min()) if not all_trades.empty else None, "t_plus_one_enforced": True},
        "safety": contract["governance"],
    })
    outputs = contract["outputs"]
    for frame, key in ((recent_base_ledger, "recent_base_ledger"), (recent_base_trades, "recent_base_trades"), (external_base_ledger, "external_base_ledger"), (external_base_trades, "external_base_trades")):
        atomic_parquet(frame, ROOT / outputs[key])
    lines = ["# 双袖带信号向2万元三仓投影 V1", "", f"- 状态：`{status}`", "", "|时段|策略年化|基准年化|基础超额|压力超额|Bootstrap下界|", "|---|---:|---:|---:|---:|---:|"]
    for label, item in (("2015—2021", external_report), ("2021—2026", recent_report)):
        lines.append(f"|{label}|{item['base_cost']['strategy']['cagr']:.2%}|{item['base_cost']['benchmark']['cagr']:.2%}|{item['base_cost']['annualized_excess']:.2%}|{item['stress_cost']['annualized_excess']:.2%}|{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}|")
    lines += ["", "## 门槛", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
    lines += ["", "历史已受研究污染；不生成仓位、订单或实盘连接。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    atomic_text("\n".join(lines), ROOT / outputs["result_markdown"])
    print(json.dumps({"状态": status, "近期基础超额": recent_report["base_cost"]["annualized_excess"], "近期压力超额": recent_report["stress_cost"]["annualized_excess"], "早期基础超额": external_report["base_cost"]["annualized_excess"], "早期压力超额": external_report["stress_cost"]["annualized_excess"], "最小成交": report["execution_audit"]["minimum_trade_notional_cny"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
