"""一次性评估冻结的点时全ETF九因子轮动。"""

from __future__ import annotations

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

from backtest.run_final_alpha_strategy import _benchmark_series, summarize  # noqa: E402
from research.csi300_all_etf_momentum_alpha_v1 import FACTOR_COLUMNS, build_all_etf_targets  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.freeze_csi300_all_etf_momentum_protocol_v1 import CONFIG_FILE, FROZEN_FILES, MANIFEST_FILE, sha256  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import (  # noqa: E402
    atomic_parquet, atomic_text, bootstrap, costs, json_safe, period_results,
)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FUND_MASTER_AND_RETURN_DOWNLOAD":
        raise RuntimeError("协议没有在基金母表与收益下载前冻结")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    if changed:
        raise RuntimeError(f"冻结后文件变化：{changed}")
    inputs = contract["inputs"]
    data_status = json.loads((ROOT / inputs["data_status"]).read_text(encoding="utf-8"))
    if data_status.get("status") != "PASS" or data_status.get("failed_symbols") != 0:
        raise RuntimeError("点时全ETF数据审计未通过")
    for key in ("master", "panel", "benchmark"):
        if sha256(ROOT / inputs[key]) != data_status["hashes"][key]:
            raise RuntimeError(f"{key}输入哈希变化")

    master = pd.read_parquet(ROOT / inputs["master"])
    panel = pd.read_parquet(ROOT / inputs["panel"])
    benchmark_raw = pd.read_parquet(ROOT / inputs["benchmark"])
    features, targets, calendar = build_all_etf_targets(panel, master, benchmark_raw, contract)
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    initial_cash = float(contract["account"]["initial_cash_cny"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, calendar, initial_cash, costs(contract, "base_slippage_bps_per_leg"))
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, calendar, initial_cash, costs(contract, "stress_slippage_bps_per_leg"))
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base, stress = summarize(base_ledger, base_trades, benchmark, initial_cash), summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    boot = bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = period_results(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in periods.values())
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else np.nan
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": boot["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
    }
    result_status = "ALL_ETF_HISTORICAL_PASS_AWAITING_TRUE_FORWARD" if all(gates.values()) else "ALL_ETF_REJECTED_FROZEN"
    selected_counts = targets["con_code"].value_counts().to_dict()
    report = json_safe({
        "project_id": contract["protocol"]["project_id"], "status": result_status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(MANIFEST_FILE), "formula_frozen_before_fund_master_and_return_download": True,
        "factor_count": len(FACTOR_COLUMNS), "master_asset_count": int(len(master)),
        "maximum_eligible_assets": int(features.groupby("date")["eligible"].sum().max()),
        "selected_signal_counts": selected_counts,
        "base_cost": base, "stress_cost": stress, "bootstrap_annualized_excess": boot,
        "predefined_periods": periods, "positive_predefined_periods": positive_periods, "gates": gates,
        "execution_audit": {"minimum_trade_notional_cny": minimum_trade, "maximum_position_count": int(base_ledger["position_count"].max()), "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()), "t_plus_one_enforced": True},
        "safety": contract["governance"],
    })
    outputs = contract["outputs"]
    for frame, key in ((features, "features"), (targets, "targets"), (base_ledger, "base_ledger"), (base_trades, "base_trades"), (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades")):
        atomic_parquet(frame, ROOT / outputs[key])
    lines = [
        "# CSI300 点时全ETF九因子轮动 V1", "", f"- 状态：`{result_status}`", f"- 母表资产：{len(master)}只", f"- 因子：{len(FACTOR_COLUMNS)}个", "",
        "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|", "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|", "",
        f"- Bootstrap 95%区间：{boot['interval_95pct']}", f"- 最小成交：{minimum_trade:.2f}元", "", "## 门槛", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
    lines += ["", "仅为冻结历史研究，不生成仓位、订单或实盘连接。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    atomic_text("\n".join(lines), ROOT / outputs["result_markdown"])
    print(json.dumps({"状态": result_status, "母表资产": len(master), "最大合格资产": report["maximum_eligible_assets"], "策略年化": base["strategy"]["cagr"], "基准年化": base["benchmark"]["cagr"], "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": boot["interval_95pct"][0], "正超额分段": positive_periods}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
