"""用Tushare fund_adj修复面板重跑完全相同的全ETF V1公式。"""

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
from scripts.download_csi300_all_etf_fund_adj_correction_v1 import CORRECTED_PANEL, CORRECTION_STATUS  # noqa: E402
from scripts.freeze_csi300_all_etf_adjustment_correction_v1 import CORRECTION_MANIFEST, FROZEN_FILES, sha256  # noqa: E402
from scripts.freeze_csi300_all_etf_momentum_protocol_v1 import CONFIG_FILE  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text, bootstrap, costs, json_safe, period_results  # noqa: E402

OUTPUT_ROOT = ROOT / "reports" / "backtest"
RESULT_JSON = OUTPUT_ROOT / "csi300_all_etf_momentum_alpha_v1r.json"
RESULT_MD = OUTPUT_ROOT / "CSI300_ALL_ETF_MOMENTUM_ALPHA_V1R.md"


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(CORRECTION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_DATA_CORRECTION_ONLY_FORMULA_UNCHANGED" or not manifest.get("formula_files_equal_original_freeze"):
        raise RuntimeError("数据修复协议未冻结或公式变化")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    if changed:
        raise RuntimeError(f"修复冻结后文件变化：{changed}")
    status = json.loads(CORRECTION_STATUS.read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or not status.get("correction_only_formula_unchanged"):
        raise RuntimeError("复权修复数据审计未通过")
    if sha256(CORRECTED_PANEL) != status["hashes"]["corrected_panel"]:
        raise RuntimeError("修复面板哈希变化")
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    panel = pd.read_parquet(CORRECTED_PANEL)
    benchmark_raw = pd.read_parquet(ROOT / contract["inputs"]["benchmark"])
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
    result_status = "ALL_ETF_V1R_HISTORICAL_PASS_AWAITING_TRUE_FORWARD" if all(gates.values()) else "ALL_ETF_V1R_REJECTED_FROZEN"
    report = json_safe({
        "project_id": "CSI300_ALL_ETF_MOMENTUM_ALPHA_V1R_DATA_CORRECTED", "status": result_status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "formula_equal_original_freeze": True,
        "correction_manifest_sha256": sha256(CORRECTION_MANIFEST), "price_source": "tushare.fund_daily", "adjustment_source": "tushare.fund_daily_pre_close_link",
        "factor_count": len(FACTOR_COLUMNS), "master_asset_count": int(len(master)),
        "maximum_eligible_assets": int(features.groupby("date")["eligible"].sum().max()), "selected_signal_counts": targets["con_code"].value_counts().to_dict(),
        "base_cost": base, "stress_cost": stress, "bootstrap_annualized_excess": boot,
        "predefined_periods": periods, "positive_predefined_periods": positive_periods, "gates": gates,
        "execution_audit": {"minimum_trade_notional_cny": minimum_trade, "maximum_position_count": int(base_ledger["position_count"].max()), "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()), "t_plus_one_enforced": True},
        "safety": contract["governance"],
    })
    outputs = {
        "features": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_features.parquet", "targets": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_targets.parquet",
        "base_ledger": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_base_ledger.parquet", "base_trades": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_base_trades.parquet",
        "stress_ledger": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_stress_ledger.parquet", "stress_trades": OUTPUT_ROOT / "csi300_all_etf_momentum_v1r_stress_trades.parquet",
    }
    for frame, key in ((features, "features"), (targets, "targets"), (base_ledger, "base_ledger"), (base_trades, "base_trades"), (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades")):
        atomic_parquet(frame, outputs[key])
    lines = ["# 全ETF九因子轮动 V1R 复权修复评估", "", f"- 状态：`{result_status}`", "- 公式：与首次冻结V1完全一致", "- 总收益链：Tushare fund_daily 除权昨收", "", "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|", "|---|---:|---:|---:|---:|---:|", f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|", f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|", "", f"- Bootstrap 95%区间：{boot['interval_95pct']}", f"- 最小成交：{minimum_trade:.2f}元", "", "## 门槛", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
    lines += ["", "仅为冻结历史研究，不生成仓位、订单或实盘连接。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), RESULT_JSON)
    atomic_text("\n".join(lines), RESULT_MD)
    print(json.dumps({"状态": result_status, "策略年化": base["strategy"]["cagr"], "基准年化": base["benchmark"]["cagr"], "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": boot["interval_95pct"][0], "正超额分段": positive_periods}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
