"""一次性打开未见股票与未见日期的双重盲测结果。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, summarize  # noqa: E402
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, build_features, predict_and_select  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.freeze_a_share_hash_holdout_model_v1 import MODEL_MANIFEST  # noqa: E402
from scripts.freeze_a_share_hash_holdout_protocol_v1 import CONFIG_FILE, FROZEN_FILES, MANIFEST_FILE, sha256  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import _costs, _json_safe, _paired_bootstrap, _periods  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402
from scripts.train_a_share_hash_holdout_alpha_v1 import rules_from_config  # noqa: E402


def verify(contract: dict) -> dict:
    protocol = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    model = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if protocol.get("state") != "FROZEN_BEFORE_STOCK_MASTER_AND_RETURNS" or model.get("state") != "TRAINED_MODEL_FROZEN_BEFORE_HOLDOUT_RETURN_DOWNLOAD":
        raise RuntimeError("双重盲测协议或模型未正确封存")
    if sha256(MANIFEST_FILE) != model["protocol_manifest_sha256"]:
        raise RuntimeError("模型封存后初始协议清单变化")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != protocol["frozen_files"][name]]
    paths = contract["paths"]
    checks = {
        paths["master"]: model["master_sha256"], paths["training_panel"]: model["training_panel_sha256"],
        paths["training_benchmark"]: model["training_benchmark_sha256"], paths["trained_model"]: model["trained_model_sha256"],
        paths["training_report"]: model["training_report_sha256"], paths["training_status"]: model["training_status_sha256"],
    }
    for relative, expected in checks.items():
        if sha256(ROOT / relative) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"封存后训练证据变化：{changed}")
    return model


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    model_manifest = verify(contract)
    paths, periods = contract["paths"], contract["periods"]
    holdout_status = json.loads((ROOT / paths["holdout_status"]).read_text(encoding="utf-8"))
    if holdout_status.get("status") != "PASS" or not holdout_status.get("model_frozen_before_download"):
        raise RuntimeError("盲测数据审计未通过")
    for key in ("master", "holdout_panel", "holdout_benchmark"):
        expected_key = key
        if sha256(ROOT / paths[key]) != holdout_status["hashes"][expected_key]:
            raise RuntimeError(f"盲测{key}输入哈希变化")
    if sha256(MODEL_MANIFEST) != holdout_status["hashes"]["model_manifest"]:
        raise RuntimeError("盲测下载后模型清单变化")

    master = pd.read_parquet(ROOT / paths["master"])
    master = master.loc[master["split_group"].eq("HOLDOUT")].copy()
    panel = pd.read_parquet(ROOT / paths["holdout_panel"])
    benchmark_raw = pd.read_parquet(ROOT / paths["holdout_benchmark"])
    model = joblib.load(ROOT / paths["trained_model"])
    rules = rules_from_config(contract, "holdout_evaluation_start", "holdout_evaluation_end")
    features, calendar_all = build_features(panel, master, benchmark_raw, rules)
    predictions, selected = predict_and_select(model, features, int(contract["selection"]["holdings"]), int(contract["selection"]["retention_prediction_rank"]))
    targets = selected[["date", "con_code", "selection_rank", "predicted_excess_log_return"]].rename(columns={"date": "signal_date"})
    targets["regime"] = "HASH_AND_TIME_HOLDOUT_TEN_FACTOR"
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    calendar = calendar_all[(calendar_all >= pd.Timestamp(periods["holdout_evaluation_start"])) & (calendar_all <= pd.Timestamp(periods["holdout_evaluation_end"]))]
    initial = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "base_slippage_bps_per_leg"), gap)
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "stress_slippage_bps_per_leg"), gap)
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial)
    base, stress = summarize(base_ledger, base_trades, benchmark, initial), summarize(stress_ledger, stress_trades, benchmark, initial)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    period_results = _periods(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in period_results.values())
    selected_codes = set(targets["con_code"].astype(str))
    selected_delisted = master.loc[master["ts_code"].astype(str).isin(selected_codes) & master["delist_date"].notna() & master["delist_date"].le(pd.Timestamp(periods["holdout_evaluation_end"])), "ts_code"].astype(str).tolist()
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else np.nan
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
        "no_selected_stock_delisted_during_holdout": len(selected_delisted) == 0,
    }
    status = "STRICT_HASH_AND_TIME_HOLDOUT_PASS_AWAITING_FORWARD" if all(gates.values()) else "STRICT_HASH_AND_TIME_HOLDOUT_REJECTED_FROZEN"
    report = _json_safe({
        "project_id": contract["protocol"]["project_id"], "status": status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "model_frozen_at": model_manifest["frozen_at"],
        "holdout_returns_downloaded_before_model_freeze": False, "factor_count": len(FEATURE_COLUMNS),
        "row_counts": {"features": len(features), "ready": int(features["signal_output"].eq("SIGNAL_READY").sum()), "predictions": len(predictions), "signal_dates": int(targets["signal_date"].nunique())},
        "base_cost": base, "stress_cost": stress, "bootstrap_annualized_excess": bootstrap,
        "predefined_periods": period_results, "positive_predefined_periods": positive_periods, "gates": gates,
        "execution_audit": {"minimum_trade_notional_cny": minimum_trade, "maximum_position_count": int(base_ledger["position_count"].max()), "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()), "selected_delisted_codes": selected_delisted, "t_plus_one_enforced": True},
        "safety": contract["governance"],
    })
    for frame, key in ((features, "features"), (predictions, "predictions"), (targets, "targets"), (base_ledger, "base_ledger"), (base_trades, "base_trades"), (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades")):
        atomic_parquet(frame, ROOT / paths[key])
    lines = ["# 全A股代码哈希与时间双重盲测 V1", "", f"- 状态：`{status}`", "- 盲测股票收益在模型封存前未下载", "", "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|", "|---|---:|---:|---:|---:|---:|", f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|", f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|", "", f"- Bootstrap 95%区间：{bootstrap['interval_95pct']}", f"- 最小成交：{minimum_trade:.2f}元", "", "## 门槛", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
    lines += ["", "研究结果不生成仓位、订单或实盘连接。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / paths["result_json"])
    atomic_text("\n".join(lines), ROOT / paths["result_markdown"])
    print(json.dumps({"状态": status, "策略年化": base["strategy"]["cagr"], "基准年化": base["benchmark"]["cagr"], "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": bootstrap["interval_95pct"][0], "正超额分段": positive_periods}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
