"""一次性评估历史沪深300成员的代码哈希留出组。"""

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
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, predict_and_select  # noqa: E402
from research.csi300_retrospective_hash_alpha_v1 import (  # noqa: E402
    build_master,
    build_point_in_time_features,
    load_continuous_benchmark,
    load_continuous_panel,
    read_codes,
)
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.freeze_csi300_retrospective_hash_alpha_v1 import CONFIG_FILE, FROZEN_FILES, sha256  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)
from scripts.train_a_share_hash_holdout_alpha_v1 import rules_from_config  # noqa: E402


def verify_frozen_state(contract: dict) -> dict:
    paths = contract["paths"]
    protocol_path = ROOT / paths["protocol_manifest"]
    model_manifest_path = ROOT / paths["model_manifest"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if protocol.get("state") != "FROZEN_BEFORE_HASH_SUBGROUP_FEATURE_AND_RETURN_EVALUATION":
        raise RuntimeError("协议冻结状态错误")
    if model_manifest.get("state") != "RETROSPECTIVE_HASH_MODEL_FROZEN_BEFORE_HOLDOUT_SUBGROUP_EVALUATION":
        raise RuntimeError("模型没有在留出组评估前封存")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != protocol["frozen_files"][name]]
    changed += [name for name, expected in protocol["input_files"].items() if sha256(ROOT / name) != expected]
    checks = {
        paths["protocol_manifest"]: model_manifest["protocol_manifest_sha256"],
        paths["trained_model"]: model_manifest["trained_model_sha256"],
        paths["training_features"]: model_manifest["training_features_sha256"],
        paths["training_report"]: model_manifest["training_report_sha256"],
    }
    changed += [name for name, expected in checks.items() if sha256(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"冻结后证据变化：{sorted(set(changed))}")
    return model_manifest


def render_report(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# 沪深300历史成员回溯性代码哈希留出 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 证据等级：`RETROSPECTIVE_HASH_HOLDOUT_SECONDARY_EVIDENCE`",
        "- 永久盲测声明：`禁止`",
        "",
        "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交额：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += [
        "",
        "即使全部门槛通过，本分支也只提供补充证据；必须等待全A永久盲测或真实前瞻期确认。",
        "本结果不生成仓位、订单或券商连接。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    model_manifest = verify_frozen_state(contract)
    inputs, paths, periods = contract["inputs"], contract["paths"], contract["periods"]
    source_paths = [ROOT / inputs["early_member_panel"], ROOT / inputs["recent_member_panel"]]
    master = build_master(read_codes(source_paths))
    holdout_master = master.loc[master["split_group"].eq("HOLDOUT")].copy()
    holdout_codes = holdout_master["ts_code"].astype(str).tolist()
    panel, panel_audit = load_continuous_panel(source_paths[0], source_paths[1], holdout_codes)
    if set(panel["con_code"].unique()).difference(holdout_codes):
        raise RuntimeError("评估面板混入训练组证券")
    benchmark_raw, benchmark_audit = load_continuous_benchmark(
        ROOT / inputs["early_benchmark"], ROOT / inputs["recent_benchmark"]
    )
    model = joblib.load(ROOT / paths["trained_model"])
    rules = rules_from_config(contract, "holdout_evaluation_start", "holdout_evaluation_end")
    features, calendar_all = build_point_in_time_features(panel, holdout_master, benchmark_raw, rules)
    predictions, selected = predict_and_select(
        model,
        features,
        int(contract["selection"]["holdings"]),
        int(contract["selection"]["retention_prediction_rank"]),
    )
    targets = selected[["date", "con_code", "selection_rank", "predicted_excess_log_return"]].rename(
        columns={"date": "signal_date"}
    )
    targets["regime"] = "RETROSPECTIVE_CODE_HASH_HOLDOUT_TEN_FACTOR"
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    start = pd.Timestamp(periods["holdout_evaluation_start"])
    end = pd.Timestamp(periods["holdout_evaluation_end"])
    calendar = calendar_all[(calendar_all >= start) & (calendar_all <= end)]
    initial = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution, targets, calendar, initial, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution, targets, calendar, initial, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial)
    base = summarize(base_ledger, base_trades, benchmark, initial)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    period_results = _periods(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in period_results.values())
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else np.nan
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
        "t_plus_one_and_lot_engine": True,
    }
    status = "SECONDARY_HASH_HOLDOUT_PASS_AWAITING_PERMANENT_BLIND" if all(gates.values()) else "SECONDARY_HASH_HOLDOUT_REJECTED_FROZEN"
    report = _json_safe({
        "project_id": contract["protocol"]["project_id"],
        "status": status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_frozen_at": model_manifest["frozen_at"],
        "evidence_class": contract["protocol"]["evidence_class"],
        "permanent_blind_claim_allowed": False,
        "factor_count": len(FEATURE_COLUMNS),
        "holdout_code_count": int(len(holdout_master)),
        "row_counts": {
            "features": int(len(features)),
            "ready": int(features["signal_output"].eq("SIGNAL_READY").sum()),
            "predictions": int(len(predictions)),
            "signal_dates": int(targets["signal_date"].nunique()),
        },
        "panel_audit": panel_audit,
        "benchmark_audit": benchmark_audit,
        "base_cost": base,
        "stress_cost": stress,
        "bootstrap_annualized_excess": bootstrap,
        "predefined_periods": period_results,
        "positive_predefined_periods": positive_periods,
        "gates": gates,
        "execution_audit": {
            "minimum_trade_notional_cny": minimum_trade,
            "maximum_position_count": int(base_ledger["position_count"].max()),
            "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()),
            "t_plus_one_enforced": True,
            "lot_size": int(contract["account"]["lot_size"]),
        },
        "safety": contract["governance"],
    })
    frames = (
        (features, "evaluation_features"),
        (predictions, "predictions"),
        (targets, "targets"),
        (base_ledger, "base_ledger"),
        (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"),
        (stress_trades, "stress_trades"),
    )
    for frame, key in frames:
        atomic_parquet(frame, ROOT / paths[key])
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / paths["result_json"])
    atomic_text(render_report(report), ROOT / paths["result_markdown"])
    print(json.dumps({
        "状态": status,
        "策略年化": base["strategy"]["cagr"],
        "H00300年化": base["benchmark"]["cagr"],
        "基础年化超额": base["annualized_excess"],
        "压力年化超额": stress["annualized_excess"],
        "Bootstrap下界": bootstrap["interval_95pct"][0],
        "正超额分段": positive_periods,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
