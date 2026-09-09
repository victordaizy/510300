"""加载已冻结早期模型，一次性评估2021—2026近期时段。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, _return_metrics, summarize  # noqa: E402
from research.csi300_temporal_transfer_alpha_v1 import (  # noqa: E402
    FEATURE_COLUMNS,
    TransferRules,
    build_price_factor_panel,
    predict_and_select,
)
from research.small_account_cross_sectional import (  # noqa: E402
    SmallAccountCosts,
    run_small_account_open_backtest,
)
from scripts.freeze_csi300_temporal_transfer_model_v1 import MODEL_MANIFEST  # noqa: E402
from scripts.freeze_csi300_temporal_transfer_protocol_v1 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    PROTOCOL_MANIFEST,
    sha256,
    tree_sha256,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _verify(contract: dict) -> dict:
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if model_manifest.get("state") != "TRAINED_MODEL_AND_EVALUATION_INPUTS_FROZEN":
        raise RuntimeError("训练模型与评估输入尚未二次冻结")
    if sha256(PROTOCOL_MANIFEST) != model_manifest["protocol_manifest_sha256"]:
        raise RuntimeError("协议清单变化")
    changed = [relative for relative in FROZEN_FILES if sha256(ROOT / relative) != protocol["frozen_files"][relative]]
    outputs = contract["outputs"]
    if sha256(ROOT / outputs["trained_model"]) != model_manifest["trained_model_sha256"]:
        changed.append(outputs["trained_model"])
    for key, relative in contract["evaluation_inputs"].items():
        actual = tree_sha256(ROOT / relative) if key == "component_history_cache" else sha256(ROOT / relative)
        if actual != model_manifest["evaluation_input_hashes"][relative]:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"二次冻结后变化：{sorted(set(changed))}")
    return model_manifest


def _rules(contract: dict) -> TransferRules:
    periods = contract["periods"]
    model = contract["model"]
    return TransferRules(
        start=pd.Timestamp(periods["evaluation_start"]), end=pd.Timestamp(periods["evaluation_end"]),
        step=int(periods["rebalance_every_trading_days"]), horizon=int(periods["target_horizon_trading_days"]),
        features=FEATURE_COLUMNS, minimum_amount=float(contract["selection"]["minimum_20d_average_amount_cny"]),
        learning_rate=float(model["learning_rate"]), max_iter=int(model["max_iter"]),
        max_leaf_nodes=int(model["max_leaf_nodes"]), min_samples_leaf=int(model["min_samples_leaf"]),
        l2_regularization=float(model["l2_regularization"]), max_bins=int(model["max_bins"]),
        random_state=int(model["random_state"]),
        target_clip=(float(model["target_clip"][0]), float(model["target_clip"][1])),
        half_life_days=float(model["training_half_life_calendar_days"]),
    )


def _load_history(directory: Path) -> pd.DataFrame:
    columns = ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume", "amount"]
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(directory.glob("*.parquet"))]
    history = pd.concat(frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    if history[["date", "con_code"]].duplicated().any():
        raise ValueError("近期完整证券历史存在重复")
    return history


def _costs(contract: dict, key: str) -> SmallAccountCosts:
    account, costs = contract["account"], contract["costs"]
    return SmallAccountCosts(
        commission_rate=float(costs["commission_rate"]), minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(costs["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(costs["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(costs["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(costs[key]), cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]), minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


def _bootstrap(strategy: pd.Series, benchmark: pd.Series, contract: dict) -> dict:
    evaluation = contract["evaluation"]
    repetitions, block = int(evaluation["bootstrap_repetitions"]), int(evaluation["bootstrap_block_length_trading_days"])
    rng = np.random.default_rng(int(evaluation["random_seed"]))
    left, right = strategy.to_numpy(float), benchmark.to_numpy(float)
    n, starts = len(left), np.arange(len(left) - block + 1)
    values = np.empty(repetitions)
    for repetition in range(repetitions):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        indices = np.concatenate([np.arange(start, start + block) for start in chosen])[:n]
        values[repetition] = np.expm1(np.log1p(left[indices]).mean() * 242) - np.expm1(np.log1p(right[indices]).mean() * 242)
    return {"repetitions": repetitions, "block_length_trading_days": block, "median": float(np.median(values)), "interval_95pct": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def _periods(ledger: pd.DataFrame, benchmark: pd.DataFrame, contract: dict) -> dict:
    result = {}
    for period in contract["evaluation"]["predefined_periods"]:
        mask = ledger["date"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
        strategy = _return_metrics(ledger.loc[mask, "daily_return"], ledger.loc[mask, "date"])
        bench = _return_metrics(benchmark.loc[mask, "daily_return"], benchmark.loc[mask, "date"])
        result[period["id"]] = {"start": period["start"], "end": period["end"], "strategy": strategy, "benchmark": bench, "annualized_excess": strategy["cagr"] - bench["cagr"]}
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _render(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# 沪深300十因子时间迁移 Alpha V1 近期评估",
        "",
        f"- 状态：`{report['status']}`",
        "- 模型训练：仅2015—2021",
        "- 近期评估：2021—2026，模型不更新",
        "- 因子：10个",
        "",
        "|成本|策略年化|基准年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap 95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += ["", "评估后禁止修改模型、周期、持股数或保留排名补救；不生成仓位或订单。", ""]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    model_manifest = _verify(contract)
    rules = _rules(contract)
    inputs, outputs = contract["evaluation_inputs"], contract["outputs"]
    history = _load_history(ROOT / inputs["component_history_cache"])
    members = pd.read_parquet(ROOT / inputs["member_panel"])
    index = pd.read_parquet(ROOT / inputs["price_index"])
    benchmark_raw = pd.read_parquet(ROOT / inputs["benchmark"])
    model = joblib.load(ROOT / outputs["trained_model"])
    features, calendar = build_price_factor_panel(history, members, index, rules)
    predictions, selected = predict_and_select(
        model, features, rules, int(contract["selection"]["holdings"]), int(contract["selection"]["retention_rank"])
    )
    targets = selected[["date", "con_code", "selection_rank", "predicted_excess_log_return"]].rename(columns={"date": "signal_date"})
    targets["regime"] = "EARLY_TRAINED_TEN_FACTOR"
    execution = history[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0).le(0)
    trade_calendar = calendar[(calendar >= rules.start) & (calendar <= rules.end)]
    initial_cash = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["selection"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, trade_calendar, initial_cash, _costs(contract, "base_slippage_bps_per_leg"), gap)
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, trade_calendar, initial_cash, _costs(contract, "stress_slippage_bps_per_leg"), gap)
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base, stress = summarize(base_ledger, base_trades, benchmark, initial_cash), summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    bootstrap = _bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = _periods(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in periods.values())
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
    }
    status = "TEMPORAL_TRANSFER_PASS_AWAITING_TRUE_FORWARD" if all(gates.values()) else "TEMPORAL_TRANSFER_REJECTED_FROZEN"
    report = _json_safe({
        "project_id": contract["protocol"]["project_id"], "status": status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_manifest_sha256": sha256(MODEL_MANIFEST), "model_frozen_at": model_manifest["frozen_at"],
        "training_recent_input_read": False, "factor_count": 10,
        "row_counts": {"feature_rows": len(features), "ready_rows": int(features["signal_output"].eq("SIGNAL_READY").sum()), "prediction_rows": len(predictions), "selected_signal_dates": int(selected["date"].nunique())},
        "base_cost": base, "stress_cost": stress, "bootstrap_annualized_excess": bootstrap,
        "predefined_periods": periods, "positive_predefined_periods": positive_periods, "gates": gates,
        "execution_audit": {"minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else None, "maximum_position_count": int(base_ledger["position_count"].max()), "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()), "t_plus_one_enforced": True},
        "safety": {"paper_signal": "DISABLED", "position_mapping": "DISABLED", "order_generation": "DISABLED", "broker_connection": "DISABLED", "live_trading": "NOT_AUTHORIZED"},
    })
    for frame, key in ((features, "evaluation_features"), (predictions, "predictions"), (base_ledger, "base_ledger"), (base_trades, "base_trades"), (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades")):
        _atomic_parquet(frame, ROOT / outputs[key])
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    _atomic_text(_render(report), ROOT / outputs["result_markdown"])
    print(json.dumps({"状态": status, "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": bootstrap["interval_95pct"][0], "正超额分段": positive_periods, "结果": str(ROOT / outputs["result_json"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

