"""固定超参数的10因子Ridge：2015—2023训练，2024—2026时间样本外。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, summarize  # noqa: E402
from research.a_share_hash_holdout_alpha_v1 import (  # noqa: E402
    FEATURE_COLUMNS,
    ModelRules,
    build_features,
    build_outcomes,
)
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


CONFIG = ROOT / "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml"
VALUATION = ROOT / "data/raw/a_share_size_value_development_v1/daily_basic_signal_dates.parquet"
MODEL_PATH = ROOT / "data/processed/a_share_fundamental_ridge_v1/model.joblib"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_fundamental_ridge_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_FUNDAMENTAL_RIDGE_DEVELOPMENT_V1.md"

MODEL_FEATURES = (
    FEATURE_COLUMNS[0],
    FEATURE_COLUMNS[1],
    FEATURE_COLUMNS[2],
    FEATURE_COLUMNS[3],
    FEATURE_COLUMNS[4],
    FEATURE_COLUMNS[6],
    "size_score",
    "ep_score",
    "bp_score",
    "low_turnover_score",
)


def code_stratum(code: str) -> int:
    return hashlib.sha256(str(code).encode("utf-8")).digest()[0] % 3


def rules(start: str, end: str) -> ModelRules:
    return ModelRules(
        signal_start=pd.Timestamp(start),
        signal_end=pd.Timestamp(end),
        rebalance_step=20,
        horizon=20,
        minimum_amount=50_000_000.0,
        maximum_one_lot=6500.0,
        learning_rate=0.01,
        max_iter=1,
        max_leaf_nodes=2,
        min_samples_leaf=2,
        l2_regularization=0.0,
        max_bins=2,
        random_state=20260819,
        target_clip=(-0.5, 0.5),
        half_life_days=1095.75,
    )


def add_fundamental_ranks(features: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    basic = valuation.rename(columns={"trade_date": "date"}).copy()
    basic["date"] = pd.to_datetime(basic["date"])
    result = result.merge(
        basic[["date", "ts_code", "pe_ttm", "pb", "turnover_rate", "total_mv"]],
        left_on=["date", "con_code"],
        right_on=["date", "ts_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_valuation"),
    )
    result["earnings_yield"] = np.where(result["pe_ttm"].gt(0.0), 1.0 / result["pe_ttm"], np.nan)
    result["book_yield"] = np.where(result["pb"].gt(0.0), 1.0 / result["pb"], np.nan)
    ready = result["signal_output"].eq("SIGNAL_READY") & result[["total_mv", "turnover_rate"]].notna().all(axis=1)
    group = result.loc[ready].groupby("date", sort=True)
    result.loc[ready, "size_score"] = group["total_mv"].rank(pct=True, ascending=False)
    result.loc[ready, "ep_score"] = group["earnings_yield"].rank(pct=True, ascending=True)
    result.loc[ready, "bp_score"] = group["book_yield"].rank(pct=True, ascending=True)
    result.loc[ready, "low_turnover_score"] = group["turnover_rate"].rank(pct=True, ascending=False)
    result.loc[ready, "ep_score"] = result.loc[ready, "ep_score"].fillna(0.5)
    result.loc[ready, "bp_score"] = result.loc[ready, "bp_score"].fillna(0.5)
    result["model_ready"] = ready & result[list(MODEL_FEATURES)].notna().all(axis=1)
    result["stratum"] = result["con_code"].astype(str).map(code_stratum)
    return result


def fit_model(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[Ridge, dict]:
    outcomes = build_outcomes(features, panel, benchmark, calendar, 20)
    data = features.loc[features["model_ready"]].merge(
        outcomes[["date", "con_code", "maturity_date", "future_excess_log_return"]],
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    ).dropna(subset=["future_excess_log_return", "maturity_date"])
    data = data.loc[pd.to_datetime(data["maturity_date"]).le(pd.Timestamp("2023-12-29"))].copy()
    y = data["future_excess_log_return"].clip(-0.5, 0.5)
    latest = pd.to_datetime(data["maturity_date"]).max()
    age = (latest - pd.to_datetime(data["maturity_date"])).dt.days.clip(lower=0)
    sample_weight = np.power(0.5, age / 1095.75)
    model = Ridge(alpha=40.0, fit_intercept=True)
    model.fit(data[list(MODEL_FEATURES)].astype(float), y.astype(float), sample_weight=sample_weight)
    diagnostics = {
        "training_rows": len(data),
        "training_signal_dates": int(data["date"].nunique()),
        "last_maturity": str(pd.Timestamp(data["maturity_date"].max()).date()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "ridge_alpha": 40.0,
        "intercept": float(model.intercept_),
        "coefficients": {name: float(value) for name, value in zip(MODEL_FEATURES, model.coef_, strict=True)},
    }
    return model, diagnostics


def select_targets(model: Ridge, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = features.loc[features["model_ready"] & features["raw_close"].ge(2.0)].copy()
    ready["predicted_excess_log_return"] = model.predict(ready[list(MODEL_FEATURES)].astype(float))
    ready.sort_values(
        ["date", "stratum", "predicted_excess_log_return", "con_code"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    targets = ready.groupby(["date", "stratum"], sort=True).head(1).copy()
    targets.rename(columns={"date": "signal_date", "predicted_excess_log_return": "score"}, inplace=True)
    targets.sort_values(["signal_date", "stratum"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = "FUNDAMENTAL_RIDGE_10_FACTOR"
    counts = targets.groupby("signal_date").size()
    if counts.min() != 3 or counts.max() != 3:
        raise RuntimeError("10因子模型每个信号日必须形成三只持仓")
    return ready.reset_index(drop=True), targets[
        ["signal_date", "con_code", "score", "stratum", "selection_rank", "regime"]
    ].reset_index(drop=True)


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    valuation = pd.read_parquet(VALUATION)
    old_master = master.loc[master["split_bucket"].isin([1, 2, 3, 4])].copy()
    old_panel = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_panel.parquet")
    old_benchmark = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_H00300.parquet")
    old_features, old_calendar = build_features(old_panel, old_master, old_benchmark, rules("2015-01-05", "2023-11-30"))
    old_features = add_fundamental_ranks(old_features, valuation)
    model, training = fit_model(old_features, old_panel, old_benchmark, old_calendar)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    recent_paths = (
        "data/raw/a_share_hash_holdout_v2/holdout_panel.parquet",
        "data/raw/a_share_bucket1_time_holdout_formula_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket2_time_holdout_lowvol_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket34_time_holdout_stratified_lowvol_v1/holdout_panel.parquet",
    )
    recent_panel = pd.concat([pd.read_parquet(ROOT / path) for path in recent_paths], ignore_index=True)
    if recent_panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("近期面板存在重复证券日期")
    recent_benchmark = pd.read_parquet(ROOT / contract["paths"]["benchmark"])
    recent_features, recent_calendar_all = build_features(
        recent_panel, master, recent_benchmark, rules("2024-01-02", "2026-08-14")
    )
    recent_features = add_fundamental_ranks(recent_features, valuation)
    predictions, targets = select_targets(model, recent_features)
    selected_codes = set(targets["con_code"].astype(str))
    execution = recent_panel.loc[
        recent_panel["con_code"].astype(str).isin(selected_codes),
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"],
    ].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    calendar = recent_calendar_all[
        (recent_calendar_all >= pd.Timestamp("2024-01-02"))
        & (recent_calendar_all <= pd.Timestamp("2026-08-14"))
    ]
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(recent_benchmark, base_ledger["date"], 20000.0)
    base = summarize(base_ledger, base_trades, benchmark, 20000.0)
    stress = summarize(stress_ledger, stress_trades, benchmark, 20000.0)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = _periods(base_ledger, benchmark, contract)
    selected_delisted = master.loc[
        master["ts_code"].astype(str).isin(selected_codes)
        & master["delist_date"].notna()
        & master["delist_date"].le(pd.Timestamp("2026-08-14")),
        "ts_code",
    ].astype(str).tolist()
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= 0.20,
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= 0.20,
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > 0.0,
        "positive_predefined_periods": sum(value["annualized_excess"] > 0 for value in periods.values()) >= 2,
        "all_trades_at_least_5000": base_trades.empty or bool(base_trades["notional"].ge(5000.0 - 1e-8).all()),
        "no_selected_stock_delisted": not selected_delisted,
    }
    report = _json_safe(
        {
            "project_id": "A_SHARE_FUNDAMENTAL_RIDGE_DEVELOPMENT_V1",
            "status": "TEMPORAL_OOS_DEVELOPMENT_PASS" if all(gates.values()) else "TEMPORAL_OOS_DEVELOPMENT_REJECTED",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "factor_count": len(MODEL_FEATURES),
            "hyperparameter_search": False,
            "training": training,
            "evaluation": {
                "start": "2024-01-02",
                "end": "2026-08-14",
                "base": base,
                "stress": stress,
                "bootstrap_annualized_excess": bootstrap,
                "predefined_periods": periods,
                "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else np.nan,
                "selected_delisted_codes": selected_delisted,
                "gates": gates,
            },
            "untouched_2005_2013_returns_read": False,
            "governance": {"development_only": True, "paper_signal": "DISABLED", "order_generation": "DISABLED"},
        }
    )
    lines = [
        "# 全A 10因子基本面Ridge开发 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 训练：2015—2023；时间样本外：2024—2026。",
        "- 模型：Ridge(alpha=40)，固定10因子，无超参数搜索。",
        "",
        "|策略年化|H00300年化|基础超额|压力超额|波动|回撤|Bootstrap下界|",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|"
        f"{stress['annualized_excess']:.2%}|{base['strategy']['annualized_volatility']:.2%}|"
        f"{base['strategy']['maximum_drawdown']:.2%}|{bootstrap['interval_95pct'][0]:.2%}|",
        "",
        "不生成仓位或订单。",
        "",
    ]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text("\n".join(lines), OUTPUT_MD)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "基础超额": base["annualized_excess"],
                "压力超额": stress["annualized_excess"],
                "Bootstrap下界": bootstrap["interval_95pct"][0],
                "门槛": gates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
