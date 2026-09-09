"""固定7因子的趋势海龟/震荡均值回归状态切换开发。"""

from __future__ import annotations

import hashlib
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
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, ModelRules, build_features  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


CONFIG = ROOT / "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_turtle_martin_regime_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_TURTLE_MARTIN_REGIME_DEVELOPMENT_V1.md"


def code_stratum(code: str) -> int:
    return hashlib.sha256(str(code).encode("utf-8")).digest()[0] % 3


def rules(start: str, end: str) -> ModelRules:
    return ModelRules(
        signal_start=pd.Timestamp(start), signal_end=pd.Timestamp(end), rebalance_step=20, horizon=20,
        minimum_amount=50_000_000.0, maximum_one_lot=6500.0, learning_rate=0.01, max_iter=1,
        max_leaf_nodes=2, min_samples_leaf=2, l2_regularization=0.0, max_bins=2,
        random_state=20260819, target_clip=(-0.5, 0.5), half_life_days=1095.75,
    )


def market_regime(benchmark: pd.DataFrame) -> pd.DataFrame:
    frame = benchmark[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame.sort_values("date", inplace=True)
    frame["ma120"] = frame["close"].rolling(120, min_periods=120).mean()
    frame["return60"] = frame["close"] / frame["close"].shift(60) - 1.0
    frame["trend_regime"] = frame["close"].gt(frame["ma120"]) & frame["return60"].gt(0.0)
    return frame[["date", "trend_regime"]]


def select_targets(features: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    ready = features.loc[features["signal_output"].eq("SIGNAL_READY") & features["raw_close"].ge(2.0)].copy()
    ready["stratum"] = ready["con_code"].astype(str).map(code_stratum)
    ready = ready.merge(market_regime(benchmark), on="date", how="left", validate="many_to_one")
    ready["trend_score"] = (
        0.35 * ready[FEATURE_COLUMNS[2]]
        + 0.25 * ready[FEATURE_COLUMNS[1]]
        + 0.20 * ready[FEATURE_COLUMNS[9]]
        + 0.20 * ready[FEATURE_COLUMNS[4]]
    )
    ready["range_score"] = (
        0.40 * ready[FEATURE_COLUMNS[3]]
        + 0.25 * ready[FEATURE_COLUMNS[4]]
        + 0.20 * ready[FEATURE_COLUMNS[8]]
        + 0.15 * ready[FEATURE_COLUMNS[6]]
    )
    ready["score"] = np.where(ready["trend_regime"], ready["trend_score"], ready["range_score"])
    ready.sort_values(["date", "stratum", "score", "con_code"], ascending=[True, True, False, True], inplace=True)
    targets = ready.groupby(["date", "stratum"], sort=True).head(1).copy()
    targets.rename(columns={"date": "signal_date"}, inplace=True)
    targets.sort_values(["signal_date", "stratum"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = np.where(targets["trend_regime"], "TURTLE_TREND", "MARTIN_RANGE")
    counts = targets.groupby("signal_date").size()
    if counts.min() != 3 or counts.max() != 3:
        raise RuntimeError("状态切换公式每个信号日必须形成三只持仓")
    return targets[["signal_date", "con_code", "score", "stratum", "selection_rank", "regime"]].reset_index(drop=True)


def evaluate(
    label: str,
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark_raw: pd.DataFrame,
    start: str,
    end: str,
    period_specs: list[dict],
    base_contract: dict,
) -> dict:
    features, calendar_all = build_features(panel, master, benchmark_raw, rules(start, end))
    targets = select_targets(features, benchmark_raw)
    selected_codes = set(targets["con_code"].astype(str))
    execution = panel.loc[
        panel["con_code"].astype(str).isin(selected_codes),
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"],
    ].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    calendar = calendar_all[(calendar_all >= pd.Timestamp(start)) & (calendar_all <= pd.Timestamp(end))]
    contract = json.loads(json.dumps(base_contract))
    contract["evaluation"]["predefined_periods"] = period_specs
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], 20000.0)
    base = summarize(base_ledger, base_trades, benchmark, 20000.0)
    stress = summarize(stress_ledger, stress_trades, benchmark, 20000.0)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = _periods(base_ledger, benchmark, contract)
    selected_delisted = master.loc[
        master["ts_code"].astype(str).isin(selected_codes)
        & master["delist_date"].notna()
        & master["delist_date"].le(pd.Timestamp(end)),
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
    regime_counts = targets.drop_duplicates("signal_date")["regime"].value_counts().to_dict()
    return _json_safe(
        {
            "label": label,
            "base": base,
            "stress": stress,
            "bootstrap_annualized_excess": bootstrap,
            "predefined_periods": periods,
            "regime_signal_counts": regime_counts,
            "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else np.nan,
            "selected_delisted_codes": selected_delisted,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    )


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    old_panel = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_panel.parquet")
    old_benchmark = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_H00300.parquet")
    old_master = master.loc[master["split_bucket"].isin([1, 2, 3, 4])].copy()
    recent_paths = (
        "data/raw/a_share_hash_holdout_v2/holdout_panel.parquet",
        "data/raw/a_share_bucket1_time_holdout_formula_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket2_time_holdout_lowvol_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket34_time_holdout_stratified_lowvol_v1/holdout_panel.parquet",
    )
    recent_panel = pd.concat([pd.read_parquet(ROOT / path) for path in recent_paths], ignore_index=True)
    recent_benchmark = pd.read_parquet(ROOT / contract["paths"]["benchmark"])
    results = [
        evaluate(
            "DEVELOPMENT_2015_2023", old_panel, old_master, old_benchmark, "2015-01-05", "2023-11-30",
            [
                {"id": "P1_2015_2017", "start": "2015-01-05", "end": "2017-12-29"},
                {"id": "P2_2018_2020", "start": "2018-01-02", "end": "2020-12-31"},
                {"id": "P3_2021_2023", "start": "2021-01-04", "end": "2023-11-30"},
            ], contract,
        ),
        evaluate(
            "DEVELOPMENT_2024_2026", recent_panel, master, recent_benchmark, "2024-01-02", "2026-08-14",
            contract["evaluation"]["predefined_periods"], contract,
        ),
    ]
    qualified = all(item["all_gates_pass"] for item in results)
    report = _json_safe(
        {
            "project_id": "A_SHARE_TURTLE_MARTIN_REGIME_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_CANDIDATE_FOUND" if qualified else "NO_DEVELOPMENT_CANDIDATE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "factor_count": 7,
            "formula": {
                "trend_gate": "H00300_close_above_MA120_and_return60_positive",
                "turtle_weights": {"momentum20": 0.35, "momentum60_skip20": 0.25, "trend_efficiency60": 0.20, "lowvol20": 0.20},
                "martin_weights": {"reversal5": 0.40, "lowvol20": 0.25, "range_compression20": 0.20, "drawdown_position60": 0.15},
            },
            "results": results,
            "qualified_for_backward_holdout": qualified,
            "untouched_2005_2013_returns_read": False,
            "governance": {"development_only": True, "paper_signal": "DISABLED", "order_generation": "DISABLED"},
        }
    )
    lines = [
        "# 海龟/类马丁状态切换开发 V1",
        "",
        "|区间|基础超额|压力超额|波动|回撤|Bootstrap下界|全部门槛|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"|{item['label']}|{item['base']['annualized_excess']:.2%}|{item['stress']['annualized_excess']:.2%}|"
            f"{item['base']['strategy']['annualized_volatility']:.2%}|{item['base']['strategy']['maximum_drawdown']:.2%}|"
            f"{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}|{'PASS' if item['all_gates_pass'] else 'FAIL'}|"
        )
    lines += ["", f"- 是否有资格进入2005—2013盲测：`{qualified}`", "- 不生成仓位或订单。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text("\n".join(lines), OUTPUT_MD)
    print(
        json.dumps(
            {item["label"]: {"基础超额": item["base"]["annualized_excess"], "压力超额": item["stress"]["annualized_excess"], "Bootstrap下界": item["bootstrap_annualized_excess"]["interval_95pct"][0], "门槛": item["all_gates_pass"]} for item in results},
            ensure_ascii=False, indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
