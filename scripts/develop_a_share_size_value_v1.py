"""在2015—2026开发期检验点时规模、价值与风险复合因子。"""

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
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


CONFIG = ROOT / "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml"
DAILY_BASIC = ROOT / "data/raw/a_share_size_value_development_v1/daily_basic_signal_dates.parquet"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_size_value_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_SIZE_VALUE_DEVELOPMENT_V1.md"
SOURCE_SIZE_VALUE = "https://www.nber.org/papers/w24458"

CANDIDATES = {
    "SIZE": {"size_score": 1.0},
    "VALUE": {"ep_score": 0.5, "bp_score": 0.5},
    "SIZE_VALUE": {"size_score": 0.5, "ep_score": 0.25, "bp_score": 0.25},
    "SIZE_BP": {"size_score": 0.6, "bp_score": 0.4},
    "SIZE_VALUE_LOWVOL": {"size_score": 0.4, "ep_score": 0.2, "bp_score": 0.2, "lowvol_score": 0.2},
    "SIZE_BP_MOMENTUM": {"size_score": 0.5, "bp_score": 0.3, "momentum_score": 0.2},
    "VALUE_LOWVOL": {"ep_score": 0.4, "bp_score": 0.4, "lowvol_score": 0.2},
    "SIZE_BP_DIVIDEND": {"size_score": 0.5, "bp_score": 0.3, "dividend_score": 0.2},
}


def code_stratum(code: str) -> int:
    return hashlib.sha256(str(code).encode("utf-8")).digest()[0] % 3


def build_features(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    valuation: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    start: str,
    end: str,
) -> pd.DataFrame:
    evaluation = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    signal_dates = set(evaluation[::20])
    source = panel[["date", "con_code", "raw_close", "total_return_close", "volume", "amount"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    source.sort_values(["con_code", "date"], inplace=True)
    pieces = []
    for _, group in source.groupby("con_code", sort=False):
        group = group.reset_index(drop=True)
        total_close = group["total_return_close"]
        log_return = np.log(total_close / total_close.shift(1))
        calculated = pd.DataFrame(
            {
                "raw_vol20": log_return.rolling(20, min_periods=20).std(),
                "raw_momentum120_skip20": np.log(total_close.shift(20) / total_close.shift(120)),
                "average_amount20": group["amount"].rolling(20, min_periods=20).mean(),
            }
        )
        selected = group["date"].isin(signal_dates)
        if selected.any():
            pieces.append(
                pd.concat(
                    [group.loc[selected].reset_index(drop=True), calculated.loc[selected].reset_index(drop=True)],
                    axis=1,
                )
            )
    features = pd.concat(pieces, ignore_index=True)
    meta = master[["ts_code", "list_date", "delist_date"]].copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta["delist_date"], errors="coerce")
    features = features.merge(meta, left_on="con_code", right_on="ts_code", how="left", validate="many_to_one")
    basic = valuation.rename(columns={"trade_date": "date"}).copy()
    basic["date"] = pd.to_datetime(basic["date"])
    features = features.merge(
        basic[["date", "ts_code", "pe_ttm", "pb", "dv_ttm", "total_mv", "circ_mv"]],
        left_on=["date", "con_code"],
        right_on=["date", "ts_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_valuation"),
    )
    features["stratum"] = features["con_code"].astype(str).map(code_stratum)
    features["earnings_yield"] = np.where(features["pe_ttm"].gt(0.0), 1.0 / features["pe_ttm"], np.nan)
    features["book_yield"] = np.where(features["pb"].gt(0.0), 1.0 / features["pb"], np.nan)
    features["eligible"] = (
        features[["raw_vol20", "raw_momentum120_skip20", "average_amount20", "total_mv", "circ_mv"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
        & features["average_amount20"].ge(50_000_000.0)
        & features["raw_close"].ge(2.0)
        & features["raw_close"].mul(100.0).le(6500.0)
        & features["volume"].gt(0.0)
        & features["date"].ge(features["list_date"])
        & (features["delist_date"].isna() | features["date"].lt(features["delist_date"]))
    )
    eligible = features["eligible"]
    groups = [features.loc[eligible, "date"], features.loc[eligible, "stratum"]]
    features.loc[eligible, "size_score"] = features.loc[eligible].groupby(["date", "stratum"])["total_mv"].rank(pct=True, ascending=False)
    features.loc[eligible, "lowvol_score"] = features.loc[eligible].groupby(["date", "stratum"])["raw_vol20"].rank(pct=True, ascending=False)
    features.loc[eligible, "momentum_score"] = features.loc[eligible].groupby(["date", "stratum"])["raw_momentum120_skip20"].rank(pct=True, ascending=True)
    features.loc[eligible, "ep_score"] = features.loc[eligible].groupby(["date", "stratum"])["earnings_yield"].rank(pct=True, ascending=True)
    features.loc[eligible, "bp_score"] = features.loc[eligible].groupby(["date", "stratum"])["book_yield"].rank(pct=True, ascending=True)
    features.loc[eligible, "dividend_score"] = features.loc[eligible].groupby(["date", "stratum"])["dv_ttm"].rank(pct=True, ascending=True)
    return features.reset_index(drop=True)


def select_targets(features: pd.DataFrame, weights: dict[str, float], candidate: str) -> pd.DataFrame:
    required = list(weights)
    ready = features.loc[features["eligible"] & features[required].notna().all(axis=1)].copy()
    ready["score"] = sum(weight * ready[column] for column, weight in weights.items())
    ready.sort_values(["date", "stratum", "score", "con_code"], ascending=[True, True, False, True], inplace=True)
    targets = ready.groupby(["date", "stratum"], sort=True).head(1).copy()
    targets.rename(columns={"date": "signal_date"}, inplace=True)
    targets.sort_values(["signal_date", "stratum"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = f"SIZE_VALUE_{candidate}"
    counts = targets.groupby("signal_date").size()
    if counts.min() != 3 or counts.max() != 3:
        raise RuntimeError(f"{candidate}未能在每个信号日形成三只持仓")
    return targets[["signal_date", "con_code", "score", "stratum", "selection_rank", "regime"]].reset_index(drop=True)


def evaluate_candidate(
    candidate: str,
    weights: dict[str, float],
    features: pd.DataFrame,
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark_raw: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    contract: dict,
) -> dict:
    targets = select_targets(features, weights, candidate)
    selected_codes = set(targets["con_code"].astype(str))
    execution = panel.loc[
        panel["con_code"].astype(str).isin(selected_codes),
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"],
    ].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
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
        & master["delist_date"].le(calendar.max()),
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
    return _json_safe(
        {
            "candidate": candidate,
            "factor_count": len(weights),
            "weights": weights,
            "base": base,
            "stress": stress,
            "bootstrap_annualized_excess": bootstrap,
            "predefined_periods": periods,
            "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else np.nan,
            "selected_delisted_codes": selected_delisted,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    )


def prepare_period(
    label: str,
    panel: pd.DataFrame,
    master: pd.DataFrame,
    valuation: pd.DataFrame,
    benchmark: pd.DataFrame,
    start: str,
    end: str,
    periods: list[dict],
    base_contract: dict,
) -> dict:
    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    calendar_all = pd.DatetimeIndex(sorted(benchmark["date"].unique()))
    calendar = calendar_all[(calendar_all >= pd.Timestamp(start)) & (calendar_all <= pd.Timestamp(end))]
    features = build_features(panel, master, valuation, calendar_all, start, end)
    contract = json.loads(json.dumps(base_contract))
    contract["evaluation"]["predefined_periods"] = periods
    contract["evaluation"]["positive_predefined_periods_minimum"] = 2
    results = []
    for candidate, weights in CANDIDATES.items():
        item = evaluate_candidate(candidate, weights, features, panel, master, benchmark, calendar, contract)
        results.append(item)
        print(
            f"{label} {candidate}：基础超额{item['base']['annualized_excess']:.2%}，"
            f"Bootstrap下界{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}",
            flush=True,
        )
    return {"label": label, "feature_rows": len(features), "results": results}


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    valuation = pd.read_parquet(DAILY_BASIC)
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
    if recent_panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("近期开发面板存在重复证券日期")
    recent_benchmark = pd.read_parquet(ROOT / contract["paths"]["benchmark"])
    period_results = [
        prepare_period(
            "DEVELOPMENT_2015_2023",
            old_panel,
            old_master,
            valuation,
            old_benchmark,
            "2015-01-05",
            "2023-11-30",
            [
                {"id": "P1_2015_2017", "start": "2015-01-05", "end": "2017-12-29"},
                {"id": "P2_2018_2020", "start": "2018-01-02", "end": "2020-12-31"},
                {"id": "P3_2021_2023", "start": "2021-01-04", "end": "2023-11-30"},
            ],
            contract,
        ),
        prepare_period(
            "DEVELOPMENT_2024_2026",
            recent_panel,
            master,
            valuation,
            recent_benchmark,
            "2024-01-02",
            "2026-08-14",
            contract["evaluation"]["predefined_periods"],
            contract,
        ),
    ]
    by_name = {
        period["label"]: {item["candidate"]: item for item in period["results"]}
        for period in period_results
    }
    qualified = [
        name
        for name in CANDIDATES
        if all(by_name[label][name]["all_gates_pass"] for label in by_name)
    ]
    report = _json_safe(
        {
            "project_id": "A_SHARE_SIZE_VALUE_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_CANDIDATE_FOUND" if qualified else "NO_DEVELOPMENT_CANDIDATE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source": SOURCE_SIZE_VALUE,
            "candidate_count": len(CANDIDATES),
            "factor_count_maximum": max(len(weights) for weights in CANDIDATES.values()),
            "period_results": period_results,
            "qualified_candidates": qualified,
            "untouched_2005_2013_returns_read": False,
            "governance": {"development_only": True, "paper_signal": "DISABLED", "order_generation": "DISABLED"},
        }
    )
    lines = [
        "# 全A规模—价值开发 V1",
        "",
        "|候选|2015—2023基础超额|2015—2023压力超额|2024—2026基础超额|2024—2026压力超额|两段均通过|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in CANDIDATES:
        old = by_name["DEVELOPMENT_2015_2023"][name]
        recent = by_name["DEVELOPMENT_2024_2026"][name]
        lines.append(
            f"|{name}|{old['base']['annualized_excess']:.2%}|{old['stress']['annualized_excess']:.2%}|"
            f"{recent['base']['annualized_excess']:.2%}|{recent['stress']['annualized_excess']:.2%}|"
            f"{'PASS' if name in qualified else 'FAIL'}|"
        )
    lines += ["", f"- 两段均通过候选：`{qualified}`", "- 2005—2013收益未读取；不生成订单。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text("\n".join(lines), OUTPUT_MD)
    print(json.dumps({"状态": report["status"], "两段均通过候选": qualified}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
