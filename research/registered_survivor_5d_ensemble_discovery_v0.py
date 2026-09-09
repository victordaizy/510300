"""把既有仍有统计证据的5日因子映射为固定二元风险分数。

只读取截至2023-12-29的数据。2021-2022固定经验分布，2023比较五个
事先定义的等权机制组合；风险分数达到0.80时下一交易日开盘转为空仓。
该文件是发现扫描，不是正式候选或交易授权。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "discovery" / "510300_registered_survivor_5d_ensemble_discovery_v0.json"
DATA_CEILING = pd.Timestamp("2023-12-29")
FIT_END = pd.Timestamp("2022-12-30")
SELECTION_START = pd.Timestamp("2023-01-03")
RISK_THRESHOLD = 0.80
BASE_ONE_WAY_COST = 0.000925
STRESS_ONE_WAY_COST = 0.00125
CASH_ANNUAL_RATE = 0.015
TRADING_DAYS_PER_YEAR = 242

FACTOR_COLUMNS = {
    "recent_reversal": "factor_etf_total_return_5d",
    "volatility_term": "factor_index_vol_term_5_20",
    "pe_valuation": "factor_official_pe_percentile_5y",
    "pb_valuation": "factor_secondary_pb_percentile_5y",
}

MECHANISMS = {
    "recent_reversal_only": ["recent_reversal"],
    "volatility_term_only": ["volatility_term"],
    "valuation_only": ["pe_valuation", "pb_valuation"],
    "reversal_plus_volatility": ["recent_reversal", "volatility_term"],
    "all_registered_survivors_equal_weight": [
        "recent_reversal",
        "volatility_term",
        "pe_valuation",
        "pb_valuation",
    ],
}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cagr(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0 or bool((values <= -1.0).any()):
        return float("nan")
    return float(np.prod(1.0 + values) ** (TRADING_DAYS_PER_YEAR / len(values)) - 1.0)


def _empirical_percentile(values: pd.Series, reference: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    output[valid] = np.searchsorted(reference, numeric[valid], side="right") / len(reference)
    return pd.Series(output, index=values.index)


def _bootstrap_excess(returns: np.ndarray, benchmark: np.ndarray, repetitions: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(20260828)
    length = len(returns)
    block = 5
    starts = np.arange(max(length - block + 1, 1))
    samples: list[float] = []
    for _ in range(repetitions):
        indices: list[int] = []
        while len(indices) < length:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block, length)))
        chosen = np.asarray(indices[:length], dtype=int)
        samples.append(_cagr(pd.Series(returns[chosen])) - _cagr(pd.Series(benchmark[chosen])))
    return {
        "median": float(np.quantile(samples, 0.50)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _evaluate(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    evaluation = frame[frame["date"] >= SELECTION_START].copy()
    evaluation = evaluation[evaluation[[score_column, "one_day_open_return", "future_5d_return"]].notna().all(axis=1)]
    evaluation["cash"] = evaluation[score_column] >= RISK_THRESHOLD
    position = (~evaluation["cash"]).astype(int)
    transitions = position.diff().abs().fillna(0.0)
    cash_daily = (1.0 + CASH_ANNUAL_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    gross = pd.Series(
        np.where(position.eq(1), evaluation["one_day_open_return"], cash_daily),
        index=evaluation.index,
    )
    base = gross - transitions * BASE_ONE_WAY_COST
    stress = gross - transitions * STRESS_ONE_WAY_COST
    benchmark_cagr = _cagr(evaluation["one_day_open_return"])
    base_cagr = _cagr(base)
    stress_cagr = _cagr(stress)
    bad_threshold = float(frame.loc[frame["date"] <= FIT_END, "future_5d_return"].quantile(0.10))
    bad = evaluation["future_5d_return"] <= bad_threshold
    bad_losses = -evaluation.loc[bad, "future_5d_return"].clip(upper=0.0).sum()
    captured = -evaluation.loc[bad & evaluation["cash"], "future_5d_return"].clip(upper=0.0).sum()
    return {
        "score": score_column,
        "evaluation_start": evaluation["date"].min().date().isoformat(),
        "evaluation_end": evaluation["date"].max().date().isoformat(),
        "rows": int(len(evaluation)),
        "cash_days": int(evaluation["cash"].sum()),
        "cash_share": float(evaluation["cash"].mean()),
        "trade_legs": int(transitions.sum()),
        "benchmark_open_to_open_cagr": benchmark_cagr,
        "base_strategy_cagr": base_cagr,
        "stress_strategy_cagr": stress_cagr,
        "base_annualized_excess": base_cagr - benchmark_cagr,
        "stress_annualized_excess": stress_cagr - benchmark_cagr,
        "bad5_recall": float(evaluation.loc[bad, "cash"].mean()) if bool(bad.any()) else None,
        "good5_false_exit": float(evaluation.loc[~bad, "cash"].mean()) if bool((~bad).any()) else None,
        "bad5_loss_capture": float(captured / bad_losses) if bad_losses > 0 else None,
        "cash_day_mean_one_day_return": float(evaluation.loc[evaluation["cash"], "one_day_open_return"].mean()),
        "cash_day_mean_future_5d_return": float(evaluation.loc[evaluation["cash"], "future_5d_return"].mean()),
        "base_bootstrap_excess": _bootstrap_excess(
            base.to_numpy(dtype=float), evaluation["one_day_open_return"].to_numpy(dtype=float)
        ),
    }


def main() -> int:
    if not INPUT_FILE.exists():
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": str(INPUT_FILE)}
        _atomic_json(payload, OUTPUT_FILE)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    columns = ["date", "etf_open", "exec_total_return_5d_gross", *FACTOR_COLUMNS.values()]
    frame = pd.read_parquet(INPUT_FILE, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] <= DATA_CEILING].sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].max() > DATA_CEILING:
        raise RuntimeError("数据上限检查失败。")
    frame["one_day_open_return"] = frame["etf_open"].shift(-2) / frame["etf_open"].shift(-1) - 1.0
    frame["future_5d_return"] = frame["exec_total_return_5d_gross"]
    fit = frame[frame["date"] <= FIT_END]
    references: dict[str, np.ndarray] = {}
    for factor_name, column in FACTOR_COLUMNS.items():
        reference = np.sort(pd.to_numeric(fit[column], errors="coerce").dropna().to_numpy(dtype=float))
        if len(reference) < 200:
            raise RuntimeError(f"{factor_name}开发参考样本不足。")
        references[factor_name] = reference
        frame[f"risk_{factor_name}"] = _empirical_percentile(frame[column], reference)
    results: list[dict[str, Any]] = []
    for mechanism, factor_names in MECHANISMS.items():
        score_column = f"score_{mechanism}"
        risk_columns = [f"risk_{name}" for name in factor_names]
        frame[score_column] = frame[risk_columns].mean(axis=1)
        result = _evaluate(frame, score_column)
        result["mechanism"] = mechanism
        result["factor_names"] = factor_names
        results.append(result)
    results.sort(key=lambda item: item["base_annualized_excess"], reverse=True)
    pairwise_correlations = {}
    risk_names = [f"risk_{name}" for name in FACTOR_COLUMNS]
    for left, right in itertools.combinations(risk_names, 2):
        pairwise_correlations[f"{left}|{right}"] = float(frame[[left, right]].corr().iloc[0, 1])
    payload = {
        "status": "DEVELOPMENT_DISCOVERY_COMPLETE_NO_2024_PLUS_READ",
        "project_id": "510300_REGISTERED_SURVIVOR_5D_ENSEMBLE_DISCOVERY_V0",
        "data_ceiling": DATA_CEILING.date().isoformat(),
        "fit_period": [frame["date"].min().date().isoformat(), FIT_END.date().isoformat()],
        "selection_period": [SELECTION_START.date().isoformat(), frame["date"].max().date().isoformat()],
        "risk_threshold": RISK_THRESHOLD,
        "base_one_way_cost": BASE_ONE_WAY_COST,
        "stress_one_way_cost": STRESS_ONE_WAY_COST,
        "results": results,
        "best": results[0],
        "pairwise_risk_correlations": pairwise_correlations,
        "validation_2024_loaded": False,
        "replication_2025_plus_loaded": False,
        "interpretation": "固定二元映射的发现扫描；若开发期未达20个百分点则不消耗后续验证样本。",
    }
    _atomic_json(payload, OUTPUT_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
