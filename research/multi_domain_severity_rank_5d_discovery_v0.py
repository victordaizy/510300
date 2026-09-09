"""把全A形态、全球流动性、融资杠杆三个风险域压缩为五日严重度排序。"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    HORIZON,
    PROJECT_ROOT,
    SELECTION_START,
    _aggregate_offsets,
    _atomic_json,
    _atomic_parquet,
    _evaluate_rule_offset,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
TAIL_COVERAGES = [0.05, 0.10, 0.20]

ALL_A_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
GLOBAL_FILE = PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
LEVERAGE_FILE = PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_multi_domain_severity_rank_5d_discovery_v0.parquet"
OUTPUT_OFFSETS = PROJECT_ROOT / "data" / "research" / "510300_multi_domain_severity_rank_5d_discovery_v0" / "offset_metrics.parquet"
OUTPUT_BLOCKS = PROJECT_ROOT / "data" / "research" / "510300_multi_domain_severity_rank_5d_discovery_v0" / "block_predictions.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_multi_domain_severity_rank_5d_discovery_v0.json"

RAW_RISK_COLUMNS = [
    "risk_raw_all_a_logistic",
    "risk_raw_global_tail",
    "risk_raw_leverage_tail",
]


def _fit_ecdf(values: pd.Series, fit_mask: pd.Series) -> pd.Series:
    fit = pd.to_numeric(values.loc[fit_mask], errors="coerce").dropna().to_numpy(dtype=float, copy=True)
    if len(fit) < 100:
        raise ValueError("风险分数拟合样本不足100行")
    fit.sort()
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    output[valid] = np.searchsorted(fit, numeric[valid], side="right") / len(fit)
    return pd.Series(output, index=values.index, dtype=float)


def _rolling_last_percentile(values: pd.Series, window: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def last_rank(array: np.ndarray) -> float:
        valid = array[np.isfinite(array)]
        if len(valid) < minimum:
            return float("nan")
        return float(np.count_nonzero(valid <= valid[-1]) / len(valid))

    return numeric.rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def _load_scores() -> pd.DataFrame:
    all_a = pd.read_parquet(
        ALL_A_FILE,
        columns=["date", "probability_logistic_l2", "probability_hist_gradient_boosting"],
    ).rename(
        columns={
            "probability_logistic_l2": "risk_raw_all_a_logistic",
            "probability_hist_gradient_boosting": "risk_raw_all_a_hgb",
        }
    )
    global_risk = pd.read_parquet(
        GLOBAL_FILE,
        columns=["date", "prob_logistic_bad_tail5", "prob_ensemble_good5", "score_mechanism_risk"],
    ).rename(
        columns={
            "prob_logistic_bad_tail5": "risk_raw_global_tail",
            "prob_ensemble_good5": "global_prob_ensemble_good5",
            "score_mechanism_risk": "risk_raw_global_mechanism",
        }
    )
    leverage = pd.read_parquet(
        LEVERAGE_FILE,
        columns=["date", "prob_logistic_bad_tail5", "prob_ensemble_good5", "score_mechanism_risk"],
    ).rename(
        columns={
            "prob_logistic_bad_tail5": "risk_raw_leverage_tail",
            "prob_ensemble_good5": "leverage_prob_ensemble_good5",
            "score_mechanism_risk": "risk_raw_leverage_mechanism",
        }
    )
    for source in [all_a, global_risk, leverage]:
        source["date"] = pd.to_datetime(source["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
        source.sort_values("date", inplace=True)
        source.drop_duplicates("date", keep="last", inplace=True)
    merged = all_a.merge(global_risk, on="date", how="inner", validate="one_to_one")
    merged = merged.merge(leverage, on="date", how="inner", validate="one_to_one")
    return merged[merged["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)


def _build_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dividends = _load_dividends()
    market = _load_market(dividends)
    market["date"] = market["date"].astype("datetime64[ns]")
    scores = _load_scores()
    frame = market.merge(scores, on="date", how="inner", validate="one_to_one")
    fit_mask = frame["date"].between(FIT_START, FIT_END)
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("多域共同拟合段或选择段覆盖不足")

    normalized_primary: list[str] = []
    for raw_column in RAW_RISK_COLUMNS:
        output = raw_column.replace("risk_raw_", "risk_fit_ecdf_")
        frame[output] = _fit_ecdf(frame[raw_column], fit_mask)
        normalized_primary.append(output)

    primary = frame[normalized_primary]
    frame["risk_pre_ecdf_mean3"] = primary.mean(axis=1)
    frame["risk_pre_ecdf_median3"] = primary.median(axis=1)
    frame["risk_pre_ecdf_max3"] = primary.max(axis=1)
    ensemble_sources = ["risk_pre_ecdf_mean3", "risk_pre_ecdf_median3", "risk_pre_ecdf_max3"]
    risk_columns = normalized_primary.copy()
    for source in ensemble_sources:
        output = source.replace("risk_pre_ecdf_", "risk_fit_ecdf_")
        frame[output] = _fit_ecdf(frame[source], fit_mask)
        risk_columns.append(output)

    rolling_columns: list[str] = []
    for column in risk_columns:
        output = column.replace("risk_fit_ecdf_", "risk_rolling252_")
        frame[output] = _rolling_last_percentile(frame[column])
        rolling_columns.append(output)

    fit_bad_threshold = float(frame.loc[fit_mask, "future5_excess_factor"].quantile(0.20))
    selection_bad = frame.loc[selection_mask, "future5_excess_factor"] <= fit_bad_threshold
    aucs: dict[str, float | None] = {}
    for column in [*risk_columns, *rolling_columns]:
        valid = frame.loc[selection_mask, column].notna() & frame.loc[selection_mask, "future5_excess_factor"].notna()
        labels = selection_bad.loc[valid]
        aucs[column] = _safe_float(
            roc_auc_score(labels.astype(int), frame.loc[selection_mask, column].loc[valid])
            if labels.nunique() == 2
            else np.nan
        )

    audit = {
        "common_rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "fit_bad20_threshold_excess_factor": fit_bad_threshold,
        "selection_bad20_auc": aucs,
    }
    return frame, [*risk_columns, *rolling_columns], audit


def _make_rules(risk_columns: list[str]) -> dict[str, tuple[Callable[[pd.Series, int], int], str, float]]:
    rules: dict[str, tuple[Callable[[pd.Series, int], int], str, float]] = {}
    for column in risk_columns:
        calibration = "FIT_ECDF" if column.startswith("risk_fit_ecdf_") else "ROLLING252_PERCENTILE"
        score_name = column.replace("risk_fit_ecdf_", "").replace("risk_rolling252_", "")
        for coverage in TAIL_COVERAGES:
            threshold = 1.0 - coverage
            name = f"SEVERITY_{score_name.upper()}_{calibration}_TOP{int(coverage * 100):02d}"
            rules[name] = (
                lambda row, previous, selected=column, cutoff=threshold: int(float(row[selected]) < cutoff),
                column,
                coverage,
            )
    return rules


def main() -> int:
    required = [ALL_A_FILE, GLOBAL_FILE, LEVERAGE_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, risk_columns, score_audit = _build_frame()
    # 精确账户模拟器会记录这些通用诊断列；将它们映射到三域主排序，决策仍使用每条规则自己的列。
    frame["prob_logistic_good5"] = 1.0 - frame["risk_fit_ecdf_all_a_logistic"]
    frame["prob_hgb_good5"] = 1.0 - frame["risk_fit_ecdf_global_tail"]
    frame["prob_ensemble_good5"] = 1.0 - frame["risk_fit_ecdf_mean3"]
    frame["prob_logistic_bad_tail5"] = frame["risk_fit_ecdf_leverage_tail"]
    frame["score_mechanism_risk"] = frame["risk_fit_ecdf_max3"]

    offset_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    rules = _make_rules(risk_columns)
    for rule_name, (rule, score_column, coverage) in rules.items():
        score_lookup = frame.set_index("date")[score_column]
        for offset in range(HORIZON):
            metrics, blocks = _evaluate_rule_offset(frame, dividends, rule_name, rule, offset)
            metrics["risk_score_column"] = score_column
            metrics["intended_tail_coverage"] = coverage
            metrics["calibration"] = (
                "FIT_ECDF" if score_column.startswith("risk_fit_ecdf_") else "ROLLING252_PERCENTILE"
            )
            blocks["risk_score_column"] = score_column
            blocks["risk_score"] = blocks["signal_date"].map(score_lookup)
            blocks["intended_tail_coverage"] = coverage
            offset_rows.append(metrics)
            block_frames.append(blocks)

    offsets = pd.DataFrame(offset_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    aggregates = _aggregate_offsets(offsets)
    aggregate_lookup = offsets.groupby("rule", sort=False).first()[
        ["risk_score_column", "intended_tail_coverage", "calibration"]
    ]
    for item in aggregates:
        metadata = aggregate_lookup.loc[item["rule"]]
        item["risk_score_column"] = str(metadata["risk_score_column"])
        item["intended_tail_coverage"] = float(metadata["intended_tail_coverage"])
        item["calibration"] = str(metadata["calibration"])
    best = aggregates[0]
    passed = bool(best["development_gate"])

    output_columns = [
        "date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "future5_excess_factor",
        "target_good5",
        *RAW_RISK_COLUMNS,
        *risk_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(offsets, OUTPUT_OFFSETS)
    _atomic_parquet(blocks, OUTPUT_BLOCKS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_MULTI_DOMAIN_SEVERITY_RANK_5D_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "holding_block_trading_days": HORIZON,
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "tail_coverages": TAIL_COVERAGES,
            "score_calibrations": ["FIT_ECDF", "ROLLING252_PERCENTILE"],
            "calendar_offsets": list(range(HORIZON)),
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个日历错位的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "score_audit": score_audit,
        "risk_columns": risk_columns,
        "rule_count": int(len(rules)),
        "best_development_rule": best,
        "all_rule_aggregates": aggregates,
        "all_offset_metrics": offsets.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "offset_metrics": str(OUTPUT_OFFSETS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "block_predictions": str(OUTPUT_BLOCKS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "开发门槛通过；下一步先冻结严重度排序和输入哈希，再读取2021年后的独立验证。"
            if passed
            else "多域严重度排序未达到冻结门槛；保留完整阈值前沿并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rule_count": len(rules),
                "best_development_rule": best,
                "validation_2021_plus_loaded": False,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
