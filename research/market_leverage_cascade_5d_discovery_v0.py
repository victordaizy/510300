"""510300 全市场融资杠杆与卖压级联的五日二元状态发现研究。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


FIT_START = pd.Timestamp("2015-01-06")
FIT_END = pd.Timestamp("2018-12-31")
RANDOM_STATE = 20260828

MARGIN_FILE = PROJECT_ROOT / "data" / "raw" / "market_margin_leverage_v0" / "market_margin_sh_sz_daily.parquet"
MARGIN_METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "market_margin_leverage_v0" / "metadata.json"
BREADTH_FILE = PROJECT_ROOT / "data" / "features" / "all_a_fragility_shape_daily_discovery_v0.parquet"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
OUTPUT_OFFSETS = PROJECT_ROOT / "data" / "research" / "510300_market_leverage_cascade_5d_discovery_v0" / "offset_metrics.parquet"
OUTPUT_BLOCKS = PROJECT_ROOT / "data" / "research" / "510300_market_leverage_cascade_5d_discovery_v0" / "block_predictions.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_market_leverage_cascade_5d_discovery_v0.json"

FEATURE_COLUMNS = [
    "etf_ret1",
    "etf_ret5",
    "etf_ret20",
    "etf_vol20",
    "etf_drawdown60",
    "etf_intraday_ret",
    "margin_balance_growth1",
    "margin_balance_growth5",
    "margin_balance_growth20",
    "margin_balance_growth63",
    "margin_balance_z252",
    "margin_buy_to_market_amount",
    "margin_net_change_to_market_amount",
    "margin_net_change5_to_market_amount",
    "margin_implied_repay_to_market_amount",
    "margin_balance_turnover_days",
    "margin_rqye_share",
    "margin_buy_z20",
    "breadth_cs_mean_ret",
    "breadth_share_down2",
    "breadth_share_down5",
    "breadth_share_limit_down",
    "breadth_share_close_near_low",
    "breadth_down_amount_share",
    "breadth_cs_dispersion",
    "breadth_top10pct_amount_share",
    "breadth_total_amount_change5",
    "interaction_deleveraging_down_amount",
    "interaction_crowded_broad_selloff",
    "interaction_repayment_broad_selloff",
]

MECHANISM_COMPONENTS = {
    "margin_balance_z252": 1.0,
    "margin_balance_growth1": -1.0,
    "margin_balance_growth5": -1.0,
    "margin_implied_repay_to_market_amount": 1.0,
    "breadth_share_down5": 1.0,
    "breadth_down_amount_share": 1.0,
    "interaction_deleveraging_down_amount": 1.0,
    "interaction_crowded_broad_selloff": 1.0,
    "etf_drawdown60": -1.0,
}


def _load_breadth() -> pd.DataFrame:
    frame = pd.read_parquet(BREADTH_FILE)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["breadth_total_amount_change5"] = pd.to_numeric(frame["total_amount_log"], errors="coerce").diff(5)
    rename = {
        "cs_mean_ret": "breadth_cs_mean_ret",
        "share_down2": "breadth_share_down2",
        "share_down5": "breadth_share_down5",
        "share_limit_down": "breadth_share_limit_down",
        "share_close_near_low": "breadth_share_close_near_low",
        "down_amount_share": "breadth_down_amount_share",
        "cs_dispersion": "breadth_cs_dispersion",
        "top10pct_amount_share": "breadth_top10pct_amount_share",
    }
    columns = ["date", "total_amount_log", "breadth_total_amount_change5", *rename]
    return frame[columns].rename(columns=rename)


def _load_margin_features(breadth: pd.DataFrame) -> pd.DataFrame:
    margin = pd.read_parquet(MARGIN_FILE)
    margin["date"] = pd.to_datetime(margin["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    margin = margin[margin["date"] <= DEVELOPMENT_CUTOFF].copy()
    margin = margin.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    margin = margin.merge(
        breadth[["date", "total_amount_log"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    balance = pd.to_numeric(margin["market_rzye"], errors="raise")
    buy = pd.to_numeric(margin["market_rzmre"], errors="raise")
    rqye = pd.to_numeric(margin["market_rqye"], errors="coerce")
    market_amount = np.exp(pd.to_numeric(margin["total_amount_log"], errors="coerce"))
    log_balance = np.log(balance)
    balance_change = balance.diff(1)
    margin["margin_balance_growth1"] = log_balance.diff(1)
    margin["margin_balance_growth5"] = log_balance.diff(5)
    margin["margin_balance_growth20"] = log_balance.diff(20)
    margin["margin_balance_growth63"] = log_balance.diff(63)
    mean252 = log_balance.rolling(252, min_periods=126).mean()
    std252 = log_balance.rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)
    margin["margin_balance_z252"] = (log_balance - mean252) / std252
    margin["margin_buy_to_market_amount"] = buy / market_amount
    margin["margin_net_change_to_market_amount"] = balance_change / market_amount
    margin["margin_net_change5_to_market_amount"] = balance.diff(5) / market_amount.rolling(5, min_periods=3).sum()
    implied_repayment = buy - balance_change
    margin["margin_implied_repay_to_market_amount"] = implied_repayment / market_amount
    margin["margin_balance_turnover_days"] = balance / market_amount.rolling(20, min_periods=10).mean()
    margin["margin_rqye_share"] = rqye / balance
    buy_log = np.log(buy.where(buy > 0.0))
    buy_mean = buy_log.rolling(20, min_periods=10).mean()
    buy_std = buy_log.rolling(20, min_periods=10).std(ddof=0).replace(0.0, np.nan)
    margin["margin_buy_z20"] = (buy_log - buy_mean) / buy_std
    output_columns = [
        "date",
        "margin_balance_growth1",
        "margin_balance_growth5",
        "margin_balance_growth20",
        "margin_balance_growth63",
        "margin_balance_z252",
        "margin_buy_to_market_amount",
        "margin_net_change_to_market_amount",
        "margin_net_change5_to_market_amount",
        "margin_implied_repay_to_market_amount",
        "margin_balance_turnover_days",
        "margin_rqye_share",
        "margin_buy_z20",
    ]
    return margin[output_columns]


def _build_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    dividends = _load_dividends()
    market = _load_market(dividends)
    market["date"] = market["date"].astype("datetime64[ns]")
    breadth = _load_breadth()
    margin = _load_margin_features(breadth)
    margin = margin.rename(columns={"date": "margin_observation_date"})
    frame = pd.merge_asof(
        market.sort_values("date"),
        margin.sort_values("margin_observation_date"),
        left_on="date",
        right_on="margin_observation_date",
        direction="backward",
        allow_exact_matches=False,
    )
    breadth_current = breadth.rename(columns={"total_amount_log": "breadth_total_amount_log"})
    frame = frame.merge(breadth_current, on="date", how="left", validate="one_to_one")
    valid_margin = frame["margin_observation_date"].notna()
    violations = int(
        (frame.loc[valid_margin, "margin_observation_date"] >= frame.loc[valid_margin, "date"]).sum()
    )
    if violations:
        raise ValueError(f"两融汇总时点违规：{violations}")
    frame["margin_observation_age_calendar_days"] = (
        frame["date"] - frame["margin_observation_date"]
    ).dt.days
    frame["interaction_deleveraging_down_amount"] = (
        -frame["margin_balance_growth1"] * frame["breadth_down_amount_share"]
    )
    frame["interaction_crowded_broad_selloff"] = (
        frame["margin_balance_z252"] * frame["breadth_share_down2"]
    )
    frame["interaction_repayment_broad_selloff"] = (
        frame["margin_implied_repay_to_market_amount"] * frame["breadth_down_amount_share"]
    )
    audit = {
        "rule": "交易日T两融汇总于T+1上午发布；T日收盘信号仅连接严格早于T的汇总日期",
        "matched_rows": int(valid_margin.sum()),
        "violations": violations,
        "maximum_observation_age_calendar_days_2019_2020": int(
            frame.loc[
                frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF),
                "margin_observation_age_calendar_days",
            ].max()
        ),
    }
    return frame.sort_values("date").reset_index(drop=True), audit


def _add_mechanism_scores(
    frame: pd.DataFrame,
    fit_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], float, float]:
    result = frame.copy()
    pieces: list[pd.Series] = []
    parameters: dict[str, Any] = {}
    for column, sign in MECHANISM_COMPONENTS.items():
        fit = pd.to_numeric(result.loc[fit_mask, column], errors="coerce")
        median = float(fit.median())
        scale = float(fit.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        values = pd.to_numeric(result[column], errors="coerce").fillna(median)
        pieces.append(sign * (values - median) / scale)
        parameters[column] = {"median": median, "scale": scale, "risk_sign": sign}
    result["score_mechanism_risk"] = pd.concat(pieces, axis=1).mean(axis=1)
    fit_scores = result.loc[fit_mask, "score_mechanism_risk"]
    return result, parameters, float(fit_scores.quantile(0.80)), float(fit_scores.quantile(0.90))


def _fit_models(frame: pd.DataFrame, fit_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    valid_fit = fit_mask & result["target_good5"].notna()
    x_fit = result.loc[valid_fit, FEATURE_COLUMNS]
    y_good = result.loc[valid_fit, "target_good5"].astype(int)
    if y_good.nunique() != 2:
        raise ValueError("拟合段五日状态标签不足两个类别")
    common_prefix = [("impute", SimpleImputer(strategy="median"))]
    logistic_good = Pipeline(
        [
            *common_prefix,
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.05, solver="liblinear", max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    hgb_good = Pipeline(
        [
            *common_prefix,
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=100,
                    max_leaf_nodes=7,
                    max_depth=2,
                    min_samples_leaf=35,
                    l2_regularization=15.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    bad_threshold = float(result.loc[valid_fit, "future5_excess_factor"].quantile(0.20))
    y_bad = (result.loc[valid_fit, "future5_excess_factor"] <= bad_threshold).astype(int)
    logistic_bad = Pipeline(
        [
            *common_prefix,
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_good.fit(x_fit, y_good)
    hgb_good.fit(x_fit, y_good)
    logistic_bad.fit(x_fit, y_bad)
    valid_prediction = result[FEATURE_COLUMNS].notna().any(axis=1)
    x_all = result.loc[valid_prediction, FEATURE_COLUMNS]
    result["prob_logistic_good5"] = np.nan
    result["prob_hgb_good5"] = np.nan
    result["prob_logistic_bad_tail5"] = np.nan
    result.loc[valid_prediction, "prob_logistic_good5"] = logistic_good.predict_proba(x_all)[:, 1]
    result.loc[valid_prediction, "prob_hgb_good5"] = hgb_good.predict_proba(x_all)[:, 1]
    result.loc[valid_prediction, "prob_logistic_bad_tail5"] = logistic_bad.predict_proba(x_all)[:, 1]
    result["prob_ensemble_good5"] = result[["prob_logistic_good5", "prob_hgb_good5"]].mean(axis=1)

    selection = result["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF) & result["target_good5"].notna()
    selection_label = result.loc[selection, "target_good5"].astype(int)
    aucs = {
        "logistic_good_selection_auc": _safe_float(
            roc_auc_score(selection_label, result.loc[selection, "prob_logistic_good5"])
        ),
        "hgb_good_selection_auc": _safe_float(
            roc_auc_score(selection_label, result.loc[selection, "prob_hgb_good5"])
        ),
        "ensemble_good_selection_auc": _safe_float(
            roc_auc_score(selection_label, result.loc[selection, "prob_ensemble_good5"])
        ),
    }
    good_coefficients = logistic_good.named_steps["model"].coef_[0]
    bad_coefficients = logistic_bad.named_steps["model"].coef_[0]
    summary = {
        "fit_rows": int(valid_fit.sum()),
        "fit_good_share": float(y_good.mean()),
        "bad_tail_threshold_excess_factor": bad_threshold,
        "selection_rolling_label_auc": aucs,
        "logistic_good_coefficients_standardized": {
            column: float(value) for column, value in zip(FEATURE_COLUMNS, good_coefficients, strict=True)
        },
        "logistic_bad_coefficients_standardized": {
            column: float(value) for column, value in zip(FEATURE_COLUMNS, bad_coefficients, strict=True)
        },
        "hgb_parameters": hgb_good.named_steps["model"].get_params(),
    }
    return result, summary


def _rules(q80: float, q90: float) -> dict[str, Callable[[pd.Series, int], int]]:
    return {
        "LEVERAGE_LOGISTIC_GOOD_050": lambda row, previous: int(float(row["prob_logistic_good5"]) >= 0.50),
        "LEVERAGE_LOGISTIC_HYSTERESIS_045_055": lambda row, previous: (
            1
            if float(row["prob_logistic_good5"]) >= 0.55
            else 0
            if float(row["prob_logistic_good5"]) <= 0.45
            else int(previous)
        ),
        "LEVERAGE_HGB_GOOD_050": lambda row, previous: int(float(row["prob_hgb_good5"]) >= 0.50),
        "LEVERAGE_ENSEMBLE_GOOD_050": lambda row, previous: int(float(row["prob_ensemble_good5"]) >= 0.50),
        "LEVERAGE_LOGISTIC_BAD_TAIL_050": lambda row, previous: int(
            float(row["prob_logistic_bad_tail5"]) < 0.50
        ),
        "LEVERAGE_CASCADE_RISK_Q80": lambda row, previous: int(float(row["score_mechanism_risk"]) < q80),
        "LEVERAGE_CASCADE_RISK_Q90": lambda row, previous: int(float(row["score_mechanism_risk"]) < q90),
    }


def main() -> int:
    required = [MARGIN_FILE, MARGIN_METADATA_FILE, BREADTH_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, point_in_time_audit = _build_frame()
    fit_mask = frame["date"].between(FIT_START, FIT_END)
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 800 or int(selection_mask.sum()) < 400:
        raise ValueError("拟合段或选择段样本不足")
    frame, mechanism_parameters, q80, q90 = _add_mechanism_scores(frame, fit_mask)
    frame, model_summary = _fit_models(frame, fit_mask)

    offset_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    for rule_name, rule in _rules(q80, q90).items():
        for offset in range(HORIZON):
            metrics, blocks = _evaluate_rule_offset(frame, dividends, rule_name, rule, offset)
            offset_rows.append(metrics)
            block_frames.append(blocks)
    offsets = pd.DataFrame(offset_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    aggregates = _aggregate_offsets(offsets)
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
        "margin_observation_date",
        "margin_observation_age_calendar_days",
        *FEATURE_COLUMNS,
        "score_mechanism_risk",
        "prob_logistic_good5",
        "prob_hgb_good5",
        "prob_ensemble_good5",
        "prob_logistic_bad_tail5",
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
        "project_id": "510300_MARKET_LEVERAGE_CASCADE_5D_DISCOVERY_V0",
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
            "calendar_offsets": list(range(HORIZON)),
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个日历错位的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "point_in_time_audit": point_in_time_audit,
        "feature_columns": FEATURE_COLUMNS,
        "mechanism_score": {
            "components": MECHANISM_COMPONENTS,
            "fit_transform_parameters": mechanism_parameters,
            "fit_q80_threshold": q80,
            "fit_q90_threshold": q90,
        },
        "models": model_summary,
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
            "开发门槛通过；下一步先冻结规则和输入哈希，再读取2021年后的独立验证。"
            if passed
            else "全市场融资杠杆与卖压级联未达到冻结门槛；保留负结果并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
