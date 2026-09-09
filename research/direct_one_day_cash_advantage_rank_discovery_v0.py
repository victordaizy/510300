"""直接学习逐日空仓相对满仓的开盘到开盘现金优势。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from all_a_fragility_shape_5d_discovery_v0 import MODEL_FEATURES as ALL_A_FEATURES
from daily_tail_state_machine_discovery_v0 import (
    PolicySpec,
    START_PERTURBATIONS,
    _aggregate,
    _evaluate_policy_start,
)
from global_liquidity_regime_5d_discovery_v0 import (
    CASH_ANNUAL_RATE,
    DEVELOPMENT_CUTOFF,
    FEATURE_COLUMNS as GLOBAL_FEATURES,
    PROJECT_ROOT,
    SELECTION_START,
    TRADING_DAYS_PER_YEAR,
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)
from market_leverage_cascade_5d_discovery_v0 import FEATURE_COLUMNS as LEVERAGE_FEATURES
from multi_domain_severity_rank_5d_discovery_v0 import _fit_ecdf, _rolling_last_percentile


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
RANDOM_STATE = 20260828
TAIL_COVERAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
EXACT_FRONTIER_LIMIT = 40

ALL_A_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
GLOBAL_FILE = PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
LEVERAGE_FILE = PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_one_day_cash_advantage_rank_discovery_v0.parquet"
)
OUTPUT_GROSS_FRONTIER = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_direct_one_day_cash_advantage_rank_discovery_v0"
    / "gross_rule_frontier.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_direct_one_day_cash_advantage_rank_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_direct_one_day_cash_advantage_rank_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_direct_one_day_cash_advantage_rank_discovery_v0.json"
)


@dataclass(frozen=True)
class ScoreSpec:
    raw_column: str
    feature_group: str
    target: str
    estimator: str


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_domain(
    path: Any,
    feature_columns: list[str],
    prefix: str,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    extras = extra_columns or []
    frame = pd.read_parquet(path, columns=["date", *feature_columns, *extras])
    frame = _normalize_dates(frame)
    renamed = {column: f"{prefix}{column}" for column in feature_columns}
    return frame.rename(columns=renamed), list(renamed.values())


def _build_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    all_a, all_a_columns = _load_domain(
        ALL_A_FILE,
        list(ALL_A_FEATURES),
        "alla__",
        extra_columns=["one_day_open_total_return"],
    )
    global_risk, global_columns = _load_domain(GLOBAL_FILE, list(GLOBAL_FEATURES), "global__")
    leverage, leverage_columns = _load_domain(LEVERAGE_FILE, list(LEVERAGE_FEATURES), "leverage__")
    frame = market.merge(all_a, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_risk, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(leverage, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)

    frame["target1_end_date"] = frame["date"].shift(-2)
    cash_factor1 = 1.0 + CASH_ANNUAL_RATE / TRADING_DAYS_PER_YEAR
    frame["signed_cash_advantage1"] = cash_factor1 - (1.0 + frame["one_day_open_total_return"])
    frame["avoidable_loss1"] = frame["signed_cash_advantage1"].clip(lower=0.0)
    frame["avoidable_loss5"] = (frame["future5_cash_factor"] - frame["future5_full_factor"]).clip(lower=0.0)

    feature_groups = {
        "ALL_DOMAINS": [*all_a_columns, *global_columns, *leverage_columns],
        "GLOBAL_ONLY": global_columns,
        "ALL_A_ONLY": all_a_columns,
        "GLOBAL_AND_ALL_A": [*global_columns, *all_a_columns],
    }
    all_feature_columns = sorted({column for columns in feature_groups.values() for column in columns})
    frame[all_feature_columns] = frame[all_feature_columns].replace([np.inf, -np.inf], np.nan)

    fit_mask = (
        frame["date"].between(FIT_START, FIT_END)
        & frame["target1_end_date"].le(FIT_END)
        & frame["signed_cash_advantage1"].notna()
    )
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("单日现金优势模型的清洗后拟合段或选择段覆盖不足")
    audit = {
        "rows": int(len(frame)),
        "purged_fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "fit_last_signal_date": frame.loc[fit_mask, "date"].max().date().isoformat(),
        "fit_last_target_end_date": frame.loc[fit_mask, "target1_end_date"].max().date().isoformat(),
        "feature_counts": {name: len(columns) for name, columns in feature_groups.items()},
        "fit_positive_cash_advantage_share": float(
            (frame.loc[fit_mask, "signed_cash_advantage1"] > 0.0).mean()
        ),
        "selection_positive_cash_advantage_share": float(
            (frame.loc[selection_mask, "signed_cash_advantage1"] > 0.0).mean()
        ),
    }
    return frame, feature_groups, audit


def _linear_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _tree_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", estimator),
        ]
    )


def _fit_scores(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[ScoreSpec], dict[str, Any]]:
    output = frame.copy()
    fit_mask = (
        output["date"].between(FIT_START, FIT_END)
        & output["target1_end_date"].le(FIT_END)
        & output["signed_cash_advantage1"].notna()
    )
    fit = output.loc[fit_mask]
    signed_target = fit["signed_cash_advantage1"].to_numpy(dtype=float)
    loss_target = fit["avoidable_loss1"].to_numpy(dtype=float)
    loss_q80 = float(np.quantile(loss_target, 0.80))
    loss_q90 = float(np.quantile(loss_target, 0.90))
    severity_weights = 1.0 + 4.0 * np.clip(loss_target / max(loss_q90, 1e-8), 0.0, 3.0)
    specs: list[ScoreSpec] = []
    summary: dict[str, Any] = {
        "fit_avoidable_loss_q80": loss_q80,
        "fit_avoidable_loss_q90": loss_q90,
        "severity_weight_range": [float(severity_weights.min()), float(severity_weights.max())],
        "models": {},
    }

    def add_regression(
        name: str,
        group: str,
        target_name: str,
        model: Pipeline,
        fit_parameters: dict[str, Any] | None = None,
    ) -> None:
        columns = feature_groups[group]
        target = signed_target if target_name == "SIGNED_CASH_ADVANTAGE1" else loss_target
        parameters = fit_parameters or {}
        model.fit(fit[columns], target, **parameters)
        raw_column = f"score_raw_{name.lower()}"
        output[raw_column] = np.asarray(model.predict(output[columns]), dtype=float)
        estimator = type(model.named_steps["model"]).__name__
        specs.append(ScoreSpec(raw_column, group, target_name, estimator))
        summary["models"][name] = {
            "feature_group": group,
            "feature_count": len(columns),
            "target": target_name,
            "estimator": estimator,
            "parameters": model.named_steps["model"].get_params(deep=False),
            "fit_parameters": sorted(parameters),
        }

    add_regression(
        "RIDGE_SIGNED_GLOBAL",
        "GLOBAL_ONLY",
        "SIGNED_CASH_ADVANTAGE1",
        _linear_pipeline(Ridge(alpha=25.0)),
    )
    add_regression(
        "RIDGE_SIGNED_ALL_A",
        "ALL_A_ONLY",
        "SIGNED_CASH_ADVANTAGE1",
        _linear_pipeline(Ridge(alpha=50.0)),
    )
    add_regression(
        "RIDGE_SIGNED_GLOBAL_ALL_A",
        "GLOBAL_AND_ALL_A",
        "SIGNED_CASH_ADVANTAGE1",
        _linear_pipeline(Ridge(alpha=75.0)),
    )
    add_regression(
        "RIDGE_SIGNED_ALL",
        "ALL_DOMAINS",
        "SIGNED_CASH_ADVANTAGE1",
        _linear_pipeline(Ridge(alpha=100.0)),
    )
    add_regression(
        "ELASTIC_SIGNED_ALL",
        "ALL_DOMAINS",
        "SIGNED_CASH_ADVANTAGE1",
        _linear_pipeline(
            ElasticNet(alpha=0.0025, l1_ratio=0.20, max_iter=10000, random_state=RANDOM_STATE)
        ),
    )
    add_regression(
        "HGB_SIGNED_ALL",
        "ALL_DOMAINS",
        "SIGNED_CASH_ADVANTAGE1",
        _tree_pipeline(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                max_iter=250,
                max_leaf_nodes=7,
                min_samples_leaf=25,
                l2_regularization=10.0,
                random_state=RANDOM_STATE,
            )
        ),
    )
    add_regression(
        "RF_SIGNED_ALL",
        "ALL_DOMAINS",
        "SIGNED_CASH_ADVANTAGE1",
        _tree_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=4,
                min_samples_leaf=15,
                max_features=0.50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    )
    add_regression(
        "EXTRA_LOSS_WEIGHTED_ALL",
        "ALL_DOMAINS",
        "AVOIDABLE_LOSS1",
        _tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=10,
                max_features=0.75,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        {"model__sample_weight": severity_weights},
    )
    add_regression(
        "GBR_Q90_LOSS_ALL",
        "ALL_DOMAINS",
        "AVOIDABLE_LOSS1",
        _tree_pipeline(
            GradientBoostingRegressor(
                loss="quantile",
                alpha=0.90,
                learning_rate=0.03,
                n_estimators=250,
                max_depth=2,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )
        ),
    )

    severe20 = (loss_target >= loss_q80).astype(int)
    severe_model = _tree_pipeline(
        HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=250,
            max_leaf_nodes=7,
            min_samples_leaf=25,
            l2_regularization=10.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )
    severe_model.fit(fit[feature_groups["ALL_DOMAINS"]], severe20)
    severe_column = "score_raw_hgb_severe20_all"
    output[severe_column] = severe_model.predict_proba(output[feature_groups["ALL_DOMAINS"]])[:, 1]
    specs.append(ScoreSpec(severe_column, "ALL_DOMAINS", "SEVERE_LOSS1_TOP20", "HistGradientBoostingClassifier"))
    summary["models"]["HGB_SEVERE20_ALL"] = {
        "feature_group": "ALL_DOMAINS",
        "feature_count": len(feature_groups["ALL_DOMAINS"]),
        "target": "SEVERE_LOSS1_TOP20",
        "estimator": "HistGradientBoostingClassifier",
        "parameters": severe_model.named_steps["model"].get_params(deep=False),
    }

    bad_label = (loss_target > 0.0).astype(int)
    bad_model = _linear_pipeline(
        LogisticRegression(
            C=0.10,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    )
    bad_model.fit(fit[feature_groups["ALL_DOMAINS"]], bad_label)
    conditional_model = _tree_pipeline(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=10,
            max_features=0.50,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )
    positive = fit["avoidable_loss1"] > 0.0
    conditional_model.fit(
        fit.loc[positive, feature_groups["ALL_DOMAINS"]],
        fit.loc[positive, "avoidable_loss1"],
    )
    expected_column = "score_raw_two_stage_expected_loss_all"
    output[expected_column] = (
        bad_model.predict_proba(output[feature_groups["ALL_DOMAINS"]])[:, 1]
        * np.clip(conditional_model.predict(output[feature_groups["ALL_DOMAINS"]]), 0.0, None)
    )
    specs.append(
        ScoreSpec(
            expected_column,
            "ALL_DOMAINS",
            "TWO_STAGE_EXPECTED_LOSS1",
            "LogisticRegression_x_RandomForestRegressor",
        )
    )
    summary["models"]["TWO_STAGE_EXPECTED_LOSS_ALL"] = {
        "feature_group": "ALL_DOMAINS",
        "feature_count": len(feature_groups["ALL_DOMAINS"]),
        "target": "TWO_STAGE_EXPECTED_LOSS1",
        "estimator": "LogisticRegression_x_RandomForestRegressor",
        "positive_fit_rows": int(positive.sum()),
    }

    baseline_column = "score_raw_global_selloff1_baseline"
    output[baseline_column] = -pd.to_numeric(output["global__global_min_ret1"], errors="coerce")
    specs.append(ScoreSpec(baseline_column, "GLOBAL_ONLY", "MECHANISM_BASELINE", "NEGATIVE_GLOBAL_MIN_RETURN1"))
    summary["models"]["GLOBAL_SELLOFF1_BASELINE"] = {
        "feature_group": "GLOBAL_ONLY",
        "feature_count": 1,
        "target": "MECHANISM_BASELINE",
        "estimator": "NEGATIVE_GLOBAL_MIN_RETURN1",
    }
    return output, specs, summary


def _calibrate_scores(
    frame: pd.DataFrame,
    specs: list[ScoreSpec],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    output = frame.copy()
    fit_mask = (
        output["date"].between(FIT_START, FIT_END)
        & output["target1_end_date"].le(FIT_END)
        & output["signed_cash_advantage1"].notna()
    )
    selection_mask = output["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    fit_severe_threshold = float(output.loc[fit_mask, "avoidable_loss1"].quantile(0.80))
    metadata: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for spec in specs:
        fit_column = spec.raw_column.replace("score_raw_", "risk_fit_ecdf_")
        rolling_column = spec.raw_column.replace("score_raw_", "risk_rolling252_")
        output[fit_column] = _fit_ecdf(output[spec.raw_column], fit_mask)
        output[rolling_column] = _rolling_last_percentile(output[spec.raw_column])
        for column, calibration in [(fit_column, "FIT_ECDF"), (rolling_column, "ROLLING252_PERCENTILE")]:
            metadata.append(
                {
                    "score_column": column,
                    "raw_column": spec.raw_column,
                    "calibration": calibration,
                    "feature_group": spec.feature_group,
                    "target": spec.target,
                    "estimator": spec.estimator,
                }
            )
        valid = selection_mask & output[spec.raw_column].notna() & output["avoidable_loss1"].notna()
        labels = output.loc[valid, "avoidable_loss1"] >= fit_severe_threshold
        raw = output.loc[valid, spec.raw_column]
        actual = output.loc[valid, "avoidable_loss1"]
        audit[spec.raw_column] = {
            "selection_severe20_auc": _safe_float(roc_auc_score(labels.astype(int), raw)),
            "selection_avoidable_loss1_spearman": _safe_float(
                spearmanr(raw, actual, nan_policy="omit").statistic
            ),
        }
    return output, metadata, {
        "fit_severe20_avoidable_loss1_threshold": fit_severe_threshold,
        "by_raw_score": audit,
    }


def _gross_frontier(frame: pd.DataFrame, metadata: list[dict[str, Any]]) -> pd.DataFrame:
    selection = frame[frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)].copy()
    total_avoidable = float(selection["avoidable_loss1"].sum())
    rows: list[dict[str, Any]] = []
    for item in metadata:
        score_column = str(item["score_column"])
        for coverage in TAIL_COVERAGES:
            selected = selection[score_column] >= 1.0 - coverage
            selected_advantage = selection.loc[selected, "signed_cash_advantage1"]
            selected_loss = selection.loc[selected, "avoidable_loss1"]
            rows.append(
                {
                    **item,
                    "intended_tail_coverage": coverage,
                    "selected_day_count": int(selected.sum()),
                    "gross_signed_cash_advantage1_sum": float(selected_advantage.sum()),
                    "gross_signed_cash_advantage1_mean": _safe_float(selected_advantage.mean()),
                    "gross_positive_day_rate": _safe_float((selected_advantage > 0.0).mean()),
                    "avoidable_loss1_capture": _safe_float(
                        float(selected_loss.sum()) / total_avoidable if total_avoidable > 0 else np.nan
                    ),
                }
            )
    frontier = pd.DataFrame(rows)
    frontier.sort_values(
        ["gross_signed_cash_advantage1_sum", "selected_day_count"],
        ascending=[False, True],
        inplace=True,
    )
    frontier.reset_index(drop=True, inplace=True)
    frontier["gross_rank"] = np.arange(1, len(frontier) + 1)
    frontier["selected_for_exact_account"] = frontier["gross_rank"] <= EXACT_FRONTIER_LIMIT
    return frontier


def _add_one_day_diagnostics(
    metrics: dict[str, Any],
    decisions: pd.DataFrame,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    lookup = frame.set_index("date")[["signed_cash_advantage1", "avoidable_loss1"]]
    output = decisions.copy()
    output["signed_cash_advantage1"] = output["signal_date"].map(lookup["signed_cash_advantage1"])
    output["avoidable_loss1"] = output["signal_date"].map(lookup["avoidable_loss1"])
    cash = output["state"].eq(0)
    bad = output["avoidable_loss1"].fillna(0.0) > 0.0
    total_avoidable = float(output["avoidable_loss1"].fillna(0.0).sum())
    captured = float(output.loc[cash, "avoidable_loss1"].fillna(0.0).sum())
    metrics.update(
        {
            "bad1_recall": _safe_float(cash[bad].mean()),
            "good1_false_exit": _safe_float(cash[~bad].mean()),
            "avoidable_loss1_capture": _safe_float(captured / total_avoidable if total_avoidable > 0 else np.nan),
            "selected_signed_cash_advantage1_sum": float(
                output.loc[cash, "signed_cash_advantage1"].fillna(0.0).sum()
            ),
        }
    )
    return metrics, output


def main() -> int:
    required = [ALL_A_FILE, GLOBAL_FILE, LEVERAGE_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, feature_groups, frame_audit = _build_frame()
    frame, specs, model_summary = _fit_scores(frame, feature_groups)
    frame, score_metadata, score_audit = _calibrate_scores(frame, specs)
    frontier = _gross_frontier(frame, score_metadata)
    exact_rules = frontier[frontier["selected_for_exact_account"]].copy()

    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for _, rule_row in exact_rules.iterrows():
        score_column = str(rule_row["score_column"])
        coverage = float(rule_row["intended_tail_coverage"])
        score_name = score_column.replace("risk_fit_ecdf_", "").replace("risk_rolling252_", "")
        policy_name = (
            f"DIRECT1_{score_name.upper()}_{rule_row['calibration']}_TOP{int(round(coverage * 100)):02d}"
        )
        spec = PolicySpec("PULSE", score_column, 1.0 - coverage)
        for perturbation in START_PERTURBATIONS:
            metrics, decisions = _evaluate_policy_start(
                frame,
                dividends,
                policy_name,
                spec,
                perturbation,
            )
            metrics.update(
                {
                    "raw_column": str(rule_row["raw_column"]),
                    "calibration": str(rule_row["calibration"]),
                    "intended_tail_coverage": coverage,
                    "feature_group": str(rule_row["feature_group"]),
                    "model_target": str(rule_row["target"]),
                    "estimator": str(rule_row["estimator"]),
                    "gross_rank": int(rule_row["gross_rank"]),
                }
            )
            metrics, decisions = _add_one_day_diagnostics(metrics, decisions, frame)
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    aggregate_metadata = metrics.groupby("policy", sort=False).first()[
        [
            "raw_column",
            "calibration",
            "intended_tail_coverage",
            "feature_group",
            "model_target",
            "estimator",
            "gross_rank",
        ]
    ]
    one_day_aggregates = metrics.groupby("policy", sort=False)[
        ["bad1_recall", "good1_false_exit", "avoidable_loss1_capture", "selected_signed_cash_advantage1_sum"]
    ].median()
    for aggregate in aggregates:
        meta = aggregate_metadata.loc[aggregate["policy"]]
        aggregate.update(
            {
                "raw_column": str(meta["raw_column"]),
                "calibration": str(meta["calibration"]),
                "intended_tail_coverage": float(meta["intended_tail_coverage"]),
                "feature_group": str(meta["feature_group"]),
                "model_target": str(meta["model_target"]),
                "estimator": str(meta["estimator"]),
                "gross_rank": int(meta["gross_rank"]),
                "bad1_recall_median": _safe_float(one_day_aggregates.loc[aggregate["policy"], "bad1_recall"]),
                "good1_false_exit_median": _safe_float(
                    one_day_aggregates.loc[aggregate["policy"], "good1_false_exit"]
                ),
                "avoidable_loss1_capture_median": _safe_float(
                    one_day_aggregates.loc[aggregate["policy"], "avoidable_loss1_capture"]
                ),
                "selected_signed_cash_advantage1_sum_median": _safe_float(
                    one_day_aggregates.loc[aggregate["policy"], "selected_signed_cash_advantage1_sum"]
                ),
            }
        )
    best = aggregates[0]
    passed = bool(best["development_gate"])

    raw_score_columns = [spec.raw_column for spec in specs]
    calibrated_columns = [item["score_column"] for item in score_metadata]
    output_columns = [
        "date",
        "target1_end_date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "one_day_open_total_return",
        "signed_cash_advantage1",
        "avoidable_loss1",
        "avoidable_loss5",
        *raw_score_columns,
        *calibrated_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(frontier, OUTPUT_GROSS_FRONTIER)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    _atomic_parquet(decisions, OUTPUT_DECISIONS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DIRECT_ONE_DAY_CASH_ADVANTAGE_RANK_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "decision_frequency": "每交易日",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "fit_target_purge": "拟合信号的开盘到开盘单日目标结束日期不得晚于2018-12-31",
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "tail_coverages": TAIL_COVERAGES,
            "score_calibrations": ["FIT_ECDF", "ROLLING252_PERCENTILE"],
            "gross_frontier_rule_count": int(len(frontier)),
            "exact_account_frontier_limit": EXACT_FRONTIER_LIMIT,
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "model_summary": model_summary,
        "score_audit": score_audit,
        "raw_score_count": int(len(specs)),
        "best_gross_rule": frontier.iloc[0].to_dict(),
        "best_development_policy": best,
        "all_exact_policy_aggregates": aggregates,
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "gross_rule_frontier": str(OUTPUT_GROSS_FRONTIER.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "直接单日现金优势排序通过开发门槛；冻结模型、覆盖率、成本和输入哈希后再读取独立验证。"
            if passed
            else "直接单日现金优势排序未达到20%开发门槛；保留毛前沿与精确路径并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "raw_score_count": len(specs),
                "gross_rule_count": len(frontier),
                "exact_rule_count": len(aggregates),
                "best_gross_rule": payload["best_gross_rule"],
                "best_development_policy": best,
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
