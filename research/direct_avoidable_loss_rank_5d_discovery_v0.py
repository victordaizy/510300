"""直接预测未来五日持仓相对现金的劣势，并按可避免损失排序。"""

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
from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    FEATURE_COLUMNS as GLOBAL_FEATURES,
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
from market_leverage_cascade_5d_discovery_v0 import FEATURE_COLUMNS as LEVERAGE_FEATURES
from multi_domain_severity_rank_5d_discovery_v0 import _fit_ecdf, _rolling_last_percentile


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
RANDOM_STATE = 20260828
TAIL_COVERAGES = [0.05, 0.10, 0.20]

ALL_A_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
GLOBAL_FILE = PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
LEVERAGE_FILE = PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_avoidable_loss_rank_5d_discovery_v0.parquet"
)
OUTPUT_OFFSETS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_direct_avoidable_loss_rank_5d_discovery_v0"
    / "offset_metrics.parquet"
)
OUTPUT_BLOCKS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_direct_avoidable_loss_rank_5d_discovery_v0"
    / "block_predictions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_direct_avoidable_loss_rank_5d_discovery_v0.json"
)


@dataclass(frozen=True)
class ScoreSpec:
    feature_group: str
    target: str
    estimator: str
    raw_column: str


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_domain(path: Any, feature_columns: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path, columns=["date", *feature_columns])
    frame = _normalize_dates(frame)
    renamed = {column: f"{prefix}{column}" for column in feature_columns}
    return frame.rename(columns=renamed), list(renamed.values())


def _build_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    all_a, all_a_columns = _load_domain(ALL_A_FILE, list(ALL_A_FEATURES), "alla__")
    global_risk, global_columns = _load_domain(GLOBAL_FILE, list(GLOBAL_FEATURES), "global__")
    leverage, leverage_columns = _load_domain(LEVERAGE_FILE, list(LEVERAGE_FEATURES), "leverage__")

    frame = market.merge(all_a, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_risk, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(leverage, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)
    frame["target_end_date"] = frame["date"].shift(-HORIZON)
    frame["signed_cash_advantage5"] = frame["future5_cash_factor"] - frame["future5_full_factor"]
    frame["avoidable_loss5"] = frame["signed_cash_advantage5"].clip(lower=0.0)

    feature_groups = {
        "ALL_DOMAINS": [*all_a_columns, *global_columns, *leverage_columns],
        "ALL_A_ONLY": all_a_columns,
        "GLOBAL_AND_LEVERAGE": [*global_columns, *leverage_columns],
    }
    fit_mask = (
        frame["date"].between(FIT_START, FIT_END)
        & frame["target_end_date"].le(FIT_END)
        & frame["signed_cash_advantage5"].notna()
    )
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("直接损失模型的清洗后拟合段或选择段覆盖不足")

    for columns in feature_groups.values():
        frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan)

    audit = {
        "common_rows": int(len(frame)),
        "purged_fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "fit_first_date": frame.loc[fit_mask, "date"].min().date().isoformat(),
        "fit_last_signal_date": frame.loc[fit_mask, "date"].max().date().isoformat(),
        "fit_last_target_end_date": frame.loc[fit_mask, "target_end_date"].max().date().isoformat(),
        "selection_first_date": frame.loc[selection_mask, "date"].min().date().isoformat(),
        "selection_last_date": frame.loc[selection_mask, "date"].max().date().isoformat(),
        "feature_counts": {name: len(columns) for name, columns in feature_groups.items()},
        "fit_bad_block_share": float((frame.loc[fit_mask, "signed_cash_advantage5"] > 0.0).mean()),
        "selection_bad_block_share": float(
            (frame.loc[selection_mask, "signed_cash_advantage5"] > 0.0).mean()
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


def _score_models(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[ScoreSpec], dict[str, Any]]:
    output = frame.copy()
    fit_mask = (
        output["date"].between(FIT_START, FIT_END)
        & output["target_end_date"].le(FIT_END)
        & output["signed_cash_advantage5"].notna()
    )
    fit = output.loc[fit_mask]
    signed_target = fit["signed_cash_advantage5"].to_numpy(dtype=float)
    loss_target = fit["avoidable_loss5"].to_numpy(dtype=float)
    loss_q80 = float(np.quantile(loss_target, 0.80))
    loss_q90 = float(np.quantile(loss_target, 0.90))
    severity_weights = 1.0 + 4.0 * np.clip(loss_target / max(loss_q90, 1e-8), 0.0, 3.0)

    specs: list[ScoreSpec] = []
    summaries: dict[str, Any] = {
        "fit_avoidable_loss_q80": loss_q80,
        "fit_avoidable_loss_q90": loss_q90,
        "severity_weight_minimum": float(severity_weights.min()),
        "severity_weight_median": float(np.median(severity_weights)),
        "severity_weight_maximum": float(severity_weights.max()),
        "models": {},
    }

    def add_regression(
        *,
        name: str,
        group: str,
        target_name: str,
        model: Any,
        fit_parameters: dict[str, Any] | None = None,
    ) -> None:
        columns = feature_groups[group]
        target = signed_target if target_name == "SIGNED_CASH_ADVANTAGE" else loss_target
        parameters = fit_parameters or {}
        model.fit(fit[columns], target, **parameters)
        raw_column = f"score_raw_{name.lower()}"
        output[raw_column] = np.asarray(model.predict(output[columns]), dtype=float)
        specs.append(ScoreSpec(group, target_name, type(model.named_steps["model"]).__name__, raw_column))
        summaries["models"][name] = {
            "feature_group": group,
            "feature_count": len(columns),
            "target": target_name,
            "estimator": type(model.named_steps["model"]).__name__,
            "parameters": model.named_steps["model"].get_params(deep=False),
            "fit_parameters": sorted(parameters),
        }

    all_columns = feature_groups["ALL_DOMAINS"]
    all_a_columns = feature_groups["ALL_A_ONLY"]
    external_columns = feature_groups["GLOBAL_AND_LEVERAGE"]

    add_regression(
        name="RIDGE_SIGNED_ALL",
        group="ALL_DOMAINS",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_linear_pipeline(Ridge(alpha=100.0)),
    )
    add_regression(
        name="ELASTIC_SIGNED_ALL",
        group="ALL_DOMAINS",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_linear_pipeline(
            ElasticNet(alpha=0.0025, l1_ratio=0.20, max_iter=10000, random_state=RANDOM_STATE)
        ),
    )
    add_regression(
        name="HGB_SIGNED_ALL",
        group="ALL_DOMAINS",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_tree_pipeline(
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
        name="HGB_LOSS_WEIGHTED_ALL",
        group="ALL_DOMAINS",
        target_name="AVOIDABLE_LOSS",
        model=_tree_pipeline(
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
        fit_parameters={"model__sample_weight": severity_weights},
    )
    add_regression(
        name="RF_SIGNED_ALL",
        group="ALL_DOMAINS",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_tree_pipeline(
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
        name="EXTRA_LOSS_WEIGHTED_ALL",
        group="ALL_DOMAINS",
        target_name="AVOIDABLE_LOSS",
        model=_tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=10,
                max_features=0.75,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        fit_parameters={"model__sample_weight": severity_weights},
    )
    add_regression(
        name="GBR_HUBER_SIGNED_ALL",
        group="ALL_DOMAINS",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_tree_pipeline(
            GradientBoostingRegressor(
                loss="huber",
                learning_rate=0.03,
                n_estimators=250,
                max_depth=2,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )
        ),
    )
    add_regression(
        name="GBR_Q90_LOSS_ALL",
        group="ALL_DOMAINS",
        target_name="AVOIDABLE_LOSS",
        model=_tree_pipeline(
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
    add_regression(
        name="RIDGE_SIGNED_ALL_A",
        group="ALL_A_ONLY",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_linear_pipeline(Ridge(alpha=50.0)),
    )
    add_regression(
        name="EXTRA_LOSS_ALL_A",
        group="ALL_A_ONLY",
        target_name="AVOIDABLE_LOSS",
        model=_tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=10,
                max_features=0.75,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    )
    add_regression(
        name="RIDGE_SIGNED_EXTERNAL",
        group="GLOBAL_AND_LEVERAGE",
        target_name="SIGNED_CASH_ADVANTAGE",
        model=_linear_pipeline(Ridge(alpha=50.0)),
    )

    severe_label = (loss_target >= loss_q80).astype(int)
    severe_classifier = _tree_pipeline(
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
    severe_classifier.fit(fit[all_columns], severe_label)
    severe_column = "score_raw_hgb_severe20_all"
    output[severe_column] = severe_classifier.predict_proba(output[all_columns])[:, 1]
    specs.append(ScoreSpec("ALL_DOMAINS", "SEVERE_LOSS_TOP20", "HistGradientBoostingClassifier", severe_column))
    summaries["models"]["HGB_SEVERE20_ALL"] = {
        "feature_group": "ALL_DOMAINS",
        "feature_count": len(all_columns),
        "target": "SEVERE_LOSS_TOP20",
        "estimator": "HistGradientBoostingClassifier",
        "parameters": severe_classifier.named_steps["model"].get_params(deep=False),
    }

    any_bad_label = (loss_target > 0.0).astype(int)
    bad_classifier = _linear_pipeline(
        LogisticRegression(
            C=0.10,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    )
    bad_classifier.fit(fit[all_columns], any_bad_label)
    conditional_loss = _tree_pipeline(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=10,
            max_features=0.50,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )
    positive_fit = fit["avoidable_loss5"] > 0.0
    conditional_loss.fit(fit.loc[positive_fit, all_columns], fit.loc[positive_fit, "avoidable_loss5"])
    bad_probability = bad_classifier.predict_proba(output[all_columns])[:, 1]
    conditional_prediction = np.clip(conditional_loss.predict(output[all_columns]), 0.0, None)
    two_stage_column = "score_raw_two_stage_expected_loss_all"
    output[two_stage_column] = bad_probability * conditional_prediction
    specs.append(ScoreSpec("ALL_DOMAINS", "TWO_STAGE_EXPECTED_LOSS", "LogisticRegression_x_RandomForestRegressor", two_stage_column))
    summaries["models"]["TWO_STAGE_EXPECTED_LOSS_ALL"] = {
        "feature_group": "ALL_DOMAINS",
        "feature_count": len(all_columns),
        "target": "TWO_STAGE_EXPECTED_LOSS",
        "estimator": "LogisticRegression_x_RandomForestRegressor",
        "positive_fit_rows": int(positive_fit.sum()),
    }

    del all_a_columns, external_columns
    return output, specs, summaries


def _add_calibrations(
    frame: pd.DataFrame,
    specs: list[ScoreSpec],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    output = frame.copy()
    fit_mask = (
        output["date"].between(FIT_START, FIT_END)
        & output["target_end_date"].le(FIT_END)
        & output["signed_cash_advantage5"].notna()
    )
    selection_mask = output["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    fit_severe_threshold = float(output.loc[fit_mask, "avoidable_loss5"].quantile(0.80))
    score_metadata: list[dict[str, Any]] = []
    score_audit: dict[str, Any] = {}
    for spec in specs:
        fit_column = spec.raw_column.replace("score_raw_", "risk_fit_ecdf_")
        rolling_column = spec.raw_column.replace("score_raw_", "risk_rolling252_")
        output[fit_column] = _fit_ecdf(output[spec.raw_column], fit_mask)
        output[rolling_column] = _rolling_last_percentile(output[spec.raw_column])
        score_metadata.extend(
            [
                {
                    "score_column": fit_column,
                    "raw_column": spec.raw_column,
                    "calibration": "FIT_ECDF",
                    "feature_group": spec.feature_group,
                    "target": spec.target,
                    "estimator": spec.estimator,
                },
                {
                    "score_column": rolling_column,
                    "raw_column": spec.raw_column,
                    "calibration": "ROLLING252_PERCENTILE",
                    "feature_group": spec.feature_group,
                    "target": spec.target,
                    "estimator": spec.estimator,
                },
            ]
        )
        valid = selection_mask & output[spec.raw_column].notna() & output["avoidable_loss5"].notna()
        labels = output.loc[valid, "avoidable_loss5"] >= fit_severe_threshold
        raw = output.loc[valid, spec.raw_column]
        actual = output.loc[valid, "avoidable_loss5"]
        auc = roc_auc_score(labels.astype(int), raw) if labels.nunique() == 2 else np.nan
        correlation = spearmanr(raw, actual, nan_policy="omit").statistic
        score_audit[spec.raw_column] = {
            "selection_severe20_auc": _safe_float(auc),
            "selection_avoidable_loss_spearman": _safe_float(correlation),
        }
    return output, score_metadata, {
        "fit_severe20_avoidable_loss_threshold": fit_severe_threshold,
        "by_raw_score": score_audit,
    }


def _make_rule(score_column: str, cutoff: float) -> Any:
    def decide(row: pd.Series, previous: int) -> int:
        del previous
        score = float(row[score_column])
        if not np.isfinite(score):
            return 1
        return int(score < cutoff)

    return decide


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
    frame, specs, model_summary = _score_models(frame, feature_groups)
    frame, score_metadata, score_audit = _add_calibrations(frame, specs)

    # 精确账户模拟器的诊断字段映射到本轮首要直接损失分数；每条规则仍使用自己的分数列。
    frame["prob_logistic_good5"] = 1.0 - frame["risk_fit_ecdf_ridge_signed_all"]
    frame["prob_hgb_good5"] = 1.0 - frame["risk_fit_ecdf_hgb_signed_all"]
    frame["prob_ensemble_good5"] = 1.0 - frame["risk_fit_ecdf_two_stage_expected_loss_all"]
    frame["prob_logistic_bad_tail5"] = frame["risk_fit_ecdf_hgb_severe20_all"]
    frame["score_mechanism_risk"] = frame["risk_fit_ecdf_extra_loss_weighted_all"]

    offset_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    for item in score_metadata:
        score_column = str(item["score_column"])
        short_name = score_column.replace("risk_fit_ecdf_", "").replace("risk_rolling252_", "")
        for coverage in TAIL_COVERAGES:
            cutoff = 1.0 - coverage
            rule_name = (
                f"DIRECT_LOSS_{short_name.upper()}_{item['calibration']}_TOP{int(coverage * 100):02d}"
            )
            rule = _make_rule(score_column, cutoff)
            for offset in range(HORIZON):
                metrics, blocks = _evaluate_rule_offset(frame, dividends, rule_name, rule, offset)
                metrics.update(
                    {
                        "score_column": score_column,
                        "raw_column": item["raw_column"],
                        "calibration": item["calibration"],
                        "intended_tail_coverage": coverage,
                        "feature_group": item["feature_group"],
                        "model_target": item["target"],
                        "estimator": item["estimator"],
                    }
                )
                blocks["score_column"] = score_column
                blocks["risk_score"] = blocks["signal_date"].map(frame.set_index("date")[score_column])
                offset_rows.append(metrics)
                block_frames.append(blocks)

    offsets = pd.DataFrame(offset_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    aggregates = _aggregate_offsets(offsets)
    aggregate_metadata = offsets.groupby("rule", sort=False).first()[
        [
            "score_column",
            "raw_column",
            "calibration",
            "intended_tail_coverage",
            "feature_group",
            "model_target",
            "estimator",
        ]
    ]
    for aggregate in aggregates:
        meta = aggregate_metadata.loc[aggregate["rule"]]
        for key in aggregate_metadata.columns:
            value = meta[key]
            aggregate[key] = float(value) if key == "intended_tail_coverage" else str(value)
    best = aggregates[0]
    passed = bool(best["development_gate"])

    score_columns = [spec.raw_column for spec in specs]
    score_columns.extend(item["score_column"] for item in score_metadata)
    feature_output = frame[
        [
            "date",
            "target_end_date",
            "etf_open",
            "etf_close",
            "benchmark_close",
            "future5_full_factor",
            "future5_cash_factor",
            "future5_excess_factor",
            "signed_cash_advantage5",
            "avoidable_loss5",
            *score_columns,
        ]
    ].copy()
    _atomic_parquet(feature_output, OUTPUT_FEATURES)
    _atomic_parquet(offsets, OUTPUT_OFFSETS)
    _atomic_parquet(blocks, OUTPUT_BLOCKS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DIRECT_AVOIDABLE_LOSS_RANK_5D_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "state_meanings": {"0": "CASH_CNY", "1": "FULL_510300"},
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "holding_block_trading_days": HORIZON,
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "fit_target_purge": "拟合信号的五日目标结束日期不得晚于2018-12-31",
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "tail_coverages": TAIL_COVERAGES,
            "score_calibrations": ["FIT_ECDF", "ROLLING252_PERCENTILE"],
            "calendar_offsets": list(range(HORIZON)),
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个日历错位的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "model_summary": model_summary,
        "score_audit": score_audit,
        "score_count_raw": int(len(specs)),
        "rule_count": int(len(aggregates)),
        "best_development_rule": best,
        "all_rule_aggregates": aggregates,
        "all_offset_metrics": offsets.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "offset_metrics": str(OUTPUT_OFFSETS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "block_predictions": str(OUTPUT_BLOCKS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "直接可避免损失排序通过开发门槛；先冻结模型、规则和输入哈希，再读取2021年后的独立验证。"
            if passed
            else "直接可避免损失排序未达到20%开发门槛；保留完整模型前沿并继续搜索。"
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
                "rule_count": len(aggregates),
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
