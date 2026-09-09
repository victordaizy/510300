"""按季度事前重训，发现510300下一交易日空仓尾部信号。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import all_a_fragility_shape_5d_discovery_v0 as all_a_module
import global_liquidity_regime_5d_discovery_v0 as global_module
import market_leverage_cascade_5d_discovery_v0 as leverage_module
from binary_state_feasibility_v1 import build_benchmark_ledger, simulate_binary_path, summarize_path
from multi_domain_severity_rank_5d_discovery_v0 import _rolling_last_percentile


PROJECT_ID = "510300_PREQUENTIAL_ONE_DAY_TAIL_DISCOVERY_V1"
FIT_START = pd.Timestamp("2016-08-15")
PREDICTION_START = pd.Timestamp("2018-01-01")
DISCOVERY_START = pd.Timestamp("2019-01-01")
DISCOVERY_END = pd.Timestamp("2023-12-31")
START_PERTURBATIONS = list(range(5))
RANDOM_STATE = 20260828
TAIL_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
MAX_EXACT_POLICIES = 50

INPUT_DIR = PROJECT_ROOT / "data" / "validation" / "510300_daily_consensus_catchup_v1"
ALL_A_FILE = INPUT_DIR / "all_a_features_2016_2025.parquet"
GLOBAL_FILE = INPUT_DIR / "global_features_2015_2025.parquet"
LEVERAGE_FILE = INPUT_DIR / "leverage_features_2015_2025.parquet"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_prequential_one_day_tail_discovery_v1.parquet"
)
OUTPUT_FRONTIER = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_prequential_one_day_tail_discovery_v1"
    / "gross_frontier.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_prequential_one_day_tail_discovery_v1"
    / "exact_start_perturbation_metrics.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_prequential_one_day_tail_discovery_v1.json"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_group: str
    training_window_rows: int | None
    model_kind: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    all_a_features = list(all_a_module.MODEL_FEATURES)
    global_features = list(global_module.FEATURE_COLUMNS)
    leverage_features = list(leverage_module.FEATURE_COLUMNS)
    all_a = _normalize_dates(
        pd.read_parquet(ALL_A_FILE, columns=["date", "one_day_open_total_return", *all_a_features])
    ).rename(columns={column: f"alla__{column}" for column in all_a_features})
    global_frame = _normalize_dates(
        pd.read_parquet(GLOBAL_FILE, columns=["date", *global_features])
    ).rename(columns={column: f"global__{column}" for column in global_features})
    leverage = _normalize_dates(
        pd.read_parquet(LEVERAGE_FILE, columns=["date", *leverage_features])
    ).rename(columns={column: f"leverage__{column}" for column in leverage_features})

    global_module.DEVELOPMENT_CUTOFF = DISCOVERY_END
    dividends = global_module._load_dividends()
    market = _normalize_dates(global_module._load_market(dividends))
    frame = market.merge(all_a, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_frame, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(leverage, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DISCOVERY_END].sort_values("date").reset_index(drop=True)
    frame["target1_end_date"] = frame["date"].shift(-2)
    cash_factor = 1.0 + global_module.CASH_ANNUAL_RATE / global_module.TRADING_DAYS_PER_YEAR
    frame["signed_cash_advantage1"] = cash_factor - (1.0 + frame["one_day_open_total_return"])
    frame["avoidable_loss1"] = frame["signed_cash_advantage1"].clip(lower=0.0)

    known_target = frame["signed_cash_advantage1"].shift(2)
    history_columns: list[str] = []
    for window, minimum in [(5, 3), (20, 10), (60, 30), (252, 126)]:
        mean_column = f"known_cash_adv_mean{window}"
        std_column = f"known_cash_adv_std{window}"
        positive_column = f"known_cash_adv_positive_share{window}"
        frame[mean_column] = known_target.rolling(window, min_periods=minimum).mean()
        frame[std_column] = known_target.rolling(window, min_periods=minimum).std(ddof=0)
        frame[positive_column] = known_target.gt(0.0).rolling(window, min_periods=minimum).mean()
        history_columns.extend([mean_column, std_column, positive_column])
    frame["known_cash_adv_lag2"] = known_target
    history_columns.append("known_cash_adv_lag2")

    groups = {
        "ALL_DOMAINS": [
            *[f"alla__{column}" for column in all_a_features],
            *[f"global__{column}" for column in global_features],
            *[f"leverage__{column}" for column in leverage_features],
            *history_columns,
        ],
        "ALL_A_AND_HISTORY": [
            *[f"alla__{column}" for column in all_a_features],
            *history_columns,
        ],
    }
    all_features = sorted({column for columns in groups.values() for column in columns})
    frame[all_features] = frame[all_features].replace([np.inf, -np.inf], np.nan)
    discovery = frame["date"].between(DISCOVERY_START, DISCOVERY_END)
    if int(discovery.sum()) < 1200:
        raise ValueError("2019-2023发现期覆盖不足")
    audit = {
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "discovery_rows": int(discovery.sum()),
        "feature_counts": {name: len(columns) for name, columns in groups.items()},
        "latest_target_end_allowed_for_each_retrain": "不晚于该季度首个信号日",
        "post_2023_rows_retained_or_used": 0,
    }
    return frame, groups, dividends, audit


def _linear_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _tree_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", model),
        ]
    )


def _make_model(kind: str) -> Pipeline:
    if kind == "RIDGE":
        return _linear_pipeline(Ridge(alpha=100.0))
    if kind == "ELASTIC":
        return _linear_pipeline(
            ElasticNet(alpha=0.001, l1_ratio=0.20, max_iter=10000, random_state=RANDOM_STATE)
        )
    if kind == "HGB_REG":
        return _tree_pipeline(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.035,
                max_iter=160,
                max_leaf_nodes=7,
                min_samples_leaf=25,
                l2_regularization=10.0,
                random_state=RANDOM_STATE,
            )
        )
    if kind == "RF_REG":
        return _tree_pipeline(
            RandomForestRegressor(
                n_estimators=240,
                max_depth=5,
                min_samples_leaf=15,
                max_features=0.50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        )
    if kind == "EXTRA_REG":
        return _tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=240,
                max_depth=6,
                min_samples_leaf=12,
                max_features=0.65,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        )
    if kind == "GBR_Q90":
        return _tree_pipeline(
            GradientBoostingRegressor(
                loss="quantile",
                alpha=0.90,
                learning_rate=0.035,
                n_estimators=180,
                max_depth=2,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )
        )
    if kind in {"HGB_POSITIVE", "HGB_SEVERE20"}:
        return _tree_pipeline(
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.035,
                max_iter=160,
                max_leaf_nodes=7,
                min_samples_leaf=25,
                l2_regularization=10.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        )
    raise ValueError(f"未知模型类型：{kind}")


def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("RIDGE_EXP_ALL", "ALL_DOMAINS", None, "RIDGE"),
        ModelSpec("RIDGE_ROLL756_ALL", "ALL_DOMAINS", 756, "RIDGE"),
        ModelSpec("ELASTIC_ROLL756_ALL", "ALL_DOMAINS", 756, "ELASTIC"),
        ModelSpec("HGB_EXP_ALL", "ALL_DOMAINS", None, "HGB_REG"),
        ModelSpec("HGB_ROLL756_ALL", "ALL_DOMAINS", 756, "HGB_REG"),
        ModelSpec("RF_ROLL756_ALL", "ALL_DOMAINS", 756, "RF_REG"),
        ModelSpec("EXTRA_ROLL756_ALL", "ALL_DOMAINS", 756, "EXTRA_REG"),
        ModelSpec("GBR_Q90_ROLL756_ALL", "ALL_DOMAINS", 756, "GBR_Q90"),
        ModelSpec("HGB_POSITIVE_ROLL756_ALL", "ALL_DOMAINS", 756, "HGB_POSITIVE"),
        ModelSpec("HGB_SEVERE20_ROLL756_ALL", "ALL_DOMAINS", 756, "HGB_SEVERE20"),
        ModelSpec("RIDGE_ROLL756_ALLA", "ALL_A_AND_HISTORY", 756, "RIDGE"),
        ModelSpec("HGB_ROLL756_ALLA", "ALL_A_AND_HISTORY", 756, "HGB_REG"),
    ]


def _fit_prequential_scores(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    output = frame.copy()
    specs = _model_specs()
    raw_columns = [f"score_raw_{spec.name.lower()}" for spec in specs]
    for column in raw_columns:
        output[column] = np.nan
    prediction_mask = output["date"].between(PREDICTION_START, DISCOVERY_END)
    periods = output.loc[prediction_mask, "date"].dt.to_period("Q")
    fit_records: list[dict[str, Any]] = []
    for period in periods.drop_duplicates().sort_values():
        block_mask = prediction_mask & output["date"].dt.to_period("Q").eq(period)
        block_indices = output.index[block_mask]
        first_signal_date = pd.Timestamp(output.loc[block_indices, "date"].min())
        train_mask = (
            output["date"].ge(FIT_START)
            & output["target1_end_date"].le(first_signal_date)
            & output["signed_cash_advantage1"].notna()
        )
        available_indices = output.index[train_mask]
        if len(available_indices) < 300:
            continue
        for spec, raw_column in zip(specs, raw_columns, strict=True):
            train_indices = available_indices
            if spec.training_window_rows is not None:
                train_indices = available_indices[-spec.training_window_rows :]
            columns = feature_groups[spec.feature_group]
            train = output.loc[train_indices]
            model = _make_model(spec.model_kind)
            if spec.model_kind == "HGB_POSITIVE":
                target = train["signed_cash_advantage1"].gt(0.0).astype(int)
                model.fit(train[columns], target)
                prediction = model.predict_proba(output.loc[block_indices, columns])[:, 1]
                threshold = 0.0
            elif spec.model_kind == "HGB_SEVERE20":
                threshold = float(train["avoidable_loss1"].quantile(0.80))
                target = train["avoidable_loss1"].ge(threshold).astype(int)
                model.fit(train[columns], target)
                prediction = model.predict_proba(output.loc[block_indices, columns])[:, 1]
            else:
                threshold = np.nan
                target = train["signed_cash_advantage1"]
                model.fit(train[columns], target)
                prediction = model.predict(output.loc[block_indices, columns])
            output.loc[block_indices, raw_column] = np.asarray(prediction, dtype=float)
            fit_records.append(
                {
                    "period": str(period),
                    "model": spec.name,
                    "first_signal_date": first_signal_date.date().isoformat(),
                    "train_rows": int(len(train)),
                    "train_first_date": train["date"].min().date().isoformat(),
                    "train_last_signal_date": train["date"].max().date().isoformat(),
                    "train_last_target_end_date": train["target1_end_date"].max().date().isoformat(),
                    "severe_threshold": _safe_float(threshold),
                }
            )
    return output, raw_columns, fit_records


def _add_calibrated_and_ensemble_scores(
    frame: pd.DataFrame,
    raw_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    risk_columns: list[str] = []
    for raw_column in raw_columns:
        risk_column = raw_column.replace("score_raw_", "risk_rolling252_")
        output[risk_column] = _rolling_last_percentile(output[raw_column])
        risk_columns.append(risk_column)
    regression_risks = [
        column
        for column in risk_columns
        if "positive" not in column and "severe20" not in column
    ]
    output["risk_ensemble_regression_mean"] = output[regression_risks].mean(axis=1)
    output["risk_ensemble_all_mean"] = output[risk_columns].mean(axis=1)
    output["risk_ensemble_all_max"] = output[risk_columns].max(axis=1)
    for threshold in [0.90, 0.95]:
        tail = int(round((1.0 - threshold) * 100))
        output[f"risk_vote_tail{tail:02d}"] = (
            output[risk_columns].ge(threshold).sum(axis=1) / len(risk_columns)
        )
    score_columns = [
        *risk_columns,
        "risk_ensemble_regression_mean",
        "risk_ensemble_all_mean",
        "risk_ensemble_all_max",
        "risk_vote_tail10",
        "risk_vote_tail05",
    ]
    return output, score_columns


def _gross_frontier(frame: pd.DataFrame, score_columns: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    discovery = frame["date"].between(DISCOVERY_START, DISCOVERY_END)
    policies: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        for coverage in TAIL_COVERAGES:
            signal = frame[score_column].ge(1.0 - coverage).fillna(False)
            selected = frame.loc[discovery & signal, ["date", "signed_cash_advantage1"]].dropna()
            if len(selected) < 8:
                continue
            policy = f"{score_column.upper()}_TOP{int(round(coverage * 100)):02d}"
            policies[policy] = signal
            year_sum = (
                selected.assign(year=selected["date"].dt.year)
                .groupby("year")["signed_cash_advantage1"]
                .sum()
                .reindex(range(2019, 2024), fill_value=0.0)
            )
            first_half = selected.loc[
                selected["date"] <= pd.Timestamp("2021-12-31"), "signed_cash_advantage1"
            ].sum()
            second_half = selected.loc[
                selected["date"] >= pd.Timestamp("2022-01-01"), "signed_cash_advantage1"
            ].sum()
            rows.append(
                {
                    "policy": policy,
                    "score_column": score_column,
                    "tail_coverage": coverage,
                    "cash_signal_days": int(len(selected)),
                    "signed_cash_advantage_sum": float(selected["signed_cash_advantage1"].sum()),
                    "signed_cash_advantage_mean": float(selected["signed_cash_advantage1"].mean()),
                    "positive_cash_advantage_share": float(
                        selected["signed_cash_advantage1"].gt(0.0).mean()
                    ),
                    "year_minimum_cash_advantage_sum": float(year_sum.min()),
                    "positive_year_count": int(year_sum.gt(0.0).sum()),
                    "first_half_cash_advantage_sum": float(first_half),
                    "second_half_cash_advantage_sum": float(second_half),
                    "stable_all_years_and_halves": bool(
                        year_sum.gt(0.0).all() and first_half > 0.0 and second_half > 0.0
                    ),
                }
            )
    frontier = pd.DataFrame(rows).sort_values(
        ["stable_all_years_and_halves", "positive_year_count", "signed_cash_advantage_sum"],
        ascending=[False, False, False],
    )
    return frontier.reset_index(drop=True), policies


def _exact_evaluate_policy(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy: str,
    cash_signal: pd.Series,
    start_perturbation: int,
) -> dict[str, Any]:
    discovery_indices = np.flatnonzero(
        frame["date"].between(DISCOVERY_START, DISCOVERY_END).to_numpy()
    )
    first_execution_index = int(discovery_indices[start_perturbation])
    last_execution_index = int(discovery_indices[-1])
    anchor_index = first_execution_index - 1
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    values = cash_signal.to_numpy(dtype=bool)
    states = np.zeros(len(sample), dtype=np.int8)
    cash_days = 0
    for execution_index in range(first_execution_index, last_execution_index + 1):
        state = 0 if values[execution_index - 1] else 1
        states[execution_index - anchor_index] = state
        cash_days += int(state == 0)
    benchmark = build_benchmark_ledger(sample, global_module.INITIAL_CAPITAL_CNY)
    base_ledger, base_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=global_module.BASE_COSTS,
        initial_capital=global_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    stress_ledger, stress_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=global_module.STRESS_COSTS,
        initial_capital=global_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    base = summarize_path(base_ledger, base_trades, benchmark, objective=global_module.OBJECTIVE)
    stress = summarize_path(stress_ledger, stress_trades, benchmark, objective=global_module.OBJECTIVE)
    row: dict[str, Any] = {
        "policy": policy,
        "start_perturbation": start_perturbation,
        "execution_days": int(last_execution_index - first_execution_index + 1),
        "cash_days": cash_days,
        "cash_day_share": float(cash_days / (last_execution_index - first_execution_index + 1)),
    }
    for prefix, summary in [("base", base), ("stress", stress)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    return row


def _select_exact_policies(frontier: pd.DataFrame) -> list[str]:
    stable = frontier[frontier["stable_all_years_and_halves"]]
    selected = list(stable["policy"])
    for _, group in frontier.groupby("score_column", sort=False):
        selected.extend(group.head(2)["policy"].tolist())
    selected.extend(frontier.head(MAX_EXACT_POLICIES)["policy"].tolist())
    return list(dict.fromkeys(selected))[:MAX_EXACT_POLICIES]


def _aggregate_exact(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in metrics.groupby("policy", sort=False):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        gates = group["stress_both_20pct_gates"].astype(bool)
        rows.append(
            {
                "policy": policy,
                "stress_annualized_excess_minimum": _safe_float(annual.min()),
                "stress_annualized_excess_median": _safe_float(annual.median()),
                "stress_rolling_242d_excess_median_minimum": _safe_float(rolling.min()),
                "stress_rolling_242d_excess_median_across_starts": _safe_float(rolling.median()),
                "all_five_start_perturbations_passed": bool(
                    len(group) == len(START_PERTURBATIONS) and gates.all()
                ),
                "cash_day_share_median": _safe_float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": _safe_float(
                    group["stress_trade_leg_count"].median()
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            bool(row["all_five_start_perturbations_passed"]),
            -999.0
            if row["stress_annualized_excess_minimum"] is None
            else float(row["stress_annualized_excess_minimum"]),
            -999.0
            if row["stress_rolling_242d_excess_median_minimum"] is None
            else float(row["stress_rolling_242d_excess_median_minimum"]),
        ),
        reverse=True,
    )


def main() -> int:
    required = [ALL_A_FILE, GLOBAL_FILE, LEVERAGE_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    frame, feature_groups, dividends, frame_audit = _load_frame()
    scored, raw_columns, fit_records = _fit_prequential_scores(frame, feature_groups)
    scored, score_columns = _add_calibrated_and_ensemble_scores(scored, raw_columns)
    frontier, policies = _gross_frontier(scored, score_columns)
    exact_policies = _select_exact_policies(frontier)
    metric_rows: list[dict[str, Any]] = []
    for policy in exact_policies:
        for perturbation in START_PERTURBATIONS:
            metric_rows.append(
                _exact_evaluate_policy(
                    scored,
                    dividends,
                    policy,
                    policies[policy],
                    perturbation,
                )
            )
    metrics = pd.DataFrame(metric_rows)
    aggregates = _aggregate_exact(metrics)
    best = aggregates[0] if aggregates else None
    passed = bool(best and best["all_five_start_perturbations_passed"])

    output_columns = [
        "date",
        "target1_end_date",
        "one_day_open_total_return",
        "signed_cash_advantage1",
        "avoidable_loss1",
        *raw_columns,
        *score_columns,
    ]
    _atomic_parquet(scored[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(frontier, OUTPUT_FRONTIER)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_POST_2023_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": PROJECT_ID,
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_cutoff": "T_CLOSE",
            "execution": "T_PLUS_1_OPEN",
            "initial_capital_cny": global_module.INITIAL_CAPITAL_CNY,
        },
        "period_contract": {
            "fit_history_start": FIT_START.date().isoformat(),
            "prequential_prediction_start": PREDICTION_START.date().isoformat(),
            "candidate_discovery_start": DISCOVERY_START.date().isoformat(),
            "candidate_discovery_end": DISCOVERY_END.date().isoformat(),
            "post_2023_values_used_in_features_model_selection_or_performance": False,
            "retraining_frequency": "CALENDAR_QUARTER",
            "target_availability_rule": "target1_end_date_not_after_quarter_first_signal_date",
        },
        "frame_audit": frame_audit,
        "model_specs": [spec.__dict__ for spec in _model_specs()],
        "fit_record_count": len(fit_records),
        "fit_records": fit_records,
        "gross_frontier_count": int(len(frontier)),
        "exact_policy_count": int(len(exact_policies)),
        "best_exact_policy": best,
        "top_exact_policies": aggregates[:20],
        "gate": {
            "stress_annualized_net_excess_minimum_each_start": 0.20,
            "stress_rolling_242d_net_excess_median_minimum_each_start": 0.20,
            "all_five_starts_required": True,
            "passed": passed,
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in required
        },
        "artifacts": {
            "features": {
                "file": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_FEATURES),
            },
            "gross_frontier": {
                "file": str(OUTPUT_FRONTIER.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_FRONTIER),
            },
            "exact_metrics": {
                "file": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_METRICS),
            },
        },
        "next_action": (
            "冻结最佳候选后才允许读取2024-2025"
            if passed
            else "拒绝本发现族并继续搜索新机制"
        ),
        "authorization": {
            "is_trading_signal": False,
            "paper_or_shadow_eligible": False,
            "broker_connection_authorized": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gross_frontier_count": payload["gross_frontier_count"],
                "exact_policy_count": payload["exact_policy_count"],
                "best_exact_policy": best,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
