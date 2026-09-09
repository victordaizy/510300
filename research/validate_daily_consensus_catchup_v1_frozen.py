"""对冻结的510300逐日共识加补跌候选执行2021-2025独立验证。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import all_a_fragility_shape_5d_discovery_v0 as all_a_module
import direct_one_day_cash_advantage_rank_discovery_v0 as direct_module
import global_liquidity_regime_5d_discovery_v0 as global_module
import market_leverage_cascade_5d_discovery_v0 as leverage_module
from binary_state_feasibility_v1 import build_benchmark_ledger, simulate_binary_path, summarize_path
from multi_domain_severity_rank_5d_discovery_v0 import _rolling_last_percentile


PROTOCOL_ID = "510300_DAILY_CONSENSUS_CATCHUP_V1"
EXPECTED_CONFIG_SHA256 = "353456161ec255b7e947a864c098d91e9c7f4adc275ee1228887b82618bd327a"
VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2025-12-31")
DEVELOPMENT_END = pd.Timestamp("2020-12-31")
START_PERTURBATIONS = list(range(5))
REPRODUCTION_TOLERANCE = 1e-12

CONFIG_FILE = PROJECT_ROOT / "config" / "510300_daily_consensus_catchup_v1.yaml"
FREEZE_RECEIPT = (
    PROJECT_ROOT / "reports" / "frozen" / "510300_daily_consensus_catchup_v1_freeze_receipt.json"
)
INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "510300_daily_consensus_catchup_v1"
    / "input_build_audit.json"
)
IMPLEMENTATION_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_daily_consensus_catchup_v1_validation_implementation_receipt.json"
)

FROZEN_ALL_A = (
    PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
)
FROZEN_GLOBAL = (
    PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
)
FROZEN_LEVERAGE = (
    PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
)
FROZEN_DIRECT_SCORES = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_one_day_cash_advantage_rank_discovery_v0.parquet"
)
FROZEN_GLOBAL_SCORES = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_global_contagion_state_machine_discovery_v0.parquet"
)
FROZEN_CANDIDATE_SIGNALS = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_consensus_residual_catchup_discovery_v0.parquet"
)

BUILT_ALL_A = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "510300_daily_consensus_catchup_v1"
    / "all_a_features_2016_2025.parquet"
)
BUILT_GLOBAL = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "510300_daily_consensus_catchup_v1"
    / "global_features_2015_2025.parquet"
)
BUILT_LEVERAGE = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "510300_daily_consensus_catchup_v1"
    / "leverage_features_2015_2025.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "510300_daily_consensus_catchup_v1"
OUTPUT_SCORES = OUTPUT_DIR / "frozen_scores_2016_2025.parquet"
OUTPUT_SIGNALS = OUTPUT_DIR / "frozen_signals_2016_2025.parquet"
OUTPUT_METRICS = OUTPUT_DIR / "validation_start_perturbation_metrics.parquet"
OUTPUT_STATES = OUTPUT_DIR / "validation_daily_states.parquet"
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "validation" / "510300_daily_consensus_catchup_v1_validation.json"
)

DIRECT_REPRODUCTION_COLUMNS = [
    "score_raw_rf_signed_all",
    "score_raw_hgb_signed_all",
    "score_raw_gbr_q90_loss_all",
    "score_raw_elastic_signed_all",
    "risk_rolling252_rf_signed_all",
    "risk_rolling252_hgb_signed_all",
    "risk_rolling252_gbr_q90_loss_all",
    "risk_rolling252_elastic_signed_all",
    "risk_rolling252_global_selloff1_baseline",
]
GLOBAL_REPRODUCTION_COLUMNS = [
    "risk_rolling252_vix_level",
    "risk_rolling252_contagion_joint",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    global_module._atomic_json(payload, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    global_module._atomic_parquet(frame, path)


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _safe_float(value: Any) -> float | None:
    return global_module._safe_float(value)


def _verify_frozen_contract() -> dict[str, Any]:
    if _sha256(CONFIG_FILE) != EXPECTED_CONFIG_SHA256:
        raise ValueError("冻结配置SHA-256漂移")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    receipt = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))
    audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("冻结协议标识不匹配")
    if config["independent_validation"]["status"] != "NOT_READ_NOT_RUN":
        raise ValueError("冻结配置的验证初始状态漂移")
    if receipt["status"] != "FROZEN_BEFORE_INDEPENDENT_VALIDATION":
        raise ValueError("冻结回执状态无效")
    if receipt["config"]["sha256"] != EXPECTED_CONFIG_SHA256:
        raise ValueError("冻结回执中的配置哈希不匹配")
    if audit["status"] != "PASS_VALIDATION_INPUTS_BUILT_DEVELOPMENT_FEATURES_REPRODUCED":
        raise ValueError("验证输入构造审计未通过")
    if audit["protocol_id"] != PROTOCOL_ID:
        raise ValueError("验证输入构造审计的协议标识不匹配")
    if audit["validation_period"] != ["2021-01-01", "2025-12-31"]:
        raise ValueError("验证输入构造审计的样本期间漂移")
    if audit["contains_performance_or_validation_signal_results"] is not False:
        raise ValueError("验证输入构造阶段不应包含信号或绩效结果")
    for path_text, expected_hash in config["development_input_sha256"].items():
        path = PROJECT_ROOT / Path(path_text)
        if _sha256(path) != expected_hash:
            raise ValueError(f"开发输入哈希漂移：{path_text}")
    for path_text, expected_hash in config["implementation_sha256"].items():
        path = PROJECT_ROOT / Path(path_text)
        if _sha256(path) != expected_hash:
            raise ValueError(f"冻结发现实现哈希漂移：{path_text}")
    for artifact in audit["artifacts"].values():
        path = PROJECT_ROOT / Path(artifact["file"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"验证特征产物哈希漂移：{artifact['file']}")
    return {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "freeze_receipt_sha256": _sha256(FREEZE_RECEIPT),
        "input_build_audit_sha256": _sha256(INPUT_AUDIT),
        "frozen_at": receipt["frozen_at"],
        "validation_status_at_freeze": receipt["independent_validation"]["status_at_freeze"],
        "validation_input_coverage": audit["coverage"],
        "validation_input_alignment": audit["alignment"],
    }


def _verify_implementation_receipt() -> dict[str, Any]:
    receipt = json.loads(IMPLEMENTATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt["status"] != "VALIDATION_IMPLEMENTATION_FROZEN_BEFORE_PERFORMANCE_RUN":
        raise ValueError("验证实现封存回执状态无效")
    if receipt["protocol_id"] != PROTOCOL_ID:
        raise ValueError("验证实现封存回执的协议标识不匹配")
    if receipt["performance_run_status_at_receipt"] != "NOT_RUN":
        raise ValueError("验证实现封存时的绩效运行状态不是NOT_RUN")
    evaluator = receipt["evaluator"]
    evaluator_path = PROJECT_ROOT / Path(evaluator["file"])
    if evaluator_path.resolve() != Path(__file__).resolve():
        raise ValueError("验证实现封存回执指向了其他验证器")
    if _sha256(evaluator_path) != evaluator["sha256"]:
        raise ValueError("验证器在封存后发生漂移")
    for manifest_name in ["implementation_sha256", "input_sha256"]:
        for path_text, expected_hash in receipt[manifest_name].items():
            path = PROJECT_ROOT / Path(path_text)
            if _sha256(path) != expected_hash:
                raise ValueError(f"验证封存清单哈希漂移：{path_text}")
    return {
        "receipt_sha256": _sha256(IMPLEMENTATION_RECEIPT),
        "frozen_at": receipt["frozen_at"],
        "evaluator_sha256": evaluator["sha256"],
        "performance_run_status_at_receipt": receipt["performance_run_status_at_receipt"],
    }


def _combine_domain(
    frozen_path: Path,
    built_path: Path,
    columns: list[str],
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    extras = extra_columns or []
    frozen = _normalize_dates(pd.read_parquet(frozen_path, columns=["date", *columns, *extras]))
    built = _normalize_dates(pd.read_parquet(built_path, columns=["date", *columns, *extras]))
    frozen = frozen[frozen["date"] <= DEVELOPMENT_END]
    built = built[built["date"] > DEVELOPMENT_END]
    combined = pd.concat([frozen, built], ignore_index=True)
    return combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _load_full_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    all_a_columns = list(all_a_module.MODEL_FEATURES)
    global_columns = list(global_module.FEATURE_COLUMNS)
    leverage_columns = list(leverage_module.FEATURE_COLUMNS)
    all_a = _combine_domain(
        FROZEN_ALL_A,
        BUILT_ALL_A,
        all_a_columns,
        extra_columns=["one_day_open_total_return"],
    )
    global_features = _combine_domain(FROZEN_GLOBAL, BUILT_GLOBAL, global_columns)
    leverage = _combine_domain(FROZEN_LEVERAGE, BUILT_LEVERAGE, leverage_columns)
    all_a = all_a.rename(columns={column: f"alla__{column}" for column in all_a_columns})
    global_features = global_features.rename(
        columns={column: f"global__{column}" for column in global_columns}
    )
    global_features = _compute_global_state_scores(global_features)
    leverage = leverage.rename(columns={column: f"leverage__{column}" for column in leverage_columns})

    global_module.DEVELOPMENT_CUTOFF = VALIDATION_END
    dividends = global_module._load_dividends()
    market = _normalize_dates(global_module._load_market(dividends))
    frame = market.merge(all_a, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_features, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(leverage, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= VALIDATION_END].sort_values("date").reset_index(drop=True)
    frame["target1_end_date"] = frame["date"].shift(-2)
    cash_factor1 = 1.0 + global_module.CASH_ANNUAL_RATE / global_module.TRADING_DAYS_PER_YEAR
    frame["signed_cash_advantage1"] = cash_factor1 - (1.0 + frame["one_day_open_total_return"])
    frame["avoidable_loss1"] = frame["signed_cash_advantage1"].clip(lower=0.0)
    frame["avoidable_loss5"] = (frame["future5_cash_factor"] - frame["future5_full_factor"]).clip(lower=0.0)
    feature_groups = {
        "ALL_DOMAINS": [
            *[f"alla__{column}" for column in all_a_columns],
            *[f"global__{column}" for column in global_columns],
            *[f"leverage__{column}" for column in leverage_columns],
        ],
        "GLOBAL_ONLY": [f"global__{column}" for column in global_columns],
        "ALL_A_ONLY": [f"alla__{column}" for column in all_a_columns],
        "GLOBAL_AND_ALL_A": [
            *[f"global__{column}" for column in global_columns],
            *[f"alla__{column}" for column in all_a_columns],
        ],
    }
    all_feature_columns = sorted(
        {column for columns in feature_groups.values() for column in columns}
    )
    frame[all_feature_columns] = frame[all_feature_columns].replace([np.inf, -np.inf], np.nan)
    return frame, feature_groups, dividends


def _compare_numeric(
    built: pd.DataFrame,
    frozen_path: Path,
    columns: list[str],
    label: str,
) -> dict[str, Any]:
    frozen = _normalize_dates(pd.read_parquet(frozen_path, columns=["date", *columns]))
    candidate = built[built["date"] <= DEVELOPMENT_END][["date", *columns]].copy()
    if candidate.empty:
        raise ValueError(f"{label}缺少开发期复现范围")
    frozen = frozen[
        frozen["date"].between(candidate["date"].min(), candidate["date"].max())
    ]
    merged = frozen.merge(candidate, on="date", how="outer", suffixes=("_frozen", "_built"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label}日期覆盖不一致")
    maxima: dict[str, float] = {}
    nan_mismatch = 0
    for column in columns:
        left = pd.to_numeric(merged[f"{column}_frozen"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_built"], errors="coerce")
        nan_mismatch += int(left.isna().ne(right.isna()).sum())
        valid = left.notna() & right.notna()
        maxima[column] = float((left[valid] - right[valid]).abs().max()) if valid.any() else 0.0
    maximum = max(maxima.values(), default=0.0)
    if maximum > REPRODUCTION_TOLERANCE or nan_mismatch:
        raise ValueError(f"{label}复现失败：maximum={maximum}, nan_mismatch={nan_mismatch}")
    return {
        "label": label,
        "status": "PASS",
        "matched_dates": int(len(merged)),
        "maximum_absolute_difference": maximum,
        "tolerance": REPRODUCTION_TOLERANCE,
        "nan_pattern_mismatch_count": nan_mismatch,
        "maximum_by_column": maxima,
    }


def _fit_frozen_models(frame: pd.DataFrame, feature_groups: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored, specs, model_summary = direct_module._fit_scores(frame, feature_groups)
    scored, _, score_audit = direct_module._calibrate_scores(scored, specs)
    reproduction = _compare_numeric(
        scored,
        FROZEN_DIRECT_SCORES,
        DIRECT_REPRODUCTION_COLUMNS,
        "DIRECT_ONE_DAY_MODEL_SCORES",
    )
    return scored, {"model_summary": model_summary, "score_audit": score_audit, "reproduction": reproduction}


def _compute_global_state_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["vix_level"] = np.exp(pd.to_numeric(output["global__vix_log_level"], errors="coerce"))
    output["raw_vix_shock"] = np.maximum(
        pd.to_numeric(output["global__vix_ret1"], errors="coerce"),
        pd.to_numeric(output["global__vix_ret5"], errors="coerce") / np.sqrt(5.0),
    )
    output["raw_global_selloff1"] = -pd.concat(
        [
            output["global__us_mean_ret1"],
            output["global__eu_mean_ret1"],
            output["global__asia_mean_ret1"],
            output["global__global_min_ret1"],
        ],
        axis=1,
    ).min(axis=1)
    output["raw_global_selloff5"] = -pd.concat(
        [
            output["global__us_mean_ret5"],
            output["global__eu_mean_ret5"],
            output["global__asia_mean_ret5"],
        ],
        axis=1,
    ).mean(axis=1)
    raw_map = {
        "vix_level": "vix_level",
        "vix_shock": "raw_vix_shock",
        "global_selloff1": "raw_global_selloff1",
        "global_selloff5": "raw_global_selloff5",
    }
    for name, raw_column in raw_map.items():
        output[f"risk_rolling252_{name}"] = _rolling_last_percentile(output[raw_column])
    output["risk_pre_contagion_joint"] = np.minimum(
        output[["risk_rolling252_vix_level", "risk_rolling252_vix_shock"]].max(axis=1),
        output[["risk_rolling252_global_selloff1", "risk_rolling252_global_selloff5"]].max(axis=1),
    )
    output["risk_rolling252_contagion_joint"] = _rolling_last_percentile(
        output["risk_pre_contagion_joint"]
    )
    return output


def _add_global_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    reproduction = _compare_numeric(
        output,
        FROZEN_GLOBAL_SCORES,
        GLOBAL_REPRODUCTION_COLUMNS,
        "GLOBAL_STATE_SCORES",
    )
    return output, reproduction


def _add_frozen_signal(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    vote_columns = {
        "member_rf_signed_top03": output["risk_rolling252_rf_signed_all"] >= 0.97,
        "member_hgb_signed_top04": output["risk_rolling252_hgb_signed_all"] >= 0.96,
        "member_gbr_q90_loss_top04": output["risk_rolling252_gbr_q90_loss_all"] >= 0.96,
        "member_elastic_signed_top01": output["risk_rolling252_elastic_signed_all"] >= 0.99,
        "member_vix_level_top05": output["risk_rolling252_vix_level"] >= 0.95,
        "member_contagion_joint_top05": output["risk_rolling252_contagion_joint"] >= 0.95,
    }
    for column, values in vote_columns.items():
        output[column] = values.fillna(False).astype(np.int8)
    member_columns = list(vote_columns)
    output["base_consensus_votes"] = output[member_columns].sum(axis=1)
    output["base_consensus_trigger"] = output["base_consensus_votes"] >= 2
    output["residual_catchup_trigger"] = (
        (output["risk_rolling252_global_selloff1_baseline"] >= 0.90)
        & (output["alla__etf_return_20d"] >= 0.0)
    )
    output["frozen_cash_signal"] = (
        output["base_consensus_trigger"] | output["residual_catchup_trigger"]
    ).astype(np.int8)
    frozen = _normalize_dates(
        pd.read_parquet(FROZEN_CANDIDATE_SIGNALS, columns=["date", "signal_catchup_021"])
    )
    frozen = frozen[frozen["date"] <= DEVELOPMENT_END]
    candidate = output[output["date"].isin(frozen["date"])][["date", "frozen_cash_signal"]]
    merged = frozen.merge(candidate, on="date", how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("冻结信号开发日期覆盖不一致")
    mismatch = int(
        pd.to_numeric(merged["signal_catchup_021"], errors="raise")
        .astype(np.int8)
        .ne(merged["frozen_cash_signal"].astype(np.int8))
        .sum()
    )
    if mismatch:
        raise ValueError(f"冻结开发信号复现失败：{mismatch}行不一致")
    audit = {
        "label": "FROZEN_CANDIDATE_SIGNAL",
        "status": "PASS",
        "matched_dates": int(len(merged)),
        "mismatch_count": mismatch,
    }
    return output, audit


def _states_and_sample(
    frame: pd.DataFrame,
    start_perturbation: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    validation_indices = np.flatnonzero(
        frame["date"].between(VALIDATION_START, VALIDATION_END).to_numpy()
    )
    if len(validation_indices) != 1212:
        raise ValueError(f"验证交易日应为1212，实际为{len(validation_indices)}")
    first_execution_index = int(validation_indices[start_perturbation])
    last_execution_index = int(validation_indices[-1])
    anchor_index = first_execution_index - 1
    if anchor_index < 0:
        raise ValueError("验证样本缺少信号锚点")
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states = np.zeros(len(sample), dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for execution_index in range(first_execution_index, last_execution_index + 1):
        signal_index = execution_index - 1
        cash_signal = int(frame.loc[signal_index, "frozen_cash_signal"])
        state = 0 if cash_signal == 1 else 1
        sample_index = execution_index - anchor_index
        states[sample_index] = state
        rows.append(
            {
                "start_perturbation": start_perturbation,
                "signal_date": pd.Timestamp(frame.loc[signal_index, "date"]),
                "execution_date": pd.Timestamp(frame.loc[execution_index, "date"]),
                "state": state,
                "cash_signal": cash_signal,
                "base_consensus_votes": int(frame.loc[signal_index, "base_consensus_votes"]),
                "base_consensus_trigger": bool(frame.loc[signal_index, "base_consensus_trigger"]),
                "residual_catchup_trigger": bool(frame.loc[signal_index, "residual_catchup_trigger"]),
            }
        )
    return sample, states, pd.DataFrame(rows)


def _calendar_year_metrics(
    strategy_ledger: pd.DataFrame,
    benchmark_ledger: pd.DataFrame,
) -> list[dict[str, Any]]:
    merged = strategy_ledger[["date", "daily_return"]].merge(
        benchmark_ledger[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        validate="one_to_one",
    )
    merged["date"] = pd.to_datetime(merged["date"], errors="raise")
    rows: list[dict[str, Any]] = []
    for year, group in merged.groupby(merged["date"].dt.year, sort=True):
        if year < VALIDATION_START.year or year > VALIDATION_END.year:
            continue
        strategy_return = float((1.0 + group["daily_return_strategy"]).prod() - 1.0)
        benchmark_return = float((1.0 + group["daily_return_benchmark"]).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "observations": int(len(group)),
                "strategy_net_return": strategy_return,
                "benchmark_total_return": benchmark_return,
                "net_excess_return": strategy_return - benchmark_return,
            }
        )
    return rows


def _evaluate_validation(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    state_frames: list[pd.DataFrame] = []
    primary_year_metrics: list[dict[str, Any]] = []
    primary_ledgers: dict[str, Any] = {}
    for perturbation in START_PERTURBATIONS:
        sample, states, state_frame = _states_and_sample(frame, perturbation)
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
        base_summary = summarize_path(base_ledger, base_trades, benchmark, objective=global_module.OBJECTIVE)
        stress_summary = summarize_path(stress_ledger, stress_trades, benchmark, objective=global_module.OBJECTIVE)
        row: dict[str, Any] = {
            "start_perturbation": perturbation,
            "execution_day_count": int(len(state_frame)),
            "cash_day_count": int(state_frame["state"].eq(0).sum()),
            "cash_day_share": float(state_frame["state"].eq(0).mean()),
            "base_consensus_cash_signal_count": int(state_frame["base_consensus_trigger"].sum()),
            "residual_catchup_cash_signal_count": int(state_frame["residual_catchup_trigger"].sum()),
        }
        for prefix, summary in [("base", base_summary), ("stress", stress_summary)]:
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        row["frozen_validation_gate"] = bool(stress_summary["both_20pct_gates"])
        metric_rows.append(row)
        state_frames.append(state_frame)
        if perturbation == 0:
            primary_year_metrics = _calendar_year_metrics(stress_ledger, benchmark)
            primary_ledgers = {
                "stress_final_equity_cny": float(stress_ledger["equity"].iloc[-1]),
                "benchmark_final_equity_cny": float(benchmark["equity"].iloc[-1]),
                "stress_trade_leg_count": int(len(stress_trades)),
            }
    metrics = pd.DataFrame(metric_rows)
    states = pd.concat(state_frames, ignore_index=True)
    annual = pd.to_numeric(metrics["stress_annualized_excess"], errors="coerce")
    rolling = pd.to_numeric(metrics["stress_rolling_242d_excess_median"], errors="coerce")
    aggregate = {
        "stress_annualized_excess_minimum": _safe_float(annual.min()),
        "stress_annualized_excess_median": _safe_float(annual.median()),
        "stress_annualized_excess_maximum": _safe_float(annual.max()),
        "stress_rolling_242d_excess_median_minimum": _safe_float(rolling.min()),
        "stress_rolling_242d_excess_median_across_perturbations": _safe_float(rolling.median()),
        "stress_rolling_242d_excess_median_maximum": _safe_float(rolling.max()),
        "all_five_start_perturbations_passed": bool(
            len(metrics) == len(START_PERTURBATIONS) and metrics["frozen_validation_gate"].all()
        ),
        "start_perturbation_pass_ratio": float(metrics["frozen_validation_gate"].mean()),
        "cash_day_share_median": float(metrics["cash_day_share"].median()),
        "stress_trade_leg_count_median": _safe_float(metrics["stress_trade_leg_count"].median()),
    }
    return metrics, states, primary_year_metrics, {**aggregate, **primary_ledgers}


def main() -> int:
    required = [
        CONFIG_FILE,
        FREEZE_RECEIPT,
        INPUT_AUDIT,
        IMPLEMENTATION_RECEIPT,
        FROZEN_ALL_A,
        FROZEN_GLOBAL,
        FROZEN_LEVERAGE,
        FROZEN_DIRECT_SCORES,
        FROZEN_GLOBAL_SCORES,
        FROZEN_CANDIDATE_SIGNALS,
        BUILT_ALL_A,
        BUILT_GLOBAL,
        BUILT_LEVERAGE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    frozen_contract = _verify_frozen_contract()
    validation_implementation = _verify_implementation_receipt()
    frame, feature_groups, dividends = _load_full_frame()
    validation_dates = frame["date"].between(VALIDATION_START, VALIDATION_END)
    if int(validation_dates.sum()) != 1212:
        raise ValueError(f"验证共同特征日期不足1212：{int(validation_dates.sum())}")
    scored, model_audit = _fit_frozen_models(frame, feature_groups)
    scored, global_score_audit = _add_global_scores(scored)
    scored, signal_audit = _add_frozen_signal(scored)
    metrics, states, year_metrics, aggregate = _evaluate_validation(scored, dividends)
    passed = bool(aggregate["all_five_start_perturbations_passed"])

    score_output_columns = [
        "date",
        "score_raw_rf_signed_all",
        "score_raw_hgb_signed_all",
        "score_raw_gbr_q90_loss_all",
        "score_raw_elastic_signed_all",
        "risk_rolling252_rf_signed_all",
        "risk_rolling252_hgb_signed_all",
        "risk_rolling252_gbr_q90_loss_all",
        "risk_rolling252_elastic_signed_all",
        "risk_rolling252_global_selloff1_baseline",
        "risk_rolling252_vix_level",
        "risk_rolling252_contagion_joint",
    ]
    signal_output_columns = [
        "date",
        "base_consensus_votes",
        "base_consensus_trigger",
        "residual_catchup_trigger",
        "frozen_cash_signal",
        "alla__etf_return_20d",
        "risk_rolling252_global_selloff1_baseline",
        "member_rf_signed_top03",
        "member_hgb_signed_top04",
        "member_gbr_q90_loss_top04",
        "member_elastic_signed_top01",
        "member_vix_level_top05",
        "member_contagion_joint_top05",
    ]
    _atomic_parquet(scored[score_output_columns], OUTPUT_SCORES)
    _atomic_parquet(scored[signal_output_columns], OUTPUT_SIGNALS)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    _atomic_parquet(states, OUTPUT_STATES)

    payload = {
        "status": (
            "VALIDATION_PASS_PAPER_SHADOW_ELIGIBLE"
            if passed
            else "FROZEN_CANDIDATE_REJECTED_CONTINUE_SEARCH"
        ),
        "protocol_id": PROTOCOL_ID,
        "validation_period": [VALIDATION_START.date().isoformat(), VALIDATION_END.date().isoformat()],
        "frozen_contract": frozen_contract,
        "validation_implementation": validation_implementation,
        "validation_results_read_after_freeze": True,
        "candidate_unchanged_from_freeze": True,
        "reproduction_audit": {
            "direct_model_scores": model_audit["reproduction"],
            "global_scores": global_score_audit,
            "candidate_signal": signal_audit,
        },
        "validation_gate": {
            "annualized_net_excess_minimum_each_start": 0.20,
            "rolling_242d_net_excess_median_minimum_each_start": 0.20,
            "required_start_perturbations": START_PERTURBATIONS,
            "passed": passed,
        },
        "validation_aggregate": aggregate,
        "all_start_perturbation_metrics": metrics.to_dict("records"),
        "calendar_year_metrics_primary_start_stress_cost": year_metrics,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "scores": {
                "file": str(OUTPUT_SCORES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_SCORES),
            },
            "signals": {
                "file": str(OUTPUT_SIGNALS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_SIGNALS),
            },
            "metrics": {
                "file": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_METRICS),
            },
            "states": {
                "file": str(OUTPUT_STATES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(OUTPUT_STATES),
            },
        },
        "next_action": (
            "进入Paper或Shadow验证；仍不授权实盘"
            if passed
            else "按冻结协议拒绝V1，不调参救援并继续搜索新机制"
        ),
        "authorization": {
            "is_trading_signal": False,
            "paper_or_shadow_eligible": passed,
            "broker_connection_authorized": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "validation_gate": payload["validation_gate"],
                "validation_aggregate": aggregate,
                "calendar_year_metrics_primary_start_stress_cost": year_metrics,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
