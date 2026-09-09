"""510300压力传导、卖盘吸收与耗竭机制图谱V1。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_STRESS_TRANSMISSION_AND_EXHAUSTION_ATLAS_V1"
CONFIG_PATH = ROOT / "config" / "510300_stress_transmission_and_exhaustion_atlas_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_stress_transmission_and_exhaustion_atlas_v1_manifest.json"
STRESS = "STRESS_DELEVERAGING"


class ContractError(RuntimeError):
    """点时数据、冻结或研究合同不满足。"""


@dataclass(frozen=True)
class DirectionSummary:
    observations: int
    median: float | None
    direction_fraction: float | None
    one_sided_sign_pvalue: float | None
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "median": self.median,
            "direction_fraction": self.direction_fraction,
            "one_sided_sign_pvalue": self.one_sided_sign_pvalue,
            "passed": self.passed,
        }


def project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    if column not in result.columns:
        raise ContractError(f"缺少日期字段：{column}")
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ContractError(f"{column}存在重复值")
    return result


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ContractError(f"{label}缺少字段：{sorted(missing)}")


def _contains_value(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"协议不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["project_id"] != PROJECT_ID:
        raise ContractError("项目编号不匹配")
    if protocol["research_stage"] != "MECHANISM_DISCOVERY_ONLY":
        raise ContractError("研究阶段不是机制发现")
    if protocol["strategy_backtest_allowed"]:
        raise ContractError("机制图谱不得允许策略回测")
    if config["scope"]["portfolio_mapping"] != "disabled":
        raise ContractError("仓位映射必须关闭")
    if config["scope"]["live_trading_authorized"]:
        raise ContractError("不得授权实盘")
    if config["prediction_questions"]["estimation"]["hyperparameter_search"] != "forbidden":
        raise ContractError("逻辑回归不得搜索超参数")
    return config


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("冻结清单不存在，禁止读取新图谱结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("冻结清单项目编号不匹配")
    if manifest.get("frozen_before_new_atlas_outcome_read") is not True:
        raise ContractError("未证明在新图谱结果读取前冻结")
    if manifest.get("new_atlas_outcome_read_before_freeze") is not False:
        raise ContractError("冻结清单显示提前读取过新图谱结果")
    mismatches: dict[str, dict[str, str]] = {}
    for group in ["tracked_files", "input_files"]:
        for relative, expected in manifest.get(group, {}).items():
            path = project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结哈希漂移：{mismatches}")
    return manifest


def causal_percentile(
    series: pd.Series,
    *,
    prior_rows: int,
    minimum_prior_rows: int,
) -> pd.Series:
    """当前观察只与严格在前的有限历史比较。"""

    if minimum_prior_rows < 1 or prior_rows < minimum_prior_rows:
        raise ValueError("因果分位窗口无效")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        reference = values[max(0, index - prior_rows) : index]
        reference = reference[np.isfinite(reference)]
        if len(reference) < minimum_prior_rows:
            continue
        output[index] = np.count_nonzero(reference <= current) / len(reference)
    return pd.Series(output, index=series.index, dtype=float)


def build_total_return_market(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    frame = _normalize_date(market)
    distribution = dividends.copy()
    distribution["ex_date"] = pd.to_datetime(
        distribution["ex_date"], errors="raise"
    ).dt.normalize()
    distribution["cash_dividend_per_share"] = pd.to_numeric(
        distribution["cash_dividend_per_share"], errors="raise"
    )
    ex_cash = (
        distribution.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    frame["cash_dividend_ex_date"] = frame["date"].map(
        lambda date: float(ex_cash.get(date, 0.0))
    )
    frame["total_return_1d_rebuilt"] = (
        (pd.to_numeric(frame["close"]) + frame["cash_dividend_ex_date"])
        / pd.to_numeric(frame["close"]).shift(1)
        - 1.0
    )
    frame["total_return_index_rebuilt"] = (
        1.0 + frame["total_return_1d_rebuilt"].fillna(0.0)
    ).cumprod()
    return frame


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    specifications = config["inputs"]

    def parquet(name: str) -> pd.DataFrame:
        item = specifications[name]
        frame = pd.read_parquet(project_path(item["path"]))
        _require_columns(frame, item["required_columns"], name)
        return frame

    def json_file(name: str) -> dict[str, Any]:
        return json.loads(project_path(specifications[name]["path"]).read_text(encoding="utf-8"))

    router_episodes = pd.read_csv(project_path(specifications["router_episodes"]["path"]))
    _require_columns(
        router_episodes,
        specifications["router_episodes"]["required_columns"],
        "router_episodes",
    )
    dividends = pd.read_csv(project_path(specifications["dividends"]["path"]))
    _require_columns(dividends, specifications["dividends"]["required_columns"], "dividends")
    return {
        "router_result": json_file("router_result"),
        "router_manifest": json_file("router_manifest"),
        "router_state_panel": _normalize_date(parquet("router_state_panel")),
        "router_episodes": router_episodes,
        "router_output_audit": json_file("router_output_audit"),
        "early_breadth": _normalize_date(parquet("early_equal_weight_breadth")),
        "current_breadth": _normalize_date(parquet("current_equal_weight_breadth")),
        "mechanism_atlas_result": json_file("mechanism_atlas_result"),
        "systemic_sell_pressure_result": json_file("systemic_sell_pressure_result"),
        "market": _normalize_date(parquet("etf_market")),
        "dividends": dividends,
        "if_features": _normalize_date(parquet("if_daily_features")),
        "if_audit": json_file("if_acquisition_audit"),
        "broad_etf_audit": json_file("broad_etf_data_quality"),
        "etf_510050": _normalize_date(parquet("etf_510050")),
        "etf_510500": _normalize_date(parquet("etf_510500")),
        "etf_159915": _normalize_date(parquet("etf_159915")),
        "etf_159919": _normalize_date(parquet("etf_159919")),
    }


def audit_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    gates = config["data_gates"]
    state = inputs["router_state_panel"]
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["signal_cutoff"])
    prohibited = [
        column
        for column in state.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    complete_stress = inputs["router_episodes"].loc[
        inputs["router_episodes"]["state"].eq(STRESS)
        & ~inputs["router_episodes"]["left_censored"].astype(bool)
        & ~inputs["router_episodes"]["right_censored"].astype(bool)
    ]
    state_calendar_pass = bool(
        len(state) == int(gates["expected_signal_rows"])
        and state["date"].min() == start
        and state["date"].max() == end
        and state["date"].is_monotonic_increasing
        and not state["date"].duplicated().any()
    )
    required_state_features = [
        "total_return_index",
        "total_return_1d",
        "downside_vol_ratio_5_20",
        "liquidity_stress_score",
        "equal_above_ma60_share",
        "breadth_change5",
        "discount_rate_pressure",
    ]
    state_feature_pass = bool(state[required_state_features].notna().all(axis=None))

    calendar = set(state["date"])
    coverage: dict[str, float] = {}
    etf_identity: dict[str, bool] = {}
    for name in ["etf_510050", "etf_510500", "etf_159915", "etf_159919"]:
        frame = inputs[name]
        expected_code = config["inputs"][name]["required_code"]
        etf_identity[name] = bool(frame["ts_code"].eq(expected_code).all())
        available = set(frame.loc[frame["date"].between(start, end), "date"])
        coverage[name] = len(calendar.intersection(available)) / len(calendar)
    etf_pass = bool(
        all(etf_identity.values())
        and min(coverage.values()) >= float(gates["minimum_broad_etf_calendar_coverage"])
    )
    if_frame = inputs["if_features"].loc[
        inputs["if_features"]["date"].between(start, end)
    ]
    if_coverage = len(calendar.intersection(set(if_frame["date"]))) / len(calendar)
    if_clock = pd.to_datetime(inputs["if_features"]["feature_asof"], errors="raise")
    if_pass = bool(
        if_coverage >= float(gates["minimum_if_calendar_coverage"])
        and inputs["if_audit"].get("status")
        == config["inputs"]["if_acquisition_audit"]["required_status"]
        and (if_clock.dt.normalize() == inputs["if_features"]["date"]).all()
        and inputs["if_features"]["eligible_contract_count"].ge(1).all()
    )
    breadth_pass = bool(
        inputs["early_breadth"]["date"].min() <= pd.Timestamp("2014-07-01")
        and inputs["early_breadth"]["date"].max() == pd.Timestamp("2021-08-11")
        and inputs["current_breadth"]["date"].min() == pd.Timestamp("2021-08-12")
        and inputs["current_breadth"]["date"].max() >= end
        and inputs["current_breadth"]["valid_for_price_breadth"].astype(bool).all()
    )
    predecessor_checks = {
        "router_rejection_preserved": _contains_value(
            inputs["router_result"],
            config["upstream_boundaries"]["rejected_router"]["required_status"],
        ),
        "router_output_recomputation_passed": inputs["router_output_audit"].get("status")
        == config["upstream_boundaries"]["rejected_router"]["required_output_audit_status"],
        "mechanism_atlas_no_strategy_status_preserved": _contains_value(
            inputs["mechanism_atlas_result"],
            config["upstream_boundaries"]["mechanism_atlas"]["required_status"],
        ),
        "systemic_sell_pressure_rejection_preserved": _contains_value(
            inputs["systemic_sell_pressure_result"],
            config["upstream_boundaries"]["systemic_sell_pressure"]["required_status"],
        ),
        "systemic_holdout_remains_unopened": inputs["systemic_sell_pressure_result"].get(
            "replication_panel_read"
        )
        is False,
    }
    checks = {
        "router_state_calendar": state_calendar_pass,
        "router_state_primary_features_complete": state_feature_pass,
        "router_state_has_no_future_columns": not prohibited,
        "exactly_61_complete_router_stress_events": len(complete_stress)
        == int(gates["expected_complete_router_stress_events"]),
        "broad_etf_identity_and_calendar": etf_pass,
        "if_identity_clock_and_calendar": if_pass,
        "breadth_warmup_and_seam": breadth_pass,
        "broad_etf_data_quality": inputs["broad_etf_audit"].get("status")
        == config["inputs"]["broad_etf_data_quality"]["required_status"],
        "predecessor_boundaries": all(predecessor_checks.values()),
        "fundamental_gap_explicit_not_proxied": config["known_data_limits"][
            "point_in_time_fundamental_expectation_2015_onward"
        ]
        == "NO_VIEW_INCOMPLETE_CONTRACT",
    }
    passed = bool(all(checks.values()))
    return {
        "project_id": PROJECT_ID,
        "status": "PASS_STRESS_TRANSMISSION_INPUT_DATA_CONTRACT"
        if passed
        else "FAIL_STRESS_TRANSMISSION_INPUT_DATA_CONTRACT",
        "passed": passed,
        "checks": checks,
        "prohibited_state_columns": prohibited,
        "complete_router_stress_events": int(len(complete_stress)),
        "broad_etf_calendar_coverage": coverage,
        "if_calendar_coverage": if_coverage,
        "predecessor_checks": predecessor_checks,
        "known_no_view": config["known_data_limits"],
        "boundaries": {
            "new_atlas_future_outcomes_read": False,
            "old_router_modified": False,
            "systemic_holdout_opened": False,
            "strategy_returns_computed": False,
            "position_mapping": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def _combine_breadth(inputs: dict[str, Any]) -> pd.DataFrame:
    early = inputs["early_breadth"][["date", "equal_above_ma60_share"]].copy()
    current = inputs["current_breadth"][["date", "equal_above_ma60_share"]].copy()
    return _normalize_date(pd.concat([early, current], ignore_index=True))


def _era_for_date(date: pd.Timestamp, config: dict[str, Any]) -> str:
    for era, bounds in config["dates"]["eras"].items():
        if pd.Timestamp(bounds[0]) <= date <= pd.Timestamp(bounds[1]):
            return era
    return "OUTSIDE_FROZEN_ERAS"


def _source_class(row: pd.Series, config: dict[str, Any]) -> str:
    specification = config["feature_construction"]["pressure_source"]
    active = float(specification["source_active_threshold"])
    margin = float(specification["source_dominance_margin"])
    liquidity = float(row["liquidity_stress_score"])
    discount = float(row["discount_rate_pressure"])
    if liquidity < active and discount < active:
        return "UNRESOLVED"
    if liquidity >= active and discount >= active:
        return "MULTI_SOURCE"
    if liquidity - discount >= margin:
        return "LEVERAGE_LIQUIDITY"
    if discount - liquidity >= margin:
        return "DISCOUNT_RATE"
    return "MIXED_BALANCED"


def build_mechanism_panel(
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> pd.DataFrame:
    percentile = config["feature_construction"]["causal_percentile"]
    prior = int(percentile["prior_rows"])
    minimum = int(percentile["minimum_prior_rows"])
    start = pd.Timestamp("2014-07-01")
    end = pd.Timestamp(config["dates"]["signal_cutoff"])

    market = build_total_return_market(inputs["market"], inputs["dividends"])
    downside = market["total_return_1d_rebuilt"].clip(upper=0.0)
    market["downside_deviation_5_rebuilt"] = np.sqrt(
        downside.pow(2).ewm(span=5, adjust=False, min_periods=5).mean()
    )
    market["downside_deviation_20_rebuilt"] = np.sqrt(
        downside.pow(2).ewm(span=20, adjust=False, min_periods=20).mean()
    )
    market["downside_ratio_rebuilt"] = market["downside_deviation_5_rebuilt"] / market[
        "downside_deviation_20_rebuilt"
    ].replace(0.0, np.nan)
    market["total_return_3d"] = market["total_return_index_rebuilt"].pct_change(3)
    market["prior_abs_total_return_3d_median"] = (
        market["total_return_3d"]
        .abs()
        .shift(1)
        .rolling(prior, min_periods=minimum)
        .median()
    )
    market["price_ma20_rebuilt"] = market["total_return_index_rebuilt"].rolling(
        20, min_periods=20
    ).mean()
    market["downside_prior20_peak_rebuilt"] = market["downside_ratio_rebuilt"].shift(1).rolling(
        20, min_periods=20
    ).max()
    market["downside_relief_fraction"] = (
        (
            market["downside_prior20_peak_rebuilt"]
            - market["downside_ratio_rebuilt"]
        )
        / market["downside_prior20_peak_rebuilt"].replace(0.0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    market["downside_pressure_percentile"] = causal_percentile(
        np.log(market["downside_ratio_rebuilt"].replace(0.0, np.nan)),
        prior_rows=prior,
        minimum_prior_rows=minimum,
    )
    market["price_resilience_percentile"] = causal_percentile(
        market["total_return_3d"], prior_rows=prior, minimum_prior_rows=minimum
    )
    market = market.loc[market["date"].between(start, end)].copy()

    breadth = _combine_breadth(inputs)
    breadth["breadth_change5_rebuilt"] = breadth["equal_above_ma60_share"].diff(5)
    breadth["breadth_level_damage_raw"] = 1.0 - breadth["equal_above_ma60_share"]
    breadth["breadth_change_damage_raw"] = -breadth["breadth_change5_rebuilt"]
    breadth["breadth_level_damage_percentile"] = causal_percentile(
        breadth["breadth_level_damage_raw"], prior_rows=prior, minimum_prior_rows=minimum
    )
    breadth["breadth_change_damage_percentile"] = causal_percentile(
        breadth["breadth_change_damage_raw"], prior_rows=prior, minimum_prior_rows=minimum
    )
    breadth["breadth_stabilization_percentile"] = causal_percentile(
        breadth["breadth_change5_rebuilt"], prior_rows=prior, minimum_prior_rows=minimum
    )
    breadth["breadth_transmission_score"] = breadth[
        ["breadth_level_damage_percentile", "breadth_change_damage_percentile"]
    ].mean(axis=1, skipna=False)

    base = market.merge(breadth, on="date", how="left", validate="one_to_one")
    etf_columns: list[str] = []
    for name, code in [
        ("etf_510050", "510050"),
        ("etf_510500", "510500"),
        ("etf_159915", "159915"),
        ("etf_159919", "159919"),
    ]:
        etf = inputs[name][["date", "pct_chg"]].copy()
        etf[f"return_1d_{code}"] = pd.to_numeric(etf["pct_chg"], errors="raise") / 100.0
        etf[f"return_5d_{code}"] = (
            1.0 + etf[f"return_1d_{code}"]
        ).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
        base = base.merge(
            etf[["date", f"return_1d_{code}", f"return_5d_{code}"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        etf_columns.append(code)
    cross_codes = ["510050", "510500", "159915"]
    cross_1d = base[[f"return_1d_{code}" for code in cross_codes]]
    cross_5d = base[[f"return_5d_{code}" for code in cross_codes]]
    base["cross_index_available_etfs_1d"] = cross_1d.notna().sum(axis=1)
    base["cross_index_available_etfs_5d"] = cross_5d.notna().sum(axis=1)
    base["cross_index_negative_share_1d"] = cross_1d.lt(0.0).sum(axis=1).div(
        base["cross_index_available_etfs_1d"].replace(0, np.nan)
    )
    base.loc[
        base["cross_index_available_etfs_1d"].lt(2),
        "cross_index_negative_share_1d",
    ] = np.nan
    base["cross_index_median_return_5d"] = cross_5d.median(axis=1, skipna=True)
    base.loc[
        base["cross_index_available_etfs_5d"].lt(2),
        "cross_index_median_return_5d",
    ] = np.nan
    base["cross_index_negative_share_percentile"] = causal_percentile(
        base["cross_index_negative_share_1d"], prior_rows=prior, minimum_prior_rows=minimum
    )
    base["cross_index_loss_percentile"] = causal_percentile(
        -base["cross_index_median_return_5d"], prior_rows=prior, minimum_prior_rows=minimum
    )
    base["cross_index_etf_pressure"] = base[
        ["cross_index_negative_share_percentile", "cross_index_loss_percentile"]
    ].mean(axis=1, skipna=False)
    base["same_index_return_3d"] = (
        1.0 + base["return_1d_159919"]
    ).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1.0
    base["same_index_etf_pressure"] = causal_percentile(
        -base["same_index_return_3d"], prior_rows=prior, minimum_prior_rows=minimum
    )

    if_frame = inputs["if_features"][
        ["date", "eligible_contract_count", "annualized_oi_weighted_basis", "near_symbol"]
    ].copy()
    if_frame["if_raw_basis_change3"] = pd.to_numeric(
        if_frame["annualized_oi_weighted_basis"], errors="raise"
    ).diff(3)
    if_frame = if_frame.loc[if_frame["date"].between(start, end)].copy()
    if_frame["if_raw_basis_deterioration"] = -if_frame["if_raw_basis_change3"]
    if_frame["if_raw_basis_pressure"] = causal_percentile(
        if_frame["if_raw_basis_deterioration"], prior_rows=prior, minimum_prior_rows=minimum
    )
    base = base.merge(if_frame, on="date", how="left", validate="one_to_one")
    base["transmission_score"] = base[
        [
            "breadth_transmission_score",
            "cross_index_etf_pressure",
            "if_raw_basis_pressure",
            "downside_pressure_percentile",
        ]
    ].mean(axis=1, skipna=False)
    base["scale_normalized_negative_price_move"] = (
        (-base["total_return_3d"]).clip(lower=0.0)
        / base["prior_abs_total_return_3d_median"].replace(0.0, np.nan)
    )
    base["marginal_negative_price_impact"] = (
        base["scale_normalized_negative_price_move"]
        / base["transmission_score"].clip(lower=0.05)
    )
    base["inverse_marginal_price_impact"] = 1.0 / (
        1.0 + base["marginal_negative_price_impact"]
    )
    memory_rows = int(
        config["feature_construction"]["absorption_exhaustion"]["pressure_memory_rows"]
    )
    memory_minimum = int(
        config["feature_construction"]["absorption_exhaustion"][
            "pressure_memory_minimum_available_rows"
        ]
    )
    base["pressure_memory"] = base["transmission_score"].rolling(
        memory_rows, min_periods=memory_minimum
    ).max()
    base["exhaustion_base_score"] = base[
        [
            "downside_relief_fraction",
            "price_resilience_percentile",
            "breadth_stabilization_percentile",
            "inverse_marginal_price_impact",
        ]
    ].mean(axis=1, skipna=False)
    base["exhaustion_score"] = base["exhaustion_base_score"] * base["pressure_memory"]

    upstream = inputs["router_state_panel"][
        [
            "date",
            "liquidity_stress_score",
            "discount_rate_pressure",
            "state",
            "price_ma20_total_return",
        ]
    ].copy()
    base = base.merge(upstream, on="date", how="inner", validate="one_to_one")
    base["pressure_source_score"] = base[
        ["liquidity_stress_score", "discount_rate_pressure"]
    ].mean(axis=1, skipna=False)
    base["source_class"] = base.apply(lambda row: _source_class(row, config), axis=1)
    base["era"] = base["date"].map(lambda date: _era_for_date(date, config))
    candidate = config["feature_construction"]["absorption_exhaustion"]
    base["exhaustion_candidate"] = (
        base["pressure_memory"].ge(
            float(candidate["point_in_time_candidate_pressure_memory_at_least"])
        )
        & base["exhaustion_score"].ge(
            float(candidate["point_in_time_candidate_score_at_least"])
        )
        & base["exhaustion_score"].diff().gt(0.0)
    )
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    panel = base.loc[base["date"].between(evaluation_start, end)].copy()
    if len(panel) != int(config["data_gates"]["expected_signal_rows"]):
        raise ContractError("机制面板交易日数量不等于冻结值")
    primary = [
        "pressure_source_score",
        "transmission_score",
        "exhaustion_score",
        "cross_index_etf_pressure",
        "breadth_transmission_score",
        "if_raw_basis_pressure",
    ]
    if panel["pressure_source_score"].isna().any():
        raise ContractError("压力来源分数不得缺失")
    warmup_rows = int(config["data_gates"]["maximum_leading_feature_warmup_rows"])
    missing_mask = panel[primary[1:]].isna().any(axis=1)
    missing_positions = set(np.flatnonzero(missing_mask.to_numpy()).tolist())
    allowed_positions = set(range(warmup_rows))
    if not missing_positions.issubset(allowed_positions):
        diagnostic = [
            "downside_relief_fraction",
            "price_resilience_percentile",
            "breadth_stabilization_percentile",
            "inverse_marginal_price_impact",
            "pressure_memory",
        ]
        missing = panel.loc[
            panel[primary].isna().any(axis=1), ["date", *primary, *diagnostic]
        ]
        raise ContractError(f"机制面板主要特征缺失：{missing.head().to_dict('records')}")
    panel["mechanism_feature_availability"] = np.where(
        missing_mask, "NO_VIEW_WARMUP", "AVAILABLE_POINT_IN_TIME"
    )
    future_columns = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    if future_columns:
        raise ContractError(f"机制面板包含未来字段：{future_columns}")
    return panel.reset_index(drop=True)


def _censored_outcomes(horizon: int) -> dict[str, Any]:
    return {
        f"future_return_{horizon}d": np.nan,
        f"future_mae_magnitude_{horizon}d": np.nan,
        f"future_mfe_{horizon}d": np.nan,
        f"future_positive_wealth_fraction_{horizon}d": np.nan,
        f"future_exit_date_{horizon}d": pd.NaT,
        f"future_censored_{horizon}d": True,
    }


def build_outcome_panel(
    config: dict[str, Any],
    mechanism_panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    prices = _normalize_date(market)
    prices = prices.loc[
        prices["date"].le(pd.Timestamp(config["dates"]["outcome_market_cutoff"]))
    ].reset_index(drop=True)
    distribution = dividends.copy()
    distribution["record_date"] = pd.to_datetime(
        distribution["record_date"], errors="raise"
    ).dt.normalize()
    distribution["ex_date"] = pd.to_datetime(
        distribution["ex_date"], errors="raise"
    ).dt.normalize()
    distribution["cash_dividend_per_share"] = pd.to_numeric(
        distribution["cash_dividend_per_share"], errors="raise"
    )
    dividend_events = {
        date: group[["record_date", "cash_dividend_per_share"]].to_dict("records")
        for date, group in distribution.groupby("ex_date")
    }
    positions = {date: index for index, date in enumerate(prices["date"])}
    horizons = [10, 20]
    rows: list[dict[str, Any]] = []
    for signal_date in mechanism_panel["date"]:
        row: dict[str, Any] = {"date": signal_date}
        signal_position = positions.get(signal_date)
        if signal_position is None or signal_position + 1 >= len(prices):
            for horizon in horizons:
                row.update(_censored_outcomes(horizon))
            rows.append(row)
            continue
        entry_position = signal_position + 1
        entry_open = float(prices.loc[entry_position, "open"])
        entry_date = prices.loc[entry_position, "date"]
        row["entry_date"] = entry_date
        row["entry_open"] = entry_open
        for horizon in horizons:
            exit_position = entry_position + horizon - 1
            if exit_position >= len(prices):
                row.update(_censored_outcomes(horizon))
                continue
            cumulative_cash = 0.0
            path: list[float] = []
            for position in range(entry_position, exit_position + 1):
                date = prices.loc[position, "date"]
                for dividend in dividend_events.get(date, []):
                    if entry_date <= dividend["record_date"]:
                        cumulative_cash += float(dividend["cash_dividend_per_share"])
                wealth = float(prices.loc[position, "close"]) + cumulative_cash
                path.append(wealth / entry_open - 1.0)
            values = np.asarray(path, dtype=float)
            row[f"future_return_{horizon}d"] = float(values[-1])
            row[f"future_mae_magnitude_{horizon}d"] = float(max(0.0, -values.min()))
            row[f"future_mfe_{horizon}d"] = float(max(0.0, values.max()))
            row[f"future_positive_wealth_fraction_{horizon}d"] = float(
                (values > 0.0).mean()
            )
            row[f"future_exit_date_{horizon}d"] = prices.loc[exit_position, "date"]
            row[f"future_censored_{horizon}d"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _repair_condition(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["total_return_3d"].gt(0.0)
        & panel["breadth_change5_rebuilt"].gt(0.0)
        & panel["total_return_index_rebuilt"].gt(panel["price_ma20_rebuilt"])
    )


def _first_independent_repair_anchor(
    panel: pd.DataFrame,
    start_position: int,
    end_position: int,
    maximum_post_rows: int,
) -> pd.Timestamp | pd.NaT:
    search_end = min(len(panel) - 1, end_position + maximum_post_rows)
    condition = _repair_condition(panel)
    for position in range(start_position, search_end + 1):
        previous = False if position == 0 else bool(condition.iloc[position - 1])
        if not previous and bool(condition.iloc[position]):
            return panel.loc[position, "date"]
    return pd.NaT


def build_router_event_atlas(
    config: dict[str, Any],
    inputs: dict[str, Any],
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = inputs["router_episodes"].copy()
    episodes["start_date"] = pd.to_datetime(episodes["start_date"]).dt.normalize()
    episodes["end_date"] = pd.to_datetime(episodes["end_date"]).dt.normalize()
    episodes = episodes.loc[
        episodes["state"].eq(STRESS)
        & ~episodes["left_censored"].astype(bool)
        & ~episodes["right_censored"].astype(bool)
    ].copy()
    positions = {date: index for index, date in enumerate(panel["date"])}
    outcome_by_date = outcomes.set_index("date")
    events: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    question_a: list[dict[str, Any]] = []
    question_b: list[dict[str, Any]] = []
    b_spec = config["prediction_questions"]["question_b_exhaustion"]
    for _, episode in episodes.iterrows():
        start_date = episode["start_date"]
        end_date = episode["end_date"]
        start_position = positions[start_date]
        end_position = positions[end_date]
        event_id = f"RSE_{int(episode['episode_id']):03d}"
        event_slice = panel.loc[start_position:end_position]
        candidates = event_slice.loc[event_slice["exhaustion_candidate"]]
        candidate_date = pd.NaT if candidates.empty else candidates["date"].iloc[0]
        candidate_position = None if pd.isna(candidate_date) else positions[candidate_date]
        independent_repair_anchor = _first_independent_repair_anchor(
            panel, start_position, end_position, 20
        )
        timing_end = min(len(panel) - 1, end_position + 20)
        timing_candidates = panel.loc[start_position:timing_end]
        timing_candidates = timing_candidates.loc[timing_candidates["exhaustion_candidate"]]
        timing_candidate_date = (
            pd.NaT if timing_candidates.empty else timing_candidates["date"].iloc[0]
        )
        exhaustion_timing_lead = (
            None
            if pd.isna(timing_candidate_date) or pd.isna(independent_repair_anchor)
            else float(
                positions[pd.Timestamp(independent_repair_anchor)]
                - positions[pd.Timestamp(timing_candidate_date)]
            )
        )
        acceleration_values = event_slice["transmission_score"].diff(3)
        acceleration_date = (
            pd.NaT
            if acceleration_values.dropna().empty
            else event_slice.loc[acceleration_values.idxmax(), "date"]
        )
        shock_end = min(len(panel) - 1, end_position + 10)
        shock_slice = panel.loc[start_position:shock_end]
        maximum_shock_date = shock_slice.loc[
            shock_slice["total_return_index_rebuilt"].idxmin(), "date"
        ]
        maximum_shock_lag = positions[maximum_shock_date] - start_position
        repair_lag_from_shock = (
            None
            if pd.isna(independent_repair_anchor)
            else positions[pd.Timestamp(independent_repair_anchor)]
            - positions[maximum_shock_date]
        )
        early_transmission = event_slice.iloc[:3]["transmission_score"].mean()
        late_transmission = event_slice.iloc[-3:]["transmission_score"].mean()
        if (
            maximum_shock_lag <= 5
            and repair_lag_from_shock is not None
            and 0 <= repair_lag_from_shock <= 10
        ):
            morphology_class = "FAST_SHOCK_THEN_REPAIR_RETROSPECTIVE"
        elif len(event_slice) >= 10 and late_transmission >= early_transmission:
            morphology_class = "PERSISTENT_TRANSMISSION_RETROSPECTIVE"
        else:
            morphology_class = "MIXED_OR_UNRESOLVED_RETROSPECTIVE"
        start_row = panel.loc[start_position]
        a_outcome = outcome_by_date.loc[start_date]
        a_label = (
            None
            if bool(a_outcome["future_censored_10d"])
            else bool(
                a_outcome["future_mae_magnitude_10d"]
                >= float(
                    config["prediction_questions"]["question_a_left_tail"][
                        "major_mae_threshold"
                    ]
                )
            )
        )
        b_label: bool | None = None
        if candidate_position is not None:
            b_outcome = outcome_by_date.loc[candidate_date]
            if not bool(b_outcome["future_censored_20d"]):
                b_label = bool(
                    b_outcome["future_return_20d"]
                    >= float(b_spec["terminal_return_threshold"])
                    and b_outcome["future_positive_wealth_fraction_20d"]
                    >= float(b_spec["positive_wealth_fraction_threshold"])
                    and b_outcome["future_mae_magnitude_20d"]
                    <= float(b_spec["maximum_mae_threshold"])
                )
        source_class = str(start_row["source_class"])
        event = {
            "event_id": event_id,
            "upstream_episode_id": int(episode["episode_id"]),
            "start_date": start_date,
            "end_date": end_date,
            "signal_rows": int(episode["signal_rows"]),
            "era": str(start_row["era"]),
            "source_class_at_start": source_class,
            "pressure_source_score_at_start": _safe_float(
                start_row["pressure_source_score"]
            ),
            "transmission_score_at_start": _safe_float(
                start_row["transmission_score"]
            ),
            "exhaustion_score_at_start": _safe_float(start_row["exhaustion_score"]),
            "exhaustion_candidate_date": candidate_date,
            "exhaustion_timing_candidate_date": timing_candidate_date,
            "independent_repair_anchor_date": independent_repair_anchor,
            "exhaustion_timing_lead_rows": exhaustion_timing_lead,
            "acceleration_date_retrospective": acceleration_date,
            "maximum_price_shock_date_retrospective": maximum_shock_date,
            "maximum_price_shock_lag_rows": int(maximum_shock_lag),
            "repair_lag_from_maximum_shock_rows": repair_lag_from_shock,
            "morphology_class_retrospective": morphology_class,
            "question_a_major_left_tail_10d": a_label,
            "question_b_sustained_repair_20d": b_label,
        }
        events.append(event)
        for phase, date, availability, used in [
            ("PRESSURE_ONSET", start_date, "POINT_IN_TIME_UPSTREAM_EVENT_START", True),
            ("TRANSMISSION_ACCELERATION", acceleration_date, "RETROSPECTIVE_ONLY", False),
            ("MAXIMUM_PRICE_SHOCK", maximum_shock_date, "RETROSPECTIVE_ONLY", False),
            ("FIRST_EXHAUSTION_CANDIDATE", candidate_date, "POINT_IN_TIME_FIRST_CROSSING", True),
            (
                "INDEPENDENT_REPAIR_ANCHOR",
                independent_repair_anchor,
                "POINT_IN_TIME_RULE_RETROSPECTIVE_EVENT_ALIGNMENT",
                False,
            ),
        ]:
            landmarks.append(
                {
                    "event_id": event_id,
                    "phase": phase,
                    "date": date,
                    "availability_class": availability,
                    "used_in_prediction": used,
                }
            )
        pre = panel.loc[max(0, start_position - 5) : start_position - 1]
        early = panel.loc[start_position : min(end_position, start_position + 2)]
        late = panel.loc[max(start_position, end_position - 2) : end_position]
        trajectories.append(
            {
                "event_id": event_id,
                "era": str(start_row["era"]),
                "source_class_at_start": source_class,
                "morphology_class_retrospective": morphology_class,
                "pre_source_score": _safe_float(pre["pressure_source_score"].mean()),
                "onset_source_score": _safe_float(early["pressure_source_score"].mean()),
                "pre_transmission_score": _safe_float(pre["transmission_score"].mean()),
                "onset_transmission_score": _safe_float(early["transmission_score"].mean()),
                "transmission_onset_delta": _safe_float(
                    early["transmission_score"].mean() - pre["transmission_score"].mean()
                ),
                "early_exhaustion_score": _safe_float(early["exhaustion_score"].mean()),
                "late_exhaustion_score": _safe_float(late["exhaustion_score"].mean()),
                "exhaustion_late_minus_early": _safe_float(
                    late["exhaustion_score"].mean() - early["exhaustion_score"].mean()
                ),
                "early_marginal_price_impact": _safe_float(
                    early["marginal_negative_price_impact"].mean()
                ),
                "late_marginal_price_impact": _safe_float(
                    late["marginal_negative_price_impact"].mean()
                ),
                "marginal_price_impact_late_minus_early": _safe_float(
                    late["marginal_negative_price_impact"].mean()
                    - early["marginal_negative_price_impact"].mean()
                ),
            }
        )
        if (
            a_label is not None
            and pd.notna(start_row["pressure_source_score"])
            and pd.notna(start_row["transmission_score"])
        ):
            question_a.append(
                {
                    "event_id": event_id,
                    "date": start_date,
                    "era": str(start_row["era"]),
                    "pressure_source_score": float(start_row["pressure_source_score"]),
                    "transmission_score": float(start_row["transmission_score"]),
                    "major_left_tail_10d": int(a_label),
                    "event_weight": 1.0,
                }
            )
        if candidate_position is not None and b_label is not None:
            candidate_row = panel.loc[candidate_position]
            question_b.append(
                {
                    "event_id": event_id,
                    "date": candidate_date,
                    "era": str(candidate_row["era"]),
                    "transmission_score": float(candidate_row["transmission_score"]),
                    "exhaustion_score": float(candidate_row["exhaustion_score"]),
                    "sustained_repair_20d": int(b_label),
                    "event_weight": 1.0,
                }
            )
    return (
        pd.DataFrame(events),
        pd.DataFrame(landmarks),
        pd.DataFrame(trajectories),
        pd.DataFrame(question_a),
        pd.DataFrame(question_b),
    )


def _crossing_timing(
    values: pd.Series,
    anchor_position: int,
    lookback: int,
    followup: int,
    threshold: float,
) -> tuple[pd.Timestamp | pd.NaT, float | None, bool]:
    start = max(0, anchor_position - lookback)
    end = min(len(values) - 1, anchor_position + followup)
    current_at_start = values.iloc[start]
    if pd.notna(current_at_start) and float(current_at_start) >= threshold:
        if start > 0:
            previous = values.iloc[start - 1]
            if pd.notna(previous) and float(previous) < threshold:
                return values.index[start], float(anchor_position - start), False
        return pd.NaT, None, True
    for position in range(start + 1, anchor_position + 1):
        previous = values.iloc[position - 1]
        current = values.iloc[position]
        if pd.notna(previous) and pd.notna(current) and previous < threshold <= current:
            return values.index[position], float(anchor_position - position), False
    for position in range(anchor_position + 1, end + 1):
        previous = values.iloc[position - 1]
        current = values.iloc[position]
        if pd.notna(previous) and pd.notna(current) and previous < threshold <= current:
            return values.index[position], float(anchor_position - position), False
    return pd.NaT, None, False


def build_price_damage_events(
    config: dict[str, Any],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    specification = config["event_definitions"]["price_damage_events"]
    peak_rows = int(specification["rolling_peak_rows"])
    threshold = float(specification["drawdown_threshold"])
    cooldown = int(specification["cooldown_trading_rows"])
    future_rows = int(specification["minimum_complete_future_rows"])
    lookback = int(specification["leading_lookback_rows"])
    followup = int(specification["lagging_followup_rows"])
    crossing_threshold = float(config["gates"]["leading_gate"]["channel_crossing_threshold"])
    drawdown = panel["total_return_index_rebuilt"] / panel[
        "total_return_index_rebuilt"
    ].rolling(peak_rows, min_periods=peak_rows).max() - 1.0
    triggers = drawdown.le(threshold) & drawdown.shift(1).gt(threshold)
    channels = {
        "source": "pressure_source_score",
        "transmission": "transmission_score",
        "cross_index_etf": "cross_index_etf_pressure",
        "same_index_etf": "same_index_etf_pressure",
        "breadth": "breadth_transmission_score",
        "if_raw_basis": "if_raw_basis_pressure",
        "h3_downside": "downside_pressure_percentile",
    }
    indexed: dict[str, pd.Series] = {}
    date_index = pd.DatetimeIndex(panel["date"])
    for label, column in channels.items():
        series = pd.Series(panel[column].to_numpy(), index=date_index)
        indexed[label] = series
    rows: list[dict[str, Any]] = []
    last_anchor = -10_000
    event_number = 0
    for position in np.flatnonzero(triggers.to_numpy()):
        if position - last_anchor < cooldown:
            continue
        last_anchor = int(position)
        event_number += 1
        complete = bool(position + future_rows < len(panel))
        row: dict[str, Any] = {
            "event_id": f"PDE_{event_number:03d}",
            "trigger_date": panel.loc[position, "date"],
            "era": panel.loc[position, "era"],
            "drawdown20_at_trigger": float(drawdown.iloc[position]),
            "future_20d_complete": complete,
        }
        for label, series in indexed.items():
            crossing_date, lead, left_censored = _crossing_timing(
                series, position, lookback, followup, crossing_threshold
            )
            row[f"{label}_cross_date"] = crossing_date
            row[f"{label}_lead_rows"] = lead
            row[f"{label}_left_censored"] = left_censored
        rows.append(row)
    return pd.DataFrame(rows)


def direction_summary(
    values: pd.Series,
    *,
    expected: str,
    minimum_observations: int,
    minimum_fraction: float,
    maximum_pvalue: float,
    minimum_median: float = 0.0,
) -> DirectionSummary:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    oriented = clean if expected == "positive" else -clean
    observations = len(oriented)
    median = _safe_float(clean.median())
    positive = int(oriented.gt(0.0).sum())
    fraction = None if observations == 0 else positive / observations
    pvalue = (
        None
        if observations == 0
        else float(binomtest(positive, observations, 0.5, alternative="greater").pvalue)
    )
    oriented_median = _safe_float(oriented.median())
    passed = bool(
        observations >= minimum_observations
        and oriented_median is not None
        and oriented_median >= minimum_median
        and fraction is not None
        and fraction >= minimum_fraction
        and pvalue is not None
        and pvalue <= maximum_pvalue
    )
    return DirectionSummary(observations, median, fraction, pvalue, passed)


def _timing_classification(
    values: pd.Series,
    passed: bool,
    minimum_observations: int,
) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < minimum_observations:
        return "INSUFFICIENT_NONCENSORED_TIMING_EVENTS"
    if passed:
        return "LEADING"
    median = float(clean.median())
    if median < 0.0:
        return "LAGGING"
    if median == 0.0:
        return "SYNCHRONOUS"
    return "MIXED_NOT_RELIABLY_LEADING"


def build_leading_results(
    config: dict[str, Any],
    price_events: pd.DataFrame,
    router_events: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    gate = config["gates"]["leading_gate"]
    rows: list[dict[str, Any]] = []
    channel_map = {
        "PRESSURE_SOURCE": "source_lead_rows",
        "TRANSMISSION": "transmission_lead_rows",
        "CROSS_INDEX_ETF": "cross_index_etf_lead_rows",
        "SAME_INDEX_ETF": "same_index_etf_lead_rows",
        "CSI300_COMPONENT_BREADTH": "breadth_lead_rows",
        "IF_RAW_BASIS": "if_raw_basis_lead_rows",
        "H3_DOWNSIDE": "h3_downside_lead_rows",
    }
    complete = price_events.loc[price_events["future_20d_complete"].astype(bool)]
    for channel, column in channel_map.items():
        summary = direction_summary(
            complete[column],
            expected="positive",
            minimum_observations=int(gate["minimum_valid_price_damage_events"]),
            minimum_fraction=float(gate["minimum_positive_lead_fraction"]),
            maximum_pvalue=float(gate["maximum_one_sided_sign_test_pvalue"]),
            minimum_median=float(gate["minimum_median_lead_trading_rows"]),
        )
        clean = pd.to_numeric(complete[column], errors="coerce").dropna()
        rows.append(
            {
                "channel": channel,
                **summary.as_dict(),
                "leading_count": int(clean.gt(0.0).sum()),
                "synchronous_count": int(clean.eq(0.0).sum()),
                "lagging_count": int(clean.lt(0.0).sum()),
                "timing_classification": _timing_classification(
                    clean, summary.passed, int(gate["minimum_valid_price_damage_events"])
                ),
            }
        )

    positions = {date: index for index, date in enumerate(panel["date"])}
    exhaustion_leads: list[float] = []
    for _, event in router_events.dropna(subset=["exhaustion_timing_lead_rows"]).iterrows():
        exhaustion_leads.append(float(event["exhaustion_timing_lead_rows"]))
    summary = direction_summary(
        pd.Series(exhaustion_leads, dtype=float),
        expected="positive",
        minimum_observations=int(gate["question_b_minimum_rows"]),
        minimum_fraction=float(gate["minimum_positive_lead_fraction"]),
        maximum_pvalue=float(gate["maximum_one_sided_sign_test_pvalue"]),
        minimum_median=float(gate["minimum_median_lead_trading_rows"]),
    )
    exhaustion_clean = pd.Series(exhaustion_leads, dtype=float)
    rows.append(
        {
            "channel": "EXHAUSTION_TO_REPAIR",
            **summary.as_dict(),
            "leading_count": int(exhaustion_clean.gt(0.0).sum()),
            "synchronous_count": int(exhaustion_clean.eq(0.0).sum()),
            "lagging_count": int(exhaustion_clean.lt(0.0).sum()),
            "timing_classification": _timing_classification(
                exhaustion_clean, summary.passed, int(gate["question_b_minimum_rows"])
            ),
        }
    )
    return pd.DataFrame(rows)


def build_event_summaries(
    router_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def summarize(grouped: Any, group_name: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for key, sample in grouped:
            question_a = pd.to_numeric(
                sample["question_a_major_left_tail_10d"], errors="coerce"
            ).dropna()
            question_b = pd.to_numeric(
                sample["question_b_sustained_repair_20d"], errors="coerce"
            ).dropna()
            morphology = sample["morphology_class_retrospective"].value_counts()
            rows.append(
                {
                    group_name: str(key),
                    "events": int(len(sample)),
                    "question_a_evaluable_events": int(len(question_a)),
                    "question_a_major_left_tail_rate": _safe_float(question_a.mean()),
                    "exhaustion_candidate_events": int(
                        sample["exhaustion_candidate_date"].notna().sum()
                    ),
                    "question_b_evaluable_events": int(len(question_b)),
                    "question_b_sustained_repair_rate": _safe_float(question_b.mean()),
                    "median_pressure_source_at_start": _safe_float(
                        sample["pressure_source_score_at_start"].median()
                    ),
                    "median_transmission_at_start": _safe_float(
                        sample["transmission_score_at_start"].median()
                    ),
                    "fast_shock_then_repair_events": int(
                        morphology.get("FAST_SHOCK_THEN_REPAIR_RETROSPECTIVE", 0)
                    ),
                    "persistent_transmission_events": int(
                        morphology.get("PERSISTENT_TRANSMISSION_RETROSPECTIVE", 0)
                    ),
                    "mixed_or_unresolved_events": int(
                        morphology.get("MIXED_OR_UNRESOLVED_RETROSPECTIVE", 0)
                    ),
                }
            )
        return pd.DataFrame(rows)

    period = summarize(router_events.groupby("era", sort=False), "era")
    source = summarize(
        router_events.groupby("source_class_at_start", sort=False),
        "source_class_at_start",
    )
    return period, source


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in frame.to_dict("records"):
        record: dict[str, Any] = {}
        for key, value in source.items():
            if value is None or pd.isna(value):
                record[key] = None
            elif isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (np.floating,)):
                record[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                record[key] = value.date().isoformat()
            else:
                record[key] = value
        records.append(record)
    return records


def _fit_logistic(frame: pd.DataFrame, predictors: list[str], target: str) -> LogisticRegression:
    if frame[target].nunique() < 2:
        raise ValueError("逻辑回归标签只有一个类别")
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(frame[predictors].to_numpy(dtype=float), frame[target].to_numpy(dtype=int))
    return model


def evaluate_question_model(
    config: dict[str, Any],
    question_id: str,
    frame: pd.DataFrame,
    predictors: list[str],
    target: str,
    primary_predictor: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate = config["gates"]["leading_gate"]
    estimation = config["prediction_questions"]["estimation"]
    minimum_rows = int(
        gate["question_a_minimum_rows"]
        if question_id == "QUESTION_A_LEFT_TAIL"
        else gate["question_b_minimum_rows"]
    )
    minimum_class = int(
        gate["question_a_minimum_each_class"]
        if question_id == "QUESTION_A_LEFT_TAIL"
        else gate["question_b_minimum_each_class"]
    )
    class_counts = frame[target].value_counts().to_dict() if not frame.empty else {}
    sample_pass = bool(
        len(frame) >= minimum_rows
        and int(class_counts.get(0, 0)) >= minimum_class
        and int(class_counts.get(1, 0)) >= minimum_class
        and frame["event_id"].is_unique
    )
    output_rows: list[dict[str, Any]] = []
    if not sample_pass:
        result = {
            "question_id": question_id,
            "status": "NOT_EVALUABLE_INSUFFICIENT_EVENT_OR_CLASS_SAMPLE",
            "rows": int(len(frame)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "sample_gate_passed": False,
            "model_gate_passed": False,
            "primary_predictor": primary_predictor,
            "primary_coefficient": None,
            "primary_bootstrap_90pct_ci": [None, None],
            "leave_one_era_out_evaluable_eras": 0,
            "leave_one_era_out_median_auc": None,
            "delete_any_event_preserves_primary_direction": False,
        }
        return result, pd.DataFrame(output_rows)

    model = _fit_logistic(frame, predictors, target)
    coefficients = dict(zip(predictors, model.coef_[0].astype(float)))
    for predictor in predictors:
        output_rows.append(
            {
                "question_id": question_id,
                "result_type": "FULL_COEFFICIENT",
                "segment": "FULL",
                "metric": predictor,
                "value": coefficients[predictor],
            }
        )

    repetitions = int(estimation["bootstrap_repetitions"])
    seed = int(estimation["bootstrap_seed"]) + (0 if question_id.endswith("LEFT_TAIL") else 1)
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {predictor: [] for predictor in predictors}
    for _ in range(repetitions):
        sample_positions = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[sample_positions]
        if sample[target].nunique() < 2:
            continue
        try:
            fitted = _fit_logistic(sample, predictors, target)
        except (ValueError, FloatingPointError):
            continue
        for index, predictor in enumerate(predictors):
            bootstrap[predictor].append(float(fitted.coef_[0][index]))
    confidence_alpha = (1.0 - float(estimation["directional_confidence_level"])) / 2.0
    intervals: dict[str, list[float | None]] = {}
    for predictor in predictors:
        values = bootstrap[predictor]
        interval = (
            [None, None]
            if len(values) < max(100, repetitions // 2)
            else [
                float(np.quantile(values, confidence_alpha)),
                float(np.quantile(values, 1.0 - confidence_alpha)),
            ]
        )
        intervals[predictor] = interval
        output_rows.extend(
            [
                {
                    "question_id": question_id,
                    "result_type": "BOOTSTRAP_CI",
                    "segment": "FULL",
                    "metric": f"{predictor}_lower",
                    "value": interval[0],
                },
                {
                    "question_id": question_id,
                    "result_type": "BOOTSTRAP_CI",
                    "segment": "FULL",
                    "metric": f"{predictor}_upper",
                    "value": interval[1],
                },
            ]
        )

    cv_rows: list[dict[str, Any]] = []
    for era in config["dates"]["eras"]:
        train = frame.loc[~frame["era"].eq(era)]
        test = frame.loc[frame["era"].eq(era)]
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        fitted = _fit_logistic(train, predictors, target)
        probability = fitted.predict_proba(test[predictors].to_numpy(dtype=float))[:, 1]
        auc = float(roc_auc_score(test[target], probability))
        brier = float(brier_score_loss(test[target], probability))
        cv_rows.append({"era": era, "rows": len(test), "auc": auc, "brier": brier})
        output_rows.extend(
            [
                {
                    "question_id": question_id,
                    "result_type": "LEAVE_ONE_ERA_OUT",
                    "segment": era,
                    "metric": "AUC",
                    "value": auc,
                },
                {
                    "question_id": question_id,
                    "result_type": "LEAVE_ONE_ERA_OUT",
                    "segment": era,
                    "metric": "BRIER",
                    "value": brier,
                },
            ]
        )
    auc_values = [row["auc"] for row in cv_rows]
    median_auc = None if not auc_values else float(np.median(auc_values))

    deletion_directions: list[bool] = []
    primary_index = predictors.index(primary_predictor)
    for event_id in frame["event_id"]:
        sample = frame.loc[~frame["event_id"].eq(event_id)]
        if sample[target].nunique() < 2:
            deletion_directions.append(False)
            continue
        fitted = _fit_logistic(sample, predictors, target)
        deletion_directions.append(bool(fitted.coef_[0][primary_index] > 0.0))
    deletion_pass = bool(deletion_directions and all(deletion_directions))
    primary_interval = intervals[primary_predictor]
    positive_ci = bool(
        primary_interval[0] is not None and float(primary_interval[0]) > 0.0
    )
    cv_pass = bool(
        len(auc_values) >= int(estimation["minimum_evaluable_eras"])
        and median_auc is not None
        and median_auc >= float(estimation["auc_threshold"])
    )
    model_pass = bool(sample_pass and positive_ci and cv_pass)
    result = {
        "question_id": question_id,
        "status": "EVALUATED",
        "rows": int(len(frame)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "sample_gate_passed": sample_pass,
        "predictors": predictors,
        "coefficients": coefficients,
        "bootstrap_90pct_ci": intervals,
        "primary_predictor": primary_predictor,
        "primary_coefficient": coefficients[primary_predictor],
        "primary_bootstrap_90pct_ci": primary_interval,
        "primary_positive_ci_passed": positive_ci,
        "leave_one_era_out": cv_rows,
        "leave_one_era_out_evaluable_eras": len(auc_values),
        "leave_one_era_out_median_auc": median_auc,
        "leave_one_era_out_auc_gate_passed": cv_pass,
        "delete_any_event_preserves_primary_direction": deletion_pass,
        "model_gate_passed": model_pass,
    }
    return result, pd.DataFrame(output_rows)


def evaluate_mechanism_gate(
    config: dict[str, Any], trajectories: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate = config["gates"]["mechanism_gate"]

    def summarize(frame: pd.DataFrame, column: str, expected: str) -> DirectionSummary:
        return direction_summary(
            frame[column],
            expected=expected,
            minimum_observations=int(gate["minimum_events_full_sample"]),
            minimum_fraction=float(gate["minimum_direction_fraction"]),
            maximum_pvalue=float(gate["maximum_one_sided_sign_test_pvalue"]),
        )

    full = {
        "transmission_onset": summarize(trajectories, "transmission_onset_delta", "positive"),
        "exhaustion_late_vs_early": summarize(
            trajectories, "exhaustion_late_minus_early", "positive"
        ),
        "marginal_price_impact_late_vs_early": summarize(
            trajectories, "marginal_price_impact_late_minus_early", "negative"
        ),
    }
    era_rows: list[dict[str, Any]] = []
    for era in config["dates"]["eras"]:
        sample = trajectories.loc[trajectories["era"].eq(era)]
        for mechanism, column, expected in [
            ("TRANSMISSION_ONSET", "transmission_onset_delta", "positive"),
            ("EXHAUSTION_LATE_VS_EARLY", "exhaustion_late_minus_early", "positive"),
            (
                "MARGINAL_PRICE_IMPACT_LATE_VS_EARLY",
                "marginal_price_impact_late_minus_early",
                "negative",
            ),
        ]:
            clean = pd.to_numeric(sample[column], errors="coerce").dropna()
            oriented = clean if expected == "positive" else -clean
            era_rows.append(
                {
                    "era": era,
                    "mechanism": mechanism,
                    "events": int(len(clean)),
                    "median": _safe_float(clean.median()),
                    "direction_fraction": _safe_float(oriented.gt(0.0).mean()),
                    "direction_pass": bool(
                        len(clean) >= int(gate["minimum_events_per_era"])
                        and _safe_float(oriented.median()) is not None
                        and float(oriented.median()) > 0.0
                        and float(oriented.gt(0.0).mean())
                        >= float(gate["minimum_direction_fraction"])
                    ),
                }
            )
    era_frame = pd.DataFrame(era_rows)
    transmission_eras = int(
        era_frame.loc[
            era_frame["mechanism"].eq("TRANSMISSION_ONSET"), "direction_pass"
        ].sum()
    )
    exhaustion_eras = int(
        era_frame.loc[
            era_frame["mechanism"].eq("EXHAUSTION_LATE_VS_EARLY"), "direction_pass"
        ].sum()
    )
    full_pass = all(summary.passed for summary in full.values())
    era_pass = bool(
        transmission_eras >= int(gate["minimum_eras_transmission_direction"])
        and exhaustion_eras >= int(gate["minimum_eras_exhaustion_direction"])
    )
    return (
        {
            "passed": bool(full_pass and era_pass),
            "full_sample": {key: value.as_dict() for key, value in full.items()},
            "transmission_eras_passing": transmission_eras,
            "exhaustion_eras_passing": exhaustion_eras,
            "era_thresholds": {
                "transmission": int(gate["minimum_eras_transmission_direction"]),
                "exhaustion": int(gate["minimum_eras_exhaustion_direction"]),
            },
            "full_sample_passed": full_pass,
            "era_replication_passed": era_pass,
        },
        era_frame,
    )


def evaluate_all_gates(
    config: dict[str, Any],
    router_events: pd.DataFrame,
    question_a_result: dict[str, Any],
    question_b_result: dict[str, Any],
    leading_results: pd.DataFrame,
    mechanism_result: dict[str, Any],
) -> dict[str, Any]:
    leading_map = leading_results.set_index("channel")["passed"].astype(bool).to_dict()
    leading_gate = bool(
        leading_map.get("TRANSMISSION", False)
        and leading_map.get("EXHAUSTION_TO_REPAIR", False)
        and question_a_result["model_gate_passed"]
        and question_b_result["model_gate_passed"]
    )
    event_gate_spec = config["gates"]["event_gate"]
    event_gate = bool(
        len(router_events) >= int(event_gate_spec["minimum_complete_router_events"])
        and question_a_result["rows"] <= len(router_events)
        and question_b_result["rows"] <= len(router_events)
        and question_a_result["delete_any_event_preserves_primary_direction"]
        and question_b_result["delete_any_event_preserves_primary_direction"]
    )
    external_spec = config["gates"]["external_replication_gate"]
    external_channels = list(external_spec["channels"])
    passing = {channel: bool(leading_map.get(channel, False)) for channel in external_channels}
    external_gate = bool(
        sum(passing.values())
        >= int(external_spec["minimum_channels_passing_leading_rule"])
        and passing["CROSS_INDEX_ETF"]
        and passing["CSI300_COMPONENT_BREADTH"]
    )
    all_pass = bool(
        leading_gate and event_gate and mechanism_result["passed"] and external_gate
    )
    return {
        "leading_gate": {
            "passed": leading_gate,
            "transmission_timing_passed": leading_map.get("TRANSMISSION", False),
            "exhaustion_timing_passed": leading_map.get(
                "EXHAUSTION_TO_REPAIR", False
            ),
            "question_a_model_passed": question_a_result["model_gate_passed"],
            "question_b_model_passed": question_b_result["model_gate_passed"],
        },
        "event_gate": {
            "passed": event_gate,
            "complete_router_events": int(len(router_events)),
            "minimum_required": int(event_gate_spec["minimum_complete_router_events"]),
            "question_a_delete_event_direction_passed": question_a_result[
                "delete_any_event_preserves_primary_direction"
            ],
            "question_b_delete_event_direction_passed": question_b_result[
                "delete_any_event_preserves_primary_direction"
            ],
        },
        "mechanism_gate": mechanism_result,
        "external_replication_gate": {
            "passed": external_gate,
            "channels": passing,
            "channels_passing": int(sum(passing.values())),
            "minimum_required": int(
                external_spec["minimum_channels_passing_leading_rule"]
            ),
            "if_semantic_limit": "RAW_ANNUALIZED_BASIS_NOT_FAIR_BASIS",
        },
        "all_four_gates_passed": all_pass,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _format_number(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    gates = report.get("gates", {})
    lines = [
        "# 510300 大盘压力传导、卖盘吸收与耗竭机制研究 V1",
        "",
        f"- 最终状态：`{report['status']}`",
        f"- 数据合同：`{'PASS' if report['data_admission']['passed'] else 'FAIL'}`",
        "- 证据等级：回顾性、历史污染的事件机制发现；不是策略回测。",
        "- 旧路由器：继续冻结拒绝，未改名、未改阈值、未改仓位。",
        "",
    ]
    if not report["data_admission"]["passed"]:
        lines.extend(
            [
                "## 数据门停止",
                "",
                "点时数据合同未通过，新图谱未来结果未读取，收益评价为`NOT_ALLOWED`。",
                "",
            ]
        )
        return "\n".join(lines)
    question_a = report["prediction_questions"]["QUESTION_A_LEFT_TAIL"]
    question_b = report["prediction_questions"]["QUESTION_B_EXHAUSTION"]
    leading_results = report["leading_results"]
    lines.extend(
        [
            "## 四道冻结门",
            "",
            "| 门 | 结果 |",
            "|---|---:|",
            f"| 领先门 | {'PASS' if gates['leading_gate']['passed'] else 'FAIL'} |",
            f"| 事件门 | {'PASS' if gates['event_gate']['passed'] else 'FAIL'} |",
            f"| 机制门 | {'PASS' if gates['mechanism_gate']['passed'] else 'FAIL'} |",
            f"| 外部复制门 | {'PASS' if gates['external_replication_gate']['passed'] else 'FAIL'} |",
            "",
            "## 两个预测问题",
            "",
            "| 问题 | 事件行 | 主系数 | 90%自助区间 | 留一时期AUC中位数 | 通过 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| A 未来10日重大左尾 | {question_a['rows']} | {_format_number(question_a.get('primary_coefficient'))} | `{question_a.get('primary_bootstrap_90pct_ci')}` | {_format_number(question_a.get('leave_one_era_out_median_auc'))} | {'是' if question_a['model_gate_passed'] else '否'} |",
            f"| B 未来20日持续修复 | {question_b['rows']} | {_format_number(question_b.get('primary_coefficient'))} | `{question_b.get('primary_bootstrap_90pct_ci')}` | {_format_number(question_b.get('leave_one_era_out_median_auc'))} | {'是' if question_b['model_gate_passed'] else '否'} |",
            "",
            "## 领先、同步与滞后",
            "",
            "| 通道 | 有效事件 | 中位有符号时差 | 领先/同步/滞后 | 冻结分类 | 通过 |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for channel, result in leading_results.items():
        lines.append(
            f"| {channel} | {result['observations']} | {_format_number(result['median_lead_rows'], 1)} | "
            f"{result['leading_count']}/{result['synchronous_count']}/{result['lagging_count']} | "
            f"`{result['timing_classification']}` | {'是' if result['passed'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 事件与来源",
            "",
            f"- 完整上游压力事件：{report['events']['router_event_count']}，每个事件等权。",
            f"- 独立价格损害事件：{report['events']['price_damage_event_count']}。",
            f"- 点时耗竭候选事件：{report['events']['exhaustion_candidate_event_count']}。",
            f"- 来源分类：`{report['events']['source_class_counts']}`。",
            f"- 回顾性路径形态：`{report['events']['morphology_class_counts']}`。",
            "- 完整点时基本面预期来源：`NO_VIEW_INCOMPLETE_CONTRACT`；未以新闻或年份补标签。",
            "- `FAST_SHOCK`与`PERSISTENT_TRANSMISSION`只描述事后路径，不等于已证明的外生冲击或去杠杆因果类型。",
            "",
            "## 边界",
            "",
            "没有组合净值、净夏普、净超额、最大回撤、仓位、当前信号、Paper、Shadow、订单、券商连接或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    inputs = load_inputs(config)
    data_audit = audit_inputs(config, inputs)
    paths = {key: project_path(value) for key, value in config["paths"].items()}
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if write:
        _atomic_json(paths["input_audit"], data_audit)
    if not data_audit["passed"]:
        report = {
            "project_id": PROJECT_ID,
            "status": config["adjudication"]["if_data_gate_fails"],
            "generated_at_asia_shanghai": generated_at,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "data_admission": data_audit,
            "return_evaluation": "NOT_ALLOWED",
            "strategy_backtest": "NOT_RUN",
            "boundaries": {
                "old_router_modified": False,
                "systemic_holdout_opened": False,
                "position_mapping": "DISABLED",
                "live_trading_authorized": False,
            },
        }
        if write:
            _atomic_json(paths["result_json"], report)
            _atomic_text(paths["result_markdown"], render_markdown(report))
        return report

    panel = build_mechanism_panel(config, inputs)
    outcomes = build_outcome_panel(config, panel, inputs["market"], inputs["dividends"])
    (
        router_events,
        phase_landmarks,
        trajectories,
        question_a_rows,
        question_b_rows,
    ) = build_router_event_atlas(config, inputs, panel, outcomes)
    price_events = build_price_damage_events(config, panel)
    leading_results = build_leading_results(
        config, price_events, router_events, panel
    )
    event_period_summary, event_source_summary = build_event_summaries(router_events)
    question_a_result, question_a_model_rows = evaluate_question_model(
        config,
        "QUESTION_A_LEFT_TAIL",
        question_a_rows,
        ["pressure_source_score", "transmission_score"],
        "major_left_tail_10d",
        "transmission_score",
    )
    question_b_result, question_b_model_rows = evaluate_question_model(
        config,
        "QUESTION_B_EXHAUSTION",
        question_b_rows,
        ["transmission_score", "exhaustion_score"],
        "sustained_repair_20d",
        "exhaustion_score",
    )
    model_results = pd.concat(
        [question_a_model_rows, question_b_model_rows], ignore_index=True
    )
    mechanism_result, era_results = evaluate_mechanism_gate(config, trajectories)
    gates = evaluate_all_gates(
        config,
        router_events,
        question_a_result,
        question_b_result,
        leading_results,
        mechanism_result,
    )
    if gates["all_four_gates_passed"]:
        status = config["adjudication"]["if_all_four_pass"]
    elif not gates["leading_gate"]["passed"]:
        status = config["adjudication"]["if_leading_gate_fails"]
    else:
        status = config["adjudication"]["if_other_gate_fails"]
    source_counts = {
        str(key): int(value)
        for key, value in router_events["source_class_at_start"].value_counts().items()
    }
    morphology_counts = {
        str(key): int(value)
        for key, value in router_events[
            "morphology_class_retrospective"
        ].value_counts().items()
    }
    report = {
        "project_id": PROJECT_ID,
        "version": config["protocol"]["version"],
        "status": status,
        "generated_at_asia_shanghai": generated_at,
        "evidence_class": config["protocol"]["evidence_class"],
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
        "data_admission": data_audit,
        "events": {
            "router_event_count": int(len(router_events)),
            "price_damage_event_count": int(len(price_events)),
            "complete_price_damage_event_count": int(
                price_events["future_20d_complete"].astype(bool).sum()
            ),
            "exhaustion_candidate_event_count": int(
                router_events["exhaustion_candidate_date"].notna().sum()
            ),
            "source_class_counts": source_counts,
            "morphology_class_counts": morphology_counts,
            "period_summary": _json_records(event_period_summary),
            "source_summary": _json_records(event_source_summary),
            "event_weighting": "ONE_EVENT_ONE_ROW_MAXIMUM_PER_PREDICTION_QUESTION",
        },
        "prediction_questions": {
            "QUESTION_A_LEFT_TAIL": question_a_result,
            "QUESTION_B_EXHAUSTION": question_b_result,
        },
        "leading_results": {
            str(row["channel"]): {
                "observations": int(row["observations"]),
                "median_lead_rows": _safe_float(row["median"]),
                "positive_lead_fraction": _safe_float(row["direction_fraction"]),
                "one_sided_sign_pvalue": _safe_float(row["one_sided_sign_pvalue"]),
                "leading_count": int(row["leading_count"]),
                "synchronous_count": int(row["synchronous_count"]),
                "lagging_count": int(row["lagging_count"]),
                "timing_classification": str(row["timing_classification"]),
                "passed": bool(row["passed"]),
            }
            for _, row in leading_results.iterrows()
        },
        "gates": gates,
        "known_data_limits": config["known_data_limits"],
        "return_evaluation": "EVENT_CONDITIONAL_PATH_DIAGNOSTICS_ONLY",
        "strategy_backtest": "NOT_RUN",
        "portfolio_metrics": {
            "net_sharpe": "NOT_COMPUTED",
            "net_excess": "NOT_COMPUTED",
            "maximum_drawdown": "NOT_COMPUTED",
        },
        "boundaries": {
            "old_router_status_overridden": False,
            "old_router_thresholds_modified": False,
            "old_router_position_mapping_reused": False,
            "retrospective_landmarks_used_in_prediction": False,
            "fundamental_source_imputed": False,
            "systemic_holdout_opened": False,
            "strategy_returns_computed": False,
            "portfolio_mapping": "DISABLED",
            "paper_signal": "DISABLED",
            "shadow_signal": "DISABLED",
            "current_signal": "NOT_CREATED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    if write:
        _atomic_parquet(paths["mechanism_panel"], panel)
        _atomic_parquet(paths["outcome_panel"], outcomes)
        _atomic_csv(paths["router_event_atlas"], router_events)
        _atomic_csv(paths["price_damage_events"], price_events)
        _atomic_csv(paths["phase_landmarks"], phase_landmarks)
        _atomic_csv(paths["event_trajectories"], trajectories)
        _atomic_csv(paths["event_period_summary"], event_period_summary)
        _atomic_csv(paths["event_source_summary"], event_source_summary)
        _atomic_csv(paths["leading_results"], leading_results)
        _atomic_csv(paths["question_a_rows"], question_a_rows)
        _atomic_csv(paths["question_b_rows"], question_b_rows)
        _atomic_csv(paths["model_results"], model_results)
        _atomic_csv(paths["era_mechanism_results"], era_results)
        _atomic_json(paths["result_json"], report)
        _atomic_text(paths["result_markdown"], render_markdown(report))
    return report


def main() -> int:
    report = run_study(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
