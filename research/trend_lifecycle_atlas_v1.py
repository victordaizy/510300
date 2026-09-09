"""510300趋势生命周期双标签机制图谱V1。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_TREND_LIFECYCLE_ATLAS_V1"
CONFIG_PATH = ROOT / "config" / "510300_trend_lifecycle_atlas_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_trend_lifecycle_atlas_v1_manifest.json"

LABEL_A = "LABEL_A_DIRECTIONAL_CHANGE"
LABEL_B = "LABEL_B_FUTURE_PATH"
EVENT_TYPES = ["UP_START", "UP_END", "DOWN_START", "DOWN_END"]


class ContractError(RuntimeError):
    """输入、冻结顺序或研究边界不满足。"""


@dataclass(frozen=True)
class LeadingResult:
    """单一标签、转移和变量的冻结领先性结果。"""

    label_system: str
    event_type: str
    feature: str
    module: str
    expected_direction: str
    event_observations: int
    positive_direction_count: int
    direction_fraction: float | None
    one_sided_sign_pvalue: float | None
    median_pre_event_z: float | None
    median_expected_direction_z: float | None
    median_onset_relative_day: float | None
    onset_event_count: int
    maximum_single_event_absolute_share: float | None
    early_event_count: int
    late_event_count: int
    early_median_expected_direction_z: float | None
    late_median_expected_direction_z: float | None
    early_late_direction_agrees: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def project_path(value: str) -> Path:
    """把协议内POSIX相对路径解析到项目根目录。"""

    return ROOT / PurePosixPath(value)


def sha256_file(path: Path) -> str:
    """计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_date(
    frame: pd.DataFrame,
    column: str = "date",
    *,
    unique: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    if column not in result.columns:
        raise ContractError(f"缺少日期字段：{column}")
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if unique and result[column].duplicated().any():
        raise ContractError(f"{column}存在重复值")
    return result


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ContractError(f"{label}缺少字段：{sorted(missing)}")


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _contains_value(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取并核对冻结协议的不可交易边界。"""

    if not path.exists():
        raise ContractError(f"协议不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ContractError("项目编号不匹配")
    if config["protocol"]["research_stage"] != "MECHANISM_DISCOVERY_ONLY":
        raise ContractError("研究阶段不是机制发现")
    if config["protocol"]["strategy_backtest_allowed"]:
        raise ContractError("第一阶段不得允许策略回测")
    if config["scope"]["return_backtest_allowed"]:
        raise ContractError("第一阶段不得读取组合收益")
    if config["scope"]["range_default_action"] != "NO_NEW_TRADE":
        raise ContractError("震荡分支默认动作必须是NO_NEW_TRADE")
    if config["scope"]["portfolio_mapping"] != "disabled":
        raise ContractError("仓位映射必须关闭")
    if config["scope"]["live_trading_authorized"]:
        raise ContractError("不得授权实盘")
    if config["data_gates"]["source_provenance_gate_for_phase_2"]:
        raise ContractError("当前历史权重来源不得误标为已通过严格点时来源门")
    return config


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """验证首次未来路径读取前的不可覆盖冻结清单。"""

    if not MANIFEST_PATH.exists():
        raise ContractError("冻结清单不存在，禁止读取新生命周期结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("冻结清单项目编号不匹配")
    if manifest.get("frozen_before_new_lifecycle_outcome_read") is not True:
        raise ContractError("未证明在新生命周期未来路径读取前冻结")
    if manifest.get("new_lifecycle_outcome_read_before_freeze") is not False:
        raise ContractError("冻结清单显示提前读取过新生命周期未来路径")
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
    """仅用当前行以前的历史值计算经验分位。"""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    for position, current in enumerate(values):
        start = max(0, position - prior_rows)
        history = values[start:position]
        history = history[np.isfinite(history)]
        if not np.isfinite(current) or len(history) < minimum_prior_rows:
            continue
        less = float(np.count_nonzero(history < current))
        equal = float(np.count_nonzero(history == current))
        output[position] = (less + 0.5 * equal) / len(history)
    return pd.Series(output, index=series.index, dtype=float)


def _read_json_input(specification: dict[str, Any]) -> dict[str, Any]:
    path = project_path(specification["path"])
    if not path.exists():
        raise ContractError(f"缺少输入：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """只加载既有点时特征源与上游裁决，不创建未来标签。"""

    specifications = config["inputs"]
    for name, specification in specifications.items():
        path = project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"缺少输入{name}：{path}")

    h00300 = _normalize_date(
        pd.read_parquet(project_path(specifications["h00300_total_return"]["path"])),
        unique=True,
    )
    etf = _normalize_date(
        pd.read_parquet(project_path(specifications["etf_market"]["path"])),
        unique=True,
    )
    cross = _normalize_date(
        pd.read_parquet(project_path(specifications["component_cross_section"]["path"])),
    )
    industry = _normalize_date(
        pd.read_parquet(project_path(specifications["industry_daily"]["path"])),
    )
    cgb = _normalize_date(
        pd.read_parquet(project_path(specifications["cgb_curve"]["path"])),
        unique=True,
    )

    return {
        "h00300": h00300,
        "etf": etf,
        "cross": cross,
        "industry": industry,
        "cgb": cgb,
        "index_driver_status": _read_json_input(specifications["index_driver_status"]),
        "membership_weight_evidence": _read_json_input(
            specifications["membership_weight_evidence"]
        ),
        "cgb_audit": _read_json_input(specifications["cgb_audit"]),
        "three_state_result": _read_json_input(specifications["three_state_result"]),
        "regime_router_result": _read_json_input(specifications["regime_router_result"]),
        "stress_atlas_result": _read_json_input(specifications["stress_atlas_result"]),
    }


def audit_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """冻结前审计身份、覆盖、时钟与上游边界；不生成Label B。"""

    specifications = config["inputs"]
    for name, frame_key in [
        ("h00300_total_return", "h00300"),
        ("etf_market", "etf"),
        ("component_cross_section", "cross"),
        ("industry_daily", "industry"),
        ("cgb_curve", "cgb"),
    ]:
        _require_columns(
            inputs[frame_key], specifications[name]["required_columns"], name
        )

    h00300 = inputs["h00300"]
    etf = inputs["etf"]
    cross = inputs["cross"]
    industry = inputs["industry"]
    gates = config["data_gates"]
    signal_cutoff = pd.Timestamp(config["dates"]["signal_cutoff"])
    feature_start = pd.Timestamp(config["dates"]["feature_history_start"])

    h00300_symbol_ok = set(h00300["symbol"].astype(str).unique()) == {
        specifications["h00300_total_return"]["required_symbol"]
    }
    etf_symbol_ok = set(etf["symbol"].astype(str).unique()) == {
        specifications["etf_market"]["required_symbol"]
    }
    h00300_calendar_ok = (
        h00300["date"].min() <= pd.Timestamp(config["dates"]["label_history_start"])
        and h00300["date"].max()
        >= pd.Timestamp(config["dates"]["label_market_cutoff"])
        and h00300["close"].gt(0).all()
    )

    cross_window = cross.loc[
        cross["date"].between(feature_start, signal_cutoff)
    ].copy()
    cross_counts = cross_window.groupby("date")["con_code"].nunique()
    cross_duplicate_rows = int(
        cross_window.duplicated(["date", "con_code"]).sum()
    )
    weight_sums = cross_window.groupby("date")["snapshot_weight"].sum()
    future_weight_rows = int(
        (
            pd.to_datetime(cross_window["weight_snapshot_date"]).dt.normalize()
            > cross_window["date"]
        ).sum()
    )
    snapshot_age = (
        cross_window["date"]
        - pd.to_datetime(cross_window["weight_snapshot_date"]).dt.normalize()
    ).dt.days
    price_coverage = (
        cross_window["snapshot_weight"]
        .where(cross_window["price_available"].astype(bool), 0.0)
        .groupby(cross_window["date"])
        .sum()
        / weight_sums
    )
    return_coverage = (
        cross_window["snapshot_weight"]
        .where(cross_window["return_available"].astype(bool), 0.0)
        .groupby(cross_window["date"])
        .sum()
        / weight_sums
    )
    industry_coverage = (
        cross_window["snapshot_weight"]
        .where(cross_window["industry_available"].astype(bool), 0.0)
        .groupby(cross_window["date"])
        .sum()
        / weight_sums
    )

    index_status_ok = (
        inputs["index_driver_status"].get("status")
        == specifications["index_driver_status"]["required_status"]
    )
    evidence = inputs["membership_weight_evidence"]
    evidence_status_ok = (
        evidence.get("status")
        == specifications["membership_weight_evidence"]["required_status"]
    )
    historical_weight_official_count = int(
        evidence.get("weights", {}).get(
            "official_historical_snapshot_verified_count", -1
        )
    )
    provenance_correctly_blocked = (
        evidence_status_ok
        and historical_weight_official_count
        == int(
            specifications["membership_weight_evidence"][
                "required_weight_official_verified_count"
            ]
        )
        and evidence.get("governance", {}).get("may_call_weights_strict_point_in_time")
        is False
    )
    cgb_ok = (
        inputs["cgb_audit"].get("status")
        == specifications["cgb_audit"]["required_status"]
    )
    upstream_checks = {
        "three_state_rejection_preserved": _contains_value(
            inputs["three_state_result"],
            config["upstream_boundaries"]["three_state_router"]["required_status"],
        ),
        "regime_router_rejection_preserved": _contains_value(
            inputs["regime_router_result"],
            config["upstream_boundaries"]["regime_transition_router"][
                "required_status"
            ],
        ),
        "stress_atlas_nonpromotion_preserved": _contains_value(
            inputs["stress_atlas_result"],
            config["upstream_boundaries"]["stress_atlas"]["required_status"],
        ),
    }
    checks = {
        "h00300_identity": h00300_symbol_ok,
        "h00300_calendar_and_positive_close": h00300_calendar_ok,
        "etf_identity": etf_symbol_ok,
        "cross_section_exactly_300_each_date": bool(
            not cross_counts.empty
            and cross_counts.eq(gates["expected_cross_section_rows_per_date"]).all()
        ),
        "cross_section_no_duplicate_date_symbol": cross_duplicate_rows == 0,
        "weight_sum_in_gate": bool(
            weight_sums.between(
                gates["minimum_weight_sum"], gates["maximum_weight_sum"]
            ).all()
        ),
        "no_future_weight_snapshot": future_weight_rows
        == gates["future_weight_snapshot_rows_allowed"],
        "weight_snapshot_age_in_gate": bool(
            snapshot_age.max() <= gates["maximum_weight_snapshot_age_calendar_days"]
        ),
        "price_coverage_has_evaluable_rows": bool(
            price_coverage.ge(gates["minimum_price_coverage_weight"]).any()
        ),
        "return_coverage_has_evaluable_rows": bool(
            return_coverage.ge(gates["minimum_return_coverage_weight"]).any()
        ),
        "industry_coverage_has_evaluable_rows": bool(
            industry_coverage.ge(gates["minimum_industry_coverage_weight"]).any()
        ),
        "industry_calendar_covers_signal_window": bool(
            industry["date"].min() <= feature_start
            and industry["date"].max() >= signal_cutoff
        ),
        "index_driver_attribution_input_status": index_status_ok,
        "historical_weight_provenance_explicitly_blocked": provenance_correctly_blocked,
        "cgb_long_history": cgb_ok,
        "upstream_boundaries": all(upstream_checks.values()),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "project_id": PROJECT_ID,
        "status": (
            "PASS_RETROSPECTIVE_ATLAS_INPUT_CONTRACT_SOURCE_PROVENANCE_BLOCKED"
            if passed
            else "FAIL_TREND_LIFECYCLE_ATLAS_INPUT_CONTRACT"
        ),
        "passed": passed,
        "checks": checks,
        "calendar": {
            "h00300_first": h00300["date"].min().date().isoformat(),
            "h00300_last": h00300["date"].max().date().isoformat(),
            "cross_section_first": cross_window["date"].min().date().isoformat(),
            "cross_section_last": cross_window["date"].max().date().isoformat(),
            "cross_section_dates": int(cross_window["date"].nunique()),
        },
        "cross_section": {
            "rows": int(len(cross_window)),
            "duplicate_date_symbol_rows": cross_duplicate_rows,
            "minimum_weight_sum": float(weight_sums.min()),
            "maximum_weight_sum": float(weight_sums.max()),
            "future_weight_snapshot_rows": future_weight_rows,
            "maximum_weight_snapshot_age_calendar_days": int(snapshot_age.max()),
            "minimum_price_coverage_weight": float(price_coverage.min()),
            "minimum_return_coverage_weight": float(return_coverage.min()),
            "minimum_industry_coverage_weight": float(industry_coverage.min()),
        },
        "source_provenance": {
            "strict_point_in_time_membership_and_weight_gate": False,
            "historical_official_weight_snapshots_verified": historical_weight_official_count,
            "allowed_evidence_class": config["protocol"]["evidence_class"],
            "phase_2_source_gate": False,
        },
        "upstream_checks": upstream_checks,
        "boundaries": {
            "new_lifecycle_future_labels_read": False,
            "strategy_returns_computed": False,
            "range_t_evaluated": False,
            "position_mapping": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def _rolling_average_pairwise_correlation(
    returns: pd.DataFrame,
    *,
    window: int,
    minimum_complete_rows: int,
    minimum_assets: int,
) -> pd.Series:
    """用标准化矩阵恒等式计算完整窗口资产的平均两两相关性。"""

    values = returns.to_numpy(dtype=float)
    result = np.full(len(returns), np.nan, dtype=float)
    for position in range(window - 1, len(returns)):
        block = values[position - window + 1 : position + 1]
        finite_counts = np.isfinite(block).sum(axis=0)
        usable = finite_counts >= minimum_complete_rows
        if int(usable.sum()) < minimum_assets:
            continue
        selected = block[:, usable]
        complete = np.isfinite(selected).all(axis=0)
        selected = selected[:, complete]
        asset_count = selected.shape[1]
        if asset_count < minimum_assets:
            continue
        means = selected.mean(axis=0)
        standard_deviations = selected.std(axis=0, ddof=0)
        nonconstant = standard_deviations > 1e-12
        selected = selected[:, nonconstant]
        standard_deviations = standard_deviations[nonconstant]
        means = means[nonconstant]
        asset_count = selected.shape[1]
        if asset_count < minimum_assets:
            continue
        standardized = (selected - means) / standard_deviations
        row_sums = standardized.sum(axis=1)
        numerator = float(np.square(row_sums).sum() - window * asset_count)
        denominator = float(window * asset_count * (asset_count - 1))
        result[position] = numerator / denominator
    return pd.Series(result, index=returns.index, dtype=float)


def _build_component_features(
    cross: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """从成员截面构造快照权重与等权参与度、集中度和相关性。"""

    settings = config["feature_construction"]
    frame = cross.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["weight_snapshot_date"] = pd.to_datetime(
        frame["weight_snapshot_date"]
    ).dt.normalize()
    frame.sort_values(["con_code", "date"], inplace=True)
    calendar = pd.Index(sorted(frame["date"].unique()))
    calendar_position = {date: index for index, date in enumerate(calendar)}
    frame["_calendar_position"] = frame["date"].map(calendar_position).astype(int)
    gaps = frame.groupby("con_code", sort=False)["_calendar_position"].diff()
    frame["_membership_segment"] = (
        gaps.ne(1).groupby(frame["con_code"], sort=False).cumsum().astype(int)
    )
    group_keys = [frame["con_code"], frame["_membership_segment"]]
    close = pd.to_numeric(frame["constituent_total_return_close"], errors="coerce")
    frame["_close"] = close
    minimum_fraction = float(settings["minimum_window_fraction"])

    feature_names: list[str] = []
    for window in settings["component_windows"]:
        window = int(window)
        minimum_rows = int(math.ceil(window * minimum_fraction))
        moving_average = (
            frame.groupby(["con_code", "_membership_segment"], sort=False)["_close"]
            .rolling(window, min_periods=minimum_rows)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .reindex(frame.index)
        )
        valid = close.notna() & moving_average.notna()
        flag = close.gt(moving_average) & valid
        weighted_numerator = frame["snapshot_weight"].where(flag, 0.0)
        weighted_denominator = frame["snapshot_weight"].where(valid, 0.0)
        daily_weighted = weighted_numerator.groupby(frame["date"]).sum() / (
            weighted_denominator.groupby(frame["date"]).sum().replace(0.0, np.nan)
        )
        daily_equal = flag.where(valid).groupby(frame["date"]).mean()
        frame[f"_ma{window}_valid"] = valid
        feature_names.extend(
            [
                f"breadth_above_ma{window}_snapshot_weighted",
                f"breadth_above_ma{window}_equal_weighted",
                f"ma{window}_coverage_weight",
            ]
        )
        if window == settings["component_windows"][0]:
            daily = pd.DataFrame(index=calendar)
        daily[f"breadth_above_ma{window}_snapshot_weighted"] = daily_weighted
        daily[f"breadth_above_ma{window}_equal_weighted"] = daily_equal
        daily[f"ma{window}_coverage_weight"] = (
            weighted_denominator.groupby(frame["date"]).sum()
            / frame["snapshot_weight"].groupby(frame["date"]).sum()
        )

    high_window = 60
    minimum_high_rows = int(math.ceil(high_window * minimum_fraction))
    rolling_high = (
        frame.groupby(["con_code", "_membership_segment"], sort=False)["_close"]
        .rolling(high_window, min_periods=minimum_high_rows)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .reindex(frame.index)
    )
    rolling_low = (
        frame.groupby(["con_code", "_membership_segment"], sort=False)["_close"]
        .rolling(high_window, min_periods=minimum_high_rows)
        .min()
        .reset_index(level=[0, 1], drop=True)
        .reindex(frame.index)
    )
    high_low_valid = close.notna() & rolling_high.notna() & rolling_low.notna()
    new_high = close.ge(rolling_high * (1.0 - 1e-12)) & high_low_valid
    new_low = close.le(rolling_low * (1.0 + 1e-12)) & high_low_valid
    high_low_denominator = frame["snapshot_weight"].where(high_low_valid, 0.0)
    daily["new_high60_weight_share"] = (
        frame["snapshot_weight"].where(new_high, 0.0).groupby(frame["date"]).sum()
        / high_low_denominator.groupby(frame["date"]).sum().replace(0.0, np.nan)
    )
    daily["new_low60_weight_share"] = (
        frame["snapshot_weight"].where(new_low, 0.0).groupby(frame["date"]).sum()
        / high_low_denominator.groupby(frame["date"]).sum().replace(0.0, np.nan)
    )
    daily["new_high_minus_new_low60"] = (
        daily["new_high60_weight_share"] - daily["new_low60_weight_share"]
    )

    contribution_window = int(settings["contribution_window_rows"])
    minimum_contribution_rows = int(
        math.ceil(contribution_window * minimum_fraction)
    )
    contribution = pd.to_numeric(
        frame["component_return_contribution_1d"], errors="coerce"
    )
    frame["_contribution"] = contribution
    rolling_contribution = (
        frame.groupby(["con_code", "_membership_segment"], sort=False)[
            "_contribution"
        ]
        .rolling(contribution_window, min_periods=minimum_contribution_rows)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .reindex(frame.index)
    )
    absolute_contribution = rolling_contribution.abs()
    daily_absolute_total = absolute_contribution.groupby(frame["date"]).transform("sum")
    contribution_share = absolute_contribution / daily_absolute_total.replace(0.0, np.nan)
    daily["contribution_hhi20"] = np.square(contribution_share).groupby(
        frame["date"]
    ).sum(min_count=1)

    constituent_return = pd.to_numeric(
        frame["constituent_return_1d"], errors="coerce"
    )
    return_valid = constituent_return.notna() & frame["return_available"].astype(bool)
    return_denominator = frame["snapshot_weight"].where(return_valid, 0.0)
    daily["extreme_down_weight"] = (
        frame["snapshot_weight"]
        .where(
            return_valid
            & constituent_return.le(settings["extreme_down_return_threshold"]),
            0.0,
        )
        .groupby(frame["date"])
        .sum()
        / return_denominator.groupby(frame["date"]).sum().replace(0.0, np.nan)
    )
    weight_sum = frame["snapshot_weight"].groupby(frame["date"]).sum()
    daily["cross_price_coverage_weight"] = (
        frame["snapshot_weight"]
        .where(frame["price_available"].astype(bool), 0.0)
        .groupby(frame["date"])
        .sum()
        / weight_sum
    )
    daily["cross_return_coverage_weight"] = (
        return_denominator.groupby(frame["date"]).sum() / weight_sum
    )
    daily["cross_industry_coverage_weight"] = (
        frame["snapshot_weight"]
        .where(frame["industry_available"].astype(bool), 0.0)
        .groupby(frame["date"])
        .sum()
        / weight_sum
    )
    daily["component_count"] = frame.groupby("date")["con_code"].nunique()
    daily["weight_sum"] = weight_sum
    daily["weight_snapshot_date"] = frame.groupby("date")[
        "weight_snapshot_date"
    ].max()
    daily["weight_snapshot_age_calendar_days"] = (
        pd.Series(daily.index, index=daily.index) - daily["weight_snapshot_date"]
    ).dt.days

    return_matrix = frame.pivot(index="date", columns="con_code", values="_contribution")
    # 相关性必须用个股收益而不是权重贡献。
    return_matrix = frame.pivot(
        index="date", columns="con_code", values="constituent_return_1d"
    ).reindex(calendar)
    daily["average_pairwise_correlation20"] = _rolling_average_pairwise_correlation(
        return_matrix,
        window=int(settings["correlation_window_rows"]),
        minimum_complete_rows=int(settings["correlation_minimum_complete_rows"]),
        minimum_assets=int(
            config["data_gates"]["minimum_valid_component_count_for_correlation"]
        ),
    )
    daily.index.name = "date"
    daily.reset_index(inplace=True)
    daily["breadth_diffusion5"] = daily[
        "breadth_above_ma60_snapshot_weighted"
    ].diff(int(settings["breadth_change_rows"]))
    return daily


def _build_industry_features(
    industry: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """构造行业20日方向扩散。"""

    frame = industry.copy()
    frame.sort_values(["industry_l1", "date"], inplace=True)
    returns = pd.to_numeric(frame["industry_return_1d"], errors="coerce")
    safe_log_return = np.log1p(returns.where(returns.gt(-1.0)))
    frame["_log_return"] = safe_log_return
    window = int(config["feature_construction"]["industry_path_rows"])
    minimum_rows = int(
        math.ceil(window * config["feature_construction"]["minimum_window_fraction"])
    )
    frame["_industry_log_return20"] = (
        frame.groupby("industry_l1", sort=False)["_log_return"]
        .rolling(window, min_periods=minimum_rows)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(frame.index)
    )
    valid = frame["_industry_log_return20"].notna()
    denominator = frame["industry_weight"].where(valid, 0.0)
    numerator = frame["industry_weight"].where(
        valid & frame["_industry_log_return20"].gt(0.0), 0.0
    )
    result = pd.DataFrame(
        {
            "industry_positive_weight20": numerator.groupby(frame["date"]).sum()
            / denominator.groupby(frame["date"]).sum().replace(0.0, np.nan),
            "industry_path_coverage_weight": denominator.groupby(frame["date"]).sum(),
        }
    ).reset_index()
    return result


def build_feature_panel(
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> pd.DataFrame:
    """建立不含未来字段的点时机制面板。"""

    dates = config["dates"]
    start = pd.Timestamp(dates["feature_history_start"])
    cutoff = pd.Timestamp(dates["signal_cutoff"])
    price = inputs["h00300"].loc[
        inputs["h00300"]["date"].between(start, cutoff), ["date", "close"]
    ].copy()
    price.rename(columns={"close": "h00300_total_return_close"}, inplace=True)
    price.sort_values("date", inplace=True)
    price.reset_index(drop=True, inplace=True)
    log_close = np.log(price["h00300_total_return_close"].astype(float))
    daily_log_return = log_close.diff()
    price["h00300_log_return_1d"] = daily_log_return
    volatility = daily_log_return.rolling(60, min_periods=40).std(ddof=1)
    price["trailing_volatility60"] = volatility
    for window in [20, 60, 120]:
        price[f"price_trend_z{window}"] = log_close.diff(window) / (
            volatility * math.sqrt(window)
        )
    for window in [20, 60]:
        path_length = daily_log_return.abs().rolling(window, min_periods=window).sum()
        price[f"signed_efficiency{window}"] = log_close.diff(window) / path_length.replace(
            0.0, np.nan
        )
    price["drawdown60"] = (
        price["h00300_total_return_close"]
        / price["h00300_total_return_close"].rolling(60, min_periods=48).max()
        - 1.0
    )
    prior_peak = (
        price["h00300_total_return_close"]
        .shift(5)
        .rolling(60, min_periods=48)
        .max()
    )
    held_floor = price["h00300_total_return_close"].rolling(5, min_periods=5).min()
    price["breakout_acceptance5_60"] = held_floor / prior_peak - 1.0
    downside_squared = np.square(daily_log_return.clip(upper=0.0))
    short_downside = np.sqrt(
        downside_squared.ewm(
            span=config["feature_construction"]["downside_short_ewm_span"],
            adjust=False,
            min_periods=config["feature_construction"]["downside_short_ewm_span"],
        ).mean()
    )
    long_downside = np.sqrt(
        downside_squared.ewm(
            span=config["feature_construction"]["downside_long_ewm_span"],
            adjust=False,
            min_periods=config["feature_construction"]["downside_long_ewm_span"],
        ).mean()
    )
    price["downside_vol_ratio5_20"] = short_downside / long_downside.replace(
        0.0, np.nan
    )

    etf = inputs["etf"].loc[
        inputs["etf"]["date"].between(start, cutoff), ["date", "amount"]
    ].copy()
    price = price.merge(etf, on="date", how="left", validate="one_to_one")
    raw_liquidity_impact = daily_log_return.mul(-1.0).clip(lower=0.0) / (
        price["amount"] / 100_000_000.0
    ).replace(0.0, np.nan)
    price["liquidity_impact_raw"] = raw_liquidity_impact
    price["liquidity_impact_pressure"] = causal_percentile(
        raw_liquidity_impact,
        prior_rows=config["feature_construction"]["causal_percentile_prior_rows"],
        minimum_prior_rows=config["feature_construction"][
            "causal_percentile_minimum_prior_rows"
        ],
    )

    component = _build_component_features(
        inputs["cross"].loc[inputs["cross"]["date"].between(start, cutoff)].copy(),
        config,
    )
    industry = _build_industry_features(
        inputs["industry"].loc[
            inputs["industry"]["date"].between(start, cutoff)
        ].copy(),
        config,
    )
    panel = price.merge(component, on="date", how="left", validate="one_to_one")
    panel = panel.merge(industry, on="date", how="left", validate="one_to_one")

    cgb = inputs["cgb"].copy()
    cgb["cgb_10y_change63"] = cgb["cgb_10y"].diff(
        config["feature_construction"]["cgb_change_rows"]
    )
    cgb["cgb_10y_change63_pressure"] = causal_percentile(
        cgb["cgb_10y_change63"],
        prior_rows=config["feature_construction"]["cgb_percentile_prior_rows"],
        minimum_prior_rows=config["feature_construction"][
            "cgb_percentile_minimum_prior_rows"
        ],
    )
    cgb = cgb[["date", "cgb_10y", "cgb_10y_change63", "cgb_10y_change63_pressure"]]
    panel.sort_values("date", inplace=True)
    cgb.sort_values("date", inplace=True)
    panel = pd.merge_asof(panel, cgb, on="date", direction="backward")

    evaluation_start = pd.Timestamp(dates["evaluation_start"])
    gates = config["data_gates"]
    required_features = [
        feature
        for module_features in config["feature_construction"]["feature_modules"].values()
        for feature in module_features
    ]
    required_features = list(dict.fromkeys(required_features))
    missing_required = set(required_features).difference(panel.columns)
    if missing_required:
        raise ContractError(f"机制面板缺少冻结变量：{sorted(missing_required)}")
    coverage_ok = (
        panel["cross_price_coverage_weight"].ge(gates["minimum_price_coverage_weight"])
        & panel["cross_return_coverage_weight"].ge(
            gates["minimum_return_coverage_weight"]
        )
        & panel["cross_industry_coverage_weight"].ge(
            gates["minimum_industry_coverage_weight"]
        )
        & panel["weight_snapshot_age_calendar_days"].le(
            gates["maximum_weight_snapshot_age_calendar_days"]
        )
    )
    features_complete = panel[required_features].notna().all(axis=1)
    panel["formal_evaluation_row"] = panel["date"].ge(evaluation_start)
    panel["valid_for_atlas"] = (
        panel["formal_evaluation_row"] & coverage_ok & features_complete
    )
    panel["data_status"] = np.select(
        [
            panel["date"].lt(evaluation_start),
            ~coverage_ok,
            ~features_complete,
        ],
        [
            "WARMUP_NOT_EVALUATED",
            "NO_VIEW_CROSS_SECTION_COVERAGE",
            "NO_VIEW_FEATURE_INCOMPLETE",
        ],
        default="PASS_ATLAS_FEATURE_ROW",
    )
    panel["feature_timestamp"] = panel["date"] + pd.Timedelta(hours=18)
    panel["historical_weight_provenance"] = (
        "RETROSPECTIVE_MONTHLY_SNAPSHOT_NOT_STRICT_PIT_OFFICIAL_ARCHIVE"
    )
    prohibited = [
        column
        for column in panel.columns
        if column.lower().startswith(("future_", "forward_"))
        or "target" in column.lower()
    ]
    if prohibited:
        raise ContractError(f"点时特征面板混入未来字段：{prohibited}")
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ContractError("点时特征面板日期不唯一或未递增")
    return panel.reset_index(drop=True)


def build_label_a_events(
    config: dict[str, Any],
    h00300: pd.DataFrame,
) -> pd.DataFrame:
    """按动态波动阈值建立方向变化事件及确认滞后。"""

    specification = config["label_a_directional_change"]
    market = h00300.loc[
        h00300["date"].between(
            pd.Timestamp(config["dates"]["label_history_start"]),
            pd.Timestamp(config["dates"]["label_market_cutoff"]),
        ),
        ["date", "close"],
    ].copy()
    market.sort_values("date", inplace=True)
    market.reset_index(drop=True, inplace=True)
    log_close = np.log(market["close"].astype(float))
    volatility = (
        log_close.diff()
        .rolling(
            int(specification["volatility_lookback_rows"]),
            min_periods=int(specification["minimum_volatility_rows"]),
        )
        .std(ddof=1)
        .shift(1)
    )
    threshold = (
        volatility
        * math.sqrt(int(specification["reference_horizon_rows"]))
        * float(specification["sigma_multiplier"])
    ).clip(
        lower=float(specification["minimum_log_threshold"]),
        upper=float(specification["maximum_log_threshold"]),
    )
    first_valid = threshold.first_valid_index()
    columns = [
        "event_id",
        "label_system",
        "event_type",
        "event_date",
        "confirmation_date",
        "event_position",
        "confirmation_position",
        "confirmation_delay_rows",
        "confirmation_threshold_log",
    ]
    if first_valid is None:
        return pd.DataFrame(columns=columns)

    state = "UNKNOWN"
    running_low_position = int(first_valid)
    running_high_position = int(first_valid)
    events: list[dict[str, Any]] = []

    def append_transition(
        event_position: int,
        confirmation_position: int,
        first_type: str,
        second_type: str,
        used_threshold: float,
    ) -> None:
        event_date = market.loc[event_position, "date"]
        confirmation_date = market.loc[confirmation_position, "date"]
        for event_type in [first_type, second_type]:
            events.append(
                {
                    "event_id": f"A_{event_type}_{event_date:%Y%m%d}",
                    "label_system": LABEL_A,
                    "event_type": event_type,
                    "event_date": event_date,
                    "confirmation_date": confirmation_date,
                    "event_position": event_position,
                    "confirmation_position": confirmation_position,
                    "confirmation_delay_rows": confirmation_position - event_position,
                    "confirmation_threshold_log": used_threshold,
                }
            )

    for position in range(int(first_valid) + 1, len(market)):
        current_threshold = threshold.iloc[position]
        if not np.isfinite(current_threshold):
            continue
        value = float(log_close.iloc[position])
        if value < float(log_close.iloc[running_low_position]):
            running_low_position = position
        if value > float(log_close.iloc[running_high_position]):
            running_high_position = position

        if state == "UNKNOWN":
            upward_move = value - float(log_close.iloc[running_low_position])
            downward_move = float(log_close.iloc[running_high_position]) - value
            if upward_move >= current_threshold:
                append_transition(
                    running_low_position,
                    position,
                    "DOWN_END",
                    "UP_START",
                    float(current_threshold),
                )
                state = "UP"
                running_high_position = position
            elif downward_move >= current_threshold:
                append_transition(
                    running_high_position,
                    position,
                    "UP_END",
                    "DOWN_START",
                    float(current_threshold),
                )
                state = "DOWN"
                running_low_position = position
        elif state == "UP":
            if value > float(log_close.iloc[running_high_position]):
                running_high_position = position
            drawdown = float(log_close.iloc[running_high_position]) - value
            if drawdown >= current_threshold:
                append_transition(
                    running_high_position,
                    position,
                    "UP_END",
                    "DOWN_START",
                    float(current_threshold),
                )
                state = "DOWN"
                running_low_position = position
        else:
            if value < float(log_close.iloc[running_low_position]):
                running_low_position = position
            rebound = value - float(log_close.iloc[running_low_position])
            if rebound >= current_threshold:
                append_transition(
                    running_low_position,
                    position,
                    "DOWN_END",
                    "UP_START",
                    float(current_threshold),
                )
                state = "UP"
                running_high_position = position

    result = pd.DataFrame(events, columns=columns)
    if result.empty:
        return result
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    signal_cutoff = pd.Timestamp(config["dates"]["signal_cutoff"])
    result = result.loc[
        result["event_date"].between(evaluation_start, signal_cutoff)
    ].copy()
    result.sort_values(["event_date", "event_type"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def build_outcome_label_panel(
    config: dict[str, Any],
    h00300: pd.DataFrame,
) -> pd.DataFrame:
    """冻结后生成与特征物理分离的20/60日未来路径标签。"""

    specification = config["label_b_future_path"]
    market = h00300.loc[
        h00300["date"].between(
            pd.Timestamp(config["dates"]["label_history_start"]),
            pd.Timestamp(config["dates"]["label_market_cutoff"]),
        ),
        ["date", "close"],
    ].copy()
    market.sort_values("date", inplace=True)
    market.reset_index(drop=True, inplace=True)
    log_close = np.log(market["close"].astype(float))
    daily_abs = log_close.diff().abs()
    volatility = (
        log_close.diff()
        .rolling(
            int(specification["volatility_lookback_rows"]),
            min_periods=int(specification["minimum_volatility_rows"]),
        )
        .std(ddof=1)
    )
    result = pd.DataFrame({"date": market["date"]})
    for horizon in specification["horizons_trading_days"]:
        horizon = int(horizon)
        forward_log_return = log_close.shift(-horizon) - log_close
        result[f"future_log_return_{horizon}d"] = forward_log_return
        result[f"future_return_z_{horizon}d"] = forward_log_return / (
            volatility * math.sqrt(horizon)
        )
        result[f"outcome_complete_{horizon}d"] = forward_log_return.notna()

    horizon = 60
    shifted_abs = daily_abs.shift(-1)
    future_path_length = (
        shifted_abs.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )
    result["future_signed_efficiency_60d"] = result[
        "future_log_return_60d"
    ] / future_path_length.replace(0.0, np.nan)
    mfe = np.full(len(market), np.nan, dtype=float)
    mae = np.full(len(market), np.nan, dtype=float)
    values = log_close.to_numpy(dtype=float)
    for position in range(len(values) - horizon):
        path = values[position + 1 : position + horizon + 1] - values[position]
        mfe[position] = max(0.0, float(np.max(path)))
        mae[position] = max(0.0, float(-np.min(path)))
    result["future_mfe_log_60d"] = mfe
    result["future_mae_log_60d"] = mae
    result["future_mfe_to_mae_ratio_60d"] = np.divide(
        mfe,
        mae,
        out=np.full(len(mfe), np.inf),
        where=mae > 1e-12,
    )
    result["future_mae_to_mfe_ratio_60d"] = np.divide(
        mae,
        mfe,
        out=np.full(len(mae), np.inf),
        where=mfe > 1e-12,
    )
    up = specification["sustained_up"]
    down = specification["sustained_down"]
    range_spec = specification["low_trend_range"]
    complete = result["outcome_complete_60d"].astype(bool)
    up_mask = (
        complete
        & result["future_return_z_20d"].ge(up["minimum_z20"])
        & result["future_return_z_60d"].ge(up["minimum_z60"])
        & result["future_signed_efficiency_60d"].ge(
            up["minimum_signed_efficiency60"]
        )
        & result["future_mfe_to_mae_ratio_60d"].ge(
            up["minimum_mfe_to_mae_ratio"]
        )
    )
    down_mask = (
        complete
        & result["future_return_z_20d"].le(down["maximum_z20"])
        & result["future_return_z_60d"].le(down["maximum_z60"])
        & result["future_signed_efficiency_60d"].le(
            down["maximum_signed_efficiency60"]
        )
        & result["future_mae_to_mfe_ratio_60d"].ge(
            down["minimum_mae_to_mfe_ratio"]
        )
    )
    range_mask = (
        complete
        & result["future_return_z_20d"].abs().le(
            range_spec["maximum_absolute_z20"]
        )
        & result["future_return_z_60d"].abs().le(
            range_spec["maximum_absolute_z60"]
        )
        & result["future_signed_efficiency_60d"].abs().le(
            range_spec["maximum_absolute_signed_efficiency60"]
        )
    )
    result["future_path_label"] = np.select(
        [~complete, up_mask, down_mask, range_mask],
        [
            specification["incomplete_horizon_label"],
            "SUSTAINED_UP",
            "SUSTAINED_DOWN",
            "LOW_TREND_RANGE",
        ],
        default=specification["unmatched_label"],
    )
    return result


def _bridge_short_false_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = mask.astype(bool).copy()
    position = 0
    while position < len(result):
        if result[position]:
            position += 1
            continue
        start = position
        while position < len(result) and not result[position]:
            position += 1
        end = position - 1
        gap_length = end - start + 1
        if (
            start > 0
            and position < len(result)
            and result[start - 1]
            and result[position]
            and gap_length <= maximum_gap
        ):
            result[start : end + 1] = True
    return result


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    position = 0
    while position < len(mask):
        if not mask[position]:
            position += 1
            continue
        start = position
        while position + 1 < len(mask) and mask[position + 1]:
            position += 1
        runs.append((start, position))
        position += 1
    return runs


def build_label_b_events(
    config: dict[str, Any],
    outcome_panel: pd.DataFrame,
) -> pd.DataFrame:
    """把未来路径日标签压缩为等权独立事件区间。"""

    specification = config["label_b_future_path"]
    minimum_rows = int(specification["minimum_episode_rows"])
    maximum_gap = int(specification["maximum_internal_gap_rows"])
    cooldown = int(specification["same_transition_cooldown_rows"])
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    signal_cutoff = pd.Timestamp(config["dates"]["signal_cutoff"])
    mappings = {
        "SUSTAINED_UP": ("UP_START", "UP_END"),
        "SUSTAINED_DOWN": ("DOWN_START", "DOWN_END"),
        "LOW_TREND_RANGE": ("RANGE_START", "RANGE_END"),
    }
    events: list[dict[str, Any]] = []
    for path_label, event_pair in mappings.items():
        raw_mask = outcome_panel["future_path_label"].eq(path_label).to_numpy()
        bridged = _bridge_short_false_gaps(raw_mask, maximum_gap)
        last_start = -10_000
        for start, end in _true_runs(bridged):
            if end - start + 1 < minimum_rows or start - last_start < cooldown:
                continue
            last_start = start
            episode_id = f"B_{path_label}_{outcome_panel.loc[start, 'date']:%Y%m%d}"
            for event_type, position in zip(event_pair, [start, end], strict=True):
                event_date = outcome_panel.loc[position, "date"]
                if not evaluation_start <= event_date <= signal_cutoff:
                    continue
                events.append(
                    {
                        "event_id": f"{episode_id}_{event_type}",
                        "episode_id": episode_id,
                        "label_system": LABEL_B,
                        "path_label": path_label,
                        "event_type": event_type,
                        "event_date": event_date,
                        "episode_start_date": outcome_panel.loc[start, "date"],
                        "episode_end_date": outcome_panel.loc[end, "date"],
                        "episode_rows": end - start + 1,
                        "event_position": position,
                    }
                )
    result = pd.DataFrame(events)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "episode_id",
                "label_system",
                "path_label",
                "event_type",
                "event_date",
                "episode_start_date",
                "episode_end_date",
                "episode_rows",
                "event_position",
            ]
        )
    result.sort_values(["event_date", "event_type"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def build_event_catalog(
    config: dict[str, Any],
    label_a_events: pd.DataFrame,
    label_b_events: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    """统一两套标签并标记完整事件窗口，不因结果删除事件。"""

    common_columns = ["event_id", "label_system", "event_type", "event_date"]
    frames = [label_a_events[common_columns].copy()]
    if not label_b_events.empty:
        frames.append(label_b_events[common_columns].copy())
    catalog = pd.concat(frames, ignore_index=True)
    catalog["event_date"] = pd.to_datetime(catalog["event_date"]).dt.normalize()
    positions = {date: index for index, date in enumerate(feature_panel["date"])}
    start = int(config["event_alignment"]["full_window_start"])
    end = int(config["event_alignment"]["full_window_end"])
    catalog["feature_position"] = catalog["event_date"].map(positions)
    catalog["event_on_feature_calendar"] = catalog["feature_position"].notna()
    catalog["full_window_available"] = catalog["feature_position"].map(
        lambda value: bool(
            pd.notna(value)
            and int(value) + start >= 0
            and int(value) + end < len(feature_panel)
        )
    )
    valid_by_date = feature_panel.set_index("date")["valid_for_atlas"]
    catalog["valid_feature_row_at_event"] = (
        catalog["event_date"].map(valid_by_date).fillna(False).astype(bool)
    )
    early_end = pd.Timestamp(config["dates"]["early_period"][1])
    catalog["era"] = np.where(
        catalog["event_date"].le(early_end), "EARLY", "LATE"
    )
    catalog.sort_values(["event_date", "label_system", "event_type"], inplace=True)
    catalog.reset_index(drop=True, inplace=True)
    if catalog["event_id"].duplicated().any():
        raise ContractError("双标签事件ID不唯一")
    return catalog


def _feature_module_map(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for module, features in config["feature_construction"]["feature_modules"].items():
        for feature in features:
            if feature in result:
                raise ContractError(f"变量重复归属机制模块：{feature}")
            result[feature] = module
    return result


def build_event_trajectories(
    config: dict[str, Any],
    feature_panel: pd.DataFrame,
    event_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """按事件自身基线构建长表轨迹，保留缺失和删失。"""

    alignment = config["event_alignment"]
    start = int(alignment["full_window_start"])
    end = int(alignment["full_window_end"])
    baseline_start, baseline_end = [int(value) for value in alignment["baseline_window"]]
    minimum_baseline = int(alignment["minimum_baseline_observations"])
    module_map = _feature_module_map(config)
    features = list(module_map)
    rows: list[dict[str, Any]] = []
    eligible_catalog = event_catalog.loc[event_catalog["full_window_available"]]
    for event in eligible_catalog.itertuples(index=False):
        position = int(event.feature_position)
        window = feature_panel.iloc[position + start : position + end + 1]
        relative_days = np.arange(start, end + 1, dtype=int)
        for feature in features:
            values = pd.to_numeric(window[feature], errors="coerce").to_numpy(dtype=float)
            baseline_mask = (relative_days >= baseline_start) & (
                relative_days <= baseline_end
            )
            baseline_values = values[baseline_mask]
            baseline_values = baseline_values[np.isfinite(baseline_values)]
            if len(baseline_values) >= minimum_baseline:
                baseline_median = float(np.median(baseline_values))
                baseline_mad = float(
                    np.median(np.abs(baseline_values - baseline_median))
                )
                baseline_scale = 1.4826 * baseline_mad
                if baseline_scale <= 1e-12:
                    baseline_scale = np.nan
            else:
                baseline_median = np.nan
                baseline_scale = np.nan
            z_values = (values - baseline_median) / baseline_scale
            for relative_day, raw_value, z_value in zip(
                relative_days, values, z_values, strict=True
            ):
                rows.append(
                    {
                        "event_id": event.event_id,
                        "label_system": event.label_system,
                        "event_type": event.event_type,
                        "event_date": event.event_date,
                        "era": event.era,
                        "relative_trading_day": int(relative_day),
                        "feature": feature,
                        "module": module_map[feature],
                        "raw_value": _safe_float(raw_value),
                        "baseline_median": _safe_float(baseline_median),
                        "baseline_scale": _safe_float(baseline_scale),
                        "event_z": _safe_float(z_value),
                        "feature_available": bool(np.isfinite(z_value)),
                    }
                )
    return pd.DataFrame(rows)


def build_trajectory_summary(
    config: dict[str, Any],
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """汇总六个预先指定相对日的中位数与四分位轨迹。"""

    if trajectories.empty:
        return pd.DataFrame(
            columns=[
                "label_system",
                "event_type",
                "feature",
                "module",
                "relative_trading_day",
                "observations",
                "median_event_z",
                "q25_event_z",
                "q75_event_z",
            ]
        )
    selected_days = set(config["event_alignment"]["event_periods_to_report"])
    selected = trajectories.loc[
        trajectories["relative_trading_day"].isin(selected_days)
        & trajectories["event_z"].notna()
    ]
    summary = (
        selected.groupby(
            [
                "label_system",
                "event_type",
                "feature",
                "module",
                "relative_trading_day",
            ],
            dropna=False,
        )["event_z"]
        .agg(
            observations="count",
            median_event_z="median",
            q25_event_z=lambda values: values.quantile(0.25),
            q75_event_z=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    return summary


def _expected_direction_map(config: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for family in config["expected_directions"].values():
        for feature in family["features"]:
            for event_type in EVENT_TYPES:
                result[(feature, event_type)] = family[event_type]
    return result


def _first_onset(
    event_trajectory: pd.DataFrame,
    *,
    sign: float,
    search_start: int,
    search_end: int,
    threshold: float,
    consecutive_rows: int,
) -> float | None:
    selected = event_trajectory.loc[
        event_trajectory["relative_trading_day"].between(search_start, search_end)
    ].sort_values("relative_trading_day")
    values = pd.to_numeric(selected["event_z"], errors="coerce").to_numpy(dtype=float)
    days = selected["relative_trading_day"].to_numpy(dtype=int)
    condition = np.isfinite(values) & (sign * values >= threshold)
    for position in range(0, len(condition) - consecutive_rows + 1):
        if condition[position : position + consecutive_rows].all():
            return float(days[position])
    return None


def evaluate_leading_results(
    config: dict[str, Any],
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """对每个标签、转移和变量执行冻结的事件等权领先门。"""

    if trajectories.empty:
        return pd.DataFrame(columns=list(LeadingResult.__dataclass_fields__))
    alignment = config["event_alignment"]
    gates = config["leading_gates"]
    expected_map = _expected_direction_map(config)
    pre_start, pre_end = [int(value) for value in alignment["pre_event_window"]]
    search_start, search_end = [
        int(value) for value in alignment["leading_search_window"]
    ]
    rows: list[dict[str, Any]] = []
    grouped = trajectories.loc[trajectories["event_type"].isin(EVENT_TYPES)].groupby(
        ["label_system", "event_type", "feature", "module"], sort=True
    )
    for (label_system, event_type, feature, module), group in grouped:
        direction = expected_map.get((feature, event_type))
        if direction is None:
            continue
        sign = 1.0 if direction == "positive" else -1.0
        event_rows: list[dict[str, Any]] = []
        for event_id, event_group in group.groupby("event_id", sort=False):
            pre_values = pd.to_numeric(
                event_group.loc[
                    event_group["relative_trading_day"].between(pre_start, pre_end),
                    "event_z",
                ],
                errors="coerce",
            ).dropna()
            if pre_values.empty:
                continue
            pre_median = float(pre_values.median())
            onset = _first_onset(
                event_group,
                sign=sign,
                search_start=search_start,
                search_end=search_end,
                threshold=float(alignment["onset_z_threshold"]),
                consecutive_rows=int(alignment["onset_consecutive_rows"]),
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "era": str(event_group["era"].iloc[0]),
                    "pre_median_z": pre_median,
                    "expected_z": sign * pre_median,
                    "onset": onset,
                }
            )
        event_frame = pd.DataFrame(event_rows)
        observations = int(len(event_frame))
        positives = (
            int(event_frame["expected_z"].gt(0.0).sum()) if observations else 0
        )
        direction_fraction = positives / observations if observations else None
        sign_pvalue = (
            float(binomtest(positives, observations, 0.5, alternative="greater").pvalue)
            if observations
            else None
        )
        median_pre = (
            float(event_frame["pre_median_z"].median()) if observations else None
        )
        median_expected = (
            float(event_frame["expected_z"].median()) if observations else None
        )
        onset_values = (
            event_frame["onset"].dropna() if observations else pd.Series(dtype=float)
        )
        median_onset = float(onset_values.median()) if len(onset_values) else None
        absolute = (
            event_frame["expected_z"].abs() if observations else pd.Series(dtype=float)
        )
        absolute_total = float(absolute.sum()) if observations else 0.0
        dominance = (
            float(absolute.max() / absolute_total)
            if absolute_total > 0.0
            else None
        )
        early = (
            event_frame.loc[event_frame["era"].eq("EARLY"), "expected_z"]
            if observations
            else pd.Series(dtype=float)
        )
        late = (
            event_frame.loc[event_frame["era"].eq("LATE"), "expected_z"]
            if observations
            else pd.Series(dtype=float)
        )
        early_median = float(early.median()) if len(early) else None
        late_median = float(late.median()) if len(late) else None
        period_minimum = int(gates["early_late_direction_must_agree_when_each_has_at_least"])
        if len(early) >= period_minimum and len(late) >= period_minimum:
            early_late_agrees = bool(early_median > 0.0 and late_median > 0.0)
        else:
            early_late_agrees = True
        passed = bool(
            observations >= int(gates["minimum_feature_event_observations"])
            and direction_fraction is not None
            and direction_fraction >= float(gates["minimum_direction_fraction"])
            and sign_pvalue is not None
            and sign_pvalue <= float(gates["maximum_one_sided_sign_pvalue"])
            and median_expected is not None
            and median_expected
            >= float(gates["minimum_absolute_median_pre_event_z"])
            and median_onset is not None
            and median_onset
            <= float(gates["latest_allowed_median_onset_relative_day"])
            and dominance is not None
            and dominance <= float(gates["maximum_single_event_absolute_share"])
            and early_late_agrees
        )
        rows.append(
            LeadingResult(
                label_system=str(label_system),
                event_type=str(event_type),
                feature=str(feature),
                module=str(module),
                expected_direction=direction,
                event_observations=observations,
                positive_direction_count=positives,
                direction_fraction=direction_fraction,
                one_sided_sign_pvalue=sign_pvalue,
                median_pre_event_z=median_pre,
                median_expected_direction_z=median_expected,
                median_onset_relative_day=median_onset,
                onset_event_count=int(len(onset_values)),
                maximum_single_event_absolute_share=dominance,
                early_event_count=int(len(early)),
                late_event_count=int(len(late)),
                early_median_expected_direction_z=early_median,
                late_median_expected_direction_z=late_median,
                early_late_direction_agrees=early_late_agrees,
                passed=passed,
            ).as_dict()
        )
    return pd.DataFrame(rows)


def build_dual_label_candidates(
    config: dict[str, Any],
    leading_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只保留两套标签同向通过且满足权重敏感性的变量。"""

    columns = [
        "event_type",
        "feature",
        "module",
        "label_a_passed",
        "label_b_passed",
        "equal_weight_sensitivity_feature",
        "equal_weight_sensitivity_passed",
        "dual_label_candidate_passed",
    ]
    if leading_results.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, {
            "transition_gates": {},
            "all_required_transitions_passed": False,
        }
    indexed = leading_results.set_index(["label_system", "event_type", "feature"])
    rows: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        features = sorted(
            leading_results.loc[
                leading_results["event_type"].eq(event_type), "feature"
            ].unique()
        )
        for feature in features:
            def passed(label: str, candidate_feature: str) -> bool:
                key = (label, event_type, candidate_feature)
                if key not in indexed.index:
                    return False
                value = indexed.loc[key, "passed"]
                if isinstance(value, pd.Series):
                    return bool(value.all())
                return bool(value)

            label_a_pass = passed(LABEL_A, feature)
            label_b_pass = passed(LABEL_B, feature)
            sensitivity_feature: str | None = None
            sensitivity_pass = True
            if (
                config["leading_gates"][
                    "snapshot_weight_feature_requires_equal_weight_sensitivity"
                ]
                and "_snapshot_weighted" in feature
            ):
                sensitivity_feature = feature.replace(
                    "_snapshot_weighted", "_equal_weighted"
                )
                sensitivity_pass = passed(LABEL_A, sensitivity_feature) and passed(
                    LABEL_B, sensitivity_feature
                )
            module_rows = leading_results.loc[
                leading_results["feature"].eq(feature), "module"
            ]
            module = str(module_rows.iloc[0]) if not module_rows.empty else "UNKNOWN"
            rows.append(
                {
                    "event_type": event_type,
                    "feature": feature,
                    "module": module,
                    "label_a_passed": label_a_pass,
                    "label_b_passed": label_b_pass,
                    "equal_weight_sensitivity_feature": sensitivity_feature,
                    "equal_weight_sensitivity_passed": sensitivity_pass,
                    "dual_label_candidate_passed": bool(
                        label_a_pass and label_b_pass and sensitivity_pass
                    ),
                }
            )
    candidates = pd.DataFrame(rows, columns=columns)
    transition_gates: dict[str, Any] = {}
    for event_type in config["leading_gates"]["required_dual_label_transitions"]:
        selected = candidates.loc[
            candidates["event_type"].eq(event_type)
            & candidates["dual_label_candidate_passed"]
        ]
        feature_count = int(len(selected))
        module_count = int(selected["module"].nunique())
        passed = bool(
            feature_count
            >= int(config["leading_gates"]["minimum_dual_label_features_per_transition"])
            and module_count
            >= int(config["leading_gates"]["minimum_distinct_modules_per_transition"])
        )
        transition_gates[event_type] = {
            "feature_count": feature_count,
            "module_count": module_count,
            "features": selected["feature"].tolist(),
            "modules": sorted(selected["module"].unique().tolist()),
            "passed": passed,
        }
    return candidates, {
        "transition_gates": transition_gates,
        "all_required_transitions_passed": bool(
            transition_gates and all(item["passed"] for item in transition_gates.values())
        ),
    }


def build_lifecycle_reference_panel(
    config: dict[str, Any],
    feature_panel: pd.DataFrame,
    outcome_panel: pd.DataFrame,
    label_a_events: pd.DataFrame,
) -> pd.DataFrame:
    """生成明确标注为事后解释用途的七状态参考图。"""

    result = feature_panel[["date", "valid_for_atlas", "data_status"]].copy()
    result = result.merge(
        outcome_panel[["date", "future_path_label"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    stress_features = [
        "contribution_hhi20",
        "average_pairwise_correlation20",
        "downside_vol_ratio5_20",
        "extreme_down_weight",
        "liquidity_impact_pressure",
        "cgb_10y_change63_pressure",
    ]
    stress_percentiles: list[pd.Series] = []
    for feature in stress_features:
        values = feature_panel[feature]
        if feature.endswith("_pressure"):
            percentile = pd.to_numeric(values, errors="coerce")
        else:
            percentile = causal_percentile(
                values,
                prior_rows=config["feature_construction"]["causal_percentile_prior_rows"],
                minimum_prior_rows=config["feature_construction"][
                    "causal_percentile_minimum_prior_rows"
                ],
            )
        stress_percentiles.append(percentile.rename(feature))
    stress_frame = pd.concat(stress_percentiles, axis=1)
    result["pressure_composite_percentile"] = stress_frame.mean(
        axis=1, skipna=True
    ).where(stress_frame.notna().sum(axis=1).ge(4))
    result["breadth_above_ma60_snapshot_weighted"] = feature_panel[
        "breadth_above_ma60_snapshot_weighted"
    ]
    result["downside_vol_ratio5_20"] = feature_panel["downside_vol_ratio5_20"]
    result["reference_state"] = config["reference_state_map"]["fallback_state"]
    reference = config["reference_state_map"]
    range_mask = (
        result["future_path_label"].eq("LOW_TREND_RANGE")
        & result["pressure_composite_percentile"].le(
            reference["R_LOW_STRESS_RANGE"]["pressure_percentile_at_most"]
        )
    )
    result.loc[range_mask, "reference_state"] = "R_LOW_STRESS_RANGE"
    mature_up_mask = (
        result["future_path_label"].eq("SUSTAINED_UP")
        & result["breadth_above_ma60_snapshot_weighted"].ge(
            reference["U1_BROAD_UPTREND"]["breadth_above_ma60_at_least"]
        )
        & result["downside_vol_ratio5_20"].lt(
            reference["U1_BROAD_UPTREND"]["downside_vol_ratio_below"]
        )
    )
    result.loc[mature_up_mask, "reference_state"] = "U1_BROAD_UPTREND"
    positions = {date: index for index, date in enumerate(result["date"])}

    def paint(event_type: str, state: str, relative_window: list[int], pressure: bool = False) -> None:
        start, end = [int(value) for value in relative_window]
        for event_date in label_a_events.loc[
            label_a_events["event_type"].eq(event_type), "event_date"
        ]:
            position = positions.get(pd.Timestamp(event_date))
            if position is None:
                continue
            left = max(0, position + start)
            right = min(len(result) - 1, position + end)
            mask = pd.Series(False, index=result.index)
            mask.iloc[left : right + 1] = True
            if pressure:
                mask &= result["pressure_composite_percentile"].ge(
                    reference[state]["pressure_percentile_at_least"]
                )
            result.loc[mask, "reference_state"] = state

    # 后写入的状态优先级更高：D1 > D0 > U2 > U0 > U1 > R > X。
    paint("UP_START", "U0_EARLY_UP", reference["U0_EARLY_UP"]["relative_window"])
    paint("UP_END", "U2_FRAGILE_LATE_UP", reference["U2_FRAGILE_LATE_UP"]["relative_window"])
    paint("DOWN_START", "D0_EARLY_DOWN", reference["D0_EARLY_DOWN"]["relative_window"])
    paint(
        "DOWN_START",
        "D1_MARKDOWN_PANIC",
        reference["D1_MARKDOWN_PANIC"]["relative_window"],
        pressure=True,
    )
    result.loc[~result["valid_for_atlas"], "reference_state"] = (
        "X_TRANSITION_NO_VIEW"
    )
    result["reference_semantics"] = reference["semantics"]
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.date().isoformat()
            elif isinstance(value, (np.bool_, bool)):
                converted[key] = bool(value)
            elif isinstance(value, (np.integer,)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                converted[key] = _safe_float(value)
            elif value is None or pd.isna(value):
                converted[key] = None
            else:
                converted[key] = value
        records.append(converted)
    return records


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


def _format_number(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """渲染面向审阅者的结论报告。"""

    lines = [
        "# 510300 趋势生命周期图谱 V1",
        "",
        f"- 最终状态：`{report['status']}`",
        f"- 数据合同：`{'PASS' if report['data_admission']['passed'] else 'FAIL'}`",
        "- 证据等级：回顾性、历史污染的双标签事件图谱；不是策略回测。",
        "- 历史权重来源门：`BLOCKED_NO_VERSION_PROVEN_STRICT_PIT_OFFICIAL_ARCHIVE`。",
        "- 收益评价：`NOT_ALLOWED_PHASE_1_EVENT_LABEL_DIAGNOSTICS_ONLY`。",
        "",
    ]
    if not report["data_admission"]["passed"]:
        lines.extend(
            [
                "## 数据门停止",
                "",
                "输入合同未通过；未生成新未来路径标签、事件轨迹或任何交易结论。",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## 双标签事件样本",
            "",
            "| 标签 | 转移 | 全部事件 | 完整[-120,+60]事件 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["events"]["counts"]:
        lines.append(
            f"| `{row['label_system']}` | `{row['event_type']}` | {row['event_count']} | {row['full_window_event_count']} |"
        )
    lines.extend(
        [
            "",
            "## 双标签领先结构门",
            "",
            "| 转移 | 双标签变量数 | 机制模块数 | 通过 |",
            "|---|---:|---:|---:|",
        ]
    )
    for event_type, gate in report["structure_gate"]["transition_gates"].items():
        lines.append(
            f"| `{event_type}` | {gate['feature_count']} | {gate['module_count']} | {'是' if gate['passed'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 通过单标签门的主要变量",
            "",
            "| 标签 | 转移 | 变量 | 模块 | 事件数 | 方向一致率 | 事前中位Z | 中位领先日 |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    passed_rows = report["leading_summary"]["passed_rows"]
    if passed_rows:
        for row in passed_rows[:30]:
            lines.append(
                f"| `{row['label_system']}` | `{row['event_type']}` | `{row['feature']}` | `{row['module']}` | "
                f"{row['event_observations']} | {_format_number(row['direction_fraction'])} | "
                f"{_format_number(row['median_expected_direction_z'])} | {_format_number(row['median_onset_relative_day'], 1)} |"
            )
    else:
        lines.append("| 无 | 无 | 无变量通过全部冻结条件 | 无 | 0 | NA | NA | NA |")
    lines.extend(
        [
            "",
            "## 七状态事后参考图",
            "",
            "| 参考状态 | 交易日 |",
            "|---|---:|",
        ]
    )
    for state, count in report["reference_states"]["counts"].items():
        lines.append(f"| `{state}` | {count} |")
    lines.extend(
        [
            "",
            "## 裁决边界",
            "",
            f"- 双标签结构门：`{'PASS' if report['structure_gate']['all_required_transitions_passed'] else 'FAIL'}`。",
            "- 历史成员/权重严格点时来源门：`FAIL`；当前结果不得直接进入路由器冻结。",
            "- H00300仅用于标签；未来若有策略，510300未复权执行、分红现金和T+1时钟仍需在新协议中另行冻结。",
            "- 未计算策略净值、净夏普、UP_CAPTURE、DOWN_CAPTURE、最大回撤或RANGE_T净边际。",
            "- 未生成当前状态概率、当前因子、Paper、Shadow、仓位、订单、券商连接或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(*, write: bool = True) -> dict[str, Any]:
    """执行冻结后的完整第一阶段图谱。"""

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
            "version": config["protocol"]["version"],
            "status": config["adjudication"]["if_data_gate_fails"],
            "generated_at_asia_shanghai": generated_at,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "data_admission": data_audit,
            "return_evaluation": "NOT_ALLOWED",
            "strategy_backtest": "NOT_RUN",
            "boundaries": {
                "new_lifecycle_future_labels_read": False,
                "position_mapping": "DISABLED",
                "live_trading_authorized": False,
            },
        }
        if write:
            _atomic_json(paths["result_json"], report)
            _atomic_text(paths["result_markdown"], render_markdown(report))
        return report

    feature_panel = build_feature_panel(config, inputs)
    label_a_events = build_label_a_events(config, inputs["h00300"])
    outcome_panel = build_outcome_label_panel(config, inputs["h00300"])
    label_b_events = build_label_b_events(config, outcome_panel)
    event_catalog = build_event_catalog(
        config, label_a_events, label_b_events, feature_panel
    )
    trajectories = build_event_trajectories(config, feature_panel, event_catalog)
    trajectory_summary = build_trajectory_summary(config, trajectories)
    leading_results = evaluate_leading_results(config, trajectories)
    dual_candidates, structure_gate = build_dual_label_candidates(
        config, leading_results
    )
    reference_panel = build_lifecycle_reference_panel(
        config, feature_panel, outcome_panel, label_a_events
    )
    source_gate = bool(
        config["data_gates"]["source_provenance_gate_for_phase_2"]
    )
    if not structure_gate["all_required_transitions_passed"]:
        status = config["adjudication"]["if_structure_fails"]
    elif not source_gate:
        status = config["adjudication"]["if_structure_passes_but_source_fails"]
    else:
        status = config["adjudication"]["if_structure_and_source_pass"]

    count_frame = (
        event_catalog.groupby(["label_system", "event_type"])
        .agg(
            event_count=("event_id", "count"),
            full_window_event_count=("full_window_available", "sum"),
        )
        .reset_index()
    )
    passed_leading = leading_results.loc[leading_results["passed"]].copy()
    passed_leading.sort_values(
        ["event_type", "label_system", "median_expected_direction_z"],
        ascending=[True, True, False],
        inplace=True,
    )
    state_counts = {
        str(key): int(value)
        for key, value in reference_panel["reference_state"].value_counts().items()
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
        "feature_panel": {
            "rows": int(len(feature_panel)),
            "first_date": feature_panel["date"].min().date().isoformat(),
            "last_date": feature_panel["date"].max().date().isoformat(),
            "formal_evaluation_rows": int(feature_panel["formal_evaluation_row"].sum()),
            "valid_atlas_rows": int(feature_panel["valid_for_atlas"].sum()),
            "no_view_rows": int(
                feature_panel["formal_evaluation_row"].sum()
                - feature_panel["valid_for_atlas"].sum()
            ),
            "future_columns": [],
        },
        "events": {
            "label_a_count": int(len(label_a_events)),
            "label_b_count": int(len(label_b_events)),
            "catalog_count": int(len(event_catalog)),
            "full_window_count": int(event_catalog["full_window_available"].sum()),
            "counts": _json_records(count_frame),
            "event_weighting": "ONE_EVENT_ONE_WEIGHT",
        },
        "leading_summary": {
            "evaluated_rows": int(len(leading_results)),
            "passed_row_count": int(leading_results["passed"].sum()),
            "passed_rows": _json_records(passed_leading),
        },
        "dual_label_candidates": {
            "passed_count": int(dual_candidates["dual_label_candidate_passed"].sum()),
            "passed_rows": _json_records(
                dual_candidates.loc[dual_candidates["dual_label_candidate_passed"]]
            ),
        },
        "structure_gate": structure_gate,
        "source_provenance_gate_for_phase_2": {
            "passed": source_gate,
            "status": "BLOCKED_NO_VERSION_PROVEN_STRICT_PIT_OFFICIAL_ARCHIVE",
        },
        "reference_states": {
            "semantics": config["reference_state_map"]["semantics"],
            "counts": state_counts,
        },
        "known_data_limits": {
            **config["feature_construction"]["unavailable_slow_features"],
            "historical_weights": "BLOCKED_NO_VERSION_PROVEN_STRICT_PIT_OFFICIAL_ARCHIVE",
        },
        "return_evaluation": config["adjudication"]["return_evaluation"],
        "strategy_backtest": "NOT_RUN",
        "portfolio_metrics": {
            "net_sharpe": "NOT_COMPUTED",
            "up_capture": "NOT_COMPUTED",
            "down_capture": "NOT_COMPUTED",
            "maximum_drawdown": "NOT_COMPUTED",
            "range_t_net_edge": "NOT_COMPUTED",
        },
        "boundaries": {
            "old_three_state_router_modified": False,
            "old_regime_router_modified": False,
            "old_stress_atlas_modified": False,
            "old_range_t_rescued": False,
            "pcf_iopv_used_as_trend_factor": False,
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
        _atomic_parquet(paths["feature_panel"], feature_panel)
        _atomic_parquet(paths["outcome_label_panel"], outcome_panel)
        _atomic_csv(paths["label_a_events"], label_a_events)
        _atomic_csv(paths["label_b_events"], label_b_events)
        _atomic_csv(paths["event_catalog"], event_catalog)
        _atomic_parquet(paths["event_trajectories"], trajectories)
        _atomic_csv(paths["trajectory_summary"], trajectory_summary)
        _atomic_csv(paths["leading_results"], leading_results)
        _atomic_csv(paths["dual_label_candidates"], dual_candidates)
        _atomic_parquet(paths["lifecycle_reference_panel"], reference_panel)
        _atomic_json(paths["result_json"], report)
        _atomic_text(paths["result_markdown"], render_markdown(report))
    return report


def main() -> int:
    report = run_study(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
