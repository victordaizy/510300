"""510300机制图谱V1。

本模块只构建点时状态面板、病例/失败对照、传导顺序和连续条件响应。
它不生成策略净值、仓位、订单、券商连接或实盘信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_mechanism_atlas_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_mechanism_atlas_v1_manifest.json"


class ContractError(RuntimeError):
    """冻结协议、输入数据或状态语义不满足契约。"""


@dataclass(frozen=True)
class RegressionResult:
    beta: np.ndarray
    standard_error: np.ndarray
    t_value: np.ndarray
    p_value: np.ndarray
    observations: int
    degrees_of_freedom: int
    r_squared: float
    condition_number: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{label}缺少字段：{missing}")


def _normalize_date(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ContractError(f"{label}存在无法解析的日期")
    return parsed


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"缺少机制图谱协议：{path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ContractError("机制图谱协议必须是YAML对象")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    matching = config["matching"]
    response = config["continuous_response"]
    adjudication = config["adjudication"]
    if protocol["project_id"] != "510300_MECHANISM_ATLAS_V1":
        raise ContractError("project_id必须固定为510300_MECHANISM_ATLAS_V1")
    if protocol["one_shot"] is not True:
        raise ContractError("机制图谱必须声明one_shot=true")
    if protocol["research_stage"] != "MECHANISM_DISCOVERY":
        raise ContractError("研究阶段必须固定为MECHANISM_DISCOVERY")
    if config["source_branch"]["required_status"] != (
        "REJECTED_FROZEN_MACD_BREADTH_DOWNSIDE_PREFLIGHT_V1_0_1_NO_RESCUE"
    ):
        raise ContractError("来源分支必须保持冻结拒绝")
    if scope["execution_asset"] != "510300.SH":
        raise ContractError("唯一风险资产必须为510300.SH")
    if scope["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("允许持有范围必须仅为510300和现金")
    forbidden = {
        "return_backtest_allowed": scope["return_backtest_allowed"],
        "leverage_allowed": scope["leverage_allowed"],
        "short_selling_allowed": scope["short_selling_allowed"],
        "derivatives_execution_allowed": scope["derivatives_execution_allowed"],
        "options_execution_allowed": scope["options_execution_allowed"],
        "futures_execution_allowed": scope["futures_execution_allowed"],
        "order_generation": scope["order_generation"],
        "broker_connection": scope["broker_connection"],
        "position_change": scope["position_change"],
        "live_trading_authorized": scope["live_trading_authorized"],
    }
    if any(value is not False for value in forbidden.values()):
        raise ContractError(f"研究边界出现非法启用项：{forbidden}")
    if scope["paper_or_shadow_mapping"] != "disabled":
        raise ContractError("Paper/Shadow映射必须禁用")
    fixed_dates = config["anchor_events"]["fixed_dates"]
    if fixed_dates != [
        "2023-07-12",
        "2024-02-29",
        "2024-05-13",
        "2025-01-27",
        "2025-07-10",
        "2026-07-01",
    ]:
        raise ContractError("六个病例日期发生变化")
    if matching["controls_per_case"] != 3 or matching["replacement"] is not False:
        raise ContractError("每个病例必须使用三个不放回对照")
    if matching["control_pool"]["failed_outcome_definition"] != (
        "forward_return_10d <= 0"
    ):
        raise ContractError("失败对照定义发生变化")
    expected_covariates = [
        "macd_histogram",
        "past_return_20d",
        "rv20_annualized",
        "distance_60d_high",
        "season_sin",
        "season_cos",
    ]
    if matching["covariates"] != expected_covariates:
        raise ContractError("匹配协变量发生变化")
    if response["interactions"] != ["MACD_X_BREADTH", "MACD_X_DOWNSIDE"]:
        raise ContractError("只允许两个预注册交互项")
    if response["horizons_trading_days"] != [5, 10, 20, 60]:
        raise ContractError("局部投影期限必须固定为5/10/20/60日")
    if response["controls"] != [
        "past_return_5d",
        "past_return_20d",
        "rv20_annualized",
        "distance_60d_high",
    ]:
        raise ContractError("局部投影控制变量发生变化")
    if config["mechanism_hypotheses"]["H4_INTRADAY_TEMPORARY_PRESSURE"]["status"] != (
        "NO_VIEW_DATA_CONTRACT_FAILED"
    ):
        raise ContractError("H4必须保留数据契约失败状态")
    if adjudication["return_evaluation"] != "NOT_ALLOWED":
        raise ContractError("机制图谱不得允许组合收益评估")
    if adjudication["net_sharpe"] != "NOT_COMPUTED":
        raise ContractError("机制图谱不得计算净夏普率")


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"输入文件不存在：{path}")
        if specification.get("type") == "json":
            try:
                loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ContractError(f"{name}不是有效JSON：{exc}") from exc
            continue
        frame = pd.read_parquet(path)
        _require_columns(frame, specification["required_columns"], name)
        key = specification.get("key", [])
        if key and frame[key].duplicated().any():
            duplicates = int(frame[key].duplicated().sum())
            raise ContractError(f"{name}主键重复：{duplicates}")
        loaded[name] = frame
    return loaded


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("机制图谱尚未冻结：缺少manifest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("manifest项目ID不一致")
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("manifest未处于FROZEN_BEFORE_FIRST_RESULT")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("机制图谱配置哈希漂移")
    mismatches: dict[str, dict[str, str | None]] = {}
    for section in ("tracked_files", "input_files"):
        records = manifest.get(section)
        if not isinstance(records, dict) or not records:
            raise ContractError(f"manifest缺少{section}")
        for relative, expected in records.items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else None
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件哈希漂移：{mismatches}")
    return {
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "tracked_file_count": len(manifest["tracked_files"]),
        "input_file_count": len(manifest["input_files"]),
        "hash_mismatches": {},
    }


def _snapshot_input_files(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        payload: dict[str, Any] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "type": specification.get("type", "parquet"),
        }
        value = inputs[name]
        if isinstance(value, pd.DataFrame):
            payload["rows"] = len(value)
            payload["columns"] = list(value.columns)
            if "date" in value.columns:
                dates = pd.to_datetime(value["date"], errors="coerce")
                if dates.notna().any():
                    payload["first_date"] = dates.min().date().isoformat()
                    payload["last_date"] = dates.max().date().isoformat()
        snapshots[name] = payload
    return snapshots


def audit_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    start = pd.Timestamp(config["windows"]["common_phase1_start"])
    end = pd.Timestamp(config["windows"]["common_phase1_end"])
    expected_rows = int(config["data_gates"]["expected_common_rows"])
    expected_components = int(config["data_gates"]["expected_component_count"])

    source_result = inputs["source_result"]
    required_status = config["source_branch"]["required_status"]
    if source_result.get("status") != required_status:
        failures.append("SOURCE_BRANCH_STATUS_CHANGED")
    if source_result.get("passed") is not False:
        failures.append("SOURCE_BRANCH_NOT_EXPLICITLY_REJECTED")
    adjudication = source_result.get("adjudication", {})
    if adjudication.get("return_evaluation") != "NOT_ALLOWED":
        failures.append("SOURCE_BRANCH_RETURN_BOUNDARY_CHANGED")

    daily = inputs["source_daily_features"].copy()
    daily["date"] = _normalize_date(daily["date"], "来源日度特征.date")
    daily = daily.loc[daily["date"].between(start, end)].sort_values("date")
    if len(daily) != expected_rows:
        failures.append(f"COMMON_DAILY_ROWS_{len(daily)}_EXPECTED_{expected_rows}")
    if daily["date"].duplicated().any():
        failures.append("COMMON_DAILY_DUPLICATE_DATE")
    common_dates = pd.DatetimeIndex(daily["date"])

    alignment: dict[str, Any] = {}
    for name in (
        "official_weighted_breadth",
        "equal_weight_breadth",
        "index_driver_summary",
        "if_term_structure",
        "etf_margin",
    ):
        frame = inputs[name].copy()
        frame["date"] = _normalize_date(frame["date"], f"{name}.date")
        dates = pd.DatetimeIndex(
            frame.loc[frame["date"].between(start, end), "date"].sort_values()
        )
        missing = common_dates.difference(dates)
        extra = dates.difference(common_dates)
        alignment[name] = {
            "common_rows": len(dates),
            "missing_common_dates": [value.date().isoformat() for value in missing],
            "extra_dates_inside_window": [value.date().isoformat() for value in extra],
        }
        if len(missing):
            failures.append(f"{name.upper()}_MISSING_COMMON_DATES")

    official = inputs["official_weighted_breadth"].copy()
    official["date"] = _normalize_date(official["date"], "官方广度.date")
    official["weight_snapshot_date"] = _normalize_date(
        official["weight_snapshot_date"], "官方广度.weight_snapshot_date"
    )
    official = official.loc[official["date"].between(start, end)]
    if (official["weight_snapshot_date"] > official["date"]).any():
        failures.append("OFFICIAL_BREADTH_FUTURE_WEIGHT_SNAPSHOT")
    if (official["component_count"] != expected_components).any():
        failures.append("OFFICIAL_BREADTH_COMPONENT_COUNT_FAILED")
    if (
        official[["ma20_weight_coverage", "ma60_weight_coverage"]].min().min()
        < float(config["data_gates"]["minimum_weight_coverage"])
    ):
        failures.append("OFFICIAL_BREADTH_WEIGHT_COVERAGE_FAILED")
    if (
        official["weight_snapshot_age_calendar_days"].max()
        > int(config["data_gates"]["maximum_weight_snapshot_age_calendar_days"])
    ):
        failures.append("OFFICIAL_BREADTH_WEIGHT_SNAPSHOT_TOO_OLD")
    if not official["valid_for_direction_model"].astype(bool).all():
        failures.append("OFFICIAL_BREADTH_DIRECTION_MODEL_INVALID")

    cross = inputs["point_in_time_cross_section"].copy()
    cross["date"] = _normalize_date(cross["date"], "点时截面.date")
    cross["weight_snapshot_date"] = _normalize_date(
        cross["weight_snapshot_date"], "点时截面.weight_snapshot_date"
    )
    cross = cross.loc[cross["date"].between(start, end)]
    component_counts = cross.groupby("date", sort=True).size()
    weight_sums = cross.groupby("date", sort=True)["snapshot_weight"].sum()
    if len(component_counts) != expected_rows:
        failures.append("POINT_IN_TIME_CROSS_SECTION_DATE_COUNT_FAILED")
    if not component_counts.eq(expected_components).all():
        failures.append("POINT_IN_TIME_CROSS_SECTION_COMPONENT_COUNT_FAILED")
    if (cross["weight_snapshot_date"] > cross["date"]).any():
        failures.append("POINT_IN_TIME_CROSS_SECTION_FUTURE_WEIGHT")
    if not weight_sums.between(0.999, 1.001).all():
        failures.append("POINT_IN_TIME_CROSS_SECTION_WEIGHT_SUM_FAILED")

    constituent = inputs["constituent_daily"].copy()
    constituent["date"] = _normalize_date(constituent["date"], "成分日线.date")
    constituent["market_cap_asof_date"] = pd.to_datetime(
        constituent["market_cap_asof_date"], errors="coerce"
    ).dt.normalize()
    constituent_common = constituent.loc[constituent["date"].between(start, end)]
    constituent_counts = constituent_common.groupby("date", sort=True).size()
    if len(constituent_counts) != expected_rows:
        failures.append("CONSTITUENT_DAILY_DATE_COUNT_FAILED")
    if not constituent_counts.eq(expected_components).all():
        failures.append("CONSTITUENT_DAILY_COMPONENT_COUNT_FAILED")
    valid_market_cap_dates = constituent_common["market_cap_asof_date"].notna()
    if (
        constituent_common.loc[valid_market_cap_dates, "market_cap_asof_date"]
        > constituent_common.loc[valid_market_cap_dates, "date"]
    ).any():
        failures.append("CONSTITUENT_DAILY_FUTURE_MARKET_CAP")

    driver = inputs["index_driver_summary"].copy()
    driver["date"] = _normalize_date(driver["date"], "指数归因摘要.date")
    driver_common = driver.loc[driver["date"].between(start, end)]

    events = inputs["source_events"].copy()
    events["date"] = _normalize_date(events["date"], "来源事件.date")
    cases = events.loc[
        events["is_selected_event"].astype(bool)
        & events["conditional_class"].eq("HIGH_QUALITY_CONVERGING")
    ]
    actual_case_dates = cases["date"].dt.strftime("%Y-%m-%d").tolist()
    fixed_case_dates = config["anchor_events"]["fixed_dates"]
    if actual_case_dates != fixed_case_dates:
        failures.append("ANCHOR_EVENT_DATES_CHANGED")
    anchor_driver = driver_common.loc[
        driver_common["date"].isin(pd.to_datetime(fixed_case_dates))
    ]
    if len(anchor_driver) != len(fixed_case_dates):
        failures.append("ANCHOR_DRIVER_ROWS_MISSING")
    elif not anchor_driver["valid_for_attribution"].astype(bool).all():
        failures.append("ANCHOR_DRIVER_ATTRIBUTION_INVALID")

    macro_audit: dict[str, Any] = {}
    for name in ("fdr007", "pmi_new_orders", "tsf_stock_yoy", "usdcny_midpoint"):
        frame = inputs[name]
        available = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
        values = pd.to_numeric(frame["first_release_value"], errors="coerce")
        invalid_available = int(available.isna().sum())
        invalid_value = int(values.isna().sum())
        macro_audit[name] = {
            "rows": len(frame),
            "invalid_available_at": invalid_available,
            "invalid_first_release_value": invalid_value,
            "latest_revision_used": False,
        }
        if invalid_available or invalid_value:
            failures.append(f"{name.upper()}_RELEASE_VINTAGE_INVALID")

    audit = {
        "project_id": config["protocol"]["project_id"],
        "status": "PASS_PHASE1_COMMON_DATA_CONTRACT" if not failures else config["adjudication"]["data_fail_status"],
        "passed": not failures,
        "generated_at": datetime.now().astimezone().isoformat(),
        "common_window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "rows": len(daily),
            "expected_rows": expected_rows,
        },
        "source_branch": {
            "status": source_result.get("status"),
            "passed": source_result.get("passed"),
            "return_evaluation": adjudication.get("return_evaluation"),
        },
        "anchor_events": {
            "expected": fixed_case_dates,
            "actual": actual_case_dates,
            "all_valid_for_attribution": bool(
                len(anchor_driver) == len(fixed_case_dates)
                and anchor_driver["valid_for_attribution"].astype(bool).all()
            ),
        },
        "official_breadth": {
            "minimum_ma20_weight_coverage": _finite_number(
                official["ma20_weight_coverage"].min()
            ),
            "minimum_ma60_weight_coverage": _finite_number(
                official["ma60_weight_coverage"].min()
            ),
            "maximum_snapshot_age_calendar_days": int(
                official["weight_snapshot_age_calendar_days"].max()
            ),
            "future_snapshot_rows": int(
                (official["weight_snapshot_date"] > official["date"]).sum()
            ),
        },
        "point_in_time_cross_section": {
            "dates": len(component_counts),
            "minimum_components": int(component_counts.min()),
            "maximum_components": int(component_counts.max()),
            "minimum_weight_sum": _finite_number(weight_sums.min()),
            "maximum_weight_sum": _finite_number(weight_sums.max()),
            "future_weight_rows": int(
                (cross["weight_snapshot_date"] > cross["date"]).sum()
            ),
        },
        "constituent_market_cap_diagnostic": {
            "valid_market_cap_asof_rows": int(valid_market_cap_dates.sum()),
            "missing_market_cap_asof_rows": int((~valid_market_cap_dates).sum()),
            "used_in_amount_expansion_feature": False,
        },
        "index_attribution": {
            "valid_days": int(driver_common["valid_for_attribution"].astype(bool).sum()),
            "no_view_days": int((~driver_common["valid_for_attribution"].astype(bool)).sum()),
            "anchor_valid_days": int(anchor_driver["valid_for_attribution"].astype(bool).sum()),
        },
        "alignment": alignment,
        "macro_release_vintage": macro_audit,
        "excluded_or_blocked_inputs": config["excluded_or_blocked_inputs"],
        "failures": failures,
        "input_snapshots": _snapshot_input_files(config, inputs),
        "boundaries": {
            "market_price_outcomes_used_in_input_audit": False,
            "strategy_backtest_run": False,
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        },
    }
    if failures:
        raise ContractError(f"机制图谱输入数据门失败：{failures}")
    return audit


def _rolling_log_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))

    def slope(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(centered, values - values.mean()) / denominator)

    return np.log(series.astype(float)).rolling(window, min_periods=window).apply(
        slope, raw=True
    )


def _prepare_macro_release(
    frame: pd.DataFrame,
    *,
    name: str,
    change_periods: int,
    change_kind: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["available_at"] = pd.to_datetime(
        output["available_at"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    if output["available_at"].isna().any():
        raise ContractError(f"{name}存在无法解析的available_at")
    output["first_release_value"] = pd.to_numeric(
        output["first_release_value"], errors="coerce"
    )
    if output["first_release_value"].isna().any():
        raise ContractError(f"{name}首发值存在缺失")
    output = output.sort_values("available_at").reset_index(drop=True)
    if change_kind == "difference":
        output["release_change"] = output["first_release_value"].diff(change_periods)
    elif change_kind == "pct_change":
        output["release_change"] = output["first_release_value"].pct_change(
            change_periods, fill_method=None
        )
    else:
        raise ContractError(f"未知宏观变化类型：{change_kind}")
    reference = "reference_period" if "reference_period" in output.columns else "date"
    return output[
        [reference, "available_at", "first_release_value", "release_change", "source"]
    ].rename(
        columns={
            reference: f"{name}_reference",
            "available_at": f"{name}_available_at",
            "first_release_value": f"{name}_value",
            "release_change": f"{name}_change",
            "source": f"{name}_source",
        }
    )


def _merge_macro_asof(
    panel: pd.DataFrame,
    release: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    signal_column = "signal_timestamp"
    available_column = f"{name}_available_at"
    left = panel.sort_values(signal_column).copy()
    right = release.sort_values(available_column).copy()
    merged = pd.merge_asof(
        left,
        right,
        left_on=signal_column,
        right_on=available_column,
        direction="backward",
        allow_exact_matches=True,
    )
    invalid = merged[available_column].notna() & (
        merged[available_column] > merged[signal_column]
    )
    if invalid.any():
        raise ContractError(f"{name}发生未来发布时间合并")
    return merged.sort_values("date").reset_index(drop=True)


def _build_cross_section_aggregates(cross: pd.DataFrame) -> pd.DataFrame:
    output = cross.copy()
    output["date"] = _normalize_date(output["date"], "点时截面.date")
    output["snapshot_weight"] = pd.to_numeric(
        output["snapshot_weight"], errors="coerce"
    )
    output["constituent_return_1d"] = pd.to_numeric(
        output["constituent_return_1d"], errors="coerce"
    )
    output["component_return_contribution_1d"] = pd.to_numeric(
        output["component_return_contribution_1d"], errors="coerce"
    )
    available = output.loc[output["return_available"].astype(bool)].copy()
    basic = (
        available.groupby("date", sort=True)
        .agg(
            constituent_return_median_1d=("constituent_return_1d", "median"),
            constituent_return_dispersion_1d=("constituent_return_1d", "std"),
            constituent_positive_share_1d=(
                "constituent_return_1d",
                lambda values: float((values > 0).mean()),
            ),
            available_constituent_count=("con_code", "count"),
        )
        .reset_index()
    )
    top10 = (
        output.sort_values(
            ["date", "snapshot_weight", "con_code"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=True, as_index=False)
        .head(10)
        .groupby("date", sort=True)["component_return_contribution_1d"]
        .sum(min_count=1)
        .rename("top10_weight_contribution_1d")
        .reset_index()
    )
    return basic.merge(top10, on="date", how="outer", validate="one_to_one")


def _build_industry_aggregates(industry: pd.DataFrame) -> pd.DataFrame:
    output = industry.copy()
    output["date"] = _normalize_date(output["date"], "行业归因.date")
    output["industry_return_1d"] = pd.to_numeric(
        output["industry_return_1d"], errors="coerce"
    )
    output["weighted_return_contribution_1d"] = pd.to_numeric(
        output["weighted_return_contribution_1d"], errors="coerce"
    )
    basic = (
        output.groupby("date", sort=True)
        .agg(
            industry_count_1d=("industry_l1", "nunique"),
            positive_industry_count_1d=(
                "industry_return_1d",
                lambda values: int((values > 0).sum()),
            ),
            industry_advancer_share_mean_1d=("industry_advancer_share_1d", "mean"),
        )
        .reset_index()
    )
    top5 = (
        output.sort_values(
            ["date", "weighted_return_contribution_1d", "industry_l1"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=True, as_index=False)
        .head(5)
        .groupby("date", sort=True)["weighted_return_contribution_1d"]
        .sum(min_count=1)
        .rename("top5_industry_contribution_1d")
        .reset_index()
    )
    return basic.merge(top5, on="date", how="left", validate="one_to_one")


def _build_amount_expansion(
    constituent: pd.DataFrame,
    cross: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    daily = constituent.copy()
    daily["date"] = _normalize_date(daily["date"], "成分日线.date")
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
    daily["total_market_cap_cny"] = pd.to_numeric(
        daily["total_market_cap_cny"], errors="coerce"
    )
    daily = daily.loc[daily["date"] <= end].sort_values(["con_code", "date"])
    daily["prior_amount_median_20"] = daily.groupby(
        "con_code", sort=False
    )["amount"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).median()
    )
    daily["amount_expansion_valid"] = (
        daily["amount"].notna()
        & daily["prior_amount_median_20"].notna()
        & daily["prior_amount_median_20"].gt(0)
    )
    daily["amount_expansion"] = (
        daily["amount"] > daily["prior_amount_median_20"]
    ).where(daily["amount_expansion_valid"])
    daily["traded_value_to_market_cap"] = (
        daily["amount"] / daily["total_market_cap_cny"].replace(0.0, np.nan)
    )
    daily = daily.loc[daily["date"].between(start, end)]

    weights = cross[["date", "con_code", "snapshot_weight"]].copy()
    weights["date"] = _normalize_date(weights["date"], "点时截面.date")
    joined = daily.merge(
        weights,
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    valid = joined["amount_expansion_valid"].astype(bool)
    joined["expansion_numeric"] = joined["amount_expansion"].astype(float)
    joined["weighted_expansion_numerator"] = (
        joined["snapshot_weight"] * joined["expansion_numeric"]
    ).where(valid)
    joined["valid_expansion_weight"] = joined["snapshot_weight"].where(valid)
    aggregate = (
        joined.groupby("date", sort=True)
        .agg(
            amount_expansion_count=("amount_expansion_valid", "sum"),
            amount_expansion_share_1d=("expansion_numeric", "mean"),
            weighted_expansion_numerator=("weighted_expansion_numerator", "sum"),
            valid_expansion_weight=("valid_expansion_weight", "sum"),
            median_traded_value_to_market_cap=(
                "traded_value_to_market_cap",
                "median",
            ),
        )
        .reset_index()
    )
    aggregate["weighted_amount_expansion_share_1d"] = (
        aggregate["weighted_expansion_numerator"]
        / aggregate["valid_expansion_weight"].replace(0.0, np.nan)
    )
    return aggregate.drop(
        columns=["weighted_expansion_numerator", "valid_expansion_weight"]
    )


def build_state_panel(
    config: dict[str, Any], inputs: dict[str, Any]
) -> pd.DataFrame:
    start = pd.Timestamp(config["windows"]["common_phase1_start"])
    end = pd.Timestamp(config["windows"]["common_phase1_end"])
    source = inputs["source_daily_features"].copy()
    source["date"] = _normalize_date(source["date"], "来源日度特征.date")
    source = source.loc[source["date"].between(start, end)].sort_values("date")
    base_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "total_return_close",
        "record_dividend_cumulative",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "macd_velocity_3d",
        "macd_acceleration_3d",
        "downside_deviation_5",
        "downside_deviation_20",
        "downside_vol_ratio_5_20",
    ]
    _require_columns(source, base_columns, "来源日度特征")
    panel = source[base_columns].reset_index(drop=True)
    total_return = panel["total_return_close"].pct_change(fill_method=None)
    log_return = np.log(panel["total_return_close"]).diff()
    panel["return_1d"] = total_return
    panel["past_return_5d"] = (
        panel["total_return_close"] / panel["total_return_close"].shift(5) - 1.0
    )
    panel["past_return_20d"] = (
        panel["total_return_close"] / panel["total_return_close"].shift(20) - 1.0
    )
    panel["trend_slope_20d"] = _rolling_log_slope(
        panel["total_return_close"], 20
    )
    panel["trend_slope_60d"] = _rolling_log_slope(
        panel["total_return_close"], 60
    )
    panel["ma20_total_return_close"] = panel["total_return_close"].rolling(
        20, min_periods=20
    ).mean()
    panel["distance_60d_high"] = (
        panel["close"] / panel["close"].rolling(60, min_periods=60).max() - 1.0
    )
    panel["distance_60d_low"] = (
        panel["close"] / panel["close"].rolling(60, min_periods=60).min() - 1.0
    )
    panel["gap_1d"] = panel["open"] / panel["close"].shift(1) - 1.0
    panel["path_efficiency_20d"] = (
        log_return.rolling(20, min_periods=20).sum().abs()
        / log_return.abs().rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    daily_rv20 = total_return.rolling(20, min_periods=20).std(ddof=1)
    panel["rv20_annualized"] = daily_rv20 * math.sqrt(242.0)
    negative_jump = total_return < (-2.0 * daily_rv20.shift(1))
    panel["negative_jump_share_20d"] = negative_jump.astype(float).rolling(
        20, min_periods=20
    ).mean()
    panel["log_downside_vol_ratio_5_20"] = np.log(
        panel["downside_vol_ratio_5_20"].replace(0.0, np.nan)
    )
    day_of_year = panel["date"].dt.dayofyear.astype(float)
    panel["season_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.2425)
    panel["season_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.2425)

    official = inputs["official_weighted_breadth"].copy()
    official["date"] = _normalize_date(official["date"], "官方广度.date")
    official_columns = [
        "date",
        "weight_snapshot_date",
        "weight_snapshot_age_calendar_days",
        "component_count",
        "weight_sum",
        "ma20_weight_coverage",
        "ma60_weight_coverage",
        "official_weighted_advancer_share_1d",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
        "official_weighted_return_dispersion_1d",
        "official_weight_hhi",
        "official_top10_weight_share",
        "official_weighted_above_ma20_share_change_5d",
        "valid_for_direction_model",
    ]
    panel = panel.merge(
        official[official_columns], on="date", how="left", validate="one_to_one"
    )

    equal = inputs["equal_weight_breadth"].copy()
    equal["date"] = _normalize_date(equal["date"], "等权广度.date")
    equal_columns = [
        "date",
        "equal_advancer_share_1d",
        "equal_positive_momentum_share_20d",
        "equal_above_ma20_share",
        "equal_above_ma60_share",
        "equal_return_1d",
        "equal_return_dispersion_1d",
        "valid_for_price_breadth",
    ]
    panel = panel.merge(
        equal[equal_columns], on="date", how="left", validate="one_to_one"
    )
    panel["equal_above_ma20_share_change_5d"] = panel[
        "equal_above_ma20_share"
    ].diff(5)
    panel["equal_above_ma60_share_change_5d"] = panel[
        "equal_above_ma60_share"
    ].diff(5)

    driver = inputs["index_driver_summary"].copy()
    driver["date"] = _normalize_date(driver["date"], "指数归因摘要.date")
    driver_columns = [
        "date",
        "price_coverage_weight",
        "return_coverage_weight",
        "industry_coverage_weight",
        "positive_industry_contribution_total_1d",
        "negative_industry_contribution_total_1d",
        "positive_industry_weight_share_1d",
        "industry_return_dispersion_1d",
        "valid_for_attribution",
        "output",
        "failure_category",
    ]
    panel = panel.merge(
        driver[driver_columns], on="date", how="left", validate="one_to_one"
    )
    panel = panel.rename(
        columns={
            "valid_for_attribution": "internal_attribution_valid",
            "output": "internal_attribution_output",
            "failure_category": "internal_attribution_failure_category",
        }
    )

    cross = inputs["point_in_time_cross_section"].copy()
    cross["date"] = _normalize_date(cross["date"], "点时截面.date")
    cross = cross.loc[cross["date"].between(start, end)]
    panel = panel.merge(
        _build_cross_section_aggregates(cross),
        on="date",
        how="left",
        validate="one_to_one",
    )
    industry = inputs["industry_attribution"].copy()
    industry["date"] = _normalize_date(industry["date"], "行业归因.date")
    industry = industry.loc[industry["date"].between(start, end)]
    panel = panel.merge(
        _build_industry_aggregates(industry),
        on="date",
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        _build_amount_expansion(inputs["constituent_daily"], cross, start, end),
        on="date",
        how="left",
        validate="one_to_one",
    )

    attribution_only_columns = [
        "top10_weight_contribution_1d",
        "constituent_return_median_1d",
        "constituent_return_dispersion_1d",
        "constituent_positive_share_1d",
        "positive_industry_count_1d",
        "industry_count_1d",
        "industry_advancer_share_mean_1d",
        "top5_industry_contribution_1d",
    ]
    invalid_attribution = ~panel["internal_attribution_valid"].fillna(False).astype(bool)
    panel.loc[invalid_attribution, attribution_only_columns] = np.nan

    term = inputs["if_term_structure"].copy()
    term["date"] = _normalize_date(term["date"], "IF期限结构.date")
    term_columns = [
        "date",
        "eligible_contract_count",
        "near_symbol",
        "near_dte",
        "annualized_near_basis",
        "annualized_near_next_curve",
        "annualized_near_far_curve",
        "annualized_oi_weighted_basis",
        "feature_asof",
    ]
    term = term[term_columns].rename(
        columns={
            "annualized_near_basis": "if_raw_annualized_near_basis",
            "annualized_near_next_curve": "if_raw_annualized_near_next_curve",
            "annualized_near_far_curve": "if_raw_annualized_near_far_curve",
            "annualized_oi_weighted_basis": "if_raw_annualized_oi_weighted_basis",
        }
    )
    panel = panel.merge(term, on="date", how="left", validate="one_to_one")
    panel["if_basis_change_3d"] = panel["if_raw_annualized_near_basis"].diff(3)
    panel["if_basis_change_5d"] = panel["if_raw_annualized_near_basis"].diff(5)

    shares = inputs["etf_fund_shares"].copy()
    shares["date"] = _normalize_date(shares["date"], "基金份额.date")
    shares = shares.sort_values("date")
    shares["fund_share_change_5d"] = shares["fund_shares"].pct_change(
        5, fill_method=None
    )
    shares["fund_share_change_20d"] = shares["fund_shares"].pct_change(
        20, fill_method=None
    )
    panel = panel.merge(
        shares[["date", "fund_shares", "fund_share_change_5d", "fund_share_change_20d"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    margin = inputs["etf_margin"].copy()
    margin["date"] = _normalize_date(margin["date"], "融资融券.date")
    panel = panel.merge(
        margin[["date", "financing_net_buy_cny", "rzrqye"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    panel["financing_net_buy_5d"] = panel["financing_net_buy_cny"].rolling(
        5, min_periods=5
    ).sum()

    nav = inputs["etf_nav"].copy()
    nav["date"] = _normalize_date(nav["date"], "ETF净值.date")
    panel = panel.merge(
        nav[["date", "unit_nav", "close_premium_bps"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    panel["signal_timestamp"] = (
        panel["date"] + pd.Timedelta(hours=15)
    ).dt.tz_localize("Asia/Shanghai")
    fdr = _prepare_macro_release(
        inputs["fdr007"],
        name="fdr007",
        change_periods=5,
        change_kind="difference",
    )
    pmi = _prepare_macro_release(
        inputs["pmi_new_orders"],
        name="pmi_new_orders",
        change_periods=3,
        change_kind="difference",
    )
    tsf = _prepare_macro_release(
        inputs["tsf_stock_yoy"],
        name="tsf_stock_yoy",
        change_periods=3,
        change_kind="difference",
    )
    fx = _prepare_macro_release(
        inputs["usdcny_midpoint"],
        name="usdcny",
        change_periods=20,
        change_kind="pct_change",
    )
    for name, release in (
        ("fdr007", fdr),
        ("pmi_new_orders", pmi),
        ("tsf_stock_yoy", tsf),
        ("usdcny", fx),
    ):
        panel = _merge_macro_asof(panel, release, name)
    panel = panel.rename(
        columns={
            "fdr007_change": "fdr007_change_5d",
            "pmi_new_orders_change": "pmi_new_orders_change_3_release",
            "tsf_stock_yoy_change": "tsf_stock_yoy_change_3_release",
            "usdcny_change": "usdcny_change_20d",
        }
    )

    panel["price_risk_valid"] = panel[
        [
            "macd_histogram",
            "macd_velocity_3d",
            "past_return_20d",
            "rv20_annualized",
            "distance_60d_high",
        ]
    ].notna().all(axis=1)
    panel["official_breadth_valid"] = panel["valid_for_direction_model"].fillna(False)
    panel["if_term_structure_valid"] = panel[
        ["if_raw_annualized_near_basis", "if_basis_change_3d"]
    ].notna().all(axis=1)
    panel["macro_slow_prior_valid"] = panel[
        ["fdr007_value", "pmi_new_orders_value", "tsf_stock_yoy_value", "usdcny_value"]
    ].notna().all(axis=1)
    if len(panel) != int(config["data_gates"]["expected_common_rows"]):
        raise ContractError("状态面板行数不满足共同窗口契约")
    if panel["date"].duplicated().any():
        raise ContractError("状态面板存在重复日期")
    return panel.sort_values("date").reset_index(drop=True)


def build_outcome_panel(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """构造与状态面板物理分离的事件结果，不计算策略净值。"""

    output = pd.DataFrame({"date": panel["date"].copy()})
    opens = panel["open"].to_numpy(dtype=float)
    closes = panel["close"].to_numpy(dtype=float)
    cumulative_dividend = panel["record_dividend_cumulative"].to_numpy(dtype=float)
    total_return_close = panel["total_return_close"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["date"]).to_numpy(dtype="datetime64[ns]")
    row_count = len(panel)
    for horizon in config["continuous_response"]["horizons_trading_days"]:
        horizon = int(horizon)
        returns = np.full(row_count, np.nan, dtype=float)
        adverse = np.full(row_count, np.nan, dtype=float)
        favorable = np.full(row_count, np.nan, dtype=float)
        future_rv = np.full(row_count, np.nan, dtype=float)
        time_to_positive = np.full(row_count, np.nan, dtype=float)
        positive_fraction = np.full(row_count, np.nan, dtype=float)
        end_dates = np.full(
            row_count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        complete = np.zeros(row_count, dtype=bool)
        for signal_index in range(row_count):
            entry_index = signal_index + 1
            exit_index = signal_index + horizon
            if exit_index >= row_count:
                continue
            entry_price = opens[entry_index]
            if not math.isfinite(entry_price) or entry_price <= 0:
                continue
            entitled_before_entry = cumulative_dividend[signal_index]
            path_wealth = (
                closes[entry_index : exit_index + 1]
                + cumulative_dividend[entry_index : exit_index + 1]
                - entitled_before_entry
            )
            path_return = path_wealth / entry_price - 1.0
            if not np.isfinite(path_return).all():
                continue
            returns[signal_index] = float(path_return[-1])
            adverse[signal_index] = float(path_return.min())
            favorable[signal_index] = float(path_return.max())
            future_close = total_return_close[signal_index : exit_index + 1]
            future_daily_return = future_close[1:] / future_close[:-1] - 1.0
            if len(future_daily_return) > 1 and np.isfinite(future_daily_return).all():
                future_rv[signal_index] = float(
                    np.std(future_daily_return, ddof=1) * math.sqrt(242.0)
                )
            positive_locations = np.flatnonzero(path_return > 0.0)
            time_to_positive[signal_index] = float(
                positive_locations[0] + 1 if len(positive_locations) else horizon + 1
            )
            positive_fraction[signal_index] = float(np.mean(path_return > 0.0))
            end_dates[signal_index] = dates[exit_index]
            complete[signal_index] = True
        output[f"forward_return_{horizon}d"] = returns
        output[f"forward_mae_{horizon}d"] = adverse
        output[f"forward_mfe_{horizon}d"] = favorable
        output[f"future_rv_{horizon}d"] = future_rv
        output[f"time_to_positive_censored_{horizon}d"] = time_to_positive
        output[f"positive_wealth_fraction_{horizon}d"] = positive_fraction
        output[f"outcome_end_date_{horizon}d"] = pd.to_datetime(end_dates)
        output[f"outcome_complete_{horizon}d"] = complete
    return output


def select_cases_and_match_controls(
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    events = events.copy()
    events["date"] = _normalize_date(events["date"], "来源事件.date")
    matching_panel = panel[
            [
                "date",
                "macd_histogram",
                "past_return_20d",
                "rv20_annualized",
                "distance_60d_high",
                "season_sin",
                "season_cos",
                "internal_attribution_valid",
            ]
        ].copy()
    overlapping = [
        column
        for column in matching_panel.columns
        if column != "date" and column in events.columns
    ]
    source = events.drop(columns=overlapping).merge(
        matching_panel,
        on="date",
        how="left",
        validate="one_to_one",
    )
    cases = source.loc[
        source["is_selected_event"].astype(bool)
        & source["conditional_class"].eq("HIGH_QUALITY_CONVERGING")
    ].sort_values("date").reset_index(drop=True)
    fixed_dates = config["anchor_events"]["fixed_dates"]
    if cases["date"].dt.strftime("%Y-%m-%d").tolist() != fixed_dates:
        raise ContractError("病例日期与冻结协议不一致")
    if not cases["internal_attribution_valid"].astype(bool).all():
        raise ContractError("病例事件日存在无效指数归因")
    cases["case_id"] = [f"CASE_{index:02d}" for index in range(1, len(cases) + 1)]

    date_to_ordinal = dict(zip(panel["date"], range(len(panel))))
    case_ordinals = [date_to_ordinal[value] for value in cases["date"]]
    pre = int(config["windows"]["dossier_pre_trading_days"])
    post = int(config["windows"]["dossier_post_trading_days"])
    gap = int(
        config["matching"]["control_pool"][
            "minimum_gap_from_any_anchor_trading_days"
        ]
    )
    start_date = pd.Timestamp(config["matching"]["control_pool"]["start_date"])
    eligible = source.loc[
        source["primary_class"].eq("CONVERGING")
        & ~source["conditional_class"].eq("HIGH_QUALITY_CONVERGING")
        & source["date"].ge(start_date)
        & pd.to_numeric(source["forward_return_10d"], errors="coerce").le(0.0)
        & source["internal_attribution_valid"].fillna(False).astype(bool)
    ].copy()
    full_window: list[bool] = []
    sufficiently_separated: list[bool] = []
    for value in eligible["date"]:
        ordinal = date_to_ordinal.get(value)
        if ordinal is None:
            full_window.append(False)
            sufficiently_separated.append(False)
            continue
        full_window.append(ordinal >= pre and ordinal <= len(panel) - post - 1)
        sufficiently_separated.append(
            min(abs(ordinal - case_ordinal) for case_ordinal in case_ordinals) > gap
        )
    eligible = eligible.loc[
        np.asarray(full_window, dtype=bool)
        & np.asarray(sufficiently_separated, dtype=bool)
    ].sort_values("date").reset_index(drop=True)

    covariates = config["matching"]["covariates"]
    if cases[covariates].isna().any().any() or eligible[covariates].isna().any().any():
        raise ContractError("病例或对照池的匹配协变量存在缺失")
    controls_per_case = int(config["matching"]["controls_per_case"])
    assignment_count = len(cases) * controls_per_case
    if len(eligible) < assignment_count:
        raise ContractError(
            f"合格失败对照不足：required={assignment_count}, actual={len(eligible)}"
        )
    pooled = pd.concat([cases[covariates], eligible[covariates]], ignore_index=True)
    means = pooled.mean(axis=0)
    standard_deviations = pooled.std(axis=0, ddof=0).replace(0.0, np.nan)
    if standard_deviations.isna().any():
        raise ContractError("匹配协变量存在零方差")
    case_z = (cases[covariates] - means) / standard_deviations
    control_z = (eligible[covariates] - means) / standard_deviations
    cost = np.linalg.norm(
        case_z.to_numpy()[:, None, :] - control_z.to_numpy()[None, :, :], axis=2
    )
    repeated_cases = np.repeat(np.arange(len(cases)), controls_per_case)
    expanded_cost = cost[repeated_cases, :]
    tie_breaker = np.arange(len(eligible), dtype=float)[None, :] * 1e-12
    assigned_rows, assigned_columns = linear_sum_assignment(expanded_cost + tie_breaker)
    if len(assigned_rows) != assignment_count:
        raise ContractError("全局匹配未返回完整分配")
    records: list[dict[str, Any]] = []
    for row_index, control_index in zip(assigned_rows, assigned_columns, strict=True):
        case_index = int(repeated_cases[row_index])
        case = cases.iloc[case_index]
        control = eligible.iloc[int(control_index)]
        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "case_date": case["date"],
            "control_date": control["date"],
            "matching_distance": float(cost[case_index, int(control_index)]),
            "control_forward_return_10d_frozen": float(control["forward_return_10d"]),
        }
        for covariate in covariates:
            record[f"case_{covariate}"] = float(case[covariate])
            record[f"control_{covariate}"] = float(control[covariate])
            record[f"difference_{covariate}"] = float(
                case[covariate] - control[covariate]
            )
        records.append(record)
    matches = pd.DataFrame(records).sort_values(
        ["case_id", "matching_distance", "control_date"]
    )
    matches["control_rank"] = matches.groupby("case_id", sort=False).cumcount() + 1
    matches["control_id"] = matches.apply(
        lambda row: f"{row['case_id']}_CONTROL_{int(row['control_rank']):02d}", axis=1
    )
    matches = matches.reset_index(drop=True)
    if matches["control_date"].duplicated().any():
        raise ContractError("不放回匹配产生重复对照")

    selected = eligible.set_index("date").loc[matches["control_date"]].reset_index()
    pooled_sample_std = pooled[covariates].std(axis=0, ddof=1).replace(0.0, np.nan)
    absolute_smd = (
        (cases[covariates].mean() - selected[covariates].mean())
        / pooled_sample_std
    ).abs()
    maximum_distance = float(matches["matching_distance"].max())
    maximum_smd = float(absolute_smd.max())
    distance_gate = maximum_distance <= float(
        config["matching"]["maximum_pair_distance"]
    )
    smd_gate = maximum_smd <= float(
        config["matching"]["maximum_absolute_standardized_mean_difference"]
    )
    balance = {
        "eligible_control_count": len(eligible),
        "case_count": len(cases),
        "assigned_control_count": len(matches),
        "unique_control_count": int(matches["control_date"].nunique()),
        "maximum_pair_distance": maximum_distance,
        "median_pair_distance": float(matches["matching_distance"].median()),
        "absolute_standardized_mean_differences": {
            name: float(value) for name, value in absolute_smd.items()
        },
        "maximum_absolute_standardized_mean_difference": maximum_smd,
        "gates": {
            "three_unique_controls_per_case": bool(
                matches.groupby("case_id")["control_date"].nunique().eq(3).all()
            ),
            "no_control_reuse": not bool(matches["control_date"].duplicated().any()),
            "maximum_pair_distance": distance_gate,
            "maximum_absolute_standardized_mean_difference": smd_gate,
        },
        "descriptive_only_due_to_outcome_conditioned_control_pool": True,
    }
    if not all(balance["gates"].values()):
        raise ContractError(f"病例/对照匹配质量门失败：{balance}")
    case_output_columns = [
        "case_id",
        "date",
        "macd_histogram",
        "past_return_20d",
        "rv20_annualized",
        "distance_60d_high",
    ]
    cases = cases[case_output_columns].rename(columns={"date": "case_date"})
    cases = cases.merge(
        outcomes,
        left_on="case_date",
        right_on="date",
        how="left",
        validate="one_to_one",
    ).drop(columns=["date"])
    return cases, matches, balance


def build_event_window_panel(
    panel: pd.DataFrame,
    cases: pd.DataFrame,
    matches: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    pre = int(config["windows"]["dossier_pre_trading_days"])
    post = int(config["windows"]["dossier_post_trading_days"])
    date_to_ordinal = dict(zip(panel["date"], range(len(panel))))
    units: list[dict[str, Any]] = []
    for row in cases.itertuples(index=False):
        units.append(
            {
                "unit_id": row.case_id,
                "unit_type": "CASE",
                "case_id": row.case_id,
                "event_date": row.case_date,
            }
        )
    for row in matches.itertuples(index=False):
        units.append(
            {
                "unit_id": row.control_id,
                "unit_type": "FAILED_CONTROL",
                "case_id": row.case_id,
                "event_date": row.control_date,
            }
        )
    windows: list[pd.DataFrame] = []
    for unit in units:
        ordinal = date_to_ordinal[pd.Timestamp(unit["event_date"])]
        start = ordinal - pre
        end = ordinal + post
        if start < 0 or end >= len(panel):
            raise ContractError(f"事件窗口不完整：{unit}")
        window = panel.iloc[start : end + 1].copy()
        window.insert(0, "relative_trading_day", np.arange(-pre, post + 1))
        window.insert(0, "event_date", pd.Timestamp(unit["event_date"]))
        window.insert(0, "case_id", unit["case_id"])
        window.insert(0, "unit_type", unit["unit_type"])
        window.insert(0, "unit_id", unit["unit_id"])
        windows.append(window)
    output = pd.concat(windows, ignore_index=True)
    expected_rows = (len(cases) + len(matches)) * (pre + post + 1)
    if len(output) != expected_rows:
        raise ContractError("事件窗口面板行数不一致")
    return output


def _first_date_where(frame: pd.DataFrame, mask: pd.Series) -> pd.Timestamp | None:
    selected = frame.loc[mask]
    if selected.empty:
        return None
    return pd.Timestamp(selected.iloc[0]["date"])


def build_chronology_table(
    windows: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    required_drop = float(
        config["transmission_chronology"]["risk_relief_required_drop"]
    )
    for unit_id, window in windows.groupby("unit_id", sort=False):
        window = window.sort_values("relative_trading_day").reset_index(drop=True)
        event_row = window.loc[window["relative_trading_day"].eq(0)].iloc[0]
        pre20 = window.loc[window["relative_trading_day"].between(-20, 0)]
        risk_values = pd.to_numeric(
            pre20["downside_vol_ratio_5_20"], errors="coerce"
        )
        if risk_values.notna().any():
            risk_index = risk_values.idxmax()
            risk_peak_date = pd.Timestamp(window.loc[risk_index, "date"])
            risk_peak_value = float(window.loc[risk_index, "downside_vol_ratio_5_20"])
            current_risk = float(event_row["downside_vol_ratio_5_20"])
            risk_relief = (
                (risk_peak_value - current_risk) / risk_peak_value
                if risk_peak_value > 0
                else np.nan
            )
        else:
            risk_peak_date = None
            risk_peak_value = np.nan
            risk_relief = np.nan

        pre10 = window.loc[window["relative_trading_day"].between(-10, 0)]
        if_confirmation_date = _first_date_where(
            pre10, pd.to_numeric(pre10["if_basis_change_3d"], errors="coerce") > 0.0
        )
        breadth_all = pd.to_numeric(
            window["official_weighted_above_ma20_share_change_5d"],
            errors="coerce",
        )
        breadth_turn_mask = (
            breadth_all.gt(0.0)
            & breadth_all.shift(1).le(0.0)
            & window["relative_trading_day"].between(-10, 0)
        )
        breadth_turn_date = _first_date_where(window, breadth_turn_mask)
        post20 = window.loc[window["relative_trading_day"].between(0, 20)]
        price_confirmation_date = _first_date_where(
            post20,
            post20["total_return_close"].gt(post20["ma20_total_return_close"])
            & post20["trend_slope_20d"].gt(0.0),
        )
        event_date = pd.Timestamp(event_row["event_date"])
        chronology_dates = [
            risk_peak_date,
            if_confirmation_date,
            breadth_turn_date,
            event_date,
            price_confirmation_date,
        ]
        all_observed = all(value is not None for value in chronology_dates)
        canonical_order = bool(
            all_observed
            and risk_relief >= required_drop
            and risk_peak_date <= if_confirmation_date
            and if_confirmation_date <= breadth_turn_date
            and breadth_turn_date <= event_date
            and event_date <= price_confirmation_date
        )
        records.append(
            {
                "unit_id": unit_id,
                "unit_type": event_row["unit_type"],
                "case_id": event_row["case_id"],
                "event_date": event_date,
                "risk_peak_date": risk_peak_date,
                "risk_peak_value": _finite_number(risk_peak_value),
                "risk_relief_fraction_at_event": _finite_number(risk_relief),
                "risk_relief_at_least_10pct": bool(
                    math.isfinite(risk_relief) and risk_relief >= required_drop
                ),
                "if_confirmation_date": if_confirmation_date,
                "breadth_turn_date": breadth_turn_date,
                "macd_turn_date": event_date,
                "price_trend_confirmation_date": price_confirmation_date,
                "all_links_observed": all_observed,
                "canonical_order_observed": canonical_order,
            }
        )
    return pd.DataFrame(records)


def _ols_hac(y: np.ndarray, x: np.ndarray, lag: int) -> RegressionResult:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ContractError("OLS输入维度不一致")
    observations, parameters = x.shape
    if observations <= parameters + lag + 5:
        raise ContractError(
            f"OLS有效样本不足：n={observations}, k={parameters}, lag={lag}"
        )
    rank = int(np.linalg.matrix_rank(x))
    if rank < parameters:
        raise ContractError(f"OLS设计矩阵秩不足：rank={rank}, k={parameters}")
    xtx_inverse = np.linalg.pinv(x.T @ x, hermitian=True)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    xu = x * residual[:, None]
    meat = xu.T @ xu
    maximum_lag = min(int(lag), observations - 1)
    for current_lag in range(1, maximum_lag + 1):
        weight = 1.0 - current_lag / (maximum_lag + 1.0)
        cross = xu[current_lag:].T @ xu[:-current_lag]
        meat += weight * (cross + cross.T)
    finite_sample = observations / (observations - parameters)
    covariance = finite_sample * xtx_inverse @ meat @ xtx_inverse
    diagonal = np.diag(covariance)
    diagonal = np.where(diagonal >= 0.0, diagonal, np.nan)
    standard_error = np.sqrt(diagonal)
    t_value = beta / standard_error
    degrees_of_freedom = observations - parameters
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df=degrees_of_freedom)
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    residual_sum_squares = float(np.sum(residual**2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    return RegressionResult(
        beta=beta,
        standard_error=standard_error,
        t_value=t_value,
        p_value=p_value,
        observations=observations,
        degrees_of_freedom=degrees_of_freedom,
        r_squared=float(r_squared),
        condition_number=float(np.linalg.cond(x)),
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().astype(float).sort_values(kind="mergesort")
    count = len(finite)
    if count == 0:
        return output
    running_maximum = 0.0
    for rank, (index, value) in enumerate(finite.items(), start=1):
        adjusted = min(1.0, (count - rank + 1) * value)
        running_maximum = max(running_maximum, adjusted)
        output.loc[index] = running_maximum
    return output


def _standardize_predictors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    mapping = {
        "M_MACD_CONVERSION": "macd_velocity_3d",
        "B_BREADTH_DIFFUSION": "official_weighted_above_ma20_share_change_5d",
        "V_DOWNSIDE_PRESSURE": "log_downside_vol_ratio_5_20",
        "CONTROL_RETURN_5D": "past_return_5d",
        "CONTROL_RETURN_20D": "past_return_20d",
        "CONTROL_RV20": "rv20_annualized",
        "CONTROL_DISTANCE_HIGH60": "distance_60d_high",
    }
    output = frame.copy()
    audit: dict[str, dict[str, float]] = {}
    for standardized, source in mapping.items():
        values = pd.to_numeric(output[source], errors="coerce")
        mean = float(values.mean())
        deviation = float(values.std(ddof=0))
        if not math.isfinite(deviation) or deviation <= 0:
            raise ContractError(f"局部投影变量零方差或无效：{source}")
        output[standardized] = (values - mean) / deviation
        audit[standardized] = {"source": source, "mean": mean, "std_ddof0": deviation}
    output["MACD_X_BREADTH"] = (
        output["M_MACD_CONVERSION"] * output["B_BREADTH_DIFFUSION"]
    )
    output["MACD_X_DOWNSIDE"] = (
        output["M_MACD_CONVERSION"] * output["V_DOWNSIDE_PRESSURE"]
    )
    return output, audit


def run_local_projections(
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = panel.merge(outcomes, on="date", how="left", validate="one_to_one")
    frame, standardization = _standardize_predictors(frame)
    predictor_names = [
        "INTERCEPT",
        "M_MACD_CONVERSION",
        "B_BREADTH_DIFFUSION",
        "V_DOWNSIDE_PRESSURE",
        "MACD_X_BREADTH",
        "MACD_X_DOWNSIDE",
        "CONTROL_RETURN_5D",
        "CONTROL_RETURN_20D",
        "CONTROL_RV20",
        "CONTROL_DISTANCE_HIGH60",
    ]
    model_columns = predictor_names[1:]
    periods = {
        "FULL": (
            pd.Timestamp(config["windows"]["common_phase1_start"]),
            pd.Timestamp(config["windows"]["common_phase1_end"]),
        ),
        "EARLY": tuple(
            pd.Timestamp(value)
            for value in config["continuous_response"]["structural_periods"]["early"]
        ),
        "LATE": tuple(
            pd.Timestamp(value)
            for value in config["continuous_response"]["structural_periods"]["late"]
        ),
    }
    outcome_templates = {
        "PRIMARY_RETURN": "forward_return_{horizon}d",
        "MAXIMUM_ADVERSE_EXCURSION": "forward_mae_{horizon}d",
        "MAXIMUM_FAVORABLE_EXCURSION": "forward_mfe_{horizon}d",
        "FUTURE_REALIZED_VOLATILITY": "future_rv_{horizon}d",
        "TIME_TO_POSITIVE_CENSORED": "time_to_positive_censored_{horizon}d",
        "POSITIVE_WEALTH_FRACTION": "positive_wealth_fraction_{horizon}d",
    }
    records: list[dict[str, Any]] = []
    for period_name, (period_start, period_end) in periods.items():
        period_mask = frame["date"].between(period_start, period_end)
        for horizon in config["continuous_response"]["horizons_trading_days"]:
            horizon = int(horizon)
            for outcome_name, template in outcome_templates.items():
                outcome_column = template.format(horizon=horizon)
                columns = [outcome_column] + model_columns
                sample = frame.loc[period_mask, columns].dropna()
                y = sample[outcome_column].to_numpy(dtype=float)
                x_values = sample[model_columns].to_numpy(dtype=float)
                x = np.column_stack([np.ones(len(sample), dtype=float), x_values])
                result = _ols_hac(y, x, lag=horizon)
                for index, term in enumerate(predictor_names):
                    records.append(
                        {
                            "period": period_name,
                            "period_start": period_start,
                            "period_end": period_end,
                            "horizon_trading_days": horizon,
                            "outcome": outcome_name,
                            "outcome_column": outcome_column,
                            "term": term,
                            "beta": float(result.beta[index]),
                            "standard_error_hac": float(result.standard_error[index]),
                            "t_value_hac": float(result.t_value[index]),
                            "p_value_hac": float(result.p_value[index]),
                            "adjusted_p_value_holm": np.nan,
                            "primary_interaction_family_member": bool(
                                period_name == "FULL"
                                and outcome_name == "PRIMARY_RETURN"
                                and term in {"MACD_X_BREADTH", "MACD_X_DOWNSIDE"}
                            ),
                            "observations": result.observations,
                            "degrees_of_freedom": result.degrees_of_freedom,
                            "r_squared": result.r_squared,
                            "condition_number": result.condition_number,
                            "hac_lag": horizon,
                        }
                    )
    results = pd.DataFrame(records)
    family_mask = results["primary_interaction_family_member"]
    family_count = int(family_mask.sum())
    expected_family = int(config["continuous_response"]["primary_interaction_tests"])
    if family_count != expected_family:
        raise ContractError(
            f"主交互检验数量不一致：expected={expected_family}, actual={family_count}"
        )
    results.loc[family_mask, "adjusted_p_value_holm"] = _holm_adjust(
        results.loc[family_mask, "p_value_hac"]
    )

    rolling_records: list[dict[str, Any]] = []
    window = int(config["continuous_response"]["rolling_window_trading_days"])
    step = int(config["continuous_response"]["rolling_step_trading_days"])
    for horizon in (10, 20):
        outcome_column = f"forward_return_{horizon}d"
        for end_position in range(window - 1, len(frame), step):
            start_position = end_position - window + 1
            sample = frame.iloc[start_position : end_position + 1][
                ["date", outcome_column] + model_columns
            ].dropna()
            if len(sample) <= len(predictor_names) + horizon + 5:
                continue
            y = sample[outcome_column].to_numpy(dtype=float)
            x = np.column_stack(
                [
                    np.ones(len(sample), dtype=float),
                    sample[model_columns].to_numpy(dtype=float),
                ]
            )
            result = _ols_hac(y, x, lag=horizon)
            for index, term in enumerate(predictor_names[1:], start=1):
                rolling_records.append(
                    {
                        "window_start": sample["date"].iloc[0],
                        "window_end": sample["date"].iloc[-1],
                        "horizon_trading_days": horizon,
                        "term": term,
                        "beta": float(result.beta[index]),
                        "standard_error_hac": float(result.standard_error[index]),
                        "p_value_hac": float(result.p_value[index]),
                        "observations": result.observations,
                    }
                )
    rolling = pd.DataFrame(rolling_records)
    audit = {
        "predictor_standardization": standardization,
        "periods": {
            name: [start.date().isoformat(), end.date().isoformat()]
            for name, (start, end) in periods.items()
        },
        "primary_interaction_family_count": family_count,
        "multiple_testing": "HOLM_FAMILYWISE_5PCT_ON_PRIMARY_INTERACTIONS",
        "rolling_window_trading_days": window,
        "rolling_step_trading_days": step,
        "rolling_result_rows": len(rolling),
    }
    return results, rolling, audit


def build_opportunity_frequency(
    panel: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    frame = panel.merge(
        outcomes[["date", "forward_return_10d", "forward_return_20d"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    frame["M_NEGATIVE_HIST_CONVERGING"] = (
        frame["macd_histogram"].lt(0.0) & frame["macd_velocity_3d"].gt(0.0)
    )
    frame["BREADTH_IMPROVING"] = frame[
        "official_weighted_above_ma20_share_change_5d"
    ].gt(0.0)
    frame["DOWNSIDE_COOLING"] = frame["downside_vol_ratio_5_20"].lt(1.0)
    frame["IF_BASIS_IMPROVING"] = frame["if_basis_change_3d"].gt(0.0)
    frame["H2_CONFIRMED"] = frame[
        [
            "M_NEGATIVE_HIST_CONVERGING",
            "BREADTH_IMPROVING",
            "DOWNSIDE_COOLING",
            "IF_BASIS_IMPROVING",
        ]
    ].all(axis=1)
    frame["H2_UNCONFIRMED"] = frame["M_NEGATIVE_HIST_CONVERGING"] & ~frame[
        "H2_CONFIRMED"
    ]
    frame["year"] = frame["date"].dt.year.astype(str)
    states = [
        "M_NEGATIVE_HIST_CONVERGING",
        "BREADTH_IMPROVING",
        "DOWNSIDE_COOLING",
        "IF_BASIS_IMPROVING",
        "H2_CONFIRMED",
        "H2_UNCONFIRMED",
    ]
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", frame)]
    groups.extend((year, group) for year, group in frame.groupby("year", sort=True))
    for period, group in groups:
        confirmed_returns = group.loc[group["H2_CONFIRMED"], "forward_return_10d"].dropna()
        unconfirmed_returns = group.loc[
            group["H2_UNCONFIRMED"], "forward_return_10d"
        ].dropna()
        conditional_edge = (
            float(confirmed_returns.mean() - unconfirmed_returns.mean())
            if len(confirmed_returns) and len(unconfirmed_returns)
            else np.nan
        )
        for state in states:
            mask = group[state].fillna(False).astype(bool)
            returns10 = group.loc[mask, "forward_return_10d"].dropna()
            returns20 = group.loc[mask, "forward_return_20d"].dropna()
            records.append(
                {
                    "period": period,
                    "state": state,
                    "available_days": len(group),
                    "opportunity_days": int(mask.sum()),
                    "opportunity_frequency": float(mask.mean()),
                    "forward_return_10d_count": len(returns10),
                    "forward_return_10d_mean": (
                        float(returns10.mean()) if len(returns10) else np.nan
                    ),
                    "forward_return_20d_count": len(returns20),
                    "forward_return_20d_mean": (
                        float(returns20.mean()) if len(returns20) else np.nan
                    ),
                    "h2_confirmed_minus_unconfirmed_10d_edge": conditional_edge,
                    "interpretation": "DESCRIPTIVE_ONLY_NOT_STRATEGY_RETURN",
                }
            )
    return pd.DataFrame(records)


def _lookup_lp(
    results: pd.DataFrame,
    *,
    period: str,
    horizon: int,
    term: str,
) -> pd.Series:
    selected = results.loc[
        results["period"].eq(period)
        & results["horizon_trading_days"].eq(horizon)
        & results["outcome"].eq("PRIMARY_RETURN")
        & results["term"].eq(term)
    ]
    if len(selected) != 1:
        raise ContractError(
            f"无法唯一定位局部投影结果：{period}/{horizon}/{term}"
        )
    return selected.iloc[0]


def build_factor_matrix(
    local_projection_results: pd.DataFrame,
) -> pd.DataFrame:
    specifications = [
        ("MACD_CONVERSION", "M_MACD_CONVERSION", 10, "MAIN"),
        ("BREADTH_DIFFUSION", "B_BREADTH_DIFFUSION", 20, "MAIN"),
        ("DOWNSIDE_PRESSURE", "V_DOWNSIDE_PRESSURE", 10, "MAIN"),
        ("MACD_X_BREADTH", "MACD_X_BREADTH", 10, "INTERACTION"),
        ("MACD_X_DOWNSIDE", "MACD_X_DOWNSIDE", 10, "INTERACTION"),
    ]
    records: list[dict[str, Any]] = []
    for factor, term, horizon, kind in specifications:
        full = _lookup_lp(
            local_projection_results, period="FULL", horizon=horizon, term=term
        )
        early = _lookup_lp(
            local_projection_results, period="EARLY", horizon=horizon, term=term
        )
        late = _lookup_lp(
            local_projection_results, period="LATE", horizon=horizon, term=term
        )
        full_sign = int(np.sign(full["beta"]))
        early_sign = int(np.sign(early["beta"]))
        late_sign = int(np.sign(late["beta"]))
        direction_consistent = bool(
            full_sign != 0 and full_sign == early_sign == late_sign
        )
        significance = (
            full["adjusted_p_value_holm"]
            if kind == "INTERACTION"
            else full["p_value_hac"]
        )
        significant = bool(pd.notna(significance) and significance <= 0.05)
        decaying = bool(
            early["p_value_hac"] <= 0.05
            and late["p_value_hac"] > 0.10
            and early_sign == late_sign
            and abs(late["beta"]) < 0.5 * abs(early["beta"])
        )
        if significant and direction_consistent:
            classification = "CONDITIONAL" if kind == "INTERACTION" else "STRUCTURAL"
        elif decaying:
            classification = "DECAYING"
        elif significant and not direction_consistent:
            classification = "REJECTED"
        else:
            classification = "DESCRIPTIVE_ONLY"
        records.append(
            {
                "factor": factor,
                "economic_mechanism": (
                    "信息与资金缓慢扩散"
                    if "BREADTH" in factor or factor == "MACD_CONVERSION"
                    else "风险承载能力与被迫去杠杆"
                ),
                "term": term,
                "primary_horizon_trading_days": horizon,
                "full_beta": float(full["beta"]),
                "full_p_value_hac": float(full["p_value_hac"]),
                "full_adjusted_p_value_holm": _finite_number(
                    full["adjusted_p_value_holm"]
                ),
                "early_beta": float(early["beta"]),
                "early_p_value_hac": float(early["p_value_hac"]),
                "late_beta": float(late["beta"]),
                "late_p_value_hac": float(late["p_value_hac"]),
                "direction_consistent_early_late": direction_consistent,
                "current_status": classification,
                "strategy_authorization": "NONE",
            }
        )
    records.extend(
        [
            {
                "factor": "IF_RAW_ANNUALIZED_BASIS",
                "economic_mechanism": "资金与衍生品价格发现",
                "term": "CHRONOLOGY_ONLY",
                "primary_horizon_trading_days": 10,
                "full_beta": np.nan,
                "full_p_value_hac": np.nan,
                "full_adjusted_p_value_holm": np.nan,
                "early_beta": np.nan,
                "early_p_value_hac": np.nan,
                "late_beta": np.nan,
                "late_p_value_hac": np.nan,
                "direction_consistent_early_late": False,
                "current_status": "DESCRIPTIVE_ONLY",
                "strategy_authorization": "NONE_RAW_BASIS_NOT_FAIR_BASIS",
            },
            {
                "factor": "MACRO_SLOW_PRIOR",
                "economic_mechanism": "现金流与折现率变化",
                "term": "DOSSIER_CONTEXT_ONLY",
                "primary_horizon_trading_days": 60,
                "full_beta": np.nan,
                "full_p_value_hac": np.nan,
                "full_adjusted_p_value_holm": np.nan,
                "early_beta": np.nan,
                "early_p_value_hac": np.nan,
                "late_beta": np.nan,
                "late_p_value_hac": np.nan,
                "direction_consistent_early_late": False,
                "current_status": "DESCRIPTIVE_ONLY",
                "strategy_authorization": "NONE",
            },
            {
                "factor": "INTRADAY_ETF_IF_IOPV_PRESSURE",
                "economic_mechanism": "暂时性价格压力与套利收敛",
                "term": "NOT_ESTIMATED",
                "primary_horizon_trading_days": 0,
                "full_beta": np.nan,
                "full_p_value_hac": np.nan,
                "full_adjusted_p_value_holm": np.nan,
                "early_beta": np.nan,
                "early_p_value_hac": np.nan,
                "late_beta": np.nan,
                "late_p_value_hac": np.nan,
                "direction_consistent_early_late": False,
                "current_status": "DATA_BLOCKED",
                "strategy_authorization": "NONE_NO_VIEW_DATA_CONTRACT_FAILED",
            },
        ]
    )
    return pd.DataFrame(records)


def _event_snapshot(
    panel: pd.DataFrame, event_date: pd.Timestamp
) -> dict[str, Any]:
    selected = panel.loc[panel["date"].eq(pd.Timestamp(event_date))]
    if len(selected) != 1:
        raise ContractError(f"无法唯一定位事件日状态：{event_date}")
    row = selected.iloc[0]
    columns = [
        "date",
        "close",
        "past_return_20d",
        "trend_slope_20d",
        "trend_slope_60d",
        "distance_60d_high",
        "path_efficiency_20d",
        "macd_histogram",
        "macd_velocity_3d",
        "macd_acceleration_3d",
        "downside_vol_ratio_5_20",
        "negative_jump_share_20d",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
        "official_weighted_above_ma20_share_change_5d",
        "equal_above_ma20_share",
        "equal_above_ma20_share_change_5d",
        "positive_industry_count_1d",
        "industry_count_1d",
        "top5_industry_contribution_1d",
        "top10_weight_contribution_1d",
        "constituent_return_median_1d",
        "constituent_return_dispersion_1d",
        "amount_expansion_share_1d",
        "weighted_amount_expansion_share_1d",
        "if_raw_annualized_near_basis",
        "if_basis_change_3d",
        "fund_share_change_5d",
        "financing_net_buy_5d",
        "close_premium_bps",
        "fdr007_value",
        "fdr007_change_5d",
        "pmi_new_orders_value",
        "pmi_new_orders_change_3_release",
        "tsf_stock_yoy_value",
        "tsf_stock_yoy_change_3_release",
        "usdcny_value",
        "usdcny_change_20d",
        "internal_attribution_valid",
    ]
    return {column: row[column] for column in columns}


def _outcome_snapshot(
    outcomes: pd.DataFrame, event_date: pd.Timestamp
) -> dict[str, Any]:
    selected = outcomes.loc[outcomes["date"].eq(pd.Timestamp(event_date))]
    if len(selected) != 1:
        raise ContractError(f"无法唯一定位事件结果：{event_date}")
    row = selected.iloc[0]
    columns: list[str] = ["date"]
    for horizon in (5, 10, 20, 60):
        columns.extend(
            [
                f"forward_return_{horizon}d",
                f"forward_mae_{horizon}d",
                f"forward_mfe_{horizon}d",
                f"future_rv_{horizon}d",
                f"time_to_positive_censored_{horizon}d",
                f"positive_wealth_fraction_{horizon}d",
                f"outcome_complete_{horizon}d",
            ]
        )
    return {column: row[column] for column in columns}


def build_case_control_comparison(
    cases: pd.DataFrame,
    matches: pd.DataFrame,
    outcomes: pd.DataFrame,
    chronology: pd.DataFrame,
) -> dict[str, Any]:
    case_dates = pd.DatetimeIndex(cases["case_date"])
    control_dates = pd.DatetimeIndex(matches["control_date"])
    result: dict[str, Any] = {
        "warning": (
            "失败对照按forward_return_10d<=0选取；差异仅用于机制尸检，"
            "不是无偏效应、因果效应或样本外策略收益。"
        )
    }
    for horizon in (5, 10, 20):
        column = f"forward_return_{horizon}d"
        case_values = outcomes.set_index("date").loc[case_dates, column].dropna()
        control_values = outcomes.set_index("date").loc[control_dates, column].dropna()
        result[f"horizon_{horizon}d"] = {
            "case_count": len(case_values),
            "control_count": len(control_values),
            "case_mean": _finite_number(case_values.mean()),
            "control_mean": _finite_number(control_values.mean()),
            "case_minus_control_mean": _finite_number(
                case_values.mean() - control_values.mean()
            ),
            "case_median": _finite_number(case_values.median()),
            "control_median": _finite_number(control_values.median()),
        }
    case_chronology = chronology.loc[chronology["unit_type"].eq("CASE")]
    control_chronology = chronology.loc[
        chronology["unit_type"].eq("FAILED_CONTROL")
    ]
    result["chronology"] = {
        "case_all_links_observed": int(case_chronology["all_links_observed"].sum()),
        "case_canonical_order_observed": int(
            case_chronology["canonical_order_observed"].sum()
        ),
        "control_all_links_observed": int(
            control_chronology["all_links_observed"].sum()
        ),
        "control_canonical_order_observed": int(
            control_chronology["canonical_order_observed"].sum()
        ),
    }
    return result


def _format_percent(value: Any, digits: int = 2) -> str:
    number = _finite_number(value)
    return "NA" if number is None else f"{number:.{digits}%}"


def _format_number(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _format_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NOT_OBSERVED"
    return pd.Timestamp(value).date().isoformat()


def render_case_dossier(
    case_row: pd.Series,
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    matches: pd.DataFrame,
    windows: pd.DataFrame,
    chronology: pd.DataFrame,
) -> str:
    case_id = str(case_row["case_id"])
    case_date = pd.Timestamp(case_row["case_date"])
    state = _event_snapshot(panel, case_date)
    outcome = _outcome_snapshot(outcomes, case_date)
    case_chronology = chronology.loc[chronology["unit_id"].eq(case_id)].iloc[0]
    case_matches = matches.loc[matches["case_id"].eq(case_id)].sort_values(
        "control_rank"
    )
    lines = [
        f"# {case_id}：{case_date.date().isoformat()} 机制尸检",
        "",
        "- 身份：冻结来源分支的高质量收敛病例；不是成功交易标签。",
        "- 研究边界：只做回顾性机制发现，不生成仓位或订单。",
        f"- 10日结果：`{_format_percent(outcome['forward_return_10d'])}`；20日结果：`{_format_percent(outcome['forward_return_20d'])}`。",
        "",
        "## 事件日状态",
        "",
        "| 模块 | 指标 | 数值 |",
        "|---|---|---:|",
        f"| 价格风险 | 过去20日总收益 | {_format_percent(state['past_return_20d'])} |",
        f"| 价格风险 | 20日趋势斜率 | {_format_number(state['trend_slope_20d'], 6)} |",
        f"| 价格风险 | 距60日高点 | {_format_percent(state['distance_60d_high'])} |",
        f"| 价格风险 | MACD柱 | {_format_number(state['macd_histogram'], 6)} |",
        f"| 价格风险 | MACD三日速度 | {_format_number(state['macd_velocity_3d'], 6)} |",
        f"| 价格风险 | 下行波动5/20比 | {_format_number(state['downside_vol_ratio_5_20'], 4)} |",
        f"| 内部结构 | 官方权重MA20广度 | {_format_percent(state['official_weighted_above_ma20_share'])} |",
        f"| 内部结构 | 官方权重MA20广度五日变化 | {_format_percent(state['official_weighted_above_ma20_share_change_5d'])} |",
        f"| 内部结构 | 等权MA20广度 | {_format_percent(state['equal_above_ma20_share'])} |",
        f"| 内部结构 | 正收益行业数/行业数 | {int(state['positive_industry_count_1d']) if pd.notna(state['positive_industry_count_1d']) else 'NA'}/{int(state['industry_count_1d']) if pd.notna(state['industry_count_1d']) else 'NA'} |",
        f"| 内部结构 | 成分股收益中位数 | {_format_percent(state['constituent_return_median_1d'])} |",
        f"| 内部结构 | 官方权重成交额扩散 | {_format_percent(state['weighted_amount_expansion_share_1d'])} |",
        f"| 资金确认 | IF原始年化近月基差 | {_format_percent(state['if_raw_annualized_near_basis'])} |",
        f"| 资金确认 | IF原始基差三日变化 | {_format_percent(state['if_basis_change_3d'])} |",
        f"| 资金确认 | ETF份额五日变化 | {_format_percent(state['fund_share_change_5d'])} |",
        f"| 资金确认 | 收盘NAV折溢价 | {_format_number(state['close_premium_bps'], 2)} bp |",
        f"| 慢速先验 | PMI新订单 | {_format_number(state['pmi_new_orders_value'], 2)} |",
        f"| 慢速先验 | 社融存量同比 | {_format_number(state['tsf_stock_yoy_value'], 2)} |",
        f"| 慢速先验 | FDR007 | {_format_number(state['fdr007_value'], 4)} |",
        f"| 慢速先验 | USD/CNY中间价20日变化 | {_format_percent(state['usdcny_change_20d'])} |",
        "",
        "## 传导顺序",
        "",
        "| 环节 | 日期/状态 |",
        "|---|---|",
        f"| 风险压力峰值 | {_format_date(case_chronology['risk_peak_date'])} |",
        f"| 风险压力至事件日下降 | {_format_percent(case_chronology['risk_relief_fraction_at_event'])} |",
        f"| IF原始基差确认 | {_format_date(case_chronology['if_confirmation_date'])} |",
        f"| 官方权重广度转正 | {_format_date(case_chronology['breadth_turn_date'])} |",
        f"| MACD收敛事件 | {_format_date(case_chronology['macd_turn_date'])} |",
        f"| 价格趋势确认 | {_format_date(case_chronology['price_trend_confirmation_date'])} |",
        f"| 标准顺序完整出现 | `{str(bool(case_chronology['canonical_order_observed'])).lower()}` |",
        "",
        "## 三个不放回失败对照",
        "",
        "| 对照 | 日期 | 匹配距离 | 冻结10日结果 |",
        "|---|---|---:|---:|",
    ]
    for row in case_matches.itertuples(index=False):
        lines.append(
            f"| {row.control_id} | {pd.Timestamp(row.control_date).date().isoformat()} | "
            f"{row.matching_distance:.3f} | {_format_percent(row.control_forward_return_10d_frozen)} |"
        )
    lines.extend(
        [
            "",
            "## 固定观察点",
            "",
            "| 相对交易日 | MACD速度 | 官方广度五日变化 | 下行波动比 | IF基差三日变化 | 价格相对事件日 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    case_window = windows.loc[windows["unit_id"].eq(case_id)].copy()
    event_close = float(
        case_window.loc[case_window["relative_trading_day"].eq(0), "close"].iloc[0]
    )
    for relative_day in (-60, -20, 0, 5, 10, 20):
        row = case_window.loc[
            case_window["relative_trading_day"].eq(relative_day)
        ].iloc[0]
        lines.append(
            f"| {relative_day:+d} | {_format_number(row['macd_velocity_3d'], 5)} | "
            f"{_format_percent(row['official_weighted_above_ma20_share_change_5d'])} | "
            f"{_format_number(row['downside_vol_ratio_5_20'], 3)} | "
            f"{_format_percent(row['if_basis_change_3d'])} | "
            f"{_format_percent(float(row['close']) / event_close - 1.0)} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- IF字段是未完成点时carry扣除的原始年化基差，不是公平基差。",
            "- 对照池按十日不涨定义，病例与对照差异不能解释为因果效应。",
            "- 本档案不判断买卖，不授权Paper、Shadow或实盘。",
            "",
        ]
    )
    return "\n".join(lines)


def render_dossier_index(
    cases: pd.DataFrame,
    chronology: pd.DataFrame,
) -> str:
    lines = [
        "# 510300 机制图谱 V1：六病例索引",
        "",
        "这六个病例来自已冻结拒绝的MACD×广度×下行波动预检，只用于机制尸检。",
        "",
        "| 病例 | 日期 | 10日结果 | 20日结果 | 标准传导顺序 |",
        "|---|---|---:|---:|---|",
    ]
    for row in cases.itertuples(index=False):
        chronology_row = chronology.loc[chronology["unit_id"].eq(row.case_id)].iloc[0]
        lines.append(
            f"| {row.case_id} | {pd.Timestamp(row.case_date).date().isoformat()} | "
            f"{_format_percent(row.forward_return_10d)} | "
            f"{_format_percent(row.forward_return_20d)} | "
            f"`{str(bool(chronology_row['canonical_order_observed'])).lower()}` |"
        )
    lines.extend(
        [
            "",
            "每个病例文件包含事件日状态、传导日期、三个不放回失败对照和固定观察点。",
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    panel: pd.DataFrame,
    cases: pd.DataFrame,
    matches: pd.DataFrame,
    matching_balance: dict[str, Any],
    chronology: pd.DataFrame,
    local_projection_results: pd.DataFrame,
    local_projection_audit: dict[str, Any],
    opportunity_frequency: pd.DataFrame,
    factor_matrix: pd.DataFrame,
    case_control_comparison: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    key_lp = local_projection_results.loc[
        local_projection_results["period"].eq("FULL")
        & local_projection_results["outcome"].eq("PRIMARY_RETURN")
        & local_projection_results["term"].isin(
            [
                "M_MACD_CONVERSION",
                "B_BREADTH_DIFFUSION",
                "V_DOWNSIDE_PRESSURE",
                "MACD_X_BREADTH",
                "MACD_X_DOWNSIDE",
            ]
        )
    ][
        [
            "horizon_trading_days",
            "term",
            "beta",
            "standard_error_hac",
            "p_value_hac",
            "adjusted_p_value_holm",
            "observations",
        ]
    ].sort_values(["horizon_trading_days", "term"])
    opportunity_all = opportunity_frequency.loc[
        opportunity_frequency["period"].eq("ALL")
    ]
    return {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": config["adjudication"]["completion_status"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "evidence_class": config["protocol"]["evidence_class"],
        "source_branch": {
            "project_id": config["source_branch"]["project_id"],
            "status": config["source_branch"]["required_status"],
            "remains_rejected": True,
            "reuse_rule": config["source_branch"]["reuse_rule"],
        },
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "return_backtest_allowed": False,
            "event_outcome_diagnostics_allowed": True,
        },
        "input_audit": audit,
        "freeze": manifest,
        "state_panel": {
            "rows": len(panel),
            "first_date": panel["date"].min().date().isoformat(),
            "last_date": panel["date"].max().date().isoformat(),
            "columns": len(panel.columns),
            "valid_internal_attribution_days": int(
                panel["internal_attribution_valid"].fillna(False).astype(bool).sum()
            ),
            "valid_macro_slow_prior_days": int(
                panel["macro_slow_prior_valid"].fillna(False).astype(bool).sum()
            ),
        },
        "matching": {
            "balance": matching_balance,
            "assignments": matches[
                [
                    "case_id",
                    "case_date",
                    "control_id",
                    "control_date",
                    "matching_distance",
                ]
            ].to_dict(orient="records"),
        },
        "case_control_comparison": case_control_comparison,
        "chronology": {
            "case_records": chronology.loc[
                chronology["unit_type"].eq("CASE")
            ].to_dict(orient="records"),
            "canonical_case_count": int(
                chronology.loc[
                    chronology["unit_type"].eq("CASE"),
                    "canonical_order_observed",
                ].sum()
            ),
        },
        "continuous_response": {
            "audit": local_projection_audit,
            "key_primary_return_coefficients": key_lp.to_dict(orient="records"),
        },
        "opportunity_frequency_all": opportunity_all.to_dict(orient="records"),
        "factor_matrix": factor_matrix.to_dict(orient="records"),
        "mechanism_hypotheses": config["mechanism_hypotheses"],
        "data_gaps": config["excluded_or_blocked_inputs"],
        "artifacts": artifact_hashes,
        "adjudication": {
            "return_evaluation": "NOT_ALLOWED",
            "portfolio_backtest": "NOT_RUN",
            "net_sharpe": "NOT_COMPUTED",
            "target_net_sharpe": config["downstream_acceptance_not_run_in_v1"][
                "target_net_sharpe"
            ],
            "cumulative_net_excess": "NOT_COMPUTED",
            "target_cumulative_net_excess": config[
                "downstream_acceptance_not_run_in_v1"
            ]["target_cumulative_net_excess"],
            "target_achieved": False,
            "reason": (
                "机制图谱只完成发现与传导诊断；任何机制若要转成策略，必须使用新冻结ID、"
                "严格样本外和至少252个新Shadow交易日。"
            ),
        },
        "boundaries": {
            "research_only": True,
            "paper_or_shadow_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
            "current_signal": "NOT_CREATED",
            "no_trade": True,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    matching = report["matching"]["balance"]
    comparison = report["case_control_comparison"]
    factors = report["factor_matrix"]
    lines = [
        "# 510300 机制图谱 V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 来源分支：`{report['source_branch']['status']}`（继续冻结拒绝）",
        f"- 共同窗口：`{report['state_panel']['first_date']}—{report['state_panel']['last_date']}`",
        f"- 状态面板：`{report['state_panel']['rows']}`个交易日、`{report['state_panel']['columns']}`个字段",
        "- 交易域：仅`510300.SH`与人民币现金；本报告不生成策略净值、仓位或订单。",
        "",
        "## 六病例与失败对照",
        "",
        f"- 病例：`{matching['case_count']}`；不放回失败对照：`{matching['assigned_control_count']}`。",
        f"- 最大匹配距离：`{matching['maximum_pair_distance']:.3f}`；最大绝对标准化均值差：`{matching['maximum_absolute_standardized_mean_difference']:.3f}`。",
        f"- 标准五环传导顺序完整出现的病例：`{report['chronology']['canonical_case_count']}/6`。",
        "- 失败对照按冻结十日结果不为正选择；下表差异只用于尸检，不能解释为因果或样本外收益。",
        "",
        "| 期限 | 病例均值 | 对照均值 | 差异 |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in (5, 10, 20):
        item = comparison[f"horizon_{horizon}d"]
        lines.append(
            f"| {horizon}日 | {_format_percent(item['case_mean'])} | "
            f"{_format_percent(item['control_mean'])} | "
            f"{_format_percent(item['case_minus_control_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 连续条件响应分类",
            "",
            "| 因子/交互 | 主要期限 | 全样本系数 | HAC p值 | Holm p值 | 早/晚方向一致 | 当前状态 |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for factor in factors:
        lines.append(
            f"| {factor['factor']} | {factor['primary_horizon_trading_days']} | "
            f"{_format_number(factor['full_beta'], 6)} | "
            f"{_format_number(factor['full_p_value_hac'], 4)} | "
            f"{_format_number(factor['full_adjusted_p_value_holm'], 4)} | "
            f"`{str(bool(factor['direction_consistent_early_late'])).lower()}` | "
            f"`{factor['current_status']}` |"
        )
    lines.extend(
        [
            "",
            "## 固定数据缺口",
            "",
            "- `IF_FAIR_BASIS`：缺少完整点时carry，现有字段只能称原始年化基差。",
            "- `OPTION_SKEW_TERM_STRUCTURE`：V1未准入，不能用短覆盖或失败分支补洞。",
            "- `PCF_IOPV_HISTORY`：质量日不足，不能覆盖六病例和长期对照。",
            "- `CSI300_EARNINGS_BREADTH`：点时数据契约缺失。",
            "- `H4_INTRADAY_TEMPORARY_PRESSURE`：`NO_VIEW_DATA_CONTRACT_FAILED`。",
            "",
            "## 裁决",
            "",
            "- 组合收益评估：`NOT_ALLOWED`",
            "- 组合回测：`NOT_RUN`",
            "- 净夏普率：`NOT_COMPUTED`",
            "- 累计净超额：`NOT_COMPUTED`",
            "- 夏普1.2与累计净超额20%：均未在本版本评估，不能声称达到。",
            "- 若未来有机制通过，必须建立新的冻结策略ID，并完成严格样本外与至少252个新Shadow交易日。",
            "- Paper、Shadow仓位映射、订单、券商连接和实盘：全部关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def write_input_audit(config: dict[str, Any], audit: dict[str, Any]) -> Path:
    path = _project_path(config["paths"]["input_audit"])
    _atomic_json(path, audit)
    return path


def run_atlas(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    inputs = load_inputs(config)
    audit = audit_inputs(config, inputs)
    panel = build_state_panel(config, inputs)
    outcomes = build_outcome_panel(panel, config)
    cases, matches, matching_balance = select_cases_and_match_controls(
        panel, outcomes, inputs["source_events"], config
    )
    windows = build_event_window_panel(panel, cases, matches, config)
    chronology = build_chronology_table(windows, config)
    local_projection_results, rolling, local_projection_audit = (
        run_local_projections(panel, outcomes, config)
    )
    opportunity_frequency = build_opportunity_frequency(panel, outcomes)
    factor_matrix = build_factor_matrix(local_projection_results)
    case_control_comparison = build_case_control_comparison(
        cases, matches, outcomes, chronology
    )

    artifact_hashes: dict[str, str] = {}
    if write:
        path_mapping = {
            "state_panel": panel,
            "outcome_panel": outcomes,
            "matched_controls": matches,
            "event_window_panel": windows,
            "local_projection_results": local_projection_results,
            "rolling_coefficients": rolling,
            "opportunity_frequency": opportunity_frequency,
            "factor_matrix": factor_matrix,
        }
        for key, frame in path_mapping.items():
            path = _project_path(config["paths"][key])
            _atomic_parquet(path, frame)
            artifact_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)

        dossier_directory = _project_path(config["paths"]["dossier_directory"])
        dossier_directory.mkdir(parents=True, exist_ok=True)
        for _, case_row in cases.iterrows():
            filename = (
                f"{case_row['case_id']}_{pd.Timestamp(case_row['case_date']).date().isoformat()}.md"
            )
            dossier_path = dossier_directory / filename
            _atomic_text(
                dossier_path,
                render_case_dossier(
                    case_row,
                    panel,
                    outcomes,
                    matches,
                    windows,
                    chronology,
                ),
            )
            artifact_hashes[dossier_path.relative_to(ROOT).as_posix()] = sha256_file(
                dossier_path
            )
        index_path = dossier_directory / "INDEX.md"
        _atomic_text(index_path, render_dossier_index(cases, chronology))
        artifact_hashes[index_path.relative_to(ROOT).as_posix()] = sha256_file(
            index_path
        )

    report = build_report(
        config,
        manifest,
        audit,
        panel,
        cases,
        matches,
        matching_balance,
        chronology,
        local_projection_results,
        local_projection_audit,
        opportunity_frequency,
        factor_matrix,
        case_control_comparison,
        artifact_hashes,
    )
    if write:
        result_json = _project_path(config["paths"]["result_json"])
        result_markdown = _project_path(config["paths"]["result_markdown"])
        _atomic_json(result_json, report)
        _atomic_text(result_markdown, render_markdown(report))
    return report


__all__ = [
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "ContractError",
    "audit_inputs",
    "build_chronology_table",
    "build_event_window_panel",
    "build_factor_matrix",
    "build_opportunity_frequency",
    "build_outcome_panel",
    "build_state_panel",
    "load_config",
    "load_inputs",
    "run_atlas",
    "run_local_projections",
    "select_cases_and_match_controls",
    "sha256_file",
    "validate_config",
    "validate_manifest",
    "write_input_audit",
]
