"""行业预期差 V1 的追加式前瞻成熟度与结果评价。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


class ForwardEvaluationError(ValueError):
    """前瞻评价输入违反冻结边界。"""


@dataclass(frozen=True)
class EvaluationRules:
    horizons_trading_days: tuple[int, ...]
    minimum_index_endpoint_weight_coverage: float
    minimum_industry_endpoint_weight_coverage_ratio: float
    minimum_fundamental_core_coverage_ratio: float
    missing_component_reweighting_allowed: bool
    partial_horizon_return_output_allowed: bool
    endpoint_carry_forward_allowed: bool
    balanced_gap_scored: bool
    unobserved_gap_scored: bool
    mixed_fundamental_scored: bool
    minimum_mature_prediction_origins_for_calibration: int
    minimum_mature_prediction_origins_for_model_comparison: int
    append_only_outputs: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationRules":
        required = set(cls.__dataclass_fields__)
        missing = required - set(raw)
        if missing:
            raise ForwardEvaluationError(f"评价规则缺少字段：{sorted(missing)}")
        values = dict(raw)
        horizons = tuple(int(value) for value in values["horizons_trading_days"])
        if not horizons or any(value <= 0 for value in horizons):
            raise ForwardEvaluationError("评价期限必须是正交易日数")
        if len(horizons) != len(set(horizons)):
            raise ForwardEvaluationError("评价期限不能重复")
        values["horizons_trading_days"] = horizons
        return cls(**{name: values[name] for name in required})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, field_name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover
        raise ForwardEvaluationError(f"{field_name} 不是合法日期：{value}") from exc
    if pd.isna(result):
        raise ForwardEvaluationError(f"{field_name} 不能为空")
    return result


def _workspace_path(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ForwardEvaluationError(f"路径越出工作区：{relative_path}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ForwardEvaluationError(f"JSON 顶层必须是对象：{path}")
    return value


def validate_evaluation_config(
    config: Mapping[str, Any],
) -> tuple[EvaluationRules, dict[str, bool]]:
    """验证评价配置没有开放偷看、回填或交易权限。"""

    if config.get("status") != "FROZEN_BEFORE_FIRST_FORWARD_OBSERVATION":
        raise ForwardEvaluationError("评价配置状态非法")
    if config.get("asset") != "510300" or config.get("benchmark") != "000300":
        raise ForwardEvaluationError("评价对象必须为 510300/000300")
    if _timestamp(config.get("entry_date"), "entry_date") <= _timestamp(
        config.get("prediction_date"), "prediction_date"
    ):
        raise ForwardEvaluationError("入场日必须晚于预测日")

    rules = EvaluationRules.from_mapping(config.get("rules", {}))
    forbidden_rule_values = {
        "missing_component_reweighting_allowed": rules.missing_component_reweighting_allowed,
        "partial_horizon_return_output_allowed": rules.partial_horizon_return_output_allowed,
        "endpoint_carry_forward_allowed": rules.endpoint_carry_forward_allowed,
        "balanced_gap_scored": rules.balanced_gap_scored,
        "unobserved_gap_scored": rules.unobserved_gap_scored,
        "mixed_fundamental_scored": rules.mixed_fundamental_scored,
    }
    enabled = [name for name, state in forbidden_rule_values.items() if state]
    if enabled:
        raise ForwardEvaluationError(f"评价配置禁止启用：{enabled}")
    if not rules.append_only_outputs:
        raise ForwardEvaluationError("评价输出必须为追加式")

    safety = dict(config.get("safety", {}))
    required_safety = {
        "may_upgrade_original_no_view",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    }
    missing_safety = required_safety - set(safety)
    if missing_safety:
        raise ForwardEvaluationError(f"安全边界缺失：{sorted(missing_safety)}")
    unsafe = [name for name in required_safety if safety[name] is not False]
    if unsafe:
        raise ForwardEvaluationError(f"安全边界必须全部为 false：{unsafe}")
    return rules, safety


def verify_hash_manifest(
    workspace_root: Path,
    manifest_relative_path: str,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """复核清单本身及其所有冻结文件。"""

    manifest_path = _workspace_path(workspace_root, manifest_relative_path)
    if not manifest_path.is_file():
        raise ForwardEvaluationError(f"冻结清单不存在：{manifest_path}")
    actual_manifest_hash = file_sha256(manifest_path)
    if expected_manifest_sha256 and actual_manifest_hash.upper() != expected_manifest_sha256.upper():
        raise ForwardEvaluationError("原预测冻结清单哈希发生变化")
    manifest = _load_json(manifest_path)
    for item in manifest.get("frozen_files", []):
        path = _workspace_path(workspace_root, str(item["path"]))
        if not path.is_file():
            raise ForwardEvaluationError(f"冻结文件缺失：{item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ForwardEvaluationError(f"冻结文件哈希不一致：{item['path']}")
    return manifest


def load_trading_calendar(calendar_frames: Sequence[pd.DataFrame]) -> pd.DatetimeIndex:
    """合并年度交易日历并拒绝重复或无序日期。"""

    if not calendar_frames:
        raise ForwardEvaluationError("没有交易日历")
    pieces: list[pd.Series] = []
    for frame in calendar_frames:
        if "trade_date" not in frame.columns:
            raise ForwardEvaluationError("交易日历缺少 trade_date")
        values = pd.to_datetime(frame["trade_date"], errors="coerce")
        if values.isna().any():
            raise ForwardEvaluationError("交易日历包含非法日期")
        pieces.append(values)
    combined = pd.concat(pieces, ignore_index=True)
    if combined.duplicated().any():
        duplicates = combined.loc[combined.duplicated()].dt.date.astype(str).tolist()
        raise ForwardEvaluationError(f"交易日历包含重复日期：{duplicates[:5]}")
    calendar = pd.DatetimeIndex(sorted(combined.unique()))
    if any(value.weekday() >= 5 for value in calendar):
        raise ForwardEvaluationError("交易日历包含周末日期")
    return calendar


def resolve_maturity_schedule(
    calendar: pd.DatetimeIndex,
    entry_date: Any,
    horizons: Sequence[int],
    pre_resolved: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按入场日为第1个交易日解析各期限成熟日。"""

    entry = _timestamp(entry_date, "entry_date").normalize()
    if entry not in calendar:
        raise ForwardEvaluationError(f"入场日不在已核验交易日历中：{entry.date()}")
    entry_position = int(calendar.get_loc(entry))
    frozen = dict(pre_resolved or {})
    schedule: list[dict[str, Any]] = []
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        maturity_position = entry_position + horizon - 1
        if maturity_position >= len(calendar):
            maturity_date = None
            calendar_state = "UNRESOLVED_CALENDAR"
            missing_calendar_days = maturity_position - len(calendar) + 1
        else:
            maturity_date = pd.Timestamp(calendar[maturity_position]).date().isoformat()
            calendar_state = "RESOLVED"
            missing_calendar_days = 0
        frozen_date = frozen.get(str(horizon))
        if frozen_date is not None and maturity_date != str(frozen_date):
            raise ForwardEvaluationError(
                f"{horizon}日成熟日与预解析结果不一致：{maturity_date} != {frozen_date}"
            )
        schedule.append(
            {
                "horizon_trading_days": horizon,
                "entry_date": entry.date().isoformat(),
                "maturity_date": maturity_date,
                "calendar_state": calendar_state,
                "missing_calendar_days": missing_calendar_days,
            }
        )
    return schedule


def _validate_daily_dates(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ForwardEvaluationError(f"{name} 缺少 date")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ForwardEvaluationError(f"{name} 包含非法日期")
    return result


def determine_observation_state(
    prediction_date: Any,
    entry_date: Any,
    schedule: Sequence[Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    constituent_daily: pd.DataFrame,
    etf_daily: pd.DataFrame,
) -> dict[str, Any]:
    """只根据日期覆盖判定成熟度，不计算未成熟收益。"""

    constituent = _validate_daily_dates(constituent_daily, "成分总收益日线")
    etf = _validate_daily_dates(etf_daily, "ETF总收益日线")
    if constituent[["date", "con_code"]].duplicated().any():
        raise ForwardEvaluationError("成分总收益日线存在重复证券日期")
    if etf["date"].duplicated().any():
        raise ForwardEvaluationError("ETF总收益日线存在重复日期")

    prediction = _timestamp(prediction_date, "prediction_date").normalize()
    entry = _timestamp(entry_date, "entry_date").normalize()
    constituent_dates = pd.DatetimeIndex(sorted(constituent["date"].unique()))
    etf_dates = pd.DatetimeIndex(sorted(etf["date"].unique()))
    common_dates = calendar.intersection(constituent_dates).intersection(etf_dates)
    forward_common = common_dates[common_dates >= entry]
    common_latest = pd.Timestamp(forward_common.max()) if len(forward_common) else None
    observation_date = max(prediction, common_latest) if common_latest is not None else prediction

    horizon_states: list[dict[str, Any]] = []
    for item in schedule:
        horizon = dict(item)
        maturity_raw = horizon["maturity_date"]
        if horizon["calendar_state"] != "RESOLVED" or maturity_raw is None:
            state = "UNRESOLVED_CALENDAR"
        elif common_latest is None or common_latest < entry:
            state = "WAITING_FOR_ENTRY"
        elif common_latest < pd.Timestamp(maturity_raw):
            state = "ACCUMULATING_NO_PEEK"
        else:
            state = "MATURE_PENDING_SCORE"
        if common_latest is None:
            observed_days = 0
        else:
            maturity_limit = (
                pd.Timestamp(maturity_raw) if maturity_raw is not None else common_latest
            )
            observed_days = int(
                len(forward_common[forward_common <= min(common_latest, maturity_limit)])
            )
        horizon["state"] = state
        horizon["observed_common_forward_trading_days"] = observed_days
        horizon_states.append(horizon)
    return {
        "observation_date": observation_date.date().isoformat(),
        "constituent_latest_date": constituent["date"].max().date().isoformat(),
        "etf_latest_date": etf["date"].max().date().isoformat(),
        "common_forward_latest_date": (
            common_latest.date().isoformat() if common_latest is not None else None
        ),
        "horizons": horizon_states,
    }


def prepare_active_industry_map(
    industry_intervals: pd.DataFrame,
    classification_date: Any,
) -> pd.DataFrame:
    """取得冻结时点有效的中信一级行业归属。"""

    required = {"ts_code", "l1_name", "in_date", "out_date"}
    missing = required - set(industry_intervals.columns)
    if missing:
        raise ForwardEvaluationError(f"行业区间缺少字段：{sorted(missing)}")
    frame = industry_intervals.copy()
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce")
    if frame["in_date"].isna().any():
        raise ForwardEvaluationError("行业区间包含非法起始日")
    cutoff = _timestamp(classification_date, "classification_date").normalize()
    active = frame.loc[
        frame["in_date"].le(cutoff)
        & (frame["out_date"].isna() | frame["out_date"].gt(cutoff)),
        ["ts_code", "l1_name", "in_date", "out_date"],
    ].copy()
    if active["ts_code"].duplicated().any():
        raise ForwardEvaluationError("冻结时点存在重复有效行业归属")
    return active.rename(columns={"ts_code": "con_code", "l1_name": "industry_l1"})


def score_price_horizon(
    horizon: int,
    entry_date: Any,
    maturity_date: Any,
    weight_snapshot_date: Any,
    historical_weights: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    forecast_rows: Sequence[Mapping[str, Any]],
    rules: EvaluationRules,
) -> dict[str, Any]:
    """在成熟后计算固定成分权重行业相对收益和预期差方向。"""

    required_weight = {"con_code", "trade_date", "weight"}
    if required_weight - set(historical_weights.columns):
        raise ForwardEvaluationError("历史权重缺少必要字段")
    weights = historical_weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"], errors="coerce")
    snapshot_date = _timestamp(weight_snapshot_date, "weight_snapshot_date").normalize()
    weights = weights.loc[weights["trade_date"].eq(snapshot_date), ["con_code", "weight"]].copy()
    if weights.empty or weights["con_code"].duplicated().any():
        raise ForwardEvaluationError("冻结权重截面为空或成分重复")
    weights["snapshot_weight"] = pd.to_numeric(weights["weight"], errors="coerce") / 100.0
    if weights["snapshot_weight"].isna().any() or (weights["snapshot_weight"] < 0).any():
        raise ForwardEvaluationError("冻结成分权重非法")

    active = prepare_active_industry_map(industry_intervals, snapshot_date)
    cross = weights[["con_code", "snapshot_weight"]].merge(
        active[["con_code", "industry_l1"]], on="con_code", how="left", validate="one_to_one"
    )
    daily = _validate_daily_dates(constituent_daily, "成分总收益日线")
    required_daily = {"con_code", "total_return_open", "total_return_close"}
    if required_daily - set(daily.columns):
        raise ForwardEvaluationError("成分总收益日线缺少必要价格字段")
    entry = _timestamp(entry_date, "entry_date").normalize()
    maturity = _timestamp(maturity_date, "maturity_date").normalize()
    entry_frame = daily.loc[
        daily["date"].eq(entry), ["con_code", "total_return_open"]
    ].rename(columns={"total_return_open": "entry_total_return_open"})
    maturity_frame = daily.loc[
        daily["date"].eq(maturity), ["con_code", "total_return_close"]
    ].rename(columns={"total_return_close": "maturity_total_return_close"})
    cross = cross.merge(entry_frame, on="con_code", how="left", validate="one_to_one")
    cross = cross.merge(maturity_frame, on="con_code", how="left", validate="one_to_one")
    cross["entry_total_return_open"] = pd.to_numeric(
        cross["entry_total_return_open"], errors="coerce"
    )
    cross["maturity_total_return_close"] = pd.to_numeric(
        cross["maturity_total_return_close"], errors="coerce"
    )
    valid_entry = cross["entry_total_return_open"].gt(0)
    valid_endpoint = cross["maturity_total_return_close"].gt(0)
    valid = valid_entry & valid_endpoint
    entry_coverage = float(cross.loc[valid_entry, "snapshot_weight"].sum())
    endpoint_coverage = float(cross.loc[valid_endpoint, "snapshot_weight"].sum())
    joint_coverage = float(cross.loc[valid, "snapshot_weight"].sum())
    if min(entry_coverage, endpoint_coverage, joint_coverage) < rules.minimum_index_endpoint_weight_coverage:
        return {
            "status": "MATURE_DATA_INCOMPLETE",
            "failure_category": "INDEX_ENDPOINT_WEIGHT_COVERAGE_BELOW_GATE",
            "horizon_trading_days": horizon,
            "entry_date": entry.date().isoformat(),
            "maturity_date": maturity.date().isoformat(),
            "entry_weight_coverage": entry_coverage,
            "endpoint_weight_coverage": endpoint_coverage,
            "joint_weight_coverage": joint_coverage,
            "industry_rows": [],
        }

    cross["component_return"] = (
        cross["maturity_total_return_close"] / cross["entry_total_return_open"] - 1.0
    ).where(valid)
    cross["component_contribution"] = cross["snapshot_weight"] * cross["component_return"]
    fixed_basket_return = float(cross["component_contribution"].sum(min_count=1))

    forecast = pd.DataFrame(list(forecast_rows))[
        ["industry_l1", "sector_weight", "expectation_gap"]
    ].copy()
    if forecast["industry_l1"].duplicated().any():
        raise ForwardEvaluationError("冻结预测包含重复行业")
    mapped = cross.loc[cross["industry_l1"].notna()].copy()
    group_rows: list[dict[str, Any]] = []
    for prediction in forecast.itertuples(index=False):
        sector = mapped.loc[mapped["industry_l1"].eq(prediction.industry_l1)]
        frozen_component_weight = float(sector["snapshot_weight"].sum())
        valid_weight = float(sector.loc[valid.loc[sector.index], "snapshot_weight"].sum())
        coverage_ratio = valid_weight / frozen_component_weight if frozen_component_weight else 0.0
        contribution = float(sector["component_contribution"].sum(min_count=1))
        if coverage_ratio < rules.minimum_industry_endpoint_weight_coverage_ratio:
            row_state = "NO_VIEW"
            industry_return = None
            relative_return = None
            direction_hit = None
            failure = "INDUSTRY_ENDPOINT_WEIGHT_COVERAGE_BELOW_GATE"
        else:
            industry_return = contribution / frozen_component_weight
            relative_return = industry_return - fixed_basket_return
            failure = "PASS"
            if prediction.expectation_gap == "POSITIVE":
                row_state = "SCORED"
                direction_hit = bool(relative_return > 0)
            elif prediction.expectation_gap == "NEGATIVE":
                row_state = "SCORED"
                direction_hit = bool(relative_return < 0)
            else:
                row_state = "ABSTAINED_AT_FORECAST"
                direction_hit = None
        group_rows.append(
            {
                "industry_l1": prediction.industry_l1,
                "forecast_sector_weight": float(prediction.sector_weight),
                "component_weight": frozen_component_weight,
                "endpoint_coverage_ratio": coverage_ratio,
                "expectation_gap": prediction.expectation_gap,
                "industry_return": industry_return,
                "fixed_basket_return": fixed_basket_return,
                "relative_return": relative_return,
                "direction_hit": direction_hit,
                "state": row_state,
                "failure_category": failure,
            }
        )

    scored = pd.DataFrame(group_rows)
    scored_rows = scored.loc[scored["state"].eq("SCORED")].copy()
    if scored_rows.empty:
        accuracy = None
        weighted_accuracy = None
    else:
        accuracy = float(scored_rows["direction_hit"].astype(float).mean())
        weights_for_score = scored_rows["forecast_sector_weight"].astype(float)
        weighted_accuracy = float(
            (scored_rows["direction_hit"].astype(float) * weights_for_score).sum()
            / weights_for_score.sum()
        )

    def _group_relative_return(gap_state: str) -> tuple[float | None, float | None]:
        selected = scored.loc[
            scored["expectation_gap"].eq(gap_state) & scored["relative_return"].notna()
        ]
        if selected.empty:
            return None, None
        equal_weighted = float(selected["relative_return"].mean())
        weights_selected = selected["forecast_sector_weight"].astype(float)
        frozen_weighted = float(
            (selected["relative_return"].astype(float) * weights_selected).sum()
            / weights_selected.sum()
        )
        return equal_weighted, frozen_weighted

    positive_equal, positive_weighted = _group_relative_return("POSITIVE")
    negative_equal, negative_weighted = _group_relative_return("NEGATIVE")
    long_short_equal = (
        positive_equal - negative_equal
        if positive_equal is not None and negative_equal is not None
        else None
    )
    long_short_weighted = (
        positive_weighted - negative_weighted
        if positive_weighted is not None and negative_weighted is not None
        else None
    )
    return {
        "status": "MATURE_SCORED",
        "failure_category": "PASS",
        "horizon_trading_days": horizon,
        "entry_date": entry.date().isoformat(),
        "maturity_date": maturity.date().isoformat(),
        "entry_weight_coverage": entry_coverage,
        "endpoint_weight_coverage": endpoint_coverage,
        "joint_weight_coverage": joint_coverage,
        "fixed_basket_return": fixed_basket_return,
        "directional_industry_count": int(len(scored_rows)),
        "direction_accuracy": accuracy,
        "frozen_weight_direction_accuracy": weighted_accuracy,
        "positive_gap_equal_weight_relative_return": positive_equal,
        "negative_gap_equal_weight_relative_return": negative_equal,
        "positive_minus_negative_equal_weight_relative_return": long_short_equal,
        "positive_gap_frozen_weight_relative_return": positive_weighted,
        "negative_gap_frozen_weight_relative_return": negative_weighted,
        "positive_minus_negative_frozen_weight_relative_return": long_short_weighted,
        "industry_rows": group_rows,
    }


def score_fundamental_horizon(
    horizon: int,
    maturity_date: Any,
    baseline_date: Any,
    sector_panel: pd.DataFrame,
    forecast_rows: Sequence[Mapping[str, Any]],
    confirmation_columns: Sequence[str],
    rules: EvaluationRules,
) -> dict[str, Any]:
    """以三项点时行业财务指标的多数方向票确认经营方向。"""

    required = {"date", "industry_l1", "sector_weight", "core_feature_coverage_ratio"} | set(
        confirmation_columns
    )
    missing = required - set(sector_panel.columns)
    if missing:
        raise ForwardEvaluationError(f"行业点时面板缺少评价字段：{sorted(missing)}")
    panel = sector_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ForwardEvaluationError("行业点时面板包含非法日期")
    baseline = _timestamp(baseline_date, "baseline_date").normalize()
    maturity = _timestamp(maturity_date, "maturity_date").normalize()
    if baseline not in set(panel["date"]):
        raise ForwardEvaluationError("行业点时面板缺少冻结基线")
    eligible_dates = panel.loc[
        panel["date"].gt(baseline) & panel["date"].le(maturity), "date"
    ]
    if eligible_dates.empty:
        return {
            "status": "MATURE_DATA_INCOMPLETE",
            "failure_category": "NO_POST_BASELINE_POINT_IN_TIME_SECTOR_SNAPSHOT",
            "horizon_trading_days": horizon,
            "maturity_date": maturity.date().isoformat(),
            "baseline_snapshot_date": baseline.date().isoformat(),
            "endpoint_snapshot_date": None,
            "industry_rows": [],
        }
    endpoint = pd.Timestamp(eligible_dates.max())
    baseline_frame = panel.loc[panel["date"].eq(baseline)].copy()
    endpoint_frame = panel.loc[panel["date"].eq(endpoint)].copy()
    if baseline_frame["industry_l1"].duplicated().any() or endpoint_frame[
        "industry_l1"
    ].duplicated().any():
        raise ForwardEvaluationError("行业点时截面存在重复行业")

    columns = ["industry_l1", "core_feature_coverage_ratio", *confirmation_columns]
    merged = baseline_frame[columns].merge(
        endpoint_frame[columns],
        on="industry_l1",
        how="outer",
        suffixes=("_baseline", "_endpoint"),
        validate="one_to_one",
    )
    forecast = pd.DataFrame(list(forecast_rows))[
        ["industry_l1", "sector_weight", f"fundamental_{horizon}d"]
    ].rename(columns={f"fundamental_{horizon}d": "predicted_fundamental"})
    merged = forecast.merge(merged, on="industry_l1", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        coverage_values = [
            row.get("core_feature_coverage_ratio_baseline"),
            row.get("core_feature_coverage_ratio_endpoint"),
        ]
        coverage_ok = all(
            pd.notna(value)
            and float(value) >= rules.minimum_fundamental_core_coverage_ratio
            for value in coverage_values
        )
        deltas: dict[str, float | None] = {}
        votes: dict[str, int | None] = {}
        for column in confirmation_columns:
            before = row.get(f"{column}_baseline")
            after = row.get(f"{column}_endpoint")
            if pd.isna(before) or pd.isna(after):
                deltas[column] = None
                votes[column] = None
            else:
                delta = float(after) - float(before)
                deltas[column] = delta
                votes[column] = 1 if delta > 0 else -1 if delta < 0 else 0
        valid_votes = [value for value in votes.values() if value is not None]
        if not coverage_ok or len(valid_votes) != len(confirmation_columns):
            observed_state = None
            state = "NO_VIEW"
            hit = None
            failure = "FUNDAMENTAL_ENDPOINT_COVERAGE_OR_FIELD_MISSING"
        else:
            positive_votes = sum(value > 0 for value in valid_votes)
            negative_votes = sum(value < 0 for value in valid_votes)
            if positive_votes > negative_votes:
                observed_state = "IMPROVEMENT"
            elif negative_votes > positive_votes:
                observed_state = "DETERIORATION"
            else:
                observed_state = "MIXED"
            predicted = str(row["predicted_fundamental"])
            if predicted in {"STRONG_IMPROVEMENT", "IMPROVEMENT"}:
                state = "SCORED"
                hit = observed_state == "IMPROVEMENT"
            elif predicted in {"STRONG_DETERIORATION", "DETERIORATION"}:
                state = "SCORED"
                hit = observed_state == "DETERIORATION"
            else:
                state = "ABSTAINED_AT_FORECAST"
                hit = None
            failure = "PASS"
        rows.append(
            {
                "industry_l1": row["industry_l1"],
                "sector_weight": float(row["sector_weight"]),
                "predicted_fundamental": row["predicted_fundamental"],
                "observed_fundamental": observed_state,
                "deltas": deltas,
                "votes": votes,
                "direction_hit": hit,
                "state": state,
                "failure_category": failure,
            }
        )
    scored = pd.DataFrame(rows)
    scored = scored.loc[scored["state"].eq("SCORED")]
    if scored.empty:
        accuracy = None
        weighted_accuracy = None
    else:
        accuracy = float(scored["direction_hit"].astype(float).mean())
        score_weights = scored["sector_weight"].astype(float)
        weighted_accuracy = float(
            (scored["direction_hit"].astype(float) * score_weights).sum()
            / score_weights.sum()
        )
    return {
        "status": "MATURE_SCORED",
        "failure_category": "PASS",
        "horizon_trading_days": horizon,
        "maturity_date": maturity.date().isoformat(),
        "baseline_snapshot_date": baseline.date().isoformat(),
        "endpoint_snapshot_date": endpoint.date().isoformat(),
        "directional_industry_count": int(len(scored)),
        "direction_accuracy": accuracy,
        "frozen_weight_direction_accuracy": weighted_accuracy,
        "industry_rows": rows,
    }


def score_etf_horizon(
    horizon: int,
    entry_date: Any,
    maturity_date: Any,
    calendar: pd.DatetimeIndex,
    etf_daily: pd.DataFrame,
) -> dict[str, Any]:
    """计算入场日开盘至成熟日收盘的 ETF 含分红毛收益。"""

    daily = _validate_daily_dates(etf_daily, "ETF总收益日线")
    required = {"open", "close", "total_return"}
    missing = required - set(daily.columns)
    if missing:
        raise ForwardEvaluationError(f"ETF总收益日线缺少字段：{sorted(missing)}")
    entry = _timestamp(entry_date, "entry_date").normalize()
    maturity = _timestamp(maturity_date, "maturity_date").normalize()
    expected_dates = calendar[(calendar >= entry) & (calendar <= maturity)]
    selected = daily.loc[daily["date"].isin(expected_dates)].sort_values("date")
    actual_dates = pd.DatetimeIndex(selected["date"])
    missing_dates = expected_dates.difference(actual_dates)
    if len(missing_dates):
        return {
            "status": "MATURE_DATA_INCOMPLETE",
            "failure_category": "ETF_TRADING_DAY_COVERAGE_INCOMPLETE",
            "horizon_trading_days": horizon,
            "entry_date": entry.date().isoformat(),
            "maturity_date": maturity.date().isoformat(),
            "missing_dates": [value.date().isoformat() for value in missing_dates],
        }
    entry_row = selected.loc[selected["date"].eq(entry)].iloc[0]
    if float(entry_row["open"]) <= 0 or float(entry_row["close"]) <= 0:
        raise ForwardEvaluationError("ETF入场日价格非法")
    wealth = float(entry_row["close"]) / float(entry_row["open"])
    later = selected.loc[selected["date"].gt(entry), "total_return"].astype(float)
    wealth *= float((1.0 + later).prod())
    return {
        "status": "MATURE_SCORED",
        "failure_category": "PASS",
        "horizon_trading_days": horizon,
        "entry_date": entry.date().isoformat(),
        "maturity_date": maturity.date().isoformat(),
        "gross_total_return": wealth - 1.0,
        "transaction_cost_applied": False,
        "interpretation": "同期执行壳层观察，不是已授权信号绩效",
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    return value


def build_forward_evaluation_report(
    config: Mapping[str, Any],
    rules: EvaluationRules,
    safety: Mapping[str, bool],
    calendar: pd.DatetimeIndex,
    observation: Mapping[str, Any],
    frozen_result: Mapping[str, Any],
    historical_weights: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    sector_panel: pd.DataFrame,
    etf_daily: pd.DataFrame,
    runtime_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """仅对已成熟期限评分，未成熟期限绝不计算部分收益。"""

    forecast_rows = frozen_result.get("industry_rows")
    if not isinstance(forecast_rows, list) or not forecast_rows:
        raise ForwardEvaluationError("原预测结果缺少行业行")
    if frozen_result.get("final_state") != "NO_VIEW":
        raise ForwardEvaluationError("首期原预测状态不再是冻结的 NO_VIEW")

    horizons: list[dict[str, Any]] = []
    for raw_state in observation["horizons"]:
        horizon_state = dict(raw_state)
        horizon = int(horizon_state["horizon_trading_days"])
        maturity = horizon_state["maturity_date"]
        if horizon_state["state"] != "MATURE_PENDING_SCORE":
            horizon_state["price_evaluation"] = {
                "status": horizon_state["state"],
                "partial_return_output": False,
            }
            horizon_state["fundamental_evaluation"] = {
                "status": horizon_state["state"],
                "partial_result_output": False,
            }
            horizon_state["etf_observation"] = {
                "status": horizon_state["state"],
                "partial_return_output": False,
            }
            horizons.append(horizon_state)
            continue

        price = score_price_horizon(
            horizon,
            config["entry_date"],
            maturity,
            config["frozen_prediction"]["weight_snapshot_date"],
            historical_weights,
            industry_intervals,
            constituent_daily,
            forecast_rows,
            rules,
        )
        fundamental = score_fundamental_horizon(
            horizon,
            maturity,
            config["frozen_prediction"]["weight_snapshot_date"],
            sector_panel,
            forecast_rows,
            config["fundamental_confirmation_columns"],
            rules,
        )
        etf = score_etf_horizon(
            horizon,
            config["entry_date"],
            maturity,
            calendar,
            etf_daily,
        )
        component_statuses = {price["status"], fundamental["status"], etf["status"]}
        horizon_state["state"] = (
            "MATURE_SCORED"
            if component_statuses == {"MATURE_SCORED"}
            else "MATURE_DATA_INCOMPLETE"
        )
        horizon_state["price_evaluation"] = price
        horizon_state["fundamental_evaluation"] = fundamental
        horizon_state["etf_observation"] = etf
        horizons.append(horizon_state)

    primary = next(
        item for item in horizons if item["horizon_trading_days"] == min(rules.horizons_trading_days)
    )
    mature_origin_count = 1 if primary["state"] == "MATURE_SCORED" else 0
    return _json_ready(
        {
            "evaluation_version": config["version"],
            "evaluation_status": config["status"],
            "prediction_model_version": frozen_result["model_version"],
            "prediction_date": config["prediction_date"],
            "information_cutoff": config["information_cutoff"],
            "entry_date": config["entry_date"],
            "observation_date": observation["observation_date"],
            "primary_state": primary["state"],
            "original_prediction_state": "NO_VIEW",
            "original_index_aggregation_state": frozen_result["industry_aggregation"]["state"],
            "original_index_direction_evaluation": "ABSTAINED_AT_FORECAST",
            "market_gate_evaluation": "NOT_SCORABLE_MISSING_AT_FORECAST",
            "national_team_evaluation": "NOT_SCORABLE_UNOBSERVED_AT_FORECAST",
            "may_upgrade_original_no_view": False,
            "mature_prediction_origin_count": mature_origin_count,
            "calibration_minimum": rules.minimum_mature_prediction_origins_for_calibration,
            "model_comparison_minimum": rules.minimum_mature_prediction_origins_for_model_comparison,
            "horizons": horizons,
            "data_cutoffs": {
                "constituent_latest_date": observation["constituent_latest_date"],
                "etf_latest_date": observation["etf_latest_date"],
                "common_forward_latest_date": observation["common_forward_latest_date"],
            },
            "runtime_sources": dict(runtime_sources),
            "safety": dict(safety),
            "interpretation": (
                "未成熟期不输出部分收益；成熟后只评价冻结行业分支，"
                "不得把原NO_VIEW改写为交易观点。"
            ),
        }
    )


def render_forward_evaluation_markdown(report: Mapping[str, Any]) -> str:
    """渲染中文成熟度报告。"""

    lines = [
        "# 行业预期差 V1 前瞻评价状态",
        "",
        f"- 预测日：`{report['prediction_date']}`",
        f"- 首个可捕获日：`{report['entry_date']}` 开盘",
        f"- 当前观察日：`{report['observation_date']}`",
        f"- 主要状态：`{report['primary_state']}`",
        f"- 原指数观点：`{report['original_prediction_state']}`（不可事后升级）",
        "",
        "## 成熟度",
        "",
        "| 期限 | 成熟日 | 日历 | 已积累共同交易日 | 状态 | 是否输出部分收益 |",
        "|---:|---|---|---:|---|---|",
    ]
    for horizon in report["horizons"]:
        lines.append(
            "| {days} | {maturity} | {calendar} | {observed} | {state} | 否 |".format(
                days=horizon["horizon_trading_days"],
                maturity=horizon["maturity_date"] or "未解析",
                calendar=horizon["calendar_state"],
                observed=horizon["observed_common_forward_trading_days"],
                state=horizon["state"],
            )
        )
    lines.extend(
        [
            "",
            "## 当前数据截止日",
            "",
            f"- 成分总收益：`{report['data_cutoffs']['constituent_latest_date']}`",
            f"- ETF含分红：`{report['data_cutoffs']['etf_latest_date']}`",
            f"- 共同前瞻日期：`{report['data_cutoffs']['common_forward_latest_date']}`",
            "",
            "## 不可回写层",
            "",
            f"- 指数方向：`{report['original_index_direction_evaluation']}`",
            f"- 市场流动性：`{report['market_gate_evaluation']}`",
            f"- 国家队：`{report['national_team_evaluation']}`",
            "",
            "## 样本边界",
            "",
            (
                f"当前成熟预测起点数为 `{report['mature_prediction_origin_count']}`；"
                f"校准至少需要 `{report['calibration_minimum']}` 个，"
                f"模型比较至少需要 `{report['model_comparison_minimum']}` 个。"
            ),
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)
