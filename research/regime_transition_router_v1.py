"""510300 大盘状态识别与机制路由 V1。

第一阶段只构造点时状态并检验条件风险结构。状态面板不含未来字段；未来路径
单独写入 outcome_panel，且永远不在本模块中映射仓位或计算策略净值。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_REGIME_TRANSITION_ROUTER_V1"
CONFIG_PATH = PROJECT_ROOT / "config" / "510300_regime_transition_router_v1.yaml"
MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "510300_regime_transition_router_v1_manifest.json"
)

NORMAL = "NORMAL_CARRY"
STRESS = "STRESS_DELEVERAGING"
RECOVERY = "RECOVERY_REPRICING"
UNCERTAIN = "UNCERTAIN"
STATES = [NORMAL, STRESS, RECOVERY, UNCERTAIN]


class ContractError(RuntimeError):
    """输入、冻结或点时合同不满足。"""


@dataclass(frozen=True)
class Contrast:
    """固定四方向条件结构及其样本量。"""

    stress_days_10d: int
    normal_days_10d: int
    recovery_days_20d: int
    stress_days_20d: int
    stress_mean_mae_minus_normal_10d: float | None
    stress_q20_return_minus_normal_10d: float | None
    recovery_mean_return_minus_stress_20d: float | None
    recovery_positive_fraction_minus_stress_20d: float | None
    stress_mean_mae_direction: bool
    stress_q20_return_direction: bool
    recovery_mean_return_direction: bool
    recovery_positive_fraction_direction: bool

    @property
    def all_directions_pass(self) -> bool:
        return bool(
            self.stress_mean_mae_direction
            and self.stress_q20_return_direction
            and self.recovery_mean_return_direction
            and self.recovery_positive_fraction_direction
        )

    @property
    def stress_directions_pass(self) -> bool:
        return bool(
            self.stress_mean_mae_direction and self.stress_q20_return_direction
        )

    @property
    def recovery_directions_pass(self) -> bool:
        return bool(
            self.recovery_mean_return_direction
            and self.recovery_positive_fraction_direction
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stress_days_10d": self.stress_days_10d,
            "normal_days_10d": self.normal_days_10d,
            "recovery_days_20d": self.recovery_days_20d,
            "stress_days_20d": self.stress_days_20d,
            "stress_mean_mae_minus_normal_10d": self.stress_mean_mae_minus_normal_10d,
            "stress_q20_return_minus_normal_10d": self.stress_q20_return_minus_normal_10d,
            "recovery_mean_return_minus_stress_20d": self.recovery_mean_return_minus_stress_20d,
            "recovery_positive_fraction_minus_stress_20d": self.recovery_positive_fraction_minus_stress_20d,
            "stress_mean_mae_direction": self.stress_mean_mae_direction,
            "stress_q20_return_direction": self.stress_q20_return_direction,
            "recovery_mean_return_direction": self.recovery_mean_return_direction,
            "recovery_positive_fraction_direction": self.recovery_positive_fraction_direction,
            "stress_directions_pass": self.stress_directions_pass,
            "recovery_directions_pass": self.recovery_directions_pass,
            "all_directions_pass": self.all_directions_pass,
        }


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    if column not in result.columns:
        raise ContractError(f"数据缺少日期字段：{column}")
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ContractError(f"{column}存在重复日期")
    return result


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
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
    """读取并验证冻结前协议结构。"""

    if not path.exists():
        raise ContractError(f"协议不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ContractError("协议项目编号不匹配")
    if config["protocol"]["research_stage"] != "PHASE_1_STATE_IDENTIFIABILITY_ONLY":
        raise ContractError("协议不是第一阶段状态可识别性研究")
    if config["scope"]["return_backtest_allowed_in_phase_1"]:
        raise ContractError("第一阶段不得允许组合收益回测")
    if config["scope"]["position_mapping"] != "disabled":
        raise ContractError("第一阶段仓位映射必须关闭")
    if config["scope"]["live_trading_authorized"]:
        raise ContractError("协议不得授权实盘")
    forbidden_input_tokens = (
        "cash_final",
        "cash_raw",
        "shadow_gate",
        "forward_return",
        "future_return",
    )
    input_paths = [str(item.get("path", "")).lower() for item in config["inputs"].values()]
    if any(token in path for path in input_paths for token in forbidden_input_tokens):
        raise ContractError("协议输入路径包含旧仓位或未来结果文件")
    return config


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """验证结果读取前冻结清单及全部哈希。"""

    if not MANIFEST_PATH.exists():
        raise ContractError("冻结清单不存在，禁止读取条件路径结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("冻结清单项目编号不匹配")
    if not manifest.get("frozen_before_outcome_read", False):
        raise ContractError("冻结清单未证明在未来条件路径读取前冻结")
    if manifest.get("outcome_read_before_freeze") is not False:
        raise ContractError("冻结清单显示冻结前读取过未来条件路径")
    mismatches: dict[str, dict[str, str]] = {}
    for group in ["tracked_files", "input_files"]:
        for relative, expected in manifest.get(group, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件哈希漂移：{mismatches}")
    return manifest


def causal_percentile(
    series: pd.Series,
    *,
    prior_rows: int,
    minimum_prior_rows: int,
) -> pd.Series:
    """计算只使用当前观察之前样本的经验分位。"""

    if minimum_prior_rows < 1 or prior_rows < minimum_prior_rows:
        raise ValueError("点时分位窗口设置无效")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        reference = values[max(0, index - prior_rows) : index]
        reference = reference[np.isfinite(reference)]
        if len(reference) < minimum_prior_rows:
            continue
        output[index] = float(np.count_nonzero(reference <= current) / len(reference))
    return pd.Series(output, index=series.index, dtype=float)


def rolling_log_slope(series: pd.Series, window: int) -> pd.Series:
    """固定窗口对数价格线性斜率。"""

    logged = np.log(pd.to_numeric(series, errors="coerce"))
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))

    def slope(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(centered, values - values.mean()) / denominator)

    return logged.rolling(window, min_periods=window).apply(slope, raw=True)


def build_total_return_market(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """从未复权收盘和现金分红构造因果总回报指数。"""

    frame = _normalize_date(market)
    _require_columns(frame, ["date", "open", "high", "low", "close"], "510300行情")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["open", "high", "low", "close"]].le(0.0).any(axis=None):
        raise ContractError("510300行情包含非正价格")
    distribution = dividends.copy()
    for column in ["record_date", "ex_date", "payment_date"]:
        distribution[column] = pd.to_datetime(
            distribution[column], errors="raise"
        ).dt.normalize()
    distribution["cash_dividend_per_share"] = pd.to_numeric(
        distribution["cash_dividend_per_share"], errors="raise"
    )
    ex_map = (
        distribution.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    frame["cash_dividend_ex_date"] = frame["date"].map(
        lambda date: float(ex_map.get(date, 0.0))
    )
    frame["total_return_1d"] = (
        (frame["close"] + frame["cash_dividend_ex_date"])
        / frame["close"].shift(1)
        - 1.0
    )
    frame["total_return_index"] = (
        1.0 + frame["total_return_1d"].fillna(0.0)
    ).cumprod()
    return frame


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """读取冻结输入；不构造任何未来结果。"""

    inputs = config["inputs"]

    def parquet(name: str) -> pd.DataFrame:
        item = inputs[name]
        frame = pd.read_parquet(_project_path(item["path"]))
        _require_columns(frame, item["required_columns"], name)
        return frame

    share = _normalize_date(parquet("share_and_close_premium"))
    historical_margin = _normalize_date(parquet("historical_margin"))
    current_margin = _normalize_date(parquet("current_margin"))
    market = _normalize_date(parquet("etf_market"))
    early_breadth = _normalize_date(parquet("early_equal_weight_breadth"))
    current_breadth = _normalize_date(parquet("current_equal_weight_breadth"))
    cgb = _normalize_date(parquet("cgb_curve"))
    dividends = pd.read_csv(_project_path(inputs["dividends"]["path"]))
    _require_columns(dividends, inputs["dividends"]["required_columns"], "dividends")

    json_inputs: dict[str, dict[str, Any]] = {}
    for name in [
        "membership_audit",
        "external_breadth_audit",
        "cgb_audit",
        "long_cycle_result",
        "mechanism_atlas_result",
        "downside_risk_result",
        "macro_stress_result",
    ]:
        json_inputs[name] = json.loads(
            _project_path(inputs[name]["path"]).read_text(encoding="utf-8")
        )
    return {
        "share": share,
        "historical_margin": historical_margin,
        "current_margin": current_margin,
        "market": market,
        "dividends": dividends,
        "early_breadth": early_breadth,
        "current_breadth": current_breadth,
        "cgb": cgb,
        **json_inputs,
    }


def audit_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """审计点时数据合同，不读取未来条件路径。"""

    gates = config["data_gates"]
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    signal_cutoff = pd.Timestamp(config["dates"]["signal_cutoff"])
    share = inputs["share"]
    prohibited = [
        column
        for column in share.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    feature_asof = pd.to_datetime(share["feature_asof"], errors="raise")
    share_clock_pass = bool(
        (feature_asof.dt.normalize() == share["date"]).all()
        and share["execution_earliest"].eq("NEXT_TRADING_DAY_OPEN").all()
    )

    seam_date = pd.Timestamp(gates["margin_seam_date"])
    historical_seam = inputs["historical_margin"].loc[
        inputs["historical_margin"]["date"].eq(seam_date)
    ]
    current_seam = inputs["current_margin"].loc[
        inputs["current_margin"]["date"].eq(seam_date)
    ]
    seam_differences: dict[str, float | None] = {}
    seam_pass = len(historical_seam) == 1 and len(current_seam) == 1
    for column in ["rzye", "rqye", "rqyl", "rzrqye"]:
        if seam_pass:
            difference = abs(
                float(historical_seam[column].iloc[0])
                - float(current_seam[column].iloc[0])
            )
            seam_differences[column] = difference
            tolerance = (
                float(gates["derived_margin_seam_absolute_tolerance"])
                if column in {"rqye", "rzrqye"}
                else float(gates["direct_margin_seam_absolute_tolerance"])
            )
            seam_pass = seam_pass and difference <= tolerance
        else:
            seam_differences[column] = None

    early_breadth = inputs["early_breadth"]
    current_breadth = inputs["current_breadth"]
    required_count = int(gates["breadth_member_count"])
    breadth_full_count_start = pd.Timestamp(gates["breadth_full_count_applies_from"])
    early_breadth_evaluation = early_breadth.loc[
        early_breadth["date"].ge(breadth_full_count_start)
    ]
    early_breadth_warmup = early_breadth.loc[
        early_breadth["date"].lt(breadth_full_count_start)
    ]
    breadth_counts_pass = bool(
        not early_breadth_evaluation.empty
        and early_breadth_evaluation["member_count"].eq(required_count).all()
        and current_breadth["component_count"].eq(required_count).all()
        and current_breadth["valid_for_price_breadth"].astype(bool).all()
    )
    breadth_seam_pass = bool(
        early_breadth["date"].max() == pd.Timestamp("2021-08-11")
        and current_breadth["date"].min() == pd.Timestamp("2021-08-12")
    )

    membership_status = config["inputs"]["membership_audit"]["required_status"]
    membership = inputs["membership_audit"]
    membership_pass = bool(
        membership.get("status") == membership_status
        and config["inputs"]["membership_audit"]["required_allowed_use_contains"]
        in str(membership.get("allowed_use", ""))
    )
    external_breadth_pass = bool(
        inputs["external_breadth_audit"].get("status")
        == config["inputs"]["external_breadth_audit"]["required_status"]
        and inputs["external_breadth_audit"].get("minimum_members_per_day")
        == required_count
        and inputs["external_breadth_audit"].get("maximum_members_per_day")
        == required_count
        and inputs["external_breadth_audit"].get("evaluation_missing_price_rows") == 0
    )

    cgb = inputs["cgb"]
    cgb_pass = bool(
        cgb["date"].min() <= pd.Timestamp("2012-01-10")
        and cgb["date"].max() >= signal_cutoff
        and not cgb[["cgb_1y", "cgb_10y"]].isna().any(axis=None)
        and cgb[["cgb_1y", "cgb_10y"]].gt(0.0).all(axis=None)
        and inputs["cgb_audit"].get("status")
        == config["inputs"]["cgb_audit"]["required_status"]
        and inputs["cgb_audit"].get("overlap_validation", {}).get("overlap_rows", 0)
        >= int(gates["minimum_cgb_overlap_validation_rows"])
        and all(
            float(value) <= 1e-10
            for value in inputs["cgb_audit"]
            .get("overlap_validation", {})
            .get("maximum_absolute_difference_pct_point", {})
            .values()
        )
    )

    expected_predecessors = {
        "long_cycle_result": config["predecessor_boundaries"][
            "liquidity_stress_long_cycle"
        ]["required_status"],
        "mechanism_atlas_result": config["predecessor_boundaries"]["mechanism_atlas"]
        ["required_status"],
        "downside_risk_result": config["predecessor_boundaries"]["downside_risk_budget"]
        ["required_status"],
        "macro_stress_result": config["predecessor_boundaries"]["macro_stress_avoidance"]
        ["required_status"],
    }
    predecessor_checks = {
        name: _contains_value(inputs[name], expected)
        for name, expected in expected_predecessors.items()
    }

    market = inputs["market"]
    evaluation_calendar = market.loc[
        market["date"].between(evaluation_start, signal_cutoff), "date"
    ]
    market_pass = bool(
        len(evaluation_calendar) == int(gates["expected_evaluation_rows"])
        and evaluation_calendar.min()
        == pd.Timestamp(gates["required_first_evaluation_date"])
        and evaluation_calendar.max()
        == pd.Timestamp(gates["required_last_signal_date"])
        and not market[["open", "high", "low", "close"]].isna().any(axis=None)
    )

    checks = {
        "no_future_columns_in_signal_inputs": not prohibited,
        "share_feature_clock": share_clock_pass,
        "margin_seam": seam_pass,
        "breadth_member_counts": breadth_counts_pass,
        "breadth_calendar_seam": breadth_seam_pass,
        "membership_allowed_use": membership_pass,
        "external_breadth_audit": external_breadth_pass,
        "cgb_long_history": cgb_pass,
        "market_calendar": market_pass,
        "predecessor_rejections_preserved": all(predecessor_checks.values()),
    }
    passed = bool(all(checks.values()))
    return {
        "project_id": PROJECT_ID,
        "status": "PASS_STATE_INPUT_DATA_CONTRACT" if passed else "FAIL_STATE_INPUT_DATA_CONTRACT",
        "passed": passed,
        "checks": checks,
        "prohibited_signal_columns": prohibited,
        "margin_seam_absolute_differences": seam_differences,
        "breadth_member_count_scope": {
            "required_count": required_count,
            "full_count_applies_from": breadth_full_count_start.date().isoformat(),
            "pre_evaluation_warmup_rows": int(len(early_breadth_warmup)),
            "pre_evaluation_warmup_minimum_reported_usable_members": int(
                early_breadth_warmup["member_count"].min()
            ),
            "pre_evaluation_warmup_maximum_reported_usable_members": int(
                early_breadth_warmup["member_count"].max()
            ),
            "evaluation_and_later_all_equal_required_count": breadth_counts_pass,
            "warmup_rows_in_formal_evaluation": 0,
        },
        "predecessor_status_checks": predecessor_checks,
        "evaluation_calendar": {
            "rows": int(len(evaluation_calendar)),
            "first_date": evaluation_calendar.min().date().isoformat(),
            "last_date": evaluation_calendar.max().date().isoformat(),
        },
        "boundaries": {
            "future_outcomes_read": False,
            "strategy_returns_computed": False,
            "position_mapping": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def _combine_margin(config: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    seam = pd.Timestamp(config["data_gates"]["margin_seam_date"])
    columns = ["date", "rzye", "rqye", "rqyl", "rzrqye"]
    combined = pd.concat(
        [
            inputs["historical_margin"].loc[
                inputs["historical_margin"]["date"].lt(seam), columns
            ],
            inputs["current_margin"].loc[
                inputs["current_margin"]["date"].ge(seam), columns
            ],
        ],
        ignore_index=True,
    )
    return _normalize_date(combined)


def _combine_breadth(inputs: dict[str, Any]) -> pd.DataFrame:
    early = inputs["early_breadth"][
        ["date", "equal_above_ma60_share", "member_count"]
    ].copy()
    early.rename(columns={"member_count": "component_count"}, inplace=True)
    early["breadth_segment"] = "EARLY_POINT_IN_TIME_MEMBERS"
    current = inputs["current_breadth"][
        ["date", "equal_above_ma60_share", "component_count"]
    ].copy()
    current["breadth_segment"] = "CURRENT_POINT_IN_TIME_MEMBERS"
    combined = pd.concat([early, current], ignore_index=True)
    return _normalize_date(combined)


def build_signal_panel(
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> pd.DataFrame:
    """构造完全点时的状态面板，不含任何未来字段。"""

    dates = config["dates"]
    market = build_total_return_market(inputs["market"], inputs["dividends"])
    market = market.loc[
        market["date"].between(
            pd.Timestamp(dates["feature_warmup_start"]),
            pd.Timestamp(dates["signal_cutoff"]),
        )
    ].copy()
    price_spec = config["features"]["breadth_and_price"]
    price_ma_rows = int(price_spec["price_ma_rows"])
    slope_rows = int(price_spec["price_log_slope_rows"])
    market["price_ma20_total_return"] = market["total_return_index"].rolling(
        price_ma_rows, min_periods=price_ma_rows
    ).mean()
    market["price_log_slope20"] = rolling_log_slope(
        market["total_return_index"], slope_rows
    )

    downside_spec = config["features"]["downside_pressure"]
    downside = market["total_return_1d"].clip(upper=0.0)
    short_span = int(downside_spec["short_ewm_span"])
    long_span = int(downside_spec["long_ewm_span"])
    market["downside_deviation_5"] = np.sqrt(
        downside.pow(2).ewm(
            span=short_span, adjust=False, min_periods=short_span
        ).mean()
    )
    market["downside_deviation_20"] = np.sqrt(
        downside.pow(2).ewm(
            span=long_span, adjust=False, min_periods=long_span
        ).mean()
    )
    market["downside_vol_ratio_5_20"] = market["downside_deviation_5"] / market[
        "downside_deviation_20"
    ].replace(0.0, np.nan)

    liquidity_spec = config["features"]["liquidity_and_leverage"]
    share = inputs["share"][["date", "fund_shares", "close_premium_to_nav"]].copy()
    margin = _combine_margin(config, inputs)
    liquidity = share.merge(margin, on="date", how="inner", validate="one_to_one")
    liquidity["fund_share_log_change20"] = np.log(
        pd.to_numeric(liquidity["fund_shares"], errors="raise")
    ).diff(int(liquidity_spec["fund_share_log_change_rows"]))
    liquidity["total_margin_log_change3"] = np.log(
        pd.to_numeric(liquidity["rzrqye"], errors="raise")
    ).diff(int(liquidity_spec["total_margin_log_change_rows"]))
    prior = int(liquidity_spec["causal_percentile_prior_rows"])
    minimum = int(liquidity_spec["causal_percentile_minimum_prior_rows"])
    liquidity["fund_share_change_percentile"] = causal_percentile(
        liquidity["fund_share_log_change20"],
        prior_rows=prior,
        minimum_prior_rows=minimum,
    )
    liquidity["total_margin_change_percentile"] = causal_percentile(
        liquidity["total_margin_log_change3"],
        prior_rows=prior,
        minimum_prior_rows=minimum,
    )
    liquidity["liquidity_stress_score"] = (
        (1.0 - liquidity["fund_share_change_percentile"])
        + (1.0 - liquidity["total_margin_change_percentile"])
    ) / 2.0

    curve = inputs["cgb"].copy()
    discount_spec = config["features"]["discount_rate"]
    curve["cgb_10y_change63"] = pd.to_numeric(
        curve["cgb_10y"], errors="raise"
    ).diff(int(discount_spec["cgb_10y_change_curve_rows"]))
    curve["discount_rate_pressure"] = causal_percentile(
        curve["cgb_10y_change63"],
        prior_rows=int(discount_spec["causal_percentile_prior_rows"]),
        minimum_prior_rows=int(
            discount_spec["causal_percentile_minimum_prior_rows"]
        ),
    )
    curve.rename(columns={"date": "cgb_observation_date"}, inplace=True)

    breadth = _combine_breadth(inputs)
    breadth["breadth_change5"] = breadth["equal_above_ma60_share"].diff(
        int(price_spec["breadth_change_rows"])
    )

    panel = market.merge(
        liquidity,
        on="date",
        how="left",
        validate="one_to_one",
    ).merge(
        breadth,
        on="date",
        how="left",
        validate="one_to_one",
    )
    panel = pd.merge_asof(
        panel.sort_values("date"),
        curve[
            [
                "cgb_observation_date",
                "cgb_1y",
                "cgb_10y",
                "cgb_10y_change63",
                "discount_rate_pressure",
            ]
        ].sort_values("cgb_observation_date"),
        left_on="date",
        right_on="cgb_observation_date",
        direction="backward",
        allow_exact_matches=True,
    )
    panel["cgb_staleness_calendar_days"] = (
        panel["date"] - panel["cgb_observation_date"]
    ).dt.days
    if panel["cgb_staleness_calendar_days"].dropna().gt(
        int(config["data_gates"]["maximum_cgb_calendar_staleness_days"])
    ).any():
        raise ContractError("中债曲线对齐陈旧天数超过冻结上限")

    panel = apply_state_machine(panel, config)
    evaluation = panel.loc[
        panel["date"].between(
            pd.Timestamp(dates["evaluation_start"]),
            pd.Timestamp(dates["signal_cutoff"]),
        )
    ].copy()
    expected = int(config["data_gates"]["expected_evaluation_rows"])
    if len(evaluation) != expected:
        raise ContractError(f"状态面板评价行数{len(evaluation)}不等于冻结值{expected}")
    future_columns = [
        column
        for column in evaluation.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    if future_columns:
        raise ContractError(f"状态面板含未来字段：{future_columns}")
    return evaluation.reset_index(drop=True)


def apply_state_machine(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """按冻结阈值运行有记忆的四状态机。"""

    frame = panel.copy().reset_index(drop=True)
    machine = config["state_machine"]
    stress_spec = machine["stress_votes"]
    recovery_spec = machine["recovery"]
    normal_spec = machine["normal"]

    frame["vote_liquidity_stress"] = frame["liquidity_stress_score"].ge(
        float(stress_spec["liquidity_score_at_least"])
    )
    frame["vote_downside_stress"] = frame["downside_vol_ratio_5_20"].ge(
        float(stress_spec["downside_ratio_at_least"])
    )
    frame["vote_discount_stress"] = frame["discount_rate_pressure"].ge(
        float(stress_spec["discount_pressure_at_least"])
    )
    vote_columns = [
        "vote_liquidity_stress",
        "vote_downside_stress",
        "vote_discount_stress",
    ]
    frame["stress_vote_count"] = frame[vote_columns].sum(axis=1).astype(np.int8)

    required_features = [
        "liquidity_stress_score",
        "downside_vol_ratio_5_20",
        "discount_rate_pressure",
        "equal_above_ma60_share",
        "breadth_change5",
        "total_return_index",
        "price_ma20_total_return",
        "price_log_slope20",
    ]
    frame["state_feature_valid"] = frame[required_features].notna().all(axis=1)
    frame["stress_candidate"] = (
        frame["state_feature_valid"]
        & frame["stress_vote_count"].ge(int(stress_spec["minimum_votes"]))
        & (
            frame["vote_liquidity_stress"]
            | frame["vote_downside_stress"]
        )
    )

    prior_peak = frame["downside_vol_ratio_5_20"].shift(1).rolling(
        20, min_periods=20
    ).max()
    frame["downside_prior20_peak"] = prior_peak
    frame["downside_change3"] = frame["downside_vol_ratio_5_20"].diff(
        int(recovery_spec["downside_change_rows"])
    )
    drop = float(recovery_spec["downside_drop_from_prior_20d_peak_at_least"])
    frame["downside_relief"] = (
        frame["downside_vol_ratio_5_20"].le((1.0 - drop) * prior_peak)
        & frame["downside_change3"].lt(0.0)
    )
    frame["breadth_repair"] = frame["equal_above_ma60_share"].ge(
        float(recovery_spec["breadth_level_at_least"])
    ) & frame["breadth_change5"].gt(0.0)
    frame["price_repair"] = frame["total_return_index"].gt(
        frame["price_ma20_total_return"]
    ) & frame["price_log_slope20"].gt(0.0)
    frame["recovery_candidate"] = (
        frame["state_feature_valid"]
        & frame["liquidity_stress_score"].le(
            float(recovery_spec["liquidity_score_at_most"])
        )
        & frame["downside_relief"]
        & frame["breadth_repair"]
        & frame["price_repair"]
        & frame["discount_rate_pressure"].lt(
            float(recovery_spec["discount_pressure_below"])
        )
    )
    frame["breadth_positive"] = frame["equal_above_ma60_share"].ge(
        float(recovery_spec["breadth_level_at_least"])
    )
    frame["price_positive"] = frame["price_repair"]
    frame["normal_candidate"] = (
        frame["state_feature_valid"]
        & frame["stress_vote_count"].eq(
            int(normal_spec["stress_vote_count_must_equal"])
        )
        & frame["breadth_positive"].eq(frame["price_positive"])
    )

    entry_confirm = int(stress_spec["consecutive_signal_rows_for_entry"])
    recovery_confirm = int(recovery_spec["consecutive_signal_rows_for_entry"])
    nonstress_exit = int(
        machine["stress_exit_without_recovery"][
            "consecutive_nonstress_rows_to_uncertain"
        ]
    )
    recent_lookback = int(recovery_spec["recent_stress_lookback_rows"])
    recovery_hold = int(recovery_spec["minimum_hold_signal_rows"])

    states: list[str] = []
    reasons: list[str] = []
    stress_streak = 0
    recovery_streak = 0
    nonstress_streak = 0
    current_state = UNCERTAIN
    recovery_age = 0
    last_stress_index: int | None = None
    for index, row in frame.iterrows():
        if not bool(row["state_feature_valid"]):
            current_state = UNCERTAIN
            reason = "MISSING_POINT_IN_TIME_FEATURE"
            stress_streak = 0
            recovery_streak = 0
            nonstress_streak = 0
            recovery_age = 0
            states.append(current_state)
            reasons.append(reason)
            continue

        stress_candidate = bool(row["stress_candidate"])
        recovery_candidate = bool(row["recovery_candidate"])
        stress_streak = stress_streak + 1 if stress_candidate else 0
        recovery_streak = recovery_streak + 1 if recovery_candidate else 0
        nonstress_streak = nonstress_streak + 1 if not stress_candidate else 0

        if stress_streak >= entry_confirm:
            current_state = STRESS
            reason = "CONFIRMED_MULTI_MECHANISM_STRESS"
            last_stress_index = index
            recovery_age = 0
        elif current_state == STRESS:
            last_stress_index = index
            if recovery_streak >= recovery_confirm:
                current_state = RECOVERY
                reason = "CONFIRMED_ASYMMETRIC_RECOVERY"
                recovery_age = 1
            elif nonstress_streak >= nonstress_exit:
                current_state = UNCERTAIN
                reason = "STRESS_DISSIPATED_WITHOUT_RECOVERY_CONFIRMATION"
                recovery_age = 0
            else:
                reason = "STRESS_HYSTERESIS"
        elif current_state == RECOVERY:
            recovery_age += 1
            if recovery_age > recovery_hold:
                if bool(row["normal_candidate"]):
                    current_state = NORMAL
                    reason = "RECOVERY_HOLD_COMPLETE_NORMAL_EVIDENCE"
                else:
                    current_state = UNCERTAIN
                    reason = "RECOVERY_HOLD_COMPLETE_CONFLICTING_EVIDENCE"
                recovery_age = 0
            else:
                reason = "RECOVERY_MINIMUM_HOLD"
        else:
            recent_stress = bool(
                last_stress_index is not None
                and index - last_stress_index <= recent_lookback
            )
            if recovery_streak >= recovery_confirm and recent_stress:
                current_state = RECOVERY
                reason = "RECENT_STRESS_CONFIRMED_RECOVERY"
                recovery_age = 1
            elif bool(row["normal_candidate"]):
                current_state = NORMAL
                reason = "NO_STRESS_VOTES_AND_BREADTH_PRICE_AGREE"
                recovery_age = 0
            else:
                current_state = UNCERTAIN
                reason = "CONFLICTING_OR_INCOMPLETE_EVIDENCE"
                recovery_age = 0
        states.append(current_state)
        reasons.append(reason)

    frame["state"] = pd.Categorical(states, categories=STATES, ordered=False)
    frame["state_reason"] = reasons
    frame["state_transition"] = frame["state"].astype(str).ne(
        frame["state"].astype(str).shift(1)
    )
    return frame


def build_outcome_panel(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """从 T+1 开盘构造独立条件路径面板。"""

    prices = _normalize_date(market)
    prices = prices.loc[
        prices["date"].le(pd.Timestamp(config["dates"]["outcome_market_cutoff"]))
    ].copy()
    for column in ["open", "close"]:
        prices[column] = pd.to_numeric(prices[column], errors="raise")
    date_to_position = {date: index for index, date in enumerate(prices["date"])}

    distribution = dividends.copy()
    distribution["record_date"] = pd.to_datetime(
        distribution["record_date"], errors="raise"
    ).dt.normalize()
    distribution["cash_dividend_per_share"] = pd.to_numeric(
        distribution["cash_dividend_per_share"], errors="raise"
    )
    record_map = (
        distribution.groupby("record_date")["cash_dividend_per_share"].sum().to_dict()
    )
    horizons = [int(value) for value in config["outcomes"]["horizons_trading_days"]]
    records: list[dict[str, Any]] = []
    for signal_date in pd.to_datetime(state_panel["date"]):
        row: dict[str, Any] = {"date": signal_date}
        signal_position = date_to_position.get(signal_date)
        if signal_position is None or signal_position + 1 >= len(prices):
            for horizon in horizons:
                row.update(_censored_outcome(horizon))
            records.append(row)
            continue
        entry_position = signal_position + 1
        entry_date = prices.loc[entry_position, "date"]
        entry_open = float(prices.loc[entry_position, "open"])
        row["entry_date"] = entry_date
        row["entry_open"] = entry_open
        for horizon in horizons:
            exit_position = entry_position + horizon - 1
            if exit_position >= len(prices):
                row.update(_censored_outcome(horizon))
                continue
            cumulative_dividend = 0.0
            path_returns: list[float] = []
            for position in range(entry_position, exit_position + 1):
                date = prices.loc[position, "date"]
                cumulative_dividend += float(record_map.get(date, 0.0))
                path_value = float(prices.loc[position, "close"]) + cumulative_dividend
                path_returns.append(path_value / entry_open - 1.0)
            path = np.asarray(path_returns, dtype=float)
            prefix = f"{horizon}d"
            row[f"future_return_{prefix}"] = float(path[-1])
            row[f"future_mae_magnitude_{prefix}"] = float(max(0.0, -path.min()))
            row[f"future_mfe_{prefix}"] = float(max(0.0, path.max()))
            row[f"future_positive_wealth_fraction_{prefix}"] = float((path > 0.0).mean())
            row[f"future_exit_date_{prefix}"] = prices.loc[exit_position, "date"]
            row[f"future_censored_{prefix}"] = False
        records.append(row)
    return pd.DataFrame(records)


def _censored_outcome(horizon: int) -> dict[str, Any]:
    prefix = f"{horizon}d"
    return {
        f"future_return_{prefix}": np.nan,
        f"future_mae_magnitude_{prefix}": np.nan,
        f"future_mfe_{prefix}": np.nan,
        f"future_positive_wealth_fraction_{prefix}": np.nan,
        f"future_exit_date_{prefix}": pd.NaT,
        f"future_censored_{prefix}": True,
    }


def build_episodes(state_panel: pd.DataFrame) -> pd.DataFrame:
    """提取连续状态区间，并保留左右删失。"""

    frame = state_panel[["date", "state"]].copy()
    state_text = frame["state"].astype(str)
    group_id = state_text.ne(state_text.shift(1)).cumsum()
    rows: list[dict[str, Any]] = []
    for episode_id, (_, group) in enumerate(frame.groupby(group_id), start=1):
        rows.append(
            {
                "episode_id": episode_id,
                "state": str(group["state"].iloc[0]),
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "signal_rows": int(len(group)),
                "left_censored": bool(group.index.min() == frame.index.min()),
                "right_censored": bool(group.index.max() == frame.index.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_contrast(frame: pd.DataFrame, minimum_days: int = 1) -> Contrast:
    """计算冻结的四个状态方向。"""

    stress10 = frame.loc[
        frame["state"].astype(str).eq(STRESS)
        & frame["future_return_10d"].notna()
    ]
    normal10 = frame.loc[
        frame["state"].astype(str).eq(NORMAL)
        & frame["future_return_10d"].notna()
    ]
    recovery20 = frame.loc[
        frame["state"].astype(str).eq(RECOVERY)
        & frame["future_return_20d"].notna()
    ]
    stress20 = frame.loc[
        frame["state"].astype(str).eq(STRESS)
        & frame["future_return_20d"].notna()
    ]
    stress_valid = len(stress10) >= minimum_days and len(normal10) >= minimum_days
    recovery_valid = len(recovery20) >= minimum_days and len(stress20) >= minimum_days
    stress_mae = (
        float(
            stress10["future_mae_magnitude_10d"].mean()
            - normal10["future_mae_magnitude_10d"].mean()
        )
        if stress_valid
        else None
    )
    stress_q20 = (
        float(
            stress10["future_return_10d"].quantile(0.20)
            - normal10["future_return_10d"].quantile(0.20)
        )
        if stress_valid
        else None
    )
    recovery_return = (
        float(
            recovery20["future_return_20d"].mean()
            - stress20["future_return_20d"].mean()
        )
        if recovery_valid
        else None
    )
    recovery_positive = (
        float(
            recovery20["future_positive_wealth_fraction_20d"].mean()
            - stress20["future_positive_wealth_fraction_20d"].mean()
        )
        if recovery_valid
        else None
    )
    return Contrast(
        stress_days_10d=int(len(stress10)),
        normal_days_10d=int(len(normal10)),
        recovery_days_20d=int(len(recovery20)),
        stress_days_20d=int(len(stress20)),
        stress_mean_mae_minus_normal_10d=stress_mae,
        stress_q20_return_minus_normal_10d=stress_q20,
        recovery_mean_return_minus_stress_20d=recovery_return,
        recovery_positive_fraction_minus_stress_20d=recovery_positive,
        stress_mean_mae_direction=bool(stress_mae is not None and stress_mae > 0.0),
        stress_q20_return_direction=bool(stress_q20 is not None and stress_q20 < 0.0),
        recovery_mean_return_direction=bool(
            recovery_return is not None and recovery_return > 0.0
        ),
        recovery_positive_fraction_direction=bool(
            recovery_positive is not None and recovery_positive > 0.0
        ),
    )


def build_period_state_metrics(
    merged: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """输出全样本、早期、晚期及周期内的状态条件统计。"""

    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        "FULL": (
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["signal_cutoff"]),
        ),
        "EARLY": tuple(pd.Timestamp(value) for value in config["dates"]["early_period"]),
        "LATE": tuple(pd.Timestamp(value) for value in config["dates"]["late_period"]),
    }
    for cycle_id, cycle in config["pressure_cycles"].items():
        periods[f"CYCLE_{cycle_id}"] = (
            pd.Timestamp(cycle["start"]),
            pd.Timestamp(cycle["end"]),
        )
    rows: list[dict[str, Any]] = []
    for period_id, (start, end) in periods.items():
        sample = merged.loc[merged["date"].between(start, end)]
        for state in STATES:
            group = sample.loc[sample["state"].astype(str).eq(state)]
            rows.append(
                {
                    "period_id": period_id,
                    "start_date": start,
                    "end_date": end,
                    "state": state,
                    "signal_days": int(len(group)),
                    "complete_10d_days": int(group["future_return_10d"].notna().sum()),
                    "mean_return_10d": _safe_float(group["future_return_10d"].mean()),
                    "return_q20_10d": _safe_float(group["future_return_10d"].quantile(0.20)),
                    "mean_mae_10d": _safe_float(
                        group["future_mae_magnitude_10d"].mean()
                    ),
                    "complete_20d_days": int(group["future_return_20d"].notna().sum()),
                    "mean_return_20d": _safe_float(group["future_return_20d"].mean()),
                    "mean_positive_wealth_fraction_20d": _safe_float(
                        group["future_positive_wealth_fraction_20d"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_state_gates(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    outcome_panel: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """按冻结门槛裁决状态可识别性。"""

    merged = state_panel[["date", "state"]].merge(
        outcome_panel, on="date", how="left", validate="one_to_one"
    )
    gates = config["state_gates"]
    state_counts = {
        state: int(state_panel["state"].astype(str).eq(state).sum()) for state in STATES
    }
    minimum_days = {
        state: int(value) for state, value in gates["minimum_state_days"].items()
    }
    days_pass = all(state_counts[state] >= minimum for state, minimum in minimum_days.items())

    complete_episodes = episodes.loc[
        ~episodes["left_censored"] & ~episodes["right_censored"]
    ]
    episode_counts = {
        state: int(complete_episodes["state"].eq(state).sum()) for state in STATES
    }
    episode_minima = {
        state: int(value) for state, value in gates["minimum_state_episodes"].items()
    }
    episodes_pass = all(
        episode_counts[state] >= minimum for state, minimum in episode_minima.items()
    )
    stress_episodes = complete_episodes.loc[complete_episodes["state"].eq(STRESS)]
    recovery_episodes = complete_episodes.loc[complete_episodes["state"].eq(RECOVERY)]
    stress_median = _safe_float(stress_episodes["signal_rows"].median())
    recovery_median = _safe_float(recovery_episodes["signal_rows"].median())
    stress_one_day_share = (
        None
        if stress_episodes.empty
        else float(stress_episodes["signal_rows"].eq(1).mean())
    )
    persistence_spec = gates["persistence"]
    persistence_pass = bool(
        stress_median is not None
        and stress_median
        >= int(persistence_spec["minimum_stress_median_episode_rows"])
        and recovery_median is not None
        and recovery_median
        >= int(persistence_spec["minimum_recovery_median_episode_rows"])
        and stress_one_day_share is not None
        and stress_one_day_share
        <= float(persistence_spec["maximum_stress_one_day_episode_share"])
    )

    full = compute_contrast(merged)
    full_pass = full.all_directions_pass
    period_contrasts: dict[str, dict[str, Any]] = {}
    early_late_pass = True
    for period_id, values in [
        ("EARLY", config["dates"]["early_period"]),
        ("LATE", config["dates"]["late_period"]),
    ]:
        start, end = (pd.Timestamp(value) for value in values)
        contrast = compute_contrast(merged.loc[merged["date"].between(start, end)])
        period_contrasts[period_id] = contrast.as_dict()
        early_late_pass = early_late_pass and contrast.all_directions_pass

    minimum_cycle_days = int(gates["held_out_cycles"]["minimum_cycle_state_days"])
    cycle_rows: list[dict[str, Any]] = []
    for cycle_id, cycle in config["pressure_cycles"].items():
        sample = merged.loc[
            merged["date"].between(pd.Timestamp(cycle["start"]), pd.Timestamp(cycle["end"]))
        ]
        contrast = compute_contrast(sample, minimum_days=minimum_cycle_days)
        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "cycle_label": cycle["label"],
                "start_date": cycle["start"],
                "end_date": cycle["end"],
                **contrast.as_dict(),
            }
        )
    cycle_contrasts = pd.DataFrame(cycle_rows)
    stress_cycle_pass_count = int(cycle_contrasts["stress_directions_pass"].sum())
    recovery_cycle_pass_count = int(cycle_contrasts["recovery_directions_pass"].sum())
    cycles_pass = bool(
        stress_cycle_pass_count
        >= int(gates["held_out_cycles"]["minimum_cycles_passing_stress_directions"])
        and recovery_cycle_pass_count
        >= int(gates["held_out_cycles"]["minimum_cycles_passing_recovery_directions"])
    )

    leave_rows: list[dict[str, Any]] = []
    for cycle_id, cycle in config["pressure_cycles"].items():
        remaining = merged.loc[
            ~merged["date"].between(
                pd.Timestamp(cycle["start"]), pd.Timestamp(cycle["end"])
            )
        ]
        contrast = compute_contrast(remaining)
        leave_rows.append(
            {
                "deleted_cycle_id": cycle_id,
                "deleted_cycle_label": cycle["label"],
                "remaining_signal_days": int(len(remaining)),
                **contrast.as_dict(),
            }
        )
    leave_one_out = pd.DataFrame(leave_rows)
    leave_one_out_pass = bool(leave_one_out["all_directions_pass"].all())

    all_pass = bool(
        days_pass
        and episodes_pass
        and persistence_pass
        and full_pass
        and early_late_pass
        and cycles_pass
        and leave_one_out_pass
    )
    result = {
        "state_day_counts": state_counts,
        "minimum_state_day_gate": {
            "passed": days_pass,
            "thresholds": minimum_days,
        },
        "complete_episode_counts": episode_counts,
        "minimum_episode_gate": {
            "passed": episodes_pass,
            "thresholds": episode_minima,
        },
        "persistence_gate": {
            "passed": persistence_pass,
            "stress_median_episode_rows": stress_median,
            "recovery_median_episode_rows": recovery_median,
            "stress_one_day_episode_share": stress_one_day_share,
            "thresholds": persistence_spec,
        },
        "full_sample_structure_gate": {
            "passed": full_pass,
            **full.as_dict(),
        },
        "early_late_structure_gate": {
            "passed": early_late_pass,
            "periods": period_contrasts,
        },
        "held_out_cycle_structure_gate": {
            "passed": cycles_pass,
            "stress_cycles_passing": stress_cycle_pass_count,
            "recovery_cycles_passing": recovery_cycle_pass_count,
            "thresholds": gates["held_out_cycles"],
        },
        "leave_one_cycle_out_gate": {
            "passed": leave_one_out_pass,
            "every_deletion_preserves_all_directions": leave_one_out_pass,
        },
        "phase_1_state_gate_passed": all_pass,
    }
    return result, cycle_contrasts, leave_one_out


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _pct(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "NA" if number is None else f"{number * 100:.{digits}f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """生成中文结论报告。"""

    gates = report.get("state_gates", {})
    full = gates.get("full_sample_structure_gate", {})
    lines = [
        "# 510300 大盘状态识别与机制路由 V1",
        "",
        f"- 最终状态：`{report['status']}`",
        f"- 数据门：`{'PASS' if report['data_admission']['passed'] else 'FAIL'}`",
        f"- 第一阶段状态门：`{'PASS' if report.get('phase_1_state_gate_passed') else 'FAIL'}`",
        "- 研究边界：状态识别与条件路径诊断；未生成仓位、策略净值、Paper、Shadow 或订单。",
        "",
    ]
    if not report["data_admission"]["passed"]:
        lines.extend(
            [
                "## 数据门停止",
                "",
                "点时数据合同未通过，因此未来条件路径未读取，收益评价为 `NOT_ALLOWED`。",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    counts = gates["state_day_counts"]
    lines.extend(
        [
            "## 状态覆盖与持续性",
            "",
            "| 状态 | 交易日 |",
            "|---|---:|",
            *[f"| `{state}` | {counts.get(state, 0)} |" for state in STATES],
            "",
            f"- 压力区间中位数：{gates['persistence_gate']['stress_median_episode_rows']} 日。",
            f"- 恢复区间中位数：{gates['persistence_gate']['recovery_median_episode_rows']} 日。",
            f"- 单日压力区间占比：{_pct(gates['persistence_gate']['stress_one_day_episode_share'])}。",
            "",
            "## 全样本条件结构",
            "",
            "| 固定方向 | 差值 | 通过 |",
            "|---|---:|---:|",
            f"| 压力减正常：未来10日平均MAE | {_pct(full.get('stress_mean_mae_minus_normal_10d'))} | {'是' if full.get('stress_mean_mae_direction') else '否'} |",
            f"| 压力减正常：未来10日收益20%分位 | {_pct(full.get('stress_q20_return_minus_normal_10d'))} | {'是' if full.get('stress_q20_return_direction') else '否'} |",
            f"| 恢复减压力：未来20日平均收益 | {_pct(full.get('recovery_mean_return_minus_stress_20d'))} | {'是' if full.get('recovery_mean_return_direction') else '否'} |",
            f"| 恢复减压力：未来20日正财富天数比例 | {_pct(full.get('recovery_positive_fraction_minus_stress_20d'))} | {'是' if full.get('recovery_positive_fraction_direction') else '否'} |",
            "",
            "## 冻结门结果",
            "",
            "| 门槛 | 通过 |",
            "|---|---:|",
            f"| 状态交易日样本 | {'是' if gates['minimum_state_day_gate']['passed'] else '否'} |",
            f"| 独立状态区间数量 | {'是' if gates['minimum_episode_gate']['passed'] else '否'} |",
            f"| 状态持续性 | {'是' if gates['persistence_gate']['passed'] else '否'} |",
            f"| 全样本四方向 | {'是' if gates['full_sample_structure_gate']['passed'] else '否'} |",
            f"| 早期与晚期均保持四方向 | {'是' if gates['early_late_structure_gate']['passed'] else '否'} |",
            f"| 五周期中至少四周期方向通过 | {'是' if gates['held_out_cycle_structure_gate']['passed'] else '否'} |",
            f"| 删除任一周期方向不反转 | {'是' if gates['leave_one_cycle_out_gate']['passed'] else '否'} |",
            "",
            "## 裁决与边界",
            "",
            f"- 第二阶段：`{report['phase_2_status']}`。",
            "- 组合收益评价：`NOT_RUN_IN_PHASE_1`。",
            "- 净夏普、净超额、最大回撤：`NOT_COMPUTED`。",
            "- 旧微观结构长周期拒绝状态未被覆盖；失败后禁止修改状态定义、变量、阈值、窗口、周期或仓位营救。",
            "- `DISCOVERY_ONLY / NO_TRADE`；本报告不覆盖用户实际持仓。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(*, write: bool = True) -> dict[str, Any]:
    """验证冻结清单后执行一次第一阶段研究。"""

    config = load_config()
    manifest = validate_manifest(config)
    inputs = load_inputs(config)
    data_audit = audit_inputs(config, inputs)
    paths = {name: _project_path(value) for name, value in config["paths"].items()}
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if write:
        _atomic_json(paths["input_audit"], data_audit)

    if not data_audit["passed"]:
        report = {
            "project_id": PROJECT_ID,
            "status": config["phase_2"]["if_data_gate_fails"],
            "generated_at_asia_shanghai": generated_at,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "data_admission": data_audit,
            "phase_1_state_gate_passed": False,
            "phase_2_status": "SKIPPED_DATA_GATE_FAILED",
            "return_evaluation": "NOT_ALLOWED",
            "strategy_backtest": "NOT_RUN",
            "boundaries": {
                "position_mapping": "DISABLED",
                "paper_signal": "DISABLED",
                "shadow_signal": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
                "live_trading_authorized": False,
            },
        }
        if write:
            _atomic_json(paths["result_json"], report)
            _atomic_text(paths["result_markdown"], render_markdown(report))
        return report

    state_panel = build_signal_panel(config, inputs)
    outcome_panel = build_outcome_panel(
        config,
        state_panel,
        inputs["market"],
        inputs["dividends"],
    )
    episodes = build_episodes(state_panel)
    state_gates, cycle_contrasts, leave_one_out = evaluate_state_gates(
        config,
        state_panel,
        outcome_panel,
        episodes,
    )
    merged = state_panel[["date", "state"]].merge(
        outcome_panel, on="date", how="left", validate="one_to_one"
    )
    period_metrics = build_period_state_metrics(merged, config)
    passed = bool(state_gates["phase_1_state_gate_passed"])
    status = (
        config["phase_2"]["if_state_gate_passes"]
        if passed
        else config["phase_2"]["if_state_gate_fails"]
    )
    report = {
        "project_id": PROJECT_ID,
        "status": status,
        "generated_at_asia_shanghai": generated_at,
        "evidence_class": config["protocol"]["evidence_class"],
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
        "data_admission": data_audit,
        "evaluation_scope": {
            "start": config["dates"]["evaluation_start"],
            "signal_cutoff": config["dates"]["signal_cutoff"],
            "signal_rows": int(len(state_panel)),
            "signal_time": config["clocks"]["unified_state_decision_time"],
            "earliest_execution": config["clocks"]["earliest_hypothetical_execution"],
        },
        "state_gates": state_gates,
        "phase_1_state_gate_passed": passed,
        "phase_2_status": "ALLOWED_NEW_FROZEN_MODULE_STUDIES" if passed else "SKIPPED_STATE_GATE_FAILED",
        "return_evaluation": "CONDITIONAL_PATH_DIAGNOSTICS_ONLY",
        "strategy_backtest": "NOT_RUN_IN_PHASE_1",
        "portfolio_metrics": {
            "net_sharpe": "NOT_COMPUTED",
            "net_excess": "NOT_COMPUTED",
            "maximum_drawdown": "NOT_COMPUTED",
        },
        "artifacts": {
            key: path.relative_to(PROJECT_ROOT).as_posix()
            for key, path in paths.items()
            if key
            in {
                "input_audit",
                "state_panel",
                "outcome_panel",
                "episodes",
                "period_state_metrics",
                "cycle_contrasts",
                "leave_one_cycle_out",
                "result_json",
                "result_markdown",
            }
        },
        "boundaries": {
            "old_microstructure_status_overridden": False,
            "old_cash_rule_read": False,
            "recent_performance_routing_used": False,
            "future_outcomes_in_state_panel": False,
            "same_history_rescue_allowed": False,
            "position_mapping": "DISABLED",
            "paper_signal": "DISABLED",
            "shadow_signal": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    if write:
        _atomic_parquet(paths["state_panel"], state_panel)
        _atomic_parquet(paths["outcome_panel"], outcome_panel)
        _atomic_csv(paths["episodes"], episodes)
        _atomic_csv(paths["period_state_metrics"], period_metrics)
        _atomic_csv(paths["cycle_contrasts"], cycle_contrasts)
        _atomic_csv(paths["leave_one_cycle_out"], leave_one_out)
        _atomic_json(paths["result_json"], report)
        _atomic_text(paths["result_markdown"], render_markdown(report))
    return report


def main() -> int:
    report = run_study(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
