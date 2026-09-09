"""510300 MACD负柱收敛×官方权重广度×下行波动的一次性机制预检。

本模块只做研究预检，不生成仓位、订单或券商连接。结果一旦写出，协议禁止
通过改参数、改窗口、反向使用或替换样本来救援失败候选。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macd_breadth_downside_preflight_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_macd_breadth_downside_preflight_v1_manifest.json"


class ContractError(RuntimeError):
    """协议或输入数据不满足冻结契约。"""


@dataclass(frozen=True)
class PairComparison:
    left_label: str
    right_label: str
    horizon: int
    left_count: int
    right_count: int
    left_mean: float | None
    right_mean: float | None
    mean_difference: float | None
    left_median: float | None
    right_median: float | None
    median_difference: float | None
    left_positive_rate: float | None
    right_positive_rate: float | None
    positive_rate_difference: float | None
    bootstrap_lower_90: float | None
    bootstrap_upper_90: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_label": self.left_label,
            "right_label": self.right_label,
            "horizon_trading_days": self.horizon,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "left_mean": self.left_mean,
            "right_mean": self.right_mean,
            "mean_difference": self.mean_difference,
            "left_median": self.left_median,
            "right_median": self.right_median,
            "median_difference": self.median_difference,
            "left_positive_rate": self.left_positive_rate,
            "right_positive_rate": self.right_positive_rate,
            "positive_rate_difference": self.positive_rate_difference,
            "bootstrap_lower_90": self.bootstrap_lower_90,
            "bootstrap_upper_90": self.bootstrap_upper_90,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"缺少协议文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ContractError("协议文件必须是YAML对象")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    dates = config["dates"]
    outcomes = config["outcomes"]
    governance = config["governance"]
    if protocol["project_id"] != "510300_MACD_BREADTH_DOWNSIDE_PREFLIGHT_V1":
        raise ContractError("project_id不符合冻结候选")
    if protocol["one_shot"] is not True:
        raise ContractError("预检必须声明one_shot=true")
    if scope["execution_asset"] != "510300.SH":
        raise ContractError("执行资产必须固定为510300.SH")
    if scope["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("允许持有范围必须仅为510300和现金")
    forbidden_true = {
        "leverage_allowed": scope["leverage_allowed"],
        "short_selling_allowed": scope["short_selling_allowed"],
        "derivatives_execution_allowed": scope["derivatives_execution_allowed"],
        "live_trading_authorized": scope["live_trading_authorized"],
        "order_generation": governance["order_generation"],
        "broker_connection": governance["broker_connection"],
        "position_change": governance["position_change"],
    }
    if any(value is not False and value != "disabled" for value in forbidden_true.values()):
        raise ContractError(f"研究安全边界存在启用项：{forbidden_true}")
    if pd.Timestamp(dates["evaluation_start"]) >= pd.Timestamp(dates["evaluation_end"]):
        raise ContractError("评价起点必须早于终点")
    if outcomes["horizons_trading_days"] != [5, 10, 20]:
        raise ContractError("预检周期必须固定为5、10、20个交易日")
    if config["event_sampling"]["minimum_gap_within_class_trading_days"] != 10:
        raise ContractError("事件类内最小间隔必须固定为10个交易日")
    if config["bootstrap"]["repetitions"] != 5000:
        raise ContractError("Bootstrap次数必须固定为5000")


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{label}缺少字段：{missing}")


def _load_parquet(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise ContractError(f"{label}文件不存在：{path}")
    frame = pd.read_parquet(path)
    _require_columns(frame, required, label)
    return frame


def _load_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise ContractError(f"{label}文件不存在：{path}")
    frame = pd.read_csv(path)
    _require_columns(frame, required, label)
    return frame


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = config["inputs"]
    market_spec = inputs["etf_daily"]
    dividend_spec = inputs["dividends"]
    breadth_spec = inputs["official_weighted_breadth"]
    market = _load_parquet(
        _project_path(market_spec["path"]),
        market_spec["required_columns"],
        "510300日线",
    )
    dividends = _load_csv(
        _project_path(dividend_spec["path"]),
        dividend_spec["required_columns"],
        "510300分红",
    )
    breadth = _load_parquet(
        _project_path(breadth_spec["path"]),
        breadth_spec["required_columns"],
        "官方权重广度",
    )
    return market, dividends, breadth


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("协议尚未冻结：缺少manifest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("manifest project_id不一致")
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("manifest未处于FROZEN_BEFORE_FIRST_RESULT")
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


def _normalize_market(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = config["dates"]
    symbol = config["scope"]["execution_asset"]
    output = market.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    if output["date"].isna().any():
        raise ContractError("510300日线存在无法解析的日期")
    if "symbol" in output.columns:
        unexpected = sorted(set(output["symbol"].dropna().astype(str)).difference({symbol}))
        if unexpected:
            raise ContractError(f"510300日线包含非目标代码：{unexpected}")
    if output["date"].duplicated().any():
        raise ContractError("510300日线存在重复日期")
    output = output.sort_values("date").reset_index(drop=True)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if output[numeric].isna().any().any():
        raise ContractError("510300日线存在无法解析的数值")
    if (output[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("510300日线存在非正价格")
    high_bad = output["high"] < output[["open", "close", "low"]].max(axis=1)
    low_bad = output["low"] > output[["open", "close", "high"]].min(axis=1)
    if bool((high_bad | low_bad).any()):
        raise ContractError("510300日线存在OHLC约束错误")
    warmup_start = pd.Timestamp(dates["feature_warmup_start"])
    evaluation_end = pd.Timestamp(dates["evaluation_end"])
    output = output.loc[output["date"].between(warmup_start, evaluation_end)].copy()
    if output.empty:
        raise ContractError("510300日线在冻结区间内为空")

    distributions = dividends.copy()
    distributions = distributions.loc[distributions["symbol"].astype(str) == symbol].copy()
    for column in ("record_date", "ex_date", "payment_date"):
        distributions[column] = pd.to_datetime(
            distributions[column], errors="coerce"
        ).dt.normalize()
    distributions["cash_dividend_per_share"] = pd.to_numeric(
        distributions["cash_dividend_per_share"], errors="coerce"
    )
    if distributions[
        ["record_date", "ex_date", "payment_date", "cash_dividend_per_share"]
    ].isna().any().any():
        raise ContractError("分红表存在无法解析的字段")
    if (distributions["cash_dividend_per_share"] <= 0).any():
        raise ContractError("分红金额必须为正数")
    distributions = distributions.sort_values("ex_date").reset_index(drop=True)

    ex_map = distributions.groupby("ex_date")["cash_dividend_per_share"].sum()
    record_map = distributions.groupby("record_date")["cash_dividend_per_share"].sum()
    output["cash_dividend_ex"] = output["date"].map(ex_map).fillna(0.0)
    output["cash_dividend_record"] = output["date"].map(record_map).fillna(0.0)
    prior_close = output["close"].shift(1)
    gross = (output["close"] + output["cash_dividend_ex"]) / prior_close
    gross.iloc[0] = 1.0
    if (gross <= 0).any() or gross.isna().any():
        raise ContractError("总收益价格构造产生无效单日收益")
    output["total_return_gross"] = gross
    output["total_return_close"] = 1000.0 * gross.cumprod()
    output["record_dividend_cumulative"] = output["cash_dividend_record"].cumsum()
    return output.reset_index(drop=True), distributions


def _normalize_breadth(breadth: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = breadth.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["weight_snapshot_date"] = pd.to_datetime(
        output["weight_snapshot_date"], errors="coerce"
    ).dt.normalize()
    if output[["date", "weight_snapshot_date"]].isna().any().any():
        raise ContractError("官方权重广度存在无法解析的日期")
    if output["date"].duplicated().any():
        raise ContractError("官方权重广度存在重复日期")
    output = output.sort_values("date").reset_index(drop=True)
    if (output["weight_snapshot_date"] > output["date"]).any():
        raise ContractError("官方权重广度使用了未来权重快照")
    contract = config["data_contract"]
    if (output["component_count"] != contract["expected_component_count"]).any():
        raise ContractError("官方权重广度并非每日严格300只成分股")
    if (
        output["weight_snapshot_age_calendar_days"]
        > contract["maximum_weight_snapshot_age_calendar_days"]
    ).any():
        raise ContractError("官方权重快照年龄超过冻结上限")
    for column in ("ma20_weight_coverage", "ma60_weight_coverage"):
        if (output[column] < contract["minimum_breadth_weight_coverage"]).any():
            raise ContractError(f"{column}低于冻结覆盖率")
    if not output["valid_for_direction_model"].astype(bool).all():
        raise ContractError("官方权重广度存在不允许进入方向模型的日期")
    return output


def build_daily_features(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    breadth: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market, distributions = _normalize_market(market, dividends, config)
    breadth = _normalize_breadth(breadth, config)
    feature = market.copy()
    total_return = feature["total_return_close"].pct_change()
    vol_window = int(config["features"]["volatility_for_macd"]["window"])
    daily_volatility = total_return.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    lag = int(config["features"]["volatility_for_macd"]["lag_days"])
    feature["lagged_realized_volatility_20d"] = daily_volatility.shift(lag)
    feature["volatility_normalized_return"] = total_return / feature[
        "lagged_realized_volatility_20d"
    ].replace(0.0, np.nan)

    macd = config["features"]["macd"]
    fast = feature["volatility_normalized_return"].ewm(
        span=int(macd["fast_ema"]), adjust=False, min_periods=int(macd["fast_ema"])
    ).mean()
    slow = feature["volatility_normalized_return"].ewm(
        span=int(macd["slow_ema"]), adjust=False, min_periods=int(macd["slow_ema"])
    ).mean()
    feature["macd_line"] = fast - slow
    feature["macd_signal"] = feature["macd_line"].ewm(
        span=int(macd["signal_ema"]), adjust=False, min_periods=int(macd["signal_ema"])
    ).mean()
    feature["macd_histogram"] = feature["macd_line"] - feature["macd_signal"]
    velocity_lag = int(macd["velocity_lag_days"])
    acceleration_lag = int(macd["acceleration_lag_days"])
    feature["macd_velocity_3d"] = feature["macd_histogram"] - feature[
        "macd_histogram"
    ].shift(velocity_lag)
    feature["macd_acceleration_3d"] = feature["macd_velocity_3d"] - feature[
        "macd_velocity_3d"
    ].shift(acceleration_lag)

    downside = total_return.clip(upper=0.0)
    short_span = int(config["features"]["downside_risk"]["short_span"])
    long_span = int(config["features"]["downside_risk"]["long_span"])
    feature["downside_deviation_5"] = np.sqrt(
        downside.pow(2).ewm(
            span=short_span, adjust=False, min_periods=short_span
        ).mean()
    )
    feature["downside_deviation_20"] = np.sqrt(
        downside.pow(2).ewm(
            span=long_span, adjust=False, min_periods=long_span
        ).mean()
    )
    feature["downside_vol_ratio_5_20"] = feature["downside_deviation_5"] / feature[
        "downside_deviation_20"
    ].replace(0.0, np.nan)

    breadth_columns = [
        "date",
        "weight_snapshot_date",
        "weight_snapshot_age_calendar_days",
        "component_count",
        "weight_sum",
        "ma20_weight_coverage",
        "ma60_weight_coverage",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
        "official_weighted_above_ma20_share_change_5d",
        "valid_for_direction_model",
    ]
    feature = feature.merge(
        breadth[breadth_columns], on="date", how="left", validate="one_to_one"
    )
    dates = config["dates"]
    evaluation_mask = feature["date"].between(
        pd.Timestamp(dates["evaluation_start"]), pd.Timestamp(dates["evaluation_end"])
    )
    evaluation = feature.loc[evaluation_mask].copy()
    expected_rows = int(config["data_contract"]["expected_evaluation_rows"])
    if len(evaluation) != expected_rows:
        raise ContractError(
            f"评价期行数不一致：expected={expected_rows}, actual={len(evaluation)}"
        )
    required_feature_columns = [
        "macd_histogram",
        "macd_velocity_3d",
        "macd_acceleration_3d",
        "downside_vol_ratio_5_20",
        "official_weighted_above_ma20_share",
        "official_weighted_above_ma60_share",
        "official_weighted_above_ma20_share_change_5d",
    ]
    if evaluation[required_feature_columns].isna().any().any():
        missing = evaluation[required_feature_columns].isna().sum()
        raise ContractError(f"评价期特征存在缺失：{missing[missing > 0].to_dict()}")

    feature["macd_transition"] = "OUTSIDE_NEGATIVE_HISTOGRAM"
    negative = feature["macd_histogram"] < 0
    converging = negative & (feature["macd_velocity_3d"] > 0)
    expanding = negative & (feature["macd_velocity_3d"] <= 0)
    feature.loc[converging, "macd_transition"] = "CONVERGING"
    feature.loc[expanding, "macd_transition"] = "EXPANDING"
    feature["breadth_state"] = np.where(
        feature["official_weighted_above_ma20_share_change_5d"] > 0,
        "IMPROVING",
        "WEAKENING",
    )
    feature["downside_state"] = np.where(
        feature["downside_vol_ratio_5_20"] < 1,
        "COOLING",
        "RISING",
    )
    feature["group_id"] = (
        feature["macd_transition"]
        + "__"
        + feature["breadth_state"]
        + "__"
        + feature["downside_state"]
    )

    feature = add_forward_outcomes(feature, config)
    evaluation = feature.loc[evaluation_mask].reset_index(drop=True)
    return evaluation, distributions


def add_forward_outcomes(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    opens = output["open"].to_numpy(dtype=float)
    closes = output["close"].to_numpy(dtype=float)
    cumulative_dividend = output["record_dividend_cumulative"].to_numpy(dtype=float)
    dates = pd.to_datetime(output["date"]).to_numpy()
    row_count = len(output)
    for horizon in config["outcomes"]["horizons_trading_days"]:
        returns = np.full(row_count, np.nan, dtype=float)
        adverse = np.full(row_count, np.nan, dtype=float)
        end_dates = np.full(
            row_count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        horizon = int(horizon)
        for signal_index in range(row_count):
            entry_index = signal_index + 1
            exit_index = signal_index + horizon
            if exit_index >= row_count:
                continue
            entry_price = opens[entry_index]
            entitled_before_entry = cumulative_dividend[signal_index]
            exit_wealth = (
                closes[exit_index]
                + cumulative_dividend[exit_index]
                - entitled_before_entry
            )
            returns[signal_index] = exit_wealth / entry_price - 1.0
            path_wealth = (
                closes[entry_index : exit_index + 1]
                + cumulative_dividend[entry_index : exit_index + 1]
                - entitled_before_entry
            )
            adverse[signal_index] = float(np.min(path_wealth / entry_price - 1.0))
            end_dates[signal_index] = dates[exit_index]
        output[f"forward_return_{horizon}d"] = returns
        output[f"forward_mae_{horizon}d"] = adverse
        output[f"outcome_end_date_{horizon}d"] = pd.to_datetime(end_dates)
    return output


def select_events(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    events = features.copy().reset_index(drop=True)
    primary_class = events["macd_transition"].where(
        events["macd_transition"].isin(["CONVERGING", "EXPANDING"])
    )
    previous = primary_class.shift(1)
    events["is_raw_state_entry"] = primary_class.notna() & primary_class.ne(previous)
    gap = int(config["event_sampling"]["minimum_gap_within_class_trading_days"])
    selected = np.zeros(len(events), dtype=bool)
    last_selected: dict[str, int] = {}
    for row_index in np.flatnonzero(events["is_raw_state_entry"].to_numpy()):
        state = str(primary_class.iloc[row_index])
        prior_index = last_selected.get(state)
        if prior_index is None or row_index - prior_index >= gap:
            selected[row_index] = True
            last_selected[state] = row_index
    events["is_selected_event"] = selected
    chosen = events.loc[events["is_selected_event"]].copy()
    chosen["primary_class"] = chosen["macd_transition"]
    chosen["conditional_class"] = "OTHER_CONVERGING_QUALITY"
    high_quality = (
        (chosen["primary_class"] == "CONVERGING")
        & (chosen["breadth_state"] == "IMPROVING")
        & (chosen["downside_state"] == "COOLING")
    )
    low_quality = (
        (chosen["primary_class"] == "CONVERGING")
        & (chosen["breadth_state"] == "WEAKENING")
        & (chosen["downside_state"] == "RISING")
    )
    chosen.loc[high_quality, "conditional_class"] = "HIGH_QUALITY_CONVERGING"
    chosen.loc[low_quality, "conditional_class"] = "LOW_QUALITY_CONVERGING"
    split_date = pd.Timestamp(config["dates"]["structural_split_date"])
    chosen["structural_period"] = np.where(
        chosen["date"] < split_date, "EARLY", "LATE"
    )
    return chosen.reset_index(drop=True)


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _circular_block_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    block_count = math.ceil(len(values) / block_length)
    starts = rng.integers(0, len(values), size=block_count)
    indices = np.concatenate(
        [
            (start + np.arange(block_length, dtype=int)) % len(values)
            for start in starts
        ]
    )[: len(values)]
    return values[indices]


def bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    config: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> tuple[float | None, float | None]:
    left = _finite(left)
    right = _finite(right)
    if len(left) == 0 or len(right) == 0:
        return None, None
    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["random_seed"]) + seed_offset)
    repetitions = int(bootstrap["repetitions"])
    block = int(bootstrap["block_length_events"])
    differences = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        left_sample = _circular_block_sample(left, rng, block)
        right_sample = _circular_block_sample(right, rng, block)
        differences[repetition] = left_sample.mean() - right_sample.mean()
    alpha = 1.0 - float(bootstrap["confidence_level"])
    return (
        float(np.quantile(differences, alpha / 2.0)),
        float(np.quantile(differences, 1.0 - alpha / 2.0)),
    )


def compare_classes(
    events: pd.DataFrame,
    class_column: str,
    left_label: str,
    right_label: str,
    horizon: int,
    config: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> PairComparison:
    outcome = f"forward_return_{horizon}d"
    left = _finite(events.loc[events[class_column] == left_label, outcome])
    right = _finite(events.loc[events[class_column] == right_label, outcome])
    lower, upper = bootstrap_mean_difference(
        left, right, config, seed_offset=seed_offset
    )

    def metric(values: np.ndarray, function: Any) -> float | None:
        return float(function(values)) if len(values) else None

    left_mean = metric(left, np.mean)
    right_mean = metric(right, np.mean)
    left_median = metric(left, np.median)
    right_median = metric(right, np.median)
    left_positive = metric(left, lambda value: np.mean(value > 0))
    right_positive = metric(right, lambda value: np.mean(value > 0))
    return PairComparison(
        left_label=left_label,
        right_label=right_label,
        horizon=horizon,
        left_count=len(left),
        right_count=len(right),
        left_mean=left_mean,
        right_mean=right_mean,
        mean_difference=(
            left_mean - right_mean
            if left_mean is not None and right_mean is not None
            else None
        ),
        left_median=left_median,
        right_median=right_median,
        median_difference=(
            left_median - right_median
            if left_median is not None and right_median is not None
            else None
        ),
        left_positive_rate=left_positive,
        right_positive_rate=right_positive,
        positive_rate_difference=(
            left_positive - right_positive
            if left_positive is not None and right_positive is not None
            else None
        ),
        bootstrap_lower_90=lower,
        bootstrap_upper_90=upper,
    )


def _period_differences(
    events: pd.DataFrame,
    class_column: str,
    left_label: str,
    right_label: str,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    outcome = f"forward_return_{horizon}d"
    payload: dict[str, dict[str, Any]] = {}
    for period in ("EARLY", "LATE"):
        subset = events.loc[events["structural_period"] == period]
        left = _finite(subset.loc[subset[class_column] == left_label, outcome])
        right = _finite(subset.loc[subset[class_column] == right_label, outcome])
        difference = float(left.mean() - right.mean()) if len(left) and len(right) else None
        payload[period] = {
            "left_count": len(left),
            "right_count": len(right),
            "left_mean": float(left.mean()) if len(left) else None,
            "right_mean": float(right.mean()) if len(right) else None,
            "mean_difference": difference,
        }
    return payload


def _group_summary(features: pd.DataFrame, horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    negative = features.loc[
        features["macd_transition"].isin(["CONVERGING", "EXPANDING"])
    ]
    for group_id, group in negative.groupby("group_id", sort=True):
        row: dict[str, Any] = {"group_id": str(group_id), "daily_row_count": len(group)}
        for horizon in horizons:
            returns = _finite(group[f"forward_return_{horizon}d"])
            adverse = _finite(group[f"forward_mae_{horizon}d"])
            row[f"return_{horizon}d_count"] = len(returns)
            row[f"return_{horizon}d_mean"] = float(returns.mean()) if len(returns) else None
            row[f"return_{horizon}d_median"] = (
                float(np.median(returns)) if len(returns) else None
            )
            row[f"return_{horizon}d_positive_rate"] = (
                float(np.mean(returns > 0)) if len(returns) else None
            )
            row[f"mae_{horizon}d_mean"] = float(adverse.mean()) if len(adverse) else None
        rows.append(row)
    return rows


def evaluate_preflight(
    features: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = compare_classes(
        events,
        "primary_class",
        "CONVERGING",
        "EXPANDING",
        10,
        config,
        seed_offset=10,
    )
    primary_periods = _period_differences(
        events, "primary_class", "CONVERGING", "EXPANDING", 10
    )
    converging = _finite(
        events.loc[events["primary_class"] == "CONVERGING", "forward_return_10d"]
    )
    expanding = _finite(
        events.loc[events["primary_class"] == "EXPANDING", "forward_return_10d"]
    )
    if len(converging) > 1 and len(expanding):
        without_best = np.delete(converging, int(np.argmax(converging)))
        delete_best_difference = float(without_best.mean() - expanding.mean())
    else:
        delete_best_difference = None

    conditional_by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in (5, 10, 20):
        comparison = compare_classes(
            events,
            "conditional_class",
            "HIGH_QUALITY_CONVERGING",
            "LOW_QUALITY_CONVERGING",
            horizon,
            config,
            seed_offset=100 + horizon,
        )
        conditional_by_horizon[str(horizon)] = comparison.as_dict()
    conditional_primary = conditional_by_horizon["10"]
    conditional_periods = _period_differences(
        events,
        "conditional_class",
        "HIGH_QUALITY_CONVERGING",
        "LOW_QUALITY_CONVERGING",
        10,
    )

    primary_rules = config["gates"]["primary_h1"]
    primary_gates = {
        "minimum_event_count_each": (
            primary.left_count >= int(primary_rules["minimum_event_count_each"])
            and primary.right_count >= int(primary_rules["minimum_event_count_each"])
        ),
        "mean_return_difference_positive": _positive(primary.mean_difference),
        "median_return_difference_positive": _positive(primary.median_difference),
        "positive_return_rate_difference_at_least_5pp": (
            primary.positive_rate_difference is not None
            and primary.positive_rate_difference
            >= float(primary_rules["positive_return_rate_difference_minimum"])
        ),
        "bootstrap_90pct_lower_bound_positive": _positive(
            primary.bootstrap_lower_90
        ),
        "both_structural_periods_mean_difference_positive": all(
            _positive(primary_periods[period]["mean_difference"])
            for period in ("EARLY", "LATE")
        ),
        "delete_best_converging_event_mean_difference_positive": _positive(
            delete_best_difference
        ),
    }

    conditional_rules = config["gates"]["conditional_confirmation"]
    conditional_gates = {
        "minimum_event_count_each": (
            int(conditional_primary["left_count"])
            >= int(conditional_rules["minimum_event_count_each"])
            and int(conditional_primary["right_count"])
            >= int(conditional_rules["minimum_event_count_each"])
        ),
        "mean_return_difference_positive": _positive(
            conditional_primary["mean_difference"]
        ),
        "median_return_difference_positive": _positive(
            conditional_primary["median_difference"]
        ),
        "positive_return_rate_difference_positive": _positive(
            conditional_primary["positive_rate_difference"]
        ),
        "bootstrap_90pct_lower_bound_positive": _positive(
            conditional_primary["bootstrap_lower_90"]
        ),
        "both_structural_periods_mean_difference_positive": all(
            _positive(conditional_periods[period]["mean_difference"])
            for period in ("EARLY", "LATE")
        ),
        "five_and_twenty_day_mean_differences_positive": all(
            _positive(conditional_by_horizon[str(horizon)]["mean_difference"])
            for horizon in (5, 20)
        ),
    }
    primary_pass = all(primary_gates.values())
    conditional_pass = all(conditional_gates.values())
    passed = primary_pass and conditional_pass
    adjudication = config["adjudication"]
    return {
        "passed": passed,
        "status": adjudication["pass_status"] if passed else adjudication["fail_status"],
        "return_evaluation": (
            adjudication["return_evaluation_if_pass"]
            if passed
            else adjudication["return_evaluation_if_fail"]
        ),
        "sharpe": adjudication["sharpe_if_preflight_only"],
        "primary_h1": {
            "comparison": primary.as_dict(),
            "structural_periods": primary_periods,
            "delete_best_converging_event_mean_difference": delete_best_difference,
            "gates": primary_gates,
            "passed": primary_pass,
        },
        "conditional_confirmation": {
            "comparisons_by_horizon": conditional_by_horizon,
            "structural_periods_10d": conditional_periods,
            "gates": conditional_gates,
            "passed": conditional_pass,
        },
        "eight_group_daily_summary": _group_summary(features, [5, 10, 20]),
    }


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _input_snapshot(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        snapshots[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return snapshots


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    features: pd.DataFrame,
    events: pd.DataFrame,
    distributions: pd.DataFrame,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    dates = config["dates"]
    return {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": evaluation["status"],
        "passed": evaluation["passed"],
        "evidence_class": config["protocol"]["evidence_class"],
        "evaluation_window": {
            "start": dates["evaluation_start"],
            "end": dates["evaluation_end"],
            "trading_rows": len(features),
        },
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "auxiliary_information": config["scope"]["auxiliary_information"],
        },
        "input_audit": {
            "snapshots": _input_snapshot(config),
            "feature_first_date": features["date"].min().date().isoformat(),
            "feature_last_date": features["date"].max().date().isoformat(),
            "feature_rows": len(features),
            "event_rows": len(events),
            "converging_event_rows": int((events["primary_class"] == "CONVERGING").sum()),
            "expanding_event_rows": int((events["primary_class"] == "EXPANDING").sum()),
            "dividend_events_in_evaluation": int(
                distributions["ex_date"].between(
                    pd.Timestamp(dates["evaluation_start"]),
                    pd.Timestamp(dates["evaluation_end"]),
                ).sum()
            ),
            "future_weight_snapshot_rows": int(
                (features["weight_snapshot_date"] > features["date"]).sum()
            ),
            "minimum_ma20_weight_coverage": float(
                features["ma20_weight_coverage"].min()
            ),
            "minimum_ma60_weight_coverage": float(
                features["ma60_weight_coverage"].min()
            ),
            "maximum_weight_snapshot_age_calendar_days": int(
                features["weight_snapshot_age_calendar_days"].max()
            ),
        },
        "freeze": manifest,
        "preflight": evaluation,
        "adjudication": {
            "return_evaluation": evaluation["return_evaluation"],
            "net_sharpe": evaluation["sharpe"],
            "target_net_sharpe": config["adjudication"]["target_net_sharpe"],
            "target_achieved": False,
            "reason": (
                "机制预检通过后仍需独立实现核心模型、成本后组合回测和真正前向验证"
                if evaluation["passed"]
                else "机制预检未通过，冻结协议禁止继续用组合回测、改参数或反向规则救援"
            ),
        },
        "boundaries": {
            "research_only": True,
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
            "parameter_rescue_after_result": "FORBIDDEN",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    primary = report["preflight"]["primary_h1"]
    conditional = report["preflight"]["conditional_confirmation"]
    pcomp = primary["comparison"]
    ccomp = conditional["comparisons_by_horizon"]["10"]

    def percent(value: float | None) -> str:
        return "NA" if value is None else f"{value:.2%}"

    lines = [
        "# 510300 MACD负柱收敛×官方权重广度×下行波动预检",
        "",
        f"- 状态：`{report['status']}`",
        f"- 通过：`{str(report['passed']).lower()}`",
        f"- 评价区间：`{report['evaluation_window']['start']}—{report['evaluation_window']['end']}`",
        f"- 交易日：`{report['evaluation_window']['trading_rows']}`",
        "- 交易资产边界：仅`510300.SH`与现金；本报告不生成仓位或订单。",
        "",
        "## 主假设 H1",
        "",
        f"- 收敛/扩张事件数：`{pcomp['left_count']}` / `{pcomp['right_count']}`",
        f"- 未来10日均值：`{percent(pcomp['left_mean'])}` / `{percent(pcomp['right_mean'])}`",
        f"- 均值差：`{percent(pcomp['mean_difference'])}`",
        f"- 中位数差：`{percent(pcomp['median_difference'])}`",
        f"- 正收益率差：`{percent(pcomp['positive_rate_difference'])}`",
        f"- 区块Bootstrap 90%区间：`[{percent(pcomp['bootstrap_lower_90'])}, {percent(pcomp['bootstrap_upper_90'])}]`",
        f"- H1通过：`{str(primary['passed']).lower()}`",
        "",
        "## 条件确认",
        "",
        f"- 高质量/低质量收敛事件数：`{ccomp['left_count']}` / `{ccomp['right_count']}`",
        f"- 未来10日均值差：`{percent(ccomp['mean_difference'])}`",
        f"- 区块Bootstrap 90%区间：`[{percent(ccomp['bootstrap_lower_90'])}, {percent(ccomp['bootstrap_upper_90'])}]`",
        f"- 条件确认通过：`{str(conditional['passed']).lower()}`",
        "",
        "## 裁决",
        "",
        f"- 收益回测许可：`{report['adjudication']['return_evaluation']}`",
        f"- 夏普率：`{report['adjudication']['net_sharpe']}`",
        f"- 夏普1.2是否实现：`{str(report['adjudication']['target_achieved']).lower()}`",
        f"- 原因：{report['adjudication']['reason']}",
        "- Paper、Shadow、订单、券商连接、实盘：全部关闭。",
        "",
    ]
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def run_preflight(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    market, dividends, breadth = load_inputs(config)
    features, distributions = build_daily_features(market, dividends, breadth, config)
    events = select_events(features, config)
    evaluation = evaluate_preflight(features, events, config)
    report = build_report(
        config, manifest, features, events, distributions, evaluation
    )
    if write:
        paths = config["paths"]
        _atomic_parquet(_project_path(paths["feature_table"]), features)
        _atomic_parquet(_project_path(paths["event_table"]), events)
        _atomic_json(_project_path(paths["result_json"]), report)
        _atomic_text(_project_path(paths["result_markdown"]), render_markdown(report))
    return report


__all__ = [
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "ContractError",
    "add_forward_outcomes",
    "bootstrap_mean_difference",
    "build_daily_features",
    "compare_classes",
    "evaluate_preflight",
    "load_config",
    "run_preflight",
    "select_events",
    "sha256_file",
    "validate_config",
    "validate_manifest",
]
