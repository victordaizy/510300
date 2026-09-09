"""以冻结日前历史行情补充市场与行业价格情境。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class HistoricalContextError(ValueError):
    """历史情境输入或边界非法。"""


@dataclass(frozen=True)
class HistoricalContextRules:
    annualization_trading_days: int
    trend_return_windows: tuple[int, ...]
    moving_average_windows: tuple[int, ...]
    volatility_windows: tuple[int, ...]
    drawdown_window: int
    valuation_percentile_window: int
    participation_percentile_window: int
    analogue_count: int
    analogue_minimum_separation_trading_days: int
    analogue_current_embargo_trading_days: int
    analogue_outcome_horizons: tuple[int, ...]
    minimum_analogue_count: int
    industry_return_windows: tuple[int, ...]
    industry_percentile_windows: tuple[int, ...]
    high_percentile: float
    extreme_percentile: float
    low_percentile: float
    allowed_market_staleness_trading_days: int
    allowed_industry_staleness_trading_days: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HistoricalContextRules":
        required = set(cls.__dataclass_fields__)
        missing = required - set(raw)
        if missing:
            raise HistoricalContextError(f"历史情境规则缺少字段：{sorted(missing)}")
        values = dict(raw)
        for name in (
            "trend_return_windows",
            "moving_average_windows",
            "volatility_windows",
            "analogue_outcome_horizons",
            "industry_return_windows",
            "industry_percentile_windows",
        ):
            values[name] = tuple(int(value) for value in values[name])
            if not values[name] or any(value <= 0 for value in values[name]):
                raise HistoricalContextError(f"{name} 必须是非空正整数列表")
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
        raise HistoricalContextError(f"{field_name} 不是合法日期：{value}") from exc
    if pd.isna(result):
        raise HistoricalContextError(f"{field_name} 不能为空")
    return result.normalize()


def validate_context_config(
    config: Mapping[str, Any],
) -> tuple[HistoricalContextRules, dict[str, bool], dict[str, bool]]:
    """验证历史情境层没有获得回填或交易权限。"""

    if config.get("status") != "FROZEN_BEFORE_ANALOGUE_OUTCOME_READ":
        raise HistoricalContextError("历史情境配置状态非法")
    if config.get("asset") != "510300" or config.get("benchmark") != "H00300":
        raise HistoricalContextError("历史情境对象必须为510300/H00300")
    rules = HistoricalContextRules.from_mapping(config.get("rules", {}))
    if not 0 < rules.low_percentile < rules.high_percentile < rules.extreme_percentile < 1:
        raise HistoricalContextError("历史分位阈值顺序非法")
    if rules.minimum_analogue_count > rules.analogue_count:
        raise HistoricalContextError("最小相似样本数不能大于目标样本数")

    prohibitions = dict(config.get("prohibitions", {}))
    required_prohibitions = {
        "may_fill_funding_liquidity",
        "may_fill_equity_flow",
        "may_infer_national_team_identity",
        "may_change_industry_expectation_gap",
        "may_upgrade_original_no_view",
        "may_call_analogue_outcomes_oos",
    }
    if required_prohibitions - set(prohibitions):
        raise HistoricalContextError("历史情境禁止项不完整")
    enabled = [name for name in required_prohibitions if prohibitions[name] is not False]
    if enabled:
        raise HistoricalContextError(f"历史情境禁止项必须为false：{enabled}")

    safety = dict(config.get("safety", {}))
    required_safety = {
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    }
    if required_safety - set(safety):
        raise HistoricalContextError("安全开关不完整")
    unsafe = [name for name in required_safety if safety[name] is not False]
    if unsafe:
        raise HistoricalContextError(f"安全开关必须为false：{unsafe}")
    return rules, prohibitions, safety


def verify_manifest(
    workspace_root: Path,
    manifest_relative_path: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = (workspace_root / manifest_relative_path).resolve()
    if not path.is_file():
        raise HistoricalContextError(f"冻结清单不存在：{path}")
    if expected_sha256 and file_sha256(path).upper() != expected_sha256.upper():
        raise HistoricalContextError("关联预测冻结清单哈希变化")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for item in manifest.get("frozen_files", []):
        frozen = (workspace_root / str(item["path"])).resolve()
        if not frozen.is_file() or file_sha256(frozen) != item["sha256"]:
            raise HistoricalContextError(f"冻结文件哈希不一致：{item['path']}")
    return manifest


def _rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def percentile(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.mean(values <= values[-1]))

    return numeric.rolling(window, min_periods=window).apply(percentile, raw=True)


def prepare_market_features(
    market_daily: pd.DataFrame,
    as_of_date: Any,
    rules: HistoricalContextRules,
) -> pd.DataFrame:
    """构造截至情境日的点时市场价格特征。"""

    required = {"date", "close", "amount", "pe_ttm"}
    missing = required - set(market_daily.columns)
    if missing:
        raise HistoricalContextError(f"市场行情缺少字段：{sorted(missing)}")
    frame = market_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise HistoricalContextError("市场行情日期非法或重复")
    cutoff = _timestamp(as_of_date, "as_of_date")
    frame = frame.loc[frame["date"].le(cutoff)].sort_values("date").reset_index(drop=True)
    for column in ("close", "amount", "pe_ttm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty or frame["date"].max() != cutoff:
        raise HistoricalContextError("市场行情没有覆盖情境截止日")
    if (frame["close"] <= 0).any():
        raise HistoricalContextError("市场全收益指数包含非正收盘值")

    daily_return = frame["close"].pct_change()
    for window in rules.trend_return_windows:
        frame[f"return_{window}d"] = frame["close"] / frame["close"].shift(window) - 1.0
    for window in rules.moving_average_windows:
        frame[f"close_vs_ma{window}"] = (
            frame["close"] / frame["close"].rolling(window, min_periods=window).mean() - 1.0
        )
    for window in rules.volatility_windows:
        frame[f"rv_{window}d"] = (
            daily_return.rolling(window, min_periods=window).std(ddof=1)
            * np.sqrt(rules.annualization_trading_days)
        )
    drawdown_window = rules.drawdown_window
    frame[f"drawdown_{drawdown_window}d"] = (
        frame["close"]
        / frame["close"].rolling(drawdown_window, min_periods=drawdown_window).max()
        - 1.0
    )
    frame[f"pe_ttm_percentile_{rules.valuation_percentile_window}d"] = (
        _rolling_last_percentile(frame["pe_ttm"], rules.valuation_percentile_window)
    )
    frame[f"amount_percentile_{rules.participation_percentile_window}d"] = (
        _rolling_last_percentile(frame["amount"], rules.participation_percentile_window)
    )
    return frame


def _percentile_label(value: float, rules: HistoricalContextRules) -> str:
    if value >= rules.extreme_percentile:
        return "EXTREME"
    if value >= rules.high_percentile:
        return "HIGH"
    if value <= rules.low_percentile:
        return "LOW"
    return "MIDDLE"


def build_current_market_context(
    features: pd.DataFrame,
    rules: HistoricalContextRules,
    feature_blocks: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """将当前价格特征映射为不含资金身份推断的情境标签。"""

    current = features.iloc[-1]
    required_features = {feature for values in feature_blocks.values() for feature in values}
    missing = required_features - set(features.columns)
    if missing:
        raise HistoricalContextError(f"特征块引用不存在字段：{sorted(missing)}")
    if current[list(required_features)].isna().any():
        raise HistoricalContextError("当前市场特征不完整")

    ret20 = float(current["return_20d"])
    ret60 = float(current["return_60d"])
    ma20 = float(current["close_vs_ma20"])
    ma60 = float(current["close_vs_ma60"])
    if ret20 > 0 and ma20 > 0 and ma60 > 0:
        trend_state = "CONFIRMED_UPTREND"
    elif ma20 > 0 and (ret60 < 0 or ma60 <= 0):
        trend_state = "SHORT_RECOVERY_NOT_CONFIRMED"
    elif ret20 < 0 and ma20 < 0 and ma60 < 0:
        trend_state = "CONFIRMED_DOWNTREND"
    else:
        trend_state = "MIXED_TREND"

    pe_percentile_name = f"pe_ttm_percentile_{rules.valuation_percentile_window}d"
    amount_percentile_name = (
        f"amount_percentile_{rules.participation_percentile_window}d"
    )
    pe_percentile = float(current[pe_percentile_name])
    amount_percentile = float(current[amount_percentile_name])
    rv20 = float(current["rv_20d"])
    rv60 = float(current["rv_60d"])
    valid_rv_history = features["rv_20d"].dropna()
    rv_percentile = float((valid_rv_history <= rv20).mean())
    valuation_state = _percentile_label(pe_percentile, rules)
    participation_state = _percentile_label(amount_percentile, rules)
    volatility_state = _percentile_label(rv_percentile, rules)
    if (
        trend_state == "SHORT_RECOVERY_NOT_CONFIRMED"
        and participation_state in {"HIGH", "EXTREME"}
        and rv20 < rv60
    ):
        price_risk_state = "ACTIVE_RECOVERY_MEDIUM_TERM_UNCONFIRMED"
    elif trend_state == "CONFIRMED_UPTREND" and rv20 <= rv60:
        price_risk_state = "TREND_SUPPORTIVE_PRICE_RISK"
    elif trend_state == "CONFIRMED_DOWNTREND":
        price_risk_state = "PRICE_RISK_ADVERSE"
    else:
        price_risk_state = "PRICE_RISK_MIXED"

    metrics = {
        "close": float(current["close"]),
        "pe_ttm": float(current["pe_ttm"]),
        "amount": float(current["amount"]),
    }
    for feature in sorted(required_features):
        metrics[feature] = float(current[feature])
    metrics["rv_20d_history_percentile"] = rv_percentile
    return {
        "as_of_date": pd.Timestamp(current["date"]).date().isoformat(),
        "trend_state": trend_state,
        "valuation_state": valuation_state,
        "participation_state": participation_state,
        "volatility_state": volatility_state,
        "price_risk_state": price_risk_state,
        "metrics": metrics,
        "can_fill_market_liquidity": False,
        "can_infer_investor_identity": False,
    }


def _robust_scale(frame: pd.DataFrame) -> pd.Series:
    median = frame.median()
    mad = (frame - median).abs().median() * 1.4826
    iqr_scale = (frame.quantile(0.75) - frame.quantile(0.25)) / 1.349
    standard = frame.std(ddof=1)
    return mad.where(mad > 1e-12, iqr_scale).where(lambda value: value > 1e-12, standard).fillna(1.0)


def select_historical_analogues(
    features: pd.DataFrame,
    feature_blocks: Mapping[str, Sequence[str]],
    rules: HistoricalContextRules,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """选择间隔去重的历史相似日，并读取已成熟的过去结果。"""

    all_features = [feature for values in feature_blocks.values() for feature in values]
    if len(all_features) != len(set(all_features)):
        raise HistoricalContextError("不同特征块不能重复使用同一字段")
    current_position = len(features) - 1
    maximum_outcome = max(rules.analogue_outcome_horizons)
    maximum_candidate_position = current_position - max(
        rules.analogue_current_embargo_trading_days, maximum_outcome
    )
    if maximum_candidate_position < 0:
        raise HistoricalContextError("历史长度不足以形成相似候选")
    candidate = features.iloc[: maximum_candidate_position + 1].copy()
    candidate = candidate.dropna(subset=all_features)
    if len(candidate) < rules.minimum_analogue_count:
        raise HistoricalContextError("完整历史相似候选不足")
    current = features.iloc[-1]
    if current[all_features].isna().any():
        raise HistoricalContextError("当前相似匹配特征不完整")

    scale = _robust_scale(candidate[all_features])
    block_distances: dict[str, pd.Series] = {}
    for block_name, block_features in feature_blocks.items():
        difference = (
            candidate[list(block_features)] - current[list(block_features)].astype(float)
        ) / scale[list(block_features)]
        block_distances[block_name] = difference.pow(2).mean(axis=1)
    distance_frame = pd.DataFrame(block_distances)
    candidate["analogue_distance"] = distance_frame.mean(axis=1)
    candidate["source_position"] = candidate.index.astype(int)
    ordered = candidate.sort_values(["analogue_distance", "date"])

    selected_rows: list[pd.Series] = []
    selected_positions: list[int] = []
    for _, row in ordered.iterrows():
        position = int(row["source_position"])
        if any(
            abs(position - previous)
            < rules.analogue_minimum_separation_trading_days
            for previous in selected_positions
        ):
            continue
        selected_rows.append(row)
        selected_positions.append(position)
        if len(selected_rows) == rules.analogue_count:
            break
    if len(selected_rows) < rules.minimum_analogue_count:
        raise HistoricalContextError("去重后的历史相似日不足")

    closes = features["close"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(features["date"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_rows, start=1):
        position = int(row["source_position"])
        record: dict[str, Any] = {
            "rank": rank,
            "analogue_date": dates.iloc[position].date().isoformat(),
            "distance": float(row["analogue_distance"]),
            "feature_snapshot": {feature: float(row[feature]) for feature in all_features},
        }
        for horizon in rules.analogue_outcome_horizons:
            endpoint = position + horizon
            if endpoint > current_position:
                raise HistoricalContextError("相似日结果尚未成熟")
            path = closes.iloc[position + 1 : endpoint + 1] / closes.iloc[position] - 1.0
            record[f"forward_return_{horizon}d"] = float(
                closes.iloc[endpoint] / closes.iloc[position] - 1.0
            )
            record[f"minimum_path_{horizon}d"] = float(path.min())
            record[f"maturity_date_{horizon}d"] = dates.iloc[endpoint].date().isoformat()
        records.append(record)

    summary: dict[str, Any] = {
        "selected_count": len(records),
        "candidate_count_before_spacing": int(len(candidate)),
        "interpretation": "DESCRIPTIVE_CONDITIONAL_SAMPLE_NOT_OOS",
    }
    for horizon in rules.analogue_outcome_horizons:
        returns = pd.Series([row[f"forward_return_{horizon}d"] for row in records])
        paths = pd.Series([row[f"minimum_path_{horizon}d"] for row in records])
        summary[f"horizon_{horizon}d"] = {
            "return_median": float(returns.median()),
            "return_q25": float(returns.quantile(0.25)),
            "return_q75": float(returns.quantile(0.75)),
            "positive_share": float(returns.gt(0).mean()),
            "minimum_path_median": float(paths.median()),
            "minimum_path_worst": float(paths.min()),
        }
    return records, summary


def build_industry_price_context(
    industry_daily: pd.DataFrame,
    forecast_rows: Sequence[Mapping[str, Any]],
    market_as_of_date: Any,
    calendar: pd.DatetimeIndex,
    rules: HistoricalContextRules,
) -> dict[str, Any]:
    """构造行业自身历史价格位置，并保留数据过期标记。"""

    required = {"date", "industry_l1", "industry_return_1d"}
    missing = required - set(industry_daily.columns)
    if missing:
        raise HistoricalContextError(f"行业收益缺少字段：{sorted(missing)}")
    daily = industry_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["industry_return_1d"] = pd.to_numeric(
        daily["industry_return_1d"], errors="coerce"
    )
    if daily[["date", "industry_l1"]].duplicated().any() or daily["date"].isna().any():
        raise HistoricalContextError("行业收益存在重复行业日期或非法日期")
    cutoff = _timestamp(market_as_of_date, "market_as_of_date")
    daily = daily.loc[daily["date"].le(cutoff)].copy()
    industry_as_of = pd.Timestamp(daily["date"].max())
    stale_trading_days = int(len(calendar[(calendar > industry_as_of) & (calendar <= cutoff)]))
    forecast = pd.DataFrame(list(forecast_rows))[
        ["industry_l1", "sector_weight", "expectation_gap"]
    ].copy()
    rows: list[dict[str, Any]] = []
    for item in forecast.itertuples(index=False):
        history = daily.loc[daily["industry_l1"].eq(item.industry_l1)].sort_values("date")
        if history.empty or history["date"].max() != industry_as_of:
            raise HistoricalContextError(f"行业收益未覆盖统一截止日：{item.industry_l1}")
        returns = history["industry_return_1d"].astype(float)
        row: dict[str, Any] = {
            "industry_l1": item.industry_l1,
            "sector_weight": float(item.sector_weight),
            "expectation_gap": item.expectation_gap,
        }
        rolling_cache: dict[int, pd.Series] = {}
        for window in set(rules.industry_return_windows) | set(
            rules.industry_percentile_windows
        ):
            rolling_cache[window] = (
                (1.0 + returns).rolling(window, min_periods=window).apply(np.prod, raw=True)
                - 1.0
            )
        for window in rules.industry_return_windows:
            value = rolling_cache[window].iloc[-1]
            row[f"return_{window}d"] = float(value) if pd.notna(value) else None
        for window in rules.industry_percentile_windows:
            history_values = rolling_cache[window].dropna()
            current_value = rolling_cache[window].iloc[-1]
            row[f"return_{window}d_history_percentile"] = (
                float((history_values <= current_value).mean())
                if pd.notna(current_value) and len(history_values)
                else None
            )
        percentile60 = row.get("return_60d_history_percentile")
        if percentile60 is None:
            position_state = "NO_VIEW"
        elif percentile60 >= rules.high_percentile:
            position_state = "EXTENDED_UP"
        elif percentile60 <= rules.low_percentile:
            position_state = "EXTENDED_DOWN"
        else:
            position_state = "MIDDLE_RANGE"
        if item.expectation_gap == "POSITIVE":
            overlay = (
                "POSITIVE_GAP_BUT_PRICE_ADVANCED"
                if position_state == "EXTENDED_UP"
                else "POSITIVE_GAP_NOT_PRICE_EXTENDED"
            )
        elif item.expectation_gap == "NEGATIVE":
            overlay = (
                "NEGATIVE_GAP_BUT_PRICE_ALREADY_WEAK"
                if position_state == "EXTENDED_DOWN"
                else "NEGATIVE_GAP_NOT_FULLY_VISIBLE_IN_PRICE"
            )
        else:
            overlay = "PRICE_CONTEXT_ONLY_NO_GAP_CHANGE"
        row["price_position_state"] = position_state
        row["expectation_gap_price_overlay"] = overlay
        rows.append(row)

    result_frame = pd.DataFrame(rows)
    for window in rules.industry_return_windows:
        result_frame[f"return_{window}d_cross_section_rank"] = result_frame[
            f"return_{window}d"
        ].rank(pct=True, method="average")
    result_rows = result_frame.sort_values("sector_weight", ascending=False).to_dict(
        orient="records"
    )
    current_eligible = stale_trading_days <= rules.allowed_industry_staleness_trading_days
    return {
        "data_as_of_date": industry_as_of.date().isoformat(),
        "market_as_of_date": cutoff.date().isoformat(),
        "stale_trading_days": stale_trading_days,
        "state": "CURRENT" if current_eligible else "STALE_HISTORICAL_CONTEXT",
        "eligible_to_fill_current_industry_state": current_eligible,
        "industry_count": len(result_rows),
        "industry_rows": result_rows,
    }


def build_historical_context_report(
    config: Mapping[str, Any],
    rules: HistoricalContextRules,
    prohibitions: Mapping[str, bool],
    safety: Mapping[str, bool],
    market_context: Mapping[str, Any],
    analogues: Sequence[Mapping[str, Any]],
    analogue_summary: Mapping[str, Any],
    industry_context: Mapping[str, Any],
    frozen_result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """组合补充报告，但不改变原预测结论。"""

    if frozen_result.get("final_state") != "NO_VIEW":
        raise HistoricalContextError("原预测不再是冻结的NO_VIEW")
    state = (
        "HISTORICAL_CONTEXT_READY"
        if industry_context["eligible_to_fill_current_industry_state"]
        else "PARTIAL_STALE_INDUSTRY_CONTEXT"
    )
    return {
        "supplement_version": config["version"],
        "status": state,
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "original_prediction_state": frozen_result["final_state"],
        "original_industry_aggregation_state": frozen_result["industry_aggregation"]["state"],
        "original_prediction_changed": False,
        "market_price_context": dict(market_context),
        "historical_analogues": list(analogues),
        "historical_analogue_summary": dict(analogue_summary),
        "industry_price_context": dict(industry_context),
        "fields_supplemented": [
            "PRICE_RISK_CONTEXT",
            "VALUATION_HISTORICAL_POSITION",
            "PARTICIPATION_HISTORICAL_POSITION",
            "DESCRIPTIVE_HISTORICAL_ANALOGUES",
            "STALE_INDUSTRY_PRICE_POSITION",
        ],
        "fields_still_unobserved": [
            "CURRENT_FUNDING_LIQUIDITY",
            "CURRENT_EQUITY_FLOW",
            "CURRENT_NATIONAL_TEAM_HOLDINGS",
            "CURRENT_NATIONAL_TEAM_ACTIVITY",
            "CURRENT_INDUSTRY_PRICE_CONTEXT_AFTER_2026-08-14",
        ],
        "prohibitions": dict(prohibitions),
        "safety": dict(safety),
        "provenance": dict(provenance),
        "interpretation": (
            "历史行情补充了价格情境，但不能把原NO_VIEW升级为方向观点；"
            "相似行情结果是描述性条件样本，不是样本外Edge。"
        ),
    }


def render_historical_context_markdown(report: Mapping[str, Any]) -> str:
    """渲染历史行情情境补充报告。"""

    market = report["market_price_context"]
    summary = report["historical_analogue_summary"]
    industry = report["industry_price_context"]
    lines = [
        "# 历史行情情境补充 V1",
        "",
        f"- 情境截止：`{report['as_of_date']}`",
        f"- 状态：`{report['status']}`",
        f"- 原预测：`{report['original_prediction_state']}`（未改变）",
        f"- 价格风险情境：`{market['price_risk_state']}`",
        f"- 趋势：`{market['trend_state']}`",
        f"- 估值历史位置：`{market['valuation_state']}`",
        f"- 成交参与位置：`{market['participation_state']}`",
        f"- 波动历史位置：`{market['volatility_state']}`",
        "",
        "## 当前关键指标",
        "",
    ]
    for name, value in market["metrics"].items():
        lines.append(f"- `{name}`：`{value:.6f}`")
    lines.extend(
        [
            "",
            "## 历史相似行情",
            "",
            f"选出 `{summary['selected_count']}` 个间隔去重的相似日。",
            "",
            "| 排名 | 相似日 | 距离 | 后20日 | 20日最差路径 | 后60日 | 60日最差路径 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["historical_analogues"]:
        lines.append(
            "| {rank} | {date} | {distance:.4f} | {r20:.2%} | {m20:.2%} | {r60:.2%} | {m60:.2%} |".format(
                rank=row["rank"],
                date=row["analogue_date"],
                distance=row["distance"],
                r20=row["forward_return_20d"],
                m20=row["minimum_path_20d"],
                r60=row["forward_return_60d"],
                m60=row["minimum_path_60d"],
            )
        )
    lines.extend(["", "相似样本分布：", ""])
    for horizon in (20, 60):
        item = summary[f"horizon_{horizon}d"]
        lines.append(
            f"- {horizon}日：中位数 `{item['return_median']:.2%}`，"
            f"四分位区间 `{item['return_q25']:.2%}` 至 `{item['return_q75']:.2%}`，"
            f"正收益比例 `{item['positive_share']:.1%}`，"
            f"最差路径样本 `{item['minimum_path_worst']:.2%}`。"
        )
    lines.extend(
        [
            "",
            "该分布是描述性条件样本，不是样本外胜率。",
            "",
            "## 行业价格位置",
            "",
            f"行业行情截止 `{industry['data_as_of_date']}`，落后市场截止 "
            f"`{industry['stale_trading_days']}` 个交易日，状态为 `{industry['state']}`。",
            "",
            "| 行业 | 权重 | 原预期差 | 20日收益 | 60日收益 | 60日历史分位 | 价格位置 | 旁注 |",
            "|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in industry["industry_rows"]:
        lines.append(
            "| {industry} | {weight:.2%} | {gap} | {r20:.2%} | {r60:.2%} | {p60:.1%} | {state} | {overlay} |".format(
                industry=row["industry_l1"],
                weight=row["sector_weight"],
                gap=row["expectation_gap"],
                r20=row["return_20d"],
                r60=row["return_60d"],
                p60=row["return_60d_history_percentile"],
                state=row["price_position_state"],
                overlay=row["expectation_gap_price_overlay"],
            )
        )
    lines.extend(
        [
            "",
            "## 仍未补齐",
            "",
            *[f"- `{field}`" for field in report["fields_still_unobserved"]],
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)
