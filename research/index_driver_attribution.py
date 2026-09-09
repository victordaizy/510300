"""510300 点时指数截面与板块贡献归因。

本模块只构造同期归因事实，不读取任何未来收益，不产生预测、仓位或订单。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttributionRules:
    """点时归因的冻结数据门槛。"""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    member_count_minimum: int
    member_count_maximum: int
    weight_sum_minimum: float
    weight_sum_maximum: float
    maximum_weight_age_days: int
    minimum_price_coverage_weight: float
    minimum_return_coverage_weight: float
    minimum_industry_coverage_weight: float
    maximum_identity_error: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "AttributionRules":
        protocol = contract["protocol"]
        gates = contract["quality_gates"]
        weight_range = gates["weight_sum_percent_range"]
        return cls(
            start_date=pd.Timestamp(protocol["data_start"]),
            end_date=pd.Timestamp(protocol["data_cutoff"]),
            member_count_minimum=int(gates["member_count_minimum"]),
            member_count_maximum=int(gates["member_count_maximum"]),
            weight_sum_minimum=float(weight_range[0]) / 100.0,
            weight_sum_maximum=float(weight_range[1]) / 100.0,
            maximum_weight_age_days=int(
                gates["maximum_weight_snapshot_age_calendar_days"]
            ),
            minimum_price_coverage_weight=float(
                gates["minimum_price_coverage_weight"]
            ),
            minimum_return_coverage_weight=float(
                gates["minimum_return_coverage_weight"]
            ),
            minimum_industry_coverage_weight=float(
                gates["minimum_industry_coverage_weight"]
            ),
            maximum_identity_error=float(
                gates["maximum_contribution_identity_absolute_error"]
            ),
        )


@dataclass(frozen=True)
class IndexDriverAttribution:
    """点时成分截面、板块贡献和每日审计摘要。"""

    cross_section: pd.DataFrame
    industry: pd.DataFrame
    daily: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name}缺少字段：{missing}")


def _prepare_weights(weights: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        weights,
        {"index_code", "con_code", "trade_date", "weight", "source"},
        "官方权重",
    )
    data = weights.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["con_code"] = data["con_code"].astype(str)
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    if data[["trade_date", "con_code"]].isna().any().any():
        raise ValueError("官方权重存在空交易日或空证券代码")
    if data[["trade_date", "con_code"]].duplicated().any():
        raise ValueError("官方权重存在重复快照证券")
    if data["weight"].isna().any() or (data["weight"] < 0).any():
        raise ValueError("官方权重存在空值或负值")
    data["weight_decimal"] = data["weight"] / 100.0
    return data.sort_values(["trade_date", "con_code"]).reset_index(drop=True)


def _prepare_constituent_daily(constituent_daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        constituent_daily,
        {"date", "con_code", "total_return_close", "is_suspended"},
        "成分股总收益日线",
    )
    data = constituent_daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["con_code"] = data["con_code"].astype(str)
    data["total_return_close"] = pd.to_numeric(
        data["total_return_close"], errors="coerce"
    )
    if data[["date", "con_code"]].isna().any().any():
        raise ValueError("成分股总收益日线存在空日期或空证券代码")
    if data[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股总收益日线存在重复证券日期")
    invalid_price = data["total_return_close"].notna() & data[
        "total_return_close"
    ].le(0)
    if invalid_price.any():
        raise ValueError("成分股总收益价格存在零值或负值")

    data.sort_values(["con_code", "date"], inplace=True)
    trading_dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    date_position = pd.Series(
        np.arange(len(trading_dates), dtype=int), index=trading_dates
    )
    data["trading_date_position"] = data["date"].map(date_position)
    grouped = data.groupby("con_code", sort=False)
    data["previous_total_return_close"] = grouped["total_return_close"].shift(1)
    data["previous_trading_date_position"] = grouped[
        "trading_date_position"
    ].shift(1)
    adjacent = data["trading_date_position"].sub(
        data["previous_trading_date_position"]
    ).eq(1)
    data["return_1d"] = np.where(
        adjacent
        & data["total_return_close"].notna()
        & data["previous_total_return_close"].notna(),
        data["total_return_close"] / data["previous_total_return_close"] - 1.0,
        np.nan,
    )
    return data.sort_values(["date", "con_code"]).reset_index(drop=True)


def _prepare_industry_intervals(industry_intervals: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        industry_intervals,
        {
            "con_code",
            "industry_l1",
            "in_date",
            "out_date",
            "classification_usage",
            "source",
        },
        "点时行业区间",
    )
    data = industry_intervals.copy()
    data["con_code"] = data["con_code"].astype(str)
    data["industry_l1"] = data["industry_l1"].astype("string")
    data["in_date"] = pd.to_datetime(data["in_date"], errors="coerce")
    data["out_date"] = pd.to_datetime(data["out_date"], errors="coerce")
    if data[["con_code", "industry_l1", "in_date"]].isna().any().any():
        raise ValueError("点时行业区间存在空证券、空一级行业或空生效日")
    invalid_interval = data["out_date"].notna() & data["out_date"].le(
        data["in_date"]
    )
    if invalid_interval.any():
        raise ValueError("点时行业区间存在非正区间")
    point_in_time = data["classification_usage"].astype(str).str.contains(
        "POINT_IN_TIME|PIT", case=False, regex=True, na=False
    )
    if not point_in_time.all():
        raise ValueError("点时行业区间混入非点时分类")
    if data[
        ["con_code", "industry_l1", "in_date", "out_date", "source"]
    ].duplicated().any():
        raise ValueError("点时行业区间存在完全重复记录")
    return data.sort_values(["con_code", "in_date"]).reset_index(drop=True)


def _active_industry_for_date(
    industry_intervals: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    active = industry_intervals.loc[
        industry_intervals["in_date"].le(date)
        & (
            industry_intervals["out_date"].isna()
            | industry_intervals["out_date"].gt(date)
        ),
        ["con_code", "industry_l1", "source", "classification_usage", "in_date", "out_date"],
    ].copy()
    if active["con_code"].duplicated().any():
        duplicated = sorted(
            active.loc[active["con_code"].duplicated(False), "con_code"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"{date.date()}存在重复有效行业区间：{duplicated[:10]}")
    return active.rename(
        columns={
            "source": "industry_source",
            "classification_usage": "industry_classification_usage",
            "in_date": "industry_in_date",
            "out_date": "industry_out_date",
        }
    )


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    selected_values = values.loc[valid].astype(float)
    selected_weights = weights.loc[valid].astype(float)
    mean = float(np.average(selected_values, weights=selected_weights))
    variance = float(
        np.average((selected_values - mean) ** 2, weights=selected_weights)
    )
    return float(np.sqrt(max(variance, 0.0)))


def _industry_rows(cross: pd.DataFrame) -> pd.DataFrame:
    mapped = cross.loc[cross["industry_available"]].copy()
    rows: list[dict[str, Any]] = []
    for industry_name, group in mapped.groupby("industry_l1", sort=True):
        return_available = group["return_available"]
        return_coverage = float(
            group.loc[return_available, "snapshot_weight"].sum()
        )
        contribution = group.loc[
            return_available, "component_return_contribution_1d"
        ].sum(min_count=1)
        industry_return = (
            float(contribution / return_coverage)
            if return_coverage > 0 and pd.notna(contribution)
            else np.nan
        )
        advancer_weight = float(
            group.loc[
                return_available & group["constituent_return_1d"].gt(0),
                "snapshot_weight",
            ].sum()
        )
        rows.append(
            {
                "date": pd.Timestamp(group["date"].iloc[0]),
                "weight_snapshot_date": pd.Timestamp(
                    group["weight_snapshot_date"].iloc[0]
                ),
                "industry_l1": str(industry_name),
                "industry_source": str(group["industry_source"].iloc[0]),
                "industry_weight": float(group["snapshot_weight"].sum()),
                "return_coverage_weight": return_coverage,
                "industry_return_1d": industry_return,
                "weighted_return_contribution_1d": (
                    float(contribution) if pd.notna(contribution) else np.nan
                ),
                "industry_advancer_share_1d": (
                    advancer_weight / return_coverage
                    if return_coverage > 0
                    else np.nan
                ),
                "component_count": int(group["con_code"].nunique()),
                "return_component_count": int(return_available.sum()),
            }
        )
    return pd.DataFrame(rows)


def _no_view_row(date: pd.Timestamp, failure_category: str) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date),
        "output": "NO_VIEW",
        "failure_category": failure_category,
        "valid_for_attribution": False,
    }


def build_index_driver_attribution(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: AttributionRules,
) -> IndexDriverAttribution:
    """构造点时指数截面及同期板块贡献，不读取未来收益。"""

    point_in_time_weights = _prepare_weights(weights)
    daily_prices = _prepare_constituent_daily(constituent_daily)
    point_in_time_industries = _prepare_industry_intervals(industry_intervals)

    research_dates = pd.DatetimeIndex(
        sorted(
            date
            for date in daily_prices["date"].dropna().unique()
            if rules.start_date <= pd.Timestamp(date) <= rules.end_date
        )
    )
    if research_dates.empty:
        raise ValueError("冻结日期范围内没有成分股交易日")

    weight_dates = pd.DatetimeIndex(
        sorted(point_in_time_weights["trade_date"].dropna().unique())
    )
    if weight_dates.empty:
        raise ValueError("官方权重没有有效快照")
    weights_by_date = {
        pd.Timestamp(date): group.copy()
        for date, group in point_in_time_weights.groupby("trade_date", sort=False)
    }
    prices_by_date = {
        pd.Timestamp(date): group.copy()
        for date, group in daily_prices.groupby("date", sort=False)
    }

    cross_frames: list[pd.DataFrame] = []
    industry_frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []

    for date in research_dates:
        snapshot_position = int(weight_dates.searchsorted(date, side="right") - 1)
        if snapshot_position < 0:
            daily_rows.append(_no_view_row(date, "NO_PRIOR_WEIGHT_SNAPSHOT"))
            continue
        snapshot_date = pd.Timestamp(weight_dates[snapshot_position])
        snapshot = weights_by_date[snapshot_date][
            ["index_code", "con_code", "weight_decimal", "source"]
        ].copy()
        snapshot.rename(
            columns={
                "weight_decimal": "snapshot_weight",
                "source": "weight_source",
            },
            inplace=True,
        )
        snapshot["weight_snapshot_date"] = snapshot_date
        snapshot["weight_snapshot_age_calendar_days"] = int(
            (date - snapshot_date).days
        )

        price = prices_by_date.get(
            date,
            pd.DataFrame(
                columns=[
                    "con_code",
                    "total_return_close",
                    "return_1d",
                    "is_suspended",
                ]
            ),
        )[
            ["con_code", "total_return_close", "return_1d", "is_suspended"]
        ].copy()
        price.rename(
            columns={
                "total_return_close": "constituent_total_return_close",
                "return_1d": "constituent_return_1d",
            },
            inplace=True,
        )
        industry = _active_industry_for_date(point_in_time_industries, date)
        cross = snapshot.merge(price, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(
            industry, on="con_code", how="left", validate="one_to_one"
        )
        cross.insert(0, "date", pd.Timestamp(date))
        cross["price_available"] = cross[
            "constituent_total_return_close"
        ].notna()
        cross["return_available"] = cross["constituent_return_1d"].notna()
        cross["industry_available"] = cross["industry_l1"].notna()
        cross["component_return_contribution_1d"] = (
            cross["snapshot_weight"] * cross["constituent_return_1d"]
        )
        cross["attributed_return_contribution_1d"] = cross[
            "component_return_contribution_1d"
        ].where(cross["industry_available"])
        cross.sort_values("con_code", inplace=True)
        cross.reset_index(drop=True, inplace=True)
        cross_frames.append(cross)

        industry_daily = _industry_rows(cross)
        if not industry_daily.empty:
            industry_frames.append(industry_daily)

        member_count = int(cross["con_code"].nunique())
        weight_sum = float(cross["snapshot_weight"].sum())
        price_coverage = float(
            cross.loc[cross["price_available"], "snapshot_weight"].sum()
        )
        return_coverage = float(
            cross.loc[cross["return_available"], "snapshot_weight"].sum()
        )
        industry_coverage = float(
            cross.loc[cross["industry_available"], "snapshot_weight"].sum()
        )
        component_contribution = cross.loc[
            cross["return_available"], "component_return_contribution_1d"
        ].sum(min_count=1)
        attributed_component_contribution = cross.loc[
            cross["return_available"] & cross["industry_available"],
            "component_return_contribution_1d",
        ].sum(min_count=1)
        industry_contribution = (
            industry_daily["weighted_return_contribution_1d"].sum(min_count=1)
            if not industry_daily.empty
            else np.nan
        )
        identity_error = (
            abs(float(industry_contribution - attributed_component_contribution))
            if pd.notna(industry_contribution)
            and pd.notna(attributed_component_contribution)
            else np.nan
        )
        unattributed_contribution = (
            float(component_contribution - attributed_component_contribution)
            if pd.notna(component_contribution)
            and pd.notna(attributed_component_contribution)
            else np.nan
        )

        valid_industry_returns = industry_daily.dropna(
            subset=["industry_return_1d", "weighted_return_contribution_1d"]
        )
        if valid_industry_returns.empty:
            leading_industry = None
            leading_industry_contribution = np.nan
            positive_contribution_total = np.nan
            negative_contribution_total = np.nan
            top1_positive_share = np.nan
            top3_positive_share = np.nan
            positive_industry_weight_share = np.nan
            industry_return_dispersion = np.nan
        else:
            leader = valid_industry_returns.loc[
                valid_industry_returns[
                    "weighted_return_contribution_1d"
                ].idxmax()
            ]
            leading_industry = str(leader["industry_l1"])
            leading_industry_contribution = float(
                leader["weighted_return_contribution_1d"]
            )
            positive = valid_industry_returns.loc[
                valid_industry_returns["weighted_return_contribution_1d"].gt(0),
                "weighted_return_contribution_1d",
            ].sort_values(ascending=False)
            negative = valid_industry_returns.loc[
                valid_industry_returns["weighted_return_contribution_1d"].lt(0),
                "weighted_return_contribution_1d",
            ]
            positive_contribution_total = float(positive.sum())
            negative_contribution_total = float(negative.sum())
            top1_positive_share = (
                float(positive.iloc[0] / positive_contribution_total)
                if positive_contribution_total > 0
                else np.nan
            )
            top3_positive_share = (
                float(positive.head(3).sum() / positive_contribution_total)
                if positive_contribution_total > 0
                else np.nan
            )
            positive_industry_weight = float(
                valid_industry_returns.loc[
                    valid_industry_returns["industry_return_1d"].gt(0),
                    "industry_weight",
                ].sum()
            )
            positive_industry_weight_share = (
                positive_industry_weight / industry_coverage
                if industry_coverage > 0
                else np.nan
            )
            industry_return_dispersion = _weighted_std(
                valid_industry_returns["industry_return_1d"],
                valid_industry_returns["industry_weight"],
            )

        failures: list[str] = []
        weight_age = int((date - snapshot_date).days)
        if weight_age > rules.maximum_weight_age_days:
            failures.append("STALE_WEIGHT_SNAPSHOT")
        if not rules.member_count_minimum <= member_count <= rules.member_count_maximum:
            failures.append("MEMBER_COUNT_OUT_OF_RANGE")
        if not rules.weight_sum_minimum <= weight_sum <= rules.weight_sum_maximum:
            failures.append("WEIGHT_SUM_OUT_OF_RANGE")
        if price_coverage < rules.minimum_price_coverage_weight:
            failures.append("PRICE_COVERAGE_BELOW_GATE")
        if return_coverage < rules.minimum_return_coverage_weight:
            failures.append("RETURN_COVERAGE_BELOW_GATE")
        if industry_coverage < rules.minimum_industry_coverage_weight:
            failures.append("INDUSTRY_COVERAGE_BELOW_GATE")
        if pd.isna(identity_error):
            failures.append("CONTRIBUTION_IDENTITY_UNAVAILABLE")
        elif identity_error > rules.maximum_identity_error:
            failures.append("CONTRIBUTION_IDENTITY_FAILED")

        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "weight_snapshot_date": snapshot_date,
                "weight_snapshot_age_calendar_days": weight_age,
                "component_count": member_count,
                "weight_sum": weight_sum,
                "price_coverage_weight": price_coverage,
                "return_coverage_weight": return_coverage,
                "industry_coverage_weight": industry_coverage,
                "snapshot_weighted_constituent_return_1d": (
                    float(component_contribution)
                    if pd.notna(component_contribution)
                    else np.nan
                ),
                "attributed_component_contribution_1d": (
                    float(attributed_component_contribution)
                    if pd.notna(attributed_component_contribution)
                    else np.nan
                ),
                "industry_contribution_sum_1d": (
                    float(industry_contribution)
                    if pd.notna(industry_contribution)
                    else np.nan
                ),
                "unattributed_contribution_1d": unattributed_contribution,
                "contribution_identity_absolute_error": identity_error,
                "leading_industry_l1": leading_industry,
                "leading_industry_contribution_1d": leading_industry_contribution,
                "positive_industry_contribution_total_1d": positive_contribution_total,
                "negative_industry_contribution_total_1d": negative_contribution_total,
                "top1_positive_contribution_share": top1_positive_share,
                "top3_positive_contribution_share": top3_positive_share,
                "positive_industry_weight_share_1d": positive_industry_weight_share,
                "industry_return_dispersion_1d": industry_return_dispersion,
                "valid_for_attribution": not failures,
                "output": "ATTRIBUTION" if not failures else "NO_VIEW",
                "failure_category": "PASS" if not failures else "|".join(failures),
            }
        )

    cross_section = (
        pd.concat(cross_frames, ignore_index=True)
        if cross_frames
        else pd.DataFrame()
    )
    industry_output = (
        pd.concat(industry_frames, ignore_index=True)
        if industry_frames
        else pd.DataFrame()
    )
    daily_output = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    return IndexDriverAttribution(
        cross_section=cross_section,
        industry=industry_output,
        daily=daily_output,
    )


__all__ = [
    "AttributionRules",
    "IndexDriverAttribution",
    "build_index_driver_attribution",
]
