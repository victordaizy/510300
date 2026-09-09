"""构造点时中信一级板块估值—盈利面板，不读取未来收益。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.build_point_in_time_fundamental_panel import (
    derive_company_metrics,
    select_latest_vintages,
)
from research.index_driver_attribution_v1_2 import (
    truncate_superseded_l1_intervals,
)
from research.index_driver_attribution_v1_3 import (
    restrict_industry_intervals_to_research_window,
)


@dataclass(frozen=True)
class SectorPanelRules:
    """点时板块面板的冻结数据门槛。"""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    minimum_snapshot_count: int
    forecast_horizon_days: int
    minimum_nonoverlapping_blocks: int
    member_count_minimum: int
    member_count_maximum: int
    weight_sum_minimum: float
    weight_sum_maximum: float
    minimum_price_coverage: float
    minimum_industry_coverage: float
    minimum_earnings_coverage: float
    minimum_book_coverage: float
    minimum_roe_coverage: float
    minimum_sector_component_count: int
    minimum_sector_feature_coverage: float
    minimum_eligible_sector_weight_coverage: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "SectorPanelRules":
        protocol = contract["protocol"]
        gates = contract["quality_gates"]
        weight_range = gates["weight_sum_range"]
        return cls(
            start_date=pd.Timestamp(protocol["data_start"]),
            end_date=pd.Timestamp(protocol["data_cutoff"]),
            minimum_snapshot_count=int(gates["minimum_snapshot_count"]),
            forecast_horizon_days=int(
                gates["forecast_horizon_trading_days_for_independence_audit"]
            ),
            minimum_nonoverlapping_blocks=int(
                gates["minimum_nonoverlapping_time_blocks"]
            ),
            member_count_minimum=int(gates["member_count_minimum"]),
            member_count_maximum=int(gates["member_count_maximum"]),
            weight_sum_minimum=float(weight_range[0]),
            weight_sum_maximum=float(weight_range[1]),
            minimum_price_coverage=float(gates["minimum_price_weight_coverage"]),
            minimum_industry_coverage=float(
                gates["minimum_industry_weight_coverage"]
            ),
            minimum_earnings_coverage=float(
                gates["minimum_earnings_weight_coverage"]
            ),
            minimum_book_coverage=float(gates["minimum_book_weight_coverage"]),
            minimum_roe_coverage=float(gates["minimum_roe_weight_coverage"]),
            minimum_sector_component_count=int(
                gates["minimum_sector_component_count"]
            ),
            minimum_sector_feature_coverage=float(
                gates["minimum_sector_core_feature_coverage"]
            ),
            minimum_eligible_sector_weight_coverage=float(
                gates["minimum_eligible_sector_weight_coverage"]
            ),
        )


@dataclass(frozen=True)
class SectorPanelResult:
    """板块特征、快照审计和窗口级可行性。"""

    sector_panel: pd.DataFrame
    snapshot_audit: pd.DataFrame
    feasibility: dict[str, Any]


def _require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name}缺少字段：{missing}")


def _weighted_value(
    group: pd.DataFrame, value_column: str
) -> tuple[float, float]:
    valid = group[value_column].notna() & group["snapshot_weight"].gt(0)
    coverage = float(group.loc[valid, "snapshot_weight"].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage
    value = float(
        np.average(
            group.loc[valid, value_column].astype(float),
            weights=group.loc[valid, "snapshot_weight"].astype(float),
        )
    )
    return value, coverage


def _active_industry(
    intervals: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    active = intervals.loc[
        intervals["in_date"].le(date)
        & (intervals["out_date"].isna() | intervals["out_date"].gt(date)),
        ["con_code", "industry_l1", "source", "in_date", "out_date"],
    ].copy()
    if active["con_code"].duplicated().any():
        duplicated = active.loc[
            active["con_code"].duplicated(False), "con_code"
        ].astype(str)
        raise ValueError(
            f"{date.date()}存在重复有效行业：{sorted(duplicated.unique())[:10]}"
        )
    return active


def _prepare_inputs(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: SectorPanelRules,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        weights, {"con_code", "trade_date", "weight", "source"}, "官方权重"
    )
    _require_columns(
        financials,
        {
            "con_code",
            "report_period",
            "available_at",
            "revenue_cny",
            "net_profit_parent_cny",
            "equity_parent_cny",
            "total_shares",
            "bps",
        },
        "点时财务事件",
    )
    _require_columns(
        constituent_daily, {"date", "con_code", "raw_close"}, "成分股日线"
    )
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
    point_weights = weights.copy()
    point_weights["trade_date"] = pd.to_datetime(
        point_weights["trade_date"], errors="coerce"
    )
    point_weights["weight"] = pd.to_numeric(
        point_weights["weight"], errors="coerce"
    )
    if point_weights[["trade_date", "con_code", "weight"]].isna().any().any():
        raise ValueError("官方权重存在空日期、证券或权重")
    if point_weights[["trade_date", "con_code"]].duplicated().any():
        raise ValueError("官方权重存在重复快照证券")
    point_weights = point_weights.loc[
        point_weights["trade_date"].between(rules.start_date, rules.end_date)
    ].copy()
    point_weights["snapshot_weight"] = point_weights["weight"] / 100.0

    events = financials.copy()
    events["report_period"] = pd.to_datetime(events["report_period"], errors="coerce")
    events["available_at"] = pd.to_datetime(events["available_at"], errors="coerce")
    if events[["con_code", "report_period", "available_at"]].isna().any().any():
        raise ValueError("点时财务事件存在空证券、报告期或可得日")

    prices = constituent_daily[["date", "con_code", "raw_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["raw_close"] = pd.to_numeric(prices["raw_close"], errors="coerce")
    if prices[["date", "con_code"]].isna().any().any():
        raise ValueError("成分股日线存在空日期或证券")
    if prices[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股日线存在重复证券日期")
    invalid_price = prices["raw_close"].notna() & prices["raw_close"].le(0)
    if invalid_price.any():
        raise ValueError("成分股日线存在非正收盘价")

    relevant = restrict_industry_intervals_to_research_window(
        industry_intervals, rules.start_date, rules.end_date
    )
    intervals = truncate_superseded_l1_intervals(relevant)
    intervals["in_date"] = pd.to_datetime(intervals["in_date"])
    intervals["out_date"] = pd.to_datetime(intervals["out_date"])
    return point_weights, events, prices, intervals


def _sector_rows(
    cross: pd.DataFrame, date: pd.Timestamp, rules: SectorPanelRules
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mapped = cross.loc[cross["industry_l1"].notna()].copy()
    for industry_name, group in mapped.groupby("industry_l1", sort=True):
        sector_weight = float(group["snapshot_weight"].sum())
        earnings_yield, earnings_weight = _weighted_value(group, "earnings_yield")
        book_yield, book_weight = _weighted_value(group, "book_yield")
        roe, roe_weight = _weighted_value(group, "ttm_roe")
        profit_growth, profit_growth_weight = _weighted_value(
            group, "ttm_profit_growth_yoy"
        )
        revenue_growth, revenue_growth_weight = _weighted_value(
            group, "ttm_revenue_growth_yoy"
        )
        price_weight = float(
            group.loc[group["raw_close"].notna(), "snapshot_weight"].sum()
        )
        coverage_ratios = {
            "price_coverage_ratio": price_weight / sector_weight,
            "earnings_coverage_ratio": earnings_weight / sector_weight,
            "book_coverage_ratio": book_weight / sector_weight,
            "roe_coverage_ratio": roe_weight / sector_weight,
            "profit_growth_coverage_ratio": profit_growth_weight / sector_weight,
            "revenue_growth_coverage_ratio": revenue_growth_weight / sector_weight,
        }
        profitable = (
            group["ttm_net_profit_parent_cny"].gt(0)
            & group["total_market_cap_cny"].gt(0)
        )
        sector_pe = (
            float(
                group.loc[profitable, "total_market_cap_cny"].sum()
                / group.loc[profitable, "ttm_net_profit_parent_cny"].sum()
            )
            if profitable.any()
            else np.nan
        )
        core_feature_coverage = min(
            coverage_ratios["earnings_coverage_ratio"],
            coverage_ratios["book_coverage_ratio"],
            coverage_ratios["roe_coverage_ratio"],
        )
        component_count = int(group["con_code"].nunique())
        eligible = (
            component_count >= rules.minimum_sector_component_count
            and core_feature_coverage >= rules.minimum_sector_feature_coverage
        )
        rows.append(
            {
                "date": date,
                "industry_l1": str(industry_name),
                "sector_weight": sector_weight,
                "component_count": component_count,
                **coverage_ratios,
                "core_feature_coverage_ratio": core_feature_coverage,
                "weighted_earnings_yield": earnings_yield,
                "weighted_book_yield": book_yield,
                "weighted_ttm_roe": roe,
                "weighted_ttm_profit_growth_yoy": profit_growth,
                "weighted_ttm_revenue_growth_yoy": revenue_growth,
                "profitable_market_cap_pe_ttm": sector_pe,
                "eligible_for_target_freeze": eligible,
                "data_output": "SECTOR_DATA_READY" if eligible else "NO_VIEW",
                "failure_category": (
                    "PASS"
                    if eligible
                    else "SECTOR_COMPONENT_OR_FEATURE_COVERAGE_BELOW_GATE"
                ),
            }
        )
    return pd.DataFrame(rows)


def _nonoverlapping_block_count(
    dates: list[pd.Timestamp], trading_dates: pd.DatetimeIndex, horizon: int
) -> int:
    position = pd.Series(np.arange(len(trading_dates)), index=trading_dates)
    selected = 0
    last_position = -horizon
    for date in dates:
        current = position.get(pd.Timestamp(date))
        if current is None:
            continue
        current_position = int(current)
        if current_position - last_position >= horizon:
            selected += 1
            last_position = current_position
    return selected


def build_sector_point_in_time_panel(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: SectorPanelRules,
) -> SectorPanelResult:
    """构造月度点时板块面板并给出不读取未来收益的数据可行性。"""

    point_weights, events, prices, intervals = _prepare_inputs(
        weights, financials, constituent_daily, industry_intervals, rules
    )
    price_by_date = {
        pd.Timestamp(date): frame.copy()
        for date, frame in prices.groupby("date", sort=False)
    }
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    sector_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot in point_weights.groupby("trade_date", sort=True):
        date = pd.Timestamp(snapshot_date)
        snapshot = snapshot[["con_code", "snapshot_weight", "source"]].copy()
        price = price_by_date.get(
            date, pd.DataFrame(columns=["con_code", "raw_close"])
        )[["con_code", "raw_close"]]
        active_industry = _active_industry(intervals, date)
        symbols = set(snapshot["con_code"].astype(str))
        vintages = select_latest_vintages(
            events.loc[events["con_code"].astype(str).isin(symbols)], date
        )
        metrics = derive_company_metrics(vintages)
        cross = snapshot.merge(price, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(
            active_industry[["con_code", "industry_l1"]],
            on="con_code",
            how="left",
            validate="one_to_one",
        )
        cross = cross.merge(metrics, on="con_code", how="left", validate="one_to_one")
        cross["earnings_yield"] = cross["ttm_eps"] / cross["raw_close"]
        cross["book_yield"] = cross["book_value_per_share"] / cross["raw_close"]
        cross["total_market_cap_cny"] = cross["raw_close"] * cross["total_shares"]
        sectors = _sector_rows(cross, date, rules)
        sector_frames.append(sectors)

        member_count = int(cross["con_code"].nunique())
        weight_sum = float(cross["snapshot_weight"].sum())
        coverage = {
            "price_weight_coverage": float(
                cross.loc[cross["raw_close"].notna(), "snapshot_weight"].sum()
            ),
            "industry_weight_coverage": float(
                cross.loc[cross["industry_l1"].notna(), "snapshot_weight"].sum()
            ),
            "earnings_weight_coverage": float(
                cross.loc[cross["earnings_yield"].notna(), "snapshot_weight"].sum()
            ),
            "book_weight_coverage": float(
                cross.loc[cross["book_yield"].notna(), "snapshot_weight"].sum()
            ),
            "roe_weight_coverage": float(
                cross.loc[cross["ttm_roe"].notna(), "snapshot_weight"].sum()
            ),
            "eligible_sector_weight_coverage": float(
                sectors.loc[
                    sectors["eligible_for_target_freeze"], "sector_weight"
                ].sum()
            ),
        }
        failures: list[str] = []
        if not rules.member_count_minimum <= member_count <= rules.member_count_maximum:
            failures.append("MEMBER_COUNT_OUT_OF_RANGE")
        if not rules.weight_sum_minimum <= weight_sum <= rules.weight_sum_maximum:
            failures.append("WEIGHT_SUM_OUT_OF_RANGE")
        gate_mapping = {
            "price_weight_coverage": rules.minimum_price_coverage,
            "industry_weight_coverage": rules.minimum_industry_coverage,
            "earnings_weight_coverage": rules.minimum_earnings_coverage,
            "book_weight_coverage": rules.minimum_book_coverage,
            "roe_weight_coverage": rules.minimum_roe_coverage,
            "eligible_sector_weight_coverage": rules.minimum_eligible_sector_weight_coverage,
        }
        for column, minimum in gate_mapping.items():
            if coverage[column] < minimum:
                failures.append(f"{column.upper()}_BELOW_GATE")
        audit_rows.append(
            {
                "date": date,
                "member_count": member_count,
                "weight_sum": weight_sum,
                "sector_count": int(len(sectors)),
                "eligible_sector_count": int(
                    sectors["eligible_for_target_freeze"].sum()
                ),
                **coverage,
                "output": "DATA_READY" if not failures else "NO_VIEW",
                "failure_category": "PASS" if not failures else "|".join(failures),
            }
        )

    panel = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()
    audit = pd.DataFrame(audit_rows).sort_values("date").reset_index(drop=True)
    valid_dates = audit.loc[audit["output"].eq("DATA_READY"), "date"].tolist()
    nonoverlapping_blocks = _nonoverlapping_block_count(
        valid_dates, trading_dates, rules.forecast_horizon_days
    )
    window_failures: list[str] = []
    if len(valid_dates) < rules.minimum_snapshot_count:
        window_failures.append("VALID_SNAPSHOT_COUNT_BELOW_GATE")
    if nonoverlapping_blocks < rules.minimum_nonoverlapping_blocks:
        window_failures.append("NONOVERLAPPING_TIME_BLOCKS_BELOW_GATE")
    feasibility = {
        "status": "READY_FOR_TARGET_FREEZE" if not window_failures else "NO_VIEW",
        "failure_category": "PASS" if not window_failures else "|".join(window_failures),
        "snapshot_count": int(len(audit)),
        "valid_snapshot_count": int(len(valid_dates)),
        "no_view_snapshot_count": int(audit["output"].eq("NO_VIEW").sum()),
        "first_snapshot_date": str(audit["date"].min().date()) if not audit.empty else None,
        "last_snapshot_date": str(audit["date"].max().date()) if not audit.empty else None,
        "nonoverlapping_60d_time_blocks": int(nonoverlapping_blocks),
        "future_return_read": False,
        "predictive_model_fitted": False,
    }
    return SectorPanelResult(panel, audit, feasibility)


__all__ = [
    "SectorPanelResult",
    "SectorPanelRules",
    "build_sector_point_in_time_panel",
]
