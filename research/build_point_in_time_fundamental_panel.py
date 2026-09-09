"""构建沪深300官方权重口径的点时基本面月度面板。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
FINANCIALS_FILE = ROOT / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time.parquet"
CONSTITUENT_DAILY_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_raw.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_point_in_time_fundamental_monthly.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_point_in_time_fundamental_status.json"

CORE_VALUE_COLUMNS = [
    "revenue_cny",
    "net_profit_parent_cny",
    "equity_parent_cny",
    "total_shares",
]

OPTIONAL_VALUE_COLUMNS = ["bps"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_latest_vintages(financials: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    """选择信号日之前每只证券、每个报告期的最新已知版本。"""

    required = {"con_code", "report_period", "available_at", *CORE_VALUE_COLUMNS}
    if missing := required - set(financials.columns):
        raise ValueError(f"点时财务数据缺少字段：{sorted(missing)}")
    data = financials.copy()
    data["report_period"] = pd.to_datetime(data["report_period"], errors="coerce")
    data["available_at"] = pd.to_datetime(data["available_at"], errors="coerce")
    data = data.loc[data["available_at"].le(pd.Timestamp(asof_date))].copy()
    if data.empty:
        return data
    data = data.sort_values(["con_code", "report_period", "available_at"])
    return data.drop_duplicates(["con_code", "report_period"], keep="last").reset_index(drop=True)


def _latest_non_null(records: pd.DataFrame, column: str) -> float:
    """返回截至信号日最近报告期的非空值，避免一张表晚于另一张表披露时丢值。"""

    if column not in records:
        return np.nan
    valid = records.loc[records[column].notna(), ["report_period", column]]
    if valid.empty:
        return np.nan
    return float(valid.sort_values("report_period").iloc[-1][column])


def validated_book_value_per_share(equity: float, shares: float, reported_bps: float) -> float:
    """优先采用经资产负债表交叉校验的BPS，异常时回退到归母权益/总股本。"""

    calculated_bps = (
        equity / shares
        if pd.notna(equity) and pd.notna(shares) and equity > 0 and shares > 0
        else np.nan
    )
    if pd.isna(reported_bps) or reported_bps <= 0:
        return calculated_bps
    if pd.isna(calculated_bps) or calculated_bps <= 0:
        return reported_bps
    ratio = reported_bps / calculated_bps
    return reported_bps if 0.5 <= ratio <= 1.5 else calculated_bps


def _ttm_at_period(
    value_map: dict[pd.Timestamp, float],
    period: pd.Timestamp,
) -> float:
    """由累计报表构造指定报告期末TTM值。"""

    current = value_map.get(period, np.nan)
    if pd.isna(current):
        return np.nan
    if period.month == 12:
        return current
    previous_annual = pd.Timestamp(year=period.year - 1, month=12, day=31)
    previous_same = period - pd.DateOffset(years=1)
    annual_value = value_map.get(previous_annual, np.nan)
    previous_same_value = value_map.get(previous_same, np.nan)
    if pd.isna(annual_value) or pd.isna(previous_same_value):
        return np.nan
    return current + annual_value - previous_same_value


def derive_company_metrics(latest_vintages: pd.DataFrame) -> pd.DataFrame:
    """从某个信号日已知报表推导公司级TTM指标。"""

    rows: list[dict[str, Any]] = []
    for con_code, records in latest_vintages.groupby("con_code", sort=False):
        records = records.sort_values("report_period").reset_index(drop=True)
        maps = {
            column: {
                pd.Timestamp(period): float(value)
                for period, value in zip(records["report_period"], records[column], strict=True)
                if pd.notna(value)
            }
            for column in CORE_VALUE_COLUMNS
        }
        valid_periods = records.loc[
            records["net_profit_parent_cny"].notna(), "report_period"
        ].drop_duplicates().sort_values(ascending=False)
        selected_period: pd.Timestamp | None = None
        ttm_profit = np.nan
        for period in valid_periods:
            candidate = _ttm_at_period(maps["net_profit_parent_cny"], pd.Timestamp(period))
            if pd.notna(candidate):
                selected_period = pd.Timestamp(period)
                ttm_profit = candidate
                break
        if selected_period is None:
            continue
        ttm_revenue = _ttm_at_period(maps["revenue_cny"], selected_period)
        prior_period = selected_period - pd.DateOffset(years=1)
        prior_ttm_profit = _ttm_at_period(maps["net_profit_parent_cny"], prior_period)
        prior_ttm_revenue = _ttm_at_period(maps["revenue_cny"], prior_period)
        equity = _latest_non_null(records, "equity_parent_cny")
        shares = _latest_non_null(records, "total_shares")
        reported_bps = _latest_non_null(records, "bps")
        book_value_per_share = validated_book_value_per_share(equity, shares, reported_bps)
        profit_growth = (
            ttm_profit / prior_ttm_profit - 1.0
            if pd.notna(prior_ttm_profit) and prior_ttm_profit > 0
            else np.nan
        )
        revenue_growth = (
            ttm_revenue / prior_ttm_revenue - 1.0
            if pd.notna(ttm_revenue) and pd.notna(prior_ttm_revenue) and prior_ttm_revenue > 0
            else np.nan
        )
        rows.append(
            {
                "con_code": con_code,
                "latest_report_period": selected_period,
                "ttm_revenue_cny": ttm_revenue,
                "ttm_net_profit_parent_cny": ttm_profit,
                "equity_parent_cny": equity,
                "total_shares": shares,
                "reported_bps": reported_bps,
                "book_value_per_share": book_value_per_share,
                "ttm_eps": ttm_profit / shares if pd.notna(shares) and shares > 0 else np.nan,
                "ttm_roe": ttm_profit / equity if pd.notna(equity) and equity > 0 else np.nan,
                "ttm_profit_growth_yoy": profit_growth,
                "ttm_revenue_growth_yoy": revenue_growth,
            }
        )
    return pd.DataFrame(rows)


def weighted_average(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    """返回有效样本归一化加权均值与原始权重覆盖率。"""

    valid = values.notna() & weights.notna() & weights.gt(0)
    coverage = float(weights.loc[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage
    return float(np.average(values.loc[valid], weights=weights.loc[valid])), coverage


def aggregate_snapshot(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    index_close: float,
) -> dict[str, Any]:
    """聚合单个官方权重快照的指数基本面。"""

    panel = weights[["con_code", "weight"]].copy()
    panel["weight"] = pd.to_numeric(panel["weight"], errors="coerce") / 100.0
    panel = panel.merge(prices[["con_code", "raw_close"]], on="con_code", how="left", validate="one_to_one")
    panel = panel.merge(metrics, on="con_code", how="left", validate="one_to_one")
    panel["earnings_yield"] = panel["ttm_eps"] / panel["raw_close"]
    panel["book_yield"] = panel["book_value_per_share"] / panel["raw_close"]
    panel["total_market_cap_cny"] = panel["raw_close"] * panel["total_shares"]
    earnings_yield, earnings_coverage = weighted_average(panel["earnings_yield"], panel["weight"])
    positive_earnings_yield = panel["earnings_yield"].where(panel["earnings_yield"].gt(0))
    profitable_earnings_yield, profitable_earnings_coverage = weighted_average(
        positive_earnings_yield, panel["weight"]
    )
    book_yield, book_coverage = weighted_average(panel["book_yield"], panel["weight"])
    roe, roe_coverage = weighted_average(panel["ttm_roe"].clip(-0.5, 0.8), panel["weight"])
    profit_growth, profit_growth_coverage = weighted_average(
        panel["ttm_profit_growth_yoy"].clip(-1.0, 1.0), panel["weight"]
    )
    revenue_growth, revenue_growth_coverage = weighted_average(
        panel["ttm_revenue_growth_yoy"].clip(-1.0, 1.0), panel["weight"]
    )
    price_coverage = float(panel.loc[panel["raw_close"].notna(), "weight"].sum())
    profitable = (
        panel["ttm_net_profit_parent_cny"].gt(0)
        & panel["total_market_cap_cny"].notna()
        & panel["total_market_cap_cny"].gt(0)
    )
    total_market_cap_pe = (
        panel.loc[profitable, "total_market_cap_cny"].sum()
        / panel.loc[profitable, "ttm_net_profit_parent_cny"].sum()
        if profitable.any()
        else np.nan
    )
    profitable_with_book = profitable & panel["book_value_per_share"].gt(0) & panel["total_shares"].gt(0)
    total_market_cap_pb = (
        panel.loc[profitable_with_book, "total_market_cap_cny"].sum()
        / (
            panel.loc[profitable_with_book, "book_value_per_share"]
            * panel.loc[profitable_with_book, "total_shares"]
        ).sum()
        if profitable_with_book.any()
        else np.nan
    )
    return {
        "date": pd.Timestamp(snapshot_date),
        "constituent_count": int(panel["con_code"].nunique()),
        "weight_sum": float(panel["weight"].sum()),
        "price_weight_coverage": price_coverage,
        "earnings_weight_coverage": earnings_coverage,
        "profitable_earnings_weight_coverage": profitable_earnings_coverage,
        "book_weight_coverage": book_coverage,
        "roe_weight_coverage": roe_coverage,
        "profit_growth_weight_coverage": profit_growth_coverage,
        "revenue_growth_weight_coverage": revenue_growth_coverage,
        "weighted_earnings_yield": earnings_yield,
        "index_weighted_earnings_yield_all": earnings_yield,
        "index_weighted_earnings_yield_profitable_only": profitable_earnings_yield,
        "component_implied_pe_ttm": 1.0 / earnings_yield if pd.notna(earnings_yield) and earnings_yield > 0 else np.nan,
        "index_weighted_implied_pe_ttm_all": 1.0 / earnings_yield if pd.notna(earnings_yield) and earnings_yield > 0 else np.nan,
        "index_weighted_implied_pe_ttm_profitable_only": 1.0 / profitable_earnings_yield if pd.notna(profitable_earnings_yield) and profitable_earnings_yield > 0 else np.nan,
        "total_market_cap_pe_ttm_profitable_only": total_market_cap_pe,
        "weighted_book_yield": book_yield,
        "component_implied_pb": 1.0 / book_yield if pd.notna(book_yield) and book_yield > 0 else np.nan,
        "total_market_cap_pb_latest_profitable_only": total_market_cap_pb,
        "weighted_ttm_roe": roe,
        "weighted_ttm_profit_growth_yoy": profit_growth,
        "weighted_ttm_revenue_growth_yoy": revenue_growth,
        "component_implied_index_eps": index_close * earnings_yield if pd.notna(earnings_yield) else np.nan,
    }


def build_panel(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    valuation: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """构建研究区间内全部官方月度权重快照。"""

    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    weights = weights.loc[weights["trade_date"].between(start, end)].copy()
    financials = financials.copy()
    constituent_daily = constituent_daily.copy()
    constituent_daily["date"] = pd.to_datetime(constituent_daily["date"])
    index_daily = index_daily.copy()
    index_daily["date"] = pd.to_datetime(index_daily["date"])
    valuation = valuation.copy()
    valuation["date"] = pd.to_datetime(valuation["date"])
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weights.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        price_rows = constituent_daily.loc[
            constituent_daily["date"].eq(snapshot_date), ["con_code", "raw_close"]
        ].drop_duplicates("con_code", keep="last")
        index_row = index_daily.loc[index_daily["date"].eq(snapshot_date), "close"]
        if index_row.empty:
            raise ValueError(f"{snapshot_date.date()}缺少指数收盘价")
        snapshot_symbols = set(snapshot_weights["con_code"].astype(str))
        vintages = select_latest_vintages(
            financials.loc[financials["con_code"].astype(str).isin(snapshot_symbols)],
            snapshot_date,
        )
        metrics = derive_company_metrics(vintages)
        rows.append(
            aggregate_snapshot(
                snapshot_weights,
                price_rows,
                metrics,
                snapshot_date,
                float(index_row.iloc[-1]),
            )
        )
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    valuation_columns = valuation[["date", "pe_ttm", "pb"]].drop_duplicates("date", keep="last")
    result = result.merge(valuation_columns, on="date", how="left", validate="one_to_one")
    result["total_market_cap_vs_vendor_pe_relative_error"] = (
        result["total_market_cap_pe_ttm_profitable_only"] / result["pe_ttm"] - 1.0
    )
    result["total_market_cap_vs_vendor_pb_relative_error"] = (
        result["total_market_cap_pb_latest_profitable_only"] / result["pb"] - 1.0
    )
    result["component_vs_vendor_pe_relative_error"] = result[
        "total_market_cap_vs_vendor_pe_relative_error"
    ]
    result["component_vs_vendor_pb_relative_error"] = result[
        "total_market_cap_vs_vendor_pb_relative_error"
    ]
    return result


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    start = pd.Timestamp(settings["project"]["start_date"])
    end = pd.Timestamp(settings["project"]["end_date"])
    panel = build_panel(
        pd.read_parquet(WEIGHTS_FILE),
        pd.read_parquet(FINANCIALS_FILE),
        pd.read_parquet(CONSTITUENT_DAILY_FILE),
        pd.read_parquet(INDEX_FILE),
        pd.read_parquet(VALUATION_FILE),
        start,
        end,
    )
    coverage_columns = [
        "price_weight_coverage",
        "earnings_weight_coverage",
        "book_weight_coverage",
        "roe_weight_coverage",
    ]
    if panel[coverage_columns].min().min() < 0.90:
        raise ValueError(
            f"核心基本面最小权重覆盖率低于90%：{panel[coverage_columns].min().to_dict()}"
        )
    if panel["constituent_count"].min() != 300 or panel["constituent_count"].max() != 300:
        raise ValueError("官方权重快照并非每期恰好300只")
    pe_pairs = panel[["total_market_cap_pe_ttm_profitable_only", "pe_ttm"]].dropna()
    pe_correlation = float(pe_pairs.corr().iloc[0, 1]) if len(pe_pairs) >= 3 else np.nan
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_FILE, index=False)
    timezone = ZoneInfo(settings["project"]["timezone"])
    report = {
        "status": "PASS",
        "checked_at": datetime.now(timezone).isoformat(),
        "snapshot_count": int(len(panel)),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "minimum_coverages": {column: float(panel[column].min()) for column in coverage_columns},
        "component_vendor_pe_correlation": pe_correlation,
        "component_vendor_pe_median_absolute_relative_error": float(
            panel["total_market_cap_vs_vendor_pe_relative_error"].abs().median()
        ),
        "component_vendor_pb_median_absolute_relative_error": float(
            panel["total_market_cap_vs_vendor_pb_relative_error"].abs().median()
        ),
        "valuation_cross_check_method": "与乐咕同口径：盈利公司总市值除以TTM归母净利润；PB使用经权益/股本校验的BPS。",
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
        "inputs": {
            WEIGHTS_FILE.relative_to(ROOT).as_posix(): _sha256(WEIGHTS_FILE),
            FINANCIALS_FILE.relative_to(ROOT).as_posix(): _sha256(FINANCIALS_FILE),
            CONSTITUENT_DAILY_FILE.relative_to(ROOT).as_posix(): _sha256(CONSTITUENT_DAILY_FILE),
        },
        "governance": "只使用快照日及以前已披露财务版本；当月官方权重不回填至快照日前。",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
