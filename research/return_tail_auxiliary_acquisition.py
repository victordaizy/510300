"""成分股、行业分类与分析师报告数据的无收益标签标准化。"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_stock_history(
    daily: pd.DataFrame,
    adjustment: pd.DataFrame,
    symbol: str,
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """合并个股未复权日线和复权因子，生成可计算总收益的价格序列。"""

    daily_required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    factor_required = {"ts_code", "trade_date", "adj_factor"}
    missing_daily = sorted(daily_required.difference(daily.columns))
    missing_factor = sorted(factor_required.difference(adjustment.columns))
    if missing_daily or missing_factor:
        raise ValueError(f"个股行情字段缺失：daily={missing_daily}，adj_factor={missing_factor}")
    prices = daily.loc[daily["ts_code"].astype(str).eq(symbol)].copy()
    factors = adjustment.loc[adjustment["ts_code"].astype(str).eq(symbol)].copy()
    if prices.empty or factors.empty:
        raise ValueError(f"{symbol}缺少行情或复权因子")
    for frame in (prices, factors):
        frame["date"] = pd.to_datetime(
            frame["trade_date"], format="%Y%m%d", errors="coerce"
        ).dt.normalize()
    for column in ("open", "high", "low", "close", "vol", "amount"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
    data = prices.merge(
        factors[["date", "adj_factor"]], on="date", how="left", validate="one_to_one"
    )
    if data["adj_factor"].isna().any():
        raise ValueError(f"{symbol}复权因子不能覆盖全部成交日")
    result = pd.DataFrame(
        {
            "date": data["date"],
            "con_code": symbol,
            "raw_open": data["open"],
            "raw_high": data["high"],
            "raw_low": data["low"],
            "raw_close": data["close"],
            "total_return_open": data["open"] * data["adj_factor"],
            "total_return_high": data["high"] * data["adj_factor"],
            "total_return_low": data["low"] * data["adj_factor"],
            "total_return_close": data["close"] * data["adj_factor"],
            "volume": data["vol"] * 100.0,
            "amount": data["amount"] * 1000.0,
            "adj_factor": data["adj_factor"],
            "price_source": "tushare_proxy.daily",
            "adjustment_source": "tushare_proxy.adj_factor",
            "retrieved_at": str(pd.Timestamp(retrieved_at)),
        }
    )
    if result[["date", "raw_close", "total_return_close"]].isna().any().any():
        raise ValueError(f"{symbol}标准化价格存在关键空值")
    if result[["raw_open", "raw_high", "raw_low", "raw_close"]].le(0).any().any():
        raise ValueError(f"{symbol}标准化价格存在非正数")
    return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def active_members(
    intervals: pd.DataFrame, dates: Iterable[pd.Timestamp | str]
) -> pd.DataFrame:
    """按左闭右开成员区间展开点时成员日历。"""

    required = {"symbol", "opt_in", "opt_out"}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"成员区间缺少字段：{missing}")
    source = intervals.copy()
    source["opt_in"] = pd.to_datetime(source["opt_in"], errors="coerce").dt.normalize()
    source["opt_out"] = pd.to_datetime(source["opt_out"], errors="coerce").dt.normalize()
    normalized_dates = sorted({pd.Timestamp(value).normalize() for value in dates})
    pieces: list[pd.DataFrame] = []
    for date in normalized_dates:
        active = source.loc[
            source["opt_in"].le(date)
            & (source["opt_out"].isna() | source["opt_out"].gt(date)),
            ["symbol"],
        ].drop_duplicates()
        active = active.rename(columns={"symbol": "con_code"})
        active["date"] = date
        pieces.append(active)
    result = pd.concat(pieces, ignore_index=True)
    counts = result.groupby("date")["con_code"].nunique()
    if counts.min() != 300 or counts.max() != 300:
        raise ValueError(f"点时成员数量异常：{int(counts.min())}至{int(counts.max())}")
    return result[["date", "con_code"]].sort_values(["date", "con_code"]).reset_index(drop=True)


def build_constituent_panel(
    histories: pd.DataFrame,
    intervals: pd.DataFrame,
    calendar_dates: Iterable[pd.Timestamp | str],
    output_dates: Iterable[pd.Timestamp | str],
) -> pd.DataFrame:
    """为点时成员生成日线；停牌日只以前一总收益价格前向填充。"""

    calendar = pd.DataFrame(
        {"date": sorted({pd.Timestamp(value).normalize() for value in calendar_dates})}
    )
    output_date_set = {pd.Timestamp(value).normalize() for value in output_dates}
    members = active_members(intervals, output_date_set)
    member_lookup = {
        symbol: set(group["date"])
        for symbol, group in members.groupby("con_code", sort=False)
    }
    price_columns = [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
        "adj_factor",
    ]
    pieces: list[pd.DataFrame] = []
    for symbol, active_dates in member_lookup.items():
        raw = histories.loc[histories["con_code"].eq(symbol)].copy()
        if raw.empty:
            raise ValueError(f"点时成员{symbol}没有历史行情")
        panel = calendar.merge(raw, on="date", how="left", validate="one_to_one")
        original_missing = panel["total_return_close"].isna()
        panel[price_columns] = panel[price_columns].ffill()
        panel = panel.loc[panel["date"].isin(active_dates)].copy()
        if panel["total_return_close"].isna().any():
            first_missing = panel.loc[panel["total_return_close"].isna(), "date"].min()
            raise ValueError(f"{symbol}在{first_missing.date()}之前没有可前向填充的价格")
        missing_after_filter = original_missing.loc[panel.index]
        panel["con_code"] = symbol
        panel["is_index_member"] = True
        panel["is_suspended"] = missing_after_filter.to_numpy()
        panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce").fillna(0.0)
        panel["amount"] = pd.to_numeric(panel["amount"], errors="coerce").fillna(0.0)
        panel["price_source"] = panel["price_source"].fillna(
            "tushare_proxy.daily_forward_fill_suspension"
        )
        panel["adjustment_source"] = panel["adjustment_source"].fillna(
            "tushare_proxy.adj_factor_forward_fill_suspension"
        )
        panel["outstanding_share"] = np.nan
        panel["total_market_cap_cny"] = np.nan
        panel["market_cap_asof_date"] = pd.NaT
        pieces.append(panel)
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "con_code"])
    if result[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股面板存在重复证券日期")
    counts = result.groupby("date")["con_code"].nunique()
    if counts.min() != 300 or counts.max() != 300:
        raise ValueError(f"成分股面板每日成员数异常：{counts.min()}至{counts.max()}")
    return result.reset_index(drop=True)


def normalize_industry_intervals(
    raw: pd.DataFrame,
    universe: set[str],
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """标准化申万行业成员区间，保留进入和退出日期。"""

    required = {
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "in_date",
        "out_date",
        "is_new",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"申万行业区间缺少字段：{missing}")
    data = raw.loc[raw["ts_code"].astype(str).isin(universe)].copy()
    data["in_date"] = pd.to_datetime(data["in_date"], format="%Y%m%d", errors="coerce").dt.normalize()
    data["out_date"] = pd.to_datetime(data["out_date"], format="%Y%m%d", errors="coerce").dt.normalize()
    data = data.dropna(subset=["ts_code", "l1_code", "l1_name", "in_date"])
    result = pd.DataFrame(
        {
            "con_code": data["ts_code"].astype(str),
            "industry_l1": data["l1_name"].astype(str),
            "industry_l1_code": data["l1_code"].astype(str),
            "industry_l2": data["l2_name"].astype(str),
            "industry_l2_code": data["l2_code"].astype(str),
            "industry_l3": data["l3_name"].astype(str),
            "industry_l3_code": data["l3_code"].astype(str),
            "in_date": data["in_date"],
            "out_date": data["out_date"],
            "is_current": data["is_new"].astype(str).str.upper().eq("Y"),
            "classification_usage": "POINT_IN_TIME_INTERVAL",
            "source": "tushare_proxy.index_member_all",
            "retrieved_at": str(pd.Timestamp(retrieved_at)),
        }
    )
    result = result.drop_duplicates(
        ["con_code", "industry_l1_code", "in_date", "out_date"], keep="last"
    )
    return result.sort_values(["con_code", "in_date", "industry_l1_code"]).reset_index(drop=True)


def industry_coverage(industry: pd.DataFrame, members: pd.DataFrame) -> dict[str, float | int]:
    """逐证券日期验证行业区间覆盖与重叠。"""

    covered = 0
    overlapping = 0
    industry_groups = {
        symbol: group for symbol, group in industry.groupby("con_code", sort=False)
    }
    for symbol, dates in members.groupby("con_code", sort=False):
        intervals = industry_groups.get(symbol, industry.iloc[0:0])
        member_dates = pd.to_datetime(dates["date"]).to_numpy(dtype="datetime64[ns]")
        active_counts = np.zeros(len(member_dates), dtype=np.int16)
        for interval in intervals.itertuples(index=False):
            interval_start = np.datetime64(pd.Timestamp(interval.in_date), "ns")
            if pd.isna(interval.out_date):
                active = member_dates >= interval_start
            else:
                interval_end = np.datetime64(pd.Timestamp(interval.out_date), "ns")
                active = (member_dates >= interval_start) & (member_dates < interval_end)
            active_counts += active.astype(np.int16)
        covered += int((active_counts >= 1).sum())
        overlapping += int((active_counts > 1).sum())
    total = int(len(members))
    return {
        "member_day_count": total,
        "covered_member_day_count": covered,
        "coverage_ratio": float(covered / max(total, 1)),
        "overlapping_member_day_count": overlapping,
        "overlap_ratio": float(overlapping / max(total, 1)),
    }


def normalize_analyst_reports(
    raw: pd.DataFrame,
    universe: set[str],
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """保留历史研报记录；明确不把它包装成供应商一致预期快照。"""

    required = {
        "ts_code",
        "report_date",
        "report_title",
        "org_name",
        "quarter",
        "eps",
        "create_time",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"分析师研报缺少字段：{missing}")
    data = raw.loc[raw["ts_code"].astype(str).isin(universe)].copy()
    data["report_date"] = pd.to_datetime(
        data["report_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    data["create_time"] = pd.to_datetime(data["create_time"], errors="coerce")
    data["eps"] = pd.to_numeric(data["eps"], errors="coerce")
    data["forecast_fiscal_year"] = pd.to_numeric(
        data["quarter"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce"
    ).astype("Int64")
    stable_columns = [
        "ts_code",
        "report_date",
        "report_title",
        "org_name",
        "author_name",
        "quarter",
        "eps",
        "create_time",
    ]
    for column in stable_columns:
        if column not in data:
            data[column] = ""
    def fingerprint(row: pd.Series) -> str:
        text = "|".join(str(row[column]) for column in stable_columns)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    data["source_record_id"] = data.apply(fingerprint, axis=1)
    data["source"] = "tushare_proxy.report_rc_historical_records"
    data["retrieved_at"] = str(pd.Timestamp(retrieved_at))
    data["consensus_vintage_qualified"] = False
    data["qualification_reason"] = (
        "单篇研报记录不是不可变的供应商历史一致预期快照，禁止替代E1输入"
    )
    data = data.drop_duplicates("source_record_id", keep="last")
    return data.sort_values(["report_date", "ts_code", "source_record_id"]).reset_index(drop=True)
