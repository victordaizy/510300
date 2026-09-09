"""按原始章程构造 M/F/T，并完成 G2 机制链检验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


class G2MechanismError(RuntimeError):
    """G2 输入、口径或不变量不满足。"""


G2_PASS_STATUS = "PASS_G2_MECHANISM_CHAIN_CONTINUE_TO_G3"
G2_FAIL_STATUS = "REJECTED_FROZEN_G2_MECHANISM_CHAIN_FAILED_NO_FEATURE_RESCUE"


@dataclass(frozen=True)
class Subperiod:
    period_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise G2MechanismError(f"{label}缺少字段：{missing}")


def normalize_dates(values: pd.Series) -> pd.Series:
    """把不同 Parquet 时间精度统一为无时区纳秒交易日。"""

    return (
        pd.to_datetime(values, errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )


def normalize_utc_timestamps(values: pd.Series) -> pd.Series:
    """把不同 Parquet 时间精度统一为 UTC 纳秒时间戳。"""

    return pd.to_datetime(values, errors="coerce", utc=True).astype(
        "datetime64[ns, UTC]"
    )


def causal_midrank(values: pd.Series) -> pd.Series:
    """当前值相对严格更早有效观察的经验中秩；当前值不进入参照集。"""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, value in numeric.items():
        if not np.isfinite(value):
            continue
        if history:
            prior = np.asarray(history, dtype=float)
            result.loc[index] = float(
                (np.count_nonzero(prior < value) + 0.5 * np.count_nonzero(prior == value))
                / len(prior)
            )
        history.append(float(value))
    return result


def required_median(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """只有所有固定通道均有效时才计算中位数。"""

    _require_columns(frame, columns, "复合分数")
    valid = frame[list(columns)].notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid] = frame.loc[valid, list(columns)].median(axis=1)
    return result


def parse_subperiods(raw: Sequence[Mapping[str, Any]]) -> list[Subperiod]:
    if len(raw) != 4:
        raise G2MechanismError("G2 必须恰好使用四个固定子期")
    result: list[Subperiod] = []
    for item in raw:
        period_id = str(item.get("id", "")).strip()
        start = pd.to_datetime(item.get("start"), errors="coerce")
        end = pd.to_datetime(item.get("end"), errors="coerce")
        if not period_id or pd.isna(start) or pd.isna(end) or start > end:
            raise G2MechanismError(f"非法子期：{item}")
        result.append(
            Subperiod(
                period_id=period_id,
                start=pd.Timestamp(start).normalize(),
                end=pd.Timestamp(end).normalize(),
            )
        )
    for previous, current in zip(result, result[1:]):
        if current.start <= previous.end:
            raise G2MechanismError("四个固定子期必须递增且不重叠")
    return result


def prepare_membership(
    frame: pd.DataFrame,
    *,
    start: str,
    cutoff: str,
) -> pd.DataFrame:
    _require_columns(frame, ["membership_date", "index_code", "symbol"], "点时成分")
    work = frame[["membership_date", "index_code", "symbol"]].copy()
    work["date"] = normalize_dates(work.pop("membership_date"))
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    work["index_code"] = work["index_code"].astype("string").str.strip()
    work = work.loc[
        work["date"].between(pd.Timestamp(start), pd.Timestamp(cutoff))
    ].copy()
    if work.empty or work[["date", "symbol"]].isna().any().any():
        raise G2MechanismError("研究区间内点时成分为空或字段非法")
    if not work["index_code"].eq("000300").all():
        raise G2MechanismError("点时成分包含非 000300 指数")
    if work.duplicated(["date", "symbol"]).any():
        raise G2MechanismError("点时成分存在 date-symbol 重复")
    counts = work.groupby("date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise G2MechanismError(
            f"点时成分每日不是 300 只：min={counts.min()}，max={counts.max()}"
        )
    return work[["date", "symbol"]].sort_values(["date", "symbol"]).reset_index(drop=True)


def assign_effective_industry(
    membership: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """按官方历史生效区间赋予当日一级行业，不使用当前分类回填。"""

    _require_columns(membership, ["date", "symbol"], "点时成分")
    _require_columns(
        history,
        ["symbol", "effective_date", "out_date", "industry_l1_code"],
        "申万行业历史",
    )
    member_work = membership.copy()
    member_work["date"] = normalize_dates(member_work["date"])
    member_work["symbol"] = (
        member_work["symbol"].astype("string").str.strip().str.upper()
    )
    events = history[
        ["symbol", "effective_date", "out_date", "industry_l1_code"]
    ].copy()
    events["symbol"] = events["symbol"].astype("string").str.strip().str.upper()
    events["effective_date"] = normalize_dates(events["effective_date"])
    events["out_date"] = normalize_dates(events["out_date"])
    events["industry_l1_code"] = (
        events["industry_l1_code"].astype("string").str.strip()
    )
    events = events.dropna(subset=["symbol", "effective_date", "industry_l1_code"])
    conflicts = (
        events.groupby(["symbol", "effective_date"])["industry_l1_code"]
        .nunique()
        .gt(1)
    )
    if conflicts.any():
        raise G2MechanismError("同一证券生效日出现多个一级行业")
    events = events.drop_duplicates(
        ["symbol", "effective_date"], keep="last"
    ).sort_values(["symbol", "effective_date"])

    pieces: list[pd.DataFrame] = []
    event_groups = {symbol: group for symbol, group in events.groupby("symbol")}
    for symbol, member_rows in member_work.groupby("symbol", sort=False):
        member_rows = member_rows.sort_values("date").copy()
        symbol_events = event_groups.get(symbol)
        if symbol_events is None or symbol_events.empty:
            member_rows["industry_l1_code"] = pd.NA
            member_rows["industry_effective_date"] = pd.NaT
            member_rows["industry_out_date"] = pd.NaT
            pieces.append(member_rows)
            continue
        matched = pd.merge_asof(
            member_rows,
            symbol_events.drop(columns="symbol").rename(
                columns={
                    "effective_date": "industry_effective_date",
                    "out_date": "industry_out_date",
                }
            ).sort_values("industry_effective_date"),
            left_on="date",
            right_on="industry_effective_date",
            direction="backward",
            allow_exact_matches=True,
        )
        valid = matched["industry_effective_date"].notna() & (
            matched["industry_out_date"].isna()
            | matched["date"].lt(matched["industry_out_date"])
        )
        matched.loc[~valid, "industry_l1_code"] = pd.NA
        pieces.append(matched)
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_constituent_return_index(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    *,
    market_dates: pd.DatetimeIndex,
    cutover_date: str,
) -> pd.DataFrame:
    """用同源重叠期切换构造每只证券的日总收益指数。"""

    _require_columns(
        historical,
        ["date", "con_code", "pre_close", "raw_close"],
        "历史成分行情",
    )
    _require_columns(
        current,
        ["date", "con_code", "total_return_close"],
        "现有成分行情",
    )
    cutoff = pd.Timestamp(cutover_date)
    old = historical[["date", "con_code", "pre_close", "raw_close"]].copy()
    old["date"] = normalize_dates(old["date"])
    old["symbol"] = old.pop("con_code").astype("string").str.strip().str.upper()
    old["daily_total_return"] = (
        pd.to_numeric(old["raw_close"], errors="coerce")
        / pd.to_numeric(old["pre_close"], errors="coerce")
        - 1.0
    )
    old = old.loc[old["date"].lt(cutoff), ["date", "symbol", "daily_total_return"]]

    new = current[["date", "con_code", "total_return_close"]].copy()
    new["date"] = normalize_dates(new["date"])
    new["symbol"] = new.pop("con_code").astype("string").str.strip().str.upper()
    new["total_return_close"] = pd.to_numeric(
        new["total_return_close"], errors="coerce"
    )
    new = new.sort_values(["symbol", "date"], kind="stable")
    new["daily_total_return"] = new.groupby("symbol")["total_return_close"].pct_change(
        fill_method=None
    )
    new = new.loc[new["date"].ge(cutoff), ["date", "symbol", "daily_total_return"]]

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.dropna(subset=["date", "symbol", "daily_total_return"])
    combined = combined.loc[
        np.isfinite(combined["daily_total_return"])
        & combined["daily_total_return"].gt(-1.0)
    ]
    if combined.duplicated(["date", "symbol"]).any():
        raise G2MechanismError("成分日收益在切换后存在重复键")
    returns = combined.pivot(
        index="date", columns="symbol", values="daily_total_return"
    ).reindex(market_dates)
    observed = returns.notna().cumsum().gt(0)
    filled_returns = returns.fillna(0.0).where(observed)
    return (1.0 + filled_returns).cumprod().where(observed)


def _wide_to_long(
    frame: pd.DataFrame,
    *,
    value_name: str,
) -> pd.DataFrame:
    return (
        frame.rename_axis("date")
        .rename_axis("symbol", axis=1)
        .stack()
        .rename(value_name)
        .reset_index()
    )


def build_etf_total_return_features(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    start: str,
    cutoff: str,
) -> pd.DataFrame:
    _require_columns(prices, ["date", "close", "symbol"], "510300 日行情")
    _require_columns(dividends, ["symbol", "ex_date", "cash_dividend_per_share"], "分红")
    market = prices[["date", "close", "symbol"]].copy()
    market["date"] = normalize_dates(market["date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.loc[
        market["symbol"].astype(str).eq("510300.SH")
        & market["date"].between(pd.Timestamp(start), pd.Timestamp(cutoff))
    ].sort_values("date")
    if market.empty or market["date"].duplicated().any():
        raise G2MechanismError("510300 日行情为空或日期重复")
    cash = dividends[["symbol", "ex_date", "cash_dividend_per_share"]].copy()
    cash["ex_date"] = normalize_dates(cash["ex_date"])
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    cash = cash.loc[cash["symbol"].astype(str).eq("510300.SH")]
    dividend_by_date = cash.groupby("ex_date")["cash_dividend_per_share"].sum()
    market["cash_dividend"] = market["date"].map(dividend_by_date).fillna(0.0)
    market["daily_total_return"] = (
        (market["close"] + market["cash_dividend"])
        / market["close"].shift(1)
        - 1.0
    )
    market["etf_total_return_index"] = (1.0 + market["daily_total_return"].fillna(0.0)).cumprod()
    market["etf_return20"] = (
        market["etf_total_return_index"]
        / market["etf_total_return_index"].shift(20)
        - 1.0
    )
    market["realized_vol20"] = market["daily_total_return"].rolling(20).std(ddof=1)
    market["price_return_risk_percentile"] = causal_midrank(-market["etf_return20"])
    market["price_vol_risk_percentile"] = causal_midrank(market["realized_vol20"])
    market["price_baseline"] = required_median(
        market,
        ["price_return_risk_percentile", "price_vol_risk_percentile"],
    )
    return market.reset_index(drop=True)


def build_internal_features(
    *,
    membership_with_industry: pd.DataFrame,
    constituent_index: pd.DataFrame,
    etf_features: pd.DataFrame,
    lookback_days: int = 20,
    change_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 F/T 原始量、因果分位数与内部机制分数。"""

    _require_columns(
        membership_with_industry,
        ["date", "symbol", "industry_l1_code"],
        "带行业点时成分",
    )
    dates = pd.DatetimeIndex(etf_features["date"])
    if not constituent_index.index.equals(dates):
        raise G2MechanismError("成分总收益指数与 510300 市场日历不一致")
    return20 = constituent_index / constituent_index.shift(lookback_days) - 1.0
    daily_return = constituent_index.pct_change(fill_method=None)
    member = membership_with_industry.copy()
    member = member.merge(
        _wide_to_long(return20, value_name="member_return20"),
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    ).merge(
        _wide_to_long(daily_return, value_name="member_daily_return"),
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    daily = (
        member.groupby("date", sort=True)
        .agg(
            member_count=("symbol", "nunique"),
            industry_count=("industry_l1_code", "count"),
            return20_count=("member_return20", "count"),
            daily_return_count=("member_daily_return", "count"),
            breadth20=("member_return20", lambda value: float((value > 0).mean())),
            member_equal_weight_return20=("member_return20", "mean"),
        )
        .reset_index()
    )
    complete20 = daily["member_count"].eq(300) & daily["return20_count"].eq(300)
    daily.loc[~complete20, ["breadth20", "member_equal_weight_return20"]] = np.nan
    etf = etf_features.set_index("date")
    daily["etf_return20"] = daily["date"].map(etf["etf_return20"])
    daily["leadership_gap20"] = (
        daily["etf_return20"] - daily["member_equal_weight_return20"]
    )

    complete_day = (
        daily.set_index("date")["member_count"].eq(300)
        & daily.set_index("date")["industry_count"].eq(300)
        & daily.set_index("date")["daily_return_count"].eq(300)
    )
    sector_daily = (
        member.dropna(subset=["industry_l1_code", "member_daily_return"])
        .groupby(["date", "industry_l1_code"], sort=True)["member_daily_return"]
        .mean()
        .unstack("industry_l1_code")
        .reindex(dates)
    )
    sector_corr20 = pd.Series(np.nan, index=dates, dtype=float)
    negative_diffusion5 = pd.Series(np.nan, index=dates, dtype=float)
    for position, date in enumerate(dates):
        if position >= lookback_days - 1:
            window_dates = dates[position - lookback_days + 1 : position + 1]
            if complete_day.reindex(window_dates).fillna(False).all():
                window = sector_daily.loc[window_dates]
                columns = window.columns[window.notna().all(axis=0)]
                if len(columns) >= 2:
                    corr = window[columns].corr().to_numpy(dtype=float)
                    values = corr[np.triu_indices_from(corr, k=1)]
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        sector_corr20.loc[date] = float(np.median(finite))
        if position >= change_days - 1:
            window_dates = dates[position - change_days + 1 : position + 1]
            if complete_day.reindex(window_dates).fillna(False).all():
                window = sector_daily.loc[window_dates]
                columns = window.columns[window.notna().all(axis=0)]
                if len(columns):
                    compound = (1.0 + window[columns]).prod(axis=0) - 1.0
                    negative_diffusion5.loc[date] = float(compound.lt(0.0).mean())

    daily = daily.set_index("date").reindex(dates)
    daily["sector_corr20"] = sector_corr20
    daily["negative_sector_diffusion5"] = negative_diffusion5
    daily["negative_delta_breadth5"] = -(
        daily["breadth20"] - daily["breadth20"].shift(change_days)
    )
    daily["delta_corr5"] = daily["sector_corr20"] - daily["sector_corr20"].shift(
        change_days
    )
    raw_to_percentile = {
        "f1_breadth_risk_percentile": -daily["breadth20"],
        "f2_leadership_risk_percentile": daily["leadership_gap20"],
        "f3_sector_corr_risk_percentile": daily["sector_corr20"],
        "t1_breadth_change_risk_percentile": daily["negative_delta_breadth5"],
        "t2_corr_change_risk_percentile": daily["delta_corr5"],
        "t3_negative_diffusion_risk_percentile": daily[
            "negative_sector_diffusion5"
        ],
    }
    for name, values in raw_to_percentile.items():
        daily[name] = causal_midrank(values)
    daily["F"] = required_median(
        daily,
        [
            "f1_breadth_risk_percentile",
            "f2_leadership_risk_percentile",
            "f3_sector_corr_risk_percentile",
        ],
    )
    daily["T"] = required_median(
        daily,
        [
            "t1_breadth_change_risk_percentile",
            "t2_corr_change_risk_percentile",
            "t3_negative_diffusion_risk_percentile",
        ],
    )
    daily["internal_score"] = np.sqrt(daily["F"] * daily["T"])
    return daily.reset_index(names="date"), sector_daily.reset_index(names="date")


def _previous_market_date_map(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(dates, index=dates).shift(1)


def _publication_asof(
    market_dates: pd.DatetimeIndex,
    available_at: pd.Series,
    values: pd.Series,
) -> pd.Series:
    releases = pd.DataFrame(
        {
            "available_at": normalize_utc_timestamps(available_at),
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna().sort_values("available_at")
    close_clock = pd.Series(market_dates).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(
        hours=15
    )
    query = pd.DataFrame(
        {
            "date": market_dates,
            "available_at": normalize_utc_timestamps(
                close_clock.dt.tz_convert("UTC")
            ),
        }
    ).sort_values("available_at")
    merged = pd.merge_asof(query, releases, on="available_at", direction="backward")
    return pd.Series(merged["value"].to_numpy(), index=merged["date"])


def build_macro_features(
    *,
    market_dates: pd.DatetimeIndex,
    pe: pd.DataFrame,
    bond: pd.DataFrame,
    dr007: pd.DataFrame,
    policy: pd.DataFrame,
    credit: pd.DataFrame,
) -> pd.DataFrame:
    """按旧文件固定定义构造 M 的三个严格因果通道。"""

    previous = _previous_market_date_map(market_dates)
    frame = pd.DataFrame(index=market_dates)

    _require_columns(pe, ["date", "index_code", "pe_official"], "官方估值")
    pe_work = pe[["date", "index_code", "pe_official"]].copy()
    pe_work["date"] = normalize_dates(pe_work["date"])
    pe_work = pe_work.loc[pe_work["index_code"].astype(str).eq("000300")]
    pe_series = pd.to_numeric(pe_work["pe_official"], errors="coerce")
    pe_by_date = pd.Series(pe_series.to_numpy(), index=pe_work["date"]).groupby(level=0).last()

    _require_columns(bond, ["date", "cgb_10y"], "十年国债收益率")
    bond_work = bond[["date", "cgb_10y"]].copy()
    bond_work["date"] = normalize_dates(bond_work["date"])
    bond_by_date = pd.Series(
        pd.to_numeric(bond_work["cgb_10y"], errors="coerce").to_numpy(),
        index=bond_work["date"],
    ).groupby(level=0).last()
    frame["erp_buffer"] = previous.map(lambda date: 100.0 / pe_by_date.get(date, np.nan) - bond_by_date.get(date, np.nan))

    _require_columns(dr007, ["date", "dr007"], "DR007")
    dr_work = dr007[["date", "dr007"]].copy()
    dr_work["date"] = normalize_dates(dr_work["date"])
    dr_by_date = pd.Series(
        pd.to_numeric(dr_work["dr007"], errors="coerce").to_numpy(),
        index=dr_work["date"],
    ).groupby(level=0).last()

    _require_columns(policy, ["published_at", "seven_day_rate_percent"], "七天逆回购利率")
    policy_at_previous_close = _publication_asof(
        market_dates,
        policy["published_at"],
        policy["seven_day_rate_percent"],
    ).shift(1)
    frame["funding_stress"] = previous.map(dr_by_date) - policy_at_previous_close.reindex(
        market_dates
    )

    _require_columns(
        credit,
        ["reference_period", "available_at", "first_release_value"],
        "社融首次发布",
    )
    credit_work = credit[
        ["reference_period", "available_at", "first_release_value"]
    ].copy()
    credit_work["reference_period"] = pd.PeriodIndex(
        credit_work["reference_period"].astype(str), freq="M"
    )
    credit_work = credit_work.sort_values("reference_period").drop_duplicates(
        "reference_period", keep="first"
    )
    credit_work["first_release_value"] = pd.to_numeric(
        credit_work["first_release_value"], errors="coerce"
    )
    value_by_period = credit_work.set_index("reference_period")["first_release_value"]
    credit_work["credit_impulse"] = [
        value - value_by_period.get(period - 3, np.nan)
        for period, value in zip(
            credit_work["reference_period"], credit_work["first_release_value"]
        )
    ]
    frame["credit_impulse"] = _publication_asof(
        market_dates,
        credit_work["available_at"],
        credit_work["credit_impulse"],
    ).reindex(market_dates)

    frame["m1_erp_risk_percentile"] = causal_midrank(-frame["erp_buffer"])
    frame["m2_funding_risk_percentile"] = causal_midrank(frame["funding_stress"])
    frame["m3_credit_risk_percentile"] = causal_midrank(-frame["credit_impulse"])
    frame["M"] = required_median(
        frame,
        [
            "m1_erp_risk_percentile",
            "m2_funding_risk_percentile",
            "m3_credit_risk_percentile",
        ],
    )
    return frame.reset_index(names="date")


def assemble_feature_panel(
    *,
    etf: pd.DataFrame,
    internal: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    panel = (
        etf.merge(internal, on="date", how="left", validate="one_to_one", suffixes=("", "_internal"))
        .merge(macro, on="date", how="left", validate="one_to_one")
        .sort_values("date")
        .reset_index(drop=True)
    )
    panel["full_mechanism_score"] = np.cbrt(panel["M"] * panel["F"] * panel["T"])
    return panel


def build_independent_units(
    *,
    events: pd.DataFrame,
    non_events: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(events, ["event_id", "event_origin_start"], "独立 BAD10 事件")
    _require_columns(non_events, ["block_id", "origin_date"], "独立非事件块")
    positive = events[["event_id", "event_origin_start"]].rename(
        columns={"event_id": "unit_id", "event_origin_start": "origin_date"}
    )
    positive["bad10"] = 1
    negative = non_events[["block_id", "origin_date"]].rename(
        columns={"block_id": "unit_id"}
    )
    negative["bad10"] = 0
    units = pd.concat([positive, negative], ignore_index=True)
    units["origin_date"] = normalize_dates(units["origin_date"])
    selected = features[
        ["date", "internal_score", "price_baseline", "M", "F", "T", "full_mechanism_score"]
    ].rename(columns={"date": "origin_date"})
    return units.merge(selected, on="origin_date", how="left", validate="many_to_one").sort_values(
        ["origin_date", "bad10"], ascending=[True, False]
    ).reset_index(drop=True)


def evaluate_g2(
    *,
    features: pd.DataFrame,
    units: pd.DataFrame,
    subperiods: Sequence[Subperiod],
    lead_days: int,
    minimum_events: int,
    minimum_non_events: int,
    required_positive_subperiods: int,
) -> dict[str, Any]:
    ordered = features.sort_values("date").reset_index(drop=True).copy()
    ordered["lead_date"] = ordered["date"].shift(-lead_days)
    ordered["internal_change_lead"] = (
        ordered["internal_score"].shift(-lead_days) - ordered["internal_score"]
    )
    period_results: list[dict[str, Any]] = []
    for period in subperiods:
        sample = ordered.loc[
            ordered["date"].between(period.start, period.end)
            & ordered["lead_date"].between(period.start, period.end),
            ["M", "internal_change_lead"],
        ].dropna()
        if len(sample) < 2 or sample.nunique().min() < 2:
            correlation = np.nan
        else:
            correlation = float(
                spearmanr(sample["M"], sample["internal_change_lead"]).statistic
            )
        period_results.append(
            {
                "subperiod_id": period.period_id,
                "start": period.start.date().isoformat(),
                "end": period.end.date().isoformat(),
                "scoreable_daily_observations": int(len(sample)),
                "spearman_M_to_internal_change_t_plus_5": (
                    correlation if np.isfinite(correlation) else None
                ),
                "strictly_positive": bool(np.isfinite(correlation) and correlation > 0.0),
            }
        )
    positive_count = sum(item["strictly_positive"] for item in period_results)
    macro_gate = positive_count >= required_positive_subperiods

    scored = units.dropna(subset=["internal_score", "price_baseline"]).copy()
    event_count = int(scored["bad10"].eq(1).sum())
    non_event_count = int(scored["bad10"].eq(0).sum())
    coverage_gate = event_count >= minimum_events and non_event_count >= minimum_non_events
    if scored["bad10"].nunique() == 2:
        internal_auc = float(roc_auc_score(scored["bad10"], scored["internal_score"]))
        price_auc = float(roc_auc_score(scored["bad10"], scored["price_baseline"]))
    else:
        internal_auc = np.nan
        price_auc = np.nan
    auc_gate = bool(
        np.isfinite(internal_auc)
        and np.isfinite(price_auc)
        and internal_auc > price_auc
    )
    passed = bool(macro_gate and coverage_gate and auc_gate)
    return {
        "gate_id": "G2_MECHANISM_CHAIN",
        "status": G2_PASS_STATUS if passed else G2_FAIL_STATUS,
        "passed": passed,
        "macro_lead_gate": {
            "passed": macro_gate,
            "positive_subperiod_count": int(positive_count),
            "required_positive_subperiods": int(required_positive_subperiods),
            "total_subperiods": int(len(subperiods)),
            "subperiods": period_results,
        },
        "internal_bad10_gate": {
            "passed": bool(coverage_gate and auc_gate),
            "coverage_passed": coverage_gate,
            "auc_passed": auc_gate,
            "scoreable_event_count": event_count,
            "minimum_event_count": int(minimum_events),
            "scoreable_non_event_count": non_event_count,
            "minimum_non_event_count": int(minimum_non_events),
            "internal_score_roc_auc": internal_auc if np.isfinite(internal_auc) else None,
            "price_baseline_roc_auc": price_auc if np.isfinite(price_auc) else None,
            "auc_difference_internal_minus_price": (
                internal_auc - price_auc
                if np.isfinite(internal_auc) and np.isfinite(price_auc)
                else None
            ),
        },
        "g3_allowed": passed,
        "feature_rescue_allowed": False,
        "window_or_sign_search_performed": False,
        "model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "next_allowed_action": (
            "FREEZE_AND_RUN_G3_FIXED_PROBABILITY_MODELS"
            if passed
            else "NONE_V1_REJECTED_AT_G2_NO_FEATURE_RESCUE"
        ),
    }


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")
