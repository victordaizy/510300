"""510300 非对称压力风险 V1 的 BAD10 标签与独立事件普查。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Bad10CensusError(RuntimeError):
    """BAD10 普查输入或不变量不满足时的封闭失败。"""


ORIGIN_COLUMNS = [
    "origin_date",
    "entry_date",
    "horizon_end_date",
    "bad10",
    "minimum_path_return",
    "first_breach_date",
    "cash_dividend_per_share_in_horizon",
]

EVENT_COLUMNS = [
    "event_id",
    "event_origin_start",
    "event_origin_end",
    "event_entry_start",
    "event_entry_end",
    "event_horizon_end",
    "bad10_origin_count",
    "worst_minimum_path_return",
    "worst_origin_date",
    "first_breach_date",
]

NON_EVENT_BLOCK_COLUMNS = [
    "block_id",
    "origin_date",
    "entry_date",
    "horizon_end_date",
    "minimum_path_return",
]


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise Bad10CensusError(f"{label}缺少必需列：{missing}")


def prepare_etf_prices(
    prices: pd.DataFrame,
    *,
    expected_symbol: str,
    start_date: str,
    observation_cutoff: str,
) -> pd.DataFrame:
    """准入未复权 510300 开收盘价，绝不使用指数替代执行路径。"""

    _require_columns(prices, ["date", "open", "close", "symbol"], "ETF 日行情")
    frame = prices[["date", "open", "close", "symbol"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string")
    frame = frame.loc[
        frame["date"].between(
            pd.Timestamp(start_date), pd.Timestamp(observation_cutoff)
        )
    ].copy()
    if frame.empty:
        raise Bad10CensusError("ETF 日行情在冻结区间内为空")
    if frame[["date", "open", "close", "symbol"]].isna().any().any():
        raise Bad10CensusError("ETF 日行情存在缺失或不可解析字段")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(False), "date"].dt.date
        raise Bad10CensusError(f"ETF 日行情存在重复交易日：{list(duplicates)[:5]}")
    unexpected = sorted(set(frame["symbol"].astype(str)).difference({expected_symbol}))
    if unexpected:
        raise Bad10CensusError(f"ETF 日行情含有非冻结证券：{unexpected}")
    if (frame[["open", "close"]] <= 0.0).any().any():
        raise Bad10CensusError("ETF 开盘价或收盘价不是严格正数")
    if not np.isfinite(frame[["open", "close"]].to_numpy(dtype=float)).all():
        raise Bad10CensusError("ETF 开盘价或收盘价含非有限值")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def prepare_cash_dividends(
    dividends: pd.DataFrame,
    *,
    expected_symbol: str,
) -> pd.DataFrame:
    """准入官方现金分红，并冻结登记日、除息日和支付日时钟。"""

    required = [
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
    ]
    _require_columns(dividends, required, "ETF 现金分红")
    frame = dividends[required].copy()
    for column in ("record_date", "ex_date", "payment_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["cash_dividend_per_share"] = pd.to_numeric(
        frame["cash_dividend_per_share"], errors="coerce"
    )
    frame["symbol"] = frame["symbol"].astype("string")
    if frame[required].isna().any().any():
        raise Bad10CensusError("ETF 现金分红存在缺失或不可解析字段")
    unexpected = sorted(set(frame["symbol"].astype(str)).difference({expected_symbol}))
    if unexpected:
        raise Bad10CensusError(f"ETF 现金分红含有非冻结证券：{unexpected}")
    if (frame["cash_dividend_per_share"] <= 0.0).any():
        raise Bad10CensusError("现金分红必须为严格正数")
    if not np.isfinite(frame["cash_dividend_per_share"].to_numpy(dtype=float)).all():
        raise Bad10CensusError("现金分红含非有限值")
    if not (frame["record_date"] <= frame["ex_date"]).all():
        raise Bad10CensusError("现金分红登记日晚于除息日")
    if not (frame["ex_date"] <= frame["payment_date"]).all():
        raise Bad10CensusError("现金分红除息日晚于支付日")
    duplicate_key = ["symbol", "record_date", "ex_date", "cash_dividend_per_share"]
    if frame.duplicated(duplicate_key).any():
        raise Bad10CensusError("ETF 现金分红存在重复事件")
    return frame.sort_values(["record_date", "ex_date"], kind="stable").reset_index(
        drop=True
    )


def build_bad10_origin_panel(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    horizon_market_days: int,
    loss_threshold: float,
) -> pd.DataFrame:
    """从 t 收盘形成原点，以 t+1 开盘为成本计算未来十日收盘财富路径。"""

    if horizon_market_days <= 0:
        raise Bad10CensusError("持有期必须为正整数")
    if not -1.0 < loss_threshold < 0.0:
        raise Bad10CensusError("损失阈值必须位于 -1 与 0 之间")
    _require_columns(prices, ["date", "open", "close"], "准入 ETF 日行情")
    _require_columns(
        dividends,
        ["record_date", "ex_date", "cash_dividend_per_share"],
        "准入 ETF 现金分红",
    )
    if len(prices) < horizon_market_days + 1:
        raise Bad10CensusError("ETF 日行情不足以形成一个完整 BAD10 原点")

    rows: list[dict[str, Any]] = []
    maximum_origin = len(prices) - horizon_market_days - 1
    for origin_index in range(maximum_origin + 1):
        origin = prices.iloc[origin_index]
        entry = prices.iloc[origin_index + 1]
        path = prices.iloc[
            origin_index + 1 : origin_index + horizon_market_days + 1
        ].copy()
        if len(path) != horizon_market_days:
            raise Bad10CensusError("内部错误：BAD10 路径长度不等于冻结期限")
        entry_date = pd.Timestamp(entry["date"])
        entry_open = float(entry["open"])
        entitled = dividends.loc[dividends["record_date"].ge(entry_date)].copy()

        wealth_returns: list[float] = []
        for observation in path.itertuples(index=False):
            observation_date = pd.Timestamp(observation.date)
            receivable = float(
                entitled.loc[
                    entitled["ex_date"].le(observation_date),
                    "cash_dividend_per_share",
                ].sum()
            )
            wealth = float(observation.close) + receivable
            wealth_returns.append(wealth / entry_open - 1.0)

        return_array = np.asarray(wealth_returns, dtype=float)
        if not np.isfinite(return_array).all():
            raise Bad10CensusError("BAD10 财富路径出现非有限值")
        minimum_index = int(np.argmin(return_array))
        minimum_path_return = float(return_array[minimum_index])
        breach_indices = np.flatnonzero(return_array <= loss_threshold)
        first_breach_date = (
            pd.Timestamp(path.iloc[int(breach_indices[0])]["date"])
            if breach_indices.size
            else pd.NaT
        )
        horizon_end_date = pd.Timestamp(path.iloc[-1]["date"])
        horizon_dividend = float(
            entitled.loc[
                entitled["ex_date"].le(horizon_end_date),
                "cash_dividend_per_share",
            ].sum()
        )
        rows.append(
            {
                "origin_date": pd.Timestamp(origin["date"]),
                "entry_date": entry_date,
                "horizon_end_date": horizon_end_date,
                "bad10": int(minimum_path_return <= loss_threshold),
                "minimum_path_return": minimum_path_return,
                "first_breach_date": first_breach_date,
                "cash_dividend_per_share_in_horizon": horizon_dividend,
            }
        )

    result = pd.DataFrame(rows, columns=ORIGIN_COLUMNS)
    if result.empty:
        raise Bad10CensusError("BAD10 原点面板为空")
    if not result["origin_date"].is_monotonic_increasing:
        raise Bad10CensusError("BAD10 原点日期不是严格递增")
    return result


def _flush_event(group: pd.DataFrame, event_number: int) -> dict[str, Any]:
    ordered = group.sort_values("origin_date", kind="stable")
    worst_index = ordered["minimum_path_return"].idxmin()
    worst = ordered.loc[worst_index]
    breach_dates = ordered["first_breach_date"].dropna()
    if breach_dates.empty:
        raise Bad10CensusError("BAD10 事件没有任何首次突破日期")
    return {
        "event_id": f"BAD10_EVENT_{event_number:04d}",
        "event_origin_start": ordered["origin_date"].min(),
        "event_origin_end": ordered["origin_date"].max(),
        "event_entry_start": ordered["entry_date"].min(),
        "event_entry_end": ordered["entry_date"].max(),
        "event_horizon_end": ordered["horizon_end_date"].max(),
        "bad10_origin_count": int(len(ordered)),
        "worst_minimum_path_return": float(worst["minimum_path_return"]),
        "worst_origin_date": pd.Timestamp(worst["origin_date"]),
        "first_breach_date": pd.Timestamp(breach_dates.min()),
    }


def merge_bad10_events(origin_panel: pd.DataFrame) -> pd.DataFrame:
    """按未来持有区间的闭区间重叠关系合并为独立压力事件。"""

    _require_columns(origin_panel, ORIGIN_COLUMNS, "BAD10 原点面板")
    bad = origin_panel.loc[origin_panel["bad10"].eq(1)].sort_values(
        "entry_date", kind="stable"
    )
    if bad.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    groups: list[pd.DataFrame] = []
    current_indices: list[int] = []
    current_end: pd.Timestamp | None = None
    for index, row in bad.iterrows():
        entry_date = pd.Timestamp(row["entry_date"])
        horizon_end = pd.Timestamp(row["horizon_end_date"])
        if current_end is None or entry_date <= current_end:
            current_indices.append(index)
            current_end = horizon_end if current_end is None else max(current_end, horizon_end)
        else:
            groups.append(bad.loc[current_indices].copy())
            current_indices = [index]
            current_end = horizon_end
    groups.append(bad.loc[current_indices].copy())

    events = pd.DataFrame(
        [_flush_event(group, number) for number, group in enumerate(groups, start=1)],
        columns=EVENT_COLUMNS,
    )
    previous_end: pd.Timestamp | None = None
    for row in events.itertuples(index=False):
        start = pd.Timestamp(row.event_entry_start)
        end = pd.Timestamp(row.event_horizon_end)
        if previous_end is not None and start <= previous_end:
            raise Bad10CensusError("独立压力事件仍存在重叠")
        previous_end = end
    return events


def _overlaps_any_event(
    entry_date: pd.Timestamp,
    horizon_end_date: pd.Timestamp,
    events: pd.DataFrame,
) -> bool:
    if events.empty:
        return False
    overlap = events["event_entry_start"].le(horizon_end_date) & events[
        "event_horizon_end"
    ].ge(entry_date)
    return bool(overlap.any())


def select_independent_non_event_blocks(
    origin_panel: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """贪心选择互不重叠、且不与任何压力事件重叠的十日非事件块。"""

    _require_columns(origin_panel, ORIGIN_COLUMNS, "BAD10 原点面板")
    if not events.empty:
        _require_columns(events, EVENT_COLUMNS, "BAD10 独立事件")
    candidates = origin_panel.loc[origin_panel["bad10"].eq(0)].sort_values(
        "entry_date", kind="stable"
    )
    rows: list[dict[str, Any]] = []
    previous_end: pd.Timestamp | None = None
    for row in candidates.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        horizon_end = pd.Timestamp(row.horizon_end_date)
        if previous_end is not None and entry_date <= previous_end:
            continue
        if _overlaps_any_event(entry_date, horizon_end, events):
            continue
        rows.append(
            {
                "block_id": f"NON_EVENT_BLOCK_{len(rows) + 1:04d}",
                "origin_date": pd.Timestamp(row.origin_date),
                "entry_date": entry_date,
                "horizon_end_date": horizon_end,
                "minimum_path_return": float(row.minimum_path_return),
            }
        )
        previous_end = horizon_end
    return pd.DataFrame(rows, columns=NON_EVENT_BLOCK_COLUMNS)


def adjudicate_g1(
    *,
    independent_event_count: int,
    independent_non_event_block_count: int,
    minimum_independent_events: int,
    minimum_independent_non_event_blocks: int,
) -> dict[str, Any]:
    event_pass = independent_event_count >= minimum_independent_events
    non_event_pass = (
        independent_non_event_block_count >= minimum_independent_non_event_blocks
    )
    passed = event_pass and non_event_pass
    return {
        "gate_id": "G1_LABEL_IDENTIFIABILITY",
        "status": (
            "PASS_G1_LABEL_IDENTIFIABILITY_ONLY_NO_FEATURE_CONSTRUCTION"
            if passed
            else "REJECTED_FROZEN_G1_LABEL_IDENTIFIABILITY_FAILED_NO_RESCUE"
        ),
        "passed": passed,
        "independent_event_count": int(independent_event_count),
        "minimum_independent_events": int(minimum_independent_events),
        "independent_event_count_passed": event_pass,
        "independent_non_event_block_count": int(independent_non_event_block_count),
        "minimum_independent_non_event_blocks": int(
            minimum_independent_non_event_blocks
        ),
        "independent_non_event_block_count_passed": non_event_pass,
        "label_change_allowed": False,
        "feature_construction_allowed": passed,
        "model_training_allowed": False,
        "portfolio_evaluation_allowed": False,
        "rescue_allowed": False,
    }


def build_census_summary(
    origin_panel: pd.DataFrame,
    events: pd.DataFrame,
    non_event_blocks: pd.DataFrame,
    *,
    horizon_market_days: int,
    loss_threshold: float,
    gate: dict[str, Any],
) -> dict[str, Any]:
    bad_count = int(origin_panel["bad10"].sum())
    return {
        "eligible_origin_count": int(len(origin_panel)),
        "bad10_origin_count": bad_count,
        "bad10_origin_prevalence": float(bad_count / len(origin_panel)),
        "independent_event_count": int(len(events)),
        "independent_non_event_block_count": int(len(non_event_blocks)),
        "first_origin_date": origin_panel["origin_date"].min().date().isoformat(),
        "last_origin_date": origin_panel["origin_date"].max().date().isoformat(),
        "first_entry_date": origin_panel["entry_date"].min().date().isoformat(),
        "last_horizon_end_date": origin_panel["horizon_end_date"].max().date().isoformat(),
        "horizon_market_days": int(horizon_market_days),
        "loss_threshold": float(loss_threshold),
        "g1": gate,
    }
