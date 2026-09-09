"""通达信公开行情协议的15分钟数据接入与历史成交重建。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

from market_data.minute import annotate_session_phase


@dataclass(frozen=True)
class TdxServer:
    """通达信行情服务器。"""

    host: str
    port: int


@dataclass(frozen=True)
class TdxMinuteRequest:
    """510300十五分钟行情请求。"""

    symbol: str = "510300.SH"
    start_date: str = "2021-08-12"
    end_date: str | None = None
    page_size: int = 800
    transaction_page_size: int = 2000
    maximum_pages: int = 40
    timeout_seconds: float = 8.0
    timezone: str = "Asia/Shanghai"
    price_divisor: float = 10.0
    volume_multiplier: float = 100.0

    def validate(self) -> None:
        if self.symbol != "510300.SH":
            raise ValueError("当前价格缩放校准仅适用于510300.SH")
        if not 1 <= self.page_size <= 800:
            raise ValueError("通达信K线单页数量必须位于1至800")
        if not 1 <= self.transaction_page_size <= 2000:
            raise ValueError("通达信历史成交单页数量必须位于1至2000")
        if self.maximum_pages <= 0:
            raise ValueError("maximum_pages必须为正数")
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须为正数")
        if self.price_divisor <= 0 or self.volume_multiplier <= 0:
            raise ValueError("价格和成交量缩放参数必须为正数")


def _market_and_code(symbol: str) -> tuple[int, str]:
    code, suffix = symbol.split(".", maxsplit=1)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"无法识别证券代码：{symbol}")
    if suffix == "SH":
        return 1, code
    if suffix == "SZ":
        return 0, code
    raise ValueError(f"无法识别交易所后缀：{symbol}")


def connect_tdx(server: TdxServer, timeout_seconds: float) -> TdxHq_API:
    """建立独立连接；调用方负责断开。"""

    api = TdxHq_API(heartbeat=True, auto_retry=True, raise_exception=True)
    api.connect(server.host, server.port, time_out=timeout_seconds)
    return api


def _base_columns(
    data: pd.DataFrame,
    request: TdxMinuteRequest,
    source: str,
    amount_is_estimated: bool,
) -> pd.DataFrame:
    output = data.copy()
    output["bar_start"] = output["bar_end"] - pd.Timedelta(minutes=15)
    output["trade_date"] = output["bar_end"].dt.normalize()
    output["symbol"] = request.symbol
    output["frequency_minutes"] = 15
    output["timestamp_meaning"] = "bar_end"
    output["adjustment"] = "raw"
    output["source"] = source
    output["timezone"] = request.timezone
    output["volume_unit"] = "share"
    output["amount_unit"] = "CNY"
    output["amount_is_estimated"] = amount_is_estimated
    output["retrieved_at"] = datetime.now(ZoneInfo(request.timezone)).isoformat()
    first_columns = [
        "bar_end", "bar_start", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "symbol", "frequency_minutes", "timestamp_meaning",
        "adjustment", "source", "timezone", "volume_unit", "amount_unit",
        "amount_is_estimated", "retrieved_at",
    ]
    remaining = [column for column in output.columns if column not in first_columns]
    output = output[first_columns + remaining]
    output = annotate_session_phase(output)
    return output.sort_values("bar_end").reset_index(drop=True)


def fetch_recent_direct_15m(
    api: TdxHq_API,
    request: TdxMinuteRequest,
) -> pd.DataFrame:
    """分页获取服务器保留的原生15分钟K线。"""

    request.validate()
    market, code = _market_and_code(request.symbol)
    pages: list[pd.DataFrame] = []
    for page_index in range(request.maximum_pages):
        offset = page_index * request.page_size
        records = api.get_security_bars(
            TDXParams.KLINE_TYPE_15MIN,
            market,
            code,
            offset,
            request.page_size,
        )
        if not records:
            break
        page = pd.DataFrame.from_records(records)
        pages.append(page)
        if len(records) < request.page_size:
            break
    if not pages:
        raise RuntimeError("通达信原生15分钟接口返回空数据")

    raw = pd.concat(pages, ignore_index=True)
    required = {"datetime", "open", "high", "low", "close", "vol", "amount"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"通达信原生15分钟接口缺少字段：{sorted(missing)}")
    data = raw.rename(columns={"datetime": "bar_end", "vol": "volume"})[
        ["bar_end", "open", "high", "low", "close", "volume", "amount"]
    ].copy()
    data["bar_end"] = pd.to_datetime(data["bar_end"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if data[["bar_end", *numeric_columns]].isna().any().any():
        raise RuntimeError("通达信原生15分钟接口存在无法解析的字段")
    start = pd.Timestamp(request.start_date)
    end = pd.Timestamp(request.end_date) + pd.Timedelta(days=1) if request.end_date else None
    data = data.loc[data["bar_end"] >= start]
    if end is not None:
        data = data.loc[data["bar_end"] < end]
    data = data.drop_duplicates("bar_end", keep="last")
    data["construction"] = "exchange_15m_kline"
    data["source_transaction_count"] = pd.NA
    return _base_columns(
        data,
        request,
        source="pytdx.get_security_bars.15m",
        amount_is_estimated=False,
    )


def _bar_end_from_transaction_time(value: str) -> str | None:
    """按交易所15分钟边界归桶，集合竞价成交纳入第一根和最后一根。"""

    hour, minute = (int(part) for part in value.split(":"))
    absolute_minute = hour * 60 + minute
    if 9 * 60 + 25 <= absolute_minute < 9 * 60 + 45:
        endpoint = 9 * 60 + 45
    elif 9 * 60 + 45 <= absolute_minute <= 11 * 60 + 30:
        endpoint = min(
            11 * 60 + 30,
            10 * 60 + ((absolute_minute - (9 * 60 + 45)) // 15) * 15,
        )
    elif 13 * 60 <= absolute_minute < 13 * 60 + 15:
        endpoint = 13 * 60 + 15
    elif 13 * 60 + 15 <= absolute_minute <= 15 * 60:
        endpoint = min(
            15 * 60,
            13 * 60 + 30 + ((absolute_minute - (13 * 60 + 15)) // 15) * 15,
        )
    else:
        return None
    return f"{endpoint // 60:02d}:{endpoint % 60:02d}"


def fetch_history_transactions(
    api: TdxHq_API,
    request: TdxMinuteRequest,
    trade_date: date | pd.Timestamp | str,
) -> list[dict]:
    """获取某交易日的历史成交记录，返回顺序统一为从早到晚。"""

    request.validate()
    market, code = _market_and_code(request.symbol)
    day = pd.Timestamp(trade_date)
    date_number = int(day.strftime("%Y%m%d"))
    pages: list[Sequence[dict]] = []
    for page_index in range(request.maximum_pages):
        offset = page_index * request.transaction_page_size
        records = api.get_history_transaction_data(
            market,
            code,
            offset,
            request.transaction_page_size,
            date_number,
        )
        if not records:
            break
        pages.append(records)
        if len(records) < request.transaction_page_size:
            break
    if not pages:
        raise RuntimeError(f"{day.date().isoformat()}历史成交接口返回空数据")
    return [dict(record) for page in reversed(pages) for record in page]


def reconstruct_15m_from_transactions(
    records: Iterable[dict],
    trade_date: date | pd.Timestamp | str,
    request: TdxMinuteRequest,
) -> pd.DataFrame:
    """由历史成交记录重建未复权15分钟OHLCV。

    通达信该历史成交端点对510300使用价格乘10、成交量以100份为单位；
    缩放值在配置和日线对账中双重校验。成交额由成交价乘成交量估算。
    """

    request.validate()
    day = pd.Timestamp(trade_date).normalize()
    raw = pd.DataFrame.from_records(records)
    required = {"time", "price", "vol"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"历史成交记录缺少字段：{sorted(missing)}")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce") / request.price_divisor
    raw["vol"] = pd.to_numeric(raw["vol"], errors="coerce")
    raw = raw.loc[raw["price"].notna() & raw["vol"].notna() & (raw["price"] > 0)].copy()
    raw["bar_end_time"] = raw["time"].astype(str).map(_bar_end_from_transaction_time)
    raw = raw.dropna(subset=["bar_end_time"])
    if raw.empty:
        raise RuntimeError(f"{day.date().isoformat()}没有可归入正常交易时段的成交记录")
    raw["volume_share"] = raw["vol"] * request.volume_multiplier
    raw["amount_estimated"] = raw["price"] * raw["volume_share"]
    grouped = raw.groupby("bar_end_time", sort=False, as_index=False).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume_share", "sum"),
        amount=("amount_estimated", "sum"),
        source_transaction_count=("price", "size"),
    )
    grouped["bar_end"] = pd.to_datetime(
        day.strftime("%Y-%m-%d") + " " + grouped["bar_end_time"]
    )
    grouped = grouped.drop(columns="bar_end_time")
    grouped["construction"] = "transaction_reconstructed_15m"
    output = _base_columns(
        grouped,
        request,
        source="pytdx.get_history_transaction_data.reconstructed_15m",
        amount_is_estimated=True,
    )
    output["post_close_separately_identifiable"] = True
    return output

