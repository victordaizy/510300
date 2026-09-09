"""分钟行情提供方接口与新浪15分钟实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SINA_MINUTE_URL = (
    "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/"
    "CN_MarketDataService.getKLineData"
)
TENCENT_MINUTE_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
TENCENT_INTRADAY_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
CLOSE_AUCTION_START = pd.Timestamp("2026-07-06")


@dataclass(frozen=True)
class MinuteDataRequest:
    """分钟行情请求。"""

    symbol: str
    frequency_minutes: int = 15
    adjustment: str = "raw"
    maximum_bars: int = 1970
    timeout_seconds: float = 15.0
    timezone: str = "Asia/Shanghai"

    def validate(self) -> None:
        if self.frequency_minutes not in {1, 5, 15, 30, 60}:
            raise ValueError("分钟频率必须属于1、5、15、30或60")
        if self.adjustment != "raw":
            raise ValueError("真实成交研究只允许未复权raw分钟价格")
        if not 1 <= self.maximum_bars <= 1970:
            raise ValueError("新浪分钟接口maximum_bars必须位于1至1970")
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须为正数")


class MinuteDataProvider(Protocol):
    """分钟行情提供方协议。"""

    provider_name: str

    def fetch(self, request: MinuteDataRequest) -> pd.DataFrame:
        """返回标准化分钟行情。"""


def _sina_symbol(symbol: str) -> str:
    code = symbol.split(".", maxsplit=1)[0]
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"无法识别证券代码：{symbol}")
    if symbol.endswith(".SZ") or code.startswith(("0", "1", "2", "3")):
        return f"sz{code}"
    return f"sh{code}"


def _tencent_symbol(symbol: str) -> str:
    return _sina_symbol(symbol)


def annotate_session_phase(data: pd.DataFrame) -> pd.DataFrame:
    """为15分钟K线标记特殊交易阶段；不声称包含盘后成交。"""

    output = data.copy()
    output["bar_position"] = "regular"
    output["session_phase"] = "intraday"
    output["contains_opening_auction"] = False
    output["contains_closing_auction"] = False
    output["post_close_included"] = False
    output["post_close_separately_identifiable"] = False
    if output.empty:
        return output
    frequency = int(output["frequency_minutes"].iloc[0])
    if frequency not in {5, 15}:
        return output
    times = output["bar_end"].dt.strftime("%H:%M:%S")
    first_time = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=frequency)).strftime(
        "%H:%M:%S"
    )
    afternoon_first_time = (
        pd.Timestamp("2000-01-01 13:00:00") + pd.Timedelta(minutes=frequency)
    ).strftime("%H:%M:%S")
    first = times == first_time
    morning = (times > first_time) & (times <= "11:30:00")
    afternoon_first = times == afternoon_first_time
    afternoon = (times > afternoon_first_time) & (times < "15:00:00")
    last = times == "15:00:00"
    current_regime = output["trade_date"] >= CLOSE_AUCTION_START
    output.loc[first, ["bar_position", "session_phase", "contains_opening_auction"]] = [
        "first", "opening_auction_mixed", True
    ]
    output.loc[morning, "session_phase"] = "continuous_am"
    output.loc[afternoon_first, ["bar_position", "session_phase"]] = [
        "afternoon_first", "continuous_pm"
    ]
    output.loc[afternoon, "session_phase"] = "continuous_pm"
    output.loc[last & ~current_regime, ["bar_position", "session_phase"]] = [
        "last", "legacy_continuous_pm_last"
    ]
    output.loc[last & current_regime, [
        "bar_position", "session_phase", "contains_closing_auction"
    ]] = ["last", "closing_auction_mixed", True]
    return output


def _parse_jsonp(text: str) -> list[dict]:
    marker = "=("
    start = text.find(marker)
    end = text.rfind(");")
    if start < 0 or end <= start:
        raise RuntimeError("新浪分钟接口返回内容不是预期JSONP格式")
    payload = text[start + len(marker) : end]
    result = json.loads(payload)
    if not isinstance(result, list):
        raise RuntimeError("新浪分钟接口JSONP主体不是记录列表")
    return result


class SinaMinuteProvider:
    """新浪CN_MarketDataService分钟K线提供方。

    该端点一次最多返回1970根，不能据此声称拥有五年分钟历史。
    时间字段按K线结束时点解释。
    """

    provider_name = "sina.CN_MarketDataService.getKLineData"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, request: MinuteDataRequest) -> pd.DataFrame:
        request.validate()
        response = self._session.get(
            SINA_MINUTE_URL,
            params={
                "symbol": _sina_symbol(request.symbol),
                "scale": str(request.frequency_minutes),
                "ma": "no",
                "datalen": str(request.maximum_bars),
            },
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        records = _parse_jsonp(response.text)
        if not records:
            raise RuntimeError("新浪分钟接口返回空数据")
        data = pd.DataFrame.from_records(records)
        required_source = ["day", "open", "high", "low", "close", "volume", "amount"]
        missing = [column for column in required_source if column not in data.columns]
        if missing:
            raise RuntimeError(f"新浪分钟接口缺少字段：{missing}")

        output = data[required_source].rename(columns={"day": "bar_end"}).copy()
        output["bar_end"] = pd.to_datetime(output["bar_end"], errors="coerce")
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        output[numeric_columns] = output[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if output["bar_end"].isna().any() or output[numeric_columns].isna().any().any():
            raise RuntimeError("新浪分钟接口存在无法解析的时间或数值")

        output = output.sort_values("bar_end").drop_duplicates("bar_end", keep="last")
        output = output.reset_index(drop=True)
        output.insert(1, "bar_start", output["bar_end"] - pd.Timedelta(minutes=request.frequency_minutes))
        output.insert(2, "trade_date", output["bar_end"].dt.normalize())
        output["symbol"] = request.symbol
        output["frequency_minutes"] = request.frequency_minutes
        output["timestamp_meaning"] = "bar_end"
        output["adjustment"] = request.adjustment
        output["source"] = self.provider_name
        output["timezone"] = request.timezone
        output["volume_unit"] = "share"
        output["amount_unit"] = "CNY"
        output["retrieved_at"] = datetime.now(ZoneInfo(request.timezone)).isoformat()
        return annotate_session_phase(output)


class TencentMinuteProvider:
    """腾讯分钟K线第二来源。

    返回窗口最多320根。第6个数值字段按手处理并乘100标准化为份；
    第8字段不是成交额，因此`amount`明确留空，禁止伪造。
    """

    provider_name = "tencent.ifzq.mkline"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, request: MinuteDataRequest) -> pd.DataFrame:
        request.validate()
        if request.maximum_bars > 320:
            raise ValueError("腾讯分钟接口maximum_bars不能超过320")
        provider_symbol = _tencent_symbol(request.symbol)
        response = self._session.get(
            TENCENT_MINUTE_URL,
            params={"param": f"{provider_symbol},m{request.frequency_minutes},,{request.maximum_bars}"},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"腾讯分钟接口错误：{payload.get('msg')}")
        symbol_data = payload.get("data", {}).get(provider_symbol, {})
        raw_bars = symbol_data.get(f"m{request.frequency_minutes}")
        if not raw_bars:
            raise RuntimeError("腾讯分钟接口返回空数据")
        records: list[dict] = []
        for raw in raw_bars:
            if len(raw) < 8:
                raise RuntimeError("腾讯分钟接口K线字段数量不足")
            records.append(
                {
                    "bar_end": pd.to_datetime(str(raw[0]), format="%Y%m%d%H%M", errors="raise"),
                    "open": float(raw[1]),
                    "close": float(raw[2]),
                    "high": float(raw[3]),
                    "low": float(raw[4]),
                    "volume": float(raw[5]) * 100.0,
                    "amount": float("nan"),
                    "turnover_field_raw": float(raw[7]),
                }
            )
        output = pd.DataFrame.from_records(records).sort_values("bar_end")
        output = output.drop_duplicates("bar_end", keep="last").reset_index(drop=True)
        output.insert(1, "bar_start", output["bar_end"] - pd.Timedelta(minutes=request.frequency_minutes))
        output.insert(2, "trade_date", output["bar_end"].dt.normalize())
        output["symbol"] = request.symbol
        output["frequency_minutes"] = request.frequency_minutes
        output["timestamp_meaning"] = "bar_end"
        output["adjustment"] = request.adjustment
        output["source"] = self.provider_name
        output["timezone"] = request.timezone
        output["volume_unit"] = "share"
        output["source_volume_unit"] = "lot_100_shares"
        output["amount_unit"] = "UNAVAILABLE"
        output["retrieved_at"] = datetime.now(ZoneInfo(request.timezone)).isoformat()
        return annotate_session_phase(output)


class TencentIntradaySnapshotProvider:
    """腾讯当日分时累计快照。

    该接口在新制度下可返回15:00之后的盘后固定价格交易累计值，但只提供
    当前交易日，不是历史回补接口。落盘后按时间戳滚动积累，可从接入日起
    建立盘后成交历史。
    """

    provider_name = "tencent.ifzq.minute.query"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, request: MinuteDataRequest) -> pd.DataFrame:
        request.validate()
        if request.frequency_minutes != 1:
            raise ValueError("腾讯当日分时累计接口仅按1分钟快照接入")
        provider_symbol = _tencent_symbol(request.symbol)
        response = self._session.get(
            TENCENT_INTRADAY_URL,
            params={"code": provider_symbol},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"腾讯当日分时接口错误：{payload.get('msg')}")
        symbol_data = payload.get("data", {}).get(provider_symbol, {}).get("data", {})
        trade_date_text = str(symbol_data.get("date", ""))
        raw_snapshots = symbol_data.get("data")
        if len(trade_date_text) != 8 or not raw_snapshots:
            raise RuntimeError("腾讯当日分时接口返回空数据或无效交易日")

        records: list[dict] = []
        for raw in raw_snapshots:
            fields = str(raw).split()
            if len(fields) != 4:
                raise RuntimeError("腾讯当日分时接口字段数量不符合预期")
            timestamp = pd.to_datetime(
                f"{trade_date_text}{fields[0]}", format="%Y%m%d%H%M", errors="raise"
            )
            records.append(
                {
                    "timestamp": timestamp,
                    "trade_date": timestamp.normalize(),
                    "price": float(fields[1]),
                    "cumulative_volume": float(fields[2]) * 100.0,
                    "cumulative_amount": float(fields[3]),
                }
            )
        output = pd.DataFrame.from_records(records).sort_values("timestamp")
        output = output.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        volume_diff = output["cumulative_volume"].diff()
        amount_diff = output["cumulative_amount"].diff()
        output["incremental_volume"] = volume_diff.fillna(output["cumulative_volume"])
        output["incremental_amount"] = amount_diff.fillna(output["cumulative_amount"])
        if (output[["incremental_volume", "incremental_amount"]] < 0).any().any():
            raise RuntimeError("腾讯当日分时累计值出现倒退，无法生成增量")

        minute_text = output["timestamp"].dt.strftime("%H:%M:%S")
        output["session_phase"] = "continuous"
        output.loc[minute_text <= "09:30:00", "session_phase"] = "opening_auction"
        output.loc[
            (minute_text > "09:30:00") & (minute_text <= "11:30:00"), "session_phase"
        ] = "continuous_am"
        output.loc[
            (minute_text >= "13:00:00") & (minute_text < "14:57:00"), "session_phase"
        ] = "continuous_pm"
        output.loc[
            (minute_text >= "14:57:00") & (minute_text <= "15:00:00"), "session_phase"
        ] = "closing_auction"
        output.loc[minute_text > "15:00:00", "session_phase"] = "post_close_fixed_price"
        output["post_close_included"] = minute_text > "15:00:00"
        output["post_close_separately_identifiable"] = minute_text > "15:00:00"
        output["symbol"] = request.symbol
        output["frequency_minutes"] = 1
        output["timestamp_meaning"] = "cumulative_snapshot_at_minute"
        output["source"] = self.provider_name
        output["timezone"] = request.timezone
        output["volume_unit"] = "share"
        output["source_volume_unit"] = "lot_100_shares"
        output["amount_unit"] = "CNY"
        output["retrieved_at"] = datetime.now(ZoneInfo(request.timezone)).isoformat()
        return output


def aggregate_one_minute_to_15m(
    one_minute: pd.DataFrame,
    expected_bar_end_times: list[str],
) -> pd.DataFrame:
    """按A股上午、下午会话边界把1分钟记录聚合成15分钟K线。

    时间标签使用bar_end语义。最后一根会自然纳入14:46—14:57以及15:00
    的收盘集合竞价记录；不会把午休或15:00后的盘后交易混入。
    """

    required = {"bar_end", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(one_minute.columns)
    if missing:
        raise ValueError(f"1分钟聚合缺少字段：{sorted(missing)}")
    data = one_minute.copy()
    data["bar_end"] = pd.to_datetime(data["bar_end"], errors="raise")
    data["trade_date"] = data["bar_end"].dt.normalize()
    output: list[dict] = []
    for trade_date, day in data.groupby("trade_date", sort=True):
        morning_left = pd.Timestamp(trade_date) + pd.Timedelta(hours=9, minutes=30)
        afternoon_left = pd.Timestamp(trade_date) + pd.Timedelta(hours=13)
        for time_text in expected_bar_end_times:
            hour, minute, second = (int(part) for part in time_text.split(":"))
            right = pd.Timestamp(trade_date) + pd.Timedelta(
                hours=hour, minutes=minute, seconds=second
            )
            left = morning_left if right.hour < 12 else afternoon_left
            window = day.loc[(day["bar_end"] > left) & (day["bar_end"] <= right)]
            if not window.empty:
                output.append(
                    {
                        "bar_end": right,
                        "trade_date": pd.Timestamp(trade_date),
                        "open": float(window.iloc[0]["open"]),
                        "high": float(window["high"].max()),
                        "low": float(window["low"].min()),
                        "close": float(window.iloc[-1]["close"]),
                        "volume": float(window["volume"].sum()),
                        "amount": float(window["amount"].sum()),
                        "source_minute_count": int(len(window)),
                    }
                )
            if right.hour < 12:
                morning_left = right
            else:
                afternoon_left = right
    return pd.DataFrame.from_records(output).sort_values("bar_end").reset_index(drop=True)
