"""ETF一级市场清单与盘中IOPV前向数据接入。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests


SSE_PCF_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_SNAPSHOT_URL = "https://yunhq.sse.com.cn:32042/v1/sh1/snap/{fund_code}"
SSE_REFERER = "https://etf.sse.com.cn/"
PCF_SQL_ID = "COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_JBXX_C"
IOPV_SELECT_FIELDS = [
    "name",
    "last",
    "chg_rate",
    "change",
    "open",
    "prev_close",
    "high",
    "low",
    "volume",
    "amount",
    "tradephase",
    "cpxxextendname",
    "iopv",
    "fp_volume",
    "fp_amount",
    "fp_phase",
    "cpxxsubtype",
]


@dataclass(frozen=True)
class EtfPrimaryMarketRequest:
    """ETF一级市场与行情快照请求。"""

    fund_code: str
    timeout_seconds: float = 15.0
    timezone: str = "Asia/Shanghai"

    def validate(self) -> None:
        if len(self.fund_code) != 6 or not self.fund_code.isdigit():
            raise ValueError(f"无法识别ETF代码：{self.fund_code}")
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须为正数")
        ZoneInfo(self.timezone)


@dataclass(frozen=True)
class ProviderSnapshot:
    """标准化记录及可审计原始载荷。"""

    record: dict[str, Any]
    raw_payload: dict[str, Any]


def _headers() -> dict[str, str]:
    return {
        "Referer": SSE_REFERER,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodexResearch/1.0",
    }


def _parse_jsonp(text: str, callback: str) -> dict[str, Any]:
    prefix = f"{callback}("
    value = text.strip()
    if not value.startswith(prefix) or not value.endswith(")"):
        raise RuntimeError("上交所PCF接口返回内容不是预期JSONP格式")
    payload = json.loads(value[len(prefix) : -1])
    if not isinstance(payload, dict):
        raise RuntimeError("上交所PCF接口载荷不是对象")
    return payload


def _nullable_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("￥", "").replace("¥", "")
    if text in {"", "-", "--", "None", "null"}:
        return None
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _parse_exchange_timestamp(date_value: Any, time_value: Any) -> datetime:
    date_text = str(date_value)
    time_text = str(time_value).zfill(6)
    if len(date_text) != 8 or len(time_text) != 6:
        raise RuntimeError("上交所行情快照日期或时间无效")
    return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")


class SsePcfProvider:
    """上交所ETF申购赎回清单基本信息。"""

    provider_name = "sse.COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_JBXX_C"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, request: EtfPrimaryMarketRequest) -> ProviderSnapshot:
        request.validate()
        callback = "jsonpCallback"
        response = self._session.get(
            SSE_PCF_URL,
            params={
                "isPagination": "false",
                "sqlId": PCF_SQL_ID,
                "FUNDID2": request.fund_code,
                "jsonCallBack": callback,
            },
            headers=_headers(),
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = _parse_jsonp(response.text, callback)
        results = payload.get("result")
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("上交所PCF接口未返回唯一基金记录")
        source = results[0]
        if str(source.get("TRADE_CODE")) != request.fund_code:
            raise RuntimeError("上交所PCF接口返回了其他基金")

        retrieved_at = datetime.now(ZoneInfo(request.timezone))
        record = {
            "trading_day": datetime.strptime(str(source["TRADING_DAY"]), "%Y%m%d").date(),
            "previous_trading_day": datetime.strptime(
                str(source["PRE_TRADING_DAY"]), "%Y%m%d"
            ).date(),
            "fund_code": request.fund_code,
            "fund_name": str(source.get("FUND_NAME", "")),
            "fund_company": str(source.get("FUND_COMP_NAME", "")),
            "file_id": str(source.get("FILE_ID", "")),
            "etf_type": str(source.get("ETF_TYPE", "")),
            "nav": _nullable_number(source.get("NAV")),
            "nav_per_creation_unit": _nullable_number(source.get("NAVPERCU")),
            "previous_cash_component": _nullable_number(source.get("PRE_CASH_COMPONENT")),
            "estimated_cash_component": _nullable_number(source.get("ESTIMATED_CASH_COMPONENT")),
            "maximum_cash_ratio": _nullable_number(source.get("MAX_CASH_RATIO")),
            "creation_redemption_unit": int(float(source["CREATION_REDEMPTION_UNIT"])),
            "component_count": int(float(source["RECORD_NUM"])),
            "publish_iopv": str(source.get("PUBLISH_IOPV", "")).strip() == "是",
            "creation_redemption_status": str(source.get("CREATION_REDEMPTION", "")),
            "creation_redemption_mechanism": str(
                source.get("CREATION_REDEMPTION_MECHANISM", "")
            ),
            "creation_limit": _nullable_number(source.get("CREATION_LIMIT")),
            "redemption_limit": _nullable_number(source.get("REDEMPTION_LIMIT")),
            "net_creation_limit": _nullable_number(source.get("NET_CREATION_LIMIT")),
            "net_redemption_limit": _nullable_number(source.get("NET_REDEMPTION_LIMIT")),
            "creation_limit_per_account": _nullable_number(
                source.get("CREATION_LIMIT_PER_ACCT")
            ),
            "redemption_limit_per_account": _nullable_number(
                source.get("REDEMPTION_LIMIT_PER_ACCT")
            ),
            "net_creation_limit_per_account": _nullable_number(
                source.get("NET_CREATION_LIMIT_PER_ACCT")
            ),
            "net_redemption_limit_per_account": _nullable_number(
                source.get("NET_REDEMPTION_LIMIT_PER_ACCT")
            ),
            "source": self.provider_name,
            "retrieved_at": retrieved_at.isoformat(),
        }
        if record["nav"] is None or record["nav"] <= 0:
            raise RuntimeError("PCF单位净值必须为正数")
        if record["creation_redemption_unit"] <= 0:
            raise RuntimeError("PCF最小申购赎回单位必须为正数")
        if record["nav_per_creation_unit"] is not None:
            implied = record["nav"] * record["creation_redemption_unit"]
            relative_error = abs(record["nav_per_creation_unit"] / implied - 1.0)
            if relative_error > 0.01:
                raise RuntimeError("PCF最小申购赎回单位净值与单位净值不一致")
        return ProviderSnapshot(record=record, raw_payload=payload)


class SseIopvSnapshotProvider:
    """上交所官方行情快照中的ETF市价与IOPV。"""

    provider_name = "sse.yunhq.sh1.snap"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, request: EtfPrimaryMarketRequest) -> ProviderSnapshot:
        request.validate()
        response = self._session.get(
            SSE_SNAPSHOT_URL.format(fund_code=request.fund_code),
            params={"select": ",".join(IOPV_SELECT_FIELDS)},
            headers=_headers(),
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != request.fund_code:
            raise RuntimeError("上交所行情快照返回了其他基金")
        values = payload.get("snap")
        if not isinstance(values, list) or len(values) != len(IOPV_SELECT_FIELDS):
            raise RuntimeError("上交所行情快照字段数量不符合预期")
        source = dict(zip(IOPV_SELECT_FIELDS, values, strict=True))
        exchange_timestamp = _parse_exchange_timestamp(payload.get("date"), payload.get("time"))
        retrieved_at = datetime.now(ZoneInfo(request.timezone))
        last_price = _nullable_number(source["last"])
        iopv = _nullable_number(source["iopv"])
        if last_price is None or last_price <= 0:
            raise RuntimeError("上交所行情快照最新价必须为正数")
        if iopv is None or iopv <= 0:
            raise RuntimeError("上交所行情快照IOPV必须为正数")
        record = {
            "exchange_timestamp": exchange_timestamp,
            "trade_date": exchange_timestamp.date(),
            "fund_code": request.fund_code,
            "name": str(source["name"]),
            "extended_name": str(source["cpxxextendname"]),
            "last_price": last_price,
            "iopv": iopv,
            "premium_discount_bps": (last_price / iopv - 1.0) * 10_000.0,
            "change_rate_pct": _nullable_number(source["chg_rate"]),
            "change_cny": _nullable_number(source["change"]),
            "open": _nullable_number(source["open"]),
            "previous_close": _nullable_number(source["prev_close"]),
            "high": _nullable_number(source["high"]),
            "low": _nullable_number(source["low"]),
            "volume_shares": _nullable_number(source["volume"]),
            "amount_cny": _nullable_number(source["amount"]),
            "trade_phase_raw": str(source["tradephase"]),
            "post_close_volume_shares": _nullable_number(source["fp_volume"]),
            "post_close_amount_cny": _nullable_number(source["fp_amount"]),
            "post_close_phase_raw": str(source["fp_phase"]),
            "product_subtype": str(source["cpxxsubtype"]),
            "source": self.provider_name,
            "retrieved_at": retrieved_at.isoformat(),
        }
        return ProviderSnapshot(record=record, raw_payload=payload)
