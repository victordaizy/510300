"""上交所 ETF 成交概况日端点，用于盘中快照的第二公开端点复核。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from market_data.etf_primary_market import EtfPrimaryMarketRequest, ProviderSnapshot


SSE_DAILY_TURNOVER_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_DAILY_TURNOVER_SQL_ID = "COMMON_SSE_CP_GPJCTPZ_GPLB_CJGK_MRGK_C"
SSE_DAILY_TURNOVER_REFERER = (
    "https://www.sse.com.cn/assortment/fund/list/etfinfo/turnover/"
    "index.shtml?FUNDID={fund_code}"
)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        result = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"上交所 ETF 成交概况字段无效：{field}") from exc
    if positive and result <= 0:
        raise RuntimeError(f"上交所 ETF 成交概况字段必须为正数：{field}")
    return result


class SseEtfDailyCrosscheckProvider:
    """读取上交所公开成交概况的单日 OHLC，复核官方盘中行情快照。"""

    provider_name = "sse.COMMON_SSE_CP_GPJCTPZ_GPLB_CJGK_MRGK_C"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(
        self,
        request: EtfPrimaryMarketRequest,
        market_date: date,
    ) -> ProviderSnapshot:
        request.validate()
        parameters = {
            "sqlId": SSE_DAILY_TURNOVER_SQL_ID,
            "SEC_CODE": request.fund_code,
            "TX_DATE": market_date.isoformat(),
        }
        response = self._session.get(
            SSE_DAILY_TURNOVER_URL,
            params=parameters,
            headers={
                "Referer": SSE_DAILY_TURNOVER_REFERER.format(
                    fund_code=request.fund_code
                ),
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "CodexResearch/1.0"
                ),
            },
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("上交所 ETF 成交概况未返回唯一日记录")
        source = results[0]
        if not isinstance(source, dict):
            raise RuntimeError("上交所 ETF 成交概况记录不是对象")
        if str(source.get("SEC_CODE")) != request.fund_code:
            raise RuntimeError("上交所 ETF 成交概况返回了其他基金")
        source_date = datetime.strptime(str(source.get("TX_DATE")), "%Y%m%d").date()
        if source_date != market_date:
            raise RuntimeError("上交所 ETF 成交概况日期与请求日期不一致")
        open_price = _number(source.get("OPEN_PRICE"), "OPEN_PRICE", positive=True)
        high_price = _number(source.get("HIGH_PRICE"), "HIGH_PRICE", positive=True)
        low_price = _number(source.get("LOW_PRICE"), "LOW_PRICE", positive=True)
        close_price = _number(source.get("CLOSE_PRICE"), "CLOSE_PRICE", positive=True)
        if high_price < max(open_price, low_price, close_price):
            raise RuntimeError("上交所 ETF 成交概况最高价关系无效")
        if low_price > min(open_price, high_price, close_price):
            raise RuntimeError("上交所 ETF 成交概况最低价关系无效")
        retrieved_at = datetime.now(ZoneInfo(request.timezone))
        record = {
            "trade_date": market_date,
            "fund_code": request.fund_code,
            "name": str(source.get("SEC_NAME", "")),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume_quoted": _number(source.get("TRADE_VOL"), "TRADE_VOL"),
            "amount_quoted": _number(source.get("TRADE_AMT"), "TRADE_AMT"),
            "source": self.provider_name,
            "retrieved_at": retrieved_at.isoformat(),
        }
        raw_payload = {
            "request": {
                "url": SSE_DAILY_TURNOVER_URL,
                "parameters": parameters,
                "referer": SSE_DAILY_TURNOVER_REFERER.format(
                    fund_code=request.fund_code
                ),
            },
            "response": payload,
        }
        return ProviderSnapshot(record=record, raw_payload=raw_payload)
