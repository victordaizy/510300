"""带严格 TLS 路由证据的上交所 PCF、IOPV 与日端点提供器 V1.3。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from market_data.etf_daily_crosscheck_v1_2 import (
    SSE_DAILY_TURNOVER_REFERER,
    SSE_DAILY_TURNOVER_SQL_ID,
    SSE_DAILY_TURNOVER_URL,
    SseEtfDailyCrosscheckProvider,
    _number,
)
from market_data.etf_primary_market import (
    EtfPrimaryMarketRequest,
    ProviderSnapshot,
    SseIopvSnapshotProvider,
    SsePcfProvider,
)
from market_data.sse_strict_transport_v1_3 import (
    StrictSseSession,
    attach_transport_evidence,
)


def _enrich(snapshot: ProviderSnapshot, session: Any) -> ProviderSnapshot:
    return ProviderSnapshot(
        record=attach_transport_evidence(snapshot.record, session.last_evidence),
        raw_payload=snapshot.raw_payload,
    )


class SsePcfProviderV1_3:
    """PCF 提供器；官方原始载荷不变，标准化记录增加传输证据。"""

    provider_name = f"{SsePcfProvider.provider_name}.strict_route_v1_3"

    def __init__(self, session: Any | None = None) -> None:
        self._provided_session = session

    def fetch(self, request: EtfPrimaryMarketRequest) -> ProviderSnapshot:
        if self._provided_session is not None:
            snapshot = SsePcfProvider(session=self._provided_session).fetch(request)
            return _enrich(snapshot, self._provided_session)
        with StrictSseSession() as session:
            snapshot = SsePcfProvider(session=session).fetch(request)
            return _enrich(snapshot, session)


class SseIopvSnapshotProviderV1_3:
    """IOPV 提供器；官方原始载荷不变，标准化记录增加传输证据。"""

    provider_name = f"{SseIopvSnapshotProvider.provider_name}.strict_route_v1_3"

    def __init__(self, session: Any | None = None) -> None:
        self._provided_session = session

    def fetch(self, request: EtfPrimaryMarketRequest) -> ProviderSnapshot:
        if self._provided_session is not None:
            snapshot = SseIopvSnapshotProvider(session=self._provided_session).fetch(
                request
            )
            return _enrich(snapshot, self._provided_session)
        with StrictSseSession() as session:
            snapshot = SseIopvSnapshotProvider(session=session).fetch(request)
            return _enrich(snapshot, session)


class SseEtfDailyCrosscheckProviderV1_3:
    """官方日端点提供器；严格路由并执行冻结的五厘 OHLC 容差。"""

    provider_name = f"{SseEtfDailyCrosscheckProvider.provider_name}.strict_route_v1_3"
    ohlc_relation_tolerance_cny = 0.005

    def __init__(self, session: Any | None = None) -> None:
        self._provided_session = session

    def fetch(
        self,
        request: EtfPrimaryMarketRequest,
        market_date: date,
    ) -> ProviderSnapshot:
        if self._provided_session is not None:
            return self._fetch_with_session(
                request,
                market_date,
                self._provided_session,
            )
        with StrictSseSession() as session:
            return self._fetch_with_session(request, market_date, session)

    def _fetch_with_session(
        self,
        request: EtfPrimaryMarketRequest,
        market_date: date,
        session: Any,
    ) -> ProviderSnapshot:
        request.validate()
        parameters = {
            "sqlId": SSE_DAILY_TURNOVER_SQL_ID,
            "SEC_CODE": request.fund_code,
            "TX_DATE": market_date.isoformat(),
        }
        response = session.get(
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
        tolerance = self.ohlc_relation_tolerance_cny
        if high_price + tolerance < max(open_price, low_price, close_price):
            raise RuntimeError("上交所 ETF 成交概况最高价关系超过冻结容差")
        if low_price - tolerance > min(open_price, high_price, close_price):
            raise RuntimeError("上交所 ETF 成交概况最低价关系超过冻结容差")

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
            "ohlc_relation_tolerance_cny": tolerance,
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
        return _enrich(
            ProviderSnapshot(record=record, raw_payload=raw_payload),
            session,
        )
