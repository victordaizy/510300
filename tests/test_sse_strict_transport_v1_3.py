from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from market_data import etf_primary_market_strict_v1_3 as strict_providers
from market_data.etf_primary_market import (
    EtfPrimaryMarketRequest,
    ProviderSnapshot,
)
from market_data.sse_strict_transport_v1_3 import (
    StrictSseRouteError,
    StrictSseSession,
    attach_transport_evidence,
)
from scripts.collect_510300_primary_market_v1_3 import validate_transport_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeResponse:
    status_code: int = 200


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.trust_env = True
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def certificate_error() -> requests.exceptions.SSLError:
    return requests.exceptions.SSLError(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate"
    )


def test_certificate_failure_uses_one_strict_direct_fallback() -> None:
    primary = FakeSession([certificate_error()])
    direct = FakeSession([FakeResponse()])
    session = StrictSseSession(primary_session=primary, direct_session=direct)

    response = session.get(
        "https://query.sse.com.cn/commonQuery.do",
        params={"FUNDID2": "510300"},
        timeout=15,
    )

    assert response.status_code == 200
    assert len(primary.calls) == 1
    assert len(direct.calls) == 1
    assert primary.calls[0]["verify"] is True
    assert direct.calls[0]["verify"] is True
    assert direct.trust_env is False
    assert session.last_evidence == {
        "schema_version": "SSE_STRICT_TRANSPORT_EVIDENCE_V1",
        "host": "query.sse.com.cn",
        "port": 443,
        "route": "STRICT_DIRECT_TLS_VERIFIED_AFTER_PROXY_CERT_FAILURE",
        "tls_verification_required": True,
        "environment_proxy_attempted": True,
        "direct_fallback_used": True,
        "primary_failure_type": "SSLError",
        "primary_failure_category": "CERTIFICATE_VERIFICATION_FAILED",
    }


def test_successful_proxy_route_does_not_call_direct() -> None:
    primary = FakeSession([FakeResponse()])
    direct = FakeSession([FakeResponse()])
    session = StrictSseSession(primary_session=primary, direct_session=direct)

    session.get("https://yunhq.sse.com.cn:32042/v1/sh1/snap/510300", timeout=15)

    assert len(primary.calls) == 1
    assert direct.calls == []
    assert session.last_evidence["route"] == "ENVIRONMENT_ROUTE_TLS_VERIFIED"
    assert session.last_evidence["direct_fallback_used"] is False


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ConnectionError("连接重置"),
        requests.exceptions.SSLError("WRONG_VERSION_NUMBER"),
    ],
)
def test_non_certificate_failure_never_bypasses_environment_route(error: Exception) -> None:
    primary = FakeSession([error])
    direct = FakeSession([FakeResponse()])
    session = StrictSseSession(primary_session=primary, direct_session=direct)

    with pytest.raises(type(error)):
        session.get("https://query.sse.com.cn/commonQuery.do", timeout=15)

    assert direct.calls == []


def test_direct_failure_preserves_both_failure_categories() -> None:
    primary = FakeSession([certificate_error()])
    direct = FakeSession([requests.exceptions.ConnectTimeout("直连超时")])
    session = StrictSseSession(primary_session=primary, direct_session=direct)

    with pytest.raises(StrictSseRouteError, match="严格直连仍失败") as captured:
        session.get("https://query.sse.com.cn/commonQuery.do", timeout=15)

    assert captured.value.evidence["primary_failure_category"] == "CERTIFICATE_VERIFICATION_FAILED"
    assert captured.value.evidence["direct_failure_type"] == "ConnectTimeout"
    assert captured.value.evidence["route"] == "FAILED_BOTH_ENVIRONMENT_AND_STRICT_DIRECT"


def test_transport_rejects_non_allowlisted_or_insecure_endpoint() -> None:
    session = StrictSseSession(
        primary_session=FakeSession([FakeResponse()]),
        direct_session=FakeSession([FakeResponse()]),
    )
    with pytest.raises(ValueError, match="HTTPS白名单"):
        session.get("https://example.com/data")
    with pytest.raises(ValueError, match="HTTPS白名单"):
        session.get("http://query.sse.com.cn/commonQuery.do")


def test_transport_forbids_disabling_tls_verification() -> None:
    session = StrictSseSession(
        primary_session=FakeSession([FakeResponse()]),
        direct_session=FakeSession([FakeResponse()]),
    )
    with pytest.raises(ValueError, match="禁止关闭TLS证书验证"):
        session.get("https://query.sse.com.cn/commonQuery.do", verify=False)


def test_transport_evidence_is_flattened_into_normalized_record_only() -> None:
    record = {"fund_code": "510300"}
    raw_payload = {"result": [{"TRADE_CODE": "510300"}]}
    evidence = {
        "schema_version": "SSE_STRICT_TRANSPORT_EVIDENCE_V1",
        "route": "STRICT_DIRECT_TLS_VERIFIED_AFTER_PROXY_CERT_FAILURE",
        "direct_fallback_used": True,
        "tls_verification_required": True,
        "primary_failure_category": "CERTIFICATE_VERIFICATION_FAILED",
    }

    enriched = attach_transport_evidence(record, evidence)

    assert enriched["transport_route"] == evidence["route"]
    assert enriched["transport_direct_fallback_used"] is True
    assert enriched["transport_tls_verification_required"] is True
    assert enriched["transport_primary_failure_category"] == "CERTIFICATE_VERIFICATION_FAILED"
    assert raw_payload == {"result": [{"TRADE_CODE": "510300"}]}


class FakeRoutedSession:
    def __init__(self) -> None:
        self.last_evidence = {
            "schema_version": "SSE_STRICT_TRANSPORT_EVIDENCE_V1",
            "route": "ENVIRONMENT_ROUTE_TLS_VERIFIED",
            "direct_fallback_used": False,
            "tls_verification_required": True,
            "primary_failure_category": None,
        }


class FakeDailyResponse:
    def __init__(self, close_price: str) -> None:
        self.payload = {
            "result": [
                {
                    "SEC_CODE": "510300",
                    "SEC_NAME": "300ETF",
                    "TX_DATE": "20260828",
                    "OPEN_PRICE": "4.684",
                    "HIGH_PRICE": "4.710",
                    "LOW_PRICE": "4.680",
                    "CLOSE_PRICE": close_price,
                    "TRADE_VOL": "60515.2",
                    "TRADE_AMT": "283726.69",
                }
            ]
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeDailySession(FakeRoutedSession):
    def __init__(self, close_price: str) -> None:
        super().__init__()
        self.response = FakeDailyResponse(close_price)

    def get(self, url: str, **kwargs: Any) -> FakeDailyResponse:
        return self.response


class FakeProvider:
    raw_payload = {"official": {"value": 1}}

    def __init__(self, session: Any) -> None:
        self.session = session

    def fetch(self, *args: Any) -> ProviderSnapshot:
        return ProviderSnapshot(
            record={"fund_code": "510300", "source": "sse.fake"},
            raw_payload=self.raw_payload,
        )


@pytest.mark.parametrize(
    ("wrapper_name", "base_name", "arguments"),
    [
        (
            "SsePcfProviderV1_3",
            "SsePcfProvider",
            (EtfPrimaryMarketRequest(fund_code="510300"),),
        ),
        (
            "SseIopvSnapshotProviderV1_3",
            "SseIopvSnapshotProvider",
            (EtfPrimaryMarketRequest(fund_code="510300"),),
        ),
    ],
)
def test_provider_wrappers_add_route_evidence_without_mutating_raw_payload(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    base_name: str,
    arguments: tuple[Any, ...],
) -> None:
    session = FakeRoutedSession()
    monkeypatch.setattr(strict_providers, base_name, FakeProvider)
    wrapper = getattr(strict_providers, wrapper_name)(session=session)

    snapshot = wrapper.fetch(*arguments)

    assert snapshot.record["transport_route"] == "ENVIRONMENT_ROUTE_TLS_VERIFIED"
    assert snapshot.record["transport_tls_verification_required"] is True
    assert snapshot.raw_payload is FakeProvider.raw_payload
    assert snapshot.raw_payload == {"official": {"value": 1}}


def test_v1_3_config_changes_transport_only_and_continues_forward_dataset() -> None:
    v1_2 = yaml.safe_load(
        (PROJECT_ROOT / "config/primary_market_forward_v1_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    v1_3 = yaml.safe_load(
        (PROJECT_ROOT / "config/primary_market_forward_v1_3.yaml").read_text(
            encoding="utf-8"
        )
    )

    validate_transport_contract(v1_3)
    assert v1_3["quality"] == v1_2["quality"]
    assert v1_3["readiness"] == v1_2["readiness"]
    assert v1_3["outputs"]["pcf_daily_file"] == v1_2["outputs"]["pcf_daily_file"]
    assert v1_3["outputs"]["iopv_snapshot_file"] == v1_2["outputs"]["iopv_snapshot_file"]
    assert v1_3["outputs"]["daily_crosscheck_file"] == v1_2["outputs"]["daily_crosscheck_file"]
    for source_name in ("pcf", "iopv", "daily_crosscheck"):
        old_source = v1_2["source_contract"]["sources"][source_name]
        new_source = v1_3["source_contract"]["sources"][source_name]
        assert new_source["source_id"] == old_source["source_id"]
        assert new_source["source_url_or_endpoint"] == old_source["source_url_or_endpoint"]
        assert new_source["access_cost_cny"] == 0


def test_v1_3_config_rejects_any_insecure_tls_permission() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config/primary_market_forward_v1_3.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["transport_contract"]["insecure_tls_allowed"] = True

    with pytest.raises(ValueError, match="insecure_tls_allowed"):
        validate_transport_contract(config)


def test_v1_3_daily_parser_honors_existing_five_mill_tolerance() -> None:
    provider = strict_providers.SseEtfDailyCrosscheckProviderV1_3(
        session=FakeDailySession("4.679")
    )

    snapshot = provider.fetch(
        EtfPrimaryMarketRequest(fund_code="510300"),
        date(2026, 8, 28),
    )

    assert snapshot.record["close"] == 4.679
    assert snapshot.record["ohlc_relation_tolerance_cny"] == 0.005
    assert snapshot.record["transport_tls_verification_required"] is True


def test_v1_3_daily_parser_rejects_relation_gap_beyond_five_mill() -> None:
    provider = strict_providers.SseEtfDailyCrosscheckProviderV1_3(
        session=FakeDailySession("4.674")
    )

    with pytest.raises(RuntimeError, match="超过冻结容差"):
        provider.fetch(
            EtfPrimaryMarketRequest(fund_code="510300"),
            date(2026, 8, 28),
        )
