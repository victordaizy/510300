"""上交所官方端点的严格 TLS 路由 V1.3。

环境代理仅在证书验证失败时允许回退一次严格直连；任何路线都禁止关闭
TLS 证书验证。该模块不负责重试业务请求、保存响应或改变研究日期。
"""

from __future__ import annotations

import ssl
from typing import Any, Mapping
from urllib.parse import urlparse

import requests


TRANSPORT_EVIDENCE_SCHEMA = "SSE_STRICT_TRANSPORT_EVIDENCE_V1"
ALLOWED_HTTPS_ENDPOINTS = {
    ("query.sse.com.cn", 443),
    ("yunhq.sse.com.cn", 32042),
}


class StrictSseRouteError(requests.exceptions.SSLError):
    """环境路线证书失败且严格直连也失败。"""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _endpoint(url: str) -> tuple[str, int]:
    parsed = urlparse(str(url))
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError(f"上交所HTTPS白名单端点端口无效：{url}") from exc
    endpoint = (host, port)
    if parsed.scheme.lower() != "https" or endpoint not in ALLOWED_HTTPS_ENDPOINTS:
        raise ValueError(f"请求不在上交所HTTPS白名单中：{url}")
    return endpoint


def _certificate_verification_failed(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        message = str(current).lower()
        if "certificate_verify_failed" in message or "certificate verify failed" in message:
            return True
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
        for argument in getattr(current, "args", ()):
            if isinstance(argument, BaseException):
                pending.append(argument)
    return False


def _base_evidence(host: str, port: int) -> dict[str, Any]:
    return {
        "schema_version": TRANSPORT_EVIDENCE_SCHEMA,
        "host": host,
        "port": port,
        "route": "NOT_ATTEMPTED",
        "tls_verification_required": True,
        "environment_proxy_attempted": True,
        "direct_fallback_used": False,
        "primary_failure_type": None,
        "primary_failure_category": None,
    }


class StrictSseSession:
    """requests 兼容会话，严格限定上交所端点和证书失败回退。"""

    def __init__(
        self,
        *,
        primary_session: Any | None = None,
        direct_session: Any | None = None,
    ) -> None:
        self._primary_session = primary_session or requests.Session()
        self._direct_session = direct_session or requests.Session()
        self._primary_session.trust_env = True
        self._direct_session.trust_env = False
        self.last_evidence: dict[str, Any] | None = None

    def get(self, url: str, **kwargs: Any) -> Any:
        host, port = _endpoint(url)
        if "verify" in kwargs and kwargs["verify"] is not True:
            raise ValueError("禁止关闭TLS证书验证或替换冻结验证模式")
        request_kwargs = dict(kwargs)
        request_kwargs["verify"] = True
        evidence = _base_evidence(host, port)
        self.last_evidence = evidence
        try:
            response = self._primary_session.get(url, **request_kwargs)
        except requests.exceptions.SSLError as primary_error:
            evidence["primary_failure_type"] = type(primary_error).__name__
            if not _certificate_verification_failed(primary_error):
                evidence["route"] = "FAILED_ENVIRONMENT_ROUTE_NO_FALLBACK"
                evidence["primary_failure_category"] = "NON_CERTIFICATE_SSL_FAILURE"
                raise
            evidence["primary_failure_category"] = "CERTIFICATE_VERIFICATION_FAILED"
            evidence["direct_fallback_used"] = True
            try:
                response = self._direct_session.get(url, **request_kwargs)
            except Exception as direct_error:
                evidence.update(
                    {
                        "route": "FAILED_BOTH_ENVIRONMENT_AND_STRICT_DIRECT",
                        "direct_failure_type": type(direct_error).__name__,
                    }
                )
                raise StrictSseRouteError(
                    "环境路线证书验证失败，严格直连仍失败："
                    f"{type(direct_error).__name__}: {direct_error}",
                    evidence,
                ) from direct_error
            evidence["route"] = "STRICT_DIRECT_TLS_VERIFIED_AFTER_PROXY_CERT_FAILURE"
            return response
        except Exception as primary_error:
            evidence.update(
                {
                    "route": "FAILED_ENVIRONMENT_ROUTE_NO_FALLBACK",
                    "primary_failure_type": type(primary_error).__name__,
                    "primary_failure_category": "NON_TLS_FAILURE",
                }
            )
            raise
        evidence["route"] = "ENVIRONMENT_ROUTE_TLS_VERIFIED"
        return response

    def close(self) -> None:
        self._primary_session.close()
        if self._direct_session is not self._primary_session:
            self._direct_session.close()

    def __enter__(self) -> "StrictSseSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def attach_transport_evidence(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把路线证据扁平写入标准化记录，不污染官方原始载荷。"""

    if not evidence:
        raise RuntimeError("上交所响应缺少严格传输路线证据")
    route = str(evidence.get("route") or "")
    if route not in {
        "ENVIRONMENT_ROUTE_TLS_VERIFIED",
        "STRICT_DIRECT_TLS_VERIFIED_AFTER_PROXY_CERT_FAILURE",
    }:
        raise RuntimeError(f"上交所响应的严格传输路线未通过：{route}")
    enriched = dict(record)
    enriched.update(
        {
            "transport_evidence_schema": str(evidence.get("schema_version") or ""),
            "transport_route": route,
            "transport_direct_fallback_used": bool(evidence.get("direct_fallback_used")),
            "transport_tls_verification_required": bool(
                evidence.get("tls_verification_required")
            ),
            "transport_primary_failure_category": evidence.get(
                "primary_failure_category"
            ),
        }
    )
    if enriched["transport_evidence_schema"] != TRANSPORT_EVIDENCE_SCHEMA:
        raise RuntimeError("严格传输证据版本不一致")
    if enriched["transport_tls_verification_required"] is not True:
        raise RuntimeError("严格传输证据未确认TLS验证")
    return enriched
