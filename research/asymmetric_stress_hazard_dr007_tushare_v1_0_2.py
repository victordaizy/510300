"""510300 非对称压力风险 V1 的 DR007 精确来源规则。

本模块只定义 Tushare ``repo_daily`` 中 ``DR007.IB`` 的解析、授权传输
选择和日度序列核验。它不读取标签、未来收益、模型或组合结果，也不允许
把 FDR007、R007、FR007 或交易所回购序列替代 DR007。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import (
    OBSERVATION_CUTOFF,
    OBSERVATION_START,
    validate_dr007_daily,
)


SOURCE_VERSION = "1.0.2"
REMEDIATION_ID = "510300_ASYMMETRIC_STRESS_HAZARD_V1_SOURCE_REMEDIATION_V1_0_2"
API_NAME = "repo_daily"
TS_CODE = "DR007.IB"
REPO_MATURITY = "DR007"
VALUE_FIELD = "weight"
VALUE_SEMANTICS = "WEIGHTED_AVERAGE_RATE_PERCENT"
AVAILABILITY_RULE = "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE"
REQUIRED_PROVIDER_FIELDS = (
    "ts_code",
    "trade_date",
    "repo_maturity",
    "weight",
)
FORBIDDEN_SUBSTITUTES = frozenset(
    {"FDR007", "R007", "FR007", "EXCHANGE_REPO_R_007"}
)


class ProviderResponseError(RuntimeError):
    """Tushare 或兼容代理返回业务错误。"""

    def __init__(self, code: object, message: object) -> None:
        self.code = str(code)
        self.provider_message = str(message or "")
        super().__init__(f"Tushare 业务错误 code={self.code}：{self.provider_message}")


class CredentialExpiredError(RuntimeError):
    """项目登记的临时代理凭据已经过期。"""


@dataclass(frozen=True)
class Transport:
    """内存中的授权传输选择；``secret`` 不得写入任何产物。"""

    credential_kind: str
    secret: str
    endpoints: tuple[str, ...]
    expires_at: str | None
    request_auth_mode: str
    license_or_account_basis: str


def sha256_bytes(payload: bytes) -> str:
    """返回字节串的 SHA-256。"""

    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """计算稳定 JSON 载荷的 SHA-256。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def calendar_year_chunks(
    start: date = OBSERVATION_START,
    cutoff: date = OBSERVATION_CUTOFF,
) -> list[tuple[date, date]]:
    """按自然年切分冻结窗口，确保每次响应远低于 2000 行上限。"""

    if start > cutoff:
        raise ValueError("DR007 冻结窗口起点晚于截止日")
    chunks: list[tuple[date, date]] = []
    for year in range(start.year, cutoff.year + 1):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(cutoff, date(year, 12, 31))
        chunks.append((chunk_start, chunk_end))
    return chunks


def _parse_expiry(value: str | None) -> pd.Timestamp | None:
    if value is None or not value.strip():
        return None
    parsed = pd.Timestamp(value.strip())
    if parsed.tzinfo is None:
        raise ValueError("TUSHARE_PROXY_TOKEN_EXPIRES_AT 必须含时区")
    return parsed.tz_convert("Asia/Shanghai")


def _validate_https_endpoint(endpoint: str, allowed: set[str]) -> str:
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("DR007 授权传输端点必须是有效 HTTPS 地址")
    if normalized not in allowed:
        raise ValueError(f"DR007 授权传输端点不在冻结允许清单：{normalized}")
    return normalized


def resolve_transport(
    contract: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    now: pd.Timestamp | None = None,
    allow_expired_proxy_for_probe: bool = False,
) -> Transport:
    """优先选择标准账号，其次选择项目授权代理，并强制 HTTPS。"""

    transport_contract = contract["transport"]
    official_allowed = set(transport_contract["allowed_official_endpoints"])
    proxy_allowed = set(transport_contract["allowed_proxy_endpoints"])
    standard = (environ.get("TUSHARE_TOKEN") or environ.get("TS_TOKEN") or "").strip()
    proxy = (environ.get("TUSHARE_PROXY_TOKEN") or "").strip()
    if standard:
        configured = environ.get(
            "TUSHARE_API_URL",
            transport_contract["official_api_base"],
        )
        endpoint = _validate_https_endpoint(configured, official_allowed)
        return Transport(
            credential_kind="STANDARD_TUSHARE_ACCOUNT_TOKEN",
            secret=standard,
            endpoints=(endpoint,),
            expires_at=None,
            request_auth_mode="TOKEN_IN_JSON_NOT_ARCHIVED",
            license_or_account_basis="USER_AUTHORIZED_TUSHARE_ACCOUNT_MINIMUM_2000_POINTS",
        )
    if not proxy:
        raise RuntimeError(
            "未设置 TUSHARE_TOKEN、TS_TOKEN 或 TUSHARE_PROXY_TOKEN"
        )
    configured = environ.get(
        "TUSHARE_PROXY_URL",
        transport_contract["default_proxy_endpoint"],
    )
    endpoints = [_validate_https_endpoint(configured, proxy_allowed)]
    for candidate in transport_contract["allowed_proxy_endpoints"]:
        normalized = _validate_https_endpoint(candidate, proxy_allowed)
        if normalized not in endpoints:
            endpoints.append(normalized)
    expiry = _parse_expiry(environ.get("TUSHARE_PROXY_TOKEN_EXPIRES_AT"))
    current = now or pd.Timestamp.now(tz="Asia/Shanghai")
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Shanghai")
    else:
        current = current.tz_convert("Asia/Shanghai")
    if expiry is not None and expiry <= current and not allow_expired_proxy_for_probe:
        raise CredentialExpiredError(
            f"Tushare 代理凭据已于 {expiry.isoformat()} 过期"
        )
    return Transport(
        credential_kind="PROJECT_AUTHORIZED_TUSHARE_COMPATIBLE_PROXY_TOKEN",
        secret=proxy,
        endpoints=tuple(endpoints),
        expires_at=expiry.isoformat() if expiry is not None else None,
        request_auth_mode="X_API_KEY_HEADER_NOT_ARCHIVED",
        license_or_account_basis="PROJECT_AUTHORIZED_TUSHARE_COMPATIBLE_PROXY_SUBSCRIPTION",
    )


def public_transport_metadata(transport: Transport) -> dict[str, Any]:
    """返回可落盘的传输元数据，严格排除凭据。"""

    return {
        "credential_kind": transport.credential_kind,
        "endpoint_hosts": [urlparse(value).hostname for value in transport.endpoints],
        "expires_at": transport.expires_at,
        "request_auth_mode": transport.request_auth_mode,
        "license_or_account_basis": transport.license_or_account_basis,
        "credential_archived": False,
    }


def build_request(
    transport: Transport,
    chunk_start: date,
    chunk_end: date,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """构造一次精确 ``DR007.IB`` 请求；调用方不得归档返回的请求对象。"""

    params = {
        "ts_code": TS_CODE,
        "start_date": chunk_start.strftime("%Y%m%d"),
        "end_date": chunk_end.strftime("%Y%m%d"),
    }
    payload: dict[str, Any] = {
        "api_name": API_NAME,
        "params": params,
        "fields": ",".join(REQUIRED_PROVIDER_FIELDS),
    }
    headers: dict[str, str] = {}
    if transport.request_auth_mode == "TOKEN_IN_JSON_NOT_ARCHIVED":
        payload["token"] = transport.secret
        url = f"{transport.endpoints[0]}/{API_NAME}"
    elif transport.request_auth_mode == "X_API_KEY_HEADER_NOT_ARCHIVED":
        headers["x-api-key"] = transport.secret
        url = transport.endpoints[0]
    else:
        raise ValueError("未知的 DR007 授权请求认证模式")
    return url, payload, headers


def parse_repo_daily_payload(
    payload: bytes,
    *,
    chunk_start: date,
    chunk_end: date,
) -> pd.DataFrame:
    """解析并严格验证一个 ``repo_daily`` 年度分块响应。"""

    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Tushare repo_daily 响应不是有效 UTF-8 JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("Tushare repo_daily 顶层响应不是对象")
    if body.get("code") not in {0, "0"}:
        raise ProviderResponseError(body.get("code"), body.get("msg"))
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tushare repo_daily 响应缺少 data 对象")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not isinstance(items, list):
        raise ValueError("Tushare repo_daily 响应缺少 fields 或 items")
    missing = sorted(set(REQUIRED_PROVIDER_FIELDS).difference(fields))
    if missing:
        raise ValueError(f"Tushare repo_daily 响应缺少字段：{missing}")
    if not items:
        raise ValueError(
            f"Tushare DR007.IB 分块为空：{chunk_start} 至 {chunk_end}"
        )
    if any(not isinstance(row, list) or len(row) != len(fields) for row in items):
        raise ValueError("Tushare repo_daily 响应行宽与字段数不一致")
    frame = pd.DataFrame.from_records(items, columns=fields)
    frame = frame.loc[:, list(REQUIRED_PROVIDER_FIELDS)].copy()
    if not frame["ts_code"].astype(str).eq(TS_CODE).all():
        unexpected = sorted(frame.loc[frame["ts_code"] != TS_CODE, "ts_code"].unique())
        raise ValueError(f"Tushare 响应混入非 DR007.IB 代码：{unexpected}")
    if not frame["repo_maturity"].astype(str).eq(REPO_MATURITY).all():
        unexpected = sorted(
            frame.loc[
                frame["repo_maturity"] != REPO_MATURITY,
                "repo_maturity",
            ].unique()
        )
        raise ValueError(f"Tushare 响应混入非 DR007 期限：{unexpected}")
    frame["date"] = pd.to_datetime(
        frame["trade_date"], format="%Y%m%d", errors="raise"
    ).dt.normalize()
    frame["dr007"] = pd.to_numeric(frame[VALUE_FIELD], errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("Tushare DR007.IB 年度响应含重复日期")
    lower = pd.Timestamp(chunk_start)
    upper = pd.Timestamp(chunk_end)
    if not frame["date"].between(lower, upper).all():
        raise ValueError("Tushare DR007.IB 响应含请求窗口外日期")
    if ((frame["dr007"] <= 0) | (frame["dr007"] >= 20)).any():
        raise ValueError("Tushare DR007.IB 加权价含越界值")
    normalized = frame[
        ["date", "dr007", "ts_code", "repo_maturity"]
    ].sort_values("date")
    normalized["provider_value_field"] = VALUE_FIELD
    normalized["value_semantics"] = VALUE_SEMANTICS
    return normalized.reset_index(drop=True)


def combine_and_validate_chunks(
    frames: Sequence[pd.DataFrame],
    *,
    market_dates: Sequence[object],
    source_identity: str = "TUSHARE_PRO_REPO_DAILY_DR007_IB",
    observation_start: date = OBSERVATION_START,
    observation_cutoff: date = OBSERVATION_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并年度分块并按冻结交易日历生成准入指标。"""

    if not frames:
        raise ValueError("没有可合并的 DR007 年度分块")
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "date",
        "dr007",
        "ts_code",
        "repo_maturity",
        "provider_value_field",
        "value_semantics",
    }
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"DR007 年度分块缺少标准化字段：{missing}")
    if combined["date"].duplicated().any():
        raise ValueError("DR007 年度分块合并后含重复日期")
    if not combined["ts_code"].eq(TS_CODE).all():
        raise ValueError("DR007 合并序列混入非 DR007.IB 代码")
    if not combined["repo_maturity"].eq(REPO_MATURITY).all():
        raise ValueError("DR007 合并序列混入非 DR007 期限")
    if not combined["provider_value_field"].eq(VALUE_FIELD).all():
        raise ValueError("DR007 合并序列不是 Tushare weight 字段")
    if not combined["value_semantics"].eq(VALUE_SEMANTICS).all():
        raise ValueError("DR007 合并序列价值语义漂移")
    combined = combined.sort_values("date").reset_index(drop=True)
    metrics = validate_dr007_daily(
        combined[["date", "dr007"]],
        market_dates=market_dates,
        source_identity=source_identity,
        observation_start=observation_start,
        observation_cutoff=observation_cutoff,
    )
    metrics.update(
        {
            "provider_api_name": API_NAME,
            "provider_ts_code": TS_CODE,
            "provider_repo_maturity": REPO_MATURITY,
            "provider_value_field": VALUE_FIELD,
            "value_semantics": VALUE_SEMANTICS,
            "availability_rule": AVAILABILITY_RULE,
            "substitute_used": False,
            "interpolation_performed": False,
        }
    )
    return combined, metrics
