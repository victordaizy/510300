"""获取并冻结 510300 非对称压力风险 V1 所需的精确 DR007。

仅允许 Tushare ``repo_daily`` 的 ``DR007.IB``、``repo_maturity=DR007``
和 ``weight`` 加权价字段。本脚本复用并校验 V1.0.1 已冻结的申万与央行
来源，不重新抓取它们；不读取标签、未来收益、模型或组合结果。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_dr007_tushare_v1_0_2 import (  # noqa: E402
    API_NAME,
    AVAILABILITY_RULE,
    CredentialExpiredError,
    REMEDIATION_ID,
    REPO_MATURITY,
    SOURCE_VERSION,
    TS_CODE,
    VALUE_FIELD,
    VALUE_SEMANTICS,
    ProviderResponseError,
    Transport,
    build_request,
    calendar_year_chunks,
    canonical_sha256,
    combine_and_validate_chunks,
    parse_repo_daily_payload,
    public_transport_metadata,
    resolve_transport,
)


CONFIG_PATH = (
    ROOT
    / "config"
    / "510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2.yaml"
)
PARENT_V1_CONFIG_PATH = (
    ROOT
    / "config"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1.yaml"
)
RAW_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2"
)
CURATED_ROOT = (
    ROOT
    / "data"
    / "curated"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2"
)
STATUS_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2.json"
)
RECEIPT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2_receipt.json"
)
REPORT_PATH = (
    ROOT
    / "reports"
    / "research"
    / "510300_ASYMMETRIC_STRESS_HAZARD_V1_SOURCE_REMEDIATION_V1_0_2.md"
)


def now_shanghai() -> str:
    """返回带时区的上海时间。"""

    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    # ``data`` 在本机可能是指向 E 盘的目录联接。这里保留工作区逻辑路径，
    # 不调用 resolve() 穿透联接，否则合法产物会被误判为位于工作区之外。
    absolute_path = Path(os.path.abspath(path))
    absolute_root = Path(os.path.abspath(ROOT))
    return absolute_path.relative_to(absolute_root).as_posix()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"冻结父级文件不存在：{relative(path)}")
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(
            f"冻结父级文件哈希漂移：{relative(path)}，期望 {expected}，实际 {actual}"
        )


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V1.0.2 DR007 来源合同不是对象")
    if payload["program"]["remediation_id"] != REMEDIATION_ID:
        raise ValueError("V1.0.2 DR007 来源合同 remediation_id 不一致")
    return payload


def verify_immutable_parent(contract: Mapping[str, Any]) -> dict[str, Any]:
    """只读核验 V1.0.1，禁止重抓或覆盖申万和央行产物。"""

    parent = contract["immutable_parent"]
    pins = (
        (ROOT / parent["status_path"], parent["status_sha256"]),
        (ROOT / parent["receipt_path"], parent["receipt_sha256"]),
        (ROOT / parent["contract_path"], parent["contract_sha256"]),
        (ROOT / parent["sw_manifest_path"], parent["sw_manifest_sha256"]),
        (ROOT / parent["pboc_manifest_path"], parent["pboc_manifest_sha256"]),
    )
    for path, digest in pins:
        require_hash(path, str(digest))
    calendar_contract = contract["market_calendar"]
    require_hash(
        ROOT / calendar_contract["path"],
        str(calendar_contract["sha256"]),
    )
    status = json.loads((ROOT / parent["status_path"]).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / parent["receipt_path"]).read_text(encoding="utf-8"))
    if status.get("status") != parent["required_status"]:
        raise ValueError("V1.0.1 父级来源修复状态漂移")
    if status.get("blocked_channels") != parent["required_blocked_channels"]:
        raise ValueError("V1.0.1 父级阻断通道不再严格等于 M2")
    if receipt.get("receipt_payload_sha256") != parent["receipt_payload_sha256"]:
        raise ValueError("V1.0.1 父级回执载荷哈希漂移")
    source_results = status.get("source_results", {})
    if source_results.get("sw_pit_industry", {}).get("status") != parent[
        "required_sw_status"
    ]:
        raise ValueError("V1.0.1 申万 PIT 准入状态漂移")
    if source_results.get("pboc_7d_reverse_repo_policy_rate", {}).get(
        "status"
    ) != parent["required_pboc_status"]:
        raise ValueError("V1.0.1 央行 7 天逆回购准入状态漂移")
    if status.get("feature_values_constructed") is not False:
        raise ValueError("V1.0.1 父级错误地包含特征值")
    return status


def make_session(contract: Mapping[str, Any]) -> requests.Session:
    transport = contract["transport"]
    retry = Retry(
        total=int(transport["max_attempts_per_endpoint"]) - 1,
        connect=int(transport["max_attempts_per_endpoint"]) - 1,
        read=int(transport["max_attempts_per_endpoint"]) - 1,
        status=int(transport["max_attempts_per_endpoint"]) - 1,
        backoff_factor=0.8,
        status_forcelist=tuple(int(value) for value in transport["retry_status_codes"]),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "510300-asymmetric-stress-hazard-dr007-v1.0.2",
        }
    )
    return session


def _decode_provider_envelope(payload: bytes) -> tuple[dict[str, Any], str]:
    errors: list[str] = []
    for encoding in ("utf-8", "gb18030"):
        try:
            value = json.loads(payload.decode(encoding))
            if not isinstance(value, dict):
                raise ValueError("顶层 JSON 不是对象")
            return value, encoding
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{encoding}:{type(exc).__name__}")
    raise ValueError(f"提供方响应无法解析为 JSON：{errors}")


def sanitize_error(error: BaseException, secret: str) -> str:
    message = f"{type(error).__name__}: {error}"
    if secret:
        message = message.replace(secret, "[REDACTED_TOKEN]")
    return message[:1500]


def archive_response(
    directory: Path,
    prefix: str,
    payload: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    path = directory / f"{prefix}_{digest}.json"
    if path.is_file() and sha256_file(path) != digest:
        raise RuntimeError("已有 DR007 响应归档哈希异常")
    if not path.is_file():
        atomic_write_bytes(path, payload)
    return artifact_record(path)


def _request_for_endpoint(
    transport: Transport,
    endpoint: str,
    chunk_start: date,
    chunk_end: date,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    scoped = replace(transport, endpoints=(endpoint,))
    return build_request(scoped, chunk_start, chunk_end)


def credential_preflight(
    contract: Mapping[str, Any],
    transport: Transport,
) -> tuple[list[dict[str, Any]], bool]:
    """每个允许端点只请求一个官方文档样例日并归档响应。"""

    sample_date = date(2020, 8, 4)
    timeout = (
        float(contract["transport"]["connect_timeout_seconds"]),
        float(contract["transport"]["read_timeout_seconds"]),
    )
    results: list[dict[str, Any]] = []
    any_success = False
    for endpoint in transport.endpoints:
        url, payload, headers = _request_for_endpoint(
            transport,
            endpoint,
            sample_date,
            sample_date,
        )
        session = make_session(contract)
        started_at = now_shanghai()
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            content = response.content
            if transport.secret.encode("utf-8") in content:
                raise RuntimeError("提供方响应意外回显凭据，拒绝归档")
            record = archive_response(
                RAW_ROOT / "credential_preflight",
                f"repo_daily_DR007_IB_20200804_{urlparse(endpoint).hostname}",
                content,
            )
            envelope, decoded_as = _decode_provider_envelope(content)
            provider_code = envelope.get("code")
            provider_message = str(envelope.get("msg") or "")
            success = response.status_code == 200 and provider_code in {0, "0"}
            if success:
                frame = parse_repo_daily_payload(
                    content,
                    chunk_start=sample_date,
                    chunk_end=sample_date,
                )
                success = (
                    len(frame) == 1
                    and frame.iloc[0]["ts_code"] == TS_CODE
                    and frame.iloc[0]["repo_maturity"] == REPO_MATURITY
                )
            any_success = any_success or success
            results.append(
                {
                    "endpoint_host": urlparse(endpoint).hostname,
                    "request_url_host": urlparse(url).hostname,
                    "sample_date": sample_date.isoformat(),
                    "started_at": started_at,
                    "completed_at": now_shanghai(),
                    "http_status": int(response.status_code),
                    "provider_code": str(provider_code),
                    "provider_message": provider_message,
                    "response_decoded_as": decoded_as,
                    "exact_series_probe_passed": bool(success),
                    "credential_echoed": False,
                    "tls_verification": True,
                    "response_artifact": record,
                }
            )
        except Exception as error:  # noqa: BLE001 - 生成脱敏失败证据
            results.append(
                {
                    "endpoint_host": urlparse(endpoint).hostname,
                    "sample_date": sample_date.isoformat(),
                    "started_at": started_at,
                    "completed_at": now_shanghai(),
                    "exact_series_probe_passed": False,
                    "credential_echoed": False,
                    "tls_verification": True,
                    "sanitized_error": sanitize_error(error, transport.secret),
                }
            )
        finally:
            session.close()
    return results, any_success


def fetch_chunk(
    contract: Mapping[str, Any],
    transport: Transport,
    chunk_start: date,
    chunk_end: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = (
        RAW_ROOT
        / "repo_daily"
        / (
            f"repo_daily_DR007_IB_{chunk_start.strftime('%Y%m%d')}_"
            f"{chunk_end.strftime('%Y%m%d')}.json"
        )
    )
    if raw_path.is_file():
        content = raw_path.read_bytes()
        frame = parse_repo_daily_payload(
            content,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        return frame, {
            "request": {
                "api_name": API_NAME,
                "ts_code": TS_CODE,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "fields": ["ts_code", "trade_date", "repo_maturity", "weight"],
            },
            "retrieval_mode": "IMMUTABLE_CACHE",
            "endpoint_host": None,
            "tls_verification": True,
            "credential_archived": False,
            "response_artifact": artifact_record(raw_path),
            "row_count": int(len(frame)),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
        }
    timeout = (
        float(contract["transport"]["connect_timeout_seconds"]),
        float(contract["transport"]["read_timeout_seconds"]),
    )
    errors: list[str] = []
    for endpoint in transport.endpoints:
        url, payload, headers = _request_for_endpoint(
            transport,
            endpoint,
            chunk_start,
            chunk_end,
        )
        session = make_session(contract)
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            content = response.content
            if transport.secret.encode("utf-8") in content:
                raise RuntimeError("提供方响应意外回显凭据，拒绝归档")
            if urlparse(str(response.url)).scheme.lower() != "https":
                raise RuntimeError("DR007 提供方响应发生非 HTTPS 跳转")
            if response.status_code != 200:
                archive_response(
                    RAW_ROOT / "failed_responses",
                    (
                        f"repo_daily_DR007_IB_{chunk_start.strftime('%Y%m%d')}_"
                        f"{chunk_end.strftime('%Y%m%d')}_{urlparse(endpoint).hostname}_"
                        f"http_{response.status_code}"
                    ),
                    content,
                )
                raise RuntimeError(f"HTTP {response.status_code}")
            envelope, _ = _decode_provider_envelope(content)
            if envelope.get("code") not in {0, "0"}:
                provider_error = ProviderResponseError(
                    envelope.get("code"),
                    envelope.get("msg"),
                )
                archive_response(
                    RAW_ROOT / "failed_responses",
                    (
                        f"repo_daily_DR007_IB_{chunk_start.strftime('%Y%m%d')}_"
                        f"{chunk_end.strftime('%Y%m%d')}_{urlparse(endpoint).hostname}_"
                        f"provider_{provider_error.code}"
                    ),
                    content,
                )
                raise provider_error
            try:
                frame = parse_repo_daily_payload(
                    content,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
            except ProviderResponseError as error:
                archive_response(
                    RAW_ROOT / "failed_responses",
                    (
                        f"repo_daily_DR007_IB_{chunk_start.strftime('%Y%m%d')}_"
                        f"{chunk_end.strftime('%Y%m%d')}_{urlparse(endpoint).hostname}_"
                        f"provider_{error.code}"
                    ),
                    content,
                )
                raise
            atomic_write_bytes(raw_path, content)
            return frame, {
                "request": {
                    "api_name": API_NAME,
                    "ts_code": TS_CODE,
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "fields": [
                        "ts_code",
                        "trade_date",
                        "repo_maturity",
                        "weight",
                    ],
                },
                "retrieval_mode": "NETWORK_TLS_VERIFIED",
                "retrieved_at": now_shanghai(),
                "endpoint_host": urlparse(endpoint).hostname,
                "tls_verification": True,
                "credential_archived": False,
                "response_artifact": artifact_record(raw_path),
                "row_count": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
            }
        except Exception as error:  # noqa: BLE001 - 跨允许端点并保留脱敏错误
            errors.append(
                f"{urlparse(endpoint).hostname}: {sanitize_error(error, transport.secret)}"
            )
        finally:
            session.close()
    raise RuntimeError(
        f"DR007.IB {chunk_start} 至 {chunk_end} 所有授权端点失败：{errors}"
    )


def acquire_full_history(
    contract: Mapping[str, Any],
    transport: Transport,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parent_status = verify_immutable_parent(contract)
    calendar_path = ROOT / contract["market_calendar"]["path"]
    calendar_column = str(contract["market_calendar"]["date_column"])
    market_dates = pd.read_parquet(calendar_path, columns=[calendar_column])[
        calendar_column
    ].drop_duplicates()
    observation_start = date.fromisoformat(str(contract["program"]["observation_start"]))
    observation_cutoff = date.fromisoformat(str(contract["program"]["observation_cutoff"]))
    frames: list[pd.DataFrame] = []
    chunks: list[dict[str, Any]] = []
    ranges = calendar_year_chunks(observation_start, observation_cutoff)
    for sequence, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        frame, receipt = fetch_chunk(contract, transport, chunk_start, chunk_end)
        frames.append(frame)
        chunks.append(receipt)
        print(
            json.dumps(
                {
                    "阶段": "DR007年度分块",
                    "年份": chunk_start.year,
                    "行数": len(frame),
                    "状态": receipt["retrieval_mode"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if sequence < len(ranges):
            time.sleep(float(contract["transport"]["request_interval_seconds"]))
    combined, metrics = combine_and_validate_chunks(
        frames,
        market_dates=market_dates,
        source_identity=str(contract["admission"]["source_identity"]),
        observation_start=observation_start,
        observation_cutoff=observation_cutoff,
    )
    raw_set_sha256 = canonical_sha256(
        {"raw_response_artifacts": [item["response_artifact"] for item in chunks]}
    )
    retrieved_at = now_shanghai()
    combined["source_identity"] = contract["admission"]["source_identity"]
    combined["source_url"] = contract["dr007_source"]["provider_documentation"]
    combined["retrieved_at"] = retrieved_at
    combined["availability_rule"] = AVAILABILITY_RULE
    combined["raw_response_set_sha256"] = raw_set_sha256
    combined["missing_value_rule"] = "NO_VIEW_NO_INTERPOLATION"
    output_path = ROOT / contract["outputs"]["curated_artifact"]
    atomic_write_parquet(output_path, combined)
    status = (
        "PASS_DR007_SOURCE_ADMITTED_FULL_MARKET_SESSION_COVERAGE"
        if metrics["missing_market_session_count"] == 0
        else "PASS_DR007_SOURCE_ADMITTED_WITH_DATE_LEVEL_NO_VIEW"
    )
    manifest = {
        "program_id": contract["program"]["program_id"],
        "remediation_id": REMEDIATION_ID,
        "version": SOURCE_VERSION,
        "source_id": "TUSHARE_PRO_REPO_DAILY_DR007_IB_WEIGHT",
        "status": status,
        "completed_at": retrieved_at,
        "provider_documentation": contract["dr007_source"][
            "provider_documentation"
        ],
        "provider_api_name": API_NAME,
        "provider_ts_code": TS_CODE,
        "provider_repo_maturity": REPO_MATURITY,
        "provider_value_field": VALUE_FIELD,
        "value_semantics": VALUE_SEMANTICS,
        "availability_rule": AVAILABILITY_RULE,
        "transport": public_transport_metadata(transport),
        "chunk_receipts": chunks,
        "raw_response_set_sha256": raw_set_sha256,
        "metrics": metrics,
        "curated_artifact": artifact_record(output_path, rows=len(combined)),
        "immutable_parent_receipt_payload_sha256": contract["immutable_parent"][
            "receipt_payload_sha256"
        ],
        "substitute_used": False,
        "interpolation_performed": False,
        "feature_values_constructed": False,
        "label_artifacts_read": False,
        "future_returns_read": False,
        "portfolio_metrics_read": False,
        "credential_archived": False,
    }
    manifest_path = ROOT / contract["outputs"]["acquisition_manifest"]
    atomic_write_json(manifest_path, manifest)
    finalize_source_admission(
        contract,
        parent_status,
        manifest,
        manifest_path,
        transport,
    )
    return combined, manifest


def _dr007_source_result(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(manifest["metrics"])
    return {
        "source_id": "dr007_daily",
        "status": manifest["status"],
        "admitted": True,
        "whole_source_blocked": False,
        "date_level_no_view_required": metrics["missing_market_session_count"] > 0,
        "availability_clock": AVAILABILITY_RULE,
        "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
        "metrics": metrics,
        "limitations": [
            "ONLY_TUSHARE_REPO_DAILY_DR007_IB_WEIGHT_ADMITTED",
            "ANY_MISSING_MARKET_SESSION_REMAINS_DATE_LEVEL_NO_VIEW",
            "FORBIDDEN_SUBSTITUTES=FDR007,R007,FR007,EXCHANGE_REPO_R_007",
        ],
        "feature_values_constructed": False,
    }


def _build_final_status(
    contract: Mapping[str, Any],
    parent_status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    transport: Transport,
) -> dict[str, Any]:
    status = copy.deepcopy(dict(parent_status))
    dr007_result = _dr007_source_result(manifest)
    channels = copy.deepcopy(parent_status["channel_results"])
    channels["M2"].update(
        {
            "admitted": True,
            "status": "PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW",
            "blockers": [],
            "date_level_no_view_required": dr007_result[
                "date_level_no_view_required"
            ],
            "feature_values_constructed": False,
        }
    )
    source_results = copy.deepcopy(parent_status["source_results"])
    source_results["dr007_daily"] = dr007_result
    status.update(
        {
            "remediation_id": REMEDIATION_ID,
            "stage_id": contract["program"]["stage_id"],
            "version": SOURCE_VERSION,
            "completed_at": now_shanghai(),
            "status": "PASS_SOURCE_REMEDIATION_ALL_REQUIRED_SOURCES_ADMITTED",
            "research_disposition": "SOURCE_READY",
            "all_required_sources_admitted": True,
            "blocked_channels": [],
            "source_results": source_results,
            "channel_results": channels,
            "parent_source_remediation": {
                "version": "1.0.1",
                "status": parent_status["status"],
                "receipt_payload_sha256": contract["immutable_parent"][
                    "receipt_payload_sha256"
                ],
                "mutated": False,
            },
            "feature_construction_allowed": True,
            "g2_allowed": True,
            "feature_values_constructed": False,
            "label_artifacts_read": False,
            "bad10_census_rerun": False,
            "label_window_threshold_changed": False,
            "model_trained": False,
            "probability_threshold_selected": False,
            "return_evaluation": "NOT_PERFORMED",
            "portfolio_metrics_read": False,
            "position_generated": False,
            "paper_shadow_generated": False,
            "broker_action_performed": False,
            "order_generated": False,
            "live_trading_authorized": False,
            "position_impact": 0,
            "rescue_allowed": False,
            "substitute_or_proxy_used": False,
            "transport_proxy_used": transport.credential_kind.startswith(
                "PROJECT_AUTHORIZED_"
            ),
            "transport_proxy_changes_series_semantics": False,
            "next_allowed_action": "CONSTRUCT_FROZEN_M_F_T_FEATURES_UNDER_G2",
        }
    )
    return status


def _build_human_report(status: Mapping[str, Any]) -> str:
    metrics = status["source_results"]["dr007_daily"]["metrics"]
    return "\n".join(
        [
            "# 510300 非对称压力风险 V1 来源修复 V1.0.2",
            "",
            "## 结论",
            "",
            f"`{status['status']}`",
            "",
            "V1.0.1 的申万 PIT 行业归属与央行 7 天逆回购操作利率保持原样；",
            "V1.0.2 仅新增并准入 Tushare repo_daily 的 DR007.IB / weight 日度序列。",
            "",
            "## DR007 冻结口径",
            "",
            f"- 观察期：{metrics['first_date']} 至 {metrics['last_date']}",
            f"- 行数：{metrics['row_count']}",
            f"- 冻结交易日：{metrics['market_session_count']}",
            f"- 已覆盖交易日：{metrics['covered_market_session_count']}",
            f"- 缺失交易日：{metrics['missing_market_session_count']}，逐日 NO_VIEW，不插值",
            "- 精确字段：DR007.IB / DR007 / weight（加权价，%）",
            "- 禁止替代：FDR007、R007、FR007、交易所 R-007",
            "",
            "## 权限边界",
            "",
            "本阶段没有构造特征、读取 BAD10/未来收益或组合指标，也没有生成仓位、订单或交易动作。",
            "来源全部准入后，下一步仅允许在另一个冻结步骤中构造既定 M/F/T 特征并进入 G2。",
            "",
        ]
    )


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finalize_source_admission(
    contract: Mapping[str, Any],
    parent_status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    transport: Transport,
) -> None:
    status = _build_final_status(contract, parent_status, manifest, transport)
    atomic_write_json(STATUS_PATH, status)
    atomic_write_bytes(REPORT_PATH, _build_human_report(status).encode("utf-8"))
    receipt_without_hash: dict[str, Any] = {
        "program_id": contract["program"]["program_id"],
        "remediation_id": REMEDIATION_ID,
        "stage_id": contract["program"]["stage_id"],
        "version": SOURCE_VERSION,
        "status": "IMMUTABLE_SOURCE_REMEDIATION_RECEIPT_COMPLETE",
        "source_remediation_status": status["status"],
        "completed_at": status["completed_at"],
        "git_head": git_head(),
        "parent_v1_0_1": {
            "status_path": contract["immutable_parent"]["status_path"],
            "status_sha256": contract["immutable_parent"]["status_sha256"],
            "receipt_path": contract["immutable_parent"]["receipt_path"],
            "receipt_sha256": contract["immutable_parent"]["receipt_sha256"],
            "receipt_payload_sha256": contract["immutable_parent"][
                "receipt_payload_sha256"
            ],
            "mutated": False,
        },
        "contract": artifact_record(CONFIG_PATH),
        "source_manifests": {
            "sw": {
                "path": contract["immutable_parent"]["sw_manifest_path"],
                "sha256": contract["immutable_parent"]["sw_manifest_sha256"],
                "reacquired": False,
            },
            "pboc": {
                "path": contract["immutable_parent"]["pboc_manifest_path"],
                "sha256": contract["immutable_parent"]["pboc_manifest_sha256"],
                "reacquired": False,
            },
            "dr007": artifact_record(manifest_path),
        },
        "outputs": {
            "status": artifact_record(STATUS_PATH),
            "human_report": artifact_record(REPORT_PATH),
        },
        "all_required_sources_admitted": True,
        "blocked_channels": [],
        "feature_construction_allowed": True,
        "g2_allowed": True,
        "feature_values_constructed": False,
        "label_artifacts_read": False,
        "bad10_census_rerun": False,
        "return_evaluation": "NOT_PERFORMED",
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "substitute_or_proxy_used": False,
        "next_allowed_action": "CONSTRUCT_FROZEN_M_F_T_FEATURES_UNDER_G2",
    }
    receipt = {
        **receipt_without_hash,
        "receipt_payload_sha256": canonical_sha256(receipt_without_hash),
    }
    atomic_write_json(RECEIPT_PATH, receipt)


def write_attempt_evidence(
    contract: Mapping[str, Any],
    transport: Transport | None,
    *,
    status: str,
    probe_results: list[dict[str, Any]],
    sanitized_error: str | None,
) -> tuple[Path, Path]:
    completed_at = now_shanghai()
    metadata = public_transport_metadata(transport) if transport else {
        "credential_kind": "NONE_OR_UNRESOLVED",
        "credential_archived": False,
    }
    payload: dict[str, Any] = {
        "program_id": contract["program"]["program_id"],
        "remediation_id": REMEDIATION_ID,
        "version": SOURCE_VERSION,
        "attempt_stage": "DR007_AUTHORIZED_CREDENTIAL_PREFLIGHT",
        "status": status,
        "completed_at": completed_at,
        "required_series": {
            "api_name": API_NAME,
            "ts_code": TS_CODE,
            "repo_maturity": REPO_MATURITY,
            "value_field": VALUE_FIELD,
            "value_semantics": VALUE_SEMANTICS,
        },
        "transport": metadata,
        "probe_results": probe_results,
        "sanitized_error": sanitized_error,
        "all_required_sources_admitted": False,
        "blocked_channels": ["M2"],
        "feature_construction_allowed": False,
        "g2_allowed": False,
        "feature_values_constructed": False,
        "label_artifacts_read": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "credential_archived": False,
        "substitute_used": False,
        "next_allowed_action": (
            "RENEW_TUSHARE_AUTHORIZED_CREDENTIAL_AND_RERUN_V1_0_2"
        ),
    }
    attempt_hash = canonical_sha256(payload)
    stamp = pd.Timestamp(completed_at).strftime("%Y%m%dT%H%M%S%z")
    safe_status = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in status
    )
    report_root = ROOT / contract["outputs"]["attempt_report_root"]
    receipt_root = ROOT / contract["outputs"]["attempt_receipt_root"]
    report_path = report_root / f"{stamp}_{safe_status}_{attempt_hash[:12]}.json"
    atomic_write_json(report_path, payload)
    receipt_without_hash = {
        "program_id": contract["program"]["program_id"],
        "remediation_id": REMEDIATION_ID,
        "version": SOURCE_VERSION,
        "status": "IMMUTABLE_DR007_CREDENTIAL_PREFLIGHT_RECEIPT_COMPLETE",
        "attempt_status": status,
        "completed_at": completed_at,
        "git_head": git_head(),
        "contract": artifact_record(CONFIG_PATH),
        "attempt_report": artifact_record(report_path),
        "parent_v1_0_1_receipt_payload_sha256": contract["immutable_parent"][
            "receipt_payload_sha256"
        ],
        "credential_archived": False,
        "all_required_sources_admitted": False,
        "blocked_channels": ["M2"],
        "feature_construction_allowed": False,
        "g2_allowed": False,
        "return_evaluation": "NOT_ALLOWED",
        "next_allowed_action": "RENEW_TUSHARE_AUTHORIZED_CREDENTIAL_AND_RERUN_V1_0_2",
    }
    receipt = {
        **receipt_without_hash,
        "receipt_payload_sha256": canonical_sha256(receipt_without_hash),
    }
    receipt_path = receipt_root / f"{stamp}_{safe_status}_{attempt_hash[:12]}_receipt.json"
    atomic_write_json(receipt_path, receipt)
    return report_path, receipt_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credential-preflight",
        action="store_true",
        help="仅对官方文档样例日做授权凭据与精确序列探针，不下载全历史",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract()
    verify_immutable_parent(contract)
    load_dotenv(ROOT / ".env")
    transport: Transport | None = None
    try:
        transport = resolve_transport(
            contract,
            os.environ,
            allow_expired_proxy_for_probe=args.credential_preflight,
        )
        if args.credential_preflight:
            probe_results, passed = credential_preflight(contract, transport)
            status = (
                "PASS_AUTHORIZED_CREDENTIAL_EXACT_DR007_IB_PROBE"
                if passed
                else "BLOCKED_AUTHORIZED_CREDENTIAL_REJECTED_OR_EXPIRED"
            )
            report_path, receipt_path = write_attempt_evidence(
                contract,
                transport,
                status=status,
                probe_results=probe_results,
                sanitized_error=None,
            )
            print(
                json.dumps(
                    {
                        "状态": status,
                        "报告": relative(report_path),
                        "回执": relative(receipt_path),
                        "凭据已归档": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 0 if passed else 3
        frame, manifest = acquire_full_history(contract, transport)
        print(
            json.dumps(
                {
                    "状态": "PASS_SOURCE_REMEDIATION_ALL_REQUIRED_SOURCES_ADMITTED",
                    "DR007行数": len(frame),
                    "来源状态": manifest["status"],
                    "最终回执": relative(RECEIPT_PATH),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except CredentialExpiredError as error:
        report_path, receipt_path = write_attempt_evidence(
            contract,
            transport,
            status="BLOCKED_AUTHORIZED_CREDENTIAL_EXPIRED_BY_LOCAL_CLOCK",
            probe_results=[],
            sanitized_error=sanitize_error(error, transport.secret if transport else ""),
        )
        print(
            json.dumps(
                {
                    "状态": "BLOCKED_AUTHORIZED_CREDENTIAL_EXPIRED_BY_LOCAL_CLOCK",
                    "报告": relative(report_path),
                    "回执": relative(receipt_path),
                    "凭据已归档": False,
                },
                ensure_ascii=False,
            )
        )
        return 3
    except (ProviderResponseError, RuntimeError, ValueError, requests.RequestException) as error:
        report_path, receipt_path = write_attempt_evidence(
            contract,
            transport,
            status="BLOCKED_DR007_AUTHORIZED_ACQUISITION_FAILED",
            probe_results=[],
            sanitized_error=sanitize_error(error, transport.secret if transport else ""),
        )
        print(
            json.dumps(
                {
                    "状态": "BLOCKED_DR007_AUTHORIZED_ACQUISITION_FAILED",
                    "报告": relative(report_path),
                    "回执": relative(receipt_path),
                    "凭据已归档": False,
                },
                ensure_ascii=False,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
