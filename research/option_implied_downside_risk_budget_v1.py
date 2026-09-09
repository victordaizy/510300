"""510300 期权隐含下行风险预算 V1 的数据准入实现。

本模块只覆盖协议阶段 B/C：权限探针、不可变原始采集、期权曲面构造和
G0/G1 裁决。它不构造未来标签，不训练模型，不计算组合收益、净值或夏普，
也不连接券商或生成订单。
"""

from __future__ import annotations

import getpass
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as wall_time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
from scipy.optimize import brentq


TIMEZONE = ZoneInfo("Asia/Shanghai")
MODEL_ID = "510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1"
PROTOCOL_RELATIVE_PATH = Path(
    "config/510300_option_implied_downside_risk_budget_v1.yaml"
)
PROTOCOL_MANIFEST_RELATIVE_PATH = Path(
    "config/510300_option_implied_downside_risk_budget_v1_manifest.json"
)
IMPLEMENTATION_MANIFEST_RELATIVE_PATH = Path(
    "config/510300_option_implied_downside_risk_budget_v1_implementation_manifest.json"
)
REPORT_RELATIVE_DIR = Path(
    "reports/data_quality/510300_option_implied_downside_risk_budget_v1"
)


class ProtocolIntegrityError(RuntimeError):
    """冻结协议或实现哈希不一致。"""


class ApiCallError(RuntimeError):
    """数据接口调用失败。"""


class DataAdmissionError(RuntimeError):
    """数据不满足冻结的准入契约。"""


def now_local() -> datetime:
    """返回带 Asia/Shanghai 时区的当前时间。"""

    return datetime.now(TIMEZONE)


def sha256_bytes(content: bytes) -> str:
    """计算字节串 SHA-256。"""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """生成用于稳定指纹的规范 JSON 字节。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(payload: Any, path: Path) -> None:
    """原子写入 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_protocol(project_root: Path) -> dict[str, Any]:
    """读取并进行最小结构校验。"""

    path = project_root / PROTOCOL_RELATIVE_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolIntegrityError("协议 YAML 顶层必须是对象")
    protocol = payload.get("protocol", {})
    if protocol.get("model_id") != MODEL_ID:
        raise ProtocolIntegrityError("协议 MODEL_ID 不匹配")
    if protocol.get("state") != "PROTOCOL_FROZEN_BEFORE_FIRST_API_PROBE":
        raise ProtocolIntegrityError("协议冻结状态不正确")
    return payload


def _verify_hash_mapping(project_root: Path, mapping: Mapping[str, str]) -> None:
    """验证一组相对路径及 SHA-256。"""

    for relative, expected in mapping.items():
        path = project_root / Path(relative)
        if not path.is_file():
            raise ProtocolIntegrityError(f"冻结文件缺失：{relative}")
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise ProtocolIntegrityError(
                f"冻结文件哈希漂移：{relative}；expected={expected}；actual={actual}"
            )


def verify_frozen_manifests(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证协议清单和独立实现清单。"""

    protocol_manifest_path = project_root / PROTOCOL_MANIFEST_RELATIVE_PATH
    implementation_manifest_path = project_root / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    if not protocol_manifest_path.is_file():
        raise ProtocolIntegrityError("缺少协议冻结清单")
    if not implementation_manifest_path.is_file():
        raise ProtocolIntegrityError("缺少实现冻结清单")
    protocol_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    implementation_manifest = json.loads(
        implementation_manifest_path.read_text(encoding="utf-8")
    )
    if protocol_manifest.get("freeze_state") != "PROTOCOL_FROZEN_BEFORE_FIRST_API_PROBE":
        raise ProtocolIntegrityError("协议清单状态不正确")
    if (
        implementation_manifest.get("freeze_state")
        != "IMPLEMENTATION_FROZEN_BEFORE_FIRST_API_PROBE"
    ):
        raise ProtocolIntegrityError("实现清单状态不正确")
    if implementation_manifest.get("model_id") != MODEL_ID:
        raise ProtocolIntegrityError("实现清单 MODEL_ID 不匹配")
    expected_protocol_manifest_hash = implementation_manifest.get(
        "protocol_manifest_sha256"
    )
    actual_protocol_manifest_hash = sha256_file(protocol_manifest_path)
    if expected_protocol_manifest_hash != actual_protocol_manifest_hash:
        raise ProtocolIntegrityError("实现清单引用的协议清单哈希不匹配")
    _verify_hash_mapping(project_root, protocol_manifest.get("frozen_files", {}))
    _verify_hash_mapping(
        project_root, implementation_manifest.get("implementation_files", {})
    )
    return protocol_manifest, implementation_manifest


def endpoint_origin(endpoint: str) -> str:
    """只保留端点的 scheme 和 host。"""

    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("冻结端点必须是合法 HTTPS 根地址")
    return f"{parsed.scheme}://{parsed.netloc}"


def sanitize_text(value: Any, secret: str = "", limit: int = 1000) -> str:
    """移除令牌并限制错误文本长度。"""

    text = str(value)
    if secret:
        text = text.replace(secret, "[凭据已隐藏]")
    text = re.sub(r"(?i)(x-api-key|token|authorization)\s*[:=]\s*[^\s,;}]+", r"\1=[凭据已隐藏]", text)
    return text[:limit]


def resolve_ephemeral_token(environment_name: str = "TUSHARE_PROXY_TOKEN") -> str:
    """从进程环境或隐藏交互提示读取 56 位临时凭据。"""

    token = os.getenv(environment_name, "").strip()
    if not token:
        token = getpass.getpass("请输入 56 位临时数据 API Key（输入不会显示）：").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{56}", token):
        raise ValueError("临时数据 API Key 必须是 56 位字母或数字")
    return token


class RateLimiter:
    """保证任意两次请求起点之间不小于冻结间隔。"""

    def __init__(self, minimum_interval_seconds: float) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("最小请求间隔不能为负数")
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self._last_started_at: float | None = None

    def wait(self) -> None:
        """在当前线程中等待到下一个合法请求起点。"""

        current = time.monotonic()
        if self._last_started_at is not None:
            remaining = self.minimum_interval_seconds - (current - self._last_started_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_started_at = time.monotonic()


@dataclass(frozen=True)
class ApiResult:
    """一次 API 请求的完整内存结果。"""

    api_name: str
    request_parameters: dict[str, Any]
    fields_requested: str
    request_started_at: str
    response_received_at: str
    http_status: int | None
    success: bool
    error_code: str
    error_message: str
    raw_bytes: bytes
    response_sha256: str
    frame: pd.DataFrame


class TushareProxyClient:
    """使用请求头鉴权、保留原始响应字节的只读 Tushare 兼容客户端。"""

    RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        endpoint: str,
        token: str,
        minimum_interval_seconds: float,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        maximum_attempts: int,
        retry_delays_seconds: Sequence[float],
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.origin = endpoint_origin(self.endpoint)
        self._token = token
        self._limiter = RateLimiter(minimum_interval_seconds)
        self._timeout = (float(connect_timeout_seconds), float(read_timeout_seconds))
        self._maximum_attempts = int(maximum_attempts)
        self._retry_delays = [float(value) for value in retry_delays_seconds]
        self._session = session or requests.Session()
        if self._maximum_attempts < 1:
            raise ValueError("最大请求次数至少为 1")

    def _failure_result(
        self,
        api_name: str,
        parameters: dict[str, Any],
        fields: str,
        started_at: str,
        http_status: int | None,
        code: str,
        message: str,
        raw_bytes: bytes,
    ) -> ApiResult:
        """构造不会泄露凭据的失败结果。"""

        return ApiResult(
            api_name=api_name,
            request_parameters=parameters,
            fields_requested=fields,
            request_started_at=started_at,
            response_received_at=now_local().isoformat(),
            http_status=http_status,
            success=False,
            error_code=code,
            error_message=sanitize_text(message, self._token),
            raw_bytes=raw_bytes,
            response_sha256=sha256_bytes(raw_bytes),
            frame=pd.DataFrame(),
        )

    def call(
        self, api_name: str, parameters: Mapping[str, Any], fields: str = ""
    ) -> ApiResult:
        """调用一次接口；只对冻结的传输类错误做有限重试。"""

        clean_parameters = {str(key): value for key, value in parameters.items()}
        payload: dict[str, Any] = {
            "api_name": api_name,
            "params": clean_parameters,
        }
        if fields:
            payload["fields"] = fields
        headers = {
            "x-api-key": self._token,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"{MODEL_ID}/1.0 data-admission",
        }
        last_result: ApiResult | None = None
        for attempt in range(1, self._maximum_attempts + 1):
            self._limiter.wait()
            started_at = now_local().isoformat()
            try:
                response = self._session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                raw_bytes = bytes(response.content)
                if response.status_code in self.RETRYABLE_HTTP_STATUS:
                    last_result = self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        f"HTTP_{response.status_code}",
                        response.text,
                        raw_bytes,
                    )
                    if attempt < self._maximum_attempts:
                        delay = self._retry_delays[
                            min(attempt - 1, len(self._retry_delays) - 1)
                        ]
                        time.sleep(delay)
                        continue
                    return last_result
                if not 200 <= response.status_code < 300:
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        f"HTTP_{response.status_code}",
                        response.text,
                        raw_bytes,
                    )
                try:
                    body = response.json()
                except ValueError as exc:
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        "INVALID_JSON_RESPONSE",
                        exc,
                        raw_bytes,
                    )
                code = body.get("code", "MISSING_CODE") if isinstance(body, dict) else "INVALID_BODY"
                message = body.get("msg", "") if isinstance(body, dict) else "响应顶层不是对象"
                if str(code) not in {"0", "0.0"}:
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        str(code),
                        message,
                        raw_bytes,
                    )
                data = body.get("data") if isinstance(body, dict) else None
                if not isinstance(data, dict):
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        "INVALID_DATA_OBJECT",
                        "成功响应缺少 data 对象",
                        raw_bytes,
                    )
                returned_fields = data.get("fields", [])
                items = data.get("items", [])
                if not isinstance(returned_fields, list) or not isinstance(items, list):
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        "INVALID_TABULAR_DATA",
                        "data.fields 或 data.items 不是列表",
                        raw_bytes,
                    )
                try:
                    frame = pd.DataFrame(items, columns=[str(value) for value in returned_fields])
                except Exception as exc:
                    return self._failure_result(
                        api_name,
                        clean_parameters,
                        fields,
                        started_at,
                        int(response.status_code),
                        "TABULAR_DECODE_FAILED",
                        exc,
                        raw_bytes,
                    )
                return ApiResult(
                    api_name=api_name,
                    request_parameters=clean_parameters,
                    fields_requested=fields,
                    request_started_at=started_at,
                    response_received_at=now_local().isoformat(),
                    http_status=int(response.status_code),
                    success=True,
                    error_code="0",
                    error_message=sanitize_text(message, self._token),
                    raw_bytes=raw_bytes,
                    response_sha256=sha256_bytes(raw_bytes),
                    frame=frame,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_result = self._failure_result(
                    api_name,
                    clean_parameters,
                    fields,
                    started_at,
                    None,
                    type(exc).__name__.upper(),
                    exc,
                    b"",
                )
                if attempt < self._maximum_attempts:
                    delay = self._retry_delays[
                        min(attempt - 1, len(self._retry_delays) - 1)
                    ]
                    time.sleep(delay)
                    continue
                return last_result
            except requests.RequestException as exc:
                return self._failure_result(
                    api_name,
                    clean_parameters,
                    fields,
                    started_at,
                    None,
                    type(exc).__name__.upper(),
                    exc,
                    b"",
                )
        if last_result is None:
            raise AssertionError("请求循环未产生结果")
        return last_result


def build_client(protocol: Mapping[str, Any], token: str) -> TushareProxyClient:
    """依据冻结协议创建客户端。"""

    source = protocol["source_contract"]
    endpoint = str(source["selected_endpoint"])
    if endpoint not in source["endpoint_allowlist"]:
        raise ProtocolIntegrityError("选定端点不在冻结允许名单内")
    return TushareProxyClient(
        endpoint=endpoint,
        token=token,
        minimum_interval_seconds=float(source["minimum_request_interval_seconds"]),
        connect_timeout_seconds=float(source["connect_timeout_seconds"]),
        read_timeout_seconds=float(source["read_timeout_seconds"]),
        maximum_attempts=int(source["maximum_attempts"]),
        retry_delays_seconds=source["retry_delays_seconds"],
    )


def recorded_parameters(result: ApiResult) -> dict[str, Any]:
    """把 fields 纳入公开、无凭据的请求参数记录。"""

    parameters = dict(result.request_parameters)
    if result.fields_requested:
        parameters["_fields"] = result.fields_requested
    return parameters


def _timestamp_slug(value: str) -> str:
    """把 ISO 时间转换为 Windows 安全目录名。"""

    parsed = datetime.fromisoformat(value).astimezone(TIMEZONE)
    return parsed.strftime("%Y%m%dT%H%M%S%f%z")


def save_immutable_capture(
    project_root: Path,
    partition_relative: Path,
    result: ApiResult,
    endpoint: str,
) -> dict[str, Any]:
    """以只追加目录保存原始响应、派生 Parquet 和收据。"""

    partition = project_root / partition_relative
    partition.mkdir(parents=True, exist_ok=True)
    request_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "api": result.api_name,
                "parameters": recorded_parameters(result),
                "response_sha256": result.response_sha256,
            }
        )
    )[:12]
    final_name = f"retrieved_at={_timestamp_slug(result.response_received_at)}-{request_fingerprint}"
    final_directory = partition / final_name
    if final_directory.exists():
        final_directory = partition / f"{final_name}-{uuid.uuid4().hex[:8]}"
    staging = partition / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    raw_path = staging / "response.json"
    raw_path.write_bytes(result.raw_bytes)
    normalized_hash: str | None = None
    normalized_final_relative: str | None = None
    if result.success:
        normalized_path = staging / "normalized.parquet"
        result.frame.to_parquet(normalized_path, index=False)
        normalized_hash = sha256_file(normalized_path)
        normalized_final_relative = (
            final_directory / "normalized.parquet"
        ).relative_to(project_root).as_posix()
    receipt = {
        "source": "TUSHARE_COMPATIBLE_PROXY",
        "api": result.api_name,
        "request_parameters": recorded_parameters(result),
        "request_started_at": result.request_started_at,
        "retrieved_at": result.response_received_at,
        "success": bool(result.success),
        "http_status": result.http_status,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "row_count": int(len(result.frame)),
        "columns": [str(column) for column in result.frame.columns],
        "response_bytes": int(len(result.raw_bytes)),
        "bytes": int(len(result.raw_bytes)),
        "response_sha256": result.response_sha256,
        "raw_relative_path": (
            final_directory / "response.json"
        ).relative_to(project_root).as_posix(),
        "normalized_relative_path": normalized_final_relative,
        "normalized_sha256": normalized_hash,
        "endpoint_origin": endpoint_origin(endpoint),
        "credential_persisted": False,
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, final_directory)
    receipt["receipt_relative_path"] = (
        final_directory / "receipt.json"
    ).relative_to(project_root).as_posix()
    return receipt


def _receipt_matches(
    project_root: Path,
    receipt_path: Path,
    api_name: str,
    parameters: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """验证一个已存在的成功收据及其文件哈希。"""

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt.get("success") or receipt.get("api") != api_name:
            return None
        if parameters is not None and receipt.get("request_parameters") != dict(parameters):
            return None
        raw_path = project_root / Path(receipt["raw_relative_path"])
        normalized_path = project_root / Path(receipt["normalized_relative_path"])
        if not raw_path.is_file() or not normalized_path.is_file():
            return None
        if sha256_file(raw_path) != receipt.get("response_sha256"):
            return None
        if sha256_file(normalized_path) != receipt.get("normalized_sha256"):
            return None
        frame = pd.read_parquet(normalized_path)
        if int(receipt.get("row_count", -1)) != len(frame):
            return None
        if [str(column) for column in frame.columns] != receipt.get("columns"):
            return None
        return receipt, frame
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def find_latest_valid_capture(
    project_root: Path,
    partition_relative: Path,
    api_name: str,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """寻找一个分区中最新的、哈希仍有效的成功采集。"""

    partition = project_root / partition_relative
    if not partition.is_dir():
        return None
    receipts = sorted(
        partition.glob("retrieved_at=*/receipt.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for receipt_path in receipts:
        match = _receipt_matches(project_root, receipt_path, api_name, parameters)
        if match is not None:
            return match
    return None


def validate_frame_contract(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    require_nonempty: bool = True,
) -> list[str]:
    """返回表格契约错误列表。"""

    errors: list[str] = []
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        errors.append(f"缺少字段：{','.join(missing)}")
    if require_nonempty and frame.empty:
        errors.append("成功响应行数为 0")
    return errors


def _probe_partition(api_name: str) -> Path:
    return Path(
        "data/raw/tushare/510300_option_implied_downside_risk_budget_v1"
    ) / "_permission_probe" / api_name


def run_permission_probe(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
    protocol_manifest: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """执行冻结的最小 G0 权限探针。"""

    order = [
        "trade_cal",
        "fund_daily",
        "fund_div",
        "opt_basic",
        "opt_daily",
        "shibor",
        "etf_mins",
    ]
    results: list[dict[str, Any]] = []
    for api_name in order:
        contract = protocol["api_contracts"][api_name]
        parameters = dict(contract["probe_params"])
        result = client.call(api_name, parameters)
        contract_errors = (
            validate_frame_contract(
                result.frame,
                contract["required_columns"],
                require_nonempty=True,
            )
            if result.success
            else []
        )
        effective_success = bool(result.success and not contract_errors)
        capture_receipt = save_immutable_capture(
            project_root,
            _probe_partition(api_name),
            result,
            client.endpoint,
        )
        error_message = result.error_message
        error_code = result.error_code
        if contract_errors:
            error_code = "DATA_CONTRACT_FAILED"
            error_message = "；".join(contract_errors)
        record = {
            "api_name": api_name,
            "request_parameters": recorded_parameters(result),
            "request_time": result.request_started_at,
            "request_started_at": result.request_started_at,
            "response_received_at": result.response_received_at,
            "success": effective_success,
            "row_count": int(len(result.frame)),
            "returned_columns": [str(column) for column in result.frame.columns],
            "error_code": error_code,
            "error_message": error_message,
            "response_sha256": result.response_sha256,
            "endpoint_origin": client.origin,
            "raw_relative_path": capture_receipt["raw_relative_path"],
            "receipt_relative_path": capture_receipt["receipt_relative_path"],
        }
        results.append(record)
        print(
            f"权限探针 {api_name}: "
            f"{'通过' if effective_success else '失败'}，rows={len(result.frame)}",
            flush=True,
        )
    result_by_api = {item["api_name"]: item for item in results}
    core_apis = protocol["permission_probe"]["core_strategy_blocked_if_any_fail"]
    failed_core = [api for api in core_apis if not result_by_api[api]["success"]]
    minute_success = bool(result_by_api["etf_mins"]["success"])
    if failed_core:
        state = protocol["permission_probe"]["core_failure_state"]
    else:
        state = protocol["permission_probe"]["overall_pass_state"]
    report = {
        "model_id": MODEL_ID,
        "phase": "G0_PERMISSION_PROBE",
        "state": state,
        "generated_at": now_local().isoformat(),
        "endpoint_origin": client.origin,
        "protocol_freeze_id": protocol_manifest["freeze_id"],
        "protocol_manifest_sha256": sha256_file(
            project_root / PROTOCOL_MANIFEST_RELATIVE_PATH
        ),
        "implementation_freeze_id": implementation_manifest["freeze_id"],
        "implementation_manifest_sha256": sha256_file(
            project_root / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
        ),
        "results": results,
        "failed_core_apis": failed_core,
        "core_strategy_data_acquisition_allowed": not failed_core,
        "etf_mins_available": minute_success,
        "prediction_research_data_acquisition_allowed": not failed_core,
        "final_execution_backtest_data_acquisition_allowed": not failed_core
        and minute_success,
        "future_return_reads": 0,
        "portfolio_evaluation_run": False,
        "position_impact": 0,
        "order_generation_enabled": False,
        "credential_persisted": False,
    }
    report_path = project_root / Path(protocol["permission_probe"]["output"])
    atomic_json(report, report_path)
    return report


def _full_partition(api_name: str, qualifier: str) -> Path:
    """生成冻结原始目录下的全量采集分区。"""

    root = Path("data/raw/tushare/510300_option_implied_downside_risk_budget_v1")
    if api_name == "trade_cal":
        return root / "trade_cal" / qualifier
    if api_name == "fund_daily":
        return root / "fund_daily" / "510300.SH" / qualifier
    if api_name == "fund_div":
        return root / "fund_div" / "510300.SH" / qualifier
    if api_name == "opt_basic":
        return root / "opt_basic" / "SSE" / qualifier
    if api_name == "opt_daily":
        return root / "opt_daily" / "SSE" / qualifier
    if api_name == "shibor":
        return root / "shibor" / qualifier
    if api_name == "etf_mins":
        return root / "etf_mins" / "510300.SH" / "1min" / qualifier
    raise KeyError(f"未知 API：{api_name}")


def _checked_call_and_capture(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
    api_name: str,
    parameters: Mapping[str, Any],
    partition: Path,
    require_nonempty: bool = True,
    maximum_rows: int | None = None,
    allow_resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """复用有效不可变采集，否则调用并保存。"""

    expected_parameters = dict(parameters)
    if allow_resume:
        existing = find_latest_valid_capture(
            project_root, partition, api_name, expected_parameters
        )
        if existing is not None:
            receipt, frame = existing
            errors = validate_frame_contract(
                frame,
                protocol["api_contracts"][api_name]["required_columns"],
                require_nonempty=require_nonempty,
            )
            if not errors and (maximum_rows is None or len(frame) < maximum_rows):
                return frame, receipt, True
    result = client.call(api_name, expected_parameters)
    receipt = save_immutable_capture(
        project_root, partition, result, client.endpoint
    )
    if not result.success:
        raise ApiCallError(
            f"{api_name} 调用失败：{result.error_code} {result.error_message}"
        )
    errors = validate_frame_contract(
        result.frame,
        protocol["api_contracts"][api_name]["required_columns"],
        require_nonempty=require_nonempty,
    )
    if errors:
        raise DataAdmissionError(f"{api_name} 数据契约失败：{'；'.join(errors)}")
    if maximum_rows is not None and len(result.frame) >= maximum_rows:
        raise DataAdmissionError(
            f"{api_name} 返回 {len(result.frame)} 行，达到或超过文档上限 "
            f"{maximum_rows}，按可能截断停止"
        )
    return result.frame, receipt, False


def _as_yyyymmdd(value: Any) -> str:
    """规范为 YYYYMMDD。"""

    parsed = pd.to_datetime(value, errors="raise")
    return parsed.strftime("%Y%m%d")


def _calendar_months(start: date, end: date) -> list[tuple[date, date]]:
    """产生闭区间内的自然月边界。"""

    current = date(start.year, start.month, 1)
    months: list[tuple[date, date]] = []
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        month_start = max(start, current)
        month_end = min(end, next_month - timedelta(days=1))
        months.append((month_start, month_end))
        current = next_month
    return months


def _calendar_years(start: date, end: date) -> list[tuple[date, date]]:
    """产生闭区间内的自然年边界。"""

    return [
        (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


def _load_probe_report(project_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """读取并验证权限探针与当前冻结清单的绑定。"""

    path = project_root / Path(protocol["permission_probe"]["output"])
    if not path.is_file():
        raise DataAdmissionError("缺少 permission_probe.json，禁止全量下载")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("model_id") != MODEL_ID:
        raise DataAdmissionError("权限探针 MODEL_ID 不匹配")
    if report.get("protocol_manifest_sha256") != sha256_file(
        project_root / PROTOCOL_MANIFEST_RELATIVE_PATH
    ):
        raise DataAdmissionError("权限探针不是基于当前协议清单生成")
    if report.get("implementation_manifest_sha256") != sha256_file(
        project_root / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    ):
        raise DataAdmissionError("权限探针不是基于当前实现清单生成")
    return report


def _checkpoint_payload(
    state: str,
    phase: str,
    completed: int,
    expected: int,
    last_key: str,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    """构造不含凭据的进度检查点。"""

    return {
        "model_id": MODEL_ID,
        "state": state,
        "phase": phase,
        "updated_at": now_local().isoformat(),
        "completed_partitions": int(completed),
        "expected_partitions": int(expected),
        "last_key": last_key,
        "failures": failures,
        "credential_persisted": False,
        "future_return_reads": 0,
        "position_impact": 0,
    }


def run_full_acquisition(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
) -> dict[str, Any]:
    """按冻结顺序执行可断点续传的全量原始采集。"""

    probe = _load_probe_report(project_root, protocol)
    if not probe.get("core_strategy_data_acquisition_allowed"):
        raise DataAdmissionError("G0 核心权限探针未通过，禁止全量下载")
    checkpoint_path = project_root / Path(protocol["raw_storage"]["checkpoint"])
    summary_path = project_root / Path(protocol["raw_storage"]["acquisition_summary"])
    start = pd.Timestamp(protocol["protocol"]["start_date"]).date()
    current_time = now_local()
    candidate_end = current_time.date()
    if current_time.timetz().replace(tzinfo=None) < wall_time(20, 15):
        candidate_end = candidate_end - timedelta(days=1)

    acquisition_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def record(api: str, key: str, receipt: Mapping[str, Any], reused: bool) -> None:
        acquisition_records.append(
            {
                "api": api,
                "key": key,
                "row_count": int(receipt["row_count"]),
                "response_sha256": receipt["response_sha256"],
                "receipt_relative_path": receipt.get("receipt_relative_path")
                or str(receipt.get("raw_relative_path", "")).replace(
                    "response.json", "receipt.json"
                ),
                "reused": bool(reused),
            }
        )

    try:
        calendar_params = {
            "exchange": "SSE",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": candidate_end.strftime("%Y%m%d"),
        }
        calendar, receipt, reused = _checked_call_and_capture(
            project_root,
            protocol,
            client,
            "trade_cal",
            calendar_params,
            _full_partition("trade_cal", "full"),
        )
        record("trade_cal", "full", receipt, reused)
        calendar_dates = pd.to_datetime(
            calendar["cal_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        open_mask = calendar["is_open"].astype(str).eq("1") & calendar_dates.notna()
        open_dates = sorted(calendar_dates.loc[open_mask].dt.date.unique().tolist())
        if not open_dates:
            raise DataAdmissionError("交易日历没有开放交易日")
        end = max(open_dates)
        if end < start:
            raise DataAdmissionError("最后开放交易日早于冻结起点")
        print(f"全量采集终点冻结为 {end:%Y-%m-%d}，开放交易日 {len(open_dates)} 个", flush=True)

        fund_params = {
            "ts_code": "510300.SH",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        _, receipt, reused = _checked_call_and_capture(
            project_root,
            protocol,
            client,
            "fund_daily",
            fund_params,
            _full_partition("fund_daily", "full"),
        )
        record("fund_daily", "full", receipt, reused)

        dividend_params = {"ts_code": "510300.SH"}
        _, receipt, reused = _checked_call_and_capture(
            project_root,
            protocol,
            client,
            "fund_div",
            dividend_params,
            _full_partition("fund_div", "full"),
        )
        record("fund_div", "full", receipt, reused)

        master_params = {"exchange": "SSE"}
        _, receipt, reused = _checked_call_and_capture(
            project_root,
            protocol,
            client,
            "opt_basic",
            master_params,
            _full_partition("opt_basic", "full"),
        )
        record("opt_basic", "full", receipt, reused)

        option_limit = int(
            protocol["api_contracts"]["opt_daily"]["documented_single_request_max_rows"]
        )
        for index, trade_date in enumerate(open_dates, start=1):
            key = trade_date.strftime("%Y%m%d")
            params = {"exchange": "SSE", "trade_date": key}
            frame, receipt, reused = _checked_call_and_capture(
                project_root,
                protocol,
                client,
                "opt_daily",
                params,
                _full_partition("opt_daily", f"trade_date={key}"),
                maximum_rows=option_limit,
            )
            returned_dates = set(frame["trade_date"].astype(str).dropna().tolist())
            if returned_dates != {key}:
                raise DataAdmissionError(
                    f"opt_daily {key} 返回日期集合异常：{sorted(returned_dates)}"
                )
            if not frame["exchange"].astype(str).eq("SSE").all():
                raise DataAdmissionError(f"opt_daily {key} 包含非 SSE 行")
            record("opt_daily", key, receipt, reused)
            atomic_json(
                _checkpoint_payload(
                    "RUNNING",
                    "OPT_DAILY",
                    index,
                    len(open_dates),
                    key,
                    failures,
                ),
                checkpoint_path,
            )
            if index == 1 or index % 50 == 0 or index == len(open_dates):
                print(
                    f"opt_daily 进度 {index}/{len(open_dates)}，日期={key}，"
                    f"{'复用' if reused else '新下载'}，rows={len(frame)}",
                    flush=True,
                )

        year_ranges = _calendar_years(start, end)
        shibor_limit = int(
            protocol["api_contracts"]["shibor"]["documented_single_request_max_rows"]
        )
        for index, (year_start, year_end) in enumerate(year_ranges, start=1):
            key = f"year={year_start.year}"
            params = {
                "start_date": year_start.strftime("%Y%m%d"),
                "end_date": year_end.strftime("%Y%m%d"),
            }
            frame, receipt, reused = _checked_call_and_capture(
                project_root,
                protocol,
                client,
                "shibor",
                params,
                _full_partition("shibor", key),
                maximum_rows=shibor_limit,
            )
            record("shibor", key, receipt, reused)
            print(
                f"shibor {index}/{len(year_ranges)}，{key}，"
                f"{'复用' if reused else '新下载'}，rows={len(frame)}",
                flush=True,
            )

        minute_probe_available = bool(probe.get("etf_mins_available"))
        month_ranges = _calendar_months(start, end) if minute_probe_available else []
        minute_limit = int(
            protocol["api_contracts"]["etf_mins"]["documented_single_request_max_rows"]
        )
        for index, (month_start, month_end) in enumerate(month_ranges, start=1):
            key = f"month={month_start:%Y%m}"
            params = {
                "ts_code": "510300.SH",
                "freq": "1min",
                "start_date": f"{month_start:%Y-%m-%d} 00:00:00",
                "end_date": f"{month_end:%Y-%m-%d} 23:59:59",
            }
            frame, receipt, reused = _checked_call_and_capture(
                project_root,
                protocol,
                client,
                "etf_mins",
                params,
                _full_partition("etf_mins", key),
                maximum_rows=minute_limit,
            )
            timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
            if timestamps.isna().any():
                raise DataAdmissionError(f"etf_mins {key} 含无效 trade_time")
            record("etf_mins", key, receipt, reused)
            atomic_json(
                _checkpoint_payload(
                    "RUNNING",
                    "ETF_MINS",
                    index,
                    len(month_ranges),
                    key,
                    failures,
                ),
                checkpoint_path,
            )
            if index == 1 or index % 12 == 0 or index == len(month_ranges):
                print(
                    f"etf_mins 进度 {index}/{len(month_ranges)}，{key}，"
                    f"{'复用' if reused else '新下载'}，rows={len(frame)}",
                    flush=True,
                )
    except Exception as exc:
        failures.append(
            {
                "type": type(exc).__name__,
                "message": sanitize_text(exc),
            }
        )
        atomic_json(
            _checkpoint_payload(
                "BLOCKED",
                "RAW_ACQUISITION",
                len(acquisition_records),
                len(acquisition_records),
                acquisition_records[-1]["key"] if acquisition_records else "",
                failures,
            ),
            checkpoint_path,
        )
        raise

    counts: dict[str, int] = {}
    rows: dict[str, int] = {}
    reused_counts: dict[str, int] = {}
    for item in acquisition_records:
        api = item["api"]
        counts[api] = counts.get(api, 0) + 1
        rows[api] = rows.get(api, 0) + int(item["row_count"])
        reused_counts[api] = reused_counts.get(api, 0) + int(bool(item["reused"]))
    summary = {
        "model_id": MODEL_ID,
        "phase": "B_RAW_DOWNLOAD",
        "state": "PASS_RAW_ACQUISITION_COMPLETE",
        "generated_at": now_local().isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "open_trade_dates": len(open_dates),
        "endpoint_origin": client.origin,
        "partition_counts": counts,
        "row_counts": rows,
        "reused_partition_counts": reused_counts,
        "records": acquisition_records,
        "etf_mins_downloaded": bool(probe.get("etf_mins_available")),
        "future_return_reads": 0,
        "portfolio_evaluation_run": False,
        "credential_persisted": False,
        "position_impact": 0,
    }
    summary["summary_fingerprint"] = sha256_bytes(canonical_json_bytes(summary))
    atomic_json(summary, summary_path)
    atomic_json(
        _checkpoint_payload(
            "COMPLETE",
            "RAW_ACQUISITION",
            len(acquisition_records),
            len(acquisition_records),
            acquisition_records[-1]["key"] if acquisition_records else "",
            failures,
        ),
        checkpoint_path,
    )
    return summary


def black76_price(
    forward: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    call_put: str,
) -> float:
    """计算 Black-76 欧式期权价格。"""

    if forward <= 0 or strike <= 0 or maturity_years <= 0 or volatility <= 0:
        raise ValueError("Black-76 输入必须严格为正")
    option_type = call_put.upper()
    if option_type not in {"C", "P"}:
        raise ValueError("call_put 必须是 C 或 P")
    sigma_root_t = volatility * math.sqrt(maturity_years)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * maturity_years) / sigma_root_t
    d2 = d1 - sigma_root_t
    discount = math.exp(-rate * maturity_years)
    normal_cdf = lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    if option_type == "C":
        return discount * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    return discount * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))


def implied_volatility_black76(
    price: float,
    forward: float,
    strike: float,
    maturity_years: float,
    rate: float,
    call_put: str,
    minimum_volatility: float,
    maximum_volatility: float,
) -> float | None:
    """在冻结区间内用 Brent 求 Black-76 隐含波动率。"""

    if not all(
        math.isfinite(value)
        for value in (price, forward, strike, maturity_years, rate)
    ):
        return None
    if price <= 0 or forward <= 0 or strike <= 0 or maturity_years <= 0:
        return None
    option_type = call_put.upper()
    if option_type not in {"C", "P"}:
        return None
    discount = math.exp(-rate * maturity_years)
    intrinsic = discount * max(
        forward - strike if option_type == "C" else strike - forward,
        0.0,
    )
    upper = discount * (forward if option_type == "C" else strike)
    epsilon = 1e-12 * max(1.0, upper)
    if price < intrinsic - epsilon or price > upper + epsilon:
        return None

    def objective(volatility: float) -> float:
        return black76_price(
            forward,
            strike,
            maturity_years,
            rate,
            volatility,
            option_type,
        ) - price

    lower_value = objective(minimum_volatility)
    upper_value = objective(maximum_volatility)
    if abs(lower_value) <= epsilon:
        return minimum_volatility
    if abs(upper_value) <= epsilon:
        return maximum_volatility
    if lower_value * upper_value > 0:
        return None
    try:
        return float(
            brentq(
                objective,
                minimum_volatility,
                maximum_volatility,
                xtol=1e-12,
                rtol=1e-12,
                maxiter=200,
            )
        )
    except (ValueError, RuntimeError, OverflowError):
        return None


def interpolate_rate(shibor_row: Mapping[str, Any], dte: int) -> float | None:
    """按冻结 SHIBOR 节点线性内插并从百分数转为小数。"""

    nodes = [(7, "1w"), (14, "2w"), (30, "1m"), (90, "3m")]
    if dte < nodes[0][0] or dte > nodes[-1][0]:
        return None
    values: list[float] = []
    for _, field in nodes:
        value = pd.to_numeric(pd.Series([shibor_row.get(field)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return None
        values.append(float(value) / 100.0)
    return float(np.interp(float(dte), [node[0] for node in nodes], values))


def interpolate_total_variance(
    x: Sequence[float],
    implied_volatility: Sequence[float],
    maturity_years: Sequence[float],
    target: float,
) -> float | None:
    """只在相邻点之间对总方差线性内插。"""

    if not (len(x) == len(implied_volatility) == len(maturity_years)) or len(x) < 1:
        return None
    rows = sorted(
        {
            float(x_value): (float(iv_value), float(t_value))
            for x_value, iv_value, t_value in zip(x, implied_volatility, maturity_years)
            if math.isfinite(float(x_value))
            and math.isfinite(float(iv_value))
            and math.isfinite(float(t_value))
            and float(iv_value) > 0
            and float(t_value) > 0
        }.items()
    )
    if not rows or target < rows[0][0] or target > rows[-1][0]:
        return None
    for x_value, (iv_value, t_value) in rows:
        if math.isclose(target, x_value, rel_tol=0.0, abs_tol=1e-12):
            return iv_value * iv_value * t_value
    lower = max((row for row in rows if row[0] < target), default=None)
    upper = min((row for row in rows if row[0] > target), default=None)
    if lower is None or upper is None:
        return None
    x1, (iv1, t1) = lower
    x2, (iv2, t2) = upper
    weight = (target - x1) / (x2 - x1)
    return (iv1 * iv1 * t1) + weight * ((iv2 * iv2 * t2) - (iv1 * iv1 * t1))


def _normalize_call_put(value: Any) -> str | None:
    """规范认购认沽字段。"""

    text = str(value).strip().upper()
    if text in {"C", "CALL", "认购"}:
        return "C"
    if text in {"P", "PUT", "认沽"}:
        return "P"
    return None


def build_contract_map(master: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    """按冻结精确标的映射生成 510300 合约表。"""

    required = protocol["api_contracts"]["opt_basic"]["required_columns"]
    errors = validate_frame_contract(master, required)
    if errors:
        raise DataAdmissionError(f"opt_basic 合约表失败：{'；'.join(errors)}")
    frame = master.copy()
    frame["call_put_normalized"] = frame["call_put"].map(_normalize_call_put)
    for column in ("list_date", "delist_date", "maturity_date"):
        frame[column] = pd.to_datetime(
            frame[column].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.normalize()
    for column in ("exercise_price", "opt_multiplier"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    mapping = protocol["contract_mapping"]
    selected = frame.loc[
        frame["exchange"].astype(str).eq(mapping["exchange"])
        & frame["opt_code"].astype(str).eq(mapping["tushare_target_opt_code"])
        & frame["call_put_normalized"].isin(["C", "P"])
        & frame["exercise_price"].gt(0)
        & frame["opt_multiplier"].gt(0)
        & frame["list_date"].notna()
        & frame["delist_date"].notna()
        & frame["maturity_date"].notna()
    ].copy()
    if selected.empty:
        raise DataAdmissionError("精确 opt_code=OP510300.SH 未映射到任何 SSE 合约")
    if selected["ts_code"].astype(str).duplicated().any():
        duplicates = sorted(
            selected.loc[
                selected["ts_code"].astype(str).duplicated(keep=False), "ts_code"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise DataAdmissionError(f"合约主表 ts_code 重复：{duplicates[:10]}")
    selected["is_standard_multiplier"] = selected["opt_multiplier"].eq(
        float(mapping["surface_standard_multiplier"])
    )
    return selected.sort_values(["list_date", "ts_code"]).reset_index(drop=True)


def _expiry_surface(
    expiry_rows: pd.DataFrame,
    underlying_close: float,
    shibor_row: Mapping[str, Any],
    trade_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """构造单个到期月份的 ATM 与 -5% 总方差。"""

    surface = protocol["surface"]
    dte = int((maturity_date - trade_date).days)
    rate = interpolate_rate(shibor_row, dte)
    if rate is None:
        return None, "NO_VIEW_RATE_TENOR_OUT_OF_RANGE"
    keys = ["exercise_price", "opt_multiplier", "call_put_normalized"]
    if expiry_rows.duplicated(keys, keep=False).any():
        return None, "NO_VIEW_DUPLICATE_PAIR_KEY"
    pivot = expiry_rows.pivot(
        index=["exercise_price", "opt_multiplier"],
        columns="call_put_normalized",
        values="settle",
    ).reset_index()
    if "C" not in pivot.columns or "P" not in pivot.columns:
        return None, "NO_VIEW_INSUFFICIENT_CALL_PUT_PAIRS"
    all_pairs = pivot.dropna(subset=["C", "P"]).copy()
    all_pairs["distance"] = (all_pairs["exercise_price"] - underlying_close).abs()
    parity_pairs = all_pairs.sort_values(["distance", "exercise_price"]).head(
        int(surface["nearest_pairs_to_underlying_close_maximum"])
    )
    if len(parity_pairs) < int(surface["minimum_pairs_per_expiry"]):
        return None, "NO_VIEW_INSUFFICIENT_CALL_PUT_PAIRS"
    maturity_years = dte / 365.0
    parity_pairs["forward_by_strike"] = parity_pairs["exercise_price"] + math.exp(
        rate * maturity_years
    ) * (parity_pairs["C"] - parity_pairs["P"])
    forward = float(parity_pairs["forward_by_strike"].median())
    if not math.isfinite(forward) or forward <= 0:
        return None, "NO_VIEW_INVALID_PARITY_FORWARD"
    mad = float((parity_pairs["forward_by_strike"] - forward).abs().median())
    dispersion = mad / forward
    if dispersion > float(surface["maximum_parity_mad_over_forward"]):
        return None, "NO_VIEW_PARITY_DISPERSION"

    minimum_iv = float(surface["minimum_implied_volatility"])
    maximum_iv = float(surface["maximum_implied_volatility"])
    iv_rows: list[dict[str, float | str]] = []
    no_arbitrage_failures = 0
    for row in all_pairs.itertuples(index=False):
        strike = float(row.exercise_price)
        k = math.log(strike / forward)
        if not (
            float(surface["minimum_log_moneyness"])
            <= k
            <= float(surface["maximum_log_moneyness"])
        ):
            continue
        option_type = "P" if k < 0 else "C"
        price = float(getattr(row, option_type))
        discount = math.exp(-rate * maturity_years)
        intrinsic = discount * max(
            forward - strike if option_type == "C" else strike - forward,
            0.0,
        )
        upper = discount * (forward if option_type == "C" else strike)
        epsilon = 1e-12 * max(1.0, upper)
        if price < intrinsic - epsilon or price > upper + epsilon:
            no_arbitrage_failures += 1
            continue
        iv = implied_volatility_black76(
            price,
            forward,
            strike,
            maturity_years,
            rate,
            option_type,
            minimum_iv,
            maximum_iv,
        )
        if iv is None:
            continue
        iv_rows.append(
            {
                "k": k,
                "iv": iv,
                "maturity_years": maturity_years,
                "option_type": option_type,
            }
        )
    if no_arbitrage_failures:
        return None, "NO_VIEW_NO_ARBITRAGE_BOUND_FAILED"
    if len(iv_rows) < int(surface["minimum_valid_strikes_per_expiry"]):
        return None, "NO_VIEW_INSUFFICIENT_VALID_STRIKES"
    iv_frame = pd.DataFrame(iv_rows).sort_values("k")
    if not (iv_frame["k"].min() < 0 < iv_frame["k"].max()):
        return None, "NO_VIEW_NO_ATM_TWO_SIDED_BRACKET"
    x = iv_frame["k"].tolist()
    iv_values = iv_frame["iv"].tolist()
    maturity_values = iv_frame["maturity_years"].tolist()
    atm_variance = interpolate_total_variance(x, iv_values, maturity_values, 0.0)
    put5_variance = interpolate_total_variance(x, iv_values, maturity_values, -0.05)
    if atm_variance is None:
        return None, "NO_VIEW_NO_ATM_STRIKE_BRACKET"
    if put5_variance is None:
        return None, "NO_VIEW_NO_PUT5_STRIKE_BRACKET"
    return (
        {
            "maturity_date": maturity_date,
            "dte": dte,
            "maturity_years": maturity_years,
            "rate": rate,
            "forward": forward,
            "parity_mad_over_forward": dispersion,
            "parity_pair_count": int(len(parity_pairs)),
            "surface_pair_count": int(len(all_pairs)),
            "valid_strike_count": int(len(iv_frame)),
            "atm_total_variance": float(atm_variance),
            "put5_total_variance": float(put5_variance),
        },
        "PASS_EXPIRY_SURFACE",
    )


def _maturity_total_variance(
    expiry_frame: pd.DataFrame, target_days: int, field: str
) -> float | None:
    """只在合法期限包围内插目标总方差。"""

    valid = expiry_frame.loc[expiry_frame[field].notna(), ["dte", field]].copy()
    if valid.empty:
        return None
    exact = valid.loc[valid["dte"].eq(target_days)]
    if not exact.empty:
        return float(exact.iloc[0][field])
    lower = valid.loc[valid["dte"].lt(target_days)].sort_values("dte").tail(1)
    upper = valid.loc[valid["dte"].gt(target_days)].sort_values("dte").head(1)
    if lower.empty or upper.empty:
        return None
    d1 = float(lower.iloc[0]["dte"])
    d2 = float(upper.iloc[0]["dte"])
    w1 = float(lower.iloc[0][field])
    w2 = float(upper.iloc[0][field])
    return w1 + (target_days - d1) / (d2 - d1) * (w2 - w1)


def build_daily_surface(
    trade_date: pd.Timestamp,
    day_options: pd.DataFrame,
    underlying_close: float,
    shibor_row: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """构造单日曲面和完整质量状态。"""

    ledger: dict[str, Any] = {
        "trade_date": trade_date,
        "state": "NO_VIEW",
        "total_target_rows": int(len(day_options)),
        "eligible_contract_rows": 0,
        "candidate_expiries": 0,
        "valid_expiries": 0,
        "expiry_state_counts_json": "{}",
    }
    if shibor_row is None:
        ledger["state"] = "NO_VIEW_MISSING_SAME_DAY_SHIBOR"
        return ledger, None
    if not math.isfinite(underlying_close) or underlying_close <= 0:
        ledger["state"] = "NO_VIEW_INVALID_UNDERLYING_CLOSE"
        return ledger, None
    frame = day_options.copy()
    for column in ("settle", "vol", "oi", "exercise_price", "opt_multiplier"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dte"] = (frame["maturity_date"] - trade_date).dt.days
    surface = protocol["surface"]
    standard_multiplier = float(protocol["contract_mapping"]["surface_standard_multiplier"])
    frame = frame.loc[
        frame["is_standard_multiplier"].eq(True)
        & frame["opt_multiplier"].eq(standard_multiplier)
        & frame["settle"].gt(0)
        & frame["vol"].gt(0)
        & frame["oi"].gt(0)
        & frame["dte"].between(
            int(surface["single_contract_day_requirements"]["minimum_dte_calendar_days"]),
            int(surface["single_contract_day_requirements"]["maximum_dte_calendar_days"]),
        )
        & frame["list_date"].le(trade_date)
        & frame["delist_date"].ge(trade_date)
    ].copy()
    ledger["eligible_contract_rows"] = int(len(frame))
    if frame.empty:
        ledger["state"] = "NO_VIEW_NO_ELIGIBLE_CONTRACTS"
        return ledger, None
    duplicate_keys = ["ts_code", "trade_date"]
    if frame.duplicated(duplicate_keys, keep=False).any():
        ledger["state"] = "NO_VIEW_DUPLICATE_TS_CODE_TRADE_DATE"
        return ledger, None
    expiry_results: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for maturity_date, expiry_rows in frame.groupby("maturity_date", sort=True):
        ledger["candidate_expiries"] += 1
        expiry, state = _expiry_surface(
            expiry_rows,
            underlying_close,
            shibor_row,
            trade_date,
            pd.Timestamp(maturity_date),
            protocol,
        )
        state_counts[state] = state_counts.get(state, 0) + 1
        if expiry is not None:
            expiry_results.append(expiry)
    ledger["valid_expiries"] = int(len(expiry_results))
    ledger["expiry_state_counts_json"] = json.dumps(
        state_counts, ensure_ascii=False, sort_keys=True
    )
    if not expiry_results:
        ledger["state"] = "NO_VIEW_NO_VALID_EXPIRY"
        return ledger, None
    expiry_frame = pd.DataFrame(expiry_results).sort_values("dte")
    w30_atm = _maturity_total_variance(expiry_frame, 30, "atm_total_variance")
    w30_put5 = _maturity_total_variance(expiry_frame, 30, "put5_total_variance")
    w60_atm = _maturity_total_variance(expiry_frame, 60, "atm_total_variance")
    if w30_atm is None or w30_put5 is None or w60_atm is None:
        ledger["state"] = "NO_VIEW_NO_MATURITY_BRACKET"
        return ledger, None
    if min(w30_atm, w30_put5, w60_atm) <= 0:
        ledger["state"] = "NO_VIEW_NONPOSITIVE_TOTAL_VARIANCE"
        return ledger, None
    iv30_atm = math.sqrt(w30_atm / (30.0 / 365.0))
    iv30_put5 = math.sqrt(w30_put5 / (30.0 / 365.0))
    iv60_atm = math.sqrt(w60_atm / (60.0 / 365.0))
    ledger["state"] = "PASS_VALID_SURFACE"
    surface_row = {
        "trade_date": trade_date,
        "iv30_atm": iv30_atm,
        "iv30_put5": iv30_put5,
        "iv60_atm": iv60_atm,
        "put_skew30": iv30_put5 - iv30_atm,
        "term_inversion30_60": iv30_atm - iv60_atm,
        "valid_expiries": int(len(expiry_results)),
    }
    return ledger, surface_row


def _read_required_full_capture(
    project_root: Path,
    api_name: str,
    qualifier: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """读取全量采集的最新有效分区。"""

    found = find_latest_valid_capture(
        project_root, _full_partition(api_name, qualifier), api_name
    )
    if found is None:
        raise DataAdmissionError(f"缺少或损坏的全量采集：{api_name}/{qualifier}")
    return found


def _duplicate_count(frame: pd.DataFrame, keys: Sequence[str]) -> int:
    """返回参与重复主键的行数。"""

    if any(key not in frame.columns for key in keys):
        return len(frame)
    return int(frame.duplicated(list(keys), keep=False).sum())


def _load_partition_series(
    project_root: Path,
    api_name: str,
    qualifier_prefix: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[str]]:
    """读取一个按日期、年或月分区的最新有效采集。"""

    base = _full_partition(api_name, "__QUALIFIER__").parent
    absolute = project_root / base
    receipts: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    invalid: list[str] = []
    if not absolute.is_dir():
        return receipts, frames, [f"目录缺失：{base.as_posix()}"]
    for partition in sorted(
        path for path in absolute.iterdir() if path.is_dir() and path.name.startswith(qualifier_prefix)
    ):
        relative = partition.relative_to(project_root)
        found = find_latest_valid_capture(project_root, relative, api_name)
        if found is None:
            invalid.append(partition.name)
        else:
            receipt, frame = found
            receipts.append(receipt)
            frames.append(frame)
    return receipts, frames, invalid


def _official_sse_cross_check(project_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """抓取上交所公告并验证冻结的标的身份事实。"""

    url = protocol["source_contract"]["documentation"]["sse_510300_option_listing"]
    output = project_root / Path(
        protocol["contract_mapping"]["official_cross_check_receipt"]
    )
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            raw_path = project_root / Path(existing["raw_relative_path"])
            if (
                existing.get("url") == url
                and existing.get("pass") is True
                and raw_path.is_file()
                and sha256_file(raw_path) == existing.get("response_sha256")
            ):
                return existing
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    response = requests.get(
        url,
        headers={"User-Agent": f"{MODEL_ID}/1.0 official-source-check"},
        timeout=(15, 60),
    )
    raw = bytes(response.content)
    text = response.text
    assertions = {
        "http_200": response.status_code == 200,
        "contains_510300": "510300" in text,
        "contains_listing_date": "2019年12月23日" in text,
        "contains_underlying_name": "华泰柏瑞沪深300" in text,
    }
    raw_path = (
        project_root
        / "data/raw/official/sse/510300_option_listing"
        / f"retrieved_at={now_local().strftime('%Y%m%dT%H%M%S%f%z')}"
        / "listing_page.html"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=False)
    raw_path.write_bytes(raw)
    report = {
        "source": "上海证券交易所",
        "url": url,
        "retrieved_at": now_local().isoformat(),
        "http_status": int(response.status_code),
        "response_bytes": len(raw),
        "response_sha256": sha256_bytes(raw),
        "raw_relative_path": raw_path.relative_to(project_root).as_posix(),
        "assertions": assertions,
        "pass": all(assertions.values()),
    }
    atomic_json(report, output)
    return report


def _execution_window_coverage(
    minute_frames: Sequence[pd.DataFrame], open_dates: Sequence[pd.Timestamp]
) -> dict[str, Any]:
    """计算冻结 09:35—09:39 五分钟窗口的完整覆盖率。"""

    if not minute_frames:
        return {
            "available": False,
            "eligible_open_dates": len(open_dates),
            "complete_dates": 0,
            "coverage": 0.0,
        }
    minutes = pd.concat(minute_frames, ignore_index=True)
    timestamps = pd.to_datetime(minutes["trade_time"], errors="coerce")
    minutes = minutes.assign(_timestamp=timestamps).loc[timestamps.notna()].copy()
    minutes["_date"] = minutes["_timestamp"].dt.normalize()
    minutes["_clock"] = minutes["_timestamp"].dt.strftime("%H:%M")
    wanted = {"09:35", "09:36", "09:37", "09:38", "09:39"}
    filtered = minutes.loc[
        minutes["_clock"].isin(wanted)
        & pd.to_numeric(minutes["amount"], errors="coerce").gt(0)
        & pd.to_numeric(minutes["vol"], errors="coerce").gt(0)
        & pd.to_numeric(minutes["close"], errors="coerce").gt(0)
    ]
    counts = filtered.groupby("_date")["_clock"].nunique()
    open_set = {pd.Timestamp(value).normalize() for value in open_dates}
    complete = {pd.Timestamp(value).normalize() for value in counts.loc[counts.eq(5)].index}
    complete_in_scope = open_set & complete
    denominator = len(open_set)
    return {
        "available": True,
        "eligible_open_dates": denominator,
        "complete_dates": len(complete_in_scope),
        "coverage": len(complete_in_scope) / denominator if denominator else 0.0,
        "duplicate_ts_code_trade_time_rows": _duplicate_count(
            minutes, ["ts_code", "trade_time"]
        ),
    }


def run_data_adjudication(
    project_root: Path,
    protocol: Mapping[str, Any],
    replay_expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """只读取允许的数据，生成 G0/G1 数据准入裁决。"""

    summary_path = project_root / Path(protocol["raw_storage"]["acquisition_summary"])
    if not summary_path.is_file():
        raise DataAdmissionError("缺少完成态 acquisition_summary.json")
    acquisition_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if acquisition_summary.get("state") != "PASS_RAW_ACQUISITION_COMPLETE":
        raise DataAdmissionError("原始采集未完成")

    calendar_receipt, calendar = _read_required_full_capture(
        project_root, "trade_cal", "full"
    )
    fund_receipt, fund_daily = _read_required_full_capture(
        project_root, "fund_daily", "full"
    )
    dividend_receipt, fund_div = _read_required_full_capture(
        project_root, "fund_div", "full"
    )
    master_receipt, opt_basic = _read_required_full_capture(
        project_root, "opt_basic", "full"
    )
    option_receipts, option_frames, invalid_option_partitions = _load_partition_series(
        project_root, "opt_daily", "trade_date="
    )
    shibor_receipts, shibor_frames, invalid_shibor_partitions = _load_partition_series(
        project_root, "shibor", "year="
    )
    minute_receipts, minute_frames, invalid_minute_partitions = _load_partition_series(
        project_root, "etf_mins", "month="
    )

    calendar_dates = pd.to_datetime(
        calendar["cal_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    open_dates = sorted(
        calendar_dates.loc[calendar["is_open"].astype(str).eq("1") & calendar_dates.notna()]
        .unique()
        .tolist()
    )
    open_keys = {pd.Timestamp(value).strftime("%Y%m%d") for value in open_dates}
    captured_option_keys = {
        str(receipt["request_parameters"].get("trade_date", ""))
        for receipt in option_receipts
    }
    missing_option_dates = sorted(open_keys - captured_option_keys)
    unexpected_option_dates = sorted(captured_option_keys - open_keys)
    if open_dates:
        first_open = pd.Timestamp(open_dates[0]).date()
        last_open = pd.Timestamp(open_dates[-1]).date()
        expected_shibor_years = {
            f"year={year_start.year}"
            for year_start, _ in _calendar_years(first_open, last_open)
        }
        expected_minute_months = {
            f"month={month_start:%Y%m}"
            for month_start, _ in _calendar_months(first_open, last_open)
        }
    else:
        expected_shibor_years = set()
        expected_minute_months = set()
    captured_shibor_years = {
        f"year={pd.Timestamp(str(receipt['request_parameters']['start_date'])).year}"
        for receipt in shibor_receipts
    }
    captured_minute_months = {
        f"month={pd.Timestamp(str(receipt['request_parameters']['start_date'])).strftime('%Y%m')}"
        for receipt in minute_receipts
    }
    missing_shibor_years = sorted(expected_shibor_years - captured_shibor_years)
    minutes_expected = bool(acquisition_summary.get("etf_mins_downloaded"))
    missing_minute_months = (
        sorted(expected_minute_months - captured_minute_months)
        if minutes_expected
        else []
    )

    contract_map = build_contract_map(opt_basic, protocol)
    contract_output = project_root / Path(protocol["contract_mapping"]["output"])
    atomic_parquet(contract_map, contract_output)
    official = _official_sse_cross_check(project_root, protocol)

    target_codes = set(contract_map["ts_code"].astype(str))
    target_option_frames: list[pd.DataFrame] = []
    all_option_duplicate_rows = 0
    all_option_rows = 0
    for frame in option_frames:
        all_option_rows += len(frame)
        all_option_duplicate_rows += _duplicate_count(frame, ["ts_code", "trade_date"])
        selected = frame.loc[frame["ts_code"].astype(str).isin(target_codes)].copy()
        if not selected.empty:
            target_option_frames.append(selected)
    option_target = (
        pd.concat(target_option_frames, ignore_index=True)
        if target_option_frames
        else pd.DataFrame(columns=protocol["api_contracts"]["opt_daily"]["required_columns"])
    )
    option_target["trade_date"] = pd.to_datetime(
        option_target["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    option_target = option_target.merge(
        contract_map,
        on="ts_code",
        how="left",
        suffixes=("", "_master"),
        validate="many_to_one",
    )

    fund = fund_daily.copy()
    fund["trade_date"] = pd.to_datetime(
        fund["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    fund["close"] = pd.to_numeric(fund["close"], errors="coerce")
    fund_by_date = fund.set_index("trade_date", drop=False)
    shibor = pd.concat(shibor_frames, ignore_index=True) if shibor_frames else pd.DataFrame()
    shibor_duplicate_rows = _duplicate_count(shibor, ["date"]) if not shibor.empty else 0
    if not shibor.empty:
        shibor["date"] = pd.to_datetime(
            shibor["date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.normalize()
        shibor = shibor.sort_values("date").drop_duplicates("date", keep="last")
        shibor_by_date = shibor.set_index("date", drop=False)
    else:
        shibor_by_date = pd.DataFrame()

    ledgers: list[dict[str, Any]] = []
    surfaces: list[dict[str, Any]] = []
    grouped_options = {
        pd.Timestamp(key): value.copy()
        for key, value in option_target.groupby("trade_date", sort=False)
        if pd.notna(key)
    }
    for index, value in enumerate(open_dates, start=1):
        trade_date = pd.Timestamp(value).normalize()
        if trade_date not in fund_by_date.index:
            ledgers.append(
                {
                    "trade_date": trade_date,
                    "state": "NO_VIEW_MISSING_FUND_DAILY",
                    "total_target_rows": 0,
                    "eligible_contract_rows": 0,
                    "candidate_expiries": 0,
                    "valid_expiries": 0,
                    "expiry_state_counts_json": "{}",
                }
            )
            continue
        fund_row = fund_by_date.loc[trade_date]
        if isinstance(fund_row, pd.DataFrame):
            fund_row = fund_row.iloc[0]
        shibor_row: Mapping[str, Any] | None = None
        if isinstance(shibor_by_date, pd.DataFrame) and not shibor_by_date.empty:
            if trade_date in shibor_by_date.index:
                row = shibor_by_date.loc[trade_date]
                shibor_row = row.iloc[0].to_dict() if isinstance(row, pd.DataFrame) else row.to_dict()
        ledger, surface_row = build_daily_surface(
            trade_date,
            grouped_options.get(trade_date, option_target.iloc[0:0].copy()),
            float(fund_row["close"]),
            shibor_row,
            protocol,
        )
        ledgers.append(ledger)
        if surface_row is not None:
            surfaces.append(surface_row)
        if index == 1 or index % 250 == 0 or index == len(open_dates):
            print(
                f"曲面准入进度 {index}/{len(open_dates)}，有效日={len(surfaces)}",
                flush=True,
            )

    ledger_frame = pd.DataFrame(ledgers).sort_values("trade_date").reset_index(drop=True)
    surface_frame = pd.DataFrame(surfaces)
    if not surface_frame.empty:
        surface_frame = surface_frame.sort_values("trade_date").reset_index(drop=True)
    atomic_parquet(
        ledger_frame,
        project_root / Path(protocol["surface"]["outputs"]["quality_ledger"]),
    )
    atomic_parquet(
        surface_frame,
        project_root / Path(protocol["surface"]["outputs"]["daily_surface"]),
    )

    valid_mask = ledger_frame["state"].eq("PASS_VALID_SURFACE")
    valid_days = int(valid_mask.sum())
    total_days = int(len(ledger_frame))
    overall_coverage = valid_days / total_days if total_days else 0.0
    ledger_frame["year"] = pd.to_datetime(ledger_frame["trade_date"]).dt.year
    start_year = pd.Timestamp(open_dates[0]).year if open_dates else 0
    end_year = pd.Timestamp(open_dates[-1]).year if open_dates else 0
    complete_years = [year for year in sorted(ledger_frame["year"].unique()) if start_year < year < end_year]
    yearly = []
    for year in sorted(ledger_frame["year"].unique()):
        year_frame = ledger_frame.loc[ledger_frame["year"].eq(year)]
        count = int(len(year_frame))
        valid = int(year_frame["state"].eq("PASS_VALID_SURFACE").sum())
        yearly.append(
            {
                "year": int(year),
                "complete_calendar_year": int(year) in complete_years,
                "open_days": count,
                "valid_surface_days": valid,
                "coverage": valid / count if count else 0.0,
            }
        )

    execution_coverage = _execution_window_coverage(minute_frames, open_dates)
    duplicate_checks = {
        "trade_cal": _duplicate_count(calendar, ["exchange", "cal_date"]),
        "fund_daily": _duplicate_count(fund_daily, ["ts_code", "trade_date"]),
        "fund_div": _duplicate_count(
            fund_div,
            protocol["api_contracts"]["fund_div"]["primary_key"],
        ),
        "opt_basic": _duplicate_count(opt_basic, ["ts_code"]),
        "opt_daily_all_sse": int(all_option_duplicate_rows),
        "shibor": shibor_duplicate_rows,
        "etf_mins": int(execution_coverage.get("duplicate_ts_code_trade_time_rows", 0)),
    }
    core_raw_hashes_complete = not (
        invalid_option_partitions
        or invalid_shibor_partitions
        or missing_option_dates
        or missing_shibor_years
    )
    minute_raw_hashes_complete = not (
        invalid_minute_partitions or missing_minute_months
    )
    core_duplicate_checks = {
        key: value for key, value in duplicate_checks.items() if key != "etf_mins"
    }
    core_g0_checks = {
        "acquisition_complete": True,
        "all_open_dates_have_opt_daily": not missing_option_dates,
        "no_unexpected_opt_daily_dates": not unexpected_option_dates,
        "all_expected_shibor_years_present": not missing_shibor_years,
        "raw_hashes_complete": core_raw_hashes_complete,
        "all_primary_keys_unique": all(
            value == 0 for value in core_duplicate_checks.values()
        ),
        "official_510300_underlying_cross_check": bool(official["pass"]),
        "exact_tushare_contract_mapping_nonempty": not contract_map.empty,
    }
    replay_facts = {
        "open_dates": total_days,
        "valid_surface_days": valid_days,
        "overall_coverage": overall_coverage,
        "yearly": yearly,
        "surface_state_counts": ledger_frame["state"].value_counts().sort_index().to_dict(),
        "duplicate_checks": duplicate_checks,
        "missing_option_dates": missing_option_dates,
        "unexpected_option_dates": unexpected_option_dates,
        "missing_shibor_years": missing_shibor_years,
        "missing_minute_months": missing_minute_months,
        "contract_rows": int(len(contract_map)),
        "standard_contract_rows": int(contract_map["is_standard_multiplier"].sum()),
        "all_sse_option_rows": int(all_option_rows),
        "target_option_rows": int(len(option_target)),
        "input_response_hashes": sorted(
            [
                calendar_receipt["response_sha256"],
                fund_receipt["response_sha256"],
                dividend_receipt["response_sha256"],
                master_receipt["response_sha256"],
                *[receipt["response_sha256"] for receipt in option_receipts],
                *[receipt["response_sha256"] for receipt in shibor_receipts],
                *[receipt["response_sha256"] for receipt in minute_receipts],
            ]
        ),
    }
    decision_fingerprint = sha256_bytes(canonical_json_bytes(replay_facts))
    replay_pass = (
        replay_expected_fingerprint == decision_fingerprint
        if replay_expected_fingerprint is not None
        else False
    )
    replay_state = (
        "PASS_CLEAN_NEW_PROCESS_REPLAY"
        if replay_pass
        else (
            "FAIL_CLEAN_NEW_PROCESS_REPLAY"
            if replay_expected_fingerprint is not None
            else "PENDING_CLEAN_NEW_PROCESS_REPLAY"
        )
    )
    core_g0_checks["clean_new_process_replay"] = replay_pass

    g1 = protocol["gates"]["g1"]
    complete_year_coverage_pass = all(
        item["coverage"] >= float(g1["minimum_each_complete_calendar_year_coverage"])
        for item in yearly
        if item["complete_calendar_year"]
    )
    g1_checks = {
        "minimum_valid_surface_days": valid_days >= int(g1["minimum_valid_surface_days"]),
        "minimum_overall_valid_coverage": overall_coverage
        >= float(g1["minimum_overall_valid_coverage"]),
        "minimum_each_complete_calendar_year_coverage": complete_year_coverage_pass,
        "legal_30d_and_60d_interpolation": bool(
            not surface_frame.empty
            and surface_frame[["iv30_atm", "iv30_put5", "iv60_atm"]].notna().all().all()
        ),
        "no_arbitrage_pass_for_every_admitted_surface": True,
        "put_call_parity_pass_for_every_admitted_surface": True,
        "missing_preserved_as_no_view": not ledger_frame.loc[
            ~valid_mask, "state"
        ].astype(str).eq("").any(),
    }
    g0_prediction_pass = all(core_g0_checks.values())
    g0_final_execution_pass = bool(
        g0_prediction_pass
        and execution_coverage["available"]
        and minute_raw_hashes_complete
        and duplicate_checks["etf_mins"] == 0
        and execution_coverage["coverage"]
        >= float(
            protocol["gates"]["g0"]["final_portfolio_additional_requirements"][
                "execution_window_coverage_minimum"
            ]
        )
    )
    g1_pass = all(g1_checks.values())
    if replay_expected_fingerprint is None:
        state = "PENDING_G0_CLEAN_PROCESS_REPLAY"
    elif not replay_pass:
        state = "BLOCKED_G0_CLEAN_PROCESS_REPLAY_MISMATCH"
    elif not g0_prediction_pass:
        state = "BLOCKED_G0_DATA_ADMISSION"
    elif not g1_pass:
        state = protocol["no_rescue"]["terminal_core_failure_state"]
    else:
        state = "PASS_G0_G1_DATA_ADMISSION_ONLY"

    report = {
        "model_id": MODEL_ID,
        "phase": "C_G0_G1_DATA_ADMISSION",
        "state": state,
        "generated_at": now_local().isoformat(),
        "decision_fingerprint": decision_fingerprint,
        "replay_state": replay_state,
        "g0": {
            "prediction_data_admission_pass": g0_prediction_pass,
            "final_execution_data_admission_pass": g0_final_execution_pass,
            "checks": core_g0_checks,
            "duplicate_primary_key_rows": duplicate_checks,
            "execution_window": execution_coverage,
            "missing_option_dates": missing_option_dates,
            "unexpected_option_dates": unexpected_option_dates,
            "missing_shibor_years": missing_shibor_years,
            "missing_minute_months": missing_minute_months,
            "invalid_option_partitions": invalid_option_partitions,
            "invalid_shibor_partitions": invalid_shibor_partitions,
            "invalid_minute_partitions": invalid_minute_partitions,
        },
        "g1": {
            "pass": g1_pass,
            "checks": g1_checks,
            "valid_surface_days": valid_days,
            "open_trade_days": total_days,
            "overall_valid_coverage": overall_coverage,
            "yearly_coverage": yearly,
            "surface_state_counts": ledger_frame["state"].value_counts().sort_index().to_dict(),
        },
        "data_counts": {
            "all_sse_option_rows": int(all_option_rows),
            "target_option_rows": int(len(option_target)),
            "contract_rows": int(len(contract_map)),
            "standard_contract_rows": int(contract_map["is_standard_multiplier"].sum()),
            "fund_daily_rows": int(len(fund_daily)),
            "fund_div_rows": int(len(fund_div)),
            "shibor_rows": int(len(shibor)),
        },
        "official_underlying_cross_check": official,
        "outputs": {
            "contract_map": protocol["contract_mapping"]["output"],
            "quality_ledger": protocol["surface"]["outputs"]["quality_ledger"],
            "daily_surface": protocol["surface"]["outputs"]["daily_surface"],
        },
        "future_return_reads": 0,
        "future_labels_created": False,
        "model_training_run": False,
        "portfolio_evaluation_run": False,
        "net_value_or_sharpe_generated": False,
        "position_impact": 0,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    output_path = project_root / Path(protocol["planned_outputs"]["g0_g1_adjudication"])
    atomic_json(report, output_path)
    return report
