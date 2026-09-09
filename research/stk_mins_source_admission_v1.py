"""510300 ``stk_mins`` 历史一分钟数据的冻结采集与来源准入。"""

from __future__ import annotations

import getpass
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


DATA_MODEL_ID = "510300_STK_MINS_SOURCE_ADMISSION_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_RELATIVE_PATH = Path("config/510300_stk_mins_source_admission_v1.yaml")
PROTOCOL_MANIFEST_RELATIVE_PATH = Path(
    "config/510300_stk_mins_source_admission_v1_manifest.json"
)
IMPLEMENTATION_MANIFEST_RELATIVE_PATH = Path(
    "config/510300_stk_mins_source_admission_v1_implementation_manifest.json"
)


class ProtocolIntegrityError(RuntimeError):
    """冻结协议或实现的哈希不匹配。"""


class ApiCallError(RuntimeError):
    """供应商调用失败。"""


class SourceAdmissionError(RuntimeError):
    """来源数据不满足冻结契约。"""


def now_local() -> datetime:
    """返回上海时区当前时间。"""

    return datetime.now(TIMEZONE)


def sha256_bytes(content: bytes) -> str:
    """计算字节串SHA-256。"""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """生成确定性的JSON字节。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(payload: Any, path: Path) -> None:
    """原子写入UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    """原子写入UTF-8文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sanitize_text(value: Any, secret: str = "", limit: int = 1200) -> str:
    """从错误文本移除临时凭据。"""

    text = str(value)
    if secret:
        text = text.replace(secret, "[凭据已隐藏]")
    text = re.sub(
        r"(?i)(x-api-key|token|authorization)\s*[:=]\s*[^\s,;}]+",
        r"\1=[凭据已隐藏]",
        text,
    )
    return text[:limit]


def endpoint_origin(endpoint: str) -> str:
    """只保留HTTPS端点的scheme和host。"""

    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("来源端点必须是合法HTTPS地址")
    return f"{parsed.scheme}://{parsed.netloc}"


def resolve_ephemeral_token(environment_name: str) -> str:
    """从环境变量或隐藏交互输入读取56位临时凭据。"""

    token = os.getenv(environment_name, "").strip()
    if not token:
        token = getpass.getpass("请输入56位临时数据API Key（输入不会显示）：").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{56}", token):
        raise ValueError("临时数据API Key必须是56位字母或数字")
    return token


def load_protocol(project_root: Path) -> dict[str, Any]:
    """读取并校验来源协议的身份。"""

    path = project_root / PROTOCOL_RELATIVE_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolIntegrityError("来源协议YAML顶层必须是对象")
    identity = payload.get("data_protocol", {})
    if identity.get("data_model_id") != DATA_MODEL_ID:
        raise ProtocolIntegrityError("来源协议DATA_MODEL_ID不匹配")
    if identity.get("state") != "SOURCE_PROTOCOL_FROZEN_BEFORE_BULK_ACQUISITION":
        raise ProtocolIntegrityError("来源协议冻结状态不正确")
    return payload


def _verify_hash_mapping(project_root: Path, mapping: Mapping[str, str]) -> None:
    """验证相对路径到哈希的映射。"""

    for relative, expected in mapping.items():
        path = project_root / Path(relative)
        if not path.is_file():
            raise ProtocolIntegrityError(f"冻结文件缺失：{relative}")
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise ProtocolIntegrityError(
                f"冻结文件哈希漂移：{relative}；expected={expected}；actual={actual}"
            )


def verify_protocol_manifest(project_root: Path) -> dict[str, Any]:
    """验证来源协议清单，不要求实现清单已经存在。"""

    path = project_root / PROTOCOL_MANIFEST_RELATIVE_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("data_model_id") != DATA_MODEL_ID:
        raise ProtocolIntegrityError("来源协议清单DATA_MODEL_ID不匹配")
    if manifest.get("freeze_state") != "SOURCE_PROTOCOL_FROZEN_BEFORE_BULK_ACQUISITION":
        raise ProtocolIntegrityError("来源协议清单冻结状态不正确")
    _verify_hash_mapping(project_root, manifest.get("frozen_files", {}))
    return manifest


def verify_frozen_manifests(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """在联网采集前验证来源协议和实现清单。"""

    protocol_manifest = verify_protocol_manifest(project_root)
    implementation_path = project_root / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    if not implementation_path.is_file():
        raise ProtocolIntegrityError("缺少来源采集实现冻结清单")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    if implementation.get("data_model_id") != DATA_MODEL_ID:
        raise ProtocolIntegrityError("来源实现清单DATA_MODEL_ID不匹配")
    if (
        implementation.get("freeze_state")
        != "IMPLEMENTATION_FROZEN_BEFORE_CREDENTIALLED_BULK_ACQUISITION"
    ):
        raise ProtocolIntegrityError("来源实现清单冻结状态不正确")
    expected = implementation.get("protocol_manifest_sha256")
    actual = sha256_file(project_root / PROTOCOL_MANIFEST_RELATIVE_PATH)
    if expected != actual:
        raise ProtocolIntegrityError("来源实现清单引用的协议清单哈希不匹配")
    _verify_hash_mapping(project_root, implementation.get("implementation_files", {}))
    return protocol_manifest, implementation


class RateLimiter:
    """保证相邻请求起点不短于冻结间隔。"""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self._last_started: float | None = None

    def wait(self) -> None:
        """必要时等待。"""

        current = time.monotonic()
        if self._last_started is not None:
            remaining = self.minimum_interval_seconds - (current - self._last_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_started = time.monotonic()


@dataclass(frozen=True)
class ApiResult:
    """一次供应商请求的内存结果。"""

    api_name: str
    request_parameters: dict[str, Any]
    request_started_at: str
    response_received_at: str
    http_status: int | None
    success: bool
    business_code: str
    error_message: str
    raw_bytes: bytes
    response_sha256: str
    frame: pd.DataFrame


class TushareProxyClient:
    """只读Tushare兼容代理客户端。"""

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
            raise ValueError("最大请求次数必须至少为1")

    def _failure(
        self,
        api_name: str,
        parameters: dict[str, Any],
        started_at: str,
        http_status: int | None,
        code: str,
        message: Any,
        raw_bytes: bytes,
    ) -> ApiResult:
        return ApiResult(
            api_name=api_name,
            request_parameters=parameters,
            request_started_at=started_at,
            response_received_at=now_local().isoformat(),
            http_status=http_status,
            success=False,
            business_code=str(code),
            error_message=sanitize_text(message, self._token),
            raw_bytes=raw_bytes,
            response_sha256=sha256_bytes(raw_bytes),
            frame=pd.DataFrame(),
        )

    def call(self, api_name: str, parameters: Mapping[str, Any]) -> ApiResult:
        """调用一次接口，只重试冻结的传输类错误。"""

        clean_parameters = {str(key): value for key, value in parameters.items()}
        payload = {"api_name": api_name, "params": clean_parameters}
        headers = {
            "x-api-key": self._token,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"{DATA_MODEL_ID}/1.0 source-admission",
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
                    last_result = self._failure(
                        api_name,
                        clean_parameters,
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
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        f"HTTP_{response.status_code}",
                        response.text,
                        raw_bytes,
                    )
                try:
                    body = response.json()
                except ValueError as exc:
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        "INVALID_JSON_RESPONSE",
                        exc,
                        raw_bytes,
                    )
                if not isinstance(body, dict):
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        "INVALID_BODY",
                        "响应顶层不是对象",
                        raw_bytes,
                    )
                code = str(body.get("code", "MISSING_CODE"))
                if code not in {"0", "0.0"}:
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        code,
                        body.get("msg", ""),
                        raw_bytes,
                    )
                data = body.get("data")
                if not isinstance(data, dict):
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        "INVALID_DATA_OBJECT",
                        "成功响应缺少data对象",
                        raw_bytes,
                    )
                fields = data.get("fields")
                items = data.get("items")
                if not isinstance(fields, list) or not isinstance(items, list):
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        "INVALID_TABULAR_DATA",
                        "data.fields或data.items不是列表",
                        raw_bytes,
                    )
                try:
                    frame = pd.DataFrame(items, columns=[str(value) for value in fields])
                except Exception as exc:
                    return self._failure(
                        api_name,
                        clean_parameters,
                        started_at,
                        int(response.status_code),
                        "TABULAR_DECODE_FAILED",
                        exc,
                        raw_bytes,
                    )
                return ApiResult(
                    api_name=api_name,
                    request_parameters=clean_parameters,
                    request_started_at=started_at,
                    response_received_at=now_local().isoformat(),
                    http_status=int(response.status_code),
                    success=True,
                    business_code="0",
                    error_message=sanitize_text(body.get("msg", ""), self._token),
                    raw_bytes=raw_bytes,
                    response_sha256=sha256_bytes(raw_bytes),
                    frame=frame,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_result = self._failure(
                    api_name,
                    clean_parameters,
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
                return self._failure(
                    api_name,
                    clean_parameters,
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
    """按冻结配置创建客户端。"""

    source = protocol["source_contract"]
    return TushareProxyClient(
        endpoint=str(source["endpoint"]),
        token=token,
        minimum_interval_seconds=float(source["minimum_request_interval_seconds"]),
        connect_timeout_seconds=float(source["connect_timeout_seconds"]),
        read_timeout_seconds=float(source["read_timeout_seconds"]),
        maximum_attempts=int(source["maximum_attempts"]),
        retry_delays_seconds=source["retry_delays_seconds"],
    )


def _timestamp_slug(value: str) -> str:
    parsed = datetime.fromisoformat(value).astimezone(TIMEZONE)
    return parsed.strftime("%Y%m%dT%H%M%S%f%z")


def save_immutable_capture(
    project_root: Path,
    partition_relative: Path,
    result: ApiResult,
    endpoint: str,
) -> dict[str, Any]:
    """保存不可变原始响应、规范化Parquet和收据。"""

    partition = project_root / partition_relative
    partition.mkdir(parents=True, exist_ok=True)
    request_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "api": result.api_name,
                "parameters": result.request_parameters,
                "response_sha256": result.response_sha256,
            }
        )
    )[:12]
    final_name = (
        f"retrieved_at={_timestamp_slug(result.response_received_at)}-"
        f"{request_fingerprint}"
    )
    final_directory = partition / final_name
    if final_directory.exists():
        final_directory = partition / f"{final_name}-{uuid.uuid4().hex[:8]}"
    staging = partition / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    raw_path = staging / "response.json"
    raw_path.write_bytes(result.raw_bytes)
    normalized_relative: str | None = None
    normalized_sha256: str | None = None
    if result.success:
        normalized_path = staging / "normalized.parquet"
        result.frame.to_parquet(normalized_path, index=False)
        normalized_sha256 = sha256_file(normalized_path)
        normalized_relative = (
            final_directory / "normalized.parquet"
        ).relative_to(project_root).as_posix()
    receipt_relative = (final_directory / "receipt.json").relative_to(
        project_root
    ).as_posix()
    receipt = {
        "source": "TUSHARE_COMPATIBLE_PROXY",
        "api": result.api_name,
        "request_parameters": result.request_parameters,
        "request_started_at": result.request_started_at,
        "retrieved_at": result.response_received_at,
        "http_status": result.http_status,
        "business_code": result.business_code,
        "success": bool(result.success),
        "error_message": result.error_message,
        "row_count": int(len(result.frame)),
        "columns": [str(column) for column in result.frame.columns],
        "response_bytes": int(len(result.raw_bytes)),
        "response_sha256": result.response_sha256,
        "raw_relative_path": (final_directory / "response.json").relative_to(
            project_root
        ).as_posix(),
        "normalized_relative_path": normalized_relative,
        "normalized_sha256": normalized_sha256,
        "receipt_relative_path": receipt_relative,
        "endpoint_origin": endpoint_origin(endpoint),
        "credential_persisted": False,
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, final_directory)
    return receipt


def _receipt_matches(
    project_root: Path,
    receipt_path: Path,
    api_name: str,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt.get("success") or receipt.get("api") != api_name:
            return None
        if receipt.get("request_parameters") != dict(parameters):
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
        if len(frame) != int(receipt.get("row_count", -1)):
            return None
        if [str(column) for column in frame.columns] != receipt.get("columns"):
            return None
        return receipt, frame
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def find_latest_valid_capture(
    project_root: Path,
    partition_relative: Path,
    api_name: str,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """寻找参数相同且哈希有效的最新成功采集。"""

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


def _required_columns(protocol: Mapping[str, Any], api_name: str) -> list[str]:
    return [str(value) for value in protocol["api_contracts"][api_name]["required_columns"]]


def normalize_api_frame(
    api_name: str,
    frame: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """按冻结字段和类型规范API结果。"""

    required = _required_columns(protocol, api_name)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SourceAdmissionError(f"{api_name}缺少字段：{','.join(missing)}")
    if frame.empty:
        raise SourceAdmissionError(f"{api_name}成功响应为空")
    output = frame[required].copy()
    if api_name == "trade_cal":
        output["exchange"] = output["exchange"].astype(str)
        output["cal_date"] = output["cal_date"].astype(str)
        output["pretrade_date"] = output["pretrade_date"].astype(str)
        output["is_open"] = pd.to_numeric(output["is_open"], errors="coerce")
        if output["is_open"].isna().any():
            raise SourceAdmissionError("trade_cal含无法解析的is_open")
        output["is_open"] = output["is_open"].astype(int)
        return output.sort_values(["exchange", "cal_date"]).reset_index(drop=True)
    output["ts_code"] = output["ts_code"].astype(str)
    if not output["ts_code"].eq("510300.SH").all():
        values = sorted(output.loc[~output["ts_code"].eq("510300.SH"), "ts_code"].unique())
        raise SourceAdmissionError(f"{api_name}返回非目标代码：{values[:10]}")
    numeric = ["open", "high", "low", "close", "vol", "amount"]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if output[numeric].isna().any().any():
        raise SourceAdmissionError(f"{api_name}含无法解析的价格或成交字段")
    if api_name == "fund_daily":
        output["trade_date"] = output["trade_date"].astype(str)
        output["pre_close"] = pd.to_numeric(output["pre_close"], errors="coerce")
        if output["pre_close"].isna().any():
            raise SourceAdmissionError("fund_daily含无法解析的pre_close")
        return output.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="coerce")
    if output["trade_time"].isna().any():
        raise SourceAdmissionError("stk_mins含无法解析的trade_time")
    return output.sort_values(["trade_time", "ts_code"]).reset_index(drop=True)


def _raw_root(protocol: Mapping[str, Any]) -> Path:
    return Path(protocol["storage"]["raw_root"])


def _partition(api_name: str, qualifier: str, protocol: Mapping[str, Any]) -> Path:
    root = _raw_root(protocol)
    if api_name == "trade_cal":
        return root / "trade_cal" / qualifier
    if api_name == "fund_daily":
        return root / "fund_daily" / "510300.SH" / qualifier
    if api_name == "stk_mins":
        return root / "stk_mins" / "510300.SH" / "1min" / qualifier
    raise KeyError(f"未知接口：{api_name}")


def _checked_call_and_capture(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
    api_name: str,
    parameters: Mapping[str, Any],
    partition: Path,
    maximum_rows: int | None = None,
    allow_resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    expected_parameters = dict(parameters)
    if allow_resume:
        existing = find_latest_valid_capture(
            project_root,
            partition,
            api_name,
            expected_parameters,
        )
        if existing is not None:
            receipt, frame = existing
            normalized = normalize_api_frame(api_name, frame, protocol)
            if maximum_rows is None or len(normalized) < maximum_rows:
                return normalized, receipt, True
    result = client.call(api_name, expected_parameters)
    if result.success:
        normalized = normalize_api_frame(api_name, result.frame, protocol)
        result = replace(result, frame=normalized)
    receipt = save_immutable_capture(project_root, partition, result, client.endpoint)
    if not result.success:
        raise ApiCallError(
            f"{api_name}调用失败：{result.business_code} {result.error_message}"
        )
    if maximum_rows is not None and len(result.frame) >= maximum_rows:
        raise SourceAdmissionError(
            f"{api_name}返回{len(result.frame)}行，达到或超过冻结上限{maximum_rows}"
        )
    return result.frame, receipt, False


def _calendar_months(start: date, end: date) -> list[tuple[date, date]]:
    current = date(start.year, start.month, 1)
    output: list[tuple[date, date]] = []
    while current <= end:
        next_month = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
        output.append((max(start, current), min(end, next_month - timedelta(days=1))))
        current = next_month
    return output


def expected_standard_times() -> tuple[str, ...]:
    """返回冻结的241个BAR_END标签。"""

    morning = pd.date_range("2000-01-01 09:30", "2000-01-01 11:30", freq="1min")
    afternoon = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="1min")
    return tuple((morning.append(afternoon)).strftime("%H:%M:%S"))


def _probe_report_path(project_root: Path, protocol: Mapping[str, Any]) -> Path:
    return project_root / Path(protocol["storage"]["permission_probe"])


def run_permission_probe(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
    protocol_manifest: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """运行正式最小权限和契约探针。"""

    probes = [
        (
            "trade_cal",
            {"exchange": "SSE", "start_date": "20170101", "end_date": "20170110"},
            "probe=20170101_20170110",
        ),
        (
            "fund_daily",
            {"ts_code": "510300.SH", "start_date": "20170101", "end_date": "20170110"},
            "probe=20170101_20170110",
        ),
        (
            "stk_mins",
            {
                "ts_code": "510300.SH",
                "freq": "1min",
                "start_date": "2017-01-03 09:00:00",
                "end_date": "2017-01-03 15:30:00",
            },
            "probe=20170103",
        ),
    ]
    records: list[dict[str, Any]] = []
    for api_name, parameters, qualifier in probes:
        result = client.call(api_name, parameters)
        contract_error = ""
        normalized = pd.DataFrame()
        if result.success:
            try:
                normalized = normalize_api_frame(api_name, result.frame, protocol)
                result = replace(result, frame=normalized)
                if api_name == "stk_mins":
                    labels = tuple(normalized["trade_time"].dt.strftime("%H:%M:%S"))
                    if labels != expected_standard_times():
                        contract_error = "分钟探针不是冻结的241个BAR_END标签"
            except SourceAdmissionError as exc:
                contract_error = str(exc)
        receipt = save_immutable_capture(
            project_root,
            _raw_root(protocol) / "_permission_probe" / api_name / qualifier,
            result,
            client.endpoint,
        )
        effective = bool(result.success and not contract_error)
        records.append(
            {
                "api_name": api_name,
                "request_parameters": parameters,
                "request_started_at": result.request_started_at,
                "response_received_at": result.response_received_at,
                "http_status": result.http_status,
                "business_code": result.business_code,
                "success": effective,
                "row_count": int(len(normalized)),
                "columns": [str(value) for value in normalized.columns],
                "error_message": contract_error or result.error_message,
                "response_sha256": result.response_sha256,
                "receipt_relative_path": receipt["receipt_relative_path"],
            }
        )
        print(
            f"来源探针 {api_name}: {'通过' if effective else '失败'}，rows={len(normalized)}",
            flush=True,
        )
    passed = all(record["success"] for record in records)
    report = {
        "data_model_id": DATA_MODEL_ID,
        "phase": "G0_PERMISSION_AND_MINIMAL_CONTRACT_PROBE",
        "state": "PASS_SOURCE_PERMISSION_PROBE" if passed else "BLOCKED_SOURCE_PERMISSION_PROBE",
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
        "results": records,
        "credential_persisted": False,
        "strategy_return_reads": 0,
        "position_impact": 0,
    }
    report["report_fingerprint"] = sha256_bytes(canonical_json_bytes(report))
    atomic_json(report, _probe_report_path(project_root, protocol))
    return report


def _load_bound_probe(project_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = _probe_report_path(project_root, protocol)
    if not path.is_file():
        raise SourceAdmissionError("缺少正式来源权限探针")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("state") != "PASS_SOURCE_PERMISSION_PROBE":
        raise SourceAdmissionError("来源权限探针未通过")
    if report.get("protocol_manifest_sha256") != sha256_file(
        project_root / PROTOCOL_MANIFEST_RELATIVE_PATH
    ):
        raise SourceAdmissionError("来源权限探针未绑定当前协议清单")
    if report.get("implementation_manifest_sha256") != sha256_file(
        project_root / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    ):
        raise SourceAdmissionError("来源权限探针未绑定当前实现清单")
    return report


def _checkpoint(
    state: str,
    phase: str,
    completed: int,
    expected: int,
    last_key: str,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "data_model_id": DATA_MODEL_ID,
        "state": state,
        "phase": phase,
        "updated_at": now_local().isoformat(),
        "completed_partitions": int(completed),
        "expected_partitions": int(expected),
        "last_key": last_key,
        "failures": [dict(value) for value in failures],
        "credential_persisted": False,
        "strategy_return_reads": 0,
        "position_impact": 0,
    }


def run_full_acquisition(
    project_root: Path,
    protocol: Mapping[str, Any],
    client: TushareProxyClient,
) -> dict[str, Any]:
    """断点续传地采集日历、日线和全部自然月分钟数据。"""

    _load_bound_probe(project_root, protocol)
    storage = protocol["storage"]
    checkpoint_path = project_root / Path(storage["checkpoint"])
    summary_path = project_root / Path(storage["acquisition_summary"])
    start = pd.Timestamp(protocol["data_protocol"]["start_date"]).date()
    end = pd.Timestamp(protocol["data_protocol"]["end_date"]).date()
    month_ranges = _calendar_months(start, end)
    expected_partitions = 2 + len(month_ranges)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def record(api_name: str, key: str, receipt: Mapping[str, Any], reused: bool) -> None:
        records.append(
            {
                "api": api_name,
                "key": key,
                "row_count": int(receipt["row_count"]),
                "response_sha256": receipt["response_sha256"],
                "normalized_sha256": receipt["normalized_sha256"],
                "receipt_relative_path": receipt["receipt_relative_path"],
                "reused": bool(reused),
            }
        )

    try:
        for api_name in ("trade_cal", "fund_daily"):
            contract = protocol["api_contracts"][api_name]
            parameters = dict(contract["parameters"])
            key = f"range={parameters['start_date']}_{parameters['end_date']}"
            frame, receipt, reused = _checked_call_and_capture(
                project_root,
                protocol,
                client,
                api_name,
                parameters,
                _partition(api_name, key, protocol),
            )
            primary_key = contract["primary_key"]
            if frame.duplicated(primary_key).any():
                raise SourceAdmissionError(f"{api_name}全量响应主键重复")
            record(api_name, key, receipt, reused)
            atomic_json(
                _checkpoint(
                    "RUNNING",
                    api_name.upper(),
                    len(records),
                    expected_partitions,
                    key,
                    failures,
                ),
                checkpoint_path,
            )
            print(
                f"{api_name}：{'复用' if reused else '新下载'}，rows={len(frame)}",
                flush=True,
            )

        maximum_rows = int(protocol["source_contract"]["documented_maximum_rows"])
        for index, (month_start, month_end) in enumerate(month_ranges, start=1):
            key = f"month={month_start:%Y%m}"
            parameters = {
                "ts_code": "510300.SH",
                "freq": "1min",
                "start_date": f"{month_start:%Y-%m-%d} 00:00:00",
                "end_date": f"{month_end:%Y-%m-%d} 23:59:59",
            }
            frame, receipt, reused = _checked_call_and_capture(
                project_root,
                protocol,
                client,
                "stk_mins",
                parameters,
                _partition("stk_mins", key, protocol),
                maximum_rows=maximum_rows,
            )
            timestamps = pd.to_datetime(frame["trade_time"], errors="raise")
            if not timestamps.dt.date.map(lambda value: month_start <= value <= month_end).all():
                raise SourceAdmissionError(f"{key}返回时间超出请求月份")
            record("stk_mins", key, receipt, reused)
            atomic_json(
                _checkpoint(
                    "RUNNING",
                    "STK_MINS",
                    len(records),
                    expected_partitions,
                    key,
                    failures,
                ),
                checkpoint_path,
            )
            if index == 1 or index % 12 == 0 or index == len(month_ranges):
                print(
                    f"stk_mins进度 {index}/{len(month_ranges)}，{key}，"
                    f"{'复用' if reused else '新下载'}，rows={len(frame)}",
                    flush=True,
                )
    except Exception as exc:
        failures.append({"type": type(exc).__name__, "message": sanitize_text(exc)})
        atomic_json(
            _checkpoint(
                "BLOCKED",
                "RAW_ACQUISITION",
                len(records),
                expected_partitions,
                records[-1]["key"] if records else "",
                failures,
            ),
            checkpoint_path,
        )
        raise

    counts: dict[str, int] = {}
    rows: dict[str, int] = {}
    reused_counts: dict[str, int] = {}
    for item in records:
        api = item["api"]
        counts[api] = counts.get(api, 0) + 1
        rows[api] = rows.get(api, 0) + int(item["row_count"])
        reused_counts[api] = reused_counts.get(api, 0) + int(item["reused"])
    summary = {
        "data_model_id": DATA_MODEL_ID,
        "phase": "RAW_ACQUISITION",
        "state": "PASS_RAW_ACQUISITION_COMPLETE",
        "generated_at": now_local().isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "expected_partitions": expected_partitions,
        "partition_counts": counts,
        "row_counts": rows,
        "reused_partition_counts": reused_counts,
        "records": records,
        "credential_persisted": False,
        "strategy_return_reads": 0,
        "position_impact": 0,
    }
    summary["summary_fingerprint"] = sha256_bytes(canonical_json_bytes(summary))
    atomic_json(summary, summary_path)
    atomic_json(
        _checkpoint(
            "COMPLETE",
            "RAW_ACQUISITION",
            len(records),
            expected_partitions,
            records[-1]["key"],
            failures,
        ),
        checkpoint_path,
    )
    return summary


def _load_acquisition_frames(
    project_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只从哈希有效且参数匹配的采集分区加载全量数据。"""

    summary_path = project_root / Path(protocol["storage"]["acquisition_summary"])
    if not summary_path.is_file():
        raise SourceAdmissionError("缺少全量采集摘要")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fingerprint = summary.pop("summary_fingerprint", None)
    if fingerprint != sha256_bytes(canonical_json_bytes(summary)):
        raise SourceAdmissionError("全量采集摘要指纹不匹配")
    summary["summary_fingerprint"] = fingerprint
    if summary.get("state") != "PASS_RAW_ACQUISITION_COMPLETE":
        raise SourceAdmissionError("全量采集尚未完成")

    frames: dict[str, list[pd.DataFrame]] = {"trade_cal": [], "fund_daily": [], "stk_mins": []}
    for record in summary["records"]:
        receipt_path = project_root / Path(record["receipt_relative_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw_path = project_root / Path(receipt["raw_relative_path"])
        normalized_path = project_root / Path(receipt["normalized_relative_path"])
        if sha256_file(raw_path) != receipt["response_sha256"]:
            raise SourceAdmissionError(f"原始响应哈希漂移：{raw_path}")
        if sha256_file(normalized_path) != receipt["normalized_sha256"]:
            raise SourceAdmissionError(f"规范化分区哈希漂移：{normalized_path}")
        frame = pd.read_parquet(normalized_path)
        if len(frame) != int(receipt["row_count"]):
            raise SourceAdmissionError(f"规范化分区行数漂移：{normalized_path}")
        frames[record["api"]].append(frame)
    if len(frames["trade_cal"]) != 1 or len(frames["fund_daily"]) != 1:
        raise SourceAdmissionError("日历或日线分区数不等于1")
    expected_months = len(
        _calendar_months(
            pd.Timestamp(protocol["data_protocol"]["start_date"]).date(),
            pd.Timestamp(protocol["data_protocol"]["end_date"]).date(),
        )
    )
    if len(frames["stk_mins"]) != expected_months:
        raise SourceAdmissionError(
            f"分钟月份分区数错误：expected={expected_months} actual={len(frames['stk_mins'])}"
        )
    return (
        frames["trade_cal"][0],
        frames["fund_daily"][0],
        pd.concat(frames["stk_mins"], ignore_index=True),
        summary,
    )


def _safe_relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def audit_minute_frames(
    calendar: pd.DataFrame,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    protocol: Mapping[str, Any],
    eligible_event_dates: Iterable[date] | None = None,
    timestamp_evidence_pass: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行纯本地、确定性的分钟来源质量裁决。"""

    calendar = normalize_api_frame("trade_cal", calendar, protocol)
    daily = normalize_api_frame("fund_daily", daily, protocol)
    minute = normalize_api_frame("stk_mins", minute, protocol)
    start = pd.Timestamp(protocol["data_protocol"]["start_date"]).normalize()
    end = pd.Timestamp(protocol["data_protocol"]["end_date"]).normalize()
    calendar_dates = pd.to_datetime(calendar["cal_date"], format="%Y%m%d", errors="coerce")
    if calendar_dates.isna().any():
        raise SourceAdmissionError("trade_cal含无法解析的cal_date")
    open_dates = sorted(
        calendar_dates[
            calendar["is_open"].eq(1)
            & calendar_dates.between(start, end, inclusive="both")
        ].dt.date.unique()
    )
    daily = daily.copy()
    daily["date"] = pd.to_datetime(
        daily["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.date
    if daily["date"].isna().any():
        raise SourceAdmissionError("fund_daily含无法解析的trade_date")
    minute = minute.copy()
    minute["date"] = minute["trade_time"].dt.date
    minute["time"] = minute["trade_time"].dt.strftime("%H:%M:%S")

    duplicate_primary_keys = int(minute.duplicated(["ts_code", "trade_time"], keep=False).sum())
    unexpected_symbol_rows = int((~minute["ts_code"].eq("510300.SH")).sum())
    finite_prices = np.isfinite(minute[["open", "high", "low", "close"]]).all(axis=1)
    valid_ohlc = (
        finite_prices
        & minute[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & minute["high"].ge(minute[["open", "close"]].max(axis=1))
        & minute["low"].le(minute[["open", "close"]].min(axis=1))
        & minute["high"].ge(minute["low"])
    )
    invalid_ohlc_rows = int((~valid_ohlc).sum())
    negative_volume_or_amount_rows = int(
        (minute["vol"].lt(0) | minute["amount"].lt(0)).sum()
    )
    expected_times = set(expected_standard_times())
    unexpected_timestamp_rows = int((~minute["time"].isin(expected_times)).sum())
    open_date_set = set(open_dates)
    unexpected_date_rows = int((~minute["date"].isin(open_date_set)).sum())

    daily_by_date = {
        row.date: row
        for row in daily.sort_values("trade_date").itertuples(index=False)
        if row.date in open_date_set
    }
    minute_groups = {key: group for key, group in minute.groupby("date", sort=True)}
    window_labels = protocol["timestamp_contract"]["research_windows"]
    tolerance = float(protocol["quality_gates"]["price_absolute_tolerance"])
    regime_boundary = pd.Timestamp(
        protocol["quality_gates"]["pre_post_close_regime_boundary"]
    ).date()
    ledger_rows: list[dict[str, Any]] = []
    for trade_date in open_dates:
        group = minute_groups.get(trade_date, minute.iloc[0:0]).sort_values("trade_time")
        labels = set(group["time"].tolist())
        reference = daily_by_date.get(trade_date)
        has_data = not group.empty
        exact_time_set = labels == expected_times and len(group) == len(expected_times)
        row: dict[str, Any] = {
            "trade_date": pd.Timestamp(trade_date),
            "minute_rows": int(len(group)),
            "has_any_minute_data": bool(has_data),
            "exact_standard_241_labels": bool(exact_time_set),
            "daily_reference_available": reference is not None,
        }
        for name, contract in window_labels.items():
            required = set(contract["bar_labels"])
            row[f"{name}_window_complete"] = bool(required.issubset(labels))
        if has_data:
            row.update(
                {
                    "minute_open": float(group.iloc[0]["open"]),
                    "minute_high": float(group["high"].max()),
                    "minute_low": float(group["low"].min()),
                    "minute_close": float(group.iloc[-1]["close"]),
                    "minute_vol_sum_shares": float(group["vol"].sum()),
                    "minute_amount_sum_cny": float(group["amount"].sum()),
                }
            )
        else:
            for field in (
                "minute_open",
                "minute_high",
                "minute_low",
                "minute_close",
                "minute_vol_sum_shares",
                "minute_amount_sum_cny",
            ):
                row[field] = math.nan
        if has_data and reference is not None:
            differences = {
                field: abs(float(row[f"minute_{field}"]) - float(getattr(reference, field)))
                for field in ("open", "high", "low", "close")
            }
            row["ohlc_max_abs_difference"] = max(differences.values())
            row["ohlc_reconciliation_pass"] = bool(
                all(value <= tolerance + 1e-12 for value in differences.values())
            )
            minute_vol_daily_unit = float(row["minute_vol_sum_shares"]) / 100.0
            minute_amount_daily_unit = float(row["minute_amount_sum_cny"]) / 1000.0
            row["volume_relative_error"] = _safe_relative_error(
                minute_vol_daily_unit, float(reference.vol)
            )
            row["amount_relative_error"] = _safe_relative_error(
                minute_amount_daily_unit, float(reference.amount)
            )
        else:
            row["ohlc_max_abs_difference"] = math.nan
            row["ohlc_reconciliation_pass"] = False
            row["volume_relative_error"] = math.nan
            row["amount_relative_error"] = math.nan
        row["pre_post_close_regime"] = bool(trade_date < regime_boundary)
        ledger_rows.append(row)
    ledger = pd.DataFrame(ledger_rows)

    open_count = len(open_dates)
    any_data_count = int(ledger["has_any_minute_data"].sum())
    standard_count = int(ledger["exact_standard_241_labels"].sum())
    ohlc_pass_count = int(ledger["ohlc_reconciliation_pass"].sum())
    pre_comparable = ledger[
        ledger["pre_post_close_regime"]
        & ledger["exact_standard_241_labels"]
        & ledger["daily_reference_available"]
    ]
    maximum_volume_error = (
        float(pre_comparable["volume_relative_error"].max())
        if not pre_comparable.empty
        else math.inf
    )
    maximum_amount_error = (
        float(pre_comparable["amount_relative_error"].max())
        if not pre_comparable.empty
        else math.inf
    )
    event_dates = sorted(set(eligible_event_dates or []))
    event_ledger = ledger[ledger["trade_date"].dt.date.isin(event_dates)] if event_dates else ledger.iloc[0:0]
    all_window_columns = [
        f"{name}_window_complete" for name in ("pre", "reaction", "entry", "exit")
    ]
    event_complete_count = (
        int(event_ledger[all_window_columns].all(axis=1).sum()) if event_dates else 0
    )
    event_coverage = event_complete_count / len(event_dates) if event_dates else None

    gates = protocol["quality_gates"]
    checks = {
        "duplicate_primary_keys": duplicate_primary_keys
        <= int(gates["duplicate_primary_keys_maximum"]),
        "unexpected_symbol_rows": unexpected_symbol_rows
        <= int(gates["unexpected_symbol_rows_maximum"]),
        "invalid_ohlc_rows": invalid_ohlc_rows <= int(gates["invalid_ohlc_rows_maximum"]),
        "negative_volume_or_amount_rows": negative_volume_or_amount_rows
        <= int(gates["negative_volume_or_amount_rows_maximum"]),
        "unexpected_timestamp_rows": unexpected_timestamp_rows == 0,
        "unexpected_date_rows": unexpected_date_rows == 0,
        "open_day_any_data_coverage": (any_data_count / open_count)
        >= float(gates["open_trading_day_any_data_coverage_minimum"]),
        "standard_241_bar_day_coverage": (standard_count / open_count)
        >= float(gates["standard_241_bar_day_coverage_minimum"]),
        "daily_ohlc_reconciliation_rate": (ohlc_pass_count / open_count)
        >= float(gates["daily_ohlc_reconciliation_rate_minimum"]),
        "pre_regime_volume_error": maximum_volume_error
        <= float(gates["pre_regime_volume_relative_error_maximum"]),
        "pre_regime_amount_error": maximum_amount_error
        <= float(gates["pre_regime_amount_relative_error_maximum"]),
        "timestamp_semantics_evidence": bool(timestamp_evidence_pass),
    }
    general_pass = all(checks.values())
    event_gate_pass = (
        event_coverage is not None
        and event_coverage
        >= float(gates["nbs_eligible_event_four_window_coverage_minimum"])
    )
    if general_pass and event_gate_pass:
        state = str(gates["full_pass_state"])
    elif general_pass and event_coverage is None:
        state = str(gates["preliminary_pass_state"])
    else:
        state = str(gates["failure_state"])
    report = {
        "data_model_id": DATA_MODEL_ID,
        "state": state,
        "checks": checks,
        "general_source_pass": general_pass,
        "event_window_gate_evaluated": event_coverage is not None,
        "event_window_gate_pass": event_gate_pass if event_coverage is not None else None,
        "coverage": {
            "expected_open_trading_days": open_count,
            "days_with_any_minute_data": any_data_count,
            "open_day_any_data_coverage": any_data_count / open_count,
            "standard_241_bar_days": standard_count,
            "standard_241_bar_day_coverage": standard_count / open_count,
            "daily_ohlc_pass_days": ohlc_pass_count,
            "daily_ohlc_reconciliation_rate": ohlc_pass_count / open_count,
            "eligible_event_dates": len(event_dates),
            "eligible_event_four_window_complete_dates": event_complete_count,
            "eligible_event_four_window_coverage": event_coverage,
        },
        "defects": {
            "duplicate_primary_key_rows": duplicate_primary_keys,
            "unexpected_symbol_rows": unexpected_symbol_rows,
            "invalid_ohlc_rows": invalid_ohlc_rows,
            "negative_volume_or_amount_rows": negative_volume_or_amount_rows,
            "unexpected_timestamp_rows": unexpected_timestamp_rows,
            "unexpected_non_open_date_rows": unexpected_date_rows,
        },
        "reconciliation": {
            "price_absolute_tolerance": tolerance,
            "pre_2026_07_06_comparable_days": int(len(pre_comparable)),
            "maximum_volume_relative_error": maximum_volume_error,
            "maximum_amount_relative_error": maximum_amount_error,
            "post_close_regime_total_comparison_excluded": True,
        },
    }
    return ledger, report


def _timestamp_evidence(project_root: Path) -> dict[str, Any]:
    """绑定既有跨源聚合证据，证明分钟标签可按BAR_END解释。"""

    evidence_files = {
        "reports/data_quality/510300_minute_cross_source.json": None,
        "reports/data_quality/510300_15m_from_1m_quality.json": None,
        "data/raw/market/510300_1m_tushare_raw.parquet": None,
        "data/raw/market/510300_15m_from_1m_raw.parquet": None,
    }
    missing: list[str] = []
    hashes: dict[str, str] = {}
    for relative in evidence_files:
        path = project_root / relative
        if not path.is_file():
            missing.append(relative)
        else:
            hashes[relative] = sha256_file(path)
    assertions = {
        "all_evidence_files_present": not missing,
        "legacy_tushare_1m_has_241_bar_session_signature": False,
        "one_minute_to_15m_has_16_bar_end_labels": False,
        "cross_source_report_supports_bar_end": False,
    }
    if not missing:
        one = pd.read_parquet(project_root / "data/raw/market/510300_1m_tushare_raw.parquet")
        one["trade_time"] = pd.to_datetime(one["trade_time"], errors="coerce")
        counts = one.groupby(one["trade_time"].dt.date).size()
        time_sets = one.groupby(one["trade_time"].dt.date)["trade_time"].apply(
            lambda values: set(values.dt.strftime("%H:%M:%S"))
        )
        assertions["legacy_tushare_1m_has_241_bar_session_signature"] = bool(
            not counts.empty
            and counts.eq(241).all()
            and all(value == set(expected_standard_times()) for value in time_sets)
        )
        fifteen = pd.read_parquet(
            project_root / "data/raw/market/510300_15m_from_1m_raw.parquet"
        )
        bar_column = "bar_end" if "bar_end" in fifteen.columns else "trade_time"
        fifteen[bar_column] = pd.to_datetime(fifteen[bar_column], errors="coerce")
        expected_15m = {
            "09:45:00", "10:00:00", "10:15:00", "10:30:00",
            "10:45:00", "11:00:00", "11:15:00", "11:30:00",
            "13:15:00", "13:30:00", "13:45:00", "14:00:00",
            "14:15:00", "14:30:00", "14:45:00", "15:00:00",
        }
        fifteen_sets = fifteen.groupby(fifteen[bar_column].dt.date)[bar_column].apply(
            lambda values: set(values.dt.strftime("%H:%M:%S"))
        )
        assertions["one_minute_to_15m_has_16_bar_end_labels"] = bool(
            not fifteen_sets.empty and all(value == expected_15m for value in fifteen_sets)
        )
        cross_source = json.loads(
            (project_root / "reports/data_quality/510300_minute_cross_source.json").read_text(
                encoding="utf-8"
            )
        )
        conclusion = str(
            cross_source.get("evidence", {})
            .get("sina_1m_to_sina_15m", {})
            .get("timestamp_conclusion", "")
        )
        assertions["cross_source_report_supports_bar_end"] = "bar_end" in conclusion.lower()
    passed = all(assertions.values())
    return {
        "state": "PASS_BAR_END_DOCUMENT_AND_SESSION_EVIDENCE"
        if passed
        else "BLOCKED_BAR_END_EVIDENCE",
        "generated_at": now_local().isoformat(),
        "evidence_file_sha256": hashes,
        "missing_files": missing,
        "assertions": assertions,
        "pass": passed,
        "interpretation": (
            "BAR_END是冻结的研究适配器语义；供应商文档只称trade_time为交易时间，"
            "本结论依赖241标签结构、1分钟向15分钟聚合及独立分钟源交叉证据。"
        ),
        "strategy_return_reads": 0,
        "position_impact": 0,
    }


def _load_eligible_event_dates(event_ledger_path: Path | None) -> list[date] | None:
    if event_ledger_path is None:
        return None
    if not event_ledger_path.is_file():
        raise SourceAdmissionError(f"事件账本不存在：{event_ledger_path}")
    frame = pd.read_parquet(event_ledger_path)
    required = {"scheduled_date", "final_event_eligibility"}
    if not required.issubset(frame.columns):
        raise SourceAdmissionError("事件账本缺少scheduled_date或final_event_eligibility")
    dates = pd.to_datetime(
        frame.loc[frame["final_event_eligibility"].eq(True), "scheduled_date"],
        errors="coerce",
    )
    if dates.isna().any():
        raise SourceAdmissionError("事件账本含无效scheduled_date")
    return sorted(dates.dt.date.unique())


def _summary_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    defects = report["defects"]
    reconciliation = report["reconciliation"]
    return f"""# 510300 STK_MINS来源准入结论

生成时间：{report['generated_at']}

```text
SOURCE_STATE={report['state']}
GENERAL_SOURCE_PASS={str(report['general_source_pass']).upper()}
EVENT_WINDOW_GATE_EVALUATED={str(report['event_window_gate_evaluated']).upper()}
STRATEGY_RETURN_READS=0
POSITION_IMPACT=0
```

## 覆盖

- 开放交易日：{coverage['expected_open_trading_days']}
- 有任意分钟数据：{coverage['days_with_any_minute_data']}，覆盖率{coverage['open_day_any_data_coverage']:.4%}
- 标准241根交易日：{coverage['standard_241_bar_days']}，覆盖率{coverage['standard_241_bar_day_coverage']:.4%}
- 日级OHLC对账通过：{coverage['daily_ohlc_pass_days']}，通过率{coverage['daily_ohlc_reconciliation_rate']:.4%}
- 合格NBS事件日：{coverage['eligible_event_dates']}
- 四窗口完整事件日：{coverage['eligible_event_four_window_complete_dates']}

## 缺陷

- 重复主键行：{defects['duplicate_primary_key_rows']}
- 非目标代码行：{defects['unexpected_symbol_rows']}
- 非法OHLC行：{defects['invalid_ohlc_rows']}
- 负成交量或成交额行：{defects['negative_volume_or_amount_rows']}
- 非法交易时刻行：{defects['unexpected_timestamp_rows']}
- 非开放日记录：{defects['unexpected_non_open_date_rows']}

## 日级总量对账

- 2026-07-06之前可比日：{reconciliation['pre_2026_07_06_comparable_days']}
- 最大成交量相对误差：{reconciliation['maximum_volume_relative_error']:.12g}
- 最大成交额相对误差：{reconciliation['maximum_amount_relative_error']:.12g}

本报告只裁决数据来源；不包含事件收益、模型、净值、Sharpe、仓位或订单。
"""


def run_source_adjudication(
    project_root: Path,
    protocol: Mapping[str, Any],
    event_ledger_path: Path | None = None,
    replay_expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """从不可变本地采集运行来源准入，绝不读取事件收益。"""

    calendar, daily, minute, acquisition = _load_acquisition_frames(project_root, protocol)
    evidence = _timestamp_evidence(project_root)
    eligible_dates = _load_eligible_event_dates(event_ledger_path)
    ledger, core = audit_minute_frames(
        calendar,
        daily,
        minute,
        protocol,
        eligible_event_dates=eligible_dates,
        timestamp_evidence_pass=bool(evidence["pass"]),
    )
    storage = protocol["storage"]
    atomic_parquet(minute.sort_values("trade_time"), project_root / storage["curated_minute"])
    atomic_parquet(ledger, project_root / storage["daily_quality_ledger"])
    evidence_path = (
        project_root
        / "reports/data_quality/510300_stk_mins_source_admission_v1/timestamp_semantics_evidence.json"
    )
    atomic_json(evidence, evidence_path)
    report = {
        "data_model_id": DATA_MODEL_ID,
        "phase": "SOURCE_ADMISSION",
        "state": core["state"],
        "generated_at": now_local().isoformat(),
        "checks": core["checks"],
        "general_source_pass": core["general_source_pass"],
        "event_window_gate_evaluated": core["event_window_gate_evaluated"],
        "event_window_gate_pass": core["event_window_gate_pass"],
        "coverage": core["coverage"],
        "defects": core["defects"],
        "reconciliation": core["reconciliation"],
        "timestamp_semantics_evidence": {
            "state": evidence["state"],
            "path": evidence_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(evidence_path),
        },
        "acquisition_summary_fingerprint": acquisition["summary_fingerprint"],
        "outputs": {
            "curated_minute": storage["curated_minute"],
            "curated_minute_sha256": sha256_file(project_root / storage["curated_minute"]),
            "daily_quality_ledger": storage["daily_quality_ledger"],
            "daily_quality_ledger_sha256": sha256_file(
                project_root / storage["daily_quality_ledger"]
            ),
        },
        "strategy_return_reads": 0,
        "strategy_model_trained": False,
        "portfolio_evaluation_run": False,
        "position_impact": 0,
    }
    fingerprint_payload = dict(report)
    fingerprint_payload.pop("generated_at")
    fingerprint_payload["timestamp_semantics_evidence"] = {
        "state": evidence["state"],
        "evidence_file_sha256": evidence["evidence_file_sha256"],
        "assertions": evidence["assertions"],
    }
    report["decision_fingerprint"] = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    if replay_expected_fingerprint is not None:
        if report["decision_fingerprint"] != replay_expected_fingerprint:
            raise SourceAdmissionError(
                "干净进程来源裁决指纹不一致："
                f"expected={replay_expected_fingerprint} actual={report['decision_fingerprint']}"
            )
        report["replay_state"] = "PASS_CLEAN_NEW_PROCESS_REPLAY"
        return report
    report_path = project_root / Path(storage["source_adjudication"])
    atomic_json(report, report_path)
    atomic_text(_summary_markdown(report), project_root / Path(storage["markdown_summary"]))
    return report

