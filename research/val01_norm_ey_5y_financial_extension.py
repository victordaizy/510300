"""补采 VAL01_NORM_EY_5Y 所需的 2012Q2 至 2015Q2 点时财务历史。

本模块只负责数据采集、点时事件构建和覆盖审计的输入准备，不计算收益、
IC、仓位或订单。原 V1 财务档案和原 VIP 检查点始终按冻结输入处理。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

from research.point_in_time_valuation_v2_audit import canonical_directory_hash
from scripts.download_csi300_point_in_time_financials import build_point_in_time_events


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "val01_norm_ey_5y_financial_extension_v1.yaml"

FORBIDDEN_PROTOCOL_FLAGS = (
    "return_calculation_enabled",
    "ic_calculation_enabled",
    "position_mapping_enabled",
    "order_generation_enabled",
    "broker_connection_enabled",
)


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """读取并校验扩展协议，防止数据阶段越权。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    enabled = [
        field for field in FORBIDDEN_PROTOCOL_FLAGS
        if config["protocol"].get(field) is not False
    ]
    if enabled:
        raise ValueError(f"财务扩展阶段禁止启用：{enabled}")
    acquisition = config["acquisition"]
    periods = build_quarter_periods(
        pd.Timestamp(acquisition["first_report_period"]),
        pd.Timestamp(acquisition["last_report_period"]),
    )
    if len(periods) != int(acquisition["expected_quarter_count"]):
        raise ValueError("配置的季度数与起止报告期不一致")
    if not config["normalization_contract"].get("fixed_parameters_only"):
        raise ValueError("标准化盈利参数必须固定")
    if config["normalization_contract"].get("parameter_search_enabled"):
        raise ValueError("数据补采阶段禁止参数搜索")
    return config


def build_quarter_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """生成闭区间内的季度末字符串。"""

    if end < start:
        raise ValueError("报告期结束日不能早于开始日")
    return [period.strftime("%Y%m%d") for period in pd.date_range(start, end, freq="QE")]


def _parse_expiry(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    return parsed


def resolve_transport(
    config: dict[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """选择标准或代理凭据，并强制 HTTPS；返回值中的令牌不得写入报告。"""

    if environ is None:
        load_dotenv(ROOT / ".env")
        environ = os.environ
    standard_token = environ.get("TUSHARE_TOKEN") or environ.get("TS_TOKEN")
    proxy_token = environ.get("TUSHARE_PROXY_TOKEN")
    acquisition = config["acquisition"]
    if standard_token:
        token = standard_token
        kind = "STANDARD_TUSHARE_TOKEN"
        api_url = environ.get("TUSHARE_API_URL", acquisition["official_api_url"])
        expiry = None
    elif proxy_token:
        token = proxy_token
        kind = "TUSHARE_PROXY_TOKEN"
        api_url = environ.get(
            "TUSHARE_PROXY_API_URL", acquisition["secure_proxy_api_url"]
        )
        expiry = _parse_expiry(environ.get("TUSHARE_PROXY_TOKEN_EXPIRES_AT"))
    else:
        raise RuntimeError("未设置 TUSHARE_TOKEN、TS_TOKEN 或 TUSHARE_PROXY_TOKEN")

    api_url = str(api_url).strip().rstrip("/")
    parsed = urlparse(api_url)
    if acquisition.get("require_https") and parsed.scheme.lower() != "https":
        raise ValueError("API 端点不是 HTTPS；协议禁止明文传输凭据")
    if not parsed.hostname:
        raise ValueError("API 端点缺少有效主机名")
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    if expiry is not None and expiry <= now:
        raise RuntimeError("Tushare 代理凭据已经过期")
    return {
        "token": token,
        "credential_kind": kind,
        "api_url": api_url,
        "api_host": parsed.hostname,
        "expires_at": expiry.isoformat() if expiry is not None else None,
    }


def sanitized_transport(transport: dict[str, Any]) -> dict[str, Any]:
    """移除敏感令牌后生成可写入报告的传输说明。"""

    return {
        key: value for key, value in transport.items() if key != "token"
    } | {"token_saved_or_echoed": False}


def _sanitize_error(error: BaseException, secrets: list[str]) -> str:
    message = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[已隐藏凭据]")
    return message


def classify_failure(error: BaseException) -> str:
    """把异常归入可审计的阻塞类别。"""

    name = type(error).__name__.lower()
    text = str(error).lower()
    if "访问频率" in text or "超速" in text or "冷却" in text or "rate limit" in text:
        return "BLOCKED_API_RATE_LIMIT_COOLDOWN"
    if "ssl" in name or "ssl" in text or "tls" in text or "certificate" in text:
        return "BLOCKED_SECURE_TRANSPORT_TLS"
    if "timeout" in name or "timed out" in text:
        return "BLOCKED_API_TIMEOUT"
    if "connection" in name or "connection" in text:
        return "BLOCKED_API_CONNECTION"
    if "permission" in text or "权限" in text or "积分" in text:
        return "BLOCKED_API_ENTITLEMENT"
    if "token" in text or "凭据" in text or "过期" in text:
        return "BLOCKED_CREDENTIAL"
    if "hash" in text or "哈希" in text:
        return "BLOCKED_FROZEN_INPUT_DRIFT"
    if "字段" in text or "报告期" in text or "schema" in text:
        return "BLOCKED_API_SCHEMA_OR_PERIOD"
    return "BLOCKED_ACQUISITION_ERROR"


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """核验原档案、权重、审计报告和原检查点目录均未漂移。"""

    rows: list[dict[str, Any]] = []
    for name, contract in config["frozen_inputs"].items():
        if "file" not in contract:
            continue
        path = ROOT / contract["file"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "dataset": name,
                "file": contract["file"],
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    checkpoint_contract = config["frozen_inputs"]["base_financial_checkpoints"]
    checkpoint_directory = ROOT / checkpoint_contract["directory"]
    directory_hash, files = canonical_directory_hash(checkpoint_directory)
    checkpoint_matches = (
        len(files) == int(checkpoint_contract["file_count"])
        and directory_hash == checkpoint_contract["content_sha256"]
    )
    result = {
        "status": "PASS" if all(row["matches"] for row in rows) and checkpoint_matches else "BLOCKED_FROZEN_INPUT_DRIFT",
        "files": rows,
        "checkpoint_directory": checkpoint_contract["directory"],
        "checkpoint_file_count": len(files),
        "expected_checkpoint_file_count": int(checkpoint_contract["file_count"]),
        "checkpoint_content_sha256": directory_hash,
        "expected_checkpoint_content_sha256": checkpoint_contract["content_sha256"],
        "checkpoint_matches": checkpoint_matches,
    }
    return result


def load_target_universe(config: dict[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    """固定五年预评价窗口的官方权重快照和证券并集。"""

    contract = config["frozen_inputs"]["historical_weights"]
    weights = pd.read_parquet(ROOT / contract["file"])
    weights["trade_date"] = pd.to_datetime(weights["trade_date"], errors="coerce")
    normalization = config["normalization_contract"]
    first_month = pd.Period(normalization["first_required_signal_month"], freq="M")
    last_month = pd.Period(normalization["last_required_signal_month"], freq="M")
    periods = weights["trade_date"].dt.to_period("M")
    selected = weights.loc[periods.between(first_month, last_month)].copy()
    snapshot_count = selected["trade_date"].nunique()
    target_symbols = set(selected["con_code"].dropna().astype(str))
    if snapshot_count != int(normalization["required_weight_snapshot_count"]):
        raise ValueError(f"目标权重快照应为 60 个，实际 {snapshot_count} 个")
    if len(target_symbols) != int(normalization["expected_target_symbol_count"]):
        raise ValueError(
            f"目标证券应为 {normalization['expected_target_symbol_count']} 只，实际 {len(target_symbols)} 只"
        )
    return selected, target_symbols


def api_fields(config: dict[str, Any], api_name: str) -> list[str]:
    return list(config["api_contracts"][api_name]["fields"])


def validate_api_response(
    data: pd.DataFrame,
    api_name: str,
    period: str,
    config: dict[str, Any],
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """校验供应商响应字段与报告期；空单证券响应可审计保留。"""

    fields = api_fields(config, api_name)
    if data is None or data.empty:
        if not allow_empty:
            raise ValueError(f"{api_name}/{period} 返回空数据")
        return pd.DataFrame(columns=fields)
    if missing := set(fields) - set(data.columns):
        raise ValueError(f"{api_name}/{period} 缺少字段：{sorted(missing)}")
    result = data[fields].copy()
    end_dates = (
        result["end_date"].astype("string").str.replace("-", "", regex=False).str[:8]
    )
    mismatches = end_dates.notna() & end_dates.ne(period)
    if mismatches.any():
        examples = sorted(end_dates.loc[mismatches].dropna().unique().tolist())[:5]
        raise ValueError(f"{api_name}/{period} 混入其他报告期：{examples}")
    if result["ts_code"].isna().any():
        raise ValueError(f"{api_name}/{period} 存在空证券代码")
    return result.reset_index(drop=True)


def validate_api_history_response(
    data: pd.DataFrame,
    api_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """校验单证券多期历史响应，保留查询窗口内的原始报告期。"""

    fields = api_fields(config, api_name)
    if data is None or data.empty:
        return pd.DataFrame(columns=fields)
    if missing := set(fields) - set(data.columns):
        raise ValueError(f"{api_name} 单证券历史响应缺少字段：{sorted(missing)}")
    result = data[fields].copy()
    if result["ts_code"].isna().any():
        raise ValueError(f"{api_name} 单证券历史响应存在空证券代码")
    parsed_periods = (
        result["end_date"].astype("string").str.replace("-", "", regex=False).str[:8]
    )
    if parsed_periods.isna().any() or parsed_periods.str.fullmatch(r"\d{8}").ne(True).any():
        raise ValueError(f"{api_name} 单证券历史响应存在无效报告期")
    return result.reset_index(drop=True)


def response_statistics(data: pd.DataFrame, target_symbols: set[str]) -> dict[str, Any]:
    """生成不改变原始响应的行数与目标覆盖统计。"""

    codes = data["ts_code"].dropna().astype(str) if "ts_code" in data else pd.Series(dtype=str)
    return {
        "row_count": int(len(data)),
        "unique_symbol_count": int(codes.nunique()),
        "exact_duplicate_row_count": int(data.duplicated().sum()),
        "target_symbol_count": int(codes.isin(target_symbols).groupby(codes).any().sum()) if not codes.empty else 0,
    }


def _canonical_counter(data: pd.DataFrame, fields: list[str]) -> Counter[tuple[str, ...]]:
    frame = data[fields].copy()
    for column in fields:
        frame[column] = frame[column].where(frame[column].notna(), "__NA__").astype(str)
    return Counter(map(tuple, frame.itertuples(index=False, name=None)))


def compare_responses(
    current: pd.DataFrame,
    frozen: pd.DataFrame,
    fields: list[str],
) -> dict[str, Any]:
    """按多重集合比较实时重取和冻结检查点，识别供应商修订。"""

    left = _canonical_counter(current, fields)
    right = _canonical_counter(frozen, fields)
    return {
        "current_row_count": int(len(current)),
        "frozen_row_count": int(len(frozen)),
        "multiset_exact_match": left == right,
        "current_only_row_count": int(sum((left - right).values())),
        "frozen_only_row_count": int(sum((right - left).values())),
    }


def _atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temp, index=False)
    temp.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _call_with_retry(
    call: Callable[[], pd.DataFrame],
    label: str,
    attempts: int,
) -> pd.DataFrame:
    error: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = call()
            return result if result is not None else pd.DataFrame()
        except BaseException as exception:  # 第三方客户端异常类型并不稳定
            error = exception
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    assert error is not None
    raise RuntimeError(f"{label} 请求失败：{type(error).__name__}: {error}") from error


def _load_or_download(
    pro: Any,
    api_name: str,
    period: str,
    checkpoint: Path,
    config: dict[str, Any],
    *,
    ts_code: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """读取断点或下载一个季度；返回数据及是否发生网络调用。"""

    allow_empty = ts_code is not None
    if checkpoint.exists():
        stored = pd.read_parquet(checkpoint)
        return validate_api_response(
            stored, api_name, period, config, allow_empty=allow_empty
        ), False

    contract = config["api_contracts"][api_name]
    query_api = contract["standard_api"] if ts_code else api_name
    parameters: dict[str, Any] = {
        "period": period,
        "fields": ",".join(api_fields(config, api_name)),
    }
    if ts_code:
        parameters["ts_code"] = ts_code
    attempts = int(config["acquisition"]["retry_attempts"])
    raw = _call_with_retry(
        lambda: pro.query(query_api, **parameters),
        f"{query_api}/{ts_code or 'ALL'}/{period}",
        attempts,
    )
    validated = validate_api_response(
        raw, api_name, period, config, allow_empty=allow_empty
    )
    _atomic_parquet(validated, checkpoint)
    return validated, True


def _load_or_download_history(
    pro: Any,
    api_name: str,
    ts_code: str,
    checkpoint: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    """一次获取单证券查询窗口内的多期历史，降低逐缺季调用数量。"""

    if checkpoint.exists():
        return validate_api_history_response(
            pd.read_parquet(checkpoint), api_name, config
        ), False
    acquisition = config["acquisition"]
    contract = config["api_contracts"][api_name]
    parameters = {
        "ts_code": ts_code,
        "start_date": acquisition["single_security_history_start_date"].replace("-", ""),
        "end_date": acquisition["single_security_history_end_date"].replace("-", ""),
        "fields": ",".join(api_fields(config, api_name)),
    }
    raw = _call_with_retry(
        lambda: pro.query(contract["standard_api"], **parameters),
        f"{contract['standard_api']}/{ts_code}/HISTORY",
        int(acquisition["retry_attempts"]),
    )
    validated = validate_api_history_response(raw, api_name, config)
    _atomic_parquet(validated, checkpoint)
    return validated, True


def find_continuity_gaps(
    period_frames: dict[str, pd.DataFrame],
    anchor: pd.DataFrame,
    target_symbols: set[str],
    periods: list[str],
) -> list[tuple[str, str]]:
    """在目标证券首次出现后识别季度连续性缺口。"""

    observed: set[tuple[str, str]] = set()
    for period, frame in period_frames.items():
        observed.update(
            (str(code), period)
            for code in frame["ts_code"].dropna().astype(str).unique()
            if str(code) in target_symbols
        )
    anchor_periods = anchor["end_date"].astype("string").str.replace("-", "", regex=False).str[:8]
    observed.update(
        (str(code), str(period))
        for code, period in zip(anchor["ts_code"], anchor_periods, strict=True)
        if pd.notna(code) and str(code) in target_symbols and pd.notna(period)
    )
    first_seen: dict[str, str] = {}
    for code, period in observed:
        first_seen[code] = min(first_seen.get(code, period), period)
    gaps: list[tuple[str, str]] = []
    for code, first_period in sorted(first_seen.items()):
        for period in periods:
            if period >= first_period and (code, period) not in observed:
                gaps.append((code, period))
    return gaps


def build_extension_events(
    api_data: dict[str, pd.DataFrame],
    target_symbols: set[str],
    retrieved_at: datetime,
) -> pd.DataFrame:
    """把三张扩展表构造成目标证券点时事件。"""

    filtered = {
        api_name: data.loc[data["ts_code"].astype(str).isin(target_symbols)].copy()
        for api_name, data in api_data.items()
    }
    events = build_point_in_time_events(
        filtered["income_vip"],
        filtered["balancesheet_vip"],
        filtered["fina_indicator_vip"],
        retrieved_at,
    )
    keys = ["con_code", "report_period", "available_at"]
    if events[keys].duplicated().any():
        raise ValueError("扩展点时事件存在重复键")
    if not set(events["con_code"].astype(str)).issubset(target_symbols):
        raise ValueError("扩展事件混入目标证券范围外代码")
    return events.sort_values(keys).reset_index(drop=True)


def combine_financial_archives(
    base: pd.DataFrame,
    extension: pd.DataFrame,
    expected_base_first_period: str,
) -> pd.DataFrame:
    """以不重叠报告期合并冻结原档案和新扩展档案。"""

    base_data = base.copy()
    extension_data = extension.copy()
    for frame in (base_data, extension_data):
        frame["report_period"] = pd.to_datetime(frame["report_period"], errors="coerce")
        frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    expected_first = pd.Timestamp(expected_base_first_period)
    if base_data["report_period"].min() != expected_first:
        raise ValueError("冻结原档案首报告期与协议不一致")
    if extension_data["report_period"].max() >= expected_first:
        raise ValueError("扩展事件与冻结原档案报告期重叠")
    if set(base_data.columns) != set(extension_data.columns):
        missing_in_extension = sorted(set(base_data.columns) - set(extension_data.columns))
        missing_in_base = sorted(set(extension_data.columns) - set(base_data.columns))
        raise ValueError(
            f"扩展和原档案列不一致：扩展缺{missing_in_extension}，原档案缺{missing_in_base}"
        )
    extension_data = extension_data[base_data.columns]
    combined = pd.concat([extension_data, base_data], ignore_index=True)
    keys = ["con_code", "report_period", "available_at"]
    if combined[keys].duplicated().any():
        raise ValueError("合并财务档案存在重复点时键")
    return combined.sort_values(keys).reset_index(drop=True)


def _make_client(transport: dict[str, Any]) -> Any:
    import tushare as ts

    pro = ts.pro_api(transport["token"])
    pro._DataApi__http_url = transport["api_url"]
    return pro


def _sleep_if_called(called: bool, config: dict[str, Any]) -> None:
    if called:
        interval = float(config["acquisition"]["request_interval_seconds"])
        minimum = float(config["acquisition"]["minimum_request_interval_seconds"])
        if interval < minimum:
            raise ValueError(f"请求间隔 {interval} 秒低于协议下限 {minimum} 秒")
        time.sleep(interval)


def run_acquisition(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行可断点续采的历史财务扩展并写出独立档案。"""

    config = config or load_config()
    started_at = datetime.now(ZoneInfo(config["protocol"]["timezone"]))
    frozen_before = verify_frozen_inputs(config)
    if frozen_before["status"] != "PASS":
        raise RuntimeError("采集前冻结输入哈希不一致")
    weights, target_symbols = load_target_universe(config)
    transport = resolve_transport(config)
    pro = _make_client(transport)
    acquisition = config["acquisition"]
    periods = build_quarter_periods(
        pd.Timestamp(acquisition["first_report_period"]),
        pd.Timestamp(acquisition["last_report_period"]),
    )
    checkpoint_root = ROOT / config["artifacts"]["checkpoint_directory"]
    api_period_frames: dict[str, dict[str, pd.DataFrame]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    network_call_count = 0

    for period in periods:
        for api_name in config["api_contracts"]:
            checkpoint = checkpoint_root / "vip" / api_name / f"{period}.parquet"
            frame, called = _load_or_download(
                pro, api_name, period, checkpoint, config
            )
            network_call_count += int(called)
            _sleep_if_called(called, config)
            api_period_frames.setdefault(api_name, {})[period] = frame
            checkpoint_rows.append(
                {
                    "api": api_name,
                    "period": period,
                    "file": checkpoint.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(checkpoint),
                    "network_call_performed": called,
                    **response_statistics(frame, target_symbols),
                }
            )
            print(
                f"财务扩展检查点 {api_name}/{period}：{len(frame)} 行",
                flush=True,
            )

    supplemental: dict[str, list[pd.DataFrame]] = {
        api_name: [] for api_name in config["api_contracts"]
    }
    continuity_report: dict[str, Any] = {}
    overlap_report: dict[str, Any] = {}
    for api_name in config["api_contracts"]:
        fields = api_fields(config, api_name)
        frozen_anchor_path = (
            ROOT
            / config["frozen_inputs"]["base_financial_checkpoints"]["directory"]
            / api_name
            / f"{acquisition['overlap_validation_period'].replace('-', '')}.parquet"
        )
        frozen_anchor = validate_api_response(
            pd.read_parquet(frozen_anchor_path),
            api_name,
            acquisition["overlap_validation_period"].replace("-", ""),
            config,
        )
        gaps = find_continuity_gaps(
            api_period_frames[api_name], frozen_anchor, target_symbols, periods
        )
        confirmed_empty: list[dict[str, str]] = []
        filled: list[dict[str, str]] = []
        boundary_history_symbols: list[str] = []
        history_checkpoint_rows: list[dict[str, Any]] = []
        gaps_by_code: dict[str, list[str]] = {}
        for code, period in gaps:
            gaps_by_code.setdefault(code, []).append(period)
        standard_limit = int(acquisition["single_security_standard_row_limit"])
        for code, code_periods in sorted(gaps_by_code.items()):
            safe_code = code.replace(".", "_")
            history_checkpoint = (
                checkpoint_root
                / "single_security_history"
                / api_name
                / f"{safe_code}.parquet"
            )
            history, called = _load_or_download_history(
                pro, api_name, code, history_checkpoint, config
            )
            network_call_count += int(called)
            _sleep_if_called(called, config)
            at_boundary = len(history) >= standard_limit
            if at_boundary:
                boundary_history_symbols.append(code)
            history_periods = (
                history["end_date"].astype("string").str.replace("-", "", regex=False).str[:8]
                if not history.empty else pd.Series(dtype="string")
            )
            history_checkpoint_rows.append(
                {
                    "con_code": code,
                    "file": history_checkpoint.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(history_checkpoint),
                    "row_count": len(history),
                    "at_standard_row_limit": at_boundary,
                    "network_call_performed": called,
                }
            )
            for period in sorted(code_periods):
                legacy_checkpoint = (
                    checkpoint_root
                    / "single_security_fallback"
                    / api_name
                    / period
                    / f"{safe_code}.parquet"
                )
                if legacy_checkpoint.exists():
                    legacy = validate_api_response(
                        pd.read_parquet(legacy_checkpoint),
                        api_name,
                        period,
                        config,
                        allow_empty=True,
                    )
                else:
                    legacy = pd.DataFrame(columns=api_fields(config, api_name))
                history_slice = history.loc[history_periods.eq(period)].copy()
                frame = legacy if not legacy.empty else history_slice
                needs_period_confirmation = frame.empty and at_boundary
                if needs_period_confirmation:
                    period_checkpoint = (
                        checkpoint_root
                        / "single_security_fallback"
                        / api_name
                        / period
                        / f"{safe_code}.parquet"
                    )
                    frame, period_called = _load_or_download(
                        pro,
                        api_name,
                        period,
                        period_checkpoint,
                        config,
                        ts_code=code,
                    )
                    network_call_count += int(period_called)
                    _sleep_if_called(period_called, config)
                if frame.empty:
                    confirmed_empty.append({"con_code": code, "period": period})
                else:
                    supplemental[api_name].append(frame)
                    filled.append({"con_code": code, "period": period})
        continuity_report[api_name] = {
            "initial_gap_count": len(gaps),
            "distinct_gap_symbol_count": len(gaps_by_code),
            "filled_gap_count": len(filled),
            "confirmed_empty_gap_count": len(confirmed_empty),
            "history_checkpoint_count": len(history_checkpoint_rows),
            "history_responses_at_standard_row_limit_count": len(boundary_history_symbols),
            "history_checkpoints": history_checkpoint_rows,
            "filled_gaps": filled,
            "confirmed_empty_gaps": confirmed_empty,
        }

        live_anchor_path = (
            checkpoint_root / "overlap_validation" / api_name
            / f"{acquisition['overlap_validation_period'].replace('-', '')}.parquet"
        )
        live_anchor, called = _load_or_download(
            pro,
            api_name,
            acquisition["overlap_validation_period"].replace("-", ""),
            live_anchor_path,
            config,
        )
        network_call_count += int(called)
        _sleep_if_called(called, config)
        overlap_report[api_name] = compare_responses(
            live_anchor, frozen_anchor, fields
        ) | {
            "current_checkpoint": live_anchor_path.relative_to(ROOT).as_posix(),
            "current_checkpoint_sha256": sha256_file(live_anchor_path),
            "frozen_checkpoint": frozen_anchor_path.relative_to(ROOT).as_posix(),
            "frozen_checkpoint_sha256": sha256_file(frozen_anchor_path),
        }

    combined_api: dict[str, pd.DataFrame] = {}
    for api_name, period_frames in api_period_frames.items():
        pieces = list(period_frames.values()) + supplemental[api_name]
        combined_api[api_name] = pd.concat(pieces, ignore_index=True)

    extension = build_extension_events(
        combined_api, target_symbols, started_at
    )
    base_contract = config["frozen_inputs"]["base_financial_archive"]
    base = pd.read_parquet(ROOT / base_contract["file"])
    extended = combine_financial_archives(
        base, extension, base_contract["first_report_period"]
    )
    extension_path = ROOT / config["artifacts"]["extension_events"]
    extended_path = ROOT / config["artifacts"]["extended_financial_archive"]
    _atomic_parquet(extension, extension_path)
    _atomic_parquet(extended, extended_path)

    frozen_after = verify_frozen_inputs(config)
    if frozen_after["status"] != "PASS":
        raise RuntimeError("采集后冻结输入哈希不一致")
    if frozen_after != frozen_before:
        raise RuntimeError("采集前后冻结输入审计结果发生变化")

    checkpoint_hash, checkpoint_files = canonical_directory_hash(checkpoint_root)
    finished_at = datetime.now(ZoneInfo(config["protocol"]["timezone"]))
    overlap_exact = all(
        item["multiset_exact_match"] for item in overlap_report.values()
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": (
            "PASS_TARGET_SCOPE_CURRENT_ARCHIVE_CAVEAT"
            if overlap_exact
            else "PASS_TARGET_SCOPE_VENDOR_REVISION_DETECTED_CAVEAT"
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "transport": sanitized_transport(transport),
        "official_documents": {
            api_name: contract["official_document"]
            for api_name, contract in config["api_contracts"].items()
        },
        "api_semantics": {
            "quarterly_vip_permission_points": 5000,
            "vip_pagination_documented": acquisition["vip_pagination_documented"],
            "completeness_strategy": acquisition["completeness_strategy"],
        },
        "target_universe": {
            "weight_snapshot_count": int(weights["trade_date"].nunique()),
            "target_symbol_count": len(target_symbols),
            "first_weight_snapshot": str(weights["trade_date"].min().date()),
            "last_weight_snapshot": str(weights["trade_date"].max().date()),
        },
        "acquisition_range": {
            "first_report_period": periods[0],
            "last_report_period": periods[-1],
            "quarter_count": len(periods),
            "vip_checkpoint_count": len(checkpoint_rows),
            "network_call_count_this_run": network_call_count,
        },
        "vip_checkpoints": checkpoint_rows,
        "target_continuity": continuity_report,
        "overlap_retrieval_cross_check": overlap_report,
        "frozen_inputs_before": frozen_before,
        "frozen_inputs_after": frozen_after,
        "extension_archive": {
            "file": extension_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(extension_path),
            "row_count": len(extension),
            "symbol_count": int(extension["con_code"].nunique()),
            "first_report_period": str(extension["report_period"].min().date()),
            "last_report_period": str(extension["report_period"].max().date()),
            "first_available_at": str(extension["available_at"].min().date()),
            "last_available_at": str(extension["available_at"].max().date()),
        },
        "extended_archive": {
            "file": extended_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(extended_path),
            "row_count": len(extended),
            "symbol_count": int(extended["con_code"].nunique()),
            "first_report_period": str(extended["report_period"].min().date()),
            "last_report_period": str(extended["report_period"].max().date()),
        },
        "checkpoint_archive": {
            "directory": checkpoint_root.relative_to(ROOT).as_posix(),
            "file_count": len(checkpoint_files),
            "content_sha256": checkpoint_hash,
        },
        "archive_caveat": "原始值在2026年统一回取；available_at重建历史可得时点，但不能证明供应商历史值从未追溯修订。",
        "governance": {
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "base_financial_archive_mutated": False,
            "base_checkpoint_archive_mutated": False,
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """把成功或阻塞报告渲染为简洁可审计的 Markdown。"""

    if not report["status"].startswith("PASS_"):
        return "\n".join(
            [
                "# VAL01_NORM_EY_5Y 财务历史扩展状态",
                "",
                f"> 状态：`{report['status']}`。未生成可供模型使用的扩展合并档案。",
                "",
                f"- 失败类别：`{report.get('failure_category')}`。",
                f"- 错误：{report.get('error')}。",
                f"- 数据时间：{report.get('checked_at')}。",
                "- 原冻结财务文件和原 132 个检查点未被改写。",
                "- 未计算收益、IC、仓位、股数或订单。",
                "",
            ]
        )
    extension = report["extension_archive"]
    extended = report["extended_archive"]
    return "\n".join(
        [
            "# VAL01_NORM_EY_5Y 财务历史扩展报告 V1",
            "",
            f"> 状态：`{report['status']}`。本阶段只通过数据采集闸门，不包含任何收益、IC或交易结论。",
            "",
            "## 范围",
            "",
            f"- 目标证券：{report['target_universe']['target_symbol_count']}只；权重快照：{report['target_universe']['weight_snapshot_count']}个月。",
            f"- 报告期：{report['acquisition_range']['first_report_period']}至{report['acquisition_range']['last_report_period']}，共{report['acquisition_range']['quarter_count']}个季度。",
            f"- VIP 基础检查点：{report['acquisition_range']['vip_checkpoint_count']}个。",
            "",
            "## 产物",
            "",
            f"- 扩展事件：{extension['row_count']}行、{extension['symbol_count']}只证券，SHA-256 `{extension['sha256']}`。",
            f"- 合并档案：{extended['row_count']}行、{extended['symbol_count']}只证券，SHA-256 `{extended['sha256']}`。",
            f"- 新检查点：{report['checkpoint_archive']['file_count']}个，目录内容哈希 `{report['checkpoint_archive']['content_sha256']}`。",
            "",
            "## 审计边界",
            "",
            "- 原财务文件及原 132 个检查点的采集前后哈希一致。",
            f"- {report['archive_caveat']}",
            "- 未计算收益、IC、仓位、股数或订单。",
            "",
        ]
    )


def write_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    json_path = ROOT / config["artifacts"]["acquisition_report_json"]
    markdown_path = ROOT / config["artifacts"]["acquisition_report_markdown"]
    _atomic_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        json_path,
    )
    _atomic_text(render_markdown(report), markdown_path)


def blocked_report(
    error: BaseException,
    config: dict[str, Any],
    transport: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成不会泄露凭据的失败报告。"""

    frozen = verify_frozen_inputs(config)
    secrets = [transport.get("token", "")] if transport else []
    category = classify_failure(error)
    return {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": category,
        "failure_category": category,
        "checked_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "error": _sanitize_error(error, secrets),
        "transport": sanitized_transport(transport) if transport else None,
        "frozen_inputs": frozen,
        "partial_checkpoint_note": "新扩展目录中的已完成检查点可用于安全断点续采；原冻结目录未改写。",
        "governance": {
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "base_financial_archive_mutated": False,
            "base_checkpoint_archive_mutated": False,
        },
    }
