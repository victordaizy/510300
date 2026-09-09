from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import yaml


PROJECT_ID = "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2"
BUNDLE_STATUS = "FROZEN_STRICT_FORWARD_ONLY_G0_GATED"
SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
CONTRACT_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
FORBIDDEN_INPUT_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "private_key",
    "secret",
    "token",
)


class ProtocolError(RuntimeError):
    """表示冻结协议、权限或不可恢复输入错误。"""


@dataclass(frozen=True)
class NoViewError(ProtocolError):
    """表示当天必须保留为 NO_VIEW 的数据合同失败。"""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"对象无法JSON序列化：{type(value).__name__}")


def canonical_json_bytes(value: Any, *, excluded_keys: Iterable[str] = ()) -> bytes:
    excluded = set(excluded_keys)

    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): clean(child)
                for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))
                if str(key) not in excluded
            }
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, (np.integer, np.floating, np.ndarray, datetime, date, Path)):
            return clean(_json_default(item))
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return json.dumps(
        clean(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any, *, excluded_keys: Iterable[str] = ()) -> str:
    return hashlib.sha256(canonical_json_bytes(value, excluded_keys=excluded_keys)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _absolute_without_link_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _approved_external_roots(root: Path, config: Mapping[str, Any]) -> tuple[Path, ...]:
    storage = config.get("storage")
    if not isinstance(storage, Mapping):
        raise ProtocolError("协议缺少显式外部数据根治理")
    if storage.get("external_root_must_match_exactly") is not True:
        raise ProtocolError("外部数据根必须启用精确匹配")
    logical_name = str(storage.get("logical_data_root", ""))
    if logical_name != "data":
        raise ProtocolError("逻辑数据根必须严格为data")
    logical_root = _absolute_without_link_resolution(root / logical_name)
    resolved_root = logical_root.resolve()
    approved_root = Path(str(storage.get("approved_resolved_data_root", ""))).resolve()
    if resolved_root != approved_root:
        raise ProtocolError(
            f"data Junction目标不匹配：actual={resolved_root}，approved={approved_root}"
        )
    return (approved_root,)


def resolve_path(
    root: Path,
    relative: str | Path,
    *,
    approved_external_roots: Iterable[Path] = (),
) -> Path:
    root_logical = _absolute_without_link_resolution(root)
    relative_path_value = Path(relative)
    if relative_path_value.is_absolute():
        raise ProtocolError(f"协议路径必须为项目相对路径：{relative}")
    candidate_logical = _absolute_without_link_resolution(root_logical / relative_path_value)
    try:
        candidate_logical.relative_to(root_logical)
    except ValueError as exc:
        raise ProtocolError(f"逻辑路径越出项目根目录：{relative}") from exc
    candidate_resolved = candidate_logical.resolve()
    allowed_resolved_roots = (root_logical.resolve(), *(path.resolve() for path in approved_external_roots))
    for allowed_root in allowed_resolved_roots:
        try:
            candidate_resolved.relative_to(allowed_root)
            return candidate_logical
        except ValueError:
            continue
    raise ProtocolError(f"解析后路径不在批准根目录：{relative} -> {candidate_resolved}")


def relative_path(root: Path, path: Path) -> str:
    root_logical = _absolute_without_link_resolution(root)
    path_logical = _absolute_without_link_resolution(path)
    try:
        return path_logical.relative_to(root_logical).as_posix()
    except ValueError as exc:
        raise ProtocolError(f"无法形成项目逻辑相对路径：{path}") from exc


def resolve_configured_path(root: Path, config: Mapping[str, Any], relative: str | Path) -> Path:
    return resolve_path(
        root,
        relative,
        approved_external_roots=_approved_external_roots(root, config),
    )


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"无法读取JSON对象：{path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON根节点不是对象：{path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"无法读取协议配置：{path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("协议配置根节点不是对象")
    if value.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ProtocolError("协议项目标识不匹配")
    validate_protocol_invariants(value)
    return value


def validate_protocol_invariants(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol.get("research_mode") != "STRICT_FORWARD_ONLY":
        raise ProtocolError("研究模式必须为STRICT_FORWARD_ONLY")
    false_fields = (
        "historical_backfill_allowed",
        "historical_tradable_backtest_allowed",
        "pre_g0_signal_collection_allowed",
        "pre_g1_model_estimation_allowed",
        "pre_g2_prediction_allowed",
        "live_trading_authorized",
        "broker_connection_authorized",
    )
    for field in false_fields:
        if protocol.get(field) is not False:
            raise ProtocolError(f"协议边界必须保持false：protocol.{field}")
    if float(protocol.get("position_impact", 1)) != 0:
        raise ProtocolError("当前仓位影响必须为0")
    if config["scope"].get("tradable_universe") != ["510300.SH", "CASH_CNY"]:
        raise ProtocolError("可执行范围必须严格为510300.SH与CASH_CNY")
    if config["scope"].get("signal_instrument_position") != 0:
        raise ProtocolError("A50信号工具仓位必须为0")
    if any(bool(value) for value in config["boundaries"].values()):
        raise ProtocolError("当前交易隔离开关必须全部关闭")

    clock = config["clock"]
    expected_clock = {
        "a50_start_window": ["15:00:05", "15:00:15"],
        "a50_end_window": ["15:03:20", "15:03:30"],
        "signal_calculation_deadline": "15:03:35",
        "research_intent_at": "15:03:45",
        "future_broker_ack_deadline": "15:04:15",
        "after_hours_matching_window": ["15:05:00", "15:30:00"],
        "target_window_next_sse_day": ["09:35:00", "09:35:30"],
    }
    for field, expected in expected_clock.items():
        if clock.get(field) != expected:
            raise ProtocolError(f"冻结时钟漂移：clock.{field}")
    if clock.get("prequeue_and_cancel_allowed") is not False:
        raise ProtocolError("主版本必须禁止预挂双向订单")

    data = config["signal_data"]
    if float(data["maximum_market_data_latency_seconds"]) != 3.0:
        raise ProtocolError("行情最大延迟必须冻结为3秒")
    if float(data["maximum_absolute_ntp_offset_milliseconds"]) != 100.0:
        raise ProtocolError("时钟偏差门必须冻结为100毫秒")
    if data.get("continuous_contract_allowed") is not False:
        raise ProtocolError("连续合约必须禁用")
    if data.get("contract_candidate_set") != "TWO_NEAREST_UNEXPIRED_LISTED_CONTRACT_MONTHS":
        raise ProtocolError("近月候选集没有精确定义为最近两个未到期挂牌月")

    costs = config["costs"]
    minimum_notional = float(costs["minimum_economic_overlay_notional_cny"])
    base_bps = round_trip_cost_bps(minimum_notional, config, scenario="BASE")
    stress_bps = round_trip_cost_bps(minimum_notional, config, scenario="STRESS")
    if not math.isclose(base_bps, 14.0, abs_tol=1e-12):
        raise ProtocolError("25,000元基础往返成本不是14bp")
    if not math.isclose(stress_bps, 24.0, abs_tol=1e-12):
        raise ProtocolError("25,000元压力往返成本不是24bp")
    if not math.isclose(float(config["signal_gates"]["research_tier_bps"]), 3 * base_bps):
        raise ProtocolError("Research Tier不是3倍基础成本")
    if not math.isclose(float(config["signal_gates"]["execution_tier_bps"]), 3 * stress_bps):
        raise ProtocolError("Execution Tier不是3倍压力成本")


def _file_inventory(
    root: Path,
    config: Mapping[str, Any],
    entries: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path, role in entries:
        path = resolve_configured_path(root, config, raw_path)
        relative = relative_path(root, path)
        if relative in seen:
            raise ProtocolError(f"冻结文件重复：{relative}")
        seen.add(relative)
        if not path.is_file():
            raise ProtocolError(f"冻结文件不存在：{relative}")
        inventory.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(inventory, key=lambda row: row["path"])


def freeze_bundle(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    if config["protocol"].get("state") != "PRE_FORWARD_ONLY_FREEZE":
        raise ProtocolError("协议不是冻结前状态")
    source_receipt_path = resolve_configured_path(root, config, config["artifacts"]["source_receipt"])
    source_receipt = load_json_object(source_receipt_path)
    if source_receipt.get("project_id") != PROJECT_ID:
        raise ProtocolError("来源回执项目标识不匹配")
    if source_receipt.get("sse_rule", {}).get("status") != "PASS_OFFICIAL_RULE_CURRENT":
        raise ProtocolError("上交所现行规则没有通过官方来源核验")

    manifest_path = resolve_configured_path(root, config, config["artifacts"]["freeze_manifest"])
    if manifest_path.exists():
        raise ProtocolError(f"冻结清单已存在，禁止覆盖：{manifest_path}")
    frozen_entries = [(path, "FROZEN_PROTOCOL_FILE") for path in config["freeze"]["files"]]
    dependency_entries = [
        (dependency["path"], dependency["role"])
        for dependency in config["immutable_dependencies"]
    ]
    inventory = _file_inventory(root, config, [*frozen_entries, *dependency_entries])
    configured_dependency_hashes = {
        dependency["path"]: str(dependency["sha256"]).lower()
        for dependency in config["immutable_dependencies"]
    }
    for entry in inventory:
        expected = configured_dependency_hashes.get(entry["path"])
        if expected is not None and entry["sha256"] != expected:
            raise ProtocolError(f"不可变依赖哈希不匹配：{entry['path']}")

    manifest: dict[str, Any] = {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_FREEZE_MANIFEST_V1",
        "project_id": PROJECT_ID,
        "protocol_version": config["protocol"]["version"],
        "bundle_status": BUNDLE_STATUS,
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "forward_start_rule": config["protocol"]["forward_start_rule"],
        "not_before_trade_date": config["protocol"]["not_before_trade_date"],
        "historical_return_read_before_freeze": False,
        "historical_backfill_allowed": False,
        "authorized_order_quantity_shares": 0,
        "position_impact": 0,
        "live_trading_authorized": False,
        "git_commit_required_before_forward_collection": True,
        "frozen_files": inventory,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest, excluded_keys=("manifest_sha256",)
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "status": BUNDLE_STATUS,
        "manifest_path": relative_path(root, manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_file_count": len(inventory),
        "position_impact": 0,
    }


def validate_manifest(
    root: Path,
    config: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_path = resolve_configured_path(root, config, config["artifacts"]["freeze_manifest"])
    manifest = load_json_object(manifest_path)
    if manifest.get("project_id") != PROJECT_ID or manifest.get("bundle_status") != BUNDLE_STATUS:
        raise ProtocolError("冻结清单身份或状态不匹配")
    actual_manifest_hash = canonical_json_sha256(manifest, excluded_keys=("manifest_sha256",))
    recorded_hash = str(manifest.get("manifest_sha256", "")).lower()
    if recorded_hash != actual_manifest_hash:
        raise ProtocolError("冻结清单自身摘要不匹配")
    if expected_manifest_sha256.lower() != actual_manifest_hash:
        raise ProtocolError("调用方预期的冻结清单摘要不匹配")
    for entry in manifest.get("frozen_files", []):
        path = resolve_configured_path(root, config, entry["path"])
        if not path.is_file():
            raise ProtocolError(f"冻结文件缺失：{entry['path']}")
        if path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"冻结文件发生漂移：{entry['path']}")
    return manifest


def validate_bundle(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    manifest = validate_manifest(root, config, expected_manifest_sha256)
    return config, manifest


def _git(*args: str, root: Path, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace") if binary else completed.stderr
        raise ProtocolError(f"Git命令失败：git {' '.join(args)}；{str(stderr).strip()}")
    return completed.stdout


def verify_frozen_git_commit(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], commit: str) -> dict[str, Any]:
    commit = str(commit).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ProtocolError("冻结提交标识格式无效")
    full_commit = str(_git("rev-parse", "--verify", f"{commit}^{{commit}}", root=root)).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", full_commit, "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise ProtocolError("冻结提交不是当前HEAD的祖先")
    expected_files = {
        str(entry["path"]): str(entry["sha256"])
        for entry in manifest["frozen_files"]
        if entry["role"] == "FROZEN_PROTOCOL_FILE"
    }
    manifest_path = config["artifacts"]["freeze_manifest"]
    expected_files[manifest_path] = sha256_file(resolve_configured_path(root, config, manifest_path))
    mismatches: list[str] = []
    for path, expected_hash in expected_files.items():
        try:
            payload = _git("show", f"{full_commit}:{path}", root=root, binary=True)
        except ProtocolError:
            mismatches.append(f"{path}:提交中缺失")
            continue
        assert isinstance(payload, bytes)
        if sha256_bytes(payload) != expected_hash:
            mismatches.append(f"{path}:提交内容不匹配")
    if mismatches:
        raise ProtocolError(f"冻结提交未完整包含协议包：{mismatches}")
    return {"commit": full_commit, "verified_file_count": len(expected_files)}


def _validate_sha256(value: Any, field: str) -> str:
    text = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(text):
        raise NoViewError("NO_VIEW_INVALID_SHA256", f"{field}不是64位SHA-256")
    return text


def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field}不是ISO日期") from exc


def _parse_timestamp(value: Any, field: str, *, no_view: bool = True) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        error = NoViewError("NO_VIEW_INVALID_TIMESTAMP", f"{field}无法解析") if no_view else ProtocolError(f"{field}无法解析")
        raise error from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        error = NoViewError("NO_VIEW_TIMEZONE_MISSING", f"{field}必须包含时区") if no_view else ProtocolError(f"{field}必须包含时区")
        raise error
    return parsed.astimezone(SHANGHAI)


def _clock_at(trade_date: date, clock: str) -> datetime:
    return datetime.combine(trade_date, time.fromisoformat(clock), tzinfo=SHANGHAI)


def _between(timestamp: datetime, trade_date: date, window: Sequence[str]) -> bool:
    return _clock_at(trade_date, window[0]) <= timestamp <= _clock_at(trade_date, window[1])


def _require_keys(payload: Mapping[str, Any], required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(payload))
    if missing:
        raise NoViewError("NO_VIEW_MISSING_REQUIRED_FIELDS", f"{context}缺少字段：{missing}")


def _reject_embedded_credentials(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in FORBIDDEN_INPUT_KEY_FRAGMENTS):
                raise ProtocolError(f"输入不得包含凭据字段：{path}.{key}")
            _reject_embedded_credentials(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_credentials(child, f"{path}[{index}]")


def load_sse_trade_dates(root: Path, config: Mapping[str, Any]) -> list[date]:
    calendar_dependency = next(
        item for item in config["immutable_dependencies"] if item["role"] == "OFFICIAL_SSE_2026_TRADING_CALENDAR"
    )
    path = resolve_configured_path(root, config, calendar_dependency["path"])
    if sha256_file(path) != str(calendar_dependency["sha256"]).lower():
        raise ProtocolError("冻结上交所交易日历发生漂移")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    dates = sorted({_parse_date(row["trade_date"], "trade_date") for row in rows})
    if not dates:
        raise ProtocolError("冻结上交所交易日历为空")
    return dates


def next_sse_trade_date(root: Path, config: Mapping[str, Any], signal_date: date) -> date:
    later = [candidate for candidate in load_sse_trade_dates(root, config) if candidate > signal_date]
    if not later:
        raise ProtocolError("冻结交易日历没有下一上交所交易日")
    return later[0]


def assert_authoritative_trade_date(trade_date: date, config: Mapping[str, Any], now: datetime) -> None:
    local_now = now.astimezone(SHANGHAI)
    if trade_date != local_now.date():
        raise ProtocolError("只允许记录上海时区当天数据，禁止补跑或回填")
    not_before = _parse_date(config["protocol"]["not_before_trade_date"], "not_before_trade_date")
    if trade_date < not_before:
        raise ProtocolError(f"权威前向日期不得早于{not_before.isoformat()}")


def evaluate_g0(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if evidence.get("project_id") != PROJECT_ID:
        raise ProtocolError("G0证据项目标识不匹配")
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        raise ProtocolError("G0证据缺少checks对象")
    required = list(config["sequential_gates"]["g0"]["required_checks"])
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for check_id in required:
        raw = checks.get(check_id)
        if not isinstance(raw, Mapping):
            row = {"check_id": check_id, "status": "NOT_PROVIDED", "evidence": "未提供证据"}
        else:
            row = {
                "check_id": check_id,
                "status": str(raw.get("status", "NOT_PROVIDED")),
                "evidence": str(raw.get("evidence", "")),
            }
            for optional in ("source_url", "source_sha256", "checked_at", "git_commit"):
                if optional in raw:
                    row[optional] = raw[optional]
        if check_id == "CODE_AND_MANIFEST_FROZEN_COMMIT_PRESENT" and row["status"] == "PASS":
            try:
                verification = verify_frozen_git_commit(root, config, manifest, str(raw.get("git_commit", "")))
                row.update(verification)
            except ProtocolError as exc:
                row["status"] = "FAIL"
                row["evidence"] = str(exc)
        if row["status"] != "PASS":
            blockers.append(check_id)
        rows.append(row)
    pass_state = config["sequential_gates"]["g0"]["pass_state"]
    fail_state = config["sequential_gates"]["g0"]["fail_state"]
    return {
        "g0_state": pass_state if not blockers else fail_state,
        "all_pass": not blockers,
        "checks": rows,
        "blockers": blockers,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"JSONL第{line_number}行损坏：{path}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"JSONL第{line_number}行不是对象：{path}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, default=_json_default)
        for row in rows
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _append_unique_jsonl(path: Path, row: Mapping[str, Any], key: str) -> None:
    rows = _read_jsonl(path)
    value = row.get(key)
    if any(existing.get(key) == value for existing in rows):
        raise ProtocolError(f"账本键已存在，禁止重复追加：{key}={value}")
    _write_jsonl(path, [*rows, dict(row)])


def _archive_raw_payload(directory: Path, trade_date: date, stem: str, raw: bytes) -> tuple[Path, str]:
    digest = sha256_bytes(raw)
    target = directory / trade_date.isoformat() / f"{stem}_{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise ProtocolError(f"原始载荷归档路径冲突：{target}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        try:
            Path(temporary_name).write_bytes(raw)
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    return target, digest


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NoViewError("NO_VIEW_INVALID_PRICE_OR_NUMBER", f"{field}不是数字") from exc
    if not math.isfinite(number) or number <= 0:
        raise NoViewError("NO_VIEW_INVALID_PRICE_OR_NUMBER", f"{field}必须为有限正数")
    return number


def _nonnegative_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NoViewError("NO_VIEW_INVALID_PRICE_OR_NUMBER", f"{field}不是数字") from exc
    if not math.isfinite(number) or number < 0:
        raise NoViewError("NO_VIEW_INVALID_PRICE_OR_NUMBER", f"{field}必须为有限非负数")
    return number


def _validate_latency(exchange: datetime, received: datetime, maximum_seconds: float, context: str) -> None:
    latency = (received - exchange).total_seconds()
    if latency < 0:
        raise NoViewError("NO_VIEW_RECEIVED_BEFORE_EXCHANGE_TIMESTAMP", f"{context}received_at早于exchange_timestamp")
    if latency > maximum_seconds:
        raise NoViewError("NO_VIEW_QUOTE_STALENESS_EXCEEDED", f"{context}到达延迟{latency:.6f}秒超过{maximum_seconds}秒")


def select_frozen_contract(payload: Mapping[str, Any], config: Mapping[str, Any], trade_date: date) -> str:
    candidates = payload.get("contract_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise NoViewError("NO_VIEW_CONTRACT_CANDIDATE_SET_INCOMPLETE", "至少需要两个挂牌合约候选")
    maximum_latency = float(config["signal_data"]["maximum_market_data_latency_seconds"])
    valid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise NoViewError("NO_VIEW_CONTRACT_IDENTITY_UNPROVEN", f"contract_candidates[{index}]不是对象")
        _require_keys(
            candidate,
            (
                "contract_id",
                "contract_month",
                "expiry_date",
                "is_continuous",
                "cumulative_volume_0900_1455",
                "volume_exchange_timestamp",
                "volume_received_at",
            ),
            f"contract_candidates[{index}]",
        )
        contract_id = str(candidate["contract_id"]).strip()
        if not contract_id or contract_id in seen_ids:
            raise NoViewError("NO_VIEW_CONTRACT_IDENTITY_UNPROVEN", "合约代码为空或重复")
        seen_ids.add(contract_id)
        if candidate["is_continuous"] is not False:
            raise NoViewError("NO_VIEW_CONTINUOUS_CONTRACT_FORBIDDEN", f"{contract_id}不是可接受的实际挂牌合约")
        contract_month = str(candidate["contract_month"])
        if not CONTRACT_MONTH_PATTERN.fullmatch(contract_month):
            raise NoViewError("NO_VIEW_CONTRACT_IDENTITY_UNPROVEN", f"{contract_id}合约月份无效")
        expiry = _parse_date(candidate["expiry_date"], f"{contract_id}.expiry_date")
        if expiry < trade_date:
            continue
        volume = _nonnegative_number(candidate["cumulative_volume_0900_1455"], f"{contract_id}.volume")
        exchange = _parse_timestamp(candidate["volume_exchange_timestamp"], f"{contract_id}.volume_exchange_timestamp")
        received = _parse_timestamp(candidate["volume_received_at"], f"{contract_id}.volume_received_at")
        if not _between(exchange, trade_date, config["clock"]["contract_volume_asof_window"]):
            raise NoViewError("NO_VIEW_CONTRACT_VOLUME_ASOF_INVALID", f"{contract_id}成交量时点不在冻结窗口")
        _validate_latency(exchange, received, maximum_latency, f"{contract_id}成交量")
        valid.append(
            {
                "contract_id": contract_id,
                "contract_month": contract_month,
                "expiry_date": expiry,
                "volume": volume,
                "received_at": received,
            }
        )
    if len(valid) < 2:
        raise NoViewError("NO_VIEW_CONTRACT_CANDIDATE_SET_INCOMPLETE", "未到期实际挂牌合约不足两个")
    valid.sort(key=lambda row: (row["contract_month"], row["expiry_date"], row["contract_id"]))
    nearby = valid[:2]
    maximum = max(row["volume"] for row in nearby)
    winners = [row for row in nearby if row["volume"] == maximum]
    if len(winners) != 1:
        raise NoViewError("NO_VIEW_CONTRACT_VOLUME_TIE", "两个近月合约成交量并列")
    frozen_at = _parse_timestamp(payload.get("contract_frozen_at"), "contract_frozen_at")
    if not _between(frozen_at, trade_date, config["clock"]["contract_freeze_window"]):
        raise NoViewError("NO_VIEW_CONTRACT_FREEZE_TIME_INVALID", "合约未在14:55冻结窗口内锁定")
    if any(frozen_at < row["received_at"] for row in nearby):
        raise NoViewError("NO_VIEW_CONTRACT_FROZEN_BEFORE_VOLUME_RECEIVED", "合约冻结早于近月成交量到达")
    selected = str(payload.get("selected_contract_id", "")).strip()
    if selected != winners[0]["contract_id"]:
        raise NoViewError("NO_VIEW_CONTRACT_SELECTION_MISMATCH", "载荷选定合约不符合冻结成交量规则")
    return selected


def _window_price(
    quotes: Sequence[Any],
    contract_id: str,
    trade_date: date,
    window: Sequence[str],
    config: Mapping[str, Any],
    label: str,
) -> tuple[float, str, int, datetime]:
    maximum_latency = float(config["signal_data"]["maximum_market_data_latency_seconds"])
    midpoint_values: list[float] = []
    trade_values: list[float] = []
    latest_received: datetime | None = None
    matched = 0
    for index, raw_quote in enumerate(quotes):
        if not isinstance(raw_quote, Mapping):
            raise NoViewError("NO_VIEW_INVALID_QUOTE", f"quotes[{index}]不是对象")
        if str(raw_quote.get("contract_id", "")).strip() != contract_id:
            continue
        exchange = _parse_timestamp(raw_quote.get("exchange_timestamp"), f"quotes[{index}].exchange_timestamp")
        if not _between(exchange, trade_date, window):
            continue
        received = _parse_timestamp(raw_quote.get("received_at"), f"quotes[{index}].received_at")
        _validate_latency(exchange, received, maximum_latency, f"{label}quotes[{index}]")
        latest_received = received if latest_received is None else max(latest_received, received)
        matched += 1
        bid_raw = raw_quote.get("bid")
        ask_raw = raw_quote.get("ask")
        if bid_raw is not None and ask_raw is not None:
            bid = _positive_number(bid_raw, f"quotes[{index}].bid")
            ask = _positive_number(ask_raw, f"quotes[{index}].ask")
            if ask < bid:
                raise NoViewError("NO_VIEW_CROSSED_QUOTE", f"{label}卖一低于买一")
            midpoint_values.append((bid + ask) / 2.0)
        elif raw_quote.get("last") is not None:
            trade_values.append(_positive_number(raw_quote["last"], f"quotes[{index}].last"))
    if midpoint_values:
        assert latest_received is not None
        return float(median(midpoint_values)), "MEDIAN_FRESH_BID_ASK_MIDPOINT", matched, latest_received
    minimum_trades = int(config["signal_data"]["minimum_trade_count_fallback"])
    if len(trade_values) >= minimum_trades:
        assert latest_received is not None
        return float(median(trade_values)), "MEDIAN_ELIGIBLE_TRADES", len(trade_values), latest_received
    raise NoViewError("NO_VIEW_INSUFFICIENT_WINDOW_QUOTES", f"{label}既无双边报价也不足{minimum_trades}笔成交")


def build_daily_observation(
    payload: Mapping[str, Any],
    raw_payload_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "trade_date",
        "collection_started_at",
        "local_ntp_offset_ms",
        "contract_candidates",
        "selected_contract_id",
        "contract_frozen_at",
        "quotes",
        "source_entitlement_id",
        "source_entitlement_evidence_sha256",
        "sse_rule_state",
        "sgx_session_state",
        "etf_state",
        "company_action_state",
        "etf_close",
        "etf_close_source_id",
        "etf_close_source_sha256",
        "signal_calculated_at",
    )
    _require_keys(payload, required, "每日A50窗口载荷")
    trade_date = _parse_date(payload["trade_date"], "trade_date")
    states = config["signal_data"]["required_daily_states"]
    for field, expected in states.items():
        if payload.get(field) != expected:
            raise NoViewError("NO_VIEW_REQUIRED_DAILY_STATE_FAILED", f"{field}={payload.get(field)}，要求={expected}")
    try:
        ntp_offset = float(payload["local_ntp_offset_ms"])
    except (TypeError, ValueError) as exc:
        raise NoViewError("NO_VIEW_CLOCK_OFFSET_UNPROVEN", "local_ntp_offset_ms不是数字") from exc
    maximum_offset = float(config["signal_data"]["maximum_absolute_ntp_offset_milliseconds"])
    if not math.isfinite(ntp_offset) or abs(ntp_offset) > maximum_offset:
        raise NoViewError("NO_VIEW_CLOCK_OFFSET_EXCEEDED", f"本地时钟偏差{ntp_offset}ms超过±{maximum_offset}ms")
    collection_started = _parse_timestamp(payload["collection_started_at"], "collection_started_at")
    if collection_started.date() != trade_date or collection_started > _clock_at(trade_date, config["clock"]["health_check_at"]):
        raise NoViewError("NO_VIEW_COLLECTION_STARTED_LATE", "采集未在14:55健康检查时点前启动")
    selected_contract = select_frozen_contract(payload, config, trade_date)
    quotes = payload["quotes"]
    if not isinstance(quotes, list):
        raise NoViewError("NO_VIEW_INVALID_QUOTE", "quotes必须为数组")
    f0, f0_method, f0_count, f0_received = _window_price(
        quotes, selected_contract, trade_date, config["clock"]["a50_start_window"], config, "A50起点"
    )
    f1, f1_method, f1_count, f1_received = _window_price(
        quotes, selected_contract, trade_date, config["clock"]["a50_end_window"], config, "A50终点"
    )
    signal_calculated_at = _parse_timestamp(payload["signal_calculated_at"], "signal_calculated_at")
    deadline = _clock_at(trade_date, config["clock"]["signal_calculation_deadline"])
    if signal_calculated_at < max(f0_received, f1_received) or signal_calculated_at > deadline:
        raise NoViewError("NO_VIEW_SIGNAL_DEADLINE_FAILED", "信号未在全部行情到达后且15:03:35前计算")
    entitlement_id = str(payload["source_entitlement_id"]).strip()
    close_source_id = str(payload["etf_close_source_id"]).strip()
    if not entitlement_id or not close_source_id:
        raise NoViewError("NO_VIEW_SOURCE_IDENTITY_UNPROVEN", "行情权利或510300收盘价来源标识为空")
    entitlement_hash = _validate_sha256(payload["source_entitlement_evidence_sha256"], "source_entitlement_evidence_sha256")
    close_hash = _validate_sha256(payload["etf_close_source_sha256"], "etf_close_source_sha256")
    raw_hash = _validate_sha256(raw_payload_sha256, "raw_payload_sha256")
    close_price = _positive_number(payload["etf_close"], "etf_close")
    return {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_OBSERVATION_V1",
        "project_id": PROJECT_ID,
        "trade_date": trade_date.isoformat(),
        "observation_state": "ELIGIBLE_RAW_FORWARD_OBSERVATION_PENDING_TARGET",
        "recorded_at": datetime.now(tz=SHANGHAI).isoformat(),
        "collection_started_at": collection_started.isoformat(),
        "local_ntp_offset_ms": ntp_offset,
        "a50_contract_id": selected_contract,
        "a50_contract_frozen_at": _parse_timestamp(payload["contract_frozen_at"], "contract_frozen_at").isoformat(),
        "a50_start_price": f0,
        "a50_start_price_method": f0_method,
        "a50_start_eligible_event_count": f0_count,
        "a50_end_price": f1,
        "a50_end_price_method": f1_method,
        "a50_end_eligible_event_count": f1_count,
        "x_log_return": math.log(f1 / f0),
        "source_entitlement_id": entitlement_id,
        "source_entitlement_evidence_sha256": entitlement_hash,
        "source_payload_sha256": raw_hash,
        "sse_rule_state": payload["sse_rule_state"],
        "sgx_session_state": payload["sgx_session_state"],
        "etf_state": payload["etf_state"],
        "company_action_state": payload["company_action_state"],
        "etf_close": close_price,
        "etf_close_source_id": close_source_id,
        "etf_close_source_sha256": close_hash,
        "signal_calculated_at": signal_calculated_at.isoformat(),
        "model_vintage_id": None,
        "mu_hat": None,
        "mu_se": None,
        "lcb_975": None,
        "ucb_975": None,
        "research_tier_state": "NOT_EVALUATED_BEFORE_G2",
        "execution_tier_state": "NOT_AUTHORIZED",
        "no_view_reason": None,
        "current_position_target": "UNSET",
        "position_impact": 0,
    }


def _no_view_observation(payload: Mapping[str, Any], raw_hash: str, error: NoViewError) -> dict[str, Any]:
    return {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_OBSERVATION_V1",
        "project_id": PROJECT_ID,
        "trade_date": str(payload["trade_date"]),
        "observation_state": "NO_VIEW",
        "recorded_at": datetime.now(tz=SHANGHAI).isoformat(),
        "source_payload_sha256": raw_hash,
        "no_view_reason": error.code,
        "no_view_detail": error.detail,
        "model_vintage_id": None,
        "mu_hat": None,
        "mu_se": None,
        "lcb_975": None,
        "ucb_975": None,
        "research_tier_state": "NO_VIEW",
        "execution_tier_state": "NOT_AUTHORIZED",
        "current_position_target": "UNSET",
        "position_impact": 0,
    }


def record_daily_capture(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    evidence_path: Path,
    payload_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    evidence = load_json_object(evidence_path)
    g0 = evaluate_g0(root, config, manifest, evidence)
    if not g0["all_pass"]:
        raise ProtocolError(f"G0未通过，禁止权威前向采集：{g0['blockers']}")
    raw = payload_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("每日载荷不是UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("每日载荷根节点不是对象")
    _reject_embedded_credentials(payload)
    if "trade_date" not in payload:
        raise ProtocolError("每日载荷缺少trade_date，无法建立NO_VIEW日期")
    trade_date = _parse_date(payload["trade_date"], "trade_date")
    current = now or datetime.now(tz=SHANGHAI)
    assert_authoritative_trade_date(trade_date, config, current)
    if trade_date not in load_sse_trade_dates(root, config):
        raise ProtocolError("输入日期不是冻结日历中的上交所交易日")
    ledger_path = resolve_configured_path(root, config, config["artifacts"]["observation_ledger"])
    if any(row.get("trade_date") == trade_date.isoformat() for row in _read_jsonl(ledger_path)):
        raise ProtocolError("该交易日已经写入观察账，禁止重复")
    archive_dir = resolve_configured_path(root, config, config["artifacts"]["raw_capture_directory"])
    archived_path, raw_hash = _archive_raw_payload(archive_dir, trade_date, "a50_window", raw)
    try:
        row = build_daily_observation(payload, raw_hash, config)
    except NoViewError as exc:
        row = _no_view_observation(payload, raw_hash, exc)
    row["raw_payload_path"] = relative_path(root, archived_path)
    row["g0_evidence_sha256"] = sha256_file(evidence_path)
    _append_unique_jsonl(ledger_path, row, "trade_date")
    return row


def build_target_maturity(
    observation: Mapping[str, Any],
    payload: Mapping[str, Any],
    raw_payload_sha256: str,
    expected_exit_date: date,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "signal_trade_date",
        "exit_trade_date",
        "window_first_exchange_timestamp",
        "window_last_exchange_timestamp",
        "received_at",
        "matured_at",
        "exit_window_bid_vwap",
        "exit_window_ask_vwap",
        "exit_window_mid",
        "exit_quote_depth_state",
        "company_action_state",
        "target_source_id",
    )
    _require_keys(payload, required, "目标成熟载荷")
    signal_date = _parse_date(payload["signal_trade_date"], "signal_trade_date")
    if signal_date.isoformat() != observation.get("trade_date"):
        raise ProtocolError("目标载荷的信号日期与观察账不匹配")
    exit_date = _parse_date(payload["exit_trade_date"], "exit_trade_date")
    if exit_date != expected_exit_date:
        raise ProtocolError("目标日期不是冻结日历中的下一上交所交易日")
    first_exchange = _parse_timestamp(payload["window_first_exchange_timestamp"], "window_first_exchange_timestamp")
    last_exchange = _parse_timestamp(payload["window_last_exchange_timestamp"], "window_last_exchange_timestamp")
    received_at = _parse_timestamp(payload["received_at"], "received_at")
    matured_at = _parse_timestamp(payload["matured_at"], "matured_at")
    target_window = config["clock"]["target_window_next_sse_day"]
    if not _between(first_exchange, exit_date, target_window) or not _between(last_exchange, exit_date, target_window):
        raise NoViewError("NO_VIEW_TARGET_WINDOW_INVALID", "09:35目标行情不在冻结窗口")
    if last_exchange < first_exchange:
        raise NoViewError("NO_VIEW_TARGET_WINDOW_INVALID", "目标行情结束时间早于开始时间")
    _validate_latency(
        last_exchange,
        received_at,
        float(config["signal_data"]["maximum_market_data_latency_seconds"]),
        "510300目标窗口",
    )
    window_end = _clock_at(exit_date, target_window[1])
    if matured_at < max(received_at, window_end):
        raise NoViewError("NO_VIEW_TARGET_MATURED_TOO_EARLY", "目标在09:35:30或行情到达前被标记成熟")
    base = {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_TARGET_V1",
        "project_id": PROJECT_ID,
        "signal_trade_date": signal_date.isoformat(),
        "exit_trade_date": exit_date.isoformat(),
        "matured_at": matured_at.isoformat(),
        "target_payload_sha256": _validate_sha256(raw_payload_sha256, "target_payload_sha256"),
        "position_impact": 0,
    }
    required_company_action = config["signal_data"]["required_daily_states"]["company_action_state"]
    if payload["company_action_state"] != required_company_action:
        return {
            **base,
            "target_state": "NO_VIEW_COMPANY_ACTION_OR_TOTAL_RETURN_UNPROVEN",
            "no_view_reason": "NO_VIEW_COMPANY_ACTION_OR_TOTAL_RETURN_UNPROVEN",
            "y_mid_log_return": None,
        }
    if payload["exit_quote_depth_state"] != config["target"]["depth_state_required"]:
        return {
            **base,
            "target_state": "NO_VIEW_TARGET_DEPTH_UNPROVEN",
            "no_view_reason": "NO_VIEW_TARGET_DEPTH_UNPROVEN",
            "y_mid_log_return": None,
        }
    source_id = str(payload["target_source_id"]).strip()
    if not source_id:
        raise NoViewError("NO_VIEW_SOURCE_IDENTITY_UNPROVEN", "目标来源标识为空")
    bid = _positive_number(payload["exit_window_bid_vwap"], "exit_window_bid_vwap")
    ask = _positive_number(payload["exit_window_ask_vwap"], "exit_window_ask_vwap")
    midpoint = _positive_number(payload["exit_window_mid"], "exit_window_mid")
    if not bid <= midpoint <= ask:
        raise NoViewError("NO_VIEW_TARGET_PRICE_ORDER_INVALID", "目标价格不满足bid<=mid<=ask")
    close_price = _positive_number(observation.get("etf_close"), "observation.etf_close")
    base_cost = round_trip_cost_fraction(
        float(config["costs"]["minimum_economic_overlay_notional_cny"]), config, scenario="BASE"
    )
    stress_cost = round_trip_cost_fraction(
        float(config["costs"]["minimum_economic_overlay_notional_cny"]), config, scenario="STRESS"
    )
    buy_gross = math.log(bid / close_price)
    sell_gross = math.log(close_price / ask)
    return {
        **base,
        "target_state": "MATURE_ELIGIBLE",
        "no_view_reason": None,
        "target_source_id": source_id,
        "window_first_exchange_timestamp": first_exchange.isoformat(),
        "window_last_exchange_timestamp": last_exchange.isoformat(),
        "received_at": received_at.isoformat(),
        "exit_quote_depth_state": payload["exit_quote_depth_state"],
        "exit_window_bid_vwap": bid,
        "exit_window_ask_vwap": ask,
        "exit_window_mid": midpoint,
        "y_mid_log_return": math.log(midpoint / close_price),
        "buy_executable_gross_log_return": buy_gross,
        "sell_executable_gross_log_return": sell_gross,
        "buy_base_cost_net_log_return": buy_gross - base_cost,
        "buy_stress_cost_net_log_return": buy_gross - stress_cost,
        "sell_base_cost_net_log_return": sell_gross - base_cost,
        "sell_stress_cost_net_log_return": sell_gross - stress_cost,
    }


def mature_target(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    evidence_path: Path,
    payload_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    evidence = load_json_object(evidence_path)
    g0 = evaluate_g0(root, config, manifest, evidence)
    if not g0["all_pass"]:
        raise ProtocolError(f"G0未通过，禁止目标成熟：{g0['blockers']}")
    raw = payload_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("目标载荷不是UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("目标载荷根节点不是对象")
    _reject_embedded_credentials(payload)
    signal_date = _parse_date(payload.get("signal_trade_date"), "signal_trade_date")
    exit_date = _parse_date(payload.get("exit_trade_date"), "exit_trade_date")
    current = now or datetime.now(tz=SHANGHAI)
    assert_authoritative_trade_date(exit_date, config, current)
    observations = _read_jsonl(resolve_configured_path(root, config, config["artifacts"]["observation_ledger"]))
    matches = [row for row in observations if row.get("trade_date") == signal_date.isoformat()]
    if len(matches) != 1 or matches[0].get("observation_state") != "ELIGIBLE_RAW_FORWARD_OBSERVATION_PENDING_TARGET":
        raise ProtocolError("找不到唯一且合格的待成熟观察")
    expected_exit = next_sse_trade_date(root, config, signal_date)
    target_ledger = resolve_configured_path(root, config, config["artifacts"]["target_ledger"])
    if any(row.get("signal_trade_date") == signal_date.isoformat() for row in _read_jsonl(target_ledger)):
        raise ProtocolError("该信号日期已有目标记录，禁止重复")
    archive_dir = resolve_configured_path(root, config, config["artifacts"]["raw_target_directory"])
    archived_path, raw_hash = _archive_raw_payload(archive_dir, exit_date, "target_window", raw)
    try:
        row = build_target_maturity(matches[0], payload, raw_hash, expected_exit, config)
    except NoViewError as exc:
        row = {
            "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_TARGET_V1",
            "project_id": PROJECT_ID,
            "signal_trade_date": signal_date.isoformat(),
            "exit_trade_date": exit_date.isoformat(),
            "target_state": "NO_VIEW",
            "no_view_reason": exc.code,
            "no_view_detail": exc.detail,
            "target_payload_sha256": raw_hash,
            "y_mid_log_return": None,
            "position_impact": 0,
        }
    row["raw_payload_path"] = relative_path(root, archived_path)
    _append_unique_jsonl(target_ledger, row, "signal_trade_date")
    return row


def round_trip_cost_fraction(notional_cny: float, config: Mapping[str, Any], *, scenario: str) -> float:
    if not math.isfinite(notional_cny) or notional_cny <= 0:
        raise ProtocolError("名义本金必须为有限正数")
    costs = config["costs"]
    commission = max(float(costs["commission_rate_per_leg"]) * notional_cny, float(costs["minimum_commission_cny_per_leg"]))
    if scenario == "BASE":
        slippage_bps = float(costs["base_slippage_bps_per_leg"])
    elif scenario == "STRESS":
        slippage_bps = float(costs["stress_slippage_bps_per_leg"])
    else:
        raise ProtocolError(f"未知成本情景：{scenario}")
    return 2.0 * commission / notional_cny + 2.0 * slippage_bps / 10000.0


def round_trip_cost_bps(notional_cny: float, config: Mapping[str, Any], *, scenario: str) -> float:
    return round_trip_cost_fraction(notional_cny, config, scenario=scenario) * 10000.0


def fit_hac_ols(x: Sequence[float], y: Sequence[float], *, minimum_observations: int) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values):
        raise ProtocolError("OLS输入必须为等长一维数组")
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    sample_n = int(len(x_values))
    if sample_n < minimum_observations:
        return {
            "status": "NO_VIEW_INSUFFICIENT_MATURE_OBSERVATIONS",
            "sample_n": sample_n,
            "minimum_sample_n": minimum_observations,
            "alpha": None,
            "beta": None,
            "covariance": None,
            "hac_lag": None,
        }
    design = np.column_stack([np.ones(sample_n), x_values])
    if np.linalg.matrix_rank(design) < 2:
        return {
            "status": "NO_VIEW_REGRESSOR_VARIATION_ZERO",
            "sample_n": sample_n,
            "minimum_sample_n": minimum_observations,
            "alpha": None,
            "beta": None,
            "covariance": None,
            "hac_lag": None,
        }
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ y_values
    residuals = y_values - design @ coefficients
    lag = int(math.floor(4.0 * (sample_n / 100.0) ** (2.0 / 9.0)))
    meat = np.zeros((2, 2), dtype=float)
    for index in range(sample_n):
        vector = design[index][:, None]
        meat += residuals[index] ** 2 * (vector @ vector.T)
    for current_lag in range(1, min(lag, sample_n - 1) + 1):
        weight = 1.0 - current_lag / (lag + 1.0)
        cross = np.zeros((2, 2), dtype=float)
        for index in range(current_lag, sample_n):
            current = design[index][:, None]
            previous = design[index - current_lag][:, None]
            cross += residuals[index] * residuals[index - current_lag] * (current @ previous.T)
        meat += weight * (cross + cross.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    covariance *= sample_n / (sample_n - 2)
    return {
        "status": "MODEL_ESTIMATED_MECHANISM_ONLY",
        "sample_n": sample_n,
        "minimum_sample_n": minimum_observations,
        "alpha": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "covariance": covariance.tolist(),
        "hac_lag": lag,
    }


def select_prior_mature_model_rows(
    observations: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    cutoff_date: date,
) -> tuple[list[float], list[float], list[str]]:
    observation_by_date = {
        str(row.get("trade_date")): row
        for row in observations
        if row.get("observation_state") == "ELIGIBLE_RAW_FORWARD_OBSERVATION_PENDING_TARGET"
    }
    x: list[float] = []
    y: list[float] = []
    included: list[str] = []
    for target in sorted(targets, key=lambda row: str(row.get("signal_trade_date", ""))):
        if target.get("target_state") != "MATURE_ELIGIBLE":
            continue
        exit_date = _parse_date(target.get("exit_trade_date"), "exit_trade_date")
        if exit_date >= cutoff_date:
            continue
        signal_date = str(target.get("signal_trade_date"))
        observation = observation_by_date.get(signal_date)
        if observation is None:
            continue
        x_value = float(observation["x_log_return"])
        y_value = float(target["y_mid_log_return"])
        if math.isfinite(x_value) and math.isfinite(y_value):
            x.append(x_value)
            y.append(y_value)
            included.append(signal_date)
    return x, y, included


def conditional_mean_interval(model: Mapping[str, Any], x_value: float, config: Mapping[str, Any]) -> dict[str, float]:
    if model.get("status") != "MODEL_ESTIMATED_MECHANISM_ONLY":
        raise ProtocolError("模型尚未估计")
    vector = np.asarray([1.0, float(x_value)], dtype=float)
    covariance = np.asarray(model["covariance"], dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ProtocolError("完整参数协方差必须为有限2x2矩阵")
    theta = np.asarray([model["alpha"], model["beta"]], dtype=float)
    mu = float(vector @ theta)
    variance = float(vector @ covariance @ vector.T)
    if variance < -1e-15:
        raise ProtocolError("条件均值方差为负")
    standard_error = math.sqrt(max(variance, 0.0))
    critical = float(config["model"]["conditional_interval_critical_value"])
    return {
        "mu_hat": mu,
        "mu_se": standard_error,
        "lcb_975": mu - critical * standard_error,
        "ucb_975": mu + critical * standard_error,
    }


def classify_signal(x_value: float, interval: Mapping[str, float], config: Mapping[str, Any]) -> dict[str, str]:
    research = float(config["signal_gates"]["research_tier_bps"]) / 10000.0
    execution = float(config["signal_gates"]["execution_tier_bps"]) / 10000.0
    lcb = float(interval["lcb_975"])
    ucb = float(interval["ucb_975"])
    research_state = "NO_RESEARCH_SIGNAL"
    execution_state = "NO_EXECUTION_SIGNAL"
    if x_value > 0 and lcb >= research:
        research_state = "RESEARCH_BUY"
    elif x_value < 0 and ucb <= -research:
        research_state = "RESEARCH_SELL"
    if x_value > 0 and lcb >= execution:
        execution_state = "EXECUTION_BUY_REQUIRES_SEPARATE_AUTHORIZATION"
    elif x_value < 0 and ucb <= -execution:
        execution_state = "EXECUTION_SELL_REQUIRES_SEPARATE_AUTHORIZATION"
    return {"research_tier_state": research_state, "execution_tier_state": execution_state}


def write_model_vintage(root: Path, config: Mapping[str, Any], model_month: str, model: Mapping[str, Any]) -> Path:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", model_month):
        raise ProtocolError("模型月份必须为YYYY-MM")
    directory = resolve_configured_path(root, config, config["artifacts"]["model_vintage_directory"])
    path = directory / f"model_vintage_{model_month}.json"
    if path.exists():
        raise ProtocolError("月度模型vintage已存在，禁止覆盖")
    payload = {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_MODEL_VINTAGE_V1",
        "project_id": PROJECT_ID,
        "model_month": model_month,
        "created_at": datetime.now(tz=SHANGHAI).isoformat(),
        **dict(model),
    }
    atomic_write_json(path, payload)
    return path


def intended_capital_return(fill_fraction: float, actual_net_return: float | None) -> float:
    if not math.isfinite(fill_fraction) or not 0.0 <= fill_fraction <= 1.0:
        raise ProtocolError("fill_fraction必须位于[0,1]")
    if fill_fraction == 0:
        return 0.0
    if actual_net_return is None or not math.isfinite(actual_net_return):
        raise ProtocolError("发生成交时必须提供有限真实净收益")
    return fill_fraction * actual_net_return


def select_t1_sell_lot(lots: Sequence[Mapping[str, Any]], trade_date: date) -> str:
    eligible: list[tuple[date, str]] = []
    for lot in lots:
        shares = int(lot.get("available_shares", 0))
        acquired = _parse_date(lot.get("acquired_date"), "lot.acquired_date")
        lot_id = str(lot.get("lot_id", "")).strip()
        if lot_id and shares > 0 and acquired < trade_date:
            eligible.append((acquired, lot_id))
    if not eligible:
        raise ProtocolError("NO_EXECUTABLE_CAPACITY：没有满足T+1的可卖库存批次")
    eligible.sort(key=lambda item: (item[0], item[1]))
    return eligible[0][1]


def assert_actual_fill_recording_authorized(config: Mapping[str, Any]) -> None:
    if not config["boundaries"].get("broker_connection_enabled") or not config["boundaries"].get("live_trading_enabled"):
        raise ProtocolError("NO_VIEW_NO_LIVE_AUTHORIZATION：当前版本禁止记录或假定真实成交")


def render_g0_status(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    evidence_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    evidence = load_json_object(evidence_path)
    g0 = evaluate_g0(root, config, manifest, evidence)
    observation_rows = _read_jsonl(resolve_configured_path(root, config, config["artifacts"]["observation_ledger"]))
    target_rows = _read_jsonl(resolve_configured_path(root, config, config["artifacts"]["target_ledger"]))
    status = {
        "schema_version": "510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_G0_STATUS_V1",
        "project_id": PROJECT_ID,
        "updated_at": datetime.now(tz=SHANGHAI).isoformat(),
        "protocol_version": config["protocol"]["version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "g0_evidence_path": relative_path(root, evidence_path),
        "g0_evidence_sha256": sha256_file(evidence_path),
        **g0,
        "authoritative_forward_start": None if not g0["all_pass"] else config["protocol"]["forward_start_rule"],
        "ledger_counts": {
            "observations": len(observation_rows),
            "eligible_observations": sum(row.get("observation_state", "").startswith("ELIGIBLE") for row in observation_rows),
            "no_view_observations": sum(row.get("observation_state") == "NO_VIEW" for row in observation_rows),
            "mature_targets": sum(row.get("target_state") == "MATURE_ELIGIBLE" for row in target_rows),
        },
        "model_trained": False,
        "probability_or_signal_generated": False,
        "paper_or_shadow_position": "NOT_AUTHORIZED",
        "current_position_target": "UNSET",
        "current_validated_high_sharpe_strategy": "NONE",
        "position_impact": 0,
        "broker_connection_authorized": False,
        "live_trading_authorized": False,
        "return_evaluation": "NOT_ALLOWED_BEFORE_SEQUENTIAL_GATES",
    }
    status_path = resolve_configured_path(root, config, config["artifacts"]["g0_status_json"])
    markdown_path = resolve_configured_path(root, config, config["artifacts"]["g0_status_markdown"])
    atomic_write_json(status_path, status)
    lines = [
        "# 510300 盘后过时收盘价捕获 V2：G0 状态",
        "",
        f"- G0：`{status['g0_state']}`",
        f"- 阻塞项：`{', '.join(status['blockers']) if status['blockers'] else 'NONE'}`",
        f"- 权威前向起点：`{status['authoritative_forward_start']}`",
        f"- 观察/NO_VIEW/成熟目标：{status['ledger_counts']['observations']}/{status['ledger_counts']['no_view_observations']}/{status['ledger_counts']['mature_targets']}",
        f"- 模型训练：`{status['model_trained']}`",
        f"- 当前目标仓位：`{status['current_position_target']}`",
        f"- 仓位影响：`{status['position_impact']}`",
        f"- 实盘授权：`{status['live_trading_authorized']}`",
        "",
        "## 顺序检查",
        "",
        "| 检查 | 状态 | 证据摘要 |",
        "|---|---|---|",
    ]
    for row in status["checks"]:
        evidence_text = str(row.get("evidence", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['check_id']}` | `{row['status']}` | {evidence_text} |")
    lines.extend(
        [
            "",
            "G0 未全部通过时，禁止权威信号采集、模型估计、收益评价、仓位映射、券商连接和订单生成。",
            "",
        ]
    )
    atomic_write_text(markdown_path, "\n".join(lines))
    return status
