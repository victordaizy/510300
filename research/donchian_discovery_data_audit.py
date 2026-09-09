"""510300简化唐奇安发现协议的数据闸门；本模块不计算任何策略收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "config" / "donchian_discovery_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY_FILE) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("唐奇安协议配置必须是YAML对象")
    validate_registry(payload)
    return payload


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    protocol = registry["protocol"]
    governance = registry["governance"]
    candidate = registry["candidate"]
    if protocol["project_id"] != "510300_DONCHIAN_DISCOVERY_V1":
        raise ValueError("项目ID不符合冻结协议")
    if protocol["state"] != "DISCOVERY_ONLY":
        raise ValueError("协议只能处于DISCOVERY_ONLY")
    if protocol["true_forward_start"] is not None:
        raise ValueError("数据和实现未冻结前true_forward_start必须为null")
    if int(protocol["candidate_budget"]) != 1:
        raise ValueError("候选预算必须严格等于1")
    if candidate["id"] != "T0_DONCHIAN_20_10":
        raise ValueError("唯一候选ID发生变化")
    if int(candidate["entry_lookback_trading_days"]) != 20:
        raise ValueError("入场窗口必须冻结为20日")
    if int(candidate["exit_lookback_trading_days"]) != 10:
        raise ValueError("退出窗口必须冻结为10日")
    if candidate["parameter_search_allowed"] or candidate["parameter_neighbors_allowed"]:
        raise ValueError("冻结协议禁止参数搜索和相邻窗口")
    disabled = (
        "article_macd_candidate_enabled",
        "martin_candidate_enabled",
        "divergence_candidate_enabled",
        "volatility_regime_switch_enabled",
        "human_segmentation_enabled",
        "covered_call_enabled",
        "options_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    )
    unexpectedly_enabled = [name for name in disabled if governance[name] is not False]
    if unexpectedly_enabled:
        raise ValueError(f"治理开关必须关闭：{unexpectedly_enabled}")
    if not protocol["return_calculation_requires_all_data_gates_pass"]:
        raise ValueError("收益计算必须依赖全部数据闸门通过")
    if not protocol["return_implementation_freeze_required"]:
        raise ValueError("收益实现必须另行冻结")
    return {
        "project_id": protocol["project_id"],
        "candidate_count": 1,
        "candidate_ids": [candidate["id"]],
        "true_forward_start": None,
        "return_calculation_requires_all_data_gates_pass": True,
    }


def _resolved(root: Path, relative: str) -> Path:
    return root / Path(relative)


def _base_file_evidence(path: Path, expected_hash: str | None) -> dict[str, Any]:
    exists = path.exists()
    actual_hash = sha256(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "sha256_matches": exists
        and expected_hash is not None
        and actual_hash == expected_hash,
        "snapshot_hash_frozen": expected_hash is not None,
    }


def _date_text(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def audit_market(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = _resolved(root, contract["file"])
    evidence = _base_file_evidence(path, contract["frozen_snapshot_sha256"])
    if not path.exists():
        return {
            "status": "BLOCKED_MISSING_MARKET_DATA",
            "return_test_allowed": False,
            "file": evidence,
        }
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - 依赖引擎错误由证据保留
        return {
            "status": "BLOCKED_UNREADABLE_MARKET_DATA",
            "return_test_allowed": False,
            "file": evidence,
            "error": f"{type(exc).__name__}: {exc}",
        }
    required = list(contract["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "status": "BLOCKED_MARKET_DATA_CONTRACT",
            "return_test_allowed": False,
            "file": evidence,
            "missing_columns": missing,
        }
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    prices = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    duplicate_dates = int(dates.duplicated().sum())
    invalid_dates = int(dates.isna().sum())
    missing_prices = int(prices.isna().sum().sum())
    nonpositive_prices = int(prices.le(0).sum().sum())
    invalid_ohlc = int(
        (
            prices["high"].lt(prices[["open", "close"]].max(axis=1))
            | prices["low"].gt(prices[["open", "close"]].min(axis=1))
            | prices["high"].lt(prices["low"])
        ).sum()
    )
    symbol_values = sorted(frame["symbol"].dropna().astype(str).unique().tolist())
    first_date = dates.min() if invalid_dates < len(dates) else None
    last_date = dates.max() if invalid_dates < len(dates) else None
    checks = {
        "snapshot_hash": evidence["sha256_matches"],
        "minimum_rows": len(frame) >= int(contract["minimum_rows"]),
        "first_date": _date_text(first_date) == contract["required_first_date"],
        "last_date": _date_text(last_date) == contract["required_last_date"],
        "no_duplicate_dates": duplicate_dates == 0,
        "valid_dates": invalid_dates == 0,
        "complete_prices": missing_prices == 0,
        "positive_prices": nonpositive_prices == 0,
        "valid_ohlc": invalid_ohlc == 0,
        "symbol": symbol_values == [contract["required_symbol"]],
        "source_present": frame["source"].notna().all(),
        "retrieved_at_present": frame["retrieved_at"].notna().all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "BLOCKED_MARKET_DATA_CONTRACT",
        "return_test_allowed": passed,
        "file": evidence,
        "rows": int(len(frame)),
        "first_date": _date_text(first_date),
        "last_date": _date_text(last_date),
        "duplicate_dates": duplicate_dates,
        "invalid_dates": invalid_dates,
        "missing_prices": missing_prices,
        "nonpositive_prices": nonpositive_prices,
        "invalid_ohlc_rows": invalid_ohlc,
        "symbols": symbol_values,
        "checks": checks,
    }


def audit_benchmark(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = _resolved(root, contract["file"])
    evidence = _base_file_evidence(path, contract["frozen_snapshot_sha256"])
    if not path.exists():
        return {
            "status": "BLOCKED_MISSING_BENCHMARK_DATA",
            "return_test_allowed": False,
            "file": evidence,
        }
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover
        return {
            "status": "BLOCKED_UNREADABLE_BENCHMARK_DATA",
            "return_test_allowed": False,
            "file": evidence,
            "error": f"{type(exc).__name__}: {exc}",
        }
    required = list(contract["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "status": "BLOCKED_BENCHMARK_DATA_CONTRACT",
            "return_test_allowed": False,
            "file": evidence,
            "missing_columns": missing,
        }
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    close = pd.to_numeric(frame["close"], errors="coerce")
    symbols = sorted(frame["symbol"].dropna().astype(str).unique().tolist())
    first_date = dates.min() if dates.notna().any() else None
    last_date = dates.max() if dates.notna().any() else None
    checks = {
        "snapshot_hash": evidence["sha256_matches"],
        "minimum_rows": len(frame) >= int(contract["minimum_rows"]),
        "first_date": _date_text(first_date) == contract["required_first_date"],
        "last_date": _date_text(last_date) == contract["required_last_date"],
        "no_duplicate_dates": int(dates.duplicated().sum()) == 0,
        "valid_dates": dates.notna().all(),
        "positive_close": close.notna().all() and close.gt(0).all(),
        "symbol": symbols == [contract["required_symbol"]],
        "retrieved_at_present": frame["retrieved_at"].notna().all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "BLOCKED_BENCHMARK_DATA_CONTRACT",
        "return_test_allowed": passed,
        "file": evidence,
        "rows": int(len(frame)),
        "first_date": _date_text(first_date),
        "last_date": _date_text(last_date),
        "symbols": symbols,
        "checks": checks,
    }


def _load_attestation(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "覆盖证明文件不存在"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"覆盖证明无法读取：{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "覆盖证明必须是JSON对象"
    return payload, None


def audit_distributions(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = _resolved(root, contract["file"])
    evidence = _base_file_evidence(path, contract["frozen_snapshot_sha256"])
    attestation_path = _resolved(root, contract["coverage_attestation_file"])
    if not path.exists():
        return {
            "status": "BLOCKED_MISSING_DISTRIBUTION_HISTORY",
            "return_test_allowed": False,
            "file": evidence,
            "coverage_attestation_file": str(attestation_path),
        }
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return {
            "status": "BLOCKED_UNREADABLE_DISTRIBUTION_HISTORY",
            "return_test_allowed": False,
            "file": evidence,
            "error": f"{type(exc).__name__}: {exc}",
        }
    required = list(contract["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "status": "BLOCKED_DISTRIBUTION_DATA_CONTRACT",
            "return_test_allowed": False,
            "file": evidence,
            "missing_columns": missing,
        }
    ex_dates = pd.to_datetime(frame["ex_date"], errors="coerce").dt.normalize()
    amounts = pd.to_numeric(frame["cash_dividend_per_share"], errors="coerce")
    sources = frame["source"].fillna("").astype(str)
    known_event_checks: list[dict[str, Any]] = []
    for event in contract["required_known_events"]:
        event_date = pd.Timestamp(event["ex_date"])
        expected_amount = float(event["cash_dividend_per_share"])
        matching = amounts[ex_dates.eq(event_date)]
        amount_matches = bool(
            matching.notna().any()
            and (matching.sub(expected_amount).abs() <= 1e-12).any()
        )
        known_event_checks.append(
            {
                "ex_date": event["ex_date"],
                "expected_cash_dividend_per_share": expected_amount,
                "present_and_amount_matches": amount_matches,
                "official_source": event["official_source"],
            }
        )
    attestation, attestation_error = _load_attestation(attestation_path)
    if attestation is None:
        attestation_checks = {
            "exists_and_readable": False,
            "distribution_file_sha256": False,
            "complete_history_confirmed": False,
            "coverage_start": False,
            "coverage_end": False,
            "official_sources_present": False,
        }
    else:
        source_list = attestation.get("official_sources", [])
        attestation_checks = {
            "exists_and_readable": True,
            "distribution_file_sha256": attestation.get("distribution_file_sha256")
            == evidence["actual_sha256"],
            "complete_history_confirmed": attestation.get("complete_history_confirmed")
            is True,
            "coverage_start": str(attestation.get("coverage_start"))
            <= contract["coverage_start_required"],
            "coverage_end": str(attestation.get("coverage_end"))
            >= contract["coverage_end_required"],
            "official_sources_present": isinstance(source_list, list)
            and len(source_list) > 0
            and all(str(source).startswith("https://") for source in source_list),
        }
    checks = {
        "snapshot_hash": evidence["sha256_matches"]
        if evidence["snapshot_hash_frozen"]
        else attestation_checks["distribution_file_sha256"],
        "valid_ex_dates": ex_dates.notna().all(),
        "unique_ex_dates": int(ex_dates.duplicated().sum()) == 0,
        "positive_amounts": amounts.notna().all() and amounts.gt(0).all(),
        "official_row_sources": sources.str.startswith("https://").all(),
        "known_events": all(
            item["present_and_amount_matches"] for item in known_event_checks
        ),
        "coverage_attestation": all(attestation_checks.values()),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "BLOCKED_INCOMPLETE_DISTRIBUTION_HISTORY",
        "return_test_allowed": passed,
        "file": evidence,
        "rows": int(len(frame)),
        "first_ex_date": _date_text(ex_dates.min()) if ex_dates.notna().any() else None,
        "last_ex_date": _date_text(ex_dates.max()) if ex_dates.notna().any() else None,
        "known_event_checks": known_event_checks,
        "coverage_attestation_file": str(attestation_path),
        "coverage_attestation_error": attestation_error,
        "coverage_attestation_checks": attestation_checks,
        "checks": checks,
    }


def audit_all(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    validation = validate_registry(registry)
    contracts = registry["data_contracts"]
    branches = {
        "market": audit_market(root, contracts["market"]),
        "benchmark": audit_benchmark(root, contracts["benchmark"]),
        "distributions": audit_distributions(root, contracts["distributions"]),
    }
    failure_categories = [
        evidence["status"]
        for evidence in branches.values()
        if evidence["status"] != "PASS"
    ]
    passed = not failure_categories
    return {
        "project_id": validation["project_id"],
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "data_cutoff": registry["protocol"]["historical_contamination_cutoff"],
        "status": "PASS" if passed else "NO_VIEW",
        "failure_categories": failure_categories,
        "return_calculation_allowed": passed,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "branches": branches,
    }
