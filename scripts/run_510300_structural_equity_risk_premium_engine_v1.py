"""运行 510300 结构性权益风险溢价引擎 V1。

阶段严格分离：
1. freeze：冻结协议与输入哈希，不读取收益值；
2. collect：采集并规范财务报表版本，不读取收益值；
3. build-states：构造 CF/DR/RC 状态并冻结结果，不读取解释性收益；
4. explain：只有状态冻结后才读取总收益，生成归因和机械周期图谱。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.structural_equity_risk_premium_engine_v1 import (  # noqa: E402
    add_cashflow_state_columns,
    add_present_value_cross_section,
    aggregate_cashflow_snapshot,
    aggregate_present_value_panel,
    average_pairwise_correlation,
    build_component_snapshot,
    build_fixed_present_value_portfolios,
    derive_balance_snapshot,
    derive_ttm_pair_snapshot,
    detect_drawdown_episodes,
    detect_sideways_windows,
    expanding_zscore,
    first_principal_component,
    prepare_statement_vintages,
    weighted_average,
)


PROGRAM_ID = "510300_STRUCTURAL_EQUITY_RISK_PREMIUM_ENGINE_V1"
CONFIG_PATH = ROOT / "config/510300_structural_equity_risk_premium_engine_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/structural_equity_risk_premium_engine_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")
ASHARE_SUFFIXES = (".SH", ".SZ")
HTTP_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class RawPage:
    payload: bytes
    payload_sha256: str
    rows: tuple[Mapping[str, Any], ...]
    page_number: int
    page_count: int
    total_rows: int
    cache_hit: bool


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("结构性引擎配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _all_registered_input_paths(config: dict[str, Any]) -> list[Path]:
    source = config["source_contract"]
    paths = [
        project_path(source["membership"]["path"]),
        project_path(source["market_cross_section"]["path"]),
        project_path(source["historical_index_weight_diagnostic"]["path"]),
        project_path(source["independent_financial_crosscheck"]["path"]),
    ]
    paths.extend(
        project_path(relative) for relative in source["protocol_history"]["paths"]
    )
    paths.extend(
        project_path(relative)
        for relative in source["market_daily_fallback"]["files"]
    )
    for contract_name in [
        "carry_forward_collection",
        "carry_forward_financial_sector_collection",
    ]:
        paths.extend(
            project_path(relative)
            for relative in source.get(contract_name, {}).get("paths", [])
        )
    for relative in source.get("carry_forward_financial_sector_raw", {}).get(
        "roots", []
    ):
        root = project_path(relative)
        if not root.is_dir():
            raise FileNotFoundError(f"金融业补源原始快照目录缺失：{root}")
        paths.extend(path for path in root.rglob("*") if path.is_file())
    paths.extend(
        project_path(relative)
        for relative in source["risk_bearing_inputs"].values()
    )
    paths.extend(
        project_path(relative)
        for relative in source["explanatory_outcomes_locked_until_state_freeze"].values()
    )
    return list(dict.fromkeys(paths))


def validate_carry_forward_financial_sector_raw(config: dict[str, Any]) -> None:
    contract = config["source_contract"].get("carry_forward_financial_sector_raw")
    if not contract:
        return
    roots = [project_path(relative) for relative in contract["roots"]]
    profile_payloads: list[Path] = []
    batch_payloads: list[Path] = []
    for root in roots:
        profile_payloads.extend(root.glob("*/profile/index.html.gz"))
        batch_payloads.extend(root.glob("*/*/batch_*.json.gz"))
    if len(profile_payloads) != int(contract["expected_profile_payloads"]):
        raise ValueError(
            "金融业补源画像快照数与继承声明不一致："
            f"{len(profile_payloads)} != {contract['expected_profile_payloads']}"
        )
    if len(batch_payloads) != int(contract["expected_batch_payloads"]):
        raise ValueError(
            "金融业补源批次快照数与继承声明不一致："
            f"{len(batch_payloads)} != {contract['expected_batch_payloads']}"
        )
    payloads = [*profile_payloads, *batch_payloads]
    for payload_path in payloads:
        metadata_path = (
            payload_path.with_name("index.metadata.json")
            if payload_path.name == "index.html.gz"
            else payload_path.with_name(
                payload_path.name.removesuffix(".json.gz") + ".metadata.json"
            )
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"金融业补源快照元数据缺失：{metadata_path}")


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"冻结清单已存在，禁止静默覆盖：{manifest_path}")
    validate_carry_forward_financial_sector_raw(config)
    registered_inputs: dict[str, Any] = {}
    outcome_relatives = set(
        config["source_contract"][
            "explanatory_outcomes_locked_until_state_freeze"
        ].values()
    )
    for path in _all_registered_input_paths(config):
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入缺失：{path}")
        relative = path.relative_to(ROOT).as_posix()
        registered_inputs[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": (
                "LOCKED_EXPLANATORY_OUTCOME_NOT_VALUE_READ"
                if relative in outcome_relatives
                else "OUTCOME_BLIND_STATE_INPUT"
            ),
        }
    config_sha256 = sha256_file(CONFIG_PATH)
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": (
            "FROZEN_AFTER_HASHED_OUTCOME_BLIND_RAW_SUPPLEMENT_BEFORE_"
            "NORMALIZATION_STATE_BUILD_AND_RETURN_READ"
            if config["source_contract"].get("carry_forward_financial_sector_raw")
            else (
                "FROZEN_BEFORE_FINANCIAL_SECTOR_SUPPLEMENT_STATE_BUILD_AND_"
                "RETURN_READ_WITH_HASHED_CARRY_FORWARD_COLLECTION"
                if config["source_contract"].get("carry_forward_collection")
                else "FROZEN_BEFORE_STATEMENT_ACQUISITION_STATE_BUILD_AND_RETURN_READ"
            )
        ),
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha256,
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered_inputs,
        "future_return_values_read": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "old_rejected_protocols_rescued": False,
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("协议尚未冻结，必须先运行 --phase freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("program_id") != PROGRAM_ID:
        raise ValueError("协议清单 PROGRAM_ID 不匹配")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("冻结后配置文件发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("冻结后配置语义发生漂移")
    for relative, expected_hash in manifest.get("implementation_files", {}).items():
        if sha256_file(project_path(relative)) != expected_hash:
            raise ValueError(f"冻结后实现文件发生漂移：{relative}")
    for relative, item in manifest.get("registered_inputs", {}).items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"冻结输入发生漂移或缺失：{relative}")
    return manifest


def quarter_ends(start: str, end: str) -> list[pd.Timestamp]:
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q-DEC")
    return [period.end_time.normalize() for period in periods]


def raw_paths(
    raw_root: Path,
    statement: str,
    report_end: pd.Timestamp,
    page: int,
) -> tuple[Path, Path]:
    folder = raw_root / statement / f"{report_end:%Y%m%d}"
    stem = f"page_{page:04d}"
    return folder / f"{stem}.json.gz", folder / f"{stem}.metadata.json"


def request_params(
    statement_config: dict[str, Any],
    report_end: pd.Timestamp,
    page: int,
    *,
    page_size: int,
) -> dict[str, str]:
    return {
        "type": statement_config["report_type"],
        "sty": ",".join(statement_config["fields"]),
        "filter": f"(REPORT_DATE='{report_end:%Y-%m-%d}')",
        "p": str(page),
        "ps": str(page_size),
        "sr": "1",
        "st": "SECURITY_CODE",
        "source": "HSF10",
        "client": "PC",
    }


def parse_payload(
    payload: bytes,
    *,
    page: int,
    cache_hit: bool,
) -> RawPage:
    document = json.loads(payload)
    if document.get("success") is not True or int(document.get("code", -1)) != 0:
        raise ValueError(
            f"财务接口失败：{document.get('code')} {document.get('message')}"
        )
    result = document.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("data"), list):
        raise ValueError("财务接口缺少 result.data")
    return RawPage(
        payload=payload,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        rows=tuple(result["data"]),
        page_number=page,
        page_count=int(result.get("pages", 0)),
        total_rows=int(result.get("count", 0)),
        cache_hit=cache_hit,
    )


def load_cached_page(
    raw_root: Path,
    statement: str,
    report_end: pd.Timestamp,
    page: int,
) -> RawPage | None:
    payload_path, metadata_path = raw_paths(raw_root, statement, report_end, page)
    if not payload_path.exists() and not metadata_path.exists():
        return None
    if not payload_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"原始财务缓存不完整：{payload_path}")
    payload = gzip.decompress(payload_path.read_bytes())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata.get("payload_sha256"):
        raise ValueError(f"原始财务缓存哈希不匹配：{payload_path}")
    return parse_payload(payload, page=page, cache_hit=True)


def fetch_page(
    session: requests.Session,
    *,
    endpoint: str,
    raw_root: Path,
    statement: str,
    statement_config: dict[str, Any],
    report_end: pd.Timestamp,
    page: int,
    page_size: int,
    timeout_seconds: int,
    attempts: int,
) -> RawPage:
    cached = load_cached_page(raw_root, statement, report_end, page)
    if cached is not None:
        return cached
    params = request_params(
        statement_config, report_end, page, page_size=page_size
    )
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(endpoint, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            parsed = parse_payload(response.content, page=page, cache_hit=False)
            payload_path, metadata_path = raw_paths(
                raw_root, statement, report_end, page
            )
            atomic_bytes(
                payload_path, gzip.compress(response.content, compresslevel=9)
            )
            atomic_json(
                metadata_path,
                {
                    "program_id": PROGRAM_ID,
                    "statement": statement,
                    "report_type": statement_config["report_type"],
                    "endpoint": endpoint,
                    "request_url": endpoint + "?" + urlencode(params),
                    "report_end": f"{report_end:%Y-%m-%d}",
                    "page_number": page,
                    "page_count": parsed.page_count,
                    "total_rows": parsed.total_rows,
                    "row_count": len(parsed.rows),
                    "payload_sha256": parsed.payload_sha256,
                    "collected_at": now_iso(),
                    "transport_tls_verified": True,
                },
            )
            return parsed
        except Exception as exception:  # noqa: BLE001
            error = exception
            if attempt < attempts:
                time.sleep(
                    min(8.0, 0.8 * (2 ** (attempt - 1)))
                    + random.random() * 0.4
                )
    raise RuntimeError(
        f"{statement} {report_end.date()} 第 {page} 页采集失败"
    ) from error


def fetch_statement_period(
    *,
    endpoint: str,
    raw_root: Path,
    statement: str,
    statement_config: dict[str, Any],
    report_end: pd.Timestamp,
    page_size: int,
    timeout_seconds: int,
    attempts: int,
    ever_members: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/139 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        }
    )
    first = fetch_page(
        session,
        endpoint=endpoint,
        raw_root=raw_root,
        statement=statement,
        statement_config=statement_config,
        report_end=report_end,
        page=1,
        page_size=page_size,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
    )
    pages = [first]
    for page in range(2, first.page_count + 1):
        pages.append(
            fetch_page(
                session,
                endpoint=endpoint,
                raw_root=raw_root,
                statement=statement,
                statement_config=statement_config,
                report_end=report_end,
                page=page,
                page_size=page_size,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
            )
        )
    all_row_count = sum(len(page.rows) for page in pages)
    if all_row_count != first.total_rows:
        raise ValueError(
            f"{statement} {report_end.date()} 行数不守恒："
            f"{all_row_count} != {first.total_rows}"
        )
    rows: list[dict[str, Any]] = []
    for page in pages:
        for raw in page.rows:
            stock_code = str(raw.get("SECUCODE") or "")
            if stock_code not in ever_members:
                continue
            row = dict(raw)
            row["source_page"] = page.page_number
            row["source_sha256"] = page.payload_sha256
            rows.append(row)
    return rows, {
        "statement": statement,
        "report_end": f"{report_end:%Y-%m-%d}",
        "pages": first.page_count,
        "source_rows": all_row_count,
        "ever_member_rows": len(rows),
        "cache_pages": int(sum(page.cache_hit for page in pages)),
        "downloaded_pages": int(sum(not page.cache_hit for page in pages)),
    }


STATEMENT_VALUE_MAP = {
    "income": {
        "TOTAL_OPERATE_INCOME": "total_operating_revenue",
        "OPERATE_PROFIT": "operating_profit",
        "PARENT_NETPROFIT": "parent_net_profit",
    },
    "cashflow": {"NETCASH_OPERATE": "operating_cashflow"},
    "balance": {
        "TOTAL_PARENT_EQUITY": "total_parent_equity",
        "TOTAL_ASSETS": "total_assets",
        "SHARE_CAPITAL": "share_capital",
    },
}


def normalize_statement(
    rows: list[dict[str, Any]], statement: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = pd.DataFrame(rows)
    value_map = STATEMENT_VALUE_MAP[statement]
    required = {
        "SECUCODE",
        "SECURITY_CODE",
        "SECURITY_TYPE_CODE",
        "ORG_CODE",
        "REPORT_DATE",
        "REPORT_TYPE",
        "REPORT_DATE_NAME",
        "NOTICE_DATE",
        "UPDATE_DATE",
        "CURRENCY",
        "source_page",
        "source_sha256",
        *value_map,
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{statement} 原始源缺字段：{missing}")
    selected = [
        "SECUCODE",
        "SECURITY_CODE",
        "SECURITY_TYPE_CODE",
        "ORG_CODE",
        "REPORT_DATE",
        "REPORT_TYPE",
        "REPORT_DATE_NAME",
        "NOTICE_DATE",
        "UPDATE_DATE",
        "CURRENCY",
        "source_page",
        "source_sha256",
        *value_map,
    ]
    frame = source[selected].rename(
        columns={
            "SECUCODE": "stock_code",
            "SECURITY_CODE": "raw_security_code",
            "SECURITY_TYPE_CODE": "security_type_code",
            "ORG_CODE": "org_code",
            "REPORT_DATE": "report_end",
            "REPORT_TYPE": "report_type",
            "REPORT_DATE_NAME": "report_date_name",
            "NOTICE_DATE": "notice_date",
            "UPDATE_DATE": "update_date",
            "CURRENCY": "currency",
            **value_map,
        }
    )
    for column in ["report_end", "notice_date", "update_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    for column in value_map.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["stock_code"].astype(str).str.endswith(ASHARE_SUFFIXES)
        & frame["report_end"].notna()
        & frame["notice_date"].notna()
    ].copy()
    frame["available_at"] = frame["notice_date"]
    frame["revision_delay_days"] = (
        frame["update_date"] - frame["notice_date"]
    ).dt.days
    frame["revision_risk_status"] = np.select(
        [
            frame["update_date"].isna(),
            frame["update_date"].gt(frame["notice_date"]),
        ],
        [
            "NO_UPDATE_DATE_RECORDED",
            "CURRENT_VALUE_MAY_INCLUDE_POST_NOTICE_REVISION",
        ],
        default="NO_LATER_UPDATE_RECORDED",
    )
    frame["statement"] = statement
    frame["source_provider"] = (
        "eastmoney.HSF10." + statement.upper() + ".SECONDARY_AGGREGATOR"
    )
    exact_key = [
        "stock_code",
        "report_end",
        "available_at",
        *value_map.values(),
    ]
    exact_duplicate = frame.duplicated(exact_key, keep="first")
    exact_removed = int(exact_duplicate.sum())
    frame = frame.loc[~exact_duplicate].copy()
    version_key = ["stock_code", "report_end", "available_at"]
    conflicts = frame.loc[frame.duplicated(version_key, keep=False)].copy()
    if not conflicts.empty:
        conflicting_keys = conflicts[version_key].drop_duplicates()
        frame = frame.merge(
            conflicting_keys.assign(_conflict=True),
            on=version_key,
            how="left",
            validate="many_to_one",
        )
        frame = frame.loc[frame["_conflict"].ne(True)].drop(columns="_conflict")
    if frame.duplicated(version_key).any():
        raise AssertionError(f"{statement} 冲突排除后版本键仍不唯一")
    frame = frame.sort_values(version_key, kind="mergesort").reset_index(drop=True)
    conflicts = conflicts.sort_values(version_key, kind="mergesort").reset_index(
        drop=True
    )
    stats = {
        "source_member_rows": len(source),
        "normalized_vintage_rows": len(frame),
        "stock_codes": int(frame["stock_code"].nunique()),
        "report_ends": int(frame["report_end"].nunique()),
        "report_end_min": str(frame["report_end"].min().date()),
        "report_end_max": str(frame["report_end"].max().date()),
        "available_at_min": str(frame["available_at"].min().date()),
        "available_at_max": str(frame["available_at"].max().date()),
        "revision_delayed_rows": int(frame["revision_delay_days"].gt(0).sum()),
        "revision_risk_status_counts": frame["revision_risk_status"]
        .value_counts()
        .to_dict(),
        "exact_duplicate_rows_removed": exact_removed,
        "conflicting_rows_excluded": len(conflicts),
        "conflicting_version_keys_excluded": int(
            conflicts[version_key].drop_duplicates().shape[0]
        ),
        "value_nonmissing": {
            column: int(frame[column].notna().sum())
            for column in value_map.values()
        },
    }
    return frame, conflicts, stats


def collect_statements(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    receipt_path = project_path(artifacts["collection_receipt"])
    output_paths = {
        "income": project_path(artifacts["statement_income"]),
        "cashflow": project_path(artifacts["statement_cashflow"]),
        "balance": project_path(artifacts["statement_balance"]),
    }
    conflict_path = project_path(artifacts["statement_conflicts"])
    if receipt_path.is_file() and all(path.is_file() for path in output_paths.values()):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        carry_forward = config["source_contract"].get("carry_forward_collection")
        if carry_forward and receipt.get("protocol_manifest_payload_sha256") != (
            carry_forward["from_manifest_payload_sha256"]
        ):
            raise ValueError("既有财务采集收据不属于冻结声明的可继承协议")
        for statement, path in output_paths.items():
            if sha256_file(path) != receipt["artifacts"][statement]["sha256"]:
                raise ValueError(f"已存在的 {statement} 产物哈希与收据不一致")
        print("财务报表点时版本已存在且哈希通过，跳过重复采集。", flush=True)
        return receipt
    if receipt_path.exists() or any(path.exists() for path in output_paths.values()):
        raise RuntimeError("财务采集产物不完整，禁止静默覆盖；请先人工审计残留文件")
    source = config["source_contract"]
    statement_contract = source["statement_acquisition"]
    membership = pd.read_parquet(project_path(source["membership"]["path"]))
    ever_members = set(membership["symbol"].astype(str).unique())
    periods = quarter_ends(
        statement_contract["financial_collection_start"],
        statement_contract["financial_collection_end"],
    )
    raw_root = project_path(artifacts["raw_statement_root"])
    tasks = [
        (statement, statement_config, report_end)
        for statement, statement_config in statement_contract["statements"].items()
        for report_end in periods
    ]
    all_rows: dict[str, list[dict[str, Any]]] = {
        statement: [] for statement in statement_contract["statements"]
    }
    task_receipts: list[dict[str, Any]] = []
    started_at = now_iso()
    with ThreadPoolExecutor(max_workers=int(statement_contract["workers"])) as executor:
        futures = {
            executor.submit(
                fetch_statement_period,
                endpoint=statement_contract["endpoint"],
                raw_root=raw_root,
                statement=statement,
                statement_config=statement_config,
                report_end=report_end,
                page_size=int(statement_contract["page_size"]),
                timeout_seconds=int(statement_contract["timeout_seconds"]),
                attempts=int(statement_contract["attempts"]),
                ever_members=ever_members,
            ): (statement, report_end)
            for statement, statement_config, report_end in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            statement, report_end = futures[future]
            rows, task_receipt = future.result()
            all_rows[statement].extend(rows)
            task_receipts.append(task_receipt)
            print(
                f"财务采集 {completed:03d}/{len(tasks)} "
                f"{statement} {report_end:%Y-%m-%d}："
                f"成员记录 {len(rows):,} 行。",
                flush=True,
            )
    normalization: dict[str, Any] = {}
    conflict_frames: list[pd.DataFrame] = []
    artifact_receipts: dict[str, Any] = {}
    for statement, rows in all_rows.items():
        frame, conflicts, stats = normalize_statement(rows, statement)
        atomic_parquet(output_paths[statement], frame)
        normalization[statement] = stats
        if not conflicts.empty:
            conflict_frames.append(conflicts.assign(statement=statement))
        artifact_receipts[statement] = {
            "path": output_paths[statement].relative_to(ROOT).as_posix(),
            "rows": len(frame),
            "sha256": sha256_file(output_paths[statement]),
        }
    combined_conflicts = (
        pd.concat(conflict_frames, ignore_index=True, sort=False)
        if conflict_frames
        else pd.DataFrame(columns=["statement", "stock_code", "report_end"])
    )
    atomic_parquet(conflict_path, combined_conflicts)
    receipt = {
        "program_id": PROGRAM_ID,
        "status": "PASS_SECONDARY_AGGREGATOR_STATEMENT_VINTAGE_COLLECTION_COMPLETE",
        "evidence_class": "DISCOVERY_EVIDENCE_ONLY_NOT_OFFICIAL_ORIGINAL_PDF",
        "started_at": started_at,
        "finished_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "availability_clock": "NOTICE_DATE",
        "update_date_usage": "DIAGNOSTIC_ONLY_NOT_INFORMATION_AVAILABILITY",
        "revision_limitation": (
            "CURRENT_SECONDARY_AGGREGATOR_VALUE_MAY_REFLECT_LATER_REVISION"
        ),
        "ever_member_count": len(ever_members),
        "normalization": normalization,
        "task_receipts": sorted(
            task_receipts,
            key=lambda item: (item["statement"], item["report_end"]),
        ),
        "artifacts": {
            **artifact_receipts,
            "conflicts": {
                "path": conflict_path.relative_to(ROOT).as_posix(),
                "rows": len(combined_conflicts),
                "sha256": sha256_file(conflict_path),
            },
        },
        "future_return_values_read": False,
        "portfolio_evaluation_run": False,
        "position_generated": False,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def verify_collection_receipt(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["artifacts"]
    receipt_path = project_path(artifacts["collection_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("财务采集收据缺失，必须先运行 --phase collect")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    carry_forward = config["source_contract"].get("carry_forward_collection")
    if carry_forward and receipt.get("protocol_manifest_payload_sha256") != (
        carry_forward["from_manifest_payload_sha256"]
    ):
        raise ValueError("财务采集收据与协议修正声明的来源清单不一致")
    expected_paths = {
        "income": artifacts["statement_income"],
        "cashflow": artifacts["statement_cashflow"],
        "balance": artifacts["statement_balance"],
        "conflicts": artifacts["statement_conflicts"],
    }
    for name, relative in expected_paths.items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != receipt["artifacts"][name][
            "sha256"
        ]:
            raise ValueError(f"财务采集产物与收据不一致：{name}")
    return receipt


def _financial_sector_paths(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "balance": project_path(artifacts["financial_sector_statement_balance"]),
        "cashflow": project_path(artifacts["financial_sector_statement_cashflow"]),
        "conflicts": project_path(artifacts["financial_sector_statement_conflicts"]),
        "receipt": project_path(artifacts["financial_sector_collection_receipt"]),
    }


def _thread_http_session() -> requests.Session:
    session = getattr(HTTP_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/139 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Referer": (
                    "https://emweb.securities.eastmoney.com/PC_HSF10/"
                    "NewFinanceAnalysis/Index"
                ),
            }
        )
        HTTP_THREAD_LOCAL.session = session
    return session


def _cached_snapshot(
    *,
    url: str,
    params: Mapping[str, str],
    payload_path: Path,
    metadata_path: Path,
    timeout_seconds: int,
    attempts: int,
    snapshot_role: str,
) -> tuple[bytes, str, bool]:
    if payload_path.exists() or metadata_path.exists():
        if not payload_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"金融业补充原始缓存不完整：{payload_path}")
        payload = gzip.decompress(payload_path.read_bytes())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(payload).hexdigest()
        if digest != metadata.get("payload_sha256"):
            raise ValueError(f"金融业补充原始缓存哈希不匹配：{payload_path}")
        if metadata.get("snapshot_role") != snapshot_role:
            raise ValueError(f"金融业补充原始缓存角色不匹配：{payload_path}")
        if metadata.get("bytes") != len(payload):
            raise ValueError(f"金融业补充原始缓存字节数不匹配：{payload_path}")
        if metadata.get("transport_tls_verified") is not True:
            raise ValueError(f"金融业补充原始缓存未确认 TLS 校验：{payload_path}")
        return payload, digest, True
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _thread_http_session().get(
                url, params=dict(params), timeout=timeout_seconds
            )
            response.raise_for_status()
            payload = response.content
            digest = hashlib.sha256(payload).hexdigest()
            atomic_bytes(payload_path, gzip.compress(payload, compresslevel=9))
            atomic_json(
                metadata_path,
                {
                    "program_id": PROGRAM_ID,
                    "snapshot_role": snapshot_role,
                    "request_url": response.url,
                    "payload_sha256": digest,
                    "bytes": len(payload),
                    "collected_at": now_iso(),
                    "transport_tls_verified": True,
                },
            )
            return payload, digest, False
        except Exception as exception:  # noqa: BLE001
            error = exception
            if attempt < attempts:
                time.sleep(
                    min(8.0, 0.8 * (2 ** (attempt - 1)))
                    + random.random() * 0.4
                )
    raise RuntimeError(f"金融业补充快照采集失败：{snapshot_role}") from error


def _eastmoney_web_code(stock_code: str) -> str:
    raw = str(stock_code).upper()
    if raw.endswith(".SH"):
        return "SH" + raw.removesuffix(".SH")
    if raw.endswith(".SZ"):
        return "SZ" + raw.removesuffix(".SZ")
    raise ValueError(f"不能转换为东方财富网页代码：{stock_code}")


def discover_financial_company_type(
    *,
    base_url: str,
    raw_root: Path,
    stock_code: str,
    allowed_company_types: Mapping[str, str],
    timeout_seconds: int,
    attempts: int,
) -> tuple[str, str, dict[str, Any]]:
    safe_code = stock_code.replace(".", "_")
    folder = raw_root / safe_code / "profile"
    payload_path = folder / "index.html.gz"
    metadata_path = folder / "index.metadata.json"
    web_code = _eastmoney_web_code(stock_code)
    payload, digest, cache_hit = _cached_snapshot(
        url=base_url.rstrip("/") + "/Index",
        params={"code": web_code, "type": "web"},
        payload_path=payload_path,
        metadata_path=metadata_path,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
        snapshot_role=f"FINANCIAL_COMPANY_TYPE_PROFILE:{stock_code}",
    )
    html = payload.decode("utf-8", errors="replace")
    match = re.search(
        r'id=["\']hidctype["\'][^>]*value=["\']([1-4])["\']',
        html,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"未能从官方 F10 页面识别公司类型：{stock_code}")
    company_type = match.group(1)
    if company_type not in allowed_company_types:
        raise ValueError(
            f"通用报表缺失证券不是冻结的金融公司类型：{stock_code}={company_type}"
        )
    return stock_code, company_type, {
        "stock_code": stock_code,
        "web_code": web_code,
        "company_type": company_type,
        "company_type_name": allowed_company_types[company_type],
        "payload_sha256": digest,
        "cache_hit": cache_hit,
    }


def resolve_effective_company_types(
    *,
    profile_types: Mapping[str, str],
    historical_overrides: Mapping[str, str],
    target_codes: Iterable[str],
    allowed_company_types: Mapping[str, str],
) -> dict[str, str]:
    targets = set(map(str, target_codes))
    profiles = {str(code): str(value) for code, value in profile_types.items()}
    overrides = {
        str(code): str(value) for code, value in historical_overrides.items()
    }
    if set(profiles) != targets:
        raise ValueError("金融业画像类型代码集合与补源目标集合不一致")
    if unknown := set(overrides).difference(targets):
        raise ValueError(f"历史公司类型覆盖含非目标证券：{sorted(unknown)}")
    allowed = set(map(str, allowed_company_types))
    invalid = {
        code: value
        for code, value in {**profiles, **overrides}.items()
        if value not in allowed
    }
    if invalid:
        raise ValueError(f"金融业公司类型不在冻结允许集合：{invalid}")
    return {code: overrides.get(code, profiles[code]) for code in sorted(targets)}


def fetch_financial_sector_batch(
    *,
    base_url: str,
    raw_root: Path,
    stock_code: str,
    company_type: str,
    statement: str,
    tab: str,
    dates: Sequence[pd.Timestamp],
    batch_number: int,
    report_date_type: int,
    report_type: int,
    timeout_seconds: int,
    attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formatted_dates = [f"{pd.Timestamp(value):%Y-%m-%d}" for value in dates]
    safe_code = stock_code.replace(".", "_")
    folder = raw_root / safe_code / statement
    stem = f"batch_{batch_number:03d}_{formatted_dates[0]}_{formatted_dates[-1]}"
    payload_path = folder / f"{stem}.json.gz"
    metadata_path = folder / f"{stem}.metadata.json"
    params = {
        "companyType": str(company_type),
        "reportDateType": str(report_date_type),
        "reportType": str(report_type),
        "dates": ",".join(formatted_dates),
        "code": _eastmoney_web_code(stock_code),
    }
    payload, digest, cache_hit = _cached_snapshot(
        url=base_url.rstrip("/") + f"/{tab}AjaxNew",
        params=params,
        payload_path=payload_path,
        metadata_path=metadata_path,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
        snapshot_role=(
            f"FINANCIAL_SECTOR_{statement.upper()}:{stock_code}:"
            f"{formatted_dates[0]}:{formatted_dates[-1]}"
        ),
    )
    document = json.loads(payload.decode("utf-8-sig"))
    data, response_status = parse_financial_sector_document(
        document,
        stock_code=stock_code,
        statement=statement,
    )
    if len(data) > len(formatted_dates):
        raise ValueError(f"金融业补充接口返回超过请求日期数：{stock_code} {statement}")
    requested = set(formatted_dates)
    returned_dates: list[str] = []
    rows: list[dict[str, Any]] = []
    for raw in data:
        returned_stock = str(raw.get("SECUCODE") or "")
        if returned_stock != stock_code:
            raise ValueError(
                f"金融业补充接口证券漂移：{stock_code} -> {returned_stock}"
            )
        report_date = pd.to_datetime(raw.get("REPORT_DATE"), errors="coerce")
        if pd.isna(report_date):
            raise ValueError(f"金融业补充接口报告期无效：{stock_code} {statement}")
        formatted_report_date = f"{pd.Timestamp(report_date):%Y-%m-%d}"
        if formatted_report_date not in requested:
            raise ValueError(
                f"金融业补充接口返回未请求报告期：{formatted_report_date}"
            )
        returned_dates.append(formatted_report_date)
        row = dict(raw)
        row["source_page"] = batch_number
        row["source_sha256"] = digest
        rows.append(row)
    if len(returned_dates) != len(set(returned_dates)):
        raise ValueError(f"金融业补充接口报告期重复：{stock_code} {statement}")
    return rows, {
        "stock_code": stock_code,
        "company_type": company_type,
        "statement": statement,
        "batch_number": batch_number,
        "requested_dates": formatted_dates,
        "returned_dates": returned_dates,
        "returned_rows": len(rows),
        "response_status": response_status,
        "payload_sha256": digest,
        "cache_hit": cache_hit,
    }


def parse_financial_sector_document(
    document: Mapping[str, Any],
    *,
    stock_code: str,
    statement: str,
) -> tuple[list[dict[str, Any]], str]:
    """解析金融业专用接口，并严格区分合法空批次与异常响应。"""

    data = document.get("data")
    if isinstance(data, list):
        return [dict(row) for row in data], (
            "DATA_LIST_NONEMPTY" if data else "DATA_LIST_EMPTY"
        )
    serializer_types = document.get("$types")
    serializer_type = document.get("$type")
    exact_marker_keys = set(document) == {"$types", "$type"}
    system_object_marker = (
        isinstance(serializer_types, Mapping)
        and any(
            str(key).startswith("System.Object, mscorlib,")
            and str(value) == str(serializer_type)
            for key, value in serializer_types.items()
        )
    )
    if exact_marker_keys and str(serializer_type) == "1" and system_object_marker:
        return [], "SERIALIZER_EMPTY_OBJECT_NO_ROWS"
    raise ValueError(f"金融业补充接口缺少 data 列表：{stock_code} {statement}")


def collect_financial_sector_supplement(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    paths = _financial_sector_paths(config)
    output_names = ["balance", "cashflow", "conflicts"]
    if paths["receipt"].is_file() and all(paths[name].is_file() for name in output_names):
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        if receipt.get("protocol_manifest_payload_sha256") != manifest[
            "manifest_payload_sha256"
        ]:
            raise ValueError("金融业补充收据不属于当前冻结清单")
        for name in output_names:
            if sha256_file(paths[name]) != receipt["artifacts"][name]["sha256"]:
                raise ValueError(f"金融业补充既有产物哈希不一致：{name}")
        print("金融业专用报表补充已存在且哈希通过，跳过重复采集。", flush=True)
        return receipt
    if paths["receipt"].exists() or any(paths[name].exists() for name in output_names):
        raise RuntimeError("金融业补充产物不完整，禁止静默覆盖；请先审计残留文件")
    source = config["source_contract"]
    contract = source["financial_sector_statement_supplement"]
    general_balance = pd.read_parquet(
        project_path(config["artifacts"]["statement_balance"]),
        columns=["stock_code"],
    )
    membership = pd.read_parquet(
        project_path(source["membership"]["path"]), columns=["symbol"]
    )
    target_codes = sorted(
        set(membership["symbol"].astype(str).unique())
        - set(general_balance["stock_code"].astype(str).unique())
    )
    if not target_codes:
        raise ValueError("没有识别到需要金融业专用报表补充的证券")
    raw_root = project_path(config["artifacts"]["raw_financial_sector_statement_root"])
    override_raw_root = project_path(
        config["artifacts"]["raw_financial_sector_statement_override_root"]
    )
    allowed_company_types = {
        str(key): str(value)
        for key, value in contract["allowed_company_types"].items()
    }
    timeout_seconds = int(contract["timeout_seconds"])
    attempts = int(contract["attempts"])
    workers = int(contract["workers"])
    profiles: dict[str, str] = {}
    profile_receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                discover_financial_company_type,
                base_url=contract["base_url"],
                raw_root=raw_root,
                stock_code=stock_code,
                allowed_company_types=allowed_company_types,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
            ): stock_code
            for stock_code in target_codes
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            stock_code, company_type, profile_receipt = future.result()
            profiles[stock_code] = company_type
            profile_receipts.append(profile_receipt)
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"金融业类型识别 {completed:03d}/{len(futures)}。",
                    flush=True,
                )
    historical_overrides = {
        str(code): str(company_type)
        for code, company_type in contract.get(
            "historical_company_type_overrides", {}
        ).items()
    }
    effective_company_types = resolve_effective_company_types(
        profile_types=profiles,
        historical_overrides=historical_overrides,
        target_codes=target_codes,
        allowed_company_types=allowed_company_types,
    )
    periods = list(
        reversed(
            quarter_ends(
                source["statement_acquisition"]["financial_collection_start"],
                source["statement_acquisition"]["financial_collection_end"],
            )
        )
    )
    batch_size = int(contract["maximum_dates_per_request"])
    date_batches = [
        periods[index : index + batch_size]
        for index in range(0, len(periods), batch_size)
    ]
    tasks = [
        (
            stock_code,
            effective_company_types[stock_code],
            statement,
            tab,
            batch_number,
            dates,
        )
        for stock_code in target_codes
        for statement, tab in contract["tables"].items()
        for batch_number, dates in enumerate(date_batches, start=1)
    ]
    rows_by_statement: dict[str, list[dict[str, Any]]] = {
        statement: [] for statement in contract["tables"]
    }
    batch_receipts: list[dict[str, Any]] = []
    started_at = now_iso()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_financial_sector_batch,
                base_url=contract["base_url"],
                raw_root=(
                    override_raw_root
                    if stock_code in historical_overrides
                    else raw_root
                ),
                stock_code=stock_code,
                company_type=company_type,
                statement=statement,
                tab=tab,
                dates=dates,
                batch_number=batch_number,
                report_date_type=int(contract["report_date_type"]),
                report_type=int(contract["report_type"]),
                timeout_seconds=timeout_seconds,
                attempts=attempts,
            ): (stock_code, statement, batch_number)
            for stock_code, company_type, statement, tab, batch_number, dates in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows, batch_receipt = future.result()
            batch_receipt["profile_company_type"] = profiles[
                batch_receipt["stock_code"]
            ]
            batch_receipt["company_type_source"] = (
                "FROZEN_HISTORICAL_OVERRIDE"
                if batch_receipt["stock_code"] in historical_overrides
                else "CURRENT_PROFILE"
            )
            rows_by_statement[batch_receipt["statement"]].extend(rows)
            batch_receipts.append(batch_receipt)
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"金融业专用报表采集 {completed:04d}/{len(futures)}。",
                    flush=True,
                )
    artifact_receipts: dict[str, Any] = {}
    normalization: dict[str, Any] = {}
    conflict_frames: list[pd.DataFrame] = []
    for statement in ["balance", "cashflow"]:
        frame, conflicts, stats = normalize_statement(
            rows_by_statement[statement], statement
        )
        frame["source_scope"] = "FINANCIAL_COMPANY_TYPE_SPECIFIC_SUPPLEMENT"
        atomic_parquet(paths[statement], frame)
        normalization[statement] = stats
        artifact_receipts[statement] = {
            "path": paths[statement].relative_to(ROOT).as_posix(),
            "rows": len(frame),
            "sha256": sha256_file(paths[statement]),
        }
        if not conflicts.empty:
            conflict_frames.append(conflicts.assign(statement=statement))
    combined_conflicts = (
        pd.concat(conflict_frames, ignore_index=True, sort=False)
        if conflict_frames
        else pd.DataFrame(columns=["statement", "stock_code", "report_end"])
    )
    atomic_parquet(paths["conflicts"], combined_conflicts)
    artifact_receipts["conflicts"] = {
        "path": paths["conflicts"].relative_to(ROOT).as_posix(),
        "rows": len(combined_conflicts),
        "sha256": sha256_file(paths["conflicts"]),
    }
    receipt = {
        "program_id": PROGRAM_ID,
        "status": "PASS_FINANCIAL_SECTOR_COMPANY_TYPE_SPECIFIC_SUPPLEMENT_COMPLETE",
        "evidence_class": "DISCOVERY_EVIDENCE_ONLY_NOT_OFFICIAL_ORIGINAL_PDF",
        "started_at": started_at,
        "finished_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "target_code_count": len(target_codes),
        "target_codes": target_codes,
        "company_type_counts": pd.Series(effective_company_types)
        .value_counts()
        .to_dict(),
        "profile_company_type_counts": pd.Series(profiles).value_counts().to_dict(),
        "historical_company_type_overrides": historical_overrides,
        "profile_receipts": sorted(
            profile_receipts, key=lambda item: item["stock_code"]
        ),
        "batch_task_count": len(tasks),
        "batch_cache_hits": int(sum(item["cache_hit"] for item in batch_receipts)),
        "batch_downloads": int(sum(not item["cache_hit"] for item in batch_receipts)),
        "batch_response_status_counts": pd.Series(
            [item["response_status"] for item in batch_receipts]
        ).value_counts().to_dict(),
        "batch_receipts_payload_sha256": canonical_hash(
            sorted(
                batch_receipts,
                key=lambda item: (
                    item["stock_code"],
                    item["statement"],
                    item["batch_number"],
                ),
            )
        ),
        "normalization": normalization,
        "artifacts": artifact_receipts,
        "availability_clock": "NOTICE_DATE",
        "update_date_usage": "DIAGNOSTIC_ONLY_NOT_INFORMATION_AVAILABILITY",
        "future_return_values_read": False,
        "portfolio_evaluation_run": False,
        "position_generated": False,
    }
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    atomic_json(paths["receipt"], receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def verify_financial_sector_supplement(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    paths = _financial_sector_paths(config)
    if not paths["receipt"].is_file():
        raise FileNotFoundError("金融业专用报表补充收据缺失，必须先运行 --phase collect")
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if receipt.get("protocol_manifest_payload_sha256") != manifest[
        "manifest_payload_sha256"
    ]:
        raise ValueError("金融业专用报表补充收据不属于当前冻结清单")
    for name in ["balance", "cashflow", "conflicts"]:
        if not paths[name].is_file() or sha256_file(paths[name]) != receipt["artifacts"][
            name
        ]["sha256"]:
            raise ValueError(f"金融业专用报表补充产物与收据不一致：{name}")
    return receipt


def monthly_origins(
    membership: pd.DataFrame,
    *,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> list[pd.Timestamp]:
    dates = pd.to_datetime(membership["membership_date"], errors="coerce").dt.normalize()
    dates = dates.loc[dates.between(start, cutoff)].dropna().drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("准入成分表在研究区间内没有开市日")
    frame = pd.DataFrame({"date": dates})
    frame["month"] = frame["date"].dt.to_period("M")
    return [pd.Timestamp(value) for value in frame.groupby("month")["date"].max()]


def _latest_market_snapshot(
    market: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    maximum_age_days: int,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    available_dates = market.loc[
        market["formation_date"].le(origin), "formation_date"
    ]
    if available_dates.empty:
        raise ValueError(f"{origin.date()} 之前没有月度市场截面")
    snapshot_date = pd.Timestamp(available_dates.max())
    if (origin - snapshot_date).days > maximum_age_days:
        raise ValueError(
            f"{origin.date()} 最近市场截面过旧：{snapshot_date.date()}"
        )
    snapshot = market.loc[market["formation_date"].eq(snapshot_date)].copy()
    return snapshot_date, snapshot


def load_monthly_fallback_closes(
    config: dict[str, Any], origins: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    """只读取冻结月末 origin 当日的原始收盘价，作为市值缺口回退输入。"""

    requested_dates = pd.DatetimeIndex(pd.to_datetime(list(origins))).normalize()
    requested_set = set(requested_dates)
    frames: list[pd.DataFrame] = []
    for relative in config["source_contract"]["market_daily_fallback"]["files"]:
        path = project_path(relative)
        frame = pd.read_parquet(path, columns=["stock_code", "date", "raw_close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.loc[
            frame["date"].isin(requested_set), ["stock_code", "date", "raw_close"]
        ].copy()
        if frame.empty:
            continue
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame["raw_close"] = pd.to_numeric(frame["raw_close"], errors="coerce")
        frames.append(frame)
    if not frames:
        raise ValueError("冻结的年度行情文件未覆盖任何月末 origin")
    fallback = pd.concat(frames, ignore_index=True, sort=False)
    conflicts = (
        fallback.dropna(subset=["raw_close"])
        .groupby(["date", "stock_code"])["raw_close"]
        .nunique()
    )
    if bool(conflicts.gt(1).any()):
        examples = [
            f"{date.date()}:{stock_code}"
            for date, stock_code in conflicts.loc[conflicts.gt(1)].index[:5]
        ]
        raise ValueError(f"月末原始收盘价存在冲突重复：{examples}")
    fallback = fallback.sort_values(
        ["date", "stock_code"], kind="mergesort"
    ).drop_duplicates(["date", "stock_code"], keep="last")
    fallback = fallback.rename(
        columns={"date": "fallback_price_date", "raw_close": "fallback_raw_close"}
    )
    return fallback.reset_index(drop=True)


def _latest_weight_snapshot(
    weights: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    maximum_age_days: int,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    eligible = weights.loc[weights["trade_date"].le(origin)]
    if eligible.empty:
        return None, pd.DataFrame(columns=["stock_code", "diagnostic_index_weight"])
    snapshot_date = pd.Timestamp(eligible["trade_date"].max())
    if (origin - snapshot_date).days > maximum_age_days:
        return None, pd.DataFrame(columns=["stock_code", "diagnostic_index_weight"])
    snapshot = weights.loc[weights["trade_date"].eq(snapshot_date), ["stock_code", "weight"]].copy()
    snapshot["diagnostic_index_weight"] = pd.to_numeric(
        snapshot["weight"], errors="coerce"
    ) / 100.0
    return snapshot_date, snapshot[["stock_code", "diagnostic_index_weight"]]


def financial_provider_crosscheck(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    independent: pd.DataFrame,
) -> dict[str, Any]:
    """与独立本地提供方做数值核对；结果不决定取舍或补值。"""

    income_latest = income.sort_values(
        ["stock_code", "report_end", "available_at"], kind="mergesort"
    ).drop_duplicates(["stock_code", "report_end"], keep="last")
    balance_latest = balance.sort_values(
        ["stock_code", "report_end", "available_at"], kind="mergesort"
    ).drop_duplicates(["stock_code", "report_end"], keep="last")
    other = independent.copy()
    other["stock_code"] = other["con_code"].astype(str)
    other["report_end"] = pd.to_datetime(
        other["report_period"], errors="coerce"
    ).dt.normalize()
    other = other.sort_values(
        ["stock_code", "report_end", "available_at"], kind="mergesort"
    ).drop_duplicates(["stock_code", "report_end"], keep="last")
    comparisons = [
        (
            "revenue",
            income_latest,
            "total_operating_revenue",
            "revenue_cny",
        ),
        (
            "parent_net_profit",
            income_latest,
            "parent_net_profit",
            "net_profit_parent_cny",
        ),
        (
            "parent_equity",
            balance_latest,
            "total_parent_equity",
            "equity_parent_cny",
        ),
        (
            "share_capital",
            balance_latest,
            "share_capital",
            "total_shares",
        ),
    ]
    result: dict[str, Any] = {}
    for label, primary, primary_column, other_column in comparisons:
        merged = primary[["stock_code", "report_end", primary_column]].merge(
            other[["stock_code", "report_end", other_column]],
            on=["stock_code", "report_end"],
            how="inner",
        )
        left = pd.to_numeric(merged[primary_column], errors="coerce")
        right = pd.to_numeric(merged[other_column], errors="coerce")
        valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
        scale = np.maximum(np.maximum(left.abs(), right.abs()), 1.0)
        relative_error = ((left - right).abs() / scale).loc[valid]
        result[label] = {
            "overlap_rows": int(valid.sum()),
            "exact_or_near_match_rate": (
                float(relative_error.le(1e-8).mean())
                if len(relative_error)
                else None
            ),
            "median_symmetric_relative_error": (
                float(relative_error.median()) if len(relative_error) else None
            ),
            "p95_symmetric_relative_error": (
                float(relative_error.quantile(0.95))
                if len(relative_error)
                else None
            ),
            "usage": "DIAGNOSTIC_ONLY_NO_VALUE_BACKFILL",
        }
    result["operating_cashflow"] = {
        "status": "NO_VIEW_NO_INDEPENDENT_LOCAL_PROVIDER_CROSSCHECK",
        "usage": "PRIMARY_SECONDARY_AGGREGATOR_SOURCE_ONLY",
    }
    return result


def prepare_independent_share_facts(independent: pd.DataFrame) -> pd.DataFrame:
    """规范已登记独立提供方的点时总股本，仅供缺口回退。"""

    required = {"con_code", "report_period", "available_at", "total_shares"}
    missing = sorted(required.difference(independent.columns))
    if missing:
        raise ValueError(f"独立总股本源缺字段：{missing}")
    shares = independent[
        ["con_code", "report_period", "available_at", "total_shares"]
    ].rename(
        columns={
            "con_code": "stock_code",
            "report_period": "independent_share_report_end",
            "available_at": "independent_share_available_at",
            "total_shares": "independent_total_shares",
        }
    )
    shares["stock_code"] = shares["stock_code"].astype(str)
    shares["independent_share_report_end"] = pd.to_datetime(
        shares["independent_share_report_end"], errors="coerce"
    ).dt.normalize()
    shares["independent_share_available_at"] = pd.to_datetime(
        shares["independent_share_available_at"], errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    shares["independent_total_shares"] = pd.to_numeric(
        shares["independent_total_shares"], errors="coerce"
    )
    shares = shares.loc[
        shares["independent_share_report_end"].notna()
        & shares["independent_share_available_at"].notna()
        & shares["independent_total_shares"].gt(0)
    ].copy()
    return shares.sort_values(
        [
            "stock_code",
            "independent_share_report_end",
            "independent_share_available_at",
        ],
        kind="mergesort",
    ).drop_duplicates(
        ["stock_code", "independent_share_report_end", "independent_share_available_at"],
        keep="last",
    )


def derive_independent_share_snapshot(
    shares: pd.DataFrame,
    member_codes: Iterable[str],
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """选择 origin 当时已可得的最新独立总股本，不允许向后取值。"""

    origin = pd.Timestamp(origin).normalize()
    codes = set(map(str, member_codes))
    eligible = shares.loc[
        shares["stock_code"].isin(codes)
        & shares["independent_share_report_end"].le(origin)
        & shares["independent_share_available_at"].le(origin)
    ].copy()
    columns = [
        "stock_code",
        "independent_share_report_end",
        "independent_share_available_at",
        "independent_total_shares",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    return (
        eligible.sort_values(
            [
                "stock_code",
                "independent_share_report_end",
                "independent_share_available_at",
            ],
            kind="mergesort",
        )
        .drop_duplicates("stock_code", keep="last")[columns]
        .reset_index(drop=True)
    )


def build_component_and_cf_states(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = config["source_contract"]
    artifacts = config["artifacts"]
    membership = pd.read_parquet(project_path(source["membership"]["path"]))
    membership["membership_date"] = pd.to_datetime(
        membership["membership_date"], errors="coerce"
    ).dt.normalize()
    membership["stock_code"] = membership["symbol"].astype(str)
    market = pd.read_parquet(project_path(source["market_cross_section"]["path"]))
    market["formation_date"] = pd.to_datetime(
        market["formation_date"], errors="coerce"
    ).dt.normalize()
    market["market_asof_date"] = pd.to_datetime(
        market["market_asof_date"], errors="coerce"
    ).dt.normalize()
    market["stock_code"] = market["stock_code"].astype(str)
    market = market.sort_values(
        ["formation_date", "stock_code"], kind="mergesort"
    ).drop_duplicates(["formation_date", "stock_code"], keep="last")
    weights = pd.read_parquet(
        project_path(source["historical_index_weight_diagnostic"]["path"])
    ).rename(columns={"con_code": "stock_code"})
    weights["trade_date"] = pd.to_datetime(
        weights["trade_date"], errors="coerce"
    ).dt.normalize()
    weights["stock_code"] = weights["stock_code"].astype(str)
    income = prepare_statement_vintages(
        pd.read_parquet(project_path(artifacts["statement_income"])),
        ["total_operating_revenue", "operating_profit", "parent_net_profit"],
    )
    general_cashflow = pd.read_parquet(
        project_path(artifacts["statement_cashflow"])
    ).assign(source_scope="GENERAL_STATEMENT_TABLE")
    financial_sector_cashflow = pd.read_parquet(
        project_path(artifacts["financial_sector_statement_cashflow"])
    )
    cashflow = prepare_statement_vintages(
        pd.concat(
            [general_cashflow, financial_sector_cashflow],
            ignore_index=True,
            sort=False,
        ),
        ["operating_cashflow"],
    )
    general_balance = pd.read_parquet(
        project_path(artifacts["statement_balance"])
    ).assign(source_scope="GENERAL_STATEMENT_TABLE")
    financial_sector_balance = pd.read_parquet(
        project_path(artifacts["financial_sector_statement_balance"])
    )
    balance = prepare_statement_vintages(
        pd.concat(
            [general_balance, financial_sector_balance],
            ignore_index=True,
            sort=False,
        ),
        ["total_parent_equity", "total_assets", "share_capital"],
    )
    independent_financials = pd.read_parquet(
        project_path(source["independent_financial_crosscheck"]["path"])
    )
    independent_shares = prepare_independent_share_facts(independent_financials)
    start = pd.Timestamp(config["program"]["study_start"])
    cutoff = pd.Timestamp(config["program"]["market_data_cutoff"])
    origins = monthly_origins(membership, start=start, cutoff=cutoff)
    fallback_closes = load_monthly_fallback_closes(config, origins)
    maximum_age = int(config["monthly_origin"]["market_snapshot_max_age_calendar_days"])
    cf_contract = config["cashflow_state"]
    growth_clip = tuple(map(float, cf_contract["growth_clip"]))
    roe_clip = tuple(map(float, cf_contract["roe_clip"]))
    roe_change_clip = tuple(map(float, cf_contract["roe_change_clip"]))
    component_frames: list[pd.DataFrame] = []
    cashflow_rows: list[dict[str, Any]] = []
    for number, origin in enumerate(origins, start=1):
        member_codes = membership.loc[
            membership["membership_date"].eq(origin), "stock_code"
        ].astype(str)
        if member_codes.nunique() != int(source["membership"]["expected_members_per_session"]):
            raise ValueError(f"{origin.date()} 准入成分数不是 300")
        snapshot_date, market_snapshot = _latest_market_snapshot(
            market, origin, maximum_age_days=maximum_age
        )
        market_members = pd.DataFrame({"stock_code": member_codes.unique()}).merge(
            market_snapshot[
                [
                    "stock_code",
                    "industry_code",
                    "industry_name",
                    "total_market_cap",
                    "market_asof_date",
                    "price_asof_date",
                    "report_end_close",
                    "total_shares",
                ]
            ],
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        fallback_snapshot = fallback_closes.loc[
            fallback_closes["fallback_price_date"].eq(origin),
            ["stock_code", "fallback_price_date", "fallback_raw_close"],
        ]
        market_members = market_members.merge(
            fallback_snapshot,
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        independent_share_snapshot = derive_independent_share_snapshot(
            independent_shares, member_codes, origin=origin
        )
        market_members = market_members.merge(
            independent_share_snapshot,
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        income_snapshot = derive_ttm_pair_snapshot(
            income,
            member_codes,
            origin=origin,
            value_columns=[
                "total_operating_revenue",
                "operating_profit",
                "parent_net_profit",
            ],
            prefix="income",
        )
        cash_snapshot = derive_ttm_pair_snapshot(
            cashflow,
            member_codes,
            origin=origin,
            value_columns=["operating_cashflow"],
            prefix="cashflow",
        )
        balance_snapshot = derive_balance_snapshot(
            balance, member_codes, origin=origin
        )
        component = build_component_snapshot(
            market_members,
            income_snapshot,
            cash_snapshot,
            balance_snapshot,
            origin=origin,
            growth_clip=growth_clip,
            roe_clip=roe_clip,
            roe_change_clip=roe_change_clip,
        )
        weight_date, diagnostic_weights = _latest_weight_snapshot(
            weights, origin, maximum_age_days=maximum_age
        )
        component = component.merge(
            diagnostic_weights,
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        component["market_snapshot_date"] = snapshot_date
        component["diagnostic_weight_snapshot_date"] = weight_date
        component["diagnostic_index_weight_status"] = np.where(
            component["diagnostic_index_weight"].notna(),
            "DIAGNOSTIC_ONLY_UNVERSIONED_NOT_PIT_ADMITTED",
            "NO_VIEW_HISTORICAL_INDEX_WEIGHT",
        )
        component["origin_completion_status"] = (
            "PARTIAL_MONTH_STATE_ONLY"
            if origin.to_period("M") == cutoff.to_period("M")
            and origin < origin.to_period("M").end_time.normalize()
            else "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"
        )
        if int(component["state_weight"].notna().sum()) < int(
            config["monthly_origin"]["minimum_market_cap_member_count"]
        ):
            component["component_data_status"] = (
                "NO_VIEW_MARKET_CAP_MEMBER_COVERAGE_BELOW_GATE"
            )
        else:
            component["component_data_status"] = (
                "PASS_PIT_COMPONENT_STATE_WITH_MARKET_CAP_PROXY_WEIGHT"
            )
        component_frames.append(component)
        aggregate = aggregate_cashflow_snapshot(
            component,
            minimum_coverages={
                key: float(value)
                for key, value in cf_contract["minimum_weight_coverage"].items()
            },
            minimum_market_cap_members=int(
                config["monthly_origin"]["minimum_market_cap_member_count"]
            ),
        )
        aggregate["market_snapshot_date"] = snapshot_date
        aggregate["origin_completion_status"] = component[
            "origin_completion_status"
        ].iloc[0]
        cashflow_rows.append(aggregate)
        print(
            f"状态构造 {number:03d}/{len(origins)} {origin.date()}："
            f"市场值覆盖 {component['state_weight'].notna().sum()}/300，"
            f"CF={aggregate['cf_data_status']}。",
            flush=True,
        )
    components = pd.concat(component_frames, ignore_index=True, sort=False)
    cashflow_panel = add_cashflow_state_columns(
        pd.DataFrame(cashflow_rows),
        minimum_periods=int(cf_contract["expanding_state_min_periods"]),
        zscore_clip=float(cf_contract["expanding_zscore_clip"]),
    )
    pca_inputs = cashflow_panel[
        [
            "weighted_revenue_yoy",
            "weighted_operating_profit_yoy",
            "weighted_operating_cashflow_yoy",
            "weighted_roe_change",
            "improvement_weight_breadth",
            "deterioration_concentration",
        ]
    ].copy()
    pca_inputs["deterioration_concentration"] *= -1.0
    cf_pca = first_principal_component(
        pca_inputs,
        sign_anchor=cashflow_panel["cf_level"],
        minimum_rows=int(config["present_value_state"]["minimum_pca_months"]),
    )
    cashflow_panel["cash_flow_expectation_factor"] = cf_pca.factor
    cashflow_panel["cash_flow_expectation_factor_status"] = cf_pca.status
    crosscheck = financial_provider_crosscheck(
        income,
        balance,
        independent_financials,
    )
    diagnostics = {
        "monthly_origins": len(origins),
        "first_origin": str(min(origins).date()),
        "last_origin": str(max(origins).date()),
        "component_rows": len(components),
        "minimum_market_cap_member_count": int(
            components.groupby("origin")["state_weight"].apply(
                lambda values: values.notna().sum()
            ).min()
        ),
        "market_cap_origins_below_gate": int(
            cashflow_panel["market_cap_gate_pass"].eq(False).sum()
        ),
        "market_cap_source_counts": components["state_market_cap_source"]
        .value_counts()
        .to_dict(),
        "share_source_counts": components["state_share_source"]
        .value_counts()
        .to_dict(),
        "price_source_counts": components["state_price_source"]
        .value_counts()
        .to_dict(),
        "cf_status_counts": cashflow_panel["cf_data_status"].value_counts().to_dict(),
        "cf_pca": {
            "status": cf_pca.status,
            "loadings": cf_pca.loadings,
            "explained_variance_ratio": cf_pca.explained_variance_ratio,
        },
        "provider_crosscheck": crosscheck,
    }
    return components, cashflow_panel, diagnostics


def merge_present_value_with_risk_free(
    present_value: pd.DataFrame,
    risk_free: pd.DataFrame,
) -> pd.DataFrame:
    required_present_value = {"origin"}
    required_risk_free = {"date", "cgb_10y"}
    if missing := required_present_value.difference(present_value.columns):
        raise ValueError(f"PV 面板缺少来源日字段：{sorted(missing)}")
    if missing := required_risk_free.difference(risk_free.columns):
        raise ValueError(f"无风险利率表缺字段：{sorted(missing)}")
    return merge_asof_state(
        present_value,
        risk_free.rename(columns={"date": "cgb_date"}),
        base_date="origin",
        source_date="cgb_date",
        columns=["cgb_10y"],
        tolerance_days=10,
    )


def build_present_value_states(
    config: dict[str, Any],
    components: pd.DataFrame,
    cashflow_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contract = config["present_value_state"]
    cross_section = add_present_value_cross_section(
        components,
        minimum_industry_members=int(contract["minimum_industry_members"]),
        winsor_limits=tuple(map(float, contract["cross_section_winsorization"])),
    )
    portfolios = build_fixed_present_value_portfolios(cross_section)
    present_value, pca = aggregate_present_value_panel(
        cross_section,
        portfolios,
        minimum_pca_months=int(contract["minimum_pca_months"]),
    )
    positive_ep_rows: list[dict[str, Any]] = []
    for origin, group in cross_section.groupby("origin", sort=True):
        positive_ep, coverage, count = weighted_average(
            group["earnings_to_price"].where(group["earnings_to_price"].gt(0)),
            group["state_weight"],
        )
        positive_ep_rows.append(
            {
                "origin": pd.Timestamp(origin),
                "cap_weighted_positive_earnings_yield": positive_ep,
                "positive_earnings_yield_weight_coverage": coverage,
                "positive_earnings_yield_member_count": count,
            }
        )
    present_value = present_value.merge(
        pd.DataFrame(positive_ep_rows), on="origin", how="left", validate="one_to_one"
    )
    cgb_path = project_path(
        config["source_contract"]["risk_bearing_inputs"]["cgb_curve"]
    )
    cgb = pd.read_parquet(cgb_path)[["date", "cgb_10y"]].copy()
    present_value = merge_present_value_with_risk_free(present_value, cgb)
    present_value["equity_risk_premium_state"] = (
        present_value["cap_weighted_positive_earnings_yield"]
        - pd.to_numeric(present_value["cgb_10y"], errors="coerce") / 100.0
    )
    present_value["equity_risk_premium_expanding_z"] = expanding_zscore(
        present_value["equity_risk_premium_state"],
        minimum_periods=int(config["cashflow_state"]["expanding_state_min_periods"]),
        clip=float(config["cashflow_state"]["expanding_zscore_clip"]),
    )
    present_value = present_value.merge(
        cashflow_panel[["origin", "cash_flow_expectation_factor"]],
        on="origin",
        how="left",
        validate="one_to_one",
    )
    present_value["present_value_data_status"] = np.select(
        [
            present_value["cross_sectional_expected_return_factor"].notna()
            & present_value["equity_risk_premium_state"].notna(),
            present_value["cap_weighted_earnings_to_price"].notna(),
        ],
        [
            "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE",
            "PARTIAL_PRESENT_VALUE_STATE_PCA_OR_RISK_FREE_NO_VIEW",
        ],
        default="NO_VIEW_PRESENT_VALUE_STATE_COVERAGE_INSUFFICIENT",
    )
    diagnostics = {
        "portfolio_ids": sorted(portfolios["portfolio_id"].unique().tolist()),
        "portfolio_count": int(portfolios["portfolio_id"].nunique()),
        "pca_status": pca.status,
        "pca_loadings": pca.loadings,
        "pca_explained_variance_ratio": pca.explained_variance_ratio,
        "status_counts": present_value["present_value_data_status"]
        .value_counts()
        .to_dict(),
        "dividend_yield_status": contract["dividend_yield_status"],
        "direct_cashflow_duration_status": contract[
            "direct_cashflow_duration_status"
        ],
    }
    return cross_section, present_value, portfolios, diagnostics


def _available_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return (
        parsed.dt.tz_convert(TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )


def _date_only(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )


def merge_asof_state(
    base: pd.DataFrame,
    source: pd.DataFrame,
    *,
    base_date: str,
    source_date: str,
    columns: Sequence[str],
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    left = base.copy()
    left[base_date] = _date_only(left[base_date])
    right = source[[source_date, *columns]].dropna(subset=[source_date]).copy()
    right[source_date] = _date_only(right[source_date])
    right = right.sort_values(source_date).drop_duplicates(source_date, keep="last")
    tolerance = (
        pd.Timedelta(days=tolerance_days) if tolerance_days is not None else None
    )
    return pd.merge_asof(
        left.sort_values(base_date),
        right,
        left_on=base_date,
        right_on=source_date,
        direction="backward",
        tolerance=tolerance,
    )


def calculate_internal_market_states(
    config: dict[str, Any],
    component_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = project_path(
        config["source_contract"]["risk_bearing_inputs"][
            "constituent_total_return_close"
        ]
    )
    prices = pd.read_parquet(path, columns=["date", "con_code", "total_return_close"])
    prices["date"] = _date_only(prices["date"])
    prices["stock_code"] = prices["con_code"].astype(str)
    prices["total_return_close"] = pd.to_numeric(
        prices["total_return_close"], errors="coerce"
    )
    prices = prices.dropna(subset=["date", "stock_code", "total_return_close"])
    prices = prices.sort_values(["date", "stock_code"], kind="mergesort")
    unique_dates = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
    rows: list[dict[str, Any]] = []
    grouped_components = list(component_panel.groupby("origin", sort=True))
    for number, (origin, component) in enumerate(grouped_components, start=1):
        eligible_dates = unique_dates[unique_dates <= pd.Timestamp(origin)]
        if len(eligible_dates) < 61:
            rows.append(
                {
                    "origin": pd.Timestamp(origin),
                    "internal_market_state_status": "NO_VIEW_LT_61_MARKET_DAYS",
                }
            )
            continue
        window_dates = eligible_dates[-61:]
        codes = set(component["stock_code"].astype(str))
        window = prices.loc[
            prices["date"].isin(window_dates) & prices["stock_code"].isin(codes)
        ]
        pivot = window.pivot(index="date", columns="stock_code", values="total_return_close")
        pivot = pivot.reindex(window_dates)
        observed = pivot.notna().sum(axis=0)
        valid_codes = observed.loc[observed.ge(50)].index
        pivot = pivot[valid_codes]
        daily_returns = pivot.pct_change(fill_method=None)
        member_return_60 = pivot.iloc[-1] / pivot.iloc[0] - 1.0
        member_return_60 = member_return_60.replace([np.inf, -np.inf], np.nan)
        member_valid = member_return_60.dropna()
        mapping = component.set_index("stock_code")
        weights = pd.to_numeric(
            mapping.reindex(member_valid.index)["state_weight"], errors="coerce"
        )
        valid_weight = weights.notna() & weights.gt(0)
        large_weight_return = (
            float(
                np.average(
                    member_valid.loc[valid_weight],
                    weights=weights.loc[valid_weight],
                )
            )
            if valid_weight.sum() >= 2
            else np.nan
        )
        median_member_return = (
            float(member_valid.median()) if len(member_valid) else np.nan
        )
        member_positive_breadth = (
            float(member_valid.gt(0).mean()) if len(member_valid) else np.nan
        )
        industry_map = mapping.reindex(daily_returns.columns)["industry_code"]
        valid_industry_columns = industry_map.dropna().index
        industry_returns = pd.DataFrame(index=daily_returns.index)
        if len(valid_industry_columns):
            transposed = daily_returns[valid_industry_columns].T
            transposed["industry_code"] = industry_map.loc[valid_industry_columns].astype(
                str
            )
            industry_returns = transposed.groupby(
                "industry_code", observed=True
            ).mean(numeric_only=True).T
        industry_return_20 = (
            (1.0 + industry_returns.tail(20)).prod(min_count=15) - 1.0
            if not industry_returns.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "origin": pd.Timestamp(origin),
                "member_average_pairwise_correlation_60d": average_pairwise_correlation(
                    daily_returns.tail(60)
                ),
                "industry_average_pairwise_correlation_60d": average_pairwise_correlation(
                    industry_returns.tail(60)
                ),
                "large_weight_minus_median_member_return_60d": (
                    large_weight_return - median_member_return
                    if np.isfinite(large_weight_return)
                    and np.isfinite(median_member_return)
                    else np.nan
                ),
                "large_weight_return_60d": large_weight_return,
                "member_median_return_60d": median_member_return,
                "member_positive_breadth_60d": member_positive_breadth,
                "industry_negative_breadth_20d": (
                    float(industry_return_20.lt(0).mean())
                    if len(industry_return_20.dropna())
                    else np.nan
                ),
                "valid_member_return_count_60d": int(len(member_valid)),
                "valid_industry_return_count_20d": int(
                    industry_return_20.notna().sum()
                ),
                "internal_market_state_status": (
                    "PASS_POINT_IN_TIME_INTERNAL_MARKET_STATE"
                    if len(member_valid) >= 240
                    else "PARTIAL_INTERNAL_MARKET_STATE_MEMBER_COVERAGE_BELOW_240"
                ),
            }
        )
        print(
            f"内部风险承载状态 {number:03d}/{len(grouped_components)} "
            f"{pd.Timestamp(origin).date()}：有效成分 {len(member_valid)}。",
            flush=True,
        )
    result = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)
    diagnostics = {
        "status_counts": result["internal_market_state_status"]
        .value_counts()
        .to_dict(),
        "minimum_valid_member_return_count_60d": int(
            pd.to_numeric(
                result.get("valid_member_return_count_60d"), errors="coerce"
            ).min()
        ),
    }
    return result, diagnostics


def _prepare_first_release(
    path: Path,
    *,
    value_name: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["state_available_date"] = _available_date(frame["available_at"])
    frame[value_name] = pd.to_numeric(frame["first_release_value"], errors="coerce")
    return frame[["state_available_date", value_name]]


def build_risk_capacity_state(
    config: dict[str, Any],
    component_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = config["source_contract"]["risk_bearing_inputs"]
    cutoff = pd.Timestamp(config["program"]["market_data_cutoff"])
    market = pd.read_parquet(project_path(paths["etf_price"])).copy()
    market["date"] = _date_only(market["date"])
    market = market.loc[market["date"].le(cutoff)].sort_values("date")
    market = market.drop_duplicates("date", keep="last")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market["amount"] = pd.to_numeric(market["amount"], errors="coerce")
    market["etf_price_return"] = market["close"].pct_change(fill_method=None)
    market["etf_amihud_20d"] = (
        market["etf_price_return"].abs()
        / market["amount"].where(market["amount"].gt(0))
    ).rolling(20, min_periods=15).mean() * 1e8
    base = market[["date", "close", "amount", "etf_price_return", "etf_amihud_20d"]].copy()

    fdr = _prepare_first_release(project_path(paths["fdr007"]), value_name="fdr007")
    base = merge_asof_state(
        base,
        fdr,
        base_date="date",
        source_date="state_available_date",
        columns=["fdr007"],
    )
    cgb = pd.read_parquet(project_path(paths["cgb_curve"])).copy()
    cgb["cgb_date"] = _date_only(cgb["date"])
    cgb["cgb_1y"] = pd.to_numeric(cgb["cgb_1y"], errors="coerce")
    cgb["cgb_10y"] = pd.to_numeric(cgb["cgb_10y"], errors="coerce")
    base = merge_asof_state(
        base,
        cgb,
        base_date="date",
        source_date="cgb_date",
        columns=["cgb_1y", "cgb_10y"],
        tolerance_days=10,
    )
    base["cgb_10y_minus_1y"] = base["cgb_10y"] - base["cgb_1y"]
    credit = pd.read_parquet(project_path(paths["credit_spread"])).copy()
    credit["credit_date"] = _date_only(credit["date"])
    credit["credit_spread_3y_bp"] = pd.to_numeric(
        credit["credit_spread_3y_bp"], errors="coerce"
    )
    base = merge_asof_state(
        base,
        credit,
        base_date="date",
        source_date="credit_date",
        columns=["credit_spread_3y_bp"],
        tolerance_days=10,
    )
    fx = _prepare_first_release(
        project_path(paths["usdcny_midpoint"]), value_name="usdcny_midpoint"
    )
    base = merge_asof_state(
        base,
        fx,
        base_date="date",
        source_date="state_available_date",
        columns=["usdcny_midpoint"],
    )
    base["usdcny_20d_log_change"] = np.log(
        base["usdcny_midpoint"].where(base["usdcny_midpoint"].gt(0))
    ).diff(20)

    margin_primary = pd.read_parquet(project_path(paths["market_margin_2015_2025"])).copy()
    margin_extension = pd.read_parquet(project_path(paths["market_margin_extension"])).copy()
    for frame, priority in [(margin_extension, 0), (margin_primary, 1)]:
        frame["margin_date"] = _date_only(frame["date"])
        frame["source_priority"] = priority
        frame["market_rzye"] = pd.to_numeric(frame["market_rzye"], errors="coerce")
    margin = pd.concat(
        [margin_extension, margin_primary], ignore_index=True, sort=False
    ).sort_values(["margin_date", "source_priority"])
    margin = margin.drop_duplicates("margin_date", keep="last")
    base = merge_asof_state(
        base,
        margin,
        base_date="date",
        source_date="margin_date",
        columns=["market_rzye"],
        tolerance_days=10,
    )
    base["financing_balance_20d_log_change"] = np.log(
        base["market_rzye"].where(base["market_rzye"].gt(0))
    ).diff(20)

    shares = pd.read_parquet(project_path(paths["etf_shares"])).copy()
    shares["share_available_date"] = _date_only(shares["feature_asof"])
    shares["share_available_date"] = shares["share_available_date"].fillna(
        _date_only(shares["date"])
    )
    shares["fund_shares"] = pd.to_numeric(shares["fund_shares"], errors="coerce")
    base = merge_asof_state(
        base,
        shares,
        base_date="date",
        source_date="share_available_date",
        columns=["fund_shares"],
    )
    base["etf_share_20d_log_change"] = np.log(
        base["fund_shares"].where(base["fund_shares"].gt(0))
    ).diff(20)

    internal, internal_diagnostics = calculate_internal_market_states(
        config, component_panel
    )
    base = merge_asof_state(
        base,
        internal,
        base_date="date",
        source_date="origin",
        columns=[
            "member_average_pairwise_correlation_60d",
            "industry_average_pairwise_correlation_60d",
            "large_weight_minus_median_member_return_60d",
            "large_weight_return_60d",
            "member_median_return_60d",
            "member_positive_breadth_60d",
            "industry_negative_breadth_20d",
            "valid_member_return_count_60d",
            "valid_industry_return_count_20d",
            "internal_market_state_status",
        ],
    )
    capacity_inputs = {
        "fdr007": -pd.to_numeric(base["fdr007"], errors="coerce"),
        "cgb_10y_minus_1y": pd.to_numeric(
            base["cgb_10y_minus_1y"], errors="coerce"
        ),
        "credit_spread_3y_bp": -pd.to_numeric(
            base["credit_spread_3y_bp"], errors="coerce"
        ),
        "usdcny_20d_log_change": -pd.to_numeric(
            base["usdcny_20d_log_change"], errors="coerce"
        ),
        "etf_amihud_20d": -pd.to_numeric(base["etf_amihud_20d"], errors="coerce"),
        "financing_balance_20d_log_change": pd.to_numeric(
            base["financing_balance_20d_log_change"], errors="coerce"
        ),
        "etf_share_20d_log_change": pd.to_numeric(
            base["etf_share_20d_log_change"], errors="coerce"
        ),
        "member_average_pairwise_correlation_60d": -pd.to_numeric(
            base["member_average_pairwise_correlation_60d"], errors="coerce"
        ),
        "industry_average_pairwise_correlation_60d": -pd.to_numeric(
            base["industry_average_pairwise_correlation_60d"], errors="coerce"
        ),
        "large_weight_minus_median_member_return_60d": -pd.to_numeric(
            base["large_weight_minus_median_member_return_60d"], errors="coerce"
        ).abs(),
        "industry_negative_breadth_20d": -pd.to_numeric(
            base["industry_negative_breadth_20d"], errors="coerce"
        ),
    }
    capacity_frame = pd.DataFrame(capacity_inputs, index=base.index)
    minimum_periods = int(
        config["risk_bearing_capacity"]["expanding_standardization_min_periods"]
    )
    zscore_clip = float(config["risk_bearing_capacity"]["expanding_zscore_clip"])
    z_columns: list[str] = []
    for column in capacity_frame.columns:
        target = f"capacity_{column}_expanding_z"
        base[target] = expanding_zscore(
            capacity_frame[column],
            minimum_periods=minimum_periods,
            clip=zscore_clip,
        )
        z_columns.append(target)
    valid_z = base[z_columns].notna().sum(axis=1)
    base["rc_capacity_expanding_median"] = base[z_columns].median(
        axis=1, skipna=True
    ).where(valid_z.ge(6))
    pca = first_principal_component(
        capacity_frame,
        sign_anchor=base["rc_capacity_expanding_median"],
        minimum_rows=minimum_periods,
    )
    base["risk_bearing_capacity_factor"] = pca.factor
    base["rc_state_status"] = np.select(
        [
            base["risk_bearing_capacity_factor"].notna(),
            base["rc_capacity_expanding_median"].notna(),
        ],
        [
            "PASS_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR",
            "PARTIAL_RC_STATE_PCA_COVERAGE_NO_VIEW",
        ],
        default="NO_VIEW_RC_STATE_WARMUP_OR_COVERAGE",
    )
    base["rejected_rule_reuse_status"] = (
        "STATE_MEASUREMENT_ONLY_DISCOVERY_EVIDENCE_ONLY"
    )
    base["portfolio_evaluation_allowed"] = False
    base["model_position_target"] = "UNSET"
    diagnostics = {
        "rows": len(base),
        "first_date": str(base["date"].min().date()),
        "last_date": str(base["date"].max().date()),
        "status_counts": base["rc_state_status"].value_counts().to_dict(),
        "pca_status": pca.status,
        "pca_loadings": pca.loadings,
        "pca_explained_variance_ratio": pca.explained_variance_ratio,
        "raw_nonmissing_counts": {
            column: int(pd.to_numeric(base[column], errors="coerce").notna().sum())
            for column in [
                "fdr007",
                "cgb_10y_minus_1y",
                "credit_spread_3y_bp",
                "usdcny_20d_log_change",
                "etf_amihud_20d",
                "financing_balance_20d_log_change",
                "etf_share_20d_log_change",
                "member_average_pairwise_correlation_60d",
                "industry_average_pairwise_correlation_60d",
                "large_weight_minus_median_member_return_60d",
                "industry_negative_breadth_20d",
            ]
        },
        "internal_market": internal_diagnostics,
        "credit_spread_clock_status": (
            "DISCOVERY_ONLY_HISTORICAL_PUBLICATION_TIMESTAMP_NOT_VERSION_PROVEN"
        ),
        "etf_share_clock_status": (
            "STATE_MEASUREMENT_ONLY_HISTORICAL_PUBLICATION_TIMESTAMP_NOT_VERSION_PROVEN"
        ),
    }
    return base, diagnostics


def _state_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "component_monthly_panel": project_path(
            artifacts["component_monthly_panel"]
        ),
        "cashflow_panel": project_path(artifacts["cashflow_panel"]),
        "present_value_panel": project_path(artifacts["present_value_panel"]),
        "present_value_portfolios": project_path(
            artifacts["present_value_portfolios"]
        ),
        "risk_capacity_panel": project_path(artifacts["risk_capacity_panel"]),
    }


def build_and_freeze_states(
    config: dict[str, Any],
    manifest: dict[str, Any],
    general_collection_receipt: dict[str, Any],
    financial_sector_collection_receipt: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = project_path(config["artifacts"]["state_freeze_receipt"])
    paths = _state_artifact_paths(config)
    if receipt_path.is_file() and all(path.is_file() for path in paths.values()):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol_manifest_payload_sha256") != manifest[
            "manifest_payload_sha256"
        ]:
            raise ValueError("既有状态冻结收据不属于当前冻结清单")
        for name, path in paths.items():
            if sha256_file(path) != receipt["artifacts"][name]["sha256"]:
                raise ValueError(f"已冻结状态产物哈希不一致：{name}")
        print("三张状态面板已冻结且哈希通过，跳过重复构造。", flush=True)
        return receipt
    if receipt_path.exists() or any(path.exists() for path in paths.values()):
        raise RuntimeError("状态产物不完整，禁止静默覆盖；请先人工审计残留文件")
    started_at = now_iso()
    components, cashflow_panel, cf_diagnostics = build_component_and_cf_states(
        config
    )
    (
        present_value_components,
        present_value_panel,
        present_value_portfolios,
        pv_diagnostics,
    ) = build_present_value_states(config, components, cashflow_panel)
    risk_capacity_panel, rc_diagnostics = build_risk_capacity_state(
        config, present_value_components
    )
    frames = {
        "component_monthly_panel": present_value_components,
        "cashflow_panel": cashflow_panel,
        "present_value_panel": present_value_panel,
        "present_value_portfolios": present_value_portfolios,
        "risk_capacity_panel": risk_capacity_panel,
    }
    artifact_receipts: dict[str, Any] = {}
    for name, frame in frames.items():
        atomic_parquet(paths[name], frame)
        artifact_receipts[name] = {
            "path": paths[name].relative_to(ROOT).as_posix(),
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": sha256_file(paths[name]),
        }
    receipt = {
        "program_id": PROGRAM_ID,
        "status": (
            "PARTIAL_PASS_MECHANISM_STATE_PRODUCTS_FROZEN_WITH_EXPLICIT_"
            "WEIGHT_DIVIDEND_AND_SOURCE_LIMITATIONS"
        ),
        "started_at": started_at,
        "frozen_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "general_collection_receipt_sha256": sha256_file(
            project_path(config["artifacts"]["collection_receipt"])
        ),
        "general_collection_status": general_collection_receipt["status"],
        "general_collection_protocol_manifest_payload_sha256": general_collection_receipt[
            "protocol_manifest_payload_sha256"
        ],
        "general_collection_carry_forward_after_outcome_blind_correction": (
            general_collection_receipt["protocol_manifest_payload_sha256"]
            != manifest["manifest_payload_sha256"]
        ),
        "financial_sector_collection_receipt_sha256": sha256_file(
            project_path(
                config["artifacts"]["financial_sector_collection_receipt"]
            )
        ),
        "financial_sector_collection_status": financial_sector_collection_receipt[
            "status"
        ],
        "financial_sector_collection_protocol_manifest_payload_sha256": (
            financial_sector_collection_receipt[
                "protocol_manifest_payload_sha256"
            ]
        ),
        "products": {
            "cashflow": config["cashflow_state"]["product_id"],
            "present_value": config["present_value_state"]["product_id"],
            "risk_capacity": config["risk_bearing_capacity"]["product_id"],
        },
        "diagnostics": {
            "cashflow": cf_diagnostics,
            "present_value": pv_diagnostics,
            "risk_capacity": rc_diagnostics,
        },
        "limitations": {
            "financial_source": (
                "GENERAL_PLUS_COMPANY_TYPE_SPECIFIC_SECONDARY_AGGREGATOR_"
                "NOTICE_DATE_PRIMARY_UPDATE_DIAGNOSTIC_CURRENT_VALUE_MAY_REFLECT_"
                "REVISION_NOT_FULL_OFFICIAL_PDF_ARCHIVE"
            ),
            "operating_cashflow_crosscheck": (
                "NO_VIEW_NO_INDEPENDENT_LOCAL_PROVIDER_CROSSCHECK"
            ),
            "weight": (
                "POINT_IN_TIME_TOTAL_MARKET_CAP_PROXY_PRIMARY_WITH_HASHED_"
                "AVAILABLE_AT_TOTAL_SHARES_AND_EXACT_ORIGIN_CLOSE_FALLBACK;"
                "UNVERSIONED_HISTORICAL_INDEX_WEIGHT_DIAGNOSTIC_ONLY"
            ),
            "component_dividend_yield": (
                "NO_VIEW_NO_POINT_IN_TIME_COMPONENT_DIVIDEND_ARCHIVE"
            ),
            "direct_cashflow_duration": "NO_VIEW_PROXY_ONLY",
            "credit_and_etf_share_history": (
                "STATE_MEASUREMENT_ONLY_HISTORICAL_PUBLICATION_CLOCK_NOT_FULLY_"
                "VERSION_PROVEN"
            ),
        },
        "artifacts": artifact_receipts,
        "state_products_frozen_before_explanatory_return_read": True,
        "future_return_values_read": False,
        "return_evaluation": "NOT_RUN_STATE_STAGE_ONLY",
        "portfolio_evaluation_run": False,
        "sharpe_calculated": False,
        "position_generated": False,
        "model_position_target": "UNSET",
        "old_rejected_protocols_rescued": False,
    }
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def verify_state_freeze(config: dict[str, Any]) -> dict[str, Any]:
    receipt_path = project_path(config["artifacts"]["state_freeze_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("状态冻结收据缺失，禁止读取解释性收益")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("future_return_values_read") is not False:
        raise ValueError("状态冻结收据的结果盲标记异常")
    if receipt.get("state_products_frozen_before_explanatory_return_read") is not True:
        raise ValueError("状态产品未确认在收益读取前冻结")
    for name, path in _state_artifact_paths(config).items():
        if not path.is_file() or sha256_file(path) != receipt["artifacts"][name][
            "sha256"
        ]:
            raise ValueError(f"冻结状态产物漂移：{name}")
    return receipt


def load_csi300_total_return(config: dict[str, Any]) -> pd.DataFrame:
    relative = config["source_contract"][
        "explanatory_outcomes_locked_until_state_freeze"
    ]["csi300_total_return"]
    frame = pd.read_parquet(project_path(relative)).copy()
    required = {"date", "close"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"沪深300全收益序列缺字段：{sorted(missing)}")
    frame["date"] = _date_only(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    cutoff = pd.Timestamp(config["program"]["market_data_cutoff"])
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(config["program"]["study_start"]), cutoff)
        & frame["close"].gt(0)
    ].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if len(frame) < 252:
        raise ValueError("沪深300全收益序列不足 252 个开市日")
    frame["daily_total_return"] = frame["close"].pct_change(fill_method=None)
    frame["wealth"] = frame["close"] / float(frame["close"].iloc[0])
    return frame.reset_index(drop=True)


def etf_total_return_crosscheck(
    config: dict[str, Any],
    csi300_total_return: pd.DataFrame,
) -> dict[str, Any]:
    outcomes = config["source_contract"][
        "explanatory_outcomes_locked_until_state_freeze"
    ]
    market = pd.read_parquet(
        project_path(config["source_contract"]["risk_bearing_inputs"]["etf_price"])
    ).copy()
    market["date"] = _date_only(market["date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    dividends = pd.read_csv(project_path(outcomes["etf_dividends"])).copy()
    date_column = "ex_date" if "ex_date" in dividends.columns else "record_date"
    if date_column not in dividends.columns or "cash_dividend_per_share" not in dividends.columns:
        raise ValueError("510300 分红表缺少日期或每份现金分红字段")
    dividends["date"] = _date_only(dividends[date_column])
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    cash = dividends.groupby("date", observed=True)[
        "cash_dividend_per_share"
    ].sum()
    market["cash_dividend_per_share"] = market["date"].map(cash).fillna(0.0)
    market["etf_total_return"] = (
        (market["close"] + market["cash_dividend_per_share"])
        / market["close"].shift(1)
        - 1.0
    )
    overlap = market[["date", "etf_total_return"]].merge(
        csi300_total_return[["date", "daily_total_return"]],
        on="date",
        how="inner",
    ).dropna()
    difference = overlap["etf_total_return"] - overlap["daily_total_return"]
    return {
        "status": "PASS_EXPLANATORY_UNDERLYING_ETF_TOTAL_RETURN_CROSSCHECK",
        "overlap_days": len(overlap),
        "daily_return_correlation": float(
            overlap[["etf_total_return", "daily_total_return"]].corr().iloc[0, 1]
        ),
        "median_absolute_daily_difference": float(difference.abs().median()),
        "cumulative_etf_total_return": float(
            (1.0 + overlap["etf_total_return"]).prod() - 1.0
        ),
        "cumulative_csi300_total_return": float(
            (1.0 + overlap["daily_total_return"]).prod() - 1.0
        ),
        "portfolio_evaluation": "NOT_RUN",
    }


def _state_panel_at_origins(
    config: dict[str, Any],
    total_return: pd.DataFrame,
) -> pd.DataFrame:
    paths = _state_artifact_paths(config)
    cashflow = pd.read_parquet(paths["cashflow_panel"])
    present_value = pd.read_parquet(paths["present_value_panel"])
    risk_capacity = pd.read_parquet(paths["risk_capacity_panel"])
    cashflow["origin"] = _date_only(cashflow["origin"])
    present_value["origin"] = _date_only(present_value["origin"])
    risk_capacity["date"] = _date_only(risk_capacity["date"])
    state = cashflow.merge(
        present_value,
        on="origin",
        how="inner",
        validate="one_to_one",
        suffixes=("_cf", "_dr"),
    ).sort_values("origin")
    rc_columns = [
        "date",
        "risk_bearing_capacity_factor",
        "rc_capacity_expanding_median",
        "member_median_return_60d",
        "member_positive_breadth_60d",
        "large_weight_return_60d",
        "large_weight_minus_median_member_return_60d",
        "industry_negative_breadth_20d",
        "rc_state_status",
    ]
    state = pd.merge_asof(
        state.sort_values("origin"),
        risk_capacity[rc_columns].sort_values("date"),
        left_on="origin",
        right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    ).drop(columns="date")
    daily = total_return[["date", "close"]].copy()
    state = pd.merge_asof(
        state.sort_values("origin"),
        daily.rename(columns={"date": "return_asof_date", "close": "tr_close"}),
        left_on="origin",
        right_on="return_asof_date",
        direction="backward",
        tolerance=pd.Timedelta(days=5),
    )
    state["monthly_total_return"] = state["tr_close"].pct_change(fill_method=None)
    daily_dates = pd.DatetimeIndex(total_return["date"])
    daily_close = total_return["close"].to_numpy(dtype=float)
    for horizon in config["program"]["primary_horizons_market_days"]:
        values: list[float] = []
        statuses: list[str] = []
        for origin in state["origin"]:
            position = int(daily_dates.searchsorted(origin, side="right") - 1)
            target = position + int(horizon)
            if position < 0 or target >= len(daily_close):
                values.append(np.nan)
                statuses.append("CENSORED_NO_VIEW")
            else:
                values.append(float(daily_close[target] / daily_close[position] - 1.0))
                statuses.append("OBSERVED_EXPLANATORY_ONLY")
        state[f"forward_total_return_{horizon}d"] = values
        state[f"forward_total_return_{horizon}d_status"] = statuses
    state["cf_news"] = pd.to_numeric(state["cf_level"], errors="coerce").diff()
    state["discount_rate_easing_news"] = -pd.to_numeric(
        state["equity_risk_premium_expanding_z"], errors="coerce"
    ).diff()
    state["risk_capacity_news"] = pd.to_numeric(
        state["risk_bearing_capacity_factor"], errors="coerce"
    ).diff()
    return state


def fit_explanatory_decomposition(
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = ["cf_news", "discount_rate_easing_news", "risk_capacity_news"]
    eligible = state[["monthly_total_return", *features]].dropna()
    if len(eligible) < 24:
        raise ValueError("完整月度状态不足 24 期，不能运行冻结的解释性 OLS")
    x = eligible[features].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    y = eligible["monthly_total_return"].to_numpy(dtype=float)
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    total_sum = float(np.square(y - y.mean()).sum())
    residual_sum = float(np.square(residual).sum())
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    result = state.copy()
    result["cashflow_news_contribution"] = coefficients[1] * result["cf_news"]
    result["discount_rate_news_contribution"] = (
        coefficients[2] * result["discount_rate_easing_news"]
    )
    result["risk_capacity_news_contribution"] = (
        coefficients[3] * result["risk_capacity_news"]
    )
    contribution_sum = result[
        [
            "cashflow_news_contribution",
            "discount_rate_news_contribution",
            "risk_capacity_news_contribution",
        ]
    ].sum(axis=1, min_count=3)
    result["decomposition_residual_including_intercept"] = (
        result["monthly_total_return"] - contribution_sum
    )
    result["ols_fitted_return"] = coefficients[0] + contribution_sum
    result["ols_residual_excluding_intercept"] = (
        result["monthly_total_return"] - result["ols_fitted_return"]
    )
    result["decomposition_status"] = np.where(
        contribution_sum.notna(),
        "OBSERVED_EXPLANATORY_OLS_NOT_ACCOUNTING_IDENTITY",
        "NO_VIEW_INCOMPLETE_STATE_INNOVATIONS",
    )
    diagnostics = {
        "status": "PASS_EXPLANATORY_FULL_SAMPLE_OLS_NO_FEATURE_SELECTION",
        "observations": len(eligible),
        "design_rank": int(rank),
        "singular_values": [float(value) for value in singular],
        "coefficients": {
            "intercept": float(coefficients[0]),
            "cf_news": float(coefficients[1]),
            "discount_rate_easing_news": float(coefficients[2]),
            "risk_capacity_news": float(coefficients[3]),
        },
        "r_squared": float(r_squared),
        "accounting_identity_claim": False,
        "predictive_claim": False,
        "portfolio_evaluation": "NOT_RUN",
    }
    return result, diagnostics


def _coalesce_monthly_flags(
    frame: pd.DataFrame,
    flag: pd.Series,
    *,
    event_type: str,
) -> list[dict[str, Any]]:
    selected = frame.loc[flag.fillna(False), "origin"].sort_values().tolist()
    if not selected:
        return []
    episodes: list[list[pd.Timestamp]] = [[pd.Timestamp(selected[0])]]
    for value in selected[1:]:
        date = pd.Timestamp(value)
        previous = episodes[-1][-1]
        month_gap = (date.year - previous.year) * 12 + date.month - previous.month
        if month_gap <= 1:
            episodes[-1].append(date)
        else:
            episodes.append([date])
    return [
        {
            "event_type": event_type,
            "start_date": episode[0],
            "end_date": episode[-1],
            "right_censored": False,
            "mechanical_observation_count": len(episode),
        }
        for episode in episodes
    ]


def build_mechanical_events(
    config: dict[str, Any],
    total_return: pd.DataFrame,
    monthly_state: pd.DataFrame,
) -> pd.DataFrame:
    contract = config["cycle_atlas"]
    drawdowns = detect_drawdown_episodes(
        total_return["date"],
        total_return["wealth"],
        activation_threshold=float(contract["drawdown_activation"]),
    )
    major = drawdowns.loc[
        drawdowns["peak_to_trough_drawdown"].le(
            float(contract["major_drawdown_threshold"])
        )
    ].copy()
    events: list[dict[str, Any]] = []
    for row in major.itertuples(index=False):
        events.append(
            {
                "event_type": "MAJOR_DRAWDOWN",
                "start_date": pd.Timestamp(row.peak_date),
                "end_date": pd.Timestamp(row.trough_date),
                "right_censored": False,
                "mechanical_magnitude": float(row.peak_to_trough_drawdown),
                "source_peak_date": pd.Timestamp(row.peak_date),
                "source_trough_date": pd.Timestamp(row.trough_date),
                "source_recovery_date": row.recovery_date,
            }
        )
        recovery_end = (
            pd.Timestamp(row.recovery_date)
            if pd.notna(row.recovery_date)
            else pd.Timestamp(total_return["date"].max())
        )
        events.append(
            {
                "event_type": (
                    "MAJOR_RECOVERY"
                    if pd.notna(row.recovery_date)
                    else "MAJOR_RECOVERY_RIGHT_CENSORED"
                ),
                "start_date": pd.Timestamp(row.trough_date),
                "end_date": recovery_end,
                "right_censored": pd.isna(row.recovery_date),
                "mechanical_magnitude": np.nan,
                "source_peak_date": pd.Timestamp(row.peak_date),
                "source_trough_date": pd.Timestamp(row.trough_date),
                "source_recovery_date": row.recovery_date,
            }
        )
    sideways_contract = contract["long_sideways_rule"]
    sideways = detect_sideways_windows(
        total_return["date"],
        total_return["wealth"],
        lookback=int(sideways_contract["lookback_market_days"]),
        absolute_return_max=float(sideways_contract["absolute_total_return_max"]),
        range_max=float(sideways_contract["peak_to_trough_range_max"]),
    )
    events.extend(sideways.to_dict("records"))

    dates = pd.DatetimeIndex(total_return["date"])
    closes = total_return["close"].to_numpy(dtype=float)
    index_return_60: list[float] = []
    for origin in monthly_state["origin"]:
        position = int(dates.searchsorted(origin, side="right") - 1)
        index_return_60.append(
            float(closes[position] / closes[position - 60] - 1.0)
            if position >= 60
            else np.nan
        )
    monthly_state = monthly_state.copy()
    monthly_state["index_return_60d"] = index_return_60
    internal = contract["internal_deterioration_rule"]
    internal_flag = (
        monthly_state["index_return_60d"].ge(float(internal["index_return_min"]))
        & monthly_state["member_median_return_60d"].le(
            float(internal["median_member_return_max"])
        )
        & monthly_state["member_positive_breadth_60d"].le(
            float(internal["positive_member_breadth_max"])
        )
    )
    events.extend(
        _coalesce_monthly_flags(
            monthly_state,
            internal_flag,
            event_type="INDEX_UP_INTERNAL_DETERIORATION",
        )
    )
    down_contract = contract["index_down_earnings_improving_rule"]
    profit_change_3m = pd.to_numeric(
        monthly_state["weighted_operating_profit_yoy"], errors="coerce"
    ).diff(3)
    down_improving_flag = (
        monthly_state["index_return_60d"].le(
            float(down_contract["index_return_max"])
        )
        & profit_change_3m.gt(
            float(down_contract["cf_operating_profit_yoy_change_3m_min"])
        )
    )
    events.extend(
        _coalesce_monthly_flags(
            monthly_state,
            down_improving_flag,
            event_type="INDEX_DOWN_EARNINGS_IMPROVING",
        )
    )
    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        raise ValueError("机械规则未识别出任何周期事件")
    event_frame["start_date"] = _date_only(event_frame["start_date"])
    event_frame["end_date"] = _date_only(event_frame["end_date"])
    event_frame = event_frame.sort_values(
        ["start_date", "end_date", "event_type"], kind="mergesort"
    ).reset_index(drop=True)
    event_frame["event_id"] = [f"CYCLE_{index:03d}" for index in range(1, len(event_frame) + 1)]
    close_by_date = total_return.set_index("date")["close"]
    event_frame["event_total_return"] = [
        float(close_by_date.loc[end] / close_by_date.loc[start] - 1.0)
        if start in close_by_date.index and end in close_by_date.index
        else np.nan
        for start, end in zip(
            event_frame["start_date"], event_frame["end_date"], strict=True
        )
    ]
    return event_frame


def _trade_date_offset(
    dates: pd.DatetimeIndex,
    anchor: pd.Timestamp,
    offset: int,
) -> pd.Timestamp | None:
    position = int(dates.searchsorted(pd.Timestamp(anchor), side="right") - 1)
    target = position + offset
    if position < 0 or target < 0 or target >= len(dates):
        return None
    return pd.Timestamp(dates[target])


def _state_at_or_before(
    state: pd.DataFrame,
    date: pd.Timestamp | None,
) -> pd.Series | None:
    if date is None:
        return None
    eligible = state.loc[state["origin"].le(pd.Timestamp(date))]
    if eligible.empty:
        return None
    return eligible.sort_values("origin").iloc[-1]


def _classify_mechanism(
    event_type: str,
    *,
    cf_change: float,
    discount_rate_easing_change: float,
    rc_change: float,
    post_rc_change: float,
) -> str:
    values = [cf_change, discount_rate_easing_change, rc_change]
    if not all(np.isfinite(value) for value in values):
        return "MIXED_OR_NO_VIEW"
    is_recovery = "RECOVERY" in event_type or event_type in {
        "INDEX_UP_INTERNAL_DETERIORATION",
    }
    is_drawdown = event_type in {
        "MAJOR_DRAWDOWN",
        "INDEX_DOWN_EARNINGS_IMPROVING",
    }
    if is_recovery:
        if cf_change > 0:
            return "FUNDAMENTAL_EXPANSION"
        if discount_rate_easing_change > 0 and rc_change >= 0:
            return "LIQUIDITY_EXPANSION"
        if discount_rate_easing_change > 0:
            return "POLICY_OR_DISCOUNT_RATE_REPAIR"
    if is_drawdown:
        if cf_change < 0 and discount_rate_easing_change < 0:
            return "TRUE_CONTRACTION"
        if cf_change >= 0 and discount_rate_easing_change < 0:
            if np.isfinite(post_rc_change) and post_rc_change > 0:
                return "LIQUIDITY_OVERSHOOT"
            return "LATE_CYCLE_DISCOUNT_RATE_SQUEEZE"
    if cf_change > 0 and discount_rate_easing_change >= 0:
        return "FUNDAMENTAL_EXPANSION"
    if cf_change <= 0 and discount_rate_easing_change > 0 and rc_change >= 0:
        return "LIQUIDITY_EXPANSION"
    return "MIXED_OR_NO_VIEW"


def build_cycle_windows_and_classification(
    config: dict[str, Any],
    events: pd.DataFrame,
    total_return: pd.DataFrame,
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(total_return["date"])
    state_columns = {
        "cf": "cf_level",
        "dr": "equity_risk_premium_expanding_z",
        "rc": "risk_bearing_capacity_factor",
    }
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        reference_dates: dict[str, pd.Timestamp | None] = {}
        for offset in config["cycle_atlas"]["event_windows_market_days"]["pre"]:
            reference_dates[f"PRE_{offset}D"] = _trade_date_offset(
                dates, event.start_date, -int(offset)
            )
        reference_dates["EVENT_START"] = pd.Timestamp(event.start_date)
        reference_dates["EVENT_END"] = pd.Timestamp(event.end_date)
        for offset in config["cycle_atlas"]["event_windows_market_days"]["post"]:
            reference_dates[f"POST_{offset}D"] = _trade_date_offset(
                dates, event.end_date, int(offset)
            )
        state_rows: dict[str, pd.Series | None] = {}
        for label, reference_date in reference_dates.items():
            observed = _state_at_or_before(state, reference_date)
            state_rows[label] = observed
            row: dict[str, Any] = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "window_label": label,
                "target_date": reference_date,
                "state_origin": observed["origin"] if observed is not None else pd.NaT,
                "window_status": "OBSERVED" if observed is not None else "NO_VIEW",
            }
            for short, column in state_columns.items():
                row[f"{short}_state"] = (
                    float(observed[column])
                    if observed is not None and pd.notna(observed[column])
                    else np.nan
                )
            rows.append(row)
        start_state = state_rows["EVENT_START"]
        end_state = state_rows["EVENT_END"]
        post_state = state_rows.get("POST_20D")
        cf_change = (
            float(end_state[state_columns["cf"]] - start_state[state_columns["cf"]])
            if start_state is not None
            and end_state is not None
            and pd.notna(start_state[state_columns["cf"]])
            and pd.notna(end_state[state_columns["cf"]])
            else np.nan
        )
        dr_easing_change = (
            float(
                -(
                    end_state[state_columns["dr"]]
                    - start_state[state_columns["dr"]]
                )
            )
            if start_state is not None
            and end_state is not None
            and pd.notna(start_state[state_columns["dr"]])
            and pd.notna(end_state[state_columns["dr"]])
            else np.nan
        )
        rc_change = (
            float(end_state[state_columns["rc"]] - start_state[state_columns["rc"]])
            if start_state is not None
            and end_state is not None
            and pd.notna(start_state[state_columns["rc"]])
            and pd.notna(end_state[state_columns["rc"]])
            else np.nan
        )
        post_rc_change = (
            float(post_state[state_columns["rc"]] - end_state[state_columns["rc"]])
            if post_state is not None
            and end_state is not None
            and pd.notna(post_state[state_columns["rc"]])
            and pd.notna(end_state[state_columns["rc"]])
            else np.nan
        )
        summaries.append(
            {
                **event._asdict(),
                "event_cf_change": cf_change,
                "event_discount_rate_easing_change": dr_easing_change,
                "event_risk_capacity_change": rc_change,
                "post_20d_risk_capacity_change": post_rc_change,
                "mechanism_class": _classify_mechanism(
                    event.event_type,
                    cf_change=cf_change,
                    discount_rate_easing_change=dr_easing_change,
                    rc_change=rc_change,
                    post_rc_change=post_rc_change,
                ),
                "classification_method": (
                    "FIXED_SIGN_MAP_NO_NEWS_NARRATIVE_NO_PERFORMANCE_SELECTION"
                ),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def _explanation_paths(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "return_decomposition": project_path(artifacts["return_decomposition"]),
        "return_decomposition_json": project_path(
            artifacts["return_decomposition_json"]
        ),
        "cycle_atlas": project_path(artifacts["cycle_atlas"]),
        "cycle_windows": project_path(artifacts["cycle_windows"]),
        "cycle_atlas_json": project_path(artifacts["cycle_atlas_json"]),
        "final_report": project_path(artifacts["final_report"]),
        "authoritative_status": project_path(artifacts["authoritative_status"]),
    }


def _format_percent(value: Any) -> str:
    number = float(value)
    return f"{number:.2%}" if np.isfinite(number) else "NO_VIEW"


def render_final_report(
    config: dict[str, Any],
    state_receipt: dict[str, Any],
    decomposition: pd.DataFrame,
    decomposition_diagnostics: dict[str, Any],
    atlas: pd.DataFrame,
    atlas_diagnostics: dict[str, Any],
    etf_crosscheck: dict[str, Any],
) -> str:
    cf = state_receipt["diagnostics"]["cashflow"]
    pv = state_receipt["diagnostics"]["present_value"]
    rc = state_receipt["diagnostics"]["risk_capacity"]
    observed_horizons = {
        horizon: int(
            decomposition[f"forward_total_return_{horizon}d_status"]
            .eq("OBSERVED_EXPLANATORY_ONLY")
            .sum()
        )
        for horizon in config["program"]["primary_horizons_market_days"]
    }
    class_counts = atlas["mechanism_class"].value_counts().to_dict()
    event_counts = atlas["event_type"].value_counts().to_dict()
    lines = [
        "# 510300 结构性权益风险溢价引擎 V1：第一阶段报告",
        "",
        "## 结论",
        "",
        "第一阶段已经完成三张状态面板、解释性收益来源分解和机械重大周期图谱。",
        "本轮没有运行组合净值、夏普率、仓位映射、阈值搜索、Paper/Shadow 或实盘。",
        "结果状态为 `STAGE_1_COMPLETE_WITH_EXPLICIT_DATA_LIMITATIONS`；它不是可交易策略，",
        "也没有让任何旧的盈利、估值、IF、期权、宏观或技术协议重新获得历史验证资格。",
        "",
        "## 冻结边界",
        "",
        f"- 项目：`{config['program']['program_id']}`。",
        f"- 协议版本：`{config['program']['version']}`；修正："
        f"`{config['program']['protocol_revision']}`。",
        "- 本次修正发生在收益读取前，只补充已登记点时总股本回退并让市值覆盖门",
        "  正确传播；原 1.0.0 清单、已采财务收据及其哈希均被保留并绑定。",
        f"- 阶段：`{config['program']['research_stage']}`。",
        f"- 范围：{config['program']['study_start']} 至 {config['program']['market_data_cutoff']}。",
        "- 标的边界：`510300.SH_OR_CASH`；但本阶段 `MODEL_POSITION_TARGET=UNSET`。",
        "- 期限固定为 `20D / 60D / 120D`。",
        "- 机械重大回撤：完整水下周期峰谷回撤至少 15%；没有按年份或表现挑事件。",
        "- 所有状态产物先写入并哈希冻结，随后才允许读取沪深300总收益。",
        "",
        "## 三张状态面板",
        "",
        "| 产品 | 覆盖/状态 | 关键说明 |",
        "|---|---:|---|",
        (
            f"| `CSI300_PIT_CASHFLOW_STATE_PANEL_V1` | {cf['monthly_origins']} 个月度 origin | "
            f"CF 状态计数：`{json.dumps(cf['cf_status_counts'], ensure_ascii=False)}` |"
        ),
        (
            f"| `CSI300_CROSS_SECTIONAL_PRESENT_VALUE_PANEL_V1` | "
            f"{pv['portfolio_count']} 个固定组合 | PCA 状态：`{pv['pca_status']}`；"
            "没有使用收益标签 |"
        ),
        (
            f"| `CSI300_RISK_BEARING_CAPACITY_PANEL_V1` | {rc['rows']} 个日度观测 | "
            f"RC 状态计数：`{json.dumps(rc['status_counts'], ensure_ascii=False)}` |"
        ),
        "",
        "CF 六个核心量为收入同比、营业利润同比、经营现金流同比、ROE 变化、",
        "改善权重广度和恶化集中度。DR 使用行业中性 E/P、B/P、S/P、经营现金流/P",
        "及 16 个冻结组合提取无收益标签的共同因子。RC 同时保留资金利率、期限结构、",
        "信用、汇率、价格冲击、融资余额、ETF 份额、成分相关性、行业相关性和内部背离。",
        "",
        "## 数据准入与明确缺口",
        "",
        "- 成分集合使用已准入的官方点时重放，每个 origin 固定 300 只。",
        "- 财务值来自带 `NOTICE_DATE/UPDATE_DATE` 的东方财富 HSF10 二级聚合源；",
        "  `NOTICE_DATE` 决定信息可得日，`UPDATE_DATE` 仅诊断修订风险。当前接口值可能",
        "  已含后续修订，因此它不是首发值版本库，也不是完整官方原始 PDF 档案。",
        "- 收入、归母利润、权益和股本与独立本地提供方做了只读核对；经营现金流没有",
        "  第二个本地提供方，因此保留 `NO_VIEW_NO_INDEPENDENT_LOCAL_PROVIDER_CROSSCHECK`。",
        "- 历史指数权重没有逐期版本凭证。主序列使用点时总市值代理权重；未版本化的",
        "  历史指数权重只作诊断，状态仍是 `BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS`。",
        "- 月度市场截面缺总市值时，使用已登记独立提供方在 origin 前可得的总股本",
        "  与 origin 当日原始收盘价回退；市值可见成分数低于 270 时 CF 必须为 `NO_VIEW`。",
        "- 成分股点时股息率没有合格档案，保留 `NO_VIEW`；现金流久期只使用代理量。",
        "- 信用利差和部分 ETF 份额历史的发布时间没有完整版本证明，只能是",
        "  `STATE_MEASUREMENT_ONLY / DISCOVERY_EVIDENCE_ONLY`。",
        "",
        "## 解释性收益来源分解",
        "",
        f"- 完整 OLS 月份：{decomposition_diagnostics['observations']}。",
        f"- 解释性 R²：{decomposition_diagnostics['r_squared']:.4f}。",
        (
            "- 系数：`"
            + json.dumps(
                decomposition_diagnostics["coefficients"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "`。"
        ),
        f"- 可观察前向期限数：`{json.dumps(observed_horizons, ensure_ascii=False)}`。",
        "- 该分解是状态创新与月度收益的全样本解释性 OLS，不是现金流恒等式、",
        "  不是预测通过，也没有根据拟合优度选择变量或惩罚参数。",
        "",
        "## 重大周期因果图谱",
        "",
        f"- 机械事件类型计数：`{json.dumps(event_counts, ensure_ascii=False)}`。",
        f"- 固定符号映射后的机制计数：`{json.dumps(class_counts, ensure_ascii=False)}`。",
        f"- 事件窗口总行数：{atlas_diagnostics['window_rows']}。",
        "- 每个事件统一保留前 120/60/20 日、事件起点、事件终点、后 20/60 日状态；",
        "  分类只使用冻结的 CF/折现率宽松/RC 变化符号，不先写新闻叙事。",
        "",
        "## 510300 与底层指数交叉检查",
        "",
        f"- 重叠日：{etf_crosscheck['overlap_days']}。",
        f"- 510300 含分红日收益与沪深300全收益日收益相关系数：{etf_crosscheck['daily_return_correlation']:.6f}。",
        f"- 日收益差绝对值中位数：{_format_percent(etf_crosscheck['median_absolute_daily_difference'])}。",
        "- 该检查只确认解释对象与可交易 ETF 的一致性，不是组合回测。",
        "",
        "## 当前决定",
        "",
        "1. 第一阶段机制识别完成，但数据状态不是无条件 `PASS`；所有缺口已进入收据。",
        "2. 现在仍不得生成 0/25/50/75/100% 仓位，不得计算净值或夏普率。",
        "3. 下一阶段若继续，应只在已冻结的 20/60/120 日条件地图上检验四种",
        "   `CF × DR` 结构，并把 RC 作为放大/削弱项；不得从最好周期反选规则。",
        "4. 夏普率 1.2 仍是最终项目目标，不是本阶段结果；本轮没有声称目标已经实现。",
        "",
    ]
    return "\n".join(lines)


def run_explanation_stage(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state_receipt: dict[str, Any],
) -> dict[str, Any]:
    paths = _explanation_paths(config)
    if paths["authoritative_status"].is_file() and all(
        path.is_file() for path in paths.values()
    ):
        status = json.loads(paths["authoritative_status"].read_text(encoding="utf-8"))
        for name in [
            "return_decomposition",
            "return_decomposition_json",
            "cycle_atlas",
            "cycle_windows",
            "cycle_atlas_json",
            "final_report",
        ]:
            if sha256_file(paths[name]) != status["artifacts"][name]["sha256"]:
                raise ValueError(f"既有解释性产物哈希不一致：{name}")
        print("解释性分解与周期图谱已存在且哈希通过，跳过重复运行。", flush=True)
        return status
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("解释性产物不完整，禁止静默覆盖；请先人工审计残留文件")
    started_at = now_iso()
    total_return = load_csi300_total_return(config)
    etf_crosscheck = etf_total_return_crosscheck(config, total_return)
    monthly_state = _state_panel_at_origins(config, total_return)
    decomposition, decomposition_diagnostics = fit_explanatory_decomposition(
        monthly_state
    )
    events = build_mechanical_events(config, total_return, monthly_state)
    atlas, windows = build_cycle_windows_and_classification(
        config, events, total_return, monthly_state
    )
    atomic_parquet(paths["return_decomposition"], decomposition)
    decomposition_receipt = {
        "program_id": PROGRAM_ID,
        "model_id": config["return_source_decomposition"]["model_id"],
        "status": "PASS_EXPLANATORY_RETURN_SOURCE_DECOMPOSITION_ONLY",
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "state_freeze_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "state_products_verified_before_return_read": True,
        "diagnostics": decomposition_diagnostics,
        "etf_underlying_crosscheck": etf_crosscheck,
        "forward_horizon_observed_counts": {
            str(horizon): int(
                decomposition[f"forward_total_return_{horizon}d_status"]
                .eq("OBSERVED_EXPLANATORY_ONLY")
                .sum()
            )
            for horizon in config["program"]["primary_horizons_market_days"]
        },
        "artifact": {
            "path": paths["return_decomposition"].relative_to(ROOT).as_posix(),
            "rows": len(decomposition),
            "sha256": sha256_file(paths["return_decomposition"]),
        },
        "portfolio_evaluation_run": False,
        "sharpe_calculated": False,
        "position_generated": False,
    }
    atomic_json(paths["return_decomposition_json"], decomposition_receipt)
    atomic_parquet(paths["cycle_atlas"], atlas)
    atomic_parquet(paths["cycle_windows"], windows)
    atlas_diagnostics = {
        "event_count": len(atlas),
        "window_rows": len(windows),
        "event_type_counts": atlas["event_type"].value_counts().to_dict(),
        "mechanism_class_counts": atlas["mechanism_class"].value_counts().to_dict(),
        "major_drawdown_count": int(atlas["event_type"].eq("MAJOR_DRAWDOWN").sum()),
        "right_censored_count": int(atlas["right_censored"].fillna(False).sum()),
    }
    atlas_receipt = {
        "program_id": PROGRAM_ID,
        "model_id": config["cycle_atlas"]["model_id"],
        "status": "PASS_MECHANICAL_MAJOR_CYCLE_CAUSAL_ATLAS_EXPLANATORY_ONLY",
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "state_freeze_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "mechanical_rules": config["cycle_atlas"],
        "diagnostics": atlas_diagnostics,
        "artifacts": {
            "atlas": {
                "path": paths["cycle_atlas"].relative_to(ROOT).as_posix(),
                "rows": len(atlas),
                "sha256": sha256_file(paths["cycle_atlas"]),
            },
            "windows": {
                "path": paths["cycle_windows"].relative_to(ROOT).as_posix(),
                "rows": len(windows),
                "sha256": sha256_file(paths["cycle_windows"]),
            },
        },
        "manual_year_selection": False,
        "news_narrative_used_for_classification": False,
        "portfolio_evaluation_run": False,
    }
    atomic_json(paths["cycle_atlas_json"], atlas_receipt)
    report = render_final_report(
        config,
        state_receipt,
        decomposition,
        decomposition_diagnostics,
        atlas,
        atlas_diagnostics,
        etf_crosscheck,
    )
    atomic_text(paths["final_report"], report)
    artifact_names = [
        "return_decomposition",
        "return_decomposition_json",
        "cycle_atlas",
        "cycle_windows",
        "cycle_atlas_json",
        "final_report",
    ]
    status = {
        "program_id": PROGRAM_ID,
        "status": "STAGE_1_COMPLETE_WITH_EXPLICIT_DATA_LIMITATIONS",
        "research_stage": "MECHANISM_IDENTIFICATION",
        "started_at": started_at,
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "state_freeze_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "state_products": state_receipt["products"],
        "return_decomposition_status": decomposition_receipt["status"],
        "cycle_atlas_status": atlas_receipt["status"],
        "data_limitations": state_receipt["limitations"],
        "artifacts": {
            name: {
                "path": paths[name].relative_to(ROOT).as_posix(),
                "sha256": sha256_file(paths[name]),
                "bytes": paths[name].stat().st_size,
            }
            for name in artifact_names
        },
        "next_stage": "CONDITIONAL_20D_60D_120D_MECHANISM_MAP_NOT_STARTED",
        "return_evaluation": "EXPLANATORY_ONLY_NOT_PORTFOLIO",
        "portfolio_evaluation_allowed": False,
        "portfolio_evaluation_run": False,
        "nav_calculated": False,
        "sharpe_calculated": False,
        "sharpe_target_1_2_achieved": False,
        "model_position_target": "UNSET",
        "position_generated": False,
        "orders_generated": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
        "old_rejected_protocols_rescued": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 510300 结构性权益风险溢价引擎 V1"
    )
    parser.add_argument(
        "--phase",
        choices=["freeze", "collect", "build-states", "explain", "all"],
        required=True,
        help="严格分阶段运行；all 仍会依次经过所有冻结门",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"collect", "all"}:
        general_collection_receipt = collect_statements(config, manifest)
        financial_sector_collection_receipt = (
            collect_financial_sector_supplement(config, manifest)
        )
        if args.phase == "collect":
            return 0
    else:
        general_collection_receipt = verify_collection_receipt(config)
        financial_sector_collection_receipt = (
            verify_financial_sector_supplement(config, manifest)
        )
    if args.phase in {"build-states", "all"}:
        state_receipt = build_and_freeze_states(
            config,
            manifest,
            general_collection_receipt,
            financial_sector_collection_receipt,
        )
        if args.phase == "build-states":
            return 0
    else:
        state_receipt = verify_state_freeze(config)
    if args.phase in {"explain", "all"}:
        run_explanation_stage(config, manifest, state_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
