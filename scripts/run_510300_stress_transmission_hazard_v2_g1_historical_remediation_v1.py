"""冻结、采集、构建并重放 510300 V2 G1 历史证据修复 V1。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
)
from research.stress_transmission_hazard_v2 import build_daily_coverage_ledger
from research.stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    G1AdmissionArtifacts,
    build_g1_admission_artifacts,
)
from research.stress_transmission_hazard_v2_g1_historical_remediation_v1 import (
    EXECUTION_ID,
    HistoricalRemediationError,
    RemediationArtifacts,
    combine_official_intervals,
    extract_szse_month_main_urls,
    extract_szse_report_month,
    find_szse_suspension_document_url,
    parse_sse_official_response,
    parse_szse_official_document,
    remediate_official_suspensions,
    sha256_bytes,
)
from research.stress_transmission_hazard_v2_mft_features_v1 import (
    MFTFeatureArtifacts,
    build_mft_feature_artifacts,
)
from scripts.build_510300_stress_transmission_hazard_v2_four_state_ledger_v1 import (
    _write_parquet_new,
    _write_text_new,
)
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "510300_stress_transmission_hazard_v2_g1_historical_remediation_v1.yaml"
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError(f"YAML 顶层必须是对象：{path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError(f"JSON 顶层必须是对象：{path}")
    return payload


def _verify_payload_sha256(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    expected = str(payload.get(field) or "")
    if not expected:
        raise EvidenceContractError(f"{label}缺少 {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if canonical_sha256(body) != expected:
        raise EvidenceContractError(f"{label}的 {field} 不一致")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_identity(contract: Mapping[str, Any], *, label: str) -> None:
    path = _project_path(str(contract["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    actual = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    expected = {"bytes": int(contract["bytes"]), "sha256": str(contract["sha256"])}
    if actual != expected:
        raise EvidenceContractError(
            f"{label}文件身份漂移：actual={actual}, expected={expected}, path={path}"
        )


def _write_bytes_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _logical_evidence(path: Path, *, physical_path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": physical_path.stat().st_size,
        "sha256": _sha256_file(physical_path),
    }


def _frozen_file_paths(config: Mapping[str, Any]) -> list[Path]:
    freeze = config["freeze_contract"]
    relative = [
        *freeze["implementation_files"],
        *freeze["governance_files"],
    ]
    paths = [_project_path(str(value)) for value in relative]
    if len(set(paths)) != len(paths):
        raise EvidenceContractError("冻结文件列表存在重复项")
    return paths


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """冻结修复规则、代码和父文件身份；不读取标签。"""

    config = _load_yaml(config_path)
    if str(config["program"]["execution_id"]) != EXECUTION_ID:
        raise EvidenceContractError("execution_id 漂移")
    if bool(config["authorization"]["g2_authorized_in_this_execution"]):
        raise EvidenceContractError("本执行不得授权 G2")
    if config["source_contract"]["event_or_label_columns_allowed_during_collection"]:
        raise EvidenceContractError("采集阶段不得准入事件或标签列")
    for name, contract in config["immutable_parent"].items():
        _verify_identity(contract, label=f"immutable_parent.{name}")
    for name, contract in config["inputs"].items():
        if "bytes" in contract and "sha256" in contract:
            _verify_identity(contract, label=f"inputs.{name}")

    frozen_files: dict[str, dict[str, Any]] = {}
    for path in _frozen_file_paths(config):
        if not path.is_file():
            raise EvidenceContractError(f"冻结文件不存在：{path}")
        frozen_files[path.relative_to(ROOT).as_posix()] = file_evidence(
            path, project_root=ROOT
        )
    manifest_path = _project_path(config["freeze_contract"]["manifest_output"])
    receipt_path = _project_path(config["freeze_contract"]["freeze_receipt_output"])
    if manifest_path.exists() or receipt_path.exists():
        raise EvidenceContractError("冻结 manifest 或收据已存在，禁止覆盖")
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": str(config["program"]["version"]),
        "frozen_at": _now(),
        "status": "FROZEN_RESULT_BLIND_FULL_SCOPE_HISTORICAL_REMEDIATION",
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "frozen_files": frozen_files,
        "immutable_parent": config["immutable_parent"],
        "inputs": {
            key: value
            for key, value in config["inputs"].items()
            if "bytes" in value and "sha256" in value
        },
        "source_contract": config["source_contract"],
        "g1_contract": config["g1_contract"],
        "forbidden_actions": config["forbidden_actions"],
        "label_or_event_read_during_freeze": False,
        "model_trained": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": f"{EXECUTION_ID}_FREEZE_RECEIPT",
        "created_at": _now(),
        "status": "PASS_FREEZE_BEFORE_COLLECTION_AND_G1_RECALCULATION",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "label_or_event_read": False,
        "model_trained": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_path, receipt)
    print("修复协议已冻结；尚未读取 BAD10 标签或采集全量官方数据。", flush=True)
    return receipt


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    manifest_path = _project_path(config["freeze_contract"]["manifest_output"])
    manifest = _read_json(manifest_path)
    expected_payload_hash = str(manifest.pop("manifest_payload_sha256"))
    actual_payload_hash = canonical_sha256(manifest)
    manifest["manifest_payload_sha256"] = expected_payload_hash
    if actual_payload_hash != expected_payload_hash:
        raise EvidenceContractError("冻结 manifest payload 摘要不一致")
    if _sha256_file(config_path) != str(manifest["config_sha256"]):
        raise EvidenceContractError("冻结后配置文件发生漂移")
    for relative, evidence in manifest["frozen_files"].items():
        path = _project_path(relative)
        actual = file_evidence(path, project_root=ROOT)
        if actual != evidence:
            raise EvidenceContractError(f"冻结文件发生漂移：{relative}")
    for name, contract in config["immutable_parent"].items():
        _verify_identity(contract, label=f"immutable_parent.{name}")
    for name, contract in config["inputs"].items():
        if "bytes" in contract and "sha256" in contract:
            _verify_identity(contract, label=f"inputs.{name}")
    return config


def _fetch_bytes(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    referer: str,
    attempts: int = 4,
) -> tuple[str, bytes, Mapping[str, str]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Referer": referer,
        "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(10, 35))
            response.raise_for_status()
            if not response.content:
                raise HistoricalRemediationError(f"官方来源返回空内容：{response.url}")
            return response.url, response.content, dict(response.headers)
        except (requests.RequestException, HistoricalRemediationError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(float(2**attempt))
    raise HistoricalRemediationError(f"官方来源请求失败：{url}；{last_error}")


def _raw_index_row(*, logical_path: Path, physical_path: Path, url: str, retrieved_at: str) -> dict[str, Any]:
    return {
        "path": logical_path.relative_to(ROOT).as_posix(),
        "bytes": int(physical_path.stat().st_size),
        "sha256": _sha256_file(physical_path),
        "source_url": url,
        "retrieved_at": retrieved_at,
    }


def _collect_sse(
    *,
    config: Mapping[str, Any],
    temp_root: Path,
    final_root: Path,
    retrieved_at: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    contract = config["source_contract"]["sse"]
    start = pd.Timestamp(contract["query_start"])
    end = pd.Timestamp(contract["query_end"])
    page_size = int(contract["page_size"])
    frames: list[pd.DataFrame] = []
    raw_rows: list[dict[str, Any]] = []
    for year in range(start.year, end.year + 1):
        window_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        window_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        page_no = 1
        page_count = 1
        while page_no <= page_count:
            params = {
                "isPagination": "true",
                "sqlId": str(contract["sql_id"]),
                "pageHelp.pageSize": page_size,
                "pageHelp.pageNo": page_no,
                "productCode": "",
                "keyWords": "",
                "startStopDate": window_start.strftime("%Y%m%d"),
                "endStopDate": window_end.strftime("%Y%m%d"),
            }
            response_url, content, _headers = _fetch_bytes(
                str(contract["endpoint"]),
                params=params,
                referer=str(contract["required_referer"]),
            )
            relative = Path("sse") / f"{year}_page_{page_no:04d}.json"
            physical = temp_root / relative
            logical = final_root / relative
            _write_bytes_new(physical, content)
            raw_rows.append(
                _raw_index_row(
                    logical_path=logical,
                    physical_path=physical,
                    url=response_url,
                    retrieved_at=retrieved_at,
                )
            )
            frames.append(
                parse_sse_official_response(
                    content,
                    source_url=response_url,
                    retrieved_at=retrieved_at,
                )
            )
            payload = json.loads(content.decode("utf-8-sig"))
            help_object = payload.get("pageHelp") or {}
            page_count = int(help_object.get("pageCount") or 1)
            if page_count < 1 or page_count > 1000:
                raise HistoricalRemediationError(
                    f"上交所分页计数非法：year={year}, page_count={page_count}"
                )
            page_no += 1
        print(f"上交所 {year} 年官方停复牌页已完成。", flush=True)
    return frames, raw_rows


def _parallel_fetch(
    urls: Iterable[str],
    *,
    referer: str,
    workers: int = 4,
) -> dict[str, tuple[str, bytes, Mapping[str, str]]]:
    ordered = sorted(set(urls))
    result: dict[str, tuple[str, bytes, Mapping[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_bytes, url, referer=referer): url for url in ordered
        }
        for future in as_completed(futures):
            requested = futures[future]
            result[requested] = future.result()
    return result


def _collect_szse(
    *,
    config: Mapping[str, Any],
    temp_root: Path,
    final_root: Path,
    retrieved_at: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    contract = config["source_contract"]["szse"]
    index_url = str(contract["official_month_index"])
    resolved_index_url, index_content, _headers = _fetch_bytes(
        index_url, referer=index_url
    )
    index_physical = temp_root / "szse" / "month_index.html"
    index_logical = final_root / "szse" / "month_index.html"
    _write_bytes_new(index_physical, index_content)
    raw_rows = [
        _raw_index_row(
            logical_path=index_logical,
            physical_path=index_physical,
            url=resolved_index_url,
            retrieved_at=retrieved_at,
        )
    ]
    all_main_urls = extract_szse_month_main_urls(index_content, index_url=index_url)
    candidate_urls = []
    for url in all_main_urls:
        match = re.search(r"/t(\d{8})_\d+\.html", url)
        if match and "20150101" <= match.group(1) <= "20260930":
            candidate_urls.append(url)
    main_responses = _parallel_fetch(candidate_urls, referer=index_url)
    first_month = str(contract["first_report_month"])
    last_month = str(contract["last_report_month"])
    selected: dict[str, tuple[str, bytes, str]] = {}
    for requested, (response_url, content, _response_headers) in main_responses.items():
        report_month = extract_szse_report_month(content)
        if first_month <= report_month <= last_month:
            if report_month in selected:
                raise HistoricalRemediationError(f"深交所月报月份重复：{report_month}")
            selected[report_month] = (response_url, content, requested)
    expected_months = pd.period_range(first_month, last_month, freq="M").astype(str).tolist()
    missing_months = sorted(set(expected_months).difference(selected))
    if missing_months:
        raise HistoricalRemediationError(f"深交所月报索引缺月：{missing_months}")

    document_urls: dict[str, str] = {}
    for report_month in expected_months:
        response_url, content, _requested = selected[report_month]
        main_name = Path(urlparse(response_url).path).name
        relative = Path("szse") / "month_main" / f"{report_month}_{main_name}"
        physical = temp_root / relative
        logical = final_root / relative
        _write_bytes_new(physical, content)
        raw_rows.append(
            _raw_index_row(
                logical_path=logical,
                physical_path=physical,
                url=response_url,
                retrieved_at=retrieved_at,
            )
        )
        document_urls[report_month] = find_szse_suspension_document_url(
            content, main_url=response_url
        )
    document_responses = _parallel_fetch(
        document_urls.values(), referer=index_url
    )
    frames: list[pd.DataFrame] = []
    for report_month in expected_months:
        requested = document_urls[report_month]
        response_url, content, _response_headers = document_responses[requested]
        name = Path(urlparse(response_url).path).name
        relative = Path("szse") / "suspension_tables" / f"{report_month}_{name}"
        physical = temp_root / relative
        logical = final_root / relative
        _write_bytes_new(physical, content)
        raw_rows.append(
            _raw_index_row(
                logical_path=logical,
                physical_path=physical,
                url=response_url,
                retrieved_at=retrieved_at,
            )
        )
        frames.append(
            parse_szse_official_document(
                content,
                source_url=response_url,
                report_month=report_month,
                retrieved_at=retrieved_at,
            )
        )
        if report_month.endswith("-12") or report_month == last_month:
            print(f"深交所官方月报已完成至 {report_month}。", flush=True)
    return frames, raw_rows


def collect(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """在冻结后全量采集交易所官方停牌历史；不读取事件或标签。"""

    config = verify_frozen_manifest(config_path)
    final_root = _project_path(config["outputs"]["raw_root"])
    if final_root.exists():
        raise EvidenceContractError(f"原始来源目录已存在，禁止覆盖：{final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = final_root.parent / f"{final_root.name}_INCOMPLETE_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=False, exist_ok=False)
    retrieved_at = _now()
    print("开始全量采集交易所官方停复牌历史；采集器未加载 BAD10 文件。", flush=True)
    sse_frames, sse_raw = _collect_sse(
        config=config,
        temp_root=temp_root,
        final_root=final_root,
        retrieved_at=retrieved_at,
    )
    szse_frames, szse_raw = _collect_szse(
        config=config,
        temp_root=temp_root,
        final_root=final_root,
        retrieved_at=retrieved_at,
    )
    intervals = combine_official_intervals([*sse_frames, *szse_frames])
    cutoff = pd.Timestamp(config["program"]["historical_cutoff"])
    intervals = intervals.loc[intervals["start_at"].dt.normalize().le(cutoff)].reset_index(drop=True)
    official_target = _project_path(config["outputs"]["official_intervals"])
    official_temp = temp_root / official_target.relative_to(final_root)
    _write_parquet_new(intervals, official_temp)

    raw_rows = [*sse_raw, *szse_raw]
    raw_index = pd.DataFrame(raw_rows).sort_values("path", kind="stable").reset_index(drop=True)
    raw_index_target = _project_path(config["outputs"]["raw_file_index"])
    raw_index_temp = temp_root / raw_index_target.relative_to(final_root)
    _write_parquet_new(raw_index, raw_index_temp)
    source_counts = intervals.groupby("source_kind").size().astype(int).to_dict()
    receipt_target = _project_path(config["outputs"]["collection_receipt"])
    receipt_temp = temp_root / receipt_target.relative_to(final_root)
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_COLLECTION_RECEIPT",
        "created_at": _now(),
        "retrieved_at": retrieved_at,
        "status": "PASS_FULL_SCOPE_EXCHANGE_OFFICIAL_SUSPENSION_COLLECTION",
        "collection_scope": config["source_contract"]["acquisition_scope"],
        "event_or_label_file_read": False,
        "raw_response_file_count": int(len(raw_index)),
        "raw_response_total_bytes": int(raw_index["bytes"].sum()),
        "official_interval_record_count": int(len(intervals)),
        "official_interval_source_counts": source_counts,
        "official_intervals": _logical_evidence(
            official_target, physical_path=official_temp
        ),
        "raw_file_index": _logical_evidence(
            raw_index_target, physical_path=raw_index_temp
        ),
        "historical_cutoff": str(config["program"]["historical_cutoff"]),
        "model_trained": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_temp, receipt)
    os.replace(temp_root, final_root)
    if not final_root.is_dir():
        raise EvidenceContractError("官方来源临时目录未能原子提交")
    print(
        f"官方停复牌采集完成：原始响应 {len(raw_index)} 个，规范区间 {len(intervals)} 条。",
        flush=True,
    )
    return receipt


def _verify_collection(config: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = _project_path(config["outputs"]["collection_receipt"])
    receipt = _read_json(receipt_path)
    _verify_payload_sha256(
        receipt, field="receipt_payload_sha256", label="官方来源采集收据"
    )
    if receipt.get("status") != "PASS_FULL_SCOPE_EXCHANGE_OFFICIAL_SUSPENSION_COLLECTION":
        raise EvidenceContractError("官方来源采集收据状态不通过")
    if bool(receipt.get("event_or_label_file_read")):
        raise EvidenceContractError("采集收据显示事件或标签曾被读取")
    for key in ("official_intervals", "raw_file_index"):
        evidence = receipt[key]
        actual = file_evidence(_project_path(evidence["path"]), project_root=ROOT)
        if actual != evidence:
            raise EvidenceContractError(f"采集输出身份漂移：{key}")
    raw_index = pd.read_parquet(_project_path(receipt["raw_file_index"]["path"]))
    required = {"path", "bytes", "sha256", "source_url", "retrieved_at"}
    missing = sorted(required.difference(raw_index.columns))
    if missing:
        raise EvidenceContractError(f"原始响应索引缺列：{missing}")
    if raw_index["path"].duplicated().any():
        raise EvidenceContractError("原始响应索引存在重复路径")
    if len(raw_index) != int(receipt["raw_response_file_count"]):
        raise EvidenceContractError("原始响应索引文件数与采集收据不一致")
    if int(pd.to_numeric(raw_index["bytes"], errors="raise").sum()) != int(
        receipt["raw_response_total_bytes"]
    ):
        raise EvidenceContractError("原始响应索引总字节数与采集收据不一致")
    for row in raw_index.itertuples(index=False):
        path = _project_path(str(row.path))
        actual = file_evidence(path, project_root=ROOT)
        expected = {
            "path": str(row.path),
            "bytes": int(row.bytes),
            "sha256": str(row.sha256),
        }
        if actual != expected:
            raise EvidenceContractError(f"原始官方响应身份漂移：{row.path}")
    intervals = pd.read_parquet(_project_path(receipt["official_intervals"]["path"]))
    if len(intervals) != int(receipt["official_interval_record_count"]):
        raise EvidenceContractError("官方区间行数与采集收据不一致")
    return receipt


def _verify_nested_input_contracts(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mft_config = _load_yaml(_project_path(config["inputs"]["mft_config"]["path"]))
    g1_config = _load_yaml(_project_path(config["inputs"]["g1_config"]["path"]))
    for name, contract in mft_config["inputs"].items():
        if name in {"classified_constituent_returns", "four_state_daily_coverage"}:
            continue
        if "bytes" in contract and "sha256" in contract:
            _verify_identity(contract, label=f"mft.inputs.{name}")
    for name, contract in g1_config["inputs"].items():
        if name == "mft_feature_panel":
            continue
        if "bytes" in contract and "sha256" in contract:
            _verify_identity(contract, label=f"g1.inputs.{name}")
    gate = g1_config["g1_gate_contract"]
    expected = config["g1_contract"]
    if int(gate["mechanism_discovery"]["minimum_independent_events"]) != int(
        expected["mechanism_minimum_independent_events"]
    ):
        raise EvidenceContractError("G1 机制事件门发生漂移")
    if int(gate["full_three_coefficient_model"]["minimum_independent_events"]) != int(
        expected["full_model_minimum_independent_events"]
    ):
        raise EvidenceContractError("G1 完整模型事件门发生漂移")
    if int(gate["mechanism_discovery"]["minimum_non_event_risk_days"]) != int(
        expected["minimum_non_event_risk_days"]
    ):
        raise EvidenceContractError("G1 非事件风险日门发生漂移")
    if int(gate["full_three_coefficient_model"]["minimum_non_event_risk_days"]) != int(
        expected["minimum_non_event_risk_days"]
    ):
        raise EvidenceContractError("G1 完整模型非事件风险日门发生漂移")
    feature = mft_config["feature_contract"]
    actual_windows = [
        int(feature["lookback_market_days"]),
        int(feature["change_market_days"]),
        int(feature["tail_volatility_market_days"]),
    ]
    if actual_windows != [int(value) for value in expected["windows_unchanged"]]:
        raise EvidenceContractError("M/F/T 20/5/60 窗口发生漂移")
    if float(feature["member_coverage_minimum"]) != float(
        expected["member_coverage_minimum"]
    ):
        raise EvidenceContractError("M/F/T 成员覆盖门发生漂移")
    if float(feature["comovement_member_ratio_minimum"]) != float(
        expected["comovement_member_ratio_minimum"]
    ):
        raise EvidenceContractError("M/F/T 共同运动成员比例门发生漂移")
    required_assets = ["510300.SH", "CASH_CNY"]
    if list(mft_config["program"]["executable_assets"]) != required_assets:
        raise EvidenceContractError("M/F/T 可执行资产范围发生漂移")
    if list(g1_config["program"]["executable_assets"]) != required_assets:
        raise EvidenceContractError("G1 可执行资产范围发生漂移")
    return mft_config, g1_config


def _build_mft(
    *,
    mft_config: Mapping[str, Any],
    classified: pd.DataFrame,
    coverage: pd.DataFrame,
) -> MFTFeatureArtifacts:
    inputs = mft_config["inputs"]
    membership = pd.read_parquet(_project_path(inputs["point_in_time_membership"]["path"]))
    h00300 = pd.read_parquet(_project_path(inputs["h00300_total_return_close"]["path"]))
    pe = pd.read_parquet(_project_path(inputs["csi300_official_pe"]["path"]))
    bond = pd.read_parquet(_project_path(inputs["china_10y_yield"]["path"]))
    dr007 = pd.read_parquet(_project_path(inputs["dr007"]["path"]))
    policy = pd.read_parquet(_project_path(inputs["reverse_repo_7d_policy_rate"]["path"]))
    source_sha256 = {
        "pe": str(inputs["csi300_official_pe"]["sha256"]),
        "bond": str(inputs["china_10y_yield"]["sha256"]),
        "dr007": str(inputs["dr007"]["sha256"]),
        "policy": str(inputs["reverse_repo_7d_policy_rate"]["sha256"]),
    }
    return build_mft_feature_artifacts(
        membership=membership,
        classified_returns=classified,
        four_state_daily_coverage=coverage,
        h00300_total_return_close=h00300,
        pe=pe,
        bond=bond,
        dr007=dr007,
        reverse_repo_7d=policy,
        source_sha256=source_sha256,
    )


def _build_g1(
    *,
    g1_config: Mapping[str, Any],
    mft_panel: pd.DataFrame,
) -> G1AdmissionArtifacts:
    inputs = g1_config["inputs"]
    etf_contract = inputs["etf_unadjusted_daily"]
    etf = pd.read_parquet(
        _project_path(etf_contract["path"]),
        columns=[
            str(etf_contract["date_column"]),
            "symbol",
            str(etf_contract["close_column"]),
        ],
    )
    dividend_contract = inputs["etf_cash_dividends"]
    dividends = pd.read_csv(
        _project_path(dividend_contract["path"]),
        usecols=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
        ],
    )
    print("来源修复和 M/F/T 已全量重建；现在才读取 BAD10 最小必要列复裁 G1。", flush=True)
    events = pd.read_csv(
        _project_path(inputs["bad10_events"]["path"]),
        usecols=[str(column) for column in inputs["bad10_events"]["read_columns"]],
    )
    samples = pd.read_csv(
        _project_path(inputs["bad10_samples"]["path"]),
        usecols=[str(column) for column in inputs["bad10_samples"]["read_columns"]],
    )
    return build_g1_admission_artifacts(
        etf_daily=etf,
        dividends=dividends,
        mft=mft_panel[["date", "F", "M", "T", "B2_feature_state", "B3_feature_state"]],
        samples=samples,
        events=events,
        config=g1_config,
    )


def _event_era_distribution(g1: G1AdmissionArtifacts) -> dict[str, dict[str, int]]:
    samples = g1.sample_eligibility_ledger
    positive = samples.loc[samples["bad10"].eq(1), ["event_id", "origin_date"]].copy()
    starts = positive.groupby("event_id")["origin_date"].min()
    events = g1.event_eligibility_ledger.set_index("event_id")
    result: dict[str, dict[str, int]] = {}
    for label, start, end in (
        ("2015-2017", 2015, 2017),
        ("2018-2020", 2018, 2020),
        ("2021-2023", 2021, 2023),
        ("2024-2026", 2024, 2026),
    ):
        ids = starts.loc[starts.dt.year.between(start, end)].index
        subset = events.reindex(ids)
        result[label] = {
            "total_events": int(len(ids)),
            "b2_identifiable_events": int(subset["b2_identifiable_event"].fillna(False).sum()),
            "b3_identifiable_events": int(subset["b3_identifiable_event"].fillna(False).sum()),
        }
    return result


def _research_frames(
    config: Mapping[str, Any],
) -> tuple[RemediationArtifacts, pd.DataFrame, MFTFeatureArtifacts, G1AdmissionArtifacts, dict[str, Any]]:
    _verify_collection(config)
    mft_config, g1_config = _verify_nested_input_contracts(config)
    classified = pd.read_parquet(
        _project_path(config["inputs"]["classified_constituent_returns"]["path"])
    )
    membership = pd.read_parquet(
        _project_path(config["inputs"]["point_in_time_membership"]["path"])
    )
    dividends = pd.read_parquet(
        _project_path(config["inputs"]["dividend_action_candidates"]["path"])
    )
    intervals = pd.read_parquet(_project_path(config["outputs"]["official_intervals"]))
    remediation = remediate_official_suspensions(
        classified_returns=classified,
        membership=membership,
        dividend_actions=dividends,
        official_intervals=intervals,
        dividend_source_sha256=str(
            config["inputs"]["dividend_action_candidates"]["sha256"]
        ),
        historical_cutoff=str(config["program"]["historical_cutoff"]),
        expected_target_rows=int(config["source_contract"]["expected_target_member_days"]),
    )
    if remediation.metrics["target_symbol_count"] != int(
        config["source_contract"]["expected_target_symbols"]
    ):
        raise HistoricalRemediationError("全量修复目标证券数漂移")
    member_input = membership.rename(columns={"membership_date": "date"})
    coverage = build_daily_coverage_ledger(
        membership=member_input,
        classified_returns=remediation.remediated_classified_returns,
        minimum_member_coverage=float(config["g1_contract"]["member_coverage_minimum"]),
        reliable_point_in_time_weights=False,
        expected_members_per_day=300,
    )
    mft = _build_mft(
        mft_config=mft_config,
        classified=remediation.remediated_classified_returns,
        coverage=coverage,
    )
    g1 = _build_g1(g1_config=g1_config, mft_panel=mft.mft_feature_panel)
    metrics = {
        "remediation": remediation.metrics,
        "mft": {
            "market_day_count": int(len(mft.mft_feature_panel)),
            "b2_view_day_count": int(mft.mft_feature_panel["B2_feature_state"].eq("VIEW_ALLOWED").sum()),
            "b3_view_day_count": int(mft.mft_feature_panel["B3_feature_state"].eq("VIEW_ALLOWED").sum()),
        },
        "g1": g1.metrics,
        "event_era_distribution": _event_era_distribution(g1),
    }
    return remediation, coverage, mft, g1, metrics


def _report_text(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    rem = metrics["remediation"]
    g1 = metrics["g1"]
    eras = metrics["event_era_distribution"]
    era_lines = [
        f"| {era} | {row['total_events']} | {row['b2_identifiable_events']} | {row['b3_identifiable_events']} |"
        for era, row in eras.items()
    ]
    passed = bool(gate["full_three_coefficient_model_prerequisite_passed"])
    recalculated_state = (
        "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
        if passed
        else str(gate["G1_DATA_AND_EVENTS"])
    )
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2：G1 历史证据修复 V1",
            "",
            f"G1 复算裁决（尚待独立新进程重放）：`{recalculated_state}`。",
            "",
            "## 全量、结果盲修复",
            "",
            f"- 父版本未观测点时成员日：{rem['target_member_day_count']:,}",
            f"- 交易所官方区间命中：{rem['official_interval_matched_member_day_count']:,}",
            f"- 严格升级为 `OFFICIAL_SUSPENSION`：{rem['promoted_official_suspension_member_day_count']:,}",
            f"- 当日公司行动候选阻断：{rem['blocked_by_action_candidate_count']:,}",
            f"- 未命中官方整日区间：{rem['unmatched_member_day_count']:,}",
            "- 采集和逐行修复未读取事件 ID 或标签，未按压力事件日期选择数据。",
            "",
            "## G1 复裁",
            "",
            f"- B2 共同样本日：{g1['b2_common_sample_day_count']:,}",
            f"- B3 共同样本日：{g1['b3_common_sample_day_count']:,}",
            f"- B2 可识别独立事件：{g1['b2_identifiable_event_count']}",
            f"- B3 可识别独立事件：{g1['b3_identifiable_event_count']}",
            f"- B2 非事件风险日：{g1['b2_eligible_non_event_risk_day_count']:,}",
            f"- B3 非事件风险日：{g1['b3_eligible_non_event_risk_day_count']:,}",
            "",
            "| 冻结时代 | 全部事件 | B2可识别 | B3可识别 |",
            "| --- | ---: | ---: | ---: |",
            *era_lines,
            "",
            "## 权限边界",
            "",
            "本执行只复裁 G1。没有拟合 B0/B1/B2/B3，没有生成概率、警报、组合收益、夏普、仓位或订单。即使 G1 通过，也停在 `G1_PASS_AWAIT_EXPLICIT_G2_AUTHORIZATION`。",
            "",
        ]
    )


def _output_frames(
    remediation: RemediationArtifacts,
    coverage: pd.DataFrame,
    mft: MFTFeatureArtifacts,
    g1: G1AdmissionArtifacts,
) -> dict[str, pd.DataFrame]:
    return {
        "target_evidence": remediation.target_evidence,
        "remediated_classified_returns": remediation.remediated_classified_returns,
        "four_state_daily_coverage": coverage,
        "internal_features": mft.internal_features,
        "member_history_ledger": mft.member_history_ledger,
        "mft_feature_panel": mft.mft_feature_panel,
        "mft_coverage": mft.coverage_ledger,
        "g1_sample_eligibility": g1.sample_eligibility_ledger,
        "g1_event_eligibility": g1.event_eligibility_ledger,
    }


def build(config_path: Path = DEFAULT_CONFIG, *, dry_run: bool = False) -> dict[str, Any]:
    """全量重建四态、M/F/T 和 G1；永不自动运行 G2。"""

    config = verify_frozen_manifest(config_path)
    remediation, coverage, mft, g1, metrics = _research_frames(config)
    if dry_run:
        result = {
            "status": "PASS_IN_MEMORY_DRY_RUN_NO_OUTPUT_WRITTEN",
            "metrics": metrics,
            "gate_result": g1.gate_result,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        return result

    outputs = config["outputs"]
    frame_map = _output_frames(remediation, coverage, mft, g1)
    curated_root = _project_path(outputs["curated_root"])
    owned = [
        curated_root,
        _project_path(outputs["report"]),
        _project_path(outputs["build_receipt"]),
    ]
    existing = [str(path) for path in owned if path.exists()]
    if existing:
        raise EvidenceContractError(f"修复构建输出已存在，禁止覆盖：{existing}")

    curated_root.parent.mkdir(parents=True, exist_ok=True)
    temp_curated_root = (
        curated_root.parent / f"{curated_root.name}_INCOMPLETE_{uuid.uuid4().hex}"
    )
    temp_curated_root.mkdir(parents=False, exist_ok=False)
    for name, frame in frame_map.items():
        logical_path = _project_path(outputs[name])
        try:
            relative = logical_path.relative_to(curated_root)
        except ValueError as exc:
            raise EvidenceContractError(
                f"表输出不在冻结 curated_root 内：{logical_path}"
            ) from exc
        physical_path = temp_curated_root / relative
        _write_parquet_new(frame, physical_path)
        persisted = pd.read_parquet(physical_path)
        try:
            pd.testing.assert_frame_equal(
                persisted.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError as exc:
            raise EvidenceContractError(f"表写后逐值不一致：{name}：{exc}") from exc
    os.replace(temp_curated_root, curated_root)
    if not curated_root.is_dir():
        raise EvidenceContractError("修复表临时目录未能原子提交")

    frame_evidence: dict[str, dict[str, Any]] = {}
    for name in frame_map:
        path = _project_path(outputs[name])
        persisted = pd.read_parquet(path)
        frame_evidence[name] = {
            **file_evidence(path, project_root=ROOT),
            "row_count": int(len(persisted)),
            "semantic_sha256": frame_semantic_sha256(persisted),
        }
    report_path = _project_path(outputs["report"])
    _write_text_new(_report_text(metrics, g1.gate_result), report_path)
    full_pass = bool(g1.gate_result["full_three_coefficient_model_prerequisite_passed"])
    g1_state = (
        "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
        if full_pass
        else str(g1.gate_result["G1_DATA_AND_EVENTS"])
    )
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_BUILD_RECEIPT",
        "created_at": _now(),
        "status": "PASS_G1_REMEDIATION_BUILD_PENDING_FRESH_PROCESS_REPLAY",
        "frozen_manifest": file_evidence(
            _project_path(config["freeze_contract"]["manifest_output"]), project_root=ROOT
        ),
        "collection_receipt": file_evidence(
            _project_path(outputs["collection_receipt"]), project_root=ROOT
        ),
        "outputs": {**frame_evidence, "report": file_evidence(report_path, project_root=ROOT)},
        "metrics": metrics,
        "g1_gate_result": {**g1.gate_result, "G1_DATA_AND_EVENTS": g1_state},
        "g2_run": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    receipt_path = _project_path(outputs["build_receipt"])
    atomic_write_json_new(receipt_path, receipt)
    print(
        f"G1 修复构建完成、待独立新进程重放：B2={g1.metrics['b2_identifiable_event_count']}，"
        f"B3={g1.metrics['b3_identifiable_event_count']}，state={g1_state}；G2 未运行。",
        flush=True,
    )
    return receipt


def replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """在新进程内重算并核对所有持久化表。"""

    config = verify_frozen_manifest(config_path)
    outputs = config["outputs"]
    replay_path = _project_path(outputs["replay_receipt"])
    status_path = _project_path(outputs["status"])
    if replay_path.exists() or status_path.exists():
        raise EvidenceContractError(
            f"重放收据或最终状态已存在，禁止覆盖：{replay_path}, {status_path}"
        )
    build_receipt = _read_json(_project_path(outputs["build_receipt"]))
    _verify_payload_sha256(
        build_receipt, field="receipt_payload_sha256", label="G1 修复构建收据"
    )
    remediation, coverage, mft, g1, metrics = _research_frames(config)
    frame_map = _output_frames(remediation, coverage, mft, g1)
    comparisons: dict[str, Any] = {}
    for name, rebuilt in frame_map.items():
        path = _project_path(outputs[name])
        persisted = pd.read_parquet(path)
        persisted_hash = frame_semantic_sha256(persisted)
        expected_hash = str(build_receipt["outputs"][name]["semantic_sha256"])
        persisted_identity = file_evidence(path, project_root=ROOT)
        expected_identity = {
            key: build_receipt["outputs"][name][key]
            for key in ("path", "bytes", "sha256")
        }
        if persisted_identity != expected_identity or persisted_hash != expected_hash:
            raise EvidenceContractError(f"持久化表身份或语义漂移：{name}")
        try:
            pd.testing.assert_frame_equal(
                persisted.reset_index(drop=True),
                rebuilt.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError as exc:
            raise EvidenceContractError(f"新进程重放逐值不一致：{name}：{exc}") from exc
        comparisons[name] = {
            "rebuilt_in_memory_semantic_sha256": frame_semantic_sha256(rebuilt),
            "persisted_semantic_sha256": persisted_hash,
            "expected_semantic_sha256": expected_hash,
            "persisted_file_identity_passed": True,
            "frame_equal_ignoring_storage_dtype_representation": True,
        }
    if canonical_sha256(metrics) != canonical_sha256(build_receipt["metrics"]):
        raise EvidenceContractError("新进程重放指标与构建收据不一致")
    recalculated_gate = {**g1.gate_result}
    full_pass = bool(
        recalculated_gate["full_three_coefficient_model_prerequisite_passed"]
    )
    g1_state = (
        "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
        if full_pass
        else str(recalculated_gate["G1_DATA_AND_EVENTS"])
    )
    recalculated_gate["G1_DATA_AND_EVENTS"] = g1_state
    if canonical_sha256(recalculated_gate) != canonical_sha256(
        build_receipt["g1_gate_result"]
    ):
        raise EvidenceContractError("新进程重放 G1 门结果与构建收据不一致")
    report_path = _project_path(outputs["report"])
    report_identity = file_evidence(report_path, project_root=ROOT)
    if report_identity != build_receipt["outputs"]["report"]:
        raise EvidenceContractError("G1 修复报告身份漂移")
    if report_path.read_text(encoding="utf-8") != _report_text(metrics, g1.gate_result):
        raise EvidenceContractError("G1 修复报告与新进程重放内容不一致")
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_REPLAY_RECEIPT",
        "created_at": _now(),
        "status": "PASS_FRESH_PROCESS_FULL_SCOPE_REMEDIATION_AND_G1_REPLAY",
        "comparisons": comparisons,
        "metrics": metrics,
        "g1_gate_result": build_receipt["g1_gate_result"],
        "g2_run": False,
        "model_trained": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(replay_path, receipt)
    status = {
        "program_id": config["program"]["program_id"],
        "execution_id": EXECUTION_ID,
        "updated_at": _now(),
        "historical_remediation": "PASS_FULL_SCOPE_RESULT_BLIND_SOURCE_REMEDIATION",
        "G1_DATA_AND_EVENTS": g1_state,
        "mechanism_discovery_prerequisite_passed": bool(
            g1.gate_result["mechanism_discovery_prerequisite_passed"]
        ),
        "full_three_coefficient_model_prerequisite_passed": full_pass,
        "b2_identifiable_event_count": int(g1.metrics["b2_identifiable_event_count"]),
        "b3_identifiable_event_count": int(g1.metrics["b3_identifiable_event_count"]),
        "b2_eligible_non_event_risk_day_count": int(
            g1.metrics["b2_eligible_non_event_risk_day_count"]
        ),
        "b3_eligible_non_event_risk_day_count": int(
            g1.metrics["b3_eligible_non_event_risk_day_count"]
        ),
        "next_allowed_step": (
            str(config["g1_contract"]["pass_next_step"])
            if full_pass
            else str(config["g1_contract"]["fail_next_step"])
        ),
        "G2": "NOT_RUN_REQUIRES_NEW_EXPLICIT_AUTHORIZATION",
        "model_trained": False,
        "current_market_probability": "NOT_GENERATED",
        "position_target": "UNSET",
        "position_impact": 0,
        "build_receipt": file_evidence(
            _project_path(outputs["build_receipt"]), project_root=ROOT
        ),
        "fresh_process_replay": "PASS",
        "replay_receipt": file_evidence(replay_path, project_root=ROOT),
    }
    atomic_write_json_new(status_path, status)
    print(
        f"全新进程重放通过；最终 G1={g1_state}，G2 未运行。", flush=True
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "collect", "build", "replay"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="仅供 build 做内存预演")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            if args.dry_run:
                raise EvidenceContractError("freeze 不支持 --dry-run")
            freeze(args.config)
        elif args.command == "collect":
            if args.dry_run:
                raise EvidenceContractError("collect 不支持 --dry-run")
            collect(args.config)
        elif args.command == "build":
            build(args.config, dry_run=args.dry_run)
        else:
            if args.dry_run:
                raise EvidenceContractError("replay 不支持 --dry-run")
            replay(args.config)
        return 0
    except (
        EvidenceContractError,
        HistoricalRemediationError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        requests.RequestException,
    ) as exc:
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
