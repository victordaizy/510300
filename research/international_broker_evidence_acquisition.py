"""国际化券商V1.1官方披露原文采集、哈希和母样本核验。

V1冻结文件只读。本模块将V1.1证据写入独立目录，不计算股票收益、
不映射仓位，也不连接券商。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests
import yaml

from research.international_broker_data_contract import (
    audit_evidence_registry,
    audit_universe,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_CONFIG_FILE = ROOT / "config" / "international_broker_evidence_v1_1.yaml"
USER_AGENT = "Mozilla/5.0 international-broker-evidence-audit/1.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_evidence_config(path: Path = EVIDENCE_CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_frozen_v1(
    root: Path = ROOT, evidence_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = evidence_config or load_evidence_config()
    manifest_path = root / config["protocol"]["base_frozen_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str | None]] = []
    for relative, expected in manifest["frozen_files"].items():
        path = root / relative
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            mismatches.append(
                {"file": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
    return {
        "status": "PASS" if not mismatches else "BLOCKED_FROZEN_V1_DRIFT",
        "manifest": manifest_path.relative_to(root).as_posix(),
        "checked_file_count": len(manifest["frozen_files"]),
        "mismatches": mismatches,
    }


def build_merged_contract(
    root: Path = ROOT, evidence_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = evidence_config or load_evidence_config()
    base_path = root / config["protocol"]["base_config"]
    merged = copy.deepcopy(yaml.safe_load(base_path.read_text(encoding="utf-8")))
    merged["protocol"]["version"] = config["protocol"]["evidence_version"]
    merged["protocol"]["state"] = config["protocol"]["state"]
    merged["protocol"]["as_of_date"] = config["protocol"]["as_of_date"]
    merged["universe"]["file"] = config["paths"]["verified_universe"]
    merged["evidence_registry"]["file"] = config["paths"][
        "verified_source_registry"
    ]
    accepted = set(merged["source_policy"]["accepted_official_domains"])
    accepted.update(config["validation"]["accepted_disclosure_domains"])
    merged["source_policy"]["accepted_official_domains"] = sorted(accepted)
    return merged


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Referer": "http://www.cninfo.com.cn/"})
    return session


def _request_bytes(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    maximum_attempts: int,
) -> tuple[bytes, str, str]:
    final_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            content = response.content
            if len(content) < 100:
                raise RuntimeError(f"响应过小：{len(content)}字节")
            return content, response.headers.get("Content-Type", ""), response.url
        except Exception as error:
            final_error = error
            if attempt < maximum_attempts:
                time.sleep(attempt)
    raise RuntimeError(f"下载失败：{url}；{final_error}")


def select_full_annual_report(
    announcements: list[dict[str, Any]], report_year: int
) -> dict[str, Any] | None:
    targets = (f"{report_year}年年度报告", f"{report_year}年度报告")
    excluded = ("摘要", "英文版", "取消", "更正公告", "提示性公告")
    candidates = [
        item
        for item in announcements
        if any(
            target
            in re.sub(r"<[^>]+>", "", str(item.get("announcementTitle", "")))
            for target in targets
        )
        and not any(
            token in re.sub(r"<[^>]+>", "", str(item.get("announcementTitle", "")))
            for token in excluded
        )
        and str(item.get("adjunctUrl", "")).lower().endswith(".pdf")
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("announcementTime") or 0),
            int(item.get("announcementId") or 0),
        ),
        reverse=True,
    )[0]


def _query_annual_report(
    session: requests.Session,
    stock_org_id: str,
    ticker_code: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    cninfo = config["cninfo"]
    final_error: Exception | None = None
    query_modes = (
        {"category": cninfo["category"], "searchkey": "", "pageSize": "30"},
        {"category": "", "searchkey": "年度报告", "pageSize": "100"},
    )
    for mode in query_modes:
        payload = {
            "pageNum": "1",
            "pageSize": mode["pageSize"],
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{ticker_code},{stock_org_id}",
            "searchkey": mode["searchkey"],
            "secid": "",
            "category": mode["category"],
            "trade": "",
            "seDate": f"{cninfo['query_start_date']}~{cninfo['query_end_date']}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        for attempt in range(1, int(cninfo["maximum_attempts"]) + 1):
            try:
                response = session.post(
                    cninfo["query_url"],
                    data=payload,
                    timeout=int(cninfo["timeout_seconds"]),
                )
                response.raise_for_status()
                records = response.json().get("announcements") or []
                selected = select_full_annual_report(
                    records, int(cninfo["report_year"])
                )
                if selected is not None:
                    return selected
                break
            except Exception as error:
                final_error = error
                if attempt < int(cninfo["maximum_attempts"]):
                    time.sleep(attempt)
    if final_error is not None:
        raise RuntimeError(f"{ticker_code}年报查询失败：{final_error}")
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"[\s\u3000]+", "", value).replace("－", "-")


def _ticker_code_present(normalized_text: str, ticker_code: str) -> bool:
    """核验股票代码，并兼容港股年报省略代码前导零的披露格式。"""
    if not ticker_code:
        return True
    candidates = {ticker_code}
    if ticker_code.startswith("0"):
        candidates.add(ticker_code.lstrip("0"))
    candidates.discard("")
    return any(candidate in normalized_text for candidate in candidates)


def verify_annual_report_pdf(
    path: Path,
    *,
    company_name: str,
    a_ticker: str,
    h_ticker: str,
    report_year: int,
    first_pages: int,
) -> dict[str, Any]:
    if path.read_bytes()[:5] != b"%PDF-":
        return {
            "status": "BLOCKED_NOT_PDF",
            "page_count": 0,
            "a_ticker_in_pdf": False,
            "h_ticker_in_pdf": False,
            "company_name_key_in_pdf": False,
            "report_year_in_pdf": False,
            "extraction_error": "文件头不是PDF",
        }
    try:
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            extracted = "\n".join(
                (page.extract_text() or "") for page in pdf.pages[:first_pages]
            )
    except Exception as error:
        return {
            "status": "BLOCKED_PDF_EXTRACTION",
            "page_count": 0,
            "a_ticker_in_pdf": False,
            "h_ticker_in_pdf": False,
            "company_name_key_in_pdf": False,
            "report_year_in_pdf": False,
            "extraction_error": f"{type(error).__name__}: {error}",
        }
    normalized = _normalize_text(extracted)
    a_code = a_ticker.split(".")[0]
    h_code = h_ticker.split(".")[0] if h_ticker else ""
    company_key = company_name.replace("股份有限公司", "").replace("集团", "")[:4]
    checks = {
        "a_ticker_in_pdf": a_code in normalized,
        "h_ticker_in_pdf": _ticker_code_present(normalized, h_code),
        "company_name_key_in_pdf": _normalize_text(company_key) in normalized,
        "report_year_in_pdf": str(report_year) in normalized,
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED_PDF_IDENTITY_MISMATCH",
        "page_count": page_count,
        **checks,
        "extraction_error": "",
    }


def _download_one_annual_report(
    row: dict[str, str],
    org_id: str,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    ticker = row["a_ticker"].strip()
    ticker_code = ticker.split(".")[0]
    report_year = int(config["cninfo"]["report_year"])
    destination = (
        root
        / config["paths"]["annual_report_root"]
        / f"{ticker_code}_{report_year}.pdf"
    )
    result: dict[str, Any] = {
        "parent_id": row["parent_id"],
        "company_name": row["company_name"],
        "a_ticker": ticker,
        "h_ticker": row["h_ticker"].strip(),
        "announcement_title": "",
        "announcement_id": "",
        "publication_date": "",
        "source_url": "",
        "resolved_url": "",
        "local_file": destination.relative_to(root).as_posix(),
        "bytes": 0,
        "sha256": "",
        "page_count": 0,
        "a_ticker_in_pdf": False,
        "h_ticker_in_pdf": False,
        "company_name_key_in_pdf": False,
        "report_year_in_pdf": False,
        "validation_status": "BLOCKED_NOT_STARTED",
        "error": "",
    }
    try:
        session = _session()
        announcement = _query_annual_report(
            session, org_id, ticker_code, config
        )
        if announcement is None:
            raise RuntimeError("查询期内未找到2025年完整年度报告")
        relative_url = str(announcement["adjunctUrl"])
        source_url = config["cninfo"]["pdf_base_url"].rstrip("/") + "/" + relative_url.lstrip("/")
        result["announcement_title"] = re.sub(
            r"<[^>]+>", "", str(announcement.get("announcementTitle", ""))
        )
        result["announcement_id"] = str(announcement.get("announcementId", ""))
        timestamp = pd.to_datetime(
            announcement.get("announcementTime"), unit="ms", utc=True, errors="coerce"
        )
        if pd.notna(timestamp):
            result["publication_date"] = timestamp.tz_convert("Asia/Shanghai").date().isoformat()
        result["source_url"] = source_url
        if not destination.exists() or destination.stat().st_size < 100:
            content, _, resolved = _request_bytes(
                session,
                source_url,
                timeout=int(config["cninfo"]["timeout_seconds"]),
                maximum_attempts=int(config["cninfo"]["maximum_attempts"]),
            )
            if content[:5] != b"%PDF-":
                raise RuntimeError("年报下载响应不是PDF")
            _atomic_bytes(destination, content)
            result["resolved_url"] = resolved
        else:
            result["resolved_url"] = source_url
        result["bytes"] = int(destination.stat().st_size)
        result["sha256"] = sha256(destination)
        verification = verify_annual_report_pdf(
            destination,
            company_name=row["company_name"],
            a_ticker=ticker,
            h_ticker=row["h_ticker"].strip(),
            report_year=report_year,
            first_pages=int(config["cninfo"]["first_pages_to_verify"]),
        )
        result.update(
            {
                "page_count": verification["page_count"],
                "a_ticker_in_pdf": verification["a_ticker_in_pdf"],
                "h_ticker_in_pdf": verification["h_ticker_in_pdf"],
                "company_name_key_in_pdf": verification["company_name_key_in_pdf"],
                "report_year_in_pdf": verification["report_year_in_pdf"],
                "validation_status": verification["status"],
                "error": verification["extraction_error"],
            }
        )
    except Exception as error:
        result["validation_status"] = "BLOCKED_ACQUISITION_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def acquire_annual_reports(
    root: Path, config: dict[str, Any], universe: pd.DataFrame
) -> pd.DataFrame:
    core = universe.loc[universe["scope_status"].eq("CORE_A_LISTED")].copy()
    session = _session()
    master = session.get(
        config["cninfo"]["stock_master_url"],
        timeout=int(config["cninfo"]["timeout_seconds"]),
    )
    master.raise_for_status()
    stock_map = {
        str(item["code"]): str(item["orgId"])
        for item in master.json().get("stockList", [])
    }
    missing_master = sorted(
        ticker.split(".")[0]
        for ticker in core["a_ticker"].astype(str)
        if ticker.split(".")[0] not in stock_map
    )
    if missing_master:
        raise RuntimeError(f"巨潮股票主表缺少核心代码：{missing_master}")

    records: list[dict[str, Any]] = []
    rows = core.to_dict("records")
    with ThreadPoolExecutor(
        max_workers=int(config["cninfo"]["workers"]),
        thread_name_prefix="券商年报证据",
    ) as pool:
        futures = {
            pool.submit(
                _download_one_annual_report,
                row,
                stock_map[row["a_ticker"].split(".")[0]],
                root,
                config,
            ): row["a_ticker"]
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            completed += 1
            print(
                f"年报证据进度：{completed}/{len(rows)} "
                f"{record['a_ticker']} {record['validation_status']}",
                flush=True,
            )
    return pd.DataFrame(records).sort_values("a_ticker").reset_index(drop=True)


def acquire_reference_snapshots(
    root: Path, config: dict[str, Any], registry: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    session = _session()
    primary = registry.loc[registry["authority_level"].eq("PRIMARY")]
    for index, row in primary.reset_index(drop=True).iterrows():
        source_id = row["source_id"]
        url = config.get("reference_url_overrides", {}).get(source_id, row["url"])
        record: dict[str, Any] = {
            "source_id": source_id,
            "source_url": url,
            "resolved_url": "",
            "local_snapshot_path": "",
            "content_type": "",
            "bytes": 0,
            "sha256": "",
            "status": "BLOCKED_NOT_STARTED",
            "capture_method": "DIRECT_HTTP",
            "error": "",
        }
        try:
            content, content_type, resolved = _request_bytes(
                session,
                url,
                timeout=int(config["cninfo"]["reference_timeout_seconds"]),
                maximum_attempts=int(config["cninfo"]["reference_maximum_attempts"]),
            )
            suffix = ".pdf" if content[:5] == b"%PDF-" or "pdf" in content_type.lower() else ".html"
            destination = (
                root
                / config["paths"]["reference_snapshot_root"]
                / f"{source_id}{suffix}"
            )
            _atomic_bytes(destination, content)
            record.update(
                {
                    "resolved_url": resolved,
                    "local_snapshot_path": destination.relative_to(root).as_posix(),
                    "content_type": content_type,
                    "bytes": int(len(content)),
                    "sha256": sha256(destination),
                    "status": "PASS",
                }
            )
        except Exception as error:
            capture_relative = config.get("browser_capture_fallbacks", {}).get(
                source_id, ""
            )
            capture_path = root / capture_relative if capture_relative else None
            if capture_path is not None and capture_path.exists():
                record.update(
                    {
                        "resolved_url": url,
                        "local_snapshot_path": capture_path.relative_to(root).as_posix(),
                        "content_type": "text/markdown; charset=utf-8",
                        "bytes": int(capture_path.stat().st_size),
                        "sha256": sha256(capture_path),
                        "status": "PASS_BROWSER_TEXT_CAPTURE",
                        "capture_method": "OFFICIAL_PAGE_BROWSER_TEXT_CAPTURE",
                        "error": f"直接下载失败，使用浏览器正文快照：{type(error).__name__}: {error}",
                    }
                )
            else:
                record["status"] = "BLOCKED_DOWNLOAD_ERROR"
                record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        print(
            f"基础来源快照：{index + 1}/{len(primary)} {source_id} {record['status']}",
            flush=True,
        )
    return pd.DataFrame(records)


def build_verified_universe(
    base_universe: pd.DataFrame, annual_manifest: pd.DataFrame
) -> pd.DataFrame:
    output = base_universe.copy()
    manifest = annual_manifest.set_index("a_ticker")
    for index, row in output.iterrows():
        if row["scope_status"] != "CORE_A_LISTED":
            continue
        ticker = row["a_ticker"].strip()
        if ticker not in manifest.index:
            output.at[index, "official_listing_verified"] = "false"
            output.at[index, "notes"] = f"{row['notes']}；V1.1缺少年报证据"
            continue
        evidence = manifest.loc[ticker]
        passed = evidence["validation_status"] == "PASS"
        output.at[index, "official_listing_verified"] = "true" if passed else "false"
        output.at[index, "listing_source_id"] = (
            f"SRC_CNINFO_2025_ANNUAL_{ticker.split('.')[0]}"
        )
        output.at[index, "notes"] = (
            f"V1.1年报原文校验={evidence['validation_status']}；"
            f"sha256={evidence['sha256']}"
        )
    return output


def build_verified_registry(
    base_registry: pd.DataFrame,
    snapshot_manifest: pd.DataFrame,
    annual_manifest: pd.DataFrame,
    retrieved_at: str,
) -> pd.DataFrame:
    columns = list(base_registry.columns)
    if "local_snapshot_path" not in columns:
        columns.append("local_snapshot_path")
    output = base_registry.reindex(columns=columns, fill_value="").copy()
    snapshots = snapshot_manifest.set_index("source_id")
    for index, row in output.iterrows():
        source_id = row["source_id"]
        if source_id not in snapshots.index:
            continue
        snapshot = snapshots.loc[source_id]
        output.at[index, "retrieved_at"] = retrieved_at
        output.at[index, "snapshot_sha256"] = snapshot["sha256"]
        output.at[index, "local_snapshot_path"] = snapshot["local_snapshot_path"]
        output.at[index, "url"] = snapshot["source_url"]
        if not str(snapshot["status"]).startswith("PASS"):
            output.at[index, "limitations"] = (
                f"{row['limitations']}；V1.1快照失败：{snapshot['error']}"
            )
        elif snapshot["status"] == "PASS_BROWSER_TEXT_CAPTURE":
            output.at[index, "limitations"] = (
                f"{row['limitations']}；V1.1仅保存官方网页浏览器正文快照，"
                "未保存原站响应字节或附件内容"
            )

    annual_rows: list[dict[str, Any]] = []
    for _, row in annual_manifest.iterrows():
        ticker_code = row["a_ticker"].split(".")[0]
        has_official_document = bool(row["source_url"] and row["sha256"])
        annual_rows.append(
            {
                "source_id": f"SRC_CNINFO_2025_ANNUAL_{ticker_code}",
                "source_name": row["announcement_title"] or f"{row['company_name']}2025年年度报告",
                "source_type": (
                    "ISSUER_ANNUAL_REPORT"
                    if has_official_document
                    else "MISSING_ANNUAL_REPORT"
                ),
                "authority_level": "PRIMARY" if has_official_document else "BLOCKED",
                "url": row["source_url"],
                "publication_date": row["publication_date"],
                "retrieved_at": retrieved_at,
                "snapshot_sha256": row["sha256"],
                "coverage": f"{row['a_ticker']} 2025年度报告及上市代码核验",
                "limitations": (
                    "仅完成原文身份和哈希核验，尚未完成海外分部财务字段抽取"
                    if row["validation_status"] == "PASS"
                    else f"年报证据未通过：{row['validation_status']} {row['error']}"
                ),
                "local_snapshot_path": row["local_file"],
            }
        )
    annual_frame = pd.DataFrame(annual_rows).reindex(columns=columns, fill_value="")
    return pd.concat([output, annual_frame], ignore_index=True)


def _render_report(report: dict[str, Any]) -> str:
    annual = report["annual_reports"]
    snapshots = report["reference_snapshots"]
    return "\n".join(
        [
            "# 国际化券商V1.1官方证据审计",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- V1冻结指纹：`{report['frozen_v1']['status']}`",
            f"- 42家年报通过：{annual['pass_count']} / {annual['expected_count']}",
            f"- 基础官方来源快照通过：{snapshots['pass_count']} / {snapshots['expected_count']}",
            f"- 母样本状态：`{report['universe']['status']}`",
            f"- 证据登记状态：`{report['evidence_registry']['status']}`",
            f"- G1状态：`{report['g1_status']}`",
            "- 收益检验：`禁止`",
            "- 510300输入：`禁止`",
            "",
            "## 年报失败",
            "",
            *(f"- {item['a_ticker']}：{item['validation_status']}；{item['error']}" for item in annual["failures"]),
            "",
            "## 基础快照失败",
            "",
            *(f"- {item['source_id']}：{item['status']}；{item['error']}" for item in snapshots["failures"]),
            "",
            "## 下一步",
            "",
            "G1通过后，仅对预登记的六家优先券商采集2021—2025年报告并定位海外子公司、并表、少数股东、利润和资本投入字段。",
            "",
        ]
    )


def run_acquisition(root: Path = ROOT) -> dict[str, Any]:
    config = load_evidence_config(root / "config" / "international_broker_evidence_v1_1.yaml")
    frozen = verify_frozen_v1(root, config)
    if frozen["status"] != "PASS":
        raise RuntimeError(f"V1冻结文件发生漂移：{frozen['mismatches']}")
    base_contract = yaml.safe_load(
        (root / config["protocol"]["base_config"]).read_text(encoding="utf-8")
    )
    base_universe = pd.read_csv(
        root / base_contract["universe"]["file"], dtype=str, keep_default_na=False
    )
    base_registry = pd.read_csv(
        root / base_contract["evidence_registry"]["file"],
        dtype=str,
        keep_default_na=False,
    )
    annual = acquire_annual_reports(root, config, base_universe)
    snapshots = acquire_reference_snapshots(root, config, base_registry)
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    verified_universe = build_verified_universe(base_universe, annual)
    verified_registry = build_verified_registry(
        base_registry, snapshots, annual, retrieved_at
    )

    paths = config["paths"]
    _atomic_csv(root / paths["annual_report_manifest"], annual)
    _atomic_csv(root / paths["reference_snapshot_manifest"], snapshots)
    _atomic_csv(root / paths["verified_universe"], verified_universe)
    _atomic_csv(root / paths["verified_source_registry"], verified_registry)

    merged = build_merged_contract(root, config)
    contract = validate_contract(merged)
    evidence, known_ids = audit_evidence_registry(root, merged)
    universe = audit_universe(root, merged, known_ids)
    annual_pass = annual["validation_status"].eq("PASS")
    snapshot_pass = snapshots["status"].astype(str).str.startswith("PASS")
    report = {
        "project_id": config["protocol"]["project_id"],
        "evidence_version": config["protocol"]["evidence_version"],
        "generated_at": retrieved_at,
        "state": config["protocol"]["state"],
        "safety_state": "RESEARCH_ONLY_NO_POSITION_CHANGE",
        "return_test_allowed": False,
        "allow_510300_input": False,
        "frozen_v1": frozen,
        "contract": contract,
        "annual_reports": {
            "expected_count": int(config["validation"]["expected_core_a_issuer_count"]),
            "pass_count": int(annual_pass.sum()),
            "failure_count": int((~annual_pass).sum()),
            "total_bytes": int(annual["bytes"].sum()),
            "failures": annual.loc[~annual_pass, ["a_ticker", "validation_status", "error"]].to_dict("records"),
        },
        "reference_snapshots": {
            "expected_count": int(len(snapshots)),
            "pass_count": int(snapshot_pass.sum()),
            "failure_count": int((~snapshot_pass).sum()),
            "failures": snapshots.loc[~snapshot_pass, ["source_id", "status", "error"]].to_dict("records"),
        },
        "universe": universe,
        "evidence_registry": evidence,
        "g1_status": (
            "PASS"
            if universe["status"] == "PASS" and evidence["status"] == "PASS"
            else "BLOCKED"
        ),
    }
    _atomic_json(root / paths["audit_json"], report)
    _atomic_text(root / paths["audit_markdown"], _render_report(report))
    return report


def main() -> int:
    report = run_acquisition()
    print(
        json.dumps(
            {
                "证据版本": report["evidence_version"],
                "年报通过": report["annual_reports"]["pass_count"],
                "年报失败": report["annual_reports"]["failure_count"],
                "基础快照通过": report["reference_snapshots"]["pass_count"],
                "母样本状态": report["universe"]["status"],
                "证据登记状态": report["evidence_registry"]["status"],
                "G1": report["g1_status"],
                "收益检验允许": report["return_test_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
