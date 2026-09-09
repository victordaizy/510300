"""国际化券商六家优先样本2021—2025年官方年报重建。"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.international_broker_evidence_acquisition import (
    ROOT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _download_one_annual_report,
    _session,
)


CONFIG_FILE = ROOT / "config" / "international_broker_priority_reconstruction_v1_2.yaml"


def load_reconstruction_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_g1_prerequisite(
    root: Path = ROOT, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    current = config or load_reconstruction_config()
    path = root / current["protocol"]["prerequisite_audit"]
    if not path.exists():
        return {"status": "BLOCKED_MISSING_G1_AUDIT", "errors": [str(path)]}
    report = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected = {
        "g1_status": "PASS",
        "return_test_allowed": False,
        "allow_510300_input": False,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            errors.append(f"{field}应为{value}，实际为{report.get(field)}")
    if report.get("annual_reports", {}).get("pass_count") != 42:
        errors.append("G1年报通过数量不等于42")
    if report.get("reference_snapshots", {}).get("pass_count") != 14:
        errors.append("G1基础快照通过数量不等于14")
    return {
        "status": "PASS" if not errors else "BLOCKED_INVALID_G1_AUDIT",
        "audit_file": path.relative_to(root).as_posix(),
        "generated_at": report.get("generated_at", ""),
        "errors": errors,
    }


def expected_report_company_name(
    ticker: str, year: int, current_name: str, config: dict[str, Any]
) -> str:
    overrides = config.get("historical_company_name_overrides", {}).get(ticker, {})
    return str(overrides.get(year, overrides.get(str(year), current_name)))


def _priority_universe(
    root: Path, config: dict[str, Any], evidence_config: dict[str, Any]
) -> pd.DataFrame:
    universe = pd.read_csv(
        root / evidence_config["paths"]["verified_universe"],
        dtype=str,
        keep_default_na=False,
    )
    tickers = list(config["priority_tickers"])
    selected = universe.loc[universe["a_ticker"].isin(tickers)].copy()
    missing = sorted(set(tickers) - set(selected["a_ticker"]))
    if missing:
        raise RuntimeError(f"V1.1母样本缺少优先券商：{missing}")
    if not selected["official_listing_verified"].str.lower().eq("true").all():
        failed = selected.loc[
            ~selected["official_listing_verified"].str.lower().eq("true"), "a_ticker"
        ].tolist()
        raise RuntimeError(f"优先券商尚未通过G1官方核验：{failed}")
    return selected.set_index("a_ticker").loc[tickers].reset_index()


def _existing_2025_rows(
    root: Path,
    config: dict[str, Any],
    evidence_config: dict[str, Any],
    universe: pd.DataFrame,
) -> list[dict[str, Any]]:
    existing = pd.read_csv(
        root / evidence_config["paths"]["annual_report_manifest"],
        dtype=str,
        keep_default_na=False,
    ).set_index("a_ticker")
    records: list[dict[str, Any]] = []
    for _, row in universe.iterrows():
        ticker = row["a_ticker"]
        if ticker not in existing.index:
            raise RuntimeError(f"V1.1年报清单缺少{ticker}")
        record = existing.loc[ticker].to_dict()
        record["a_ticker"] = ticker
        record["report_year"] = 2025
        record["current_company_name"] = row["company_name"]
        record["report_company_name_expected"] = expected_report_company_name(
            ticker, 2025, row["company_name"], config
        )
        record["evidence_reuse"] = "V1_1_VERIFIED"
        records.append(record)
    return records


def _download_historical_one(
    row: dict[str, str],
    year: int,
    stock_org_id: str,
    root: Path,
    config: dict[str, Any],
    evidence_config: dict[str, Any],
) -> dict[str, Any]:
    per_year = copy.deepcopy(evidence_config)
    per_year["cninfo"]["report_year"] = year
    per_year["cninfo"]["query_start_date"] = (
        f"{year + 1}-{config['acquisition']['publication_window_start']}"
    )
    per_year["cninfo"]["query_end_date"] = (
        f"{year + 1}-{config['acquisition']['publication_window_end']}"
    )
    per_year["paths"]["annual_report_root"] = (
        f"{config['paths']['raw_report_root']}/{year}"
    )
    verification_row = dict(row)
    verification_row["company_name"] = expected_report_company_name(
        row["a_ticker"], year, row["company_name"], config
    )
    record = _download_one_annual_report(
        verification_row, stock_org_id, root, per_year
    )
    record["report_year"] = year
    record["current_company_name"] = row["company_name"]
    record["report_company_name_expected"] = verification_row["company_name"]
    record["evidence_reuse"] = "NEW_DOWNLOAD"
    return record


def acquire_priority_history(
    root: Path, config: dict[str, Any], evidence_config: dict[str, Any]
) -> pd.DataFrame:
    universe = _priority_universe(root, config, evidence_config)
    session = _session()
    master = session.get(
        evidence_config["cninfo"]["stock_master_url"],
        timeout=int(evidence_config["cninfo"]["timeout_seconds"]),
    )
    master.raise_for_status()
    stock_map = {
        str(item["code"]): str(item["orgId"])
        for item in master.json().get("stockList", [])
    }
    records = _existing_2025_rows(root, config, evidence_config, universe)
    tasks = [
        (row, int(year))
        for row in universe.to_dict("records")
        for year in config["report_years"]
        if int(year) != 2025
    ]
    with ThreadPoolExecutor(
        max_workers=int(config["acquisition"]["workers"]),
        thread_name_prefix="优先券商历史年报",
    ) as pool:
        futures = {
            pool.submit(
                _download_historical_one,
                row,
                year,
                stock_map[row["a_ticker"].split(".")[0]],
                root,
                config,
                evidence_config,
            ): (row["a_ticker"], year)
            for row, year in tasks
        }
        completed = 0
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            completed += 1
            print(
                f"历史年报进度：{completed}/{len(tasks)} "
                f"{record['a_ticker']} {record['report_year']} "
                f"{record['validation_status']}",
                flush=True,
            )
    frame = pd.DataFrame(records)
    return frame.sort_values(["a_ticker", "report_year"]).reset_index(drop=True)


def build_priority_source_registry(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        code = item["a_ticker"].split(".")[0]
        year = int(item["report_year"])
        rows.append(
            {
                "source_id": f"SRC_CNINFO_ANNUAL_{code}_{year}",
                "a_ticker": item["a_ticker"],
                "report_year": year,
                "source_name": item["announcement_title"],
                "authority_level": "PRIMARY",
                "url": item["source_url"],
                "publication_date": item["publication_date"],
                "local_file": item["local_file"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "validation_status": item["validation_status"],
                "evidence_reuse": item["evidence_reuse"],
            }
        )
    return pd.DataFrame(rows)


def _render_report(report: dict[str, Any]) -> str:
    failures = report["reports"]["failures"]
    return "\n".join(
        [
            "# 国际化券商V1.2优先样本历史年报审计",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- G1前置条件：`{report['g1_prerequisite']['status']}`",
            f"- 年报通过：{report['reports']['pass_count']} / {report['reports']['expected_count']}",
            f"- 覆盖券商：{report['reports']['ticker_count']}",
            f"- 覆盖年度：{report['reports']['year_min']}—{report['reports']['year_max']}",
            f"- G2原文状态：`{report['g2_document_status']}`",
            "- 收益检验：`禁止`",
            "- 510300输入：`禁止`",
            "",
            "## 失败项",
            "",
            *(
                f"- {item['a_ticker']} {item['report_year']}："
                f"{item['validation_status']}；{item['error']}"
                for item in failures
            ),
            "",
            "## 边界",
            "",
            "本审计只证明六家优先券商的历史年报原文齐备且身份通过；不代表海外分部口径已重建，也不授权收益检验。",
            "",
        ]
    )


def run_reconstruction(root: Path = ROOT) -> dict[str, Any]:
    config = load_reconstruction_config(
        root / "config" / "international_broker_priority_reconstruction_v1_2.yaml"
    )
    g1 = verify_g1_prerequisite(root, config)
    if g1["status"] != "PASS":
        raise RuntimeError(f"G1前置条件未通过：{g1['errors']}")
    evidence_config = yaml.safe_load(
        (root / config["protocol"]["base_evidence_config"]).read_text(encoding="utf-8")
    )
    manifest = acquire_priority_history(root, config, evidence_config)
    registry = build_priority_source_registry(manifest)
    paths = config["paths"]
    _atomic_csv(root / paths["manifest"], manifest)
    _atomic_csv(root / paths["source_registry"], registry)

    passed = manifest["validation_status"].eq("PASS")
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    expected_count = len(config["priority_tickers"]) * len(config["report_years"])
    report = {
        "project_id": config["protocol"]["project_id"],
        "reconstruction_version": config["protocol"]["reconstruction_version"],
        "generated_at": generated_at,
        "state": config["protocol"]["state"],
        "safety_state": "RESEARCH_ONLY_NO_POSITION_CHANGE",
        "return_test_allowed": False,
        "allow_510300_input": False,
        "g1_prerequisite": g1,
        "reports": {
            "expected_count": expected_count,
            "pass_count": int(passed.sum()),
            "failure_count": int((~passed).sum()),
            "ticker_count": int(manifest["a_ticker"].nunique()),
            "year_min": int(manifest["report_year"].min()),
            "year_max": int(manifest["report_year"].max()),
            "total_bytes": int(pd.to_numeric(manifest["bytes"]).sum()),
            "failures": manifest.loc[
                ~passed,
                ["a_ticker", "report_year", "validation_status", "error"],
            ].to_dict("records"),
        },
        "g2_document_status": (
            "PASS" if int(passed.sum()) == expected_count else "BLOCKED"
        ),
    }
    _atomic_json(root / paths["audit_json"], report)
    _atomic_text(root / paths["audit_markdown"], _render_report(report))
    return report


def main() -> int:
    report = run_reconstruction()
    print(
        json.dumps(
            {
                "重建版本": report["reconstruction_version"],
                "年报通过": report["reports"]["pass_count"],
                "年报失败": report["reports"]["failure_count"],
                "G2原文": report["g2_document_status"],
                "收益检验允许": report["return_test_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
