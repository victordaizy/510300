"""验证并生成国际化券商首版母子公司及海外财务证据表。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import yaml

from research.international_broker_evidence_acquisition import (
    ROOT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    sha256,
)


CONFIG_FILE = ROOT / "config" / "international_broker_subsidiary_map_v1_2.yaml"


def load_subsidiary_map_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_document_prerequisite(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    path = root / config["protocol"]["prerequisite_audit"]
    if not path.exists():
        return {"status": "BLOCKED_MISSING_G2_DOCUMENT_AUDIT", "errors": [str(path)]}
    report = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("g2_document_status") != "PASS":
        errors.append("G2原文状态不是PASS")
    if report.get("reports", {}).get("pass_count") != 30:
        errors.append("G2原文通过数量不等于30")
    if report.get("return_test_allowed") is not False:
        errors.append("G2原文审计错误地允许收益检验")
    return {
        "status": "PASS" if not errors else "BLOCKED_INVALID_G2_DOCUMENT_AUDIT",
        "audit_file": path.relative_to(root).as_posix(),
        "errors": errors,
    }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("，", ",")


def _validate_record_pages(
    pdf_path: Path, pages: list[int], required_terms: list[str]
) -> tuple[list[str], str]:
    with pdfplumber.open(pdf_path) as pdf:
        invalid_pages = [page for page in pages if page < 1 or page > len(pdf.pages)]
        if invalid_pages:
            return [f"PDF页码越界：{invalid_pages}"], ""
        page_text = "\n".join(
            pdf.pages[page - 1].extract_text() or "" for page in pages
        )
    normalized = _normalize(page_text)
    reversed_normalized = _normalize(page_text[::-1])
    missing = [
        term
        for term in required_terms
        if _normalize(str(term)) not in normalized
        and _normalize(str(term)) not in reversed_normalized
    ]
    return [f"证据页缺少术语：{missing}"] if missing else [], normalized


def build_subsidiary_map(
    root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = pd.read_csv(
        root / config["protocol"]["priority_manifest"],
        dtype=str,
        keep_default_na=False,
    )
    manifest["report_year"] = manifest["report_year"].astype(int)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in config["records"]:
        row = dict(item)
        matched = manifest.loc[
            manifest["a_ticker"].eq(row["parent_ticker"])
            & manifest["report_year"].eq(int(row["report_year"]))
        ]
        errors: list[str] = []
        if len(matched) != 1:
            errors.append(f"年报清单匹配数量应为1，实际为{len(matched)}")
            local_file = ""
            source_sha256 = ""
            source_url = ""
        else:
            evidence = matched.iloc[0]
            local_file = evidence["local_file"]
            source_sha256 = evidence["sha256"]
            source_url = evidence["source_url"]
            pdf_path = root / local_file
            if not pdf_path.exists():
                errors.append("年报原文不存在")
            elif sha256(pdf_path) != source_sha256:
                errors.append("年报原文哈希与清单不一致")
            else:
                page_errors, _ = _validate_record_pages(
                    pdf_path, row["pdf_pages"], row["required_terms"]
                )
                errors.extend(page_errors)
            if evidence["validation_status"] != "PASS":
                errors.append("年报原文身份状态不是PASS")

        no_platform = row["platform_status"].startswith("NO_CONSOLIDATED")
        if no_platform:
            if row["consolidated_flag"] or row["ownership_pct"] is not None:
                errors.append("无并表平台行不得填写持股或并表=true")
        else:
            ownership = float(row["ownership_pct"])
            if not (0 < ownership <= 100):
                errors.append("持股比例必须在(0,100]范围")
            if row["consolidated_flag"] is not True:
                errors.append("已确认平台必须并表")

        row["source_id"] = (
            f"SRC_CNINFO_ANNUAL_{row['parent_ticker'].split('.')[0]}_"
            f"{row['report_year']}"
        )
        row["source_url"] = source_url
        row["source_local_file"] = local_file
        row["source_sha256"] = source_sha256
        row["pdf_pages"] = ",".join(str(value) for value in row["pdf_pages"])
        row["printed_pages"] = ",".join(
            str(value) for value in row["printed_pages"]
        )
        row["required_terms"] = "|".join(str(value) for value in row["required_terms"])
        row["field_validation_status"] = "PASS" if not errors else "BLOCKED"
        row["validation_errors"] = "；".join(errors)
        records.append(row)
        if errors:
            failures.append(
                {
                    "parent_ticker": row["parent_ticker"],
                    "subsidiary_name": row["subsidiary_name"],
                    "errors": errors,
                }
            )
    return pd.DataFrame(records), failures


def _render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 国际化券商V1.2首版母子公司证据审计",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- G2原文前置条件：`{report['document_prerequisite']['status']}`",
            f"- 字段行通过：{report['map']['pass_count']} / {report['map']['row_count']}",
            f"- 已确认并表境外平台的母公司：{report['map']['confirmed_platform_parent_count']} / 6",
            f"- 未发现并表境外平台的母公司：{report['map']['no_platform_parent_count']} / 6",
            f"- 字段图状态：`{report['g2_subsidiary_map_status']}`",
            "- 多年利润重建：`未完成`",
            "- 收益检验：`禁止`",
            "- 510300输入：`禁止`",
            "",
            "## 初步筛选结论",
            "",
            "- 中信证券、中金公司、中信建投、华泰证券、国泰海通：2025年报确认存在上市公司并表的境外平台。",
            "- 国泰海通必须拆分国泰海通金融控股与吸收合并取得的海通国际控股；后者2025年披露净资产及利润均为负。",
            "- 中银证券：2025年报未发现并表境外平台；中银国际控股是其股东，不是其子公司。",
            "",
            "## 边界",
            "",
            "不同公司使用人民币、港元或美元及元、百万元、一亿元等不同单位，未换算前禁止横向排名。营业利润不等同税前利润；空缺字段保持空缺。",
            "",
        ]
    )


def run_subsidiary_map(root: Path = ROOT) -> dict[str, Any]:
    config = load_subsidiary_map_config(
        root / "config" / "international_broker_subsidiary_map_v1_2.yaml"
    )
    prerequisite = verify_document_prerequisite(root, config)
    if prerequisite["status"] != "PASS":
        raise RuntimeError(f"G2原文前置条件失败：{prerequisite['errors']}")
    table, failures = build_subsidiary_map(root, config)
    _atomic_csv(root / config["paths"]["output_table"], table)
    confirmed = table["platform_status"].eq(
        "CONFIRMED_CONSOLIDATED_OVERSEAS_PLATFORM"
    )
    no_platform = table["platform_status"].str.startswith("NO_CONSOLIDATED")
    passed = table["field_validation_status"].eq("PASS")
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report = {
        "project_id": config["protocol"]["project_id"],
        "map_version": config["protocol"]["map_version"],
        "generated_at": generated_at,
        "state": config["protocol"]["state"],
        "safety_state": "RESEARCH_ONLY_NO_POSITION_CHANGE",
        "return_test_allowed": False,
        "allow_510300_input": False,
        "document_prerequisite": prerequisite,
        "map": {
            "row_count": int(len(table)),
            "pass_count": int(passed.sum()),
            "failure_count": int((~passed).sum()),
            "confirmed_platform_row_count": int(confirmed.sum()),
            "confirmed_platform_parent_count": int(
                table.loc[confirmed, "parent_ticker"].nunique()
            ),
            "no_platform_parent_count": int(
                table.loc[no_platform, "parent_ticker"].nunique()
            ),
            "negative_net_asset_platforms": table.loc[
                pd.to_numeric(table["net_assets"], errors="coerce").lt(0),
                ["parent_ticker", "subsidiary_name", "net_assets", "financial_currency", "financial_unit"],
            ].to_dict("records"),
            "failures": failures,
        },
        "multi_year_profit_reconstruction_ready": False,
        "license_audit_ready": False,
        "g2_subsidiary_map_status": "PASS" if not failures else "BLOCKED",
    }
    _atomic_json(root / config["paths"]["audit_json"], report)
    _atomic_text(root / config["paths"]["audit_markdown"], _render_report(report))
    return report


def main() -> int:
    report = run_subsidiary_map()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
