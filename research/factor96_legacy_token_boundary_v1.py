"""定向核查旧字段的数值边界疑点，不修改字段或计算市场收益。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd
import requests

from research.factor96_earnings_cashflow_measurement_v1 import digest, now, save
from research.factor96_financial_row_parser_v2 import extract_official_pdf_facts


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_1"
BASE = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2"
OUT = ROOT / "reports/research/510300_factor96_legacy_token_boundary_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_candidates(facts):
    """仅标记保存跨度之后仍有千分组的旧记录，金额不自动改写。"""
    rows = []
    for row in facts.loc[facts.repair_scope.eq("UNCHANGED_LEGACY_NOT_IN_SUMMARY_PATH")].itertuples():
        try:
            locator = json.loads(row.source_locator)
        except (TypeError, ValueError):
            continue
        span = locator.get("value_span")
        text = locator.get("line_window", "")
        if not span or len(span) != 2 or text[span[0]:span[1]] != row.source_raw_value:
            continue
        tail = text[span[1]:]
        if not re.match(r"[,，]\d{3}", tail):
            continue
        rows.append({"announcement_id": str(row.announcement_id), "ts_code": row.ts_code,
            "report_period": str(row.report_period.date()), "period_type": row.period_type,
            "metric_id": row.metric_id, "old_value_cny": row.verified_value,
            "source_raw_value": row.source_raw_value, "source_page": int(row.source_page),
            "source_locator": locator, "following_text": tail,
            "kind": "INTEGER_GROUP_CONTINUATION" if "." not in row.source_raw_value else "POSSIBLE_ADJACENT_COLUMN_JOIN",
            "pdf_sha256": row.official_pdf_sha256, "pdf_bytes": int(row.official_pdf_size_bytes),
            "url": row.official_pdf_url})
    return rows


def prepare():
    assert not OUT.exists()
    facts_path = SOURCE / "repaired_verified_facts.parquet"
    rows = find_candidates(pd.read_parquet(facts_path))
    assert len(rows) == 5 and len({r["announcement_id"] for r in rows}) == 5
    assert any(r["announcement_id"] == "1207690337" for r in rows)
    for name in ["code", "inputs", "pdf", "fetch_receipts", "parsed"]:
        (OUT / name).mkdir(parents=True, exist_ok=False)
    for path in [Path(__file__), ROOT / "research/factor96_financial_row_parser_v2.py"]:
        shutil.copy2(path, OUT / "code" / path.name)
    shutil.copy2(facts_path, OUT / "inputs/fields_before.parquet")
    shutil.copy2(SOURCE / "source_repair_result.json", OUT / "inputs/source_repair_result.json")
    targets = {t["sha256"]: t for t in read(BASE / "batch_targets.json")}
    for row in rows:
        target = targets.get(row["pdf_sha256"])
        local = ROOT / target["pdf_relative_path"] if target else None
        row["existing_same_hash_path"] = local.relative_to(ROOT).as_posix() if local and local.exists() and digest(local) == row["pdf_sha256"] else None
        old = ROOT / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/checkpoints_v1_6" / row["report_period"][:4] / (row["announcement_id"] + ".json.gz")
        if old.exists():
            shutil.copy2(old, OUT / "inputs" / old.name)
    save(OUT / "candidate_fields.json", rows)
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_FACTOR96_LEGACY_TOKEN_BOUNDARY_V1",
        "known_before_freeze": "已经查看V2.0.1财务诊断极值及旧字段定位；亨通光电2019FY资产疑似千分组截断；已在33143项未修复旧字段中扫描同类定位跨度，得到5项候选，尚未取得这5份原文或观察市场收益。",
        "question": "旧字段数值跨度是否提前截断千分组，或粘连下一列数字？",
        "selection": "仅旧字段，source_locator的value_span与source_raw_value逐字一致且其后紧接逗号加3位数字；列粘连只作候选，不能凭该模式自动修值。",
        "scope": "5字段、5文档；只取原记录的URL和完全相同SHA256，缺失、版本不一致与解析未知保持未知。每文档至多一次逻辑HTTP，不重试，不搜索替代年份。",
        "maximum_new_logical_http_requests": sum(r["existing_same_hash_path"] is None for r in rows),
        "parser": "冻结现有V2通用表格解析器，仅作原文核对；不改解析器、原字段版本、财务公式或旧失败。",
        "admission": "任何确认旧金额错误都使V2.0.1混合来源测量不能接入T11；需新来源版本统一处理完整候选范围并重新固定公式复算。",
        "limits": "该扫描只覆盖存有精确定位跨度的这一模式，不证明余下字段完全正确。当前原文不证明历史首次公开版本。",
        "new_accounts": 0, "returns_read": False, "orders_authorized": False, "goal_achieved": False})
    files = [p for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]})
    print("已冻结5字段的原文核查范围，不自动修改财务金额。", flush=True)


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    for path in [Path(__file__), ROOT / "research/factor96_financial_row_parser_v2.py"]:
        assert digest(path) == digest(OUT / "code" / path.name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json"), "new_accounts": 0})
    results = []
    for row in read(OUT / "candidate_fields.json"):
        aid = row["announcement_id"]
        receipt = {"at": now(), "announcement_id": aid, "url": row["url"], "expected_sha256": row["pdf_sha256"], "logical_http_requests": 0}
        pdf = OUT / "pdf" / (aid + ".pdf")
        try:
            if row["existing_same_hash_path"]:
                raw = (ROOT / row["existing_same_hash_path"]).read_bytes()
                receipt["route"] = "LOCAL_REUSE"
            else:
                receipt["logical_http_requests"] = 1
                response = requests.get(row["url"], timeout=(15, 90), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
                receipt.update({"route": "OFFICIAL_URL_ONCE", "http_status": response.status_code,
                    "final_url": response.url, "content_type": response.headers.get("Content-Type")})
                response.raise_for_status()
                raw = response.content
            checksum = hashlib.sha256(raw).hexdigest()
            receipt.update({"bytes": len(raw), "sha256": checksum})
            assert raw[:4] == b"%PDF" and checksum == row["pdf_sha256"] and len(raw) == row["pdf_bytes"], "原文身份不符，不能替代旧档案"
            pdf.write_bytes(raw)
            receipt["status"] = "PASS_SAME_HASH_PDF"
        except (requests.RequestException, AssertionError, OSError) as error:
            receipt.update({"status": "NO_VIEW_SOURCE_NOT_CONFIRMED", "error_type": type(error).__name__, "error": str(error)})
        save(OUT / "fetch_receipts" / (aid + ".json"), receipt)
        result = {**row, "fetch_status": receipt["status"], "new_value_cny": None, "status": "NO_VIEW"}
        if receipt["status"] == "PASS_SAME_HASH_PDF":
            parsed = extract_official_pdf_facts(raw, period_type=row["period_type"], report_period=row["report_period"])
            save(OUT / "parsed" / (aid + ".json"), parsed)
            value = next((r for r in parsed["metrics"] if r["metric_id"] == row["metric_id"]), None)
            if value is not None:
                result.update({"new_value_cny": value["metric_value_cny"], "new_source_page": value["source_page"],
                    "new_source_locator": value["source_locator"],
                    "status": "VALUE_DIFFERS_SAME_PDF" if abs(value["metric_value_cny"] - row["old_value_cny"]) > .011 else "SAME_VALUE_WITHIN_CENT_TOLERANCE"})
        results.append(result)
        save(OUT / ("case_" + aid + ".json"), result)
        print(f"{aid}：{result['status']}，原值{row['old_value_cny']}，本次原文解析值{result['new_value_cny']}。", flush=True)
    output = {"at": now(), "status": "COMPLETED_BOUNDED_SOURCE_PROBE",
        "candidate_fields": len(results), "same_hash_pdfs": sum(r["fetch_status"] == "PASS_SAME_HASH_PDF" for r in results),
        "differing_fields": sum(r["status"] == "VALUE_DIFFERS_SAME_PDF" for r in results),
        "unknown_fields": sum(r["status"] == "NO_VIEW" for r in results), "cases": results,
        "new_accounts": 0, "returns_read": False, "goal_achieved": False}
    save(OUT / "result.json", output)
    print("5字段核查结束，旧字段及既有测量结果没有回写。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
