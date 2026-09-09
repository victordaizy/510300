"""按银行合并表标题和累计期间表头识别财务数值，拒绝未经证明的版式。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import ROOT
from research.csi300_pit_fundamental_underreaction_official_facts_v1 import extract_pdf_page_texts
from research.csi300_pit_fundamental_underreaction_official_facts_v1_6 import extract_metrics_from_page_texts
from research.intraday_overnight_increment_v1 import now, require, write_json

OUT = ROOT / "reports/research/510300_bank_financial_scope_adapter_v1"
COLLECT = ROOT / "reports/research/510300_original_earnings_source_completion_v1"
PROTOCOL = ROOT / "config/510300_bank_financial_scope_adapter_v1_manifest.json"
NUMBER = r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?"


def normalize(text):
    return unicodedata.normalize("NFKC", str(text)).replace("−", "-").replace("—", "-").replace("－", "-")


def amount(text):
    text = text.replace(",", "")
    return -float(text[1:-1]) if text.startswith("(") and text.endswith(")") else float(text)


def row_values(text: str, label: str):
    label_pattern = r"\s*".join(re.escape(c) for c in normalize(label))
    pattern = label_pattern + r"\s+" + r"\s+".join([f"({NUMBER})"] * 4)
    found = re.search(pattern, text)
    return found.groups() if found else None


def parse_consolidated_q3_page(text: str, year: int) -> dict:
    text = normalize(text)
    compact = re.sub(r"\s+", "", text)
    heading = "未经审计合并利润表"
    at = compact.find(heading)
    if at < 0 or at > 350:
        return {"status": "NO_VIEW_NOT_EXPLICIT_CONSOLIDATED_INCOME_STATEMENT", "facts": []}
    end = compact.find("营业收入", at)
    if end < 0:
        return {"status": "NO_VIEW_NO_MATCHING_INCOME_HEADER", "facts": []}
    header = compact[at:end]
    periods = re.findall(r"(1-9|7-9)月期间", header)
    years = [int(x) for x in re.findall(r"(20\d{2})年", header)]
    if "人民币百万元" not in header or years != [year, year-1, year, year-1] or periods not in (["1-9", "7-9"], ["7-9", "1-9"]):
        return {"status": "NO_VIEW_UNPROVEN_CURRENCY_OR_PERIOD_COLUMN_LAYOUT", "facts": [], "header": header}
    column = 0 if periods[0] == "1-9" else 2
    labels = {"OPERATING_REVENUE_YTD": "营业收入合计", "OPERATING_EXPENSE_YTD": "营业支出合计",
              "OPERATING_PROFIT_YTD": "营业利润", "PARENT_NET_PROFIT_YTD": "本行股东的净利润",
              "MINORITY_NET_PROFIT_YTD": "少数股东的净利润", "BASIC_AND_DILUTED_EPS_YTD": "基本及稀释每股收益(人民币元)"}
    rows = {key: row_values(text, label) for key, label in labels.items()}
    total_profit = re.search(r"(?<!的)净利润\s+" + r"\s+".join([f"({NUMBER})"] * 4), text)
    rows["TOTAL_NET_PROFIT_YTD"] = total_profit.groups() if total_profit else None
    missing = [key for key, values in rows.items() if values is None]
    if missing:
        return {"status": "NO_VIEW_REQUIRED_ROWS_FOR_IDENTITY_MISSING", "facts": [], "missing": missing}
    values = {key: np.array([amount(x) for x in row]) for key, row in rows.items()}
    revenue_error = float(np.max(np.abs(values["OPERATING_REVENUE_YTD"] + values["OPERATING_EXPENSE_YTD"] - values["OPERATING_PROFIT_YTD"])))
    owner_error = float(np.max(np.abs(values["PARENT_NET_PROFIT_YTD"] + values["MINORITY_NET_PROFIT_YTD"] - values["TOTAL_NET_PROFIT_YTD"])))
    if revenue_error > 1.01 or owner_error > 1.01:
        return {"status": "NO_VIEW_CONSOLIDATED_ACCOUNTING_IDENTITY_MISMATCH", "facts": [], "revenue_identity_error_million": revenue_error, "owner_identity_error_million": owner_error}
    facts = []
    for key in ["OPERATING_REVENUE_YTD", "OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD", "BASIC_AND_DILUTED_EPS_YTD"]:
        multiplier = 1 if key == "BASIC_AND_DILUTED_EPS_YTD" else 1_000_000
        facts.append({"metric_id": key, "metric_value": float(values[key][column] * multiplier),
                      "unit": "CNY_PER_SHARE" if multiplier == 1 else "CNY", "statement_scope": "CONSOLIDATED_ONLY",
                      "period_scope": "YEAR_TO_DATE_1_9", "selected_numeric_column_one_based": column+1,
                      "source_label": labels[key], "source_raw_value": rows[key][column], "source_unit_multiplier": multiplier,
                      "all_numeric_cells_for_verification": list(rows[key]), "previous_year_comparative_is_original_prior_vintage": False})
    return {"status": "PASS_EXPLICIT_CONSOLIDATED_Q3_PERIOD_HEADER_AND_IDENTITIES", "facts": facts, "header": header,
            "revenue_identity_error_million": revenue_error, "owner_identity_error_million": owner_error}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not PROTOCOL.exists(), "银行合并期间识别已登记")
    paths = [Path(__file__), ROOT / "tests/test_bank_financial_scope_adapter_v1.py", ROOT / "docs/510300_BANK_FINANCIAL_SCOPE_ADAPTER_V1.md",
             COLLECT / "batch_01_result.json", COLLECT / "bank_eps_source_example_1211361859.json"]
    write_json(PROTOCOL, {"registered_at": now(), "purpose": "SOURCE_EXTRACTION_ONLY_NOT_PORTFOLIO", "files": [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "scope": "只接受合并利润表明确列出1至9月和7至9月、两期年份及百万元单位，且两个会计等式在全部四列成立的版式。", "original_nonfinancial_parser_changed": False}, exclusive=True)


def run():
    require(PROTOCOL.exists(), "来源处理规则尚未登记")
    require(not (OUT / "result.json").exists(), "银行首批范围识别已完成")
    for item in json.loads(PROTOCOL.read_text(encoding="utf-8"))["files"]:
        require(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], "来源处理文件变化")
    records = json.loads((COLLECT / "batch_01_result.json").read_text(encoding="utf-8"))["rows"]
    facts, diagnostics, old_example = [], [], None
    for n, record in enumerate(records, 1):
        source = Path(r"E:\ResearchData\New project 8") / record["raw_path"]
        content = source.read_bytes()
        require(hashlib.sha256(content).hexdigest() == record["sha256"], "新收集原始报告哈希变化")
        pages, text_receipt = extract_pdf_page_texts(content)
        year = int(record["report_period"][:4])
        outputs = [parse_consolidated_q3_page(text, year) for text in pages]
        admitted = [(i+1, result) for i, result in enumerate(outputs) if result["status"].startswith("PASS_")]
        require(len(admitted) <= 1, "一份报告出现多个相同完整累计表，需要消除歧义")
        for page_number, result in admitted:
            for fact in result["facts"]:
                facts.append({"announcement_id": record["announcement_id"], "ts_code": record["ts_code"], "sec_name": record["sec_name"],
                              "report_period": record["report_period"], "event_publication_date": record["event_publication_date"],
                              "official_pdf_url": record["official_pdf_url"], "official_pdf_sha256": record["sha256"],
                              "source_page": page_number, "status": result["status"], **fact})
        diagnostics.append({"announcement_id": record["announcement_id"], "ts_code": record["ts_code"], "sec_name": record["sec_name"],
                            "page_count": len(pages), "admitted_fact_count": 4 * len(admitted),
                            "status": "PASS_MATCHING_EXPLICIT_LAYOUT" if admitted else "NO_VIEW_LAYOUT_NOT_YET_SUPPORTED",
                            "page_statuses": [{"page": i+1, **result} for i, result in enumerate(outputs) if result["status"] != "NO_VIEW_NOT_EXPLICIT_CONSOLIDATED_INCOME_STATEMENT"]})
        if str(record["announcement_id"]) == "1211361859":
            old = extract_metrics_from_page_texts(pages, period_type="Q3", text_engine="PDFIUM")
            old_example = {"legacy_parse_result": old, "raw_page_18": pages[17], "raw_page_20": pages[19],
                           "status": "REJECT_LEGACY_NONFINANCIAL_PARSER_APPLICATION_TO_THIS_BANK_REPORT",
                           "legacy_value_is_not_admitted_financial_factor": True, "original_study_excluded_financials": True,
                           "new_consolidated_result": admitted, "visually_checked_pages": [18,20]}
            expected = {"OPERATING_REVENUE_YTD": 251410000000, "PARENT_NET_PROFIT_YTD": 93615000000, "BASIC_AND_DILUTED_EPS_YTD": 3.62, "OPERATING_PROFIT_YTD": 116581000000}
            require(admitted and all(np.isclose(f["metric_value"], expected[f["metric_id"]], rtol=0, atol=.01) for f in admitted[0][1]["facts"]), "已核实原始实例未复现")
        write_json(OUT / "document_records" / (str(record["announcement_id"]) + ".json"), diagnostics[-1])
        print(f"金融报表范围与期间识别 {n}/{len(records)}：{record['sec_name']}，提取{4*len(admitted)}项", flush=True)
    require(old_example is not None, "未找到已核实的银行原始实例")
    write_json(OUT / "legacy_parser_bank_scope_failure_example.json", old_example, exclusive=True)
    pd.DataFrame(facts).to_parquet(OUT / "explicit_consolidated_q3_facts.parquet", index=False)
    pd.DataFrame(facts).to_csv(OUT / "合并累计盈利与每股盈利_已识别数值.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "PARTIAL_BANK_SOURCE_LAYOUT_REBUILD_CONTINUES", "documents_examined": len(records),
              "documents_with_supported_layout": sum(d["admitted_fact_count"] > 0 for d in diagnostics), "fact_rows": len(facts),
              "all_financial_sector_reports_validated": False, "new_portfolio_evaluation": False,
              "legacy_scope_and_period_contamination_prevented": True, "next_step": "扩展其余原始银行、券商、保险版式；继续核对普通股权益扣除与原始去年同期，不将当前表内比较列回填去年。"}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融公司合并范围、累计期间及原始金额识别")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.freeze != args.run, "请选择登记或执行")
    freeze() if args.freeze else run()
