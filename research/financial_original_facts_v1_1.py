"""金融原始季报的主体、累计期间和股东利润识别；不读取策略收益。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"E:\ResearchData\New project 8")
INVENTORY = ROOT / "reports/research/510300_financial_original_layout_inventory_v1"
REPAIR = ROOT / "reports/research/510300_financial_report_subject_repair_v1"
OUT = ROOT / "reports/research/510300_financial_original_facts_v1_1"
MANIFEST = ROOT / "config/510300_financial_original_facts_v1_1_manifest.json"
NAMES = {
    "600999.SH": "招商证券股份有限公司", "601601.SH": "中国太平洋保险(集团)股份有限公司",
    "601328.SH": "交通银行股份有限公司", "601288.SH": "中国农业银行股份有限公司",
    "601939.SH": "中国建设银行股份有限公司", "600036.SH": "招商银行股份有限公司",
    "600030.SH": "中信证券股份有限公司", "601628.SH": "中国人寿保险股份有限公司",
    "000776.SZ": "广发证券股份有限公司", "601211.SH": "国泰君安证券股份有限公司",
    "000001.SZ": "平安银行股份有限公司", "601318.SH": "中国平安保险(集团)股份有限公司",
}
BROKERS = {"600999.SH", "600030.SH", "000776.SZ", "601211.SH"}
NUMBER = r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?"
LABELS = {
    "OPERATING_REVENUE_YTD": ["营业收入合计", "营业总收入", "营业收入"],
    "OPERATING_EXPENSE_YTD": ["营业支出合计", "营业总支出", "营业支出"],
    "OPERATING_PROFIT_YTD": ["营业利润"],
    "PRE_IMPAIRMENT_OPERATING_PROFIT_YTD": ["减值损失前营业利润"],
    "CREDIT_IMPAIRMENT_YTD": ["信用减值损失"],
    "OTHER_IMPAIRMENT_YTD": ["其他资产减值损失"],
    "TOTAL_NET_PROFIT_YTD": ["净利润"],
    "PARENT_NET_PROFIT_YTD": ["归属于母公司股东的净利润", "其中:归属于母公司所有者(或股东)的净利润", "归属于母公司所有者的净利润", "归属于本行股东的净利润", "本行股东的净利润"],
    "MINORITY_NET_PROFIT_YTD": ["少数股东损益", "少数股东的净利润"],
    "BASIC_EPS_YTD": ["基本及稀释每股收益", "基本和稀释每股收益", "基本每股收益", "基本/稀释每股收益"],
    "DILUTED_EPS_YTD": ["稀释每股收益"],
}
CORE = ["OPERATING_REVENUE_YTD", "OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD", "BASIC_EPS_YTD"]


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def norm(text):
    return unicodedata.normalize("NFKC", text).replace("−", "-").replace("—", "-").replace("\r", "")


def compact(text):
    return re.sub(r"\s+", "", norm(text))


def number(token):
    value = token.replace(",", "")
    return -Decimal(value[1:-1]) if value.startswith("(") else Decimal(value)


def identity(pages, source):
    first = next((i for i, p in enumerate(pages[:2]) if compact(p)), None)
    if first is None:
        return {"status": "NO_VIEW_NO_TEXT_TITLE_PAGE"}
    title, front = compact(pages[first]), compact("\n".join(pages[:4]))
    year = str(pd.Timestamp(source["report_period"]).year)
    quarter = "第三季度" if pd.Timestamp(source["report_period"]).month == 9 else "第一季度"
    passed = NAMES[source["ts_code"]] in title and source["ts_code"].split(".")[0] in front and year in title and (quarter in title or quarter.replace("第", "") in title)
    return {"status": "PASS_FIRST_TEXT_TITLE_SUBJECT_CODE_PERIOD" if passed else "NO_VIEW_WRONG_SUBJECT_CODE_OR_PERIOD",
            "identity_page": first + 1, "skipped_empty_text_pages": first, "raw_identity_page": pages[first]}


def statement_lines(pages, year):
    start = None
    for pi, page in enumerate(pages):
        lines = norm(page).splitlines()
        for li, line in enumerate(lines):
            c = compact(line)
            explicit = re.fullmatch(r"(?:\d+、)?(?:未经审计)?合并(?:年初到报告期末)?利润表", c)
            life = c == f"{year}年三季度利润表(未经审计)" and "本集团本集团本公司本公司" in compact(page)
            if explicit or life or c == f"{year}年三季度合并利润表":
                start = (pi, li)
                break
        if start:
            break
    if start is None:
        return []
    result = []
    for pi in range(start[0], min(start[0] + 4, len(pages))):
        for li, line in enumerate(norm(pages[pi]).splitlines()):
            if pi == start[0] and li < start[1]:
                continue
            c = compact(line)
            boundary = ("现金流量表" in c or "资产负债表" in c or "母公司利润表" in c or re.fullmatch(r"(?:未经审计)?(?:银行|本行|本公司)?利润表(?:\(续\))?", c))
            if boundary:
                return result
            if line.strip():
                result.append({"page": pi + 1, "text": line})
    return result


def header_and_layout(lines, year):
    idx = next((i for i, r in enumerate(lines) if re.search(r"^(?:一、)?营业(?:总)?收入", compact(r["text"]))), None)
    if idx is None:
        raise ValueError("未找到合并利润表收入起始行")
    header = compact("\n".join(r["text"] for r in lines[:idx]))
    after = header.split("项目", 1)[-1] if "项目" in header else header
    if "项目" not in header and "审计类型:未经审计" in header:
        after = header.split("审计类型:未经审计", 1)[1]
    direct = [(int(y), p) for y, p in re.findall(r"(20\d{2})年(?:前三季度)?\(?(1-9|7-9)月\)?", after)]
    width = len(direct)
    group_columns = "本集团本集团本公司本公司" in after
    if width in (2, 4):
        wanted = [i for i, x in enumerate(direct) if x == (year, "1-9") and (not group_columns or i < 2)]
        if len(wanted) == 1 and all(y in (year, year - 1) for y, _ in direct):
            return {"header": header, "width": width, "column": wanted[0], "layout": "EXPLICIT_YEAR_PERIOD_COLUMNS", "group_company_columns": group_columns}
    years = [int(x) for x in re.findall(r"(20\d{2})年", header)]
    if years[-4:] == [year, year - 1, year, year - 1]:
        groups = re.findall(r"(1-9|7-9)月期间", header)
        if groups in (["1-9", "7-9"], ["7-9", "1-9"]):
            col = 0 if groups[0] == "1-9" else 2
            return {"header": header, "width": 4, "column": col, "layout": "TWO_EXPLICIT_PERIOD_GROUPS", "group_company_columns": False}
        durations = re.findall(r"(?<!\d)(3|9)个月(?:期间)?", header)
        if durations in (["3", "9"], ["9", "3"]) and "9月30日" in header:
            return {"header": header, "width": 4, "column": 2 if durations[0] == "3" else 0, "layout": "THREE_AND_NINE_MONTH_GROUPS", "group_company_columns": False}
    if "合并年初到报告期末利润表" in header and "本期发生额上期发生额" in header:
        return {"header": header, "width": 2, "column": 0, "layout": "EXPLICIT_YTD_CURRENT_PRIOR", "group_company_columns": False}
    if f"{year}年1-9月" in header and "本期金额(7-9月)上期金额(7-9月)年初至报告期期末金额(1-9月)上年年初至报告期期末金额(1-9月)" in header:
        return {"header": header, "width": 4, "column": 2, "layout": "EXPLICIT_QUARTER_AND_YTD_RELATIVE", "group_company_columns": False}
    raise ValueError("累计年度及列顺序尚未证明：" + header)


def row_matches(lines, label):
    text = "\n".join(row["text"] for row in lines)
    # 标签允许换行，括号中的填报说明和单位属于标签，不能混入数值。
    prefix = r"(?m)^\s*(?:(?:[一二三四五六七八九十]+、|\([一二三四五六七八九十]+\)|\d+[.、]|-)\s*)?"
    label_re = r"\s*".join(re.escape(ch) for ch in norm(label))
    suffix = r"(?:\s*\((?=[^)]*[A-Za-z\u4e00-\u9fff])[^)]{0,100}\))?\s*[:：]?\s*"
    cells = r"(?:人民币\s*)?" + NUMBER + r"(?:%\)?)?(?:\s*元)?"
    pattern = prefix + label_re + suffix + r"(?P<cells>" + cells + r"(?:[ \t]+" + cells + r")*)[ \t]*$"
    result = []
    for match in re.finditer(pattern, text):
        raw = match.group("cells")
        tokens = re.findall(NUMBER, raw)
        line_index = text[:match.start("cells")].count("\n")
        result.append({"source_label": label, "raw_text": match.group(0).strip(), "source_page": lines[line_index]["page"],
                       "cells": tokens, "values": [number(x) for x in tokens]})
    return result


def get_row(lines, labels, widths):
    found = []
    for label in labels:
        found += [r for r in row_matches(lines, label) if len(r["cells"]) in widths]
    unique = {tuple(r["values"]): r for r in found}
    if len(unique) != 1:
        raise ValueError(f"行缺失或数值冲突：{labels[0]}，匹配{len(unique)}种")
    return next(iter(unique.values()))


def equality(left, right, label, tolerance):
    if len(left) != len(right):
        raise ValueError(label + "列数不同")
    error = max(abs(a - b) for a, b in zip(left, right))
    if error > tolerance:
        raise ValueError(f"{label}不成立，最大差额{error}")
    return {"check": label, "maximum_error_in_reported_units": str(error), "tolerance_in_reported_units": str(tolerance)}


def make_fact(metric, row, layout, multiplier, currency):
    eps = "EPS" in metric
    ci = layout["column"]
    return {"metric_id": metric, "metric_value": float(row["values"][ci] * (1 if eps else multiplier)),
            "unit": ("CNY_PER_SHARE" if eps else "CNY") if currency else ("REPORTED_YUAN_PER_SHARE_CURRENCY_UNSPECIFIED" if eps else "REPORTED_YUAN_CURRENCY_UNSPECIFIED"),
            "statement_scope": "CONSOLIDATED", "period_scope": "YEAR_TO_DATE", "source_page": row["source_page"],
            "source_label": row["source_label"], "source_raw_value": row["cells"][ci], "source_raw_row": row["raw_text"],
            "selected_numeric_column_one_based": ci + 1, "all_numeric_cells_for_verification": row["cells"],
            "source_unit_multiplier": 1 if eps else multiplier, "currency_explicit_in_statement": currency,
            "previous_year_comparative_is_original_prior_vintage": False,
            "strategy_input_status": "PASS_EXPLICIT_CNY_CURRENT_VINTAGE" if currency else "NO_VIEW_CURRENCY_NOT_EXPLICIT"}


def parse_summary_q1(pages, source):
    year = int(str(source["report_period"])[:4])
    candidates = [(i + 1, p) for i, p in enumerate(pages) if "统称\"本集团\"" in compact(p).replace("“", '"').replace("”", '"') and "以人民币百万元列示" in compact(p)]
    if len(candidates) != 1:
        raise ValueError("建行摘要中的集团范围、人民币百万元不唯一")
    pi, page = candidates[0]
    h = compact(page)
    expected = f"截至{year}年3月31日止三个月截至{year-1}年3月31日止三个月"
    if expected not in h:
        raise ValueError("建行摘要未证明本期与上期累计三个月")
    lines = [{"page": pi, "text": x} for x in norm(page).splitlines() if x.strip()]
    layout = {"column": 0, "width": 3, "layout": "EXPLICIT_CONSOLIDATED_Q1_SUMMARY", "header": expected}
    facts = []
    for metric in ["OPERATING_REVENUE_YTD", "PARENT_NET_PROFIT_YTD", "BASIC_EPS_YTD"]:
        fact = make_fact(metric, get_row(lines, LABELS[metric], [3]), layout, 1000000, True)
        fact["statement_scope"] = "EXPLICIT_CONSOLIDATED_SUMMARY"
        facts.append(fact)
    return {"status": "PASS_THREE_EXPLICIT_CONSOLIDATED_SUMMARY_FACTS", "facts": facts, "layout": layout,
            "identities": [], "missing_core_metrics": ["OPERATING_PROFIT_YTD"], "summary_is_full_income_statement": False}


def parse_document(pages, source):
    checked = identity(pages, source)
    if not checked["status"].startswith("PASS_"):
        return {"status": checked["status"], "identity": checked, "facts": []}
    try:
        year = int(str(source["report_period"])[:4])
        if source["ts_code"] == "601939.SH" and pd.Timestamp(source["report_period"]).month == 3:
            return {**parse_summary_q1(pages, source), "identity": checked}
        lines = statement_lines(pages, year)
        if not lines:
            raise ValueError("没有明确的集团利润表起点")
        layout = header_and_layout(lines, year)
        header = layout["header"]
        currency = "人民币" in header
        if "百万元" in header:
            multiplier = 1000000
        elif "单位:元" in header or "单位:人民币元" in header:
            multiplier = 1
        else:
            raise ValueError("原始金额单位尚未证明")
        rows = {}
        for metric in CORE + ["OPERATING_EXPENSE_YTD", "TOTAL_NET_PROFIT_YTD"]:
            if metric == "PARENT_NET_PROFIT_YTD" and source["ts_code"] == "000001.SZ":
                continue
            widths = [2] if layout["group_company_columns"] and metric in ["PARENT_NET_PROFIT_YTD", "BASIC_EPS_YTD"] else [layout["width"]]
            rows[metric] = get_row(lines, LABELS[metric], widths)
        tolerance = Decimal("1.01") if multiplier == 1000000 else Decimal("0.02")
        rev = rows["OPERATING_REVENUE_YTD"]["values"]
        exp = rows["OPERATING_EXPENSE_YTD"]["values"]
        operating = rows["OPERATING_PROFIT_YTD"]["values"]
        checks = []
        if source["ts_code"] == "000001.SZ":
            for m in ["PRE_IMPAIRMENT_OPERATING_PROFIT_YTD", "CREDIT_IMPAIRMENT_YTD", "OTHER_IMPAIRMENT_YTD"]:
                rows[m] = get_row(lines, LABELS[m], [layout["width"]])
            pre = rows["PRE_IMPAIRMENT_OPERATING_PROFIT_YTD"]["values"]
            checks += [equality([a+b for a,b in zip(rev, exp)], pre, "收入加负数经营支出等于减值前利润", tolerance)]
            checks += [equality([a+b+c for a,b,c in zip(pre, rows["CREDIT_IMPAIRMENT_YTD"]["values"], rows["OTHER_IMPAIRMENT_YTD"]["values"])], operating, "减值前利润扣除减值后等于营业利润", tolerance)]
            summary = [{"page": i+1, "text": x} for i,p in enumerate(pages[:5]) for x in norm(p).splitlines() if x.strip()]
            parent = get_row(summary, LABELS["PARENT_NET_PROFIT_YTD"], [4])
            # 摘要为本季、同比、累计、同比；仅比较两个实际期间，百分比不作利润。
            checks += [equality([parent["values"][0], parent["values"][2]], [rows["TOTAL_NET_PROFIT_YTD"]["values"][0], rows["TOTAL_NET_PROFIT_YTD"]["values"][2]], "摘要归母利润与合并表本季及累计净利润相符", tolerance)]
            rows["PARENT_NET_PROFIT_YTD"] = parent
        else:
            direction = -1 if source["ts_code"] in BROKERS else 1
            checks += [equality([a+direction*b for a,b in zip(rev, exp)], operating, "收入按原始报表支出符号计算营业利润", tolerance)]
            width = 2 if layout["group_company_columns"] else layout["width"]
            rows["MINORITY_NET_PROFIT_YTD"] = get_row(lines, LABELS["MINORITY_NET_PROFIT_YTD"], [width])
            checks += [equality([a+b for a,b in zip(rows["PARENT_NET_PROFIT_YTD"]["values"], rows["MINORITY_NET_PROFIT_YTD"]["values"])], rows["TOTAL_NET_PROFIT_YTD"]["values"][:width], "归母净利润加少数股东损益等于集团净利润", tolerance)]
        facts = [make_fact(m, rows[m], layout, multiplier, currency) for m in CORE]
        return {"status": "PASS_CONSOLIDATED_CURRENT_YTD_NUMERICAL_FACTS" if currency else "PARTIAL_NUMERICAL_FACTS_CURRENCY_NOT_EXPLICIT",
                "identity": checked, "facts": facts, "layout": layout, "identities": checks,
                "statement_pages": sorted({r["page"] for r in lines}), "all_rows_for_verification": rows,
                "restatement_word_in_statement": "重述" in "\n".join(r["text"] for r in lines), "missing_core_metrics": []}
    except ValueError as error:
        return {"status": "NO_VIEW_UNPROVEN_STATEMENT_LAYOUT_OR_IDENTITY", "identity": checked, "facts": [], "reason": str(error)}


def ordinary_eps_diagnostic(pages, source, parsed):
    if source["ts_code"] != "000001.SZ" or not parsed["facts"]:
        return None
    facts = {r["metric_id"]: r for r in parsed["facts"]}
    texts = [{"page": i+1, "text": x} for i,p in enumerate(pages[:5]) for x in norm(p).splitlines() if x.strip()]
    selectors = {"shares": "截至披露前一交易日的公司总股本", "preferred_dividend": "支付的优先股股利", "perpetual_interest": "支付的永续债利息", "reported_latest_share_eps": "用最新股本计算的全面摊薄每股收益"}
    rows = {k: get_row(texts, [v], [1]) for k,v in selectors.items()}
    values = {k: r["values"][0] for k,r in rows.items()}
    parent = Decimal(str(facts["PARENT_NET_PROFIT_YTD"]["metric_value"]))
    ordinary = parent - values["preferred_dividend"] - values["perpetual_interest"]
    corrected = ordinary / values["shares"]
    naive = parent / values["shares"]
    if abs(corrected - values["reported_latest_share_eps"]) > Decimal("0.005"):
        raise ValueError("扣除其他权益工具回报后仍不符合原披露全面摊薄每股收益舍入范围")
    return {"announcement_id": source["announcement_id"], "ts_code": source["ts_code"], "report_period": source["report_period"],
            "status": "PASS_ORDINARY_PROFIT_AND_LATEST_SHARE_EPS_RECONCILIATION", "original_rows": rows,
            "parent_profit_cny": str(parent), "ordinary_profit_after_other_equity_returns_cny": str(ordinary),
            "naive_parent_profit_divided_by_latest_shares": str(naive), "corrected_ordinary_profit_divided_by_latest_shares": str(corrected),
            "relative_overstatement_if_other_equity_returns_not_deducted": str(naive/corrected-1),
            "reported_latest_share_eps": str(values["reported_latest_share_eps"]), "reported_basic_eps": facts["BASIC_EPS_YTD"]["metric_value"],
            "latest_share_count_is_proven_ytd_weighted_average_share_count": False,
            "preferred_and_perpetual_returns_are_ordinary_shareholder_cash_returns": False}


def input_files():
    return sorted((INVENTORY / "page_texts").glob("*.json")) + sorted((REPAIR / "page_texts").glob("*.json"))


def freeze():
    paths = [Path(__file__), ROOT / "docs/510300_FINANCIAL_ORIGINAL_FACTS_V1_1.md", ROOT / "tests/test_financial_original_facts_v1_1.py", REPAIR / "result.json"] + input_files()
    old = ROOT / "reports/research/510300_financial_original_facts_v1"
    paths += [ROOT / "research/financial_original_facts_v1.py", old / "result.json"] + sorted((old / "document_records").glob("*.json"))
    write_json(MANIFEST, {"registered_at": now(), "purpose": "SOURCE_FACTS_AND_DEFINITION_REPAIR_NO_RETURN_READ",
               "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}, exclusive=True)


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("本次金融原始事实提取已有结果，保留冻结记录")
    for item in json.loads(MANIFEST.read_text("utf-8"))["files"]:
        if hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("金融原始事实提取输入变化")
    facts, documents, eps = [], [], []
    for f in input_files():
        data = json.loads(f.read_text("utf-8"))
        s = data["source"]
        pdf = BASE/s["raw_path"]
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != s["sha256"]:
            raise RuntimeError("原始财报与来源记录不一致")
        parsed = parse_document(data["pages"], s)
        diagnostic = ordinary_eps_diagnostic(data["pages"], s, parsed)
        if diagnostic:
            eps.append(diagnostic)
        write_json(OUT/"document_records"/(s["announcement_id"]+".json"), {"source": s, **parsed}, exclusive=True)
        common = {k:s[k] for k in ["announcement_id", "ts_code", "sec_name", "report_period", "event_publication_date", "official_pdf_url", "sha256"]}
        facts += [{**common, **fact} for fact in parsed["facts"]]
        documents.append({**common, "status": parsed["status"], "fact_count": len(parsed["facts"]), "reason": parsed.get("reason")})
        print(f"金融原始报表：{s['sec_name']} {str(s['report_period'])[:10]}，{len(parsed['facts'])}项，{parsed['status']}", flush=True)
    frame = pd.DataFrame(facts)
    frame.to_parquet(OUT/"current_original_financial_facts.parquet", index=False)
    frame.to_csv(OUT/"金融原始累计盈利与每股收益.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(documents).to_csv(OUT/"逐份原始财报处理结果.csv", index=False, encoding="utf-8-sig")
    write_json(OUT/"ordinary_share_eps_reconciliation.json", {"rows": eps}, exclusive=True)
    result = {"completed_at": now(), "status": "FINANCIAL_SOURCE_REBUILD_PARTIAL_ADMISSION_RESEARCH_CONTINUES", "documents_examined": len(documents),
              "documents_with_numerical_facts": sum(d["fact_count"] > 0 for d in documents), "current_vintage_fact_rows": len(facts),
              "explicit_cny_fact_rows": int(frame.strategy_input_status.eq("PASS_EXPLICIT_CNY_CURRENT_VINTAGE").sum()),
              "ordinary_eps_reconciliations": len(eps), "document_status_counts": pd.Series([d["status"] for d in documents]).value_counts().to_dict(),
              "new_portfolio_evaluation": False, "all_index_financial_constituents_covered": False, "source_revisions_backfilled_into_history": False,
              "goal_sharpe_1_2_achieved": False, "documents": documents}
    write_json(OUT/"result.json", result, exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k != "documents"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融合并累计盈利和普通股每股收益原始口径重建")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
