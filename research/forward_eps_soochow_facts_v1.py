"""读取东吴原件的明确年度EPS预测，保留股本定义差异和公开日期。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now, norm, compact, decimal
from research.forward_eps_guosen_history_v1 import NUM, identity, exact_row
from research.forward_eps_guosen_layout_v2 import annual_header
from research.forward_eps_guosen_history_v2 import parquet_frame


def report_date(first_page: str) -> str:
    """旧版日期在作者栏，新版在首页标题下，均要求完整独立日期行。"""
    text = norm(first_page)
    dates = set()
    pattern = r"(?m)^\s*(?:\[Table_Author\]\s*)?(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*$"
    for match in re.finditer(pattern, text):
        dates.add(date(*map(int, match.groups())).isoformat())
    if len(dates) != 1:
        raise ValueError("原件独立完整落款日未唯一识别")
    return dates.pop()


def optional_snapshot(text: str, label: str, unit: str) -> dict | None:
    pattern = r"(?m)^\s*" + re.escape(label) + r"\s*\(" + re.escape(unit) + r"\)\s*(" + NUM + r")\s*$"
    rows = [{"raw": m.group(0).strip(), "value": str(decimal(m.group(1)))} for m in re.finditer(pattern, text)]
    return rows[0] if len(rows) == 1 else None


def parse(pages: list[str], metadata: dict, row: dict) -> dict:
    if metadata["info_code"] != row["infoCode"] or str(metadata["company_code"]) != "80000031":
        raise ValueError("原件编号或东吴机构身份不符")
    if row["ts_code"][:6] not in [str(x.get("stock")) for x in metadata["security"]]:
        raise ValueError("目录证券与详情证券不符")
    front = compact("\n".join(pages[:2]))
    if row["ts_code"][:6] not in front or not ("东吴证券" in front or "dwzq.com.cn" in front.lower()):
        raise ValueError("原件首页证券与机构身份不符")
    internal = report_date(pages[0])
    clocks = [internal, str(metadata["notice_date"])[:10], str(metadata["eitime"])[:10], str(row["publishDate"])[:10]]
    for value in clocks:
        date.fromisoformat(value)
    available = max(clocks)
    tables = []
    for page_number, original in enumerate(pages[:3], 1):
        text = norm(original)
        for marker in re.finditer(r"盈利预测(?:与|预)估值", text):
            tail = text[marker.end():]
            first_metric = re.search(r"(?:营业总收入|营业收入|归母净利润)\s*\(", tail)
            if first_metric is None:
                continue
            head = re.sub(r"\[Table_[A-Za-z]+\]", "", tail[:first_metric.start()])
            header, header_raw = annual_header(head)
            years = [int(y) for y, flag in header]
            if not 3 <= len(years) <= 6 or years != list(range(years[0], years[0] + len(years))):
                raise ValueError("预测列数或连续绝对年度不符")
            forecast_columns = [i for i, (_, flag) in enumerate(header) if flag.upper() == "E"]
            if not forecast_columns or forecast_columns != list(range(forecast_columns[0], len(header))):
                raise ValueError("预测年度标志缺失或混入实际列")
            stop = re.search(r"\n\s*(?:事件|投资要点)|\[Table_Tag\]|\[Table_Summary\]", tail)
            body = tail[:stop.start()] if stop else tail
            eps = exact_row(body, ["EPS-最新摊薄", "每股收益"], len(header), "元/股")
            optional, errors = {}, {}
            for key, labels, unit in [("net_profit", ["归母净利润"], "百万元"),
                                      ("pe", ["P/E"], "(?:倍|现价&最新摊薄)")]:
                try:
                    optional[key] = exact_row(body, labels, len(header), unit)
                except ValueError as exc:
                    errors[key] = str(exc)
            tables.append({"page": page_number, "header": [[int(y), flag.upper()] for y, flag in header],
                           "header_raw": header_raw.strip(), "eps": eps, "optional": optional,
                           "optional_errors": errors})
    if len(tables) != 1:
        raise ValueError("前三页明确盈利预测表不是唯一一组：" + str(len(tables)))
    table = tables[0]
    first = norm(pages[0])
    shares = optional_snapshot(first, "总股本", "百万股")
    quote = optional_snapshot(first, "收盘价", "元")
    facts = []
    for i, (year, flag) in enumerate(table["header"]):
        if flag != "E":
            continue
        basis_explicit = table["eps"]["label"] == "EPS-最新摊薄"
        fact = {
            "report_id": row["infoCode"], "ts_code": row["ts_code"], "sec_name": row["stockName"],
            "institution_code": "80000031", "institution": "东吴证券", "target_fiscal_year": year,
            "eps_value_exact": table["eps"]["values"][i], "eps_unit_as_reported": "元/股",
            "source_eps_label": table["eps"]["label"], "eps_currency_iso_independently_proven": False,
            "eps_definition": "最新摊薄每股收益" if basis_explicit else "原文每股收益，未明确注明基本或摊薄定义",
            "latest_diluted_basis_explicit": basis_explicit,
            "source_page": table["page"], "source_eps_row": table["eps"]["raw"],
            "source_eps_cell": table["eps"]["cells"][i], "header": table["header"],
            "header_raw": table["header_raw"], "selected_column_one_based": i + 1,
            "report_internal_date": internal, "provider_notice_date": metadata["notice_date"],
            "provider_eitime": metadata["eitime"], "directory_publish_date": row["publishDate"],
            "conservative_information_date": available, "historical_immutable_snapshot_proven": False,
            "target_fiscal_year_already_ended": year < int(internal[:4]),
            "share_snapshot_million": shares["value"] if shares else None,
            "share_snapshot_raw": shares["raw"] if shares else None,
            "share_snapshot_is_rounded_report_value": True,
            "report_reference_close_exact": quote["value"] if quote else None,
            "report_reference_close_raw": quote["raw"] if quote else None,
            "actual_future_eps_used_as_predictor": False,
        }
        for key in ["net_profit", "pe"]:
            value = table["optional"].get(key)
            fact[key + "_value_exact"] = value["values"][i] if value else None
            fact[key + "_source_raw"] = value["raw"] if value else None
            fact[key + "_source_label"] = value["label"] if value else None
        facts.append(fact)
    return {"facts": facts, "table": table, "report_internal_date": internal,
            "conservative_information_date": available, "date_fields_differ": len(set(clocks)) > 1,
            "share_snapshot": shares, "reference_quote": quote}


def paths(pilot: bool) -> tuple[Path, Path]:
    source = "510300_forward_eps_soochow_source_pilot_v1" if pilot else "510300_forward_eps_soochow_originals_v1"
    output = "510300_forward_eps_soochow_pilot_facts_v1" if pilot else "510300_forward_eps_soochow_facts_v1"
    return ROOT / "reports/research" / source, ROOT / "reports/research" / output


def freeze(pilot: bool) -> None:
    source, out = paths(pilot)
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "tests/test_forward_eps_soochow_facts_v1.py",
             ROOT / "docs/510300_FORWARD_EPS_SOOCHOW_FACTS_V1.md", source / "selected_before_originals.parquet",
             ROOT / "research/forward_eps_guosen_history_v1.py", ROOT / "research/forward_eps_guosen_layout_v2.py",
             ROOT / "research/forward_eps_guosen_history_v2.py", ROOT / "research/financial_annual_components_v1.py"]
    save(out / "manifest.json", {"registered_at": now(), "pilot": pilot,
         "new_strategy_returns_read": False, "source_rules_fixed_before_full_extraction": True,
         "files": [identity(p) for p in files]}, exclusive=True)
    print("东吴原件EPS规则已登记", "样例" if pilot else "完整历史", flush=True)


def run(pilot: bool) -> None:
    source, out = paths(pilot)
    for item in read(out / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "EPS来源冻结输入变化"
    if not (source / "result.json").exists():
        raise RuntimeError("原件仍未完成，等待原采集进程")
    if (out / "result.json").exists():
        raise FileExistsError("东吴EPS提取已经完成")
    facts, records = [], []
    for row in pd.read_parquet(source / "selected_before_originals.parquet").to_dict("records"):
        aid = row["infoCode"]
        path = source / "document_records" / f"{aid}.json"
        record = {"report_id": aid, "ts_code": row["ts_code"], "status": "NO_VIEW_ORIGINAL_UNAVAILABLE"}
        if path.exists():
            original = read(path)
            pages = read(source / "page_texts" / f"{aid}.json")["pages"]
            try:
                parsed = parse(pages, original["provider_metadata"], row)
                for fact in parsed["facts"]:
                    assert compact(fact["source_eps_row"]) in compact(pages[fact["source_page"] - 1])
                    fact["raw_pdf_path"] = original["source"]["raw_pdf_path"]
                    fact["pdf_sha256"] = original["source"]["pdf_sha256"]
                record.update({"status": "ANNUAL_FORECAST_EPS_EXTRACTED", "source_record": path.relative_to(ROOT).as_posix(), **parsed})
            except ValueError as exc:
                record.update({"status": "NO_VIEW_LAYOUT_OR_BASIS_UNRESOLVED", "reason": str(exc)})
        save(out / "document_facts" / f"{aid}.json", record, exclusive=True)
        facts.extend(record.get("facts", []))
        records.append(record)
    frame = parquet_frame(facts)
    if len(frame):
        assert not frame.duplicated(["report_id", "target_fiscal_year"]).any()
        frame.to_parquet(out / "annual_eps_forecast_vintages.parquet", index=False)
        frame[["report_id", "ts_code", "institution", "target_fiscal_year", "eps_value_exact", "eps_definition", "net_profit_value_exact", "pe_value_exact", "conservative_information_date"]].to_csv(out / "年度前瞻EPS_原件明细.csv", index=False, encoding="utf-8-sig")
    save(out / "document_outcomes.json", {"rows": records}, exclusive=True)
    result = {"study_id": "510300_FORWARD_EPS_SOOCHOW_FACTS_V1", "completed_at": now(), "pilot": pilot,
              "status": "ABSOLUTE_YEAR_EPS_EXTRACTED_WITH_EXPLICIT_GAPS", "selected_reports": len(records),
              "parsed_reports": sum(x["status"] == "ANNUAL_FORECAST_EPS_EXTRACTED" for x in records),
              "forecast_eps_facts": len(facts), "companies_with_eps": int(frame.ts_code.nunique()) if len(frame) else 0,
              "reports_with_date_differences": sum(bool(x.get("date_fields_differ")) for x in records),
              "unresolved_reason_counts": dict(Counter(x.get("reason") for x in records if x.get("reason"))),
              "status_counts": dict(Counter(x["status"] for x in records)),
              "market_consensus": False, "exact_next_twelve_month_eps_constructed": False,
              "new_models_fit": 0, "new_accounts_generated": 0, "goal_achieved": False}
    save(out / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze(args.pilot) if args.freeze else run(args.pilot)
