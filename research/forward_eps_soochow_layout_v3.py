"""补充没有总标题的财务表及归母行换行，保留第二版已成功事实。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pypdfium2 as pdfium

from research import forward_eps_soochow_facts_v1 as original
from research import forward_eps_soochow_layout_v2 as previous
from research.financial_annual_components_v1 import norm, compact, decimal
from research.forward_eps_guosen_history_v1 import NUM


def date_with_author(first: str) -> str:
    try:
        return original.report_date(first)
    except ValueError:
        text = norm(first)
        pattern = r"(?m)^\s*(?:\[Table_Author\]\s*)?(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*\n\s*(?:(?:首席|高级|资深)?证券分析师|分析师)"
        values = {date(*map(int, m.groups())).isoformat() for m in re.finditer(pattern, text)}
        if len(values) != 1:
            raise ValueError("原件落款日与作者栏关系不唯一")
        return values.pop()


def headers(text: str) -> list[dict]:
    result = []
    for line in re.finditer(r"(?m)^.*$", text):
        matches = list(re.finditer(r"(?<!\d)(20\d{2})\s*([AEae]?)(?!\d)", line.group()))
        if not 3 <= len(matches) <= 6:
            continue
        years = [(int(m.group(1)), m.group(2).upper()) for m in matches]
        raw_years = [x[0] for x in years]
        predicted = [i for i, (_, flag) in enumerate(years) if flag == "E"]
        if raw_years != list(range(raw_years[0], raw_years[0] + len(raw_years))):
            continue
        if not predicted or predicted != list(range(predicted[0], len(years))):
            continue
        tail = line.group()[matches[-1].end():].strip()
        if tail:
            continue
        result.append({"position": line.start(), "header": [[y, flag] for y, flag in years],
                       "header_raw": line.group()[matches[0].start():].strip(), "line": line.group().strip()})
    return result


def rows_with_headers(text: str, field: str) -> list[dict]:
    head = headers(text)
    if field == "eps":
        pattern = r"(?<![A-Za-z一-龥])(?P<label>EPS-最新摊薄|每股收益|EPS)(?:\s*\((?P<unit>元/股|元)\))?(?=\s)"
    elif field == "pe":
        pattern = r"(?<![A-Za-z一-龥])(?P<label>P/E|PE)(?:\s*\((?P<unit>倍|现价&最新摊薄)\))?(?=\s)"
    elif field == "profit":
        pattern = r"(?<![一-龥])(?P<label>归属于母公司的净利\s*润|归属母公司净利\s*润|归母净利\s*润|净利润)(?:\s*\((?P<unit>人民币百万元|百万元)\))?(?=\s)"
    else:
        raise ValueError("未知预测行类型")
    result = []
    for match in re.finditer(pattern, text):
        eligible = [x for x in head if x["position"] < match.start()]
        if not eligible:
            continue
        selected = eligible[-1]
        width = len(selected["header"])
        end = text.find("\n", match.end())
        rest = text[match.end():end if end >= 0 else len(text)]
        pattern_values = r"^\s+" + r"\s+".join("(" + NUM + ")" for _ in range(width)) + r"(?=\s*(?:[^\d\s,.%+\-()]|$))"
        values = re.match(pattern_values, rest)
        if values is None:
            continue
        if field == "profit":
            label = compact(match.group("label"))
            unit = match.group("unit")
            if unit is None:
                if label == "净利润":
                    continue
                headings = list(re.finditer(r"(?:利润表|损益表)\s*\((?:人民币)?百万元\)", text[:match.start()]))
                if not headings:
                    continue
                unit = headings[-1].group()
            label_canonical = "净利润" if label == "净利润" else "归母净利润"
        else:
            unit = match.group("unit")
            label_canonical = match.group("label")
        result.append({"label": label_canonical, "label_as_reported": match.group("label"),
                       "unit_as_reported": unit, "header": selected["header"], "header_raw": selected["header_raw"],
                       "raw": text[match.start():match.end()] + rest[:values.end()],
                       "cells": list(values.groups()), "values": [str(decimal(x)) for x in values.groups()]})
    return result


def choose_optional(candidates: list[dict], header: list) -> dict | None:
    matching = [r for r in candidates if r["header"] == header]
    if not matching:
        return None
    keys = {(r["label"], tuple(r["values"])) for r in matching}
    return matching[0] if len(keys) == 1 else None


def appendix_page_selected(text: str) -> bool:
    packed = compact(text)
    named = re.search(r"(?:财务预测表|盈利预测表)", packed)
    financial = re.search(r"(?:资产负债表|利润表|损益表)", packed)
    eps = re.search(r"(?:每股收益|EPS)", packed)
    forecast = re.search(r"20\d{2}[Ee]", packed)
    return bool(named or (financial and eps and forecast))


def appendices(pdf_path: Path, pages: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    eps, profits, pe = [], [], []
    selected = [i for i, text in enumerate(pages) if appendix_page_selected(text)]
    if not selected:
        return eps, profits, pe
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for number in selected:
            page = document[number]
            text_page = page.get_textpage()
            width, height = page.get_size()
            try:
                for label, left, right in [("左半页", 0, width / 2), ("右半页", width / 2, width)]:
                    text = norm(text_page.get_text_bounded(left=left, bottom=0, right=right, top=height))
                    coordinate = {"page": number + 1, "source_region": label, "bbox_pdf_points": [left, 0, right, height]}
                    eps.extend({**item, **coordinate} for item in rows_with_headers(text, "eps"))
                    profits.extend({**item, **coordinate} for item in rows_with_headers(text, "profit"))
                    pe.extend({**item, **coordinate} for item in rows_with_headers(text, "pe"))
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    return eps, profits, pe


def parse(pages: list[str], metadata: dict, row: dict, pdf_path: Path) -> dict:
    try:
        kept = previous.parse(pages, metadata, row, pdf_path)
        kept["preserved_previous_adapter"] = kept["adapter"]
        kept["adapter"] = "V2_SUCCESS_PRESERVED_EXACTLY"
        return kept
    except ValueError as exc:
        v2_reason = str(exc)
    if metadata["info_code"] != row["infoCode"] or str(metadata["company_code"]) != "80000031":
        raise ValueError("原件机构或编号不符")
    if row["ts_code"][:6] not in [str(x.get("stock")) for x in metadata["security"]]:
        raise ValueError("详情证券不符")
    front = compact("\n".join(pages[:2]))
    if row["ts_code"][:6] not in front or not ("东吴证券" in front or "dwzq.com.cn" in front.lower()):
        raise ValueError("原件首页证券与机构身份未确认")
    internal = date_with_author(pages[0])
    clocks = [internal, str(metadata["notice_date"])[:10], str(metadata["eitime"])[:10], str(row["publishDate"])[:10]]
    for clock in clocks:
        date.fromisoformat(clock)
    eps, profits, pe = [], [], []
    for number, page_text in enumerate(pages[:3], 1):
        text = norm(page_text)
        for marker in re.finditer(r"盈利预测(?:与|预)估值", text):
            tail = re.sub(r"\[Table_[A-Za-z]+\]", "", text[marker.end():])
            stop = re.search(r"\n\s*(?:事件|事项|投资要点|点评)", tail)
            table = tail[:stop.start()] if stop else tail
            coordinate = {"page": number, "source_region": "完整页明确摘要表", "bbox_pdf_points": None}
            eps.extend({**item, **coordinate} for item in rows_with_headers(table, "eps"))
            profits.extend({**item, **coordinate} for item in rows_with_headers(table, "profit"))
            pe.extend({**item, **coordinate} for item in rows_with_headers(table, "pe"))
    if not eps:
        eps, profits, pe = appendices(pdf_path, pages)
    if not eps:
        raise ValueError("明确旧版或分栏预测EPS未识别")
    signatures = {(tuple(tuple(v) for v in item["header"]), tuple(item["values"])) for item in eps}
    if len(signatures) != 1:
        raise ValueError("原件包含多组不一致的预测EPS表")
    table = min(eps, key=lambda r: (r["page"], r["source_region"]))
    selected_profit = choose_optional(profits, table["header"])
    selected_pe = choose_optional(pe, table["header"])
    shares = original.optional_snapshot(norm(pages[0]), "总股本", "百万股")
    quote = original.optional_snapshot(norm(pages[0]), "收盘价", "元")
    facts = []
    for i, (year, flag) in enumerate(table["header"]):
        if flag != "E":
            continue
        basis_explicit = table["label"] == "EPS-最新摊薄"
        fact = {"report_id": row["infoCode"], "ts_code": row["ts_code"], "sec_name": row["stockName"],
                "institution_code": "80000031", "institution": "东吴证券", "target_fiscal_year": year,
                "eps_value_exact": table["values"][i], "eps_unit_as_reported": table["unit_as_reported"] or "原文EPS未单独注明单位",
                "source_eps_label": table["label_as_reported"], "eps_currency_iso_independently_proven": False,
                "eps_definition": "最新摊薄每股收益" if basis_explicit else "原文每股收益或EPS，未明确基本或摊薄定义",
                "latest_diluted_basis_explicit": basis_explicit, "source_page": table["page"],
                "source_eps_row": table["raw"].strip(), "source_eps_cell": table["cells"][i],
                "header": table["header"], "header_raw": table["header_raw"], "selected_column_one_based": i + 1,
                "report_internal_date": internal, "provider_notice_date": metadata["notice_date"],
                "provider_eitime": metadata["eitime"], "directory_publish_date": row["publishDate"],
                "conservative_information_date": max(clocks), "historical_immutable_snapshot_proven": False,
                "target_fiscal_year_already_ended": year < int(internal[:4]),
                "share_snapshot_million": shares["value"] if shares else None,
                "share_snapshot_raw": shares["raw"] if shares else None,
                "share_snapshot_is_rounded_report_value": True,
                "report_reference_close_exact": quote["value"] if quote else None,
                "report_reference_close_raw": quote["raw"] if quote else None,
                "actual_future_eps_used_as_predictor": False,
                "source_region": table["source_region"], "source_bbox_pdf_points": table["bbox_pdf_points"]}
        for key, optional in [("net_profit", selected_profit), ("pe", selected_pe)]:
            fact[key + "_value_exact"] = optional["values"][i] if optional else None
            fact[key + "_source_raw"] = optional["raw"].strip() if optional else None
            fact[key + "_source_label"] = optional["label"] if optional else None
            fact[key + "_source_label_as_reported"] = optional["label_as_reported"] if optional else None
            fact[key + "_source_unit_as_reported"] = optional["unit_as_reported"] if optional else None
            fact[key + "_source_page"] = optional["page"] if optional else None
        facts.append(fact)
    return {"facts": facts, "table": table, "report_internal_date": internal,
            "conservative_information_date": max(clocks), "date_fields_differ": len(set(clocks)) > 1,
            "share_snapshot": shares, "reference_quote": quote,
            "adapter": "ADDITIONAL_FINANCIAL_APPENDIX_AND_WRAPPED_PROFIT_V3", "previous_v2_reason": v2_reason}
