"""财务单元格解析V2：独立报表切断合并状态，新表头不沿用旧表，支持本年列。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import re
import unicodedata

import pdfplumber
import pypdfium2 as pdfium


PARSER_VERSION = "FACTOR96_FINANCIAL_TABLE_CELL_CONTEXT_V2"
METRICS = ("PARENT_NET_PROFIT_YTD", "OPERATING_CASH_FLOW_YTD", "TOTAL_ASSETS_END")
UNITS = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "百万元": Decimal(1000000), "亿元": Decimal(100000000)}
UNIT_RE = re.compile(r"(?:金额)?单位\s*(?:为|[:：])?\s*(?:人民币)?(百万元|千元|万元|亿元|元)")
INLINE_UNIT_RE = re.compile(r"[（(](百万元|千元|万元|亿元|元)(?:人民币)?[）)]")
NUMBER_RE = re.compile(r"\(?[-+]?((?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?\)?")
HEADER_RE = re.compile(r"本报告期|报告期末|本期|当期|本年|当年|期末|年初|上年|上期|去年|期初|增减|变动|同比|增长|附注|20\d{2}|第一季度|第二季度|第三季度|第四季度")
SECTION_RE = re.compile(r"(?:合并(?:年初[至到]报告期末)?|母公司|本公司|公司)?(?:资产负债表|利润表|现金流量表|股东权益变动表|所有者权益变动表)(?:[（(]续[）)])?|(?:合并)?财务报表附注|非经常性损益项目|主要会计数据|主要财务数据|主要会计资料|会计数据和财务指标摘要")
LABELS = {
    "TOTAL_ASSETS_END": re.compile(r"(?:^|[、.\d)])(?:总资产|资产总计)(?:[（(]|$|注|\s)"),
    "PARENT_NET_PROFIT_YTD": re.compile(r"归属于(?:上市公司|母公司)(?:普通股)?(?:所有者|股东)(?:的)?净利润"),
    "OPERATING_CASH_FLOW_YTD": re.compile(r"经营活动(?:产生(?:[（(]使用[）)])?|使用)(?:的)?现金流量净额"),
}


def compact(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", value).replace("−", "-").replace("，", ",")


def parse_amount_cell(value):
    text = compact(value)
    if NUMBER_RE.fullmatch(text) is None or (text.startswith("(") != text.endswith(")")):
        return None
    try:
        amount = Decimal(text.strip("()").replace(",", ""))
    except InvalidOperation:
        return None
    if text.startswith("("):
        amount = -abs(amount)
    decimals = len(text.rstrip(")").split(".")[1]) if "." in text else 0
    return amount, Decimal(10) ** (-decimals)


@dataclass
class Context:
    section: str = "UNKNOWN"
    unit: str | None = None
    unit_page: int | None = None
    scope_page: int | None = None
    ytd_explicit: bool = False
    headers: list | None = None

    def advance(self, text, page_number):
        text = compact(text)
        matches = list(SECTION_RE.finditer(text))
        if matches:
            marker = matches[-1].group(0)
            statement = any(word in marker for word in ["资产负债表", "利润表", "现金流量表", "权益变动表"])
            if "附注" in marker or "非经常性" in marker or "权益变动表" in marker or (statement and not marker.startswith("合并")):
                section = "EXCLUDED"
            elif "资产负债表" in marker:
                section = "BALANCE"
            elif "现金流量表" in marker:
                section = "CASH"
            elif "利润表" in marker:
                section = "INCOME"
            else:
                section = "SUMMARY"
            explicit_continuation = "(续)" in marker and section == self.section and section != "EXCLUDED"
            if not explicit_continuation:
                self.unit, self.unit_page, self.headers = None, None, None
                self.ytd_explicit = False
            self.section, self.scope_page = section, page_number
            text = text[matches[-1].start():]
        if re.search(r"年初[至到]|1[-—－–至]9月|1月[-—－–至]9月|一至九月", text):
            self.ytd_explicit = True
        units = list(UNIT_RE.finditer(text))
        if units:
            self.unit, self.unit_page = units[-1].group(1), page_number


def header_is_current(header, metric, period_type, year, ytd_explicit):
    h = compact(header)
    if not h or any(word in h for word in ["上年", "上期", "去年", "期初", "年初余额", "增减", "变动", "增长", "同比", "附注"]):
        return False
    if period_type == "FY" and any(word in h for word in ["第一季度", "第二季度", "第三季度", "第四季度"]):
        return False
    years = set(re.findall(r"20\d{2}", h))
    if years and year is not None and str(year) not in years:
        return False
    named_current = bool(re.search(r"本报告期|报告期末|本期|当期|本年|当年|期末|年初[至到]", h))
    if not named_current and not years:
        return False
    if period_type == "Q3" and metric != "TOTAL_ASSETS_END":
        return "年初" in h or bool(re.search(r"1[-—－–至]9月", h)) or ytd_explicit
    return True


def _headers_above(table, rows, row_index, bbox, inherited):
    midpoint = (bbox[0] + bbox[2]) / 2
    choices = []
    for j in range(row_index):
        for k, cell in enumerate(table.rows[j].cells):
            if cell is None or not (cell[0] - .5 <= midpoint <= cell[2] + .5):
                continue
            text = compact(rows[j][k])
            if HEADER_RE.search(text) and len(text) <= 95:
                choices.append((cell[1], text))
    if choices:
        top = max(v[0] for v in choices)
        return " ".join(text for y, text in choices if top - y < 32)
    for left, right, header in inherited or []:
        if left - 2 <= midpoint <= right + 2:
            return header
    return ""


def _table_headers(table, rows):
    result = []
    for j, row in enumerate(rows[:6]):
        for k, raw in enumerate(row):
            cell = table.rows[j].cells[k]
            text = compact(raw)
            if cell is not None and HEADER_RE.search(text) and len(text) <= 95:
                result.append((cell[0], cell[2], text))
    return result


def _identify_metric(label, section):
    label = compact(label)
    for metric, pattern in LABELS.items():
        if not pattern.search(label):
            continue
        expected = {"TOTAL_ASSETS_END": "BALANCE", "PARENT_NET_PROFIT_YTD": "INCOME", "OPERATING_CASH_FLOW_YTD": "CASH"}[metric]
        if section in {expected, "SUMMARY"}:
            return metric
    return None


def _extract_table(page, number, table, context, period_type, year, seen):
    rows = table.extract()
    output = []
    for j, row in enumerate(rows):
        values = []
        for k, raw in enumerate(row):
            cell = table.rows[j].cells[k]
            parsed = parse_amount_cell(raw)
            if cell is not None and parsed is not None:
                values.append((k, cell, parsed, raw))
        if not values:
            continue
        leftmost = min(v[1][0] for v in values)
        if leftmost <= table.bbox[0] + 2:
            continue
        for k, cell, (amount, resolution), raw in values:
            key = (number, tuple(round(v, 3) for v in cell))
            if key in seen:
                continue
            # 数值单元格可能跨过标签的两行；按它的实际纵向边界恢复整个标签。
            label = page.crop((table.bbox[0], cell[1], leftmost, cell[3])).extract_text() or ""
            metric = _identify_metric(label, context.section)
            if metric is None:
                continue
            header = _headers_above(table, rows, j, cell, context.headers)
            if not header_is_current(header, metric, period_type, year, context.ytd_explicit):
                continue
            inline = INLINE_UNIT_RE.search(compact(label))
            unit = inline.group(1) if inline else context.unit
            if unit not in UNITS:
                continue
            seen.add(key)
            output.append({"metric_id": metric, "value_cny": amount * UNITS[unit], "resolution_cny": resolution * UNITS[unit],
                "raw_value": str(raw), "label": label, "unit": unit, "unit_page": number if inline else context.unit_page,
                "source_page": number, "section": context.section, "scope_page": context.scope_page,
                "current_column_header": header, "row_cells": row, "cell_bbox": list(cell), "table_bbox": list(table.bbox),
                "ytd_explicit": context.ytd_explicit})
    header = _table_headers(table, rows)
    if header:
        context.headers = header
    return output


def select_consistent_candidates(candidates):
    if not candidates:
        return None, "NO_VIEW_NO_EXPLICIT_CURRENT_CELL"
    ordered = sorted(candidates, key=lambda c: (c["resolution_cny"], c["section"] == "SUMMARY", c["source_page"]))
    best = ordered[0]
    for candidate in ordered[1:]:
        tolerance = (best["resolution_cny"] + candidate["resolution_cny"]) / 2 + Decimal("0.011")
        if abs(best["value_cny"] - candidate["value_cny"]) > tolerance:
            return None, "NO_VIEW_CONFLICTING_EXPLICIT_CURRENT_CELLS"
    return best, "PASS_CURRENT_CELL_SCOPE_UNIT_AND_DUPLICATE_CONSISTENCY"


def extract_official_pdf_facts(content, *, period_type=None, report_period=None, table_scope_pages=None):
    if not content.startswith(b"%PDF"):
        raise ValueError("不是PDF响应")
    period_type = str(period_type or "").upper()
    if period_type not in {"Q1", "H1", "Q3", "FY"}:
        raise ValueError("必须明确Q1/H1/Q3/FY报告期间")
    candidates, examined, errors, seen = [], [], [], set()
    context = Context()
    document = pdfium.PdfDocument(content)
    texts = []
    for i in range(len(document)):
        page = document[i]
        textpage = page.get_textpage()
        texts.append(textpage.get_text_range())
        textpage.close()
        page.close()
    document.close()
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        year = int(str(report_period)[:4]) if report_period else None
        if year is None:
            match = re.search(r"(20\d{2})年(?:第[一二三四1234]季度|年度|半年度)", compact(" ".join(texts[:3])))
            year = int(match.group(1)) if match else None
        for number, (page, text) in enumerate(zip(pdf.pages, texts), 1):
            reduced = compact(text)
            if (table_scope_pages is not None and number not in table_scope_pages) or not any(token in reduced for token in ["总资产", "资产总计", "归属于", "经营活动产生", "经营活动使用"]):
                context.advance(text, number)
                page.close()
                continue
            tables = page.find_tables()
            cursor = 0.
            examined.append(number)
            for table in sorted(tables, key=lambda t: (t.bbox[1], -(t.bbox[2] - t.bbox[0]))):
                if table.bbox[1] < cursor - 1:
                    continue
                if table.bbox[1] > cursor:
                    context.advance(page.crop((0, cursor, page.width, table.bbox[1])).extract_text() or "", number)
                candidates.extend(_extract_table(page, number, table, context, period_type, year, seen))
                cursor = table.bbox[3]
            if cursor < page.height:
                context.advance(page.crop((0, cursor, page.width, page.height)).extract_text() or "", number)
            # 候选只保留数值和坐标；及时释放单页对象缓存，长年报不累计整本文本对象。
            page.close()
        page_count = len(pdf.pages)
    metrics, decisions = [], {}
    for metric in METRICS:
        group = [c for c in candidates if c["metric_id"] == metric]
        chosen, status = select_consistent_candidates(group)
        decisions[metric] = {"status": status, "candidate_count": len(group)}
        if chosen is None:
            continue
        locator = {k: str(v) if isinstance(v, Decimal) else v for k, v in chosen.items()}
        metrics.append({"metric_id": metric, "metric_value_cny": float(chosen["value_cny"]),
            "statement_scope": "CONSOLIDATED_ONLY", "value_period_scope": "PERIOD_END" if metric == "TOTAL_ASSETS_END" else "YEAR_TO_DATE",
            "source_page": chosen["source_page"], "source_locator": json.dumps(locator, ensure_ascii=False),
            "source_label": chosen["label"], "source_raw_value": chosen["raw_value"], "source_unit": chosen["unit"],
            "source_unit_multiplier": float(UNITS[chosen["unit"]]), "source_method": PARSER_VERSION,
            "verification_status": status})
    return {"parser_version": PARSER_VERSION, "official_pdf_sha256": hashlib.sha256(content).hexdigest(),
        "official_pdf_size_bytes": len(content), "pdf_page_count": page_count, "period_type": period_type,
        "report_year": year, "metrics": metrics, "missing_metrics": [m for m in METRICS if m not in {r['metric_id'] for r in metrics}],
        "decisions": decisions, "candidates": [{k: str(v) if isinstance(v, Decimal) else v for k, v in c.items()} for c in candidates],
        "examined_table_pages": examined, "errors": errors, "network_requests": 0}
