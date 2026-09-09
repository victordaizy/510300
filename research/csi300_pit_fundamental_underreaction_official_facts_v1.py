from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence

import pdfplumber
import pypdfium2 as pdfium


PARSER_VERSION = "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_PDFIUM_PDFPLUMBER_V1_2_0"

REQUIRED_METRICS = (
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
    "OPERATING_CASH_FLOW_YTD",
    "ACCOUNTS_RECEIVABLE_END",
    "INVENTORY_END",
    "TOTAL_ASSETS_END",
    "TOTAL_LIABILITIES_END",
)

UNIT_MULTIPLIERS = {
    "元": Decimal("1"),
    "千元": Decimal("1000"),
    "万元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "亿元": Decimal("100000000"),
}
UNIT_PATTERN = r"百万元|千元|万元|亿元|元"
GLOBAL_UNIT_RE = re.compile(
    rf"(?:金额)?单位\s*(?:为|[:：])?\s*(?:人民币)?\s*(?P<unit>{UNIT_PATTERN})"
)
ROW_UNIT_RE = re.compile(rf"[（(]\s*(?P<unit>{UNIT_PATTERN})\s*[）)]")
NUMBER_RE = re.compile(
    r"(?<![\d.])(?P<number>[\-−－—]?\(?\d[\d,]*(?:\.\d+)?\)?)(?![\d.])"
    r"|(?<!\S)(?P<placeholder>[\-－—])(?=\s|$)"
)
CJK_WHITESPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")


@dataclass(frozen=True)
class MetricEvidence:
    metric_id: str
    metric_value_cny: float
    statement_scope: str
    value_period_scope: str
    source_page: int
    source_locator: str
    source_label: str
    source_raw_value: str
    source_unit: str
    source_unit_multiplier: float
    source_method: str
    verification_status: str


@dataclass(frozen=True)
class ParsedCandidate:
    value_cny: Decimal
    page_number: int
    label: str
    raw_value: str
    unit: str
    line_window: str
    match_span: tuple[int, int]
    value_span: tuple[int, int]
    selected_amount_index: int


@dataclass(frozen=True)
class SectionRange:
    name: str
    page_indices: tuple[int, ...]
    unit: str | None


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CJK_WHITESPACE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def logical_line_windows(page_text: str, maximum_lines: int = 5) -> Iterable[str]:
    lines = [normalize_text(line) for line in normalize_text(page_text).splitlines()]
    lines = [line for line in lines if line]
    for index in range(len(lines)):
        for width in range(1, maximum_lines + 1):
            end = index + width
            if end <= len(lines):
                yield " ".join(lines[index:end])


def parse_decimal(raw_value: str) -> Decimal:
    text = raw_value.strip().replace(",", "")
    if text in {"-", "－", "—"}:
        return Decimal("0")
    text = text.replace("−", "-").replace("－", "-").replace("—", "-")
    negative_parentheses = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"无法解析金额：{raw_value}") from error
    return -value if negative_parentheses else value


def _unit_candidates(text: str) -> list[str]:
    global_units = [match.group("unit") for match in GLOBAL_UNIT_RE.finditer(text)]
    if global_units:
        return global_units
    return [match.group("unit") for match in ROW_UNIT_RE.finditer(text)]


def detect_unique_unit(page_texts: Sequence[str], page_indices: Iterable[int]) -> str | None:
    units: list[str] = []
    for page_index in page_indices:
        units.extend(_unit_candidates(normalize_text(page_texts[page_index])))
    unique = list(dict.fromkeys(units))
    return unique[0] if len(unique) == 1 else None


def extract_pdf_page_texts(content: bytes) -> tuple[list[str], dict[str, Any]]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("响应没有%PDF-文件头")
    document = pdfium.PdfDocument(content)
    page_texts: list[str] = []
    page_errors: list[str] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            text_page = None
            try:
                text_page = page.get_textpage()
                page_texts.append(normalize_text(text_page.get_text_range()))
            except Exception as error:  # noqa: BLE001 - 逐页错误进入检查点
                page_texts.append("")
                page_errors.append(
                    f"PAGE_{page_index + 1}:{type(error).__name__}:{error}"
                )
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()
    finally:
        document.close()
    return page_texts, {
        "pdf_page_count": len(page_texts),
        "text_character_count": int(sum(len(text) for text in page_texts)),
        "page_errors": page_errors,
    }


def _heading_precedes_markers(
    page_texts: Sequence[str],
    index: int,
    headings: Sequence[str],
    markers: Sequence[str],
) -> bool:
    page_compact = compact_text(page_texts[index])
    combined = compact_text(
        "\n".join(page_texts[index : min(len(page_texts), index + 2)])
    )
    heading_positions = [
        page_compact.find(compact_text(heading))
        for heading in headings
        if page_compact.find(compact_text(heading)) >= 0
    ]
    marker_positions = [combined.find(compact_text(marker)) for marker in markers]
    return bool(
        heading_positions
        and all(position >= 0 for position in marker_positions)
        and min(heading_positions) < min(marker_positions)
    )


def locate_section(
    page_texts: Sequence[str],
    *,
    name: str,
    heading: str | Sequence[str],
    required_marker_groups: Sequence[Sequence[str]],
    terminators: Sequence[str],
) -> SectionRange:
    headings = (heading,) if isinstance(heading, str) else tuple(heading)
    heading_compact = [compact_text(item) for item in headings]
    start: int | None = None
    for page_index, page_text in enumerate(page_texts):
        if not any(item in compact_text(page_text) for item in heading_compact):
            continue
        if any(
            _heading_precedes_markers(page_texts, page_index, headings, marker_group)
            for marker_group in required_marker_groups
        ):
            start = page_index
            break
    if start is None:
        return SectionRange(name=name, page_indices=(), unit=None)

    end = len(page_texts)
    compact_terminators = [compact_text(item) for item in terminators]
    for page_index in range(start + 1, len(page_texts)):
        page_compact = compact_text(page_texts[page_index])
        if any(marker in page_compact for marker in compact_terminators):
            end = page_index
            break
    # 财务报表经常在同一页结束上一张表并开始下一张表，因此保留终止标题所在页。
    page_indices = tuple(range(start, min(len(page_texts), end + 1)))
    unit = detect_unique_unit(page_texts, page_indices[:3])
    return SectionRange(name=name, page_indices=page_indices, unit=unit)


def locate_statement_sections(page_texts: Sequence[str]) -> dict[str, SectionRange]:
    balance = locate_section(
        page_texts,
        name="CONSOLIDATED_BALANCE_SHEET",
        heading=("合并资产负债表", "合并及公司资产负债表"),
        required_marker_groups=(("流动资产", "货币资金"),),
        terminators=(
            "母公司资产负债表",
            "合并利润表",
            "合并及公司利润表",
            "合并损益表",
            "合并年初到报告期末利润表",
            "合并本报告期利润表",
        ),
    )
    income = locate_section(
        page_texts,
        name="CONSOLIDATED_INCOME_STATEMENT",
        heading=(
            "合并利润表",
            "合并及公司利润表",
            "合并年初到报告期末利润表",
            "合并本报告期利润表",
        ),
        required_marker_groups=(
            ("营业收入", "营业成本"),
            ("营业总收入", "营业总成本"),
            ("营业收入", "营业利润"),
            ("营业总收入", "营业利润"),
        ),
        terminators=(
            "母公司利润表",
            "合并现金流量表",
            "合并及公司现金流量表",
            "合并年初到报告期末现金流量表",
            "合并本报告期现金流量表",
        ),
    )
    cash = locate_section(
        page_texts,
        name="CONSOLIDATED_CASH_FLOW_STATEMENT",
        heading=(
            "合并现金流量表",
            "合并及公司现金流量表",
            "合并年初到报告期末现金流量表",
            "合并本报告期现金流量表",
        ),
        required_marker_groups=(
            ("经营活动产生的现金流量", "销售商品", "经营活动产生的现金流量净额"),
        ),
        terminators=("母公司现金流量表", "合并所有者权益变动表"),
    )
    return {"balance": balance, "income": income, "cash": cash}


def _number_matches_after(text: str, start: int) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in NUMBER_RE.finditer(text, start):
        suffix = text[match.end() : match.end() + 1]
        if suffix == "%":
            continue
        matches.append(match)
    return matches


def _raw_amount(match: re.Match[str]) -> str:
    return str(match.group("number") or match.group("placeholder") or "")


def _contains_unrelated_label_before_first_amount(text: str) -> bool:
    compact = compact_text(text)
    compact = re.sub(
        r"(?:净)?亏损(?:总额)?以[“\"']?[-－—][”\"']?号填列",
        "",
        compact,
    )
    compact = compact.replace("附注", "").replace("注", "")
    compact = re.sub(r"人民币?(?:百万元|千元|万元|亿元|元)|百万元|千元|万元|亿元|元", "", compact)
    compact = re.sub(r"[（）()\[\]【】一二三四五六七八九十百0-9.,，:：、\-－—“”\"']", "", compact)
    return bool(re.search(r"[\u3400-\u9fff]", compact))


def _choose_amount_match(
    matches: Sequence[re.Match[str]],
    *,
    skip_statement_note_number: bool,
    amount_index: int,
) -> re.Match[str] | None:
    if not matches:
        return None
    first = matches[0]
    usable = list(matches)
    if skip_statement_note_number and len(usable) >= 2:
        raw_first = _raw_amount(first)
        raw_second = _raw_amount(usable[1])
        first_is_plain_integer = not any(char in raw_first for char in ",.()")
        try:
            first_value = abs(parse_decimal(raw_first))
            second_value = abs(parse_decimal(raw_second))
        except ValueError:
            first_value = Decimal("1000")
            second_value = Decimal("0")
        if (
            first_is_plain_integer
            and first_value <= 999
            and (raw_second in {"-", "－", "—"} or second_value >= 1000)
        ):
            usable = usable[1:]
    return usable[amount_index] if len(usable) > amount_index else None


def find_amount(
    page_texts: Sequence[str],
    page_indices: Iterable[int],
    *,
    label_patterns: Sequence[str],
    fallback_unit: str | None,
    skip_statement_note_number: bool,
    amount_index: int = 0,
    required_page_marker_any: Sequence[str] = (),
) -> ParsedCandidate | None:
    compiled = [re.compile(pattern) for pattern in label_patterns]
    for page_index in page_indices:
        page_text = page_texts[page_index]
        if required_page_marker_any and not any(
            compact_text(marker) in compact_text(page_text)
            for marker in required_page_marker_any
        ):
            continue
        page_units = _unit_candidates(page_text)
        page_unit = page_units[0] if len(set(page_units)) == 1 else None
        for window in logical_line_windows(page_text):
            for pattern in compiled:
                label_match = pattern.search(window)
                if label_match is None:
                    continue
                tail_limit = min(len(window), label_match.end() + 260)
                amount_matches = _number_matches_after(window[:tail_limit], label_match.end())
                if amount_matches and _contains_unrelated_label_before_first_amount(
                    window[label_match.end() : amount_matches[0].start()]
                ):
                    continue
                amount_match = _choose_amount_match(
                    amount_matches,
                    skip_statement_note_number=skip_statement_note_number,
                    amount_index=amount_index,
                )
                if amount_match is None:
                    continue
                row_tail = window[label_match.start() : min(len(window), label_match.end() + 80)]
                row_unit_matches = [
                    match.group("unit") for match in ROW_UNIT_RE.finditer(row_tail)
                ]
                unit = (
                    row_unit_matches[0]
                    if len(set(row_unit_matches)) == 1
                    else page_unit or fallback_unit
                )
                if unit not in UNIT_MULTIPLIERS:
                    continue
                raw_value = _raw_amount(amount_match)
                value_cny = parse_decimal(raw_value) * UNIT_MULTIPLIERS[unit]
                return ParsedCandidate(
                    value_cny=value_cny,
                    page_number=page_index + 1,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=window[:500],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=amount_index,
                )
    return None


def _summary_page_indices(
    page_texts: Sequence[str], sections: dict[str, SectionRange]
) -> tuple[int, ...]:
    section_starts = [
        section.page_indices[0]
        for section in sections.values()
        if section.page_indices
    ]
    first_statement = min(section_starts) if section_starts else len(page_texts)
    return tuple(range(min(first_statement, 35)))


def _metric_evidence(
    metric_id: str,
    candidate: ParsedCandidate,
    *,
    value_period_scope: str,
    section_name: str,
) -> MetricEvidence:
    locator = json.dumps(
        {
            "page": candidate.page_number,
            "section": section_name,
            "line_window": candidate.line_window,
            "label_span": list(candidate.match_span),
            "value_span": list(candidate.value_span),
            "parser_version": PARSER_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=candidate.page_number,
        source_locator=locator,
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(UNIT_MULTIPLIERS[candidate.unit]),
        source_method=(
            "PDFIUM_TEXT_LOGICAL_ROW_AMOUNT_INDEX_"
            f"{candidate.selected_amount_index}_AFTER_OPTIONAL_NOTE"
        ),
        verification_status="PASS_OFFICIAL_ORIGINAL_PDF_LABEL_VALUE_UNIT_VERIFIED",
    )


def find_explicit_blank_current_and_prior_inventory(
    page_texts: Sequence[str],
    table_rows: Iterable[tuple[int, int, int, Sequence[Any]]],
    *,
    fallback_unit: str | None,
) -> MetricEvidence | None:
    """把官方表格中当前期与上期都明确为空的“存货”单元格严格记为零。

    该规则只接受表格坐标证据，不使用线性文本中的空白，也不接受任一金额列含有
    数字、横线或其他字符。这样不会把提取失败、跨行错位或未知空白静默改成零。
    """

    for page_index, table_index, row_index, raw_cells in table_rows:
        cells = [normalize_text(cell) for cell in raw_cells]
        for label_index, cell in enumerate(cells):
            if compact_text(cell) != "存货":
                continue
            amount_cells = cells[label_index + 1 :]
            if len(amount_cells) < 2 or any(compact_text(value) for value in amount_cells):
                continue
            unit = detect_unique_unit(page_texts, [page_index]) or fallback_unit
            if unit not in UNIT_MULTIPLIERS:
                continue
            locator = json.dumps(
                {
                    "page": page_index + 1,
                    "section": "CONSOLIDATED_BALANCE_SHEET",
                    "table_index": table_index,
                    "row_index": row_index,
                    "label_cell_index": label_index,
                    "row_cells": cells,
                    "validation_rule": (
                        "EXACT_INVENTORY_LABEL_AND_ALL_TRAILING_CURRENT_AND_PRIOR_"
                        "AMOUNT_CELLS_EXPLICITLY_BLANK"
                    ),
                    "parser_version": PARSER_VERSION,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return MetricEvidence(
                metric_id="INVENTORY_END",
                metric_value_cny=0.0,
                statement_scope="CONSOLIDATED_ONLY",
                value_period_scope="PERIOD_END",
                source_page=page_index + 1,
                source_locator=locator,
                source_label=cell,
                source_raw_value="[CURRENT_AND_PRIOR_AMOUNT_CELLS_BLANK]",
                source_unit=unit,
                source_unit_multiplier=float(UNIT_MULTIPLIERS[unit]),
                source_method=(
                    "PDFPLUMBER_TABLE_EXPLICIT_BLANK_CURRENT_AND_PRIOR_CELLS_AS_ZERO"
                ),
                verification_status=(
                    "PASS_OFFICIAL_ORIGINAL_PDF_TABLE_BLANK_CURRENT_AND_PRIOR_"
                    "CELLS_VERIFIED_AS_ZERO"
                ),
            )
    return None


def extract_pdf_table_rows(
    content: bytes, page_indices: Iterable[int]
) -> tuple[list[tuple[int, int, int, Sequence[Any]]], list[str]]:
    rows: list[tuple[int, int, int, Sequence[Any]]] = []
    errors: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as document:
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(document.pages):
                errors.append(f"PAGE_{page_index + 1}:OUT_OF_RANGE")
                continue
            try:
                tables = document.pages[page_index].extract_tables()
            except Exception as error:  # noqa: BLE001 - 错误进入解析收据，不改变缺失状态
                errors.append(f"PAGE_{page_index + 1}:{type(error).__name__}:{error}"[:1000])
                continue
            for table_index, table in enumerate(tables):
                for row_index, row in enumerate(table or []):
                    if isinstance(row, (list, tuple)):
                        rows.append((page_index, table_index, row_index, row))
    return rows, errors


def extract_metrics_from_page_texts(
    page_texts: Sequence[str], *, period_type: str | None = None
) -> dict[str, Any]:
    sections = locate_statement_sections(page_texts)
    known_statement_units = [
        section.unit for section in sections.values() if section.unit is not None
    ]
    statement_unit_consensus = (
        known_statement_units[0] if len(set(known_statement_units)) == 1 else None
    )
    summary_pages = _summary_page_indices(page_texts, sections)
    summary_unit = detect_unique_unit(page_texts, summary_pages)
    summary_text_without_period_end = compact_text(
        "\n".join(page_texts[index] for index in summary_pages)
    ).replace("本报告期末", "")
    q3_summary_has_current_and_ytd_columns = bool(
        str(period_type or "").upper() == "Q3"
        and "本报告期" in summary_text_without_period_end
        and any(
            marker in summary_text_without_period_end
            for marker in ("年初至报告期末", "年初至本报告期末")
        )
    )
    income_section_text = compact_text(
        "\n".join(page_texts[index] for index in sections["income"].page_indices)
    )
    q3_current_quarter_markers = (
        "7-9月",
        "7—9月",
        "7－9月",
        "7至9月",
        "本报告期金额",
    )
    q3_year_to_date_markers = (
        "1-9月",
        "1—9月",
        "1－9月",
        "1至9月",
        "年初至报告期",
        "年前三季度",
    )
    q3_income_has_current_and_ytd_columns = bool(
        str(period_type or "").upper() == "Q3"
        and any(marker in income_section_text for marker in q3_current_quarter_markers)
        and any(marker in income_section_text for marker in q3_year_to_date_markers)
    )
    income_amount_index = 2 if q3_income_has_current_and_ytd_columns else 0

    definitions = {
        "OPERATING_REVENUE_YTD": {
            "section": sections["income"],
            "patterns": (
                r"其中[:：]?营业收入",
                r"一[、.]营业收入",
                r"(?<!总)营业收入",
            ),
            "period_scope": "YEAR_TO_DATE",
        },
        "OPERATING_PROFIT_YTD": {
            "section": sections["income"],
            "patterns": (r"[一二三四五六七八九十][、.]营业利润", r"营业利润"),
            "period_scope": "YEAR_TO_DATE",
        },
        "PARENT_NET_PROFIT_YTD": {
            "section": sections["income"],
            "patterns": (
                r"(?:\d+[.、]?\s*)?归属于母公司(?:所有者|股东)的净利润",
                r"归属于上市公司股东的净利润",
            ),
            "period_scope": "YEAR_TO_DATE",
        },
        "OPERATING_CASH_FLOW_YTD": {
            "section": sections["cash"],
            "patterns": (r"经营活动产生的现金流量净额",),
            "period_scope": "YEAR_TO_DATE",
        },
        "ACCOUNTS_RECEIVABLE_END": {
            "section": sections["balance"],
            "patterns": (r"(?<!应收票据及)(?<!其他)应收账款",),
            "period_scope": "PERIOD_END",
        },
        "INVENTORY_END": {
            "section": sections["balance"],
            "patterns": (r"(?<!周转)存货",),
            "period_scope": "PERIOD_END",
        },
        "TOTAL_ASSETS_END": {
            "section": sections["balance"],
            "patterns": (r"(?<!流动)(?<!非流动)资产总计",),
            "period_scope": "PERIOD_END",
        },
        "TOTAL_LIABILITIES_END": {
            "section": sections["balance"],
            "patterns": (r"(?<!流动)(?<!非流动)负债合计",),
            "period_scope": "PERIOD_END",
        },
    }

    evidence: dict[str, MetricEvidence] = {}
    for metric_id, definition in definitions.items():
        section: SectionRange = definition["section"]
        candidate = find_amount(
            page_texts,
            section.page_indices,
            label_patterns=definition["patterns"],
            fallback_unit=section.unit or statement_unit_consensus,
            skip_statement_note_number=True,
            amount_index=(
                income_amount_index if section.name == "CONSOLIDATED_INCOME_STATEMENT" else 0
            ),
        )
        if candidate is not None:
            evidence[metric_id] = _metric_evidence(
                metric_id,
                candidate,
                value_period_scope=definition["period_scope"],
                section_name=section.name,
            )

    if "ACCOUNTS_RECEIVABLE_END" not in evidence:
        note_candidate = find_amount(
            page_texts,
            range(len(page_texts)),
            label_patterns=(
                r"(?<!应收票据及)(?<!其他)应收账款(?:[（(][a-zA-Z一二三四五六七八九十\d]+[）)])?",
            ),
            fallback_unit=statement_unit_consensus,
            skip_statement_note_number=True,
            required_page_marker_any=(
                "合并财务报表项目注释",
                "合并财务报表主要项目注释",
                "合并财务报表附注",
            ),
        )
        if note_candidate is not None:
            evidence["ACCOUNTS_RECEIVABLE_END"] = _metric_evidence(
                "ACCOUNTS_RECEIVABLE_END",
                note_candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_STATEMENT_NOTES",
            )

    core_candidate = find_amount(
        page_texts,
        summary_pages,
        label_patterns=(
            r"归属于上市公司股东的扣除非经常性损益的净利润",
            r"归属于上市公司股东的扣除非经常性损益的净利(?:润)?(?:[（(]元[）)])?",
            r"归属于上市公司股东的扣除非经",
            r"归属于母公司(?:所有者|股东)的扣除非经常性损益的净利润",
            r"归属于母公司(?:所有者|股东)的扣除非经常性损益的净利(?:润)?(?:[（(]元[）)])?",
            r"扣除非经常性损益后的净利润",
        ),
        fallback_unit=summary_unit,
        skip_statement_note_number=False,
        amount_index=1 if q3_summary_has_current_and_ytd_columns else 0,
        required_page_marker_any=(
            ("年初至报告期末", "年初到报告期末", "年初至本报告期末")
            if str(period_type or "").upper() == "Q3"
            else ()
        ),
    )
    if core_candidate is not None:
        evidence["CORE_PARENT_NET_PROFIT_YTD"] = _metric_evidence(
            "CORE_PARENT_NET_PROFIT_YTD",
            core_candidate,
            value_period_scope="YEAR_TO_DATE",
            section_name="MAIN_FINANCIAL_HIGHLIGHTS",
        )

    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in evidence]
    return {
        "parser_version": PARSER_VERSION,
        "document_complete": not missing_metrics,
        "document_status": (
            "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
            if not missing_metrics
            else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
        ),
        "metrics": [asdict(evidence[metric]) for metric in REQUIRED_METRICS if metric in evidence],
        "missing_metrics": missing_metrics,
        "sections": {
            key: {
                "name": section.name,
                "start_page": section.page_indices[0] + 1 if section.page_indices else None,
                "end_page": section.page_indices[-1] + 1 if section.page_indices else None,
                "unit": section.unit,
            }
            for key, section in sections.items()
        },
    }


def extract_official_pdf_facts(
    content: bytes, *, period_type: str | None = None
) -> dict[str, Any]:
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    page_texts, text_receipt = extract_pdf_page_texts(content)
    result = extract_metrics_from_page_texts(page_texts, period_type=period_type)
    table_fallback_receipt: dict[str, Any] = {
        "rule": "EXPLICIT_BLANK_CURRENT_AND_PRIOR_INVENTORY_TABLE_CELLS_AS_ZERO",
        "status": "NOT_REQUIRED_TEXT_EXTRACTION_PRESENT",
        "page_errors": [],
    }
    if "INVENTORY_END" in result["missing_metrics"]:
        balance = locate_statement_sections(page_texts)["balance"]
        table_rows, table_errors = extract_pdf_table_rows(content, balance.page_indices)
        blank_inventory = find_explicit_blank_current_and_prior_inventory(
            page_texts,
            table_rows,
            fallback_unit=balance.unit,
        )
        table_fallback_receipt = {
            "rule": "EXPLICIT_BLANK_CURRENT_AND_PRIOR_INVENTORY_TABLE_CELLS_AS_ZERO",
            "status": (
                "PASS_EXPLICIT_BLANK_TABLE_CELLS_VERIFIED_AS_ZERO"
                if blank_inventory is not None
                else "NO_MATCH_INVENTORY_REMAINS_MISSING"
            ),
            "examined_balance_table_row_count": len(table_rows),
            "page_errors": table_errors,
        }
        if blank_inventory is not None:
            metrics = {row["metric_id"]: row for row in result["metrics"]}
            metrics[blank_inventory.metric_id] = asdict(blank_inventory)
            result["metrics"] = [
                metrics[metric] for metric in REQUIRED_METRICS if metric in metrics
            ]
            result["missing_metrics"] = [
                metric for metric in REQUIRED_METRICS if metric not in metrics
            ]
            result["document_complete"] = not result["missing_metrics"]
            result["document_status"] = (
                "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
                if result["document_complete"]
                else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
            )
    result.update(
        {
            "official_pdf_sha256": pdf_sha256,
            "official_pdf_size_bytes": len(content),
            "table_fallback_receipt": table_fallback_receipt,
            **text_receipt,
        }
    )
    return result
