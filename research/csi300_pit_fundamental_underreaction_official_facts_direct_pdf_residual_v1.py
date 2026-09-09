from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_v1 as direct
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser


RESIDUAL_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_DIRECT_PDF_RESIDUAL_DOCUMENT_STATISTICS_V1_5_0"
)
RESIDUAL_VERIFICATION_STATUS = (
    "PASS_OFFICIAL_ORIGINAL_PDF_DIRECT_RESIDUAL_RECONCILED"
)

SUMMARY_METRICS = (
    "OPERATING_REVENUE_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
    "OPERATING_CASH_FLOW_YTD",
)
FY_QUARTERLY_CONTAMINATION_METRICS = set(SUMMARY_METRICS)

SUMMARY_LABEL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "OPERATING_REVENUE_YTD": (
        re.compile(r"(?:^|[^总])营业收入"),
        re.compile(r"营业总收入"),
    ),
    "PARENT_NET_PROFIT_YTD": (
        re.compile(r"归属于上市公司(?:普通股)?股东的净利润"),
        re.compile(r"归属于母公司(?:所有者|股东)的净利润"),
        re.compile(r"归属于上市公司(?:普通股)?股东(?=\s*[-(（0-9])"),
    ),
    "CORE_PARENT_NET_PROFIT_YTD": (
        re.compile(
            r"归属于上市公司(?:普通股)?股东的扣除非经常性损益"
            r"(?:后的|的)?净(?:亏损[/／])?利润"
        ),
        re.compile(r"扣除非经常性损益后的净利润"),
        re.compile(
            r"归属于母公司(?:普通股)?股东的扣除非经常性损益"
            r"(?:后的|的)?净利润"
        ),
    ),
    "OPERATING_CASH_FLOW_YTD": (
        re.compile(
            r"经营活动(?:[（(]使用[）)]\s*[/／]\s*产生|"
            r"产生(?:[（(]使用[）)])?|使用)的现金流量净额"
        ),
        re.compile(r"经营活动.*?现金流量净额"),
    ),
}

INCOME_OPERATING_PROFIT_PATTERNS = (
    r"(?:[一二三四五六七八九十][、.]?\s*)?营业利润"
    r"(?:\s*[（(](?:亏损|损失)[^）)]{0,50}[）)])?",
    r"(?:[一二三四五六七八九十][、.]?\s*)?营业亏损",
    r"(?:[一二三四五六七八九十][、.]?\s*)?营业损失[/／]利润",
)
INCOME_NONOPERATING_INCOME_PATTERNS = (r"(?:加[:：]?\s*)?营业外收入",)
INCOME_NONOPERATING_EXPENSE_PATTERNS = (r"(?:减[:：]?\s*)?营业外支出",)
INCOME_PROFIT_TOTAL_PATTERNS = (
    r"(?:[一二三四五六七八九十][、.]?\s*)?利润总额"
    r"(?:\s*[（(]亏损总额[^）)]{0,50}[）)])?",
)
CASH_NET_PATTERNS = (
    r"经营活动产生的现金流量净额",
    r"经营活动使用的现金流量净额",
    r"经营活动[（(]使用[）)]\s*[/／]\s*产生的现金流量净额",
)
CASH_INFLOW_PATTERNS = (r"经营活动现金流入小计",)
CASH_OUTFLOW_PATTERNS = (r"经营活动现金流出小计",)
COMBINED_RECEIVABLE_PATTERNS = (r"应收票据(?:及|和)应收账款",)
PURE_BILLS_PATTERNS = (
    r"(?<!及)(?<!和)应收票据(?!及应收账款)(?!和应收账款)",
)

_UNIT_RE = re.compile(r"(?:人民币)?(?P<unit>亿元|百万元|万元|千元|元)")
_PAGE_UNIT_RE = re.compile(
    r"(?:金额)?单位\s*[:：]\s*(?:人民币)?(?P<unit>亿元|百万元|万元|千元|元)"
)
_PERCENT_RE = re.compile(r"[%％]|个百分点")
_NUMBER_FULL_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$"
)


def _compact(value: Any) -> str:
    return parser._v1_4.compact_financial_text(value)


def _normalize_cell(value: Any) -> str:
    return parser._v1_4.normalize_financial_text(value or "").strip()


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(metric["metric_id"]): metric
        for metric in result.get("metrics") or []
        if metric.get("metric_id")
    }


def _metric_value(result: dict[str, Any], metric_id: str) -> Decimal | None:
    metric = _metric_map(result).get(metric_id)
    if metric is None:
        return None
    try:
        return Decimal(str(metric["metric_value_cny"]))
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None


def _metric_anchor_unit(result: dict[str, Any], metric_id: str) -> str:
    metric = _metric_map(result).get(metric_id) or {}
    unit = str(metric.get("source_unit") or "元")
    return unit if unit in parser._base.UNIT_MULTIPLIERS else "元"


def _same_display(left: Decimal, right: Decimal, unit: str) -> bool:
    return parser._v1_4._same_display_identity(left, right, unit)


def _pages(start: int, end: int, count: int) -> tuple[int, ...]:
    return tuple(range(max(0, start), min(count, end)))


def _is_summary_page(page_text: str) -> bool:
    compact = _compact(page_text)
    main = any(
        marker in compact
        for marker in (
            "主要会计数据",
            "主要财务数据",
            "主要会计资料",
            "主要财务指标",
        )
    )
    metric_count = sum(
        marker in compact
        for marker in (
            "营业收入",
            "归属于上市公司股东的净利润",
            "扣除非经常性损益",
            "现金流量净额",
            "总资产",
        )
    )
    return main or metric_count >= 3


def _is_fy_quarterly_breakdown(page_text: str) -> bool:
    return parser._is_explicit_quarterly_breakdown(page_text)


def _source_page_is_quarterly(
    metric: dict[str, Any],
    page_texts: Sequence[str],
) -> bool:
    try:
        page_index = int(metric.get("source_page") or 0) - 1
    except (TypeError, ValueError):
        return False
    return (
        0 <= page_index < len(page_texts)
        and _is_fy_quarterly_breakdown(page_texts[page_index])
    )


def _metric_matches_label(metric_id: str, label: str) -> bool:
    compact = _compact(label)
    if metric_id == "OPERATING_REVENUE_YTD" and (
        "归属于" in compact
        or "扣除与主营业务无关" in compact
        or "不具备商业实质" in compact
        or ("扣除" in compact and "后的营业收入" in compact)
    ):
        return False
    if metric_id == "CORE_PARENT_NET_PROFIT_YTD" and (
        "非经常性损益项目" in compact or "非经常性损益影响" in compact
    ):
        return False
    return any(pattern.search(compact) for pattern in SUMMARY_LABEL_PATTERNS[metric_id])


def _unit_from_text(value: str) -> str | None:
    matches = [match.group("unit") for match in _UNIT_RE.finditer(value)]
    units = [unit for unit in matches if unit in parser._base.UNIT_MULTIPLIERS]
    return units[-1] if units else None


def _table_unit(page_text: str, label: str, table: Sequence[Sequence[Any]]) -> str | None:
    label_unit = _unit_from_text(label)
    if label_unit is not None:
        return label_unit
    table_prefix = " ".join(
        _normalize_cell(cell)
        for row in table[:5]
        for cell in row
        if cell is not None
    )
    table_match = _PAGE_UNIT_RE.search(table_prefix)
    if table_match is not None:
        return table_match.group("unit")
    page_match = _PAGE_UNIT_RE.search(page_text)
    if page_match is not None:
        return page_match.group("unit")
    unit = parser._base.detect_unique_unit((page_text,), (0,))
    return unit if unit in parser._base.UNIT_MULTIPLIERS else None


def _amount_from_cell(value: Any, unit: str) -> tuple[Decimal, str] | None:
    text = _normalize_cell(value)
    if not text or _PERCENT_RE.search(text):
        return None
    normalized = (
        text.replace("，", ",")
        .replace("．", ".")
        .replace("－", "-")
        .replace("—", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    normalized = re.sub(r"\s+", "", normalized)
    negative = normalized.startswith("(") and normalized.endswith(")")
    if negative:
        normalized = normalized[1:-1]
    if _NUMBER_FULL_RE.fullmatch(normalized):
        raw = f"({normalized})" if negative else normalized
    else:
        # PDF 表格偶尔把相邻两行金额合并在一个单元格中。当前逻辑行的金额
        # 永远是该单元格中第一个完整金额；后续金额由下一逻辑行处理。
        matches = parser._base._number_matches_after(text, 0)
        if not matches:
            return None
        raw = parser._raw_amount_with_closing_parenthesis(text, matches[0])
        if not parser._well_formed_summary_amount(raw):
            return None
    try:
        value_decimal = parser._base.parse_decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    return value_decimal * parser._base.UNIT_MULTIPLIERS[unit], raw


def _row_numeric_columns(
    row: Sequence[Any],
    unit: str,
) -> list[tuple[int, Decimal, str]]:
    numeric: list[tuple[int, Decimal, str]] = []
    for column_index, cell in enumerate(row):
        parsed = _amount_from_cell(cell, unit)
        if parsed is None:
            continue
        value, raw = parsed
        numeric.append((column_index, value, raw))
    return numeric


def _row_has_numeric_value(row: Sequence[Any]) -> bool:
    for cell in row:
        text = _normalize_cell(cell)
        if not text or _PERCENT_RE.search(text):
            continue
        normalized = (
            text.replace("，", ",")
            .replace("．", ".")
            .replace("－", "-")
            .replace("（", "(")
            .replace("）", ")")
        )
        normalized = re.sub(r"\s+", "", normalized).strip("()")
        if _NUMBER_FULL_RE.fullmatch(normalized) or parser._base._number_matches_after(
            text,
            0,
        ):
            return True
    return False


def _label_context(
    table: Sequence[Sequence[Any]],
    row_index: int,
) -> str:
    current = table[row_index]
    current_text = "".join(
        _normalize_cell(cell)
        for cell in current
        if cell is not None and not _row_has_numeric_value((cell,))
    )
    if current_text:
        prefix_parts: list[str] = []
        for index in range(row_index - 1, max(-1, row_index - 3), -1):
            row = table[index]
            if _row_has_numeric_value(row):
                break
            prefix_parts.insert(
                0,
                "".join(_normalize_cell(cell) for cell in row if cell is not None),
            )
        return "".join(prefix_parts) + current_text

    parts: list[str] = []
    for index in range(row_index - 1, max(-1, row_index - 3), -1):
        row = table[index]
        if _row_has_numeric_value(row):
            break
        parts.insert(0, "".join(_normalize_cell(cell) for cell in row if cell is not None))
    for index in range(row_index + 1, min(len(table), row_index + 3)):
        row = table[index]
        if _row_has_numeric_value(row):
            break
        parts.append("".join(_normalize_cell(cell) for cell in row if cell is not None))
    return "".join(parts)


def _column_header(
    table: Sequence[Sequence[Any]],
    row_index: int,
    column_index: int,
) -> str:
    return _compact(
        " ".join(
            _normalize_cell(table[index][column_index])
            for index in range(min(row_index, 7))
            if column_index < len(table[index]) and table[index][column_index] is not None
        )
    )


def _select_summary_amount(
    table: Sequence[Sequence[Any]],
    row_index: int,
    numeric: Sequence[tuple[int, Decimal, str]],
    *,
    period_type: str | None,
    page_text: str,
) -> tuple[int, Decimal, str] | None:
    if not numeric:
        return None
    if str(period_type or "").upper() != "Q3":
        return numeric[0]

    ytd_markers = (
        "年初至报告期末",
        "年初至本报告期末",
        "本年初至报告期末",
        "1-9月",
        "1—9月",
        "1－9月",
        "1~9月",
        "1～9月",
        "前三季度",
    )
    for candidate in numeric:
        header = _column_header(table, row_index, candidate[0])
        if any(marker in header for marker in ytd_markers) and not any(
            marker in header for marker in ("同比", "比上年", "增减", "变动")
        ):
            return candidate
    compact_page = _compact(page_text)
    has_current_quarter = any(
        marker in compact_page
        for marker in ("本报告期", "本季度", "7-9月", "7—9月", "7－9月")
    )
    has_ytd = any(marker in compact_page for marker in ytd_markers)
    if has_current_quarter and has_ytd and len(numeric) >= 2:
        return numeric[1]
    if len(numeric) == 1 and has_ytd:
        return numeric[0]
    return None


def _summary_evidence(
    metric_id: str,
    candidate: dict[str, Any],
    *,
    validation: dict[str, Any],
    source_context: dict[str, Any],
) -> parser._base.MetricEvidence:
    locator = {
        "page": candidate["page_number"],
        "section": "DIRECT_OFFICIAL_PDF_MAIN_ACCOUNTING_DATA",
        "table_index": candidate.get("table_index"),
        "row_index": candidate.get("row_index"),
        "value_column_index": candidate.get("column_index"),
        "label_context": candidate["label"][:700],
        "row_cells": candidate.get("row_cells"),
        "validation": validation,
        "source_context": source_context,
        "parser_version": RESIDUAL_PARSER_VERSION,
        "text_engine": candidate.get(
            "engine",
            "PDFPLUMBER_DIRECT_TABLE_DOCUMENT_STATISTICS",
        ),
    }
    return parser._base.MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate["value_cny"]),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope="YEAR_TO_DATE",
        source_page=int(candidate["page_number"]),
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=str(candidate["label"]),
        source_raw_value=str(candidate["raw_value"]),
        source_unit=str(candidate["unit"]),
        source_unit_multiplier=float(
            parser._base.UNIT_MULTIPLIERS[str(candidate["unit"])]
        ),
        source_method=str(
            candidate.get(
                "source_method",
                "PDFPLUMBER_DIRECT_MAIN_ACCOUNTING_DATA_"
                "SEMANTIC_YTD_COLUMN_RECONCILED",
            )
        ),
        verification_status=RESIDUAL_VERIFICATION_STATUS,
    )


def _summary_table_candidates(
    page_texts: Sequence[str],
    page_tables: dict[int, Sequence[Sequence[Sequence[Any]]]],
    *,
    period_type: str | None,
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_index, tables in sorted(page_tables.items()):
        if not 0 <= page_index < len(page_texts):
            continue
        page_text = page_texts[page_index]
        if not _is_summary_page(page_text):
            continue
        for table_index, raw_table in enumerate(tables):
            table = [list(row or []) for row in raw_table or []]
            if not table:
                continue
            table_text = " ".join(
                _normalize_cell(cell)
                for row in table
                for cell in row
                if cell is not None
            )
            if (
                str(period_type or "").upper() == "FY"
                and _is_fy_quarterly_breakdown(table_text)
            ):
                continue
            for row_index, row in enumerate(table):
                label = _label_context(table, row_index)
                matched_metrics = [
                    metric_id
                    for metric_id in SUMMARY_METRICS
                    if _metric_matches_label(metric_id, label)
                ]
                if not matched_metrics:
                    continue
                unit_context = "\n".join(
                    page_texts[index]
                    for index in range(max(0, page_index - 1), page_index + 1)
                )
                unit = _table_unit(unit_context, label, table)
                if unit not in parser._base.UNIT_MULTIPLIERS:
                    continue
                numeric = _row_numeric_columns(row, unit)
                selected = _select_summary_amount(
                    table,
                    row_index,
                    numeric,
                    period_type=period_type,
                    page_text=page_text,
                )
                if selected is None:
                    continue
                column_index, value_cny, raw_value = selected
                if str(period_type or "").upper() == "FY" and _is_fy_quarterly_breakdown(
                    page_text
                ):
                    compact_page = _compact(page_text)
                    breakdown_positions = [
                        compact_page.find(marker)
                        for marker in (
                            "分季度主要财务",
                            "第一季度",
                            "第二季度",
                        )
                        if compact_page.find(marker) >= 0
                    ]
                    breakdown_position = (
                        min(breakdown_positions) if breakdown_positions else -1
                    )
                    raw_position = compact_page.find(_compact(raw_value))
                    if (
                        breakdown_position >= 0
                        and raw_position >= 0
                        and raw_position > breakdown_position
                    ):
                        continue
                for metric_id in matched_metrics:
                    candidates[metric_id].append(
                        {
                            "metric_id": metric_id,
                            "value_cny": value_cny,
                            "page_number": page_index + 1,
                            "table_index": table_index,
                            "row_index": row_index,
                            "column_index": column_index,
                            "label": label,
                            "raw_value": raw_value,
                            "unit": unit,
                            "row_cells": [
                                _normalize_cell(cell) if cell is not None else None
                                for cell in row
                            ],
                            "engine": "PDFPLUMBER_DIRECT_TABLE_DOCUMENT_STATISTICS",
                            "source_method": (
                                "PDFPLUMBER_DIRECT_MAIN_ACCOUNTING_DATA_"
                                "SEMANTIC_YTD_COLUMN_RECONCILED"
                            ),
                        }
                    )
    return candidates


def _parsed_candidate_as_summary_candidate(
    metric_id: str,
    candidate: parser._base.ParsedCandidate,
    *,
    method: str,
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value_cny": candidate.value_cny,
        "page_number": candidate.page_number,
        "table_index": None,
        "row_index": None,
        "column_index": candidate.selected_amount_index,
        "label": candidate.label,
        "raw_value": candidate.raw_value,
        "unit": candidate.unit,
        "row_cells": None,
        "engine": "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
        "source_method": method,
    }


def _supplement_summary_text_candidates(
    candidates: dict[str, list[dict[str, Any]]],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> None:
    upper = min(40, len(page_texts))
    compact_pages = [_compact(page_texts[index]) for index in range(upper)]
    layout_pages = [
        parser._normalize_summary_numeric_layout(page_texts[index])
        for index in range(upper)
    ]
    main_markers = ("主要会计数据", "主要财务数据", "主要会计资料")
    for page_index in range(upper):
        compact_page = compact_pages[page_index]
        page_text = layout_pages[page_index]
        previous_compact = compact_pages[page_index - 1] if page_index > 0 else ""
        if not any(
            marker in compact_page or marker in previous_compact
            for marker in main_markers
        ):
            continue
        unit_context = "\n".join(
            page_texts[index]
            for index in range(max(0, page_index - 1), page_index + 1)
        )
        for metric_id in SUMMARY_METRICS:
            if candidates.get(metric_id):
                continue
            for pattern in SUMMARY_LABEL_PATTERNS[metric_id]:
                for label_match in pattern.finditer(page_text):
                    if parser._candidate_follows_explicit_quarterly_breakdown(
                        page_text,
                        label_match.start(),
                    ):
                        continue
                    if (
                        metric_id == "PARENT_NET_PROFIT_YTD"
                        and "净利润" not in label_match.group(0)
                    ):
                        next_page_prefix = (
                            compact_pages[page_index + 1][:800]
                            if page_index + 1 < upper
                            else ""
                        )
                        if "的净利润" not in next_page_prefix:
                            continue
                    if metric_id == "OPERATING_REVENUE_YTD":
                        label_prefix = _compact(page_text[
                            max(0, label_match.start() - 100) : label_match.start()
                        ])
                        if (
                            "扣除与主营业务无关" in label_prefix
                            or "不具备商业实质" in label_prefix
                            or ("扣除" in label_prefix and "后的" in label_prefix)
                        ):
                            continue
                    numeric_matches = parser._summary_numeric_matches(
                        page_text,
                        label_match.end(),
                        maximum_characters=600,
                    )
                    if not numeric_matches:
                        continue
                    amount_index = 0
                    if str(period_type or "").upper() == "Q3":
                        selected_index = parser._q3_summary_amount_index(
                            page_text,
                            numeric_matches,
                            context_text=unit_context + page_text[: label_match.start()],
                        )
                        if selected_index is None:
                            ytd_context = parser._has_q3_ytd_context(
                                unit_context + page_text[: label_match.start()]
                            )
                            current_placeholders = page_text[
                                label_match.end() : numeric_matches[0].start()
                            ].count("—")
                            if ytd_context and current_placeholders >= 2:
                                selected_index = 0
                        if (
                            selected_index is None
                            or selected_index >= len(numeric_matches)
                        ):
                            continue
                        amount_index = selected_index
                    amount_match = numeric_matches[amount_index]
                    raw_value = parser._raw_amount_with_closing_parenthesis(
                        page_text,
                        amount_match,
                    )
                    normalized_raw = raw_value.strip("()（）").replace(",", "")
                    if (
                        re.fullmatch(r"\d{4}", normalized_raw)
                        and 1900 <= int(normalized_raw) <= 2100
                    ):
                        continue
                    unit = _unit_from_text(
                        page_text[
                            max(0, label_match.start() - 80) : min(
                                len(page_text),
                                label_match.end() + 40,
                            )
                        ]
                    )
                    if unit is None:
                        unit = parser._summary_unit_near_candidate(
                            layout_pages,
                            page_index,
                            label_match.start(),
                        )
                    if unit is None:
                        page_unit_match = _PAGE_UNIT_RE.search(unit_context)
                        unit = (
                            page_unit_match.group("unit")
                            if page_unit_match is not None
                            else None
                        )
                    if unit not in parser._base.UNIT_MULTIPLIERS:
                        continue
                    try:
                        value_cny = (
                            parser._base.parse_decimal(raw_value)
                            * parser._base.UNIT_MULTIPLIERS[unit]
                        )
                    except (InvalidOperation, ValueError):
                        continue
                    candidates[metric_id].append(
                        {
                            "metric_id": metric_id,
                            "value_cny": value_cny,
                            "page_number": page_index + 1,
                            "table_index": None,
                            "row_index": None,
                            "column_index": amount_index,
                            "label": label_match.group(0),
                            "label_position": label_match.start(),
                            "raw_value": raw_value,
                            "unit": unit,
                            "row_cells": None,
                            "engine": "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
                            "source_method": (
                                "PDFIUM_DIRECT_MAIN_ACCOUNTING_DATA_"
                                "SEMANTIC_YTD_COLUMN_RECONCILED"
                            ),
                        }
                    )
    core = parser._multiline_summary_core_candidate(
        page_texts,
        period_type=period_type,
    )
    if core is not None and not parser._candidate_follows_explicit_quarterly_breakdown(
        page_texts[core.page_number - 1],
        core.match_span[0],
    ):
        candidates["CORE_PARENT_NET_PROFIT_YTD"].append(
            _parsed_candidate_as_summary_candidate(
                "CORE_PARENT_NET_PROFIT_YTD",
                core,
                method=(
                    "PDFIUM_DIRECT_MULTILINE_MAIN_ACCOUNTING_DATA_"
                    "SEMANTIC_YTD_COLUMN_RECONCILED"
                ),
            )
        )


def _unique_summary_candidates(
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidates:
        return None, {
            "raw_candidate_count": 0,
            "unique_value_count": 0,
        }
    earliest_page = min(int(candidate["page_number"]) for candidate in candidates)
    primary_page_candidates = [
        candidate
        for candidate in candidates
        if int(candidate["page_number"]) == earliest_page
    ]
    table_candidates = [
        candidate
        for candidate in primary_page_candidates
        if str(candidate.get("engine") or "").startswith("PDFPLUMBER")
    ]
    primary_candidates = table_candidates or primary_page_candidates
    if not table_candidates and any(
        candidate.get("label_position") is not None
        for candidate in primary_candidates
    ):
        first_label_position = min(
            int(candidate["label_position"])
            for candidate in primary_candidates
            if candidate.get("label_position") is not None
        )
        primary_candidates = [
            candidate
            for candidate in primary_candidates
            if int(candidate.get("label_position", first_label_position))
            == first_label_position
        ]
    grouped: dict[Decimal, list[dict[str, Any]]] = defaultdict(list)
    for candidate in primary_candidates:
        grouped[candidate["value_cny"]].append(candidate)
    if len(grouped) != 1:
        return None, {
            "raw_candidate_count": len(candidates),
            "primary_page_number": earliest_page,
            "primary_candidate_count": len(primary_candidates),
            "unique_value_count": len(grouped),
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda item: (
            int(item["page_number"]),
            int(item["table_index"] if item.get("table_index") is not None else -1),
            int(item["row_index"] if item.get("row_index") is not None else -1),
        ),
    )
    return selected, {
        "raw_candidate_count": len(candidates),
        "primary_page_number": earliest_page,
        "primary_candidate_count": len(primary_candidates),
        "identical_value_occurrence_count": len(occurrences),
        "unique_value_count": 1,
    }


def _add_summary_metrics_and_correct_fy_quarterly_values(
    result: dict[str, Any],
    page_texts: Sequence[str],
    page_tables: dict[int, Sequence[Sequence[Sequence[Any]]]],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    metric_map_before = _metric_map(result)
    contaminated = []
    if str(period_type or "").upper() == "FY":
        contaminated = sorted(
            metric_id
            for metric_id in FY_QUARTERLY_CONTAMINATION_METRICS
            if metric_id in metric_map_before
            and _source_page_is_quarterly(metric_map_before[metric_id], page_texts)
        )

    candidates = _summary_table_candidates(
        page_texts,
        page_tables,
        period_type=period_type,
    )
    _supplement_summary_text_candidates(
        candidates,
        page_texts,
        period_type=period_type,
    )
    selected_by_metric: dict[str, dict[str, Any]] = {}
    candidate_stats: dict[str, dict[str, Any]] = {}
    for metric_id in SUMMARY_METRICS:
        selected, stats = _unique_summary_candidates(candidates.get(metric_id) or [])
        candidate_stats[metric_id] = stats
        if selected is not None:
            selected_by_metric[metric_id] = selected

    removed: list[str] = []
    corrected: list[str] = []
    admitted: list[str] = []
    before_missing = set(result.get("missing_metrics") or [])
    for metric_id in contaminated:
        if metric_id not in selected_by_metric:
            parser._v1_4._drop_metrics(result, (metric_id,))
            removed.append(metric_id)

    scale_mismatch: list[str] = []
    for metric_id, selected in selected_by_metric.items():
        old_metric = metric_map_before.get(metric_id)
        if old_metric is None:
            continue
        old_raw = re.sub(r"\s+", "", str(old_metric.get("source_raw_value") or ""))
        new_raw = re.sub(r"\s+", "", str(selected.get("raw_value") or ""))
        same_source_page = int(old_metric.get("source_page") or 0) == int(
            selected["page_number"]
        )
        old_unit = str(old_metric.get("source_unit") or "")
        if (
            same_source_page
            and old_raw
            and old_raw == new_raw
            and old_unit != str(selected["unit"])
            and Decimal(str(old_metric.get("metric_value_cny")))
            != selected["value_cny"]
        ):
            scale_mismatch.append(metric_id)

    for metric_id, selected in selected_by_metric.items():
        should_put = (
            metric_id in before_missing
            or metric_id in contaminated
            or metric_id in scale_mismatch
        )
        if not should_put:
            continue
        old_metric = metric_map_before.get(metric_id)
        validation = {
            "rule": (
                "FY_ANNUAL_MAIN_ACCOUNTING_DATA_EXCLUDES_QUARTERLY_BREAKDOWN"
                if metric_id in contaminated
                else "MAIN_ACCOUNTING_DATA_SEMANTIC_YTD_COLUMN"
            ),
            "period_type": period_type,
            "explicit_quarterly_breakdown_rejected": (
                str(period_type or "").upper() == "FY"
            ),
            "quarterly_contamination_detected": metric_id in contaminated,
            "source_unit_scale_mismatch_detected": metric_id in scale_mismatch,
            "old_metric_value_cny": (
                old_metric.get("metric_value_cny") if old_metric else None
            ),
            "old_source_page": old_metric.get("source_page") if old_metric else None,
            **candidate_stats[metric_id],
        }
        parser._v1_4._put_metric(
            result,
            _summary_evidence(
                metric_id,
                selected,
                validation=validation,
                source_context=source_context,
            ),
        )
        if metric_id in contaminated or metric_id in scale_mismatch:
            corrected.append(metric_id)
        elif metric_id in before_missing:
            admitted.append(metric_id)

    return {
        "status": (
            "PASS_SUMMARY_METRICS_ADMITTED_OR_CORRECTED"
            if admitted or corrected or removed
            else "NO_MATCH_SUMMARY_RESIDUAL_OR_CORRECTION"
        ),
        "contaminated_metric_ids": contaminated,
        "source_unit_scale_mismatch_metric_ids": sorted(scale_mismatch),
        "corrected_metric_ids": corrected,
        "removed_unreplaced_metric_ids": removed,
        "admitted_metric_ids": admitted,
        "candidate_stats": candidate_stats,
    }


def _nearest_income_scope(
    compact_pages: Sequence[str],
    page_index: int,
) -> str | None:
    context = "".join(
        compact_pages[index]
        for index in range(max(0, page_index - 5), page_index + 1)
    )
    consolidated_position = max(
        context.rfind("合并利润表"),
        context.rfind("合并损益表"),
        context.rfind("合并及公司利润表"),
    )
    standalone_company_positions = [
        match.start()
        for match in re.finditer(r"(?<!合并及)(?<!合并)公司利润表", context)
    ]
    parent_position = max(
        [context.rfind("母公司利润表"), *standalone_company_positions],
        default=-1,
    )
    if consolidated_position >= 0 and consolidated_position > parent_position:
        return "EXPLICIT_CONSOLIDATED_INCOME_HEADING"
    if parent_position >= 0 and parent_position > consolidated_position:
        return "EXPLICIT_PARENT_INCOME_HEADING"
    return None


def _flexible_anchor_match(
    pair: tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None,
    result: dict[str, Any],
    metric_id: str,
) -> bool:
    expected = _metric_value(result, metric_id)
    return (
        pair is not None
        and expected is not None
        and _same_display(pair[0].value_cny, expected, _metric_anchor_unit(result, metric_id))
    )


def _income_candidates(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> list[dict[str, Any]]:
    expected_parent = _metric_value(result, "PARENT_NET_PROFIT_YTD")
    expected_revenue = _metric_value(result, "OPERATING_REVENUE_YTD")
    if expected_parent is None and expected_revenue is None:
        return []
    compact_pages = [_compact(text) for text in page_texts]
    operating_pages = [
        index
        for index, compact in enumerate(compact_pages)
        if "营业利润" in compact or "营业亏损" in compact
    ]
    candidates: list[dict[str, Any]] = []
    for operating_page in operating_pages:
        amount_index, q3_context = direct._q3_amount_index(
            page_texts,
            operating_page,
            period_type,
        )
        if amount_index is None:
            continue
        section_pages = _pages(operating_page - 2, operating_page + 4, len(page_texts))
        explicit_unit = direct._unique_explicit_unit(page_texts, section_pages)
        explicit_scope = _nearest_income_scope(compact_pages, operating_page)
        if explicit_scope == "EXPLICIT_PARENT_INCOME_HEADING":
            continue
        units: Iterable[str] = (
            (explicit_unit,)
            if explicit_unit is not None
            else tuple(parser._base.UNIT_MULTIPLIERS)
        )
        for unit in units:
            if unit is None:
                continue
            operating_profit = direct._whole_page_pair(
                page_texts,
                (operating_page,),
                INCOME_OPERATING_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            if operating_profit is None:
                continue
            revenue = direct._whole_page_pair(
                page_texts,
                section_pages,
                direct.REVENUE_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            net_profit = direct._whole_page_pair(
                page_texts,
                section_pages,
                direct.NET_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            parent_profit = direct._whole_page_pair(
                page_texts,
                section_pages,
                direct.PARENT_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            minority_profit = direct._whole_page_pair(
                page_texts,
                section_pages,
                direct.MINORITY_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            nonoperating_income = direct._whole_page_pair(
                page_texts,
                section_pages,
                INCOME_NONOPERATING_INCOME_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            nonoperating_expense = direct._whole_page_pair(
                page_texts,
                section_pages,
                INCOME_NONOPERATING_EXPENSE_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            profit_total = direct._whole_page_pair(
                page_texts,
                section_pages,
                INCOME_PROFIT_TOTAL_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            parent_match = _flexible_anchor_match(
                parent_profit,
                result,
                "PARENT_NET_PROFIT_YTD",
            )
            revenue_match = _flexible_anchor_match(
                revenue,
                result,
                "OPERATING_REVENUE_YTD",
            )
            dual_summary_anchor = parent_match and revenue_match
            if not parent_match and not revenue_match:
                continue
            profit_bridge_current: bool | None = None
            profit_bridge_prior: bool | None = None
            if all(
                pair is not None
                for pair in (
                    nonoperating_income,
                    nonoperating_expense,
                    profit_total,
                )
            ):
                assert nonoperating_income is not None
                assert nonoperating_expense is not None
                assert profit_total is not None
                profit_bridge_current = _same_display(
                    operating_profit[0].value_cny
                    + nonoperating_income[0].value_cny
                    - nonoperating_expense[0].value_cny,
                    profit_total[0].value_cny,
                    unit,
                )
                profit_bridge_prior = _same_display(
                    operating_profit[1].value_cny
                    + nonoperating_income[1].value_cny
                    - nonoperating_expense[1].value_cny,
                    profit_total[1].value_cny,
                    unit,
                )
            dual_profit_bridge = bool(
                profit_bridge_current and profit_bridge_prior
            )
            parent_identity_current: bool | None = None
            parent_identity_prior: bool | None = None
            if all(pair is not None for pair in (net_profit, parent_profit, minority_profit)):
                assert net_profit is not None
                assert parent_profit is not None
                assert minority_profit is not None
                parent_identity_current = _same_display(
                    net_profit[0].value_cny,
                    parent_profit[0].value_cny + minority_profit[0].value_cny,
                    unit,
                )
                parent_identity_prior = _same_display(
                    net_profit[1].value_cny,
                    parent_profit[1].value_cny + minority_profit[1].value_cny,
                    unit,
                )
                if (
                    not parent_identity_current or not parent_identity_prior
                ) and not dual_summary_anchor and not dual_profit_bridge:
                    continue
            if explicit_scope is None and not (
                dual_summary_anchor
                or dual_profit_bridge
                or (parent_identity_current and parent_identity_prior)
            ):
                continue
            candidates.append(
                {
                    "unit": unit,
                    "amount_index": amount_index,
                    "q3_context": q3_context,
                    "scope_anchor": (
                        explicit_scope
                        or (
                            "DUAL_PERIOD_SUMMARY_REVENUE_AND_PARENT_ANCHORS"
                            if dual_summary_anchor
                            else (
                                "DUAL_PERIOD_OPERATING_TO_TOTAL_PROFIT_BRIDGE_"
                                "PLUS_SUMMARY_ANCHOR"
                                if dual_profit_bridge
                                else "DUAL_PERIOD_PARENT_IDENTITY"
                            )
                        )
                    ),
                    "operating_profit": operating_profit,
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "parent_profit": parent_profit,
                    "minority_profit": minority_profit,
                    "parent_anchor_match": parent_match,
                    "revenue_anchor_match": revenue_match,
                    "parent_identity_current": parent_identity_current,
                    "parent_identity_prior": parent_identity_prior,
                    "profit_bridge_current": profit_bridge_current,
                    "profit_bridge_prior": profit_bridge_prior,
                    "nonoperating_income": nonoperating_income,
                    "nonoperating_expense": nonoperating_expense,
                    "profit_total": profit_total,
                }
            )
    return candidates


def _income_candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    revenue = candidate.get("revenue")
    parent = candidate.get("parent_profit")
    return (
        candidate["unit"],
        candidate["amount_index"],
        candidate["operating_profit"][0].value_cny,
        revenue[0].value_cny if revenue is not None else None,
        parent[0].value_cny if parent is not None else None,
    )


def _add_income_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = {
        "OPERATING_REVENUE_YTD",
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    }.intersection(result.get("missing_metrics") or [])
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_INCOME_RESIDUAL",
            "admitted_metric_ids": [],
        }
    candidates = _income_candidates(result, page_texts, period_type=period_type)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[_income_candidate_key(candidate)].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_RESIDUAL_INCOME_TUPLE_NOT_UNIQUE",
            "raw_candidate_count": len(candidates),
            "unique_fact_tuple_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda item: int(item["operating_profit"][0].page_number),
    )
    validation = {
        "rule": (
            "ANNUAL_OR_PERIOD_SUMMARY_ANCHOR_PLUS_EXPLICIT_CONSOLIDATED_SCOPE_"
            "OR_DUAL_PERIOD_PARENT_IDENTITY"
        ),
        "scope_anchor": selected["scope_anchor"],
        "summary_parent_profit_cny": str(
            _metric_value(result, "PARENT_NET_PROFIT_YTD")
        ),
        "summary_revenue_cny": str(_metric_value(result, "OPERATING_REVENUE_YTD")),
        "parent_profit_anchor_match_passed": selected["parent_anchor_match"],
        "revenue_anchor_match_passed": selected["revenue_anchor_match"],
        "parent_plus_minority_identity_current": selected["parent_identity_current"],
        "parent_plus_minority_identity_prior": selected["parent_identity_prior"],
        "operating_plus_nonoperating_identity_current": selected[
            "profit_bridge_current"
        ],
        "operating_plus_nonoperating_identity_prior": selected[
            "profit_bridge_prior"
        ],
        "selected_amount_index": selected["amount_index"],
        **selected["q3_context"],
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
        "unique_fact_tuple_count": 1,
    }
    metric_pairs = {
        "OPERATING_REVENUE_YTD": selected.get("revenue"),
        "OPERATING_PROFIT_YTD": selected.get("operating_profit"),
        "PARENT_NET_PROFIT_YTD": selected.get("parent_profit"),
    }
    admitted: list[str] = []
    for metric_id in (
        "OPERATING_REVENUE_YTD",
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    ):
        pair = metric_pairs[metric_id]
        if metric_id not in result.get("missing_metrics", []) or pair is None:
            continue
        if metric_id == "PARENT_NET_PROFIT_YTD" and not (
            selected["parent_anchor_match"]
            or (
                selected["parent_identity_current"]
                and selected["parent_identity_prior"]
            )
        ):
            continue
        parser._v1_4._put_metric(
            result,
            direct._direct_evidence(
                metric_id,
                pair[0],
                value_period_scope="YEAR_TO_DATE",
                section_name="DIRECT_RESIDUAL_CONSOLIDATED_INCOME_STATEMENT",
                validation={
                    **validation,
                    "residual_parser_version": RESIDUAL_PARSER_VERSION,
                },
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    return {
        "status": (
            "PASS_RESIDUAL_INCOME_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_RESIDUAL_INCOME_DETAILS"
        ),
        "source_pages": sorted(
            {
                pair[0].page_number
                for pair in metric_pairs.values()
                if pair is not None
            }
        ),
        "source_unit": selected["unit"],
        "admitted_metric_ids": admitted,
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
    }


def _nearest_cash_scope(compact_pages: Sequence[str], page_index: int) -> str | None:
    context = "".join(
        compact_pages[index]
        for index in range(max(0, page_index - 4), page_index + 1)
    )
    consolidated_position = max(
        context.rfind("合并现金流量表"),
        context.rfind("合并年初到报告期末现金流量表"),
        context.rfind("合并年初至报告期末现金流量表"),
        context.rfind("合并及公司现金流量表"),
    )
    parent_position = max(
        context.rfind("母公司现金流量表"),
        context.rfind("公司现金流量表"),
    )
    if consolidated_position >= 0 and consolidated_position > parent_position:
        return "EXPLICIT_CONSOLIDATED_CASH_FLOW_HEADING"
    return None


def _add_cash_flow_statement_metric(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    metric_id = "OPERATING_CASH_FLOW_YTD"
    if metric_id not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_OPERATING_CASH_FLOW_PRESENT",
            "admitted_metric_ids": [],
        }
    compact_pages = [_compact(text) for text in page_texts]
    net_pages = [
        index
        for index, compact in enumerate(compact_pages)
        if "经营活动" in compact and "现金流量净额" in compact
    ]
    candidates: list[dict[str, Any]] = []
    for page_index in net_pages:
        scope = _nearest_cash_scope(compact_pages, page_index)
        if scope is None:
            continue
        section_pages = _pages(page_index - 2, page_index + 3, len(page_texts))
        unit = direct._unique_explicit_unit(page_texts, section_pages)
        if unit not in parser._base.UNIT_MULTIPLIERS:
            continue
        net = direct._whole_page_pair(
            page_texts,
            section_pages,
            CASH_NET_PATTERNS,
            unit=unit,
            amount_index=0,
        )
        inflow = direct._whole_page_pair(
            page_texts,
            section_pages,
            CASH_INFLOW_PATTERNS,
            unit=unit,
            amount_index=0,
        )
        outflow = direct._whole_page_pair(
            page_texts,
            section_pages,
            CASH_OUTFLOW_PATTERNS,
            unit=unit,
            amount_index=0,
        )
        if net is None or inflow is None or outflow is None:
            continue
        current_identity = _same_display(
            inflow[0].value_cny - outflow[0].value_cny,
            net[0].value_cny,
            unit,
        )
        prior_identity = _same_display(
            inflow[1].value_cny - outflow[1].value_cny,
            net[1].value_cny,
            unit,
        )
        if not current_identity or not prior_identity:
            continue
        candidates.append(
            {
                "unit": unit,
                "scope": scope,
                "net": net,
                "inflow": inflow,
                "outflow": outflow,
            }
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                candidate["unit"],
                candidate["net"][0].value_cny,
                candidate["net"][1].value_cny,
            )
        ].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_CASH_FLOW_IDENTITY_TUPLE_NOT_UNIQUE",
            "raw_candidate_count": len(candidates),
            "unique_fact_tuple_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(occurrences, key=lambda item: item["net"][0].page_number)
    validation = {
        "rule": (
            "EXPLICIT_CONSOLIDATED_CASH_FLOW_HEADING_PLUS_DUAL_PERIOD_"
            "INFLOW_MINUS_OUTFLOW_EQUALS_NET"
        ),
        "scope_anchor": selected["scope"],
        "current_period_identity_passed": True,
        "prior_period_identity_passed": True,
        "inflow_current_cny": str(selected["inflow"][0].value_cny),
        "outflow_current_cny": str(selected["outflow"][0].value_cny),
        "net_current_cny": str(selected["net"][0].value_cny),
        "inflow_prior_cny": str(selected["inflow"][1].value_cny),
        "outflow_prior_cny": str(selected["outflow"][1].value_cny),
        "net_prior_cny": str(selected["net"][1].value_cny),
        "raw_candidate_count": len(candidates),
        "unique_fact_tuple_count": 1,
        "residual_parser_version": RESIDUAL_PARSER_VERSION,
    }
    parser._v1_4._put_metric(
        result,
        direct._direct_evidence(
            metric_id,
            selected["net"][0],
            value_period_scope="YEAR_TO_DATE",
            section_name="DIRECT_RESIDUAL_CONSOLIDATED_CASH_FLOW_STATEMENT",
            validation=validation,
            source_context=source_context,
        ),
    )
    return {
        "status": "PASS_CASH_FLOW_IDENTITY_METRIC_ADMITTED",
        "source_page": selected["net"][0].page_number,
        "source_unit": selected["unit"],
        "admitted_metric_ids": [metric_id],
    }


def _page_has_display_value(
    page_text: str,
    value_cny: Decimal,
    unit: str,
) -> bool:
    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    for amount_match in parser._base._number_matches_after(page_text, 0):
        raw = parser._raw_amount_with_closing_parenthesis(page_text, amount_match)
        if re.search(r"\d", raw) is None:
            continue
        try:
            observed = parser._base.parse_decimal(raw) * multiplier
        except (InvalidOperation, ValueError):
            continue
        if _same_display(observed, value_cny, unit):
            return True
    return False


def _derived_receivable_evidence(
    value_cny: Decimal,
    *,
    source_page: int,
    unit: str,
    validation: dict[str, Any],
    source_context: dict[str, Any],
) -> parser._base.MetricEvidence:
    raw_value = str(value_cny / parser._base.UNIT_MULTIPLIERS[unit])
    locator = {
        "page": source_page,
        "section": "CONSOLIDATED_NOTES_RECEIVABLE_DERIVED_AND_EXPLICITLY_DISPLAYED",
        "validation": validation,
        "source_context": source_context,
        "parser_version": RESIDUAL_PARSER_VERSION,
        "text_engine": "PDFIUM_DIRECT_DOCUMENT_STATISTICS",
    }
    return parser._base.MetricEvidence(
        metric_id="ACCOUNTS_RECEIVABLE_END",
        metric_value_cny=float(value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope="PERIOD_END",
        source_page=source_page,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label="应收账款",
        source_raw_value=raw_value,
        source_unit=unit,
        source_unit_multiplier=float(parser._base.UNIT_MULTIPLIERS[unit]),
        source_method=(
            "PDFIUM_DIRECT_COMBINED_RECEIVABLE_MINUS_BILLS_DUAL_PERIOD_"
            "EXPLICIT_NOTE_DISPLAY_RECONCILED"
        ),
        verification_status=RESIDUAL_VERIFICATION_STATUS,
    )


def _add_derived_receivable_metric(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    metric_id = "ACCOUNTS_RECEIVABLE_END"
    if metric_id not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_ACCOUNTS_RECEIVABLE_PRESENT",
            "admitted_metric_ids": [],
        }
    balance_candidates = direct._balance_candidates(result, page_texts)
    grouped_balance: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in balance_candidates:
        grouped_balance[direct._balance_identity_key(candidate)].append(candidate)
    if len(grouped_balance) != 1:
        return {
            "status": "NO_MATCH_DERIVED_RECEIVABLE_BALANCE_SCOPE_NOT_UNIQUE",
            "balance_candidate_count": len(balance_candidates),
            "unique_balance_identity_count": len(grouped_balance),
            "admitted_metric_ids": [],
        }
    balance_occurrences = next(iter(grouped_balance.values()))
    combined_by_value: dict[tuple[Decimal, Decimal], Any] = {}
    for balance in balance_occurrences:
        combined = direct._pair(
            page_texts,
            _pages(
                int(balance["asset_page"]) - 4,
                int(balance["asset_page"]) + 1,
                len(page_texts),
            ),
            COMBINED_RECEIVABLE_PATTERNS,
            unit=str(balance["unit"]),
        )
        if combined is not None:
            combined_by_value[direct._pair_values(combined)] = combined
    if len(combined_by_value) != 1:
        return {
            "status": "NO_MATCH_DERIVED_RECEIVABLE_COMBINED_ROW_NOT_UNIQUE",
            "combined_pair_count": len(combined_by_value),
            "admitted_metric_ids": [],
        }
    combined = next(iter(combined_by_value.values()))
    selected_balance = min(
        balance_occurrences,
        key=lambda candidate: int(candidate["asset_page"]),
    )
    statement_unit = str(selected_balance["unit"])
    first_note_page = int(selected_balance["asset_page"]) + 1
    derived_candidates: list[dict[str, Any]] = []
    for page_index in range(first_note_page, len(page_texts)):
        compact = _compact(page_texts[page_index])
        if "应收票据" not in compact:
            continue
        context = "".join(
            _compact(page_texts[index])
            for index in range(max(first_note_page, page_index - 1), min(len(page_texts), page_index + 2))
        )
        if "应收票据及应收账款" not in context and "应收票据和应收账款" not in context:
            continue
        note_unit = direct._unique_explicit_unit(
            page_texts,
            _pages(page_index - 1, page_index + 2, len(page_texts)),
        ) or statement_unit
        if note_unit not in parser._base.UNIT_MULTIPLIERS:
            continue
        bills = direct._pair(
            page_texts,
            (page_index,),
            PURE_BILLS_PATTERNS,
            unit=note_unit,
        )
        if bills is None:
            continue
        current = combined[0].value_cny - bills[0].value_cny
        prior = combined[1].value_cny - bills[1].value_cny
        if current < 0 or prior < 0:
            continue
        display_pages = _pages(page_index, page_index + 3, len(page_texts))
        current_display_pages = [
            display_page + 1
            for display_page in display_pages
            if "应收账款" in _compact(page_texts[display_page])
            and _page_has_display_value(page_texts[display_page], current, note_unit)
        ]
        prior_display_pages = [
            display_page + 1
            for display_page in display_pages
            if "应收账款" in _compact(page_texts[display_page])
            and _page_has_display_value(page_texts[display_page], prior, note_unit)
        ]
        if not current_display_pages or not prior_display_pages:
            continue
        explicit_pages = sorted(set(current_display_pages + prior_display_pages))
        derived_candidates.append(
            {
                "current": current,
                "prior": prior,
                "bills": bills,
                "note_unit": note_unit,
                "explicit_pages": explicit_pages,
                "current_display_pages": current_display_pages,
                "prior_display_pages": prior_display_pages,
            }
        )
    grouped: dict[tuple[Decimal, Decimal], list[dict[str, Any]]] = defaultdict(list)
    for candidate in derived_candidates:
        grouped[(candidate["current"], candidate["prior"])].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_DERIVED_RECEIVABLE_VALUE_NOT_UNIQUE",
            "raw_candidate_count": len(derived_candidates),
            "unique_value_pair_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda candidate: (
            candidate["explicit_pages"][0],
            candidate["bills"][0].page_number,
        ),
    )
    validation = {
        "rule": (
            "COMBINED_RECEIVABLE_MINUS_PURE_BILLS_EQUALS_NET_RECEIVABLE_"
            "AND_DERIVED_VALUES_EXPLICITLY_DISPLAYED_FOR_BOTH_PERIODS"
        ),
        "statement_combined_current_cny": str(combined[0].value_cny),
        "statement_combined_prior_cny": str(combined[1].value_cny),
        "note_bills_current_cny": str(selected["bills"][0].value_cny),
        "note_bills_prior_cny": str(selected["bills"][1].value_cny),
        "derived_receivable_current_cny": str(selected["current"]),
        "derived_receivable_prior_cny": str(selected["prior"]),
        "current_period_identity_passed": True,
        "prior_period_identity_passed": True,
        "explicit_display_pages": selected["explicit_pages"],
        "current_period_explicit_display_pages": selected[
            "current_display_pages"
        ],
        "prior_period_explicit_display_pages": selected[
            "prior_display_pages"
        ],
        "statement_combined_source_page": combined[0].page_number,
        "note_bills_source_page": selected["bills"][0].page_number,
        "raw_candidate_count": len(derived_candidates),
        "unique_value_pair_count": 1,
        "residual_parser_version": RESIDUAL_PARSER_VERSION,
    }
    parser._v1_4._put_metric(
        result,
        _derived_receivable_evidence(
            selected["current"],
            source_page=selected["explicit_pages"][0],
            unit=selected["note_unit"],
            validation=validation,
            source_context=source_context,
        ),
    )
    return {
        "status": "PASS_DERIVED_RECEIVABLE_DUAL_PERIOD_RECONCILED_AND_ADMITTED",
        "statement_combined_source_page": combined[0].page_number,
        "note_bills_source_page": selected["bills"][0].page_number,
        "explicit_display_pages": selected["explicit_pages"],
        "admitted_metric_ids": [metric_id],
    }


def _nearest_statement_pair(
    page_texts: Sequence[str],
    page_indices: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    for page_index in page_indices:
        pair = direct._statement_page_pair(
            page_texts[page_index],
            page_number=page_index + 1,
            patterns=patterns,
            unit=unit,
            amount_index=0,
        )
        if pair is not None:
            return pair
    return None


def _add_residual_balance_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = direct.BALANCE_TARGET_METRICS.intersection(
        result.get("missing_metrics") or []
    )
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_BALANCE_RESIDUAL",
            "admitted_metric_ids": [],
        }

    compact_pages = [_compact(page_text) for page_text in page_texts]
    expected_assets = _metric_value(result, "TOTAL_ASSETS_END")
    total_patterns = (
        r"负债(?:及|和)(?:股东权益|所有者权益)"
        r"(?:\s*[（(]或股东权益[）)])?\s*(?:总计|合计)",
    )
    equity_patterns = (
        r"(?<!公司)(?<!母公司)(?<!少数)"
        r"(?:股东权益|所有者权益)(?:\s*[（(]或股东权益[）)])?\s*合计",
    )
    candidates: list[dict[str, Any]] = []
    for asset_page, compact_page in enumerate(compact_pages):
        if re.search(direct.ASSET_PATTERNS[0], compact_page) is None:
            continue
        scope = direct._nearest_balance_scope(compact_pages, asset_page)
        if scope != "EXPLICIT_CONSOLIDATED_BALANCE_HEADING":
            continue
        identity_pages = _pages(asset_page, asset_page + 6, len(page_texts))
        unit_pages = _pages(asset_page - 5, asset_page + 6, len(page_texts))
        unit = direct._unique_explicit_unit(page_texts, unit_pages)
        if unit not in parser._base.UNIT_MULTIPLIERS:
            continue
        assets = _nearest_statement_pair(
            page_texts,
            (asset_page,),
            direct.ASSET_PATTERNS,
            unit=unit,
        )
        liabilities = _nearest_statement_pair(
            page_texts,
            identity_pages,
            direct.LIABILITY_PATTERNS,
            unit=unit,
        )
        equity = _nearest_statement_pair(
            page_texts,
            identity_pages,
            equity_patterns,
            unit=unit,
        )
        total = _nearest_statement_pair(
            page_texts,
            identity_pages,
            total_patterns,
            unit=unit,
        )
        if any(pair is None for pair in (assets, liabilities, equity, total)):
            continue
        assert assets is not None
        assert liabilities is not None
        assert equity is not None
        assert total is not None
        current_identity = _same_display(
            assets[0].value_cny,
            liabilities[0].value_cny + equity[0].value_cny,
            unit,
        ) and _same_display(
            assets[0].value_cny,
            total[0].value_cny,
            unit,
        )
        prior_identity = _same_display(
            assets[1].value_cny,
            liabilities[1].value_cny + equity[1].value_cny,
            unit,
        ) and _same_display(
            assets[1].value_cny,
            total[1].value_cny,
            unit,
        )
        if not current_identity or not prior_identity:
            continue
        if expected_assets is not None and not _same_display(
            assets[0].value_cny,
            expected_assets,
            _metric_anchor_unit(result, "TOTAL_ASSETS_END"),
        ):
            continue
        detail_pages = tuple(
            range(asset_page, max(-1, asset_page - 6), -1)
        )
        receivable = _nearest_statement_pair(
            page_texts,
            detail_pages,
            direct.RECEIVABLE_PATTERNS,
            unit=unit,
        )
        inventory = _nearest_statement_pair(
            page_texts,
            detail_pages,
            direct.INVENTORY_PATTERNS,
            unit=unit,
        )
        candidates.append(
            {
                "asset_page": asset_page,
                "unit": unit,
                "scope": scope,
                "assets": assets,
                "liabilities": liabilities,
                "equity": equity,
                "total": total,
                "receivable": receivable,
                "inventory": inventory,
            }
        )

    if not candidates:
        return {
            "status": "NO_MATCH_RESIDUAL_BALANCE_DUAL_PERIOD_IDENTITY",
            "raw_candidate_count": 0,
            "admitted_metric_ids": [],
        }
    first_asset_page = min(int(candidate["asset_page"]) for candidate in candidates)
    primary_candidates = [
        candidate
        for candidate in candidates
        if int(candidate["asset_page"]) == first_asset_page
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in primary_candidates:
        grouped[
            (
                candidate["unit"],
                candidate["assets"][0].value_cny,
                candidate["liabilities"][0].value_cny,
                candidate["equity"][0].value_cny,
                candidate["total"][0].value_cny,
            )
        ].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_RESIDUAL_BALANCE_PRIMARY_IDENTITY_NOT_UNIQUE",
            "raw_candidate_count": len(candidates),
            "primary_candidate_count": len(primary_candidates),
            "unique_identity_tuple_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = occurrences[0]
    validation = {
        "rule": (
            "EXPLICIT_CONSOLIDATED_BALANCE_SCOPE_PLUS_DUAL_PERIOD_"
            "ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY_AND_TOTAL_ROW"
        ),
        "scope_anchor_kind": selected["scope"],
        "source_unit": selected["unit"],
        "asset_page": int(selected["asset_page"]) + 1,
        "first_column_identity_passed": True,
        "second_column_identity_passed": True,
        "total_assets_current_cny": str(selected["assets"][0].value_cny),
        "total_liabilities_current_cny": str(
            selected["liabilities"][0].value_cny
        ),
        "total_equity_current_cny": str(selected["equity"][0].value_cny),
        "liabilities_equity_total_current_cny": str(
            selected["total"][0].value_cny
        ),
        "total_assets_prior_cny": str(selected["assets"][1].value_cny),
        "total_liabilities_prior_cny": str(
            selected["liabilities"][1].value_cny
        ),
        "total_equity_prior_cny": str(selected["equity"][1].value_cny),
        "liabilities_equity_total_prior_cny": str(
            selected["total"][1].value_cny
        ),
        "raw_candidate_count": len(candidates),
        "primary_candidate_count": len(primary_candidates),
        "identical_primary_occurrence_count": len(occurrences),
        "residual_parser_version": RESIDUAL_PARSER_VERSION,
    }
    metric_pairs = {
        "ACCOUNTS_RECEIVABLE_END": selected["receivable"],
        "INVENTORY_END": selected["inventory"],
        "TOTAL_ASSETS_END": selected["assets"],
        "TOTAL_LIABILITIES_END": selected["liabilities"],
    }
    admitted: list[str] = []
    for metric_id in (
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
    ):
        pair = metric_pairs[metric_id]
        if metric_id not in result.get("missing_metrics", []) or pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            direct._direct_evidence(
                metric_id,
                pair[0],
                value_period_scope="PERIOD_END",
                section_name="DIRECT_RESIDUAL_CONSOLIDATED_BALANCE_STATEMENT",
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    return {
        "status": (
            "PASS_RESIDUAL_BALANCE_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_RESIDUAL_BALANCE_DETAILS"
        ),
        "source_pages": sorted(
            {
                pair[0].page_number
                for pair in metric_pairs.values()
                if pair is not None
            }
        ),
        "source_unit": selected["unit"],
        "admitted_metric_ids": admitted,
        "raw_candidate_count": len(candidates),
        "primary_candidate_count": len(primary_candidates),
    }


def add_direct_pdf_residual_statistics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    page_tables: dict[int, Sequence[Sequence[Sequence[Any]]]],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    normalized = [
        parser._v1_4.normalize_financial_text(page_text) for page_text in page_texts
    ]
    before_metrics = _metric_map(result)
    before_missing = list(result.get("missing_metrics") or [])
    summary_receipt = _add_summary_metrics_and_correct_fy_quarterly_values(
        result,
        normalized,
        page_tables,
        period_type=period_type,
        source_context=source_context,
    )
    balance_receipt = _add_residual_balance_metrics(
        result,
        normalized,
        source_context=source_context,
    )
    income_receipt = _add_income_metrics(
        result,
        normalized,
        period_type=period_type,
        source_context=source_context,
    )
    cash_receipt = _add_cash_flow_statement_metric(
        result,
        normalized,
        source_context=source_context,
    )
    receivable_receipt = _add_derived_receivable_metric(
        result,
        normalized,
        source_context=source_context,
    )
    parser._restamp_result(result)
    after_metrics = _metric_map(result)
    after_missing = list(result.get("missing_metrics") or [])
    admitted = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    corrected = [
        metric_id
        for metric_id in parser.REQUIRED_METRICS
        if metric_id in before_metrics
        and metric_id in after_metrics
        and before_metrics[metric_id] != after_metrics[metric_id]
    ]
    removed = [
        metric_id
        for metric_id in before_metrics
        if metric_id not in after_metrics
    ]
    changed_metric_ids = [
        metric_id
        for metric_id in parser.REQUIRED_METRICS
        if metric_id in admitted or metric_id in corrected
    ]
    return {
        "status": (
            "PASS_DIRECT_PDF_RESIDUAL_STATISTICS_CHANGED"
            if changed_metric_ids or removed
            else "NO_MATCH_DIRECT_PDF_RESIDUAL_STATISTICS"
        ),
        "residual_parser_version": RESIDUAL_PARSER_VERSION,
        "summary": summary_receipt,
        "balance": balance_receipt,
        "income": income_receipt,
        "cash_flow": cash_receipt,
        "derived_receivable": receivable_receipt,
        "admitted_metric_ids": admitted,
        "corrected_metric_ids": corrected,
        "removed_metric_ids": removed,
        "changed_metric_ids": changed_metric_ids,
        "changed_metrics": [after_metrics[metric_id] for metric_id in changed_metric_ids],
        "remaining_missing_metrics": after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def metric_evidence_to_dict(evidence: parser._base.MetricEvidence) -> dict[str, Any]:
    return asdict(evidence)
