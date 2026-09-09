from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_3 as _v1_3


PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_"
    "SEMANTIC_TABLE_RECONCILIATION_IMAGE_APPENDIX_V1_8_0"
)
MANUAL_APPENDIX_PATH = (
    _v1_3.ROOT
    / "config/csi300_pit_fundamental_underreaction_official_facts_v1_4_manual_appendix.json"
)
ADMITTED_VERIFICATION_STATUS = _v1_3.ADMITTED_VERIFICATION_STATUS
REQUIRED_METRICS = _v1_3.REQUIRED_METRICS
MetricEvidence = _v1_3.MetricEvidence
SectionRange = _v1_3.SectionRange


TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "併": "并",
        "資": "资",
        "產": "产",
        "負": "负",
        "債": "债",
        "賬": "账",
        "應": "应",
        "收": "收",
        "貨": "货",
        "幣": "币",
        "營": "营",
        "業": "业",
        "潤": "润",
        "淨": "净",
        "現": "现",
        "額": "额",
        "歸": "归",
        "屬": "属",
        "東": "东",
        "權": "权",
        "益": "益",
        "計": "计",
        "總": "总",
        "長": "长",
        "遞": "递",
        "延": "延",
        "稅": "税",
        "費": "费",
        "損": "损",
        "綜": "综",
        "開": "开",
        "報": "报",
        "表": "表",
        "註": "注",
        "項": "项",
        "庫": "库",
        "點": "点",
        "數": "数",
        "據": "据",
        "減": "减",
        "處": "处",
        "置": "置",
        "價": "价",
        "值": "值",
        "動": "动",
        "與": "与",
        "經": "经",
        "續": "续",
        "別": "别",
        "類": "类",
        "時": "时",
        "間": "间",
        "構": "构",
        "單": "单",
        "萬": "万",
        "億": "亿",
        "圓": "元",
        "於": "于",
        "為": "为",
        "財": "财",
        "務": "务",
        "職": "职",
        "聯": "联",
        "預": "预",
    }
)


def normalize_financial_text(value: Any) -> str:
    return _base.normalize_text(value).translate(TRADITIONAL_TO_SIMPLIFIED)


def compact_financial_text(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_financial_text(value))


@dataclass(frozen=True)
class TableCandidate:
    value_cny: Decimal
    page_index: int
    table_index: int
    row_index: int
    label_cell_index: int
    value_cell_index: int
    label: str
    raw_value: str
    unit: str
    row_cells: tuple[str, ...]
    column_header: str


@dataclass(frozen=True)
class SummaryDuplicate:
    value_cny: Decimal
    unit: str


def _refresh_result(result: dict[str, Any]) -> None:
    metrics = {str(metric["metric_id"]): metric for metric in result.get("metrics") or []}
    result["metrics"] = [metrics[key] for key in REQUIRED_METRICS if key in metrics]
    result["missing_metrics"] = [key for key in REQUIRED_METRICS if key not in metrics]
    result["document_complete"] = not result["missing_metrics"]
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if result["document_complete"]
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )
    result["parser_version"] = PARSER_VERSION


def _put_metric(result: dict[str, Any], evidence: MetricEvidence) -> None:
    metrics = {str(metric["metric_id"]): metric for metric in result.get("metrics") or []}
    metrics[evidence.metric_id] = asdict(evidence)
    result["metrics"] = list(metrics.values())
    _refresh_result(result)


def _drop_metrics(result: dict[str, Any], metric_ids: Iterable[str]) -> None:
    removed = set(metric_ids)
    result["metrics"] = [
        metric
        for metric in result.get("metrics") or []
        if str(metric.get("metric_id")) not in removed
    ]
    _refresh_result(result)


def _stamp_inherited_metrics(result: dict[str, Any]) -> None:
    for metric in result.get("metrics") or []:
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {"legacy_source_locator": str(metric.get("source_locator") or "")}
        previous = locator.get("parser_version")
        if previous and previous != PARSER_VERSION:
            locator.setdefault("base_parser_version", previous)
        locator["parser_version"] = PARSER_VERSION
        metric["source_locator"] = json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metric["verification_status"] = ADMITTED_VERIFICATION_STATUS
    _refresh_result(result)


def _normalize_cell(value: Any) -> str:
    return normalize_financial_text(value)


def _parse_amount_cell(value: Any) -> tuple[Decimal, str] | None:
    raw = _normalize_cell(value)
    compact = re.sub(r"\s+", "", raw)
    compact = compact.replace("，", ",").replace("．", ".")
    compact = compact.replace("−", "-").replace("－", "-").replace("—", "-")
    if compact in {"-", "--"}:
        return Decimal("0"), raw
    if re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", compact) is None:
        return None
    negative = compact.startswith("(") and compact.endswith(")")
    normalized = compact.strip("()").replace(",", "")
    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    return (-abs(amount) if negative else amount), raw


def _unit_near_page(
    page_texts: Sequence[str],
    page_index: int,
    section_pages: Sequence[int] = (),
) -> str | None:
    for pages in (
        (page_index,),
        tuple(
            index
            for index in range(max(0, page_index - 1), min(len(page_texts), page_index + 2))
        ),
        tuple(section_pages[:4]),
    ):
        if not pages:
            continue
        normalized = [normalize_financial_text(text) for text in page_texts]
        unit = _base.detect_unique_unit(normalized, pages)
        if unit in _base.UNIT_MULTIPLIERS:
            return unit
    return None


def _group_table_rows(
    rows: Iterable[tuple[int, int, int, Sequence[Any]]],
) -> dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]]:
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for page_index, table_index, row_index, cells in rows:
        groups[(page_index, table_index)].append(
            (row_index, tuple(_normalize_cell(cell) for cell in cells))
        )
    for group in groups.values():
        group.sort(key=lambda item: item[0])
    return dict(groups)


def _column_header(
    group: Sequence[tuple[int, tuple[str, ...]]],
    row_index: int,
    column_index: int,
) -> str:
    fragments: list[str] = []
    for candidate_row_index, cells in group:
        if candidate_row_index >= row_index:
            break
        if column_index < len(cells) and cells[column_index]:
            fragments.append(cells[column_index])
    return compact_financial_text(" ".join(fragments[-5:]))


def _column_score(
    header: str,
    *,
    period_type: str | None,
    role: str,
    amount_position: int,
    amount_count: int,
) -> int:
    compact = compact_financial_text(header)
    score = 0
    if "合并" in compact:
        score += 20
    if "公司" in compact and "合并" not in compact:
        score -= 20
    if any(marker in compact for marker in ("本期", "期末", "年末", "本年", "本报告期")):
        score += 5
    if any(marker in compact for marker in ("上期", "期初", "年初", "上年", "同期")):
        score -= 5
    if str(period_type or "").upper() == "Q3" and role in {"INCOME", "SUMMARY"}:
        if any(
            marker in compact
            for marker in ("前三季度", "1-9月", "1—9月", "1－9月", "年初至报告期末")
        ):
            score += 16
        if any(marker in compact for marker in ("第三季度", "7-9月", "7—9月", "7－9月")):
            score -= 12
    if role == "BALANCE" and amount_position == 0:
        score += 2
    if role in {"INCOME", "CASH", "SUMMARY"} and amount_position == 0:
        score += 1
    if (
        str(period_type or "").upper() == "Q3"
        and role == "INCOME"
        and amount_count >= 4
        and amount_position == 2
    ):
        score += 8
    return score


def _table_candidates(
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    patterns: Sequence[str],
    *,
    period_type: str | None,
    role: str,
    section_pages: Sequence[int],
    reject_tokens: Sequence[str] = (),
) -> list[TableCandidate]:
    compiled = [re.compile(pattern) for pattern in patterns]
    candidates: list[TableCandidate] = []
    for (page_index, table_index), group in sorted(groups.items()):
        if section_pages and page_index not in section_pages:
            continue
        unit = _unit_near_page(page_texts, page_index, section_pages)
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        for row_index, cells in group:
            for label_cell_index, cell in enumerate(cells):
                label = compact_financial_text(cell)
                if not label or any(token in label for token in reject_tokens):
                    continue
                if not any(pattern.search(label) for pattern in compiled):
                    continue
                amounts: list[tuple[int, Decimal, str]] = []
                for value_cell_index in range(label_cell_index + 1, len(cells)):
                    parsed = _parse_amount_cell(cells[value_cell_index])
                    if parsed is None:
                        continue
                    amount, raw = parsed
                    amounts.append((value_cell_index, amount, raw))
                if not amounts:
                    continue
                ranked: list[tuple[int, int, Decimal, str, str]] = []
                for position, (value_cell_index, amount, raw) in enumerate(amounts):
                    header = _column_header(group, row_index, value_cell_index)
                    score = _column_score(
                        header,
                        period_type=period_type,
                        role=role,
                        amount_position=position,
                        amount_count=len(amounts),
                    )
                    ranked.append((score, -position, amount, raw, header))
                best = max(ranked, key=lambda item: (item[0], item[1]))
                best_position = -best[1]
                value_cell_index, amount, raw = amounts[best_position]
                candidates.append(
                    TableCandidate(
                        value_cny=amount * _base.UNIT_MULTIPLIERS[unit],
                        page_index=page_index,
                        table_index=table_index,
                        row_index=row_index,
                        label_cell_index=label_cell_index,
                        value_cell_index=value_cell_index,
                        label=cell,
                        raw_value=raw,
                        unit=unit,
                        row_cells=cells,
                        column_header=best[4],
                    )
                )
    return candidates


def _best_candidate(candidates: Sequence[TableCandidate]) -> TableCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.page_index,
            item.table_index,
            item.row_index,
            item.label_cell_index,
        ),
    )


def _first_pattern_candidate(
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    patterns: Sequence[str],
    *,
    period_type: str | None,
    role: str,
    section_pages: Sequence[int],
    reject_tokens: Sequence[str] = (),
) -> TableCandidate | None:
    for pattern in patterns:
        candidate = _best_candidate(
            _table_candidates(
                groups,
                page_texts,
                (pattern,),
                period_type=period_type,
                role=role,
                section_pages=section_pages,
                reject_tokens=reject_tokens,
            )
        )
        if candidate is not None:
            return candidate
    return None


def _table_evidence(
    metric_id: str,
    candidate: TableCandidate,
    *,
    value_period_scope: str,
    section_name: str,
    validation: dict[str, Any],
    text_engine: str = "PDFPLUMBER",
) -> MetricEvidence:
    locator = {
        "page": candidate.page_index + 1,
        "section": section_name,
        "table_index": candidate.table_index,
        "row_index": candidate.row_index,
        "label_cell_index": candidate.label_cell_index,
        "value_cell_index": candidate.value_cell_index,
        "row_cells": list(candidate.row_cells),
        "column_header": candidate.column_header,
        "validation": validation,
        "parser_version": PARSER_VERSION,
        "text_engine": text_engine,
    }
    return MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=candidate.page_index + 1,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method=(
            f"{text_engine}_TABLE_SEMANTIC_COLUMN_"
            "CURRENT_CONSOLIDATED_PERIOD_RECONCILED"
        ),
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _empty_section(name: str) -> SectionRange:
    return SectionRange(name=name, page_indices=(), unit=None)


def _enhanced_section(
    page_texts: Sequence[str],
    *,
    name: str,
    headings: Sequence[str],
    next_headings: Sequence[str],
    marker_groups: Sequence[Sequence[str]],
    company_headings: Sequence[str],
    maximum_pages: int,
) -> SectionRange:
    compact_pages = [compact_financial_text(text) for text in page_texts]
    starts: list[int] = []
    for page_index, compact in enumerate(compact_pages):
        if not any(compact_financial_text(heading) in compact for heading in headings):
            continue
        if "合并" not in compact:
            continue
        if not any(all(marker in compact for marker in group) for group in marker_groups):
            continue
        if len(_base.NUMBER_RE.findall(page_texts[page_index])) < 6:
            continue
        starts.append(page_index)
    if not starts:
        return _empty_section(name)
    start = starts[0]
    end = min(len(page_texts), start + maximum_pages)
    compact_terminators = [compact_financial_text(item) for item in next_headings]
    compact_company_headings = [compact_financial_text(item) for item in company_headings]
    for page_index in range(start + 1, end):
        compact = compact_pages[page_index]
        if any(marker in compact for marker in compact_terminators):
            end = page_index
            break
        if any(marker in compact for marker in compact_company_headings) and "合并" not in compact:
            end = page_index
            break
    pages = tuple(range(start, max(start + 1, end)))
    unit = _unit_near_page(page_texts, start, pages)
    return SectionRange(name=name, page_indices=pages, unit=unit)


def _section_has_consolidated_statement_content(
    page_texts: Sequence[str],
    section: SectionRange,
    *,
    headings: Sequence[str],
    marker_groups: Sequence[Sequence[str]],
) -> bool:
    if not section.page_indices:
        return False
    compact = "".join(
        compact_financial_text(page_texts[page_index])
        for page_index in section.page_indices
    )
    explicit_heading = "合并" in compact and any(
        compact_financial_text(heading) in compact for heading in headings
    )
    parallel_consolidated_columns = (
        compact.count("合并数") >= 2
        and compact.count("公司数") >= 2
    ) or (
        compact.count("合并") >= 2
        and compact.count("公司") >= 2
    ) or "合并及公司" in compact
    if not explicit_heading and not parallel_consolidated_columns:
        return False
    if not any(all(marker in compact for marker in group) for group in marker_groups):
        return False
    number_count = sum(
        len(_base.NUMBER_RE.findall(page_texts[page_index]))
        for page_index in section.page_indices
    )
    return number_count >= 12


def locate_statement_sections(page_texts: Sequence[str]) -> dict[str, SectionRange]:
    normalized = [normalize_financial_text(text) for text in page_texts]
    sections = _v1_3.locate_statement_sections(normalized)
    definitions = {
        "balance": {
            "name": "CONSOLIDATED_BALANCE_SHEET",
            "headings": ("合并资产负债表", "合并及公司资产负债表"),
            "next_headings": ("合并利润表", "合并及公司利润表", "利润表"),
            "marker_groups": (("流动资产", "货币资金", "资产总计"),),
            "company_headings": ("母公司资产负债表", "公司资产负债表"),
            "maximum_pages": 8,
        },
        "income": {
            "name": "CONSOLIDATED_INCOME_STATEMENT",
            "headings": (
                "合并利润表",
                "合并及公司利润表",
                "合并及公司利润及利润分配表",
            ),
            "next_headings": ("合并现金流量表", "合并及公司现金流量表", "现金流量表"),
            "marker_groups": (
                ("营业收入", "营业利润", "净利润"),
                ("营业总收入", "营业利润", "净利润"),
                ("营业收入", "营业利润", "利润总额"),
                ("营业总收入", "营业利润", "利润总额"),
                ("营业收入", "营业外收入", "所得税费用"),
                ("营业总收入", "营业外收入", "所得税费用"),
            ),
            "company_headings": ("母公司利润表", "公司利润表"),
            "maximum_pages": 8,
        },
        "cash": {
            "name": "CONSOLIDATED_CASH_FLOW_STATEMENT",
            "headings": ("合并现金流量表", "合并及公司现金流量表"),
            "next_headings": ("合并所有者权益变动表", "合并股东权益变动表", "所有者权益变动表"),
            "marker_groups": (("经营活动", "现金流量", "销售商品"),),
            "company_headings": ("母公司现金流量表", "公司现金流量表"),
            "maximum_pages": 8,
        },
    }
    enhanced = {
        key: _enhanced_section(
            normalized,
            **definition,
        )
        for key, definition in definitions.items()
    }
    for key, definition in definitions.items():
        candidate = enhanced[key]
        current = sections.get(key) or _empty_section(str(definition["name"]))
        current_valid = _section_has_consolidated_statement_content(
            normalized,
            current,
            headings=definition["headings"],
            marker_groups=definition["marker_groups"],
        )
        if candidate.page_indices:
            sections[key] = candidate
        elif not current_valid:
            sections[key] = _empty_section(str(definition["name"]))
    return sections


def _summary_pages(page_texts: Sequence[str], sections: dict[str, SectionRange]) -> tuple[int, ...]:
    section_starts = [
        section.page_indices[0]
        for section in sections.values()
        if section.page_indices
    ]
    upper = min(section_starts) if section_starts else len(page_texts)
    limit = min(upper, 40)
    selected: set[int] = set()
    for page_index in range(limit):
        compact = compact_financial_text(page_texts[page_index])
        main_marker = any(
            marker in compact
            for marker in ("主要会计数据", "主要财务数据", "主要会计资料")
        )
        metric_marker_count = sum(
            marker in compact
            for marker in (
                "营业收入",
                "归属于上市公司股东的净利润",
                "扣除非经常性损益",
                "经营活动产生的现金流量净额",
                "总资产",
            )
        )
        if main_marker or metric_marker_count >= 2:
            selected.add(page_index)
            if page_index + 1 < limit:
                selected.add(page_index + 1)
    return tuple(sorted(selected))


SUMMARY_DEFINITIONS: dict[str, tuple[Sequence[str], Sequence[str]]] = {
    "OPERATING_REVENUE_YTD": (
        (r"^营业收入$", r"^营业总收入$"),
        (),
    ),
    "PARENT_NET_PROFIT_YTD": (
        (
            r"^归属于上市公司(?:普通股)?股东的净利润$",
            r"^归属于母公司(?:所有者|股东)的净利润$",
            r"^归属于母公司普通股股东(?:的净利润)?$",
            r"^母公司股东(?:的净利润)?$",
        ),
        ("扣除", "综合收益"),
    ),
    "CORE_PARENT_NET_PROFIT_YTD": (
        (
            r"^归属于上市公司(?:普通股)?股东的扣除非经常性损益的净利润$",
            r"^归属于上市公司(?:普通股)?股东的扣除$",
            r"^归属于母公司(?:所有者|股东)的扣除非经常性损益的净利润$",
            r"^扣除非经常性损益后的净利润$",
            r"^归属于上市公司(?:普通股)?股东的扣除非经常性损益的净(?:（亏损）/|\(亏损\)/)?利润$",
            r"^归属于上市公司(?:普通股)?股东的扣除非经常性损益的净(?:亏损/)?利润$",
        ),
        (),
    ),
    "OPERATING_CASH_FLOW_YTD": (
        (
            r"^经营活动产生的现金流量净额$",
            r"^经营活动产生(?:（使用）|\(使用\))的现金流量净额$",
            r"^经营活动使用的现金流量净额$",
        ),
        (),
    ),
    "TOTAL_ASSETS_END": ((r"^总资产$",), ()),
}


def _income_section_has_specific_operating_revenue(
    page_texts: Sequence[str],
) -> bool:
    income = locate_statement_sections(page_texts)["income"]
    return any(
        re.search(r"其中[:：]?营业收入", compact_financial_text(page_texts[page_index]))
        is not None
        for page_index in income.page_indices
    )


def _add_summary_overrides(
    result: dict[str, Any],
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    summary_pages: Sequence[int],
    *,
    period_type: str | None,
) -> dict[str, SummaryDuplicate]:
    duplicates: dict[str, SummaryDuplicate] = {}
    specific_operating_revenue = _income_section_has_specific_operating_revenue(
        page_texts
    )
    for metric_id, (patterns, rejects) in SUMMARY_DEFINITIONS.items():
        candidate = _first_pattern_candidate(
            groups,
            page_texts,
            patterns,
            period_type=period_type,
            role="SUMMARY",
            section_pages=summary_pages,
            reject_tokens=rejects,
        )
        if candidate is None:
            continue
        if (
            metric_id == "OPERATING_REVENUE_YTD"
            and specific_operating_revenue
            and "营业总收入" in compact_financial_text(candidate.label)
        ):
            continue
        if metric_id == "CORE_PARENT_NET_PROFIT_YTD" and compact_financial_text(candidate.label).endswith("扣除"):
            next_page = (
                compact_financial_text(page_texts[candidate.page_index + 1])
                if candidate.page_index + 1 < len(page_texts)
                else ""
            )
            if "非经常性损益的净利润" not in next_page:
                continue
        evidence = _table_evidence(
            metric_id,
            candidate,
            value_period_scope=("PERIOD_END" if metric_id == "TOTAL_ASSETS_END" else "YEAR_TO_DATE"),
            section_name="MAIN_FINANCIAL_HIGHLIGHTS",
            validation={
                "rule": "SUMMARY_TABLE_LABEL_COLUMN_AND_CURRENT_PERIOD_HEADER",
                "summary_page_limit": 40,
            },
        )
        duplicate = SummaryDuplicate(candidate.value_cny, candidate.unit)
        existing = next(
            (
                metric
                for metric in result.get("metrics") or []
                if str(metric.get("metric_id")) == metric_id
            ),
            None,
        )
        existing_value = (
            Decimal(str(existing["metric_value_cny"])) if existing is not None else None
        )
        if existing_value is None or not _same_display_identity(
            existing_value,
            duplicate.value_cny,
            duplicate.unit,
        ):
            _put_metric(result, evidence)
        duplicates[metric_id] = duplicate
    return duplicates


BALANCE_PATTERNS: dict[str, Sequence[str]] = {
    "ACCOUNTS_RECEIVABLE_END": (r"^应收账款$",),
    "INVENTORY_END": (r"^存货$",),
    "TOTAL_ASSETS_END": (r"^资产总计$",),
    "TOTAL_LIABILITIES_END": (r"^负债(?:合计|总计)$",),
    "TOTAL_EQUITY_END": (
        r"^(?:所有者权益|股东权益|所有者权益（或股东权益）)合计$",
    ),
    "TOTAL_LIABILITIES_EQUITY_END": (
        r"^负债(?:和|及)所有者权益(?:（或股东权益）)?总计$",
        r"^负债(?:和|及)股东权益总计$",
    ),
}


def _same_display_identity(left: Decimal, right: Decimal, unit: str) -> bool:
    tolerance = _base.UNIT_MULTIPLIERS[unit]
    return abs(left - right) <= tolerance


def _text_evidence(
    metric_id: str,
    candidate: _base.ParsedCandidate,
    *,
    value_period_scope: str,
    section_name: str,
    validation: dict[str, Any],
    text_engine: str = "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
) -> MetricEvidence:
    locator = {
        "page": candidate.page_number,
        "section": section_name,
        "line_window": candidate.line_window[:700],
        "label_span": list(candidate.match_span),
        "value_span": list(candidate.value_span),
        "selected_amount_index": candidate.selected_amount_index,
        "validation": validation,
        "parser_version": PARSER_VERSION,
        "text_engine": text_engine,
    }
    return MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=candidate.page_number,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method=(
            f"{text_engine}_LOGICAL_ROW_SEMANTIC_CURRENT_CONSOLIDATED_PERIOD_RECONCILED"
        ),
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _summary_text_candidate(
    page_texts: Sequence[str],
    summary_pages: Sequence[int],
    patterns: Sequence[str],
    *,
    period_type: str | None,
    metric_id: str,
) -> _base.ParsedCandidate | None:
    q3 = str(period_type or "").upper() == "Q3"
    compiled = [re.compile(pattern) for pattern in patterns]
    inline_unit_prefix = re.compile(
        rf"^\s*[（(]\s*(?P<unit>{_base.UNIT_PATTERN})(?:人民币)?\s*[）)]"
        r"\s*(?:注\s*\d+)?\s*"
    )
    for page_index in summary_pages:
        compact = compact_financial_text(page_texts[page_index])
        previous = (
            compact_financial_text(page_texts[page_index - 1])
            if page_index > 0
            else ""
        )
        if not any(
            marker in compact or marker in previous
            for marker in ("主要会计数据", "主要财务数据", "主要会计资料")
        ):
            continue
        if q3 and not (
            any(marker in compact for marker in ("7-9月", "7—9月", "7－9月", "本报告期"))
            and any(
                marker in compact
                for marker in ("1-9月", "1—9月", "1－9月", "年初至报告期末")
            )
        ):
            continue

        # 部分深交所季报把“（千元人民币）注3”单独放在标签下一行。
        # 基础解析器会把注号视为首个数字；这里仅接纳这一精确前缀。基础金额
        # 扫描会过滤百分比，因此Q3中要求两个金额之间确有同比百分号，才把第
        # 二个非百分比金额解释为年初至报告期末值。
        for window in _base.logical_line_windows(page_texts[page_index]):
            for pattern in compiled:
                label_match = pattern.search(window)
                if label_match is None:
                    continue
                prefix = inline_unit_prefix.match(window[label_match.end() :])
                if prefix is None:
                    continue
                value_start = label_match.end() + prefix.end()
                amount_matches = _base._number_matches_after(
                    window[: min(len(window), value_start + 260)],
                    value_start,
                )
                amount_index = 1 if q3 else 0
                if len(amount_matches) <= amount_index:
                    continue
                if q3:
                    comparison_region = window[
                        amount_matches[0].end() : amount_matches[1].start()
                    ]
                    if "%" not in comparison_region:
                        continue
                amount_match = amount_matches[amount_index]
                unit = prefix.group("unit")
                raw_value = _base._raw_amount(amount_match)
                return _base.ParsedCandidate(
                    value_cny=(
                        _base.parse_decimal(raw_value) * _base.UNIT_MULTIPLIERS[unit]
                    ),
                    page_number=page_index + 1,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=window[:500],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=amount_index,
                )
        if q3:
            continue
        unit = _unit_near_page(page_texts, page_index, (page_index,))
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        candidate = _v1_3._find_candidate(
            page_texts,
            (page_index,),
            patterns,
            unit=unit,
            skip_note=False,
        )
        if candidate is not None:
            return candidate
    return None


def _add_text_summary_overrides(
    result: dict[str, Any],
    page_texts: Sequence[str],
    summary_pages: Sequence[int],
    *,
    period_type: str | None,
) -> dict[str, SummaryDuplicate]:
    duplicates: dict[str, SummaryDuplicate] = {}
    specific_operating_revenue = _income_section_has_specific_operating_revenue(
        page_texts
    )
    text_patterns: dict[str, Sequence[str]] = {
        "OPERATING_REVENUE_YTD": (r"(?<!总)营业收入", r"营业总收入"),
        "PARENT_NET_PROFIT_YTD": (
            r"归属于上市公司(?:普通股)?股东的净利润",
            r"归属于母公司(?:所有者|股东)的净利润",
        ),
        "CORE_PARENT_NET_PROFIT_YTD": (
            r"归属于上市公司(?:普通股)?股东的扣除非经常性损益的净(?:（亏损）[/／]|\(亏损\)[/／]|亏损[/／])?利润",
            r"归属于上市公司(?:普通股)?股东的扣除非经常性损益的净利润",
            r"扣除非经常性损益后的净利润",
        ),
        "OPERATING_CASH_FLOW_YTD": (
            r"经营活动产生(?:（使用）|\(使用\))?的现金流量净额",
            r"经营活动使用的现金流量净额",
        ),
        "TOTAL_ASSETS_END": (r"总资产",),
    }
    for metric_id, patterns in text_patterns.items():
        candidate = _summary_text_candidate(
            page_texts,
            summary_pages,
            patterns,
            period_type=period_type,
            metric_id=metric_id,
        )
        if candidate is None:
            continue
        if (
            metric_id == "OPERATING_REVENUE_YTD"
            and specific_operating_revenue
            and "营业总收入" in compact_financial_text(candidate.label)
        ):
            continue
        duplicate = SummaryDuplicate(candidate.value_cny, candidate.unit)
        duplicates[metric_id] = duplicate
        existing = next(
            (
                metric
                for metric in result.get("metrics") or []
                if str(metric.get("metric_id")) == metric_id
            ),
            None,
        )
        existing_value = (
            Decimal(str(existing["metric_value_cny"])) if existing is not None else None
        )
        if existing_value is not None and _same_display_identity(
            existing_value,
            candidate.value_cny,
            candidate.unit,
        ):
            continue
        _put_metric(
            result,
            _text_evidence(
                metric_id,
                candidate,
                value_period_scope=(
                    "PERIOD_END" if metric_id == "TOTAL_ASSETS_END" else "YEAR_TO_DATE"
                ),
                section_name="MAIN_FINANCIAL_HIGHLIGHTS",
                validation={
                    "rule": "EXACT_MAIN_FINANCIAL_HIGHLIGHTS_LOGICAL_ROW",
                    "summary_page_limit": 40,
                },
            ),
        )
    return duplicates


def _logical_candidate(
    page_texts: Sequence[str],
    pages: Sequence[int],
    patterns: Sequence[str],
    *,
    unit: str | None,
    amount_index: int = 0,
) -> _base.ParsedCandidate | None:
    if not pages or unit not in _base.UNIT_MULTIPLIERS:
        return None
    return _v1_3._find_candidate(
        page_texts,
        pages,
        patterns,
        unit=unit,
        amount_index=amount_index,
    )


def _first_logical_pattern_candidate(
    page_texts: Sequence[str],
    pages: Sequence[int],
    patterns: Sequence[str],
    *,
    unit: str | None,
    amount_index: int = 0,
) -> _base.ParsedCandidate | None:
    """按标签优先级跨整段查找，避免先出现的宽泛标签抢占精确标签。"""

    for pattern in patterns:
        candidate = _logical_candidate(
            page_texts,
            pages,
            (pattern,),
            unit=unit,
            amount_index=amount_index,
        )
        if candidate is not None:
            return candidate
    return None


def _add_logical_balance_overrides(
    result: dict[str, Any],
    page_texts: Sequence[str],
    section: SectionRange,
) -> None:
    if not section.page_indices or section.unit not in _base.UNIT_MULTIPLIERS:
        return
    definitions: dict[str, Sequence[str]] = {
        "ACCOUNTS_RECEIVABLE_END": (r"(?<!应收票据及)(?<!其他)应收账款",),
        "INVENTORY_END": (r"(?<!周转)存货",),
        "TOTAL_ASSETS_END": (r"(?<!流动)(?<!非流动)资产总计",),
        "TOTAL_LIABILITIES_END": (r"(?<!流动)(?<!非流动)负债(?:合计|总计)",),
        "TOTAL_EQUITY_END": (
            r"(?<!母公司)(?<!少数)(?:所有者权益|(?<!普通股)股东权益)(?:（或股东权益）)?合计",
        ),
        "TOTAL_LIABILITIES_EQUITY_END": (
            r"负债(?:和|及)(?:所有者权益|股东权益)(?:（或股东权益）)?总计",
        ),
    }
    selected = {
        metric_id: _logical_candidate(
            page_texts,
            section.page_indices,
            patterns,
            unit=section.unit,
        )
        for metric_id, patterns in definitions.items()
    }
    # 应收账款与存货本身不参与资产负债恒等式，但在已经通过“合并资产负债表”
    # 章节定位后，首个数值列就是本期合并数。先独立接纳这两个精确行，避免因为
    # 跨页的权益合计或负债和权益总计未被文本层识别而连带丢失明细项目。
    detail_validation = {
        "rule": "EXACT_CONSOLIDATED_BALANCE_LOGICAL_ROW_CURRENT_PERIOD_COLUMN",
        "statement_scope": "CONSOLIDATED_ONLY",
        "selected_amount_semantics": "CURRENT_PERIOD_CONSOLIDATED_FIRST_NUMERIC_COLUMN",
    }
    for metric_id in ("ACCOUNTS_RECEIVABLE_END", "INVENTORY_END"):
        candidate = selected.get(metric_id)
        if candidate is None:
            continue
        _put_metric(
            result,
            _text_evidence(
                metric_id,
                candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_BALANCE_SHEET",
                validation=detail_validation,
            ),
        )

    required = (
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
        "TOTAL_EQUITY_END",
    )
    if any(selected[key] is None for key in required):
        return
    assets = selected["TOTAL_ASSETS_END"]
    liabilities = selected["TOTAL_LIABILITIES_END"]
    equity = selected["TOTAL_EQUITY_END"]
    total = selected["TOTAL_LIABILITIES_EQUITY_END"]
    assert assets is not None and liabilities is not None and equity is not None
    if not _same_display_identity(
        assets.value_cny,
        liabilities.value_cny + equity.value_cny,
        section.unit,
    ):
        return
    if total is not None and not _same_display_identity(
        assets.value_cny,
        total.value_cny,
        section.unit,
    ):
        return
    validation = {
        "rule": (
            "TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY_AND_BALANCE_TOTAL"
            if total is not None
            else "TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY"
        ),
        "total_assets_cny": str(assets.value_cny),
        "total_liabilities_cny": str(liabilities.value_cny),
        "total_equity_cny": str(equity.value_cny),
        "liabilities_equity_total_cny": (
            str(total.value_cny) if total is not None else None
        ),
    }
    for metric_id in (
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
    ):
        candidate = selected.get(metric_id)
        if candidate is None:
            continue
        _put_metric(
            result,
            _text_evidence(
                metric_id,
                candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_BALANCE_SHEET",
                validation=validation,
            ),
        )


def _add_logical_income_overrides(
    result: dict[str, Any],
    page_texts: Sequence[str],
    section: SectionRange,
    summary_duplicates: dict[str, SummaryDuplicate],
    *,
    period_type: str | None,
) -> None:
    if not section.page_indices:
        return
    income_pages, amount_index = _v1_3._income_selection(
        page_texts,
        section,
        period_type,
    )
    unit = section.unit or _unit_near_page(page_texts, income_pages[0], income_pages)
    definitions: dict[str, Sequence[str]] = {
        "OPERATING_REVENUE_YTD": (
            r"其中[:：]?营业收入",
            r"(?:一[、.]?)?营业收入",
            r"(?:一[、.]?)?营业总收入",
        ),
        "OPERATING_PROFIT_YTD": (
            r"(?:[一二三四五六七八九十][、.]?\s*)?营业利润\s*[（(](?:亏损|损失)以[^）)]{0,50}填列[）)]?",
            r"(?:[一二三四五六七八九十][、.]?\s*)?营业(?:损失[/／])?利润",
            r"(?:[一二三四五六七八九十][、.]?\s*)?营业[（(](?:亏损|损失)[）)][/／]利润",
            r"(?:[一二三四五六七八九十][、.]?\s*)?营业(?:亏损|损失)[/／]利润",
        ),
        "PARENT_NET_PROFIT_YTD": (
            r"归属于母公司(?:普通股)?(?:所有者|股东)的净利润",
            r"归属于本公司股东的净利润",
            r"归属于上市公司股东的净利润",
            r"母公司股东(?:的净利润)?",
            r"归属于母公司普通股股东",
        ),
        "NET_PROFIT_YTD": (r"(?:[一二三四五六七八九十][、.]?\s*)?净利润",),
        "MINORITY_PROFIT_YTD": (r"少数股东损益",),
        "PERPETUAL_PROFIT_YTD": (r"归属于(?:永续票据|永续债)持有者",),
        "NONOPERATING_INCOME_YTD": (r"加[:：]?营业外收入", r"营业外收入"),
        "NONOPERATING_EXPENSE_YTD": (r"减[:：]?营业外支出", r"营业外支出"),
        "PROFIT_TOTAL_YTD": (r"(?:[一二三四五六七八九十][、.]?\s*)?利润总额",),
    }
    selected = {
        metric_id: _logical_candidate(
            page_texts,
            income_pages,
            patterns,
            unit=unit,
            amount_index=amount_index,
        )
        for metric_id, patterns in definitions.items()
    }
    selected["OPERATING_REVENUE_YTD"] = _first_logical_pattern_candidate(
        page_texts,
        income_pages,
        definitions["OPERATING_REVENUE_YTD"],
        unit=unit,
        amount_index=amount_index,
    )
    op_valid = False
    if all(
        selected[key] is not None
        for key in (
            "OPERATING_PROFIT_YTD",
            "NONOPERATING_INCOME_YTD",
            "NONOPERATING_EXPENSE_YTD",
            "PROFIT_TOTAL_YTD",
        )
    ):
        op = selected["OPERATING_PROFIT_YTD"]
        income = selected["NONOPERATING_INCOME_YTD"]
        expense = selected["NONOPERATING_EXPENSE_YTD"]
        total = selected["PROFIT_TOTAL_YTD"]
        assert op is not None and income is not None and expense is not None and total is not None
        op_valid = _same_display_identity(
            op.value_cny + income.value_cny - expense.value_cny,
            total.value_cny,
            op.unit,
        )
    parent_valid = False
    parent = selected.get("PARENT_NET_PROFIT_YTD")
    if parent is not None:
        duplicate = summary_duplicates.get("PARENT_NET_PROFIT_YTD")
        if duplicate is not None and _same_display_identity(
            parent.value_cny,
            duplicate.value_cny,
            duplicate.unit,
        ):
            parent_valid = True
        elif selected.get("NET_PROFIT_YTD") is not None and selected.get("MINORITY_PROFIT_YTD") is not None:
            net = selected["NET_PROFIT_YTD"]
            minority = selected["MINORITY_PROFIT_YTD"]
            assert net is not None and minority is not None
            components = parent.value_cny + minority.value_cny
            perpetual = selected.get("PERPETUAL_PROFIT_YTD")
            if perpetual is not None:
                components += perpetual.value_cny
            parent_valid = _same_display_identity(components, net.value_cny, parent.unit)
    for metric_id in (
        "OPERATING_REVENUE_YTD",
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    ):
        candidate = selected.get(metric_id)
        if candidate is None:
            continue
        duplicate = summary_duplicates.get(metric_id)
        if duplicate is not None and not _same_display_identity(
            candidate.value_cny,
            duplicate.value_cny,
            duplicate.unit,
        ):
            continue
        _put_metric(
            result,
            _text_evidence(
                metric_id,
                candidate,
                value_period_scope="YEAR_TO_DATE",
                section_name="CONSOLIDATED_INCOME_STATEMENT",
                validation={
                    "rule": (
                        "OPERATING_PROFIT_RECONCILES_TO_PROFIT_TOTAL"
                        if metric_id == "OPERATING_PROFIT_YTD" and op_valid
                        else (
                            "PARENT_PROFIT_MATCHES_SUMMARY_OR_RECONCILES_TO_NET_PROFIT"
                            if metric_id == "PARENT_NET_PROFIT_YTD" and parent_valid
                            else "EXACT_CONSOLIDATED_STATEMENT_LOGICAL_ROW_WITH_SEMANTIC_PERIOD_COLUMN"
                        )
                    ),
                    "summary_duplicate_cny": (
                        str(duplicate.value_cny) if duplicate is not None else None
                    ),
                },
            ),
        )


def _add_logical_cash_override(
    result: dict[str, Any],
    page_texts: Sequence[str],
    section: SectionRange,
    summary_duplicates: dict[str, SummaryDuplicate],
    *,
    period_type: str | None,
) -> None:
    del period_type
    if not section.page_indices:
        return
    unit = section.unit or _unit_near_page(
        page_texts,
        section.page_indices[0],
        section.page_indices,
    )
    candidate = _logical_candidate(
        page_texts,
        section.page_indices,
        (
            r"经营活动产生(?:（使用）|\(使用\))?的现金流量净额",
            r"经营活动使用的现金流量净额",
        ),
        unit=unit,
    )
    if candidate is None:
        return
    duplicate = summary_duplicates.get("OPERATING_CASH_FLOW_YTD")
    if duplicate is not None and not _same_display_identity(
        candidate.value_cny,
        duplicate.value_cny,
        duplicate.unit,
    ):
        return
    _put_metric(
        result,
        _text_evidence(
            "OPERATING_CASH_FLOW_YTD",
            candidate,
            value_period_scope="YEAR_TO_DATE",
            section_name="CONSOLIDATED_CASH_FLOW_STATEMENT",
            validation={
                "rule": "STATEMENT_VALUE_MATCHES_SUMMARY_DUPLICATE_WHEN_AVAILABLE",
                "summary_duplicate_cny": (
                    str(duplicate.value_cny) if duplicate is not None else None
                ),
            },
        ),
    )


def _add_balance_overrides(
    result: dict[str, Any],
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    section: SectionRange,
    *,
    period_type: str | None,
) -> None:
    if not section.page_indices:
        return
    selected: dict[str, TableCandidate] = {}
    for metric_id, patterns in BALANCE_PATTERNS.items():
        rejects = ("归属于", "少数") if metric_id == "TOTAL_EQUITY_END" else ()
        candidate = _best_candidate(
            _table_candidates(
                groups,
                page_texts,
                patterns,
                period_type=period_type,
                role="BALANCE",
                section_pages=section.page_indices,
                reject_tokens=rejects,
            )
        )
        if candidate is not None:
            selected[metric_id] = candidate
    required = (
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
        "TOTAL_EQUITY_END",
        "TOTAL_LIABILITIES_EQUITY_END",
    )
    if any(key not in selected for key in required):
        return
    unit = selected["TOTAL_ASSETS_END"].unit
    if any(selected[key].unit != unit for key in required):
        return
    assets = selected["TOTAL_ASSETS_END"].value_cny
    liabilities = selected["TOTAL_LIABILITIES_END"].value_cny
    equity = selected["TOTAL_EQUITY_END"].value_cny
    total = selected["TOTAL_LIABILITIES_EQUITY_END"].value_cny
    if not _same_display_identity(assets, liabilities + equity, unit):
        return
    if not _same_display_identity(assets, total, unit):
        return
    validation = {
        "rule": "TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY_AND_BALANCE_TOTAL",
        "total_assets_cny": str(assets),
        "total_liabilities_cny": str(liabilities),
        "total_equity_cny": str(equity),
        "liabilities_equity_total_cny": str(total),
    }
    for metric_id in (
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
    ):
        candidate = selected.get(metric_id)
        if candidate is None:
            continue
        _put_metric(
            result,
            _table_evidence(
                metric_id,
                candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_BALANCE_SHEET",
                validation=validation,
            ),
        )


INCOME_PATTERNS: dict[str, tuple[Sequence[str], Sequence[str]]] = {
    "OPERATING_REVENUE_YTD": (
        (r"^(?:一、)?营业收入$", r"^(?:一、)?营业总收入$"),
        (),
    ),
    "OPERATING_PROFIT_YTD": (
        (
            r"^(?:[一二三四五六七八九十]、)?营业利润.*$",
            r"^(?:[一二三四五六七八九十]、)?营业\(损失\)/利润$",
        ),
        ("综合收益",),
    ),
    "PARENT_NET_PROFIT_YTD": (
        (
            r"^(?:其中：)?归属于(?:母公司|本公司)(?:普通股)?(?:所有者|股东)?的?净.*利润.*$",
            r"^归属于上市公司股东的净利润$",
            r"^母公司股东(?:的净利润)?$",
            r"^归属于母公司普通股股东$",
        ),
        ("综合收益", "扣除"),
    ),
    "NET_PROFIT_YTD": (
        (r"^(?:[一二三四五六七八九十]、)?净利润.*$",),
        ("归属于", "持续经营", "综合收益"),
    ),
    "MINORITY_PROFIT_YTD": ((r"^(?:\d+[.、])?少数股东损益.*$",), ("综合收益",)),
    "PERPETUAL_PROFIT_YTD": (
        (r"^归属于(?:永续票据|永续债)持有者.*$",),
        ("综合收益",),
    ),
    "NONOPERATING_INCOME_YTD": ((r"^加[:：]?营业外收入$", r"^营业外收入$"), ()),
    "NONOPERATING_EXPENSE_YTD": ((r"^减[:：]?营业外支出$", r"^营业外支出$"), ()),
    "PROFIT_TOTAL_YTD": (
        (r"^(?:[一二三四五六七八九十]、)?利润.*总额.*$",),
        ("营业利润",),
    ),
}


def _income_candidates(
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    section: SectionRange,
    period_type: str | None,
) -> dict[str, TableCandidate]:
    selected: dict[str, TableCandidate] = {}
    for key, (patterns, rejects) in INCOME_PATTERNS.items():
        candidate = _first_pattern_candidate(
            groups,
            page_texts,
            patterns,
            period_type=period_type,
            role="INCOME",
            section_pages=section.page_indices,
            reject_tokens=rejects,
        )
        if candidate is not None:
            selected[key] = candidate
    return selected


def _add_income_overrides(
    result: dict[str, Any],
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    section: SectionRange,
    summary_duplicates: dict[str, SummaryDuplicate],
    *,
    period_type: str | None,
) -> None:
    if not section.page_indices:
        return
    selected = _income_candidates(groups, page_texts, section, period_type)
    op_valid = False
    if all(
        key in selected
        for key in (
            "OPERATING_PROFIT_YTD",
            "NONOPERATING_INCOME_YTD",
            "NONOPERATING_EXPENSE_YTD",
            "PROFIT_TOTAL_YTD",
        )
    ):
        unit = selected["OPERATING_PROFIT_YTD"].unit
        values = {key: selected[key].value_cny for key in selected}
        op_valid = _same_display_identity(
            values["OPERATING_PROFIT_YTD"]
            + values["NONOPERATING_INCOME_YTD"]
            - values["NONOPERATING_EXPENSE_YTD"],
            values["PROFIT_TOTAL_YTD"],
            unit,
        )
    parent_valid = False
    if "PARENT_NET_PROFIT_YTD" in selected:
        parent = selected["PARENT_NET_PROFIT_YTD"].value_cny
        duplicate = summary_duplicates.get("PARENT_NET_PROFIT_YTD")
        if duplicate is not None and _same_display_identity(
            parent,
            duplicate.value_cny,
            duplicate.unit,
        ):
            parent_valid = True
        elif all(key in selected for key in ("NET_PROFIT_YTD", "MINORITY_PROFIT_YTD")):
            components = parent + selected["MINORITY_PROFIT_YTD"].value_cny
            if "PERPETUAL_PROFIT_YTD" in selected:
                components += selected["PERPETUAL_PROFIT_YTD"].value_cny
            parent_valid = _same_display_identity(
                components,
                selected["NET_PROFIT_YTD"].value_cny,
                selected["PARENT_NET_PROFIT_YTD"].unit,
            )
    for metric_id in ("OPERATING_REVENUE_YTD", "OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD"):
        candidate = selected.get(metric_id)
        if candidate is None:
            continue
        duplicate = summary_duplicates.get(metric_id)
        if (
            metric_id == "OPERATING_REVENUE_YTD"
            and duplicate is not None
            and not _same_display_identity(
                candidate.value_cny,
                duplicate.value_cny,
                duplicate.unit,
            )
        ):
            continue
        if metric_id == "OPERATING_PROFIT_YTD" and not op_valid:
            validation = {"rule": "EXACT_CONSOLIDATED_STATEMENT_TABLE_ROW_WITH_SEMANTIC_PERIOD_COLUMN"}
        elif metric_id == "PARENT_NET_PROFIT_YTD" and not parent_valid:
            validation = {"rule": "EXACT_CONSOLIDATED_STATEMENT_TABLE_ROW_WITH_SEMANTIC_PERIOD_COLUMN"}
        else:
            validation = {
                "rule": (
                    "OPERATING_PROFIT_RECONCILES_TO_PROFIT_TOTAL"
                    if metric_id == "OPERATING_PROFIT_YTD"
                    else (
                        "PARENT_PROFIT_MATCHES_SUMMARY_OR_RECONCILES_TO_NET_PROFIT"
                        if metric_id == "PARENT_NET_PROFIT_YTD"
                        else "STATEMENT_VALUE_MATCHES_SUMMARY_DUPLICATE_WHEN_AVAILABLE"
                    )
                ),
                "summary_duplicate_cny": (
                    str(duplicate.value_cny) if duplicate is not None else None
                ),
            }
        _put_metric(
            result,
            _table_evidence(
                metric_id,
                candidate,
                value_period_scope="YEAR_TO_DATE",
                section_name="CONSOLIDATED_INCOME_STATEMENT",
                validation=validation,
            ),
        )


def _add_cash_override(
    result: dict[str, Any],
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    section: SectionRange,
    summary_duplicates: dict[str, SummaryDuplicate],
    *,
    period_type: str | None,
) -> None:
    if not section.page_indices:
        return
    candidate = _best_candidate(
        _table_candidates(
            groups,
            page_texts,
            (
                r"^经营活动产生(?:（使用）|\(使用\))?的现金流量净额$",
                r"^经营活动使用的现金流量净额$",
            ),
            period_type=period_type,
            role="CASH",
            section_pages=section.page_indices,
        )
    )
    if candidate is None:
        return
    duplicate = summary_duplicates.get("OPERATING_CASH_FLOW_YTD")
    if duplicate is not None and not _same_display_identity(
        candidate.value_cny,
        duplicate.value_cny,
        duplicate.unit,
    ):
        return
    _put_metric(
        result,
        _table_evidence(
            "OPERATING_CASH_FLOW_YTD",
            candidate,
            value_period_scope="YEAR_TO_DATE",
            section_name="CONSOLIDATED_CASH_FLOW_STATEMENT",
            validation={
                "rule": "STATEMENT_VALUE_MATCHES_SUMMARY_DUPLICATE_WHEN_AVAILABLE",
                "summary_duplicate_cny": (
                    str(duplicate.value_cny) if duplicate is not None else None
                ),
            },
        ),
    )


def _note_metric_candidate(
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    metric_id: str,
    patterns: Sequence[str],
    *,
    period_type: str | None,
) -> TableCandidate | None:
    pages = tuple(range(len(page_texts)))
    candidates = _table_candidates(
        groups,
        page_texts,
        patterns,
        period_type=period_type,
        role="BALANCE",
        section_pages=pages,
        reject_tokens=("长期", "其他", "票据及", "保理") if metric_id == "ACCOUNTS_RECEIVABLE_END" else (),
    )
    filtered: list[TableCandidate] = []
    for candidate in candidates:
        page_compact = compact_financial_text(page_texts[candidate.page_index])
        if not any(marker in page_compact for marker in ("财务报表附注", "财务报表注释", "合并财务报表")):
            continue
        amount_cells = [
            _parse_amount_cell(cell)
            for cell in candidate.row_cells[candidate.label_cell_index + 1 :]
        ]
        if sum(value is not None for value in amount_cells) < 2:
            continue
        intervening = candidate.row_cells[
            candidate.label_cell_index + 1 : candidate.value_cell_index
        ]
        if any(
            compact_financial_text(cell)
            and re.fullmatch(
                r"(?:附注|注释)?[（()）\[\]【】一二三四五六七八九十百千万\d.,、/／-]+",
                compact_financial_text(cell),
            )
            is None
            for cell in intervening
        ):
            continue
        group = groups[(candidate.page_index, candidate.table_index)]
        headers = "".join(
            _column_header(group, candidate.row_index, column_index)
            for column_index in range(candidate.label_cell_index + 1, len(candidate.row_cells))
        )
        if not any(marker in headers for marker in ("期末", "年末", "本期")):
            continue
        if not any(marker in headers for marker in ("期初", "年初", "上年")):
            continue
        filtered.append(candidate)
    return _best_candidate(filtered)


def _repair_note_balance_metrics(
    result: dict[str, Any],
    groups: dict[tuple[int, int], list[tuple[int, tuple[str, ...]]]],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> None:
    for metric_id, patterns in (
        ("ACCOUNTS_RECEIVABLE_END", (r"^应收账款$",)),
        ("INVENTORY_END", (r"^存货$",)),
    ):
        if metric_id not in result.get("missing_metrics", []):
            continue
        candidate = _note_metric_candidate(
            groups,
            page_texts,
            metric_id,
            patterns,
            period_type=period_type,
        )
        if candidate is None:
            continue
        _put_metric(
            result,
            _table_evidence(
                metric_id,
                candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_STATEMENT_NOTES",
                validation={
                    "rule": "EXACT_NOTE_TABLE_ROW_CURRENT_PERIOD_COLUMN",
                    "used_only_as_balance_statement_repair": True,
                },
            ),
        )


def _prune_untrusted_statement_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    sections: dict[str, SectionRange],
) -> list[str]:
    roles = {
        "OPERATING_REVENUE_YTD": "income",
        "OPERATING_PROFIT_YTD": "income",
        "PARENT_NET_PROFIT_YTD": "income",
        "OPERATING_CASH_FLOW_YTD": "cash",
        "ACCOUNTS_RECEIVABLE_END": "balance",
        "INVENTORY_END": "balance",
        "TOTAL_ASSETS_END": "balance",
        "TOTAL_LIABILITIES_END": "balance",
    }
    summary_allowed = {
        "OPERATING_REVENUE_YTD",
        "PARENT_NET_PROFIT_YTD",
        "OPERATING_CASH_FLOW_YTD",
        "TOTAL_ASSETS_END",
    }
    balance_admission_rules = {
        "BILLS_PLUS_ACCOUNTS_RECEIVABLE_EQUALS_COMBINED_TOTAL_CURRENT_AND_PRIOR",
        "EXACT_ACCOUNTS_RECEIVABLE_LABEL_AND_ALL_TRAILING_CURRENT_AND_PRIOR_AMOUNT_CELLS_EXPLICITLY_BLANK",
        "EXACT_INVENTORY_LABEL_AND_ALL_TRAILING_CURRENT_AND_PRIOR_AMOUNT_CELLS_EXPLICITLY_BLANK",
        "EXACT_NOTE_TABLE_ROW_CURRENT_PERIOD_COLUMN",
        "TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY_AND_BALANCE_TOTAL",
        "TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY",
    }
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    for metric in result.get("metrics") or []:
        metric_id = str(metric.get("metric_id") or "")
        role = roles.get(metric_id)
        if role is None:
            kept.append(metric)
            continue
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {}
        section_name = str(locator.get("section") or "")
        validation = locator.get("validation") or {}
        validation_rule = str(
            validation.get("rule")
            or locator.get("validation_rule")
            or ""
        )
        if metric_id in summary_allowed and section_name.startswith("MAIN_FINANCIAL"):
            kept.append(metric)
            continue
        if role == "balance" and validation_rule in balance_admission_rules:
            kept.append(metric)
            continue
        try:
            page_index = int(locator.get("page") or metric.get("source_page")) - 1
        except (TypeError, ValueError):
            page_index = -1
        trusted_pages = set(sections[role].page_indices)
        trusted = page_index in trusted_pages
        if trusted and 0 <= page_index < len(page_texts):
            compact = compact_financial_text(page_texts[page_index])
            company_markers = {
                "balance": ("母公司资产负债表", "公司资产负债表"),
                "income": ("母公司利润表", "公司利润表"),
                "cash": ("母公司现金流量表", "公司现金流量表"),
            }[role]
            if any(marker in compact for marker in company_markers) and "合并" not in compact:
                trusted = False
        if trusted:
            kept.append(metric)
        else:
            removed.append(metric_id)
    result["metrics"] = kept
    _refresh_result(result)
    return removed


def _manual_appendix_documents() -> dict[str, dict[str, Any]]:
    if not MANUAL_APPENDIX_PATH.exists():
        return {}
    payload = json.loads(MANUAL_APPENDIX_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("V1.4哈希绑定人工核验附录schema_version不匹配")
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("V1.4哈希绑定人工核验附录documents格式错误")
    return {str(key): value for key, value in documents.items()}


def _validate_manual_appendix_entry(
    entry: dict[str, Any],
    *,
    content: bytes,
    pdf_sha256: str,
) -> None:
    if int(entry.get("official_pdf_size_bytes") or -1) != len(content):
        raise ValueError("V1.4人工核验附录PDF字节数不匹配")
    facts = entry.get("facts") or []
    metric_ids = [str(fact.get("metric_id") or "") for fact in facts]
    if sorted(metric_ids) != sorted(REQUIRED_METRICS) or len(set(metric_ids)) != len(REQUIRED_METRICS):
        raise ValueError("V1.4人工核验附录必须恰好覆盖九项必需事实")
    values = {
        str(fact["metric_id"]): (
            Decimal(str(fact["raw_value"]))
            * _base.UNIT_MULTIPLIERS[str(fact["unit"])]
        )
        for fact in facts
    }
    validation = entry.get("validation") or {}
    balance = validation.get("balance_identity") or {}
    if balance:
        assets = values["TOTAL_ASSETS_END"]
        liabilities = values["TOTAL_LIABILITIES_END"]
        equity = Decimal(str(balance["total_equity_cny"]))
        total = Decimal(str(balance["liabilities_equity_total_cny"]))
        if assets != liabilities + equity or assets != total:
            raise ValueError("V1.4人工核验附录资产负债恒等式失败")
    profit = validation.get("profit_identity") or {}
    if profit:
        operating = values["OPERATING_PROFIT_YTD"]
        nonoperating_income = Decimal(str(profit["nonoperating_income_cny"]))
        nonoperating_expense = Decimal(str(profit["nonoperating_expense_cny"]))
        profit_total = Decimal(str(profit["profit_total_cny"]))
        parent = values["PARENT_NET_PROFIT_YTD"]
        minority = Decimal(str(profit["minority_profit_cny"]))
        net_profit = Decimal(str(profit["net_profit_cny"]))
        if operating + nonoperating_income - nonoperating_expense != profit_total:
            raise ValueError("V1.4人工核验附录营业利润恒等式失败")
        if parent + minority != net_profit:
            raise ValueError("V1.4人工核验附录归母利润恒等式失败")
    if str(entry.get("official_pdf_sha256") or pdf_sha256) != pdf_sha256:
        raise ValueError("V1.4人工核验附录PDF哈希字段不匹配")


def _apply_manual_appendix(
    result: dict[str, Any],
    entry: dict[str, Any],
    *,
    content: bytes,
    pdf_sha256: str,
) -> dict[str, Any]:
    _validate_manual_appendix_entry(
        entry,
        content=content,
        pdf_sha256=pdf_sha256,
    )
    existing = {
        str(metric["metric_id"]): Decimal(str(metric["metric_value_cny"]))
        for metric in result.get("metrics") or []
    }
    facts = entry.get("facts") or []
    fact_by_metric = {str(fact["metric_id"]): fact for fact in facts}
    duplicate_checks: dict[str, bool] = {}
    for metric_id in entry.get("duplicate_metric_ids") or []:
        fact = fact_by_metric[str(metric_id)]
        expected = Decimal(str(fact["raw_value"])) * _base.UNIT_MULTIPLIERS[str(fact["unit"])]
        if str(metric_id) in existing:
            duplicate_checks[str(metric_id)] = _same_display_identity(
                existing[str(metric_id)],
                expected,
                str(fact["unit"]),
            )
    if duplicate_checks and not all(duplicate_checks.values()):
        raise ValueError("V1.4人工核验附录与独立文本重复值不一致")
    for fact in facts:
        metric_id = str(fact["metric_id"])
        unit = str(fact["unit"])
        raw_value = str(fact["raw_value"])
        value_cny = Decimal(raw_value) * _base.UNIT_MULTIPLIERS[unit]
        locator = {
            "page": int(fact["page"]),
            "section": str(fact["section"]),
            "official_pdf_sha256": pdf_sha256,
            "appendix_path": MANUAL_APPENDIX_PATH.relative_to(_v1_3.ROOT).as_posix(),
            "appendix_document_key": str(entry["document_key"]),
            "validation": entry.get("validation") or {},
            "duplicate_checks": duplicate_checks,
            "parser_version": PARSER_VERSION,
            "text_engine": "HASH_BOUND_OFFICIAL_PDF_VISUAL_OR_TEXT_TRANSCRIPTION",
        }
        _put_metric(
            result,
            MetricEvidence(
                metric_id=metric_id,
                metric_value_cny=float(value_cny),
                statement_scope="CONSOLIDATED_ONLY",
                value_period_scope=str(fact["value_period_scope"]),
                source_page=int(fact["page"]),
                source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
                source_label=str(fact["source_label"]),
                source_raw_value=raw_value,
                source_unit=unit,
                source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
                source_method="HASH_BOUND_OFFICIAL_PDF_VERIFIED_TRANSCRIPTION",
                verification_status=ADMITTED_VERIFICATION_STATUS,
            ),
        )
    return {
        "status": "PASS_HASH_BOUND_OFFICIAL_PDF_VERIFIED_TRANSCRIPTION",
        "document_key": str(entry["document_key"]),
        "official_pdf_sha256": pdf_sha256,
        "admitted_metric_ids": list(REQUIRED_METRICS),
        "duplicate_checks": duplicate_checks,
        "balance_identity_passed": bool((entry.get("validation") or {}).get("balance_identity")),
        "profit_identity_passed": bool((entry.get("validation") or {}).get("profit_identity")),
    }


def _merge_complete_v1_3_fallback(
    result: dict[str, Any],
    legacy_result: dict[str, Any],
    *,
    pdf_sha256: str,
) -> dict[str, Any]:
    legacy_parser_version = str(legacy_result.get("parser_version") or _v1_3.PARSER_VERSION)
    if str(legacy_result.get("official_pdf_sha256") or "") != pdf_sha256:
        raise ValueError("V1.3完整文档回退的官方PDF哈希不一致")
    legacy_metrics = legacy_result.get("metrics") or []
    legacy_metric_ids = {str(metric.get("metric_id") or "") for metric in legacy_metrics}
    if not legacy_result.get("document_complete") or legacy_metric_ids != set(REQUIRED_METRICS):
        return {
            "status": "REJECTED_V1_3_FALLBACK_DOCUMENT_NOT_COMPLETE",
            "legacy_parser_version": legacy_parser_version,
            "legacy_document_complete": bool(legacy_result.get("document_complete")),
            "legacy_missing_metrics": list(legacy_result.get("missing_metrics") or []),
            "filled_metric_ids": [],
        }

    existing = {str(metric["metric_id"]) for metric in result.get("metrics") or []}
    filled: list[str] = []
    for metric in legacy_metrics:
        metric_id = str(metric["metric_id"])
        if metric_id in existing:
            continue
        admitted = dict(metric)
        try:
            locator = json.loads(str(admitted.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {"legacy_source_locator": str(admitted.get("source_locator") or "")}
        locator["v1_4_admission"] = {
            "rule": "FILL_MISSING_ONLY_FROM_V1_3_COMPLETE_NINE_METRIC_DOCUMENT",
            "legacy_parser_version": legacy_parser_version,
            "official_pdf_sha256": pdf_sha256,
        }
        locator.setdefault("base_parser_version", legacy_parser_version)
        locator["parser_version"] = PARSER_VERSION
        admitted["source_locator"] = json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        admitted["verification_status"] = ADMITTED_VERIFICATION_STATUS
        result.setdefault("metrics", []).append(admitted)
        filled.append(metric_id)
    _refresh_result(result)
    return {
        "status": (
            "PASS_V1_3_COMPLETE_DOCUMENT_FILLED_V1_4_MISSING_METRICS"
            if filled
            else "PASS_V1_3_COMPLETE_DOCUMENT_NO_FILL_NEEDED"
        ),
        "legacy_parser_version": legacy_parser_version,
        "legacy_document_complete": True,
        "legacy_missing_metrics": [],
        "filled_metric_ids": filled,
    }


def extract_metrics_from_page_texts(
    page_texts: Sequence[str],
    *,
    period_type: str | None = None,
    text_engine: str = "PDFIUM",
) -> dict[str, Any]:
    normalized = [normalize_financial_text(text) for text in page_texts]
    result = _v1_3.extract_metrics_from_page_texts(
        normalized,
        period_type=period_type,
        text_engine=text_engine,
    )
    _stamp_inherited_metrics(result)
    sections = locate_statement_sections(normalized)
    result["sections"] = {
        key: {
            "name": section.name,
            "start_page": section.page_indices[0] + 1 if section.page_indices else None,
            "end_page": section.page_indices[-1] + 1 if section.page_indices else None,
            "unit": section.unit,
            "text_engine": text_engine,
        }
        for key, section in sections.items()
    }
    _refresh_result(result)
    return result


def extract_official_pdf_facts(
    content: bytes,
    *,
    period_type: str | None = None,
) -> dict[str, Any]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("响应没有%PDF-文件头")
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    manual_entry = _manual_appendix_documents().get(pdf_sha256)
    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    normalized = [normalize_financial_text(text) for text in page_texts]
    result = extract_metrics_from_page_texts(
        normalized,
        period_type=period_type,
        text_engine="PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
    )
    result["pdfplumber_secondary_text_receipt"] = {
        "status": "NOT_NEEDED_V1_4_PRIMARY_TEXT_AND_RELEVANT_TABLE_PATH"
    }
    result["ocr_fallback_receipt"] = {
        "status": (
            "SKIPPED_HASH_BOUND_VERIFIED_TRANSCRIPTION"
            if manual_entry is not None
            else "NOT_NEEDED_BEFORE_V1_4_RELEVANT_TABLE_AND_LOGICAL_ROW_PATH"
        )
    }
    _stamp_inherited_metrics(result)
    traditional_applied = any(
        original != converted
        for original, converted in zip(page_texts, normalized, strict=True)
    )
    sections = locate_statement_sections(normalized)
    pruned_before_overrides = _prune_untrusted_statement_metrics(
        result,
        normalized,
        sections,
    )
    summary_pages = _summary_pages(normalized, sections)
    table_pages = (
        ()
        if manual_entry is not None
        else tuple(
            sorted(
                set(summary_pages).union(
                    *(set(section.page_indices) for section in sections.values())
                )
            )
        )
    )
    rows, table_errors = _base.extract_pdf_table_rows(content, table_pages)
    groups = _group_table_rows(rows)
    summary_duplicates = _add_text_summary_overrides(
        result,
        normalized,
        summary_pages,
        period_type=period_type,
    )
    summary_duplicates.update(_add_summary_overrides(
        result,
        groups,
        normalized,
        summary_pages,
        period_type=period_type,
    ))
    _add_logical_balance_overrides(
        result,
        normalized,
        sections["balance"],
    )
    _add_balance_overrides(
        result,
        groups,
        normalized,
        sections["balance"],
        period_type=period_type,
    )
    _add_logical_income_overrides(
        result,
        normalized,
        sections["income"],
        summary_duplicates,
        period_type=period_type,
    )
    _add_income_overrides(
        result,
        groups,
        normalized,
        sections["income"],
        summary_duplicates,
        period_type=period_type,
    )
    _add_logical_cash_override(
        result,
        normalized,
        sections["cash"],
        summary_duplicates,
        period_type=period_type,
    )
    _add_cash_override(
        result,
        groups,
        normalized,
        sections["cash"],
        summary_duplicates,
        period_type=period_type,
    )
    _repair_note_balance_metrics(
        result,
        groups,
        normalized,
        period_type=period_type,
    )
    pruned_after_overrides = _prune_untrusted_statement_metrics(
        result,
        normalized,
        sections,
    )
    manual_receipt: dict[str, Any] = {"status": "NOT_APPLICABLE_PDF_HASH_NOT_REGISTERED"}
    if manual_entry is not None:
        manual_receipt = _apply_manual_appendix(
            result,
            manual_entry,
            content=content,
            pdf_sha256=pdf_sha256,
        )
    legacy_receipt: dict[str, Any] = {
        "status": "NOT_NEEDED_V1_4_DOCUMENT_ALREADY_COMPLETE_OR_HASH_APPENDIX_APPLIED",
        "filled_metric_ids": [],
    }
    if manual_entry is None and not result.get("document_complete"):
        try:
            legacy_result = _v1_3.extract_official_pdf_facts(
                content,
                period_type=period_type,
            )
            legacy_receipt = _merge_complete_v1_3_fallback(
                result,
                legacy_result,
                pdf_sha256=pdf_sha256,
            )
        except Exception as error:  # noqa: BLE001 - 回退失败必须落账但不抹掉V1.4证据
            legacy_receipt = {
                "status": "FAILED_V1_3_COMPLETE_DOCUMENT_FALLBACK",
                "error": f"{type(error).__name__}: {error}"[:2000],
                "filled_metric_ids": [],
            }
    result["sections"] = {
        key: {
            "name": section.name,
            "start_page": section.page_indices[0] + 1 if section.page_indices else None,
            "end_page": section.page_indices[-1] + 1 if section.page_indices else None,
            "unit": section.unit,
            "text_engine": "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
        }
        for key, section in sections.items()
    }
    result["v1_4_semantic_table_receipt"] = {
        "status": "PASS_TABLES_EXAMINED" if rows else "NO_TEXT_TABLE_ROWS",
        "examined_page_numbers": [page + 1 for page in table_pages],
        "table_row_count": len(rows),
        "page_errors": table_errors,
        "summary_duplicate_metric_ids": sorted(summary_duplicates),
        "traditional_financial_token_normalization_applied": traditional_applied,
        "pruned_untrusted_metric_ids_before_overrides": pruned_before_overrides,
        "pruned_untrusted_metric_ids_after_overrides": pruned_after_overrides,
    }
    result["hash_bound_manual_appendix_receipt"] = manual_receipt
    result["v1_3_complete_document_fallback_receipt"] = legacy_receipt
    result.update(
        {
            "parser_version": PARSER_VERSION,
            "official_pdf_sha256": pdf_sha256,
            "official_pdf_size_bytes": len(content),
            **text_receipt,
        }
    )
    _stamp_inherited_metrics(result)
    return result


__all__ = [
    "ADMITTED_VERIFICATION_STATUS",
    "PARSER_VERSION",
    "REQUIRED_METRICS",
    "extract_metrics_from_page_texts",
    "extract_official_pdf_facts",
    "locate_statement_sections",
]
