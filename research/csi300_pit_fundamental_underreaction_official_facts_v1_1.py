from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base


PARSER_VERSION = "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_PDFIUM_PDFPLUMBER_V1_4_0"
ADMITTED_VERIFICATION_STATUS = "PASS_OFFICIAL_ORIGINAL_PDF_LABEL_VALUE_UNIT_VERIFIED"
REQUIRED_METRICS = _base.REQUIRED_METRICS
MetricEvidence = _base.MetricEvidence
ParsedCandidate = _base.ParsedCandidate
SectionRange = _base.SectionRange

_BASE_EXTRACT_METRICS = _base.extract_metrics_from_page_texts
_BASE_FIND_AMOUNT = _base.find_amount
_base.PARSER_VERSION = PARSER_VERSION

CURRENCY_UNIT_RE = re.compile(
    rf"(?:人民币|RMB)\s*(?P<unit>{_base.UNIT_PATTERN})",
    flags=re.IGNORECASE,
)


def _unit_candidates(text: str) -> list[str]:
    normalized = _base.normalize_text(text)
    global_units = [
        match.group("unit") for match in _base.GLOBAL_UNIT_RE.finditer(normalized)
    ]
    if global_units:
        return global_units
    currency_units = [
        match.group("unit") for match in CURRENCY_UNIT_RE.finditer(normalized)
    ]
    if currency_units:
        return currency_units
    return [match.group("unit") for match in _base.ROW_UNIT_RE.finditer(normalized)]


def detect_unique_unit(
    page_texts: Sequence[str], page_indices: Iterable[int]
) -> str | None:
    units: list[str] = []
    for page_index in page_indices:
        units.extend(_unit_candidates(page_texts[page_index]))
    unique = list(dict.fromkeys(units))
    return unique[0] if len(unique) == 1 else None


def _same_page_statement_table(
    page_compact: str,
    compact_headings: Sequence[str],
    marker_group: Sequence[str],
) -> bool:
    compact_markers = [_base.compact_text(marker) for marker in marker_group]
    for heading in compact_headings:
        search_from = 0
        while True:
            heading_position = page_compact.find(heading, search_from)
            if heading_position < 0:
                break
            heading_end = heading_position + len(heading)
            marker_positions = [
                page_compact.find(marker, heading_end) for marker in compact_markers
            ]
            if all(position >= 0 for position in marker_positions):
                header_region = page_compact[heading_end : min(marker_positions)]
                if "单位" in header_region or "人民币" in header_region:
                    return True
            search_from = heading_end
    return False


def _small_integer(raw_value: str) -> bool:
    text = raw_value.strip().replace("−", "-").replace("－", "-").replace("—", "-")
    text = text.strip("()")
    return bool(re.fullmatch(r"\d{1,3}", text))


def _note_prefix(text: str) -> bool:
    compact = _base.compact_text(text)
    if not compact:
        return False
    compact = compact.replace("附注", "").replace("注释", "").replace("注", "")
    compact = re.sub(r"[（()）\[\]【】一二三四五六七八九十百A-Za-z.:：、]", "", compact)
    return compact == ""


def _match_is_parenthesized(window: str, match: re.Match[str]) -> bool:
    raw_value = _base._raw_amount(match)
    if raw_value.startswith("(") and raw_value.endswith(")"):
        return True
    before = window[match.start() - 1 : match.start()]
    after = window[match.end() : match.end() + 1]
    if raw_value.endswith((")", "）")) and before in {"(", "（"}:
        return True
    return before in {"(", "（"} and after in {")",
        "）",
    }


def _strip_statement_note_matches(
    window: str,
    label_end: int,
    matches: Sequence[re.Match[str]],
) -> list[re.Match[str]]:
    """从财务报表金额列前移除可证明的附注编号，不猜测小额金额。"""

    usable = list(matches)
    if len(usable) < 2:
        return usable

    first_raw = _base._raw_amount(usable[0])
    prefix = window[label_end : usable[0].start()]

    # 例如“经营活动产生的现金流量净额（六）51(1) 118,554 99,618”。
    if len(usable) >= 3:
        second_raw = _base._raw_amount(usable[1])
        between = _base.compact_text(window[usable[0].end() : usable[1].start()])
        if (
            _small_integer(first_raw)
            and not first_raw.startswith("(")
            and _match_is_parenthesized(window, usable[1])
            and _small_integer(second_raw)
            and between in {"", "(", "（"}
        ):
            return usable[2:]

    # 例如“营业收入四(42) 258,409,403 267,490,414”。括号负数若没有
    # 附注前缀且只有本期/上期两个金额，不会被该规则删除。
    if first_raw.startswith("(") and first_raw.endswith(")") and _small_integer(first_raw):
        second_raw = _base._raw_amount(usable[1])
        try:
            second_value = abs(_base.parse_decimal(second_raw))
        except ValueError:
            second_value = Decimal("0")
        if (
            (len(usable) >= 3 or _note_prefix(prefix))
            and (second_raw in {"-", "－", "—"} or second_value >= 1000)
        ):
            return usable[1:]

    # 无括号的附注号必须同时具有显式附注前缀，避免把真实的小额金额删掉。
    if _small_integer(first_raw) and _note_prefix(prefix):
        second_raw = _base._raw_amount(usable[1])
        try:
            second_value = abs(_base.parse_decimal(second_raw))
        except ValueError:
            second_value = Decimal("0")
        if second_raw in {"-", "－", "—"} or second_value >= 1000:
            return usable[1:]

    # 兼容没有“附注”字样的传统报表列：裸整数小于等于999，后一列为
    # 千位以上金额或占位符。该规则与冻结 V1 行为一致。
    first_is_plain_integer = not any(char in first_raw for char in ",.()")
    if first_is_plain_integer and _small_integer(first_raw):
        second_raw = _base._raw_amount(usable[1])
        try:
            second_value = abs(_base.parse_decimal(second_raw))
        except ValueError:
            second_value = Decimal("0")
        if second_raw in {"-", "－", "—"} or second_value >= 1000:
            return usable[1:]

    return usable


def _nearest_explicit_unit(
    window: str,
    label_match: re.Match[str],
    amount_match: re.Match[str],
) -> str | None:
    start = max(0, label_match.start() - 240)
    end = min(len(window), amount_match.start() + 40)
    region = window[start:end]
    mentions: list[tuple[int, int, str]] = []
    for pattern in (_base.GLOBAL_UNIT_RE, CURRENCY_UNIT_RE, _base.ROW_UNIT_RE):
        for match in pattern.finditer(region):
            mentions.append((match.start(), match.end(), match.group("unit")))
    if not mentions:
        return None
    # 同一短窗口内离金额最近的明确单位优先，避免页面上其他统计表的单位串入。
    return max(mentions, key=lambda item: (item[1], item[0]))[2]


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
            _base.compact_text(marker) in _base.compact_text(page_text)
            for marker in required_page_marker_any
        ):
            continue
        page_units = _unit_candidates(page_text)
        page_unit = page_units[0] if len(set(page_units)) == 1 else None
        for window in _base.logical_line_windows(page_text):
            for pattern in compiled:
                label_match = pattern.search(window)
                if label_match is None:
                    continue
                tail_limit = min(len(window), label_match.end() + 260)
                amount_matches = _base._number_matches_after(
                    window[:tail_limit], label_match.end()
                )
                if amount_matches and _base._contains_unrelated_label_before_first_amount(
                    window[label_match.end() : amount_matches[0].start()]
                ):
                    continue
                usable = (
                    _strip_statement_note_matches(window, label_match.end(), amount_matches)
                    if skip_statement_note_number
                    else list(amount_matches)
                )
                if len(usable) <= amount_index:
                    continue
                amount_match = usable[amount_index]
                unit = (
                    _nearest_explicit_unit(window, label_match, amount_match)
                    or fallback_unit
                    or page_unit
                )
                if unit not in _base.UNIT_MULTIPLIERS:
                    continue
                raw_value = _base._raw_amount(amount_match)
                return ParsedCandidate(
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
    return None


def _section_candidates(
    page_texts: Sequence[str],
    *,
    name: str,
    headings: Sequence[str],
    required_marker_groups: Sequence[Sequence[str]],
    terminators: Sequence[str],
) -> list[SectionRange]:
    compact_headings = [_base.compact_text(item) for item in headings]
    starts: list[int] = []
    for page_index, page_text in enumerate(page_texts):
        page_compact = _base.compact_text(page_text)
        if not any(item in page_compact for item in compact_headings):
            continue
        heading_ends = [
            page_compact.rfind(item) + len(item)
            for item in compact_headings
            if page_compact.rfind(item) >= 0
        ]
        heading_near_page_end = bool(
            heading_ends and len(page_compact) - max(heading_ends) <= 80
        )
        next_page_compact = (
            _base.compact_text(page_texts[page_index + 1])
            if page_index + 1 < len(page_texts)
            else ""
        )
        next_page_has_statement_header = bool(
            "项目" in next_page_compact
            and (
                "单位" in page_compact
                or "人民币" in page_compact
                or "单位" in next_page_compact
                or "人民币" in next_page_compact
            )
        )
        if any(
            _base._heading_precedes_markers(
                page_texts, page_index, headings, marker_group
            )
            and (
                _same_page_statement_table(
                    page_compact,
                    compact_headings,
                    marker_group,
                )
                or (
                    heading_near_page_end
                    and page_index + 1 < len(page_texts)
                    and next_page_has_statement_header
                    and all(
                        _base.compact_text(marker)
                        in next_page_compact
                        for marker in marker_group
                    )
                )
            )
            for marker_group in required_marker_groups
        ):
            starts.append(page_index)

    clustered_starts: list[int] = []
    for start in starts:
        if clustered_starts and start - clustered_starts[-1] <= 3:
            continue
        clustered_starts.append(start)

    compact_terminators = [_base.compact_text(item) for item in terminators]
    candidates: list[SectionRange] = []
    for start in clustered_starts:
        end = len(page_texts)
        for page_index in range(start + 1, len(page_texts)):
            page_compact = _base.compact_text(page_texts[page_index])
            if any(marker in page_compact for marker in compact_terminators):
                end = page_index
                break
        page_indices = tuple(range(start, min(len(page_texts), end + 1)))
        candidates.append(
            SectionRange(
                name=name,
                page_indices=page_indices,
                unit=detect_unique_unit(page_texts, page_indices[:3]),
            )
        )
    return candidates


def _empty_section(name: str) -> SectionRange:
    return SectionRange(name=name, page_indices=(), unit=None)


def locate_statement_sections(page_texts: Sequence[str]) -> dict[str, SectionRange]:
    balance = _section_candidates(
        page_texts,
        name="CONSOLIDATED_BALANCE_SHEET",
        headings=("合并资产负债表", "合并及公司资产负债表"),
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
    income = _section_candidates(
        page_texts,
        name="CONSOLIDATED_INCOME_STATEMENT",
        headings=(
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
    cash = _section_candidates(
        page_texts,
        name="CONSOLIDATED_CASH_FLOW_STATEMENT",
        headings=(
            "合并现金流量表",
            "合并及公司现金流量表",
            "合并年初到报告期末现金流量表",
            "合并本报告期现金流量表",
        ),
        required_marker_groups=(
            ("经营活动产生的现金流量", "销售商品", "经营活动产生的现金流量净额"),
            ("经营活动产生的现金流量", "销售商品", "经营活动产生的现金净额"),
            ("经营活动", "销售商品", "现金流量净额"),
        ),
        terminators=("母公司现金流量表", "合并所有者权益变动表"),
    )

    triples = [
        (balance_item, income_item, cash_item)
        for balance_item in balance
        for income_item in income
        for cash_item in cash
        if balance_item.page_indices[0]
        <= income_item.page_indices[0]
        <= cash_item.page_indices[0]
    ]
    if triples:
        selected = min(
            triples,
            key=lambda items: (
                items[2].page_indices[0] - items[0].page_indices[0],
                sum(item.unit is None for item in items),
                items[0].page_indices[0],
            ),
        )
        return {"balance": selected[0], "income": selected[1], "cash": selected[2]}

    return {
        "balance": balance[0] if balance else _empty_section("CONSOLIDATED_BALANCE_SHEET"),
        "income": income[0] if income else _empty_section("CONSOLIDATED_INCOME_STATEMENT"),
        "cash": cash[0] if cash else _empty_section("CONSOLIDATED_CASH_FLOW_STATEMENT"),
    }


def _statement_unit_consensus(sections: dict[str, SectionRange]) -> str | None:
    units = [section.unit for section in sections.values() if section.unit is not None]
    return units[0] if units and len(set(units)) == 1 else None


def _q3_income_amount_index(
    page_texts: Sequence[str], section: SectionRange, period_type: str | None
) -> int:
    if str(period_type or "").upper() != "Q3":
        return 0
    text = _base.compact_text(
        "\n".join(page_texts[index] for index in section.page_indices)
    )
    current_markers = ("7-9月", "7—9月", "7－9月", "7至9月", "本报告期金额")
    ytd_markers = ("1-9月", "1—9月", "1－9月", "1至9月", "年初至报告期", "年前三季度")
    return 2 if any(item in text for item in current_markers) and any(
        item in text for item in ytd_markers
    ) else 0


def _add_evidence(result: dict[str, Any], evidence: MetricEvidence) -> None:
    metrics = {row["metric_id"]: row for row in result.get("metrics") or []}
    metrics[evidence.metric_id] = asdict(evidence)
    result["parser_version"] = PARSER_VERSION
    result["metrics"] = [metrics[item] for item in REQUIRED_METRICS if item in metrics]
    result["missing_metrics"] = [item for item in REQUIRED_METRICS if item not in metrics]
    result["document_complete"] = not result["missing_metrics"]
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if result["document_complete"]
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )


def _text_metric_fallbacks(
    result: dict[str, Any], page_texts: Sequence[str], period_type: str | None
) -> None:
    sections = locate_statement_sections(page_texts)
    consensus = _statement_unit_consensus(sections)
    if "OPERATING_REVENUE_YTD" in result["missing_metrics"]:
        section = sections["income"]
        candidate = _base.find_amount(
            page_texts,
            section.page_indices,
            label_patterns=(r"一[、.]营业总收入", r"营业总收入"),
            fallback_unit=section.unit or consensus,
            skip_statement_note_number=True,
            amount_index=_q3_income_amount_index(page_texts, section, period_type),
        )
        if candidate is not None:
            _add_evidence(
                result,
                _base._metric_evidence(
                    "OPERATING_REVENUE_YTD",
                    candidate,
                    value_period_scope="YEAR_TO_DATE",
                    section_name=section.name,
                ),
            )

    if "OPERATING_CASH_FLOW_YTD" in result["missing_metrics"]:
        section = sections["cash"]
        candidate = _base.find_amount(
            page_texts,
            section.page_indices,
            label_patterns=(
                r"经营活动产生\s*[/／]?\s*(?:[（(]使用[）)])?\s*的现金流量净额",
                r"经营活动产生的现金(?:流量)?净额",
            ),
            fallback_unit=section.unit or consensus,
            skip_statement_note_number=True,
        )
        if candidate is not None:
            _add_evidence(
                result,
                _base._metric_evidence(
                    "OPERATING_CASH_FLOW_YTD",
                    candidate,
                    value_period_scope="YEAR_TO_DATE",
                    section_name=section.name,
                ),
            )

    if (
        "CORE_PARENT_NET_PROFIT_YTD" in result["missing_metrics"]
        and str(period_type or "").upper() != "Q3"
    ):
        summary_pages = _base._summary_page_indices(page_texts, sections)
        summary_unit = detect_unique_unit(page_texts, summary_pages) or consensus
        candidate = _base.find_amount(
            page_texts,
            summary_pages,
            label_patterns=(
                r"归属于(?:上市公司|本公司|公司|母公司)(?:所有者|股东)?的扣除非经常性损益的净利润",
                r"(?:上市公司)?股东的扣除非经常性损益的净利润",
                r"扣除非经常性损益后的净利润",
            ),
            fallback_unit=summary_unit,
            skip_statement_note_number=False,
            amount_index=0,
        )
        if candidate is not None:
            _add_evidence(
                result,
                _base._metric_evidence(
                    "CORE_PARENT_NET_PROFIT_YTD",
                    candidate,
                    value_period_scope="YEAR_TO_DATE",
                    section_name="MAIN_FINANCIAL_HIGHLIGHTS",
                ),
            )


CORE_PROFIT_PATTERNS = (
    r"归属于上市公司股东的扣除非经常性损益的净利润",
    r"归属于上市公司股东的扣除非经常性损益的净利(?:润)?(?:[（(]元[）)])?",
    r"归属于上市公司股东的扣除非经",
    r"归属于母公司(?:所有者|股东)的扣除非经常性损益的净利润",
    r"归属于母公司(?:所有者|股东)的扣除非经常性损益的净利(?:润)?(?:[（(]元[）)])?",
    r"扣除非经常性损益后的净利润",
)


def _remove_metric(result: dict[str, Any], metric_id: str) -> None:
    result["metrics"] = [
        row for row in result.get("metrics") or [] if row["metric_id"] != metric_id
    ]
    present = {row["metric_id"] for row in result["metrics"]}
    result["missing_metrics"] = [item for item in REQUIRED_METRICS if item not in present]
    result["document_complete"] = not result["missing_metrics"]
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if result["document_complete"]
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )


def _refresh_core_profit_text(
    result: dict[str, Any],
    page_texts: Sequence[str],
    period_type: str | None,
) -> None:
    if str(period_type or "").upper() == "Q3":
        return
    sections = locate_statement_sections(page_texts)
    summary_pages = list(_base._summary_page_indices(page_texts, sections))
    quarter_markers = ("第一季度", "第二季度", "第三季度", "第四季度")
    preferred_pages = [
        page
        for page in summary_pages
        if not all(marker in _base.compact_text(page_texts[page]) for marker in quarter_markers)
    ]
    summary_unit = detect_unique_unit(page_texts, preferred_pages)
    candidate = find_amount(
        page_texts,
        preferred_pages,
        label_patterns=CORE_PROFIT_PATTERNS,
        fallback_unit=summary_unit or _statement_unit_consensus(sections),
        skip_statement_note_number=False,
        amount_index=0,
    )
    if candidate is not None:
        _add_evidence(
            result,
            _base._metric_evidence(
                "CORE_PARENT_NET_PROFIT_YTD",
                candidate,
                value_period_scope="YEAR_TO_DATE",
                section_name="MAIN_FINANCIAL_HIGHLIGHTS",
            ),
        )
        return

    existing = next(
        (
            row
            for row in result.get("metrics") or []
            if row["metric_id"] == "CORE_PARENT_NET_PROFIT_YTD"
        ),
        None,
    )
    if existing is not None:
        page_index = int(existing["source_page"]) - 1
        if page_index in summary_pages and all(
            marker in _base.compact_text(page_texts[page_index]) for marker in quarter_markers
        ):
            _remove_metric(result, "CORE_PARENT_NET_PROFIT_YTD")


def extract_metrics_from_page_texts(
    page_texts: Sequence[str], *, period_type: str | None = None
) -> dict[str, Any]:
    result = _BASE_EXTRACT_METRICS(page_texts, period_type=period_type)
    result["parser_version"] = PARSER_VERSION
    _text_metric_fallbacks(result, page_texts, period_type)
    _refresh_core_profit_text(result, page_texts, period_type)
    return result


def _numeric_cells_after(cells: Sequence[str], label_index: int) -> list[tuple[str, Decimal]]:
    values: list[tuple[str, Decimal]] = []
    for cell in cells[label_index + 1 :]:
        raw = _base.normalize_text(cell)
        match = _base.NUMBER_RE.fullmatch(raw)
        if match is None:
            continue
        raw_value = str(match.group("number") or match.group("placeholder") or "")
        values.append((raw_value, _base.parse_decimal(raw_value)))
    return values


def find_core_profit_from_summary_tables(
    page_texts: Sequence[str],
    table_rows: Sequence[tuple[int, int, int, Sequence[Any]]],
    *,
    period_type: str | None,
    fallback_unit: str | None,
) -> MetricEvidence | None:
    normalized_rows = [
        (page, table, row, [_base.normalize_text(cell) for cell in cells])
        for page, table, row, cells in table_rows
    ]
    q3_text = _base.compact_text("\n".join(page_texts))
    q3_has_current_and_ytd = bool(
        str(period_type or "").upper() == "Q3"
        and "本报告期" in q3_text.replace("本报告期末", "")
        and any(item in q3_text for item in ("年初至报告期末", "年初至本报告期末"))
    )

    for index, (page, table, row, cells) in enumerate(normalized_rows):
        for label_index, label_cell in enumerate(cells):
            label = _base.compact_text(label_cell)
            joined_label = label
            continuation: dict[str, Any] | None = None
            core_label = "扣除非经常性损益" in label and (
                "净利润" in label or label.endswith("的净")
            )
            if not core_label and label.endswith(("归属于上市公司", "归属于上市公司股")):
                if index + 1 < len(normalized_rows):
                    next_page, next_table, next_row, next_cells = normalized_rows[index + 1]
                    next_label = _base.compact_text(next_cells[label_index]) if label_index < len(next_cells) else ""
                    candidate_label = label + next_label
                    if (
                        next_page in {page, page + 1}
                        and "扣除非经常性损益" in candidate_label
                        and "净利润" in candidate_label
                    ):
                        joined_label = candidate_label
                        core_label = True
                        continuation = {
                            "page": next_page + 1,
                            "table_index": next_table,
                            "row_index": next_row,
                            "label": next_label,
                        }
            if not core_label:
                continue

            numeric = _numeric_cells_after(cells, label_index)
            if not numeric:
                continue
            selected_index = len(numeric) // 2 if q3_has_current_and_ytd else 0
            if selected_index >= len(numeric):
                continue
            raw_value, decimal_value = numeric[selected_index]
            unit = detect_unique_unit(page_texts, [page]) or fallback_unit
            if unit not in _base.UNIT_MULTIPLIERS:
                continue
            locator = json.dumps(
                {
                    "page": page + 1,
                    "section": "MAIN_FINANCIAL_HIGHLIGHTS",
                    "table_index": table,
                    "row_index": row,
                    "label_cell_index": label_index,
                    "row_cells": cells,
                    "continuation": continuation,
                    "q3_current_and_ytd_columns": q3_has_current_and_ytd,
                    "numeric_cell_count": len(numeric),
                    "selected_numeric_cell_index": selected_index,
                    "parser_version": PARSER_VERSION,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return MetricEvidence(
                metric_id="CORE_PARENT_NET_PROFIT_YTD",
                metric_value_cny=float(decimal_value * _base.UNIT_MULTIPLIERS[unit]),
                statement_scope="CONSOLIDATED_ONLY",
                value_period_scope="YEAR_TO_DATE",
                source_page=page + 1,
                source_locator=locator,
                source_label=joined_label,
                source_raw_value=raw_value,
                source_unit=unit,
                source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
                source_method=(
                    "PDFPLUMBER_SUMMARY_TABLE_CURRENT_YTD_MIDPOINT_COLUMN"
                    if q3_has_current_and_ytd
                    else "PDFPLUMBER_SUMMARY_TABLE_FIRST_CURRENT_PERIOD_COLUMN"
                ),
                verification_status=(
                    ADMITTED_VERIFICATION_STATUS
                ),
            )
    return None


def extract_official_pdf_facts(
    content: bytes, *, period_type: str | None = None
) -> dict[str, Any]:
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    result = extract_metrics_from_page_texts(page_texts, period_type=period_type)
    sections = locate_statement_sections(page_texts)
    table_receipts: dict[str, Any] = {}

    summary_pages = _base._summary_page_indices(page_texts, sections)
    rows, errors = _base.extract_pdf_table_rows(content, summary_pages)
    core = find_core_profit_from_summary_tables(
        page_texts,
        rows,
        period_type=period_type,
        fallback_unit=(
            detect_unique_unit(page_texts, summary_pages)
            or _statement_unit_consensus(sections)
        ),
    )
    table_receipts["core_parent_net_profit"] = {
        "status": "PASS" if core is not None else "NO_MATCH",
        "examined_table_row_count": len(rows),
        "page_errors": errors,
    }
    if core is not None:
        _add_evidence(result, core)

    if "INVENTORY_END" in result["missing_metrics"]:
        balance = sections["balance"]
        rows, errors = _base.extract_pdf_table_rows(content, balance.page_indices)
        inventory = _base.find_explicit_blank_current_and_prior_inventory(
            page_texts,
            rows,
            fallback_unit=balance.unit,
        )
        table_receipts["blank_inventory"] = {
            "status": "PASS" if inventory is not None else "NO_MATCH",
            "examined_table_row_count": len(rows),
            "page_errors": errors,
        }
        if inventory is not None:
            _add_evidence(result, inventory)

    result.update(
        {
            "parser_version": PARSER_VERSION,
            "official_pdf_sha256": pdf_sha256,
            "official_pdf_size_bytes": len(content),
            "table_fallback_receipt": table_receipts,
            **text_receipt,
        }
    )
    for metric in result.get("metrics") or []:
        metric["verification_status"] = ADMITTED_VERIFICATION_STATUS
    return result


_base._unit_candidates = _unit_candidates
_base.detect_unique_unit = detect_unique_unit
_base.locate_statement_sections = locate_statement_sections
_base.find_amount = find_amount
