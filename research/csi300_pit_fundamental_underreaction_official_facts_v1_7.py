from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_4 as _v1_4
from research import csi300_pit_fundamental_underreaction_official_facts_v1_6 as _v1_6


PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_"
    "POSITIONAL_BROAD_SUMMARY_CORE_CROSSPAGE_SUPPLEMENT_IDENTITY_"
    "ADJACENT_NOTE_EXTENDED_PROFIT_GENERIC_STATEMENT_SUMMARY_SCOPE_"
    "QUARTERLY_IMAGE_IDENTITY_V1_16_0"
)
REQUIRED_METRICS = _v1_6.REQUIRED_METRICS
ADMITTED_VERIFICATION_STATUS = _v1_6.ADMITTED_VERIFICATION_STATUS
SectionRange = _v1_6.SectionRange

CONSOLIDATED_NOTE_MARKERS = (
    "合并财务报表附注",
    "合并财务报表主要项目附注",
    "合并财务报表主要项目注释",
)
CORE_PARENT_EXTENDED_LABEL_PATTERNS = (
    r"归属于母公司(?:普通股)?股东的扣除非经常性损益的净"
    r"\s*(?:[（(][亏虧][损損][）)]\s*[/／])?\s*利润"
    r"(?:\s*[/／]\s*[（(][亏虧][损損][）)])?",
    r"扣除非经常性损益[后後]归属于母公司(?:普通股)?股东的净利润"
    r"(?:[（(][亏虧][损損][）)])?",
    r"归属于母公司(?:普通股)?股东的扣除非经常性损益的净"
    r"(?=\s*[-(（0-9])",
    r"归属于母公司(?:普通股)?股东的扣除非经常性损益(?:的)?"
    r"(?=\s*[-(（0-9])",
    r"扣除非经常性损益后的归属于上市公司(?:普通股)?股东的净利润",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性损益的净利润",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性损益后的净利润",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性损益的净"
    r"(?=\s*[-(（0-9])",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性损益(?:的)?"
    r"(?=\s*[-(（0-9])",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性损"
    r"(?=\s*[-(（0-9])",
    r"归属于上市公司(?:普通股)?股东的扣除非经常性"
    r"(?=\s*[-(（0-9])",
    r"归属于上市公司(?:普通股)?股东的扣除非经常"
    r"(?=\s*[-(（0-9])",
    r"扣除非經常性損益後歸屬於母公司(?:普通股)?股東的淨利潤"
    r"(?:[（(][亏虧][损損][）)])?",
)
SUMMARY_TRADITIONAL_CURRENCY_UNIT_RE = re.compile(
    r"(?:人民幣)?(?P<unit>百萬元|萬元|億元)"
)
TRADITIONAL_UNIT_MAP = {
    "百萬元": "百万元",
    "萬元": "万元",
    "億元": "亿元",
}


def _broad_summary_page_candidates(page_texts: Sequence[str]) -> tuple[int, ...]:
    selected = set(_v1_6._summary_page_candidates(page_texts))
    upper = min(40, len(page_texts))
    for page_index in range(upper):
        compact = _v1_4.compact_financial_text(page_texts[page_index])
        previous = (
            _v1_4.compact_financial_text(page_texts[page_index - 1])
            if page_index > 0
            else ""
        )
        context = previous + compact
        has_revenue = any(
            marker in context for marker in ("营业收入", "营业总收入")
        )
        has_parent_profit = (
            "归属于上市公司股东的净利润" in context
            or "归属于母公司股东的净利润" in context
        )
        has_summary_peer = any(
            marker in context
            for marker in (
                "经营活动产生的现金流量净额",
                "基本每股收益",
                "主要财务数据",
                "主要会计数据",
                "主要财务指标",
            )
        )
        if has_revenue and has_parent_profit and has_summary_peer:
            selected.add(page_index)
            if page_index + 1 < upper:
                selected.add(page_index + 1)
    return tuple(sorted(selected))


def _has_q3_ytd_context(value: str) -> bool:
    compact = _v1_4.compact_financial_text(value)
    return any(
        marker in compact
        for marker in (
            *_v1_6.Q3_YTD_MARKERS,
            "1~9月",
            "1～9月",
            "年初至本报告期末",
            "本年初至报告期末",
            "前三季度",
            "9个月期间",
            "九个月期间",
        )
    ) or re.search(r"截至\d{4}年9月30日止?9个月", compact) is not None or (
        "1月1日至" in compact and "9月30日" in compact
    )


def _q3_summary_amount_index(
    amount_text: str,
    amount_matches: Sequence[re.Match[str]],
    *,
    context_text: str | None = None,
) -> int | None:
    compact = _v1_4.compact_financial_text(context_text or amount_text)
    has_ytd = _has_q3_ytd_context(compact)
    if not has_ytd:
        return None
    has_modern_ytd_header = any(
        marker in compact
        for marker in (
            "年初至报告期末",
            "年初至本报告期末",
            "本年初至报告期末",
        )
    )
    has_explicit_current_quarter = any(
        marker in compact
        for marker in (
            "7-9月",
            "7—9月",
            "7－9月",
            "7~9月",
            "7～9月",
            "本季度",
        )
    ) or (
        has_modern_ytd_header
        and re.search(r"本报告期(?!末)", compact) is not None
    ) or (
        "7月1日至" in compact and "9月30日" in compact
    )
    if not has_explicit_current_quarter:
        return 0 if amount_matches else None
    if len(amount_matches) < 2:
        return None
    for amount_index in range(len(amount_matches) - 1):
        comparison_region = amount_text[
            amount_matches[amount_index].end() : amount_matches[amount_index + 1].start()
        ]
        if "%" in comparison_region:
            return amount_index + 1
    if len(amount_matches) >= 4 and len(amount_matches) % 2 == 0:
        return len(amount_matches) // 2
    return None


SUMMARY_NEXT_ROW_MARKERS = (
    "经营活动产生的现金流量净额",
    "经营活动使用的现金流量净额",
    "加权平均净资产收益率",
    "基本每股收益",
    "稀释每股收益",
    "归属于母公司股东的净利润率",
    "归属于上市公司股东的净利润率",
)


def _normalize_summary_numeric_layout(value: str) -> str:
    """只合并 PDFIUM 在数字内部产生的换行，不合并正常列间空白。"""

    normalized = value
    for _ in range(4):
        previous = normalized
        normalized = re.sub(r"(?<=\d)\s+,(?=\d)", ",", normalized)
        normalized = re.sub(r"(,\d{1,2})\s+(?=\d)", r"\1", normalized)
        normalized = re.sub(r"\.\s+(?=\d{1,2}(?:\D|$))", ".", normalized)
        normalized = re.sub(r"-\s+(?=\d)", "-", normalized)
        if normalized == previous:
            break
    return normalized


def _well_formed_summary_amount(raw_value: str) -> bool:
    normalized = raw_value.strip().replace("（", "(").replace("）", ")")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    return (
        re.fullmatch(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
            normalized,
        )
        is not None
    )


def _summary_numeric_matches(
    text: str,
    label_end: int,
    *,
    maximum_characters: int,
) -> list[re.Match[str]]:
    region_end = min(len(text), label_end + maximum_characters)
    for marker in SUMMARY_NEXT_ROW_MARKERS:
        marker_position = text.find(marker, label_end)
        if marker_position >= 0:
            region_end = min(region_end, marker_position)
    matches = _base._number_matches_after(text[:region_end], label_end)
    numeric_matches: list[re.Match[str]] = []
    for match in matches:
        raw_value = _raw_amount_with_closing_parenthesis(text, match)
        if re.search(r"\d", raw_value) is None:
            continue
        suffix = text[match.end() : match.end() + 5]
        if re.match(r"^\s*[)）]?\s*%", suffix):
            continue
        if not _well_formed_summary_amount(raw_value):
            return []
        numeric_matches.append(match)
    if len(numeric_matches) >= 2:
        first_raw = _raw_amount_with_closing_parenthesis(text, numeric_matches[0])
        if first_raw in {f"({number})" for number in range(1, 10)}:
            numeric_matches = numeric_matches[1:]
    return numeric_matches


def _summary_context_before_candidate(
    page_texts: Sequence[str],
    page_index: int,
    candidate_start: int,
) -> str:
    current_prefix = page_texts[page_index][:candidate_start]
    if _has_q3_ytd_context(current_prefix):
        return current_prefix
    context_start = max(0, page_index - 1)
    return "\n".join(
        [
            *page_texts[context_start:page_index],
            current_prefix,
        ]
    )


def _normalized_summary_unit(raw_unit: str) -> str:
    return TRADITIONAL_UNIT_MAP.get(raw_unit, raw_unit)


def _summary_unit_near_candidate(
    page_texts: Sequence[str],
    page_index: int,
    candidate_start: int,
) -> str | None:
    page_text = page_texts[page_index]
    region_start = max(0, candidate_start - 1500)
    unit_region = page_text[region_start:candidate_start]
    unit_mentions: list[tuple[int, int, str]] = []
    for unit_pattern in (
        _base.GLOBAL_UNIT_RE,
        _v1_6._v1_3._v1_1.CURRENCY_UNIT_RE,
        _base.ROW_UNIT_RE,
        SUMMARY_TRADITIONAL_CURRENCY_UNIT_RE,
    ):
        for unit_match in unit_pattern.finditer(unit_region):
            unit_mentions.append(
                (
                    unit_match.start(),
                    unit_match.end(),
                    _normalized_summary_unit(unit_match.group("unit")),
                )
            )
    if unit_mentions:
        unit = max(unit_mentions, key=lambda item: (item[1], item[0]))[2]
        return unit if unit in _base.UNIT_MULTIPLIERS else None

    unit = _v1_4._unit_near_page(page_texts, page_index, (page_index,))
    if unit in _base.UNIT_MULTIPLIERS:
        return unit

    adjacent_start = max(0, page_index - 1)
    adjacent_text = "\n".join(page_texts[adjacent_start : page_index + 1])
    traditional_units = {
        _normalized_summary_unit(match.group("unit"))
        for match in SUMMARY_TRADITIONAL_CURRENCY_UNIT_RE.finditer(adjacent_text)
    }
    return (
        next(iter(traditional_units))
        if len(traditional_units) == 1
        and next(iter(traditional_units)) in _base.UNIT_MULTIPLIERS
        else None
    )


def _is_explicit_quarterly_breakdown(page_text: str) -> bool:
    compact = _v1_4.compact_financial_text(page_text)
    quarter_markers = sum(
        marker in compact
        for marker in ("第一季度", "第二季度", "第三季度", "第四季度")
    )
    return "分季度主要财务" in compact or quarter_markers >= 2


def _candidate_follows_explicit_quarterly_breakdown(
    page_text: str,
    candidate_start: int,
) -> bool:
    """只拒绝季度明细标题之后的候选，保留同页更早的年度摘要主表。"""

    return _is_explicit_quarterly_breakdown(page_text[:candidate_start])


def _raw_amount_with_closing_parenthesis(
    text: str,
    amount_match: re.Match[str],
) -> str:
    raw_value = _base._raw_amount(amount_match).replace("（", "(").replace("）", ")")
    suffix = text[amount_match.end() : amount_match.end() + 1]
    if raw_value.startswith("(") and not raw_value.endswith(")") and suffix in {
        ")",
        "）",
    }:
        raw_value += ")"
    return raw_value


def _restamp_result(result: dict[str, Any]) -> None:
    for metric in result.get("metrics") or []:
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {
                "legacy_source_locator": str(metric.get("source_locator") or "")
            }
        previous = str(
            locator.get("parser_version") or result.get("parser_version") or ""
        )
        if previous and previous != PARSER_VERSION:
            locator.setdefault("base_parser_version", previous)
        locator["parser_version"] = PARSER_VERSION
        metric["source_locator"] = json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metric["verification_status"] = ADMITTED_VERIFICATION_STATUS
    _v1_4._refresh_result(result)
    result["parser_version"] = PARSER_VERSION


def _candidate_pair(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
) -> tuple[_base.ParsedCandidate, _base.ParsedCandidate] | None:
    return _v1_6._candidate_pair(
        page_texts,
        pages,
        patterns,
        unit=unit,
        skip_note=False,
    )


def _substantial_amount_pair(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
) -> tuple[_base.ParsedCandidate, _base.ParsedCandidate] | None:
    compiled = [re.compile(pattern) for pattern in patterns]
    for page_index in pages:
        page_text = page_texts[page_index]
        for window in _base.logical_line_windows(page_text, maximum_lines=3):
            for pattern in compiled:
                label_match = pattern.search(window)
                if label_match is None:
                    continue
                tail_limit = min(len(window), label_match.end() + 320)
                candidates: list[tuple[re.Match[str], str]] = []
                for amount_match in _base._number_matches_after(
                    window[:tail_limit],
                    label_match.end(),
                ):
                    raw_value = _raw_amount_with_closing_parenthesis(
                        window,
                        amount_match,
                    )
                    value = _base.parse_decimal(raw_value)
                    if value == value.to_integral_value() and abs(value) <= 999:
                        continue
                    candidates.append((amount_match, raw_value))
                if len(candidates) < 2:
                    continue
                leading_values = [
                    _base.parse_decimal(raw_value)
                    for _, raw_value in candidates[:2]
                ]
                if any(
                    value == value.to_integral_value()
                    and 1900 <= value <= 2100
                    for value in leading_values
                ):
                    continue
                parsed: list[_base.ParsedCandidate] = []
                for amount_index, (amount_match, raw_value) in enumerate(candidates[:2]):
                    parsed.append(
                        _base.ParsedCandidate(
                            value_cny=(
                                _base.parse_decimal(raw_value)
                                * _base.UNIT_MULTIPLIERS[unit]
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
                    )
                return parsed[0], parsed[1]
    return None


def _has_consolidated_note_context(
    compact_pages: Sequence[str],
    page_index: int,
) -> bool:
    context_start = max(0, page_index - 1)
    context = "".join(compact_pages[context_start : page_index + 1])
    return any(marker in context for marker in CONSOLIDATED_NOTE_MARKERS)


def _multiline_summary_core_candidate(
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> _base.ParsedCandidate | None:
    normalized_period = str(period_type or "").upper()
    layout_page_texts = tuple(
        _normalize_summary_numeric_layout(page_text) for page_text in page_texts
    )
    compiled = [
        re.compile(pattern)
        for pattern in (
            *CORE_PARENT_EXTENDED_LABEL_PATTERNS,
            *_v1_6.CORE_PARENT_INVERSE_LABEL_PATTERNS,
        )
    ]
    for page_index in _broad_summary_page_candidates(page_texts):
        page_text = layout_page_texts[page_index]
        for pattern in compiled:
            for label_match in pattern.finditer(page_text):
                if _candidate_follows_explicit_quarterly_breakdown(
                    page_text,
                    label_match.start(),
                ):
                    continue
                numeric_matches = _summary_numeric_matches(
                    page_text,
                    label_match.end(),
                    maximum_characters=500,
                )
                if not numeric_matches:
                    continue
                amount_index = 0
                if normalized_period == "Q3":
                    selected_index = _q3_summary_amount_index(
                        page_text,
                        numeric_matches,
                        context_text=_summary_context_before_candidate(
                            layout_page_texts,
                            page_index,
                            label_match.start(),
                        ),
                    )
                    if selected_index is None or selected_index >= len(numeric_matches):
                        continue
                    amount_index = selected_index
                amount_match = numeric_matches[amount_index]
                intervening = page_text[label_match.end() : amount_match.start()]
                if _base._contains_unrelated_label_before_first_amount(intervening):
                    continue
                if any(
                    marker in _v1_4.compact_financial_text(intervening)
                    for marker in ("加权平均净资产收益率", "基本每股收益")
                ):
                    continue
                unit = _summary_unit_near_candidate(
                    layout_page_texts,
                    page_index,
                    label_match.start(),
                )
                if unit not in _base.UNIT_MULTIPLIERS:
                    continue
                raw_value = _raw_amount_with_closing_parenthesis(
                    page_text,
                    amount_match,
                )
                return _base.ParsedCandidate(
                    value_cny=(
                        _base.parse_decimal(raw_value) * _base.UNIT_MULTIPLIERS[unit]
                    ),
                    page_number=page_index + 1,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=page_text[
                        max(0, label_match.start() - 240) : min(
                            len(page_text), amount_match.end() + 240
                        )
                    ],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=amount_index,
                )
    return None


def _cross_page_summary_core_candidate(
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> tuple[_base.ParsedCandidate, int, str] | None:
    summary_pages = set(_broad_summary_page_candidates(page_texts))
    prefix_patterns = (
        (
            re.compile(
                r"归属于上市公司(?:普通股)?股东的扣除非(?=\s*[-(（0-9])"
            ),
            "经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司(?:普通股)?股东的扣(?=\s*[-(（0-9])"),
            "除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司(?:普通股)?股东的(?=\s*[-(（0-9])"),
            "扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司(?:普通股)?股东(?=\s*[-(（0-9])"),
            "的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司(?=\s*[-(（0-9])"),
            "股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司(?=\s*[-(（0-9])"),
            "司股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公(?=\s*[-(（0-9])"),
            "司股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市公司股(?=\s*[-(（0-9])"),
            "东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上市(?=\s*[-(（0-9])"),
            "公司股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
        (
            re.compile(r"归属于上(?=\s*[-(（0-9])"),
            "市公司股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益的净利润",
        ),
    )
    for page_index in sorted(summary_pages):
        if page_index + 1 >= len(page_texts):
            continue
        compact_next = _v1_4.compact_financial_text(page_texts[page_index + 1])
        unit = _v1_4._unit_near_page(page_texts, page_index, (page_index,))
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        normalized_lines = [
            _base.normalize_text(line)
            for line in _base.normalize_text(page_texts[page_index]).splitlines()
        ]
        normalized_lines = [line for line in normalized_lines if line]
        for prefix_pattern, suffix, full_label in prefix_patterns:
            if suffix not in compact_next[:600]:
                continue
            for line_index in range(len(normalized_lines)):
                preceding_text = "\n".join(normalized_lines[:line_index])
                for width in range(1, 17):
                    end = line_index + width
                    if end > len(normalized_lines):
                        continue
                    window = _normalize_summary_numeric_layout(
                        " ".join(normalized_lines[line_index:end])
                    )
                    prefix_match = prefix_pattern.search(window)
                    if prefix_match is None:
                        continue
                    if _is_explicit_quarterly_breakdown(preceding_text):
                        continue
                    numeric_matches = _summary_numeric_matches(
                        window,
                        prefix_match.end(),
                        maximum_characters=260,
                    )
                    if not numeric_matches:
                        continue
                    amount_index = 0
                    if str(period_type or "").upper() == "Q3":
                        selected_index = _q3_summary_amount_index(
                            window,
                            numeric_matches,
                            context_text="\n".join(
                                [
                                    *page_texts[max(0, page_index - 1) : page_index],
                                    preceding_text,
                                ]
                            ),
                        )
                        if (
                            selected_index is None
                            or selected_index >= len(numeric_matches)
                        ):
                            continue
                        amount_index = selected_index
                        if (
                            amount_index >= len(numeric_matches) - 1
                            and "%"
                            not in window[numeric_matches[amount_index].end() :]
                        ):
                            continue
                    amount_match = numeric_matches[amount_index]
                    raw_value = _raw_amount_with_closing_parenthesis(
                        window,
                        amount_match,
                    )
                    candidate = _base.ParsedCandidate(
                        value_cny=(
                            _base.parse_decimal(raw_value)
                            * _base.UNIT_MULTIPLIERS[unit]
                        ),
                        page_number=page_index + 1,
                        label=full_label,
                        raw_value=raw_value,
                        unit=unit,
                        line_window=window[:700],
                        match_span=(prefix_match.start(), prefix_match.end()),
                        value_span=(amount_match.start(), amount_match.end()),
                        selected_amount_index=amount_index,
                    )
                    return candidate, page_index + 2, suffix
    return None


def _supplemental_core_parent_profit_candidate(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> tuple[_base.ParsedCandidate, dict[str, Any]] | None:
    if str(period_type or "").upper() != "FY":
        return None
    existing_parent = _v1_6._metric_cny(result, "PARENT_NET_PROFIT_YTD")
    if existing_parent is None:
        return None

    parent_pattern = re.compile(
        r"(?<!扣除非经常性损益后)归属于母公司普通股股东的净利润"
    )
    core_pattern = re.compile(
        r"扣除非经常性损益后归属于母公司普通股股东的净利润"
    )
    for page_index in range(max(0, len(page_texts) - 30), len(page_texts)):
        page_text = _normalize_summary_numeric_layout(page_texts[page_index])
        compact = _v1_4.compact_financial_text(page_text)
        if (
            "补充资料" not in compact
            or "净资产收益率和每股收益" not in compact
            or re.search(r"\d{4}年度人民币元", compact[:160]) is None
        ):
            continue
        parent_match = parent_pattern.search(page_text)
        core_match = core_pattern.search(page_text)
        if parent_match is None or core_match is None:
            continue
        parent_amounts = _summary_numeric_matches(
            page_text,
            parent_match.end(),
            maximum_characters=40,
        )
        core_amounts = _summary_numeric_matches(
            page_text,
            core_match.end(),
            maximum_characters=40,
        )
        if not parent_amounts or not core_amounts:
            continue
        parent_amount = parent_amounts[0]
        core_amount = core_amounts[0]
        parent_raw = _raw_amount_with_closing_parenthesis(page_text, parent_amount)
        core_raw = _raw_amount_with_closing_parenthesis(page_text, core_amount)
        parent_cny = _base.parse_decimal(parent_raw)
        core_cny = _base.parse_decimal(core_raw)
        existing_parent_decimal = Decimal(str(existing_parent))
        if parent_cny != existing_parent_decimal:
            continue
        nonrecurring_cny = parent_cny - core_cny
        if nonrecurring_cny != nonrecurring_cny.to_integral_value():
            continue
        if page_index == 0:
            continue
        previous_compact = _v1_4.compact_financial_text(page_texts[page_index - 1])
        nonrecurring_raw = f"{abs(int(nonrecurring_cny)):,}"
        if (
            "非经常性损益明细表" not in previous_compact
            or nonrecurring_raw not in previous_compact
        ):
            continue
        candidate = _base.ParsedCandidate(
            value_cny=core_cny,
            page_number=page_index + 1,
            label=core_match.group(0),
            raw_value=core_raw,
            unit="元",
            line_window=page_text[
                max(0, parent_match.start() - 180) : min(
                    len(page_text), core_amount.end() + 260
                )
            ],
            match_span=(core_match.start(), core_match.end()),
            value_span=(core_amount.start(), core_amount.end()),
            selected_amount_index=0,
        )
        validation = {
            "supplemental_parent_profit_cny": float(parent_cny),
            "existing_statement_parent_profit_cny": float(existing_parent_decimal),
            "nonrecurring_profit_cny": float(nonrecurring_cny),
            "identity": "PARENT_PROFIT_MINUS_NONRECURRING_EQUALS_CORE_PROFIT",
            "identity_residual_cny": float(
                parent_cny - nonrecurring_cny - core_cny
            ),
            "unit_anchor": "SUPPLEMENT_PAGE_HEADER_RENMINBI_YUAN",
            "nonrecurring_source_page": page_index,
        }
        return candidate, validation
    return None


def _add_extended_core_parent_profit(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> dict[str, Any]:
    if "CORE_PARENT_NET_PROFIT_YTD" not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_METRIC_ALREADY_PRESENT",
            "admitted_metric_ids": [],
        }
    candidate = _multiline_summary_core_candidate(
        page_texts,
        period_type=period_type,
    )
    next_page: int | None = None
    next_page_suffix: str | None = None
    section = "MAIN_FINANCIAL_HIGHLIGHTS"
    supplemental_validation: dict[str, Any] = {}
    method = "EXACT_EXTENDED_MULTILINE_LABEL_IN_MAIN_FINANCIAL_HIGHLIGHTS"
    if candidate is None:
        cross_page = _cross_page_summary_core_candidate(
            page_texts,
            period_type=period_type,
        )
        if cross_page is not None:
            candidate, next_page, next_page_suffix = cross_page
            method = "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"
    if candidate is None:
        supplemental = _supplemental_core_parent_profit_candidate(
            result,
            page_texts,
            period_type=period_type,
        )
        if supplemental is not None:
            candidate, supplemental_validation = supplemental
            method = "EXACT_SUPPLEMENTAL_CORE_PARENT_PROFIT_TRIPLE_IDENTITY"
            section = "SUPPLEMENTAL_NET_ASSET_RETURN_AND_EPS"
    if candidate is None:
        return {
            "status": "NO_MATCH_EXTENDED_OR_CROSS_PAGE_CORE_LABEL_NOT_FOUND",
            "examined_page_numbers": [
                page + 1 for page in _v1_6._summary_page_candidates(page_texts)
            ],
            "admitted_metric_ids": [],
        }

    locator = {
        "page": candidate.page_number,
        "section": section,
        "parser_version": PARSER_VERSION,
        "line_window": candidate.line_window[:700],
        "label_span": list(candidate.match_span),
        "value_span": list(candidate.value_span),
        "next_page": next_page,
        "next_page_label_suffix": next_page_suffix,
        "validation": {
            "rule": method,
            "period_type": str(period_type or "").upper() or None,
            "selected_amount_index": candidate.selected_amount_index,
            "summary_page_limit": 40,
            **supplemental_validation,
        },
    }
    evidence = _base.MetricEvidence(
        metric_id="CORE_PARENT_NET_PROFIT_YTD",
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope="YEAR_TO_DATE",
        source_page=candidate.page_number,
        source_locator=json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method=f"PDFIUM_{method}",
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )
    _v1_4._put_metric(result, evidence)
    return {
        "status": "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED",
        "source_page": candidate.page_number,
        "next_page": next_page,
        "source_label": candidate.label,
        "source_raw_value": candidate.raw_value,
        "source_unit": candidate.unit,
        "method": method,
        "admitted_metric_ids": ["CORE_PARENT_NET_PROFIT_YTD"],
    }


def _add_adjacent_note_net_receivable_reconciliation(
    result: dict[str, Any],
    page_texts: Sequence[str],
) -> dict[str, Any]:
    if "ACCOUNTS_RECEIVABLE_END" not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_METRIC_ALREADY_PRESENT",
            "admitted_metric_ids": [],
        }

    sections = _v1_4.locate_statement_sections(page_texts)
    balance = sections["balance"]
    unit = balance.unit
    if not balance.page_indices or unit not in _base.UNIT_MULTIPLIERS:
        return {
            "status": "NO_MATCH_NO_VALIDATED_CONSOLIDATED_BALANCE_SECTION",
            "admitted_metric_ids": [],
        }

    combined = _substantial_amount_pair(
        page_texts,
        balance.page_indices,
        (r"应收票据(?:及|和)应收账款",),
        unit=unit,
    )
    if combined is None:
        return {
            "status": "NO_MATCH_COMBINED_BALANCE_ROW_NOT_FOUND",
            "admitted_metric_ids": [],
        }

    compact_pages = [_v1_4.compact_financial_text(text) for text in page_texts]
    examined_blocks: list[dict[str, Any]] = []
    for summary_page, compact in enumerate(compact_pages):
        if not any(
            marker in compact
            for marker in ("应收票据及应收账款", "应收票据和应收账款")
        ):
            continue
        if not _has_consolidated_note_context(compact_pages, summary_page):
            continue

        bills = _substantial_amount_pair(
            page_texts,
            (summary_page,),
            (r"(?<!及)(?<!和)应收票据(?!及应收账款)(?!和应收账款)",),
            unit=unit,
        )
        if bills is None:
            continue

        detail_end = min(len(page_texts), summary_page + 3)
        for detail_page in range(summary_page, detail_end):
            detail_compact = compact_pages[detail_page]
            if "应收账款" not in detail_compact or "坏账准备" not in detail_compact:
                continue
            gross = _substantial_amount_pair(
                page_texts,
                (detail_page,),
                (
                    r"(?<!票据及)(?<!票据和)应收账款"
                    r"(?!总体)(?!按)(?!主要)(?!的)",
                ),
                unit=unit,
            )
            allowance = _substantial_amount_pair(
                page_texts,
                (detail_page,),
                (r"减[:：]?坏账准备",),
                unit=unit,
            )
            if gross is None or allowance is None:
                continue

            net_current = gross[0].value_cny - allowance[0].value_cny
            net_prior = gross[1].value_cny - allowance[1].value_cny
            if net_current < 0 or net_prior < 0:
                continue
            current_pass = _v1_4._same_display_identity(
                combined[0].value_cny,
                bills[0].value_cny + net_current,
                unit,
            )
            prior_pass = _v1_4._same_display_identity(
                combined[1].value_cny,
                bills[1].value_cny + net_prior,
                unit,
            )
            examined_blocks.append(
                {
                    "summary_page": summary_page + 1,
                    "detail_page": detail_page + 1,
                    "current_period_identity_passed": current_pass,
                    "prior_period_identity_passed": prior_pass,
                }
            )
            if not current_pass or not prior_pass:
                continue

            multiplier = _base.UNIT_MULTIPLIERS[unit]
            net_display = net_current / multiplier
            locator = {
                "page": detail_page + 1,
                "section": (
                    "CONSOLIDATED_NOTES_ADJACENT_PAGE_COMBINED_RECEIVABLE_BREAKDOWN"
                ),
                "parser_version": PARSER_VERSION,
                "statement_combined_source_page": combined[0].page_number,
                "note_summary_source_page": summary_page + 1,
                "note_detail_source_page": detail_page + 1,
                "validation": {
                    "rule": (
                        "ADJACENT_NOTE_BILLS_PLUS_GROSS_ACCOUNTS_RECEIVABLE_"
                        "MINUS_ALLOWANCE_EQUALS_COMBINED_TOTAL_CURRENT_AND_PRIOR"
                    ),
                    "current_period_identity_passed": True,
                    "prior_period_identity_passed": True,
                    "bills_current_cny": str(bills[0].value_cny),
                    "gross_accounts_receivable_current_cny": str(
                        gross[0].value_cny
                    ),
                    "allowance_current_cny": str(allowance[0].value_cny),
                    "net_accounts_receivable_current_cny": str(net_current),
                    "combined_current_cny": str(combined[0].value_cny),
                    "bills_prior_cny": str(bills[1].value_cny),
                    "gross_accounts_receivable_prior_cny": str(
                        gross[1].value_cny
                    ),
                    "allowance_prior_cny": str(allowance[1].value_cny),
                    "net_accounts_receivable_prior_cny": str(net_prior),
                    "combined_prior_cny": str(combined[1].value_cny),
                },
            }
            evidence = _base.MetricEvidence(
                metric_id="ACCOUNTS_RECEIVABLE_END",
                metric_value_cny=float(net_current),
                statement_scope="CONSOLIDATED_ONLY",
                value_period_scope="PERIOD_END",
                source_page=detail_page + 1,
                source_locator=json.dumps(
                    locator,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                source_label="应收账款减：坏账准备",
                source_raw_value=(
                    f"{gross[0].raw_value}-{allowance[0].raw_value}={net_display}"
                ),
                source_unit=unit,
                source_unit_multiplier=float(multiplier),
                source_method=(
                    "PDFIUM_ADJACENT_NOTE_GROSS_RECEIVABLE_MINUS_ALLOWANCE_"
                    "RECONCILED_TO_COMBINED_BALANCE_BOTH_PERIODS"
                ),
                verification_status=ADMITTED_VERIFICATION_STATUS,
            )
            _v1_4._put_metric(result, evidence)
            return {
                "status": (
                    "PASS_ADJACENT_NOTE_NET_ACCOUNTS_RECEIVABLE_"
                    "RECONCILED_AND_ADMITTED"
                ),
                "balance_source_page": combined[0].page_number,
                "note_summary_source_page": summary_page + 1,
                "note_detail_source_page": detail_page + 1,
                "current_period_identity_passed": True,
                "prior_period_identity_passed": True,
                "admitted_metric_ids": ["ACCOUNTS_RECEIVABLE_END"],
            }

    return {
        "status": "NO_MATCH_ADJACENT_NOTE_DUAL_PERIOD_IDENTITY_NOT_PROVEN",
        "examined_blocks": examined_blocks,
        "admitted_metric_ids": [],
    }


def _add_dual_period_text_liability(
    result: dict[str, Any],
    page_texts: Sequence[str],
) -> dict[str, Any]:
    if "TOTAL_LIABILITIES_END" not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_METRIC_ALREADY_PRESENT",
            "admitted_metric_ids": [],
        }
    sections = _v1_4.locate_statement_sections(page_texts)
    balance = sections["balance"]
    unit = balance.unit
    if not balance.page_indices or unit not in _base.UNIT_MULTIPLIERS:
        return {
            "status": "NO_MATCH_NO_VALIDATED_CONSOLIDATED_BALANCE_SECTION",
            "admitted_metric_ids": [],
        }
    definitions = {
        "assets": (r"(?<!流动)(?<!非流动)资产总计",),
        "liabilities": (r"(?<!流动)(?<!非流动)负债(?:合计|总计)",),
        "equity": (
            r"(?<!公司)(?<!母公司)(?<!少数)"
            r"(?:股东权益|所有者权益)(?:（或股东权益）)?合计",
        ),
        "total": (
            r"负债(?:及|和)(?:股东权益|所有者权益)"
            r"(?:（或股东权益）)?总计",
        ),
    }
    selected = {
        key: _v1_6._candidate_pair(
            page_texts,
            balance.page_indices,
            patterns,
            unit=unit,
            skip_note=True,
        )
        for key, patterns in definitions.items()
    }
    if any(pair is None for pair in selected.values()):
        return {
            "status": "NO_MATCH_DUAL_PERIOD_BALANCE_IDENTITY_ROWS_INCOMPLETE",
            "present_identity_rows": sorted(
                key for key, pair in selected.items() if pair is not None
            ),
            "admitted_metric_ids": [],
        }
    assets = selected["assets"]
    liabilities = selected["liabilities"]
    equity = selected["equity"]
    total = selected["total"]
    assert assets is not None
    assert liabilities is not None
    assert equity is not None
    assert total is not None
    current_pass = (
        _v1_4._same_display_identity(
            assets[0].value_cny,
            liabilities[0].value_cny + equity[0].value_cny,
            unit,
        )
        and _v1_4._same_display_identity(
            assets[0].value_cny,
            total[0].value_cny,
            unit,
        )
    )
    prior_pass = (
        _v1_4._same_display_identity(
            assets[1].value_cny,
            liabilities[1].value_cny + equity[1].value_cny,
            unit,
        )
        and _v1_4._same_display_identity(
            assets[1].value_cny,
            total[1].value_cny,
            unit,
        )
    )
    existing_assets = _v1_6._metric_cny(result, "TOTAL_ASSETS_END")
    existing_assets_pass = existing_assets is None or _v1_4._same_display_identity(
        assets[0].value_cny,
        existing_assets,
        unit,
    )
    if not current_pass or not prior_pass or not existing_assets_pass:
        return {
            "status": "NO_MATCH_DUAL_PERIOD_BALANCE_IDENTITY_NOT_PROVEN",
            "current_period_identity_passed": current_pass,
            "prior_period_identity_passed": prior_pass,
            "existing_assets_match_passed": existing_assets_pass,
            "admitted_metric_ids": [],
        }
    validation = {
        "rule": (
            "TEXT_TOTAL_ASSETS_EQUALS_LIABILITIES_PLUS_EXACT_TOTAL_EQUITY_"
            "AND_LIABILITIES_EQUITY_TOTAL_CURRENT_AND_PRIOR"
        ),
        "current_period_identity_passed": True,
        "prior_period_identity_passed": True,
        "existing_assets_match_passed": True,
        "total_assets_current_cny": str(assets[0].value_cny),
        "total_liabilities_current_cny": str(liabilities[0].value_cny),
        "total_equity_current_cny": str(equity[0].value_cny),
        "liabilities_equity_total_current_cny": str(total[0].value_cny),
        "total_assets_prior_cny": str(assets[1].value_cny),
        "total_liabilities_prior_cny": str(liabilities[1].value_cny),
        "total_equity_prior_cny": str(equity[1].value_cny),
        "liabilities_equity_total_prior_cny": str(total[1].value_cny),
    }
    _v1_4._put_metric(
        result,
        _v1_4._text_evidence(
            "TOTAL_LIABILITIES_END",
            liabilities[0],
            value_period_scope="PERIOD_END",
            section_name="CONSOLIDATED_BALANCE_SHEET",
            validation=validation,
        ),
    )
    return {
        "status": "PASS_DUAL_PERIOD_TEXT_LIABILITY_IDENTITY_ADMITTED",
        "source_page": liabilities[0].page_number,
        "source_raw_value": liabilities[0].raw_value,
        "source_unit": unit,
        "current_period_identity_passed": True,
        "prior_period_identity_passed": True,
        "admitted_metric_ids": ["TOTAL_LIABILITIES_END"],
    }


def _whole_page_amount_pair(
    page_text: str,
    *,
    page_number: int,
    patterns: Sequence[str],
    unit: str,
    amount_index: int,
) -> tuple[_base.ParsedCandidate, _base.ParsedCandidate] | None:
    for pattern_text in patterns:
        pattern = re.compile(pattern_text)
        label_match = pattern.search(page_text)
        if label_match is None:
            continue
        amount_matches = [
            match
            for match in _base._number_matches_after(
                page_text[: min(len(page_text), label_match.end() + 700)],
                label_match.end(),
            )
            if re.search(
                r"\d",
                _raw_amount_with_closing_parenthesis(page_text, match),
            )
            is not None
        ]
        if amount_index + 1 >= len(amount_matches):
            continue
        selected: list[_base.ParsedCandidate] = []
        for selected_index in (amount_index, amount_index + 1):
            amount_match = amount_matches[selected_index]
            raw_value = _raw_amount_with_closing_parenthesis(
                page_text,
                amount_match,
            )
            selected.append(
                _base.ParsedCandidate(
                    value_cny=(
                        _base.parse_decimal(raw_value)
                        * _base.UNIT_MULTIPLIERS[unit]
                    ),
                    page_number=page_number,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=page_text[
                        max(0, label_match.start() - 240) : min(
                            len(page_text),
                            amount_match.end() + 240,
                        )
                    ],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=selected_index,
                )
            )
        return selected[0], selected[1]
    return None


def _q3_income_amount_index_from_context(
    page_texts: Sequence[str],
    page_index: int,
    *,
    period_type: str | None,
) -> int:
    if str(period_type or "").upper() != "Q3":
        return 0
    context = "".join(
        _v1_4.compact_financial_text(page_texts[index])
        for index in range(max(0, page_index - 4), page_index + 1)
    )
    has_current_quarter = any(
        marker in context for marker in _v1_6._v1_3.CURRENT_QUARTER_MARKERS
    )
    has_ytd = any(marker in context for marker in _v1_6._v1_3.YTD_MARKERS)
    return 2 if has_current_quarter and has_ytd else 0


def _income_statement_unit(
    page_texts: Sequence[str],
    page_index: int,
) -> str | None:
    unit = _v1_4._unit_near_page(page_texts, page_index, (page_index,))
    if unit in _base.UNIT_MULTIPLIERS:
        return unit
    for header_index in range(page_index, max(-1, page_index - 5), -1):
        header = _v1_4.compact_financial_text(page_texts[header_index])
        if "合并利润表" not in header:
            continue
        unit = _base.detect_unique_unit(page_texts, (header_index,))
        if unit in _base.UNIT_MULTIPLIERS:
            return unit
        return None
    return None


def _add_extended_parent_net_profit(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> dict[str, Any]:
    if "PARENT_NET_PROFIT_YTD" not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_METRIC_ALREADY_PRESENT",
            "admitted_metric_ids": [],
        }
    parent_patterns = (
        r"(?:\d+[.、]?\s*)?归属于母公司(?:普通股)?(?:所有者|股东)的净利润",
        r"归属于本公司股东的净利润",
    )
    net_patterns = (
        r"(?:[一二三四五六七八九十][、.]?)净利润",
        r"(?<!持续经营)(?<!终止经营)(?<!母公司)(?<!的)净利润"
        r"(?:\s*[（(]净亏损[^）)]{0,40}[）)])?",
    )
    minority_patterns = (
        r"少数股东损益",
        r"少数股东损失",
    )
    for page_index, page_text in enumerate(page_texts):
        context = "".join(
            _v1_4.compact_financial_text(page_texts[index])
            for index in range(max(0, page_index - 4), page_index + 1)
        )
        if "合并利润表" not in context:
            continue
        unit = _income_statement_unit(page_texts, page_index)
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        amount_index = _q3_income_amount_index_from_context(
            page_texts,
            page_index,
            period_type=period_type,
        )
        parent = _whole_page_amount_pair(
            page_text,
            page_number=page_index + 1,
            patterns=parent_patterns,
            unit=unit,
            amount_index=amount_index,
        )
        net = _whole_page_amount_pair(
            page_text,
            page_number=page_index + 1,
            patterns=net_patterns,
            unit=unit,
            amount_index=amount_index,
        )
        minority = _whole_page_amount_pair(
            page_text,
            page_number=page_index + 1,
            patterns=minority_patterns,
            unit=unit,
            amount_index=amount_index,
        )
        cross_page_suffix: str | None = None
        if parent is None and page_index + 1 < len(page_texts):
            prefix_patterns = (
                r"(?:\d+[.、]?\s*)?归属于母公司(?:普通股)?"
                r"(?:所有者|股东)的净(?=\s*[-(（0-9])",
            )
            parent = _whole_page_amount_pair(
                page_text,
                page_number=page_index + 1,
                patterns=prefix_patterns,
                unit=unit,
                amount_index=amount_index,
            )
            next_compact = _v1_4.compact_financial_text(page_texts[page_index + 1])
            if parent is not None and "利润" in next_compact[:300]:
                cross_page_suffix = "利润"
            else:
                parent = None
        if minority is None and page_index + 1 < len(page_texts):
            minority = _whole_page_amount_pair(
                page_texts[page_index + 1],
                page_number=page_index + 2,
                patterns=minority_patterns,
                unit=unit,
                amount_index=amount_index,
            )
        if parent is None or net is None or minority is None:
            continue
        current_pass = _v1_4._same_display_identity(
            net[0].value_cny,
            parent[0].value_cny + minority[0].value_cny,
            unit,
        )
        prior_pass = _v1_4._same_display_identity(
            net[1].value_cny,
            parent[1].value_cny + minority[1].value_cny,
            unit,
        )
        if not current_pass or not prior_pass:
            continue
        validation = {
            "rule": (
                "NET_PROFIT_EQUALS_PARENT_PLUS_MINORITY_CURRENT_AND_PRIOR_"
                "WITH_EXTENDED_OR_CROSS_PAGE_PARENT_LABEL"
            ),
            "current_period_identity_passed": True,
            "prior_period_identity_passed": True,
            "q3_selected_amount_index": amount_index,
            "net_profit_current_cny": str(net[0].value_cny),
            "parent_profit_current_cny": str(parent[0].value_cny),
            "minority_profit_current_cny": str(minority[0].value_cny),
            "net_profit_prior_cny": str(net[1].value_cny),
            "parent_profit_prior_cny": str(parent[1].value_cny),
            "minority_profit_prior_cny": str(minority[1].value_cny),
            "cross_page_label_suffix": cross_page_suffix,
            "minority_source_page": minority[0].page_number,
        }
        _v1_4._put_metric(
            result,
            _v1_4._text_evidence(
                "PARENT_NET_PROFIT_YTD",
                parent[0],
                value_period_scope="YEAR_TO_DATE",
                section_name="CONSOLIDATED_INCOME_STATEMENT",
                validation=validation,
            ),
        )
        return {
            "status": "PASS_EXTENDED_PARENT_NET_PROFIT_IDENTITY_ADMITTED",
            "source_page": parent[0].page_number,
            "source_raw_value": parent[0].raw_value,
            "source_unit": unit,
            "q3_selected_amount_index": amount_index,
            "current_period_identity_passed": True,
            "prior_period_identity_passed": True,
            "cross_page_label_suffix": cross_page_suffix,
            "admitted_metric_ids": ["PARENT_NET_PROFIT_YTD"],
        }
    return {
        "status": "NO_MATCH_EXTENDED_PARENT_NET_PROFIT_IDENTITY_NOT_PROVEN",
        "admitted_metric_ids": [],
    }


def _generic_issuer_statement_section(
    page_texts: Sequence[str],
    *,
    statement_kind: str,
) -> SectionRange:
    compact_pages = [
        _v1_4.compact_financial_text(page_text) for page_text in page_texts
    ]
    if statement_kind == "BALANCE":
        name = "SUMMARY_RECONCILED_GENERIC_ISSUER_BALANCE_SHEET"
        heading = "资产负债表"
        forbidden_headings = ("合并资产负债表", "母公司资产负债表")
        marker_groups = (("流动资产", "资产总计"),)
        next_headings = ("利润表", "现金流量表", "母公司资产负债表")
    elif statement_kind == "INCOME":
        name = "SUMMARY_RECONCILED_GENERIC_ISSUER_INCOME_STATEMENT"
        heading = "利润表"
        forbidden_headings = ("合并利润表", "母公司利润表")
        marker_groups = (
            ("营业收入", "营业利润", "净利润"),
            ("营业总收入", "营业利润", "净利润"),
        )
        next_headings = ("现金流量表", "母公司利润表")
    else:
        raise ValueError(f"未知通用报表类型：{statement_kind}")

    for start, compact in enumerate(compact_pages):
        if heading not in compact:
            continue
        if any(forbidden in compact for forbidden in forbidden_headings):
            continue
        if not any(marker in compact for marker in ("财务报表", "编制单位", "审计类型")):
            continue
        end = min(len(page_texts), start + 8)
        for page_index in range(start + 1, end):
            candidate = compact_pages[page_index]
            if any(next_heading in candidate for next_heading in next_headings):
                end = page_index
                break
        pages = tuple(range(start, max(start + 1, end)))
        section_text = "".join(compact_pages[index] for index in pages)
        if not any(
            all(marker in section_text for marker in marker_group)
            for marker_group in marker_groups
        ):
            continue
        if sum(
            len(_base.NUMBER_RE.findall(page_texts[index])) for index in pages
        ) < 12:
            continue
        unit = _v1_4._unit_near_page(page_texts, start, pages)
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        return SectionRange(name=name, page_indices=pages, unit=unit)
    return SectionRange(name=name, page_indices=(), unit=None)


def _summary_reconciled_statement_evidence(
    metric_id: str,
    candidate: _base.ParsedCandidate,
    *,
    value_period_scope: str,
    section_name: str,
    validation: dict[str, Any],
) -> _base.MetricEvidence:
    locator = {
        "page": candidate.page_number,
        "section": section_name,
        "line_window": candidate.line_window[:700],
        "label_span": list(candidate.match_span),
        "value_span": list(candidate.value_span),
        "selected_amount_index": candidate.selected_amount_index,
        "validation": validation,
        "parser_version": PARSER_VERSION,
        "text_engine": "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
    }
    return _base.MetricEvidence(
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
            "PDFIUM_GENERIC_ISSUER_STATEMENT_DUAL_PERIOD_"
            "SUMMARY_SCOPE_RECONCILED"
        ),
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _generic_balance_candidates(
    page_texts: Sequence[str],
    section: SectionRange,
) -> dict[str, tuple[_base.ParsedCandidate, _base.ParsedCandidate] | None]:
    assert section.unit in _base.UNIT_MULTIPLIERS
    definitions = {
        "assets": (r"(?<!流动)(?<!非流动)资产总计",),
        "liabilities": (r"(?<!流动)(?<!非流动)负债(?:合计|总计)",),
        "equity": (
            r"(?<!公司)(?<!母公司)(?<!少数)"
            r"(?:股东权益|所有者权益)(?:（或股东权益）)?合计",
        ),
        "total": (
            r"负债(?:及|和)(?:股东权益|所有者权益)"
            r"(?:（或股东权益）)?总计",
        ),
        "receivable": (
            r"(?<!应收票据及)(?<!应收票据和)(?<!其他)"
            r"应收账款(?!及合同资产)(?!和合同资产)",
        ),
        "inventory": (r"(?<!周转)存货(?!跌价)(?!减值)",),
    }
    return {
        key: _substantial_amount_pair(
            page_texts,
            section.page_indices,
            patterns,
            unit=str(section.unit),
        )
        for key, patterns in definitions.items()
    }


def _add_summary_reconciled_generic_statement_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> dict[str, Any]:
    target_metrics = {
        "OPERATING_PROFIT_YTD",
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_LIABILITIES_END",
    }
    if not target_metrics.intersection(result.get("missing_metrics") or []):
        return {
            "status": "NOT_NEEDED_NO_GENERIC_STATEMENT_TARGET_METRIC_MISSING",
            "admitted_metric_ids": [],
        }

    admitted: list[str] = []
    balance_receipt: dict[str, Any] = {
        "status": "NO_MATCH_GENERIC_ISSUER_BALANCE_SCOPE_NOT_PROVEN"
    }
    balance = _generic_issuer_statement_section(
        page_texts,
        statement_kind="BALANCE",
    )
    if balance.page_indices and balance.unit in _base.UNIT_MULTIPLIERS:
        selected = _generic_balance_candidates(page_texts, balance)
        identity_keys = ("assets", "liabilities", "equity", "total")
        if all(selected[key] is not None for key in identity_keys):
            assets = selected["assets"]
            liabilities = selected["liabilities"]
            equity = selected["equity"]
            total = selected["total"]
            assert assets is not None
            assert liabilities is not None
            assert equity is not None
            assert total is not None
            unit = str(balance.unit)
            current_identity_passed = (
                _v1_4._same_display_identity(
                    assets[0].value_cny,
                    liabilities[0].value_cny + equity[0].value_cny,
                    unit,
                )
                and _v1_4._same_display_identity(
                    assets[0].value_cny,
                    total[0].value_cny,
                    unit,
                )
            )
            prior_identity_passed = (
                _v1_4._same_display_identity(
                    assets[1].value_cny,
                    liabilities[1].value_cny + equity[1].value_cny,
                    unit,
                )
                and _v1_4._same_display_identity(
                    assets[1].value_cny,
                    total[1].value_cny,
                    unit,
                )
            )
            summary_assets = _v1_6._metric_cny(result, "TOTAL_ASSETS_END")
            summary_assets_match_passed = (
                summary_assets is not None
                and _v1_4._same_display_identity(
                    assets[0].value_cny,
                    summary_assets,
                    unit,
                )
            )
            if (
                current_identity_passed
                and prior_identity_passed
                and summary_assets_match_passed
            ):
                validation = {
                    "rule": (
                        "GENERIC_ISSUER_BALANCE_TITLE_PLUS_SUMMARY_TOTAL_ASSETS_"
                        "ANCHOR_PLUS_DUAL_PERIOD_ASSETS_LIABILITIES_EQUITY_IDENTITY"
                    ),
                    "generic_title_is_not_explicit_parent_or_consolidated_title": True,
                    "summary_total_assets_match_passed": True,
                    "current_period_identity_passed": True,
                    "prior_period_identity_passed": True,
                    "summary_total_assets_cny": str(summary_assets),
                    "total_assets_current_cny": str(assets[0].value_cny),
                    "total_liabilities_current_cny": str(
                        liabilities[0].value_cny
                    ),
                    "total_equity_current_cny": str(equity[0].value_cny),
                    "liabilities_equity_total_current_cny": str(
                        total[0].value_cny
                    ),
                    "total_assets_prior_cny": str(assets[1].value_cny),
                    "total_liabilities_prior_cny": str(
                        liabilities[1].value_cny
                    ),
                    "total_equity_prior_cny": str(equity[1].value_cny),
                    "liabilities_equity_total_prior_cny": str(
                        total[1].value_cny
                    ),
                }
                detail_definitions = {
                    "ACCOUNTS_RECEIVABLE_END": selected["receivable"],
                    "INVENTORY_END": selected["inventory"],
                    "TOTAL_LIABILITIES_END": liabilities,
                }
                for metric_id, pair in detail_definitions.items():
                    if metric_id not in result.get("missing_metrics", []) or pair is None:
                        continue
                    _v1_4._put_metric(
                        result,
                        _summary_reconciled_statement_evidence(
                            metric_id,
                            pair[0],
                            value_period_scope="PERIOD_END",
                            section_name=balance.name,
                            validation=validation,
                        ),
                    )
                    admitted.append(metric_id)
                balance_receipt = {
                    "status": "PASS_GENERIC_ISSUER_BALANCE_SCOPE_RECONCILED",
                    "source_pages": [page + 1 for page in balance.page_indices],
                    "source_unit": unit,
                    "current_period_identity_passed": True,
                    "prior_period_identity_passed": True,
                    "summary_total_assets_match_passed": True,
                    "admitted_metric_ids": sorted(
                        metric_id
                        for metric_id in admitted
                        if metric_id.endswith("_END")
                    ),
                }
            else:
                balance_receipt = {
                    "status": "NO_MATCH_GENERIC_ISSUER_BALANCE_IDENTITY_NOT_PROVEN",
                    "current_period_identity_passed": current_identity_passed,
                    "prior_period_identity_passed": prior_identity_passed,
                    "summary_total_assets_match_passed": summary_assets_match_passed,
                }
        else:
            balance_receipt = {
                "status": "NO_MATCH_GENERIC_ISSUER_BALANCE_IDENTITY_ROWS_INCOMPLETE",
                "present_identity_rows": sorted(
                    key for key in identity_keys if selected[key] is not None
                ),
            }

    income_receipt: dict[str, Any] = {
        "status": "NO_MATCH_GENERIC_ISSUER_INCOME_SCOPE_NOT_PROVEN"
    }
    if "OPERATING_PROFIT_YTD" in result.get("missing_metrics", []):
        income = _generic_issuer_statement_section(
            page_texts,
            statement_kind="INCOME",
        )
        if income.page_indices and income.unit in _base.UNIT_MULTIPLIERS:
            unit = str(income.unit)
            normalized_period = str(period_type or "").upper()
            context = "".join(
                _v1_4.compact_financial_text(page_texts[index])
                for index in income.page_indices
            )
            has_q3_ytd_context = _has_q3_ytd_context(context)
            has_q3_current_quarter_context = any(
                marker in context
                for marker in _v1_6._v1_3.CURRENT_QUARTER_MARKERS
            )
            if normalized_period == "Q3" and not has_q3_ytd_context:
                amount_index: int | None = None
            elif (
                normalized_period == "Q3"
                and has_q3_current_quarter_context
                and has_q3_ytd_context
            ):
                amount_index = 2
            else:
                amount_index = 0

            operating_profit = None
            net_profit = None
            parent_profit = None
            minority_profit = None
            if amount_index is not None:
                for page_index in income.page_indices:
                    page_text = page_texts[page_index]
                    if operating_profit is None:
                        operating_profit = _whole_page_amount_pair(
                            page_text,
                            page_number=page_index + 1,
                            patterns=(
                                r"(?:[一二三四五六七八九十][、.]?)?营业利润"
                                r"(?:\s*[（(]亏损[^）)]{0,40}[）)])?",
                            ),
                            unit=unit,
                            amount_index=amount_index,
                        )
                    if net_profit is None:
                        net_profit = _whole_page_amount_pair(
                            page_text,
                            page_number=page_index + 1,
                            patterns=(
                                r"(?:[一二三四五六七八九十][、.]?)净利润",
                                r"(?<!持续经营)(?<!终止经营)(?<!母公司)(?<!的)净利润"
                                r"(?:\s*[（(]净亏损[^）)]{0,40}[）)])?",
                            ),
                            unit=unit,
                            amount_index=amount_index,
                        )
                    if parent_profit is None:
                        parent_profit = _whole_page_amount_pair(
                            page_text,
                            page_number=page_index + 1,
                            patterns=(
                                r"归属于母公司(?:普通股)?(?:所有者|股东)的净利润",
                                r"归属于本公司股东的净利润",
                            ),
                            unit=unit,
                            amount_index=amount_index,
                        )
                    if minority_profit is None:
                        minority_profit = _whole_page_amount_pair(
                            page_text,
                            page_number=page_index + 1,
                            patterns=(r"少数股东(?:损益|损失)",),
                            unit=unit,
                            amount_index=amount_index,
                        )

            summary_parent = _v1_6._metric_cny(result, "PARENT_NET_PROFIT_YTD")
            explicit_parent_label_present = any(
                marker in context
                for marker in (
                    "归属于母公司所有者的净利润",
                    "归属于母公司股东的净利润",
                    "归属于本公司股东的净利润",
                )
            )
            minority_label_present = any(
                marker in context for marker in ("少数股东损益", "少数股东损失")
            )
            parent_identity_current = None
            parent_identity_prior = None
            if (
                net_profit is not None
                and parent_profit is not None
                and minority_profit is not None
            ):
                parent_identity_current = _v1_4._same_display_identity(
                    net_profit[0].value_cny,
                    parent_profit[0].value_cny + minority_profit[0].value_cny,
                    unit,
                )
                parent_identity_prior = _v1_4._same_display_identity(
                    net_profit[1].value_cny,
                    parent_profit[1].value_cny + minority_profit[1].value_cny,
                    unit,
                )

            scope_anchor = None
            scope_anchor_kind = None
            if (
                parent_profit is not None
                and summary_parent is not None
                and _v1_4._same_display_identity(
                    parent_profit[0].value_cny,
                    summary_parent,
                    unit,
                )
                and parent_identity_current is not False
                and parent_identity_prior is not False
            ):
                scope_anchor = parent_profit
                scope_anchor_kind = "EXPLICIT_PARENT_PROFIT_ROW"
            elif (
                net_profit is not None
                and summary_parent is not None
                and not explicit_parent_label_present
                and _v1_4._same_display_identity(
                    net_profit[0].value_cny,
                    summary_parent,
                    unit,
                )
                and (
                    not minority_label_present
                    or (
                        minority_profit is not None
                        and minority_profit[0].value_cny == 0
                        and minority_profit[1].value_cny == 0
                    )
                )
            ):
                scope_anchor = net_profit
                scope_anchor_kind = "NET_PROFIT_EQUALS_PARENT_SUMMARY_NO_NONZERO_MINORITY"

            if operating_profit is not None and scope_anchor is not None:
                validation = {
                    "rule": (
                        "GENERIC_ISSUER_INCOME_TITLE_PLUS_PARENT_PROFIT_SUMMARY_"
                        "ANCHOR_PLUS_EXACT_DUAL_PERIOD_COLUMN_SEMANTICS"
                    ),
                    "generic_title_is_not_explicit_parent_or_consolidated_title": True,
                    "summary_parent_profit_match_passed": True,
                    "summary_parent_profit_cny": str(summary_parent),
                    "scope_anchor_kind": scope_anchor_kind,
                    "scope_anchor_current_cny": str(scope_anchor[0].value_cny),
                    "scope_anchor_prior_cny": str(scope_anchor[1].value_cny),
                    "q3_ytd_context_present": has_q3_ytd_context,
                    "q3_current_quarter_context_present": (
                        has_q3_current_quarter_context
                    ),
                    "selected_amount_index": amount_index,
                    "parent_plus_minority_identity_current": (
                        parent_identity_current
                    ),
                    "parent_plus_minority_identity_prior": parent_identity_prior,
                }
                _v1_4._put_metric(
                    result,
                    _summary_reconciled_statement_evidence(
                        "OPERATING_PROFIT_YTD",
                        operating_profit[0],
                        value_period_scope="YEAR_TO_DATE",
                        section_name=income.name,
                        validation=validation,
                    ),
                )
                admitted.append("OPERATING_PROFIT_YTD")
                income_receipt = {
                    "status": "PASS_GENERIC_ISSUER_INCOME_SCOPE_RECONCILED",
                    "source_pages": [page + 1 for page in income.page_indices],
                    "source_unit": unit,
                    "summary_parent_profit_match_passed": True,
                    "scope_anchor_kind": scope_anchor_kind,
                    "q3_selected_amount_index": amount_index,
                    "admitted_metric_ids": ["OPERATING_PROFIT_YTD"],
                }
            else:
                income_receipt = {
                    "status": "NO_MATCH_GENERIC_ISSUER_INCOME_ANCHOR_NOT_PROVEN",
                    "operating_profit_row_present": operating_profit is not None,
                    "scope_anchor_present": scope_anchor is not None,
                    "q3_ytd_context_present": has_q3_ytd_context,
                    "q3_selected_amount_index": amount_index,
                }

    return {
        "status": (
            "PASS_GENERIC_ISSUER_STATEMENT_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_GENERIC_ISSUER_STATEMENT_SCOPE_NOT_PROVEN"
        ),
        "balance": balance_receipt,
        "income": income_receipt,
        "admitted_metric_ids": sorted(admitted),
        "remaining_missing_metrics": list(result.get("missing_metrics") or []),
    }


def find_quarterly_image_statement_pages(
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> tuple[int, ...]:
    if str(period_type or "").upper() not in {"Q1", "Q3"}:
        return ()
    if len(page_texts) > 80:
        return ()
    normalized = [_v1_4.compact_financial_text(text) for text in page_texts]
    anchor = next(
        (
            index
            for index, compact in enumerate(normalized)
            if any(marker in compact for marker in ("附录", "附表"))
            and "财务报表" in compact
        ),
        None,
    )
    anchorless_terminal_run = anchor is None
    scan_start = 0 if anchor is None else anchor + 1
    low_text = [
        index
        for index in range(scan_start, len(page_texts))
        if len(normalized[index]) <= 30
    ]
    if not low_text:
        return ()
    runs: list[list[int]] = []
    for page_index in low_text:
        if not runs or page_index != runs[-1][-1] + 1:
            runs.append([page_index])
        else:
            runs[-1].append(page_index)
    eligible = []
    for run in runs:
        if len(run) < 4 or run[-1] < len(page_texts) - 2:
            continue
        if not anchorless_terminal_run:
            if run[0] == anchor + 1:
                eligible.append(run)
            continue
        preceding_index = run[0] - 1
        if preceding_index < 0:
            continue
        preceding = normalized[preceding_index]
        whole_document_prefix = "".join(normalized[: run[0]])
        has_terminal_signature = (
            "法定代表人" in preceding
            and any(marker in preceding for marker in ("日期", "年月日"))
        )
        has_quarterly_report_context = any(
            marker in whole_document_prefix
            for marker in ("第一季度报告", "第三季度报告")
        )
        if has_quarterly_report_context and (
            has_terminal_signature or len(run) >= 6
        ):
            eligible.append(run)
    if len(eligible) != 1:
        return ()
    return tuple(eligible[0])


def _quarterly_inventory_pairs_with_exact_glyph_confusion(
    outputs: dict[float, list[dict[str, Any]]],
    by_scale: dict[float, dict[str, list[_v1_6.OcrObservationV1_6]]],
) -> list[tuple[_v1_6.OcrObservationV1_6, _v1_6.OcrObservationV1_6]]:
    exact = _v1_6._agreed_pairs(by_scale, "INVENTORY_END")
    if exact:
        return exact
    first_scale, second_scale = _v1_6._v1_3.OCR_RENDER_SCALES
    first_rows = by_scale[first_scale].get("INVENTORY_END") or []
    repaired_second: list[_v1_6.OcrObservationV1_6] = []
    for record in outputs[second_scale]:
        for row in _v1_6._v1_3._ocr_rows(record):
            compact = _v1_4.compact_financial_text(row["compact"])
            if re.match(r"^(?:子贝|存贝|子货)", compact) is None:
                continue
            numeric: list[tuple[str, Decimal]] = []
            for segment in row["segments"]:
                raw_value = str(segment["text"])
                parsed = _v1_6._normalize_ocr_number(raw_value)
                if parsed is not None:
                    numeric.append((raw_value, parsed))
            if not numeric:
                continue
            raw_value, value = numeric[0]
            repaired_second.append(
                _v1_6.OcrObservationV1_6(
                    metric_id="INVENTORY_END",
                    page_index=int(row["page_index"]),
                    render_scale=float(row["render_scale"]),
                    raw_value=raw_value,
                    value=value,
                    row_text=compact[:1000],
                    pixel_width=int(row["pixel_width"]),
                    pixel_height=int(row["pixel_height"]),
                )
            )
    pairs: list[
        tuple[_v1_6.OcrObservationV1_6, _v1_6.OcrObservationV1_6]
    ] = []
    for first in first_rows:
        for second in repaired_second:
            if first.page_index == second.page_index and first.value == second.value:
                pairs.append((first, second))
    return pairs


def _infer_quarterly_image_structure(
    result: dict[str, Any],
    by_scale: dict[float, dict[str, list[_v1_6.OcrObservationV1_6]]],
    pages: Sequence[int],
) -> dict[str, Any] | None:
    expected_assets = _v1_6._metric_cny(result, "TOTAL_ASSETS_END")
    expected_parent = _v1_6._metric_cny(result, "PARENT_NET_PROFIT_YTD")
    if expected_assets is None or expected_parent is None:
        return None
    candidates: list[dict[str, Any]] = []
    liability_pairs = _v1_6._agreed_pairs(by_scale, "TOTAL_LIABILITIES_END")
    equity_pairs = _v1_6._agreed_pairs(by_scale, "TOTAL_EQUITY_END")
    for unit in _base.UNIT_MULTIPLIERS:
        parent_pair = _v1_6._anchor_pair(
            by_scale,
            "PARENT_NET_PROFIT_YTD",
            expected_parent,
            unit,
        )
        if parent_pair is None:
            continue
        for liability_pair in liability_pairs:
            for equity_pair in equity_pairs:
                if liability_pair[1].page_index not in pages:
                    continue
                if equity_pair[1].page_index not in pages:
                    continue
                if abs(
                    liability_pair[1].page_index - equity_pair[1].page_index
                ) > 2:
                    continue
                if max(
                    liability_pair[1].page_index,
                    equity_pair[1].page_index,
                ) >= parent_pair[1].page_index:
                    continue
                assets_display = (
                    liability_pair[1].value + equity_pair[1].value
                )
                if not _v1_6._display_matches_cny(
                    assets_display,
                    expected_assets,
                    unit,
                ):
                    continue
                candidates.append(
                    {
                        "unit": unit,
                        "parent_pair": parent_pair,
                        "liability_pair": liability_pair,
                        "equity_pair": equity_pair,
                        "assets_display": assets_display,
                    }
                )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _add_quarterly_image_ocr(
    result: dict[str, Any],
    content: bytes,
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> dict[str, Any]:
    target_missing = set(result.get("missing_metrics") or []).intersection(
        {
            "OPERATING_PROFIT_YTD",
            "ACCOUNTS_RECEIVABLE_END",
            "INVENTORY_END",
            "TOTAL_LIABILITIES_END",
        }
    )
    if not target_missing:
        return {
            "status": "NOT_NEEDED_NO_QUARTERLY_IMAGE_TARGET_METRIC_MISSING",
            "admitted_metric_ids": [],
        }
    pages = find_quarterly_image_statement_pages(
        page_texts,
        period_type=period_type,
    )
    if not pages:
        return {
            "status": "NOT_APPLICABLE_NO_FROZEN_QUARTERLY_IMAGE_STATEMENT_RUN",
            "admitted_metric_ids": [],
        }
    try:
        outputs, render_receipt = _v1_6._v1_3._run_windows_ocr(content, pages)
    except Exception as error:  # noqa: BLE001 - OCR失败显式保留不完整状态
        return {
            "status": "NO_VIEW_QUARTERLY_WINDOWS_OCR_FAILED",
            "page_numbers": [page + 1 for page in pages],
            "error": f"{type(error).__name__}: {error}"[:3000],
            "admitted_metric_ids": [],
        }
    by_scale = {
        scale: _v1_6._ocr_observations(records)
        for scale, records in outputs.items()
    }
    structure = _infer_quarterly_image_structure(result, by_scale, pages)
    if structure is None:
        render_receipt.update(
            {
                "status": "NO_VIEW_QUARTERLY_IMAGE_UNIT_OR_STRUCTURE_NOT_UNIQUE",
                "admitted_metric_ids": [],
            }
        )
        return render_receipt

    unit = str(structure["unit"])
    parent_pair = structure["parent_pair"]
    liability_pair = structure["liability_pair"]
    equity_pair = structure["equity_pair"]
    assets_display = structure["assets_display"]
    unit_source_page = parent_pair[1].page_index + 1
    balance_pages = set(
        range(
            pages[0],
            max(
                liability_pair[1].page_index,
                equity_pair[1].page_index,
            )
            + 1,
        )
    )
    validation_common = {
        "rule": (
            "SUMMARY_PARENT_PROFIT_DUAL_SCALE_ANCHOR_PLUS_DUAL_SCALE_"
            "LIABILITIES_EQUITY_IDENTITY_UNIQUELY_IDENTIFIES_UNIT_AND_PAGES"
        ),
        "summary_parent_profit_cny": str(
            _v1_6._metric_cny(result, "PARENT_NET_PROFIT_YTD")
        ),
        "summary_total_assets_cny": str(
            _v1_6._metric_cny(result, "TOTAL_ASSETS_END")
        ),
        "parent_profit_anchor_page": parent_pair[1].page_index + 1,
        "liability_page": liability_pair[1].page_index + 1,
        "equity_page": equity_pair[1].page_index + 1,
        "assets_display_from_liabilities_plus_equity": str(assets_display),
        "unit_uniqueness_candidate_count": 1,
        "dual_render_scale_exact_value_agreement_required": True,
    }
    admitted: list[str] = []

    for metric_id in ("ACCOUNTS_RECEIVABLE_END", "INVENTORY_END"):
        if metric_id not in result.get("missing_metrics", []):
            continue
        if metric_id == "INVENTORY_END":
            pairs = _quarterly_inventory_pairs_with_exact_glyph_confusion(
                outputs,
                by_scale,
            )
            method_suffix = (
                "DUAL_SCALE_EXACT_WITH_FROZEN_INVENTORY_LABEL_GLYPH_CONFUSION"
            )
        else:
            pairs = _v1_6._agreed_pairs(by_scale, metric_id)
            method_suffix = "DUAL_SCALE_EXACT"
        pair = _v1_6._pair_on_pages(pairs, balance_pages)
        if pair is None:
            continue
        validation = {
            **validation_common,
            "same_validated_consolidated_balance_page_range": True,
            "inventory_label_glyph_confusion_rule": (
                "EXACT_SCALE_2_25_存货_AND_SCALE_3_0_子贝_SAME_PAGE_SAME_VALUE"
                if metric_id == "INVENTORY_END"
                and "子贝" in pair[1].row_text
                else None
            ),
        }
        _v1_4._put_metric(
            result,
            _v1_6._ocr_evidence(
                metric_id,
                pair,
                unit=unit,
                unit_source_page=unit_source_page,
                validation=validation,
                method_suffix=method_suffix,
            ),
        )
        admitted.append(metric_id)

    if "TOTAL_LIABILITIES_END" in result.get("missing_metrics", []):
        validation = {
            **validation_common,
            "liability_row_dual_scale_match_passed": True,
            "equity_row_dual_scale_match_passed": True,
            "summary_assets_equal_ocr_liabilities_plus_equity": True,
        }
        _v1_4._put_metric(
            result,
            _v1_6._ocr_evidence(
                "TOTAL_LIABILITIES_END",
                liability_pair,
                unit=unit,
                unit_source_page=unit_source_page,
                validation=validation,
            ),
        )
        admitted.append("TOTAL_LIABILITIES_END")

    if "OPERATING_PROFIT_YTD" in result.get("missing_metrics", []):
        operating_profit_pair = _v1_6._pair_on_pages(
            _v1_6._agreed_pairs(by_scale, "OPERATING_PROFIT_YTD"),
            {parent_pair[1].page_index},
        )
        if operating_profit_pair is not None:
            validation = {
                **validation_common,
                "same_consolidated_income_page_as_parent_profit_anchor": True,
            }
            _v1_4._put_metric(
                result,
                _v1_6._ocr_evidence(
                    "OPERATING_PROFIT_YTD",
                    operating_profit_pair,
                    unit=unit,
                    unit_source_page=unit_source_page,
                    validation=validation,
                ),
            )
            admitted.append("OPERATING_PROFIT_YTD")

    render_receipt.update(
        {
            "status": (
                "PASS_QUARTERLY_IMAGE_OCR_METRICS_ADMITTED"
                if admitted
                else "NO_VIEW_QUARTERLY_IMAGE_OCR_NO_METRIC_ADMITTED"
            ),
            "unit": unit,
            "unit_source_rule": (
                "PARENT_PROFIT_ANCHOR_AND_ASSETS_LIABILITIES_EQUITY_IDENTITY"
            ),
            "unit_source_page": unit_source_page,
            "quarterly_image_statement_pages": [page + 1 for page in pages],
            "admitted_metric_ids": sorted(admitted),
            "remaining_missing_metrics": list(result.get("missing_metrics") or []),
        }
    )
    return render_receipt


def extract_metrics_from_page_texts(
    page_texts: Sequence[str],
    *,
    period_type: str | None = None,
    text_engine: str = "PDFIUM",
) -> dict[str, Any]:
    normalized = [_v1_4.normalize_financial_text(text) for text in page_texts]
    result = _v1_6.extract_metrics_from_page_texts(
        normalized,
        period_type=period_type,
        text_engine=text_engine,
    )
    result["v1_7_extended_core_parent_profit_receipt"] = (
        _add_extended_core_parent_profit(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_extended_parent_net_profit_receipt"] = (
        _add_extended_parent_net_profit(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_summary_reconciled_generic_statement_receipt"] = (
        _add_summary_reconciled_generic_statement_metrics(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_adjacent_note_receivable_reconciliation_receipt"] = (
        _add_adjacent_note_net_receivable_reconciliation(result, normalized)
    )
    result["v1_7_dual_period_text_liability_receipt"] = (
        _add_dual_period_text_liability(result, normalized)
    )
    result["v1_7_quarterly_image_ocr_receipt"] = {
        "status": "NOT_APPLICABLE_PDF_CONTENT_REQUIRED",
        "admitted_metric_ids": [],
    }
    _restamp_result(result)
    return result


def extract_official_pdf_facts(
    content: bytes,
    *,
    period_type: str | None = None,
) -> dict[str, Any]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("响应没有%PDF-文件头")
    result = _v1_6.extract_official_pdf_facts(
        content,
        period_type=period_type,
    )
    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    normalized = [_v1_4.normalize_financial_text(text) for text in page_texts]
    result["v1_7_extended_core_parent_profit_receipt"] = (
        _add_extended_core_parent_profit(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_extended_parent_net_profit_receipt"] = (
        _add_extended_parent_net_profit(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_summary_reconciled_generic_statement_receipt"] = (
        _add_summary_reconciled_generic_statement_metrics(
            result,
            normalized,
            period_type=period_type,
        )
    )
    result["v1_7_adjacent_note_receivable_reconciliation_receipt"] = (
        _add_adjacent_note_net_receivable_reconciliation(result, normalized)
    )
    result["v1_7_dual_period_text_liability_receipt"] = (
        _add_dual_period_text_liability(result, normalized)
    )
    result["v1_7_quarterly_image_ocr_receipt"] = _add_quarterly_image_ocr(
        result,
        content,
        normalized,
        period_type=period_type,
    )
    result["v1_7_text_reextraction_receipt"] = {
        "status": "PASS_REEXTRACTED_ONLY_AFTER_FROZEN_V1_6_RESULT",
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
    }
    _restamp_result(result)
    return result


__all__ = [
    "ADMITTED_VERIFICATION_STATUS",
    "PARSER_VERSION",
    "REQUIRED_METRICS",
    "SectionRange",
    "extract_metrics_from_page_texts",
    "extract_official_pdf_facts",
]
