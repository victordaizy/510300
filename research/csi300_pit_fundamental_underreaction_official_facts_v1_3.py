from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Iterator, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_1 as _v1_1
from research import csi300_pit_fundamental_underreaction_official_facts_v1_2 as _v1_2


PARSER_VERSION = "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_PDFIUM_PDFPLUMBER_WINDOWS_OCR_V1_7_0"
ADMITTED_VERIFICATION_STATUS = _v1_2.ADMITTED_VERIFICATION_STATUS
REQUIRED_METRICS = _v1_2.REQUIRED_METRICS
MetricEvidence = _v1_2.MetricEvidence
ParsedCandidate = _base.ParsedCandidate
SectionRange = _v1_2.SectionRange
ROOT = Path(__file__).resolve().parents[1]
WINDOWS_OCR_SCRIPT = ROOT / "scripts/windows_ocr_financial_statement_v1_3.ps1"

OCR_RENDER_SCALES = (2.25, 3.0)
OCR_MAX_CONSOLIDATED_STATEMENT_PAGES = 12
OCR_LANGUAGE_TAG = "zh-Hans-CN"
OCR_ENGINE_NAME = "Windows.Media.Ocr.OcrEngine"

QUARTER_TABLE_SPLIT_RE = re.compile(
    r"(?:[一二三四五六七八九十\d]+[、.]?\s*)?分季度主要财务指标"
)
YTD_MARKERS = (
    "年初至报告期末",
    "年初到报告期末",
    "年初至本报告期末",
    "1-9月",
    "1—9月",
    "1－9月",
    "1至9月",
    "前三季度",
)
CURRENT_QUARTER_MARKERS = (
    "本报告期利润表",
    "本报告期金额",
    "7-9月",
    "7—9月",
    "7－9月",
    "7至9月",
)


@contextmanager
def _v1_3_versions() -> Iterator[None]:
    previous = (
        _base.PARSER_VERSION,
        _v1_1.PARSER_VERSION,
        _v1_2.PARSER_VERSION,
    )
    _base.PARSER_VERSION = PARSER_VERSION
    _v1_1.PARSER_VERSION = PARSER_VERSION
    _v1_2.PARSER_VERSION = PARSER_VERSION
    try:
        yield
    finally:
        _base.PARSER_VERSION, _v1_1.PARSER_VERSION, _v1_2.PARSER_VERSION = previous


def _empty_section(name: str) -> SectionRange:
    return SectionRange(name=name, page_indices=(), unit=None)


def _section_payload(section: SectionRange, *, text_engine: str = "PDFIUM") -> dict[str, Any]:
    return {
        "name": section.name,
        "start_page": section.page_indices[0] + 1 if section.page_indices else None,
        "end_page": section.page_indices[-1] + 1 if section.page_indices else None,
        "unit": section.unit,
        "text_engine": text_engine,
    }


def _refresh_result(result: dict[str, Any]) -> None:
    metrics = {row["metric_id"]: row for row in result.get("metrics") or []}
    result["metrics"] = [metrics[item] for item in REQUIRED_METRICS if item in metrics]
    result["missing_metrics"] = [item for item in REQUIRED_METRICS if item not in metrics]
    result["document_complete"] = not result["missing_metrics"]
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if result["document_complete"]
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )
    result["parser_version"] = PARSER_VERSION


def _add_metric_dict(result: dict[str, Any], metric: dict[str, Any]) -> None:
    metrics = {row["metric_id"]: row for row in result.get("metrics") or []}
    metrics[str(metric["metric_id"])] = metric
    result["metrics"] = list(metrics.values())
    _refresh_result(result)


def _add_evidence(result: dict[str, Any], evidence: MetricEvidence) -> None:
    _add_metric_dict(result, asdict(evidence))


def _stamp_metric(metric: dict[str, Any], *, text_engine: str | None = None) -> None:
    locator: dict[str, Any]
    try:
        locator = json.loads(str(metric.get("source_locator") or "{}"))
    except json.JSONDecodeError:
        locator = {"legacy_source_locator": str(metric.get("source_locator") or "")}
    previous_version = locator.get("parser_version")
    if previous_version and previous_version != PARSER_VERSION:
        locator.setdefault("base_parser_version", previous_version)
    locator["parser_version"] = PARSER_VERSION
    if text_engine:
        locator["text_engine"] = text_engine
    metric["source_locator"] = json.dumps(
        locator,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    method = str(metric.get("source_method") or "")
    if text_engine == "PDFPLUMBER" and method.startswith("PDFIUM_"):
        method = "PDFPLUMBER_" + method.removeprefix("PDFIUM_")
    metric["source_method"] = method
    metric["verification_status"] = ADMITTED_VERIFICATION_STATUS


def _stamp_result(result: dict[str, Any], *, text_engine: str | None = None) -> None:
    for metric in result.get("metrics") or []:
        _stamp_metric(metric, text_engine=text_engine)
    _refresh_result(result)


def _candidate_evidence(
    metric_id: str,
    candidate: ParsedCandidate,
    *,
    value_period_scope: str,
    section_name: str,
    text_engine: str,
    validation: dict[str, Any] | None = None,
    actual_page_number: int | None = None,
) -> MetricEvidence:
    page_number = actual_page_number or candidate.page_number
    locator: dict[str, Any] = {
        "page": page_number,
        "section": section_name,
        "line_window": candidate.line_window[:700],
        "label_span": list(candidate.match_span),
        "value_span": list(candidate.value_span),
        "selected_amount_index": candidate.selected_amount_index,
        "text_engine": text_engine,
        "parser_version": PARSER_VERSION,
    }
    if validation:
        locator["validation"] = validation
    return MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=page_number,
        source_locator=json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method=(
            f"{text_engine}_TEXT_LOGICAL_ROW_AMOUNT_INDEX_"
            f"{candidate.selected_amount_index}_AFTER_PROVABLE_OPTIONAL_NOTE"
        ),
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _generic_column_section(
    page_texts: Sequence[str],
    *,
    name: str,
    headings: Sequence[str],
    marker_groups: Sequence[Sequence[str]],
    terminators: Sequence[str],
) -> SectionRange:
    starts: list[int] = []
    for page_index, page_text in enumerate(page_texts):
        compact = _base.compact_text(page_text)
        if "母公司" + headings[0] in compact:
            continue
        if not any(_base.compact_text(heading) in compact for heading in headings):
            continue
        if "合并" not in compact or "公司" not in compact:
            continue
        if not any(token in compact for token in ("单位", "人民币")):
            continue
        next_compact = (
            _base.compact_text(page_texts[page_index + 1])
            if page_index + 1 < len(page_texts)
            else ""
        )
        if not any(
            all(_base.compact_text(marker) in compact + next_compact for marker in group)
            for group in marker_groups
        ):
            continue
        starts.append(page_index)
    if not starts:
        return _empty_section(name)

    start = starts[0]
    compact_terminators = [_base.compact_text(item) for item in terminators]
    end = len(page_texts)
    for page_index in range(start + 1, len(page_texts)):
        compact = _base.compact_text(page_texts[page_index])
        if any(marker in compact for marker in compact_terminators):
            end = page_index
            break
    pages = tuple(range(start, max(start + 1, end)))
    unit = _v1_1.detect_unique_unit(page_texts, pages[:3])
    return SectionRange(name=name, page_indices=pages, unit=unit)


def _content_classified_balance_pair(page_texts: Sequence[str]) -> SectionRange:
    for page_index in range(len(page_texts) - 1):
        first = _base.compact_text(page_texts[page_index])
        second = _base.compact_text(page_texts[page_index + 1])
        first_markers = (
            "流动资产",
            "货币资金",
            "应收账款",
            "存货",
            "资产总计",
            "合并",
            "公司",
            "附注",
        )
        second_markers = (
            "流动负债",
            "负债合计",
            "股东权益合计",
            "负债及股东权益总计",
        )
        if not all(marker in first for marker in first_markers):
            continue
        if not all(marker in second for marker in second_markers):
            continue
        if len(_base.NUMBER_RE.findall(page_texts[page_index])) < 20:
            continue
        if len(_base.NUMBER_RE.findall(page_texts[page_index + 1])) < 20:
            continue
        unit = _v1_1.detect_unique_unit(page_texts, (page_index, page_index + 1))
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        return SectionRange(
            name="CONSOLIDATED_BALANCE_SHEET",
            page_indices=(page_index, page_index + 1),
            unit=unit,
        )
    return _empty_section("CONSOLIDATED_BALANCE_SHEET")


def locate_statement_sections(page_texts: Sequence[str]) -> dict[str, SectionRange]:
    with _v1_3_versions():
        sections = _v1_2.locate_statement_sections(page_texts)

    if not sections["balance"].page_indices:
        sections["balance"] = _generic_column_section(
            page_texts,
            name="CONSOLIDATED_BALANCE_SHEET",
            headings=("资产负债表",),
            marker_groups=(("流动资产", "货币资金", "资产总计"),),
            terminators=(
                "年初至报告期末利润表",
                "本报告期利润表",
                "利润表",
                "现金流量表",
            ),
        )
    if not sections["balance"].page_indices:
        sections["balance"] = _content_classified_balance_pair(page_texts)

    if not sections["income"].page_indices:
        sections["income"] = _generic_column_section(
            page_texts,
            name="CONSOLIDATED_INCOME_STATEMENT",
            headings=("年初至报告期末利润表", "本报告期利润表", "利润表"),
            marker_groups=(
                ("营业收入", "营业成本", "营业利润"),
                ("营业总收入", "营业总成本", "营业利润"),
            ),
            terminators=("现金流量表", "所有者权益变动表", "股东权益变动表"),
        )

    if not sections["cash"].page_indices:
        sections["cash"] = _generic_column_section(
            page_texts,
            name="CONSOLIDATED_CASH_FLOW_STATEMENT",
            headings=("现金流量表",),
            marker_groups=(
                (
                    "经营活动产生的现金流量",
                    "销售商品",
                    "经营活动产生的现金流量净额",
                ),
            ),
            terminators=("所有者权益变动表", "股东权益变动表"),
        )
    return sections


def _statement_unit_consensus(sections: dict[str, SectionRange]) -> str | None:
    units = [section.unit for section in sections.values() if section.unit]
    return units[0] if units and len(set(units)) == 1 else None


def _income_selection(
    page_texts: Sequence[str],
    section: SectionRange,
    period_type: str | None,
) -> tuple[tuple[int, ...], int]:
    if str(period_type or "").upper() != "Q3":
        return section.page_indices, 0
    ytd_starts = [
        page
        for page in section.page_indices
        if any(marker in _base.compact_text(page_texts[page]) for marker in YTD_MARKERS)
    ]
    if ytd_starts:
        start = ytd_starts[0]
        end = section.page_indices[-1] + 1
        for page in section.page_indices:
            if page <= start:
                continue
            compact = _base.compact_text(page_texts[page])
            if any(
                heading in compact
                for heading in (
                    "母公司利润表",
                    "本报告期利润表",
                    "年初至报告期末利润表",
                    "年初至本报告期末利润表",
                )
            ):
                end = page
                break
        selected = tuple(page for page in section.page_indices if start <= page < end)
        header = _base.compact_text(page_texts[start])
        combined_columns = any(marker in header for marker in CURRENT_QUARTER_MARKERS)
        return selected, 2 if combined_columns else 0
    return section.page_indices, _v1_1._q3_income_amount_index(
        page_texts,
        section,
        period_type,
    )


def _find_candidate(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str | None,
    amount_index: int = 0,
    skip_note: bool = True,
) -> ParsedCandidate | None:
    if skip_note:
        chained = _find_candidate_after_multiple_note_references(
            page_texts,
            pages,
            patterns,
            unit=unit,
            amount_index=amount_index,
        )
        if chained is not None:
            return chained
    with _v1_3_versions():
        return _v1_1.find_amount(
            page_texts,
            pages,
            label_patterns=patterns,
            fallback_unit=unit,
            skip_statement_note_number=skip_note,
            amount_index=amount_index,
        )


def _reference_only_gap(text: str) -> bool:
    compact = _base.compact_text(text)
    compact = compact.replace("附注", "").replace("注释", "").replace("注", "")
    compact = re.sub(
        r"[（()）\[\]【】,，、./／:：;；一二三四五六七八九十百千万第项号-]",
        "",
        compact,
    )
    return compact == ""


def _small_reference_number(match: re.Match[str]) -> bool:
    raw = _base._raw_amount(match).strip()
    normalized = raw.replace("−", "-").replace("－", "-").replace("—", "-")
    normalized = normalized.strip("()（）")
    return bool(re.fullmatch(r"\d{1,3}", normalized))


def _multiple_note_reference_count(
    window: str,
    label_end: int,
    matches: Sequence[re.Match[str]],
) -> int:
    """Return a proven multi-reference prefix length, never a guessed single note."""

    if len(matches) < 3:
        return 0
    count = 0
    previous_end = label_end
    for match in matches:
        if not _small_reference_number(match):
            break
        if not _reference_only_gap(window[previous_end : match.start()]):
            break
        count += 1
        previous_end = match.end()
    if count < 2 or count >= len(matches):
        return 0
    next_match = matches[count]
    if not _reference_only_gap(window[previous_end : next_match.start()]):
        return 0
    raw_next = _base._raw_amount(next_match)
    try:
        next_value = abs(_base.parse_decimal(raw_next))
    except ValueError:
        return 0
    if raw_next not in {"-", "－", "—"} and next_value < 1000 and "," not in raw_next:
        return 0
    return count


def _find_candidate_after_multiple_note_references(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str | None,
    amount_index: int,
) -> ParsedCandidate | None:
    """Parse rows such as ``(39),十五(12) amount`` with exact span evidence."""

    compiled = [re.compile(pattern) for pattern in patterns]
    for page_index in pages:
        page_text = page_texts[page_index]
        page_units = _v1_1._unit_candidates(page_text)
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
                if not amount_matches:
                    continue
                if _base._contains_unrelated_label_before_first_amount(
                    window[label_match.end() : amount_matches[0].start()]
                ):
                    continue
                note_count = _multiple_note_reference_count(
                    window,
                    label_match.end(),
                    amount_matches,
                )
                selected_index = note_count + amount_index
                if note_count == 0 or selected_index >= len(amount_matches):
                    continue
                amount_match = amount_matches[selected_index]
                selected_unit = (
                    _v1_1._nearest_explicit_unit(window, label_match, amount_match)
                    or unit
                    or page_unit
                )
                if selected_unit not in _base.UNIT_MULTIPLIERS:
                    continue
                raw_value = _base._raw_amount(amount_match)
                return ParsedCandidate(
                    value_cny=(
                        _base.parse_decimal(raw_value)
                        * _base.UNIT_MULTIPLIERS[selected_unit]
                    ),
                    page_number=page_index + 1,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=selected_unit,
                    line_window=window[:500],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=amount_index,
                )
    return None


def _metric_selects_proven_multiple_note_reference(metric: dict[str, Any]) -> bool:
    try:
        locator = json.loads(str(metric.get("source_locator") or "{}"))
        window = str(locator["line_window"])
        label_end = int(locator["label_span"][1])
        value_span = tuple(int(item) for item in locator["value_span"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    amount_matches = _base._number_matches_after(window, label_end)
    note_count = _multiple_note_reference_count(window, label_end, amount_matches)
    return any(match.span() == value_span for match in amount_matches[:note_count])


def _remove_proven_note_reference_metrics(result: dict[str, Any]) -> None:
    original = list(result.get("metrics") or [])
    result["metrics"] = [
        metric
        for metric in original
        if not _metric_selects_proven_multiple_note_reference(metric)
    ]
    if len(result["metrics"]) != len(original):
        _refresh_result(result)


def _validated_balance_metrics(
    page_texts: Sequence[str],
    section: SectionRange,
    *,
    text_engine: str,
) -> dict[str, MetricEvidence]:
    if not section.page_indices or section.unit not in _base.UNIT_MULTIPLIERS:
        return {}
    definitions: dict[str, Sequence[str]] = {
        "ACCOUNTS_RECEIVABLE_END": (r"(?<!应收票据及)(?<!其他)应收账款",),
        "INVENTORY_END": (r"(?<!周转)存货",),
        "TOTAL_ASSETS_END": (r"(?<!流动)(?<!非流动)资产总计",),
        "TOTAL_LIABILITIES_END": (r"(?<!流动)(?<!非流动)负债合计",),
    }
    candidates = {
        metric: _find_candidate(
            page_texts,
            section.page_indices,
            patterns,
            unit=section.unit,
        )
        for metric, patterns in definitions.items()
    }
    equity = _find_candidate(
        page_texts,
        section.page_indices,
        (r"(?<!母公司)(?<!少数)股东权益合计",),
        unit=section.unit,
    )
    assets = candidates["TOTAL_ASSETS_END"]
    liabilities = candidates["TOTAL_LIABILITIES_END"]
    if assets is None or liabilities is None or equity is None:
        return {}
    if assets.value_cny != liabilities.value_cny + equity.value_cny:
        return {}
    validation = {
        "rule": "TOTAL_ASSETS_EQUALS_TOTAL_LIABILITIES_PLUS_TOTAL_EQUITY",
        "total_assets_cny": str(assets.value_cny),
        "total_liabilities_cny": str(liabilities.value_cny),
        "total_equity_cny": str(equity.value_cny),
    }
    return {
        metric: _candidate_evidence(
            metric,
            candidate,
            value_period_scope="PERIOD_END",
            section_name=section.name,
            text_engine=text_engine,
            validation=validation,
        )
        for metric, candidate in candidates.items()
        if candidate is not None
    }


def _add_standard_missing_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    text_engine: str,
) -> None:
    sections = locate_statement_sections(page_texts)
    consensus = _statement_unit_consensus(sections)

    for metric, evidence in _validated_balance_metrics(
        page_texts,
        sections["balance"],
        text_engine=text_engine,
    ).items():
        if metric in result["missing_metrics"]:
            _add_evidence(result, evidence)

    income_pages, income_index = _income_selection(
        page_texts,
        sections["income"],
        period_type,
    )
    definitions: dict[str, dict[str, Any]] = {
        "OPERATING_REVENUE_YTD": {
            "section": sections["income"],
            "pages": income_pages,
            "patterns": (r"其中[:：]?营业收入", r"一[、.]营业收入", r"一[、.]营业总收入", r"营业总收入", r"(?<!总)营业收入"),
            "scope": "YEAR_TO_DATE",
            "index": income_index,
        },
        "OPERATING_PROFIT_YTD": {
            "section": sections["income"],
            "pages": income_pages,
            "patterns": (
                r"(?:[一二三四五六七八九十][、.]?\s*)?营业利润\s*[/／]\s*[（(]亏损[）)]",
                r"(?:[一二三四五六七八九十][、.]?\s*)?营业利润\s*[（(]亏损以[^）)]{0,40}填列[）)]",
                r"(?:[一二三四五六七八九十][、.]?\s*)?营业利润",
            ),
            "scope": "YEAR_TO_DATE",
            "index": income_index,
        },
        "PARENT_NET_PROFIT_YTD": {
            "section": sections["income"],
            "pages": income_pages,
            "patterns": (
                r"(?:\d+[.、]?\s*)?归属于母公司(?:普通股)?(?:所有者|股东)的净利润\s*[/／]\s*[（(]亏损[）)]",
                r"(?:\d+[.、]?\s*)?归属于母公司(?:普通股)?(?:所有者|股东)的净利润\s*[（(]净亏损以[^）)]{0,40}号填(?:列)?[）)]?",
                r"按所有权归属分类归属于母公司普通股股东",
                r"(?:\d+[.、]?\s*)?归属于母公司(?:普通股)?(?:所有者|股东)的净利润",
                r"归属于上市公司股东的净利润",
            ),
            "scope": "YEAR_TO_DATE",
            "index": income_index,
        },
        "OPERATING_CASH_FLOW_YTD": {
            "section": sections["cash"],
            "pages": sections["cash"].page_indices,
            "patterns": (
                r"经营活动产生\s*[/／]?\s*(?:[（(]使用[）)])?\s*的现金流量净额",
                r"经营活动使用的现金流量净额",
                r"经营活动产生的现金流量净额",
            ),
            "scope": "YEAR_TO_DATE",
            "index": 0,
        },
    }
    for metric_id, definition in definitions.items():
        if metric_id not in result["missing_metrics"]:
            continue
        section: SectionRange = definition["section"]
        if not section.page_indices:
            continue
        candidate = _find_candidate(
            page_texts,
            definition["pages"],
            definition["patterns"],
            unit=section.unit or consensus,
            amount_index=int(definition["index"]),
        )
        if candidate is None:
            continue
        _add_evidence(
            result,
            _candidate_evidence(
                metric_id,
                candidate,
                value_period_scope=str(definition["scope"]),
                section_name=section.name,
                text_engine=text_engine,
            ),
        )


def _add_core_before_quarter_table(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    text_engine: str,
) -> None:
    if "CORE_PARENT_NET_PROFIT_YTD" not in result["missing_metrics"]:
        return
    if str(period_type or "").upper() == "Q3":
        return
    sections = locate_statement_sections(page_texts)
    summary_pages = list(_base._summary_page_indices(page_texts, sections))
    truncated = list(page_texts)
    for page in summary_pages:
        truncated[page] = QUARTER_TABLE_SPLIT_RE.split(truncated[page], maxsplit=1)[0]
    unit = _v1_1.detect_unique_unit(truncated, summary_pages) or _statement_unit_consensus(sections)
    candidate = _find_candidate(
        truncated,
        summary_pages,
        _v1_1.CORE_PROFIT_PATTERNS,
        unit=unit,
        skip_note=False,
    )
    if candidate is None:
        return
    _add_evidence(
        result,
        _candidate_evidence(
            "CORE_PARENT_NET_PROFIT_YTD",
            candidate,
            value_period_scope="YEAR_TO_DATE",
            section_name="MAIN_FINANCIAL_HIGHLIGHTS_BEFORE_QUARTER_TABLE",
            text_engine=text_engine,
            validation={
                "rule": "ANNUAL_SUMMARY_REGION_TRUNCATED_BEFORE_QUARTER_TABLE_HEADING"
            },
        ),
    )


def _adjust_candidate_page(candidate: ParsedCandidate, page_number: int) -> ParsedCandidate:
    return ParsedCandidate(
        value_cny=candidate.value_cny,
        page_number=page_number,
        label=candidate.label,
        raw_value=candidate.raw_value,
        unit=candidate.unit,
        line_window=candidate.line_window,
        match_span=candidate.match_span,
        value_span=candidate.value_span,
        selected_amount_index=candidate.selected_amount_index,
    )


def _note_bounds(page_texts: Sequence[str]) -> tuple[int, int] | None:
    start = next(
        (
            index
            for index, text in enumerate(page_texts)
            if any(
                marker in _base.compact_text(text)
                for marker in ("合并财务报表主要项目注释", "合并财务报表项目注释")
            )
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index
            for index in range(start + 1, len(page_texts))
            if "母公司财务报表主要项目注释" in _base.compact_text(page_texts[index])
        ),
        len(page_texts),
    )
    return start, end


def _fragment_candidate(
    fragment: str,
    patterns: Sequence[str],
    *,
    unit: str,
    amount_index: int,
    actual_page_number: int,
) -> ParsedCandidate | None:
    candidate = _find_candidate(
        (fragment,),
        (0,),
        patterns,
        unit=unit,
        amount_index=amount_index,
    )
    return _adjust_candidate_page(candidate, actual_page_number) if candidate else None


def _add_reconciled_combined_receivable_note(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    text_engine: str,
) -> None:
    if "ACCOUNTS_RECEIVABLE_END" not in result["missing_metrics"]:
        return
    bounds = _note_bounds(page_texts)
    if bounds is None:
        return
    sections = locate_statement_sections(page_texts)
    unit = _statement_unit_consensus(sections)
    if unit not in _base.UNIT_MULTIPLIERS:
        return
    for page_index in range(*bounds):
        text = page_texts[page_index]
        compact = _base.compact_text(text)
        if not all(
            marker in compact
            for marker in ("应收票据及应收账款", "应收票据", "应收账款", "合计")
        ):
            continue
        start_match = re.search(r"应收票据及应收账款", text)
        if start_match is None:
            continue
        fragment = text[start_match.start() :]
        candidates: dict[str, ParsedCandidate | None] = {}
        for key, pattern in {
            "bills": (r"(?<!及)应收票据(?!及应收账款)",),
            "receivable": (r"(?<!票据及)应收账款",),
            "total": (r"合计",),
        }.items():
            candidates[key + "_current"] = _fragment_candidate(
                fragment,
                pattern,
                unit=unit,
                amount_index=0,
                actual_page_number=page_index + 1,
            )
            candidates[key + "_prior"] = _fragment_candidate(
                fragment,
                pattern,
                unit=unit,
                amount_index=1,
                actual_page_number=page_index + 1,
            )
        if any(candidate is None for candidate in candidates.values()):
            continue
        typed = {key: value for key, value in candidates.items() if value is not None}
        if typed["bills_current"].value_cny + typed["receivable_current"].value_cny != typed["total_current"].value_cny:
            continue
        if typed["bills_prior"].value_cny + typed["receivable_prior"].value_cny != typed["total_prior"].value_cny:
            continue
        candidate = typed["receivable_current"]
        _add_evidence(
            result,
            _candidate_evidence(
                "ACCOUNTS_RECEIVABLE_END",
                candidate,
                value_period_scope="PERIOD_END",
                section_name="CONSOLIDATED_NOTES_COMBINED_RECEIVABLE_BREAKDOWN",
                text_engine=text_engine,
                validation={
                    "rule": "BILLS_PLUS_ACCOUNTS_RECEIVABLE_EQUALS_COMBINED_TOTAL_CURRENT_AND_PRIOR",
                    "bills_current_cny": str(typed["bills_current"].value_cny),
                    "accounts_receivable_current_cny": str(candidate.value_cny),
                    "combined_current_cny": str(typed["total_current"].value_cny),
                    "bills_prior_cny": str(typed["bills_prior"].value_cny),
                    "accounts_receivable_prior_cny": str(typed["receivable_prior"].value_cny),
                    "combined_prior_cny": str(typed["total_prior"].value_cny),
                },
            ),
        )
        return


def extract_metrics_from_page_texts(
    page_texts: Sequence[str],
    *,
    period_type: str | None = None,
    text_engine: str = "PDFIUM",
) -> dict[str, Any]:
    with _v1_3_versions():
        result = _v1_2.extract_metrics_from_page_texts(
            page_texts,
            period_type=period_type,
        )
    _remove_proven_note_reference_metrics(result)
    _stamp_result(result, text_engine=text_engine)
    _add_standard_missing_metrics(
        result,
        page_texts,
        period_type=period_type,
        text_engine=text_engine,
    )
    _add_core_before_quarter_table(
        result,
        page_texts,
        period_type=period_type,
        text_engine=text_engine,
    )
    _add_reconciled_combined_receivable_note(
        result,
        page_texts,
        text_engine=text_engine,
    )
    sections = locate_statement_sections(page_texts)
    result["sections"] = {
        key: _section_payload(section, text_engine=text_engine)
        for key, section in sections.items()
    }
    _stamp_result(result, text_engine=text_engine)
    return result


def _blank_accounts_receivable_evidence(
    page_texts: Sequence[str],
    table_rows: Iterable[tuple[int, int, int, Sequence[Any]]],
    *,
    fallback_unit: str | None,
) -> MetricEvidence | None:
    for page_index, table_index, row_index, raw_cells in table_rows:
        cells = [_base.normalize_text(cell) for cell in raw_cells]
        for label_index, cell in enumerate(cells):
            if _base.compact_text(cell) != "应收账款":
                continue
            trailing = cells[label_index + 1 :]
            if len(trailing) < 2 or any(_base.compact_text(value) for value in trailing):
                continue
            unit = _v1_1.detect_unique_unit(page_texts, (page_index,)) or fallback_unit
            if unit not in _base.UNIT_MULTIPLIERS:
                continue
            locator = json.dumps(
                {
                    "page": page_index + 1,
                    "section": "CONSOLIDATED_BALANCE_SHEET",
                    "table_index": table_index,
                    "row_index": row_index,
                    "label_cell_index": label_index,
                    "row_cells": cells,
                    "validation_rule": "EXACT_ACCOUNTS_RECEIVABLE_LABEL_AND_ALL_TRAILING_CURRENT_AND_PRIOR_AMOUNT_CELLS_EXPLICITLY_BLANK",
                    "parser_version": PARSER_VERSION,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return MetricEvidence(
                metric_id="ACCOUNTS_RECEIVABLE_END",
                metric_value_cny=0.0,
                statement_scope="CONSOLIDATED_ONLY",
                value_period_scope="PERIOD_END",
                source_page=page_index + 1,
                source_locator=locator,
                source_label=cell,
                source_raw_value="[CURRENT_AND_PRIOR_AMOUNT_CELLS_BLANK]",
                source_unit=unit,
                source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
                source_method="PDFPLUMBER_TABLE_EXPLICIT_BLANK_CURRENT_AND_PRIOR_ACCOUNTS_RECEIVABLE_CELLS_AS_ZERO",
                verification_status=ADMITTED_VERIFICATION_STATUS,
            )
    return None


def _extract_pdfplumber_page_texts(content: bytes) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    errors: list[str] = []
    with _base.pdfplumber.open(io.BytesIO(content)) as document:
        for page_index, page in enumerate(document.pages):
            try:
                texts.append(_base.normalize_text(page.extract_text() or ""))
            except Exception as error:  # noqa: BLE001 - fallback failure remains incomplete
                texts.append("")
                errors.append(f"PAGE_{page_index + 1}:{type(error).__name__}:{error}"[:1000])
    return texts, errors


def _text_corruption_ratio(page_texts: Sequence[str]) -> float:
    joined = "".join(page_texts)
    nonspace = sum(not char.isspace() for char in joined)
    if nonspace == 0:
        return 0.0
    bad = joined.count("\ufffd") + sum("\ue000" <= char <= "\uf8ff" for char in joined)
    return bad / nonspace


def _should_try_pdfplumber_text(
    page_texts: Sequence[str],
    result: dict[str, Any],
) -> bool:
    if result.get("document_complete"):
        return False
    sections = locate_statement_sections(page_texts)
    all_missing = all(not section.page_indices for section in sections.values())
    return _text_corruption_ratio(page_texts) >= 0.02 or (
        len(page_texts) <= 80 and all_missing
    )


def _merge_missing_metrics(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> None:
    present = {row["metric_id"] for row in primary.get("metrics") or []}
    for metric in secondary.get("metrics") or []:
        if metric["metric_id"] in present:
            continue
        _add_metric_dict(primary, dict(metric))
        present.add(metric["metric_id"])
    primary_sections = primary.get("sections") or {}
    for key, section in (secondary.get("sections") or {}).items():
        if not (primary_sections.get(key) or {}).get("start_page") and section.get("start_page"):
            primary_sections[key] = section
    primary["sections"] = primary_sections
    _refresh_result(primary)


def _find_statement_image_pages(page_texts: Sequence[str]) -> tuple[int, ...]:
    contents_pages = [
        index
        for index, text in enumerate(page_texts)
        if all(
            marker in _base.compact_text(text)
            for marker in (
                "合并资产负债表",
                "合并利润表",
                "合并现金流量表",
                "公司资产负债表",
            )
        )
    ]
    for contents_page in contents_pages:
        compact = _base.compact_text(page_texts[contents_page])
        consolidated_match = re.search(r"合并资产负债表(\d+)", compact)
        company_match = re.search(r"公司资产负债表(\d+)", compact)
        if consolidated_match is None or company_match is None:
            continue
        count = int(company_match.group(1)) - int(consolidated_match.group(1))
        if count < 4 or count > OCR_MAX_CONSOLIDATED_STATEMENT_PAGES:
            continue
        start = next(
            (
                index
                for index in range(contents_page + 1, min(len(page_texts), contents_page + 7))
                if len(_base.compact_text(page_texts[index])) <= 20
            ),
            None,
        )
        if start is None or start + count > len(page_texts):
            continue
        pages = tuple(range(start, start + count))
        if all(len(_base.compact_text(page_texts[index])) <= 20 for index in pages):
            return pages
    return ()


def _find_image_statement_unit(
    page_texts: Sequence[str],
    pages: Sequence[int],
) -> tuple[str | None, int | None]:
    if not pages:
        return None, None
    search_pages = range(max(0, pages[0] - 3), min(len(page_texts), pages[-1] + 25))
    pattern = re.compile(
        r"(?:均以|金额单位为|单位[:：]?)(?:人民币)?(?P<unit>百万元|千元|万元|亿元|元)(?:为单位)?"
    )
    hits: list[tuple[int, str]] = []
    for page in search_pages:
        for match in pattern.finditer(_base.compact_text(page_texts[page])):
            hits.append((page, match.group("unit")))
    unique = list(dict.fromkeys(unit for _, unit in hits))
    if len(unique) != 1:
        return None, None
    unit = unique[0]
    source_page = next(page + 1 for page, found in hits if found == unit)
    return unit, source_page


def _run_windows_ocr(
    content: bytes,
    pages: Sequence[int],
    *,
    render_scales: Sequence[float] | None = None,
) -> tuple[dict[float, list[dict[str, Any]]], dict[str, Any]]:
    if not WINDOWS_OCR_SCRIPT.exists():
        raise FileNotFoundError(WINDOWS_OCR_SCRIPT)
    selected_scales = tuple(
        float(scale) for scale in (render_scales or OCR_RENDER_SCALES)
    )
    if not selected_scales or len(set(selected_scales)) != len(selected_scales):
        raise ValueError("OCR渲染倍率必须非空且互不重复")
    outputs: dict[float, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="csi300_v1_3_ocr_") as temporary:
        temporary_path = Path(temporary)
        document = _base.pdfium.PdfDocument(content)
        try:
            for scale in selected_scales:
                image_paths: list[Path] = []
                for page_index in pages:
                    page = document[page_index]
                    try:
                        destination = temporary_path / (
                            f"page_{page_index + 1:04d}_scale_{str(scale).replace('.', '_')}.png"
                        )
                        bitmap = page.render(scale=scale)
                        try:
                            bitmap.to_pil().save(destination)
                        finally:
                            bitmap.close()
                        image_paths.append(destination)
                    finally:
                        page.close()
                list_path = temporary_path / f"images_{str(scale).replace('.', '_')}.json"
                list_path.write_text(
                    json.dumps([str(path.resolve()) for path in image_paths], ensure_ascii=False),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(WINDOWS_OCR_SCRIPT),
                        "-ImageListPath",
                        str(list_path),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    check=False,
                    timeout=300,
                )
                if completed.returncode != 0:
                    error = completed.stderr.decode("utf-8", errors="replace")[:3000]
                    raise RuntimeError(f"Windows OCR failed at scale {scale}: {error}")
                records = [
                    json.loads(line)
                    for line in completed.stdout.decode("utf-8").splitlines()
                    if line.strip()
                ]
                if len(records) != len(pages):
                    raise RuntimeError(
                        f"Windows OCR page count mismatch at scale {scale}: "
                        f"{len(records)} != {len(pages)}"
                    )
                for page_index, record in zip(pages, records, strict=True):
                    record["page_index"] = page_index
                    record["render_scale"] = scale
                outputs[scale] = records
        finally:
            document.close()
    first = outputs[selected_scales[0]][0]
    receipt = {
        "status": "PASS_WINDOWS_OCR_RENDERED",
        "engine": first.get("ocr_engine"),
        "language_tag": first.get("language_tag"),
        "max_image_dimension": first.get("max_image_dimension"),
        "render_scales": list(selected_scales),
        "page_numbers": [page + 1 for page in pages],
        "temporary_images_retained": False,
    }
    return outputs, receipt


def _line_runs(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(words, key=lambda word: float(word["x"]))
    if not ordered:
        return []
    heights = [float(word["height"]) for word in ordered if float(word["height"]) > 0]
    median_height = sorted(heights)[len(heights) // 2] if heights else 20.0
    split_gap = max(20.0, median_height * 1.2)
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for word in ordered[1:]:
        previous = groups[-1][-1]
        gap = float(word["x"]) - (float(previous["x"]) + float(previous["width"]))
        if gap > split_gap:
            groups.append([word])
        else:
            groups[-1].append(word)
    runs: list[dict[str, Any]] = []
    for group in groups:
        left = min(float(word["x"]) for word in group)
        top = min(float(word["y"]) for word in group)
        right = max(float(word["x"]) + float(word["width"]) for word in group)
        bottom = max(float(word["y"]) + float(word["height"]) for word in group)
        runs.append(
            {
                "text": "".join(str(word["text"]) for word in group).replace(" ", ""),
                "x": left,
                "right": right,
                "cy": (top + bottom) / 2,
                "height": bottom - top,
            }
        )
    return runs


def _ocr_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for line in record.get("lines") or []:
        segments.extend(_line_runs(line.get("words") or []))
    segments.sort(key=lambda segment: (float(segment["cy"]), float(segment["x"])))
    groups: list[dict[str, Any]] = []
    for segment in segments:
        candidates = [
            group
            for group in groups[-8:]
            if abs(float(group["cy"]) - float(segment["cy"]))
            <= max(8.0, min(22.0, 0.55 * max(float(group["height"]), float(segment["height"]))))
        ]
        if not candidates:
            groups.append(
                {
                    "cy": float(segment["cy"]),
                    "height": float(segment["height"]),
                    "segments": [segment],
                }
            )
            continue
        group = min(candidates, key=lambda candidate: abs(float(candidate["cy"]) - float(segment["cy"])))
        group["segments"].append(segment)
        centers = sorted(float(item["cy"]) for item in group["segments"])
        heights = sorted(float(item["height"]) for item in group["segments"])
        group["cy"] = centers[len(centers) // 2]
        group["height"] = heights[len(heights) // 2]
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group["segments"], key=lambda segment: float(segment["x"]))
        rows.append(
            {
                "page_index": int(record["page_index"]),
                "render_scale": float(record["render_scale"]),
                "pixel_width": int(record["pixel_width"]),
                "pixel_height": int(record["pixel_height"]),
                "segments": ordered,
                "compact": "".join(str(segment["text"]) for segment in ordered),
            }
        )
    return rows


OCR_NUMBER_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "．": ".",
        "·": ".",
        "。": ".",
        "（": "(",
        "）": ")",
        "－": "-",
        "—": "-",
        "−": "-",
        "]": "1",
        "】": "1",
        "L": "1",
        "I": "1",
        "l": "1",
        "丨": "1",
    }
)


def _normalize_ocr_number(value: str) -> Decimal | None:
    text = re.sub(r"\s+", "", value).translate(OCR_NUMBER_TRANSLATION)
    text = text.replace("巧", "15").replace("、", ",")
    if re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", text) is None:
        return None
    negative = text.startswith("(") or text.endswith(")")
    normalized = text.strip("()").replace(",", "")
    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    return -abs(amount) if negative else amount


@dataclass(frozen=True)
class OcrObservation:
    key: str
    page_index: int
    render_scale: float
    raw_value: str
    value: Decimal
    row_text: str
    pixel_width: int
    pixel_height: int


OCR_ROW_PATTERNS: dict[str, re.Pattern[str]] = {
    "OPERATING_REVENUE_YTD": re.compile(r"(?<!总)营业收入"),
    "OPERATING_PROFIT_YTD": re.compile(r"(?<!非)营业利润"),
    "PARENT_NET_PROFIT_YTD": re.compile(r"归属于母公司股东的净利润"),
    "OPERATING_CASH_FLOW_YTD": re.compile(r"经营活动(?:产生|使用)的现金流量净额"),
    "ACCOUNTS_RECEIVABLE_END": re.compile(r"(?<!应收票据及)(?<!其他)应收账款"),
    "INVENTORY_END": re.compile(r"(?<!周转)存货"),
    "TOTAL_ASSETS_END": re.compile(r"(?<!流动)(?<!非流动)资产总计"),
    "TOTAL_LIABILITIES_END": re.compile(r"(?<!流动)(?<!非流动)负债合计"),
    "TOTAL_EQUITY_END": re.compile(r"(?<!母公司)(?<!少数)股东权益合计"),
    "TOTAL_LIABILITIES_EQUITY_END": re.compile(r"负债(?:和|及)股东权益总计"),
    "NONOPERATING_INCOME_YTD": re.compile(r"加[:：]营业外收入"),
    "NONOPERATING_EXPENSE_YTD": re.compile(r"减[:：]营业外支出"),
    "PROFIT_TOTAL_YTD": re.compile(r"(?<!营业)利润总额"),
    "NET_PROFIT_YTD": re.compile(r"^净利润"),
    "MINORITY_PROFIT_YTD": re.compile(r"少数股东损益"),
}


def _ocr_observations(records: Sequence[dict[str, Any]]) -> dict[str, list[OcrObservation]]:
    observations: dict[str, list[OcrObservation]] = {key: [] for key in OCR_ROW_PATTERNS}
    for record in records:
        for row in _ocr_rows(record):
            compact = str(row["compact"])
            for key, pattern in OCR_ROW_PATTERNS.items():
                match = pattern.search(compact)
                if match is None:
                    continue
                if key == "ACCOUNTS_RECEIVABLE_END" and "应收票据及应收账款" in compact:
                    continue
                numeric: list[tuple[str, Decimal]] = []
                for segment in row["segments"]:
                    value = _normalize_ocr_number(str(segment["text"]))
                    if value is not None:
                        numeric.append((str(segment["text"]), value))
                if len(numeric) >= 3 and numeric[0][1] == numeric[0][1].to_integral_value() and abs(numeric[0][1]) <= 999:
                    numeric = numeric[1:]
                if not numeric:
                    continue
                raw_value, value = numeric[0]
                observations[key].append(
                    OcrObservation(
                        key=key,
                        page_index=int(row["page_index"]),
                        render_scale=float(row["render_scale"]),
                        raw_value=raw_value,
                        value=value,
                        row_text=compact[:700],
                        pixel_width=int(row["pixel_width"]),
                        pixel_height=int(row["pixel_height"]),
                    )
                )
    for values in observations.values():
        values.sort(key=lambda item: (item.page_index, item.render_scale))
    return observations


def _agreed_ocr_observation(
    by_scale: dict[float, dict[str, list[OcrObservation]]],
    key: str,
) -> tuple[OcrObservation, OcrObservation] | None:
    first_scale, second_scale = OCR_RENDER_SCALES
    first_values = by_scale[first_scale].get(key) or []
    second_values = by_scale[second_scale].get(key) or []
    for first in first_values:
        for second in second_values:
            if first.page_index == second.page_index and first.value == second.value:
                return first, second
    return None


def _summary_duplicate_values(
    page_texts: Sequence[str],
    unit: str,
) -> dict[str, Decimal]:
    pages = tuple(
        index
        for index, text in enumerate(page_texts[:60])
        if "主要会计数据" in _base.compact_text(text)
        and "单位" in _base.compact_text(text)
    )
    if not pages:
        return {}
    patterns = {
        "OPERATING_REVENUE_YTD": (r"(?<!总)营业收入",),
        "PARENT_NET_PROFIT_YTD": (r"归属于上市公司股东的净利润",),
        "OPERATING_CASH_FLOW_YTD": (r"经营活动产生的现金流量净额",),
        "TOTAL_ASSETS_END": (r"总资产",),
    }
    values: dict[str, Decimal] = {}
    for key, label_patterns in patterns.items():
        candidate = _find_candidate(
            page_texts,
            pages,
            label_patterns,
            unit=unit,
            skip_note=False,
        )
        if candidate is not None:
            values[key] = candidate.value_cny / _base.UNIT_MULTIPLIERS[unit]
    return values


def _ocr_metric_evidence(
    metric_id: str,
    agreed: tuple[OcrObservation, OcrObservation],
    *,
    unit: str,
    unit_source_page: int,
    validations: dict[str, Any],
) -> MetricEvidence:
    first, second = agreed
    scope = "PERIOD_END" if metric_id.endswith("_END") else "YEAR_TO_DATE"
    section = (
        "CONSOLIDATED_BALANCE_SHEET"
        if scope == "PERIOD_END"
        else (
            "CONSOLIDATED_CASH_FLOW_STATEMENT"
            if metric_id == "OPERATING_CASH_FLOW_YTD"
            else "CONSOLIDATED_INCOME_STATEMENT"
        )
    )
    locator = {
        "page": second.page_index + 1,
        "section": section,
        "ocr_engine": OCR_ENGINE_NAME,
        "ocr_language_tag": OCR_LANGUAGE_TAG,
        "render_scales": list(OCR_RENDER_SCALES),
        "raw_value_by_scale": {
            str(first.render_scale): first.raw_value,
            str(second.render_scale): second.raw_value,
        },
        "normalized_value_by_scale": {
            str(first.render_scale): str(first.value),
            str(second.render_scale): str(second.value),
        },
        "row_text_by_scale": {
            str(first.render_scale): first.row_text,
            str(second.render_scale): second.row_text,
        },
        "pixel_dimensions_by_scale": {
            str(first.render_scale): [first.pixel_width, first.pixel_height],
            str(second.render_scale): [second.pixel_width, second.pixel_height],
        },
        "unit_source_page": unit_source_page,
        "validation": validations,
        "parser_version": PARSER_VERSION,
    }
    return MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(second.value * _base.UNIT_MULTIPLIERS[unit]),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=scope,
        source_page=second.page_index + 1,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=second.row_text.split(str(second.raw_value), 1)[0],
        source_raw_value=second.raw_value,
        source_unit=unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
        source_method="WINDOWS_MEDIA_OCR_DUAL_RENDER_SCALE_EXACT_VALUE_AGREEMENT",
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _add_ocr_fallback(
    result: dict[str, Any],
    content: bytes,
    page_texts: Sequence[str],
) -> dict[str, Any]:
    pages = _find_statement_image_pages(page_texts)
    if not pages:
        return {"status": "NOT_APPLICABLE_NO_FROZEN_IMAGE_STATEMENT_RUN"}
    removed_metric_ids = [
        str(metric["metric_id"])
        for metric in result.get("metrics") or []
        if metric.get("metric_id") != "CORE_PARENT_NET_PROFIT_YTD"
    ]
    result["metrics"] = [
        metric
        for metric in result.get("metrics") or []
        if metric.get("metric_id") == "CORE_PARENT_NET_PROFIT_YTD"
    ]
    _refresh_result(result)
    unit, unit_source_page = _find_image_statement_unit(page_texts, pages)
    if unit not in _base.UNIT_MULTIPLIERS or unit_source_page is None:
        return {
            "status": "NO_VIEW_OCR_UNIT_NOT_UNIQUE",
            "page_numbers": [page + 1 for page in pages],
            "discarded_pre_ocr_non_core_metric_ids": removed_metric_ids,
        }
    try:
        outputs, receipt = _run_windows_ocr(content, pages)
    except Exception as error:  # noqa: BLE001 - OCR failure remains explicit incomplete
        return {
            "status": "NO_VIEW_WINDOWS_OCR_FAILED",
            "page_numbers": [page + 1 for page in pages],
            "error": f"{type(error).__name__}: {error}"[:3000],
            "discarded_pre_ocr_non_core_metric_ids": removed_metric_ids,
        }
    receipt["discarded_pre_ocr_non_core_metric_ids"] = removed_metric_ids
    by_scale = {
        scale: _ocr_observations(records)
        for scale, records in outputs.items()
    }
    agreed = {
        key: _agreed_ocr_observation(by_scale, key)
        for key in OCR_ROW_PATTERNS
    }
    required_auxiliary = (
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
        "TOTAL_EQUITY_END",
        "TOTAL_LIABILITIES_EQUITY_END",
        "OPERATING_PROFIT_YTD",
        "NONOPERATING_INCOME_YTD",
        "NONOPERATING_EXPENSE_YTD",
        "PROFIT_TOTAL_YTD",
        "NET_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
        "MINORITY_PROFIT_YTD",
    )
    if any(agreed[key] is None for key in required_auxiliary):
        receipt.update(
            {
                "status": "NO_VIEW_OCR_AUXILIARY_ROWS_NOT_DUAL_SCALE_AGREED",
                "missing_agreement_keys": [key for key in required_auxiliary if agreed[key] is None],
            }
        )
        return receipt
    values = {key: pair[1].value for key, pair in agreed.items() if pair is not None}
    balance_pass = (
        values["TOTAL_ASSETS_END"]
        == values["TOTAL_LIABILITIES_END"] + values["TOTAL_EQUITY_END"]
        == values["TOTAL_LIABILITIES_EQUITY_END"]
    )
    profit_pass = (
        values["OPERATING_PROFIT_YTD"]
        + values["NONOPERATING_INCOME_YTD"]
        - values["NONOPERATING_EXPENSE_YTD"]
        == values["PROFIT_TOTAL_YTD"]
        and values["PARENT_NET_PROFIT_YTD"] + values["MINORITY_PROFIT_YTD"]
        == values["NET_PROFIT_YTD"]
    )
    if not balance_pass or not profit_pass:
        receipt.update(
            {
                "status": "NO_VIEW_OCR_ACCOUNTING_IDENTITY_FAILED",
                "balance_identity_passed": balance_pass,
                "profit_identity_passed": profit_pass,
            }
        )
        return receipt
    duplicates = _summary_duplicate_values(page_texts, unit)
    duplicate_checks: dict[str, bool] = {}
    for key, expected in duplicates.items():
        pair = agreed.get(key)
        if pair is not None:
            duplicate_checks[key] = pair[1].value == expected
    if duplicate_checks and not all(duplicate_checks.values()):
        receipt.update(
            {
                "status": "NO_VIEW_OCR_SUMMARY_DUPLICATE_MISMATCH",
                "summary_duplicate_checks": duplicate_checks,
            }
        )
        return receipt
    validations = {
        "dual_render_scale_exact_value_agreement": True,
        "total_assets_equals_liabilities_plus_equity": True,
        "total_assets_equals_liabilities_and_equity_total": True,
        "operating_profit_plus_nonoperating_income_minus_expense_equals_profit_total": True,
        "parent_profit_plus_minority_profit_equals_net_profit": True,
        "summary_duplicate_checks": duplicate_checks,
    }
    for metric_id in REQUIRED_METRICS:
        if metric_id not in result["missing_metrics"]:
            continue
        pair = agreed.get(metric_id)
        if pair is None:
            continue
        _add_evidence(
            result,
            _ocr_metric_evidence(
                metric_id,
                pair,
                unit=unit,
                unit_source_page=unit_source_page,
                validations=validations,
            ),
        )
    receipt.update(
        {
            "status": (
                "PASS_WINDOWS_OCR_REQUIRED_METRICS_ADMITTED"
                if result["document_complete"]
                else "PARTIAL_WINDOWS_OCR_SOME_REQUIRED_METRICS_STILL_MISSING"
            ),
            "unit": unit,
            "unit_source_page": unit_source_page,
            "balance_identity_passed": True,
            "profit_identity_passed": True,
            "summary_duplicate_checks": duplicate_checks,
            "admitted_metric_ids": [
                metric_id
                for metric_id in REQUIRED_METRICS
                if agreed.get(metric_id) is not None
            ],
            "discarded_pre_ocr_non_core_metric_ids": removed_metric_ids,
        }
    )
    return receipt


def extract_official_pdf_facts(
    content: bytes,
    *,
    period_type: str | None = None,
) -> dict[str, Any]:
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    result = extract_metrics_from_page_texts(
        page_texts,
        period_type=period_type,
        text_engine="PDFIUM",
    )
    sections = locate_statement_sections(page_texts)
    table_receipts: dict[str, Any] = {}

    summary_pages = _base._summary_page_indices(page_texts, sections)
    rows, errors = _base.extract_pdf_table_rows(content, summary_pages)
    with _v1_3_versions():
        core = _v1_2.find_core_profit_from_summary_tables(
            page_texts,
            rows,
            period_type=period_type,
            fallback_unit=(
                _v1_1.detect_unique_unit(page_texts, summary_pages)
                or _statement_unit_consensus(sections)
            ),
        )
    table_receipts["core_parent_net_profit"] = {
        "status": "PASS" if core is not None else "NO_MATCH",
        "examined_table_row_count": len(rows),
        "page_errors": errors,
    }
    if core is not None and "CORE_PARENT_NET_PROFIT_YTD" in result["missing_metrics"]:
        _add_evidence(result, core)

    balance_rows: list[tuple[int, int, int, Sequence[Any]]] = []
    balance_errors: list[str] = []
    if sections["balance"].page_indices and any(
        metric in result["missing_metrics"]
        for metric in ("INVENTORY_END", "ACCOUNTS_RECEIVABLE_END")
    ):
        balance_rows, balance_errors = _base.extract_pdf_table_rows(
            content,
            sections["balance"].page_indices,
        )
    if "INVENTORY_END" in result["missing_metrics"]:
        inventory = _base.find_explicit_blank_current_and_prior_inventory(
            page_texts,
            balance_rows,
            fallback_unit=sections["balance"].unit,
        )
        table_receipts["blank_inventory"] = {
            "status": "PASS" if inventory is not None else "NO_MATCH",
            "examined_table_row_count": len(balance_rows),
            "page_errors": balance_errors,
        }
        if inventory is not None:
            _add_evidence(result, inventory)
    if "ACCOUNTS_RECEIVABLE_END" in result["missing_metrics"]:
        blank_receivable = _blank_accounts_receivable_evidence(
            page_texts,
            balance_rows,
            fallback_unit=sections["balance"].unit,
        )
        table_receipts["blank_accounts_receivable"] = {
            "status": "PASS" if blank_receivable is not None else "NO_MATCH",
            "examined_table_row_count": len(balance_rows),
            "page_errors": balance_errors,
        }
        if blank_receivable is not None:
            _add_evidence(result, blank_receivable)

    pdfplumber_receipt: dict[str, Any] = {"status": "NOT_NEEDED"}
    if _should_try_pdfplumber_text(page_texts, result):
        secondary_texts, secondary_errors = _extract_pdfplumber_page_texts(content)
        secondary = extract_metrics_from_page_texts(
            secondary_texts,
            period_type=period_type,
            text_engine="PDFPLUMBER",
        )
        before = set(result["missing_metrics"])
        _merge_missing_metrics(result, secondary)
        admitted = sorted(before.difference(result["missing_metrics"]))
        pdfplumber_receipt = {
            "status": "PASS_SECONDARY_TEXT_METRICS_ADMITTED" if admitted else "NO_NEW_MATCH",
            "trigger_corruption_ratio": _text_corruption_ratio(page_texts),
            "page_errors": secondary_errors,
            "admitted_metric_ids": admitted,
        }

    ocr_receipt: dict[str, Any] = {"status": "NOT_NEEDED"}
    if not result["document_complete"]:
        ocr_receipt = _add_ocr_fallback(result, content, page_texts)

    result.update(
        {
            "parser_version": PARSER_VERSION,
            "official_pdf_sha256": pdf_sha256,
            "official_pdf_size_bytes": len(content),
            "table_fallback_receipt": table_receipts,
            "pdfplumber_secondary_text_receipt": pdfplumber_receipt,
            "ocr_fallback_receipt": ocr_receipt,
            **text_receipt,
        }
    )
    _stamp_result(result)
    return result


__all__ = [
    "ADMITTED_VERIFICATION_STATUS",
    "PARSER_VERSION",
    "REQUIRED_METRICS",
    "extract_metrics_from_page_texts",
    "extract_official_pdf_facts",
    "locate_statement_sections",
]
