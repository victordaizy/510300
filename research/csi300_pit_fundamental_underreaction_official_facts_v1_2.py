from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_1 as _v1_1


PARSER_VERSION = "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_PDFIUM_PDFPLUMBER_V1_6_0"
ADMITTED_VERIFICATION_STATUS = _v1_1.ADMITTED_VERIFICATION_STATUS
REQUIRED_METRICS = _v1_1.REQUIRED_METRICS
MetricEvidence = _v1_1.MetricEvidence
SectionRange = _v1_1.SectionRange


def _same_page_statement_table(
    page_compact: str,
    compact_headings: Sequence[str],
    marker_group: Sequence[str],
) -> bool:
    """接受起始页已有真实表头、后续字段延续到紧邻下一页的主体报表。"""

    compact_markers = [_base.compact_text(marker) for marker in marker_group]
    for heading in compact_headings:
        search_from = 0
        while True:
            heading_position = page_compact.find(heading, search_from)
            if heading_position < 0:
                break
            heading_end = heading_position + len(heading)
            present_marker_positions = [
                position
                for marker in compact_markers
                if (position := page_compact.find(marker, heading_end)) >= 0
            ]
            if present_marker_positions:
                header_region = page_compact[heading_end : min(present_marker_positions)]
                if "单位" in header_region or "人民币" in header_region:
                    return True
            search_from = heading_end
    return False


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
            heading_ends and len(page_compact) - max(heading_ends) <= 120
        )
        next_page_compact = (
            _base.compact_text(page_texts[page_index + 1])
            if page_index + 1 < len(page_texts)
            else ""
        )
        split_table_header = bool(
            heading_near_page_end
            and ("项目" in page_compact or "项目" in next_page_compact)
            and (
                "单位" in page_compact
                or "人民币" in page_compact
                or "单位" in next_page_compact
                or "人民币" in next_page_compact
            )
        )
        if any(
            _base._heading_precedes_markers(
                page_texts,
                page_index,
                headings,
                marker_group,
            )
            and (
                _same_page_statement_table(
                    page_compact,
                    compact_headings,
                    marker_group,
                )
                or (
                    split_table_header
                    and all(
                        _base.compact_text(marker) in next_page_compact
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
                unit=_v1_1.detect_unique_unit(page_texts, page_indices[:3]),
            )
        )
    return candidates


def _q3_table_has_current_and_ytd(
    normalized_rows: Sequence[tuple[int, int, int, Sequence[str]]],
    row_index_in_sequence: int,
    numeric_count: int,
) -> bool:
    page, table, _, _ = normalized_rows[row_index_in_sequence]
    header_text = _base.compact_text(
        " ".join(
            " ".join(cells)
            for candidate_page, candidate_table, _, cells in normalized_rows[
                max(0, row_index_in_sequence - 8) : row_index_in_sequence
            ]
            if candidate_page == page and candidate_table == table
        )
    ).replace("本报告期末", "")
    has_ytd = any(
        marker in header_text
        for marker in ("年初至报告期末", "年初至本报告期末", "年初至报告期期末")
    )
    has_current = "本报告期" in header_text
    if has_ytd:
        return has_current
    return numeric_count >= 4


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
    for index, (page, table, row, cells) in enumerate(normalized_rows):
        for label_index, label_cell in enumerate(cells):
            label = _base.compact_text(label_cell)
            joined_label = label
            continuation: dict[str, Any] | None = None
            core_label = "扣除非经常性损益" in label and (
                "净利润" in label or label.endswith("的净")
            )
            if not core_label and label.endswith(("归属于上市公司", "归属于上市公司股")):
                for lookahead in range(index + 1, min(len(normalized_rows), index + 5)):
                    next_page, next_table, next_row, next_cells = normalized_rows[lookahead]
                    if next_page not in {page, page + 1}:
                        break
                    next_label = (
                        _base.compact_text(next_cells[label_index])
                        if label_index < len(next_cells)
                        else ""
                    )
                    candidate_label = label + next_label
                    if (
                        "扣除非经常性损益" in candidate_label
                        and "净利润" in candidate_label
                    ):
                        joined_label = candidate_label
                        core_label = True
                        continuation = {
                            "page": next_page + 1,
                            "table_index": next_table,
                            "row_index": next_row,
                            "label": next_label,
                            "lookahead_rows": lookahead - index,
                        }
                        break
            if not core_label:
                continue

            numeric = _v1_1._numeric_cells_after(cells, label_index)
            if not numeric:
                continue
            q3_current_and_ytd = bool(
                str(period_type or "").upper() == "Q3"
                and _q3_table_has_current_and_ytd(normalized_rows, index, len(numeric))
            )
            selected_index = len(numeric) // 2 if q3_current_and_ytd else 0
            if selected_index >= len(numeric):
                continue
            raw_value, decimal_value = numeric[selected_index]
            unit = _v1_1.detect_unique_unit(page_texts, [page]) or fallback_unit
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
                    "q3_current_and_ytd_columns": q3_current_and_ytd,
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
                    if q3_current_and_ytd
                    else "PDFPLUMBER_SUMMARY_TABLE_FIRST_CURRENT_PERIOD_COLUMN"
                ),
                verification_status=ADMITTED_VERIFICATION_STATUS,
            )
    return None


@contextmanager
def _patched_v1_1_runtime() -> Iterable[None]:
    previous_v1_1_version = _v1_1.PARSER_VERSION
    previous_base_version = _base.PARSER_VERSION
    previous_statement_rule = _v1_1._same_page_statement_table
    previous_section_candidates = _v1_1._section_candidates
    previous_core_table_rule = _v1_1.find_core_profit_from_summary_tables
    _v1_1.PARSER_VERSION = PARSER_VERSION
    _v1_1._same_page_statement_table = _same_page_statement_table
    _v1_1._section_candidates = _section_candidates
    _v1_1.find_core_profit_from_summary_tables = find_core_profit_from_summary_tables
    _base.PARSER_VERSION = PARSER_VERSION
    try:
        yield
    finally:
        _v1_1.PARSER_VERSION = previous_v1_1_version
        _v1_1._same_page_statement_table = previous_statement_rule
        _v1_1._section_candidates = previous_section_candidates
        _v1_1.find_core_profit_from_summary_tables = previous_core_table_rule
        _base.PARSER_VERSION = previous_base_version


def locate_statement_sections(page_texts: Sequence[str]) -> dict[str, SectionRange]:
    with _patched_v1_1_runtime():
        return _v1_1.locate_statement_sections(page_texts)


def detect_unique_unit(
    page_texts: Sequence[str], page_indices: Iterable[int]
) -> str | None:
    with _patched_v1_1_runtime():
        return _v1_1.detect_unique_unit(page_texts, page_indices)


def extract_metrics_from_page_texts(
    page_texts: Sequence[str], *, period_type: str | None = None
) -> dict[str, Any]:
    with _patched_v1_1_runtime():
        result = _v1_1.extract_metrics_from_page_texts(
            page_texts,
            period_type=period_type,
        )
    result["parser_version"] = PARSER_VERSION
    return result


def extract_official_pdf_facts(
    content: bytes, *, period_type: str | None = None
) -> dict[str, Any]:
    with _patched_v1_1_runtime():
        result = _v1_1.extract_official_pdf_facts(content, period_type=period_type)
    result["parser_version"] = PARSER_VERSION
    for metric in result.get("metrics") or []:
        metric["verification_status"] = ADMITTED_VERIFICATION_STATUS
    return result
