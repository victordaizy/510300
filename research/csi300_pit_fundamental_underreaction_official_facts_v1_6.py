from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_3 as _v1_3
from research import csi300_pit_fundamental_underreaction_official_facts_v1_4 as _v1_4
from research import csi300_pit_fundamental_underreaction_official_facts_v1_5 as _v1_5


PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_"
    "ANCHORED_IMAGE_OCR_INVERSE_CORE_LABEL_NET_RECEIVABLE_RECONCILIATION_V1_10_0"
)
REQUIRED_METRICS = _v1_5.REQUIRED_METRICS
ADMITTED_VERIFICATION_STATUS = _v1_5.ADMITTED_VERIFICATION_STATUS
SectionRange = _v1_5.SectionRange

CORE_PARENT_INVERSE_LABEL_PATTERNS = (
    r"扣除非经常性损益后归属于上市公司(?:普通股)?股东的净利润",
    r"扣除非经常损益后归属于上市公司(?:普通股)?股东的净利润",
    r"归属于上市公司(?:普通股)?股东的扣除非经常(?:性)?损益后的净利润",
)
SUMMARY_CONTEXT_MARKERS = (
    "主要会计数据",
    "主要财务数据",
    "主要会计资料",
    "会计数据和财务指标摘要",
    "会计数据及财务指标摘要",
    "会计资料和财务指标摘要",
)
Q3_CURRENT_MARKERS = ("7-9月", "7—9月", "7－9月", "本报告期")
Q3_YTD_MARKERS = ("1-9月", "1—9月", "1－9月", "年初至报告期末")

OCR_NUMBER_TRANSLATION_V1_6 = str.maketrans(
    {
        **{
            "，": ",",
            "．": ".",
            "·": ".",
            "。": ".",
            "（": "(",
            "）": ")",
            "－": "-",
            "—": "-",
            "−": "-",
            "、": ",",
            "》": ",",
            "〉": ",",
            "＞": ",",
            "：": ",",
            ":": ",",
        },
        **{
            "]": "1",
            "】": "1",
            "L": "1",
            "I": "1",
            "l": "1",
            "丨": "1",
            "乙": "2",
            "艺": "2",
        },
    }
)

OCR_ROW_PATTERNS_V1_6: dict[str, re.Pattern[str]] = {
    "OPERATING_REVENUE_YTD": re.compile(r"(?<!总)营业收入"),
    "OPERATING_PROFIT_YTD": re.compile(r"(?<!非)营业利润"),
    "PARENT_NET_PROFIT_YTD": re.compile(r"属于母公司股东的净利润"),
    "OPERATING_CASH_FLOW_YTD": re.compile(r"经营活动(?:产生|使用)的现金流量净额"),
    "ACCOUNTS_RECEIVABLE_END": re.compile(r"(?<!应收票据及)(?<!其他)应收账款"),
    "INVENTORY_END": re.compile(r"(?<!周转)存货"),
    "TOTAL_ASSETS_END": re.compile(r"(?<!流动)(?<!非流动)资产总计"),
    "TOTAL_LIABILITIES_END": re.compile(r"(?<!流动)(?<!非流动)负[债偾](?:合计|总计)"),
    "TOTAL_EQUITY_END": re.compile(
        r"(?<!母公司)(?<!少数)(?:股东|所有者)权益(?:合计|总计)"
    ),
    "TOTAL_LIABILITIES_EQUITY_END": re.compile(
        r"负[债偾](?:和|及)(?:股东|所有者)权益总计"
    ),
}


@dataclass(frozen=True)
class OcrObservationV1_6:
    metric_id: str
    page_index: int
    render_scale: float
    raw_value: str
    value: Decimal
    row_text: str
    pixel_width: int
    pixel_height: int


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(metric["metric_id"]): metric
        for metric in result.get("metrics") or []
    }


def _restamp_result(result: dict[str, Any]) -> None:
    for metric in result.get("metrics") or []:
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {"legacy_source_locator": str(metric.get("source_locator") or "")}
        previous = str(locator.get("parser_version") or result.get("parser_version") or "")
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


def _summary_page_candidates(page_texts: Sequence[str]) -> tuple[int, ...]:
    selected: set[int] = set()
    for page_index in range(min(40, len(page_texts))):
        compact = _v1_4.compact_financial_text(page_texts[page_index])
        previous = (
            _v1_4.compact_financial_text(page_texts[page_index - 1])
            if page_index > 0
            else ""
        )
        if any(marker in compact or marker in previous for marker in SUMMARY_CONTEXT_MARKERS):
            selected.add(page_index)
            if page_index + 1 < min(40, len(page_texts)):
                selected.add(page_index + 1)
    return tuple(sorted(selected))


def _q3_core_candidate(
    page_texts: Sequence[str],
    pages: Sequence[int],
) -> _base.ParsedCandidate | None:
    compiled = [re.compile(pattern) for pattern in CORE_PARENT_INVERSE_LABEL_PATTERNS]
    for page_index in pages:
        compact = _v1_4.compact_financial_text(page_texts[page_index])
        if not (
            any(marker in compact for marker in Q3_CURRENT_MARKERS)
            and any(marker in compact for marker in Q3_YTD_MARKERS)
        ):
            continue
        unit = _v1_4._unit_near_page(page_texts, page_index, (page_index,))
        if unit not in _base.UNIT_MULTIPLIERS:
            continue
        for window in _base.logical_line_windows(page_texts[page_index]):
            for pattern in compiled:
                label_match = pattern.search(window)
                if label_match is None:
                    continue
                amount_matches = _base._number_matches_after(
                    window[: min(len(window), label_match.end() + 320)],
                    label_match.end(),
                )
                if len(amount_matches) < 2:
                    continue
                comparison_region = window[
                    amount_matches[0].end() : amount_matches[1].start()
                ]
                if "%" not in comparison_region:
                    continue
                amount_match = amount_matches[1]
                raw_value = _base._raw_amount(amount_match)
                return _base.ParsedCandidate(
                    value_cny=(
                        _base.parse_decimal(raw_value) * _base.UNIT_MULTIPLIERS[unit]
                    ),
                    page_number=page_index + 1,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=window[:700],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=1,
                )
    return None


def _add_inverse_core_parent_profit(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> dict[str, Any]:
    if "CORE_PARENT_NET_PROFIT_YTD" not in result.get("missing_metrics", []):
        return {"status": "NOT_NEEDED_METRIC_ALREADY_PRESENT", "admitted_metric_ids": []}
    pages = _summary_page_candidates(page_texts)
    if not pages:
        return {"status": "NO_MATCH_NO_EXACT_SUMMARY_CONTEXT", "admitted_metric_ids": []}
    if str(period_type or "").upper() == "Q3":
        candidate = _q3_core_candidate(page_texts, pages)
    else:
        candidate = None
        for page_index in pages:
            unit = _v1_4._unit_near_page(page_texts, page_index, (page_index,))
            if unit not in _base.UNIT_MULTIPLIERS:
                continue
            candidate = _v1_3._find_candidate(
                page_texts,
                (page_index,),
                CORE_PARENT_INVERSE_LABEL_PATTERNS,
                unit=unit,
                skip_note=False,
            )
            if candidate is not None:
                break
    if candidate is None:
        return {
            "status": "NO_MATCH_EXACT_INVERSE_CORE_LABEL_NOT_FOUND",
            "examined_page_numbers": [page + 1 for page in pages],
            "admitted_metric_ids": [],
        }
    _v1_4._put_metric(
        result,
        _v1_4._text_evidence(
            "CORE_PARENT_NET_PROFIT_YTD",
            candidate,
            value_period_scope="YEAR_TO_DATE",
            section_name="MAIN_FINANCIAL_HIGHLIGHTS",
            validation={
                "rule": "EXACT_MAIN_FINANCIAL_HIGHLIGHTS_INVERSE_ORDER_CORE_PARENT_PROFIT_LABEL",
                "q3_selected_amount_index": candidate.selected_amount_index,
                "summary_page_limit": 40,
            },
        ),
    )
    return {
        "status": "PASS_EXACT_INVERSE_CORE_LABEL_ADMITTED",
        "examined_page_numbers": [page + 1 for page in pages],
        "source_page": candidate.page_number,
        "source_label": candidate.label,
        "source_raw_value": candidate.raw_value,
        "source_unit": candidate.unit,
        "admitted_metric_ids": ["CORE_PARENT_NET_PROFIT_YTD"],
    }


def _candidate_pair(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
    skip_note: bool,
) -> tuple[_base.ParsedCandidate, _base.ParsedCandidate] | None:
    current = _v1_3._find_candidate(
        page_texts,
        pages,
        patterns,
        unit=unit,
        amount_index=0,
        skip_note=skip_note,
    )
    prior = _v1_3._find_candidate(
        page_texts,
        pages,
        patterns,
        unit=unit,
        amount_index=1,
        skip_note=skip_note,
    )
    if current is None or prior is None or current.page_number != prior.page_number:
        return None
    return current, prior


def _add_net_receivable_reconciliation(
    result: dict[str, Any],
    page_texts: Sequence[str],
) -> dict[str, Any]:
    if "ACCOUNTS_RECEIVABLE_END" not in result.get("missing_metrics", []):
        return {"status": "NOT_NEEDED_METRIC_ALREADY_PRESENT", "admitted_metric_ids": []}
    sections = _v1_4.locate_statement_sections(page_texts)
    balance = sections["balance"]
    unit = balance.unit
    if not balance.page_indices or unit not in _base.UNIT_MULTIPLIERS:
        return {"status": "NO_MATCH_NO_VALIDATED_CONSOLIDATED_BALANCE_SECTION", "admitted_metric_ids": []}
    combined = _candidate_pair(
        page_texts,
        balance.page_indices,
        (r"应收票据(?:及|和)应收账款",),
        unit=unit,
        skip_note=True,
    )
    if combined is None:
        return {"status": "NO_MATCH_COMBINED_BALANCE_ROW_NOT_FOUND", "admitted_metric_ids": []}

    note_pages = []
    for page_index, text in enumerate(page_texts):
        compact = _v1_4.compact_financial_text(text)
        if not any(marker in compact for marker in ("财务报表附注", "财务报表注释")):
            continue
        if all(
            marker in compact
            for marker in ("应收票据", "应收账款", "坏账准备")
        ) and any(marker in compact for marker in ("应收票据及应收账款", "应收票据和应收账款")):
            note_pages.append(page_index)

    for page_index in note_pages:
        bills = _candidate_pair(
            page_texts,
            (page_index,),
            (r"(?<!及)(?<!和)应收票据(?!及应收账款)(?!和应收账款)",),
            unit=unit,
            skip_note=False,
        )
        gross = _candidate_pair(
            page_texts,
            (page_index,),
            (r"(?<!票据及)(?<!票据和)应收账款",),
            unit=unit,
            skip_note=False,
        )
        allowance = _candidate_pair(
            page_texts,
            (page_index,),
            (r"减[:：]?坏账准备",),
            unit=unit,
            skip_note=False,
        )
        if bills is None or gross is None or allowance is None:
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
        if not current_pass or not prior_pass:
            continue
        multiplier = _base.UNIT_MULTIPLIERS[unit]
        net_display = net_current / multiplier
        locator = {
            "page": page_index + 1,
            "section": "CONSOLIDATED_NOTES_COMBINED_RECEIVABLE_BREAKDOWN",
            "parser_version": PARSER_VERSION,
            "statement_combined_source_page": combined[0].page_number,
            "validation": {
                "rule": "BILLS_PLUS_GROSS_ACCOUNTS_RECEIVABLE_MINUS_ALLOWANCE_EQUALS_COMBINED_TOTAL_CURRENT_AND_PRIOR",
                "current_period_identity_passed": True,
                "prior_period_identity_passed": True,
                "bills_current_cny": str(bills[0].value_cny),
                "gross_accounts_receivable_current_cny": str(gross[0].value_cny),
                "allowance_current_cny": str(allowance[0].value_cny),
                "net_accounts_receivable_current_cny": str(net_current),
                "combined_current_cny": str(combined[0].value_cny),
                "bills_prior_cny": str(bills[1].value_cny),
                "gross_accounts_receivable_prior_cny": str(gross[1].value_cny),
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
            source_page=page_index + 1,
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label="应收账款减：坏账准备",
            source_raw_value=(
                f"{gross[0].raw_value}-{allowance[0].raw_value}={net_display}"
            ),
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "PDFIUM_NOTE_GROSS_RECEIVABLE_MINUS_ALLOWANCE_"
                "RECONCILED_TO_COMBINED_BALANCE_BOTH_PERIODS"
            ),
            verification_status=ADMITTED_VERIFICATION_STATUS,
        )
        _v1_4._put_metric(result, evidence)
        return {
            "status": "PASS_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED",
            "balance_source_page": combined[0].page_number,
            "note_source_page": page_index + 1,
            "current_period_identity_passed": True,
            "prior_period_identity_passed": True,
            "admitted_metric_ids": ["ACCOUNTS_RECEIVABLE_END"],
        }
    return {
        "status": "NO_MATCH_DUAL_PERIOD_NET_RECEIVABLE_IDENTITY_NOT_PROVEN",
        "examined_note_page_numbers": [page + 1 for page in note_pages],
        "admitted_metric_ids": [],
    }


def _low_text_runs(page_texts: Sequence[str]) -> list[tuple[int, ...]]:
    low = [
        index
        for index, text in enumerate(page_texts)
        if len(_v1_4.compact_financial_text(text)) <= 30
    ]
    runs: list[list[int]] = []
    for page_index in low:
        if not runs or page_index != runs[-1][-1] + 1:
            runs.append([page_index])
        else:
            runs[-1].append(page_index)
    return [tuple(run) for run in runs if 4 <= len(run) <= 24]


def find_image_statement_pages(page_texts: Sequence[str]) -> tuple[int, ...]:
    inherited = _v1_3._find_statement_image_pages(page_texts)
    if inherited:
        return inherited
    candidates: list[tuple[int, ...]] = []
    for run in _low_text_runs(page_texts):
        before = "".join(
            _v1_4.compact_financial_text(page_texts[index])
            for index in range(max(0, run[0] - 8), run[0])
        )
        after = "".join(
            _v1_4.compact_financial_text(page_texts[index])
            for index in range(run[-1] + 1, min(len(page_texts), run[-1] + 5))
        )
        audit_context = "审计报告" in before and any(
            marker in before for marker in ("我们审计了", "已审财务报表", "财务报表")
        )
        note_context = any(
            marker in after
            for marker in ("财务报表附注", "财务报表注释", "公司基本情况")
        )
        if audit_context and note_context:
            candidates.append(run)
    if not candidates:
        return ()
    return min(candidates, key=lambda run: (run[0], len(run)))


def _normalize_ocr_number(value: str) -> Decimal | None:
    text = re.sub(r"\s+", "", value).translate(OCR_NUMBER_TRANSLATION_V1_6)
    text = text.replace("巧", "15")
    if re.fullmatch(r"\(?-?[\d,.]+\)?", text) is None:
        return None
    negative = text.startswith("(") or text.endswith(")")
    unsigned = text.strip("()").lstrip("-")
    if "." in unsigned:
        integer, fractional = unsigned.rsplit(".", 1)
        integer = integer.replace(",", "").replace(".", "")
        normalized = f"{integer}.{fractional.replace(',', '')}"
    else:
        pieces = unsigned.split(",")
        if len(pieces) >= 3 and len(pieces[-1]) == 2:
            normalized = f"{''.join(pieces[:-1])}.{pieces[-1]}"
        else:
            normalized = "".join(pieces)
    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    if value.strip().startswith("-") or negative:
        return -abs(amount)
    return amount


def _ocr_observations(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[OcrObservationV1_6]]:
    observations: dict[str, list[OcrObservationV1_6]] = {
        key: [] for key in OCR_ROW_PATTERNS_V1_6
    }
    for record in records:
        for row in _v1_3._ocr_rows(record):
            compact = str(row["compact"])
            for metric_id, pattern in OCR_ROW_PATTERNS_V1_6.items():
                if pattern.search(compact) is None:
                    continue
                if (
                    metric_id == "ACCOUNTS_RECEIVABLE_END"
                    and any(
                        marker in compact
                        for marker in ("应收票据及应收账款", "应收票据和应收账款")
                    )
                ):
                    continue
                numeric: list[tuple[str, Decimal]] = []
                for segment in row["segments"]:
                    raw = str(segment["text"])
                    parsed = _normalize_ocr_number(raw)
                    if parsed is not None:
                        numeric.append((raw, parsed))
                if (
                    len(numeric) >= 3
                    and numeric[0][1] == numeric[0][1].to_integral_value()
                    and abs(numeric[0][1]) <= 999
                ):
                    numeric = numeric[1:]
                if not numeric:
                    continue
                raw_value, parsed_value = numeric[0]
                observations[metric_id].append(
                    OcrObservationV1_6(
                        metric_id=metric_id,
                        page_index=int(row["page_index"]),
                        render_scale=float(row["render_scale"]),
                        raw_value=raw_value,
                        value=parsed_value,
                        row_text=compact[:1000],
                        pixel_width=int(row["pixel_width"]),
                        pixel_height=int(row["pixel_height"]),
                    )
                )
    for values in observations.values():
        values.sort(key=lambda item: (item.page_index, item.value, item.render_scale))
    return observations


def _agreed_pairs(
    by_scale: dict[float, dict[str, list[OcrObservationV1_6]]],
    metric_id: str,
) -> list[tuple[OcrObservationV1_6, OcrObservationV1_6]]:
    first_scale, second_scale = _v1_3.OCR_RENDER_SCALES
    result: list[tuple[OcrObservationV1_6, OcrObservationV1_6]] = []
    for first in by_scale[first_scale].get(metric_id) or []:
        for second in by_scale[second_scale].get(metric_id) or []:
            if first.page_index == second.page_index and first.value == second.value:
                result.append((first, second))
    return result


def _metric_cny(result: dict[str, Any], metric_id: str) -> Decimal | None:
    metric = _metric_map(result).get(metric_id)
    return Decimal(str(metric["metric_value_cny"])) if metric is not None else None


def _display_matches_cny(value: Decimal, expected_cny: Decimal, unit: str) -> bool:
    return _v1_4._same_display_identity(
        value * _base.UNIT_MULTIPLIERS[unit],
        expected_cny,
        unit,
    )


def _anchor_pair(
    by_scale: dict[float, dict[str, list[OcrObservationV1_6]]],
    metric_id: str,
    expected_cny: Decimal | None,
    unit: str,
) -> tuple[OcrObservationV1_6, OcrObservationV1_6] | None:
    if expected_cny is None:
        return None
    for pair in _agreed_pairs(by_scale, metric_id):
        if _display_matches_cny(pair[1].value, expected_cny, unit):
            return pair
    return None


def _pair_on_pages(
    pairs: Sequence[tuple[OcrObservationV1_6, OcrObservationV1_6]],
    pages: set[int],
) -> tuple[OcrObservationV1_6, OcrObservationV1_6] | None:
    return next((pair for pair in pairs if pair[1].page_index in pages), None)


def _ocr_evidence(
    metric_id: str,
    pair: tuple[OcrObservationV1_6, OcrObservationV1_6],
    *,
    unit: str,
    unit_source_page: int,
    validation: dict[str, Any],
    method_suffix: str = "DUAL_SCALE_EXACT",
) -> _base.MetricEvidence:
    first, second = pair
    scope = "PERIOD_END" if metric_id.endswith("_END") else "YEAR_TO_DATE"
    section = (
        "CONSOLIDATED_BALANCE_SHEET"
        if scope == "PERIOD_END"
        else "CONSOLIDATED_INCOME_STATEMENT"
    )
    locator = {
        "page": second.page_index + 1,
        "section": section,
        "parser_version": PARSER_VERSION,
        "ocr_engine": _v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": _v1_3.OCR_LANGUAGE_TAG,
        "render_scales": list(_v1_3.OCR_RENDER_SCALES),
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
        "validation": validation,
    }
    return _base.MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(second.value * _base.UNIT_MULTIPLIERS[unit]),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=scope,
        source_page=second.page_index + 1,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=second.row_text.split(second.raw_value, 1)[0],
        source_raw_value=second.raw_value,
        source_unit=unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
        source_method=f"WINDOWS_MEDIA_OCR_ANCHORED_IMAGE_STATEMENT_{method_suffix}",
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _single_scale_identity_liability_evidence(
    observation: OcrObservationV1_6,
    peer: OcrObservationV1_6,
    *,
    unit: str,
    unit_source_page: int,
    validation: dict[str, Any],
) -> _base.MetricEvidence:
    locator = {
        "page": observation.page_index + 1,
        "section": "CONSOLIDATED_BALANCE_SHEET",
        "parser_version": PARSER_VERSION,
        "ocr_engine": _v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": _v1_3.OCR_LANGUAGE_TAG,
        "render_scales": list(_v1_3.OCR_RENDER_SCALES),
        "accepted_scale": observation.render_scale,
        "accepted_raw_value": observation.raw_value,
        "accepted_normalized_value": str(observation.value),
        "peer_scale": peer.render_scale,
        "peer_raw_value": peer.raw_value,
        "peer_normalized_value": str(peer.value),
        "row_text_by_scale": {
            str(observation.render_scale): observation.row_text,
            str(peer.render_scale): peer.row_text,
        },
        "unit_source_page": unit_source_page,
        "validation": validation,
    }
    return _base.MetricEvidence(
        metric_id="TOTAL_LIABILITIES_END",
        metric_value_cny=float(observation.value * _base.UNIT_MULTIPLIERS[unit]),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope="PERIOD_END",
        source_page=observation.page_index + 1,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=observation.row_text.split(observation.raw_value, 1)[0],
        source_raw_value=observation.raw_value,
        source_unit=unit,
        source_unit_multiplier=float(_base.UNIT_MULTIPLIERS[unit]),
        source_method=(
            "WINDOWS_MEDIA_OCR_ANCHORED_IMAGE_STATEMENT_SINGLE_SCALE_EXACT_"
            "PLUS_DUAL_SCALE_BALANCE_IDENTITY"
        ),
        verification_status=ADMITTED_VERIFICATION_STATUS,
    )


def _add_anchored_image_ocr(
    result: dict[str, Any],
    content: bytes,
    page_texts: Sequence[str],
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
        return {"status": "NOT_NEEDED_NO_IMAGE_TARGET_METRIC_MISSING", "admitted_metric_ids": []}
    pages = find_image_statement_pages(page_texts)
    if not pages:
        return {"status": "NOT_APPLICABLE_NO_IMAGE_STATEMENT_RUN", "admitted_metric_ids": []}
    unit, unit_source_page = _v1_3._find_image_statement_unit(page_texts, pages)
    if unit not in _base.UNIT_MULTIPLIERS or unit_source_page is None:
        return {
            "status": "NO_VIEW_IMAGE_STATEMENT_UNIT_NOT_UNIQUE",
            "page_numbers": [page + 1 for page in pages],
            "admitted_metric_ids": [],
        }
    try:
        outputs, render_receipt = _v1_3._run_windows_ocr(content, pages)
    except Exception as error:  # noqa: BLE001 - OCR失败必须显式保留不完整状态
        return {
            "status": "NO_VIEW_WINDOWS_OCR_FAILED",
            "page_numbers": [page + 1 for page in pages],
            "error": f"{type(error).__name__}: {error}"[:3000],
            "admitted_metric_ids": [],
        }
    by_scale = {
        scale: _ocr_observations(records)
        for scale, records in outputs.items()
    }
    assets_anchor = _anchor_pair(
        by_scale,
        "TOTAL_ASSETS_END",
        _metric_cny(result, "TOTAL_ASSETS_END"),
        unit,
    )
    parent_anchor = _anchor_pair(
        by_scale,
        "PARENT_NET_PROFIT_YTD",
        _metric_cny(result, "PARENT_NET_PROFIT_YTD"),
        unit,
    )
    admitted: list[str] = []
    corrected_existing: list[str] = []
    validation_common = {
        "rule": "EXISTING_OFFICIAL_SUMMARY_VALUE_ANCHORS_CONSOLIDATED_IMAGE_STATEMENT_PAGE",
        "assets_anchor_page": assets_anchor[1].page_index + 1 if assets_anchor else None,
        "parent_profit_anchor_page": parent_anchor[1].page_index + 1 if parent_anchor else None,
        "dual_render_scale_exact_value_agreement_required_for_detail_rows": True,
    }

    if assets_anchor is not None:
        asset_page = assets_anchor[1].page_index
        balance_asset_pages = set(range(max(pages[0], asset_page - 7), asset_page + 1))
        balance_liability_pages = set(
            range(asset_page, min(pages[-1], asset_page + 4) + 1)
        )
        for metric_id in ("ACCOUNTS_RECEIVABLE_END", "INVENTORY_END"):
            if metric_id not in result.get("missing_metrics", []):
                continue
            pair = _pair_on_pages(_agreed_pairs(by_scale, metric_id), balance_asset_pages)
            if pair is None:
                continue
            validation = {
                **validation_common,
                "anchor_metric_id": "TOTAL_ASSETS_END",
                "anchor_metric_value_cny": str(_metric_cny(result, "TOTAL_ASSETS_END")),
                "anchor_dual_scale_match_passed": True,
                "detail_row_dual_scale_match_passed": True,
            }
            _v1_4._put_metric(
                result,
                _ocr_evidence(
                    metric_id,
                    pair,
                    unit=unit,
                    unit_source_page=unit_source_page,
                    validation=validation,
                ),
            )
            admitted.append(metric_id)

        if "TOTAL_LIABILITIES_END" in result.get("missing_metrics", []):
            liability_pair = _pair_on_pages(
                _agreed_pairs(by_scale, "TOTAL_LIABILITIES_END"),
                balance_liability_pages,
            )
            equity_pair = _pair_on_pages(
                _agreed_pairs(by_scale, "TOTAL_EQUITY_END"),
                balance_liability_pages,
            )
            total_pair = _pair_on_pages(
                _agreed_pairs(by_scale, "TOTAL_LIABILITIES_EQUITY_END"),
                balance_liability_pages,
            )
            assets_display = assets_anchor[1].value
            if (
                liability_pair is not None
                and equity_pair is not None
                and assets_display == liability_pair[1].value + equity_pair[1].value
                and (total_pair is None or total_pair[1].value == assets_display)
            ):
                validation = {
                    **validation_common,
                    "liability_row_dual_scale_match_passed": True,
                    "equity_row_dual_scale_match_passed": True,
                    "assets_equal_liabilities_plus_equity": True,
                    "assets_equal_liabilities_and_equity_total": (
                        total_pair is not None and total_pair[1].value == assets_display
                    ),
                }
                _v1_4._put_metric(
                    result,
                    _ocr_evidence(
                        "TOTAL_LIABILITIES_END",
                        liability_pair,
                        unit=unit,
                        unit_source_page=unit_source_page,
                        validation=validation,
                    ),
                )
                admitted.append("TOTAL_LIABILITIES_END")
            elif equity_pair is not None:
                expected = assets_display - equity_pair[1].value
                if expected >= 0 and (
                    total_pair is None or total_pair[1].value == assets_display
                ):
                    first_scale, second_scale = _v1_3.OCR_RENDER_SCALES
                    first_rows = [
                        row
                        for row in by_scale[first_scale]["TOTAL_LIABILITIES_END"]
                        if row.page_index in balance_liability_pages
                    ]
                    second_rows = [
                        row
                        for row in by_scale[second_scale]["TOTAL_LIABILITIES_END"]
                        if row.page_index in balance_liability_pages
                    ]
                    accepted: tuple[OcrObservationV1_6, OcrObservationV1_6] | None = None
                    for row in [*second_rows, *first_rows]:
                        if row.value != expected:
                            continue
                        peers = (
                            first_rows if row.render_scale == second_scale else second_rows
                        )
                        peer = next(
                            (candidate for candidate in peers if candidate.page_index == row.page_index),
                            None,
                        )
                        if peer is not None:
                            accepted = row, peer
                            break
                    if accepted is not None:
                        observation, peer = accepted
                        validation = {
                            **validation_common,
                            "liability_row_one_scale_exact_identity_value": True,
                            "liability_label_observed_at_both_scales_same_page": True,
                            "equity_row_dual_scale_match_passed": True,
                            "assets_equal_liabilities_plus_equity": True,
                            "assets_equal_liabilities_and_equity_total": (
                                total_pair is not None and total_pair[1].value == assets_display
                            ),
                            "derived_identity_value_used_only_to_validate_ocr_row": True,
                        }
                        _v1_4._put_metric(
                            result,
                            _single_scale_identity_liability_evidence(
                                observation,
                                peer,
                                unit=unit,
                                unit_source_page=unit_source_page,
                                validation=validation,
                            ),
                        )
                        admitted.append("TOTAL_LIABILITIES_END")

    if parent_anchor is not None:
        income_page = parent_anchor[1].page_index
        for metric_id in (
            "OPERATING_REVENUE_YTD",
            "OPERATING_PROFIT_YTD",
            "PARENT_NET_PROFIT_YTD",
        ):
            pair = _pair_on_pages(
                _agreed_pairs(by_scale, metric_id),
                {income_page},
            )
            if pair is None:
                continue
            existing_cny = _metric_cny(result, metric_id)
            observed_cny = pair[1].value * _base.UNIT_MULTIPLIERS[unit]
            if existing_cny is not None and _v1_4._same_display_identity(
                observed_cny,
                existing_cny,
                unit,
            ):
                continue
            validation = {
                **validation_common,
                "anchor_metric_id": "PARENT_NET_PROFIT_YTD",
                "anchor_metric_value_cny": str(_metric_cny(result, "PARENT_NET_PROFIT_YTD")),
                "anchor_dual_scale_match_passed": True,
                "target_row_dual_scale_match_passed": True,
                "same_consolidated_income_page_as_parent_profit_anchor": True,
                "replaced_existing_metric_value_cny": (
                    str(existing_cny) if existing_cny is not None else None
                ),
            }
            _v1_4._put_metric(
                result,
                _ocr_evidence(
                    metric_id,
                    pair,
                    unit=unit,
                    unit_source_page=unit_source_page,
                    validation=validation,
                ),
            )
            admitted.append(metric_id)
            if existing_cny is not None:
                corrected_existing.append(metric_id)

    render_receipt.update(
        {
            "status": (
                "PASS_ANCHORED_IMAGE_OCR_METRICS_ADMITTED"
                if admitted
                else "NO_VIEW_ANCHORED_IMAGE_OCR_NO_METRIC_ADMITTED"
            ),
            "unit": unit,
            "unit_source_page": unit_source_page,
            "assets_anchor_page": assets_anchor[1].page_index + 1 if assets_anchor else None,
            "parent_profit_anchor_page": parent_anchor[1].page_index + 1 if parent_anchor else None,
            "admitted_metric_ids": sorted(admitted),
            "corrected_existing_metric_ids": sorted(corrected_existing),
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
    result = _v1_5.extract_metrics_from_page_texts(
        normalized,
        period_type=period_type,
        text_engine=text_engine,
    )
    result["v1_6_inverse_core_label_receipt"] = _add_inverse_core_parent_profit(
        result,
        normalized,
        period_type=period_type,
    )
    result["v1_6_net_receivable_reconciliation_receipt"] = (
        _add_net_receivable_reconciliation(result, normalized)
    )
    _restamp_result(result)
    return result


def extract_official_pdf_facts(
    content: bytes,
    *,
    period_type: str | None = None,
) -> dict[str, Any]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("响应没有%PDF-文件头")

    original_image_finder = _v1_3._find_statement_image_pages
    _v1_3._find_statement_image_pages = lambda _page_texts: ()
    try:
        result = _v1_5.extract_official_pdf_facts(
            content,
            period_type=period_type,
        )
    finally:
        _v1_3._find_statement_image_pages = original_image_finder

    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    normalized = [_v1_4.normalize_financial_text(text) for text in page_texts]
    result["v1_6_inverse_core_label_receipt"] = _add_inverse_core_parent_profit(
        result,
        normalized,
        period_type=period_type,
    )
    result["v1_6_net_receivable_reconciliation_receipt"] = (
        _add_net_receivable_reconciliation(result, normalized)
    )
    result["v1_6_anchored_image_ocr_receipt"] = _add_anchored_image_ocr(
        result,
        content,
        normalized,
    )
    result["v1_6_text_reextraction_receipt"] = {
        "status": "PASS_REEXTRACTED_ONLY_AFTER_FROZEN_V1_5_RESULT",
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
    }
    _restamp_result(result)
    return result


__all__ = [
    "ADMITTED_VERIFICATION_STATUS",
    "CORE_PARENT_INVERSE_LABEL_PATTERNS",
    "PARSER_VERSION",
    "REQUIRED_METRICS",
    "SectionRange",
    "extract_metrics_from_page_texts",
    "extract_official_pdf_facts",
    "find_image_statement_pages",
]
