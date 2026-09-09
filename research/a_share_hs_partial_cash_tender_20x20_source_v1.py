from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

import pandas as pd


PROTOCOL_ID = "A_SHARE_HS_OFFICIAL_PARTIAL_CASH_TENDER_20X20_V1"


class PartialTenderSourceError(ValueError):
    """正式部分现金要约来源或解析结果违反冻结合同。"""


RATIO_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "APPENDIX_QUANTITY_AND_RATIO",
        re.compile(
            r"预定收购股份数量.{0,180}?比例[：:]?"
            r"(?P<ratio>[0-9]+(?:\.[0-9]+)?)%"
        ),
    ),
    (
        "NUMBERED_QUANTITY_TOTAL_SHARE_RATIO",
        re.compile(
            r"预定收购(?:的)?股份数量[：:]?[0-9,]+股.{0,140}?"
            r"占(?:被收购公司已发行股份|被收购公司总股本|上市公司总股本|"
            r"公司总股本|总股本).{0,40}?"
            r"(?P<ratio>[0-9]+(?:\.[0-9]+)?)%"
        ),
    ),
    (
        "NARRATIVE_OFFER_QUANTITY_TOTAL_SHARE_RATIO",
        re.compile(
            r"(?:本次要约收购.{0,80}?)?收购股份数量(?:为|：|:)"
            r"[0-9,]+股.{0,100}?占.{0,40}?总股本.{0,30}?"
            r"(?P<ratio>[0-9]+(?:\.[0-9]+)?)%"
        ),
    ),
    (
        "OFFER_PLAN_QUANTITY_RATIO",
        re.compile(
            r"要约收购的股份情况.{0,240}?预定收购.{0,100}?"
            r"(?P<ratio>[0-9]+(?:\.[0-9]+)?)%"
        ),
    ),
    (
        "DISCLOSURE_FORM_QUANTITY_RATIO",
        re.compile(
            r"预定收购股份.{0,80}?数量[：:]?[0-9,]+股.{0,80}?"
            r"比例[：:]?(?P<ratio>[0-9]+(?:\.[0-9]+)?)%"
        ),
    ),
)

MINIMUM_ACCEPTANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"本次要约收购的生效条件(?:为|如下)"),
    re.compile(r"预受要约.{0,140}?(?:不低于|不少于|至少)[0-9,]+股"),
    re.compile(r"最低(?:预受|接受|申报).{0,100}?(?:数量|比例)"),
    re.compile(r"预受要约股份.{0,160}?未达到.{0,80}?自始不生效"),
    re.compile(r"未达到.{0,100}?生效条件.{0,100}?要约收购(?:不生效|自始不生效)"),
)

REMAINING_CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"本次要约收购尚需(?:取得|履行|通过)"),
    re.compile(r"尚需获得.{0,100}?(?:批准|核准|同意)"),
    re.compile(r"尚待.{0,100}?(?:批准|核准|同意|完成)"),
    re.compile(r"以.{0,100}?(?:获得|取得).{0,80}?(?:批准|核准|同意)为前提"),
)

CASH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"以现金(?:方式)?支付"),
    re.compile(r"支付方式[：:]?现金支付"),
    re.compile(r"现金对价"),
)

GUARANTEE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"履约保证金"),
    re.compile(r"履约保证"),
    re.compile(r"银行保函"),
)

NO_OTHER_CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"无其他约定条件"),
    re.compile(r"无其他生效条件"),
    re.compile(r"本次要约收购不附带任何条件"),
)

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "TERMINATION": ("终止要约收购", "撤回要约", "要约收购失败", "自始不生效"),
    "COMPLETION": ("完成交割", "交割完成", "清算过户", "股份过户完成", "收购完成"),
    "RESULT": ("要约收购结果", "收购公司股份结果", "期限届满", "股份结果"),
    "FORMAL_REPORT": ("要约收购报告书",),
}


def normalize_document_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<<<PAGE_[0-9]+>>>", "", text)
    return re.sub(r"\s+", "", text)


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _json_array(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise PartialTenderSourceError(f"字段不是合法JSON数组：{value}") from error
    if not isinstance(parsed, list):
        raise PartialTenderSourceError(f"字段不是JSON数组：{value}")
    return parsed


def _positive_decimal(value: Any) -> float | None:
    try:
        parsed = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return float(parsed)


def _matching_evidence(
    text: str, patterns: Iterable[re.Pattern[str]]
) -> list[str]:
    evidence: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            item = match.group(0)[:500]
            if item not in evidence:
                evidence.append(item)
    return evidence


def extract_offer_ratio_candidates(value: Any) -> list[dict[str, Any]]:
    text = normalize_document_text(value)
    by_ratio: dict[float, dict[str, Any]] = {}
    for pattern_id, pattern in RATIO_PATTERNS:
        for match in pattern.finditer(text):
            parsed = _positive_decimal(match.group("ratio"))
            if parsed is None or parsed >= 100.0:
                continue
            candidate = by_ratio.setdefault(
                parsed,
                {
                    "ratio_percent": parsed,
                    "pattern_ids": [],
                    "evidence": [],
                },
            )
            if pattern_id not in candidate["pattern_ids"]:
                candidate["pattern_ids"].append(pattern_id)
            snippet = match.group(0)[:500]
            if snippet not in candidate["evidence"]:
                candidate["evidence"].append(snippet)
    return [by_ratio[key] for key in sorted(by_ratio)]


def extract_offer_price_candidates(value: Any) -> list[float]:
    candidates: list[float] = []
    for item in _json_array(value):
        parsed = _positive_decimal(item)
        if parsed is not None and parsed not in candidates:
            candidates.append(parsed)
    return sorted(candidates)


def _parse_cn_date(value: Any) -> str | None:
    match = re.fullmatch(
        r"\s*([12][0-9]{3})\s*年\s*([0-9]{1,2})\s*月\s*([0-9]{1,2})\s*日\s*",
        str(value or ""),
    )
    if match is None:
        return None
    year, month, day = (int(item) for item in match.groups())
    try:
        return pd.Timestamp(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_offer_period_candidates(value: Any) -> list[list[str]]:
    periods: list[list[str]] = []
    for item in _json_array(value):
        if not isinstance(item, list) or len(item) != 2:
            continue
        start = _parse_cn_date(item[0])
        end = _parse_cn_date(item[1])
        if start is None or end is None or start > end:
            continue
        candidate = [start, end]
        if candidate not in periods:
            periods.append(candidate)
    return sorted(periods)


def classify_lifecycle_roles(title: Any) -> list[str]:
    normalized = normalize_title(title)
    roles: list[str] = []
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            roles.append(role)
    return roles or ["OTHER_OFFER_RELATED"]


def validate_source_inputs(
    text_review: pd.DataFrame,
    adjudication: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    expected_rows = {
        "formal_report_text_review": 111,
        "formal_report_adjudication_ledger": 111,
        "cninfo_metadata_archive": 680,
    }
    actual_rows = {
        "formal_report_text_review": len(text_review),
        "formal_report_adjudication_ledger": len(adjudication),
        "cninfo_metadata_archive": len(metadata),
    }
    if actual_rows != expected_rows:
        raise PartialTenderSourceError(
            f"官方来源行数漂移：{actual_rows}!={expected_rows}"
        )
    required_text = {
        "announcement_id",
        "ts_code",
        "official_pdf_date",
        "extracted_text",
        "offer_price_candidates_json",
        "offer_period_candidates_json",
        "pdf_path",
        "pdf_sha256",
    }
    required_adjudication = {
        "announcement_id",
        "ts_code",
        "final_decision",
        "reason_code",
        "market_price_read",
        "future_return_read",
        "tender_outcome_read",
    }
    required_metadata = {
        "announcement_id",
        "ts_code",
        "official_pdf_date",
        "announcement_title",
        "pdf_path",
        "pdf_sha256",
        "status",
    }
    for name, frame, required in (
        ("formal_report_text_review", text_review, required_text),
        ("formal_report_adjudication_ledger", adjudication, required_adjudication),
        ("cninfo_metadata_archive", metadata, required_metadata),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise PartialTenderSourceError(f"{name}缺少字段：{missing}")
    partial = adjudication[
        adjudication["reason_code"].astype(str).str.contains("PARTIAL", regex=False)
    ].copy()
    if len(partial) != 76 or partial["ts_code"].nunique() != 65:
        raise PartialTenderSourceError(
            f"部分要约来源池漂移：rows={len(partial)},codes={partial['ts_code'].nunique()}"
        )
    years = pd.to_datetime(partial["official_pdf_date"], errors="coerce").dt.year
    if years.isna().any() or years.nunique() != 10:
        raise PartialTenderSourceError("部分要约来源年份覆盖漂移")
    for column in ("market_price_read", "future_return_read", "tender_outcome_read"):
        if partial[column].fillna(False).astype(bool).any():
            raise PartialTenderSourceError(f"部分要约来源池提前读取禁止字段：{column}")
    return {
        "status": "PASS_PARTIAL_TENDER_SOURCE_INPUTS_VALIDATED",
        "row_counts": actual_rows,
        "partial_formal_report_rows": len(partial),
        "partial_unique_security_count": partial["ts_code"].nunique(),
        "partial_calendar_year_count": years.nunique(),
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
    }


def build_machine_source_ledger(
    text_review: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> pd.DataFrame:
    left = adjudication.copy()
    right = text_review.copy()
    left["_announcement_id"] = left["announcement_id"].astype(str)
    right["_announcement_id"] = right["announcement_id"].astype(str)
    partial = left[
        left["reason_code"].astype(str).str.contains("PARTIAL", regex=False)
    ].copy()
    merged = partial.merge(
        right,
        on="_announcement_id",
        how="left",
        suffixes=("_adjudication", "_text"),
        validate="one_to_one",
    )
    if len(merged) != 76 or merged["extracted_text"].isna().any():
        raise PartialTenderSourceError("部分要约正式报告与文本提取未一一匹配")
    records: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        text = normalize_document_text(row["extracted_text"])
        ratio_candidates = extract_offer_ratio_candidates(text)
        price_candidates = extract_offer_price_candidates(
            row["offer_price_candidates_json"]
        )
        period_candidates = extract_offer_period_candidates(
            row["offer_period_candidates_json"]
        )
        minimum_evidence = _matching_evidence(text, MINIMUM_ACCEPTANCE_PATTERNS)
        remaining_evidence = _matching_evidence(text, REMAINING_CONDITION_PATTERNS)
        cash_evidence = _matching_evidence(text, CASH_PATTERNS)
        guarantee_evidence = _matching_evidence(text, GUARANTEE_PATTERNS)
        no_condition_evidence = _matching_evidence(
            text, NO_OTHER_CONDITION_PATTERNS
        )
        resolved_ratio = (
            float(ratio_candidates[0]["ratio_percent"]) / 100.0
            if len(ratio_candidates) == 1
            else None
        )
        resolved_price = price_candidates[0] if len(price_candidates) == 1 else None
        resolved_period = period_candidates[0] if len(period_candidates) == 1 else None
        machine_exclusion_reasons: list[str] = []
        if resolved_ratio is not None and resolved_ratio < 0.20:
            machine_exclusion_reasons.append("OFFER_RATIO_BELOW_20_PERCENT")
        if minimum_evidence:
            machine_exclusion_reasons.append("MINIMUM_ACCEPTANCE_CONDITION_DETECTED")
        if remaining_evidence:
            machine_exclusion_reasons.append("REMAINING_MATERIAL_CONDITION_DETECTED")
        if not cash_evidence:
            machine_exclusion_reasons.append("CASH_ONLY_NOT_MACHINE_CONFIRMED")
        if not guarantee_evidence:
            machine_exclusion_reasons.append("PERFORMANCE_GUARANTEE_NOT_MACHINE_CONFIRMED")
        ambiguity_reasons: list[str] = []
        if len(ratio_candidates) != 1:
            ambiguity_reasons.append("OFFER_RATIO_REQUIRES_MANUAL_ADJUDICATION")
        if len(price_candidates) != 1:
            ambiguity_reasons.append("OFFER_PRICE_REQUIRES_MANUAL_ADJUDICATION")
        if len(period_candidates) != 1:
            ambiguity_reasons.append("OFFER_PERIOD_REQUIRES_MANUAL_ADJUDICATION")
        if machine_exclusion_reasons:
            machine_status = "MACHINE_EXCLUSION_OR_MANUAL_OVERRIDE_REVIEW_REQUIRED"
        elif ambiguity_reasons:
            machine_status = "MANUAL_SOURCE_ADJUDICATION_REQUIRED"
        elif resolved_ratio is not None and resolved_ratio >= 0.20:
            machine_status = "POTENTIAL_20X20_SOURCE_EVENT_MANUAL_CONFIRMATION_REQUIRED"
        else:
            machine_status = "MACHINE_BELOW_20_PERCENT_SOURCE_EXCLUSION"
        records.append(
            {
                "announcement_id": str(row["announcement_id_adjudication"]),
                "ts_code": str(row["ts_code_adjudication"]),
                "sec_name": row.get("sec_name_adjudication"),
                "official_pdf_date": str(row["official_pdf_date_adjudication"]),
                "announcement_title": row.get("announcement_title_adjudication"),
                "pdf_path": row.get("pdf_path_adjudication"),
                "pdf_sha256": row.get("pdf_sha256_adjudication"),
                "prior_reason_code": row.get("reason_code"),
                "ratio_candidates_json": json.dumps(
                    ratio_candidates, ensure_ascii=False, separators=(",", ":")
                ),
                "ratio_candidate_count": len(ratio_candidates),
                "machine_offer_quantity_ratio": resolved_ratio,
                "price_candidates_json": json.dumps(
                    price_candidates, ensure_ascii=False, separators=(",", ":")
                ),
                "price_candidate_count": len(price_candidates),
                "machine_offer_price_cny": resolved_price,
                "period_candidates_json": json.dumps(
                    period_candidates, ensure_ascii=False, separators=(",", ":")
                ),
                "period_candidate_count": len(period_candidates),
                "machine_offer_start_date": (
                    resolved_period[0] if resolved_period is not None else None
                ),
                "machine_offer_end_date": (
                    resolved_period[1] if resolved_period is not None else None
                ),
                "minimum_acceptance_condition_detected": bool(minimum_evidence),
                "minimum_acceptance_evidence_json": json.dumps(
                    minimum_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "remaining_material_condition_detected": bool(remaining_evidence),
                "remaining_material_condition_evidence_json": json.dumps(
                    remaining_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "cash_only_machine_confirmed": bool(cash_evidence),
                "cash_evidence_json": json.dumps(
                    cash_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "performance_guarantee_machine_confirmed": bool(guarantee_evidence),
                "performance_guarantee_evidence_json": json.dumps(
                    guarantee_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "no_other_condition_text_detected": bool(no_condition_evidence),
                "no_other_condition_evidence_json": json.dumps(
                    no_condition_evidence, ensure_ascii=False, separators=(",", ":")
                ),
                "machine_exclusion_reasons_json": json.dumps(
                    machine_exclusion_reasons,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "ambiguity_reasons_json": json.dumps(
                    ambiguity_reasons, ensure_ascii=False, separators=(",", ":")
                ),
                "machine_status": machine_status,
                "manual_review_priority": (
                    "HIGH"
                    if resolved_ratio is None or resolved_ratio >= 0.20
                    else "LOW"
                ),
                "market_price_read": False,
                "future_return_read": False,
                "tender_outcome_read": False,
            }
        )
    result = pd.DataFrame(records).sort_values(
        ["official_pdf_date", "ts_code", "announcement_id"], kind="mergesort"
    )
    return result.reset_index(drop=True)


def build_lifecycle_candidate_ledger(
    metadata: pd.DataFrame,
    machine_source: pd.DataFrame,
) -> pd.DataFrame:
    first_dates = (
        machine_source.groupby("ts_code", as_index=False)["official_pdf_date"]
        .min()
        .rename(columns={"official_pdf_date": "first_partial_formal_pdf_date"})
    )
    candidates = metadata.copy()
    candidates["announcement_id"] = candidates["announcement_id"].astype(str)
    candidates["official_pdf_date"] = pd.to_datetime(
        candidates["official_pdf_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    candidates = candidates.merge(first_dates, on="ts_code", how="inner")
    candidates = candidates[
        candidates["official_pdf_date"] >= candidates["first_partial_formal_pdf_date"]
    ].copy()
    candidates["candidate_roles_json"] = candidates["announcement_title"].map(
        lambda item: json.dumps(
            classify_lifecycle_roles(item),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    candidates["source_status"] = candidates["status"]
    candidates["market_price_read"] = False
    candidates["future_return_read"] = False
    candidates["tender_outcome_read"] = False
    columns = [
        "announcement_id",
        "ts_code",
        "sec_name",
        "official_pdf_date",
        "announcement_title",
        "candidate_roles_json",
        "pdf_path",
        "pdf_sha256",
        "source_status",
        "first_partial_formal_pdf_date",
        "market_price_read",
        "future_return_read",
        "tender_outcome_read",
    ]
    return candidates[columns].sort_values(
        ["ts_code", "official_pdf_date", "announcement_id"], kind="mergesort"
    ).reset_index(drop=True)


def canonical_frame_sha256_payload(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.where(pd.notna(normalized), None)
    records = normalized.to_dict("records")
    return json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
