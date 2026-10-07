"""提取回购执行披露的金额及原文证据；不确定的方案或金额保持待核实。"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_index_documents_v1 as sources

OUT = ROOT / "reports/research/510300_corporate_repurchase_execution_extractor_v1"
STUDY = "510300_CORPORATE_REPURCHASE_EXECUTION_EXTRACTOR_V1"
DATE = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"
NUMBER = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
LABEL = r"(?:累计(?:已)?回购金额|实际回购金额|回购(?:资金)?总金额|成交总金额|成交金额|成交总额|(?:支付|使用)(?:的)?(?:资金)?(?:总)?金额|(?:支付|使用)(?:的)?(?:资金)?总额)"
NOTE = r"(?:[（(][^()（）]{0,25}[）)])?"
AMOUNT = re.compile(LABEL + NOTE + r"(?:为|约为|合计|共计|达|约|共|计|：|:)?" + NOTE + r"(?:人民币|RMB|￥)?" + NUMBER + r"(亿元|万元|元)")


def normalized(text):
    compact = re.sub(r"\s+", "", text).replace("，", ",").replace("Ａ", "A").replace("Ｈ", "H")
    # 页脚的固定责任声明有时插入金额中间；只移除该明确语句，原始文本不变。
    return re.sub(r"(?:本公司|公司)(?:及董事会全体成员|及全体董事)?保证.{0,140}?重大遗漏。?", "", compact)


def page_text(pages):
    cleaned = []
    for number, page in enumerate(pages, 1):
        lines = page.splitlines()
        nonempty = [i for i, value in enumerate(lines) if value.strip()]
        for index in (nonempty[:1] + nonempty[-1:]):
            value = re.sub(r"\s+", "", lines[index])
            if value in [str(number), f"{number}/{len(pages)}"]:
                lines[index] = ""
        cleaned.append("\n".join(lines))
    return normalized("\n".join(cleaned))


def money(number, unit):
    clean = number.replace(",", "")
    exponent = {"元": 2, "万元": 6, "亿元": 10}[unit]
    value = Decimal(clean) * (10 ** exponent)
    if value != value.to_integral_value():
        raise ValueError("披露金额精度小于分，不能默默舍入")
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return int(value), max(1, 10 ** (exponent - decimals))


def calendar(parts):
    return pd.Timestamp(*map(int, parts))


def meeting_anchors(text):
    anchors = []
    for match in re.finditer(DATE, text[:1800]):
        following = text[match.end():match.end() + 420]
        preceding = text[max(0, match.start() - 130):match.start()]
        # 并列两个日期对应董事会和股东会时，董事会取第一个，不能误取较近的股东会日。
        if re.search(DATE + r"[,、及和]$", preceding):
            continue
        opening = re.match(r"(?:[,、及和](?:20\d{2}年)?\d{1,2}月\d{1,2}日)*,?(?:分别)?召开(?:的|了)?.{0,45}?董事会", following)
        if not opening:
            continue
        proposal = re.search(r"审议(?:并)?通过(?:了)?《([^》]+)》", following)
        preceding_titles = re.findall(r"《([^》]+)》", preceding)
        bare_proposal = re.search(r"审议(?:并)?通过(?:了)?([^。]{1,45})", following)
        related = ("回购" in proposal.group(1)) if proposal else (
            (bare_proposal is not None and "回购" in bare_proposal.group(1)) or
            ("审议通过" in following[:150] and bool(preceding_titles) and "回购" in preceding_titles[-1]))
        if not related:
            continue
        anchors.append({"date": calendar(match.groups()), "evidence": text[match.start():match.end() + 210]})
    unique = {row["date"]: row for row in anchors}
    return sorted(unique.values(), key=lambda row: row["date"])


def budget_terms(text):
    """识别金额边界作为方案身份的补充；每股价格不属于回购预算。"""
    expression = re.compile(r"(不低于|不少于|不超过|不高于|不多于)(?:人民币)?" + NUMBER + r"(亿元|万元|元)")
    terms = []
    for match in expression.finditer(text[:2200]):
        if re.match(r"(?:[（(]含[^）)]*[）)])?[/／]股", text[match.end():match.end() + 18]):
            continue
        before = text[max(0, match.start() - 90):match.start()]
        if match.group(3) == "元" and "价格" in before[-15:]:
            continue
        value, precision = money(match.group(2), match.group(3))
        terms.append({"side": "FLOOR" if match.group(1) in ["不低于", "不少于"] else "CAP",
            "cents": value, "reporting_resolution_cents": precision,
            "evidence": before + match.group(0) + text[match.end():match.end() + 20]})
    floors = {item["cents"] for item in terms if item["side"] == "FLOOR"}
    caps = {item["cents"] for item in terms if item["side"] == "CAP"}
    unique = len(floors) == len(caps) == 1 and next(iter(floors)) <= next(iter(caps))
    return {"evidence": terms, "unambiguous_pair": unique,
        "floor_cents": next(iter(floors)) if unique else None,
        "cap_cents": next(iter(caps)) if unique else None}


def amount_candidates(text, first_execution):
    candidates = []
    for match in AMOUNT.finditer(text):
        sentence_start = max(text.rfind("。", 0, match.start()), text.rfind("；", 0, match.start())) + 1
        prefix = text[sentence_start:match.start()]
        context = text[max(sentence_start, match.start() - 550):match.end() + 40]
        value, precision = money(match.group(1), match.group(2))
        cumulative = max(prefix.rfind("累计"), prefix.rfind("实际回购"))
        first = prefix.rfind("首次")
        monthly_matches = list(re.finditer(r"本月|当月|\d{1,2}月份|20\d{2}年\d{1,2}月,", prefix))
        monthly = monthly_matches[-1].start() if monthly_matches else -1
        dated_interval = re.search(DATE + r"[-—至](?:20\d{2}年)?\d{1,2}月\d{1,2}日", prefix)
        label = match.group(0)
        if label.startswith(("累计", "实际回购")):
            role = "EXPLICIT_CUMULATIVE_LABEL"
        elif (monthly >= 0 and monthly > prefix.rfind("截至")) or (dated_interval and dated_interval.start() > prefix.rfind("截至")):
            role = "MONTHLY_PERIOD_NOT_SCHEME_CUMULATIVE"
        elif re.search(r"不低于|不超过|预计|拟(?:使用|用于)|计划(?:使用|用于)", prefix[-160:]):
            role = "PLAN_OR_LIMIT_NOT_ACTUAL"
        elif cumulative >= 0 and cumulative >= max(first, monthly) and len(prefix) - cumulative <= 550:
            role = "CUMULATIVE_EXECUTION_CONTEXT"
        elif first_execution and first >= 0 and first >= monthly:
            role = "FIRST_EXECUTION_CONTEXT"
        elif re.search(r"截至" + DATE, prefix) and "回购" in prefix and monthly < 0:
            role = "AS_OF_EXECUTION_CONTEXT"
        elif first_execution and "回购" in text[max(0, match.start() - 400):match.start()] and monthly < 0:
            role = "FIRST_EXECUTION_TITLE_AND_NEARBY_CONTEXT"
        else:
            role = "UNRESOLVED_AMOUNT_CONTEXT"
        cutoff = None
        date_matches = list(re.finditer(r"截至" + DATE, prefix))
        if date_matches:
            cutoff = calendar(date_matches[-1].groups())
        elif re.search(r"截至(20\d{2})年(\d{1,2})月底", prefix):
            match_end = list(re.finditer(r"截至(20\d{2})年(\d{1,2})月底", prefix))[-1]
            cutoff = pd.Timestamp(int(match_end.group(1)), int(match_end.group(2)), 1) + pd.offsets.MonthEnd(0)
        elif first_execution and role.startswith("FIRST_EXECUTION"):
            dates = list(re.finditer(DATE, text[max(0, match.start() - 450):match.start()]))
            if dates:
                cutoff = calendar(dates[-1].groups())
        candidates.append({"reported_cents": value, "reporting_resolution_cents": precision,
            "label": label, "role": role, "economic_cutoff_candidate": cutoff,
            "text_start": match.start(), "text_end": match.end(), "evidence": context})
    return candidates


def parse(row):
    record = {k: row.get(k) for k in ["symbol", "document_id", "title", "source_url", "raw_path", "raw_sha256",
        "text_path", "catalogue_date", "title_category", "reference_snapshot", "in_latest_prior_saved_snapshot",
        "reference_weight_percent", "after_last_saved_snapshot"]}
    record.update(status="NOT_PARSED", scheme=None, cumulative_cents=None, known_at=None,
        economic_cutoff=None, amount_candidates=[], anchor_candidates=[])
    if row["status"] != "PDF_TEXT_SAVED":
        record["status"] = "NO_VIEW_MISSING_SOURCE_TEXT"
        return record
    text = page_text(read(ROOT / row["text_path"]))
    title = normalized(row["title"])
    if any(word in title for word in ["子公司", "联营公司", "合营公司", "补偿股份", "股份补偿", "美元债券", "股票质押"]):
        record["status"] = "EXCLUDED_OTHER_ENTITY_OR_NON_OPEN_MARKET_REPURCHASE"
        return record
    if "H股" in title and "A股" not in title:
        record["status"] = "EXCLUDED_H_SHARE_DOCUMENT"
        return record
    if "注销完成" in title and not any(v in title for v in ["实施结果", "回购结果"]):
        record["status"] = "EXCLUDED_CANCELLATION_ONLY_DOCUMENT"
        return record
    if row["title_category"] != "EXECUTION_DISCLOSURE_CANDIDATE":
        record["status"] = "PLAN_OR_OTHER_DOCUMENT_NO_ACTUAL_AMOUNT"
        return record
    if row["symbol"][:6] not in text[:1000]:
        record["status"] = "NO_VIEW_ISSUER_CODE_NOT_CONFIRMED"
        return record
    first = "首次" in title
    anchors = meeting_anchors(text)
    record["anchor_candidates"] = anchors
    record["budget_terms"] = budget_terms(text)
    if len(anchors) == 1:
        record["scheme"] = row["symbol"] + "_BOARD_" + anchors[0]["date"].strftime("%Y%m%d")
    record["classification"] = "FIRST_EXECUTION" if first else ("COMPLETION" if "结果" in title or "完成" in title or "实施完" in title else "PROGRESS")
    signed = list(re.finditer(DATE, text[-260:]))
    day = pd.Timestamp(row["catalogue_date"])
    if signed:
        day = max(day, calendar(signed[-1].groups()))
    record["known_at"] = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    included_prior = list(re.finditer(r"(?:含|包括)公司于" + DATE + r".{0,100}?期间回购", text))
    if len(anchors) == 1 and any(calendar(m.groups()) < anchors[0]["date"] for m in included_prior):
        record["status"] = "NO_VIEW_CUMULATIVE_INCLUDES_PRIOR_SCHEME"
        record["prior_scheme_evidence"] = [m.group(0) for m in included_prior]
        return record
    candidates = amount_candidates(text, first)
    record["amount_candidates"] = candidates
    accepted = [c for c in candidates if c["role"] in ["EXPLICIT_CUMULATIVE_LABEL", "CUMULATIVE_EXECUTION_CONTEXT", "FIRST_EXECUTION_CONTEXT", "AS_OF_EXECUTION_CONTEXT", "FIRST_EXECUTION_TITLE_AND_NEARBY_CONTEXT"]]
    explicit_zero = list(re.finditer(r"(?:尚未|暂未|未)(?:(?:实施|开始|通过|开展|进行).{0,80}?回购|回购)", text))
    zero_evidence = []
    for match in explicit_zero:
        beginning = text.rfind("。", 0, match.start()) + 1
        segment = text[beginning:match.end() + 15]
        dates = list(re.finditer(r"截至" + DATE, segment))
        if dates:
            zero_evidence.append({"economic_cutoff": calendar(dates[-1].groups()), "evidence": segment})
        elif "截至本公告披露日" in segment:
            zero_evidence.append({"economic_cutoff": day, "evidence": segment, "precision": "AS_OF_ANNOUNCEMENT_NOT_LAST_BUY_DAY"})
    dates = [c["economic_cutoff_candidate"] for c in accepted if c["economic_cutoff_candidate"] is not None]
    dates += [c["economic_cutoff"] for c in zero_evidence]
    latest_cutoff = max(dates) if dates else None
    if latest_cutoff is not None:
        record["earlier_dated_amount_evidence"] = [c for c in accepted if c["economic_cutoff_candidate"] is not None and c["economic_cutoff_candidate"] < latest_cutoff]
        record["earlier_dated_zero_evidence"] = [c for c in zero_evidence if c["economic_cutoff"] < latest_cutoff]
        accepted = [c for c in accepted if c["economic_cutoff_candidate"] is None or c["economic_cutoff_candidate"] == latest_cutoff]
        zero_evidence = [c for c in zero_evidence if c["economic_cutoff"] == latest_cutoff]
    finest = min((c["reporting_resolution_cents"] for c in accepted), default=None)
    precise = [c for c in accepted if c["reporting_resolution_cents"] == finest]
    precise_values = {c["reported_cents"] for c in precise}
    compatible = len(precise_values) == 1 and all(
        abs(c["reported_cents"] - precise[0]["reported_cents"]) * 2 <= c["reporting_resolution_cents"] for c in accepted)
    if zero_evidence and not accepted:
        record.update(cumulative_cents=0, reporting_resolution_cents=1,
            economic_cutoff=zero_evidence[-1]["economic_cutoff"], amount_evidence=zero_evidence,
            amount_method="EXPLICIT_NOT_STARTED")
    elif compatible and not zero_evidence:
        chosen = precise[-1]
        record.update(cumulative_cents=chosen["reported_cents"], reporting_resolution_cents=finest,
            economic_cutoff=chosen["economic_cutoff_candidate"] if chosen["economic_cutoff_candidate"] is not None else latest_cutoff,
            amount_evidence=accepted, amount_method="DIRECT_PRECISE_VALUE_WITH_COMPATIBLE_CUMULATIVE_EVIDENCE",
            coarse_summary_precision_reconciled=len({c["reported_cents"] for c in accepted}) > 1)
    else:
        record["status"] = "NO_VIEW_AMBIGUOUS_AMOUNT" if accepted or zero_evidence else "NO_VIEW_ACTUAL_AMOUNT_NOT_IDENTIFIED"
        return record
    if record["economic_cutoff"] is not None and record["economic_cutoff"] > day:
        record["status"] = "NO_VIEW_ECONOMIC_DATE_AFTER_PUBLICATION"
        return record
    if len(anchors) == 1 and record["economic_cutoff"] is not None and record["economic_cutoff"] < anchors[0]["date"]:
        record["status"] = "NO_VIEW_SOURCE_ECONOMIC_DATE_BEFORE_APPROVAL"
        return record
    terms = record["budget_terms"]
    record["within_reported_plan_cap"] = record["cumulative_cents"] <= terms["cap_cents"] if terms["unambiguous_pair"] else None
    record["scheme_candidate_with_budget"] = (
        record["scheme"] + f"_CNYCENTS_{terms['floor_cents']}_{terms['cap_cents']}"
        if record["scheme"] and terms["unambiguous_pair"] else None)
    record["status"] = "EXTRACTED_AMOUNT_WITH_BOARD_ANCHOR" if record["scheme"] else "EXTRACTED_AMOUNT_SCHEME_UNRESOLVED"
    record["same_board_date_may_contain_multiple_schemes"] = True
    record["admitted_to_trading_feature"] = False
    return record


def checks():
    assert money("1,234.56", "元") == (123456, 1)
    assert money("123.45", "万元") == (123450000, 10000)
    text = normalized("计划不超过10亿元。2025年2月份公司支付的总金额为50万元。截至2025年2月28日，公司累计回购100股，支付的总金额为200万元。")
    found = amount_candidates(text, False)
    assert found[0]["role"] == "MONTHLY_PERIOD_NOT_SCHEME_CUMULATIVE"
    assert found[1]["role"] == "CUMULATIVE_EXECUTION_CONTEXT" and found[1]["reported_cents"] == 200000000
    assert found[1]["economic_cutoff_candidate"] == pd.Timestamp("2025-02-28")
    anchors = meeting_anchors("公司于2025年4月8日召开第五届董事会第七次会议，审议通过回购公司股份的议案。")
    assert anchors[0]["date"] == pd.Timestamp("2025-04-08")
    assert not meeting_anchors("截至2025年4月8日公司已回购100股。公司董事会。")
    paired = meeting_anchors("于2024年12月9日、2024年12月25日召开董事会和股东大会,审议通过了《关于回购股份的议案》。")
    assert len(paired) == 1 and paired[0]["date"] == pd.Timestamp("2024-12-09")
    dividend = meeting_anchors("公司于2025年3月27日召开董事会,审议通过了《关于年度利润分配的议案》。随后调整回购价格。")
    assert not dividend
    assert page_text(["1\n金额为人民币", "2\n210,594,617.17元。"] ) == "金额为人民币210,594,617.17元。"
    terms = budget_terms("回购价格不超过人民币25元/股，回购资金总金额不低于人民币2亿元，不超过人民币4亿元。")
    assert terms["unambiguous_pair"] and terms["floor_cents"] == 20_000_000_000 and terms["cap_cents"] == 40_000_000_000
    return {"amount_unit_and_reported_resolution": True, "monthly_vs_cumulative": True,
        "cutoff_separate_from_publication": True, "board_anchor_not_any_body_date": True,
        "price_limit_not_budget_cap": True, "page_number_not_amount_digit": True,
        "parallel_board_and_shareholder_dates": True, "dividend_meeting_not_repurchase_meeting": True}


def run(dry=False):
    tests = checks()
    paths = sorted((sources.OUT / "documents").glob("*.json"))
    rows = [read(path) for path in paths]
    parsed = [parse(row) for row in rows]
    counts = pd.Series([row["status"] for row in parsed]).value_counts().to_dict()
    pilot_path = ROOT / "reports/research/510300_corporate_repurchase_fact_ledger_v1/results/facts.json"
    gold = {row["document_id"]: row for row in read(pilot_path) if row["cumulative_cents"] is not None}
    pilot_checks = []
    for row in parsed:
        if row["document_id"] in gold:
            expected = gold[row["document_id"]]["cumulative_cents"]
            pilot_checks.append({"document_id": row["document_id"], "expected": expected,
                "extracted": row["cumulative_cents"], "agrees": row["cumulative_cents"] == expected,
                "status": row["status"]})
    print({"文档": len(rows), "状态": counts, "已核实试点金额匹配": sum(r["agrees"] for r in pilot_checks), "试点金额总数": len(pilot_checks)}, flush=True)
    if dry:
        for row in pilot_checks:
            if not row["agrees"]:
                print(row, flush=True)
        return
    assert (sources.OUT / "result.json").exists(), "原文采集未到终态，正式结果暂不写入"
    assert len(pilot_checks) == 36 and all(row["agrees"] for row in pilot_checks), "试点事实金额出现解析回归"
    examples = read(OUT / "inspected_examples.json")["examples"]
    indexed = {row["document_id"]: row for row in parsed}
    assert all(indexed[row["document_id"]]["cumulative_cents"] == row["expected_cumulative_cents"] for row in examples)
    for name in ["code", "results"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "purpose": "通用文本提取与原文证据定位，不将董事会日期自动当成完整方案身份，不直接生成交易特征。",
        "sources_sha256": digest(sources.OUT / "documents.json"), "code_sha256": digest(Path(__file__)),
        "implementation_checks": tests, "new_accounts": 0, "new_fits": 0,
        "rules": "只接受同段明确累计、实际或首购上下文；月度金额与累计分开；港股与纯注销文件排除；未知和多义金额不填零；披露精度独立保存。",
        "remaining_admission": "多方案同董事会日期、金额区间及精度、跨公告身份与修订、A/H混合和正文一致性仍需核实。"}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / "results/parsed_documents.json", parsed, True)
    save(OUT / "results/pilot_amount_comparison.json", pilot_checks, True)
    result = {"at": now(), "study_id": STUDY, "status": "TEXT_EXTRACTION_COMPLETED_FEATURE_ADMISSION_PENDING",
        "source_documents": len(rows), "status_counts": counts,
        "extracted_amounts": sum(row["status"].startswith("EXTRACTED_") for row in parsed),
        "pilot_actual_disclosures": len(pilot_checks), "pilot_amounts_matched": sum(row["agrees"] for row in pilot_checks),
        "additional_inspected_examples_matched": len(examples),
        "trading_feature_admitted_documents": 0, "new_accounts": 0, "new_fits": 0,
        "independent_forward_observations": 0, "current_market_view": "NO_VIEW", "goal_status": "active",
        "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)


if __name__ == "__main__":
    run(dry=len(sys.argv) > 1 and sys.argv[1] == "inspect")
