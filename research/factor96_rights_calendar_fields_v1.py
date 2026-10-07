"""定位配股条款候选，并建立最早发行公告对应的四文档时序示例。"""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd

from research.factor96_issuance_catalogue_v1 import digest, read, save

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"
OUT = ROOT / "reports/research/510300_factor96_rights_calendar_fields_v1"
CASE_IDS = ["1201854885", "1201897631", "1201900999", "1201908424"]
CUES = {
    "PAYMENT_OR_CLEARING": r"配股缴款|认购缴款|缴款起止",
    "LISTING": r"新增股份上市|新增股票上市|获配股票上市|上市时间|上市流通日",
    "RECORD_DATE": r"股权登记日",
    "PRICE": r"配股价格|发行价格",
    "ACTUAL_SUBSCRIPTION": r"有效认购股份|有效认购资金|有效认购数量|认购金额",
    "PROCEEDS": r"募集资金",
    "REVISION_OR_UNKNOWN": r"现更正为|公告原文为|另行公告|终止本次配股|终止公司配股",
}
DATE = re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日")
NUMBER = r"(?<![\d.,])([\d][\d,]*(?:\.\d+)?)"
MONEY = re.compile(NUMBER + r"(亿元|万元|元)(?!/股)")
SHARES = re.compile(NUMBER + r"(亿股|万股|股)")


def now():
    return datetime.now().astimezone().isoformat()


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def number_literals(text, regex):
    values = []
    for match in regex.finditer(text):
        amount, unit = match.groups()
        scale = Decimal("100000000") if unit.startswith("亿") else Decimal("10000") if unit.startswith("万") else Decimal(1)
        number = Decimal(amount.replace(",", "")) * scale
        values.append({"literal": match.group(0), "normalized_number": format(number, "f"), "unit": unit, "start": match.start(), "end": match.end()})
    return values


def extract_candidates(pages, document_id):
    output = []
    for page in pages:
        text = normalized(page["text"])
        for kind, pattern in CUES.items():
            for hit in re.finditer(pattern, text):
                start, end = max(0, hit.start() - 65), min(len(text), hit.end() + 220)
                evidence = text[start:end]
                dates = []
                for date_match in DATE.finditer(evidence):
                    y, m, d = map(int, date_match.groups())
                    try:
                        value = datetime(y, m, d).date().isoformat()
                    except ValueError:
                        value = None
                    dates.append({"literal": date_match.group(0), "value": value, "start": date_match.start(), "end": date_match.end()})
                output.append({"document_id": document_id, "page": page["page"], "cue_kind": kind,
                    "cue": hit.group(0), "normalized_start": start, "normalized_end": end, "evidence": evidence,
                    "date_literals": dates, "money_literals": number_literals(evidence, MONEY),
                    "share_literals": number_literals(evidence, SHARES), "event_field_admitted": False,
                    "note": "邻近文本中的数字仅作定位，尚未判定属于计划、历史事实、当前结果或引用旧文；缴款清算区间不自动成为缴款窗口。"})
    return output


def asof_case(versions, at):
    eligible = [v for v in versions if pd.Timestamp(v["known_at"]) <= pd.Timestamp(at)]
    if not eligible:
        return {"as_of": at, "status": "NO_VIEW", "actual_subscribed_shares": None,
                "actual_subscription_cents": None, "announced_listing_date": None, "source_documents": []}
    state = {"as_of": at, "status": "KNOWN_SOURCE_CLAIMS_ONLY", "actual_subscribed_shares": None,
             "actual_subscription_cents": None, "announced_listing_date": None, "source_documents": []}
    for version in sorted(eligible, key=lambda v: v["known_at"]):
        state.update(version["updates"])
        state["source_documents"].append(version["document_id"])
    day = pd.Timestamp(at).date().isoformat()
    if day < state["payment_start"]:
        state["payment_phase"] = "ANNOUNCED_WINDOW_NOT_STARTED"
    elif day <= state["payment_end"]:
        state["payment_phase"] = "ANNOUNCED_PAYMENT_WINDOW_DATE"
    else:
        state["payment_phase"] = "ANNOUNCED_PAYMENT_WINDOW_ENDED"
    if state["announced_listing_date"] is None:
        state["listing_phase"] = "NO_KNOWN_LISTING_DATE"
    elif day < state["announced_listing_date"]:
        state["listing_phase"] = "ANNOUNCED_LISTING_DATE_AHEAD"
    else:
        state["listing_phase"] = "ANNOUNCED_DATE_REACHED_NOT_IMPLEMENTATION_PROOF"
    state["trading_feature_admitted"] = False
    return state


def case_versions(documents):
    by_id = {r["document_id"]: r for r in documents}
    updates = [
        {"record_date": "2015-12-28", "payment_start": "2015-12-29", "payment_end": "2016-01-05",
         "scheduled_clearing_date": "2016-01-06", "scheduled_exrights_resumption_date": "2016-01-07",
         "plan_max_shares": 1560000000, "issue_price_cents": 819, "stated_proceeds_budget_cap_cents": 1500000000000,
         "plan_shares_times_price_cents": 1560000000 * 819, "scheduled_result_publication_date": None,
         "result_date_conflict": ["2015-01-07", "2016-01-07"]},
        {"actual_subscribed_shares": 1496665905, "actual_subscription_cents": 1225769376195,
         "announced_listing_date": None, "listing_source_status": "EXPLICITLY_TO_BE_ANNOUNCED"},
        {"actual_subscribed_shares": 1496671674, "actual_subscription_cents": 1225774101006,
         "correction_of_document_id": "1201897631"},
        {"announced_listing_date": "2016-01-18", "listing_announced_shares": 1496671674},
    ]
    roles = ["ISSUANCE_NOTICE", "INITIAL_SUBSCRIPTION_RESULT", "CORRECTED_SUBSCRIPTION_RESULT", "LISTING_NOTICE"]
    versions = []
    for key, fields, role in zip(CASE_IDS, updates, roles):
        row = by_id[key]
        versions.append({"document_id": key, "role": role, "catalogue_timestamp": row["catalogue_timestamp"],
            "known_at": row["catalogue_date"] + "T23:59:59+08:00", "source_url": row["source_url"], "updates": fields})
    assert versions[1]["updates"]["actual_subscribed_shares"] * 819 == versions[1]["updates"]["actual_subscription_cents"]
    assert versions[2]["updates"]["actual_subscribed_shares"] * 819 == versions[2]["updates"]["actual_subscription_cents"]
    return versions


def case_evidence(documents):
    by_id = {r["document_id"]: r for r in documents}
    specifications = [
        (CASE_IDS[0], 1, "ISSUE_PRICE", r"本次配股价格为8\.19元/股"),
        (CASE_IDS[0], 1, "MAX_SHARES", r"可配售股份总数为1,560,000,000股"),
        (CASE_IDS[0], 1, "RECORD_DATE", r"股权登记日2015年12月28日"),
        (CASE_IDS[0], 3, "PAYMENT_WINDOW", r"2015年12月29日至2016年1月5日\(T\+1日-T\+5日\)配股缴款起止日期"),
        (CASE_IDS[0], 2, "HALT_WITH_CLEARING_NOT_PAYMENT", r"2015年12月29日\(T\+1日\)至2016年1月6日\(T\+6日\)"),
        (CASE_IDS[0], 3, "CLEARING_DATE", r"2016年1月6日\(T\+6日\)获取保荐机构清算数据发行成功,登记公司网上清算"),
        (CASE_IDS[0], 3, "EXRIGHTS_RESUMPTION", r"2016年1月7日\(T\+7日\)刊登发行结果公告发行成功的除权基准日"),
        (CASE_IDS[0], 1, "CONFLICTING_RESULT_YEAR", r"本次发行结果将于2015年1月7日"),
        (CASE_IDS[0], 3, "PROCEEDS_BUDGET_CAP", r"拟募集资金不超过150亿元人民币"),
        (CASE_IDS[1], 1, "INITIAL_SHARES", r"有效认购股份\(股\)1,496,665,905"),
        (CASE_IDS[1], 1, "INITIAL_MONEY", r"有效认购资金总额\(人民币元\)12,257,693,761\.95"),
        (CASE_IDS[1], 2, "LISTING_UNKNOWN", r"本次配股的获配股份上市时间将另行公告"),
        (CASE_IDS[2], 1, "EXPLICIT_OLD_DOCUMENT_REFERENCE", r"配股发行结果公告》\(临2016-003\)"),
        (CASE_IDS[2], 1, "QUOTED_OLD_SHARES", r"有效认购股份\(股\)1,496,665,905"),
        (CASE_IDS[2], 2, "CORRECTION_BOUNDARY", r"现更正为:"),
        (CASE_IDS[2], 2, "CORRECTED_SHARES", r"有效认购股份\(股\)1,496,671,674"),
        (CASE_IDS[2], 2, "CORRECTED_MONEY", r"有效认购资金总额\(人民币元\)12,257,741,010\.06"),
        (CASE_IDS[3], 2, "LISTING_DATE", r"新增股份上市时间:2016年1月18日"),
        (CASE_IDS[3], 2, "LISTING_SHARES", r"本次配股新增上市股份1,496,671,674股"),
    ]
    evidence = []
    for key, page_number, field, pattern in specifications:
        pages = read(SOURCE / by_id[key]["text_snapshot"])["pages"]
        page = pages[page_number - 1]
        text = normalized(page["text"])
        match = re.search(pattern, text)
        assert match is not None, (key, page_number, field, pattern)
        evidence.append({"document_id": key, "page": page_number, "field": field, "pattern": pattern,
            "normalized_start": match.start(), "normalized_end": match.end(), "evidence": match.group(0)})
    # 更正后的数量金额必须位于同页更正界线之后，原文引用不作新版本值。
    positions = {r["field"]: r["normalized_start"] for r in evidence if r["document_id"] == CASE_IDS[2] and r["page"] == 2}
    assert positions["CORRECTED_SHARES"] > positions["CORRECTION_BOUNDARY"]
    assert positions["CORRECTED_MONEY"] > positions["CORRECTION_BOUNDARY"]
    return evidence


def freeze():
    assert not (OUT / "freeze.json").exists()
    assert (SOURCE / "result.json").exists()
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    manifest = SOURCE / "transport_completion_v1/effective_documents.json"
    if not manifest.exists():
        manifest = SOURCE / "documents.json"
    documents = read(manifest)
    seed = min([r for r in documents if r["title"] == "配股发行公告" and r["status"] == "PDF_TEXT_SAVED"],
               key=lambda r: (r["catalogue_date"], r["document_id"]))
    assert seed["document_id"] == CASE_IDS[0]
    evidence = case_evidence(documents)
    versions = case_versions(documents)
    for name, source in {"inputs/documents.json": manifest, "inputs/source_result.json": SOURCE / "result.json",
        "inputs/source_freeze.json": SOURCE / "freeze.json"}.items():
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
    if (SOURCE / "transport_completion_v1/result.json").exists():
        shutil.copy2(SOURCE / "transport_completion_v1/result.json", OUT / "inputs/source_transport_completion_result.json")
    save(OUT / "inputs/case_evidence_specifications.json", evidence, True)
    save(OUT / "inputs/case_versions_specifications.json", versions, True)
    for file in [Path(__file__), ROOT / "tests/test_factor96_rights_calendar_fields_v1.py"]:
        target = OUT / "code" / file.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(file, target)
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_CALENDAR_FIELDS_V1",
        "phase": "SOURCE_CLAUSE_CANDIDATES_AND_ONE_CHRONOLOGICAL_CASE_NO_RETURNS",
        "scope": "全部已成功提取文本的固定491文档，以固定词组定位日期、金额和股数原文候选，所有候选event_field_admitted=false。",
        "case_selection": "按目录日期和文档ID选择最早的严格配股发行公告，得到兴业证券2015-12-24；再读取该发行的结果、更正、上市四文档，不按收益选案例。",
        "schema_seen_before_freeze": CASE_IDS,
        "case_roles": "计划可配数、价格、预算上限，缴款日期，含清算的停牌期，除权复牌日，初始实际认购，更正金额与股数，另行公告的上市日分开。",
        "conflict_policy": "首份公告正文称结果日2015-01-07，表格为2016-01-07，两值均保留，计划结果发布日期置空，不自动修正年份。",
        "revision_policy": "2016-01-08更正只在该文档目录日结束后生效；其引用旧文不是新值；此前仍用初始结果。",
        "time_policy": "目录时间统一保守到同日23:59:59；仅作已保存历史来源的时序示例，不证明历史第一版HTTP可得，届时是否实际上市另需事实。",
        "not_established": ["全体配股事件身份与版本链", "其他发行方式日历", "自由流通市值", "T13收益或入场"],
        "new_accounts": 0, "new_returns": 0, "new_models": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "external_review": "NOT_PERFORMED"}, True)
    files = [{"scope": "local", "path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]
    for row in documents:
        if row["status"] == "PDF_TEXT_SAVED":
            files.extend([{"scope": "source", "path": row["raw_snapshot"], "sha256": row["raw_sha256"]},
                          {"scope": "source", "path": row["text_snapshot"], "sha256": row["text_sha256"]}])
    save(OUT / "freeze.json", {"at": now(), "files": files}, True)
    print("配股字段合同已冻结：全体原文定位候选及最早发行的四文档时序示例。", flush=True)


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        parent = OUT if item["scope"] == "local" else SOURCE
        assert digest(parent / item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    documents, candidates = read(OUT / "inputs/documents.json"), []
    for row in documents:
        if row["status"] == "PDF_TEXT_SAVED":
            candidates.extend(extract_candidates(read(SOURCE / row["text_snapshot"])["pages"], row["document_id"]))
    save(OUT / "clause_candidates.json", candidates, True)
    pd.DataFrame([{k: r[k] for k in ["document_id", "page", "cue_kind", "cue", "evidence"]} for r in candidates]).to_csv(
        OUT / "配股日期金额股数原文候选.csv", index=False, encoding="utf-8-sig")
    versions = case_versions(documents)
    assert versions == read(OUT / "inputs/case_versions_specifications.json")
    evidence = case_evidence(documents)
    assert evidence == read(OUT / "inputs/case_evidence_specifications.json")
    save(OUT / "case_versions.json", versions, True)
    save(OUT / "case_evidence.json", evidence, True)
    dates = ["2015-12-23T23:59:59+08:00", "2015-12-24T23:59:59+08:00", "2016-01-05T14:00:00+08:00",
        "2016-01-06T23:59:59+08:00", "2016-01-07T23:59:59+08:00", "2016-01-08T23:59:59+08:00",
        "2016-01-13T23:59:59+08:00", "2016-01-18T23:59:59+08:00"]
    snapshots = [asof_case(versions, date) for date in dates]
    save(OUT / "case_asof_snapshots.json", snapshots, True)
    pd.DataFrame(snapshots).to_csv(OUT / "兴业证券配股已知信息时序.csv", index=False, encoding="utf-8-sig")
    result = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_CALENDAR_FIELDS_V1", "status": "CLAUSE_CANDIDATES_AND_ONE_VERSIONED_CALENDAR_CASE_COMPLETE",
        "source_documents": len(documents), "documents_with_candidates": len({r["document_id"] for r in candidates}),
        "clause_candidates": len(candidates), "case_documents": len(versions), "case_evidence_anchors": len(evidence),
        "asof_snapshots": len(snapshots), "correction_delta_shares": 1496671674 - 1496665905,
        "correction_delta_cents": 1225774101006 - 1225769376195,
        "calendar_cases_fully_generalized": False, "full_M06_calendar_established": False,
        "T13": "NOT_RUN", "new_accounts": 0, "new_returns": 0, "new_models": 0, "new_network_requests": 0,
        "goal_status": "active", "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run()
