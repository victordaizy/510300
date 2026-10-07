"""固定剩余配股文档与旧筛选未选标题，检查中间版本候选，不计算收益。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re

from research.factor96_rights_event_chains_v1 import read, save, digest, normalized, now, SOURCE, ROOT
from research.factor96_rights_event_review_v1 import APPROVAL_RE

BASE = ROOT / "reports/research/510300_factor96_rights_event_chains_v1/effective_v1_0_1"
OUT = ROOT / "reports/research/510300_factor96_rights_intermediate_v1"


def title_class(title):
    title = normalized(title)
    if "超额配股" in title or "H股配股" in title or "境外" in title:
        return "H_SHARE_OR_OVERSUBSCRIPTION_TITLE"
    if "保荐" in title or "法律意见" in title or "律师" in title or "核查" in title:
        return "SUPPORTING_DOCUMENT_TITLE"
    if "转股" in title:
        return "CONVERTIBLE_CONVERSION_HALT_TITLE"
    if "发行的公告" in title or "配股发行公告" in title:
        return "ISSUANCE_IMPLEMENTATION_TITLE"
    if "发行结果" in title or "获配" in title and "上市" in title:
        return "RESULT_OR_LISTING_TITLE"
    if "提示" in title and not any(word in title for word in ("摊薄", "填补", "申请文件", "财务数据", "问询", "说明书")):
        return "PAYMENT_OR_RESUMPTION_REMINDER_TITLE"
    if "说明书" in title:
        return "FILING_DRAFT_PROSPECTUS_TITLE" if "申报稿" in title else "PROSPECTUS_OR_ABSTRACT_TITLE"
    if "募集资金" in title and any(word in title for word in ("节余", "结余", "结项", "补充流动")):
        return "HISTORICAL_PROCEEDS_USE_TITLE"
    if any(word in title for word in ("终止", "失效", "撤回")):
        return "TERMINATION_OR_EXPIRY_TITLE"
    if any(word in title for word in ("延长", "延期", "调整", "更正", "修订")):
        return "PLAN_OR_ADMINISTRATIVE_CHANGE_TITLE"
    if any(word in title for word in ("预案", "方案", "比例", "数量", "条件", "摊薄", "填补", "承诺", "可行性")):
        return "PRE_IMPLEMENTATION_PLAN_TITLE"
    if any(word in title for word in ("反馈", "审核", "核准", "批复", "受理", "会后", "问询", "申请")):
        return "APPLICATION_OR_REGULATORY_STAGE_TITLE"
    return "OTHER_UNRESOLVED_TITLE"


def freeze():
    documents = read(SOURCE / "transport_completion_v1/effective_documents.json")
    previous = read(BASE / "reviewed_document_facts.json")
    used = {row["document_id"] for row in previous}
    local = [{**row, "title_class": title_class(row["title"])} for row in documents if row["document_id"] not in used]
    with (SOURCE / "本阶段未选中的配股标题.csv").open(encoding="utf-8-sig") as handle:
        excluded = [{**row, "title_class": title_class(row["title"]),
                     "already_reviewed_in_base": row["document_id"] in used} for row in csv.DictReader(handle)]
    assert len(local) == 365 and len(excluded) == 717
    implementation = [row for row in excluded if row["title_class"] in
                      ("ISSUANCE_IMPLEMENTATION_TITLE", "RESULT_OR_LISTING_TITLE", "PAYMENT_OR_RESUMPTION_REMINDER_TITLE")]
    targets = [row for row in implementation if not row["already_reviewed_in_base"]]
    save(OUT/"protocol.json", {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_INTERMEDIATE_V1",
        "previous_goal_turn_classification": "PROGRESS_36_ISSUANCE_CHAINS_AND_ONE_TITLE_OMISSION_RESOLVED",
        "scope": "剩余365份已存原文全部做身份及日历候选定位；717条旧未选标题全部重新分类。正文结果不按收益选择。",
        "remaining_local_documents": len(local), "unselected_title_rows": len(excluded),
        "additional_original_targets_from_titles": len(targets),
        "target_selection": "原未选标题里，明确发行、发行结果、获配上市及实际配股提示/复牌标题，排除已读天齐原件；不按公司收益或日历差异选择。",
        "source_transport": "固定目标公开PDF，逐个请求，单请求10秒连接/30秒读取、40MiB；仅传输或5xx最多2次，403或429终止新请求；保存失败，不替换目标。",
        "identity_rule": "文档核准文号与同证券的36既有A股事件匹配；多重、缺失、H股及历史引用分别保留，不能仅凭时间相邻确认身份。",
        "comparison": "实际提示文件只与该文件公开代理时点前已知的条款比较；无法唯一抽取的值不得默认沿用原值。说明书历史引用只做候选，不自动覆盖发行条款。",
        "clock": "目录日末23:59:59+08:00代理；历史首版与完整期间覆盖尚未证明。所有来源记录仍不准入交易特征。",
        "new_accounts": 0, "new_returns": 0, "new_models": 0,
        "full_M06_calendar_established": False, "orders_authorized": False, "delivery_package_required": False})
    save(OUT/"remaining_local_documents.json", local)
    save(OUT/"unselected_title_review.json", excluded)
    save(OUT/"additional_source_targets.json", targets)
    save(OUT/"freeze.json", {"at": now(), "files": [{"path": str(p), "sha256": digest(p)} for p in
         [OUT/"protocol.json", OUT/"remaining_local_documents.json", OUT/"unselected_title_review.json", OUT/"additional_source_targets.json", BASE/"reviewed_document_facts.json", BASE/"event_chains.json"]]})
    print("范围已固定：365份本地文档、717条未选标题、",len(targets),"个新增定向原文目标。")
    print("本地标题分类",dict(Counter(row["title_class"] for row in local)))
    print("未选标题分类",dict(Counter(row["title_class"] for row in excluded)))
    print("新增目标",[(r["symbol"],r["document_id"],r["title"]) for r in targets])


def extract():
    for row in read(OUT/"freeze.json")["files"]:
        assert digest(row["path"]) == row["sha256"]
    events = read(BASE/"event_chains.json")
    lookup = {(r["symbol"], r["approval_key"]): r["event_id"] for r in events}
    rows = read(OUT/"remaining_local_documents.json")
    extra = OUT/"additional_documents.json"
    if extra.exists():
        rows += [r for r in read(extra) if r["status"] == "PDF_TEXT_SAVED"]
    index, candidates = [], []
    cues = {"RECORD_DATE": r"股权登记日", "PAYMENT_WINDOW": r"(?:配股)?(?:缴款|认购)(?:起止日期|时间|期限|期)",
            "PRICE": r"配股价格|发行价格", "QUANTITY": r"可配(?:售)?(?:A股)?(?:股份|股票)?(?:数量|总数|总额|总数量)",
            "REVISION": r"更正为|更正后|修改本次|调整.{0,8}(?:配股|日期)|缴款期为|另行公告"}
    for row in rows:
        text_path = Path(row["text_snapshot"])
        if not text_path.is_absolute():
            text_path = SOURCE/text_path
        assert digest(text_path) == row["text_sha256"]
        pages = read(text_path)["pages"]
        identities, anchors = [], []
        for page in pages:
            text = normalized(page["text"])
            for match in APPROVAL_RE.finditer(text):
                key = "_".join(match.groups())
                identities.append({"approval_key": key, "matched_event_id": lookup.get((row["symbol"], key)),
                    "page": page["page"], "start": match.start(), "end": match.end(), "literal": match.group(0),
                    "context": text[max(0,match.start()-110):match.end()+170]})
            for field, pattern in cues.items():
                for match in re.finditer(pattern, text):
                    start, end = max(0,match.start()-100), min(len(text),match.end()+230)
                    anchors.append({"document_id": row["document_id"], "field": field, "page": page["page"],
                                    "start": start, "end": end, "cue_start": match.start(), "cue_end": match.end(),
                                    "text": text[start:end], "admitted_event_field": False})
        candidate_events = sorted({r["matched_event_id"] for r in identities if r["matched_event_id"]})
        index.append({"document_id": row["document_id"], "symbol": row["symbol"], "title": row["title"],
            "title_class": row["title_class"], "known_at": row["catalogue_date"]+"T23:59:59+08:00",
            "pages": len(pages), "text_path": str(text_path), "identity_anchors": identities,
            "candidate_event_ids": candidate_events, "identity_confirmed": False,
            "field_candidate_count": len(anchors), "trading_feature_admitted": False})
        candidates.extend(anchors)
    save(OUT/"intermediate_document_index.json",index)
    save(OUT/"intermediate_field_candidates.json",candidates)
    print("原文定位完成：",len(index),"文档",sum(r["pages"] for r in index),"页，",len(candidates),"段候选。")
    print("候选事件匹配分布",dict(Counter(len(r["candidate_event_ids"]) for r in index)))
    print("实际提醒候选",[(r["document_id"],r["symbol"],len(r["candidate_event_ids"])) for r in index if r["title_class"]=="PAYMENT_OR_RESUMPTION_REMINDER_TITLE" and len(r["candidate_event_ids"])!=1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else extract()
