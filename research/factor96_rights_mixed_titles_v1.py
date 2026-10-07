"""补读既有目录的十五份H股与混合配股标题，保留A/H及发行人边界。"""
import argparse
from copy import deepcopy
import json
import pandas as pd

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, now
import research.factor96_rights_issue_documents_v1 as collector

OUT = ROOT / "reports/research/510300_factor96_rights_mixed_titles_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_rights_lock_followup_v1"
CATALOGUE = ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1/unique_documents.parquet"
REUSED_ID = "1212493239"


def freeze():
    status = read(ROOT / "reports/research/510300_factor96_program_v1/status.json")
    assert status["latest_round"] == "510300_FACTOR96_RIGHTS_LOCK_FOLLOWUP_V1"
    frame = pd.read_parquet(CATALOGUE)
    rights = frame[frame.title.fillna("").str.contains("配股|供股")]
    mixed = rights[rights.title_role.eq("OTHER_FINANCING_OR_MIXED_TITLE")].sort_values(
        ["catalogue_date", "symbol", "document_id"])
    h_titles = rights[rights.title.fillna("").str.contains(r"[hHＨｈ]股|境外|[aAＡａ][+＋及和、/][hHＨｈ]")]
    assert len(mixed) == 15 and set(mixed.document_id) == set(h_titles.document_id)
    prior_ids = {r["document_id"] for r in read(PREVIOUS / "combined_source_facts.json")}
    assert set(mixed.document_id) & prior_ids == {REUSED_ID}
    targets = []
    for raw in mixed.to_dict("records"):
        row = deepcopy(raw)
        row.update(review_role="H_AND_MIXED_TITLE_BODY_REVIEW",
                   reuse_existing_source=row["document_id"] == REUSED_ID,
                   source_date_basis="既有巨潮目录日期，日末代理",
                   historical_first_publication_verified=False, trading_feature_admitted=False)
        targets.append(row)
    save(OUT / "targets.json", targets)
    save(OUT / "protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_RIGHTS_MIXED_TITLES_V1",
        "previous_goal_turn_classification": "PROGRESS_EXPLICIT_CITIC_TRANCHE_AND_EXTENSION_SOURCES",
        "selection": "既有77026份去重目录中标题含配股或供股、角色为OTHER_FINANCING_OR_MIXED_TITLE的全部15份；与H/境外标题交叉核对一致。",
        "scope": "9个发行人；复用中信1份已读来源，新取14份；逐份全文核对正文中的A股字段、H股字段、承诺及主体。",
        "coverage_boundary": "只关闭这15份目录标题的正文缺口；不宣称全部发行、全部H股披露或供给因子完整。",
        "quantity_rule": "A/H各自保存；超额配股、子公司发行、国有股转持与母公司新增A股分开。公告登记无限售不自动推定无承诺限制。",
        "clock_rule": "目录日末为可得代理；后披露的结果不得回填到A股上市前；签署日和H股上市日分别保存。",
        "conflict_rule": "原股数、新配股数、募资总额和旧披露差异并列保存，不静默订正。",
        "collection": "已有公开静态PDF，逐份顺序请求，复用原收集器的超时、限额、失败留存与403/429停止规则。",
        "measurement": "先保存字段与比较记录，不计算收益、不改变M06分母、不把题名修复当策略通过。",
        "new_accounts": 0, "new_models": 0, "orders_authorized": False,
        "delivery_package_required": False, "goal_achieved": False,
    })
    paths = [OUT / "targets.json", OUT / "protocol.json", CATALOGUE,
             PREVIOUS / "combined_source_facts.json", PREVIOUS / "combined_claim_addenda.json",
             PREVIOUS / "mixed_title_coverage_gap.json", __file__]
    save(OUT / "freeze.json", {"at": now(), "files": [
        {"path": str(p), "sha256": digest(p)} for p in paths]})
    print("已冻结15份混合标题正文补读，复用1份、新取14份。", flush=True)


def collect():
    for f in read(OUT / "freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    save(OUT / "run_started.json", {"at": now(), "target_sha256": digest(OUT / "targets.json")})
    collector.OUT = OUT
    old = {r["document_id"]: r for r in read(PREVIOUS / "documents.json")}
    rows = []
    for target in read(OUT / "targets.json"):
        if target["reuse_existing_source"]:
            row = deepcopy(old[target["document_id"]])
            for key in ("raw_snapshot", "text_snapshot", "receipt_snapshot"):
                row[key] = str(PREVIOUS / row[key])
            row.update(review_role=target["review_role"], reuse_existing_source=True)
        else:
            row = collector.extract(collector.download(target))
        rows.append(row)
        save(OUT / "documents" / f"{row['document_id']}.json", row)
        print(json.dumps({"公告": row["document_id"], "状态": row["status"],
                          "页数": row.get("pages"), "复用": row.get("reuse_existing_source", False)},
                         ensure_ascii=False), flush=True)
    save(OUT / "documents.json", rows)
    save(OUT / "collection_result.json", {
        "at": now(), "target_documents": len(rows), "new_targets": 14, "reused_documents": 1,
        "pdf_text_saved": sum(r["status"] == "PDF_TEXT_SAVED" for r in rows),
        "new_pages": sum(r.get("pages", 0) for r in rows if not r.get("reuse_existing_source")),
        "all_pages": sum(r.get("pages", 0) for r in rows),
        "raw_http_responses": len(list((OUT / "receipts").glob("*.json"))),
        "goal_achieved": False, "T13": "NOT_RUN", "new_accounts": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "collect"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else collect()
