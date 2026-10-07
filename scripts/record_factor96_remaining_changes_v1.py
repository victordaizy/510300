"""把回购条款进展与接口阻碍登记到当前研究状态，不增加回测数或生成交付包。"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_remaining_changes_v1 import OUT, digest, now, read, save


def update(path, value):
    temporary = path.with_name(path.name + ".remaining-changes.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    assert not (OUT / "program_update_receipt.json").exists(), "本轮已登记，不重复累计"
    verification = read(OUT / "saved_verification_receipt.json")
    assert verification["status"] == "PASS_SAVED_NOTICE_ANCHORS_CLOCKS_AND_THREE_LOCAL_ORIGIN_RELATIONS"
    for row in verification["input_identities"]:
        assert digest(ROOT / row["path"]) == row["sha256"]
    cards = {r["document_id"]: r for r in read(OUT / "review_cards.json")}
    links = read(OUT / "local_original_link_addendum.json")["links"]
    links_by_change = defaultdict(list)
    for link in links:
        links_by_change[link["change_document_id"]].append(link)
    prior = ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1/updated_change_ledger.json"
    ledger = read(prior)
    assert len(ledger) == 54
    for row in ledger:
        card = cards.get(row["document_id"])
        if card is None:
            continue
        assert row["review_status"] == "NO_EXISTING_PRIOR_ROOT_CANDIDATE_FULL_TARGET_REVIEW_PENDING"
        row.update(review_status="NOTICE_TARGET_AND_APPROVAL_REVIEWED_ORIGINAL_CHAIN_PARTIAL",
                   action="PURPOSE_CHANGE", approval_state=card["approval_state"],
                   proposed_purpose=card["proposed_new_purpose"], notice_review=card,
                   current_notice_evidence_path=OUT.relative_to(ROOT).as_posix() + "/review_cards.json")
        row["source_original_links"] = links_by_change[row["document_id"]]
        row["confirmed_target_roots"] = [link["root_id"] for link in row["source_original_links"]
                                         if link["relation"] == "CONFIRMED_TARGET_ORIGINAL"]
        row["pooled_origin_reference_roots"] = [link["root_id"] for link in row["source_original_links"]
                                               if link["relation"] == "POOL_REFERENCE_ALLOCATION_UNKNOWN"]
        row["new_executed_cashflow"] = "NOT_A_NEW_MARKET_PURCHASE_EVENT"
    assert not any(r["review_status"] == "NO_EXISTING_PRIOR_ROOT_CANDIDATE_FULL_TARGET_REVIEW_PENDING" for r in ledger)
    save("combined_change_ledger.json", ledger)
    reference = OUT.relative_to(ROOT).as_posix()
    probe = "reports/research/510300_factor96_free_float_source_probe_v1"
    auth = read(ROOT / probe / "single_auth_validation_receipt.json")
    assert auth["provider_code"] == 40101 and auth["rows"] == 0
    blocker = {"status": "CURRENT_CREDENTIAL_EXPIRED", "provider_code": 40101,
               "receipt": probe + "/single_auth_validation_receipt.json",
               "required_external_change": "在本机更新可用数据接口凭据，或取得口径一致的历史自由流通股本来源。",
               "no_new_requests_without_changed_credentials_or_source": True}
    summary = {
        "at": now(), "study_id": "510300_FACTOR96_REPURCHASE_REMAINING_CHANGES_V1",
        "status": "REMAINING_31_NOTICE_CLAUSES_COMPLETE_T12_SOURCE_GATES_PENDING",
        "new_reviewed_change_notices": len(cards), "total_reviewed_change_candidates": len(ledger),
        "remaining_unreviewed_change_candidates": 0,
        "change_documents_with_confirmed_target_root": sum(bool(r["confirmed_target_roots"]) for r in ledger),
        "distinct_confirmed_target_roots": len({root for r in ledger for root in r["confirmed_target_roots"]}),
        "confirmed_change_to_target_root_edges": sum(len(r["confirmed_target_roots"]) for r in ledger),
        "new_pooled_original_reference_links": 2, "remaining_unique_original_targets": 32,
        "new_cancellation_completions_established": 0, "new_market_purchase_events_from_these_changes": 0,
        "free_float_source_blocker": blocker, "full_M02_lifecycle_established": False,
        "strict_T12": "NOT_RUN_FREE_FLOAT_AND_COMPLETE_PURPOSE_CLOCK_CHAIN", "new_accounts": 0,
        "new_returns": 0, "new_free_float_rows": 0, "goal_status": "active", "goal_achieved": False,
        "delivery_package_created": False, "orders_authorized": False,
    }
    save("round_status.json", summary)
    program = ROOT / "reports/research/510300_factor96_program_v1"
    files = [program / n for n in ("status.json", "strategy_progress.json", "factor_progress.json")]
    files.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    values = [read(p) for p in files]
    status, strategies, factors, mandate = values
    before = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]
    protected = {k: status[k] for k in ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
                                      "completed_total_fixed_questions", "independent_forward_observations",
                                      "last_completed_account_experiment", "last_completed_account_result")}
    assert protected["cumulative_admitted_account_scenarios"] == 552
    assert protected["cumulative_executed_account_scenarios"] == 704
    assert not status["delivery_package_required"] and not mandate["delivery_package_required"]
    note = ("补齐31份回购变更公告条款，54份候选已完成对象辨认；新增1条明确目标原方案关系、2条不分配数量的合并来源关系。"
            "本轮29份待股东会、1份董事长提议待双重审批、1份董事会权限内改激励用途；不新增买入或注销完成事件。"
            "仍有32个原方案目标待取得。旧daily_basic采集未请求free_share，官方字段存在，但当前服务实测40101凭据过期、0行；"
            "T12及T13仍未回测，本轮0新账户，夏普目标未完成。")
    for row in strategies:
        if row["id"] == "T12":
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE",
                       current_evidence=note, source_gate_path=reference + "/round_status.json")
        elif row["id"] == "T13":
            row["free_float_source_probe_path"] = probe + "/single_auth_validation_receipt.json"
    for row in factors:
        if row["id"] == "M01":
            row.update(current_note="官方daily_basic包含free_share（万股），旧采集FIELDS未请求；当前接口40101凭据过期，尚无合格自由流通分母。",
                       current_evidence_path=probe + "/single_auth_validation_receipt.json")
        elif row["id"] == "M02":
            row.update(current_note=note, current_evidence_path=reference + "/round_status.json",
                       current_status="NOTICE_SEMANTICS_COMPLETE_FULL_LIFECYCLE_PENDING_NOT_RUN")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=summary["study_id"], latest_result=reference + "/round_status.json",
                  latest_progress_receipt=reference + "/saved_verification_receipt.json",
                  latest_continuation_classification="PROGRESS_31_NOTICE_CLAUSES_AND_THREE_ORIGINS_DATA_CREDENTIAL_PENDING",
                  current_research_phase="REPURCHASE_NOTICE_SEMANTICS_COMPLETE_ORIGINAL_CHAIN_AND_FREE_FLOAT_PENDING",
                  last_source_result=note, admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                  new_reviewed_change_notices_this_round=31, new_explicit_version_links_this_round=1,
                  new_pooled_origin_references_this_round=2, reused_repurchase_pdf_documents_this_round=34,
                  new_source_documents_this_round=1, new_archived_source_http_responses_this_round=2,
                  source_field_candidate_kind="31份用途变更和3份旧原方案；1份官方字段文档及1份接口拒绝响应，不含新股本数据。",
                  repurchase_change_candidates_semantically_reviewed=54,
                  repurchase_remaining_original_targets=32, free_float_source_blocker=blocker,
                  latest_free_float_source_probe=probe + "/single_auth_validation_receipt.json",
                  latest_repurchase_change_ledger=reference + "/combined_change_ledger.json",
                  delivery_package_required=False, latest_user_delivery_instruction="不需要交付包",
                  goal_status="active", goal_achieved=False)
    for key, value in protected.items():
        assert status[key] == value
    mandate.update(as_of_date="2026-09-28", current_round=summary["study_id"],
                   current_protocol=reference + "/protocol.json", latest_progress_receipt=reference + "/saved_verification_receipt.json",
                   last_research_result=note, last_source_result=note,
                   latest_continuation_report=reference + "/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], free_float_source_blocker=blocker,
                   goal_status="active", goal_achieved=False, delivery_package_required=False)
    assert mandate["scheduled_pcf_iopv_collection"] == "PAUSED_BY_USER"
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    assert not mandate["orders_authorized"] and not mandate["option_research_authorized"]
    report = (
        "截至2026-09-28，本轮没有新增策略回测，也没有实现净夏普1.2。用户不需要交付包，本轮仅保留本地代码和研究记录。\n\n"
        "补齐31份回购用途变更公告：29份董事会通过但待股东会，1份仅董事长提议、董事会与股东会均待审议，"
        "另1份是在董事会权限内由员工持股改为限制性股票激励。30份涉及拟注销，不能记为已完成注销或新的市场买入。\n\n"
        "已复用3份本地原方案：豪威集团2026-04-03方案明确对应8750205股拟变更库存；海尔智家2025及2026方案"
        "只确定为合并库存来源，公告未提供逐方案拟注销股数，分配数继续留空。\n\n"
        f"合并此前记录后，54份变更候选均已辨认对象和审批状态，{summary['change_documents_with_confirmed_target_root']}份有明确目标原方案关联。"
        "另有32个不同原方案目标尚待取得；当前记录只覆盖这些公告，不代表所有948条执行披露的完整生命周期。\n\n"
        "自由流通股本缺列的直接原因之一是旧daily_basic采集FIELDS没有请求free_share；官方接口说明确有此字段，单位万股。"
        "单次实测当前服务返回40101、凭据过期、0行。尚未取得股本分母，不用float_share或circ_mv代替；"
        "无凭据或来源变化不重复发送相同请求。官方字段说明：https://tushare.pro/document/2?doc_id=32 。\n\n"
        "已核对31篇原文的785处保存位置和93个公告前后查询；对仅拟议、同批股份多版本、合并库存、未来计划使用分别保留原状态。"
        "这只确认本地条款与查询一致，不是策略有效性证明。\n\n"
        "T12仍未开跑：缺自由流通分母、完整原方案及用途/终止/减额时钟。T13也仍有事件身份与分母缺口。"
        "上一轮T11日期代理主账户（20万元、压力成本、全区间）的净夏普-0.351761、年化-0.270499%、最大回撤4.703664%；"
        "其固定失败结果保留，不改参数救援。\n\n"
        "累计有效研究账户情景552个、另有152个无效实现情景，共执行704个；已完成9个固定研究问题，合格候选0、独立前向观测0。"
        "后续工作是补齐32个原方案并取得可用分母，再决定是否具备T12固定规则测试条件；不能把待补数据算成现金收益或已通过策略。\n"
    )
    (OUT / "研究进展.md").write_text(report, encoding="utf-8")
    for path, value in zip(files, values):
        update(path, value)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save("program_update_receipt.json", {
        "at": now(), "before": before,
        "after": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files],
        "account_counts_unchanged": protected, "new_accounts": 0, "goal_achieved": False,
        "delivery_package_created": False, "current_data_service_requires_external_change": True,
        "goal_has_other_meaningful_source_work": True,
    })
    print("31份公告与3份原方案的进展已登记；未新增回测，目标未完成，没有制作交付包。")


if __name__ == "__main__":
    main()
