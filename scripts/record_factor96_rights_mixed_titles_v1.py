"""核对股类数量、来源日期和旧状态保留后，登记本轮十五份标题补读。"""
from datetime import datetime, timedelta
import json
import pandas as pd

from research.factor96_rights_mixed_titles_review_v1 import (
    OUT, PREVIOUS, ROOT, REUSED_ID, VISUAL_PAGES, read, save, digest, now, normalized,
    source_view, comparisons,
)
from scripts.record_factor96_remaining_changes_v1 import update


def verify():
    for f in read(OUT / "freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    assert digest(OUT / "targets.json") == read(OUT / "run_started.json")["target_sha256"]
    rows = read(OUT / "documents.json")
    old = read(PREVIOUS / "combined_source_facts.json")
    facts = read(OUT / "reviewed_source_facts.json")
    combined = read(OUT / "combined_source_facts.json")
    assert len(old) == 458 and len(facts) == 14 and len(combined) == 472
    assert combined == old + facts and len({f["document_id"] for f in combined}) == 472
    for row in rows:
        assert digest(OUT / row["raw_snapshot"]) == row["raw_sha256"]
        assert digest(OUT / row["text_snapshot"]) == row["text_sha256"]
    byid = {f["document_id"]: f for f in combined}
    pages = {}; anchors = 0
    for rid in [r["document_id"] for r in rows]:
        f = byid[rid]
        pages[rid] = {p["page"]: normalized(p["text"]) for p in read(f["text_path"])["pages"]}
        assert f["event_id"] is None and f["updates"] == {} and not f["trading_feature_admitted"]
        assert f["effective_sellable_new_shares"] is None
        for evidence in f["claim_evidence"].values():
            for e in evidence:
                assert pages[rid][e["page"]][e["normalized_start"]:e["normalized_end"]] == e["literal"]
                anchors += 1
    scopes = read(OUT / "read_scope.json")
    assert len(scopes) == 15 and sum(r["total_pages"] for r in scopes) == 27
    for r in scopes:
        expected = [{"page": p, "literal": t} for p, t in pages[r["document_id"]].items()]
        assert r["full_document_read"] and r["read_text"] == expected
    claims = {f["document_id"]: f["reviewed_claims"] for f in facts}
    table_checks = 0
    for rid, c in claims.items():
        table = c.get("reported_share_table")
        if not table:
            continue
        assert table["A_before"] + table["H_before"] == table["total_before"]
        assert table["A_after"] + table["H_after"] == table["total_after"]
        if "A_delta" in table:
            assert table["A_before"] + table["A_delta"] == table["A_after"]
            assert table["H_before"] + table["H_delta"] == table["H_after"]
            assert table["A_delta"] + table["H_delta"] == table["total_delta"]
            assert table["total_before"] + table["total_delta"] == table["total_after"]
        else:
            assert table["A_after"] - table["A_before"] == table["A_change_calculated"] == 0
            assert table["H_after"] - table["H_before"] == c["own_H_overallotment_terms"]["new_H_shares"]
        table_checks += 1
    assert table_checks == 6
    for rid in ("1206163925", "1208196458", "1213526873"):
        h = claims[rid]["H_subscription_result"]
        assert h["basic_allotment"] + h["extra_allotment"] == h["actual_H_shares"]
        assert h["actual_H_shares"] == claims[rid]["reported_share_table"]["H_delta"]
        if "extra_application_shares" in h:
            assert h["basic_allotment"] + h["extra_application_shares"] == h["total_application_shares"]
            assert h["total_application_shares"] > h["actual_H_shares"]
    assert claims["1213526873"]["H_subscription_result"]["actual_H_shares"] == 82428
    assert claims["1213526873"]["H_subscription_result"]["planned_H_shares"] == 287582400
    zj = claims["1217394908"]["H_subscription_result"]
    assert zj["basic_application_allotted"] + zj["extra_application_allotted"] == zj["total_applications_allotted"]
    assert (zj["total_applications_allotted"] + zj["underwriter_arranged_investors_allotted"] +
            zj["underwriter_taken_shares"]) == zj["actual_H_shares"] == 1366200000
    assert claims["1217394908"]["reported_share_table"]["A_delta"] == 0
    assert zj["announced_H_listing_date"] is None
    transfer = claims["1203474456"]["A_to_H_transfer"]
    assert transfer["A_before"] + transfer["A_delta"] == transfer["A_after"]
    assert transfer["A_restricted_delta"] + transfer["A_unrestricted_delta"] == transfer["A_delta"] == -4893380
    assert transfer["A_delta"] + transfer["H_delta_from_conversion"] == transfer["total_share_delta_from_conversion"] == 0
    basis = claims["1203474456"]["transfer_table_basis"]
    assert (basis["initial_H_listing_shares_reported"] + claims["1203474456"]["own_H_overallotment"]["additional_H_issue_shares"]
            == basis["existing_H_in_conversion_table"])
    assert transfer["A_before"] + basis["existing_H_in_conversion_table"] == 8713933800
    assert transfer["A_after"] + basis["existing_H_in_conversion_table"] + transfer["H_delta_from_conversion"] == 8713933800
    for name in ("招商局集团有限公司", "中国远洋海运集团有限公司"):
        values = claims["1208196458"]["group_level_A_changes"][name]
        assert values["before"] + values["delta"] == values["after"]
    gap = read(OUT / "cms_actor_quantity_gap.json")
    assert gap["commitment_constrained_new_A_shares"] is None
    assert not set(gap["group_A_changes"]) & set(gap["named_commitment_actors"])
    assert claims["1212441974"]["H_approval_and_A_pending"]["A_CSRC_approval_state"] == "PENDING"
    assert claims["1206163925"]["source_signature_anomaly"]["normalized_signature_date"] is None
    assert claims["1206497156"]["subsidiary_overallotment_expiry"]["not_parent_A_rights_termination"]
    assert claims["1214874892"]["subsidiary_H_overallotment"]["parent_A_issue_not_implied"]
    saved_comparisons = read(OUT / "A_H_source_comparisons.json")
    assert saved_comparisons == comparisons(facts, old)
    assert len(saved_comparisons) == 5 and all(not r["later_document_adds_new_A_event"] for r in saved_comparisons)
    views = read(OUT / "source_publication_queries.json")
    for q in views:
        f = byid[q["document_id"]]
        assert datetime.fromisoformat(q["at"]["as_of"]) - datetime.fromisoformat(q["before"]["as_of"]) == timedelta(seconds=1)
        assert q["before"]["status"] == "NO_VIEW" and q["before"]["reviewed_claims"] == {}
        assert q["at"]["reviewed_claims"] == f["reviewed_claims"]
        for side in ("before", "at"):
            assert q[side] == source_view(f, q[side]["as_of"])
    coverage = read(OUT / "mixed_title_coverage.json")
    assert coverage["fixed_catalogue_mixed_titles"] == coverage["reviewed"] == 15
    assert coverage["remaining_in_fixed_scope"] == 0 and not coverage["full_issuance_coverage"]
    visual = [{"document_id": rid, "page": p, "path": str(OUT / "page_previews" / f"{rid}_p{p}.png"),
               "sha256": digest(OUT / "page_previews" / f"{rid}_p{p}.png")} for rid, p in VISUAL_PAGES]
    result = read(OUT / "result.json")
    assert result["new_source_fields"] == sum(len(f["reviewed_claims"]) for f in facts) == 25
    receipt = {"at": now(), "status": "PASS_A_H_QUANTITIES_SCOPE_AND_DISCLOSURE_BOUNDARIES",
               "prior_facts_unchanged": 458, "new_facts": 14, "anchors_checked_including_reused": anchors,
               "share_tables_arithmetically_checked": table_checks, "cross_source_A_comparisons": 5,
               "source_boundary_views": len(views) * 2, "full_documents_read": 15, "full_pages_read": 27,
               "visual_pages": visual, "review_code_sha256": digest(ROOT / "research/factor96_rights_mixed_titles_review_v1.py"),
               "result_sha256": digest(OUT / "result.json"), "goal_achieved": False,
               "account_validation_performed": False, "external_review": "NOT_PERFORMED"}
    save(OUT / "saved_recomputation_receipt.json", receipt)
    return receipt


def record():
    assert not (OUT / "program_update_receipt.json").exists()
    receipt = verify()
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / n for n in ("status.json", "strategy_progress.json", "factor_progress.json")]
    paths.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in paths]; status, strategies, factors, mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RIGHTS_LOCK_FOLLOWUP_V1"
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys = ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
            "cumulative_invalid_implementation_account_scenarios", "last_completed_account_experiment",
            "last_completed_account_result", "independent_forward_observations", "completed_total_fixed_questions",
            "qualified_candidates", "orders_authorized", "latest_rights_event_chains", "latest_rights_publication_queries",
            "latest_rights_holding_claim_addenda", "rights_events_with_issuance_result_listing_in_scope")
    protected = {k: status[k] for k in keys}; before = [{"path": str(p), "sha256": digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix(); result = read(OUT / "result.json")
    note = ("既有目录15份H股或境外混合标题全文已读完，共27页；新增14份PDF25页，复用中信1份2页。"
            "新增25组字段，来源458份增至472份。5份H股结果均含A股上下文：金风、招商、东方的A股新增量与旧来源一致；"
            "浙商A股变动0仅指后续H股阶段；中信旧量冲突及限售承诺原样保留。东方实际H股配售82428股，不用计划287582400股替代。"
            "国泰君安4893380股由A转H，其中受限3036620股、无限售1856760股，属于股类转换；H超额新发48933800股另存。"
            "福耀两份仅H股超额发行，万科与中集两份属于子公司；不加入母公司A股配股事件。招商集团认购量与承诺法人名称未直接连接，"
            "不能据此量化三家承诺法人的限售股。6张股本表、5组A/H比较和30个来源边界视图已复核。"
            "此15份标题缺口归零，不代表完整发行覆盖；36条旧配股链及所有账户不变，T13未回测，目标未实现。")
    for row in strategies:
        if row["id"] == "T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE", current_evidence=note,
                       current_evidence_path=relative + "/result.json", source_gate_path=relative + "/result.json")
    for row in factors:
        if row["id"] == "M06":
            row.update(current_status="MIXED_H_TITLES_REVIEWED_A_H_SCOPES_SEPARATED_NOT_COMPUTED",
                       current_note=note, current_evidence_path=relative + "/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_", "source_field_candidates", "reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=result["study_id"], latest_result=relative + "/result.json",
        latest_progress_receipt=relative + "/saved_recomputation_receipt.json", last_source_result=note,
        latest_continuation_classification="PROGRESS_15_MIXED_TITLE_BODIES_REVIEWED_WITH_A_H_QUANTITY_SEPARATION",
        current_research_phase="T13_REMAINING_HOLDING_IDENTITY_AND_FULL_ISSUANCE_GAPS",
        new_source_documents_this_round=14, new_searchable_text_documents_this_round=14,
        new_archived_source_http_responses_this_round=result["raw_http_responses"],
        new_reviewed_rights_document_facts_this_round=14, new_mixed_title_source_claim_fields_this_round=25,
        reused_rights_pdf_documents_this_round=1, source_field_candidates_this_round=14,
        source_field_candidate_kind="混合标题全文补读：A/H股数量、审批边界、集团主体与股类转换。",
        latest_rights_document_facts=relative + "/combined_source_facts.json",
        latest_rights_mixed_title_facts=relative + "/reviewed_source_facts.json",
        latest_rights_mixed_title_coverage=relative + "/mixed_title_coverage.json",
        latest_rights_mixed_title_coverage_gap=relative + "/mixed_title_coverage.json",
        latest_rights_mixed_title_comparisons=relative + "/A_H_source_comparisons.json",
        latest_rights_mixed_title_source_queries=relative + "/source_publication_queries.json",
        latest_rights_CMS_actor_quantity_gap=relative + "/cms_actor_quantity_gap.json",
        latest_rights_unresolved_holding_links=relative + "/unresolved_holding_links.json",
        rights_mixed_title_fixed_scope_reviewed=15, rights_mixed_title_fixed_scope_remaining=0,
        rights_local_original_pdfs_including_supplement=status["rights_local_original_pdfs_including_supplement"] + 14,
        next_independent_source_action=result["next_source_action"], goal_status="active", goal_achieved=False,
        delivery_package_required=False)
    assert all(status[k] == value for k, value in protected.items())
    mandate.update(current_round=result["study_id"], current_protocol=relative + "/protocol.json",
        latest_progress_receipt=relative + "/saved_recomputation_receipt.json", last_research_result=note, last_source_result=note,
        latest_continuation_report=relative + "/研究进展.md", latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    paragraphs = [
        "只操作510300、完整账户成本后夏普至少1.2的目标仍未实现。合格候选0、独立前向观测0。本轮补读15份原先按H股或境外标题排除的公告，不新增回测或交付包。",
        "范围来自既有77026份公告目录中标题含配股或供股、角色为OTHER_FINANCING_OR_MIXED_TITLE的全部15份，与H股/境外标题过滤结果一致，涉及9个发行人。新下载14份25页，复用中信1份2页；27页正文全部阅读，另查看5页PDF原图。新增14份来源、25组字段，累计472份来源，原458份保持不变。",
        "金风2019年4月30日、招商2020年8月19日、东方2022年5月28日H股配股结果均同时披露A股新增量，与此前A股上市公告一致，分别545352788、1702997123、1502907061股。这些后续披露不产生新A股事件，也不回填到较早上市前。浙商2023年7月27日H股结果中的A股增量为0，表头限定为H股完成前后；其原A股4829739185股配股已于7月6日上市，不能据H阶段的0覆盖原记录。中信上一轮的121480144股承诺限售及旧收购总数冲突均保留。",
        "东方H股实际仅配售82428股，由80436股和1992股构成，与计划287582400股、早先H股核准上限308124000股分别保存。其2022年2月24日公告明确当时A股发行仍待核准，H股批复不能作为A股已获批的证据。金风、招商的额外认购申请量也显著大于最后配售量，全部分开；浙商的股东申请、安排投资者和包销商承接合计1366200000股，只属于H股。",
        "国泰君安2017年5月4日公告另含A股转H股4893380股：限售A股减少3036620股，无限售A股减少1856760股，H股增加4893380股。股类转换本身不增加总股本；该表已经包含额外发行的48933800股H股，不能再把它加一次。福耀两份公告的A股数量均未变；万科万物云和中集中集车辆的公告属于子公司发行与超额配股权状态，不是母公司A股配股。",
        "招商的集团层面A股增加数量分别865808166股、153100965股，但原不减持承诺列的是深圳市招融投资控股有限公司、深圳市集盛投资发展有限公司、中国远洋运输有限公司。不能只凭集团名称和股数就把全部集团数量指定给三家承诺法人，新受限股份数量仍未知。东方申能持股前后数量也没有被差额法升级成明确限售认购数量。",
        "金风H股结果末页署期原文为2019年4日29日，原图确认，未静默改为正常年月日；来源时钟仍用巨潮目录2019年4月30日日末。全部新增来源继续采用目录日末代理，保存披露前一秒及披露时点的30个视图，不宣称已经证明每份公告的最早公开时刻。",
        "6张股本表的加总、5组A/H来源比较和既有事件保留检查通过。此15份混合标题的正文缺口归零，不表示全部发行事件和约束已齐。下一步连接浪潮2016年10月承诺与具体配股事项；其他已存持有约束缺口及历史自由流通股分母问题保留。M06严格供给压力未计算，T13未回测。",
    ]
    with (OUT / "研究进展.md").open("x", encoding="utf-8") as handle:
        handle.write("\n\n".join(paragraphs) + "\n")
    for path, value in zip(paths, objects):
        update(path, value)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": now(), "before": before,
        "after": [{"path": str(p), "sha256": digest(p)} for p in paths], "protected_state": protected,
        "new_accounts": 0, "goal_achieved": False, "delivery_package_created": False})
    print(json.dumps({"累计来源": 472, "混合标题剩余": 0, "原文锚点": receipt["anchors_checked_including_reused"],
                      "时点视图": 30, "新增账户": 0, "目标实现": False}, ensure_ascii=False))


if __name__ == "__main__":
    record()
