"""核对全文关键词覆盖与部分阅读边界，登记来源范围差异，不重跑账户。"""
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json
import re

import pandas as pd

from research.factor96_rights_prospectus_constraints_review_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, VISUAL_PAGES
from research.factor96_rights_prospectus_constraints_v1 import PATTERN
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    facts=read(OUT/"reviewed_context_source_facts.json");prior=read(PREVIOUS/"combined_source_facts.json")
    combined=read(OUT/"combined_source_facts.json");result=read(OUT/"result.json")
    assert len(facts)==6 and len(prior)==439 and len(combined)==445 and combined[:439]==prior and combined[439:]==facts
    assert len({f["document_id"] for f in combined})==445
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    targets=read(OUT/"targets.json");candidate_index={};lookup={}
    for name,kind in (("source_candidates","CURRENT_SECTION"),("frontmatter_candidates","FRONTMATTER"),("remainder_candidates","REMAINDER")):
        for row in read(OUT/(name+".json")):
            for j,c in enumerate(row["candidates"]):
                key=(row["document_id"],kind,j);assert key not in candidate_index
                candidate_index[key]=c;lookup.setdefault((row["document_id"],c["page"]),[]).append(c)
    assert len(candidate_index)==381
    reviewed=read(OUT/"reviewed_candidates.json");pending=read(OUT/"remaining_holding_candidates.json")["candidates"]
    reviewed_keys={(c["document_id"],c["candidate_set"],c["candidate_index"]) for c in reviewed}
    pending_keys={(c["document_id"],c["candidate_set"],c["candidate_index"]) for c in pending}
    assert len(reviewed)==len(reviewed_keys)==133 and len(pending)==len(pending_keys)==248
    assert not reviewed_keys&pending_keys and reviewed_keys|pending_keys==set(candidate_index)
    assert all(c["reviewed"] for c in reviewed) and not any(c["reviewed"] for c in pending)
    pages_by_id={};pdf_pages=0;matches=0
    for target in targets:
        pages={p["page"]:normalized(p["text"]) for p in read(target["text_path"])["pages"]}
        pages_by_id[target["document_id"]]=pages;pdf_pages+=len(pages)
        assert digest(target["text_path"])==target["text_sha256"]
        for page,t in pages.items():
            cs=lookup.get((target["document_id"],page),[])
            for m in re.finditer(PATTERN,t):
                assert any(c["start"]<=m.start() and m.end()<=c["end"] for c in cs),(target["document_id"],page,m.group())
                matches+=1
    assert pdf_pages==result["target_pdf_pages"]==15747
    for item in reviewed+pending:
        c=candidate_index[(item["document_id"],item["candidate_set"],item["candidate_index"])]
        assert (item["page"],item["start"],item["end"],item["literal"])==(c["page"],c["start"],c["end"],c["literal"])
        assert pages_by_id[item["document_id"]][item["page"]][item["start"]:item["end"]]==item["literal"]
    anchors=0
    for f in facts:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        assert f["event_id"] is None and f["updates"]=={} and not f["trading_feature_admitted"]
        assert f["effective_sellable_new_shares"] is None and f["strict_M06_pressure"] is None
        assert set(f["reviewed_claims"])==set(f["claim_evidence"])
        for evidence in f["claim_evidence"].values():
            for e in evidence:
                assert pages_by_id[f["document_id"]][e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
    assert sum(len(f["reviewed_claims"]) for f in facts)==10
    legacy=facts[0]["reviewed_claims"]["legacy_private_placement_lock"]
    assert Decimal(legacy["reported_subscription_value"])*10000==legacy["original_subscription_shares"]==104156064
    queries=read(OUT/"source_publication_queries.json");idx={f["document_id"]:f for f in facts};assert len(queries)==6
    for q in queries:
        f=idx[q["document_id"]]
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        for side in ("before","at"):assert q[side]==source_view(f,q[side]["as_of"])
    comp=read(OUT/"windpower_response_vs_prospectus_scope.json");old=next(f for f in prior if f["document_id"]==comp["response_document_id"])
    new=idx[comp["prospectus_document_id"]]
    assert comp["response_claims_preserved"]==old["reviewed_claims"]["holding_commitments"]
    assert comp["prospectus_claims"]==new["reviewed_claims"]["legacy_convertible_non_reduction_commitments"]
    assert old["known_at"]<new["known_at"]==comp["relationship_known_at"]
    assert not comp["independent_confirmation_of_2022_commitment"] and not comp["2022_commitment_revocation_proved"]
    assert not comp["2022_commitment_effective_period_resolved"] and comp["effective_sellable_new_shares"] is None
    with (OUT/"6份存量与歧义约束来源.csv").open(encoding="utf-8-sig",newline="") as h:rows=list(csv.DictReader(h))
    assert len(rows)==6 and all(json.loads(r["结构化字段"])==idx[r["公告ID"]]["reviewed_claims"] for r in rows)
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"document_id":rid,"page":p,
        "path":str(OUT/"page_previews"/f"{rid}_p{p}.png"),"sha256":digest(OUT/"page_previews"/f"{rid}_p{p}.png")} for rid,p in VISUAL_PAGES],
        "scope":"实际查看4页：浪潮2016年10月其他承诺条目与跨页6个月语句；节能风电2020非公开发行及2021可转债承诺日期、履行状态。"})
    receipt={"at":now(),"status":"PASS_KEYWORD_SCAN_COVERAGE_PARTIAL_READING_AND_SEPARATE_COMMITMENT_SCOPES",
        "keyword_scanned_documents":74,"keyword_scanned_pdf_pages":pdf_pages,"keyword_matches_covered_by_saved_candidates":matches,
        "saved_candidate_intervals_verified":381,"read_candidates":133,"pending_candidates":248,
        "new_source_facts":6,"prior_source_facts_unchanged":439,"literal_anchors_verified":anchors,"source_publication_queries_recomputed":12,
        "legacy_share_unit_conversion_checked":1,"current_issue_quantity_computed":0,"CSV_rows_recomputed":6,
        "result_sha256":digest(OUT/"result.json"),"review_code_sha256":digest(ROOT/"research/factor96_rights_prospectus_constraints_review_v1.py"),
        "meaning":"关键词扫描已覆盖固定文件文本，但人工条款阅读仅133段；未读248段不作无约束判断。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved();program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_RESPONSES_V1" and not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios","cumulative_invalid_implementation_account_scenarios",
        "last_completed_account_experiment","last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
        "qualified_candidates","orders_authorized","latest_rights_event_chains","rights_remaining_plan_admin_documents")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths];relative=OUT.relative_to(ROOT).as_posix()
    note=("固定74份正式配股说明书全文15747页完成关键词扫描，保存381段候选，实际已读133段、余248段。"
        "当前发行章节仅复现原18条一般持有期条款，前置7段为术语、担保物、套保或旧减持。"
        "后文新增6份存量/歧义约束来源10项字段，439份旧记录不变，累计445份。"
        "浪潮2020说明书中2016年10月条目含配股后6个月不减持，未核实所属配股；"
        "节能风电正式说明书相似承诺明确归于2021可转债并报告已履行，不能独立佐证2022配股回复，亦不能证明2022承诺被撤销。"
        "旧定增、重组、条件性限制与当前配股分开，新增确认的当次约束为0，可卖数量未知。T13未运行，目标未实现。")
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
            current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="PROSPECTUS_CONSTRAINT_SCOPE_REVIEW_248_CANDIDATES_PENDING_NOT_RUN",current_note=note,current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_PROSPECTUS_CONSTRAINTS_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_LEGACY_SCOPE_DISTINCTIONS_6_SOURCES_AND_248_CANDIDATES_PENDING",
        current_research_phase="T13_CURRENT_VS_LEGACY_HOLDING_CONTEXT_REVIEW_PENDING",
        reused_rights_pdf_documents_this_round=74,new_reviewed_rights_document_facts_this_round=6,new_holding_context_claim_fields_this_round=10,
        source_field_candidates_this_round=381,source_field_candidate_kind="74份正式说明书全文关键词381段，已读133段；6份存量或歧义条款来源、10项字段、4页原图。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",latest_rights_prospectus_context_facts=relative+"/reviewed_context_source_facts.json",
        latest_rights_prospectus_context_queries=relative+"/source_publication_queries.json",
        latest_rights_commitment_scope_comparison=relative+"/windpower_response_vs_prospectus_scope.json",
        latest_rights_pending_holding_candidates=relative+"/remaining_holding_candidates.json",rights_remaining_prospectus_constraint_candidates=248,
        next_independent_source_action="REVIEW_REMAINING_248_PROSPECTUS_HOLDING_CONTEXTS_AND_RESOLVE_COMMITMENT_IDENTITY",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",latest_progress_receipt=relative+"/saved_recomputation_receipt.json",
        last_research_result=note,last_source_result=note,latest_continuation_report=relative+"/研究进展.md",
        latest_continuation_classification=status["latest_continuation_classification"],research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、完整账户成本后夏普达到1.2的目标仍未实现。本轮核对正式配股说明书的持有约束来源，发现旧融资承诺与当次配股不能直接等同，保存6份文件的10项条款。没有新增回测、收益或模型，没有制作交付包。",
        "固定74份正式说明书与摘要，全文共15747页。按当前发行章节、前置内容、后续正文依次用同一组持有/限售/减持关键词扫描，共381段候选。已读当前章节18段、前置7段、后文优先98段及补充10段，共133段77183个规范化字符，另读必要跨页正文并查看4张原页。余248段未读；全文扫描不代表全文已核读。",
        "当前发行章节扩大关键词后仍只命中此前保存的18条持有期语句，没有新增当次约束。前置7段分别是大小非释义、客户担保物限售风险、铝价套保，以及国海旧股东已减持的记录，未计为本次配股新锁定。",
        "浪潮2020年3月说明书第106至107页中，浪潮集团‘其他承诺’条目时间写2016年10月、期限写长期有效，跨页第四项写配股完成后6个月内不减持。两页图像确认属于同一表格行。该条目对应哪次配股仍未确定，未直接归给2020年发行，也未倒填到2017年公开时点。",
        "节能风电2022年11月正式说明书第133至134页，将相似的发行前后各六个月不减持承诺明确归于2021年可转债，承诺日为2021年4月8日，表列期限2020年12月21日至2021年12月25日、已履行完毕。它与此前2022年7月配股回复的来源表述分别保存，不能独立佐证2022配股承诺的效力，也不能因后文未列同一条款就断定2022承诺撤销。需要对应2022年配股的原始承诺函或明确连接事项与期限的同期证据。",
        "东北证券2012年非公开发行60个月锁定含衍生股份安排；南山铝业2016重组中怡力电业旧股份明确延锁至2020年6月6日；宁波银行华侨银行2020年非公开发行股份为上市后五年不转让；东方证券同时有2017非公开发行48个月锁定和违反关联交易承诺后不得转让的条件性限制。它们可能与存量持股状态有关，但是否覆盖当前获配股份未确认，不转换为新增受限股数。",
        "新增6份后来源记录为445份，原439份不变；当前已确认的新当次配股持有约束为0。候选已读与未读分区、原文定位、六份来源可得时点以及东北旧认购万股单位已核对。旧36条实施事件和全部账户结果保持不变，达标候选0、独立前向观测0。",
        "下一步继续阅读固定文件剩余248段候选，并核对承诺对应的发行事项。完整发行覆盖、持有约束量化、历史成员与自由流通股分母仍未齐，严格M06未计算，T13继续NOT_RUN。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as h:h.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"已读候选":133,"待读候选":248,"来源字段":10,"来源总数":445,"原文锚点":receipt["literal_anchors_verified"],"目标实现":False},ensure_ascii=False))


if __name__=="__main__":record()
