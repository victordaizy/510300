"""复算监管回复保存字段，登记剩余数据条件，保持交易账户与既有结果不变。"""
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

import pandas as pd

from research.factor96_rights_responses_review_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, VISUAL_PAGES
from scripts.record_factor96_rights_preplans_v1 import count_money
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    facts=read(OUT/"reviewed_response_source_facts.json");prior=read(PREVIOUS/"combined_source_facts.json")
    combined=read(OUT/"combined_source_facts.json");result=read(OUT/"result.json")
    assert len(facts)==18 and len(prior)==421 and len(combined)==439
    assert combined[:421]==prior and combined[421:]==facts and len({f["document_id"] for f in combined})==439
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    idx={f["document_id"]:f for f in facts};old={f["document_id"]:f for f in prior}
    pages_by_id={};anchors=0;money_checks=0;quantities=[];allocation_sums=[]
    for f in facts:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        pages={p["page"]:normalized(p["text"]) for p in read(f["text_path"])["pages"]};pages_by_id[f["document_id"]]=pages
        assert f["event_id"] is None and f["updates"]=={} and not f["trading_feature_admitted"]
        for k in ("actual_issuance_calendar","actual_issue_price","actual_issuance_quantity","effective_sellable_new_shares","strict_M06_pressure"):
            assert f[k] is None
        c=f["reviewed_claims"];assert set(c)==set(f["claim_evidence"])
        for evidence in f["claim_evidence"].values():
            for e in evidence:
                assert pages[e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
        money_checks+=count_money(c)
        for q in c.get("plan_quantity_scenarios",[]):
            if q["basis_shares"] is not None:
                calc=Decimal(q["basis_shares"])*Decimal(q["ratio_per_10"])/10
                assert abs(Decimal(q["reported_A_shares"])-calc)<1
            else:calc=None
            assert q["conditional_on_basis"] and not q["actual_issuance_quantity"]
            quantities.append({"document_id":f["document_id"],"basis_times_ratio":str(calc) if calc is not None else None,
                "reported_A_shares":q["reported_A_shares"],"version":q["version"],"basis_missing_not_backfilled":calc is None})
        if "proposed_allocations" in c:
            total=sum(a["amount_cents"] for a in c["proposed_allocations"])
            reported=c.get("plan_gross_proceeds",c.get("regulatory_question_budget",{}).get("amount"))
            assert total==reported["amount_cents"]
            allocation_sums.append({"document_id":f["document_id"],"allocation_sum_cents":total,"reported_proposal_amount_cents":reported["amount_cents"],
                "budget_role":"REGULATORY_QUESTION" if "regulatory_question_budget" in c else "CURRENT_REPLY","actual_cash_inferred":False})
    assert len(quantities)==6 and sum(q["basis_times_ratio"] is not None for q in quantities)==5 and len(allocation_sums)==5
    reviewed=read(OUT/"reviewed_source_candidates.json");assert len(reviewed)==144
    for c in reviewed:assert c["reviewed"] and pages_by_id[c["document_id"]][c["page"]][c["start"]:c["end"]]==c["literal"]
    assert result["candidate_characters"]==sum(len(c["literal"]) for c in reviewed)
    queries=read(OUT/"response_publication_queries.json");assert len(queries)==18
    for q in queries:
        f=idx[q["document_id"]]
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        for side in ("before","at"):assert source_view(f,q[side]["as_of"])==q[side]
    pairs=read(OUT/"same_day_admin_comparisons.json");assert len(pairs)==15
    for p in pairs:
        f,a=idx[p["response_id"]],old[p["admin_id"]]
        assert f["known_at"]==a["known_at"]==p["relationship_known_at"] and f["symbol"]==a["symbol"]
    wind=read(OUT/"earlier_windpower_holding_source.json")
    wf=idx[wind["response_id"]]
    assert old[wind["listing_id"]]["updates"]==wind["listing_classification"]
    assert wind["earlier_than_saved_issuance_notice_days"]==130 and wind["earlier_than_saved_listing_days"]==153
    for n in ("announcement","listing"):
        assert datetime.fromisoformat(wind[n+"_known_at"])-datetime.fromisoformat(wind["response_known_at"])==timedelta(days=wind["earlier_than_saved_issuance_notice_days"] if n=="announcement" else wind["earlier_than_saved_listing_days"])
    assert wind["effective_sellable_new_shares"] is None and not wind["automatic_subtraction_from_listing_total"]
    assert wind["retrospective_comparison_only"] and not wind["first_ever_disclosure_verified"]
    for h in wf["reviewed_claims"]["holding_commitments"]:
        assert all(h[k] is None for k in ("calendar_start","calendar_end","restricted_new_shares"))
    assert not wf["reviewed_claims"]["question_summary_and_quoted_commitment_scope_difference"]["automatic_equivalence_or_broader_ban_inferred"]
    terminal=read(OUT/"dongwu_2017_terminal_context.json")
    assert terminal["relationship_known_at"]==old[terminal["terminal_document_id"]]["known_at"]>idx[terminal["response_id"]]["known_at"]
    assert terminal["old_termination_preserved"] and terminal["separate_from_dongwu_2020"]
    versions=read(OUT/"response_version_relationships.json");assert len(versions)==4
    for v in versions:
        a,b=idx[v["earlier_response"]],idx[v["later_response"]]
        assert a["symbol"]==b["symbol"] and a["known_at"]<b["known_at"]==v["relationship_known_at"]
        assert not v["new_issuance_inferred"] and not v["missing_field_means_revocation"]
    gold=old["1205782946"]["reviewed_claims"]["gross_proceeds_cap_revision"]
    deduction=idx["1205804813"]["reviewed_claims"]["financial_investment_budget_deduction"]["amount_cents"]
    assert gold["before"]["amount_cents"]-gold["after"]["amount_cents"]==deduction
    tq=idx["1206438323"]["reviewed_claims"];a,b=tq["plan_quantity_scenarios"]
    assert a["basis_shares"]-b["basis_shares"]==tq["historical_incentive_cancellation_not_new_lock"]["shares"]==64906
    assert a["reported_A_shares"]-b["reported_A_shares"]==19472
    assert b["reported_A_shares"]==idx["1206556837"]["reviewed_claims"]["plan_quantity_scenarios"][0]["reported_A_shares"]
    enn=idx["1203759861"]["reviewed_claims"]
    assert enn["regulatory_question_old_budget"]["amount"]["amount_cents"]-enn["plan_gross_proceeds"]["amount_cents"]==enn["working_capital_use_removed"]["amount_cents"]
    with (OUT/"18份监管回复条款.csv").open(encoding="utf-8-sig",newline="") as h:rows=list(csv.DictReader(h))
    assert len(rows)==18 and all(json.loads(r["结构化字段"])==idx[r["公告ID"]]["reviewed_claims"] for r in rows)
    remaining=read(OUT/"remaining_response_documents.json");assert remaining["count"]==0 and remaining["documents"]==[]
    assert result["source_fields_reviewed"]==sum(len(f["reviewed_claims"]) for f in facts)==62
    save(OUT/"quantity_and_allocation_checks.json",{"quantities":quantities,"allocation_sums":allocation_sums,
        "goldwind_budget_deduction_cross_source":{"earlier_admin_id":"1205782946","later_reply_id":"1205804813",
            "deduction_cents":deduction,"relationship_known_at":idx["1205804813"]["known_at"],"earlier_admin_unchanged":True},
        "tianqi_previous_to_revised_quantity":{"basis_decrease":64906,"scenario_share_decrease":19472,"actual_supply_delta_inferred":False}})
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"document_id":rid,"page":n,
        "path":str(OUT/"page_previews"/f"{rid}_p{n}.png"),"sha256":digest(OUT/"page_previews"/f"{rid}_p{n}.png")} for rid,n in VISUAL_PAGES],
        "scope":"实际查看节能风电第29、30页，确认承诺主体、前后六个月起止语义及不买卖总结和不减持原文差别；隆基第4页确认旧新基数和数量。"})
    receipt={"at":now(),"status":"PASS_RESPONSE_FIELDS_AMOUNTS_TIME_SCOPE_AND_EARLIER_HOLDING_SOURCE",
        "new_PDF_text_pairs_verified":18,"prior_source_facts_unchanged":421,"literal_anchors_verified":anchors,
        "currency_unit_conversions_recomputed":money_checks,"quantity_scenarios_checked":6,"quantities_with_explicit_basis_recomputed":5,
        "allocation_sums_recomputed":5,"publication_queries_recomputed":36,"candidate_fragments_verified":144,
        "same_day_notice_pairs_checked":15,"version_pairs_checked":4,"CSV_rows_recomputed":18,
        "result_sha256":digest(OUT/"result.json"),"review_code_sha256":digest(ROOT/"research/factor96_rights_responses_review_v1.py"),
        "interpretation":"通过的是来源字段、金额和可得时点复算；不是完整供给、有效可售量或交易策略验证。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved();program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_DRAFTS_V1" and not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios","cumulative_invalid_implementation_account_scenarios",
        "last_completed_account_experiment","last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
        "qualified_candidates","orders_authorized","latest_rights_event_chains")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths];relative=OUT.relative_to(ROOT).as_posix()
    note=("固定最后18份监管回复已完成相关条款阅读，初始137段和补充7段、62项字段、3张原页。"
        "新奥旧提问24亿与当次23亿回复分开；隆基和天齐保留修订前后基数数量；首创同业融资不归为自身发行。"
        "节能风电2022-07-08回复新增2组不减持承诺，分别覆盖持股5%以上股东/控股股东及相关方、全体董监高及一致行动人，"
        "相对发行期首日前六个月至发行完成后六个月；比已存发行公告早130天、上市公告早153天，首次公开仍未证明。"
        "原文不减持与提问/总结不买卖分别保留，受限股数与解禁日期未知，不从上市无限售分类减造可卖量。"
        "旧东吴2017方案终止保持。原421份不变，累计439份，固定18份待审文件归零；不等于全市场覆盖。"
        "完整发行覆盖、历史成员、持有约束量化和自由流通分母尚缺，T13仍未运行。")
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
            current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="18_RESPONSES_REVIEWED_EARLIER_HOLDING_SOURCE_ADDED_NOT_RUN",current_note=note,current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_RESPONSES_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_18_RESPONSE_CLAUSES_AND_ONE_EARLIER_HOLDING_SOURCE",
        current_research_phase="T13_HOLDING_COVERAGE_ISSUANCE_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=18,new_reviewed_rights_document_facts_this_round=18,new_response_source_claim_fields_this_round=62,
        new_earlier_holding_source_documents_this_round=1,source_field_candidates_this_round=144,
        source_field_candidate_kind="18份监管回复137段初始候选与7段统一补扫、62项字段、3张原页；非1373页全文核读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",latest_rights_response_source_facts=relative+"/reviewed_response_source_facts.json",
        latest_rights_response_publication_queries=relative+"/response_publication_queries.json",
        latest_rights_earlier_response_holding=relative+"/earlier_windpower_holding_source.json",
        latest_rights_remaining_documents=relative+"/remaining_response_documents.json",rights_remaining_plan_admin_documents=0,
        next_independent_source_action="T13_HOLDING_SOURCE_COVERAGE_AND_ISSUANCE_DENOMINATOR_GAPS",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",latest_progress_receipt=relative+"/saved_recomputation_receipt.json",
        last_research_result=note,last_source_result=note,latest_continuation_report=relative+"/研究进展.md",
        latest_continuation_classification=status["latest_continuation_classification"],research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、完整账户成本后净夏普达到1.2的目标仍未实现。本轮处理固定剩余18份监管回复，新增62项有原文定位的字段，并找到节能风电较早的不减持承诺。没有新增回测、账户收益、模型或实盘交易，也没有制作交付包。",
        "18份共1373页。固定关键词产生137段候选，另因首份回复存在‘不存在减持’而统一补扫18份，新增7段。已读这些144段及必要跨页内容，首份11页补充回复全文已读；没有宣称1373页全文或会计核查意见已获验证。另查看节能风电第29、30页及隆基第4页原图。",
        "节能风电2022年7月8日回复载明，两组主体分别承诺不减持且不安排减持计划：持股5%以上股东、控股股东中国节能及其一致行动人中节能资本，并覆盖有控制关系的关联方和一致行动人；全体董监高及其一致行动人。期间为发行定价基准日即发行期首日前六个月至发行完成后六个月。比库内2022年11月15日发行公告早130天，比12月8日上市公告早153天。正文说已经披露，但未给出可验证的首次公开日期，因此只按7月该来源日末代理保存。",
        "节能风电提问和总结写‘不再进行相关股票买卖’，逐字引用的承诺则写‘不减持’和不安排减持计划，两类表述分别保留，没有自动扩展买入禁令。后续上市公告将1462523613股归为无限售条件流通股，同时另有激励获配股份自愿锁定安排。七月承诺、十二月分类与激励约束是不同层次；缺少受限数量、重叠关系、具体起止日及后续变更，实际可卖新增股数仍未知。",
        "新奥2017年8月回复仍保留监管提问的24亿元旧预算，但前言和回复已改成23亿元，删除1亿元补流；另一个24.7亿元是此前已终止的非公开发行。浪潮31亿元四项用途、南山50亿元两项用途、东吴65亿元四项用途、华安回复四项用途与提问40亿元均分别复算。华安的40亿元保留为提问角色，未把提问独立当成实际融资凭证。",
        "隆基回复将2018年6月末2791679915股、每10股不超过3股、837503974股旧测算，调整为9月末2791680001股、每10股3股、837504000股。天齐保留1142052851股到1141987945股基数变化，以及342615855股到342596383股测算变化；64906股为激励回购注销。后续天齐回复虽再次报告342596383股，但未在该条款给出基数，未从早稿回填。所有这些仍是条件测算。",
        "金风回复明确因财务性投资扣减25581.70万元，与早先50亿元降至47.44183亿元公告差额一致。东吴当前回复属于2017配股方案，其后来终止来源保持，与2020方案分开。国海已经减持的旧股东、首创2015和2016增持承诺、同行融资金额、36个月类金融投入限制、铝价和外汇锁定等都没有计入当次配股的新增供给或锁定量。",
        "原421份来源记录未改，追加后439份。复算新增字段定位、金额单位、5组用途合计、6组数量情景中的5个明确基数，以及36次可得时点查询；同日提示与回复合并理解为同一事项。累计有效账户情景552、无效实现152、执行合计704不变，达标候选0、独立前向观测0。",
        "固定的18份监管回复待处理数归零，只代表这一批文件已处理，不代表全市场发行覆盖完整。下一步合并已有持有约束来源覆盖表，检查计划与实际事件的可用连接及其他发行类型，并继续处理历史成员和自由流通股分母缺口。严格M06仍未计算，T13保持NOT_RUN；接口凭据未变化时不重复请求。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as h:h.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"回复":18,"字段":62,"锚点":receipt["literal_anchors_verified"],"金额换算":receipt["currency_unit_conversions_recomputed"],
        "当前来源":439,"固定剩余回复":0,"目标实现":False},ensure_ascii=False))


if __name__=="__main__":record()
