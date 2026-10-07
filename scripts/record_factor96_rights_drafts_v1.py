"""核对申报稿来源字段及版本变化，更新研究进度并保持实际账户不变。"""
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

import pandas as pd

from research.factor96_rights_drafts_review_v1 import OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, VISUAL_PAGES
from scripts.record_factor96_rights_preplans_v1 import count_money
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    facts=read(OUT/"reviewed_draft_source_facts.json");prior=read(PREVIOUS/"combined_source_facts.json")
    combined=read(OUT/"combined_source_facts.json");result=read(OUT/"result.json")
    assert len(facts)==7 and len(prior)==414 and len(combined)==421
    assert combined[:414]==prior and combined[414:]==facts and len({f["document_id"] for f in combined})==421
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    idx={f["document_id"]:f for f in facts};old_idx={f["document_id"]:f for f in prior}
    pages_by_id={};anchors=0;money_checks=0;quantities=[]
    for f in facts:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        pages={p["page"]:normalized(p["text"]) for p in read(f["text_path"])["pages"]};pages_by_id[f["document_id"]]=pages
        assert f["event_id"] is None and f["updates"]=={} and not f["trading_feature_admitted"]
        for k in ("actual_issuance_calendar","actual_issue_price","actual_issuance_quantity","effective_sellable_new_shares","strict_M06_pressure"):
            assert f[k] is None
        c=f["reviewed_claims"]
        assert set(c)==set(f["claim_evidence"])
        for evidence in f["claim_evidence"].values():
            for e in evidence:
                assert pages[e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
        money_checks+=count_money(c)
        q=c["plan_quantity_scenario"];ratio=Decimal(c["plan_ratio_per_10"]["value"])
        calc=Decimal(q["basis_shares"])*ratio/10
        assert abs(Decimal(q["total_shares"])-calc)<1
        assert q["A_shares"]+(q["H_shares"] or 0)==q["total_shares"]
        assert c["plan_gross_proceeds_cap"]["scope"]==q["scope"] and q["conditional_on_basis"]
        quantities.append({"document_id":f["document_id"],"exact_basis_times_ratio":str(calc),"reported_total_shares":q["total_shares"],
            "scope":q["scope"],"rounding_rule_overridden":False})
        assert c["remaining_approvals"]==["SSE_REVIEW","CSRC_REGISTRATION"]
        assert c["final_price_not_determined"] and c["net_proceeds_not_determined"]
        calendar=c["calendar_template"]
        assert not calendar["actual_calendar_admitted"] and [calendar[k] for k in ("record_date","payment_start_date","payment_end_date")]==[None,None,None]
        assert (calendar["payment_relative_start"],calendar["payment_relative_end"],calendar["result_relative_day"])==(1,5,7)
        holding=c["general_holding_clause"]
        assert holding["regulatory_exceptions"] and holding["effective_sellable_shares"] is None
    reviewed=read(OUT/"reviewed_source_candidates.json")
    assert len(reviewed)==30
    for c in reviewed:assert c["reviewed"] and pages_by_id[c["document_id"]][c["page"]][c["start"]:c["end"]]==c["literal"]
    queries=read(OUT/"draft_publication_queries.json")
    assert len(queries)==7
    for q in queries:
        f=idx[q["document_id"]]
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        assert q["at"]["as_of"]==f["known_at"]
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        for side in ("before","at"):assert source_view(f,q[side]["as_of"])==q[side]
    pairs=read(OUT/"same_day_admin_comparisons.json");assert len(pairs)==5
    for p in pairs:
        f,a=idx[p["draft_id"]],old_idx[p["admin_id"]]
        assert f["known_at"]==a["known_at"]==p["relationship_known_at"] and f["symbol"]==a["symbol"]
    versions=read(OUT/"jianfa_version_comparisons.json");assert len(versions)==5
    for p in versions:
        a,b=idx[p["earlier_source"]],idx[p["later_source"]]
        assert a["known_at"]<b["known_at"]==p["relationship_known_at"] and not p["new_executed_supply_inferred"]
        ca,cb=a["reviewed_claims"],b["reviewed_claims"]
        assert p["gross_budget_change_cents"]==cb["plan_gross_proceeds_cap"]["amount_cents"]-ca["plan_gross_proceeds_cap"]["amount_cents"]
        assert p["conditional_A_quantity_change"]==cb["plan_quantity_scenario"]["A_shares"]-ca["plan_quantity_scenario"]["A_shares"]
        for k,v in p["changed_source_fields"].items():assert v["earlier"]==ca.get(k) and v["later"]==cb.get(k)
    assert [f["reviewed_claims"]["plan_gross_proceeds_cap"]["amount_cents"] for f in facts[1:]]==[850000000000,778769000000,778769000000,778769000000,498000000000,498000000000]
    assert versions[0]["gross_budget_change_cents"]==-71231000000
    assert versions[3]["gross_budget_change_cents"]==-280769000000 and versions[3]["conditional_A_quantity_change"]==-384892
    assert idx["1219892998"]["reviewed_claims"]["validity_extension_shareholder_approval"]=="PENDING"
    assert idx["1221204347"]["reviewed_claims"]["resolution_validity"]["calendar_end"]=="2025-05-21"
    bank=read(OUT/"citic_bank_preplan_comparison.json")
    assert bank["budget_unchanged"] and bank["quantity_unchanged"]
    assert old_idx[bank["preplan_id"]]["known_at"]<idx[bank["draft_id"]]["known_at"]==bank["relationship_known_at"]
    profile=read(OUT/"jianfa_profile_and_basis_scope.json")
    c=idx[profile["source_id"]]["reviewed_claims"]
    assert c["issuer_profile_registered_capital"]["amount_cents"]==profile["registered_capital_cny"]*100
    assert c["plan_quantity_scenario"]["basis_shares"]==profile["quantity_scenario_basis_shares"]
    assert not profile["automatic_rebase_applied"]
    with (OUT/"7份申报稿发行条款.csv").open(encoding="utf-8-sig",newline="") as h:rows=list(csv.DictReader(h))
    assert len(rows)==7 and all(json.loads(r["结构化字段"])==idx[r["公告ID"]]["reviewed_claims"] for r in rows)
    remaining=read(OUT/"remaining_18_response_documents.json")
    assert len(remaining["documents"])==remaining["count"]==18 and not set(idx)&{r["document_id"] for r in remaining["documents"]}
    assert result["source_fields_reviewed"]==sum(len(f["reviewed_claims"]) for f in facts)==79
    assert sum(b-a+1 for a,b in (f["operative_page_range"] for f in facts))==62
    save(OUT/"conditional_quantity_checks.json",quantities)
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"document_id":rid,"page":n,
        "path":str(OUT/"page_previews"/f"{rid}_p{n}.png"),"sha256":digest(OUT/"page_previews"/f"{rid}_p{n}.png")} for rid,n in VISUAL_PAGES],
        "scope":"实际查看4页，确认银行及建发的一般持有条款及法规例外、建发77.8769亿元预算和延长至2025年5月21日的报告文字。"})
    receipt={"at":now(),"status":"PASS_NEW_DRAFT_FIELDS_VERSIONS_AND_NO_ACTUAL_CALENDAR",
        "new_PDF_text_pairs_verified":7,"prior_source_facts_unchanged":414,"literal_anchors_verified":anchors,
        "currency_unit_conversions_recomputed":money_checks,"conditional_quantity_scenarios_checked":7,
        "publication_queries_recomputed":14,"jianfa_version_pairs_recomputed":5,"same_day_update_pairs_checked":5,
        "general_holding_clauses_with_exceptions_retained":7,"blank_calendars_excluded":7,"CSV_rows_recomputed":7,
        "result_sha256":digest(OUT/"result.json"),"code_sha256":digest(ROOT/"research/factor96_rights_drafts_review_v1.py"),
        "interpretation":"新增来源字段、数量和版本变化一致；仍为未证实实施的申报稿，不是完整供给覆盖、有效可售数量或策略验证。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved();program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_USE_DILUTION_V1" and not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios","cumulative_invalid_implementation_account_scenarios",
        "last_completed_account_experiment","last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
        "qualified_candidates","orders_authorized","latest_rights_event_chains")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths];relative=OUT.relative_to(ROOT).as_posix()
    note=("完成7份申报稿当次发行概况30段候选及跨页条款，79项来源字段。建发6版保留85亿、77.8769亿、49.8亿预算变化，"
        "3.5股比例由上限改明确，历史股本测算数量1051809860改1051424968股；没有重复计算融资。"
        "2024年4月续期仍待股东大会，9月稿报告已于5月6日通过并延至2025-05-21，不回填更早时点。"
        "中信银行及建发共7份均披露法规例外下一般不设持有期，保留例外及有效可售数量未知。"
        "7份各自披露时点仍待审核注册，日历留白和相对日程均不生成实际发行。原414份不变，新增后421份，余18份监管回复。"
        "完整发行、较早持有期和自由流通分母仍未齐，T13未运行。")
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
            current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="7_DRAFTS_REVIEWED_18_RESPONSE_DOCUMENTS_PENDING_NOT_RUN",current_note=note,current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_DRAFTS_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_7_DRAFT_VERSIONS_HOLDING_EXCEPTIONS_AND_18_REPLIES_REMAINING",
        current_research_phase="T13_18_RESPONSES_EARLIER_HOLDING_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=7,new_reviewed_rights_document_facts_this_round=7,new_draft_source_claim_fields_this_round=79,
        source_field_candidates_this_round=30,source_field_candidate_kind="7份申报稿62页章节范围中的30段相关候选及补读，79项字段、4页原图；非2237页全文阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",latest_rights_draft_source_facts=relative+"/reviewed_draft_source_facts.json",
        latest_rights_draft_publication_queries=relative+"/draft_publication_queries.json",latest_rights_draft_version_comparisons=relative+"/jianfa_version_comparisons.json",
        latest_rights_remaining_documents=relative+"/remaining_18_response_documents.json",rights_remaining_plan_admin_documents=18,
        next_independent_source_action="T13_18_REGULATORY_RESPONSE_REPORTS_AND_EARLIER_HOLDING_COVERAGE",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",latest_progress_receipt=relative+"/saved_recomputation_receipt.json",
        last_research_result=note,last_source_result=note,latest_continuation_report=relative+"/研究进展.md",
        latest_continuation_classification=status["latest_continuation_classification"],research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、完整账户成本后净夏普达到1.2的目标仍未实现。此次继续先完成15份用途和摊薄报告，再完成7份申报稿，共22份关键条款，待审长文件从40份减至18份。没有新增账户、收益或模型，合格候选和独立前向观测仍为0；没有制作交付包。",
        "7份申报稿包括中信银行1份和建发6个版本，共2237页。固定当次发行概况的62页章节范围，实际阅读30段候选及必要跨页内容，查看4张原页，保存79项来源字段；没有声明2237页或62页均逐页全文阅读。仅用目录日末作为可得日期代理，不把所引用会议、批准或财报日期当成公开时点。",
        "建发2023年6月申报稿写每10股不超过3.5股、拟募资不超过85亿元，以2023年3月末3005171030股测算1051809860股。2023年8月、9月两次稿将比例明确为3.5股，基准日改6月末，股本和测算数量未变，募资上限降至778769万元即77.8769亿元。2024年4月及9月稿预算降至498000万元即49.8亿元，以2023年末3004071338股测算1051424968股。五组相邻版本分别保存，变化不是新增五笔融资。",
        "建发2024年4月稿报告董事会提出将有效期延长12个月，但仍待拟于5月召开的股东大会。2024年9月稿报告相关议案已于5月6日获股东大会通过，并明确延至2025年5月21日。后者仅从9月该来源可得时点保存，不反向填入4月或声称本轮取得了5月6日当日公告。注册资本资料在9月稿写2947095201元，而数量情景仍明确使用2023年末股本3004071338股；两种字段分开，不擅自重新计算配股数量。",
        "中信银行2023年3月稿保留400亿元A/H整体募集预算、每10股3股及2022年末股本测算。合计14680453097股，其中A股10215804204股、H股4464648893股，与此前二次修订预案数字相同。银行监管部门751号批复与仍待上交所审核和证监会同意注册分别保存；论证分析报告拟提交4月12日股东会议也未当作已经通过。",
        "7份申报稿在各自披露时点均写明尚需上交所审核和证监会注册，日历中的年月日仍为空白，价格由未来协商确定。T或R相对交易日模板只保留相对位置，未补造登记、缴款或上市日期。7份同时明确除相关法律法规规定外不设持有期，新增的是带法规例外的一般条款，不是已确认全部可出售；有效可售股数仍为空。",
        "同日5份建发申报更新提示与对应长稿并列，旧事项来源不覆盖。前一阶段15份用途/摊薄报告中发现的4项原文冲突仍保留：万向62.6亿与同日62.50亿、6.84元假设价不能由同页预算股数复算，以及东吴两版表格的万元单位与数值不一致。前一阶段6个假设完成日期或月份和2个假设价格仍未进入实际发行。",
        "原414份来源不变，新增7份后421份；与本次继续开始时相比新增22份。新增14次申报稿可得时点查询、7个条件数量及版本变动已核对，之前的实际发行事件和账户未重跑。累计有效账户情景552、无效实现情景152、执行合计704保持不变。",
        "下一步处理余下18份监管回复中的当次发行条款及较早持有期来源。完整发行覆盖、历史成员和自由流通分母仍未齐，严格M06未计算，T13保持NOT_RUN。本轮推进的是可用来源，尚未产生满足夏普目标的策略。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as h:h.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"申报稿":7,"字段":79,"锚点":receipt["literal_anchors_verified"],"本次继续累计新增来源":22,"当前来源":421,"剩余回复":18,"目标实现":False},ensure_ascii=False))


if __name__=="__main__":record()
