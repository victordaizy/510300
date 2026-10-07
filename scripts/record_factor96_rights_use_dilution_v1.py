"""验证新增用途及摊薄来源记录，不消除原文矛盾，不重跑旧账户。"""
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json

import pandas as pd

from research.factor96_rights_use_dilution_review_v1 import (
    OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, VISUAL_PAGES)
from scripts.record_factor96_rights_preplans_v1 import count_money
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    facts=read(OUT/"reviewed_use_dilution_source_facts.json")
    prior=read(PREVIOUS/"combined_source_facts.json")
    combined=read(OUT/"combined_source_facts.json")
    assert len(facts)==15 and len(prior)==399 and len(combined)==414
    assert combined[:399]==prior and combined[399:]==facts
    assert len({f["document_id"] for f in combined})==414
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"])==f["sha256"]
    idx={f["document_id"]:f for f in facts}
    old_idx={f["document_id"]:f for f in prior}
    all_pages={};anchors=0;money_checks=0;alloc_checks=[];quantity_checks=[]
    for f in facts:
        assert digest(f["raw_path"])==f["raw_sha256"] and digest(f["text_path"])==f["text_sha256"]
        pages={p["page"]:normalized(p["text"]) for p in read(f["text_path"])["pages"]}
        all_pages[f["document_id"]]=pages
        assert f["updates"]=={} and f["event_id"] is None and not f["trading_feature_admitted"]
        for field in ("actual_issuance_calendar","actual_issue_price","actual_issuance_quantity","strict_M06_pressure"):
            assert f[field] is None
        assert set(f["reviewed_claims"])==set(f["claim_evidence"])
        for evidence in f["claim_evidence"].values():
            for e in evidence:
                assert pages[e["page"]][e["normalized_start"]:e["normalized_end"]]==e["literal"]
                anchors+=1
        c=f["reviewed_claims"];money_checks+=count_money(c)
        if "proposed_use_components" in c:
            total=sum(r["proposed_amount"]["amount_cents"] for r in c["proposed_use_components"])
            assert total==c["plan_gross_proceeds_cap"]["amount_cents"]
            alloc_checks.append({"document_id":f["document_id"],"proposed_components_sum_cents":total,
                "scope":"只加顶层用途或互斥明细；不重复加入小计、子用途及项目总投资。"})
        if "dilution_quantity_assumption" in c:
            q=c["dilution_quantity_assumption"]
            ratio=Decimal(q["ratio_per_10"])/10 if "ratio_per_10" in q else Decimal(q["ratio_percent"])/100
            for field in ("basis","new","post"):
                if field not in q:continue
                v=q[field];scale=10000 if v["reported_unit"]=="万股" else 1
                assert Decimal(v["reported_value"])*scale==v["nominal_shares"]
                assert Decimal(v["unit_resolution_shares"])==Decimal(scale)/(Decimal(10)**v["reported_decimal_places"])
            calc=Decimal(q["basis"]["nominal_shares"])*ratio
            residual=Decimal(q["new"]["nominal_shares"])-calc
            precision=Decimal(q["basis"]["unit_resolution_shares"])*ratio+Decimal(q["new"]["unit_resolution_shares"])
            assert abs(residual)<=precision
            post_residual=None
            if "post" in q:
                post_residual=q["post"]["nominal_shares"]-q["basis"]["nominal_shares"]-q["new"]["nominal_shares"]
                assert abs(post_residual)<=sum(Decimal(q[n]["unit_resolution_shares"]) for n in ("basis","new","post"))
            quantity_checks.append({"document_id":f["document_id"],"reported_new_minus_ratio_calculation":str(residual),
                "unit_precision_bound_shares":str(precision),"reported_post_minus_basis_and_new":post_residual,
                "actual_issuance_quantity_admitted":False})
            d=c["dilution_completion_assumption"]
            assert not d["actual_calendar_admitted"]
            if d["precision"]!="DAY":assert d["date"] is None
    assert len(alloc_checks)==9 and len(quantity_checks)==6
    z=idx["1209640416"]["reviewed_claims"]
    assert sum(r["proposed_amount"]["amount_cents"] for r in z["proposed_use_components"][:4])==z["ningbo_subtotal"]["amount_cents"]
    assert z["project_total_investment"]["amount_cents"]==89000000000
    assert z["plan_gross_proceeds_cap"]["amount_cents"]==72000000000
    reviewed=read(OUT/"reviewed_source_candidates.json")
    assert len(reviewed)==55 and read(OUT/"additional_holding_candidates.json")==[]
    for c in reviewed:
        assert c["reviewed"] and all_pages[c["document_id"]][c["page"]][c["start"]:c["end"]]==c["literal"]
    queries=read(OUT/"use_dilution_publication_queries.json")
    assert len(queries)==15
    for q in queries:
        f=idx[q["document_id"]]
        assert q["before"]["status"]=="NO_VIEW" and q["before"]["reviewed_claims"]=={}
        assert q["at"]["as_of"]==f["known_at"]
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"])==timedelta(seconds=1)
        for side in ("before","at"):assert source_view(f,q[side]["as_of"])==q[side]
    pairs=read(OUT/"same_day_preplan_comparisons.json")
    assert len(pairs)==14 and sum(p["numeric_budget_matches"] for p in pairs)==13
    for p in pairs:
        f,old=idx[p["source_id"]],old_idx[p["preplan_id"]]
        assert f["known_at"]==old["known_at"]==p["relationship_known_at"] and f["symbol"]==old["symbol"]
        assert p["source_amount_cents"]==f["reviewed_claims"][p["source_field"]]["amount_cents"]
        assert p["preplan_amount_cents"]==old["reviewed_claims"]["plan_gross_proceeds"]["amount_cents"]
        assert p["numeric_budget_matches"]==(p["source_amount_cents"]==p["preplan_amount_cents"])
    conflicts=read(OUT/"preserved_source_conflicts.json")
    assert len(conflicts)==4 and all(not r["correction_applied"] for r in conflicts)
    assert conflicts[0]["difference_cents"]==1000000000
    for rid in ("1204532638","1205275253"):
        c=idx[rid]["reviewed_claims"]
        assert c["dilution_table_amount_conflict"]["amount_cents"] is None
        assert c["dilution_assumed_proceeds"]["amount_cents"]==650000000000
    prices=read(OUT/"hypothetical_price_checks.json")
    assert len(prices)==2 and [p["within_half_cent_rounding"] for p in prices]==[True,False]
    for p in prices:
        c=idx[p["source_id"]]["reviewed_claims"]
        calc=Decimal(c["dilution_assumed_proceeds"]["amount_cents"])/100/c["dilution_quantity_assumption"]["new"]["nominal_shares"]
        assert calc==Decimal(p["budget_divided_by_reported_nominal_shares"])
        assert Decimal(p["reported_hypothetical_price"])-calc==Decimal(p["difference_cny_per_share"])
        assert not p["actual_price_admitted"]
    earlier=read(OUT/"earlier_budget_source_comparison.json")
    assert earlier["source_known_at"]<earlier["compared_later_admin_known_at"]==earlier["relationship_known_at"]
    assert earlier["numeric_budget_matches"] and not earlier["earlier_state_asset_approval_inferred"]
    corroboration=read(OUT/"zhongke_numeric_corroboration.json")
    assert corroboration["dilution_assumption_new_shares"]==corroboration["preplan_cap_visually_recovered_shares"]==213040000
    assert not corroboration["actual_quantity_admitted"]
    versions=read(OUT/"version_comparisons.json")
    assert len(versions)==4
    for p in versions:
        a,b=idx[p["earlier_source"]],idx[p["later_source"]]
        assert a["known_at"]<b["known_at"]==p["relationship_known_at"]
        for field,v in p["changed_source_fields"].items():
            assert v["earlier"]==a["reviewed_claims"].get(field) and v["later"]==b["reviewed_claims"].get(field)
    terminals=read(OUT/"terminal_source_comparisons.json")
    assert len(terminals)==2
    for t in terminals:
        end=old_idx[t["terminal_source_id"]]
        assert t["relationship_known_at_no_earlier_than"]==end["known_at"] and t["terminal_status"]==end["administrative_status"]
        assert not t["joined_to_executed_event"] and not t["future_approval_backfilled"]
    with (OUT/"15份用途与摊薄条款.csv").open(encoding="utf-8-sig",newline="") as handle:csv_rows=list(csv.DictReader(handle))
    assert len(csv_rows)==15
    assert all(json.loads(r["结构化字段"])==idx[r["公告ID"]]["reviewed_claims"] for r in csv_rows)
    remaining=read(OUT/"remaining_25_long_documents.json")
    assert remaining["count"]==len(remaining["documents"])==25
    assert not set(idx)&{r["document_id"] for r in remaining["documents"]}
    result=read(OUT/"result.json")
    assert result["source_fields_reviewed"]==sum(len(f["reviewed_claims"]) for f in facts)==53
    assert result["candidate_role_counts"]==dict(Counter(r["review_role"] for r in reviewed))
    save(OUT/"allocation_and_assumption_checks.json",{"proposed_allocations":alloc_checks,"quantity_assumptions":quantity_checks})
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"document_id":rid,"page":n,
        "path":str(OUT/"page_previews"/f"{rid}_p{n}.png"),"sha256":digest(OUT/"page_previews"/f"{rid}_p{n}.png")} for rid,n in VISUAL_PAGES],
        "scope":"实际查看7页，确认万向两版金额/假设价格、东吴两版单位冲突、长春用途表、中科用途表和数量；不是275页全文阅读。"})
    receipt={"at":now(),"status":"PASS_SOURCE_RECORDS_WITH_FOUR_CONFLICTS_PRESERVED",
        "PDF_text_pairs_verified":15,"prior_source_facts_unchanged":399,"literal_anchors_verified":anchors,
        "currency_unit_conversions_recomputed":money_checks,"proposed_use_totals_checked":9,
        "hypothetical_quantity_precision_checks":6,"hypothetical_price_checks":2,"source_conflicts_preserved":4,
        "saved_publication_queries_recomputed":30,"same_day_preplan_pairs_checked":14,
        "CSV_rows_recomputed":15,"new_actual_issuance_calendars":0,"old_accounts_rerun":False,
        "result_sha256":digest(OUT/"result.json"),"code_sha256":digest(ROOT/"research/factor96_rights_use_dilution_review_v1.py"),
        "interpretation":"保存字段、单位及日期代理回放一致，四项原文冲突仍未解决；不是来源完全无误、完整供给覆盖或策略验证。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved()
    program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/name for name in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths];status,strategies,factors,mandate=objects
    assert status["latest_round"]=="510300_FACTOR96_RIGHTS_PREPLANS_V1"
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios",
        "cumulative_invalid_implementation_account_scenarios","last_completed_account_experiment","last_completed_account_result",
        "independent_forward_observations","completed_total_fixed_questions","qualified_candidates","orders_authorized","latest_rights_event_chains")
    protected={k:status[k] for k in keys};before=[{"path":str(p),"sha256":digest(p)} for p in paths]
    relative=OUT.relative_to(ROOT).as_posix()
    note=("完成9份用途可行性报告和6份摊薄说明的55段候选及必要跨页条款，形成53项来源字段。"
        "9组拟用募集资金分项与总额分开核对，6个假设完成日期/月份、2个假设价、2份万股精度股数均不进入实际供给。"
        "原页确认并保留4项冲突：万向62.6亿与同日62.50亿、2018假设6.84元不能由同页预算股数复算、东吴两版金额表单位不一致。"
        "中科同日摊薄文本213040000股佐证预案原图数字，但保留假设与上限的不同角色。"
        "长春2015年10月用途预算早于对照的11月国资事项，不回填后来批准。原399份来源不变，新加15份后414份。"
        "余18份监管回复、7份申报稿；更早持有期、完整发行覆盖及自由流通分母仍待齐，T13未运行。")
    for r in strategies:
        if r["id"]=="T13":r.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
            current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for r in factors:
        if r["id"]=="M06":r.update(current_status="15_USE_DILUTION_REPORTS_REVIEWED_25_LONG_DOCUMENTS_PENDING_NOT_RUN",current_note=note,
            current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_USE_DILUTION_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_15_USE_DILUTION_REPORTS_FOUR_SOURCE_CONFLICTS_AND_25_REMAINING",
        current_research_phase="T13_18_RESPONSES_7_DRAFTS_EARLIER_HOLDING_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=15,new_reviewed_rights_document_facts_this_round=15,
        new_use_dilution_source_claim_fields_this_round=53,new_preserved_source_conflicts_this_round=4,source_field_candidates_this_round=55,
        source_field_candidate_kind="15份用途与摊薄报告55段候选及补充原页、53项来源字段、7张原页；非275页全文阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_use_dilution_source_facts=relative+"/reviewed_use_dilution_source_facts.json",
        latest_rights_use_dilution_publication_queries=relative+"/use_dilution_publication_queries.json",
        latest_rights_use_dilution_conflicts=relative+"/preserved_source_conflicts.json",
        latest_rights_remaining_documents=relative+"/remaining_25_long_documents.json",rights_remaining_plan_admin_documents=25,
        next_independent_source_action="T13_18_REGULATORY_RESPONSES_7_DRAFT_PROSPECTUSES_AND_EARLIER_HOLDING_COVERAGE",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、完整账户成本后净夏普达到1.2的目标仍未实现。本轮完成15份用途及摊薄资料的关键条款复核，没有新增账户、收益或模型。合格候选和独立前向观测仍为0；按用户要求没有制作交付包。",
        "本轮固定剩余40份中的全部9份用途可行性报告和6份摊薄说明，共275页。阅读55段候选及必要的跨页条款，实际查看7张关键页面，形成53项结构化来源字段。没有声明全文275页逐页阅读，没有验证企业经营预测和投资可行性结论。剩余文件减少到25份。",
        "9份用途报告的分项投入与募集预算逐项分开保存。长春的180000万元分为疫苗基地40000、新研发80000、补流60000；新奥项目总投资376305.87万元与拟用募集资金230000万元分开；中科项目总投资89000万元、拟用募集资金72000万元，宁波39000万元小计不重复加到四项明细。金风报告的整体预算474418.30万元与同日A/H预案对照，不直接算作A股融资金额。",
        "原页确认万向2017年摊薄说明写募集资金不超过62.6亿元，而同日预案与用途报告写62.50亿元，相差1000万元。2018年摊薄说明写56.60亿元、82594.78万股、假设价格6.84元；预算除以所列股数约6.852733元，不能以两位小数取整解释成6.84元。这两项矛盾保留原文，不自行改数字，也不转为实际金额或定价。",
        "东吴2018年3月和8月两版摊薄说明正文均写计划65亿元、假设到账650000万元，但第4页表格均将6,500,000,000标为万元。原页确认了这一万元单位与数值的冲突。问题表格的货币转换值保持空，不将异常数值计入融资，也不擅自把万元改为元。连同万向两项差异，本轮共保留4项未消解的来源冲突。",
        "6份摊薄说明的日期分别作为假设保存：新奥2017年10月31日、万向2017年10月31日和2018年10月31日、东吴2018年6月末和9月末、中科2021年6月。月或月末表述没有补造成精确日期，全部排除出实际发行日历。万向两版数量用万股两位小数，保留相当于100股的显示精度，不与预案精确股数强行相等；7.58元与6.84元均只是测算假设价。",
        "中科同日摊薄说明的文本和原图明确列出213040000股，与上轮从完整预案原图读到的上限相同；原预案文本的漏字仍保留。新的来源用于数值交叉佐证，摊薄假设和预案上限的角色不合并。长春2015年10月28日用途报告的18亿元预算早于对照的11月13日国资事项来源；这不能把后来国资同意提前，也没有证明历史最早公开。",
        "新增14组同日预案对照中13组预算数值一致，万向2017年冲突单独保存。4组文档版本变化及万向、东吴两组后验终止/失效关系也分别保存，既有终止状态未覆盖。没有新增确认的持有期条款；这一结果不表示没有约束。",
        "原399份来源记录完全不变，新增15份后共414份。新增30次公告前后查询、9组用途加总及6组股数精度检查通过，检查的是保存记录与原文/单位的一致性；四项原文冲突仍未解决。未重跑旧账户，累计有效账户情景552、无效实现情景152、执行合计704保持不变。",
        "下一步处理18份监管回复和7份申报稿中的当次发行条款及较早持有期证据。完整发行覆盖、历史成员和自由流通分母仍有缺口，严格M06未计算，T13仍NOT_RUN，不能把资料核对进展当成夏普目标已经实现。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as handle:handle.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,"after":[{"path":str(p),"sha256":digest(p)} for p in paths],
        "protected_counts_and_authority":protected,"new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"来源新增":15,"字段":53,"锚点":receipt["literal_anchors_verified"],"原文冲突保留":4,
        "剩余长文件":25,"目标实现":False},ensure_ascii=False))


if __name__=="__main__":
    record()
