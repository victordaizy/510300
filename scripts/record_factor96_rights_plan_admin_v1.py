"""复算计划事项字段与可得日期，登记本轮来源进展，不运行回测账户。"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd

from research.factor96_rights_plan_admin_v1 import OUT, PREVIOUS, read, save, digest, normalized, now
from research.factor96_rights_plan_admin_review_v1 import source_view, VISUAL_PAGES
from research.factor96_rights_event_review_v1 import state_at
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"])==item["sha256"]
    old=read(PREVIOUS/"combined_source_facts.json")
    new=read(OUT/"reviewed_admin_source_facts.json")
    all_facts=read(OUT/"combined_source_facts.json")
    assert len(old)==318 and len(new)==58 and len(all_facts)==376
    assert all_facts[:318]==old and all_facts[318:]==new
    assert len({f["document_id"] for f in all_facts})==376
    indexes={f["document_id"]:f for f in new}
    pages={}
    anchors=money_checks=0
    for r in new:
        assert r["event_id"] is None and not r["updates"] and not r["trading_feature_admitted"]
        assert not r["historical_first_publication_verified"]
        assert digest(r["raw_path"])==r["raw_sha256"] and digest(r["text_path"])==r["text_sha256"]
        texts={p["page"]:normalized(p["text"]) for p in read(r["text_path"])["pages"]}
        pages[r["document_id"]]=texts
        assert set(r["claim_evidence"])==set(r["reviewed_claims"])
        for hits in r["claim_evidence"].values():
            assert hits
            for hit in hits:
                assert texts[hit["page"]][hit["normalized_start"]:hit["normalized_end"]]==hit["literal"]
                anchors+=1

    def check_money(value):
        nonlocal money_checks
        if isinstance(value,dict):
            if "amount_cents" in value:
                unit=value["reported_unit"]
                assert Decimal(value["reported_value"])*{"元":100,"万元":1000000,"亿元":10000000000}[unit]==value["amount_cents"]
                money_checks+=1
            for child in value.values():
                check_money(child)
        elif isinstance(value,list):
            for child in value:
                check_money(child)

    for r in new:
        check_money(r["reviewed_claims"])
    candidates=[c for row in read(OUT/"source_candidates.json") for c in row["candidates"]]
    reviewed=read(OUT/"reviewed_source_candidates.json")
    assert len(candidates)==len(reviewed)==140
    for a,b in zip(candidates,reviewed):
        assert b["reviewed"]
        assert all(b[k]==v for k,v in a.items() if k!="reviewed")
        assert pages[b["document_id"]][b["page"]][b["start"]:b["end"]]==b["literal"]
        anchors+=1
    queries=read(OUT/"admin_publication_queries.json")
    assert len(queries)==58
    for pair in queries:
        fact=indexes[pair["document_id"]]
        for side in ("before","at"):
            view=pair[side]
            assert source_view(fact,view["as_of"])==view
            assert view["actual_event_id"] is None and view["actual_issuance_calendar"] is None
        assert pair["before"]["status"]=="NO_VIEW" and not pair["before"]["reviewed_claims"]
        assert pair["at"]["reviewed_claims"]==fact["reviewed_claims"]
    # 原36组实施来源逐份未改动，连同保存的回放作一次复算，保证计划不会混入实施字段。
    groups=defaultdict(list)
    for f in all_facts:
        if f["event_id"]:
            groups[f["event_id"]].append(f)
    assert len(groups)==36
    for pair in read(PREVIOUS/"publication_boundary_queries.json"):
        for side in ("before","at"):
            view=pair[side]
            assert state_at(groups[pair["event_id"]],view["as_of"])==view
    for case in read(OUT/"terminal_source_comparisons.json"):
        terminal=next(r for r in old if r["document_id"]==case["terminal_source_id"])
        assert next(r for r in all_facts if r["document_id"]==terminal["document_id"])==terminal
        assert case["relationship_known_at_no_earlier_than"]==terminal["known_at"]
        assert not case["earlier_active_status_inferred"] and not case["joined_to_executed_event"]
    for rid in ("1205782946","1214794484"):
        value=indexes[rid]["reviewed_claims"]["plan_A_H_quantity_scenario"]
        assert value["A_shares"]+value["H_shares"]==value["total_shares"]
    for rid in ("1203759865","1205782946"):
        value=indexes[rid]["reviewed_claims"]["gross_proceeds_cap_revision"]
        assert value["after"]["amount_cents"]<value["before"]["amount_cents"]
    tianqi=indexes["1206375058"]["reviewed_claims"]
    assert tianqi["quantity_change_shareholder_vote"]=="NOT_REQUIRED_BY_PRIOR_AUTHORITY"
    assert tianqi["validity_change_shareholder_vote"]=="PENDING"
    assert indexes["1203642857"]["reviewed_claims"]["profit_distribution_ratio_unresolved"]
    for r in new:
        if r["reviewed_claims"].get("quantity_detail_referenced_not_numeric"):
            assert not any(k in r["reviewed_claims"] for k in ("plan_quantity_scenario","plan_quantity_revision","plan_A_H_quantity_scenario"))
    with (OUT/"58份方案与审核事项.csv").open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==58
    for row in rows:
        assert json.loads(row["结构化来源字段"])==indexes[row["公告ID"]]["reviewed_claims"]
    remaining=read(OUT/"remaining_63_long_documents.json")
    assert remaining["count"]==len(remaining["documents"])==63
    assert not {r["document_id"] for r in remaining["documents"]}&set(indexes)
    images=[OUT/"page_previews"/f"{rid}_p{page}.png" for rid,page in VISUAL_PAGES]
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"path":str(p),"sha256":digest(p)} for p in images],
        "scope":"8张关键页已实际查看，确认前后列、A/H数量、分项批准状态及万向原页文字冲突；不声称全文147页均已逐页阅读。"})
    receipt={"at":now(),"status":"PASS_SAVED_ADMIN_FIELDS_UNITS_AND_PUBLICATION_REPLAY",
        "new_source_pdf_text_pairs_verified":58,"prior_source_facts_unchanged":318,
        "literal_anchors_verified":anchors,"currency_unit_conversions_recomputed":money_checks,
        "saved_admin_queries_recomputed":116,"existing_actual_event_queries_unchanged_and_recomputed":586,
        "terminal_comparisons_preserved":3,"new_actual_calendars":0,"CSV_rows_recomputed":58,
        "remaining_long_documents":63,"result_sha256":digest(OUT/"result.json"),
        "code_sha256":digest(ROOT/"research/factor96_rights_plan_admin_review_v1.py"),
        "interpretation":"来源字段、单位及目录日期代理回放一致；不是完整供给数据、真实首次公开证明或策略验证。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt=verify_saved()
    program=ROOT/"reports/research/510300_factor96_program_v1"
    paths=[program/name for name in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects=[read(p) for p in paths]
    status,strategies,factors,mandate=objects
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys=("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios",
          "cumulative_invalid_implementation_account_scenarios","last_completed_account_experiment",
          "last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
          "qualified_candidates","orders_authorized")
    protected={k:status[k] for k in keys}
    before=[{"path":str(p),"sha256":digest(p)} for p in paths]
    relative=OUT.relative_to(ROOT).as_posix()
    note=("剩余121份资料中完成58份事项公告的140段候选及关键补充原页复核，形成159项来源字段。"
          "新奥募资上限24亿改23亿，金风50亿改47.44183亿；A/H数量独立。"
          "方案提议、董事会授权、股东大会待审、监管核准/注册待办分别记录；4份历史资金再用途、5份申报更新不记新增供给。"
          "9份仅提示比例数量变更未列数值，仍保留未知。原318份来源不变，新增58份行政记录，共376份，未新建实际发行日历。"
          "余63份长预案/申报稿/回复等资料、较早持有期及完整历史版本、其他发行方式和自由流通分母待齐；T13仍NOT_RUN。")
    for row in strategies:
        if row["id"]=="T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                       current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for row in factors:
        if row["id"]=="M06":
            row.update(current_status="58_PLAN_ADMIN_NOTICES_REVIEWED_63_LONG_DOCUMENTS_PENDING_NOT_RUN",current_note=note,
                       current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_PLAN_ADMIN_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_58_PLAN_ADMIN_NOTICE_CLAIMS_AND_REMAINING_63_LONG_DOCUMENTS",
        current_research_phase="T13_LONG_PLAN_REPORT_VERSIONS_EARLIER_HOLDING_CLAUSES_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=58,new_reviewed_rights_document_facts_this_round=58,
        new_admin_source_claim_fields_this_round=159,source_field_candidates_this_round=140,
        source_field_candidate_kind="58份事项公告140段相关候选及补充原页；159项结构化声明；8张关键页目检，非147页全文阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_admin_source_facts=relative+"/reviewed_admin_source_facts.json",
        latest_rights_admin_publication_queries=relative+"/admin_publication_queries.json",
        latest_rights_remaining_documents=relative+"/remaining_63_long_documents.json",
        rights_remaining_plan_admin_documents=63,
        next_independent_source_action="T13_63_LONG_PLAN_REPORT_DOCUMENTS_AND_EARLIER_HOLDING_CLAUSE_COVERAGE",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k]==v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、成本后完整账户净夏普达到1.2的目标仍未实现。本轮完成58份方案和审核事项公告的来源复核，没有新增账户、收益或模型；合格候选与独立前向观测仍为0。按用户要求不制作交付包。",
        "上一轮补齐36份上市公告分类。本轮先固定剩余121份文件，从中选择58份事项公告（含一份此前被错分为预案的3页修订说明），其余63份长预案、回复报告和申报稿保留待审。阅读140段候选及必要的补充原页，形成159项结构化声明，实际查看8张关键页面；不声称147页逐页全文阅读。没有新网络请求。",
        "明确保存的计划变化包括：西部证券每10股不超过3股改为2.6股，按同一股本测算的上限838670886股改为726848101股；新奥拟募资上限240000万元改为230000万元，去掉补充流动资金用途；金风拟募资上限500000万元改为474418.30万元，A/H测算股数分别552167067和123511560，合计675678627。计划上限、条件股本测算和实际认购金额是不同字段。",
        "天齐2019方案更新股本基数后，测算股数由342615855变为342596383，同时将每10股不超过3股明确为3股。数量调整已获此前股东大会授权，无需再审；自动延长有效期条款的删除却仍需股东大会审议。按具体事项分别保存批准状态，不用单一已批准标签掩盖差异。浙商A/H数量与180亿元合计募资上限也分开，续期至2023-11-22仍是待表决的提议。",
        "3份公告只是申请延期回复；有效期续期公告也分别保留董事会提议、股东大会待审、以未来表决日起算等条件。国海2019年公告引用2018年股东大会确认过去决议一直有效的说法，只从2019年该公告的目录日期代理时点保存，不倒填2017年状态。发审委通过、国资委同意、证监会核准以及审核注册程序分别记录。",
        "9份修订说明只说比例数量已经明确，没有在该公告中给出数值，保持未知，不从之后的实际配股补齐。5份建发2023至2024年的申报更新公告均明确仍待上交所审核及证监会注册，不能创建实施日历。中信银行2023年3页文件只修改审核注册流程表述，未提供实际注册决定或登记日。",
        "4份历史募集资金再用途文件（巨化、隆基两份、中科三环）均排除出新增配股供给。它们处理已发行项目结余和待支付款项，不能把公告中回顾的往年募资再计一次。对东吴2017、万向数量方案和河钢2019补存三组后验来源对照，原有终止或失效记录完全不变，未合并到后来的已实施配股，也不向更早时点倒填关系。",
        "万向2017年6月公告直接写出股本基准2753159454股及配股上限825947836股；原页同时写每1股送红股2股，与前后股本变化不一致。原图确认该文字确实存在，保留来源冲突，直接记录明确披露的配股上限，不自行改写分红比例或据此生成除权收益。",
        "原318份来源记录逐份不变，58份新行政来源加入后共376份。新增116次公告前后查询和原36组实施事件的586次回放已复算；新行政来源不自动加入实施事件，不创建实际发行日历，严格M06仍未计算。校验支持字段、单位和日期代理一致，不代表首版公开证明、完整供给覆盖或策略验证。",
        "累计有效账户情景552、无效实现情景152、执行合计704，合格策略仍为0。剩余63份包括23份完整预案、18份回复报告、9份用途可行性报告、6份摊薄说明和7份申报稿；下一步继续其当次发行条款、较早持有期披露和版本关系。其他发行方式、历史成员及自由流通分母等缺口仍在，T13保持NOT_RUN。",
    ]
    with (OUT/"研究进展.md").open("x",encoding="utf-8") as handle:
        handle.write("\n\n".join(paragraphs)+"\n")
    for path,obj in zip(paths,objects):
        update(path,obj)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json",{"at":now(),"before":before,
        "after":[{"path":str(p),"sha256":digest(p)} for p in paths],"protected_counts_and_authority":protected,
        "new_accounts":0,"goal_achieved":False,"delivery_package_created":False})
    print(json.dumps({"新增来源":58,"结构化字段":159,"原来源不变":318,
        "来源锚点复算":receipt["literal_anchors_verified"],"新时点查询":116,"旧时点回放":586,
        "剩余长文件":63,"新账户":0,"合格候选":0},ensure_ascii=False))


if __name__=="__main__":
    record()
