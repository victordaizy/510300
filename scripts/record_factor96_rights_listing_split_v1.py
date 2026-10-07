"""复算已保存分类及披露时点，并更新本地研究进度；不运行账户。"""
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd

from research.factor96_rights_listing_split_v1 import OUT, PREVIOUS, read, save, digest, normalized, now
from research.factor96_rights_listing_split_review_v1 import VISUAL_PAGES
from research.factor96_rights_event_review_v1 import state_at
from scripts.record_factor96_remaining_changes_v1 import update


def verify_saved():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"])==item["sha256"],item["path"]
    original={r["document_id"]:r for r in read(PREVIOUS/"combined_source_facts_with_restriction_addendum.json")}
    current=read(OUT/"combined_source_facts.json")
    by_id={r["document_id"]:r for r in current}
    assert len(current)==len(by_id)==318 and set(by_id)==set(original)
    reviews=read(OUT/"reviewed_listing_classifications.json")
    assert len(reviews)==len({r["document_id"] for r in reviews})==36
    targets={r["document_id"]:r for r in reviews}
    page_cache={}
    for rid,row in targets.items():
        assert digest(row["raw_path"])==row["raw_sha256"]
        assert digest(row["text_path"])==row["text_sha256"]
        page_cache[rid]={p["page"]:normalized(p["text"]) for p in read(row["text_path"])["pages"]}
    anchors=0

    def check_anchor(a):
        nonlocal anchors
        text=page_cache[a["document_id"]][a["page"]]
        start=a.get("normalized_start",a.get("start"))
        end=a.get("normalized_end",a.get("end"))
        assert text[start:end]==a["literal"]
        anchors+=1

    for row in reviews:
        fact=by_id[row["document_id"]]
        u,r=row["listing_reported_unrestricted_new_shares"],row["listing_reported_restricted_new_shares"]
        assert u+r==row["listing_announced_total_shares"]==fact["updates"]["listing_announced_total_shares"]
        for key in ("listing_reported_unrestricted_new_shares","listing_reported_restricted_new_shares"):
            assert row[key]==fact["updates"][key]
        for a in row["classification_evidence"]:
            check_anchor(a)
        for c in row["scope_exceptions"]+row["non_reduction_commitments"]:
            for a in c["evidence"]:
                check_anchor(a)
        assert row["effective_sellable_new_shares"] is None
        assert fact["updates"]["effective_sellable_new_shares"] is None
        assert not row["trading_feature_admitted"] and not row["historical_holding_commitment_full_version_coverage"]
    unchanged=extended=0
    for rid,old in original.items():
        fact=by_id[rid]
        if rid not in targets:
            assert fact==old
            unchanged+=1
            continue
        assert old["role"]=="LISTING_NOTICE"
        for key,value in old.items():
            if key not in {"updates","evidence","notes"}:
                assert fact[key]==value,(rid,key)
        for key,value in old["updates"].items():
            assert fact["updates"][key]==value,(rid,key)
        for key,value in old["evidence"].items():
            assert fact["evidence"][key]==value,(rid,key)
        assert fact["notes"][:len(old["notes"])]==old["notes"]
        for hits in fact["evidence"].values():
            for a in hits:
                check_anchor(a)
        extended+=1
    assert (unchanged,extended)==(282,36)
    originals=[c for row in read(OUT/"source_candidates.json") for c in row["candidates"]]
    reviewed=read(OUT/"reviewed_classification_candidates.json")
    assert len(originals)==len(reviewed)==93
    extras=read(OUT/"additional_constraint_candidates.json")
    reviewed_extras=read(OUT/"reviewed_additional_constraints.json")
    assert len(extras)==len(reviewed_extras)==43
    for a,b in zip(originals+extras,reviewed+reviewed_extras):
        assert b["reviewed"]
        for key,value in a.items():
            if key!="reviewed":
                assert b[key]==value
        check_anchor(b)
    groups=defaultdict(list)
    for fact in current:
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    saved=read(OUT/"publication_boundary_queries.json")
    assert len(saved)==293 and len(groups)==36
    for pair in saved:
        for side in ("before","at"):
            snapshot=pair[side]
            assert state_at(groups[pair["event_id"]],snapshot["as_of"])==snapshot
            assert not snapshot["trading_feature_admitted"] and snapshot["strict_M06_pressure"] is None
            assert all(v["known_at"]<=snapshot["as_of"] for v in snapshot["field_sources"].values())
        added={f["document_id"] for f in groups[pair["event_id"]] if f["known_at"]==pair["boundary"]}
        assert not added.intersection(pair["before"]["source_documents"])
    for row in reviews:
        facts=groups[row["event_id"]]
        before=(datetime.fromisoformat(row["known_at"])-timedelta(seconds=1)).isoformat()
        a,b=state_at(facts,before),state_at(facts,row["known_at"])
        for key in ("listing_reported_unrestricted_new_shares","listing_reported_restricted_new_shares"):
            assert key not in a and b[key]==row[key]
            assert b["field_sources"][key]["document_id"]==row["document_id"]
        if row["non_reduction_commitments"]:
            assert "listing_disclosed_non_reduction_commitments" not in a
            assert len(b["listing_disclosed_non_reduction_commitments"])==1
    changchun=state_at(groups["000661.SZ_A_RIGHTS_2016_460"],"2016-05-06T23:59:59+08:00")
    assert changchun["controller_voluntary_lock_commitment_months"]==6
    assert changchun["registered_unrestricted_new_shares"]==changchun["listing_reported_unrestricted_new_shares"]==38772000
    assert changchun["controller_actual_subscription_shares"]==8778110 and changchun["effective_sellable_new_shares"] is None
    for event in read(OUT/"event_chains.json"):
        facts=groups[event["event_id"]]
        assert event["latest_reviewed_state"]==state_at(facts,max(f["known_at"] for f in facts))
    with (OUT/"36份上市股份分类.csv").open(encoding="utf-8-sig",newline="") as handle:
        table=list(csv.DictReader(handle))
    assert len(table)==36
    for row in table:
        source=targets[row["公告ID"]]
        assert int(row["公告列示无限售新增"])+int(row["公告列示限售新增"])==int(row["新增总股数"])
        assert int(row["公告列示无限售新增"])==source["listing_reported_unrestricted_new_shares"]
        assert int(row["公告列示限售新增"])==source["listing_reported_restricted_new_shares"]
        assert row["即时可卖量"]=="未知，未计算"
    images=[OUT/"page_previews"/f"{rid}_p{page}.png" for rid,page in VISUAL_PAGES]
    save(OUT/"visual_review_receipt.json",{"at":now(),"scope":"已实际查看9张关键页，核对表格列及脚注；未声称逐页读完359页。",
        "pages":[{"path":str(p),"sha256":digest(p)} for p in images]})
    receipt={"at":now(),"status":"PASS_SAVED_CLASSIFICATION_SCOPE_AND_PUBLICATION_REPLAY",
        "source_pdf_text_pairs_verified":36,"literal_anchors_verified":anchors,
        "source_candidates_read_and_preserved":136,"quantity_sum_checks":36,
        "old_nonlisting_facts_unchanged":unchanged,"old_listing_facts_additively_extended":extended,
        "all_prior_updates_and_evidence_preserved":True,"saved_global_queries_recomputed":586,
        "specific_listing_before_at_checks":36,"CSV_rows_recomputed":36,"new_accounts":0,
        "result_sha256":digest(OUT/"result.json"),"code_sha256":digest(ROOT/"research/factor96_rights_listing_split_review_v1.py"),
        "interpretation":"仅证明保存字段、原文口径和日期代理回放一致；不证明首版公开时刻、完整卖出限制或策略有效。"}
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
    note=("36份既存A股配股上市公告完成原文分类复核，其中35份新增复核、1份长春复用。"
          "5份公告新增限售为正，31份按公告口径为零；各自两类相加均等于新增总数。"
          "另存11份股东6个月不减持条款、1份激励配股继续锁定安排，3份无限售口径含高管锁定、2份高管另行管理限定；4份同期转股独立。"
          "这里保存公告列示分类，未推算即时可卖量；长春既有承诺及登记字段原样保留。"
          "121份方案/审核文件、历史承诺全版本、其他发行方式及自由流通分母仍待齐，M06未计算、T13仍NOT_RUN。")
    for row in strategies:
        if row["id"]=="T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                       source_gate_path=relative+"/result.json",current_evidence_path=relative+"/result.json")
    for row in factors:
        if row["id"]=="M06":
            row.update(current_status="36_LISTING_REPORTED_SPLITS_AND_CONSTRAINTS_REVIEWED_NOT_RUN",current_note=note,
                       current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key]=0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_LISTING_SPLIT_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_36_LISTING_REPORTED_SPLITS_AND_ADDITIONAL_CONSTRAINTS",
        current_research_phase="T13_PLAN_ADMIN_VERSIONS_EARLIER_HOLDING_CLAUSES_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=36,new_classification_documents_reviewed_this_round=35,
        reused_classification_documents_this_round=1,new_extended_listing_facts_this_round=36,
        new_listing_six_month_non_reduction_claims_this_round=11,new_incentive_allotment_lock_claims_this_round=1,
        source_field_candidates_this_round=136,
        source_field_candidate_kind="36上市公告93段分类候选及43段追加约束均已阅读；9张关键页实际目检，非359页全文阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_event_chains=relative+"/event_chains.json",latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_publication_queries=relative+"/publication_boundary_queries.json",
        latest_rights_listing_classifications=relative+"/reviewed_listing_classifications.json",
        latest_rights_listing_constraint_review=relative+"/reviewed_additional_constraints.json",
        rights_remaining_plan_admin_documents=121,
        next_independent_source_action="T13_121_PLAN_ADMIN_VERSION_REVIEW_AND_EARLIER_HOLDING_CLAUSE_COVERAGE",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    for key,value in protected.items():
        assert status[key]==value
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs=[
        "只操作510300、成本后完整账户净夏普达到1.2的目标仍未实现。本轮补齐固定36组配股的上市公告分类口径，没有新增账户或绩效结果；合格候选与独立前向观测仍为0。按用户要求不制作交付包。",
        "本轮固定36份既存上市公告，其中35份新增股份分类复核，长春1份复用。文件共359页，实际阅读93段分类候选、43段追加约束，并查看9张关键页面；不声称全文逐页读完。没有新网络请求。",
        "36份均得到公告列示的新增无限售与限售数量，二者与原已核实的新增总股数逐份相加一致。23份按全数无限售明确语句取数，7份按明确两类增量取数，6份按股份变动表取数；没有直接用混入转股的总股本前后之差。",
        "5组存在公告列示的新增限售：东北证券2016年41662425股、长春高新2016年13695股、浪潮信息2017年1186709股、国轩高科2017年114575348股、天齐锂业2017年849930股。其余31份在各自公告分类口径中新增限售为0，这不是证明无其他持股限制。",
        "上市公告另存11组自上市日起6个月内不减持条款：兴业2016、太平洋、中金黄金、特变电工、天齐2017、新奥、广汇、南山、隆基、东吴2020及招商。适用股东逐个保留；东吴条款限定获配股份，招商限定A股，不统一改写成全部股东全部持股。未推算确切终止日和受限新增数量，也不把较晚上市公告的条款提前提供给发行公告日期。",
        "3份文件的无限售叙述或表格包含高管锁定股：天齐2017、国元2020、宁波银行2021。金风2019、天齐2019另明确高管认购股份按规定通过登记机构管理。节能风电2022表内新增股份按无限售列示，但脚注明确股权激励配股股份继续按原激励股票期限锁定；具体受限量及终止日未在本次核对中得到。",
        "因此使用新字段listing_reported_unrestricted_new_shares和listing_reported_restricted_new_shares保存公告口径，不统称最终登记可交易状态。此前长春登记字段和超达6个月承诺保留原值与原证据。全部36组effective_sellable_new_shares均保持未知，不能从公告无限售数量直接减去某个股东获配量就生成可卖供给。",
        "太平洋的存量225000000股限制、国元原定增346809486股计划解禁与本次配股分开。特变、南山、隆基及江苏的网下认购也不作为新增限售。隆基9209股、华安802股、财通170股、节能风电3324股同期可转债转股单列，不并入明确披露的配股总数。A/H发行只记录当次A股量，浙商披露的未完成H股不提前累计。",
        "318份来源文件数量不变：282份非上市记录完全不变，36份上市记录只增加新字段和说明，所有原有字段及证据不变。36组事件、293组公开边界的586次保存回放已复算；36份公告的新分类均在各自目录日末代理时点才出现，长春先前承诺仍存在。日期代理不等于经过验证的历史首版公开时间。",
        "本轮复算只支持数据来源、口径和时点一致，不代表T13已回测或达到夏普目标。累计有效账户情景552、无效实现情景152、执行总数704，合格策略仍为0。未新增模型、收益曲线或实盘授权。",
        "剩余121份方案及审核文件待核对；本轮新增的减持和锁定条款还需要核对是否在更早文件中已披露，并核实历史变更。其他发行方式、历史自由流通分母和成员覆盖也仍不完整。M06严格压力为NOT_COMPUTED，T13保持NOT_RUN，下一轮继续这些可独立推进的来源工作。",
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
    print(json.dumps({"来源锚点复算":receipt["literal_anchors_verified"],"公开时点回放":586,
        "新分类时点核对":36,"旧非上市事实不变":282,"旧上市事实补充":36,"账户新增":0,"合格候选":0},ensure_ascii=False))


if __name__=="__main__":
    record()
