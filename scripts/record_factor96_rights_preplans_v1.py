"""检查新增23份预案字段并更新研究进度；保留已有账户及实施事件结果。"""
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
import csv
import json
import re

import pandas as pd

from research.factor96_rights_preplans_review_v1 import (
    OUT, PREVIOUS, ROOT, read, save, digest, normalized, now, source_view, VISUAL_PAGES)
from scripts.record_factor96_remaining_changes_v1 import update


def count_money(value):
    count = 0
    if isinstance(value, dict):
        if {"reported_value","reported_unit","amount_cents"} <= value.keys():
            scale = {"元":Decimal(100),"万元":Decimal(1000000),"亿元":Decimal(10000000000)}[value["reported_unit"]]
            assert Decimal(value["reported_value"])*scale == value["amount_cents"]
            count += 1
        for v in value.values():
            count += count_money(v)
    elif isinstance(value, list):
        for v in value:
            count += count_money(v)
    return count


def verify_saved():
    facts = read(OUT/"reviewed_preplan_source_facts.json")
    prior = read(PREVIOUS/"combined_source_facts.json")
    combined = read(OUT/"combined_source_facts.json")
    result = read(OUT/"result.json")
    assert len(facts) == 23 and len(prior) == 376 and len(combined) == 399
    assert combined[:376] == prior and combined[376:] == facts
    assert len({f["document_id"] for f in combined}) == 399
    for f in read(OUT/"freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    all_pages, anchors, money_checks = {}, 0, 0
    quantity_checks = []
    for f in facts:
        assert digest(f["raw_path"]) == f["raw_sha256"] and digest(f["text_path"]) == f["text_sha256"]
        pages = {p["page"]:normalized(p["text"]) for p in read(f["text_path"])["pages"]}
        all_pages[f["document_id"]] = pages
        assert f["updates"] == {} and f["event_id"] is None and not f["trading_feature_admitted"]
        assert f["actual_record_date"] is None and f["actual_payment_calendar"] is None and f["actual_issue_price"] is None
        assert f["effective_sellable_new_shares"] is None and f["strict_M06_pressure"] is None
        assert set(f["reviewed_claims"]) == set(f["claim_evidence"])
        for es in f["claim_evidence"].values():
            for e in es:
                assert pages[e["page"]][e["normalized_start"]:e["normalized_end"]] == e["literal"]
                anchors += 1
        c = f["reviewed_claims"]
        money_checks += count_money(c)
        ratio = Decimal(c["plan_ratio_per_10"]["reported_value"])
        qp = f["claim_evidence"]["plan_quantity_scenarios"]
        number_evidence = "\n".join(e["literal"] for e in qp)
        for n,q in enumerate(c["plan_quantity_scenarios"]):
            for name in ("basis_shares","reported_total_shares","A_shares","H_shares"):
                value = q[name]
                if value is None:
                    continue
                if f["document_id"] == "1209640415" and name in ("reported_total_shares","A_shares"):
                    v = f["visual_evidence"][0]
                    assert v["value"] == value == 213040000 and v["page"] == 3 and v["visually_reviewed"]
                    assert digest(v["path"]) == v["sha256"]
                    assert "213,040,000" in v["transcribed_literal"]
                else:
                    assert f"{value:,}股" in number_evidence, (f["document_id"],name,value)
            date = re.search(r"\d{4}-\d{2}-\d{2}",q["basis_asof"])
            if date:
                yy,mm,dd = map(int,date.group().split("-"))
                assert any(f"{yy}年{mm}月{dd}日" in pages[e["page"]] for e in qp)
            calculated = Decimal(q["basis_shares"])*ratio/10
            residual = Decimal(q["reported_total_shares"])-calculated
            assert abs(residual) < 1
            if q["H_shares"] is not None:
                assert q["A_shares"] + q["H_shares"] == q["reported_total_shares"]
                assert c["plan_gross_proceeds"]["scope"] == "A_AND_H_COMBINED"
            else:
                assert q["A_shares"] == q["reported_total_shares"]
            quantity_checks.append({"document_id":f["document_id"],"scenario_index":n,
                "basis_times_ratio_exact":str(calculated),"reported_total_shares":q["reported_total_shares"],
                "reported_minus_exact":str(residual),"interpretation":"保留原文股数；差额用于检查量纲，不采用统一取整规则覆盖原文。"})
    assert len(quantity_checks) == 25
    reviewed = read(OUT/"reviewed_source_candidates.json")
    extras = read(OUT/"reviewed_additional_holding_candidates.json")
    assert len(reviewed) == 127 and len(extras) == 3
    for c in reviewed + extras:
        assert c["reviewed"] and all_pages[c["document_id"]][c["page"]][c["start"]:c["end"]] == c["literal"]
    assert dict(Counter(r["review_role"] for r in reviewed)) == result["candidate_role_counts"]
    idx = {f["document_id"]:f for f in facts}
    queries = read(OUT/"preplan_publication_queries.json")
    assert len(queries) == 23
    for q in queries:
        f = idx[q["document_id"]]
        assert q["before"]["status"] == "NO_VIEW" and q["before"]["reviewed_claims"] == {}
        assert q["at"]["as_of"] == f["known_at"]
        assert datetime.fromisoformat(q["at"]["as_of"])-datetime.fromisoformat(q["before"]["as_of"]) == timedelta(seconds=1)
        for name in ("before","at"):
            assert source_view(f,q[name]["as_of"]) == q[name]
    previous_idx = {f["document_id"]:f for f in prior}
    same_day = read(OUT/"same_day_admin_comparisons.json")
    assert len(same_day) == 20 and sum(r["quantity_not_numeric_in_admin"] for r in same_day) == 9
    for pair in same_day:
        f,a = idx[pair["preplan_id"]],previous_idx[pair["admin_id"]]
        assert f["known_at"] == a["known_at"] == pair["relationship_known_at"] and f["symbol"] == a["symbol"]
        assert pair["new_preplan_quantity_source"] == f["reviewed_claims"]["plan_quantity_scenarios"]
        assert pair["quantity_not_numeric_in_admin"] == a["reviewed_claims"].get("quantity_detail_referenced_not_numeric",False)
    # 关注经济意义明确的差异，而不是将所有文字差异都升级为新增供给。
    assert idx["1203978536"]["reviewed_claims"]["plan_gross_proceeds"]["amount_cents"] == 625000000000
    assert idx["1205272661"]["reviewed_claims"]["plan_gross_proceeds"]["amount_cents"] == 566000000000
    k1,k2 = (idx[r]["reviewed_claims"]["plan_quantity_scenarios"] for r in ("1207103022","1207129939"))
    assert k1[0]["A_shares"]-k2[0]["A_shares"] == 194768
    assert k1[1] == k2[1] and k1[1]["A_shares"] == 201570048
    assert idx["1215889023"]["reviewed_claims"]["plan_quantity_scenarios"] == idx["1215952169"]["reviewed_claims"]["plan_quantity_scenarios"]
    assert idx["1215889023"]["reviewed_claims"]["remaining_approvals"] != idx["1215952169"]["reviewed_claims"]["remaining_approvals"]
    versions = read(OUT/"preplan_version_comparisons.json")
    assert len(versions) == 5
    for pair in versions:
        a,b = idx[pair["earlier_source"]],idx[pair["later_source"]]
        assert a["plan_source_group"] == b["plan_source_group"] == pair["plan_source_group"]
        assert a["known_at"] < pair["relationship_known_at"] == b["known_at"]
        for field,values in pair["changed_source_fields"].items():
            assert values["earlier"] == a["reviewed_claims"].get(field)
            assert values["later"] == b["reviewed_claims"].get(field)
    for terminal in read(OUT/"terminal_source_comparisons.json"):
        end = previous_idx[terminal["terminal_source_id"]]
        assert terminal["relationship_known_at_no_earlier_than"] == end["known_at"]
        assert terminal["terminal_status"] == end["administrative_status"]
        assert not terminal["future_approval_backfilled"] and not terminal["joined_to_executed_event"]
    with (OUT/"23份配股预案条款.csv").open(encoding="utf-8-sig",newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 23
    for row in csv_rows:
        c = idx[row["公告ID"]]["reviewed_claims"]
        parsed = json.loads(row["其他来源字段"])
        parsed.update(plan_ratio_per_10=json.loads(row["配股比例"]),plan_quantity_scenarios=json.loads(row["数量情景"]),
                      plan_gross_proceeds=json.loads(row["募资预算"]))
        assert parsed == c
    remaining = read(OUT/"remaining_40_long_documents.json")
    assert remaining["count"] == len(remaining["documents"]) == 40
    assert not set(idx) & {r["document_id"] for r in remaining["documents"]}
    assert result["source_fields_reviewed"] == sum(len(f["reviewed_claims"]) for f in facts) == 129
    assert len({f["plan_source_group"] for f in facts}) == result["source_plan_groups"] == 18
    save(OUT/"quantity_arithmetic_checks.json",quantity_checks)
    save(OUT/"visual_review_receipt.json",{"at":now(),"pages":[{"document_id":rid,"page":n,
        "path":str(OUT/"page_previews"/f"{rid}_p{n}.png"),
        "sha256":digest(OUT/"page_previews"/f"{rid}_p{n}.png")} for rid,n in VISUAL_PAGES],
        "scope":"实际查看4页：中科缺失数字、万向预算表、科伦回购专户两情景、中信银行待审核注册。不是688页全文阅读。",
        "manual_numeric_recovery":{"document_id":"1209640415","page":3,"shares":213040000,
            "status":"VISIBLE_IN_ORIGINAL_PDF_MISSING_IN_OLD_TEXT","automatic_literal_recomputation":False}})
    receipt = {"at":now(),"status":"PASS_NEW_PREPLAN_SOURCES_UNITS_AND_PUBLICATION_QUERIES",
        "new_PDF_text_pairs_verified":23,"prior_source_facts_unchanged":376,"literal_anchors_verified":anchors,
        "currency_unit_conversions_recomputed":money_checks,"quantity_scenarios_checked":25,
        "A_H_additions_checked":6,"manual_visual_numeric_recoveries":1,"new_publication_queries_recomputed":46,
        "same_day_source_pairs_checked":20,"preplan_version_pairs_checked":5,"terminal_source_comparisons_preserved":3,
        "CSV_rows_recomputed":23,"prior_actual_event_outputs_unchanged_not_rerun":True,
        "result_sha256":digest(OUT/"result.json"),"code_sha256":digest(ROOT/"research/factor96_rights_preplans_review_v1.py"),
        "interpretation":"新增来源、单位和日期代理回放一致；一项数值经原图人工确认，不声称自动文本复算。不是完整供给覆盖、首次披露证明或策略检验。"}
    save(OUT/"saved_recomputation_receipt.json",receipt)
    return receipt


def record():
    assert not (OUT/"program_update_receipt.json").exists()
    receipt = verify_saved()
    program = ROOT/"reports/research/510300_factor96_program_v1"
    paths = [program/n for n in ("status.json","strategy_progress.json","factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in paths]
    status,strategies,factors,mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RIGHTS_PLAN_ADMIN_V1"
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    keys = ("cumulative_admitted_account_scenarios","cumulative_executed_account_scenarios",
            "cumulative_invalid_implementation_account_scenarios","last_completed_account_experiment",
            "last_completed_account_result","independent_forward_observations","completed_total_fixed_questions",
            "qualified_candidates","orders_authorized","latest_rights_event_chains")
    protected = {k:status[k] for k in keys}
    before = [{"path":str(p),"sha256":digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix()
    note = ("固定23份完整预案并完成127段相关候选及3段附加持有期候选复核，形成129项来源字段。"
        "20组同日说明与预案对照补齐9份仅提示数量变更的数字；原事项记录不覆盖。"
        "5组计划版本、3组后验终止对照分别保存。万向预算62.50亿变56.60亿；科伦回购专户两情景、6份A/H合计口径分开。"
        "中科三环原PDF确认213040000股上限，旧文本漏提数字保留；3个摊薄假设完成日排除。"
        "原376份来源不变，新增23份预案来源后399份；没有新增持有期条款确认、实施日历或账户。"
        "余40份长文件及较早持有期完整覆盖、其他发行方式、历史成员和自由流通分母待齐，T13仍NOT_RUN。")
    for row in strategies:
        if row["id"] == "T13":
            row.update(current_status="NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",current_evidence=note,
                       current_evidence_path=relative+"/result.json",source_gate_path=relative+"/result.json")
    for row in factors:
        if row["id"] == "M06":
            row.update(current_status="23_PREPLANS_REVIEWED_40_LONG_DOCUMENTS_PENDING_NOT_RUN",current_note=note,
                       current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_","source_field_candidates","reused_")):
            status[key] = 0
    status.update(at=now(),latest_round="510300_FACTOR96_RIGHTS_PREPLANS_V1",latest_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_source_result=note,
        latest_continuation_classification="PROGRESS_23_PREPLAN_CLAUSES_NINE_NUMERIC_GAPS_AND_FIVE_VERSION_PAIRS",
        current_research_phase="T13_40_LONG_REPORTS_DRAFTS_EARLIER_HOLDING_AND_FREE_FLOAT_PENDING",
        reused_rights_pdf_documents_this_round=23,new_reviewed_rights_document_facts_this_round=23,
        new_preplan_source_claim_fields_this_round=129,source_field_candidates_this_round=127,
        source_field_candidate_kind="23份完整预案127段候选及3段附加条款，129项来源字段；实际查看4页，非688页全文阅读。",
        admitted_account_scenarios_this_round=0,invalid_implementation_accounts_this_round=0,
        latest_rights_document_facts=relative+"/combined_source_facts.json",
        latest_rights_preplan_source_facts=relative+"/reviewed_preplan_source_facts.json",
        latest_rights_preplan_publication_queries=relative+"/preplan_publication_queries.json",
        latest_rights_preplan_version_comparisons=relative+"/preplan_version_comparisons.json",
        latest_rights_remaining_documents=relative+"/remaining_40_long_documents.json",
        rights_remaining_plan_admin_documents=40,
        next_independent_source_action="T13_40_FEASIBILITY_DILUTION_RESPONSE_AND_DRAFT_DOCUMENTS_AND_EARLIER_HOLDING_COVERAGE",
        goal_status="active",goal_achieved=False,delivery_package_required=False)
    assert all(status[k] == v for k,v in protected.items())
    mandate.update(current_round=status["latest_round"],current_protocol=relative+"/protocol.json",
        latest_progress_receipt=relative+"/saved_recomputation_receipt.json",last_research_result=note,last_source_result=note,
        latest_continuation_report=relative+"/研究进展.md",latest_continuation_classification=status["latest_continuation_classification"],
        research_execution_state=status["current_research_phase"],goal_status="active",goal_achieved=False)
    paragraphs = [
        "只操作510300、完整账户成本后净夏普达到1.2的目标仍未实现。本轮补充23份配股预案的当次发行条款，没有新增账户、收益或模型；合格候选和独立前向观测仍为0。遵照用户要求，不制作交付包。",
        "从既有63份待审长文件中固定全部23份完整预案，共688页；实际阅读127段相关候选、3段附加持有期候选及必要跨页条款，查看4张关键原页。不声称688页逐页全文阅读，也未验证企业经营预测和投资可行性。129项来源字段按各公告目录日末记录，未把会议或签署日期提前当作公开时点。",
        "20组同日修订说明与预案并列保存，其中9份修订说明原先只提示比例数量变化，预案补充了相应数值。这9份为天风、招商、山西、国元、江苏银行、中信证券、财通、浙商银行和中信银行。原58份事项来源记录完全保留，不在旧公告上假装存在未披露数字，不把同日两份文件算两次融资。",
        "万向2017年修订预案写拟募资不超过62.50亿元，2018年二次修订写预计56.60亿元，债券偿还用途从159000万元降至100000万元；比例由每10股不超过3股改为3股，825947836股测算不变。两份预案均明确尚需股东大会及证监会批准。2019年的核准失效记录保留，未将该计划升级为实际发行。",
        "科伦2019年11月与12月两版预案分别按扣除回购专户后的1430497345股和1429106145股测算，可配股数从200269628股变为200074860股。另列处置完回购专户股份后的201570048股替代情景，两个情景不能相加。与东吴2017方案、万向2017方案一样，后验终止或失效来源保持原样，关系仅从终止公告可得时点开放。",
        "中科三环原文本写成本次配股数量不超过股，漏掉数字。实际查看原PDF第3页可清楚读到213040000股，与1065200000股乘以2/10相符。该值作为原图人工确认字段保存，旧文本不修改；它是当时上限，不是后来的实际认购数量。",
        "6份A/H预案分别保留合计、A股和H股数量及整体募资预算，不把H股计入A股供给。浙商银行2022年的银保监复163号和中信银行2022年的751号批复，与仍待证监会核准的流程分开。中信银行2023年2月二次修订转为上交所审核及证监会同意注册，数量和400亿元预算未变，程序表述变化不代表已经获准实施。",
        "山西、国元、财通预案中的2020年6月30日、2019年12月31日、2021年11月30日只是摊薄回报测算假设日期，全部排除出实际日历。财通摊薄章节仍使用2020年末股本和1076700000股假设，发行概况则用2021年6月末股本和1076702852股，按章节范围保存，不混为同一个实际数字。",
        "附加持有期检索只找到万向两版的核准后6个月内发行窗口和国海对股东历史减持的描述，均不是新股锁定期。本轮没有新增确认的当次持有期条款；这不证明不存在约束。此前上市及说明书来源里的持有期记录不变，完整的更早披露覆盖仍未完成。",
        "原376份来源记录保持不变，新加23份后为399份。新增46次公告前后查询、25个数量情景的量纲检查及6组A/H加总检查通过；一项中科数值依赖已查看原图的人工确认。只校验新增字段，没有重跑已有36组实际发行事件和账户。累计有效账户情景552、无效实现情景152、执行合计704，合格候选仍为0。",
        "剩余40份包括9份用途可行性报告、6份摊薄说明、18份监管回复和7份申报稿。下一步继续这些文件的当次发行条款和更早持有期来源。完整发行方式、历史成员和自由流通分母仍有缺口，严格M06未计算，T13仍未进入正式回测。",
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
    print(json.dumps({"新增预案来源":23,"来源字段":129,"锚点核对":receipt["literal_anchors_verified"],
        "时点查询":46,"剩余长文件":40,"账户累计":status["cumulative_admitted_account_scenarios"],"目标达到":False},ensure_ascii=False))


if __name__ == "__main__":
    record()
