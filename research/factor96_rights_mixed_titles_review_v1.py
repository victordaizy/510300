"""登记混合标题正文中的实际股份、审批范围及股类转换，按披露时点保存。"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
import json

from research.factor96_rights_mixed_titles_v1 import OUT, PREVIOUS, ROOT, REUSED_ID, read, save, digest, now
from research.factor96_rights_event_chains_v1 import normalized
from research.factor96_rights_event_review_v1 import Document, base_fact
from research.factor96_rights_plan_admin_review_v1 import claim
from research.factor96_rights_preplans_review_v1 import anchor
from research.factor96_rights_prospectus_constraints_review_v1 import source_view

VISUAL_PAGES = [("1213526873", 1), ("1203474456", 2), ("1208196458", 2),
                ("1217394908", 1), ("1206163925", 2)]
CLASSES = {
    "1200900123": "OWN_H_OVERALLOTMENT_WITH_UNCHANGED_A_CAPITAL",
    "1200933220": "OWN_H_OVERALLOTMENT_WITH_UNCHANGED_A_CAPITAL",
    "1203474456": "OWN_H_OVERALLOTMENT_WITH_A_TO_H_CONVERSION",
    "1205194964": "H_APPLICATION_ACCEPTED_APPROVAL_PENDING",
    "1205674572": "H_APPROVAL_NOT_A_APPROVAL",
    "1206163925": "H_RIGHTS_RESULT_WITH_A_CONTEXT",
    "1206497156": "SUBSIDIARY_H_OVERALLOTMENT_EXPIRY_NOT_PARENT_RIGHTS",
    "1207054840": "H_APPROVAL_NOT_A_APPROVAL",
    "1208196458": "H_RIGHTS_RESULT_WITH_A_CONTEXT",
    "1212441974": "H_APPROVAL_NOT_A_APPROVAL",
    "1212493239": "H_RIGHTS_RESULT_WITH_A_CONTEXT",
    "1213526873": "H_RIGHTS_RESULT_WITH_A_CONTEXT",
    "1214874892": "SUBSIDIARY_H_OVERALLOTMENT_NOT_PARENT_RIGHTS",
    "1216315910": "H_APPROVAL_NOT_A_APPROVAL",
    "1217394908": "H_RIGHTS_RESULT_WITH_A_CONTEXT",
}


def specs():
    return {
      "1200900123": [
        claim("own_H_overallotment_terms", {"exercised_date": "2015-04-23", "new_H_shares": 65951600,
              "H_price_HKD": "16.80", "announced_H_listing_date": "2015-04-28", "listing_time": "09:00",
              "listing_state": "EXPECTED_AT_THIS_PUBLICATION"},
              (1, "已于2015年4月23日全部行使本公司H股招股说明书中所述的超额配股权"),
              (1, "共计65,951,600股H股"), (1, "作价每股H股16.80港元"),
              (2, "预计该等超额配售股份将于2015年4月28日上午九时开始在香港联交所主板上市及买卖")),
        claim("reported_share_table", {"scope": "OWN_H_OVERALLOTMENT", "A_before": 2002986332,
              "A_after": 2002986332, "A_change_calculated": 0, "H_before": 439679600,
              "H_after": 505631200, "total_before": 2442665932, "total_after": 2508617532},
              (2, "境内上市内资股(A股)2,002,986,33282.00%2,002,986,33279.84%"),
              (2, "境外上市外资股(H股)439,679,60018.00%505,631,20020.16%"),
              (2, "股份总数2,442,665,932100.00%2,508,617,532100.00%")),
      ],
      "1200933220": [
        claim("reported_H_listing", {"new_H_shares": 65951600, "reported_listing_date": "2015-04-28",
              "state": "SOURCE_REPORTS_ALREADY_LISTED", "actual_sales_not_proved": True},
              (1, "超额配售股份65,951,600股境外上市外资股(H股),已于2015年4月28日在香港联交所主板挂牌并开始上市交易")),
        claim("reported_share_table", {"scope": "OWN_H_OVERALLOTMENT", "A_before": 2002986332,
              "A_delta": 0, "A_after": 2002986332, "H_before": 439679600, "H_delta": 65951600,
              "H_after": 505631200, "total_before": 2442665932, "total_delta": 65951600,
              "total_after": 2508617532},
              (1, "境内上市内资股(A股)2,002,986,33282.00%02,002,986,33279.84%"),
              (1, "境外上市外资股(H股)439,679,60018.00%+65,951,600505,631,20020.16%"),
              (1, "股份总数2,442,665,932100.00%+65,951,6002,508,617,532100.00%")),
      ],
      "1203474456": [
        claim("own_H_overallotment", {"exercised_date": "2017-04-28", "additional_H_issue_shares": 48933800,
              "announced_H_listing_date": "2017-05-09", "A_rights_issue": False},
              (1, "2017年4月28日"), (1, "要求公司额外发行48,933,800股H股股份"),
              (1, "该等股份预计于2017年5月9日在香港联交所主板上市交易")),
        claim("A_to_H_transfer", {"announced_A_account_cancellation_date": "2017-05-04",
              "shares": 4893380, "state_shareholders": 54, "A_before": 7521000000, "A_after": 7516106620,
              "A_delta": -4893380, "H_delta_from_conversion": 4893380,
              "total_share_delta_from_conversion": 0, "A_restricted_delta": -3036620,
              "A_unrestricted_delta": -1856760, "not_new_A_issuance": True},
              (1, "上述54家国有股东"),
              (2, "合计应转持股份数为4,893,380股,并将于2017年5月4日从各国有股东的A股证券账户中注销"),
              (2, "于超额配售股份上市前转为H股登记到社保基金会"),
              (2, "本公司A股股数为7,521,000,000股,本次变动后,本公司A股股数为7,516,106,620股"),
              (2, "有限售条件股份合计2,786,452,89031.98-3,036,620-3,036,6202,783,416,27031.94"),
              (2, "人民币普通股4,734,547,11054.33-1,856,760-1,856,7604,732,690,35054.31"),
              (3, "三、股份总数8,713,933,800100.008,713,933,800100.00")),
        claim("transfer_table_basis", {"existing_H_in_conversion_table": 1192933800,
              "initial_H_listing_shares_reported": 1144000000,
              "difference_equals_additional_H_issue": True,
              "table_already_includes_additional_H_issue": True,
              "do_not_add_overallotment_to_this_table_twice": True},
              (1, "1,144,000,000股境外上市外资股(H股)"),
              (2, "境外上市的外资股1,192,933,80013.691,192,933,80013.69")),
      ],
      "1205194964": [
        claim("H_application_acceptance", {"receipt_date": "2018-07-17", "acceptance_number": "181057",
              "share_class": "H", "CSRC_approval_state": "PENDING", "acceptance_is_not_approval": True},
              (1, "2018年7月17日,公司收到中国证监会出具的《中国证监会行政许可申请受理单》(受理序号:181057号)"),
              (1, "公司本次H股配股事项尚需获得中国证监会的核准")),
      ],
      "1205674572": [
        claim("H_approval", {"approval_key": "2018_2057", "receipt_date": "2018-12-17",
              "maximum_H_shares": 130012168, "maximum_not_actual": True, "A_approval_not_implied": True},
              (1, "2018年12月17日"), (1, "(证监许可)[2018]2057号"),
              (1, "获准向境外上市外资股股东配售不超过130,012,168股境外上市外资股")),
      ],
      "1206163925": [
        claim("H_subscription_result", {"planned_H_shares": 123511559, "actual_H_shares": 123511559,
              "basic_allotment": 109836772, "extra_allotment": 13674787,
              "extra_application_shares": 1254274911, "total_application_shares": 1364111683,
              "applications_not_actual_allotment": True, "payment_end": "2019-04-23",
              "announced_H_listing_date": "2019-05-03"},
              (1, "本次H股可配售股份数量为123,511,559股,实际配售股份数量为123,511,559股"),
              (1, "本次H股配股缴款已于2019年4月23日结束"), (1, "实际认配股份为109,836,772股"),
              (1, "额外H股配股实际可配售数量为13,674,787股"),
              (1, "所申请的额外H股配股股份1,254,274,911股最终将按照H股额外实际可配售数量13,674,787股进行分配"),
              (1, "H股配股有效申购和额外有效申请共计1,364,111,683股"),
              (2, "缴足股款的H股配股股份将于2019年5月3日开始买卖")),
        claim("reported_share_table", {"scope": "BOTH_A_AND_H_RIGHTS", "A_before": 2906142460,
              "A_delta": 545352788, "A_after": 3451495248, "H_before": 650060840,
              "H_delta": 123511559, "H_after": 773572399, "total_before": 3556203300,
              "total_delta": 668864347, "total_after": 4225067647},
              (2, "A股2,906,142,46081.72%545,352,7883,451,495,24881.69%"),
              (2, "H股650,060,84018.28%123,511,559773,572,39918.31%"),
              (2, "股份总数3,556,203,300100.00%668,864,3474,225,067,647100.00%")),
        claim("source_signature_anomaly", {"literal": "2019年4日29日", "normalized_signature_date": None,
              "catalogue_day_still_used": "2019-04-30", "silent_month_correction": False},
              (2, "2019年4日29日")),
      ],
      "1206497156": [
        claim("subsidiary_overallotment_expiry", {"parent": "中国国际海运集装箱(集团)股份有限公司",
              "issuer": "中集车辆(集团)股份有限公司", "share_class": "H", "expiry_date": "2019-08-02",
              "option_exercised": False, "not_parent_A_rights_termination": True},
              (1, "分拆下属子公司中集车辆(集团)股份有限公司"),
              (1, "中集车辆向国际包销商授出的超额配股权于稳定价格期间未获行使且已于2019年8月2日失效")),
      ],
      "1207054840": [
        claim("H_approval", {"approval_key": "2019_1946", "maximum_H_shares": 294120354,
              "HKEX_approval_state": "PENDING", "A_approval_not_implied": True},
              (1, "证监许可[2019]1946号"),
              (1, "配售不超过294,120,354股境外上市外资股"),
              (1, "本次发行上市尚需取得香港联交所的批准")),
      ],
      "1208196458": [
        claim("H_subscription_result", {"approval_key": "2019_1946", "planned_H_shares": 294120354,
              "actual_H_shares": 294120354, "basic_allotment": 285292771, "extra_allotment": 8827583,
              "extra_application_shares": 956409484, "total_application_shares": 1241702255,
              "applications_not_actual_allotment": True, "payment_end": "2020-08-11",
              "announced_H_listing_date": "2020-08-20", "retrospective_A_listing_date": "2020-07-31"},
              (1, "证监许可[2019]1946号"),
              (1, "本次H股可配售股份数量为294,120,354股,实际配售股份数量为294,120,354股"),
              (1, "H股配股发行已于2020年8月11日结束"), (1, "实际认配股份为285,292,771股"),
              (1, "共计956,409,484股"), (1, "额外H股配股申请中实际配股股份数量为8,827,583股"),
              (1, "H股配股有效申购和额外有效申请共计1,241,702,255股"),
              (1, "缴足股款的H股配股股份将于2020年8月20日开始在香港联交所买卖"),
              (2, "A股配股股票已于7月31日在上海证券交易所上市")),
        claim("reported_share_table", {"scope": "BOTH_A_AND_H_RIGHTS", "A_before": 5719008149,
              "A_delta": 1702997123, "A_after": 7422005272, "H_before": 980401180,
              "H_delta": 294120354, "H_after": 1274521534, "total_before": 6699409329,
              "total_delta": 1997117477, "total_after": 8696526806,
              "new_A_registered_as_unrestricted": True, "effective_A_sellability_unknown": True},
              (1, "无限售条件的流通股(A股)5,719,008,14985.371,702,997,1237,422,005,27285.34"),
              (1, "H股980,401,18014.63294,120,3541,274,521,53414.66"),
              (2, "合计6,699,409,329100.001,997,117,4778,696,526,806100.00")),
        claim("group_level_A_changes", {"招商局集团有限公司": {"before": 2886027221, "delta": 865808166, "after": 3751835387},
              "中国远洋海运集团有限公司": {"before": 510336550, "delta": 153100965, "after": 663437515},
              "scope": "实际控制人及主要股东集团汇总表", "not_automatically_mapped_to_legal_holder_commitments": True,
              "new_A_shares_subject_to_specific_holder_commitment": None},
              (2, "实际控制人及主要股东名称"),
              (2, "招商局集团有限公司A股2,886,027,22143.08865,808,1663,751,835,38743.14"),
              (2, "中国远洋海运集团有限公司A股510,336,5507.62153,100,965663,437,5157.63")),
      ],
      "1212441974": [
        claim("H_approval_and_A_pending", {"approval_key": "2022_348", "maximum_H_shares": 308124000,
              "validity_months_from_approval": 12, "approval_date_not_specified": True,
              "HKEX_approval_state": "PENDING", "A_CSRC_approval_state": "PENDING"},
              (1, "证监许可[2022]348号"), (1, "核准公司增发不超过308,124,000股境外上市外资股"),
              (1, "本批复自核准之日起12个月内有效"),
              (1, "本次H股配股发行上市尚需取得香港联交所的批准。本次A股配股发行尚需取得中国证监会核准")),
      ],
      "1213526873": [
        claim("H_subscription_result", {"approval_key": "2022_348", "planned_H_shares": 287582400,
              "actual_H_shares": 82428, "basic_allotment": 80436, "extra_allotment": 1992,
              "payment_end": "2022-05-20", "announced_H_listing_date": "2022-05-31",
              "retrospective_A_listing_date": "2022-05-13", "under_subscription_is_H_only": True},
              (1, "证监许可[2022]348号"),
              (1, "本次H股可配售股份数量为287,582,400股,实际配售股份数量为82,428股"),
              (1, "本次H股配股缴款已于2022年5月20日结束"), (1, "实际认配股份为80,436股"),
              (1, "共计1,992股"), (1, "缴足股款的H股配股股份将于2022年5月31日开始在香港联交所买卖"),
              (2, "公司本次A股配股股票已于2022年5月13日在上海证券交易所上市")),
        claim("reported_share_table", {"scope": "BOTH_A_AND_H_RIGHTS", "A_before": 5966575803,
              "A_delta": 1502907061, "A_after": 7469482864, "H_before": 1027080000,
              "H_delta": 82428, "H_after": 1027162428, "total_before": 6993655803,
              "total_delta": 1502989489, "total_after": 8496645292,
              "new_A_registered_as_unrestricted": True, "effective_A_sellability_unknown": True},
              (2, "无限售条件的流通股(A股)5,966,575,80385.311,502,907,0617,469,482,86487.91"),
              (2, "H股1,027,080,00014.6982,4281,027,162,42812.09"),
              (2, "股份总额6,993,655,803100.001,502,989,4898,496,645,292100.00")),
        claim("largest_holder_reported_holdings", {"actor": "申能(集团)有限公司", "A_before": 1767522422,
              "A_after": 2262428700, "source_reports_full_subscription_commitment_fulfilled": True,
              "holding_delta_not_promoted_to_explicit_locked_allotment": True},
              (2, "申能(集团)有限公司履行了其全额认购A股配股的相关承诺"),
              (2, "其所持公司A股股份由1,767,522,422股增至2,262,428,700股")),
      ],
      "1214874892": [
        claim("subsidiary_H_overallotment", {"parent": "万科企业股份有限公司", "issuer": "万物云空间科技服务股份有限公司",
              "exercised_date": "2022-10-22", "new_subsidiary_H_shares": 11334700,
              "price_HKD": "49.35", "announced_H_listing_date": "2022-10-26", "listing_time": "09:00",
              "parent_A_issue_not_implied": True, "parent_holding_pct_before_approx": "56.60",
              "parent_holding_pct_after_approx": "56.06"},
              (1, "拟分拆所属子公司万物云空间科技服务股份有限公司"),
              (1, "已于2022年10月22日部分行使"), (1, "合计11,334,700股万物云H股股份"),
              (1, "按每股万物云H股股份49.35港元"),
              (1, "持有万物云已发行股本的比例将由约56.60%降至约56.06%"),
              (2, "超额配发股份预计将于2022年10月26日上午9时正在香港联交所主板开始上市及买卖")),
      ],
      "1216315910": [
        claim("H_approval", {"approval_key": "2023_717", "maximum_H_shares": 1366200000,
              "validity_months_from_approval": 12, "approval_date_not_specified": True,
              "HKEX_approval_state": "PENDING", "A_approval_not_implied": True},
              (1, "证监许可〔2023〕717号"), (1, "核准本公司发行不超过1,366,200,000股境外上市外资股"),
              (1, "批复自核准之日起12个月内有效"), (1, "本公司H股配股发行上市尚需取得香港联交所的批准")),
      ],
      "1217394908": [
        claim("H_subscription_result", {"approval_key": "2023_717", "planned_H_shares": 1366200000,
              "actual_H_shares": 1366200000, "basic_application_allotted": 559900119,
              "extra_application_allotted": 74774340, "total_applications_allotted": 634674459,
              "underwriter_arranged_investors_allotted": 288031000, "underwriter_taken_shares": 443494541,
              "payment_end": "2023-07-19", "announced_H_listing_date": None,
              "retrospective_A_listing_date": "2023-07-06"},
              (1, "证监许可〔2023〕717号"),
              (1, "本次H股可配售股份数量为1,366,200,000股,实际已全部完成配售"),
              (1, "本行H股配股已于2023年7月19日完成股东认购缴款"),
              (1, "认购配股数额为634,674,459股"), (1, "认购配股数额为559,900,119股"),
              (1, "认购配股数额为74,774,340股"), (1, "上述配股股份将全部配售予申请人"),
              (1, "包销商已促成相关投资者认购288,031,000股H股配股股份"),
              (1, "包销责任即认购443,494,541股H股配股股份"),
              (2, "本公司本次A股配股股票已于2023年7月6日在上海证券交易所上市")),
        claim("reported_share_table", {"scope": "H_STAGE_AFTER_A_RIGHTS_COMPLETED",
              "A_before": 21544435963, "A_delta": 0, "A_after": 21544435963,
              "H_before": 4554000000, "H_delta": 1366200000, "H_after": 5920200000,
              "total_before": 26098435963, "total_delta": 1366200000, "total_after": 27464635963,
              "A_zero_in_this_stage_does_not_reverse_prior_A_issue": True},
              (1, "H股配股完成前H股配股数额(股)H股配股完成后"),
              (1, "A股无限售流通股21,544,435,96382.55%021,544,435,96378.44%"),
              (2, "H股无限售流通股4,554,000,00017.45%1,366,200,0005,920,200,00021.56%"),
              (2, "合计26,098,435,963100.00%1,366,200,00027,464,635,963100.00%")),
      ],
    }


def comparisons(facts, prior):
    new = {f["document_id"]: f for f in facts}
    old = {f["document_id"]: f for f in prior}
    pairs = [("1206163925", "1205995906"), ("1208196458", "1208080267"),
             ("1213526873", "1213301939"), ("1217394908", "1217194963"),
             (REUSED_ID, "1212332253")]
    results = []
    for rid, earlier in pairs:
        f = new.get(rid, old.get(rid)); a = old[earlier]
        assert f["known_at"] > a["known_at"]
        quantity = a["updates"]["listing_announced_total_shares"]
        if rid == REUSED_ID:
            observed = f["reviewed_claims"]["a_h_registration_classification"]["registered_new_unrestricted_A"]
            relation = "LATER_A_COUNT_MATCHES_EARLIER_WITH_SEPARATE_LOCK_TRANCHE"
        else:
            table = f["reviewed_claims"]["reported_share_table"]
            observed = table["A_delta"]
            relation = "H_STAGE_ZERO_A_DELTA_AFTER_PRIOR_A_ISSUE" if rid == "1217394908" else "LATER_A_COUNT_MATCHES_EARLIER"
        if rid == "1217394908":
            assert observed == 0 and quantity == 4829739185
        else:
            assert observed == quantity
        results.append({"symbol": f["symbol"], "H_source_document_id": rid,
            "H_source_known_at": f["known_at"], "earlier_A_listing_document_id": earlier,
            "earlier_A_known_at": a["known_at"], "existing_event_id": a["event_id"],
            "earlier_announced_A_listing_date": a["updates"]["announced_listing_date"],
            "earlier_new_A_shares": quantity, "H_table_A_delta": observed,
            "relation": relation, "later_document_adds_new_A_event": False,
            "earlier_A_event_overwritten": False, "effective_sellable_new_A_shares": None})
    return results


def finalize():
    for f in read(OUT / "freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    rows = read(OUT / "documents.json"); definitions = specs()
    assert len(rows) == 15 and all(r["status"] == "PDF_TEXT_SAVED" for r in rows)
    prior = read(PREVIOUS / "combined_source_facts.json")
    facts = []; scopes = []; reviews = []
    for raw in rows:
        rid = raw["document_id"]
        row = {**raw, "raw_snapshot": str(OUT / raw["raw_snapshot"]),
               "text_snapshot": str(OUT / raw["text_snapshot"])}
        doc = Document(row); pages = dict(doc.pages)
        scopes.append({"document_id": rid, "total_pages": len(pages), "full_document_read": True,
                       "read_text": [{"page": p, "literal": text} for p, text in doc.pages]})
        reviews.append({"document_id": rid, "symbol": row["symbol"], "title": row["title"],
                        "former_title_role": row["title_role"], "body_classification": CLASSES[rid],
                        "full_body_reviewed": True, "source_reused": rid == REUSED_ID,
                        "new_A_rights_event_admitted": False, "unknown_constraints_are_not_zero": True})
        if rid == REUSED_ID:
            continue
        f = base_fact(doc, None)
        f.update(reviewed_claims={}, claim_evidence={}, effective_sellable_new_shares=None,
                 strict_M06_pressure=None, full_body_reviewed=True,
                 body_classification=CLASSES[rid], source_date_basis=row["source_date_basis"],
                 notes=["上下文来源字段，不自动进入A股事件；仅从本公告目录日末起使用。",
                        "已读全部正文；仅结构化当前数量、类别、审批与时钟范围，其他叙述不自动成为模型输入。"])
        for c in definitions[rid]:
            f["reviewed_claims"][c["field"]] = c["value"]
            f["claim_evidence"][c["field"]] = [anchor(rid, pages, p, text) for p, text in c["proof"]]
        facts.append(f)
    assert len(facts) == 14 and len(prior) == 458
    assert not {f["document_id"] for f in facts} & {f["document_id"] for f in prior}
    reused = next(f for f in prior if f["document_id"] == REUSED_ID)
    views = [{"document_id": f["document_id"],
              "before": source_view(f, (datetime.fromisoformat(f["known_at"]) - timedelta(seconds=1)).isoformat()),
              "at": source_view(f, f["known_at"])} for f in facts + [reused]]
    cmp = comparisons(facts, prior)
    cms = next(f for f in prior if f["document_id"] == "1208080267")
    holder_gap = {"new_source": "1208196458", "commitment_source": "1208080267",
                  "new_source_known_at": "2020-08-19T23:59:59+08:00",
                  "group_A_changes": {"招商局集团有限公司": 865808166, "中国远洋海运集团有限公司": 153100965},
                  "named_commitment_actors": cms["updates"]["listing_disclosed_non_reduction_commitments"][0]["actors"],
                  "commitment_months": 6, "legal_holder_to_group_quantity_link": "NOT_ESTABLISHED",
                  "commitment_constrained_new_A_shares": None,
                  "reason": "集团汇总认购表与具体承诺法人名称不一致，不能直接把集团数量全部映射为三家法人的限售量。"}
    queue = deepcopy(read(PREVIOUS / "unresolved_holding_links.json"))
    queue["at"] = now()
    for item in queue["items"]:
        if item["id"] == "CITIC_INHERITED_RIGHTS_LOCK":
            item.update(status="EXPLICIT_QUANTITY_EXTENSION_AND_FIXED_MIXED_TITLE_SCOPE_REVIEWED",
                        need="15份既有混合标题已读完；保持3月3日前数量未知及旧收购总数冲突，未宣称全部其他版本完整。")
    queue["items"].append({"id": "CMS_GROUP_VS_LEGAL_COMMITMENT_ACTOR", "priority": 6,
                           "sources": ["1208196458", "1208080267"],
                           "need": holder_gap["reason"], "status": "IDENTITY_AND_QUANTITY_LINK_UNKNOWN"})
    queue["count"] = len(queue["items"])
    coll = read(OUT / "collection_result.json")
    coverage = {"at": now(), "prior_gap": str(PREVIOUS / "mixed_title_coverage_gap.json"),
                "fixed_catalogue_mixed_titles": 15, "reviewed": 15, "remaining_in_fixed_scope": 0,
                "body_classes": dict(Counter(CLASSES.values())),
                "H_results_with_A_context": 5, "full_issuance_coverage": False,
                "scope": "既有目录标题含配股或供股且被归类OTHER_FINANCING_OR_MIXED_TITLE的15份，不扩展为全部发行完整。"}
    result = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_MIXED_TITLES_V1",
              "status": "PROGRESS_FIXED_15_MIXED_TITLES_FULLY_REVIEWED_WITH_A_H_SCOPE_SEPARATION",
              "new_pdf_documents": 14, "reused_pdf_documents": 1, "new_pdf_pages": coll["new_pages"],
              "fully_read_documents": 15, "source_pages_read": coll["all_pages"],
              "visual_pages_reviewed": len(VISUAL_PAGES), "new_context_source_facts": 14,
              "prior_source_facts_unchanged": 458, "combined_source_facts": 472,
              "new_source_fields": sum(len(f["reviewed_claims"]) for f in facts),
              "raw_http_responses": coll["raw_http_responses"], "mixed_title_scope_remaining": 0,
              "A_H_comparisons": len(cmp), "new_A_events": 0, "actual_event_count_unchanged": 36,
              "orient_actual_new_H_shares": 82428, "guotai_A_to_H_transfer_shares": 4893380,
              "CMS_specific_holder_constraint_quantity_resolved": False,
              "new_accounts": 0, "new_returns": 0, "new_models": 0, "qualified_candidates": 0,
              "independent_forward_observations": 0, "effective_sellable_new_quantities_computed": 0,
              "full_issuance_coverage": False, "free_float_denominator_available": False,
              "strict_M06_pressure": "NOT_COMPUTED", "T13": "NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE",
              "orders_authorized": False, "delivery_package_created": False, "goal_achieved": False,
              "next_source_action": "RESOLVE_INSPUR_2016_COMMITMENT_ISSUE_IDENTITY_FROM_PRIOR_PLAN_SOURCES"}
    outputs = {"reviewed_source_facts": facts, "combined_source_facts": prior + facts,
               "source_publication_queries": views, "read_scope": scopes,
               "title_body_classifications": reviews, "A_H_source_comparisons": cmp,
               "cms_actor_quantity_gap": holder_gap, "mixed_title_coverage": coverage,
               "unresolved_holding_links": queue, "result": result}
    for name, obj in outputs.items():
        save(OUT / (name + ".json"), obj)
    print(json.dumps({"新增来源": 14, "累计来源": 472, "已读标题": 15,
                      "新字段": result["new_source_fields"], "新增A股事件": 0,
                      "原配股事件保留": 36, "目标实现": False}, ensure_ascii=False))


if __name__ == "__main__":
    finalize()
