"""固定126文档范围的已读字段及按公开时钟回放；不接入价格或收益。"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import csv
import json
import re

from research.factor96_rights_event_chains_v1 import OUT, SOURCE, read, save, digest, now, normalized

# 每行依次为原文ID、登记日、缴款起止日、价格分、计划最多配售股数。
# 日期逐项对应发行公告的认购方法或日程表；没有以停牌清算日代替缴款截止日。
NOTICE_ROWS = """
1201854885 2015-12-28 2015-12-29 2016-01-05 819 1560000000
1201906334 2016-01-14 2016-01-15 2016-01-21 424 1059140107
1202123520 2016-04-06 2016-04-07 2016-04-13 908 391433206
1202174216 2016-04-18 2016-04-19 2016-04-25 4568 39397971
1202315250 2016-05-13 2016-05-16 2016-05-20 622 529781184
1203195217 2017-03-29 2017-03-30 2017-04-07 687 726848101
1203566327 2017-05-31 2017-06-01 2017-06-07 717 505983018
1203573138 2017-05-31 2017-06-01 2017-06-07 717 505983018
1203703600 2017-07-18 2017-07-19 2017-07-25 1034 299784814
1204136107 2017-11-16 2017-11-17 2017-11-23 1369 262926000
1204217312 2017-12-15 2017-12-18 2017-12-22 1106 149153497
1204372704 2018-02-01 2018-02-02 2018-02-08 933 246446260
1204477665 2018-03-19 2018-03-20 2018-03-26 255 1566427405
1205513862 2018-10-23 2018-10-24 2018-10-30 170 2775330868
1205903343 2019-03-20 2019-03-21 2019-03-27 702 552167067
1205988166 2019-04-08 2019-04-09 2019-04-15 465 837241060
1207208640 2020-01-03 2020-01-06 2020-01-10 325 1264662591
1207347975 2020-03-10 2020-03-11 2020-03-17 1292 154710260
1207348215 2020-03-10 2020-03-11 2020-03-17 360 1554000000
1207356249 2020-03-12 2020-03-13 2020-03-19 680 899129490
1207919684 2020-06-16 2020-06-17 2020-06-23 500 848617545
1208008817 2020-07-09 2020-07-10 2020-07-16 746 1715702444
1208445105 2020-09-18 2020-09-21 2020-09-25 229 1705634462
1208521700 2020-10-13 2020-10-14 2020-10-20 544 1009634114
1208849529 2020-12-08 2020-12-09 2020-12-15 459 3463356846
1210115273 2021-06-01 2021-06-02 2021-06-08 368 1086315441
1210543902 2021-07-26 2021-07-27 2021-08-02 733 1090021619
1211627064 2021-11-23 2021-11-24 2021-11-30 1997 600801628
1211866504 2021-12-14 2021-12-15 2021-12-21 719 1151645218
1212175292 2022-01-18 2022-01-19 2022-01-25 1443 1597267249
1212338416 2022-02-15 2022-02-16 2022-02-22 450 159780000
1212698781 2022-03-30 2022-03-31 2022-04-08 680 1076704995
1212951464 2022-04-20 2022-04-21 2022-04-27 846 1670641224
1214276228 2022-08-16 2022-08-17 2022-08-23 520 2009001502
1215110334 2022-11-17 2022-11-18 2022-11-24 228 1503764986
1217034932 2023-06-14 2023-06-15 2023-06-21 202 5014409033
"""

# 金额为公告中有效认购总额，单位元；不使用扣费后的募集净额。
RESULT_ROWS = """
1201897631 1496665905 12257693761.95
1201900999 1496671674 12257741010.06
1201936414 1013743887 4298274080.88
1202179746 383286883 3480244897.64
1202249012 38785695 1771730547.60
1202335054 507908392 3159190198.24
1203274632 706270150 4852075930.50
1203603207 480765103 3447085788.51
1203736747 289969457 2998284185.38
1204165417 260230819 3562559912.11
1204253339 147696201 1633519983.06
1204415082 243570740 2272515004.20
1204527139 1515101106 3863507820.30
1204597968 1515678586 3864980394.30
1205564951 2699378625 4588943662.50
1205954988 545352788 3828376571.76
1206042055 833419462 3875400498.30
1207194799 335111438 2932225082.50
1207247024 1228983542 3994196511.50
1207383380 151866908 1962120451.36
1207383673 1485967280 5349482208.00
1207391500 880518908 5987528574.40
1207962044 761046394 3805231970.00
1208046671 1702997123 12704358537.58
1208506447 1655142470 3790276256.30
1208597439 998330844 5430919791.36
1208919065 3225083672 14803134054.48
1210209866 1076601364 3961893019.52
1210651292 1083382346 7941192596.18
1211763892 595574506 11893622884.82
1211991537 1126983743 8103013112.17
1212286046 1552021645 22395672337.35
1212443412 150525773 677365978.50
1212886504 1054713257 7172050147.60
1213214263 1502907061 12714593736.06
1214390207 1939315620 10084441224.00
1215213853 1462523613 3334553837.64
1217140784 4829739185 9756073153.70
"""

# 文号多义的五份文件分别含H股核准号或历史非公开发行号，以下仅取当次A股配股文号。
APPROVAL_OVERRIDES = {"1205903343": "2019_284", "1208008817": "2020_723",
                      "1212175292": "2021_3729", "1217034932": "2023_339",
                      "1208636972": "2020_762"}
EXPLICIT_CORRECTION_ROOTS = {"1201900999": ("2015_1631", "1201897631"),
                             "1203573137": ("2017_461", "1203566327")}
APPROVAL_RE = re.compile(r"证监许可[^0-9]{0,3}(20[0-9]{2})[^0-9]{0,3}([0-9]{1,5})号")
DATE_RE = re.compile(r"(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日")


def date_literal(value):
    y, m, d = map(int, value.split("-"))
    return f"{y}年{m}月{d}日"


class Document:
    def __init__(self, row):
        self.row = row
        self.pages = [(p["page"], normalized(p["text"])) for p in read(SOURCE / row["text_snapshot"])["pages"]]
        self.text = "\n".join(t for _, t in self.pages)

    def anchors(self, pattern, required=True):
        hits = []
        for page, text in self.pages:
            for match in re.finditer(pattern, text):
                hits.append({"document_id": self.row["document_id"], "page": page,
                             "normalized_start": match.start(), "normalized_end": match.end(),
                             "literal": match.group(0),
                             "context": text[max(0, match.start()-95):match.end()+145]})
        if required:
            assert hits, (self.row["document_id"], pattern)
        return hits

    def literal(self, value):
        return self.anchors(re.escape(value))

    def number(self, value, places=0):
        decimal = Decimal(str(value))
        raw = format(decimal, f".{places}f")
        grouped = format(decimal, f",.{places}f")
        choices = [raw, grouped]
        if decimal == decimal.to_integral_value():
            choices.extend([str(int(decimal)), format(int(decimal), ",")])
        pattern = "(?:" + "|".join(re.escape(x) for x in dict.fromkeys(choices)) + ")"
        return self.anchors(pattern)


def base_fact(doc, approval):
    row = doc.row
    return {"document_id": row["document_id"], "symbol": row["symbol"], "title": row["title"],
            "role": row["review_role"], "approval_key": approval,
            "event_id": f"{row['symbol']}_A_RIGHTS_{approval}" if approval else None,
            "known_at": row["catalogue_date"] + "T23:59:59+08:00",
            "catalogue_timestamp": row["catalogue_timestamp"], "source_url": row["source_url"],
            "raw_sha256": row["raw_sha256"], "text_sha256": row["text_sha256"],
            "raw_path": str(SOURCE / row["raw_snapshot"]), "text_path": str(SOURCE / row["text_snapshot"]),
            "historical_first_publication_verified": False, "trading_feature_admitted": False,
            "updates": {}, "evidence": {}, "notes": []}


def reviewed_facts():
    targets = read(OUT / "targets.json")
    assert digest(OUT / "targets.json") == read(OUT / "freeze.json")["targets_sha256"]
    notices = {r[0]: r[1:] for line in NOTICE_ROWS.strip().splitlines() if (r := line.split())}
    results = {r[0]: r[1:] for line in RESULT_ROWS.strip().splitlines() if (r := line.split())}
    facts, docs = [], {}
    for row in targets:
        assert digest(SOURCE / row["text_snapshot"]) == row["text_sha256"]
        doc = docs[row["document_id"]] = Document(row)
        keys = list(dict.fromkeys("_".join(m.groups()) for m in APPROVAL_RE.finditer(doc.text)))
        approval = APPROVAL_OVERRIDES.get(row["document_id"], keys[0] if len(keys) == 1 else None)
        if row["document_id"] in EXPLICIT_CORRECTION_ROOTS:
            approval = EXPLICIT_CORRECTION_ROOTS[row["document_id"]][0]
        fact = base_fact(doc, approval)
        if approval and row["document_id"] not in EXPLICIT_CORRECTION_ROOTS:
            year, number = approval.split("_")
            fact["evidence"]["approval_key"] = doc.anchors(r"证监许可[^0-9]{0,3}"+year+r"[^0-9]{0,3}"+number+"号")
        if len(keys) > 1:
            fact["notes"].append("同文档存在其他核准文号；依据当次A股配股段落选择，其他文号未用于创建发行事件。")
            fact["other_approval_keys"] = [key for key in keys if key != approval]
        if row["document_id"] in notices:
            record, start, end, cents, maximum = notices[row["document_id"]]
            assert record < start <= end
            fact["updates"] = {"record_date": record, "payment_start": start, "payment_end": end,
                               "issue_price_cents": int(cents), "plan_max_shares": int(maximum),
                               "plan_max_subscription_cents_calculated": int(cents)*int(maximum)}
            for field in ("record_date", "payment_start", "payment_end"):
                fact["evidence"][field] = doc.literal(date_literal(fact["updates"][field]))
            fact["evidence"]["issue_price_cents"] = doc.number(Decimal(cents)/100, 2)
            fact["evidence"]["plan_max_shares"] = doc.number(maximum)
            fact["notes"].append("计划股数乘发行价格为计算上限，不等于公告募集预算，不等于实际募集额；起止日之间仅按交易所允许时间认购。")
        if row["document_id"] in results:
            shares, amount = results[row["document_id"]]
            fact["updates"].update(actual_subscribed_shares=int(shares), actual_subscription_cents=int(Decimal(amount)*100))
            fact["evidence"]["actual_subscribed_shares"] = doc.number(shares)
            fact["evidence"]["actual_subscription_cents"] = doc.number(amount, 2)
            fact["updates"]["subscription_scope"] = "A_SHARE_TOTAL_REPORTED"
        if row["review_role"] == "LISTING_NOTICE":
            # 只识别已逐项阅读的“上市时间”字段；不提取公司概况中的IPO上市日期。
            hits = doc.anchors(r"上市时间[:为]?(20[0-9]{2})年([0-9]{1,2})月([0-9]{1,2})日")
            values = {datetime(*map(int, DATE_RE.search(h["literal"]).groups())).date().isoformat() for h in hits}
            assert len(values) == 1, (row["document_id"], values)
            fact["updates"]["announced_listing_date"] = next(iter(values))
            fact["evidence"]["announced_listing_date"] = hits
            fact["notes"].append("上市日期为公告安排；新增股份可能含限售或高管锁定部分，本字段不证明全量当天可卖或实际卖出。")
        if row["review_role"] in ("TERMINATION_OR_EXPIRY", "OVERSUBSCRIPTION_OUTSIDE_A_RIGHTS"):
            fact["event_id"] = None
        if row["document_id"] == "1208015874":
            fact["event_id"] = None
            fact["notes"].append("更正发行条件说明所引非公开发行监管比例20%为30%，正文明确不影响配股发行条件和方案；不据此修改2022年配股数量。")
            fact["evidence"]["no_issuance_terms_changed"] = doc.literal("上述更正内容不会对公司配股发行条件和发行方案产生影响")
        facts.append(fact)
    return facts, docs


def supplement_facts(facts, docs):
    by_id = {row["document_id"]: row for row in facts}
    original = by_id["1201854885"]
    original["updates"]["scheduled_result_publication_date"] = None
    original["source_conflicts"] = {"scheduled_result_publication_date": ["2015-01-07", "2016-01-07"]}
    original["evidence"]["conflicting_result_year"] = docs["1201854885"].literal("本次发行结果将于2015年1月7日")
    for key, (_, previous) in EXPLICIT_CORRECTION_ROOTS.items():
        by_id[key]["revises_document_id"] = previous
    by_id["1201900999"]["evidence"]["explicit_prior_notice_reference"] = docs["1201900999"].literal("临2016-003")
    by_id["1201900999"]["notes"].append("2016-01-08更正仅自目录日末生效；其中引用旧值不替代更正后表格。")
    by_id["1203573137"]["evidence"]["explicit_prior_notice_reference"] = docs["1203573137"].literal("2017年5月25日")
    by_id["1203566327"]["updates"].update(ratio_per_10="1.5627", plan_unrestricted_shares=500004386, plan_restricted_shares=5978632)
    for key in ("1203573137", "1203573138"):
        by_id[key]["updates"].update(ratio_per_10="1.56269", plan_unrestricted_shares=500004424, plan_restricted_shares=5978594)
        by_id[key]["notes"].append("更正配股比例及无限售/限售内部划分；总计划505983018股和价格7.17元未改变，不生成第二次发行。")
    for key in ("1203566327", "1203573137", "1203573138"):
        for field in ("ratio_per_10", "plan_unrestricted_shares", "plan_restricted_shares"):
            value = by_id[key]["updates"][field]
            by_id[key]["evidence"][field] = docs[key].literal(value) if isinstance(value, str) else docs[key].number(value)
    western = by_id["1203236200"]
    western["updates"] = {"corrected_reminder_rights_code": "082673"}
    western["revises_external_notice"] = {"date": "2017-03-30", "notice_number": "2017-013",
                                          "document_id": None, "standalone_original_obtained_in_this_scope": False}
    western["evidence"]["prior_notice"] = docs["1203236200"].literal("2017-013")
    western["evidence"]["old_wrong_code"] = docs["1203236200"].literal("086273")
    western["evidence"]["corrected_reminder_rights_code"] = docs["1203236200"].literal("082673")
    western["notes"].append("更正的是3月30日首次提示公告的配股代码；3月27日发行公告原已为082673，价格与缴款窗口无变更。")
    first, final = by_id["1204527139"], by_id["1204597968"]
    first["updates"].update(subscription_scope="DOMESTIC_INVESTORS_ONLY_CONNECT_PENDING",
                            connect_payment_dates=["2018-03-29", "2018-04-03", "2018-04-04"],
                            announced_expected_resumption_date="2018-04-10")
    first["evidence"]["connect_payment_dates"] = docs["1204527139"].literal("公司本次配股沪港通投资者缴款期为2018年3月29日、2018年4月3日以及2018年4月4日")
    first["evidence"]["expected_resumption"] = docs["1204527139"].literal("预计2018年4月10日复牌")
    first["notes"].append("境内认购结果同时新增沪股通分段缴款日，不能在原境内窗口结束后认定全部缴款结束。")
    final["updates"].update(connect_actual_subscribed_shares=577480, connect_payment_reported_complete=True)
    final["follows_document_id"] = "1204527139"
    final["evidence"]["connect_actual_subscribed_shares"] = docs["1204597968"].number(577480)
    final["notes"].append("第二份结果包含沪股通新增认购，为同次发行的新范围结果；不是重复发行，也不是机械更正旧境内值。")
    assert final["updates"]["actual_subscribed_shares"] - first["updates"]["actual_subscribed_shares"] == 577480
    # 本轮缺少天齐2019年的事前发行公告。价格仅在已取得的结果文档公开后可见。
    by_id["1207194799"]["updates"]["issue_price_cents"] = 875
    by_id["1207194799"]["evidence"]["issue_price_cents"] = docs["1207194799"].literal("每股人民币8.75元")
    by_id["1207194799"]["notes"].append("本轮固定范围缺少事前发行公告；没有从事后报告追溯填入过去的计划缴款窗口。")
    # 原文误写了市场后缀，维持目录证券身份并显式保留原文问题。
    by_id["1205995906"]["source_conflicts"] = {"body_stock_symbol": "002202.SH", "catalogue_symbol": "002202.SZ"}
    by_id["1205995906"]["evidence"]["body_symbol_conflict"] = docs["1205995906"].literal("002202.SH")
    by_id["1205995906"]["evidence"]["actual_exchange"] = docs["1205995906"].literal("深圳证券交易所")
    termination_states = {
        "1203106432": ("BOARD_TERMINATION_REPORTED", "董事会决定终止公司2016年度配股"),
        "1204046376": ("BOARD_TERMINATION_PENDING_SHAREHOLDERS", "本议案尚需提交至公司股东大会审议"),
        "1204930109": ("BOARD_TERMINATION_PENDING_SHAREHOLDERS", "该事项尚需提交公司2017年度股东大会审议"),
        "1204930111": ("SUPPORTING_OPINION_NOT_NEW_DECISION", "同意将相关议案提交公司2017年度股东大会审议"),
        "1205354373": ("BOARD_TERMINATION_WITH_PRIOR_AUTHORIZATION", "根据公司于2017年第一次临时股东大会审议通过的授权事项"),
        "1205825785": ("BOARD_TERMINATION_REGULATOR_WITHDRAWAL_PENDING", "公司申请撤回本次配股申请文件尚需取得中国证监会的同意"),
        "1206233867": ("APPROVAL_EXPIRED_UNIMPLEMENTED", "批复到期自动失效"),
        "1207329075": ("BOARD_TERMINATION_SHAREHOLDERS_NOT_REQUIRED", "无须提交公司股东大会审议"),
        "1208713385": ("BOARD_TERMINATION_SHAREHOLDERS_NOT_REQUIRED", "无须提交公司股东大会审议"),
        "1208713386": ("SUPPORTING_OPINION_NOT_NEW_DECISION", "我们同意终止本次配股"),
    }
    for key, (status, phrase) in termination_states.items():
        by_id[key]["administrative_status"] = status
        by_id[key]["evidence"]["administrative_status"] = docs[key].literal(phrase)
        by_id[key]["notes"].append("计划的终止或到期文件，本轮不生成实施配股的缴款/上市供给记录；待批准程序不自动当作最终批准。")
    for key in ("1202808343", "1206286002", "1206381460"):
        by_id[key]["administrative_status"] = "H_SHARE_IPO_OVERSUBSCRIPTION_OUTSIDE_A_RIGHTS"
        by_id[key]["evidence"]["scope"] = docs[key].literal("H股")
    # 认购结果与上市公告为同次发行不同阶段；上市增加股份单独从原文核对，不据以回写早期结果。
    grouped = defaultdict(list)
    for fact in facts:
        if fact["event_id"]:
            grouped[fact["event_id"]].append(fact)
    for event, rows in grouped.items():
        actual = [r for r in rows if "actual_subscribed_shares" in r["updates"]]
        assert actual, event
        latest = max(actual, key=lambda r: (r["known_at"], r["document_id"]))
        price_rows = [r for r in rows if "issue_price_cents" in r["updates"]]
        prices = {r["updates"]["issue_price_cents"] for r in price_rows}
        assert len(prices) == 1, event
        price = next(iter(prices))
        for row in actual:
            assert row["updates"]["actual_subscribed_shares"] * price == row["updates"]["actual_subscription_cents"], row["document_id"]
        for row in rows:
            if row["role"] == "LISTING_NOTICE":
                value = latest["updates"]["actual_subscribed_shares"]
                row["updates"]["listing_announced_total_shares"] = value
                row["evidence"]["listing_announced_total_shares"] = docs[row["document_id"]].number(value)
                row["listing_unrestricted_or_locked_breakdown_fully_reviewed"] = False
    return facts


def state_at(rows, at):
    current = {"as_of": at, "status": "NO_VIEW", "source_documents": [], "field_sources": {},
               "actual_subscribed_shares": None, "actual_subscription_cents": None,
               "announced_listing_date": None, "strict_M06_pressure": None,
               "trading_feature_admitted": False}
    eligible = [r for r in rows if r["known_at"] <= at]
    for row in sorted(eligible, key=lambda r: (r["known_at"], r["document_id"])):
        current["status"] = "REVIEWED_SOURCE_CLAIMS_DATE_PROXY_ONLY"
        current["source_documents"].append(row["document_id"])
        for key, value in row["updates"].items():
            current[key] = value
            current["field_sources"][key] = {"document_id": row["document_id"], "known_at": row["known_at"]}
    # 是否位于起止日之间只表示日期范围，不能证明当天/盘中允许缴款或供给压力已经退潮。
    day = at[:10]
    if "payment_start" in current:
        current["main_payment_calendar_phase"] = ("BEFORE_RANGE" if day < current["payment_start"] else
            "WITHIN_ANNOUNCED_DATE_RANGE" if day <= current["payment_end"] else "AFTER_ANNOUNCED_MAIN_RANGE")
    if "connect_payment_dates" in current:
        current["known_connect_payment_dates_on_or_after_day"] = [d for d in current["connect_payment_dates"] if d >= day]
    if current["announced_listing_date"]:
        current["listing_calendar_phase"] = ("ANNOUNCED_DATE_AHEAD" if day < current["announced_listing_date"]
                                             else "ANNOUNCED_DATE_REACHED_NOT_IMPLEMENTATION_PROOF")
    return current


def build():
    facts, docs = reviewed_facts()
    facts = supplement_facts(facts, docs)
    grouped = defaultdict(list)
    for row in facts:
        if row["event_id"]:
            grouped[row["event_id"]].append(row)
    events, boundaries = [], []
    for event_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: (r["known_at"], r["document_id"]))
        assert all(row["evidence"] for row in rows)
        dates = sorted({r["known_at"] for r in rows})
        for date in dates:
            before = (datetime.fromisoformat(date)-timedelta(seconds=1)).isoformat()
            previous, current = state_at(rows, before), state_at(rows, date)
            assert not set(r["document_id"] for r in rows if r["known_at"] == date) & set(previous["source_documents"])
            assert all(source["known_at"] <= date for source in current["field_sources"].values())
            boundaries.append({"event_id": event_id, "boundary": date, "before": previous, "at": current})
        notice_ids = [r["document_id"] for r in rows if r["role"] in ("ISSUANCE_NOTICE", "REVISED_ISSUANCE_NOTICE")]
        final = state_at(rows, dates[-1])
        assert final["listing_announced_total_shares"] == final["actual_subscribed_shares"]
        events.append({"event_id": event_id, "symbol": rows[0]["symbol"], "approval_key": rows[0]["approval_key"],
                       "document_ids": [r["document_id"] for r in rows], "issuance_notice_ids": notice_ids,
                       "initial_issuance_notice_present_in_scope": bool(notice_ids),
                       "first_source_known_at_in_scope": dates[0], "latest_source_known_at_in_scope": dates[-1],
                       "latest_reviewed_state": final, "full_intermediate_version_coverage": False,
                       "free_float_denominator": None, "strict_M06_pressure": None,
                       "trading_feature_admitted": False})
    assert len(events) == 36 and len(facts) == 126
    assert sum(bool(e["issuance_notice_ids"]) for e in events) == 35
    assert len([r for r in facts if "actual_subscribed_shares" in r["updates"]]) == 38
    # 针对真实来源差异验证：更正不倒填，新增沪股通窗口不倒填，2019天齐不回填事前信息。
    xingye = grouped["601377.SH_A_RIGHTS_2015_1631"]
    assert state_at(xingye, "2016-01-07T23:59:59+08:00")["actual_subscribed_shares"] == 1496665905
    assert state_at(xingye, "2016-01-08T23:59:59+08:00")["actual_subscribed_shares"] == 1496671674
    assert state_at(xingye, "2016-01-12T23:59:59+08:00")["announced_listing_date"] is None
    guanghui = grouped["600256.SH_A_RIGHTS_2018_157"]
    assert "connect_payment_dates" not in state_at(guanghui, "2018-03-27T23:59:59+08:00")
    assert state_at(guanghui, "2018-03-28T23:59:59+08:00")["connect_payment_dates"] == ["2018-03-29", "2018-04-03", "2018-04-04"]
    assert state_at(guanghui, "2018-04-09T23:59:59+08:00")["actual_subscribed_shares"] == 1515101106
    assert state_at(guanghui, "2018-04-10T23:59:59+08:00")["actual_subscribed_shares"] == 1515678586
    tbea = grouped["600089.SH_A_RIGHTS_2017_461"]
    assert state_at(tbea, "2017-05-26T23:59:59+08:00")["ratio_per_10"] == "1.5627"
    assert state_at(tbea, "2017-05-27T23:59:59+08:00")["ratio_per_10"] == "1.56269"
    tianqi = grouped["002466.SZ_A_RIGHTS_2019_1849"]
    assert state_at(tianqi, "2019-12-25T23:59:59+08:00")["status"] == "NO_VIEW"
    assert "payment_start" not in state_at(tianqi, "2020-01-02T23:59:59+08:00")
    result = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_EVENT_CHAINS_V1",
              "status": "FIXED_SCOPE_A_RIGHTS_IDENTITIES_AND_REVIEWED_CALENDAR_VERSIONS_COMPLETE",
              "reviewed_documents": len(facts), "original_local_documents_reused": len(docs),
              "source_candidate_clauses": sum(len(r["candidates"]) for r in read(OUT / "source_candidates.json")),
              "identified_A_rights_events": len(events), "events_with_initial_notice_in_scope": 35,
              "events_with_result_and_listing": 36, "result_amount_versions": 38,
              "issuance_calendar_documents": 36, "listing_announcements": 36,
              "publication_boundary_pairs": len(boundaries), "publication_boundary_queries": 2*len(boundaries),
              "administrative_source_statuses": dict(Counter(r["administrative_status"] for r in facts if "administrative_status" in r)),
              "unlinked_non_terms_correction_documents": ["1208015874"],
              "missing_initial_notice_events_in_scope": [e["event_id"] for e in events if not e["issuance_notice_ids"]],
              "full_M06_calendar_established": False, "historical_first_publication_verified": False,
              "free_float_denominator_established": False, "full_intermediate_version_coverage": False,
              "new_accounts": 0, "new_returns": 0, "new_models": 0, "new_network_requests": 0,
              "goal_status": "active", "goal_achieved": False, "T13": "NOT_RUN",
              "orders_authorized": False, "delivery_package_created": False, "external_review": "NOT_PERFORMED"}
    # 全部验证结束后才写输出，避免把部分构建误认为完整结果。
    save(OUT / "reviewed_document_facts.json", facts)
    save(OUT / "event_chains.json", events)
    save(OUT / "publication_boundary_queries.json", boundaries)
    save(OUT / "result.json", result)
    flat = [{"发行身份": e["event_id"], "证券代码": e["symbol"],
             "事前发行公告在本轮范围内": e["initial_issuance_notice_present_in_scope"],
             "首份已存来源时钟": e["first_source_known_at_in_scope"], "文件数": len(e["document_ids"]),
             "登记日": e["latest_reviewed_state"].get("record_date"),
             "主要缴款开始": e["latest_reviewed_state"].get("payment_start"),
             "主要缴款结束": e["latest_reviewed_state"].get("payment_end"),
             "额外沪股通缴款日": "|".join(e["latest_reviewed_state"].get("connect_payment_dates", [])),
             "最终已披露认购股数": e["latest_reviewed_state"]["actual_subscribed_shares"],
             "最终已披露认购金额元": str(Decimal(e["latest_reviewed_state"]["actual_subscription_cents"])/100),
             "公告安排上市日": e["latest_reviewed_state"]["announced_listing_date"],
             "严格M06压力": "NOT_COMPUTED", "交易特征准入": False,
             "文档ID": "|".join(e["document_ids"])} for e in events]
    with (OUT / "36组配股发行与日历.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save(OUT / "implementation_receipt.json", {"at": now(), "code_path": str(Path(__file__)),
         "code_sha256": digest(Path(__file__)), "checks": "原文锚点、38认购金额与股数乘价格、36上市数量、全部公开前后边界和四类具体历史差异",
         "new_accounts": 0})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    build()
