"""把31份既有用途变更原文的明确条款落库，不推断新增回购或已完成注销。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_remaining_changes_v1"

# 序号沿用事先登记的全部31篇范围；股数指本次拟改变用途的库存，不是新买入股数。
# 多方案未披露逐方案分配时，只保存合计，不把合计重复分配给各方案。
CARDS = [
    (0, ["2022-09-29"], 64679497, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "64,679,497.00股", "尚需提交公司股东大会审议"),
    (1, ["2022-08-30"], 467966, "EMPLOYEE_PLAN", "RESTRICTED_STOCK_INCENTIVE", "BOARD_NO_SHAREHOLDER_REQUIRED", "467,966股", "无需提交公司股东大会审议"),
    (2, ["2021-10-31"], 7699963, "INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "7,699,963股", "尚需提交公司股东大会审议"),
    (3, ["2021-08-18"], 17485676, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "17,485,676股", "尚需提交股东大会审议"),
    (4, ["2021-07-21", "2021-07-29"], 3444001, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "3,444,001股", "尚须提交公司股东大会审议"),
    (5, ["2021-11-23"], 43900, "INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "43,900股", "尚需以特别议案形式提交公司股东大会审议"),
    (6, ["2022-03-30"], 15242153, "RESTRICTED_STOCK_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "15,242,153股", "仍需本公司股东大会审议通过"),
    (7, ["2022-04-06", "2022-09-16", "2022-10-10"], 16465985, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "16,465,985股", "尚需提交公司股东会审议"),
    (8, ["2022-10-28"], 2858043, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "2,858,043股", "尚需提交公司股东大会审议"),
    (9, ["2022-03-10"], 11090741, "EMPLOYEE_AND_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "11,090,741股", "尚需提交公司股东大会审议"),
    (10, ["2024-01-23"], 11213200, "UNKNOWN_IN_THIS_NOTICE", "CANCELLATION", "CHAIR_PROPOSAL_PENDING_BOARD_AND_SHAREHOLDERS", "11,213,200股", "尚需经公司董事会、股东大会审议"),
    (11, ["2024-01-23"], 11213200, "MAINTAIN_VALUE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "11,213,200股", "尚需提交至公司股东大会审议"),
    (12, ["2023-06-27"], 26527300, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "26,527,300股", "尚需提交公司股东大会审议"),
    (13, ["2022-03-30"], 1472684, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "1,472,684股", "尚需提交股东会审议"),
    (14, ["2022-08-30"], 26600, "RESTRICTED_STOCK_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "26,600股", "尚需提交公司股东大会审议"),
    (15, ["2022-08-05"], 729970, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "729,970股", "尚需提交公司股东会审议"),
    (16, ["2022-02-11"], 1280000, "INCENTIVE_EMPLOYEE_OR_CANCELLATION", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "1,280,000股", "尚需提交公司股东会"),
    (17, ["2021-10-25"], 5586721, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "5,586,721股", "尚需提交公司股东会审议"),
    (18, ["2022-10-31"], 1310991, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "1,310,991股", "尚需提交公司股东会审议"),
    (19, ["2023-11-03"], 22242535, "MAINTAIN_VALUE_THEN_SELL", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "22,242,535股", "尚需提交公司2025年度股东周年大会审议"),
    (20, ["2023-08-16"], 3921163, "CURRENT_PURPOSE_UNKNOWN_ORIGINAL_MAINTAIN_VALUE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "3,921,163股", "尚需提交至公司股东会审议"),
    (21, ["2022-04-25"], 7272164, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "7,272,164股", "尚须提交公司股东会审议"),
    (22, ["2023-01-16"], 191, "PARTNER_PLAN_OR_RESTRICTED_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "191股", "尚需提交股东会审议"),
    (23, ["2023-08-30", "2024-01-30"], 47786169, "MAINTAIN_VALUE_THEN_SELL", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "47,786,169股", "尚需提交公司股东会审议"),
    (24, ["2023-04-27", "2025-03-27", "2026-03-26"], 74541486, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "74,541,486股", "尚需提交股东会审议"),
    (25, ["2023-06-21"], 9751415, "EMPLOYEE_PLAN", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "9,751,415股", "尚需提交公司股东会审议"),
    (26, ["2023-08-15"], 29721264, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "29,721,264股", "尚需提交股东会审议"),
    (27, ["2026-04-03"], 8750205, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "8,750,205股", "尚需提交至公司股东会审议"),
    (28, ["2022-11-07"], 32543837, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "32,543,837股", "尚需提交公司股东会审议"),
    (29, ["2022-11-11"], 12539547, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "12,539,547股", "尚需提交公司股东会审议"),
    (30, ["2024-02-26"], 739967, "EMPLOYEE_OR_INCENTIVE", "CANCELLATION", "BOARD_PENDING_SHAREHOLDERS", "739,967股", "尚需提交公司股东会审议"),
]


NOTES = {
    0: "64679497股回购库存与76624634股重整账户余股分开；合计拟注销141304131股不能全记成回购库存。",
    1: "员工持股改为限制性股票激励，本次不属于注销；后来的注销提议不能倒填到本次。",
    4: "2021-07-21方案剩余6701股，2021-07-29方案剩余3437300股，两批合计3444001股。",
    7: "三份旧方案的合并库存16465985股，未披露注销股份逐方案分配，不能给每份方案重复分配总数。",
    8: "注销2858043股对应2022方案，2024方案库存5175000股不在本次注销范围。",
    9: "剩余库存28452226股中仅11090741股拟注销，不能把全部剩余库存都记为注销。",
    10: "仅董事长提议，董事会和股东会均待审议；未在此文明确的旧用途保持未知，不从后文回填。",
    11: "对应先前提议的同一2024方案和11213200股；两份公告不能重复算作新增股份或新增资金。",
    14: "2024年先由员工持股转限制性激励，本次只对2022方案余下26600股拟注销。",
    15: "正文提到终止员工持股计划，不等于在本公告日终止全部回购；本次对象为729970股库存。",
    16: "本次处理已终止回购事项的库存；历史2022年终止被本公告回述，不倒填成已留存的历史终止证据。",
    17: "本次5586721股与其他限制性股票回购注销另计；修章合计5924111股不能当成本次回购库存。",
    20: "原始方案为维护价值，部分股份后来用于激励；当前变更前现行用途链未齐，不能直接用原始用途填缺。",
    22: "本次只是191股余股，不是累计回购2174560股；小量仍按原文保留，不按研究方便放大或删除。",
    23: "来自海通证券两批库存77074467股，吸收合并按1:0.62转换为47786169股；不是国泰海通当期新回购。另有67516831股库存不在本次范围。",
    24: "公告列三次原方案，74541486股拟注销、26695900股保留；本文件没有逐方案注销分配，原方案列表只作共同来源参照。",
    27: "原文只写2026-04-03公司审议通过方案，具体审批机构需原件确认，不自行补成董事会日期。",
    30: "拟注销739967股的计算涉及预计2026-09-23上市流通使用的69425股；本公告日尚未到该日期，不自动记为已完成交付。",
}


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def locate(pages, phrase):
    needle, hits = normalize(phrase), []
    for number, page in enumerate(pages, 1):
        text = normalize(page)
        for m in re.finditer(re.escape(needle), text):
            if needle[0].isdigit() and m.start() and text[m.start()-1].isdigit():
                continue
            if needle[-1].isdigit() and m.end() < len(text) and text[m.end()].isdigit():
                continue
            hits.append({"page": number, "start": m.start(), "end": m.end(), "phrase": needle,
                         "context": text[max(0,m.start()-70):m.end()+100]})
    assert hits, "原文没有定位到复核短语："+phrase
    return hits


def build():
    protocol = read(OUT / "protocol.json")
    targets = protocol["targets"]
    assert len(targets) == len(CARDS) == 31
    assert {r[0] for r in CARDS} == set(range(31))
    rows = []
    for index, dates, shares, old, new, approval, share_proof, approval_proof in CARDS:
        source = targets[index]
        for kind in ("raw", "text"):
            assert digest(ROOT / source[kind+"_path"]) == source[kind+"_sha256"]
        pages = read(ROOT / source["text_path"])
        proof = {"affected_shares": locate(pages, share_proof), "approval": locate(pages, approval_proof)}
        for date in dates:
            y, m, d = map(int, date.split("-"))
            proof[date] = locate(pages, f"{y}年{m}月{d}日")
        if new == "CANCELLATION":
            purpose = "注销"
        else:
            purpose = "用于A股限制性股票激励计划"
        proof["new_purpose"] = locate(pages, purpose)
        value = float(share_proof[:-1].replace(",", ""))
        assert value == shares
        row = {"document_id": source["document_id"], "symbol": source["symbol"], "title": source["title"],
               "known_at": source["known_at"], "source_url": source["source_url"], "raw_sha256": source["raw_sha256"],
               "reported_original_approval_dates": dates, "original_document_ids": [],
               "original_identity_status": "REPORTED_ORIGIN_REFERENCES_ORIGINAL_DOCUMENT_MATCH_PENDING",
               "original_reference_scope": "POOLED_REFERENCES_ALLOCATION_UNKNOWN" if len(dates) > 1 else "SINGLE_REPORTED_ORIGIN",
               "affected_inventory_shares": shares, "old_purpose": old, "proposed_new_purpose": new,
               "approval_state": approval, "effective_new_purpose_from_this_notice": new if approval == "BOARD_NO_SHAREHOLDER_REQUIRED" else None,
               "cancellation_completed_by_this_notice": False, "event_represents_new_market_purchase": False,
               "new_purchase_cashflow_from_this_notice": None, "historical_first_publication_verified": False,
               "full_plan_lifecycle_established": False, "trading_feature_admitted": False,
               "note": NOTES.get(index, "明确记录库存用途变更及待履行程序，不计作新回购。"), "evidence": proof}
        if index == 0:
            row["separate_restructuring_account_shares"] = 76624634
            row["proposed_total_cancellation_shares"] = 141304131
            proof["separate_inventory"] = locate(pages, "76,624,634股")
            proof["total_cancellation"] = locate(pages, "141,304,131股")
            assert shares+row["separate_restructuring_account_shares"] == row["proposed_total_cancellation_shares"]
        if index == 4:
            row["explicit_original_allocations"] = {"2021-07-21": 6701, "2021-07-29": 3437300}
            row["original_reference_scope"] = "SEPARATE_REPORTED_ALLOCATIONS"
            proof["first_origin_remaining"] = locate(pages, "尚剩余股票6,701股")
            proof["second_origin_remaining"] = locate(pages, "尚剩余股票3,437,300股")
            assert sum(row["explicit_original_allocations"].values()) == shares
        if index == 11:
            row["prior_proposal_document_id"] = targets[10]["document_id"]
            row["duplicate_economic_inventory_not_new_flow"] = True
        if index == 14:
            row["prior_purpose_change_document_id"] = targets[1]["document_id"]
            proof["prior_change_reference"] = locate(pages, "2024-043")
        if index == 23:
            row["predecessor_issuer_name"] = "海通证券股份有限公司"
            row["predecessor_issuer_symbol"] = None
            row["original_reference_scope"] = "PREDECESSOR_POOLED_INVENTORY_AFTER_MERGER"
            row["predecessor_shares"] = 77074467
            row["reported_exchange_ratio"] = .62
            row["separate_retained_inventory_shares"] = 67516831
            proof["predecessor_shares"] = locate(pages, "77,074,467股")
            proof["exchange_ratio"] = locate(pages, "1:0.62")
            proof["unaffected_inventory"] = locate(pages, "67,516,831股")
            assert abs(77074467*.62-shares) < 1
        if index == 24:
            row["pool_inventory_before"] = 101237386
            row["pool_inventory_after_proposed_cancellation"] = 26695900
            proof["pool_inventory"] = locate(pages, "101,237,386股")
            assert row["pool_inventory_before"]-shares == row["pool_inventory_after_proposed_cancellation"]
        if index == 27:
            row["reported_original_date_type"] = "PLAN_APPROVAL_REPORTED_APPROVING_BODY_NOT_EXPLICIT"
        if index == 30:
            row["announced_future_use_shares"] = 69425
            row["announced_future_listing_date"] = "2026-09-23"
            row["future_use_completed_by_this_notice"] = False
            row["inventory_measurement_note"] = "拟注销数是扣除公告所述未来使用后的数值，不自动视为当前结算库存。"
            proof["future_use"] = locate(pages, "将使用上述已回购的库存股69,425股")
            assert pd.Timestamp(source["known_at"]).date() < pd.Timestamp("2026-09-23").date()
            assert 1218700-409308-69425 == shares
        rows.append(row)
    return rows


def at_document_clock(row, timestamp):
    if pd.Timestamp(timestamp) < pd.Timestamp(row["known_at"]):
        return {"status": "NO_VIEW_BEFORE_DOCUMENT", "document_id": None}
    return {"status": row["approval_state"], "document_id": row["document_id"],
            "effective_new_purpose": row["effective_new_purpose_from_this_notice"],
            "cancellation_completed": False, "new_market_purchase": False,
            "scope": "THIS_REVIEWED_NOTICE_ONLY_NOT_COMPLETE_LIFECYCLE"}


def run():
    assert not (OUT / "result.json").exists(), "不覆盖已完成来源记录"
    rows = build()
    save("review_cards.json", rows)
    save("analysis_freeze.json", {"at": now(), "scope": "原文语义与时钟，未读新策略收益",
        "identities": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [OUT/"protocol.json", OUT/"review_cards.json", Path(__file__)]]})
    queries = []
    for row in rows:
        stamp = pd.Timestamp(row["known_at"])
        for when in [stamp-pd.Timedelta(seconds=1), stamp, stamp+pd.Timedelta(days=365)]:
            queries.append({"document_id": row["document_id"], "query_at": when.isoformat(), "result": at_document_clock(row, when)})
        assert queries[-3]["result"]["status"] == "NO_VIEW_BEFORE_DOCUMENT"
        assert queries[-2]["result"]["status"] == row["approval_state"]
        assert not queries[-1]["result"]["cancellation_completed"], "时间经过不能自动完成注销"
    save("asof_notice_queries.json", queries)
    output = [{k: v for k, v in r.items() if k != "evidence"} for r in rows]
    for row in output:
        row["reported_original_approval_dates"] = "|".join(row["reported_original_approval_dates"])
        row["original_document_ids"] = "|".join(row["original_document_ids"])
    pd.DataFrame(output).to_csv(OUT / "remaining_change_fields.csv", index=False, encoding="utf-8-sig")
    save("result.json", {"at": now(), "study_id": "510300_FACTOR96_REPURCHASE_REMAINING_CHANGES_V1",
        "status": "ALL_31_REMAINING_NOTICE_TARGETS_REVIEWED_ORIGINAL_MATCH_AND_COMPLETE_LIFECYCLE_PENDING",
        "documents": len(rows), "approval_counts": dict(Counter(r["approval_state"] for r in rows)),
        "purpose_counts": dict(Counter(r["proposed_new_purpose"] for r in rows)),
        "newly_reported_origin_references": sum(len(r["reported_original_approval_dates"]) for r in rows),
        "original_documents_matched_this_round": 0, "cancellation_completions_established": 0,
        "new_market_purchase_events_from_these_changes": 0, "asof_queries_checked": len(queries),
        "known_before_document_rows": 0, "historical_first_publication_verified": False,
        "T12": "NOT_RUN_FREE_FLOAT_AND_COMPLETE_PURPOSE_CLOCK_CHAIN", "new_accounts": 0,
        "new_returns": 0, "new_network_requests": 0, "delivery_package_required": False,
        "goal_achieved": False, "orders_authorized": False})
    print("31份用途变更已定位原文并登记；30份拟注销、1份改激励用途，均不作为新增回购资金。", flush=True)


if __name__ == "__main__":
    run()
