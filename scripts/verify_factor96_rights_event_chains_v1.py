"""从已保存输出复核数值、原文定位和信息时序，不重新运行研究或访问网络。"""
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_rights_event_chains_v1 import OUT, read, save, digest, normalized, now


def main():
    effective = OUT / "effective_v1_0_1"
    facts = read(effective / "reviewed_document_facts.json")
    events = read(effective / "event_chains.json")
    queries = read(effective / "publication_boundary_queries.json")
    by_id = {row["document_id"]: row for row in facts}
    assert len(facts) == len(by_id) == 127
    groups = defaultdict(list)
    anchors = 0
    for row in facts:
        assert digest(row["raw_path"]) == row["raw_sha256"]
        assert digest(row["text_path"]) == row["text_sha256"]
        pages = {p["page"]: normalized(p["text"]) for p in read(row["text_path"])["pages"]}
        for entries in row["evidence"].values():
            for item in entries:
                assert item["document_id"] == row["document_id"]
                assert pages[item["page"]][item["normalized_start"]:item["normalized_end"]] == item["literal"]
                anchors += 1
        assert row["trading_feature_admitted"] is False
        if row["event_id"]:
            groups[row["event_id"]].append(row)
    amounts = 0
    for rows in groups.values():
        prices = []
        for row in sorted(rows, key=lambda r: (r["known_at"], r["document_id"])):
            fields = row["updates"]
            if "issue_price_cents" in fields:
                prices.append(fields["issue_price_cents"])
            if "actual_subscribed_shares" in fields:
                assert prices, row["document_id"]
                assert fields["actual_subscribed_shares"] * prices[-1] == fields["actual_subscription_cents"]
                amounts += 1
    for query in queries:
        for side in ("before", "at"):
            stored = query[side]
            eligible = sorted([r for r in groups[query["event_id"]] if r["known_at"] <= stored["as_of"]],
                              key=lambda r: (r["known_at"], r["document_id"]))
            assert [r["document_id"] for r in eligible] == stored["source_documents"]
            expected, origins = {}, {}
            for row in eligible:
                expected.update(row["updates"])
                origins.update({key: {"document_id": row["document_id"], "known_at": row["known_at"]} for key in row["updates"]})
            assert origins == stored["field_sources"]
            assert all(stored[key] == value for key, value in expected.items())
            for field in ("actual_subscribed_shares", "actual_subscription_cents", "announced_listing_date"):
                if field not in expected:
                    assert stored[field] is None
            assert stored["strict_M06_pressure"] is None and stored["trading_feature_admitted"] is False
    for event in events:
        actual = groups[event["event_id"]]
        assert {r["document_id"] for r in actual} == set(event["document_ids"])
        assert len(event["document_ids"]) == len(set(event["document_ids"]))
        assert event["issuance_notice_ids"]
    program = read(ROOT / "reports/research/510300_factor96_program_v1/status.json")
    protected = read(OUT / "program_update_receipt.json")["protected_counts_and_authority"]
    assert all(program[key] == value for key, value in protected.items())
    assert not program["goal_achieved"] and not program["delivery_package_required"]
    assert program["latest_result"].endswith("510300_factor96_rights_event_chains_v1/effective_v1_0_1/result.json")
    assert amounts == 38 and len(groups) == 36 and len(queries) == 112
    prior_case = read(ROOT / "reports/research/510300_factor96_rights_calendar_fields_v1/result.json")
    assert prior_case["case_documents"] == 4
    receipt = {"at": now(), "status": "PASS_SAVED_SOURCE_RECOMPUTATION",
               "raw_and_text_document_pairs_checked": len(facts), "literal_source_anchors_checked": anchors,
               "subscription_amounts_recomputed_using_price_known_by_each_result": amounts,
               "saved_publication_states_recomputed": 2*len(queries), "effective_events": len(groups),
               "prior_single_case_reused": 1, "additional_events_beyond_prior_single_case": 35,
               "account_statistics_unchanged": protected, "new_network_requests": 0,
               "result_sha256": digest(effective / "result.json"), "code_sha256": digest(Path(__file__)),
               "meaning": "通过的是保存数据、原文定位和时序复算；没有检验策略有效性，尚不能交易。"}
    save(effective / "saved_recomputation_receipt.json", receipt)
    print("已保存来源复算通过：", len(facts), "份原文与文本，", anchors, "个定位，", amounts,
          "个金额版本，", 2*len(queries), "个时点；原1案例复用，另35事件完成本次还原。")


if __name__ == "__main__":
    main()
