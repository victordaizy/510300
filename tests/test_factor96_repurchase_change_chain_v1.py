"""检验公告时钟、未生效提案、同日多版本及跨页数字四类真实风险。"""
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_change_chain_v1 import anchors, latest_reviewed_change, prior_candidates, read

OUT = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"


def reviewed_rows():
    cards = read(OUT / "review_cards.json")
    changes = read(OUT / "inputs/changes.json")
    nodes = {r["document_id"]: r for r in read(OUT / "inputs/plan_nodes.json")}
    rows = []
    for change in changes:
        if change["document_id"] not in cards:
            continue
        card = cards[change["document_id"]]
        roots = [nodes[t["original_id"]]["root_id"] for t in card["targets"] if t["original_id"]]
        rows.append(change | card | {"confirmed_target_roots": roots})
    return rows


def test_termination_cannot_enter_before_publication_clock():
    rows = reviewed_rows()
    notice = next(r for r in rows if r["document_id"] == "1221333277")
    root = "002129.SZ_ORIGINAL_1218151247"
    before = latest_reviewed_change(rows, root, pd.Timestamp(notice["known_at"]) - pd.Timedelta(seconds=1))
    after = latest_reviewed_change(rows, root, notice["known_at"])
    assert before["status"] == "NO_REVIEWED_CHANGE_BEFORE_QUERY"
    assert after["approved_terms"] == {"terminated": True}
    assert after["effective_new_purpose"] is None


def test_chair_then_board_both_remain_unapproved_by_shareholders():
    rows = reviewed_rows()
    root = "300496.SZ_ORIGINAL_1223031157"
    proposal = next(r for r in rows if r["document_id"] == "1225080515")
    board = next(r for r in rows if r["document_id"] == "1225140967")
    first = latest_reviewed_change(rows, root, proposal["known_at"])
    second = latest_reviewed_change(rows, root, board["known_at"])
    assert first["status"] == "CHAIR_PROPOSAL_PENDING_BOARD_AND_SHAREHOLDERS"
    assert second["status"] == "BOARD_APPROVED_PENDING_SHAREHOLDERS"
    assert first["effective_new_purpose"] is second["effective_new_purpose"] is None


def test_same_clock_versions_do_not_select_or_sum_one_notice():
    rows = reviewed_rows()
    row = next(r for r in rows if r["document_id"] == "1225525350")
    actual = latest_reviewed_change(rows, "300347.SZ_ORIGINAL_1219108067", row["known_at"])
    assert actual["status"] == "SAME_CLOCK_MULTIPLE_DOCUMENTS_UNRESOLVED"
    assert actual["document_ids"] == ["1225525350", "1225530137"]
    assert actual["effective_new_purpose"] is None


def test_increased_budget_is_terms_not_new_executed_money():
    rows = reviewed_rows()
    row = next(r for r in rows if r["document_id"] == "1224960309")
    value = latest_reviewed_change(rows, "002241.SZ_ORIGINAL_1223056790", row["known_at"])
    assert value["approved_terms"]["new_floor_cny"] == 1_000_000_000
    assert value["approved_terms"]["new_cap_cny"] == 1_500_000_000
    assert row["budget_increase_is_new_executed_cashflow"] is False
    assert row["reported_cumulative_amount_cny"] == "950080648.62"


def test_mentioned_later_plans_are_not_changed_targets():
    rows = reviewed_rows()
    for key in ["1224735615", "1225508551", "1225519613"]:
        row = next(r for r in rows if r["document_id"] == key)
        assert row["confirmed_target_roots"] == []
        assert row["review_status"] == "CONTEXT_ONLY_PRIOR_ROOT_MATCH_REJECTED"
        assert all(t["original_id"] is None for t in row["targets"])


def test_candidate_match_rejects_future_original():
    changes = read(OUT / "inputs/changes.json")
    item = next(r for r in changes if r["document_id"] == "1221333277")
    metadata = next(r for r in read(OUT / "inputs/metadata.json") if r["document_id"] == item["document_id"])
    original = deepcopy(next(r for r in read(OUT / "inputs/plan_nodes.json") if r["document_id"] == "1218151247"))
    assert prior_candidates(item, metadata, [original])
    original["known_at"] = (pd.Timestamp(item["known_at"]) + pd.Timedelta(seconds=1)).isoformat()
    assert prior_candidates(item, metadata, [original]) == []


def test_page_number_cannot_be_joined_into_share_count():
    pages = ["拟变更已回购且尚未使用的", "公司1,258,707 股公司股份的用途"]
    assert anchors(pages, "1,258,707股")[0]["page"] == 2
    with pytest.raises(ValueError):
        anchors(pages, "的21,258,707股")


def test_numeric_quote_cannot_match_suffix_of_a_larger_value():
    with pytest.raises(ValueError):
        anchors(["21,258,707股"], "1,258,707股")


def test_partial_inventory_allocation_is_explicit_and_reconciles():
    rows = reviewed_rows()
    for key in ["1224772736", "1225266149", "1225371015", "1225451851"]:
        row = next(r for r in rows if r["document_id"] == key)
        assert sum(t["proposed_affected_shares"] for t in row["targets"]) == row["proposed_total_shares"]
        assert len(row["confirmed_target_roots"]) == 1
        assert row["review_status"] == "MULTI_TARGET_SOME_PRIOR_ORIGINALS_MISSING"
