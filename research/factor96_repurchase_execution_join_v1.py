"""把已知用途变更接入948条执行披露，按公开批次保留增量和未知，不替代自由流通分母。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from research.factor96_repurchase_32_originals_v1 import ROOT, digest, now, read, save
from research.factor96_repurchase_remaining_changes_v1 import locate

OUT = ROOT / "reports/research/510300_factor96_repurchase_execution_join_v1"
BASE = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
LATEST = ROOT / "reports/research/510300_factor96_repurchase_32_originals_v1"
TRINA_ROOT = "688599.SH_ORIGINAL_1220457662"


def freeze():
    assert not (OUT / "protocol.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [BASE / "inputs/execution_events.json", BASE / "execution_purpose_fields.json",
              BASE / "inputs/documents.json", LATEST / "combined_change_ledger.json", LATEST / "original_roots.json",
              ROOT / "reports/research/510300_funding_repurchase_demand_daily_v1/result.json",
              ROOT / "reports/research/510300_repurchase_fixed_maturity_account_v1/result.json", Path(__file__)]
    save(OUT / "protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_REPURCHASE_EXECUTION_JOIN_V1",
        "previous_goal_turn": "PROGRESS_32_ORIGINAL_TERMS_AND_SOURCE_CONFLICTS_DOCUMENTED",
        "scope": "全部948条直接方案身份支持的执行披露，以及54份已读变更对象，逐条按当时公开时钟相连。",
        "preflight": "原方案补件32个与948执行的方案根无交集；54变更中11条执行已有先前变更，均对应天合光能延期。发现两组同日公开的重复/不同统计日文件并已读原文。",
        "cash": "继承已冻结可比累计额差，不恢复任何修订冲突隔离记录；金额只表示已披露的历史实施，不是未来买单。",
        "batch": "同方案同公开日末为一个批次；有明确统计日期时按最新统计日取累计状态。同统计日同金额只计一次。未知日或冲突不按公告编号制造先后。",
        "purpose": "使用当前执行文件自身的明确用途；批次中最新统计日的文档须一致明确，否则保持未知。未知不从未来公告或旧原方案填入。",
        "change_clock": "变更仅在其已公开时间后进入对应方案；未被已读范围识别不等于不存在其他变更。",
        "term_case": "天合原文董事会2024-06-25、期限12个月；2025-05-24公布延至2026-03-24；2026-03-25才公开3月23日已完成，不倒填完成状态。",
        "strict_M01": "自由流通市值未取得，M01归一化值保持空，不用成交额或普通流通市值代替。",
        "prior_research": "资金回购需求日频家族及固定到期变体已经失败，复用其来源不复活旧规则，不运行替代分母或参数搜索。",
        "new_accounts": 0, "new_network_requests": 0, "new_returns": 0, "goal_achieved": False,
        "delivery_package_required": False, "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in inputs],
    })
    print("已固定948条执行披露的公开批次与用途关联规则，不读取新策略收益。")


def trina_terms_at(when):
    t = pd.Timestamp(when)
    if t < pd.Timestamp("2024-06-26T23:59:59+08:00"):
        return {"status": "NO_VIEW"}
    amended = t >= pd.Timestamp("2025-05-24T23:59:59+08:00")
    completed = t >= pd.Timestamp("2026-03-25T23:59:59+08:00")
    end = pd.Timestamp("2026-03-24" if amended else "2025-06-24")
    return {"status": "COMPLETION_PUBLICLY_REPORTED" if completed else "ORIGINAL_TERMS_OR_AMENDMENT_KNOWN",
            "end_date": end.date().isoformat(), "funding": "OWN_AND_RAISED" if amended else "OWN_FUNDS",
            "deadline_method": "EXPLICIT_AMENDMENT" if amended else "BOARD_PLUS_12_MONTHS_MINUS_ONE_DAY_CALENDAR_INFERENCE",
            "remaining_calendar_days_under_known_terms": max(0, (end - t.tz_localize(None).normalize()).days),
            "completed_as_disclosed": completed, "actual_completion_date": "2026-03-23" if completed else None,
            "completion_source": "1225028659" if completed else None,
            "latest_terms_source": "1223645277" if amended else "1220457662",
            "scope": "THIS_REVIEWED_PLAN_HISTORY_ONLY"}


def run():
    assert not (OUT / "result.json").exists()
    for item in read(OUT / "protocol.json")["inputs"]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    events = read(BASE / "inputs/execution_events.json")
    by_id = {e["document_id"]: e for e in events}
    purpose = {p["document_id"]: p for p in read(BASE / "execution_purpose_fields.json")}
    changes = read(LATEST / "combined_change_ledger.json")
    docs = {d["document_id"]: d for d in read(BASE / "inputs/documents.json")}
    selected = {
        "1220457662": ["2024年6月25日", "自董事会审议通过本次回购方案之日起12个月内", "无需提交公司股东大会审议"],
        "1223645277": ["2025年6月24日止延期至2026年3月24日止", "无需提交公司股东会审议"],
        "1225028659": ["2026年3月23日，公司本次股份回购计划实施完毕", "1,001,275,293.60元"],
        "1224401364": ["截至2025年7月31日，公司尚未实施本次回购股份"],
        "1224401365": ["10,473,100元", "2025年8月5日"],
        "1225536562": ["30,021,592元", "2026年8月31日"],
        "1225536563": ["30,021,592元", "2026年8月31日"],
    }
    proofs = []
    for key, phrases in selected.items():
        doc = docs[key]
        for kind in ("raw", "text"):
            assert digest(BASE / doc[kind + "_snapshot"]) == doc[kind + "_sha256"]
        pages = read(BASE / doc["text_snapshot"])
        proofs.append({"document_id": key, "raw_sha256": doc["raw_sha256"], "source_url": doc["source_url"],
                       "evidence": [locate(pages, phrase) for phrase in phrases]})
    save(OUT / "focused_source_clauses.json", proofs)
    joined, grouped = [], defaultdict(list)
    for event in events:
        key, root, stamp = event["document_id"], event["root_id"], pd.Timestamp(event["known_at"])
        p = purpose[key]
        assert p["root_id"] == root and pd.Timestamp(p["known_at"]) == stamp
        previous = by_id.get(event["previous_confirmed_id"])
        if previous is not None:
            assert previous["root_id"] == root and pd.Timestamp(previous["known_at"]) <= stamp
        delta = event["reported_increment_cents"]
        if delta is not None and previous is not None:
            assert delta == event["cumulative_cents"] - previous["cumulative_cents"]
        prior_changes = [c for c in changes if root in c["confirmed_target_roots"] and pd.Timestamp(c["known_at"]) <= stamp]
        latest = max(prior_changes, key=lambda c: c["known_at"]) if prior_changes else None
        row = {"document_id": key, "symbol": event["symbol"], "root_id": root, "known_at": event["known_at"],
               "economic_cutoff": event["economic_cutoff"], "classification": event["classification"],
               "reported_cumulative_cents": event["cumulative_cents"], "reported_increment_cents": delta,
               "prior_increment_status": event["change_status"], "prior_usable_positive_disclosure": event["usable_positive_disclosure"],
               "previous_confirmed_id": event["previous_confirmed_id"],
               "previous_is_strictly_earlier_publication": previous is not None and pd.Timestamp(previous["known_at"]) < stamp,
               "same_document_purpose_status": p["purpose"]["status"], "same_document_purposes": p["purpose"]["selected_purposes"],
               "reported_execution_to_floor_ratio": p["reported_execution_to_floor_ratio"],
               "known_target_change_ids": [c["document_id"] for c in prior_changes],
               "latest_reviewed_target_change": latest["document_id"] if latest else None,
               "latest_reviewed_change_action": latest["action"] if latest else None,
               "latest_reviewed_change_approval": latest["approval_state"] if latest else None,
               "complete_change_coverage_established": False, "strict_M01_value": None,
               "free_float_denominator_status": "MISSING_CURRENT_CREDENTIAL_EXPIRED", "trading_feature_admitted": False}
        if root == TRINA_ROOT:
            row["known_term_state"] = trina_terms_at(event["known_at"])
        joined.append(row)
        grouped[root, event["known_at"]].append(event)
    batches = []
    for (root, stamp), rows in sorted(grouped.items()):
        frontier = rows
        if len(rows) > 1:
            assert all(r["economic_cutoff"] is not None for r in rows)
            last_cutoff = max(pd.Timestamp(r["economic_cutoff"]) for r in rows)
            frontier = [r for r in rows if pd.Timestamp(r["economic_cutoff"]) == last_cutoff]
            assert len({r["cumulative_cents"] for r in frontier}) == 1
        chosen = min(frontier, key=lambda r: (r["classification"] != "FIRST_EXECUTION", r["document_id"]))
        increment = chosen["reported_increment_cents"]
        batch_status = chosen["change_status"]
        external = [e for e in events if e["root_id"] == root and pd.Timestamp(e["known_at"]) < pd.Timestamp(stamp)]
        prior = max(external, key=lambda e: (e["known_at"], e["economic_cutoff"] or "", e["document_id"])) if external else None
        if len(rows) > 1:
            assert not any("CONFLICT" in r["change_status"] for r in rows)
            if prior:
                increment = chosen["cumulative_cents"] - prior["cumulative_cents"]
            else:
                assert any(r["classification"] == "FIRST_EXECUTION" for r in rows)
                increment = chosen["cumulative_cents"]
            assert sum(r["reported_increment_cents"] or 0 for r in rows) == increment
            batch_status = "REVIEWED_SAME_CLOCK_BATCH_POSITIVE_INCREMENT"
        selected_purposes = [purpose[r["document_id"]]["purpose"]["selected_purposes"] for r in frontier]
        consensus = selected_purposes[0] if all(x is not None and x == selected_purposes[0] for x in selected_purposes) else None
        batches.append({"batch_id": root + "_" + stamp[:10], "root_id": root, "symbol": chosen["symbol"], "known_at": stamp,
                        "document_ids": [r["document_id"] for r in rows], "frontier_document_ids": [r["document_id"] for r in frontier],
                        "frontier_cumulative_cents": chosen["cumulative_cents"], "reported_increment_cents": increment,
                        "increment_status": batch_status, "positive_disclosed_increment": increment is not None and increment > max(r["reporting_resolution_cents"] for r in rows),
                        "purpose_consensus": consensus, "previous_publication_document_id": prior["document_id"] if prior else None,
                        "strict_M01_value": None, "trading_feature_admitted": False,
                        "source_clock_grade": "HISTORICAL_CATALOGUE_DATE_END_PROXY"})
    assert len(joined) == 948 and len(batches) == 946
    duplicates = [b for b in batches if len(b["document_ids"]) > 1]
    assert sorted(b["reported_increment_cents"] for b in duplicates) == [1047310000, 3002159200]
    assert sum(r["latest_reviewed_target_change"] is not None for r in joined) == 11
    assert {r["latest_reviewed_target_change"] for r in joined if r["latest_reviewed_target_change"]} == {"1223645277"}
    queries = [{"query_at": t, "state": trina_terms_at(t)} for t in (
        "2024-06-26T23:59:58+08:00", "2024-06-26T23:59:59+08:00",
        "2025-05-24T23:59:58+08:00", "2025-05-24T23:59:59+08:00",
        "2026-03-23T23:59:59+08:00", "2026-03-24T23:59:59+08:00", "2026-03-25T23:59:59+08:00")]
    assert queries[2]["state"]["end_date"] == "2025-06-24" and queries[3]["state"]["end_date"] == "2026-03-24"
    assert not queries[4]["state"]["completed_as_disclosed"] and not queries[5]["state"]["completed_as_disclosed"]
    assert queries[6]["state"]["completed_as_disclosed"]
    save(OUT / "execution_context_rows.json", joined)
    save(OUT / "publication_batches.json", batches)
    save(OUT / "same_clock_examples.json", duplicates)
    save(OUT / "trina_term_clock_queries.json", queries)
    pd.DataFrame(joined).to_csv(OUT / "948条执行公告关联.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(batches).to_csv(OUT / "946个公开批次.csv", index=False, encoding="utf-8-sig")
    event_roots = {e["root_id"] for e in events}
    result = {"at": now(), "study_id": "510300_FACTOR96_REPURCHASE_EXECUTION_JOIN_V1",
              "status": "EXECUTION_CONTEXT_AND_PUBLICATION_BATCHES_BUILT_STRICT_M01_BLOCKED",
              "execution_rows": len(joined), "publication_batches": len(batches), "same_clock_multiple_document_batches": len(duplicates),
              "positive_increment_batches": sum(b["positive_disclosed_increment"] for b in batches),
              "batch_purpose_known": sum(b["purpose_consensus"] is not None for b in batches),
              "batch_purpose_unknown": sum(b["purpose_consensus"] is None for b in batches),
              "prior_increment_statuses_preserved": dict(Counter(r["prior_increment_status"] for r in joined)),
              "new_32_roots_overlap_with_execution_roots": len(event_roots & {r["root_id"] for r in read(LATEST / "original_roots.json")}),
              "all_change_roots_overlap_with_execution_roots": len(event_roots & {r for c in changes for r in c["confirmed_target_roots"]}),
              "execution_rows_with_previously_known_reviewed_change": 11, "unique_prior_changes_affecting_rows": 1,
              "new_network_requests": 0, "new_accounts": 0, "new_returns": 0, "strict_M01_non_null_rows": 0,
              "full_lifecycle_established": False, "goal_achieved": False, "delivery_package_created": False}
    save(OUT / "result.json", result)
    print(f"已关联948条执行公告，形成946个公开批次；正向披露批次{result['positive_increment_batches']}个，自由流通分母仍缺失。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回购执行披露与已知条款关联")
    parser.add_argument("action", choices=["freeze", "run"])
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
