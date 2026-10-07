"""修正待股东会初始方案被显示为已生效用途的问题，旧查询保留并明确弃用。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_32_originals_v1 import OUT, digest, now, read, save
from research.factor96_repurchase_32_originals_analysis_v1 import reviewed_history_at
from scripts.record_factor96_remaining_changes_v1 import update


def corrected_history_at(root, edges, when):
    state = reviewed_history_at(root, edges, when)
    if state["status"] == "NO_VIEW_BEFORE_ORIGINAL_SOURCE":
        return state
    if root["initial_approval_state"] == "BOARD_PENDING_SHAREHOLDERS":
        approved_changes = [e for e in edges if e["root_id"] == root["root_id"]
                            and pd.Timestamp(e["known_at"]) <= pd.Timestamp(when)
                            and e["effective_new_purpose_from_notice"] is not None]
        if not approved_changes:
            state["last_reviewed_effective_purpose"] = None
        if len(state["source_document_ids"]) == 1:
            state["proposed_purpose"] = root["initial_purpose"]
    return state


def main():
    assert not (OUT / "approval_clock_correction_v1_0_1.json").exists()
    roots = {r["root_id"]: r for r in read(OUT / "original_roots.json")}
    edges = read(OUT / "origin_to_change_edges.json")
    old_path = OUT / "asof_reviewed_history.json"
    old, new, changed = read(old_path), [], []
    for query in old:
        repaired = deepcopy(query)
        repaired["result"] = corrected_history_at(roots[query["root_id"]], edges, query["query_at"])
        if repaired != query:
            assert roots[query["root_id"]]["initial_approval_state"] == "BOARD_PENDING_SHAREHOLDERS"
            assert repaired["result"]["last_reviewed_effective_purpose"] is None
            changed.append({"root_id": query["root_id"], "query_at": query["query_at"],
                            "before": query["result"], "after": repaired["result"]})
        new.append(repaired)
    assert len(changed) == 12 and len({r["root_id"] for r in changed}) == 3
    assert len(new) == len(old) == 166
    new_path = OUT / "asof_reviewed_history_v1_0_1.json"
    save(new_path, new)
    result = read(OUT / "result.json")
    result.update(at=now(), effective_asof_query_path=new_path.relative_to(ROOT).as_posix(),
                  original_asof_query_status="SUPERSEDED_INITIAL_PENDING_APPROVAL_MAPPING_ERROR",
                  approval_clock_correction_path=(OUT / "approval_clock_correction_v1_0_1.json").relative_to(ROOT).as_posix())
    save(OUT / "result_v1_0_1.json", result)
    correction = {"at": now(), "status": "CORRECTED_INITIAL_PENDING_APPROVAL_PURPOSE_VISIBILITY",
                  "reason": "原方案仍待股东会批准时，初始用途只能作为拟议用途，不能显示为已生效用途。",
                  "invalidated_query_path": old_path.relative_to(ROOT).as_posix(), "invalidated_query_sha256": digest(old_path),
                  "authoritative_query_path": new_path.relative_to(ROOT).as_posix(), "corrected_query_sha256": digest(new_path),
                  "changed_rows": changed, "changed_roots": 3, "unchanged_rows": len(old) - len(changed),
                  "raw_sources_and_original_terms_changed": False, "new_accounts": 0, "new_returns": 0,
                  "code_sha256": digest(Path(__file__)), "goal_achieved": False}
    save(OUT / "approval_clock_correction_v1_0_1.json", correction)
    authoritative_result = (OUT / "result_v1_0_1.json").relative_to(ROOT).as_posix()
    program = ROOT / "reports/research/510300_factor96_program_v1"
    status_path = program / "status.json"
    status = read(status_path)
    status.update(at=now(), latest_result=authoritative_result,
                  latest_progress_receipt=correction["authoritative_query_path"],
                  latest_repurchase_original_asof_queries=correction["authoritative_query_path"],
                  latest_approval_clock_correction=result["approval_clock_correction_path"])
    update(status_path, status)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate["latest_progress_receipt"] = result["approval_clock_correction_path"]
    update(mandate_path, mandate)
    for filename in ("strategy_progress.json", "factor_progress.json"):
        path = program / filename
        rows = read(path)
        for row in rows:
            if row["id"] in ("T12", "M02"):
                row["current_evidence_path"] = authoritative_result
                if row["id"] == "T12":
                    row["source_gate_path"] = authoritative_result
        update(path, rows)
        csv_name = "18策略当前进度.csv" if filename.startswith("strategy") else "96因子当前进度.csv"
        pd.DataFrame(rows).to_csv(program / csv_name, index=False, encoding="utf-8-sig")
    report = OUT / "研究进展.md"
    with report.open("a", encoding="utf-8") as f:
        f.write("\n保存后核对发现3个初始待股东会方案的用途被提前显示为已生效，已修正12条查询。原文、预算和关联未变；旧asof_reviewed_history.json明确弃用，现行查询为asof_reviewed_history_v1_0_1.json，现行结果为result_v1_0_1.json。\n")
    print("3个初始待批方案的12条查询已修正，其余154条未变；旧结果保留并标注弃用，无回测变动。")


if __name__ == "__main__":
    main()
