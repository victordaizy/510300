"""完整交付和真实解压复核后，追加两对口径诊断索引。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_profit_attribution_pair_adjudication_v1 import OUT, identity, now, read, save

VERIFY = ROOT / "reports/research/510300_eps_profit_pair_delivery_verification_20260914"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
DELIVERY = ROOT / "deliverables/510300_EPS两对归母口径核实_V1_GPT审阅_20260914.delivery.json"


def main() -> None:
    delivery = read(DELIVERY)
    extracted_path = VERIFY / "extracted_packet_verification.json"
    extracted = read(extracted_path)
    numerical = read(VERIFY / "extracted_offline_recomputation_v1_1.json")
    result_path = OUT / "result.json"
    result = read(result_path)
    if extracted["status"] != "PASS_EXTRACTED_PACKET_OFFLINE_RECOMPUTATION":
        raise ValueError("实际解压交付核对未通过")
    if numerical["status"] != "PASS_FOUR_ORIGINAL_PDFS_TWO_SEMANTIC_PAIRS_AND_CONDITIONAL_BOUNDS_RECOMPUTED":
        raise ValueError("实际解压数值复算未通过")
    if identity(Path(delivery["path"]))["sha256"] != delivery["sha256"] or extracted["zip_sha256"] != delivery["sha256"]:
        raise ValueError("交付ZIP身份不符")
    if result["goal_achieved"] or result["new_model_fits"] or result["new_accounts_generated"]:
        raise ValueError("本轮来源诊断不应生成新模型、账户或达标声明")
    before_bytes = INDEX.read_bytes()
    before = json.loads(before_bytes)
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    key = "supplemental_eps_profit_attribution_pair_adjudication_v1"
    if key in before:
        raise FileExistsError("本轮诊断已经登记")
    stamp = now()
    after = dict(before)
    after[key] = {"study_id": result["study_id"], "status": result["status"], "registered_at": stamp,
        "result": result_path.relative_to(ROOT).as_posix(), "result_sha256": identity(result_path)["sha256"],
        "original_pdfs": 4, "pairs": 2, "origin": result["origin"], "target_fiscal_year": 2026,
        "old_stock_coverage": result["before"]["covered_stock_weight"],
        "conditional_stock_coverage": result["conditional_after"]["covered_stock_weight"],
        "conditional_direction_bounds_include_zero": result["conditional_after"]["bounds_include_zero"],
        "both_positive_best_case_lower_bound": result["both_recovered_revisions_positive_best_case_lower_bound"],
        "independent_iso_currency_proof_added": False, "global_net_profit_alias_admitted": False,
        "original_factors_or_accounts_changed": False, "new_model_fits": 0, "new_accounts": 0,
        "goal_achieved": False, "review_zip": Path(delivery["path"]).relative_to(ROOT).as_posix(),
        "review_zip_sha256": delivery["sha256"], "review_zip_bytes": delivery["bytes"],
        "extracted_packet_verification": extracted_path.relative_to(ROOT).as_posix(),
        "extracted_packet_verification_sha256": identity(extracted_path)["sha256"],
        "full_upstream_659_report_reconstruction_in_this_packet": False,
        "current_goal_turn_classification": "PROGRESS", "same_external_blocker_consecutive_turns": 0,
        "followup_route": (OUT / "goal_followup_and_next_route.json").relative_to(ROOT).as_posix(),
        "existing_eps_failures_preserved": True, "paused_source_queues_resumed": False,
        "original_daily_observer_unchanged": True, "legacy_round_and_account_counters_not_redefined": True,
        "position_impact": 0}
    after["updated_at"] = stamp
    with (VERIFY / "latest_research_before_pair_registration.json").open("xb") as stream:
        stream.write(before_bytes)
    temporary = INDEX.with_name(".latest_research_eps_profit_pair_registration_20260914.json")
    save(temporary, after)
    if identity(INDEX)["sha256"] != before_sha:
        raise ValueError("登记期间总索引改变，未覆盖")
    os.replace(temporary, INDEX)
    persisted = read(INDEX)
    if {k: v for k, v in persisted.items() if k not in {key, "updated_at"}} != {k: v for k, v in before.items() if k != "updated_at"}:
        raise ValueError("旧研究字段未完整保留")
    record = {"status": "PASS_SUPPLEMENTAL_PAIR_DIAGNOSTIC_REGISTERED_WITH_OTHER_FIELDS_PRESERVED",
              "registered_at": stamp, "index": INDEX.relative_to(ROOT).as_posix(), "before_sha256": before_sha,
              "after_sha256": identity(INDEX)["sha256"], "added_key": key,
              "all_other_fields_except_updated_at_preserved": True, "new_model_fits": 0, "new_accounts": 0,
              "current_goal_turn_classification": "PROGRESS", "goal_achieved": False}
    save(VERIFY / "registration_receipt.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
