"""两项本轮研究实际交付复核后，追加总索引并保存旧状态。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_explicit_revision_novelty_diagnostic_v1 import read, save, identity, now


def main():
    verification = ROOT / "reports/research/510300_eps_revision_novelty_delivery_verification_20260914"
    receipt_path = verification / "extracted_packet_verification.json"
    receipt = read(receipt_path)
    delivery = read(ROOT / "deliverables/510300_EPS明示修正去重与增长分歧来源_V1_GPT审阅_20260914.delivery.json")
    if receipt["status"] != "PASS_EXTRACTED_PACKET_BOTH_OFFLINE_RECOMPUTATIONS":
        raise ValueError("实际解压复算未通过")
    if identity(Path(delivery["path"]))["sha256"] != delivery["sha256"] or receipt["zip_sha256"] != delivery["sha256"]:
        raise ValueError("ZIP身份不一致")
    index = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    before_bytes = index.read_bytes()
    before = json.loads(before_bytes)
    after = dict(before)
    stamp = now()
    additions = ["eps_explicit_revision_novelty_diagnostic_v1", "eps_growth_disagreement_source_feasibility_v1"]
    keys = []
    for slug in additions:
        key = "supplemental_" + slug
        if key in before:
            raise FileExistsError("研究已登记")
        keys.append(key)
        path = ROOT / "reports/research" / ("510300_" + slug) / "result.json"
        result = read(path)
        if result["goal_achieved"] or result["new_model_fits"] or result["new_accounts_generated"]:
            raise ValueError("本轮来源检查不应有新模型、账户或目标达成声明")
        after[key] = {"study_id": result["study_id"], "status": result["status"], "registered_at": stamp,
            "result": path.relative_to(ROOT).as_posix(), "result_sha256": identity(path)["sha256"],
            "review_zip": Path(delivery["path"]).relative_to(ROOT).as_posix(), "review_zip_sha256": delivery["sha256"],
            "review_zip_bytes": delivery["bytes"], "extracted_packet_verification": receipt_path.relative_to(ROOT).as_posix(),
            "extracted_packet_verification_sha256": identity(receipt_path)["sha256"],
            "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False, "current_goal_turn_classification": "PROGRESS",
            "same_external_blocker_consecutive_turns": 0, "old_frozen_results_preserved": True, "position_impact": 0,
            "paused_source_queues_resumed": False,
            "followup_route": "reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1/goal_followup_and_next_route.json"}
        if "disagreement" in slug:
            after[key].update({"eligible_source_months": result["eligible_months"], "paired_company_months": result["available_paired_company_months"],
                               "source_screen_passed": result["source_screen_passed"], "predictive_value": "NOT_COMPUTED"})
    after["updated_at"] = stamp
    with (verification / "latest_research_before_registration.json").open("xb") as stream:
        stream.write(before_bytes)
    temporary = index.with_name(".latest_research_eps_novelty_disagreement_20260914.json")
    save(temporary, after)
    if identity(index)["sha256"] != hashlib.sha256(before_bytes).hexdigest():
        raise ValueError("总索引在登记期间变化，未覆盖")
    os.replace(temporary, index)
    persisted = read(index)
    if {k: v for k, v in persisted.items() if k not in set(keys + ["updated_at"])} != {k: v for k, v in before.items() if k != "updated_at"}:
        raise ValueError("旧字段未完整保留")
    record = {"status": "PASS_TWO_SUPPLEMENTAL_STUDIES_REGISTERED_OTHER_FIELDS_PRESERVED", "registered_at": stamp,
              "added_keys": keys, "before_sha256": hashlib.sha256(before_bytes).hexdigest(), "after_sha256": identity(index)["sha256"],
              "all_other_fields_except_updated_at_preserved": True, "new_model_fits": 0, "new_accounts": 0,
              "current_goal_turn_classification": "PROGRESS", "goal_achieved": False}
    save(verification / "registration_receipt.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
