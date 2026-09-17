"""实际解压复算通过后追加本轮结果，原样保留其他研究状态。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_growth_disagreement_increment_v1 import OUT, read, save, identity, now


def main():
    verification = ROOT / "reports/research/510300_eps_growth_disagreement_delivery_verification_20260914"
    receipt_path = verification / "extracted_packet_verification.json"
    receipt = read(receipt_path)
    delivery = read(ROOT / "deliverables/510300_EPS增长分歧增量_V1_GPT审阅_20260914.delivery.json")
    if receipt["status"] != "PASS_ACTUAL_EXTRACTED_PACKET_SOURCES_PREDICTIONS_AND_ALL_MONTHS":
        raise ValueError("实际解压复算未通过")
    if identity(Path(delivery["path"]))["sha256"] != delivery["sha256"] or receipt["zip_sha256"] != delivery["sha256"]:
        raise ValueError("ZIP身份不一致")
    result = read(OUT / "result.json")
    if result["goal_achieved"] or result["new_model_fits"] != 72 or result["new_accounts_generated"] != 0:
        raise ValueError("本轮实际工作计数或目标状态不符")
    index = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    before_bytes = index.read_bytes()
    before = json.loads(before_bytes)
    key = "supplemental_eps_growth_disagreement_increment_v1"
    if key in before:
        raise FileExistsError("研究已经登记")
    after, stamp = dict(before), now()
    after[key] = {"study_id": result["study_id"], "status": result["status"], "registered_at": stamp,
        "result": (OUT / "result.json").relative_to(ROOT).as_posix(), "result_sha256": identity(OUT / "result.json")["sha256"],
        "review_zip": Path(delivery["path"]).relative_to(ROOT).as_posix(), "review_zip_sha256": delivery["sha256"],
        "review_zip_bytes": delivery["bytes"], "extracted_packet_verification": receipt_path.relative_to(ROOT).as_posix(),
        "extracted_packet_verification_sha256": identity(receipt_path)["sha256"],
        "source_eligible_months": 50, "predicted_origins": 36, "mature_paired_origins": 33,
        "new_model_fits": 72, "new_accounts": 0, "prediction_increment_gate_pass": False,
        "account_stage_status": result["account_stage"]["status"], "goal_achieved": False,
        "current_goal_turn_classification": "PROGRESS", "same_external_blocker_consecutive_turns": 0,
        "old_frozen_results_preserved": True, "position_impact": 0, "paused_source_queues_resumed": False,
        "followup_route": (OUT / "goal_followup_and_next_route.json").relative_to(ROOT).as_posix()}
    after["updated_at"] = stamp
    with (verification / "latest_research_before_registration.json").open("xb") as stream:
        stream.write(before_bytes)
    temporary = index.with_name(".latest_research_eps_growth_disagreement_increment_20260914.json")
    save(temporary, after)
    if identity(index)["sha256"] != hashlib.sha256(before_bytes).hexdigest():
        raise ValueError("总索引在登记期间变化，未覆盖")
    os.replace(temporary, index)
    persisted = read(index)
    if {k: v for k, v in persisted.items() if k not in {key, "updated_at"}} != {k: v for k, v in before.items() if k != "updated_at"}:
        raise ValueError("旧字段未完整保留")
    record = {"status": "PASS_GROWTH_DISAGREEMENT_REGISTERED_OTHER_FIELDS_PRESERVED", "registered_at": stamp,
        "added_key": key, "before_sha256": hashlib.sha256(before_bytes).hexdigest(), "after_sha256": identity(index)["sha256"],
        "all_other_fields_except_updated_at_preserved": True, "new_model_fits_in_study": 72,
        "new_accounts": 0, "current_goal_turn_classification": "PROGRESS", "goal_achieved": False}
    save(verification / "registration_receipt.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
