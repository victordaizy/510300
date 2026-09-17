"""登记A/H固定用途终态及其审阅包，保留总索引已有项目字段。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDY = ROOT / "reports/research/510300_ah_premium_increment_v1"
OUT = ROOT / "reports/research/510300_ah_premium_delivery_verification_20260913"
FIELD = "supplemental_ah_premium_increment_v1"


def main() -> None:
    before_bytes = INDEX.read_bytes()
    before = json.loads(before_bytes)
    if FIELD in before:
        raise FileExistsError("A/H本轮已登记")
    result = json.loads((STUDY / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((STUDY / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "deliverables/510300_AH溢价信息增量_V1_GPT审阅_20260913.delivery.json").read_text(encoding="utf-8"))
    if result["status"] != "REJECTED_FROZEN_NO_RELIABLE_AH_INCREMENT" or verification["accounts_checked"] != 6:
        raise ValueError("研究终态或核对状态不同")
    zip_path = ROOT / delivery["zip"]
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != delivery["sha256"]:
        raise ValueError("交付包哈希与回执不同")
    moment = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    stress = next(a for a in result["accounts"] if a["cost"] == "STRESS" and a["model"] == "M1")
    entry = {"registered_at": moment, "study_id": result["study_id"], "status": result["status"],
             "result": (STUDY / "result.json").relative_to(ROOT).as_posix(),
             "result_sha256": hashlib.sha256((STUDY / "result.json").read_bytes()).hexdigest(),
             "report": (STUDY / "研究结论与下一步.md").relative_to(ROOT).as_posix(),
             "review_zip": delivery["zip"], "review_zip_sha256": delivery["sha256"],
             "mature_paired_origins": result["evaluation"]["mature_evaluation_origins"],
             "relative_mse_improvement": result["evaluation"]["overall"]["relative_mse_improvement"],
             "positive_eras": result["evaluation"]["positive_eras"], "model_fits": result["model_fits"],
             "accounts": 6, "stress_net_cagr": stress["annualized_return"], "stress_net_sharpe": stress["net_sharpe"],
             "disposition": "CLOSED_THIS_FIXED_INFORMATION_USE_NO_PARAMETER_RESCUE", "goal_achieved": False,
             "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    after = copy.deepcopy(before)
    after[FIELD] = entry
    after["updated_at"] = moment
    unchanged = copy.deepcopy(after)
    unchanged.pop(FIELD)
    unchanged["updated_at"] = before["updated_at"]
    if unchanged != before:
        raise AssertionError("登记意外更改其他研究")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index_before_ah_registration.json").write_bytes(before_bytes)
    if INDEX.read_bytes() != before_bytes:
        raise RuntimeError("其他任务更新了索引，停止以保留并发改动")
    encoded = (json.dumps(after, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = INDEX.with_name(INDEX.name + ".ah_registration.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    os.replace(temporary, INDEX)
    if json.loads(INDEX.read_bytes()) != after:
        raise AssertionError("回读不一致")
    receipt = {"status": "AH_TERMINAL_AND_REVIEW_ZIP_REGISTERED", "registered_at": moment,
               "only_semantic_changes": [FIELD, "updated_at"],
               "index_before_sha256": hashlib.sha256(before_bytes).hexdigest(),
               "index_after_sha256": hashlib.sha256(encoded).hexdigest(), "entry": entry,
               "new_models": 0, "new_accounts": 0, "old_observer_changed": False}
    with (OUT / "registration_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
