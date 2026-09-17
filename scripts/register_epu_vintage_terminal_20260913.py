"""只追加本轮 EPU 终态及交付索引，保留既有研究和原观察任务。"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "reports/research/510300_epu_vintage_delivery_verification_20260913"
STUDY = ROOT / "reports/research/510300_epu_vintage_increment_v1"
FIELD = "supplemental_epu_vintage_increment_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    before_bytes = INDEX.read_bytes()
    before = json.loads(before_bytes)
    if FIELD in before:
        raise FileExistsError("EPU 研究已登记，不覆盖")
    result = json.loads((STUDY / "result.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "deliverables/510300_EPU历史版本信息增量_V1_GPT审阅_20260913.delivery.json").read_text(encoding="utf-8"))
    restored = json.loads((OUT / "restored_packet_verification_receipt.json").read_text(encoding="utf-8"))
    if result["status"] != "REJECTED_FROZEN_NO_RELIABLE_EPU_INCREMENT" or restored["exit_code"] != 0:
        raise ValueError("研究终态或解压后重算未通过")
    if sha(ROOT / delivery["zip"]) != delivery["sha256"] or restored["zip_sha256"] != delivery["sha256"]:
        raise ValueError("审阅包版本不同")
    moment = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    stress = next(a for a in result["accounts"] if a["cost"] == "STRESS" and a["model"] == "M1")
    entry = {"registered_at": moment, "study_id": result["study_id"], "status": result["status"],
             "goal_turn_classification": "PROGRESS", "goal_achieved": False,
             "result": (STUDY / "result.json").relative_to(ROOT).as_posix(), "result_sha256": sha(STUDY / "result.json"),
             "report": (STUDY / "研究结论与下一步.md").relative_to(ROOT).as_posix(),
             "review_zip": delivery["zip"], "review_zip_sha256": delivery["sha256"],
             "mature_calendar_months": result["evaluation"]["mature_calendar_months"], "paired_months": result["evaluation"]["paired_months"],
             "latest_value_vintages_used": result["evaluation"]["distinct_used_latest_value_vintages"],
             "relative_mse_improvement": result["evaluation"]["relative_mse_improvement"],
             "model_fits": result["model_fits"], "accounts": result["new_accounts"],
             "stress_net_cagr": stress["annualized_return"], "stress_net_sharpe": stress["net_sharpe"],
             "disposition": "CLOSED_THIS_FIXED_INFORMATION_USE_NO_PARAMETER_RESCUE",
             "independent_validation": "NOT_ESTABLISHED", "position_impact": 0,
             "next_source_check": "EX_ANTE_SUPPLY_DISCLOSURE_DEDUPLICATION_AND_FREE_PIT_FEASIBILITY_ONLY_NOT_RUN"}
    after = copy.deepcopy(before)
    after[FIELD], after["updated_at"] = entry, moment
    unchanged = copy.deepcopy(after)
    unchanged.pop(FIELD)
    unchanged["updated_at"] = before["updated_at"]
    if unchanged != before:
        raise AssertionError("登记意外修改了既有研究")
    with (OUT / "index_before_epu_registration.json").open("xb") as stream:
        stream.write(before_bytes)
    if INDEX.read_bytes() != before_bytes:
        raise RuntimeError("检测到其他任务更新索引，保留并发改动")
    encoded = (json.dumps(after, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = INDEX.with_name(INDEX.name + ".epu_registration.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    os.replace(temporary, INDEX)
    if json.loads(INDEX.read_bytes()) != after:
        raise AssertionError("索引回读不一致")
    receipt = {"status": "EPU_TERMINAL_AND_VERIFIED_REVIEW_ZIP_REGISTERED", "registered_at": moment,
               "only_semantic_changes": [FIELD, "updated_at"], "index_before_sha256": hashlib.sha256(before_bytes).hexdigest(),
               "index_after_sha256": hashlib.sha256(encoded).hexdigest(), "entry": entry,
               "new_models": 0, "new_accounts": 0, "old_observer_changed": False}
    with (OUT / "registration_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
