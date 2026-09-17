"""将已交付且解压复核通过的新研究补充进总索引。"""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
OUT=ROOT/"reports/research/510300_unlock_delivery_verification_20260914"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    result_path=ROOT/"reports/research/510300_unlock_announcement_increment_v1/result.json"
    result=json.loads(result_path.read_text(encoding="utf-8"))
    delivery_path=ROOT/"deliverables/510300_解禁公告信息增量_V1_GPT审阅_20260914.delivery.json"
    delivery=json.loads(delivery_path.read_text(encoding="utf-8"))
    verified_path=OUT/"extracted_packet_verification.json"
    verified=json.loads(verified_path.read_text(encoding="utf-8"))
    assert verified["status"]=="PASS_EXTRACTED_PACKET_SOURCE_MODEL_ACCOUNT_RECOMPUTATION"
    assert sha(Path(delivery["path"]))==delivery["sha256"]==verified["zip_sha256"]
    assert result["status"]=="REJECTED_FROZEN_NO_RELIABLE_UNLOCK_ANNOUNCEMENT_INCREMENT"
    before_bytes=INDEX.read_bytes()
    before_sha=hashlib.sha256(before_bytes).hexdigest()
    before=json.loads(before_bytes.decode("utf-8"))
    key="supplemental_unlock_announcement_increment_v1"
    assert key not in before
    stress=next(x for x in result["accounts"] if x["cost"]=="STRESS" and x["model"]=="M1")
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    addition={"study_id":result["study_id"],"registered_at":stamp,"status":result["status"],"result":result_path.relative_to(ROOT).as_posix(),"result_sha256":sha(result_path),"source_admission":"reports/research/510300_unlock_announcement_source_admission_v1/source_admission.json","source_scope":"COMPLETE_33722_ANNOUNCEMENT_DATE_RECORDS_WITH_ALL_FAILURES_PRESERVED","model_fits":result["model_fits"],"new_accounts":result["new_accounts"],"evaluation_origins":1604,"mature_paired_origins":1599,"stress_M1_net_cagr":stress["annualized_return"],"stress_M1_net_sharpe":stress["net_sharpe"],"stress_M1_max_drawdown":stress["max_drawdown"],"goal_achieved":False,"independent_validation":"NOT_ESTABLISHED","review_zip":Path(delivery["path"]).relative_to(ROOT).as_posix(),"review_zip_sha256":delivery["sha256"],"review_zip_bytes":delivery["bytes"],"extracted_packet_verification":verified_path.relative_to(ROOT).as_posix(),"extracted_packet_verification_sha256":sha(verified_path),"source_and_model_data_not_recomputed_for_rescue":True,"post_result_parameter_rescue":False,"original_daily_observer_status_unchanged":True,"legacy_round_and_account_counters_not_redefined":True,"current_goal_turn_classification":"PROGRESS","followup_route":"reports/research/510300_unlock_delivery_verification_20260914/goal_followup_and_next_route.json","position_impact":0}
    updated=dict(before)
    updated[key]=addition
    updated["updated_at"]=stamp
    backup=OUT/"latest_research_before_unlock_registration.json"
    with backup.open("xb") as stream:stream.write(before_bytes)
    temporary=INDEX.with_name(".latest_research_unlock_registration_20260914.json")
    save(temporary,updated)
    assert sha(INDEX)==before_sha
    assert temporary.resolve().parent==INDEX.resolve().parent
    os.replace(temporary,INDEX)
    after=json.loads(INDEX.read_text(encoding="utf-8"))
    unchanged={k:v for k,v in after.items() if k not in {key,"updated_at"}}
    expected={k:v for k,v in before.items() if k!="updated_at"}
    assert json.dumps(unchanged,sort_keys=True,ensure_ascii=False)==json.dumps(expected,sort_keys=True,ensure_ascii=False)
    receipt={"status":"PASS_SUPPLEMENTAL_RESEARCH_REGISTERED_WITH_ALL_OTHER_FIELDS_PRESERVED","registered_at":stamp,"index":INDEX.relative_to(ROOT).as_posix(),"before_sha256":before_sha,"after_sha256":sha(INDEX),"backup":backup.relative_to(ROOT).as_posix(),"added_key":key,"all_other_fields_except_updated_at_preserved":True,"goal_achieved":False,"source_result_and_frozen_study_files_modified":False}
    save(OUT/"registration_receipt.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
