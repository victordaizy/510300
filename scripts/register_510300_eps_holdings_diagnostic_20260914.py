"""在完整交付与解压复核后登记新增诊断，保持旧研究状态不变。"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_eps_holdings_delivery_verification_20260914"
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path,obj):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2)
        stream.write("\n")


def main():
    delivery_path=ROOT/"deliverables/510300_EPS实际持仓覆盖诊断_V1_GPT审阅_20260914.delivery.json"
    delivery=read(delivery_path)
    receipt_path=OUT/"extracted_packet_verification.json"
    receipt=read(receipt_path)
    numerical=read(OUT/"extracted_offline_recomputation.json")
    result_path=ROOT/"reports/research/510300_eps_disclosed_holdings_diagnostic_v1/result.json"
    result=read(result_path)
    if receipt["status"]!="PASS_EXTRACTED_PACKET_OFFLINE_RECOMPUTATION" or numerical["status"]!="PASS_OFFLINE_ORIGINAL_TABLE_EPS_FACT_AND_COVERAGE_RECOMPUTATION":
        raise ValueError("解压后的数值复核未通过")
    if sha(Path(delivery["path"]))!=delivery["sha256"] or receipt["zip_sha256"]!=delivery["sha256"]:
        raise ValueError("交付ZIP身份不一致")
    if result["goal_achieved"] or result["new_accounts_generated"] or result["new_model_fits"]:
        raise ValueError("来源诊断不应产生新账户或目标达成状态")
    before_bytes=INDEX.read_bytes()
    before=json.loads(before_bytes)
    before_sha=hashlib.sha256(before_bytes).hexdigest()
    key="supplemental_eps_disclosed_holdings_diagnostic_v1"
    if key in before:
        raise FileExistsError("本诊断已经登记")
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    updated=dict(before)
    updated[key]={"study_id":result["study_id"],"registered_at":stamp,"status":result["status"],
        "result":result_path.relative_to(ROOT).as_posix(),"result_sha256":sha(result_path),
        "report_years":[2017,2020,2024],"original_fund_reports":3,"holding_rows":990,
        "reused_earnings_originals":659,"new_labels":0,"new_model_fits":0,"new_accounts":0,
        "revision_stock_weight_coverage":[x["covered_stock_weight"] for x in result["coverage"] if x["field"]=="profit_revision"],
        "all_three_revision_direction_bounds_include_zero":True,"goal_achieved":False,
        "current_official_index_weights_admitted":False,"historical_http_first_publication_proven":False,
        "new_strategy_selected":False,"existing_eps_failures_preserved":True,"paused_source_queues_resumed":False,
        "review_zip":Path(delivery["path"]).relative_to(ROOT).as_posix(),"review_zip_sha256":delivery["sha256"],
        "review_zip_bytes":delivery["bytes"],"extracted_packet_verification":receipt_path.relative_to(ROOT).as_posix(),
        "extracted_packet_verification_sha256":sha(receipt_path),"current_goal_turn_classification":"PROGRESS",
        "same_external_blocker_consecutive_turns":0,"followup_route":"reports/research/510300_eps_disclosed_holdings_diagnostic_v1/goal_followup_and_next_route.json",
        "original_daily_observer_unchanged":True,"legacy_round_and_account_counters_not_redefined":True,"position_impact":0}
    updated["updated_at"]=stamp
    backup=OUT/"latest_research_before_eps_holdings_registration.json"
    with backup.open("xb") as stream:
        stream.write(before_bytes)
    temporary=INDEX.with_name(".latest_research_eps_holdings_registration_20260914.json")
    save(temporary,updated)
    if sha(INDEX)!=before_sha:
        raise ValueError("研究索引在登记期间被其他任务修改，未覆盖")
    os.replace(temporary,INDEX)
    after=read(INDEX)
    remaining={k:v for k,v in after.items() if k not in {key,"updated_at"}}
    if remaining!={k:v for k,v in before.items() if k!="updated_at"}:
        raise ValueError("旧研究字段未被完整保留")
    record={"status":"PASS_SUPPLEMENTAL_DIAGNOSTIC_REGISTERED_WITH_OTHER_FIELDS_PRESERVED","registered_at":stamp,
            "index":INDEX.relative_to(ROOT).as_posix(),"before_sha256":before_sha,"after_sha256":sha(INDEX),
            "added_key":key,"all_other_fields_except_updated_at_preserved":True,"new_models":0,"new_accounts":0,
            "current_goal_turn_classification":"PROGRESS","goal_achieved":False}
    save(OUT/"registration_receipt.json",record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
