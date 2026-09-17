"""只读重核来源；统一日期存储精度后仍逐值严格比较。"""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_510300_unlock_source_admission_v1 import OUT, RESULT, reconstruct, sha, now
from scripts.repair_510300_unlock_announcement_source_v1_1 import check


def normalize_date_storage(frame):
    result = frame.copy()
    changed = []
    for name in result:
        if pd.api.types.is_datetime64_any_dtype(result[name].dtype):
            before = str(result[name].dtype)
            result[name] = result[name].dt.as_unit("ns")
            changed.append({"column": name, "before": before, "comparison_dtype": str(result[name].dtype)})
    return result, changed


def main():
    saved = json.loads((OUT/"source_admission.json").read_text(encoding="utf-8"))
    check(sha(ROOT/saved["verifier"]["path"]) == saved["verifier"]["sha256"], "原已保存的来源构造代码变化")
    check(sha(RESULT) == saved["source_completion"]["sha256"], "来源完成回执变化")
    for item in saved["outputs"]:
        check(sha(ROOT/item["path"]) == item["sha256"], "保存来源输出变化")
    diagnostic, eligible, company_days, years = reconstruct()
    provenance = diagnostic.pop("response_provenance")
    for key, value in diagnostic.items():
        check(value == saved[key], "来源诊断重算不同："+key)
    check(provenance == json.loads((OUT/"admitted_response_provenance.json").read_text(encoding="utf-8")), "原始响应身份清单变化")
    normalization = []
    for name, recreated in [("eligible_with_membership_diagnostic.parquet",eligible), ("company_day_activity_diagnostic.parquet",company_days)]:
        stored = pd.read_parquet(OUT/name)
        left, ln = normalize_date_storage(recreated)
        right, rn = normalize_date_storage(stored)
        pd.testing.assert_frame_equal(left,right,check_exact=True)
        normalization.append({"file":name,"recreated_dates":ln,"stored_dates":rn,"all_cell_values_equal":True})
    actual_years = pd.read_csv(OUT/"year_coverage.csv",dtype={"year":str})
    pd.testing.assert_frame_equal(years,actual_years,check_exact=True)
    print(json.dumps({"status":"PASS_OFFLINE_RAW_SOURCE_AND_MEMBERSHIP_RECOMPUTATION_AFTER_DATE_UNIT_NORMALIZATION","verified_at":now(),"all_records_checked":diagnostic["all_records"],"raw_responses_checked":diagnostic["raw_responses_rechecked"],"membership_sessions":diagnostic["membership_sessions"],"date_storage_normalization":normalization,"previous_dtype_failure_preserved":True,"source_data_or_model_inputs_modified":False,"new_network_calls":0,"new_return_labels":0,"new_model_fits":0,"new_accounts":0,"new_random_samples":0,"verifier_sha256":sha(Path(__file__))},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
