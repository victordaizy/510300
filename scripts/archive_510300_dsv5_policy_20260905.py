"""追加归档决定并保留全部旧版本文件身份，不修改V1原件。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import identity, load_json, now, verify, write_once


def main() -> None:
    policy_dir = "reports/research/510300_price_path_dsv5_risk_budget_policy_v1"
    decision_path = ROOT / policy_dir / "ARCHIVE_DECISION_20260905.json"
    if decision_path.exists():
        print(json.dumps(load_json(decision_path), ensure_ascii=False, indent=2))
        return
    paths = set()
    patterns = [
        "config/510300_price_path_dsv5_risk_budget_policy_v1*",
        "docs/510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1*",
        "research/price_path_dsv5_risk_budget_policy_v1*",
        "scripts/*510300_price_path_dsv5_risk_budget_policy_v1*",
        "tests/test_510300_price_path_dsv5_risk_budget_policy_v1*",
    ]
    for pattern in patterns:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for path in [policy_dir, "data/research/510300_price_path_dsv5_risk_budget_policy_v1",
                 "data/forward/510300_price_path_dsv5_risk_budget_policy_v1"]:
        paths.update(p for p in (ROOT / path).rglob("*") if p.is_file())
    preservation = {"recorded_at": now(), "identities": [identity(ROOT, p) for p in sorted(paths)],
                    "v1_files_overwritten": False, "historical_policy_redesign": "NOT_ALLOWED"}
    keep = "reports/research/510300_dsv5_policy_archive_20260905_preservation.json"
    write_once(ROOT / keep, preservation)
    verify(ROOT, preservation["identities"])
    previous = load_json(ROOT / policy_dir / "status.json")
    report = {"MODEL_ID": "510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1",
              "STATE": "ARCHIVED_NO_VIEW_INSUFFICIENT_POLICY_VARIATION",
              "B1_DSV5_PRICE_RISK_FORECAST": "RETAIN_AS_FORECAST_BENCHMARK_ONLY",
              "B1_POSITION_MAPPING": "DISABLED", "HISTORICAL_POLICY_REDESIGN": "NOT_ALLOWED",
              "CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY": "NONE", "POSITION_IMPACT": 0,
              "LIVE_TRADING_AUTHORIZED": False, "DSV5_G1_AUTOMATION_REQUIRED_STATE": "PAUSED",
              "POLICY_FORWARD_RECORDING": "CLOSED", "archived_at": now(),
              "evidence": {"non100_origins": previous["G1"]["non100_origins"],
                           "all_origins": previous["G1"]["primary_origin_count"]},
              "preservation_manifest": identity(ROOT, keep),
              "supersedes_status_interpretation_without_overwriting": identity(ROOT, f"{policy_dir}/status.json")}
    write_once(decision_path, report)
    write_once(ROOT / "reports/research/510300_nbs_1000_negative_information_drift_v1/ARCHIVE_DECISION_20260905.json", {
        "MODEL_ID": "510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1",
        "STATE": "ARCHIVED_BLOCKED_BEFORE_RETURN_READ", "PARENT_RETURN_VALUES_READ": False,
        "archived_at": now(), "original_stop_receipt": identity(ROOT,
            "reports/research/510300_nbs_1000_negative_information_drift_v1/G0_G1_ADJUDICATION.json"),
        "v1_source_and_raw_identity_manifest": identity(ROOT, "config/510300_stk_mins_source_admission_v2_manifest.json"),
        "v1_files_overwritten": False, "position_impact": 0})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
