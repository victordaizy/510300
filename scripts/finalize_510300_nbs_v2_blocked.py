"""在来源硬门失败后登记终止尝试；不加载任何事件价格或收益。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import ContractError, committed, csv_bytes, git, identity, load_json, now, verify, write_once

REPORT = "reports/research/510300_nbs_1000_negative_information_drift_v2"
PROTOCOL = "config/510300_nbs_1000_negative_information_drift_v2.yaml"
MANIFEST = "config/510300_nbs_1000_negative_information_drift_v2_manifest.json"
SCOPE = [PROTOCOL, "docs/510300_NBS_V2_PROTOCOL_AND_STOP_20260905.md",
         "scripts/finalize_510300_nbs_v2_blocked.py",
         "data/curated/510300_nbs_1000_negative_information_drift_v2/event_ledger_pre_return.csv"]


def main() -> None:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="冻结或关闭被来源阻断的NBS V2")
    parser.add_argument("phase", choices=["freeze", "close"])
    args = parser.parse_args()
    source_path = "reports/data_quality/510300_stk_mins_source_admission_v2/source_adjudication.json"
    source = load_json(ROOT / source_path)
    if source["source_pass"] or source["event_return_reads"] != 0:
        raise ContractError("该终止入口只允许来源失败且收益尚未读取的状态")
    if args.phase == "freeze":
        paths = SCOPE + [source_path, "config/510300_stk_mins_source_admission_v2_manifest.json"]
        report = {"model_id": "510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V2", "frozen_at": now(),
                  "source_state_known_at_freeze": source["state"], "event_return_values_read": False,
                  "identities": [identity(ROOT, p) for p in paths], "committed_scope": SCOPE}
        write_once(ROOT / MANIFEST, report)
    else:
        manifest = load_json(ROOT / MANIFEST)
        verify(ROOT, manifest["identities"])
        source_manifest = load_json(ROOT / "config/510300_stk_mins_source_admission_v2_manifest.json")
        verify(ROOT, source_manifest["identities"])
        committed(ROOT, SCOPE + [MANIFEST])
        commit = git(ROOT, "rev-parse", "HEAD")
        claim = {"model_id": "510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V2",
                 "claimed_at": now(), "freeze_commit": commit,
                 "state": "ONE_SHOT_ATTEMPT_TERMINATED_BEFORE_RETURN_READ",
                 "source_gate_pass": False, "return_read_right_consumed": False,
                 "event_return_values_read_before_claim": False, "restart_allowed": False,
                 "manifest": identity(ROOT, MANIFEST), "position_impact": 0}
        claim_path = ROOT / REPORT / "one_shot_claim.json"
        if claim_path.exists():
            raise ContractError("一次性终止尝试已登记，禁止再次消费")
        write_once(claim_path, claim)
        report = {"MODEL_ID": claim["model_id"], "STATE": "ARCHIVED_BLOCKED_G0_BEFORE_RETURN_READ",
                  "NBS_FAMILY": "PERMANENTLY_CLOSED_NO_RESCUE", "G0": "FAIL_SOURCE_VWAP_CONTRACT",
                  "G1": "PASS_36_17_17_16", "G2": "NOT_RUN_BLOCKED_BY_G0",
                  "G3": "NOT_RUN_BLOCKED_BY_G0", "G4": "NOT_RUN_BLOCKED_BY_G0",
                  "EVENT_RETURN_VALUES_READ": False, "EVENT_LABELS_CREATED": 0,
                  "NBS_B0_B1_TRAINED": False, "PLACEBO_RETURN_VALUES_READ": False,
                  "HIGH_MARGIN_SIGNAL_DENSITY_READ": False, "PORTFOLIO_EVALUATION": "NOT_ALLOWED",
                  "SHARPE": "NOT_COMPUTED", "ALPHA_EFFICACY_TESTED": False,
                  "NEXT_RESEARCH_MODE": "STRICT_FORWARD_ONLY", "POSITION_IMPACT": 0,
                  "LIVE_TRADING_AUTHORIZED": False, "freeze_commit": commit, "closed_at": now(),
                  "source": identity(ROOT, source_path), "claim": identity(ROOT, claim_path)}
        write_once(ROOT / REPORT / "status.json", report)
        write_once(ROOT / REPORT / "execution_receipt.json", report)
        write_once(ROOT / "config/510300_historical_alpha_stop_gate_20260905.json", {
            "STATE": "STRICT_FORWARD_ONLY", "effective_at": now(),
            "reason": "NBS_V2_FINAL_HISTORICAL_ALPHA_ATTEMPT_BLOCKED_BY_FROZEN_SOURCE_GATE",
            "additional_510300_historical_alpha_experiments_allowed": False,
            "closed_nbs_family_reopen_allowed": False,
            "allowed_forward_research": ["PCF_IOPV", "A50_AFTER_HOURS_INFORMATION_LAG",
                                         "MACRO_FIRST_RELEASE_VINTAGE", "B1_DSV5_FORECAST_OBSERVATORY",
                                         "ACTUAL_BROKER_EXECUTION_COST_OBSERVATION"],
            "broker_connection_authorized": False, "live_trading_authorized": False,
            "position_impact": 0, "nbs_final_status": identity(ROOT, ROOT / REPORT / "status.json")})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
