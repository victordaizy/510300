from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "scripts/run_a_share_hs_cash_option_floor_alpha_v1_0_2_final_free_source_repair.py"
)
CONFIG_PATH = (
    ROOT
    / "config/a_share_hs_official_cash_option_floor_alpha_v1_0_2_final_free_source_repair.json"
)
MANIFEST_PATH = (
    ROOT
    / "config/a_share_hs_official_cash_option_floor_alpha_v1_0_2_final_free_source_repair_protocol_manifest.json"
)
RECEIPT_PATH = (
    ROOT
    / "reports/audit/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_FINAL_FREE_SOURCE_REPAIR.json"
)


def load_module():
    specification = importlib.util.spec_from_file_location("cash_option_v1_0_2_final_repair", SCRIPT_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_frozen_contract_keeps_one_attempt_and_zero_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["attempt_id"] == "CASH_OPTION_V1_0_2_FINAL_FREE_REPAIR_20260826"
    assert config["maximum_formal_attempts"] == 1
    assert config["data_purchase_budget_cny"] == 0
    assert config["terminal_status_on_any_required_gate_failure"] == (
        "NO_VIEW_FREE_DATA_INSUFFICIENT"
    )
    assert config["governance"]["second_source_repair_allowed"] is False
    assert config["governance"]["source_hopping_after_attempt"] is False
    assert config["governance"]["price_values_may_be_read_before_full_admission"] is False
    assert config["governance"]["return_values_may_be_read_before_full_admission"] is False


def test_protocol_and_frozen_inputs_verify() -> None:
    module = load_module()
    assert MANIFEST_PATH.is_file()
    result = module.verify_protocol()
    assert result["status"] == "PASS_FINAL_FREE_SOURCE_REPAIR_PROTOCOL_VERIFIED"
    assert result["failure_count"] == 0
    assert result["frozen_input_count"] == 9


def test_terminal_receipt_is_no_view_and_consumes_only_attempt() -> None:
    module = load_module()
    assert RECEIPT_PATH.is_file()
    result = module.verify_terminal_receipt()
    assert result["status"] == "PASS_FINAL_FREE_SOURCE_REPAIR_TERMINAL_RECEIPT_VERIFIED"
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["final_status"] == "NO_VIEW_FREE_DATA_INSUFFICIENT"
    assert receipt["attempt_number"] == 1
    assert receipt["maximum_formal_attempts"] == 1
    assert receipt["attempt_consumed"] is True
    assert receipt["terminal_decision"]["second_repair_allowed"] is False
    assert receipt["terminal_decision"]["v1_0_3_or_later_repair_allowed"] is False


def test_no_external_request_or_price_evaluation_occurs_after_failed_preflight() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    execution = receipt["external_source_execution"]
    decision = receipt["terminal_decision"]
    assert execution["status"] == "SKIPPED_MANDATORY_PREFLIGHT_FAILED"
    assert execution["external_request_attempted"] is False
    assert execution["eastmoney_subject_requests_attempted"] == 0
    assert execution["baostock_requests_attempted"] == 0
    assert execution["eastmoney_csi300_requests_attempted"] == 0
    assert execution["raw_files_created"] == 0
    assert decision["price_screen_executed"] is False
    assert decision["price_values_read"] is False
    assert decision["return_values_read"] is False
    assert decision["qualified_event_count_known"] is False


def test_frozen_official_lifecycle_inventory_is_complete_but_not_adjudicated() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    inventory = receipt["official_lifecycle_evidence_inventory"]
    assert inventory["event_count"] == 21
    assert inventory["result_role_present_event_count"] == 21
    assert inventory["settlement_role_present_event_count"] == 10
    assert inventory["formal_corporate_action_adjudication_complete_count"] == 0
    assert inventory["formal_lifecycle_adjudication_complete_count"] == 0
    assert inventory["role_inventory_is_final_adjudication"] is False
    assert all(event["formal_lifecycle_adjudication_complete"] is False for event in inventory["events"])


def test_second_run_is_idempotent_and_remains_terminal() -> None:
    command = [str(ROOT / ".venv/Scripts/python.exe"), str(SCRIPT_PATH), "--mode", "run"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 3
    payload = json.loads(completed.stdout)
    assert payload["final_status"] == "NO_VIEW_FREE_DATA_INSUFFICIENT"
    assert payload["terminal_receipt_reused"] is True
