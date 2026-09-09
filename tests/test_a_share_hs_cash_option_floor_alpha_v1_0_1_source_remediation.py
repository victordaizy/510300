from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_a_share_hs_cash_option_floor_alpha_v1_0_1_source_remediation import (
    CONFIG_PATH,
    SourceRemediationError,
    validate_remediation_config,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_source_only_remediation_config_passes() -> None:
    validate_remediation_config(config())


def test_parent_threshold_inheritance_cannot_change() -> None:
    mutated = copy.deepcopy(config())
    mutated["preserved_parent_contract"]["minimum_net_conditional_floor_return"] = (
        "INHERIT_PARENT_0_07"
    )
    with pytest.raises(SourceRemediationError, match="minimum_net_conditional_floor_return"):
        validate_remediation_config(mutated)


def test_trigger_must_remain_exact_40101_attempt() -> None:
    mutated = copy.deepcopy(config())
    mutated["trigger"]["required_error_code"] = "OTHER"
    with pytest.raises(SourceRemediationError, match="required_error_code"):
        validate_remediation_config(mutated)


def test_eastmoney_must_remain_unadjusted() -> None:
    mutated = copy.deepcopy(config())
    mutated["remediated_market_data_contract"]["subject_secondary_request"]["fqt"] = "1"
    with pytest.raises(SourceRemediationError, match="subject_secondary_request.fqt"):
        validate_remediation_config(mutated)


def test_pre_freeze_price_screen_cannot_be_claimed_complete() -> None:
    mutated = copy.deepcopy(config())
    mutated["information_observed_before_remediation_freeze"]["price_screen_completed"] = True
    with pytest.raises(SourceRemediationError, match="price_screen_completed"):
        validate_remediation_config(mutated)


def test_no_execution_authorization_can_be_enabled() -> None:
    mutated = copy.deepcopy(config())
    mutated["governance"]["live_authorized"] = True
    with pytest.raises(SourceRemediationError, match="live_authorized"):
        validate_remediation_config(mutated)
