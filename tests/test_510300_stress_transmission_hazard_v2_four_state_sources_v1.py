from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.project_evidence_contract_v1 import EvidenceContractError
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)
from scripts.probe_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    validate_endpoint_frames,
)


ROOT = Path(__file__).resolve().parents[1]


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "daily": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20200312"],
                "open": [12.0],
                "high": [12.3],
                "low": [11.8],
                "close": [12.1],
                "pre_close": [12.0],
                "change": [0.1],
                "pct_chg": [(12.1 / 12.0 - 1.0) * 100.0],
                "vol": [100.0],
                "amount": [1200.0],
            }
        ),
        "suspend_d": pd.DataFrame(
            {
                "ts_code": ["000977.SZ"],
                "trade_date": ["20200312"],
                "suspend_timing": [None],
                "suspend_type": ["S"],
            }
        ),
        "dividend": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "end_date": ["20191231"],
                "ann_date": ["20200101"],
                "div_proc": ["实施"],
                "stk_div": [0.0],
                "stk_bo_rate": [0.0],
                "stk_co_rate": [0.0],
                "cash_div": [0.1],
                "cash_div_tax": [0.1],
                "record_date": ["20200527"],
                "ex_date": ["20200528"],
                "pay_date": ["20200528"],
                "div_listdate": [None],
                "imp_ann_date": ["20200521"],
                "base_date": ["20191231"],
                "base_share": [100.0],
            }
        ),
        "adj_factor": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20200312"],
                "adj_factor": [100.0],
            }
        ),
    }


def test_source_addendum_keeps_scope_and_performance_closed() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    assert config["program"]["executable_assets"] == ["510300.SH", "CASH_CNY"]
    assert config["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["program"]["position_impact"] == 0
    assert config["four_state_mapping"]["blanket_zero_fill_allowed"] is False


def test_provider_suspension_candidate_cannot_authorize_zero() -> None:
    config = load_config(DEFAULT_CONFIG)
    suspend = config["provider_contract"]["suspend_d"]
    official = config["four_state_mapping"]["OFFICIAL_SUSPENSION"]
    assert suspend["official_exchange_evidence_eligible"] is False
    assert suspend["zero_return_authorization"] is False
    assert official["provider_suspend_d_sufficient"] is False


def test_adjustment_factor_is_crosscheck_only() -> None:
    config = load_config(DEFAULT_CONFIG)
    factor = config["provider_contract"]["adj_factor"]
    assert factor["role"] == "ACTION_CANDIDATE_AND_CROSSCHECK_ONLY"
    assert factor["adjusted_price_construction_allowed"] is False
    assert factor["direct_return_construction_allowed"] is False


def test_probe_frames_require_frozen_schema_and_daily_identity() -> None:
    config = load_config(DEFAULT_CONFIG)
    normalized = validate_endpoint_frames(config, _frames())
    assert set(normalized) == {"daily", "suspend_d", "dividend", "adj_factor"}

    broken = _frames()
    broken["daily"] = broken["daily"].assign(pct_chg=99.0)
    with pytest.raises(EvidenceContractError, match="恒等式失败"):
        validate_endpoint_frames(config, broken)


def test_probe_rejects_unknown_suspension_type() -> None:
    config = load_config(DEFAULT_CONFIG)
    broken = _frames()
    broken["suspend_d"] = broken["suspend_d"].assign(suspend_type="UNKNOWN")
    with pytest.raises(EvidenceContractError, match="未知类型"):
        validate_endpoint_frames(config, broken)


def test_frozen_addendum_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        pytest.skip("四态来源附录尚未冻结")
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["addendum_id"] == config["program"]["addendum_id"]
    assert manifest["return_evaluation"] == "NOT_ALLOWED"
    assert manifest["portfolio_results_read"] is False
