from __future__ import annotations

import copy

import pytest

from research.digital_asset_basis_ranked_spot_long_v12 import (
    load_contract as load_base_contract,
)
from research.digital_asset_basis_ranked_spot_long_v12r_schema_correction import (
    load_contract,
    validate_contract,
)


def test_schema_correction_preserves_objective_and_adds_required_net_limit() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["risk"]["maximum_gross_exposure"] == 1.0
    assert contract["risk"]["maximum_absolute_net_exposure"] == 1.0
    assert not any(contract["safety"].values())


def test_only_protocol_output_and_one_risk_field_differ_from_v12() -> None:
    base = load_base_contract()
    corrected = load_contract()
    for section in (
        "account",
        "historical_partition",
        "universe",
        "information_timing",
        "signal",
        "delivery",
        "prefreeze_coverage_gate",
        "portfolio",
        "costs",
        "inputs",
        "visible_gates",
        "historical_evidence_limits",
        "governance",
        "safety",
    ):
        assert corrected[section] == base[section]
    corrected_risk = copy.deepcopy(corrected["risk"])
    assert corrected_risk.pop("maximum_absolute_net_exposure") == 1.0
    assert corrected_risk == base["risk"]


def test_expanding_schema_correction_is_rejected() -> None:
    contract = load_contract()
    contract["signal"]["target_long_gross"] = 1.0
    with pytest.raises(ValueError, match="模式修正被扩大或损坏"):
        validate_contract(contract)
