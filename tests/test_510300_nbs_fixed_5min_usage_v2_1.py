"""新用途合同只采用被授权的测量变更，测试全部使用合成数据。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.nbs_v2_common import ContractError
from research.nbs_fixed_5min_usage_v2_1 import PASS, admit, g3_statistics, inherited_checks, window_measurements
from tests.test_510300_nbs_v2_source import fixture


def test_only_two_explicitly_authorized_bar_checks_are_demoted():
    prior = {"checks": {"all_bar_vwap_defined": False, "all_bar_vwap_inside_low_high": False,
                        "daily_amount_error": False, "duplicate_keys_zero": True}}
    kept, diagnostics = inherited_checks(prior)
    assert kept == {"daily_amount_error": False, "duplicate_keys_zero": True}
    assert len(diagnostics) == 2
    assert not all(kept.values())


def test_offsetting_bar_defects_can_pass_only_the_fixed_window_contract():
    cfg, minute, _, _ = fixture()
    i = minute.index[minute.trade_time.eq("2024-01-02 10:01:00")][0]
    minute.loc[i, "amount"] += 1
    minute.loc[i + 1, "amount"] -= 1
    result = window_measurements(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001 + 1e-12)
    assert result["pass"].all()
    assert minute.loc[i, "amount"] / minute.loc[i, "vol"] > minute.loc[i, "high"] + .001


@pytest.mark.parametrize("kind", ["missing", "duplicate", "negative", "unpaired_zero", "zero_window", "nonfinite", "range"])
def test_actual_window_defects_remain_hard_failures(kind):
    cfg, minute, _, _ = fixture()
    i = minute.index[minute.trade_time.eq("2024-01-02 10:01:00")][0]
    select = minute.trade_time.str[-8:].isin(cfg["windows"]["reaction"])
    if kind == "missing":
        minute = minute.drop(index=i)
    elif kind == "duplicate":
        minute.loc[i, "trade_time"] = minute.loc[i + 1, "trade_time"]
    elif kind == "negative":
        minute.loc[i, ["vol", "amount"]] = [-100, -400]
    elif kind == "unpaired_zero":
        minute.loc[i, "vol"] = 0
    elif kind == "zero_window":
        minute.loc[select, ["vol", "amount"]] = 0
    elif kind == "nonfinite":
        minute.loc[i, "amount"] = np.nan
    else:
        minute.loc[select, "amount"] = 500
    result = window_measurements(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001)
    assert not result.loc[result.window.eq("reaction"), "pass"].iloc[0]


def test_no_trade_minute_preserves_aggregated_original_amount_and_volume():
    cfg, minute, _, _ = fixture()
    minute.loc[minute.trade_time.eq("2024-01-02 10:01:00"), ["vol", "amount"]] = 0
    result = window_measurements(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001)
    row = result.loc[result.window.eq("reaction")].iloc[0]
    assert row["pass"]
    assert row.vwap == 4 and row.volume_shares == 400 and row.amount_cny == 1600


@pytest.mark.parametrize("key,value", [("explicit_user_authorization", False), ("g1_pass", False),
                                      ("source_pass", False), ("scope", "ALL_MINUTE_STRATEGIES")])
def test_scope_authority_and_prior_gates_cannot_be_bypassed(key, value):
    report = {"state": PASS, "source_pass": True, "g1_pass": True, "checks": {"all_windows": True},
              "scope": "EXACT_EIGHT_FIXED_5MIN_WINDOWS_ONLY", "explicit_user_authorization": True}
    report[key] = value
    with pytest.raises(ContractError):
        admit(report, lambda: pytest.fail("不得读取真实标签"))


def test_g3_keeps_42bp_and_stops_before_margins_when_density_fails():
    root = Path(__file__).resolve().parents[1]
    model = yaml.safe_load((root / "config/510300_nbs_1000_negative_information_drift_v2.yaml").read_text("utf-8"))
    prediction = pd.DataFrame({"X": [1., 1., 1.], "beta": [.01, -.01, .01],
                               "lcb": [.0041999, .01, .0042], "era": ["ERA_1", "ERA_2", "ERA_3"]})
    result, selected = g3_statistics(prediction, model)
    assert result["signal_count"] == 1
    assert not result["g3_pass"]
    assert result["state"] == "NO_VIEW_INSUFFICIENT_HIGH_MARGIN_SIGNAL_DENSITY"
    assert "stress_net_avoided_loss" not in selected.columns


def test_g3_synthetic_high_margin_signals_pass_fixed_stress_gates():
    root = Path(__file__).resolve().parents[1]
    model = yaml.safe_load((root / "config/510300_nbs_1000_negative_information_drift_v2.yaml").read_text("utf-8"))
    prediction = pd.DataFrame({"X": 1., "beta": .01, "lcb": .005,
                               "era": ["ERA_1"] * 6 + ["ERA_2"] * 6,
                               "Y": np.linspace(.006, .008, 12)})
    result, selected = g3_statistics(prediction, model)
    assert result["g3_pass"]
    assert result["stress_net_hit_rate"] == 1
    np.testing.assert_allclose(selected.stress_net_avoided_loss, prediction.Y - .0024)
