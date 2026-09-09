"""只使用合成价格验证附加诊断和原逐分钟硬门之间的区别。"""
import numpy as np
import pandas as pd

from research.nbs_v2_data_remediation import aggregate_measurement
from research.stk_mins_source_admission_v2 import measure
from tests.test_510300_nbs_v2_source import fixture


def test_window_pass_does_not_repair_original_bar_gate():
    cfg, minute, daily, events = fixture()
    i = minute.index[minute.trade_time.eq("2024-01-02 10:01:00")][0]
    minute.loc[i, "amount"] += 1
    minute.loc[i + 1, "amount"] -= 1
    dates = pd.DatetimeIndex(["2024-01-02"])
    result = aggregate_measurement(minute, dates, cfg["windows"], .001 + 1e-12)
    strict = measure(minute, daily, dates, minute.copy(), events, cfg)[0]
    assert result.diagnostic_pass.all()
    assert not strict["source_pass"]
    assert not strict["checks"]["all_bar_vwap_inside_low_high"]


def test_zero_trade_pair_contributes_no_synthetic_price():
    cfg, minute, _, _ = fixture()
    minute.loc[minute.trade_time.eq("2024-01-02 10:01:00"), ["vol", "amount"]] = 0
    result = aggregate_measurement(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001)
    reaction = result.loc[result.window.eq("reaction")].iloc[0]
    assert reaction.diagnostic_pass
    assert reaction.volume_shares == 400
    assert reaction.amount_cny == 1600
    assert reaction.vwap == 4


def test_zero_entire_window_inconsistent_zero_and_missing_label_fail():
    cfg, original, _, _ = fixture()
    for kind in ["zero", "inconsistent_zero", "missing", "nonfinite"]:
        minute = original.copy()
        selected = minute.trade_time.str[-8:].isin(cfg["windows"]["reaction"])
        if kind == "zero":
            minute.loc[selected, ["vol", "amount"]] = 0
        elif kind == "inconsistent_zero":
            minute.loc[selected, "vol"] = 0
        elif kind == "missing":
            minute = minute.loc[minute.trade_time.ne("2024-01-02 10:02:00")]
        else:
            minute.loc[selected, "amount"] = np.nan
        result = aggregate_measurement(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001)
        assert not result.loc[result.window.eq("reaction"), "diagnostic_pass"].iloc[0]


def test_aggregate_range_violation_is_not_hidden():
    cfg, minute, _, _ = fixture()
    minute.loc[minute.trade_time.str[-8:].isin(cfg["windows"]["reaction"]), "amount"] = 500
    result = aggregate_measurement(minute, pd.DatetimeIndex(["2024-01-02"]), cfg["windows"], .001)
    assert not result.loc[result.window.eq("reaction"), "diagnostic_pass"].iloc[0]
