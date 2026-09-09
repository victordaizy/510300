"""测量合同边界测试，全为合成数据，不访问真实事件标签。"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from research.stk_mins_source_admission_v2 import measure, standard_clocks, rebuild_ledger


def fixture():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/510300_stk_mins_source_admission_v2.yaml").read_text("utf-8"))
    cfg["quality_gates"]["independent_overlap_start"] = "2024-01-02"
    cfg["quality_gates"]["independent_overlap_end"] = "2024-01-02"
    minute = pd.DataFrame([{"ts_code": "510300.SH", "trade_time": "2024-01-02 " + t,
                            "open": 4., "high": 4., "low": 4., "close": 4.,
                            "vol": 100., "amount": 400.} for t in sorted(standard_clocks())])
    daily = pd.DataFrame([{"ts_code": "510300.SH", "trade_date": "20240102", "vol": 241.,
                           "amount": 96.4, "high": 4.02, "low": 3.98, "close": 4.01}])
    events = pd.DataFrame([{"scheduled_date": "2024-01-02", "final_event_eligibility": True}])
    return cfg, minute, daily, events


def evaluate(mutate=None):
    cfg, minute, daily, events = fixture()
    independent = minute.copy()
    if mutate:
        minute = mutate(minute)
    return measure(minute, daily, pd.DatetimeIndex(["2024-01-02"]), independent, events, cfg)[0]


def test_ohlc_diagnostic_does_not_block_valid_vwap():
    result = evaluate()
    assert result["source_pass"]
    assert result["ohlc_diagnostic_only"]["high"]["above_0_001_days"] == 1


def test_vwap_outside_tick_tolerance_blocks_even_if_totals_pass():
    def mutate(frame):
        frame.loc[5, "amount"] += .2
        frame.loc[6, "amount"] -= .2
        return frame
    result = evaluate(mutate)
    assert result["checks"]["daily_amount_error"]
    assert not result["checks"]["all_bar_vwap_inside_low_high"]
    assert not result["source_pass"]


def test_duplicate_and_missing_window_never_hidden_by_bar_count():
    def mutate(frame):
        frame.loc[frame.trade_time.eq("2024-01-02 10:02:00"), "trade_time"] = "2024-01-02 10:01:00"
        return frame
    result = evaluate(mutate)
    assert not result["checks"]["all_event_windows"]
    assert not result["checks"]["duplicate_keys_zero"]


def test_nonfinite_measurement_and_unit_error_block():
    def mutate(frame):
        frame["vol"] /= 100
        frame.loc[0, "amount"] = np.nan
        return frame
    result = evaluate(mutate)
    assert not result["checks"]["daily_volume_error"]
    assert not result["checks"]["all_bar_vwap_defined"]


def test_equal_eras_do_not_change_event_identity_or_exclusions():
    dates = pd.bdate_range("2020-01-01", periods=86)
    parent = pd.DataFrame({"event_id": range(86), "scheduled_date": dates,
                           "model_event_ordinal": range(1, 87), "final_event_eligibility": True})
    quality = pd.DataFrame({"date": dates, "standard_241": True, "pre_complete": True})
    result = rebuild_ledger(parent, quality)
    assert result.era.value_counts().to_dict() == {"TRAINING_ORIGIN": 36, "ERA_1": 17, "ERA_2": 17, "ERA_3": 16}
    assert result.event_id.equals(parent.event_id)
