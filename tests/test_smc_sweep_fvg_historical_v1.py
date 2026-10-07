"""检验因果时间顺序、午休边界、入场延迟和分红归属。"""
import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smc_study", ROOT / "research/smc_sweep_fvg_historical_v1.py")
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def bars():
    rows = [
        (4.02, 4.03, 3.990, 3.995),
        (3.995, 4.010, 3.994, 4.005),
        (4.005, 4.040, 4.004, 4.035),
        (4.035, 4.055, 4.032, 4.045),
        (4.045, 4.052, 4.020, 4.040),
    ]
    b = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    b["time"] = pd.date_range("2022-01-04 09:35", periods=len(b), freq="5min")
    b["session"] = "AM"
    return b


def test_late_gap_cannot_rewrite_earlier_confirmation():
    b = bars()
    event, signals = STUDY.detect_day(b, 4., 4.02, pd.Timestamp("2022-01-04"))
    assert event["confirmation_time"] == pd.Timestamp("2022-01-04 09:45")
    assert [s["policy"] for s in signals] == ["P0_RECLAIM", "P1_RANGE_BREAK"]
    assert not event["fvg_at_confirmation"]


def test_gap_known_at_confirmation_then_later_retest():
    b = bars()
    b.loc[2, ["high", "close"]] = [4.029, 4.025]
    event, signals = STUDY.detect_day(b, 4., 4.02, pd.Timestamp("2022-01-04"))
    assert event["state"] == "RETEST_CONFIRMED"
    assert [s["policy"] for s in signals] == STUDY.BASE_POLICIES
    assert signals[2]["signal_time"] < signals[3]["signal_time"]
    assert signals[2]["gap_lower"] == 4.01
    for s in signals:
        _, prefix = STUDY.detect_day(b[b.time.le(s["signal_time"])], 4., 4.02, pd.Timestamp("2022-01-04"))
        assert s["policy"] in {p["policy"] for p in prefix}


def test_no_lunch_crossing_or_open_gap_relabel():
    b = bars()
    b.loc[1:, "session"] = "PM"
    event, signals = STUDY.detect_day(b, 4., 4.02, pd.Timestamp("2022-01-04"))
    assert event["state"] == "SWEEP_UNRECLAIMED" and signals == []
    event, signals = STUDY.detect_day(bars(), 4., 3.99, pd.Timestamp("2022-01-04"))
    assert event["state"] == "OPEN_BELOW_REFERENCE" and signals == []


def test_first_sweep_failure_does_not_restart_later():
    b = bars()
    b.loc[2, ["open", "high", "low", "close"]] = [4.005, 4.010, 3.980, 3.985]
    event, signals = STUDY.detect_day(b, 4., 4.02, pd.Timestamp("2022-01-04"))
    assert event["state"] == "STRUCTURE_FAILED_BEFORE_CONFIRM"
    assert len(signals) == 1


def test_delayed_entry_uses_three_minutes_and_rejects_lunch():
    sig = {"signal_time": pd.Timestamp("2022-01-04 10:00"), "structure_low": 3.99}
    mi = pd.DataFrame({"open": [4.01], "vol": [1000]}, index=[pd.Timestamp("2022-01-04 10:03")])
    row, state = STUDY.entry_for_signal(sig, mi)
    assert row.open == 4.01 and state == "MODEL_FILL_PROXY_NOT_ACTUAL_EXECUTION"
    sig["signal_time"] = pd.Timestamp("2022-01-04 11:30")
    assert STUDY.entry_for_signal(sig, mi)[0] is None


def test_lots_costs_and_dividend_entitlement():
    result = STUDY.trade_outcome(4., 4., 0., "STRESS")
    assert result["quantity"] % 100 == 0
    assert result["net_return"] < -.002
    d = pd.DataFrame({"record_date": pd.to_datetime(["2022-01-04"]), "ex_date": pd.to_datetime(["2022-01-05"]), "payment_date": pd.to_datetime(["2022-01-10"]), "cash_dividend_per_share": [.1]})
    assert STUDY.dividend_between(d, pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-05")) == .1
    assert STUDY.dividend_between(d, pd.Timestamp("2022-01-05"), pd.Timestamp("2022-01-06")) == 0.


def test_context_does_not_use_current_daily_close_or_current_volume_in_baseline():
    days = pd.bdate_range("2020-01-02", periods=300)
    daily = pd.DataFrame({"close": 4 + .02 * np.sin(np.arange(300) / 3)}, index=days)
    b = pd.DataFrame({"date": days, "time": days + pd.Timedelta(hours=10), "session": "AM", "slot": 5, "close": daily.close.to_numpy(), "volume": 100.})
    b.loc[299, "volume"] = 500.
    div = pd.DataFrame({"ex_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})
    features = STUDY.context_features(b, daily, div)
    assert features.relative_volume.iloc[-1] == 5.
    assert features.context_available.iloc[-1]
    changed_daily = daily.copy()
    changed_daily.iloc[-1, 0] = 40.
    changed = STUDY.context_features(b, changed_daily, div)
    assert features.rv20_relative_prior.iloc[-1] == changed.rv20_relative_prior.iloc[-1]
    prefix = STUDY.context_features(b.iloc[:-1], daily.iloc[:-1], div)
    columns = ["macd_hist_bps", "macd_hist_change_bps", "rv20_relative_prior", "relative_volume"]
    assert np.allclose(features[columns].iloc[:-1], prefix[columns], equal_nan=True)


def test_exit_open_does_not_observe_later_intraminute_extreme():
    day = pd.Timestamp("2022-01-04")
    sig = pd.DataFrame([{ "policy": "P0_RECLAIM", "event_id": "20220104", "date": day, "signal_time": day + pd.Timedelta(hours=10), "structure_low": 3.99 }])
    minute = pd.DataFrame({"time": pd.to_datetime(["2022-01-04 10:03", "2022-01-04 10:08"]), "open": [4., 4.01], "close": [4., 4.01], "high": [4.001, 5.], "low": [3.999, 3.], "vol": [1000, 1000]})
    daily = pd.DataFrame({"open": [4., 4., 4.01], "close": [4., 4., 4.01]}, index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]))
    div = pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "cash_dividend_per_share": pd.Series(dtype=float)})
    labels, _ = STUDY.labels(sig, minute, daily, div)
    short = labels[labels.horizon.eq("M5") & labels.cost.eq("STRESS")].iloc[0]
    assert np.isclose(short.raw_price_mfe, 4.01 / 4. - 1)
    assert np.isclose(short.raw_price_mae, 3.999 / 4. - 1)
    overnight = labels[labels.horizon.eq("T1_OPEN")].iloc[0]
    assert not overnight.raw_path_complete and pd.isna(overnight.raw_price_mae)
