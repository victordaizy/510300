"""验证真实来源差分、时钟隔离、当前行排除和周原点去重。"""
import numpy as np
import pandas as pd

from scripts.collect_rmb_residual_sources_v1 import normalize_dukascopy, normalize_treasury
from research.rmb_residual_state_v1 import build_models, weekly_origins


def synthetic_sources():
    dates = pd.bdate_range("2017-01-02", periods=800)
    t = np.arange(len(dates))
    usd = np.sin(t*.07)*.01
    spread = np.cos(t*.013)*.12
    return pd.DataFrame({"date": dates, "available_at": dates.tz_localize("Asia/Shanghai")+pd.Timedelta(days=2, hours=23, minutes=59),
        "usd_change5": usd, "spread_change5": spread, "fx_change5": .001+.3*usd-.004*spread+.0004*np.sin(t*.317)})


def test_quote_delta_decoder_keeps_real_gaps_and_xml_units():
    payload = {"timestamp": 1451606400000, "shift": 86400000, "multiplier": .00001,
        "open": 6.5, "high": 6.6, "low": 6.4, "close": 6.51,
        "times": [2, 3], "opens": [0, 10], "highs": [0, 10], "lows": [0, 10], "closes": [0, 10], "volumes": [3., 4.]}
    frame = normalize_dukascopy(payload)
    assert len(frame) == 2
    assert frame.date.tolist() == [pd.Timestamp("2016-01-03"), pd.Timestamp("2016-01-06")]
    assert np.allclose(frame.close, [6.51, 6.5101])
    xml = b'<feed xmlns:m="urn:m" xmlns:d="urn:d"><m:properties><d:NEW_DATE>2020-01-02T00:00:00</d:NEW_DATE><d:BC_10YEAR>1.88</d:BC_10YEAR></m:properties></feed>'
    rates = normalize_treasury(xml)
    assert rates.us_10y.iloc[0] == 1.88


def test_current_fx_observation_never_enters_its_own_regression():
    source = synthetic_sources()
    day = pd.Timestamp("2019-10-01")
    a, training = build_models([day], source)
    idx = int(a.source_idx.iloc[0])
    modified = source.copy()
    modified.loc[idx, "fx_change5"] += .02
    b, _ = build_models([day], modified)
    assert a.source_admitted.iloc[0]
    assert max(training[0]["training_indices"]) < idx
    assert a[['intercept','beta_usd','beta_spread']].equals(b[['intercept','beta_usd','beta_spread']])
    assert np.isclose(b.residual_fx_change5.iloc[0]-a.residual_fx_change5.iloc[0], .02)


def test_future_unavailable_sources_do_not_change_current_model():
    source = synthetic_sources()
    day = pd.Timestamp("2019-10-01")
    a, _ = build_models([day], source)
    clock = day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)
    changed = source.copy()
    changed.loc[changed.available_at.gt(clock), ['fx_change5','usd_change5','spread_change5']] = 99.
    b, _ = build_models([day], changed)
    pd.testing.assert_frame_equal(a, b)
    assert a.source_available_at.iloc[0] <= clock


def test_weekly_cohort_uses_iso_week_and_first_available_trade_day():
    days = pd.to_datetime(['2019-12-27','2019-12-30','2019-12-31','2020-01-02','2020-01-03','2020-01-06'])
    assert weekly_origins(days).tolist() == [0, 1, 5]
