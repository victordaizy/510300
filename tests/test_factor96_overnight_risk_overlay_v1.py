"""核对隔夜财富分解、日前方差、阈值时钟和真实整手覆盖层状态。"""
import numpy as np
import pandas as pd

from research import intraday_overnight_increment_v1 as engine
from research.factor96_overnight_risk_overlay_v1 import (
    predictable_variance, risk_features, lag_features, overlay_transition, reduced_quantity, simulate,
)


def synthetic_market(n=360):
    random = np.random.default_rng(932)
    closes = 3*np.exp(np.cumsum(random.normal(.0002, .009, n)))
    opens = np.r_[3, closes[:-1]]*np.exp(random.normal(0, .004, n))
    return pd.DataFrame({"date": pd.bdate_range("2017-01-02", periods=n), "open": opens,
                         "close": closes, "dividend": 0., "previous_close": np.r_[3, closes[:-1]]})


def test_dividend_adjusted_sessions_compound_exactly():
    market = synthetic_market()
    market.loc[170, "dividend"] = .2
    f = risk_features(market)
    compound = (1+f.overnight_return)*(1+f.intraday_return)-1
    np.testing.assert_allclose(compound.iloc[1:], f.session_total_return.iloc[1:], atol=1e-14)


def test_pure_ex_dividend_price_drop_is_not_overnight_loss():
    market = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3),
                           "open": [10., 9., 9.], "close": [10., 9., 9.], "dividend": [0., 1., 0.]})
    result = risk_features(market)
    assert result.overnight_return.iloc[1] == 0
    assert result.intraday_return.iloc[1] == 0
    assert result.session_total_return.iloc[1] == 0


def test_d06_uses_both_sessions_in_denominator():
    market = synthetic_market()
    f = risk_features(market)
    i = 200
    a, b = f.overnight_return.iloc[i-59:i+1], f.intraday_return.iloc[i-59:i+1]
    expected = np.minimum(a, 0).pow(2).sum()/(a.pow(2)+b.pow(2)).sum()
    assert np.isclose(f.D06.iloc[i], expected)
    assert np.isclose(f.overnight_tail_mean.iloc[i], np.sort(a)[:3].mean())
    assert f.D06.dropna().between(0, 1).all()


def test_forecast_does_not_use_current_shock():
    returns = np.full(100, .01)
    returns[70] = -.20
    f = predictable_variance(returns)
    assert np.isclose(f[60], .0001) and np.isclose(f[70], .0001)
    assert np.isclose(f[71], .94*.0001+.06*.04)
    assert np.isnan(f[:60]).all()


def test_forecast_resets_after_missing_return():
    returns = np.full(180, .01)
    returns[80] = np.nan
    f = predictable_variance(returns)
    assert np.isfinite(f[80])
    assert np.isnan(f[81:141]).all()
    assert np.isclose(f[141], .0001)


def test_zero_variance_is_unknown_not_zero_risk_ratio():
    market = synthetic_market()
    market[["open", "close", "previous_close"]] = 3.
    f = risk_features(market)
    assert f.D06.isna().all() and f.P02.isna().all()
    assert not f.d06_known.any() and not f.p02_known.any()


def test_percentiles_exclude_current_day():
    market = synthetic_market()
    f = risk_features(market)
    for col in ["D06", "P02"]:
        assert np.isclose(f[col+"_q90"].iloc[350], f[col].iloc[98:350].quantile(.9))
        assert np.isclose(f[col+"_q70"].iloc[350], f[col].iloc[98:350].quantile(.7))
    changed = market.copy()
    changed.loc[350:, "close"] *= 1.8
    later = risk_features(changed)
    for col in ["D06_q90", "D06_q70", "P02_q90", "P02_q70", "forecast_variance"]:
        assert np.isclose(f[col].iloc[350], later[col].iloc[350])


def test_entire_feature_prefix_ignores_future_data():
    market = synthetic_market()
    full, prefix = risk_features(market), risk_features(market.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)


def test_full_triggers_on_either_high_variable():
    for d, p in [(True, False), (False, True), (True, True)]:
        assert overlay_transition(False, 0, True, d, p, False, False, "FULL", True) == (True, 0)
    assert overlay_transition(False, 0, True, True, True, False, False, "FULL", False) == (False, 0)


def test_mid_band_does_not_restore_and_both_low_needed():
    assert overlay_transition(True, 1, True, False, False, False, False, "FULL", True) == (True, 0)
    assert overlay_transition(True, 1, True, False, False, True, False, "FULL", True) == (True, 0)
    assert overlay_transition(True, 0, True, False, False, True, True, "FULL", True) == (True, 1)
    assert overlay_transition(True, 1, True, False, False, True, True, "FULL", True) == (False, 0)


def test_missing_resets_recovery_without_clearing_cut():
    assert overlay_transition(True, 1, False, False, False, True, True, "FULL", True) == (True, 0)


def test_single_condition_recovery_and_lot_floor():
    assert overlay_transition(True, 1, True, False, True, True, False, "D06_ONLY", True) == (False, 0)
    assert overlay_transition(True, 1, True, True, False, False, True, "P02_ONLY", True) == (False, 0)
    assert reduced_quantity(100, True) == 0 and reduced_quantity(1300, True) == 600


def test_extra_lag_moves_signal_but_not_current_es():
    market = synthetic_market()
    price = pd.DataFrame({"date": market.date, "wealth": market.close, "es95": np.arange(len(market))*.001,
                          "pressure5": 0., "trend20": 0., "log_rv5_rv60": 0.})
    risk = risk_features(market)
    f = lag_features(market, price, risk, 1)
    assert f.stat_idx.iloc[-1] == len(market)-2 and f.stat_date.iloc[-1] == market.date.iloc[-2]
    assert f.D06.iloc[-1] == risk.D06.iloc[-2] and f.es95.iloc[-1] == price.es95.iloc[-1]


def test_zero_share_cut_keeps_cycle_and_restores_after_two_low_days():
    dates = pd.bdate_range("2020-01-02", periods=12)
    market = pd.DataFrame({"date": dates, "open": 3., "close": 3., "previous_close": 3., "dividend": 0.})
    x = pd.DataFrame({"date": dates, "price_long": True, "common_known": True, "d06_known": True, "p02_known": True,
        "d06_high": False, "p02_high": False, "d06_low": True, "p02_low": True, "es95": .03,
        "stat_date": dates, "stat_idx": np.arange(12), "D06": .2, "D06_q90": .4, "D06_q70": .3,
        "P02": .5, "P02_q90": 2., "P02_q70": 1., "forecast_variance": .0001, "overnight_tail_mean": -.01})
    x.loc[0:2, ["d06_high", "d06_low"]] = [True, False]
    x.loc[8:, "price_long"] = False
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate(market, x, dividends, "FULL", 1000, "STRESS", dates[1], dates[-1], engine)
    assert decisions.iloc[0].initial_entry and not decisions.iloc[0].overlay_active
    assert ledger.shares.iloc[0] == 100
    cut = decisions[decisions.origin.eq(dates[1])].iloc[0]
    assert cut.overlay_active and cut.filled_quantity == -100 and cut.base_live
    waiting = decisions[decisions.origin.eq(dates[3])].iloc[0]
    assert waiting.overlay_active and waiting.clear_streak == 1 and not waiting.initial_entry
    restored = decisions[decisions.origin.eq(dates[4])].iloc[0]
    assert not restored.overlay_active and restored.filled_quantity == 100 and not restored.initial_entry
    expiry = decisions[decisions.date.eq(dates[6])].iloc[0]
    assert expiry.base_exit_pending and expiry.filled_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7
    sold = ledger.filled_quantity.lt(0)
    assert (ledger.loc[sold, "filled_quantity"].abs() <= ledger.loc[sold, "sellable_before"]).all()
