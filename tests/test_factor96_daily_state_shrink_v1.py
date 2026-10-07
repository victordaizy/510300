"""验证新校准时序和账户边界，不以历史收益挑选规则。"""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from research import factor96_daily_state_shrink_v1 as study


@pytest.mark.parametrize("mu,variance,expected", [
    (-.01, .01, 0.), (0., .01, 0.), (.004999, .01, 0.),
    (.005, .01, .125), (.01, .01, .25), (.015, .01, .375),
    (.02, .01, .5), (.5, .01, .5), (.01, 0., 0.), (np.nan, .01, 0.),
])
def test_five_levels_and_exact_decimal_boundary(mu, variance, expected):
    assert study.fraction(mu, variance) == expected


def test_monitor_stop_and_registered_recovery():
    assert study.monitor([3., 3.])["monitor_paused"]
    state = study.monitor([3., 3., -3., -3., -3., -3., -3.])
    assert not state["monitor_paused"] and state["monitor_restore_count"] == 1
    assert study.monitor([-2.]*30)["cusum"] == 0


def test_monitor_is_fixed_twenty_observation_window():
    assert study.monitor([100.]+[-1.]*20) == study.monitor([-1.]*20)


def synthetic_forecast(n=610):
    rng = np.random.default_rng(7621)
    dates = pd.bdate_range("2016-01-04", periods=n)
    f = pd.DataFrame({"date": dates, "A01": rng.normal(size=n), "E01": rng.uniform(size=n), "F01": rng.normal(size=n)})
    labels = pd.DataFrame({"gross_return5": .002+.005*f.A01+.02*rng.normal(size=n)})
    return f, labels


def test_nested_clock_and_current_two_year_lower_bound():
    f, y = synthetic_forecast()
    out, outer, inner = study.forecast(f, y)
    assert len(outer) > 0 and out.model_known.any()
    for row in outer:
        assert row["latest_training_exit_idx"] < row["decision_idx"]
        assert f.date.iloc[row["training_indices"][0]] >= row["lower"]
    for row in inner:
        assert row["training_last_idx"]+6 < row["validation_idx"]
        assert row["validation_exit_idx"] < row["decision_idx"]
        assert row["validation_idx"] % 5 == 0
        assert f.date.iloc[row["training_first_idx"]] >= row["lower"]


def test_future_labels_and_features_cannot_change_earlier_forecasts():
    f, y = synthetic_forecast(560)
    first, models, _ = study.forecast(f, y)
    changed_f, changed_y = f.copy(), y.copy()
    changed_f.loc[520:, study.FEATURES] *= 100
    changed_y.loc[520:, "gross_return5"] *= -200
    second, later_models, _ = study.forecast(changed_f, changed_y)
    pd.testing.assert_frame_equal(first.iloc[:520], second.iloc[:520])
    assert [r for r in models if r["decision_idx"] < 520] == [r for r in later_models if r["decision_idx"] < 520]


def test_missing_current_feature_is_no_view_and_zero_request():
    f, y = synthetic_forecast(500)
    f.loc[499, "E01"] = np.nan
    result, _, _ = study.forecast(f, y)
    assert result.loc[499, "model_status"] == "NO_VIEW"
    assert all(result.loc[499, "fraction_"+p] == 0 for p in study.POLICIES)


def test_external_lag_and_missing_breadth_not_zero():
    dates = pd.bdate_range("2020-01-01", periods=45)
    d = pd.DataFrame({"date": dates})
    price = pd.DataFrame({"date": dates, "wealth": 1+np.arange(45)/100, "es95": .02})
    cumulative = pd.DataFrame(.1, index=dates, columns=range(300))
    members = pd.DataFrame(True, index=dates, columns=range(300))
    cumulative.iloc[30, :7] = np.nan
    cover = pd.DataFrame({"date": dates, "aggregation_state": "VIEW_ALLOWED"})
    margin = pd.DataFrame({"date": dates, "market_rzye": 100+np.arange(45)})
    f = study.build_features(d, price, cumulative, members, cover, margin, 1)
    assert f.loc[30, "E01"] == 1
    assert pd.isna(f.loc[31, "E01"]) and f.loc[31, "feature_status"] == "NO_VIEW"
    assert f.loc[30, "external_stat_date"] == dates[29]
    assert f.loc[30, "F01"] == pytest.approx(129/124-1)


def engine():
    path = study.SOURCE/"code/account_engine.py"
    spec = importlib.util.spec_from_file_location("test_daily_shrink_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def market_and_forecasts(prices):
    dates = pd.bdate_range("2021-01-01", periods=len(prices))
    d = pd.DataFrame({"date": dates, "open": prices, "close": prices, "previous_close": [prices[0]]+prices[:-1], "dividend": 0.})
    f = pd.DataFrame({"date": dates, "wealth": np.array(prices)/prices[0], "es95": .02,
        "model_known": True, "feature_status": "VIEW_ALLOWED", "model_status": "VIEW_ALLOWED",
        "external_stat_idx": np.arange(len(prices))-1, "external_stat_date": pd.Series(dates).shift(), "fraction_FULL": .5})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return d, f, div


def test_no_addition_when_price_falls_below_cycle_entry():
    d, f, div = market_and_forecasts([10., 10., 9.5, 9.5, 9.5, 9.5])
    f.loc[:1, "fraction_FULL"] = .125
    ledger, decisions = study.simulate(d, f, div, "FULL", 200000, "STRESS", str(d.date.iloc[1].date()), str(d.date.iloc[-1].date()), engine())
    assert ledger.iloc[0].filled_quantity > 0
    assert decisions.iloc[2].origin_wealth < decisions.iloc[2].entry_wealth_before
    assert decisions.iloc[2].requested_quantity <= 0
    assert ledger.accounting_error.abs().max() < 1e-6


def test_twenty_day_exit_and_no_same_open_reentry():
    d, f, div = market_and_forecasts([10.]*25)
    ledger, decisions = study.simulate(d, f, div, "FULL", 200000, "BASE", str(d.date.iloc[1].date()), str(d.date.iloc[-1].date()), engine())
    closing = ledger[ledger.idx == 21].iloc[0]
    assert closing.filled_quantity < 0 and closing.shares == 0
    assert ledger[ledger.idx == 22].iloc[0].filled_quantity > 0
    assert (ledger.filled_quantity.clip(upper=0).abs() <= ledger.sellable_before).all()


def test_gap_down_cancels_increase_even_if_previous_close_allows_it():
    d, f, div = market_and_forecasts([10., 10., 9.6, 9.6, 9.6])
    f.loc[0, "fraction_FULL"] = .125
    ledger, decisions = study.simulate(d, f, div, "FULL", 200000, "BASE", str(d.date.iloc[1].date()), str(d.date.iloc[-1].date()), engine())
    row = decisions[decisions.origin_idx == 1].iloc[0]
    assert row.pre_open_request > 0 and row.requested_quantity == 0


def test_no_view_exit_is_visible_and_pending():
    d, f, div = market_and_forecasts([10.]*7)
    f.loc[2, ["model_known", "model_status"]] = [False, "NO_VIEW"]
    ledger, decisions = study.simulate(d, f, div, "FULL", 200000, "BASE", str(d.date.iloc[1].date()), str(d.date.iloc[-1].date()), engine())
    assert ledger[ledger.idx == 3].iloc[0].filled_quantity < 0
    assert decisions[decisions.origin_idx == 2].iloc[0].model_status == "NO_VIEW"
