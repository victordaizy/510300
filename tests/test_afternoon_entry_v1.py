"""检验下午信号的可见信息、实际入场及原退出和登记日权益。"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research.afternoon_entry_inputs_v1 import partial_entry
from research.afternoon_entry_account_v1 import simulate_afternoon_entry
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=136)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
        "feature_valid": True, "overnight_log": np.tile([-.001, .001], 68), "intraday_log": .002})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "minute_participation_cap": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.zeros(len(data), int), "exit": {1: np.zeros(len(data), bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, div, cfg, cost, str(dates[130].date()), rule, spec]


def snapshots(data, volume=1e7, price=10.):
    return {date: {"signal_price": 10., "execution_price": price, "execution_volume": volume,
        "signal_label": date + pd.Timedelta(hours=14, minutes=29), "execution_label": date + pd.Timedelta(hours=14, minutes=46)} for date in data.date}


def preview(t, price, time):
    return {"entry_signal": True, "factor_status": "SYNTHETIC_COMPLETE_INPUT"}


def test_partial_factor_is_manual_past59_plus_known_current_and_dividend():
    data, t = setup()[0], 130
    data.loc[t, ["open", "dividend"]] = [9.9, .1]
    answer = partial_entry(data, t, 9.95)
    history = (data.intraday_log.iloc[t-60:t] - data.overnight_log.iloc[t-60:t]).to_numpy()
    current = np.log(10.05 / 10.)
    values = np.r_[history[1:], current]
    assert np.isclose(answer["previous_d60"], history.sum() / (history.std(ddof=1) * np.sqrt(60)))
    assert np.isclose(answer["partial_d60"], values.sum() / (values.std(ddof=1) * np.sqrt(60)))
    changed = data.copy()
    changed.loc[t:, ["close", "intraday_log", "overnight_log"]] = 9999.
    changed.loc[t:, "feature_valid"] = False
    assert answer == partial_entry(changed, t, 9.95)


@pytest.mark.parametrize("kind", ["missing_history", "missing_validity", "zero_scale"])
def test_incomplete_or_constant_history_preserves_no_view(kind):
    data = setup()[0]
    if kind == "missing_history":
        data.loc[100, "intraday_log"] = np.nan
    elif kind == "missing_validity":
        data["feature_valid"] = data.feature_valid.astype(object)
        data.loc[129, "feature_valid"] = pd.NA
    else:
        data["intraday_log"], data["overnight_log"] = 0., 0.
    answer = partial_entry(data, 130, 10.)
    assert not answer["entry_signal"] and answer["factor_status"].startswith("NO_VIEW")
    assert np.isnan(answer["partial_d60"])


def test_no_minute_recovers_original_account_and_daily_decisions():
    args = setup()
    args[5]["entry"][129:132] = 1
    args[5]["exit"][1][131] = True
    args[3].update(commission=.0002, minimum=5., slippage=.0005)
    old = simulate_rearmed_exit(*args)
    new = simulate_afternoon_entry(*args, preview=preview, snapshots={})
    for left, right in zip(old, new[:3]):
        assert_frame_equal(left, right[left.columns])


def test_entry_profit_begins_at_later_actual_fill_not_signal_or_open():
    args = setup()
    ledger, decisions, cycles, afternoon = simulate_afternoon_entry(*args, preview=preview, snapshots=snapshots(args[0], price=9.8))
    assert ledger.filled_quantity.iloc[0] == 10000
    assert np.isclose(ledger.price_pnl.iloc[0], 2000.)
    assert np.isclose(ledger.equity.iloc[0], 102000.)
    assert cycles.entry_origin.iloc[0] == args[0].date.iloc[130] + pd.Timedelta(hours=14, minutes=31)
    assert ledger.execution_clock.iloc[0] == "AFTERNOON_LATER_BAR_OPEN_PROXY"
    assert ledger.accounting_error.abs().max() < 1e-6


@pytest.mark.parametrize("price,volume,expected", [(np.nan, 1e7, "UNFILLED_NO_EXECUTION_PRICE"), (10., 100., "UNFILLED_MINUTE_CAPACITY"), (11., 1e7, "UNFILLED_UP_LIMIT")])
def test_future_price_capacity_and_limit_do_not_erase_request(price, volume, expected):
    args = setup()
    ledger, decisions, cycles, afternoon = simulate_afternoon_entry(*args, preview=preview, snapshots=snapshots(args[0], volume, price))
    assert (afternoon.requested_quantity == 10000).all()
    if expected == "UNFILLED_UP_LIMIT":
        assert afternoon.execution_status.str.contains("LIMIT").all()
    else:
        assert (afternoon.execution_status == expected).all()
    assert afternoon.filled_quantity.eq(0).all() and ledger.shares.eq(0).all()


def test_actual_fill_affordability_and_same_day_exit_waits_for_next_open():
    args = setup()
    args[3].update(commission=.0002, minimum=5., slippage=.0005)
    args[5]["exit"][1][130:] = True
    ledger, decisions, cycles, afternoon = simulate_afternoon_entry(*args, preview=preview, snapshots=snapshots(args[0], price=10.2))
    first = ledger.iloc[0]
    assert 0 < first.filled_quantity < afternoon.requested_quantity.iloc[0]
    assert decisions[decisions.origin_index.eq(130)].requested_quantity.iloc[0] == -first.filled_quantity
    assert ledger.filled_quantity.iloc[1] == -first.filled_quantity
    assert cycles.holding_intervals.ge(1).all()
    assert ledger.cash.ge(-1e-8).all() and ledger.accounting_error.abs().max() < 1e-6
    same_day = afternoon[afternoon.date.eq(args[0].date.iloc[131])].iloc[0]
    assert same_day.factor_status == "WAIT_PREVIOUS_ENTRY_CONDITION_TO_DISAPPEAR" and same_day.requested_quantity == 0
    next_day = afternoon[afternoon.date.eq(args[0].date.iloc[132])].iloc[0]
    assert next_day.factor_status == "WAIT_ORIGINAL_EXIT_COOLDOWN" and next_day.requested_quantity == 0


def test_actual_record_day_entry_receives_dividend_even_after_next_open_exit():
    args = setup()
    data = args[0]
    args[1] = pd.DataFrame({"record_date": [data.date.iloc[130]], "ex_date": [data.date.iloc[131]],
        "payment_date": [data.date.iloc[132]], "cash_dividend_per_share": [.1]})
    data.loc[131:, ["open", "close", "previous_close"]] = 9.9
    data.loc[131, "previous_close"], data.loc[131, "dividend"] = 10., .1
    args[5]["exit"][1][130] = True
    snap = {data.date.iloc[130]: snapshots(data)[data.date.iloc[130]]}
    ledger, decisions, cycles, afternoon = simulate_afternoon_entry(*args, preview=preview, snapshots=snap)
    assert ledger.dividend_recognized.sum() == 1000. and ledger.dividend_paid.sum() == 1000.
    assert np.isclose(ledger.equity.iloc[-1], 100000.) and np.isclose(cycles.net_profit_cny.iloc[0], 0.)


def test_final_open_has_no_afternoon_entry_and_old_signal_cannot_rearm_intraday():
    args = setup()
    args[5]["entry"][:] = 1
    args[5]["exit"][1][130] = True
    ledger, decisions, cycles, afternoon = simulate_afternoon_entry(*args, preview=preview, snapshots=snapshots(args[0]))
    assert len(cycles) == 1 and ledger.filled_quantity.gt(0).sum() == 1
    assert afternoon.factor_status.eq("WAIT_PREVIOUS_ENTRY_CONDITION_TO_DISAPPEAR").all()
    assert args[0].date.iloc[-1] not in set(afternoon.date)
