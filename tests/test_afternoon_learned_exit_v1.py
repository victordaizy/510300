"""下午退出必须遵守历史模型时钟、交易日确认及账户现金权益。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.afternoon_exit_account_v1 import simulate_afternoon_exit
from research.afternoon_learned_exit_v1 import AfternoonPreview, partial_values
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=138)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "wealth": 1., "total_simple": 0., "mom5": 0., "mom20": 0., "sma120": 0., "vol20": 0.})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "minute_participation_cap": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(len(data), dtype=int), "exit": {1: np.zeros(len(data), dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, dividends, cfg, cost, str(dates[130].date()), rule, spec]


def model(data, t, value):
    return {"fit_index": t, "fit_time": str(data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5)), "latest_exit_index": t - 1,
            "status": "FIT_COMPLETE", "model": {"kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8,
                "feature_clip": 5., "coefficients": [0.] * 8, "intercept": value}}


def snapshots(data, volume=1e6, price=9.8):
    return {date: {"signal_price": 9.9, "execution_price": price, "execution_volume": volume,
                   "signal_label": date + pd.Timedelta(hours=14, minutes=29),
                   "execution_label": date + pd.Timedelta(hours=14, minutes=46)} for date in data.date}


def controllers(data):
    models = [model(data, 125, -.01), model(data, 131, .01)]
    daily = ExitController(data, models)
    return daily, AfternoonPreview(data, models, daily)


def test_current_close_and_future_daily_data_do_not_change_afternoon_factors():
    args = setup()
    data, t = args[0], 131
    cycle = {"entry_index": 130, "entry_cost_cny": 100000., "mode": 1}
    original = partial_values(data, t, cycle, 99000., 100000., 9.9)
    changed = data.copy()
    for name in ["open", "close", "wealth", "total_simple", "mom5", "mom20", "sma120", "vol20"]:
        changed.loc[t:, name] = 99999.
    np.testing.assert_array_equal(original, partial_values(changed, t, cycle, 99000., 100000., 9.9))
    assert np.isclose(original[4], np.log(.99))
    assert np.isclose(original[6], .99 / ((119 + .99) / 120) - 1)
    assert np.isclose(original[7], np.std([0.] * 19 + [-.01], ddof=1) * np.sqrt(242))


def test_month_start_afternoon_cannot_use_that_days_close_model_or_double_count():
    data = setup()[0]
    daily, preview = controllers(data)
    daily.cycle_id, daily.negative_count = 7, 1
    cycle = {"cycle_id": 7, "entry_index": 130, "entry_cost_cny": 100000., "mode": 1}
    for _ in range(2):
        answer = preview(131, cycle, 99000., 100000., 9.9, data.date.iloc[131] + pd.Timedelta(hours=14, minutes=31))
        assert answer["continuation_prediction"] == -.01 and answer["learned_exit_requested"]
        assert answer["learning_fit_time"] == pd.Timestamp(preview.models[0]["fit_time"])
        assert daily.negative_count == 1 and daily.cycle_id == 7


def test_missing_historical_wealth_is_not_silently_dropped_from_mean():
    data = setup()[0]
    data.loc[129, "wealth"] = np.nan
    cycle = {"entry_index": 130, "entry_cost_cny": 100000., "mode": 1}
    assert np.isnan(partial_values(data, 131, cycle, 99000., 100000., 9.9)).all()


def test_missing_minute_data_recovers_original_full_account():
    args = setup()
    args[3].update(commission=.0002, minimum=5., slippage=.0005)
    daily, preview = controllers(args[0])
    original = simulate_rearmed_exit(*args, ExitController(args[0], daily.models))
    revised = simulate_afternoon_exit(*args, daily, preview, {})
    for left, right in zip(original, revised[:3]):
        assert_frame_equal(left, right[left.columns])


def test_afternoon_sells_only_old_inventory_and_records_actual_intraday_price():
    args = setup()
    daily, preview = controllers(args[0])
    ledger, decisions, cycles, afternoon = simulate_afternoon_exit(*args, daily, preview, snapshots(args[0]))
    assert afternoon.iloc[0].learning_status == "NOT_ALLOWED_BUY_DAY_T_PLUS_ONE"
    assert len(cycles) == 1 and cycles.holding_intervals.iloc[0] == 1
    assert cycles.exit_date.iloc[0] == args[0].date.iloc[131]
    assert np.isclose(cycles.net_profit_cny.iloc[0], -2000.)
    assert np.isclose(ledger.equity.iloc[-1], 98000.)
    assert ledger.accounting_error.abs().max() < 1e-6


def test_future_execution_volume_does_not_change_request_and_failed_exit_stays_pending():
    args = setup()
    for volume in [1e6, 100.]:
        daily, preview = controllers(args[0])
        ledger, decisions, cycles, afternoon = simulate_afternoon_exit(*args, daily, preview, snapshots(args[0], volume))
        signal = afternoon[afternoon.date == args[0].date.iloc[131]].iloc[0]
        assert signal.learned_exit_requested and signal.requested_quantity == -10000
        expected_day = 131 if volume == 1e6 else 132
        assert cycles.exit_date.iloc[0] == args[0].date.iloc[expected_day]
        if volume == 100.:
            assert signal.execution_status == "UNFILLED_MINUTE_CAPACITY"
            close = decisions[decisions.origin_index == 131].iloc[0]
            assert close.continuation_prediction > 0 and close.requested_quantity < 0


def test_missing_future_execution_price_preserves_known_signal_and_exit_request():
    args = setup()
    daily, preview = controllers(args[0])
    ledger, decisions, cycles, afternoon = simulate_afternoon_exit(*args, daily, preview, snapshots(args[0], price=np.nan))
    signal = afternoon[afternoon.learned_exit_requested].iloc[0]
    assert signal.execution_status == "UNFILLED_NO_EXECUTION_PRICE"
    assert cycles.exit_date.iloc[0] == args[0].date.iloc[132]


def test_afternoon_sale_preserves_already_registered_dividend_receivable():
    args = setup()
    data = args[0]
    args[1] = pd.DataFrame([{"record_date": data.date.iloc[130], "ex_date": data.date.iloc[131],
                             "payment_date": data.date.iloc[134], "cash_dividend_per_share": .2}])
    data.loc[131, "dividend"] = .2
    daily, preview = controllers(data)
    ledger, decisions, cycles, afternoon = simulate_afternoon_exit(*args, daily, preview, snapshots(data))
    assert ledger.iloc[1].shares == 0 and ledger.iloc[1].dividend_receivable == 2000.
    assert ledger.dividend_paid.sum() == 2000. and cycles.dividend_cny.iloc[0] == 2000.
    assert np.isclose(ledger.equity.iloc[-1], 100000.)


def test_afternoon_sale_before_record_close_does_not_receive_new_dividend():
    args = setup()
    data = args[0]
    args[1] = pd.DataFrame([{"record_date": data.date.iloc[131], "ex_date": data.date.iloc[132],
                             "payment_date": data.date.iloc[134], "cash_dividend_per_share": .2}])
    data.loc[132, "dividend"] = .2
    daily, preview = controllers(data)
    ledger, decisions, cycles, afternoon = simulate_afternoon_exit(*args, daily, preview, snapshots(data))
    assert ledger.dividend_recognized.sum() == 0 and ledger.dividend_paid.sum() == 0
