"""验证原有仓位复现、预测续期、缺失不续期和受限退出持续请求。"""
import numpy as np
import pandas as pd
from research.event_clock_account_v1 import simulate_event_account
from research.forward_eps_utility_exit_account_v1 import simulate_utility_exit_account


def fixture(n=9):
    dates = pd.bdate_range('2025-01-01', periods=n)
    data = pd.DataFrame({'date': dates, 'open': 10., 'close': 10., 'previous_close': 10.,
                         'dividend': 0., 'sma120': .1, 'variance60': .0001})
    dividends = pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    config = {'evaluation_start': str(dates[1].date()), 'initial_capital': 200000., 'lot': 100,
              'tick': .001, 'limit_fraction': .1, 'gamma': 4., 'weights': [0., .25, .5, .75, 1.],
              'horizon': 60, 'forecast_validity_trading_days': 3}
    cost = {'commission': 0., 'minimum': 0., 'slippage': 0.}
    prediction = np.full(n, np.nan)
    prediction[0] = .1
    mask = np.zeros(n, bool)
    mask[0] = True
    return data, dividends, config, cost, prediction, mask


def test_disabled_exits_match_original_including_partial_positions_and_dividend():
    args = list(fixture())
    data, dividends, config, cost, prediction, mask = args
    prediction[2], prediction[4], prediction[6] = .006, .012, -.1
    mask[[2,4,6]] = True
    data.loc[3:, ['open','close','previous_close']] = 9.9
    data.loc[3,'previous_close'] = 10.
    data.loc[3,'dividend'] = .1
    args[1] = pd.DataFrame({'record_date':[data.date.iloc[2]],'ex_date':[data.date.iloc[3]],
                            'payment_date':[data.date.iloc[5]],'cash_dividend_per_share':[.1]})
    actual, decisions = simulate_utility_exit_account(*args)
    expected, _ = simulate_event_account(data,args[1],config,cost,config['evaluation_start'],'EPS',
                                          prediction=prediction,horizon=60,event_mask=mask)
    for column in ['equity','cash','shares','dividend_receivable','filled_quantity','net_return']:
        np.testing.assert_array_equal(actual[column], expected[column])
    assert len(set(actual.shares)) >= 3
    assert not decisions.new_exit_trigger.any()


def test_new_valid_forecast_renews_clock_instead_of_forcing_holding_expiry():
    args = list(fixture())
    args[4][2], args[5][2] = .1, True
    ledger, decisions = simulate_utility_exit_account(*args, expiry_enabled=True)
    assert decisions.loc[decisions.new_exit_trigger,'origin_index'].tolist() == [5]
    assert ledger.loc[ledger.filled_quantity.lt(0),'date'].tolist() == [args[0].date.iloc[6]]


def test_missing_month_end_does_not_reset_forecast_age():
    args = list(fixture())
    args[5][2] = True
    ledger, decisions = simulate_utility_exit_account(*args, expiry_enabled=True)
    assert decisions.loc[decisions.new_exit_trigger,'origin_index'].tolist() == [3]
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[3]),'shares'].iloc[0] > 0
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[4]),'shares'].iloc[0] == 0


def test_limit_blocked_exit_is_not_cancelled_by_recovery():
    args = list(fixture())
    data = args[0]
    data.loc[[0,1],'sma120'] = -.1
    data.loc[2,'open'] = 9.
    ledger, decisions = simulate_utility_exit_account(*args, trend_enabled=True)
    assert ledger.loc[ledger.date.eq(data.date.iloc[2]),'status'].iloc[0] == 'UNFILLED_DIRECTIONAL_LIMIT'
    assert decisions.loc[decisions.origin_index.eq(2),'signal_state'].iloc[0] == 'EXIT_PENDING_UNTIL_FILLED'
    assert ledger.loc[ledger.date.eq(data.date.iloc[3]),'shares'].iloc[0] == 0


def test_reentry_uses_original_next_month_end_without_new_entry_filter():
    args = list(fixture())
    args[0].loc[[0,1],'sma120'] = -.1
    args[4][4], args[5][4] = .1, True
    ledger, decisions = simulate_utility_exit_account(*args, trend_enabled=True)
    assert decisions.loc[decisions.requested_quantity.gt(0),'origin_index'].tolist() == [0,4]
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[3]),'shares'].iloc[0] == 0
