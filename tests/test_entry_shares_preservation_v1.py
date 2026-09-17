"""检查固定实际份额、零与未知、受阻重判、分红及未来隔离。"""
import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account
from research.entry_shares_preservation_account_v1 import preservation_request, simulate_preserved_shares


def fixture():
    dates = pd.bdate_range('2020-01-01', periods=10)
    frame = pd.DataFrame({'date': dates, 'open': 10., 'close': 10., 'previous_close': 10., 'dividend': 0., 'variance60': .0004})
    dividends = pd.DataFrame(columns=['record_date', 'ex_date', 'payment_date', 'cash_dividend_per_share'])
    cfg = {'initial_capital': 100000., 'lot': 100, 'tick': .001, 'limit_fraction': .1, 'weight_band': .1}
    cost = {'commission': .0002, 'minimum': 5., 'slippage': .0005}
    return frame, dividends, cfg, cost


def run(targets, frame=None, dividends=None):
    base, div, cfg, cost = fixture()
    frame, dividends = (base if frame is None else frame), (div if dividends is None else dividends)
    return simulate_preserved_shares(frame, dividends, cfg, cost, str(frame.date.iloc[1].date()),
        'ENTRY_SHARES_PRESERVATION', targets=np.array(targets, float), event_mask=np.ones(len(frame), bool))


def test_empty_account_enters_below_band_and_known_positive_never_rebalances():
    cfg = fixture()[2]
    account = Account(100000.)
    assert preservation_request(account, 10., .01, cfg)['requested_quantity'] == 100
    account.shares, account.cash = 2000, 80000.
    for target in [1., .9, .01, 1e-12]:
        result = preservation_request(account, 10., target, cfg)
        assert result['requested_quantity'] == 0 and result['reference_weight'] == target
        assert result['action'] == '已知正目标，保持本笔实际份额'
    assert preservation_request(account, 10., 0., cfg)['requested_quantity'] == -2000
    with pytest.raises(ValueError, match='超出范围'):
        preservation_request(account, 10., 1.1, cfg)


def test_positive_tiny_unknown_zero_exit_and_new_entry_use_actual_own_account():
    targets = [.3, .9, 1e-12, np.nan, 0., 0., .2, .8, .5, .5]
    ledger, decisions = run(targets)
    assert ledger.iloc[:4].shares.tolist() == [3000]*4
    assert ledger.iloc[4].shares == 0 and ledger.iloc[4].filled_quantity == -3000
    assert ledger.iloc[6].filled_quantity > 0 and ledger.iloc[6].filled_quantity != 3000
    assert ledger.iloc[7].filled_quantity == 0
    unknown = decisions[decisions.origin_index.eq(3)].iloc[0]
    assert pd.isna(unknown.reference_weight) and unknown.requested_quantity == 0
    assert ledger.iloc[-1].shares == 0 and ledger.iloc[-1].filled_quantity < 0


def test_blocked_zero_exit_is_reconsidered_using_recovered_positive_target():
    frame = fixture()[0]
    frame.loc[2, 'open'] = 9.
    ledger, decisions = run([.3, 0., .2, 0., 0., 0., 0., 0., 0., 0.], frame)
    assert ledger.iloc[1].status == 'UNFILLED_DIRECTIONAL_LIMIT'
    assert ledger.iloc[2].shares == 3000 and ledger.iloc[2].requested_quantity == 0
    assert ledger.iloc[3].shares == 0 and ledger.iloc[3].filled_quantity == -3000
    assert decisions.iloc[2].reference_weight == .2


def test_failed_buy_does_not_fix_nonexistent_shares_and_next_request_uses_latest_target():
    frame = fixture()[0]
    frame.loc[1, 'open'] = 11.
    ledger, decisions = run([.5, .2, .9, .9, .9, .9, .9, .9, .9, .9], frame)
    assert ledger.iloc[0].status == 'UNFILLED_DIRECTIONAL_LIMIT' and ledger.iloc[0].shares == 0
    assert decisions.iloc[0].requested_quantity == 5000 and decisions.iloc[1].requested_quantity == 2000
    assert ledger.iloc[1].filled_quantity == 2000
    assert ledger.iloc[2:-1].filled_quantity.eq(0).all()


def test_dividends_do_not_reinvest_and_terminal_unknown_still_exits():
    frame = fixture()[0]
    frame.loc[3:, ['open', 'close']] = 9.9
    frame.loc[4:, 'previous_close'] = 9.9
    frame.loc[3, 'dividend'] = .1
    dividends = pd.DataFrame({'record_date': [frame.date.iloc[2]], 'ex_date': [frame.date.iloc[3]],
        'payment_date': [frame.date.iloc[5]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = run([.4]*8+[np.nan, np.nan], frame, dividends)
    assert ledger.iloc[:-1].shares.eq(4000).all()
    assert ledger.iloc[2].dividend_recognized == pytest.approx(400.)
    assert ledger.iloc[4].dividend_paid == pytest.approx(400.)
    assert ledger.filled_quantity.gt(0).sum() == 1 and ledger.filled_quantity.lt(0).sum() == 1
    assert pd.isna(decisions.iloc[-1].reference_weight) and ledger.iloc[-1].requested_quantity == -4000
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.iloc[-1].shares == 0


def test_future_targets_and_prices_do_not_change_earlier_decisions_or_fills():
    targets = np.full(10, .3)
    original_ledger, original_decisions = run(targets)
    frame = fixture()[0]
    frame.loc[6:, ['open', 'close', 'previous_close']] = 10.2
    targets[5:] = [0., .8, np.nan, .01, .9]
    changed_ledger, changed_decisions = run(targets, frame)
    pd.testing.assert_frame_equal(original_ledger.iloc[:5], changed_ledger.iloc[:5])
    pd.testing.assert_frame_equal(original_decisions.iloc[:5], changed_decisions.iloc[:5])
