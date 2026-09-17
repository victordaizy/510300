"""未来隔离、训练确认隔离、规则选择和完整退出的针对性测试。"""
import numpy as np
import pandas as pd
import pytest

from research.intraday_overnight_increment_v1 import Account
from research.single_factor_direct_stump_v1 import (
    CONFIG, read, POLICY_ORDER, policies_from_training, targets_for_policy, confirm_choices, request, learn_at,
)


def synthetic():
    rng = np.random.default_rng(71991)
    date = pd.bdate_range('2011-01-03', periods=720)
    close = 4 * np.exp(np.cumsum(rng.normal(.0003, .012, len(date))))
    data = pd.DataFrame({'date': date, 'close': close, 'open': close * np.exp(rng.normal(0, .002, len(date)))})
    data['previous_close'] = data.close.shift(1)
    data['dividend'] = 0.0
    data['total_simple'] = data.close.pct_change()
    data['vol20'] = data.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    data['variance60'] = data.total_simple.rolling(60).var(ddof=1)
    data['five_day_total_return'] = data.close / data.close.shift(5) - 1
    dividends = pd.DataFrame({key: pd.Series(dtype='datetime64[ns]') for key in ['record_date', 'ex_date', 'payment_date']})
    dividends['cash_dividend_per_share'] = pd.Series(dtype=float)
    return data, dividends


@pytest.fixture(scope='module')
def learned():
    data, dividends = synthetic()
    cfg = read(CONFIG)
    record, ledger = learn_at(data, dividends, 650, cfg, data.date.iloc[651])
    return data, dividends, cfg, record, ledger


def test_future_changes_leave_all_earlier_training_and_confirmation_results_unchanged(learned):
    data, dividends, cfg, first, ledgers = learned
    changed = data.copy()
    changed.loc[651:, ['open', 'close', 'five_day_total_return', 'vol20']] *= 11
    second, new_ledgers = learn_at(changed, dividends, 650, cfg, data.date.iloc[651])
    assert first == second
    pd.testing.assert_frame_equal(ledgers, new_ledgers)


def test_confirmation_prices_cannot_change_training_thresholds_or_training_scores(learned):
    data, dividends, cfg, first, _ = learned
    changed = data.copy()
    first_confirmation = first['confirmation_start_index']
    changed.loc[first_confirmation:, ['open', 'close']] *= 1.04
    changed['previous_close'] = changed.close.shift(1)
    changed['total_simple'] = changed.close.pct_change()
    changed['vol20'] = changed.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    changed['variance60'] = changed.total_simple.rolling(60).var(ddof=1)
    changed['five_day_total_return'] = changed.close / changed.close.shift(5) - 1
    second, _ = learn_at(changed, dividends, 650, cfg, data.date.iloc[651])
    assert first['policies'] == second['policies']
    assert first['training_scores'] == second['training_scores']
    assert first['training_end'] < first['confirmation_start'] <= first['confirmation_end'] == first['fit_origin']


def test_confirmation_cannot_select_a_different_training_loser():
    cfg = read(CONFIG)
    training = {key: {'utility': -.02, 'fills': 4} for key in POLICY_ORDER}
    training.update(CASH={'utility': 0, 'fills': 0}, PASSIVE={'utility': .01, 'fills': 1},
                    LOW_Q25={'utility': .05, 'fills': 5}, LOW_Q75={'utility': .04, 'fills': 5})
    confirmation = {key: {'utility': 0} for key in POLICY_ORDER}
    confirmation.update(PASSIVE={'utility': .02}, LOW_Q25={'utility': -.01}, LOW_Q75={'utility': .9})
    result = confirm_choices(training, confirmation, cfg)
    assert result['training_winner'] == 'LOW_Q25'
    assert result['primary_policy'] == 'PASSIVE'
    assert not result['feature_policy_accepted']


def test_threshold_sides_partition_equal_values_and_use_real_total_return():
    cfg = read(CONFIG)
    policies = policies_from_training(np.arange(100) / 100, cfg)
    assert policies['LOW_Q50']['threshold'] == .495
    data = pd.DataFrame({'five_day_total_return': [.494, .495, np.nan], 'vol20': [.2, .2, .2]})
    low = targets_for_policy(data, policies['LOW_Q50'], cfg)
    high = targets_for_policy(data, policies['HIGH_Q50'], cfg)
    np.testing.assert_allclose(low, [.5, 0, np.nan], equal_nan=True)
    np.testing.assert_allclose(high, [0, .5, np.nan], equal_nan=True)


def test_training_fills_follow_fridays_or_initialization_and_zero_target_exits(learned):
    data, _, cfg, record, ledger = learned
    for (phase, policy), group in ledger.groupby(['phase', 'policy']):
        assert len(group) == (363 if phase == 'TRAINING' else 121)
        assert group.date.max() <= record['fit_origin']
        traded = group[group.filled_quantity.ne(0)]
        allowed = traded.origin.dt.weekday.eq(4) | traded.date.eq(group.date.iloc[0])
        assert allowed.all()
    account = Account(cash=199600, shares=100)
    assert request(account, 4, 0, cfg)['requested_quantity'] == -100
    assert request(account, 4, .0001, cfg)['requested_quantity'] == -100
