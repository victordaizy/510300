"""覆盖实际成交基准、损失边界、未知、受阻保护与恢复进入。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account
from research.account_cycle_loss_exit_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, CycleLossPolicy, loss_frames, simulate_loss_account
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), decision_clock='15:05:00',
        costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def run(raw, data=None, div=None, model=PRIMARY):
    base, dividends, _, cost = fixture()
    data = base if data is None else data
    return simulate_loss_account(data, dividends if div is None else div, settings(), cost, str(data.date.iloc[1].date()), model,
        targets=np.array(raw, float), event_mask=np.ones(len(data), bool))


def test_actual_entry_basis_failed_buy_and_partial_trades_do_not_reset():
    cfg, policy, account = settings(), CycleLossPolicy(PRIMARY), Account(100000.)
    first = policy.request(account, 10., .5, cfg)
    assert first['requested_quantity'] == 5000 and np.isnan(first['cycle_basis'])
    # 首次申请未成交，仍为空仓，不开始周期；下一次申请采用新的事前净值。
    account.cash = 100100.
    retry = policy.request(account, 10., .5, cfg)
    assert retry['cycle_number'] == 0 and np.isnan(retry['cycle_basis'])
    account.cash, account.shares = 50000., 5000
    held = policy.request(account, 10., .8, cfg)
    assert held['cycle_basis'] == 100100. and held['cycle_return'] < 0 and held['cycle_number'] == 1
    account.cash, account.shares = 20000., 8000
    changed = policy.request(account, 10., .4, cfg)
    assert changed['cycle_basis'] == 100100. and changed['requested_quantity'] < 0


def test_exact_loss_threshold_includes_cost_and_triggers_on_first_close():
    cfg = settings()
    for model, setting in SETTINGS.items():
        policy, account = CycleLossPolicy(model), Account(100000.)
        policy.request(account, 10., 1., cfg)
        account.shares = 5000
        account.cash = 100000*(1-setting['loss_fraction'])-50000
        answer = policy.request(account, 10., 1., cfg)
        assert answer['cycle_basis'] == 100000. and answer['triggered_protection'] == 1
        assert answer['requested_quantity'] == -5000 and answer['reference_weight'] == 0
    data = fixture()[0]
    data.loc[1, 'close'] = 9.804
    ledger, decisions = run([1.]*10, data=data)
    assert ledger.iloc[0].equity < 98000
    assert decisions.iloc[1].triggered_protection == 1
    assert decisions.iloc[1].cycle_basis == 100000.


def test_unknown_source_does_not_cancel_protection_and_requires_known_zero():
    cfg, policy, account = settings(), CycleLossPolicy(PRIMARY), Account(100000.)
    unknown = policy.request(account, 10., np.nan, cfg)
    assert unknown['requested_quantity'] == 0 and np.isnan(unknown['reference_weight'])
    policy.request(account, 10., .5, cfg)
    account.cash, account.shares = 50000., 5000
    held = policy.request(account, 10., np.nan, cfg)
    assert np.isnan(held['reference_weight']) and held['requested_quantity'] == 0
    stopped = policy.request(account, 9.5, np.nan, cfg)
    assert stopped['triggered_protection'] == 1 and stopped['requested_quantity'] == -5000
    # 假设卖出受阻，价格恢复与来源归零都不能撤销尚未完成的保护卖出。
    zero = policy.request(account, 10., 0., cfg)
    assert zero['requested_quantity'] == -5000 and zero['waiting_source_zero'] == 0
    recovered = policy.request(account, 10., .8, cfg)
    assert recovered['requested_quantity'] == -5000
    account.cash, account.shares = 99000., 0
    reentry = policy.request(account, 10., .5, cfg)
    assert reentry['requested_quantity'] > 0 and reentry['protection_active'] == 0


def test_after_protective_sale_positive_and_unknown_wait_until_zero():
    cfg, policy, account = settings(), CycleLossPolicy(PRIMARY), Account(100000.)
    policy.request(account, 10., .5, cfg)
    account.cash, account.shares = 50000., 5000
    policy.request(account, 9.5, .8, cfg)
    account.cash, account.shares = 97500., 0
    for raw in [.8, np.nan, .3]:
        answer = policy.request(account, 10., raw, cfg)
        assert answer['requested_quantity'] == 0 and answer['reference_weight'] == 0 and answer['waiting_source_zero'] == 1
    assert policy.request(account, 10., 0., cfg)['protection_active'] == 0
    assert policy.request(account, 10., .3, cfg)['requested_quantity'] > 0


def test_actual_limit_block_unknown_retry_dividend_and_terminal():
    data = fixture()[0]
    data.loc[1, 'close'] = 9.7
    data.loc[2, 'open'] = 9.0
    data.loc[3:, ['open', 'close', 'previous_close']] = 9.7
    ledger, decisions = run([1., np.nan, np.nan, .8, 0., .8, .8, .8, .8, .8], data=data)
    assert decisions.iloc[1].triggered_protection == 1
    assert ledger.iloc[1].status == 'UNFILLED_DIRECTIONAL_LIMIT' and ledger.iloc[2].shares == 0
    assert decisions.iloc[2].requested_quantity < 0 and decisions.iloc[3].waiting_source_zero == 1
    assert ledger.iloc[-1].shares == 0 and ledger.accounting_error.abs().max() < 1e-6
    data = fixture()[0]
    data.loc[3:, ['open', 'close']] = 9.9
    data.loc[3, 'dividend'] = .1
    data.loc[4:, 'previous_close'] = 9.9
    div = pd.DataFrame([{'record_date': data.date.iloc[2], 'ex_date': data.date.iloc[3],
                         'payment_date': data.date.iloc[5], 'cash_dividend_per_share': .1}])
    ledger, decisions = run([.5]*10, data, div)
    assert ledger.dividend_recognized.sum() > 0 and ledger.dividend_paid.sum() > 0
    assert decisions.triggered_protection.sum() == 0 and ledger.shares.iloc[-1] == 0


def test_fixed_sources_cost_alignment_and_future_isolation():
    data = fixture()[0]
    cfg, parents = settings(), {}
    for cost in cfg['costs']:
        parents[cost] = {m: pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
            'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
            'reference_weight': .5, 'source_cost': cost, 'source_model': m}) for m in MODELS}
    frames, summaries = loss_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    assert len(CANDIDATES) == 6 and len(summaries) == 12
    bad = copy.deepcopy(parents)
    bad['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        loss_frames(data, bad, cfg, str(data.date.iloc[1].date()))
    raw = [.5, .5, np.nan, .8, .7, .6, .4, .3, 0., .3]
    old, decisions = run(raw)
    data.loc[6:, ['open', 'close', 'previous_close']] = 9.7
    raw[5:] = [0., 0., .8, .8, .8]
    newer, changed = run(raw, data)
    pd.testing.assert_frame_equal(old.iloc[:5], newer.iloc[:5])
    pd.testing.assert_frame_equal(decisions.iloc[:5], changed.iloc[:5])
