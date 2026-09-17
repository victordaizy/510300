"""检查含分红收益边界、只拦买入及过去决定不依赖未来。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account
from research.close_return_buy_strength_gate_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, gate_frames, gate_request, market_factors, simulate_gate_account
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), decision_clock='15:05:00',
        costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def rising_frame():
    frame = fixture()[0]
    frame['close'] = 10.*1.02**(np.arange(len(frame))+1)
    frame['previous_close'] = np.r_[10., frame.close.iloc[:-1]]
    frame['open'] = frame.previous_close
    return frame


def run(values, frame=None):
    data, div, _, cost = fixture()
    data = rising_frame() if frame is None else frame
    return simulate_gate_account(data, div, settings(), cost, str(data.date.iloc[1].date()), PRIMARY,
        targets=np.asarray(values, float), event_mask=np.ones(len(data), bool))


def test_dividend_neutral_zero_one_percent_boundaries_and_missing():
    data = pd.DataFrame({'close': [.1, 101., 101.001, 100., 99., 100., 100.],
        'previous_close': [.3, 100., 100., 100., 100., np.nan, 100.],
        'dividend': [.2, 0., 0., 0., 0., 0., np.nan]})
    factors = market_factors(data)
    np.testing.assert_array_equal(factors.buy_allowed_r00, [True, True, True, True, False, False, False])
    np.testing.assert_array_equal(factors.buy_allowed_r01, [False, True, True, False, False, False, False])
    assert abs(factors.daily_total_simple.iloc[0]) < 1e-12 and np.isnan(factors.daily_total_simple.iloc[-1])


def test_gate_blocks_entry_and_addition_but_keeps_positive_target_and_all_sells():
    cfg = settings()
    for model in CANDIDATES:
        account = Account(100000.)
        blocked = gate_request(account, 10., .5, cfg, model, False, .02)
        assert blocked['requested_quantity'] == 0 and blocked['reference_weight'] == .5 and blocked['normal_requested_quantity'] == 5000
        assert gate_request(account, 10., .5, cfg, model, True, 0.)['requested_quantity'] == 5000
        account.cash, account.shares = 70000., 3000
        assert gate_request(account, 10., .8, cfg, model, False, np.nan)['requested_quantity'] == 0
        assert gate_request(account, 10., .05, cfg, model, False, np.nan)['requested_quantity'] < 0
        assert gate_request(account, 10., 0., cfg, model, False, .05)['requested_quantity'] == -3000


def test_new_close_retries_unknown_keeps_shares_and_terminal_exits():
    data = fixture()[0]
    data['close'] = 9.8*1.02**np.arange(len(data))
    data['previous_close'] = np.r_[10., data.close.iloc[:-1]]
    data['open'] = data.previous_close
    ledger, decisions = run([.5, .5, np.nan, .8, .2, 0., .5, .5, .5, .5], data)
    assert decisions.iloc[0].buy_suppressed == 1 and ledger.shares.iloc[0] == 0
    assert ledger.shares.iloc[1] > 0 and ledger.shares.iloc[2] == ledger.shares.iloc[1]
    assert decisions.iloc[2].signal_state == 'NO_VIEW_KEEP_EXISTING_SHARES'
    assert ledger.shares.iloc[5] == 0 and ledger.shares.iloc[6] > 0 and ledger.shares.iloc[-1] == 0


def test_four_candidates_share_market_factor_but_preserve_source_cost_and_dates():
    data = fixture()[0]
    cfg, parents = settings(), {}
    for cost in cfg['costs']:
        parents[cost] = {m: pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
            'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
            'reference_weight': .5, 'source_cost': cost, 'source_model': m}) for m in MODELS}
    frames, summaries = gate_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    assert len(CANDIDATES) == 4 and len(summaries) == 8
    pd.testing.assert_series_equal(frames['BASE'].daily_total_simple, frames['STRESS'].daily_total_simple)
    bad = copy.deepcopy(parents)
    bad['BASE'][MODELS[0]].loc[0, 'execution_date'] = data.date.iloc[2]
    with pytest.raises(ValueError, match='下一开盘'):
        gate_frames(data, bad, cfg, str(data.date.iloc[1].date()))
    bad = copy.deepcopy(parents)
    bad['STRESS'][MODELS[1]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        gate_frames(data, bad, cfg, str(data.date.iloc[1].date()))


def test_future_prices_and_targets_do_not_change_past_decisions_or_accounts():
    data = rising_frame()
    targets = np.array([.5, .6, np.nan, .8, .6, .5, .3, .2, 0., 0.])
    old, decisions = run(targets)
    changed = data.copy()
    changed.loc[6:, ['open', 'close', 'previous_close']] = 10.4
    targets[5:] = [0., .8, .8, .8, .8]
    newer, revised = run(targets, changed)
    pd.testing.assert_frame_equal(old.iloc[:5], newer.iloc[:5])
    pd.testing.assert_frame_equal(decisions.iloc[:5], revised.iloc[:5])
