"""检验上一原点确认、一次锚点、未知跨越与未来后缀隔离。"""
import numpy as np
import pandas as pd
from research.downtrend_break_inputs_v1 import break_states, account_signals, PRIMARY


def market():
    shape = [11., 12., 13., 12., 11., 10., 11., 12., 12.5, 12., 11., 9., 10., 11., 11.5,
             12.6, 12.6, 12.4, 12.6, 12.6, 12.4, 12.6]
    return pd.DataFrame({"date": pd.date_range("2020-01-02", periods=len(shape), freq="B"),
                         "close": shape, "dividend": 0., "atr20": .1})


def test_break_uses_the_last_high_already_confirmed_at_the_previous_close():
    data = market()
    states, _ = break_states(data)
    assert states.prior_complete_downtrend.iloc[14]
    assert not states.break_event.iloc[:15].any()
    assert states.break_event.iloc[15]
    row = states.iloc[15]
    assert [row.prior_H1_index, row.prior_L1_index, row.prior_H2_index, row.prior_L2_index] == [2, 5, 8, 11]
    assert row.prior_H2_confirmation_index == 10
    assert row.prior_L2_confirmation_index == 13
    assert row.initial_broken_high_stop == 12.5
    assert row.previous_cash_adjusted_close <= row.initial_broken_high_stop < row.known_cash_adjusted_close
    assert account_signals(data, states, PRIMARY).entry_event.iloc[15]


def test_same_confirmed_high_is_only_allowed_its_first_eligible_cross():
    states, _ = break_states(market())
    assert states.break_candidate.iloc[18]
    assert states.anchor_consumed_before_today.iloc[18]
    assert not states.break_event.iloc[18]
    assert states.break_status.iloc[18] == "ANCHOR_ALREADY_CONSUMED"
    assert states.break_event.sum() == 1
    assert states.consumed_anchor_count.iloc[-1] == 1


def test_unknown_prior_quote_does_not_fabricate_a_cross_on_recovery():
    data = market()
    data.loc[14, "close"] = np.nan
    states, _ = break_states(data)
    assert states.break_status.iloc[15] == "NO_VIEW_CURRENT_OR_PRIOR_QUOTES"
    assert not states.break_event.iloc[15:17].any()
    assert states.break_event.iloc[18]
    unknown_cash = market()
    unknown_cash.loc[14, "dividend"] = np.nan
    unknown, _ = break_states(unknown_cash)
    assert not unknown.break_event.iloc[14:].any()


def test_future_quotes_dividends_and_outcomes_cannot_change_past_anchor_consumption():
    data = market()
    full, _ = break_states(data)
    other = data.copy()
    other["future_return"] = np.arange(len(data))*100.
    other["retrospective_bottom"] = np.arange(len(data))[::-1]
    pd.testing.assert_frame_equal(break_states(other)[0], full, check_exact=True)
    for n in range(1, len(data)+1):
        prefix, _ = break_states(data.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix, full.iloc[:n].reset_index(drop=True), check_exact=True)
        changed = data.copy()
        changed.loc[n:, "close"] = 100.+np.arange(len(data)-n)
        changed.loc[n:, "dividend"] = 10.
        pd.testing.assert_frame_equal(break_states(changed)[0].iloc[:n].reset_index(drop=True), prefix, check_exact=True)
