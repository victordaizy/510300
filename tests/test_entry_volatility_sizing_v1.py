"""验证初始预算、持仓期间不调整、缺失和未成交。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.entry_sized_rearmed_account_v1 import simulate_entry_sized_exit
from research.entry_volatility_sizing_v1 import entry_fraction
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .2})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, dividends, cfg, cost, str(dates[1].date()), rule, spec]


def test_full_fraction_recovers_original_complete_account():
    args = setup()
    args[3].update(commission=.0002, minimum=5., slippage=.0005)
    original, original_decisions, original_cycles = simulate_rearmed_exit(*args)
    revised, decisions, cycles = simulate_entry_sized_exit(*args, fraction_provider=lambda t: 1.)
    assert_frame_equal(original, revised)
    assert_frame_equal(original_cycles, cycles[original_cycles.columns])


def test_budget_is_set_only_at_entry_and_later_volatility_does_not_add_shares():
    args = setup()
    args[0].loc[1:, "vol20"] = .05
    calls = []
    def provider(t):
        calls.append(t)
        return entry_fraction(args[0], t, "ENTRY_VOL10")
    ledger, decisions, cycles = simulate_entry_sized_exit(*args, fraction_provider=provider)
    assert calls == [0]
    assert ledger[ledger.filled_quantity > 0].filled_quantity.to_list() == [5000]
    assert (ledger.iloc[:-1].shares == 5000).all()
    assert cycles.entry_fraction.iloc[0] == .5
    assert cycles.entry_budget_cny.iloc[0] == 50000.


def test_unfilled_order_preserves_permission_to_recalculate_next_close():
    args = setup()
    args[0].loc[1, "open"] = 11.
    ledger, decisions, cycles = simulate_entry_sized_exit(*args, fraction_provider=lambda t: .5)
    assert ledger.iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger[ledger.filled_quantity > 0].date.iloc[0] == args[0].date.iloc[2]
    assert cycles.entry_fraction.iloc[0] == .5


def test_missing_volatility_is_not_a_zero_forecast_and_can_recover():
    args = setup()
    args[0].loc[0, "vol20"] = np.nan
    provider = lambda t: entry_fraction(args[0], t, "ENTRY_VOL10")
    ledger, decisions, cycles = simulate_entry_sized_exit(*args, fraction_provider=provider)
    assert decisions.iloc[0].entry_sizing_status == "NO_VIEW_ENTRY_SIZING_INPUT"
    assert pd.isna(decisions.iloc[0].entry_fraction)
    assert decisions.iloc[0].entry_rearmed
    assert ledger[ledger.filled_quantity > 0].date.iloc[0] == args[0].date.iloc[2]
    assert entry_fraction(args[0], 0, "ENTRY_HALF") == .5


def test_low_volatility_is_capped_at_full_budget_without_leverage():
    data = pd.DataFrame({"vol20": [.2, .1, .05, 0., -1., np.inf]})
    assert [entry_fraction(data, t, "ENTRY_VOL10") for t in range(6)] == [.5, 1., 1., None, None, None]
