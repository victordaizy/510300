"""覆盖部分退出的实际数量、保护优先、登记权益、风险分摊与重新进入。"""
import numpy as np
import pandas as pd
from research.partial_learned_exit_account_v1 import simulate_partial_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


class NegativeAfterTwoCloses:
    def __init__(self):
        self.calls, self.counts = [], {}

    def __call__(self, t, cycle, current_value, peak_value):
        self.calls.append(t)
        key = cycle["cycle_id"]
        self.counts[key] = self.counts.get(key, 0) + 1
        return {"learned_exit_requested": self.counts[key] >= 2,
                "learning_status": "PREDICTION_AVAILABLE", "continuation_prediction": -.02}


def setup():
    dates = pd.bdate_range("2020-01-01", periods=12)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(len(dates), dtype=int), "exit": {1: np.zeros(len(dates), dtype=bool)}}
    rule["exit"][1][5] = True
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, div, cfg, cost, str(dates[1].date()), rule, spec, NegativeAfterTwoCloses()]


def test_only_one_half_reduction_then_original_exit_without_reentry():
    args = setup()
    ledger, decisions, cycles = simulate_partial_exit(*args)
    trades = ledger[ledger.filled_quantity.ne(0)]
    assert trades.filled_quantity.to_list() == [10000, -5000, -5000]
    assert trades.date.to_list() == args[0].date.iloc[[1, 3, 6]].to_list()
    assert args[-1].calls == [1, 2]
    assert cycles.partial_exit_fills.to_list() == [1]
    assert cycles.partial_sold_quantity.to_list() == [5000]
    assert not decisions.iloc[-1].entry_rearmed
    assert ledger.accounting_error.abs().max() < 1e-7


def test_original_protection_has_priority_over_learning_reduction():
    args = setup()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_partial_exit(*args)
    assert ledger.loc[ledger.filled_quantity.ne(0), "filled_quantity"].to_list() == [10000, -10000]
    assert cycles.partial_exit_fills.iloc[0] == 0
    assert args[-1].calls == [1]


def test_unfilled_reduction_keeps_locked_quantity_and_never_halves_twice():
    args = setup()
    args[0].loc[3, "open"] = 9.
    ledger, decisions, cycles = simulate_partial_exit(*args)
    rejected = ledger[ledger.date.eq(args[0].date.iloc[3])].iloc[0]
    assert rejected.status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert rejected.requested_quantity == -5000
    trades = ledger[ledger.filled_quantity.ne(0)]
    assert trades.filled_quantity.to_list() == [10000, -5000, -5000]
    assert trades.date.to_list() == args[0].date.iloc[[1, 4, 6]].to_list()
    assert args[-1].calls == [1, 2]


def test_new_protection_overrides_blocked_half_order_with_full_sale():
    args = setup()
    args[0].loc[3, "open"] = 9.
    args[5]["exit"][1][3] = True
    ledger, decisions, cycles = simulate_partial_exit(*args)
    sale = ledger[ledger.filled_quantity.lt(0)].iloc[0]
    assert sale.date == args[0].date.iloc[4]
    assert sale.filled_quantity == -10000
    assert sale.execution_action_kind == "FULL_EXIT"
    assert cycles.partial_exit_fills.iloc[0] == 0


def test_recorded_rights_stay_full_but_remaining_risk_is_proportional():
    args = setup()
    d = args[0]
    args[1] = pd.DataFrame([{"record_date": d.date.iloc[2], "ex_date": d.date.iloc[4],
                             "payment_date": d.date.iloc[7], "cash_dividend_per_share": .1}])
    d.loc[4:, ["open", "close"]] = 9.9
    d.loc[4, "dividend"] = .1
    d["previous_close"] = d.close.shift(1).fillna(10.)
    ledger, decisions, cycles = simulate_partial_exit(*args)
    at_ex = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
    risk = decisions[decisions.origin_index.eq(4)].iloc[0]
    assert at_ex.dividend_recognized == 1000.
    assert at_ex.dividend_receivable == 1000.
    assert risk.remaining_risk_cost == 50000.
    assert risk.remaining_risk_dividends == 500.
    assert abs(risk.remaining_risk_value / risk.remaining_risk_cost - 1) < 1e-12
    assert cycles.dividend_cny.iloc[0] == 1000.
    assert cycles.net_profit_cny.sum() == ledger.equity.iloc[-1] - 100000.
    assert ledger.accounting_error.abs().max() < 1e-7


def test_old_cycle_dividend_is_not_assigned_to_new_cycle():
    args = setup()
    d = args[0]
    args[-1] = None
    args[5]["exit"][1][:] = False
    args[5]["exit"][1][2] = True
    args[5]["entry"][3] = 0
    args[1] = pd.DataFrame([{"record_date": d.date.iloc[1], "ex_date": d.date.iloc[7],
                             "payment_date": d.date.iloc[8], "cash_dividend_per_share": .1}])
    d.loc[7:, ["open", "close"]] = 9.9
    d.loc[7, "dividend"] = .1
    d["previous_close"] = d.close.shift(1).fillna(10.)
    ledger, decisions, cycles = simulate_partial_exit(*args)
    assert cycles.entry_date.to_list() == d.date.iloc[[1, 6]].to_list()
    assert cycles.dividend_cny.to_list() == [1000., 0.]
    risk = decisions[decisions.origin_index.eq(7)].iloc[0]
    assert risk.remaining_risk_dividends == 0.
    assert abs(cycles.net_profit_cny.sum() - (ledger.equity.iloc[-1] - 100000.)) < 1e-7


def test_one_lot_exits_fully_instead_of_emitting_zero_reduction():
    args = setup()
    args[2]["initial_capital"] = 1000.
    ledger, decisions, cycles = simulate_partial_exit(*args)
    assert ledger.loc[ledger.filled_quantity.ne(0), "filled_quantity"].to_list() == [100, -100]
    assert "不足一手" in cycles.exit_reasons.iloc[0]


def test_validation_full_exit_mode_matches_original_account_economics():
    args = setup()
    args[0].loc[4:, ["open", "close"]] = 10.2
    args[0]["previous_close"] = args[0].close.shift(1).fillna(10.)
    args[3].update(commission=.0002, minimum=5., slippage=.0005)
    args[5]["entry"][4] = 0
    old = simulate_rearmed_exit(*args)[0]
    args[-1] = NegativeAfterTwoCloses()
    new = simulate_partial_exit(*args, learned_sell_fraction=1.)[0]
    fields = ["date", "cash", "shares", "equity", "net_return", "dividend_receivable", "commission", "slippage_cost", "filled_quantity", "status"]
    pd.testing.assert_frame_equal(old[fields], new[fields])


def test_change_in_later_prices_cannot_change_earlier_trades():
    original = setup()
    earlier = simulate_partial_exit(*original)[0]
    altered = setup()
    altered[0].loc[8:, ["open", "close"]] = 20.
    later = simulate_partial_exit(*altered)[0]
    pd.testing.assert_frame_equal(earlier[earlier.date.lt(original[0].date.iloc[8])], later[later.date.lt(original[0].date.iloc[8])])
