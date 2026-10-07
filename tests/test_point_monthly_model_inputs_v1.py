"""训练参考必须自然成熟，真实末端及后续价格不能提前制造训练标签。"""
import numpy as np
import pandas as pd

from research import point_monthly_model_inputs_v1 as study
from research.training_reference_observation_v1 import observe_training_reference


def inputs():
    dates = pd.bdate_range("2024-03-04", periods=8)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .01, "vol20": .1})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "costs": {"BASE": cost}}
    return data, dividends, cfg


def run_reference(data, dividends, cfg, next_date, exit_origin=None):
    exits = np.zeros(len(data), bool)
    if exit_origin is not None:
        exits[exit_origin] = True
    rule = {"entry": np.ones(len(data), int), "exit": {1: exits}}
    spec = {"cooldown": 2, "modes": {1: {"loss": .06, "trail": .08, "take": None, "days": 60}}}
    def recorder(t, cycle, current_value, peak_value):
        return {"learning_cycle_id": cycle["cycle_id"], "learned_exit_requested": False,
                **dict(zip(study.FEATURES, study.state_values(data, t, cycle, current_value, peak_value)))}
    return observe_training_reference(data, dividends, cfg, cfg["costs"]["BASE"], data.date.iloc[1], rule, spec, recorder,
                                      next_execution_date=next_date)


def test_unfinished_cycle_has_no_labels():
    data, dividends, cfg = inputs()
    ledger, decisions, cycles = run_reference(data, dividends, cfg, data.date.iloc[-1]+pd.offsets.BDay(1))
    assert ledger.shares.iloc[-1] > 0
    assert ledger.mark_clock.eq("CLOSE").all()
    assert cycles.exit_date.isna().all()
    assert study.mature_samples(data, dividends, cfg, decisions, cycles).empty


def test_last_real_day_natural_exit_creates_only_mature_labels():
    data, dividends, cfg = inputs()
    ledger, decisions, cycles = run_reference(data, dividends, cfg, data.date.iloc[-1]+pd.offsets.BDay(1), exit_origin=6)
    assert ledger.shares.iloc[-1] == 0
    assert cycles.exit_date.iloc[-1] == data.date.iloc[-1]
    samples = study.mature_samples(data, dividends, cfg, decisions, cycles)
    assert len(samples) == 5
    assert samples.exit_index.eq(7).all()
    before, ids = study.training_rows(samples, 6, {"recent_cycles": 20})
    assert before.empty and not ids
    after, ids = study.training_rows(samples, 7, {"recent_cycles": 20})
    assert len(after) == 5 and ids == [1]


def test_observed_training_states_do_not_change_with_future_bars():
    data, dividends, cfg = inputs()
    short = run_reference(data.iloc[:6].copy(), dividends, cfg, data.date.iloc[6])
    data.loc[6:, ["open", "close"]] = 10.2
    data.loc[7, "previous_close"] = 10.2
    full = run_reference(data, dividends, cfg, data.date.iloc[-1]+pd.offsets.BDay(1))
    for index, clock in ((0, "date"), (1, "origin")):
        prefix = full[index].loc[full[index][clock].le(data.date.iloc[5])].reset_index(drop=True)
        pd.testing.assert_frame_equal(short[index], prefix)


def test_training_reference_keeps_original_reentry_after_cooldown():
    data, dividends, cfg = inputs()
    ledger, _, cycles = run_reference(data, dividends, cfg, data.date.iloc[-1]+pd.offsets.BDay(1), exit_origin=3)
    assert len(cycles) == 2
    assert cycles.entry_index.to_list() == [1, 7]
    assert ledger.shares.iloc[-1] > 0


def test_real_last_month_first_close_is_an_eligible_fit_clock():
    dates = pd.bdate_range("2024-01-30", "2024-03-01")
    data = pd.DataFrame({"date": dates})
    schedule = study.monthly_schedule(data, dates[1])
    assert schedule[0] == 0
    assert data.date.iloc[schedule[-1]] == pd.Timestamp("2024-03-01")
    assert len(schedule) == 3
