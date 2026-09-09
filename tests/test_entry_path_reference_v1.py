"""检验单次参考路径与既有账户的一致性，以及整组训练的成熟时钟。"""
import numpy as np
import pandas as pd
import pytest

from research.entry_path_reference_v1 import PathInputs, single_entry_path, signal_episodes, mature_episodes, episode_training_rows
from research.learned_cycle_exit_account_v1 import simulate_learned_exit
from research.learned_cycle_exit_v1 import FEATURES, state_values


def fixture(days=9):
    dates = pd.bdate_range("2020-01-01", periods=days)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    config = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    spec = {"loss": None, "trail": None, "take": None, "days": 3}
    return data, dividends, config, cost, spec


@pytest.mark.parametrize("payment_day", [2, 5])
def test_reference_matches_existing_account_costs_dividend_and_states(payment_day):
    data, _, config, cost, spec = fixture()
    data.loc[2:, ["open", "close"]] = 9.8
    data.loc[3:, "previous_close"] = 9.8
    data.loc[2, "dividend"] = .2
    dividends = pd.DataFrame([{"record_date": data.date.iloc[1], "ex_date": data.date.iloc[2],
                               "payment_date": data.date.iloc[payment_day], "cash_dividend_per_share": .2}])
    exit_flag = np.zeros(len(data), bool)
    metadata, ledger, states = single_entry_path(PathInputs(data, dividends), 0, config, cost, exit_flag, spec)
    entry = np.zeros(len(data), int)
    entry[0] = 1

    def recorder(t, cycle, current_value, peak_value):
        return {"learning_cycle_id": cycle["cycle_id"], "learned_exit_requested": False,
                **dict(zip(FEATURES, state_values(data, t, cycle, current_value, peak_value)))}

    full, decisions, cycles = simulate_learned_exit(data, dividends, config, cost, str(data.date.iloc[1].date()),
        {"entry": entry, "exit": {1: exit_flag}}, {"cooldown": 2, "modes": {1: spec}}, recorder)
    columns = ["date", "cash", "shares", "equity", "net_return", "commission", "slippage_cost", "filled_quantity", "status", "dividend_receivable"]
    pd.testing.assert_frame_equal(ledger[columns], full.iloc[:len(ledger)][columns].reset_index(drop=True))
    old_states = decisions[decisions.learning_cycle_id.notna()]
    np.testing.assert_allclose(states[FEATURES], old_states[FEATURES])
    assert metadata["net_profit_cny"] == pytest.approx(cycles.iloc[0].net_profit_cny)
    assert metadata["natural_exit"] and metadata["resolution_index"] == 4
    assert metadata["max_accounting_error"] < 1e-6
    assert metadata["remaining_dividend_receivable"] == pytest.approx(0 if payment_day == 2 else metadata["entry_quantity"] * .2)


def test_unfilled_initial_buy_does_not_retry_or_create_label_state():
    data, dividends, config, cost, spec = fixture()
    data.loc[1, "open"] = 11.
    meta, ledger, states = single_entry_path(PathInputs(data, dividends), 0, config, cost, np.zeros(len(data), bool), spec)
    assert meta["resolution_index"] == 1 and not meta["natural_exit"]
    assert len(ledger) == 1 and ledger.iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert states.empty and meta["final_observed_nav"] == config["initial_capital"]


def test_pending_exit_survives_limit_block_until_actual_sale():
    data, dividends, config, cost, spec = fixture()
    spec["days"] = 2
    data.loc[3, "open"] = 9.
    meta, ledger, states = single_entry_path(PathInputs(data, dividends), 0, config, cost, np.zeros(len(data), bool), spec)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.iloc[3].filled_quantity < 0 and meta["resolution_index"] == 4
    assert states.iloc[1:].natural_continue.eq(False).all()
    assert "最长" in meta["exit_reasons"]


@pytest.mark.parametrize("origin", [0, 7])
def test_terminal_forced_exit_or_entry_cannot_make_path_mature(origin):
    data, dividends, config, cost, spec = fixture()
    spec["days"] = None
    meta, ledger, states = single_entry_path(PathInputs(data, dividends), origin, config, cost, np.zeros(len(data), bool), spec)
    assert not meta["natural_exit"] and meta["resolution_index"] is None
    assert meta["status"].startswith("CENSORED")
    assert ledger.iloc[-1].mark_clock == "OPEN_TERMINAL"


def test_incomplete_data_does_not_certify_episode_closed():
    episodes = signal_episodes([0, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1])
    assert [(r["start_index"], r["end_index"], r["closed_index"]) for r in episodes] == [(1, 2, 3), (4, 5, None), (7, 7, None)]
    with pytest.raises(ValueError, match="长度"):
        signal_episodes([1], [1, 1])


def test_whole_episode_waits_for_latest_path_and_closed_signal():
    episodes = signal_episodes([0, 1, 1, 0, 1, 1, 0], [1] * 7)
    paths = pd.DataFrame([{"episode_id": 1, "entry_index": 2, "natural_exit": True, "resolution_index": 4},
                          {"episode_id": 1, "entry_index": 3, "natural_exit": True, "resolution_index": 8},
                          {"episode_id": 2, "entry_index": 5, "natural_exit": True, "resolution_index": 6},
                          {"episode_id": 2, "entry_index": 6, "natural_exit": False, "resolution_index": None}])
    groups = mature_episodes(episodes, paths, 2).set_index("episode_id")
    assert groups.loc[1, "group_mature_index"] == 8 and groups.loc[1, "reference_left_truncated"]
    assert pd.isna(groups.loc[2, "group_mature_index"])


def test_future_path_blocks_early_samples_and_group_and_path_weights_are_balanced():
    samples = pd.DataFrame([
        {"episode_id": 1, "path_id": 1, "origin_index": 2, "exit_index": 4, "group_mature_index": 4},
        {"episode_id": 1, "path_id": 2, "origin_index": 3, "exit_index": 4, "group_mature_index": 4},
        {"episode_id": 1, "path_id": 2, "origin_index": 2, "exit_index": 4, "group_mature_index": 4},
        {"episode_id": 2, "path_id": 3, "origin_index": 5, "exit_index": 6, "group_mature_index": 12},
        {"episode_id": 2, "path_id": 4, "origin_index": 7, "exit_index": 12, "group_mature_index": 12},
        {"episode_id": 3, "path_id": 5, "origin_index": 8, "exit_index": 10, "group_mature_index": None},
    ])
    early, ids = episode_training_rows(samples, 10, 20)
    assert ids == [1] and set(early.path_id) == {1, 2}
    np.testing.assert_allclose(early.groupby("path_id").sample_weight.sum(), [.5, .5])
    late, ids = episode_training_rows(samples, 12, 20)
    assert ids == [1, 2]
    np.testing.assert_allclose(late.groupby("episode_id").sample_weight.sum(), 1.)
    _, ids = episode_training_rows(samples, 12, 1)
    assert ids == [2]
