import numpy as np

from research.sequential_regime_strategy_selection_inputs_v1 import sequential_choice


def test_insufficient_same_state_samples_stays_cash():
    state = np.array(["上升稳定"] * 4, dtype=object)
    returns = {"A": np.full(4, 0.01), "B": np.full(4, 0.02)}
    choice, score, annual, samples = sequential_choice(state, returns, 242, 10, 5, 1.2, 0.1)
    assert (choice == "现金").all()
    assert np.isnan(score).all() and np.isnan(annual).all() and (samples == 0).all()


def test_uses_only_current_and_prior_same_state_returns():
    state = np.array(["上升稳定", "压力回撤", "上升稳定", "上升稳定"], dtype=object)
    returns = {
        "A": np.array([0.01, -0.02, 0.02, 0.01]),
        "B": np.array([0.001, 0.01, 0.001, 0.001]),
    }
    choice, _, _, samples = sequential_choice(state, returns, 242, 10, 3, 0.1, 0.001)
    assert choice[:2].tolist() == ["现金", "现金"]
    assert samples[3] == 3
    assert choice[3] == "A"


def test_no_candidate_passing_threshold_stays_cash():
    state = np.array(["上升稳定"] * 6, dtype=object)
    returns = {"A": np.array([0.001, -0.001, 0.001, -0.001, 0.001, -0.001]), "B": np.zeros(6)}
    choice, _, _, _ = sequential_choice(state, returns, 242, 10, 3, 1.2, 0.1)
    assert (choice == "现金").all()
