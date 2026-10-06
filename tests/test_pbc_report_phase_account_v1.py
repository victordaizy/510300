"""检查季度重复计数、未来标签和实际绝对账户门的关键语义。"""
import numpy as np
import pandas as pd

from research.pbc_report_phase_account_v1 import balance_quarter_weights, economic_gate, training_pool


def test_quarter_total_weight_is_equal_despite_duplicate_origins():
    pool = pd.DataFrame({"uniqueness_weight": [.2, .4, .1, .6]})
    result = balance_quarter_weights(pool, ["2014Q1", "2014Q1", "2014Q1", "2014Q2"])
    totals = result.groupby("report_quarter").fit_weight.sum()
    assert np.isclose(totals.iloc[0], totals.iloc[1])


def test_future_mature_reference_and_unknown_text_cannot_enter_training():
    data = pd.DataFrame({"first_passage_feature_known": [True] * 6, "joint_features_known": [True, False, True, True, True, True],
                         "pbc_quarter": ["2014Q1"] * 6})
    outcomes = pd.DataFrame({"origin_index": [0, 1, 2], "status": ["MATURE_REFERENCE"] * 3,
                             "mature_idx": [3, 3, 5], "entry_idx": [1, 2, 3], "exit_idx": [3, 3, 5]})
    pool = training_pool(data, outcomes, 3)
    assert pool.origin_index.tolist() == [0]


def test_high_sharpe_without_absolute_return_or_defined_payoff_is_rejected():
    good = {"net_cagr": .11, "net_sharpe": 1.6, "max_drawdown": .09, "p_times_b": 1.1,
            "standard_expectancy_loss_units": .1, "mean_cycle_net_return": .01}
    controls = [{"net_cagr": .04, "net_sharpe": 1.2}]
    assert economic_gate(good, controls)
    assert not economic_gate({**good, "net_cagr": .0999}, controls)
    assert not economic_gate({**good, "p_times_b": np.nan}, controls)
    assert not economic_gate({**good, "p_times_b": 1.0}, controls)
