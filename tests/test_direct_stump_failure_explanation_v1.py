"""验证跨月规则生效时点与账面费用分解。"""
import numpy as np
import pandas as pd
from research.direct_stump_failure_explanation_v1 import active_fit_indices, decomposition


def test_non_friday_fit_waits_for_friday_decision_and_next_day_execution():
    # 周一更新编号3；此前周五编号1判断仍生效，直到新周五编号7收盘判断。
    decisions = pd.DataFrame({'origin_index': np.arange(1, 10),
                              'weekly_event': [True, False, False, False, False, False, True, False, False],
                              'initialization_event': [True] + [False] * 8})
    actual = active_fit_indices(decisions, [0, 3, 9])
    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 0, 0, 1, 1])
    changed_future = active_fit_indices(decisions, [0, 3, 90])
    np.testing.assert_array_equal(actual, changed_future)


def test_cost_decomposition_closes_without_becoming_zero_cost_backtest():
    ledger = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3), 'equity': [1000., 1010., 1005.],
                           'net_return': [0., .01, 1005 / 1010 - 1], 'shares_before': [0., 50., 50.],
                           'commission': [1., 0., 1.], 'slippage_cost': [1., 0., .5]})
    market = pd.DataFrame({'previous_close': [10., 10., 10.2], 'total_simple': [0., .02, -.01]})
    daily, parts = decomposition(ledger, market, 1000., 242)
    expected_cost = -(2 / 1000 + 1.5 / 1010) / 3 * 242
    np.testing.assert_allclose(parts['friction_part'], expected_cost, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sum(parts[k] for k in ['average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']),
                               ledger.net_return.mean() * 242, atol=1e-12, rtol=0)
    np.testing.assert_allclose(daily.prior_weight, [0, .5, 510 / 1010], atol=1e-12, rtol=0)
