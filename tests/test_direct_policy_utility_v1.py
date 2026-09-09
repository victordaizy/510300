"""直接仓位学习的梯度、时间隔离和账户边界检查。"""
import unittest

import numpy as np

from research.adaptive_allocation_v1 import factors, spec, simulate, eligible_training
from research.direct_policy_utility_v1 import objective, smooth_policy, fit_policy, predict_sequence, simulate_policy
from tests.test_adaptive_allocation_v1 import synthetic


def test_config():
    config = spec()
    config.update({"minimum_train_samples": 100, "smoothing_alpha": 0.2, "feature_clip": 5.0,
                   "coefficient_bound": 2.0, "intercept_bound": 4.0, "l2_penalty": 0.05,
                   "turnover_smoothing_epsilon": 0.001, "annual_volatility_floor": 0.02,
                   "training_turnover_cost": 0.0014, "mean_variance_gamma": 10.0,
                   "training_carry_initial_weight": 0.5,
                   "optimizer": {"method": "L-BFGS-B", "maxiter": 250, "maxfun": 2000,
                                 "ftol": 1e-9, "gtol": 1e-6, "maxls": 40}})
    return config


class DirectPolicyTests(unittest.TestCase):
    def test_analytic_gradient_matches_independent_finite_difference(self):
        rng = np.random.default_rng(71)
        design = np.column_stack((np.ones(140), rng.normal(size=(140, 4))))
        returns = rng.normal(0.0004, 0.015, 140)
        theta = rng.normal(0, 0.15, 5)
        starts = np.array([0, 90])
        config = test_config()
        for utility in ("SHARPE", "MEANVAR"):
            _, analytic = objective(theta, design, returns, starts, utility, config)
            numerical = []
            for j in range(len(theta)):
                perturbation = np.zeros(len(theta))
                perturbation[j] = 1e-6
                a = objective(theta + perturbation, design, returns, starts, utility, config)[0]
                b = objective(theta - perturbation, design, returns, starts, utility, config)[0]
                numerical.append((a - b) / 2e-6)
            np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=2e-6)

    def test_smoothing_preserves_prefix_and_uses_no_future_feature(self):
        rng = np.random.default_rng(8)
        design = np.column_stack((np.ones(150), rng.normal(size=(150, 3))))
        theta = np.array([0.2, -0.1, 0.3, 0.5])
        a, _ = smooth_policy(design, theta, 0.2, np.array([0]))
        b, _ = smooth_policy(design[:75], theta, 0.2, np.array([0]))
        np.testing.assert_array_equal(a[:75], b)
        self.assertTrue(((a > 0) & (a < 1)).all())

    def test_mature_training_ignores_future_labels_and_fit_is_bounded(self):
        prices, dividends = synthetic()
        data, columns = factors(prices, dividends)
        origin = 850
        train = eligible_training(data, origin, 1, 756)
        self.assertTrue((train + 2 <= origin).all())
        rng = np.random.default_rng(4)
        labels = rng.normal(0.0004, 0.015, len(data))
        modified = labels.copy()
        modified[origin - 1:] = 99
        x = data[columns[:4]].to_numpy(float)
        a = fit_policy(x[train], labels[train], train, "SHARPE", test_config())
        b = fit_policy(x[train], modified[train], train, "SHARPE", test_config())
        np.testing.assert_array_equal(a["theta"], b["theta"])
        self.assertLessEqual(abs(a["theta"][0]), 4)
        self.assertTrue((abs(a["theta"][1:]) <= 2).all())
        self.assertTrue(a["usable"])

    def test_missing_prediction_does_not_update_smoothed_state(self):
        model = {"usable": True, "theta": np.array([0.0, 1.0]), "mean": np.zeros(1), "scale": np.ones(1)}
        x = np.array([[1.0], [np.nan], [2.0]])
        a, final = predict_sequence(model, x, np.array([True, False, True]), 0.5, test_config())
        b, expected = predict_sequence(model, x[[0, 2]], np.ones(2, dtype=bool), 0.5, test_config())
        self.assertTrue(np.isnan(a[1]))
        np.testing.assert_array_equal(a[[0, 2]], b)
        self.assertEqual(final, expected)

    def test_actual_account_matches_shared_engine_and_no_view_keeps_inventory(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        config = test_config()
        config["evaluation_start"] = str(data.date.iloc[700].date())
        target = 0.5 + 0.3 * np.sin(np.arange(len(data)) / 31)
        a, _ = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", target)
        b, _ = simulate(data, dividends, config, config["costs"]["BASE"], config["evaluation_start"], "TEST", targets=target)
        for column in ("equity", "cash", "shares", "net_return", "commission", "slippage_cost", "dividend_receivable"):
            np.testing.assert_allclose(a[column], b[column], atol=1e-10, rtol=0)
        target[700] = np.nan
        missing, decisions = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", target)
        self.assertGreater(missing.shares.iloc[0], 0)
        self.assertEqual(missing.shares.iloc[1], missing.shares.iloc[0])
        self.assertEqual(missing.filled_quantity.iloc[1], 0)
        self.assertEqual(decisions.view.iloc[1], "NO_VIEW")
        self.assertTrue(np.isnan(decisions.reference_weight.iloc[1]))


if __name__ == "__main__":
    unittest.main()
