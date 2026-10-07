"""验证宏观公布钟、严格融资滞后、共同训练池及树预测复算。"""
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from research import macro_technical_first_passage_inputs_v1 as module


def view_fixture():
    dates = pd.bdate_range("2018-01-02", periods=130)
    data = pd.DataFrame({"date": dates, "first_passage_feature_known": True})
    for name in module.TECH:
        data[name] = .1
    orders = pd.DataFrame({"reference_period": ["2017-11", "2017-12", "2018-01"],
                           "available_at": ["2017-11-30T09:00:00+08:00", "2017-12-31T09:00:00+08:00", "2018-02-01T17:00:00+08:00"],
                           "first_release_value": [49., 50., 80.], "source_url": ["旧源", "旧源", "晚发布源"]})
    gap = np.linspace(.1, .3, len(dates))
    funding = pd.DataFrame({"date": dates, "fund_stat_date": dates - pd.Timedelta(days=1),
                            "available_at": [str(d.date()) + "T09:30:00+08:00" for d in dates],
                            "policy_known_at": "2017-12-01T09:00:00+08:00", "fund_known": True,
                            "gap_pp": gap, "dr007": 2. + gap, "rate": 2.})
    margin = pd.DataFrame({"date": dates, "market_rzye": np.arange(len(dates)) + 10000.,
                           "market_rzmre": np.arange(len(dates)) + 1000.})
    return data, orders, funding, margin


def training_fixture():
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2017-01-03", periods=500)
    data = pd.DataFrame(rng.normal(size=(500, len(module.TECH + module.MACRO))), columns=module.TECH + module.MACRO)
    data["date"] = dates
    data["first_passage_feature_known"] = True
    data["joint_features_known"] = True
    classes = np.asarray(["LOSS", "PROFIT", "TIMEOUT"])[np.arange(500) % 3]
    outcomes = pd.DataFrame({"origin_index": np.arange(500), "entry_idx": np.arange(500) + 1,
                             "exit_idx": np.arange(500) + 2, "mature_idx": np.arange(500) + 2,
                             "status": "MATURE_REFERENCE", "event_class": classes,
                             "reference_net_return": np.asarray([-.02, .04, .001])[np.arange(500) % 3]})
    return data, outcomes


class MacroClockTests(unittest.TestCase):
    def test_late_pmi_release_is_not_used_at_close(self):
        data, orders, funding, margin = view_fixture()
        d = module.views(data, orders, funding, margin).set_index("date")
        self.assertEqual(d.loc["2018-02-01", "pmi_orders_level"], 0.)
        self.assertEqual(d.loc["2018-02-02", "pmi_orders_level"], 30.)

    def test_today_financing_is_available_only_next_session(self):
        data, orders, funding, margin = view_fixture()
        before = module.views(data, orders, funding, margin)
        changed = margin.copy()
        changed.loc[90, "market_rzye"] *= 2.
        after = module.views(data, orders, funding, changed)
        self.assertEqual(before.financing_net_change5.iloc[90], after.financing_net_change5.iloc[90])
        self.assertNotEqual(before.financing_net_change5.iloc[91], after.financing_net_change5.iloc[91])

    def test_missing_financing_day_is_not_skipped(self):
        data, orders, funding, margin = view_fixture()
        result = module.views(data, orders, funding, margin.drop(index=90))
        self.assertTrue(pd.isna(result.financing_net_change5.iloc[91]))
        self.assertFalse(result.margin_known.iloc[91])

    def test_unknown_funding_does_not_become_zero(self):
        data, orders, funding, margin = view_fixture()
        funding.loc[90, "fund_known"] = False
        result = module.views(data, orders, funding, margin)
        self.assertTrue(pd.isna(result.funding_gap_pp.iloc[90]))
        self.assertTrue(pd.isna(result.funding_gap_change5.iloc[95]))

    def test_after_source_end_does_not_carry_forward(self):
        data, orders, funding, margin = view_fixture()
        result = module.views(data, orders, funding.iloc[:100], margin.iloc[:100])
        self.assertFalse(result.joint_features_known.iloc[102])
        self.assertTrue(pd.isna(result.funding_gap_pp.iloc[102]))

    def test_common_pool_excludes_future_and_unknown_origins(self):
        data, outcomes = training_fixture()
        data.loc[300, "joint_features_known"] = False
        outcomes.loc[301, "mature_idx"] = 451
        result = module.common_pool(data, outcomes, 450)
        self.assertNotIn(300, result.origin_index.tolist())
        self.assertNotIn(301, result.origin_index.tolist())
        self.assertTrue(result.mature_idx.le(450).all())

    def test_future_labels_do_not_change_fitted_pair(self):
        data, outcomes = training_fixture()
        before = module.fit_pair(data, outcomes, 450)
        changed = outcomes.copy()
        changed.loc[changed.mature_idx.gt(450), "reference_net_return"] = 1000.
        changed.loc[changed.mature_idx.gt(450), "event_class"] = "PROFIT"
        after = module.fit_pair(data, changed, 450)
        self.assertEqual(before, after)

    def test_saved_tree_predicts_same_probabilities(self):
        data, outcomes = training_fixture()
        x = data[module.TECH].to_numpy(float)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=module.SEED).fit(x, outcomes.event_class)
        model = module.serialize(tree, module.TECH, [{"positive_probability": 0.}] * 3)
        for i in [0, 55, 321, 499]:
            p, leaf, used = module.predict(model, x[i])
            np.testing.assert_allclose(p, tree.predict_proba(x[i:i + 1])[0], rtol=0., atol=1e-15)
            self.assertEqual(leaf, tree.apply(x[i:i + 1])[0])
            self.assertTrue(all(name in module.TECH for name in used))


if __name__ == "__main__":
    unittest.main()
