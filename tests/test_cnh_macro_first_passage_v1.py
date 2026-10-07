"""新接入时钟、共同成熟成员以及实际外汇树路径的必要检验。"""
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from research import cnh_macro_first_passage_inputs_v1 as module


def fixture():
    rng = np.random.default_rng(206)
    dates = pd.bdate_range("2017-01-03", periods=500)
    data = pd.DataFrame(rng.normal(size=(500, len(module.TECH + module.MACRO))), columns=module.TECH + module.MACRO)
    data["date"] = dates
    data["decision_time"] = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    data["first_passage_feature_known"] = True
    data["joint_features_known"] = True
    data["atr14"] = .02
    fixing = np.full(500, 7.)
    close = fixing * np.exp(rng.normal(0, .001, 500))
    fx = pd.DataFrame({"date": dates, "fx_stat_date": dates - pd.Timedelta(days=3),
        "cnh_bid_close": close, "fixing_value": fixing,
        "fixing_available_at": dates.tz_localize("Asia/Shanghai") - pd.Timedelta(days=3),
        "bar_end_utc": dates.tz_localize("UTC") - pd.Timedelta(days=2),
        "conservative_available_at": dates.tz_localize("Asia/Shanghai") - pd.Timedelta(minutes=1),
        "source_age_calendar_days": 3., "both_fields_known": True,
        module.FX[0]: 10000. * np.log(close / fixing), module.FX[1]: rng.normal(size=500)})
    classes = np.asarray(["LOSS", "PROFIT", "TIMEOUT"])[np.arange(500) % 3]
    outcomes = pd.DataFrame({"origin_index": np.arange(500), "entry_idx": np.arange(500) + 1,
        "exit_idx": np.arange(500) + 2, "mature_idx": np.arange(500) + 2,
        "status": "MATURE_REFERENCE", "event_class": classes,
        "reference_net_return": np.asarray([-.02, .04, .001])[np.arange(500) % 3]})
    return data, fx, outcomes


class CnhMacroTests(unittest.TestCase):
    def test_after_close_fx_is_not_known(self):
        data, fx, _ = fixture()
        fx.loc[300, "conservative_available_at"] = data.decision_time.iloc[300] + pd.Timedelta(minutes=1)
        joined = module.attach_fx(data, fx)
        self.assertFalse(joined.joint_features_known.iloc[300])
        self.assertTrue(joined.loc[300, module.FX].isna().all())

    def test_missing_or_stale_fx_is_not_filled_or_removed(self):
        data, fx, _ = fixture()
        fx.loc[300, module.FX[1]] = np.nan
        fx.loc[301, "source_age_calendar_days"] = 8.
        joined = module.attach_fx(data, fx)
        self.assertEqual(len(joined), 500)
        self.assertTrue(joined.loc[[300, 301], module.FX].isna().all().all())
        self.assertTrue((~joined.joint_features_known.iloc[[300, 301]]).all())

    def test_all_three_models_share_only_mature_known_members(self):
        data, fx, outcomes = fixture()
        fx.loc[300, "both_fields_known"] = False
        outcomes.loc[301, "mature_idx"] = 451
        joined = module.attach_fx(data, fx)
        record = module.fit_three(joined, outcomes, 450)
        self.assertEqual(record["status"], "FIT_COMPLETE")
        self.assertNotIn(300, record["training_origins"])
        self.assertNotIn(301, record["training_origins"])
        self.assertLessEqual(record["latest_mature_idx"], 450)
        self.assertEqual(set(record["models"]), set(module.POLICIES))
        for saved in record["models"].values():
            self.assertEqual(saved["node_rows"][0], record["training_rows"])

    def test_future_labels_cannot_change_any_model(self):
        data, fx, outcomes = fixture()
        joined = module.attach_fx(data, fx)
        before = module.fit_three(joined, outcomes, 450)
        changed = outcomes.copy()
        changed.loc[changed.mature_idx.gt(450), "reference_net_return"] = 1000.
        changed.loc[changed.mature_idx.gt(450), "event_class"] = "PROFIT"
        after = module.fit_three(joined, changed, 450)
        self.assertEqual(before, after)

    def test_actual_fx_path_and_saved_probabilities_agree(self):
        data, fx, _ = fixture()
        joined = module.attach_fx(data, fx)
        columns = module.FEATURES[module.POLICIES[2]]
        x = joined[columns].to_numpy(float)
        labels = np.where(joined[module.FX[0]].lt(-4), "LOSS",
                          np.where(joined[module.FX[0]].gt(4), "PROFIT", "TIMEOUT"))
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=module.SEED).fit(x, labels)
        saved = module.parent.serialize(tree, columns, [{"positive_probability": 0.}] * 3)
        for index in [0, 55, 321, 499]:
            p, leaf, path = module.parent.predict(saved, x[index])
            np.testing.assert_allclose(p, tree.predict_proba(x[index:index + 1])[0], rtol=0., atol=1e-15)
            self.assertEqual(leaf, tree.apply(x[index:index + 1])[0])
            self.assertTrue(any(name in module.FX for name in path))


if __name__ == "__main__":
    unittest.main()
