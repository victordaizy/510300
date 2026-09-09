"""分类决策的概率边界与未来不回写检查。"""
import unittest

import numpy as np

from research.return_classification_v1 import classifier, consensus


class ReturnClassificationTests(unittest.TestCase):
    def test_consensus_hysteresis_and_future_prefix(self):
        probability = np.array([np.nan, 0.5, 0.53, 0.52, 0.49, 0.48, 0.47, 0.51, 0.55])
        actual = consensus(probability, 0.52, 0.48)
        np.testing.assert_array_equal(actual, [0, 0, 1, 1, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(actual[:7], consensus(probability[:7], 0.52, 0.48))

    def test_all_models_accept_magnitude_weights_and_valid_probability(self):
        rng = np.random.default_rng(100)
        x = rng.normal(size=(400, 4))
        forward = 0.02 * x[:, 0] + rng.normal(0, 0.01, 400)
        labels = (forward > 0).astype(int)
        weights = np.abs(forward) / np.abs(forward).mean()
        for kind in ("LOGIT", "HGB", "ET"):
            model = classifier(kind, 61)
            model.fit(x, labels, sample_weight=weights)
            probability = model.predict_proba(np.array([[-3, 0, 0, 0], [3, 0, 0, 0]]))[:, 1]
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            self.assertLess(probability[0], probability[1])


if __name__ == "__main__":
    unittest.main()
