import unittest

import numpy as np
import pandas as pd

from forward_prediction import evaluate, prepare_data, walk_forward


class ForwardPredictionTest(unittest.TestCase):
    def test_missing_predictions_feed_back_but_observations_take_priority(self):
        class Model:
            def predict(self, state, verbose=0):
                return state[:, -1, :] + 1

        # First prediction uses training; a missing point feeds the prediction back.
        result = walk_forward(Model(), np.array([1., 2., 3.]),
                              np.array([np.nan, 10., np.nan]))
        np.testing.assert_array_equal(result, [4., 5., 11.])

    def test_preparation_preserves_missing_test_values_and_uses_training_only(self):
        data = pd.DataFrame({
            "Data": pd.date_range("2024-01-01", periods=20, freq="6h"),
            "Vazao": [1_000_000.] * 16 + [-1, 100_000_000., -1, 100_000_000.],
        })
        _, train, test, split = prepare_data(data)
        self.assertEqual(split, 16)
        np.testing.assert_array_equal(train, np.ones(16))
        self.assertTrue(np.isnan(test[[0, 2]]).all())
        np.testing.assert_array_equal(test[[1, 3]], [100, 100])

    def test_metrics_ignore_missing_actual_values(self):
        result = evaluate(np.array([10., np.nan, 20.]), np.array([12., 999., 18.]))
        self.assertEqual(result["observed_test_intervals"], 2)
        self.assertEqual(result["RMSE"], 2.)
        self.assertEqual(result["MAE"], 2.)


if __name__ == "__main__":
    unittest.main()
