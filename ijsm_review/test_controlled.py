import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

from controlled_models import Network, online
from run_controlled import preparation, windows, validation_arrays, fit_statistical


class ControlledProtocolTest(unittest.TestCase):
    def test_failed_final_statistical_fit_retries_without_changing_configuration(self):
        first = Mock(mle_retvals={"converged": False}, params=np.array([.5]))
        second = Mock(mle_retvals={"converged": True})
        first.model.fit.return_value = second
        with patch("run_controlled.SARIMAX") as constructor:
            constructor.return_value.fit.return_value = first
            fitted, messages = fit_statistical(np.array([1., 2., np.nan, 4.]),
                                              dict(order=(1, 0, 0)), retry=True)
        self.assertIs(fitted, second)
        self.assertIn("Powell", messages[0])
        first.model.fit.assert_called_once()

    def test_online_prediction_precedes_observation_and_recurses_missing(self):
        class Increment:
            def predict(self, x):
                return x[:, -1] + 1
        np.testing.assert_array_equal(online(Increment(), [1., 2., 3.], [np.nan, 10., np.nan]), [4., 5., 11.])
        np.testing.assert_array_equal(online(None, [1., 2., 3.], [np.nan, 10., np.nan]), [3., 3., 10.])

    def test_missing_targets_are_never_interpolated_into_training_labels(self):
        raw = np.array([1., 2., np.nan, 4., np.nan, 6.])
        inputs, lower, scale = preparation(raw)
        x, y = windows(inputs, (raw-lower)/scale, 2)
        self.assertEqual(len(y), 2)
        np.testing.assert_allclose(y, [.6, 1.])
        self.assertEqual(x.shape, (2, 2))

    def test_validation_filling_cannot_see_future_observations(self):
        x, y = validation_arrays(np.array([1., 2., 3.]), np.array([np.nan, 100., 101.]), 2)
        np.testing.assert_array_equal(x[0], [3., 3.])
        np.testing.assert_array_equal(y, [100., 101.])

    def test_training_scaler_is_independent_of_test_values(self):
        inputs, lower, scale = preparation(np.array([10., np.nan, 20.]))
        self.assertEqual((lower, scale), (10., 10.))
        np.testing.assert_array_equal(inputs, [0., .5, 1.])

    def test_neural_architectures_forward_backward_both_contexts(self):
        torch.set_num_threads(1)
        for context in (3, 28):
            for name, config in (("MLP", dict(hidden=(64, 32))),
                                 ("Transformer", dict(layers=2)),
                                 ("LSTM", dict(units=32)), ("GRU", dict(units=32)),
                                 ("ForwardingLSTM", dict(units=64))):
                with self.subTest(context=context, name=name):
                    model = Network(name, context, config)
                    prediction = model(torch.randn(4, context))
                    self.assertEqual(tuple(prediction.shape), (4,))
                    prediction.square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
