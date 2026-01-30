
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for time series models."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self._model = None

    @abstractmethod
    def train(self, train_data, **kwargs):
        """
        Train the model.

        :param train_data: A pandas Series or DataFrame with the training data.
        """
        pass

    @abstractmethod
    def predict(self, test_data, horizon, **kwargs):
        """
        Make predictions for a given horizon.

        :param test_data: A pandas Series or DataFrame with the test data.
        :param horizon: The number of steps to forecast ahead.
        :return: A numpy array of predictions.
        """
        pass

    def save(self, model_dir):
        """Save the trained model."""
        pass

    def load(self, model_dir):
        """Load a trained model."""
        pass
