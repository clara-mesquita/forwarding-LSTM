
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

from code.models.base_model import BaseModel

warnings.filterwarnings("ignore")

class SARIMAModel(BaseModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self._model = None

    def train(self, train_data):
        # SARIMA works directly on the time series data
        try:
            model = sm.tsa.SARIMAX(
                train_data['Vazao'],
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self._model = model.fit(disp=False)
        except Exception as e:
            print(f"Error training SARIMA: {e}")
            # Fallback to a simpler model if the main one fails
            try:
                model = sm.tsa.SARIMAX(train_data['Vazao'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
                self._model = model.fit(disp=False)
            except Exception as e2:
                print(f"Fallback SARIMA failed: {e2}")
                self._model = None

    def predict(self, test_data, horizon):
        if self._model is None:
            return np.full(horizon, np.nan) # Return NaNs if model failed to train
            
        start = len(test_data)
        end = start + horizon - 1
        
        # The predict method of statsmodels is based on the index of the training data
        # To forecast future values, we need to specify start and end points
        predictions = self._model.predict(start=len(test_data), end=len(test_data) + horizon - 1)
        return predictions.values
