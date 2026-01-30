
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from code.models.base_model import BaseModel

def create_dataset(X, look_back=3):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        v = X[i : (i + look_back)]
        Xs.append(v)
        ys.append(X[i + look_back])
    return np.array(Xs), np.array(ys)

class RecursiveLSTM(BaseModel):
    def __init__(self, look_back=3, units=64, epochs=100, batch_size=32, patience=10, learning_rate=1e-3):
        super().__init__("RecursiveLSTM")
        self.look_back = look_back
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=(input_shape[1], input_shape[2])))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(LSTM(units=self.units))
        model.add(Dense(1))
        model.compile(
            loss=Huber(delta=0.25),
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=[RootMeanSquaredError()],
        )
        return model

    def train(self, train_data):
        train_scaled = self.scaler.fit_transform(train_data[['Vazao']])
        X_train, y_train = create_dataset(train_scaled, self.look_back)
        
        self._model = self._build_model(X_train.shape)
        
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        
        self._model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0,
        )

    def predict(self, test_data, horizon):
        train_data = self.scaler.transform(test_data[['Vazao']])
        
        # Taking the last 'look_back' points from the training data to start the prediction
        state = train_data[-self.look_back:].reshape(1, self.look_back, 1)
        
        predictions = []
        
        for _ in range(horizon):
            pred_scaled = self._model.predict(state, verbose=0)
            predictions.append(pred_scaled[0, 0])
            
            # Roll the state and append the new prediction
            new_state = np.roll(state, -1, axis=1)
            new_state[0, -1, 0] = pred_scaled[0, 0]
            state = new_state
            
        predictions_unscaled = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_unscaled.flatten()
