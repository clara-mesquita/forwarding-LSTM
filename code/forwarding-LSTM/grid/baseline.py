import statsmodels.api as sm
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices("GPU")
print("GPUs detectadas:", gpus)

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Usando GPU {}".format(gpus[0]))
    except RuntimeError as e:
        print(e)

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(16, 6))
plt.rc("font", size=13)


def create_lstm(units, train, learning_rate):
    model = Sequential()
    model.add(Input(shape=(train.shape[1], train.shape[2])))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(
        loss=Huber(delta=0.25),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    return model


def create_gru(units, train, learning_rate):
    model = Sequential()
    model.add(Input(shape=(train.shape[1], train.shape[2])))
    model.add(GRU(units=units))
    model.add(Dense(1))
    model.compile(
        loss=Huber(delta=0.25),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    return model


def bits_para_megabits(df, col_vaz):
    df[col_vaz] = df[col_vaz] / 1_000_000
    df[col_vaz] = df[col_vaz].replace(-1, np.nan)
    return df


def linear_interpolation(df, limit_direction="both", method="linear"):
    df_imputed = df.interpolate(method=method, limit_direction=limit_direction)
    df["Vazao"] = df["Vazao"].fillna(df_imputed["Vazao"])
    return df


def main():
    protocols = ["cubic", "bbr"]
    for protocol in protocols:
        DATASET_DIR = "datasets/amostras/"
        RESULTS_DIR = f"results2/{protocol}"
        MODEL_DIR = os.path.join(RESULTS_DIR, "models")
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation.json")

        if not os.path.exists(DATASET_DIR):
            print(f"Directory not found, skipping: {DATASET_DIR}")
            continue

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        evaluation_metrics = {}

        for filename in os.listdir(DATASET_DIR):
            if filename.endswith(".csv"):
                file_path = os.path.join(DATASET_DIR, filename)
                dataset_name = os.path.splitext(filename)[0]

                print(f"Processing {dataset_name}...")

                try:
                    df = pd.read_csv(file_path, index_col="Timestamp")
                    if "0" in df.columns:
                        df.drop("0", axis=1, inplace=True)

                    df = bits_para_megabits(df, "Vazao")

                    train_size = int(len(df.index) * 0.8)
                    train_data = df.iloc[:train_size].copy()
                    test_data = df.iloc[train_size:].copy()

                    train_data = linear_interpolation(train_data)

                    scaler = MinMaxScaler().fit(train_data[["Vazao"]])
                    train_scaled = scaler.transform(train_data[["Vazao"]])
                    test_scaled = scaler.transform(test_data[["Vazao"]])

                    X_train = train_scaled[:-1]
                    y_train = train_scaled[1:]

                    X_test = test_scaled[:-1]
                    y_test = test_scaled[1:]

                    # Reshape for LSTM and GRU
                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, 1))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, 1))

                    # Models
                    models = {
                        "LSTM": (
                            create_lstm,
                            {
                                "units": [32, 64, 128],
                                "learning_rate": [1e-2, 1e-3, 1e-4],
                            },
                        ),
                        "GRU": (
                            create_gru,
                            {
                                "units": [32, 64, 128],
                                "learning_rate": [1e-2, 1e-3, 1e-4],
                            },
                        ),
                        "SARIMA": None,  # SARIMA has a different grid search
                        "RandomForest": (
                            RandomForestRegressor,
                            {
                                "n_estimators": [50, 100, 200],
                                "max_depth": [None, 10, 20],
                                "min_samples_leaf": [1, 2, 4],
                            },
                        ),
                    }

                    for model_name, (model_builder, param_grid) in models.items():
                        print(f"Training {model_name}...")

                        if model_name in ["LSTM", "GRU"]:
                            best_score = float("inf")
                            best_params = {}
                            for units in param_grid["units"]:
                                for lr in param_grid["learning_rate"]:
                                    model = model_builder(units, X_train_reshaped, lr)
                                    early_stop = EarlyStopping(
                                        monitor="val_loss",
                                        patience=10,
                                        restore_best_weights=True,
                                    )
                                    history = model.fit(
                                        X_train_reshaped,
                                        y_train,
                                        epochs=100,
                                        validation_split=0.2,
                                        batch_size=32,
                                        callbacks=[early_stop],
                                        verbose=0,
                                    )
                                    val_loss = min(history.history["val_loss"])
                                    if val_loss < best_score:
                                        best_score = val_loss
                                        best_params = {
                                            "units": units,
                                            "learning_rate": lr,
                                        }

                            print(f"Best params for {model_name}: {best_params}")
                            model = model_builder(
                                best_params["units"],
                                X_train_reshaped,
                                best_params["learning_rate"],
                            )
                            model.fit(
                                X_train_reshaped,
                                y_train,
                                epochs=100,
                                validation_split=0.2,
                                batch_size=32,
                                callbacks=[early_stop],
                                verbose=0,
                            )
                            predictions = model.predict(X_test_reshaped)

                        elif model_name == "RandomForest":
                            grid = GridSearchCV(
                                model_builder(),
                                param_grid,
                                cv=TimeSeriesSplit(n_splits=5),
                                n_jobs=-1,
                                verbose=0,
                            )
                            grid.fit(X_train, y_train.ravel())
                            print(f"Best params for {model_name}: {grid.best_params_}")
                            model = grid.best_estimator_
                            predictions = model.predict(X_test)
                            predictions = predictions.reshape(-1, 1)

                        elif model_name == "SARIMA":
                            # SARIMA requires a different approach for grid search
                            p_values = d_values = q_values = [0, 1]
                            P_values = D_values = Q_values = [0, 1]
                            s_values = [12]
                            best_score, best_cfg = float("inf"), None

                            for p in p_values:
                                for d in d_values:
                                    for q in q_values:
                                        for P in P_values:
                                            for D in D_values:
                                                for Q in Q_values:
                                                    for s in s_values:
                                                        order = (p, d, q)
                                                        seasonal_order = (P, D, Q, s)
                                                        try:
                                                            model = sm.tsa.SARIMAX(
                                                                y_train.ravel(),
                                                                order=order,
                                                                seasonal_order=seasonal_order,
                                                            )
                                                            model_fit = model.fit(
                                                                disp=False
                                                            )
                                                            if (
                                                                model_fit.aic
                                                                < best_score
                                                            ):
                                                                best_score, best_cfg = (
                                                                    model_fit.aic,
                                                                    (
                                                                        order,
                                                                        seasonal_order,
                                                                    ),
                                                                )
                                                        except Exception:
                                                            continue
                            print(f"Best SARIMA config: {best_cfg}")
                            model = sm.tsa.SARIMAX(
                                y_train.ravel(),
                                order=best_cfg[0],
                                seasonal_order=best_cfg[1],
                            )
                            model_fit = model.fit(disp=False)
                            predictions = model_fit.predict(
                                start=len(y_train), end=len(y_train) + len(y_test) - 1
                            )
                            predictions = predictions.reshape(-1, 1)

                        predictions_unscaled = scaler.inverse_transform(predictions)
                        y_test_unscaled = scaler.inverse_transform(y_test)

                        rmse = np.sqrt(
                            mean_squared_error(y_test_unscaled, predictions_unscaled)
                        )
                        mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
                        nrmse = rmse / (y_test_unscaled.mean() + 1e-12)

                        evaluation_metrics[f"{dataset_name}_{model_name}"] = {
                            "RMSE": rmse,
                            "MAE": mae,
                            "NRMSE": nrmse,
                        }

                        # Save model
                        if model_name in ["LSTM", "GRU"]:
                            model.save(
                                os.path.join(
                                    MODEL_DIR, f"{dataset_name}_{model_name}.h5"
                                )
                            )

                except Exception as e:
                    print(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)


if __name__ == "__main__":
    main()
