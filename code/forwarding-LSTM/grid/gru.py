import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import time
import logging
from fuzzy_analysis import compute_fuzzy_accuracy as fuzzy_accuracy

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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(16, 6))
plt.rc("font", size=13)


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
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_gru.json")
        LOG_FILE = os.path.join(RESULTS_DIR, "experiments_gru.log")

        if not os.path.exists(DATASET_DIR):
            print(f"Directory not found, skipping: {DATASET_DIR}")
            continue

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        evaluation_metrics = {}

        for filename in os.listdir(DATASET_DIR):
            if filename.endswith(".csv"):
                start_time = time.time()
                file_path = os.path.join(DATASET_DIR, filename)
                dataset_name = os.path.splitext(filename)[0]
                model_name = "GRU"

                print(f"Processing {dataset_name} with {model_name}...")
                logging.info(f"Processing {dataset_name} with {model_name}...")

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

                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, 1))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, 1))

                    param_grid = {
                        "units": [32, 64, 128],
                        "learning_rate": [1e-2, 1e-3, 1e-4],
                    }
                    model_builder = create_gru

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
                    early_stop = EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
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

                    predictions_unscaled = scaler.inverse_transform(predictions)
                    y_test_unscaled = scaler.inverse_transform(y_test)

                    rmse = np.sqrt(
                        mean_squared_error(y_test_unscaled, predictions_unscaled)
                    )
                    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
                    nrmse = rmse / (y_test_unscaled.mean() + 1e-12)
                    
                    y_test_flat = y_test_unscaled.flatten()
                    predictions_flat = predictions_unscaled.flatten()

                    fuzzy_acc, _ = fuzzy_accuracy(y_test_flat, predictions_flat)

                    end_time = time.time()
                    execution_time = end_time - start_time

                    metrics = {
                        "RMSE": rmse,
                        "MAE": mae,
                        "NRMSE": nrmse,
                        "Fuzzy_Accuracy": fuzzy_acc,
                        "Execution_Time": execution_time,
                    }
                    evaluation_metrics[f"{dataset_name}_{model_name}"] = metrics
                    
                    logging.info(f"Model: {model_name}, Dataset: {dataset_name}, Metrics: {metrics}")


                    model.save(
                        os.path.join(
                            MODEL_DIR, f"{dataset_name}_{model_name}.h5"
                        )
                    )

                except Exception as e:
                    print(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )
                    logging.error(f"Error processing file {filename} with model {model_name}: {e}")


        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)


if __name__ == "__main__":
    main()