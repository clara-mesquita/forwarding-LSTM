import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
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


def positional_encoding(length, d_model):
    positions = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = positions * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)


def create_transformer(units, train, learning_rate, heads=4, num_layers=1, dropout=0.1):
    head_dim = units // heads
    ff_dim = 4 * units

    inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = Dense(units)(inputs)
    x = x + positional_encoding(train.shape[1], units)

    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=heads, key_dim=head_dim)(x, x)
        attn = Dropout(dropout)(attn)
        x = LayerNormalization(epsilon=1e-6)(x + attn)

        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dense(units)(ff)
        ff = Dropout(dropout)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)

    out = Dense(1)(x[:, -1, :])
    model = tf.keras.Model(inputs, out)
    model.compile(
        loss=Huber(delta=0.25),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    return model


def create_dataset(X, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i : i + look_back])
        ys.append(X[i + look_back])
    return np.array(Xs), np.array(ys)


def bits_para_megabits(df, col_vaz):
    df[col_vaz] = df[col_vaz] / 1_000_000
    df[col_vaz] = df[col_vaz].replace(-1, np.nan)
    return df


def linear_interpolation(df, limit_direction="both", method="linear"):
    df_imputed = df.interpolate(method=method, limit_direction=limit_direction)
    df["Vazao"] = df["Vazao"].fillna(df_imputed["Vazao"])
    return df


def main():
    LOOK_BACK = 24

    sources = ["rnp", "geant"]
    for source in sources:
        DATASET_DIR = f"datasets/data/{source}"
        RESULTS_DIR = f"results2/{source}"
        MODEL_DIR = os.path.join(RESULTS_DIR, "models")
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_transformer.json")
        LOG_FILE = os.path.join(RESULTS_DIR, "experiments_transformer.log")

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
                model_name = "Transformer"

                print(f"Processing {dataset_name} with {model_name}...")
                logging.info(f"Processing {dataset_name} with {model_name}...")

                try:
                    df = pd.read_csv(file_path, index_col="Data")
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

                    X_train, y_train = create_dataset(train_scaled, LOOK_BACK)
                    X_test, y_test = create_dataset(test_scaled, LOOK_BACK)

                    param_grid = {
                        "units": [32, 64, 128],
                        "learning_rate": [1e-2, 1e-3, 1e-4],
                    }
                    model_builder = create_transformer

                    best_score = float("inf")
                    best_params = {}
                    for units in param_grid["units"]:
                        for lr in param_grid["learning_rate"]:
                            model = model_builder(units, X_train, lr)
                            early_stop = EarlyStopping(
                                monitor="val_loss",
                                patience=10,
                                restore_best_weights=True,
                            )
                            history = model.fit(
                                X_train,
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
                        X_train,
                        best_params["learning_rate"],
                    )
                    early_stop = EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
                    )
                    model.fit(
                        X_train,
                        y_train,
                        epochs=100,
                        validation_split=0.2,
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=0,
                    )
                    predictions = model.predict(X_test)

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

                    logging.info(
                        f"Model: {model_name}, Dataset: {dataset_name}, Metrics: {metrics}"
                    )

                    model.save(
                        os.path.join(
                            MODEL_DIR, f"{dataset_name}_{model_name}.keras"
                        )
                    )

                except Exception as e:
                    print(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )
                    logging.error(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)


if __name__ == "__main__":
    main()
