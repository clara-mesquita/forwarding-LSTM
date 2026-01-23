import statsmodels.api as sm
import json
import os
import warnings
import time
import logging
from fuzzy_analysis import compute_fuzzy_accuracy as fuzzy_accuracy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(16, 6))
plt.rc("font", size=13)


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
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_sarima.json")
        LOG_FILE = os.path.join(RESULTS_DIR, "experiments_sarima.log")

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
                model_name = "SARIMA"

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

                    y_train = train_scaled[1:]
                    y_test = test_scaled[1:]

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


                except Exception as e:
                    print(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )
                    logging.error(f"Error processing file {filename} with model {model_name}: {e}")

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)


if __name__ == "__main__":
    main()