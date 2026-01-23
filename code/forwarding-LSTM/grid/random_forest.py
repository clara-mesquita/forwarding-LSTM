import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_random_forest.json")

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
                model_name = "RandomForest"

                print(f"Processing {dataset_name} with {model_name}...")

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

                    model_builder = RandomForestRegressor
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_leaf": [1, 2, 4],
                    }

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

                except Exception as e:
                    print(
                        f"Error processing file {filename} with model {model_name}: {e}"
                    )

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)


if __name__ == "__main__":
    main()
