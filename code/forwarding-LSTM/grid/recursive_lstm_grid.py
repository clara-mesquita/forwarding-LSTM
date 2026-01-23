import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(16, 6))
plt.rc("font", size=13)


def grid_search_cv(
    modelo,
    units,
    X_train,
    learning_rates,
    y_train,
    epochs_list,
    batch_sizes,
    patiences,
    model_name,
):
    best_loss = float("inf")
    best_params = {}
    for lr in learning_rates:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for patience in patiences:
                    model = modelo(units, X_train, lr)
                    histories = fit_model_with_cross_validation(
                        model,
                        X_train,
                        y_train,
                        model_name,
                        patience,
                        epochs,
                        batch_size,
                    )
                    mean_history = calculate_mean_history(histories)
                    val_loss = min(mean_history["val_loss"])
                    print(
                        f"Val Loss: {val_loss}, learning rate: {lr}, epochs: {epochs}, batch_size: {batch_size}, patience: {patience}"
                    )
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_params = {
                            "learning_rate": lr,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "patience": patience,
                        }
    print(
        f"O modelo {model_name} tem como melhores parametros os seguintes: {best_params}"
    )
    return best_params


def fit_model_with_cross_validation(
    model, xtrain, ytrain, model_name, patience, epochs, batch_size
):
    tscv = TimeSeriesSplit(n_splits=5)
    histories = []
    for fold, (train_index, val_index) in enumerate(tscv.split(xtrain)):
        x_train_fold, x_val_fold = xtrain[train_index], xtrain[val_index]
        y_train_fold, y_val_fold = ytrain[train_index], ytrain[val_index]
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-5,
        )
        history = model.fit(
            x_train_fold,
            y_train_fold,
            epochs=epochs,
            validation_data=(x_val_fold, y_val_fold),
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )
        print(f"\n\nTREINAMENTO - Fold {fold + 1} do modelo: {model_name}")
        histories.append(history)
    return histories


def calculate_mean_history(histories):
    mean_history = {
        "loss": [],
        "root_mean_squared_error": [],
        "val_loss": [],
        "val_root_mean_squared_error": [],
    }
    for fold_history in histories:
        for key in mean_history.keys():
            mean_history[key].append(fold_history.history[key])
    for key, values in mean_history.items():
        max_len = max(len(val) for val in values)
        for i in range(len(values)):
            if len(values[i]) < max_len:
                values[i] += [values[i][-1]] * (max_len - len(values[i]))
    for key, values in mean_history.items():
        mean_history[key] = [sum(vals) / len(vals) for vals in zip(*values)]
    return mean_history


def create_dataset(X, look_back=3):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        v = X[i : (i + look_back)]
        Xs.append(v)
        ys.append(X[i + look_back])
    return np.array(Xs), np.array(ys)


def create_lstm(units, train, learning_rate):
    model = Sequential()
    model.add(Input(shape=(train.shape[1], train.shape[2])))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(
        loss=Huber(delta=0.25),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    return model

def fit_model(model, xtrain, ytrain, model_name, patience, epochs, batch_size):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        xtrain,
        ytrain,
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop],
    )
    print(f"\n\nTREINAMENTO: {model_name}")
    return history

def prediction(model, xtest, ytest, myscaler, model_name, link, results_path):
    pred = model.predict(xtest)
    pred = myscaler.inverse_transform(pred)
    dataframe_prediction = pd.DataFrame(
        {"Prediction": pred.flatten(), "Test": ytest.flatten()}
    )
    save_path = os.path.join(
        results_path, "forecast", f"prediction_{model_name}_{link}.csv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataframe_prediction.to_csv(save_path)
    return pred

def validate_missing_data_prediction(predictions, actual, mask, model_name):
    predictions = predictions.flatten()
    actual = actual.flatten()
    mask = mask.astype(bool).flatten()

    preds_valid = predictions[mask]
    acts_valid = actual[mask]

    if len(acts_valid) == 0:
        print(f"{model_name}: NÃO HÁ PONTOS REAIS PARA AVALIAÇÃO NESTA JANELA!")
        return np.nan, np.nan, np.nan, model_name

    errors = preds_valid - acts_valid
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    nrmse_median = rmse / ((acts_valid.max() - acts_valid.min()) + 1e-12)
    nrmse = rmse / (acts_valid.mean() + 1e-12)
    mae = np.abs(errors).mean()

    print(f"{model_name} (missing-aware):")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE: {nrmse * 100:.4f}%")

    return rmse, mae, nrmse, nrmse_median, model_name

def bits_para_megabits(df, col_vaz):
    df[col_vaz] = df[col_vaz] / 1_000_000
    df[col_vaz] = df[col_vaz].replace(-1, np.nan)
    return df

def linear_interpolation(df, limit_direction="both", method="linear"):
    df_imputed = df.interpolate(method=method, limit_direction=limit_direction)
    df["Vazao"] = df["Vazao"].fillna(df_imputed["Vazao"])
    return df

def plot_loss_cv(mean_history, model_name, link, results_path):
    epochs = range(1, len(mean_history["loss"]) + 1)
    plt.figure(figsize=(15, 6), dpi=100)
    plt.plot(epochs, mean_history["loss"], label="Train Loss")
    plt.plot(epochs, mean_history["val_loss"], label="Validation Loss")
    plt.title(f"Mean Training and Validation Loss for {link} {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_path = os.path.join(
        results_path, "plots", "loss", f"{link}_{model_name}_loss_cv.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_rmse_cv(mean_history, model_name, link, results_path):
    epochs = range(1, len(mean_history["root_mean_squared_error"]) + 1)
    plt.figure(figsize=(15, 6), dpi=100)
    plt.plot(epochs, mean_history["root_mean_squared_error"], label="Train RMSE")
    plt.plot(
        epochs, mean_history["val_root_mean_squared_error"], label="Validation RMSE"
    )
    plt.title(f"Mean Training and Validation RMSE for {link} {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    save_path = os.path.join(
        results_path, "plots", "rmse", f"{link}_{model_name}_rmse_cv.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_future(prediction, y_test, model_name, link, results_path):
    plt.figure(figsize=(15, 6), dpi=100)
    range_future = len(y_test)
    plt.plot(np.arange(range_future), y_test, label="Test data")
    plt.plot(np.arange(range_future), prediction, label=model_name)
    plt.title(f"Test data vs prediction for {link}")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Mbits/s")
    save_path = os.path.join(
        results_path, "plots", "predictions", f"{link}_{model_name}_future.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def save_model_and_metrics(model, metrics, model_dir, metrics_path, file_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, f"{file_name}.keras"))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

def walk_forward_validation_hybrid(
    model, scaler, train_data, test_data, mask_test, look_back, window_size
):
    state = train_data[-look_back:].reshape(1, look_back, 1)
    predictions = []
    for i in range(0, len(test_data), window_size):
        end_idx = min(i + window_size, len(test_data))
        window_data = test_data[i:end_idx]
        window_mask = mask_test[i:end_idx]
        window_preds = []
        current_state = state
        for j in range(len(window_data)):
            pred_scaled = model.predict(current_state, verbose=0)
            pred_orig = scaler.inverse_transform(pred_scaled)[0, 0]
            window_preds.append(pred_orig)
            if window_mask[j]:
                new_point = window_data[j].item()
            else:
                new_point = pred_scaled[0, 0]
            current_state = np.roll(current_state, -1, axis=1)
            current_state[0, -1, 0] = new_point
        predictions.extend(window_preds)
        state = current_state
    return predictions

def main():
    # Configuration
    LOOK_BACK = 3
    WINDOW_SIZE = 28

    protocols = ["cubic", "bbr"]

    for protocol in protocols:
        DATASET_DIR = f"datasets/treated/{protocol}"
        RESULTS_DIR = f"results/{protocol}"
        MODEL_DIR = os.path.join(RESULTS_DIR, "models")
        METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation.json")
        LOG_FILE = os.path.join(RESULTS_DIR, "experiments_recursive_lstm.log")

        if not os.path.exists(DATASET_DIR):
            print(f"Directory not found, skipping: {DATASET_DIR}")
            continue

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

                print(f"Processing {dataset_name}...")
                logging.info(f"Processing {dataset_name}...")

                try:
                    # Data loading and preprocessing
                    df = pd.read_csv(file_path, index_col="Timestamp")
                    if "0" in df.columns:
                        df.drop("0", axis=1, inplace=True)

                    df = bits_para_megabits(df, "Vazao")
                    mask_total = ~(df["Vazao"].isna() | (df["Vazao"] == -1))

                    train_size = int(len(df.index) * 0.8)
                    train_data = df.iloc[:train_size].copy()
                    test_data = df.iloc[train_size:].copy()

                    train_data = linear_interpolation(train_data)

                    scaler = MinMaxScaler().fit(train_data[["Vazao"]])
                    train_scaled = scaler.transform(train_data[["Vazao"]])
                    test_scaled = scaler.transform(test_data[["Vazao"]])
                    mask_test = mask_total.iloc[train_size:].values

                    X_train, y_train = create_dataset(train_scaled, LOOK_BACK)

                    # Grid Search
                    best_params = grid_search_cv(
                        create_lstm,
                        64,
                        X_train,
                        [1e-3],
                        y_train,
                        [100, 300, 500],
                        [32, 64, 128],
                        [10],
                        "LSTM",
                    )

                    # Final model training
                    model = create_lstm(64, X_train, best_params["learning_rate"])
                    history = model.fit(
                        X_train,
                        y_train,
                        epochs=best_params["epochs"],
                        batch_size=best_params["batch_size"],
                        validation_split=0.2,
                        callbacks=[
                            EarlyStopping(
                                monitor="val_loss",
                                patience=best_params["patience"],
                                restore_best_weights=True,
                            )
                        ],
                    )

                    # Walk-forward validation
                    predictions = walk_forward_validation_hybrid(
                        model,
                        scaler,
                        train_scaled,
                        test_scaled,
                        mask_test,
                        LOOK_BACK,
                        WINDOW_SIZE,
                    )

                    # Evaluation
                    min_len = min(len(predictions), len(test_scaled) - LOOK_BACK)
                    y_test_valid = test_scaled[LOOK_BACK : LOOK_BACK + min_len]
                    mask_eval = mask_test[LOOK_BACK : LOOK_BACK + min_len]
                    y_test_orig = scaler.inverse_transform(y_test_valid)
                    predictions_arr = np.array(predictions[:min_len]).reshape(-1, 1)

                    rmse, mae, nrmse, _, _ = validate_missing_data_prediction(
                        predictions_arr, y_test_orig, mask_eval, "LSTM"
                    )

                    y_test_flat = y_test_orig.flatten()
                    predictions_flat = predictions_arr.flatten()
                    
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

                    evaluation_metrics[dataset_name] = metrics
                    logging.info(f"Model: LSTM, Dataset: {dataset_name}, Metrics: {metrics}")


                    # Save results and plots
                    save_model_and_metrics(
                        model, evaluation_metrics, MODEL_DIR, METRICS_PATH, dataset_name
                    )
                    plot_loss_cv(
                        calculate_mean_history([history]),
                        "LSTM",
                        dataset_name,
                        RESULTS_DIR,
                    )
                    plot_rmse_cv(
                        calculate_mean_history([history]),
                        "LSTM",
                        dataset_name,
                        RESULTS_DIR,
                    )
                    plot_future(
                        predictions_arr, y_test_orig, "LSTM", dataset_name, RESULTS_DIR
                    )

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    logging.error(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    main()