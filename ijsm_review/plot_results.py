"""Plot completed Forwarding LSTM and benchmark results.

Run from anywhere: python ijsm_review/plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"
MODELS = ("Forwarding LSTM", "iTransformer", "TimeBridge", "TimeMixer++")
COLORS = ("#176b87", "#e08e45", "#7057a3", "#4d9b74")
METRICS = (("RMSE", "RMSE (Mbit/s)"), ("MAE", "MAE (Mbit/s)"),
           ("NRMSE_mean", "NRMSE"), ("SMAPE", "SMAPE (%)"))


def load_metrics(dataset):
    rows = []
    lstm_dir = RESULTS / "forwarding_lstm" / dataset
    for path in sorted(lstm_dir.glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": dataset, "model": MODELS[0],
                     "link": path.name.removesuffix("_metrics.json"), **payload["metrics"]})
    if not rows:
        raise FileNotFoundError(f"No Forwarding LSTM metrics in {lstm_dir}")
    for model, folder in zip(MODELS[1:], ("iTransformer", "TimeBridge", "TimeMixerPP")):
        path = RESULTS / "benchmarks" / dataset / folder / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in pd.read_csv(path).to_dict("records"):
            rows.append({"dataset": dataset, "model": model, **row})
    frame = pd.DataFrame(rows)
    for column, _ in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["observed_test_intervals"] = pd.to_numeric(frame["observed_test_intervals"], errors="coerce")
    frame["Fuzzy_Accuracy"] = pd.to_numeric(frame["Fuzzy_Accuracy"], errors="coerce")
    counts = frame.groupby("link")["model"].nunique()
    if (counts != len(MODELS)).any():
        raise ValueError(f"Incomplete model results for {dataset}: {counts[counts != len(MODELS)].to_dict()}")
    return frame


def link_labels(links, dataset):
    if dataset == "sbrc":
        return {link: link.split("_bbr_")[0].upper() for link in links}
    return {link: f"G{index:02d}" for index, link in enumerate(sorted(links), 1)}


def plot_overview(frame, dataset):
    valid = frame[frame["observed_test_intervals"] > 0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, METRICS):
        grouped = valid.groupby("model")[metric].mean().reindex(MODELS)
        bars = ax.bar(MODELS, grouped, color=COLORS, width=.7)
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Média entre links avaliáveis")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max(grouped) * 1.18 if max(grouped) > 0 else 1)
    fig.suptitle(f"{dataset.upper()} | desempenho no período de teste", fontsize=15)
    fig.savefig(OUT / f"{dataset}_resumo_metricas.png", dpi=220)
    plt.close(fig)


def plot_by_link(frame, dataset):
    valid = frame[frame["observed_test_intervals"] > 0]
    labels = link_labels(valid["link"].unique(), dataset)
    pd.DataFrame({"codigo": list(labels.values()), "link": list(labels.keys())}).to_csv(
        OUT / f"{dataset}_legenda_links.csv", index=False)
    for metric, title in METRICS:
        table = valid.pivot(index="link", columns="model", values=metric)
        table = table.reindex(sorted(table.index)).reindex(columns=MODELS)
        x = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(max(10, len(table) * 1.15), 5.5), constrained_layout=True)
        width = .19
        for index, model in enumerate(MODELS):
            ax.bar(x + (index - 1.5) * width, table[model], width,
                   label=model, color=COLORS[index])
        ax.set_xticks(x, [labels[link] for link in table.index])
        ax.set_ylabel(title)
        ax.set_title(f"{dataset.upper()} | {title} por link (menor é melhor)")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.10), frameon=False)
        fig.savefig(OUT / f"{dataset}_{metric.lower()}_por_link.png", dpi=220)
        plt.close(fig)


def load_predictions(dataset, model):
    if model == MODELS[0]:
        paths = sorted((RESULTS / "forwarding_lstm" / dataset).glob("*_predictions.csv"))
        return pd.concat([pd.read_csv(path).assign(link=path.name.removesuffix("_predictions.csv"))
                          for path in paths], ignore_index=True)
    folder = "TimeMixerPP" if model == "TimeMixer++" else model
    return pd.read_csv(RESULTS / "benchmarks" / dataset / folder / "predictions.csv")


def plot_example(dataset, frame):
    counts = frame[frame.model == MODELS[0]].set_index("link")["observed_test_intervals"]
    link = counts.idxmax()
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for index, model in enumerate(MODELS):
        data = load_predictions(dataset, model)
        data = data[data.link == link].sort_values("Data")
        dates = pd.to_datetime(data.Data)
        ax.plot(dates, data.Prediction_Mbps, label=model, color=COLORS[index],
                alpha=.9, linewidth=1.3)
        if index == 0:
            observed = data.Observed.astype(str).str.lower().eq("true")
            ax.scatter(dates[observed], data.loc[observed, "Actual_Mbps"],
                       label="Observado", color="#20242a", s=22, zorder=5)
    ax.set_title(f"{dataset.upper()} | exemplo de previsão: {link}", fontsize=11)
    ax.set_ylabel("Vazão (Mbit/s)")
    ax.set_xlabel("Data")
    ax.grid(alpha=.22)
    ax.legend(ncol=3, fontsize=9)
    fig.autofmt_xdate()
    fig.savefig(OUT / f"{dataset}_exemplo_previsoes.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in ("geant", "sbrc"):
        frame = load_metrics(dataset)
        frames.append(frame)
        plot_overview(frame, dataset)
        plot_by_link(frame, dataset)
        plot_example(dataset, frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(OUT / "metricas_por_link.csv", index=False)
    summary = (all_metrics[all_metrics.observed_test_intervals > 0]
               .groupby(["dataset", "model"])[[name for name, _ in METRICS]]
               .mean().reset_index())
    summary.to_csv(OUT / "resumo_metricas.csv", index=False)
    print(f"Saved charts and tables in {OUT}")


if __name__ == "__main__":
    main()
