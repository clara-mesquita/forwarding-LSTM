"""Plot the 2026-11-09 RNP and GEANT grid evaluation exports."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/journal_2026-11-09"
MODELS = ("Forwarding LSTM", "LSTM", "GRU", "SARIMA", "Transformer")
NETWORKS = {"rnp": "RNP", "geant": "GÉANT"}
ERRORS = {"NRMSE": "NRMSE", "RMSE": "RMSE (Mbit/s)", "MAE": "MAE (Mbit/s)"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def draw_points(ax, tables, metric, label, panel):
    for index, table in enumerate(tables):
        values = np.array([row[metric] for row in table.values()])
        color = "#0072B2" if index == 0 else "#555555"
        ax.scatter(values, index + np.linspace(-.16, .16, len(values)),
                   s=17, facecolors="none", edgecolors=color, linewidths=.7)
        ax.scatter(np.median(values), index, s=27, marker="D", color=color,
                   edgecolors="white", linewidths=.45, zorder=3)
    ax.set(yticks=range(len(MODELS)), yticklabels=MODELS, ylim=(4.5, -.5), xlabel=label)
    ax.set_title(panel, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "ytick.major.size": 0, "savefig.facecolor": "white"})
    data = {}
    labels = {}
    with (OUT / "plotted_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["network", "link_code", "link", "model", *ERRORS,
                         "Execution_Time_s", "Fuzzy_Accuracy", "source"])
        for network, title in NETWORKS.items():
            tables = []
            for model in MODELS:
                source = (Path("results", network, "evaluation.json") if model == MODELS[0]
                          else Path("results2", network, f"evaluation_{model.lower()}.json"))
                records = json.loads((ROOT / source).read_text(encoding="utf-8"))
                table = {key.removesuffix("_" + model): row for key, row in records.items()}
                assert len(table) == len(records) == (6 if network == "rnp" else 10), source
                assert not tables or table.keys() == tables[0].keys(), source
                for row in table.values():
                    assert all(np.isfinite(row[m]) and row[m] >= 0 for m in ERRORS), source
                    assert np.isfinite(row["Execution_Time"]) and row["Execution_Time"] > 0, source
                tables.append(table)
                for index, link in enumerate(sorted(table), 1):
                    code = link.split("_bbr_")[0].upper() if network == "rnp" else f"G{index:02}"
                    labels[link] = code
                    row = table[link]
                    writer.writerow([title, code, link, model, *(row[m] for m in ERRORS),
                                     row["Execution_Time"], row["Fuzzy_Accuracy"], source.as_posix()])
            data[title] = tables

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.65), sharex="col", layout="constrained")
    for r, (title, tables) in enumerate(data.items()):
        for c, (metric, label) in enumerate(ERRORS.items()):
            panel = f"({chr(97 + 3*r + c)})"
            draw_points(axes[r, c], tables, metric, label, f"{panel}  {title}" if c == 0 else panel)
            if c:
                axes[r, c].tick_params(labelleft=False)
    save(fig, "figure_1_forecast_errors")

    matrices = {title: np.array([[table[link]["NRMSE"] for table in tables]
                                 for link in sorted(tables[0])]) for title, tables in data.items()}
    norm = Normalize(min(m.min() for m in matrices.values()), max(m.max() for m in matrices.values()))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
    for index, (ax, (title, matrix)) in enumerate(zip(axes, matrices.items())):
        heat = ax.imshow(matrix, cmap="cividis", norm=norm, aspect="auto")
        ax.set_xticks(range(len(MODELS)), ["Forwarding\nLSTM", *MODELS[1:]], fontsize=7)
        ax.set_yticks(range(len(matrix)), [labels[link] for link in sorted(data[title][0])])
        ax.set_title(f"({chr(97 + index)})  {title}", loc="left", fontweight="bold")
        ax.tick_params(axis="x", length=0)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm(value) < .52 else "black")
    fig.colorbar(heat, ax=axes, label="NRMSE", shrink=.86)
    save(fig, "figure_2_nrmse_by_link")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), sharex=True, layout="constrained")
    for index, (ax, (title, tables)) in enumerate(zip(axes, data.items())):
        ax.set_xscale("log")
        draw_points(ax, tables, "Execution_Time", "Wall-clock runtime (s)", f"({chr(97 + index)})  {title}")
        ax.set_xticks([30, 60, 120, 240, 480], ["30", "60", "120", "240", "480"])
        ax.minorticks_off()
        ax.set_xlim(25, 660)
        if index:
            ax.tick_params(labelleft=False)
    save(fig, "figure_3_experiment_runtime")


if __name__ == "__main__":
    main()
