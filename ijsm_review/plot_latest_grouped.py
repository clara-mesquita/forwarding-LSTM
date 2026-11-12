"""Export one grouped-by-link journal figure per metric."""
import csv

import numpy as np
from matplotlib.patches import Patch

from plot_latest_journal import ERRORS, MODELS, NETWORKS, OUT, plt, save

COLORS = ("#0072B2", "#E69F00", "#009E73", "#777777", "#CC79A7")
METRICS = {**ERRORS, "Cohen_Kappa": "Cohen’s κ"}


def main():
    with (OUT / "plotted_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (OUT / "replayed_kappa.csv").open(encoding="utf-8", newline="") as stream:
        kappas = {(r["network"], r["link"], r["model"]): r["Cohen_Kappa"]
                  for r in csv.DictReader(stream)}
    assert len(rows) == len(kappas) == 80
    for row in rows:
        row["Cohen_Kappa"] = kappas[row["network"], row["link"], row["model"]]

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "ytick.major.size": 0})
    for metric, label in METRICS.items():
        kappa = metric == "Cohen_Kappa"
        values = [float(r[metric]) for r in rows if r[metric]]
        limits = (min(-.05, np.floor(20 * min(values)) / 20),
                  max(.05, np.ceil(20 * max(values)) / 20)) if kappa else (0, max(values) * 1.06)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.15), sharex=True)
        fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.92, wspace=.25)
        for index, (title, ax) in enumerate(zip(NETWORKS.values(), axes)):
            table = {(r["link_code"], r["model"]): r[metric] for r in rows if r["network"] == title}
            links = sorted({link for link, _ in table})
            y = np.arange(len(links))
            for i, (model, color) in enumerate(zip(MODELS, COLORS)):
                entries = [table[link, model] for link in links]
                positions = y + (i - 2) * .15
                valid = np.array([bool(value) for value in entries])
                widths = [float(value) for value in entries if value]
                ax.barh(positions[valid], widths, height=.135, color=color)
                if kappa:
                    zero = valid & np.array([value != "" and float(value) == 0 for value in entries])
                    ax.scatter(np.zeros(zero.sum()), positions[zero], color=color, s=6, zorder=4)
                    ax.plot(np.full((~valid).sum(), .97), positions[~valid], "x",
                            transform=ax.get_yaxis_transform(), color=color, markersize=3)
            ax.set(yticks=y, yticklabels=links, ylim=(len(links)-.5, -.5), xlabel=label)
            ax.set_title(f"({chr(97 + index)})  {title}", loc="left", fontweight="bold")
            ax.set_xlim(limits)
            ax.grid(axis="x", color="#dddddd", linewidth=.6)
            ax.set_axisbelow(True)
            if kappa:
                ax.axvline(0, color="#888888", linewidth=.6)
        fig.legend(handles=[Patch(color=color, label=model) for model, color in zip(MODELS, COLORS)],
                   loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .03))
        save(fig, f"grouped_by_link_{metric.lower()}")


if __name__ == "__main__":
    main()
