"""Minimal horizontal-bar figures; captions live outside the figure artwork."""
import json
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_metrics_by_link import short_link

ROOT = Path(__file__).resolve().parent / "results" / "controlled" / "plots"
SOURCE = ROOT / "five_models_single_plots_vertical" / "metrics.csv"
OUT = ROOT / "five_models_journal"
MODELS = ["SARIMA", "GRU", "LSTM", "ForwardingLSTM", "Transformer"]
NAMES = ["SARIMA", "GRU", "LSTM", "Forwarding LSTM", "Transformer"]
# Grey plus colors from the Okabe–Ito palette.
COLORS = ["#888888", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]
METRICS = [("Cohen_Kappa", "Cohen’s κ", "cohen_kappa"),
           ("SMAPE", "SMAPE (%)", "smape"),
           ("RMSE", "RMSE (Mbit/s)", "rmse"),
           ("MAE", "MAE (Mbit/s)", "mae")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 13, "xtick.labelsize": 11,
                         "ytick.labelsize": 12, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": True})
    data = pd.read_csv(SOURCE)
    excluded_rnp = sorted(data.loc[(data.dataset == "sbrc") &
                                  data.link.str.startswith("es-pr_bbr_"), "link"].unique())
    if len(excluded_rnp) != 1:
        raise ValueError("Expected exactly one ES–PR link to exclude")
    data = data[~((data.dataset == "sbrc") & data.link.isin(excluded_rnp))].copy()
    if set(data.model) != set(MODELS):
        raise ValueError("Unexpected model selection")
    if data.groupby("dataset").link.nunique().to_dict() != {"geant": 6, "sbrc": 5}:
        raise ValueError("Expected the requested six GEANT and five RNP links")
    data.to_csv(OUT / "metrics.csv", index=False)
    captions = ["# Figure captions", "",
                "Each file is a standalone figure with one metric, dataset and context. No titles, subtitles or footnotes are embedded in the artwork.", "",
                "Figures use a compact 7.2 × 7.4-inch canvas, cropped to artwork with minimal padding; PDF/SVG are vector, PNG is 600 dpi. Axis labels are 13 pt, link labels 12 pt, numeric ticks 11 pt and legend 10.5 pt before resizing.", "",
                "Values are means across stochastic seeds per link, calculated on observed test intervals. These are descriptive means, without confidence intervals. Smaller errors are better; larger kappa is better.", "",
                "Cohen’s kappa is unweighted with fixed classes floor(throughput in Mbit/s / 100), extended below zero and above 2000 Mbit/s. A × at the left edge denotes an undefined value, not zero. ES–PR is omitted from the RNP figures at the user's request; it has no observed test values.", "",
                "The displayed GEANT links exclude the four user-selected directions: Ankara → Lisboa, LUT → Viena, Lisboa → RESTENA and Anella → Poznań. This is a selected subset, not the full GEANT evaluation.", ""]
    files = []
    for dataset, title, prefix in (("geant", "GEANT", "geant"), ("sbrc", "RNP", "rnp")):
        group = data[data.dataset == dataset]
        links = sorted(group.link.unique())
        for metric, label, filename in METRICS:
            maximum = group[metric].max()
            for context in (3, 28):
                part = group[group.context == context]
                table = part.pivot(index="link", columns="model", values=metric).reindex(index=links, columns=MODELS)
                fig, ax = plt.subplots(figsize=(7.2, 7.4 if dataset == "geant" else 6.4))
                fig.subplots_adjust(left=.27, right=.99, bottom=.18, top=.995)
                positions = np.arange(len(links))
                height = .17
                for index, (model, name, color) in enumerate(zip(MODELS, NAMES, COLORS)):
                    y = positions+(index-2)*height
                    ax.barh(y, table[model], height=height*.96, label=name, color=color,
                            edgecolor="none", zorder=3)
                    missing = table[model].isna().to_numpy()
                    if missing.any():
                        # x is in axes coordinates, y in data coordinates: the
                        # symbol does not represent a measured numerical value.
                        ax.plot(np.full(missing.sum(), .012), y[missing], "x",
                                transform=ax.get_yaxis_transform(), color="#777777",
                                markersize=4, markeredgewidth=.8, clip_on=False, zorder=4)
                ax.set_yticks(positions, [short_link(link, dataset).replace("\n", " ") for link in links])
                ax.set_ylim(len(links)-.44, -.44)
                ax.set_xlabel(label, labelpad=5)
                ax.tick_params(axis="both", length=0, pad=4)
                for side in ("top", "right", "left"):
                    ax.spines[side].set_visible(False)
                ax.spines["bottom"].set_color("#bbbbbb")
                ax.spines["bottom"].set_linewidth(.5)
                ax.grid(axis="x", color="#e5e5e5", linewidth=.4, zorder=0)
                ax.set_axisbelow(True)
                if metric == "Cohen_Kappa":
                    ax.set_xlim(-1, 1)
                    ax.set_xticks([-1, -.5, 0, .5, 1])
                    ax.axvline(0, color="#999999", linewidth=.5, zorder=2)
                else:
                    ax.set_xlim(0, maximum*1.07 if maximum > 0 else 1)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                           bbox_to_anchor=(.52, .002), ncol=3, frameon=False, fontsize=10.5,
                           handlelength=1.1, handletextpad=.4, columnspacing=1.1,
                           labelspacing=.4, borderaxespad=0)
                base = f"{prefix}_context_{context}_{filename}"
                for extension in ("png", "pdf", "svg"):
                    path = OUT / f"{base}.{extension}"
                    fig.savefig(path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=.035)
                    files.append(path.name)
                plt.close(fig)
                captions.extend([f"## {base}", "",
                                 f"{label} by link for {title}, using a {context}-interval history ({18 if context == 3 else 168} h) to predict the next 6-h interval. Bars indicate means across seeds. SARIMA uses its full fitted history irrespective of context. See the general notes above for omitted GEANT directions and undefined values.", ""])
    (OUT / "CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    manifest = dict(models=MODELS, contexts=[3, 28], datasets={"geant": 6, "rnp": 5},
                    excluded_rnp_links=excluded_rnp,
                    canvas_inches_by_dataset={"geant": [7.2, 7.4], "rnp": [7.2, 6.4]},
                    orientation="horizontal bars, compact portrait canvas", size_inches=[7.2, 7.4],
                    fonts_pt=dict(axis_label=13, link_label=12, numeric_tick=11, legend=10.5),
                    cropped_to_artwork=True,
                    png_dpi=600, files=files, source=str(SOURCE),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved 16 figures as PNG/PDF/SVG: {OUT}")


if __name__ == "__main__":
    main()
