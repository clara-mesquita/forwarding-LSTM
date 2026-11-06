"""Reproduce the supplied Matplotlib template styling with current metrics."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_metrics_by_link import short_link

ROOT = Path(__file__).resolve().parent / "results" / "controlled" / "plots"
SOURCE = ROOT / "five_models_journal" / "metrics.csv"
TEMPLATES = ROOT / "template"
OUT = ROOT / "five_models_template_style"
MODELS = ["SARIMA", "GRU", "LSTM", "ForwardingLSTM", "Transformer"]
NAMES = ["SARIMA", "GRU", "LSTM", "Forwarding LSTM", "Transformer"]
# Exact solid-fill RGB values sampled from the two supplied PNG templates.
ERROR_COLORS = ["#047016", "#ff8800", "#3c1cec", "#e874f3", "#00a6a6"]
KAPPA_COLORS = ["#2dc653", "#d06500", "#3a86ff", "#ffbe0b", "#9b5de5"]
METRICS = [("Cohen_Kappa", "COHEN KAPPA", "cohen_kappa", "COHEN KAPPA"),
           ("SMAPE", "SMAPE", "smape", "SMAPE (%)"),
           ("RMSE", "RMSE", "rmse", "RMSE (Mbit/s)"),
           ("MAE", "MAE", "mae", "MAE (Mbit/s)")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    data = pd.read_csv(SOURCE)
    assert set(data.model) == set(MODELS)
    assert data.groupby("dataset").link.nunique().to_dict() == {"geant": 6, "sbrc": 5}
    data.to_csv(OUT / "metrics.csv", index=False)
    files = []
    captions = ["# Figures styled after the supplied templates", "",
                "The MAE reference supplies the error-figure palette; the Cohen Kappa reference supplies the kappa palette. The first four colors were sampled exactly from the source images. A complementary fifth color was added for Transformer, which is absent from the templates.", "",
                "The references' horizontal grouping, upward category ordering, black frame, dashed vertical grid, sans-serif type, internal legend and English title pattern are reproduced. Actual current data and model names are retained. The kappa reference omits category text; the recreated figures include link labels for identification.", "",
                "All figures show SARIMA, GRU, LSTM, Forwarding LSTM and Transformer. GEANT omits Ankara–Lisboa, LUT–Viena, Lisboa–RESTENA and Anella–Poznań as requested. RNP omits ES–PR. Five stochastic seeds are averaged per link; SARIMA is deterministic. Error metrics use observed test intervals only. Cohen's kappa is unweighted with 100-Mbit/s classes extended beyond 0–2000; × denotes an undefined value, not zero.", "",
                "Context 3 means 18 hours of history; context 28 means 7 days. Forecast horizon is 6 hours. SARIMA uses its full fitted historical state. Context is identified by filenames/captions, not an extra subtitle. Each dataset/metric uses matching axis limits across contexts.", ""]
    for dataset, title, prefix in (("geant", "GEANT", "geant"), ("sbrc", "RNP", "rnp")):
        group = data[data.dataset == dataset]
        links = sorted(group.link.unique())
        for metric, heading, filename, xlabel in METRICS:
            colors = KAPPA_COLORS if metric == "Cohen_Kappa" else ERROR_COLORS
            if metric == "Cohen_Kappa":
                values = group[metric].dropna()
                left = min(0., float(values.min()))
                right = max(0., float(values.max()))
                span = max(right-left, .2)
                limits = (left-.08*span, right+.08*span)
            else:
                limits = (0, float(group[metric].max())*1.06)
            for context in (3, 28):
                table = group[group.context == context].pivot(index="link", columns="model", values=metric).reindex(index=links, columns=MODELS)
                fig, ax = plt.subplots(figsize=(6, 5))
                positions, width = np.arange(len(links)), .16
                for index, (model, name, color) in enumerate(zip(MODELS, NAMES, colors)):
                    y = positions+(index-2)*width
                    ax.barh(y, table[model], height=width, label=name, color=color, edgecolor="none")
                    missing = table[model].isna().to_numpy()
                    if missing.any():
                        ax.plot(np.full(missing.sum(), .012), y[missing], "x", transform=ax.get_yaxis_transform(),
                                color="#555555", markersize=3, markeredgewidth=.7)
                labels = [short_link(link, dataset).replace("\n", " ") if dataset == "geant"
                          else link.split("_bbr_")[0].upper() for link in links]
                ax.set_yticks(positions, labels)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Communication Point")
                ax.set_title(f"{heading} ({title}) Comparison by Communication Point", pad=6)
                ax.set_xlim(*limits)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
                ax.grid(axis="x", linestyle="--", color="#b0b0b0", alpha=.5, linewidth=.8)
                ax.set_axisbelow(False)
                ax.tick_params(direction="out", length=3.5, width=.8)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(.8)
                ax.legend(loc="best",
                          frameon=False, handlelength=2, borderpad=.3, labelspacing=.45)
                fig.tight_layout(pad=.65)
                base = f"{prefix}_context_{context}_{filename}"
                for extension in ("png", "pdf", "svg"):
                    path = OUT / f"{base}.{extension}"
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.03, facecolor="white")
                    files.append(path.name)
                plt.close(fig)
                captions.extend([f"## {base}", "", f"{heading} for {title}, context {context}; one-step online prediction, mean across seeds. See the shared methodology and exclusions above.", ""])
    (OUT / "CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    manifest = dict(models=MODELS, datasets={"geant": 6, "rnp": 5}, contexts=[3, 28],
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    templates={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in TEMPLATES.glob("*.png")},
                    error_palette=ERROR_COLORS, kappa_palette=KAPPA_COLORS,
                    fonts_pt=dict(title=12, axis_label=12, tick_label=11, legend=11),
                    canvas_inches=[6, 5], png_dpi=300, files=files)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved 16 figures in PNG/PDF/SVG: {OUT}")


if __name__ == "__main__":
    main()
