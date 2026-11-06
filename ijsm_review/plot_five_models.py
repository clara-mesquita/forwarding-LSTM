"""One chart per image: four metrics, two datasets, two history lengths."""
import hashlib
import json
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_metrics_by_link import short_link

ROOT = Path(__file__).resolve().parent / "results" / "controlled" / "plots"
SOURCE = ROOT / "metricas_selecionadas" / "metricas_medias_todos_links.csv"
OUT = ROOT / "five_models_single_plots"
MODELS = ["SARIMA", "GRU", "LSTM", "ForwardingLSTM", "Transformer"]
NAMES = ["SARIMA", "GRU", "LSTM", "Forwarding LSTM", "Transformer"]
COLORS = ["#7f7f7f", "#8c564b", "#9467bd", "#17becf", "#bcbd22"]
REQUESTED_EXCLUSIONS = {"Ankara\n→ Lisboa", "LUT\n→ Viena",
                        "Lisboa\n→ RESTENA", "Anella\n→ Poznań"}
METRICS = [("Cohen_Kappa", "Cohen’s kappa", "cohen_kappa", True),
           ("SMAPE", "SMAPE (%)", "smape", False),
           ("RMSE", "RMSE (Mbit/s)", "rmse", False),
           ("MAE", "MAE (Mbit/s)", "mae", False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical", action="store_true", help="Create portrait images")
    parser.add_argument("--exclude-requested-links", action="store_true",
                        help="Exclude the four GEANT directions specified by the user")
    args = parser.parse_args()
    global OUT
    if args.vertical:
        OUT = ROOT / "five_models_single_plots_vertical"
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    data = data[data.model.isin(MODELS)].copy()
    excluded_links = []
    if args.exclude_requested_links:
        excluded_links = sorted(link for link in data.loc[data.dataset == "geant", "link"].unique()
                                if short_link(link, "geant") in REQUESTED_EXCLUSIONS)
        if len(excluded_links) != 4:
            raise ValueError(f"Expected four requested GEANT links, found {excluded_links}")
        data = data[~((data.dataset == "geant") & data.link.isin(excluded_links))].copy()
    data.to_csv(OUT / "metrics.csv", index=False)
    files = []
    for dataset, title, prefix in (("geant", "GEANT", "geant"), ("sbrc", "RNP", "rnp")):
        group = data[data.dataset == dataset]
        links = sorted(group.link.unique())
        expected = 10-len(excluded_links) if dataset == "geant" else 6
        if len(links) != expected or len(group) != expected * 2 * len(MODELS):
            raise ValueError(f"Incomplete metrics: {dataset}")
        for metric, label, filename, higher in METRICS:
            maximum = group[metric].max()
            for context in (3, 28):
                part = group[group.context == context]
                table = part.pivot(index="link", columns="model", values=metric).reindex(index=links, columns=MODELS)
                fig, ax = plt.subplots(figsize=(9, 12.5) if args.vertical else (15, 6.8))
                fig.subplots_adjust(left=.105 if args.vertical else .075,
                                    right=.98 if args.vertical else .985,
                                    top=.86 if args.vertical else .84,
                                    bottom=.29 if args.vertical else .28)
                x, width = np.arange(len(links)), .16
                for index, (model, name, color) in enumerate(zip(MODELS, NAMES, COLORS)):
                    ax.bar(x + (index - 2)*width, table[model], width,
                           label=name, color=color, edgecolor="white", linewidth=.5)
                ax.set_xticks(x, [short_link(link, dataset) for link in links],
                              fontsize=8 if args.vertical else 10,
                              rotation=40 if args.vertical else 0,
                              ha="right" if args.vertical else "center")
                ax.set_ylabel(label, fontsize=12)
                ax.set_xlabel("Link", fontsize=12, labelpad=10)
                ax.grid(axis="y", alpha=.22)
                ax.set_axisbelow(True)
                ax.set_xlim(-.6, len(links)-.4)
                if higher:
                    ax.set_ylim(-1, 1.05)
                    ax.axhline(0, linewidth=.6, color="#555555")
                else:
                    ax.set_ylim(0, maximum * 1.12 if maximum > 0 else 1)
                fig.suptitle(f"{title} | {label} por link", y=.96 if args.vertical else .97,
                             fontsize=18 if args.vertical else 19)
                subtitle = f"Contexto {context} ({'18 horas' if context == 3 else '7 dias'}) • média das sementes • {'maior é melhor' if higher else 'menor é melhor'}"
                fig.text(.5, .90, textwrap.fill(subtitle, 65) if args.vertical else subtitle,
                         ha="center", fontsize=11 if args.vertical else 12)
                fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3 if args.vertical else 5,
                           frameon=False, bbox_to_anchor=(.5, .09), fontsize=10 if args.vertical else 11)
                if higher:
                    note = "Kappa sem ponderação; classes fixas de 100 Mbit/s. Valores indefinidos são ausentes."
                else:
                    note = "Métricas calculadas sobre todos os intervalos observados do teste."
                if dataset == "sbrc":
                    note += " ES → PR: sem observações no teste."
                elif excluded_links:
                    note += " GEANT: 6 links exibidos, conforme seleção solicitada."
                fig.text(.5, .035 if args.vertical else .045,
                         textwrap.fill(note, 95) if args.vertical else note, ha="center", fontsize=9)
                base = f"{prefix}_context_{context}_{filename}"
                for extension in ("png", "pdf"):
                    path = OUT / f"{base}.{extension}"
                    fig.savefig(path, dpi=220)
                    files.append(path.name)
                plt.close(fig)
    manifest = dict(models=MODELS, datasets={"geant": 10-len(excluded_links), "rnp": 6}, contexts=[3, 28],
                    excluded_geant_links=excluded_links,
                    metrics=[key for key, *_ in METRICS], source=str(SOURCE),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    files=files)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Single plots: five models\n\n"
        "16 PNG images and 16 PDF versions. Each image contains exactly one chart: one dataset, one context, one metric.\n\n"
        f"Models: SARIMA, GRU, LSTM, Forwarding LSTM, Transformer. Showing {10-len(excluded_links)} GEANT and 6 RNP links.\n\n"
        + ("User-requested GEANT exclusions: Ankara → Lisboa; LUT → Viena; Lisboa → RESTENA; Anella → Poznań.\n\n" if excluded_links else "") +
        "Metrics are averaged across seeds per link, using observed test intervals. Kappa is unweighted with fixed 100-Mbit/s classes extending beyond 0–2000; undefined values remain missing. ES–PR has no observed test values.\n\n"
        "For each dataset/metric, the two contexts use the same y-axis limits.\n",
        encoding="utf-8")
    print(f"Saved {len(files)//2} PNG charts and PDF versions: {OUT}")


if __name__ == "__main__":
    main()
