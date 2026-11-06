"""Four metric figures with all GEANT and RNP links; exclude Persistence and Ridge."""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from plot_metrics_by_link import short_link, MODELS as ALL_MODELS, NAMES as ALL_NAMES, COLORS as ALL_COLORS

ROOT = Path(__file__).resolve().parent / "results" / "controlled"
OUT = ROOT / "plots" / "metricas_selecionadas"
MODELS = [m for m in ALL_MODELS if m not in ("Persistence", "Ridge")]
NAMES = {m: n for m, n in zip(ALL_MODELS, ALL_NAMES)}
COLORS = {m: c for m, c in zip(ALL_MODELS, ALL_COLORS)}
METRICS = [("Cohen_Kappa", "Cohen’s kappa", "cohen_kappa", True),
           ("SMAPE", "SMAPE (%)", "smape", False),
           ("RMSE", "RMSE (Mbit/s)", "rmse", False),
           ("MAE", "MAE (Mbit/s)", "mae", False)]


def kappa_from_predictions(path):
    frame = pd.read_csv(path)
    observed = frame.Observed.astype(str).str.lower().eq("true")
    actual = frame.loc[observed, "Actual_Mbps"].to_numpy()
    predicted = frame.loc[observed, "Prediction_Mbps"].to_numpy()
    if not len(actual):
        return np.nan, 0
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"Nonfinite observed pair: {path}")
    # Classes are fixed a priori, never fitted to test values; extend 100-Mbit/s
    # classes beyond 0..2000 rather than discarding out-of-range pairs.
    truth_classes = np.floor(actual / 100).astype(np.int64)
    predicted_classes = np.floor(predicted / 100).astype(np.int64)
    if len(np.unique(np.r_[truth_classes, predicted_classes])) == 1:
        return np.nan, len(actual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(cohen_kappa_score(truth_classes, predicted_classes, weights=None))
    return value, len(actual)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "metrics_all.csv")
    raw = raw[raw.model.isin(MODELS)].copy()
    kappas, counts = [], []
    for row in raw.itertuples(index=False):
        path = ROOT / row.dataset / f"context_{row.context}" / row.model / row.link / f"seed_{row.seed}" / "predictions.csv"
        value, count = kappa_from_predictions(path)
        kappas.append(value)
        counts.append(count)
    raw["Cohen_Kappa"], raw["kappa_observed_pairs"] = kappas, counts
    raw.to_csv(OUT / "metricas_por_execucao.csv", index=False)
    metrics = [key for key, *_ in METRICS]
    means = raw.groupby(["dataset", "context", "link", "model"])[metrics].mean().reset_index()
    means.to_csv(OUT / "metricas_medias_todos_links.csv", index=False)
    plotted = means.copy()
    pd.DataFrame({"link": sorted(means.loc[means.dataset == "geant", "link"].unique()),
                  "retained": True, "criterion": "Todos os links; sem filtro de desempenho"}).to_csv(
                      OUT / "selecao_links_geant.csv", index=False)
    plotted.to_csv(OUT / "metricas_exibidas.csv", index=False)
    for metric, label, filename, higher in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(25, 12))
        fig.subplots_adjust(left=.055, right=.985, bottom=.20, top=.88, wspace=.14, hspace=.43)
        for row, (dataset, title) in enumerate((("geant", "GEANT"), ("sbrc", "RNP"))):
            part = plotted[plotted.dataset == dataset]
            links = sorted(part.link.unique())
            for col, context in enumerate((3, 28)):
                ax = axes[row, col]
                subtitle = f"{title}: todos os {len(links)} links"
                ax.set_title(f"{subtitle} | contexto {context}", fontsize=14, pad=12)
                table = part[part.context == context].pivot(index="link", columns="model", values=metric).reindex(index=links, columns=MODELS)
                x, width = np.arange(len(links)), .105
                for index, model in enumerate(MODELS):
                    ax.bar(x+(index-3.5)*width, table[model], width, label=NAMES[model],
                           color=COLORS[model], edgecolor="white", linewidth=.35)
                ax.set_xticks(x, [short_link(link, dataset) for link in links], fontsize=11)
                ax.set_ylabel(label, fontsize=12)
                ax.set_xlabel("Link", fontsize=11)
                ax.grid(axis="y", alpha=.22)
                ax.set_axisbelow(True)
                if higher:
                    ax.set_ylim(-1, 1.08)
                    ax.axhline(0, color="#777777", linewidth=.7)
                else:
                    ax.set_ylim(bottom=0)
                if dataset == "sbrc":
                    ax.text(.99, .97, "ES → PR: sem observações no teste", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color="#555555")
        fig.suptitle(f"{label} por link | {'maior é melhor' if higher else 'menor é melhor'}", fontsize=22, y=.975)
        fig.text(.5, .928, "GEANT: subconjunto selecionado por desempenho do Forwarding LSTM; não representa o conjunto completo.", ha="center", fontsize=13)
        fig.legend([Patch(facecolor=COLORS[m]) for m in MODELS], [NAMES[m] for m in MODELS],
                   loc="lower center", bbox_to_anchor=(.5, .078), ncol=4, frameon=False, fontsize=13)
        fig.text(.5, .053, "Filtro GEANT: ≥5 vitórias estritas nas 8 combinações (4 métricas × 2 contextos), após média das sementes. Empates não contam.", ha="center", fontsize=11)
        fig.text(.5, .027, "Kappa sem ponderação: classes fixas de 100 Mbit/s, estendidas além de 0–2000. Kappa indefinido é ausente; não equivale a zero.", ha="center", fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{filename}_por_link.{extension}", dpi=210)
        plt.close(fig)
    notes = {"excluded_models": ["Persistence", "Ridge"], "kappa": "unweighted; class=floor(Mbit/s / 100); all observed pairs; undefined values remain missing",
             "geant_filter": "none; all links included",
             "retained_geant_links": sorted(means.loc[means.dataset == "geant", "link"].unique()),
             "removed_geant_links": []}
    (OUT / "criterios.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved four metric figures: {OUT}")


if __name__ == "__main__":
    main()
