"""Grouped metric columns by link: one GEANT figure and one RNP figure."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "results" / "controlled"
OUT = ROOT / "plots"
MODELS = ["Persistence", "Ridge", "RandomForest", "MLP", "LSTM", "GRU",
          "ARIMA", "SARIMA", "Transformer", "ForwardingLSTM"]
NAMES = ["Persistência", "Regressão Ridge", "Random Forest", "MLP", "LSTM",
         "GRU", "ARIMA", "SARIMA", "Transformer padrão", "Forwarding LSTM"]
METRICS = [("RMSE", "RMSE (Mbit/s)"), ("MAE", "MAE (Mbit/s)"),
           ("NRMSE_mean", "NRMSE"), ("SMAPE", "SMAPE (%)")]
COLORS = list(plt.get_cmap("tab10").colors)


def short_link(link, dataset):
    if dataset == "sbrc":
        return link.split("_bbr_")[0].upper().replace("-", " → ")
    endpoints = {"perfsonar-ankara.ulakbim.gov.tr": "Ankara",
                 "perfsonar-sonda.rediris.es": "RedIRIS",
                 "psmall.lut.ac.uk": "LUT",
                 "psmp-gn-bw-lis-pt.geant.org": "Lisboa",
                 "perfsonar.restena.lu": "RESTENA",
                 "psmp-gn-bw-poz-pl.geant.org": "Poznań",
                 "pspmp-anella.csuc.cat": "Anella",
                 "psmp-gn-bw-vie-at.geant.org": "Viena"}
    source, destination = link.removeprefix("esmond data ").split(" to ")
    destination = destination.split(" ")[0]
    return f"{endpoints.get(source, source)}\n→ {endpoints.get(destination, destination)}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "metrics_all.csv")
    # Each seed's metric already uses every observed test interval. Average seeds
    # within each link, retaining links with no observations as missing, not zero.
    means = data.groupby(["dataset", "context", "model", "link"])[[m for m, _ in METRICS]].mean().reset_index()
    for dataset, title, filename in (("geant", "GEANT", "geant"), ("sbrc", "RNP", "rnp")):
        group = means[means.dataset == dataset]
        links = sorted(group.link.unique())
        fig, axes = plt.subplots(4, 2, figsize=(28, 19), sharex="col")
        fig.subplots_adjust(left=.055, right=.985, top=.90, bottom=.14, hspace=.28, wspace=.14)
        x = np.arange(len(links))
        width = .085
        for col, context in enumerate((3, 28)):
            part = group[group.context == context]
            for row, (metric, label) in enumerate(METRICS):
                ax = axes[row, col]
                table = part.pivot(index="link", columns="model", values=metric).reindex(index=links, columns=MODELS)
                for index, model in enumerate(MODELS):
                    ax.bar(x + (index - 4.5)*width, table[model], width,
                           color=COLORS[index], linewidth=.35, edgecolor="white")
                ax.set_ylabel(label, fontsize=12)
                ax.grid(axis="y", alpha=.22)
                ax.set_axisbelow(True)
                ax.set_ylim(bottom=0)
                ax.set_xlim(-.6, len(links)-.4)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xticks(x, [short_link(link, dataset) for link in links], fontsize=11)
                if row == 0:
                    ax.set_title(f"Contexto {context}: {'18 horas' if context == 3 else '7 dias'} de histórico", fontsize=16, pad=12)
                if row == 3:
                    ax.set_xlabel("Link", fontsize=12, labelpad=10)
        fig.suptitle(f"{title} | comparação das métricas por link", fontsize=23, y=.975)
        fig.text(.5, .943, "Todos os intervalos observados do teste • média das sementes por link • menor é melhor", ha="center", fontsize=14)
        fig.legend([Patch(facecolor=color) for color in COLORS], NAMES,
                   loc="lower center", bbox_to_anchor=(.5, .053), ncol=5,
                   frameon=False, fontsize=13, columnspacing=2)
        note = "Previsão de um intervalo de seis horas. ARIMA e SARIMA utilizam o histórico completo de treinamento."
        if dataset == "sbrc":
            note += " ES → PR: sem observações no teste; métricas ausentes."
        fig.text(.5, .024, note, ha="center", fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{filename}_metricas_colunas_por_link.{extension}", dpi=200)
        plt.close(fig)
        group.to_csv(OUT / f"{filename}_metricas_colunas_por_link.csv", index=False)
        print(OUT / f"{filename}_metricas_colunas_por_link.png")


if __name__ == "__main__":
    main()
