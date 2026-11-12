"""Generate plots for completed controlled experiments, averaging seeds per link first."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent/"results"/"controlled"
OUT = ROOT/"plots"
NAMES = {"ForwardingLSTM": "Forwarding LSTM", "RandomForest": "Random Forest",
         "Persistence": "Persistência", "Ridge": "Regressão Ridge", "Transformer": "Transformer padrão"}
ORDER = ["Persistence", "Ridge", "RandomForest", "MLP", "LSTM", "GRU", "ARIMA", "SARIMA", "Transformer", "ForwardingLSTM"]
METRICS = [("RMSE", "RMSE (Mbit/s)"), ("MAE", "MAE (Mbit/s)"),
           ("NRMSE_mean", "NRMSE"), ("SMAPE", "SMAPE (%)")]


def plot_grouped_links(part, dataset, context, out):
    models = [model for model in ORDER if model in set(part.model)]
    labels = sorted(part.link.unique())
    x = np.arange(len(labels))
    width = .85 / len(models)
    for metric, label in METRICS:
        table = part.pivot(index="link", columns="model", values=metric).reindex(index=labels, columns=models)
        fig, ax = plt.subplots(figsize=(max(14, len(labels)*1.8), 6), constrained_layout=True)
        for index, model in enumerate(models):
            ax.bar(x + (index-(len(models)-1)/2)*width, table[model], width,
                   label=NAMES.get(model, model))
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{dataset.upper()} | context {context} | lower is better\nMean across seeds for each link")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=8)
        for suffix in ("png", "pdf"):
            fig.savefig(out/f"{dataset}_contexto_{context}_{metric.lower()}_grouped_links.{suffix}", dpi=220)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT/"metrics_all.csv")
    valid = data[data.observed_test_intervals > 0]
    columns = [m for m, _ in METRICS]
    links = valid.groupby(["dataset", "context", "model", "link"])[columns].mean().reset_index()
    summary = links.groupby(["dataset", "context", "model"])[columns].mean().reset_index()
    summary.to_csv(OUT/"medias_entre_links.csv", index=False)
    valid.groupby(["dataset", "context", "model", "link"])[columns].agg(["mean", "std", "count"]).to_csv(OUT/"variacao_sementes_por_link.csv")
    seed_macro = valid.groupby(["dataset", "context", "model", "seed"])[columns].mean()
    seed_macro.to_csv(OUT/"medias_por_semente.csv")
    seed_macro.groupby(["dataset", "context", "model"])[columns].agg(["mean", "std", "count"]).to_csv(OUT/"variacao_sementes_media_links.csv")
    lines = ["# Resultados da comparação controlada", "",
             "Cada linha abaixo é a média entre links avaliáveis, após a média das sementes por link.",
             "GEANT: dez links. SBRC: cinco links avaliáveis; es-pr recebe previsões, mas não tem observações no teste.",
             "Modelos estocásticos usam cinco sementes; os determinísticos, uma execução.", "",
             "| Conjunto | Contexto | Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | NRMSE | SMAPE (%) |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset.upper()} | {row.context} | {NAMES.get(row.model, row.model)} | {row.RMSE:.3f} | {row.MAE:.3f} | {row.NRMSE_mean:.4f} | {row.SMAPE:.3f} |")
    lines.extend(["", "São execuções únicas de treinamento por semente; diferenças pequenas precisam ser interpretadas com a variação entre sementes e links.",
                  "ARIMA/SARIMA usam histórico completo, independentemente da coluna contexto.",
                  "LSTM/GRU/Forwarding LSTM são portas PyTorch com validação comum; os resultados antigos ficam em pastas separadas.",
                  "A auditoria dos artefatos está em audit.json. As figuras estão em plots/."])
    (ROOT/"RESULTADOS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for (dataset, context), group in summary.groupby(["dataset", "context"]):
        table = group.set_index("model").reindex(ORDER)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        colors = ["#176b87" if m == "ForwardingLSTM" else "#a3bdca" for m in ORDER]
        for ax, (metric, label) in zip(axes.flat, METRICS):
            bars = ax.barh([NAMES.get(m, m) for m in ORDER], table[metric], color=colors)
            ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.set_xlim(0, table[metric].max()*1.2)
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(f"{dataset.upper()} | contexto de {context} intervalos | menor é melhor\nMédia por link, após média das sementes", fontsize=14)
        fig.savefig(OUT/f"{dataset}_contexto_{context}_comparacao.png", dpi=220)
        fig.savefig(OUT/f"{dataset}_contexto_{context}_comparacao.pdf")
        plt.close(fig)

        part = links[(links.dataset == dataset) & (links.context == context)]
        plot_grouped_links(part, dataset, context, OUT)
        pivot = part.pivot(index="link", columns="model", values="NRMSE_mean").reindex(columns=ORDER)
        labels = ([link.split("_bbr_")[0].upper() for link in pivot.index] if dataset == "sbrc"
                  else [f"G{i:02d}" for i in range(1, len(pivot)+1)])
        pd.DataFrame({"codigo": labels, "link": pivot.index}).to_csv(OUT/f"{dataset}_legenda_links.csv", index=False)
        fig, ax = plt.subplots(figsize=(13, max(4, len(pivot)*.55)), constrained_layout=True)
        heat = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(ORDER)), [NAMES.get(m, m) for m in ORDER], rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for row in range(len(pivot)):
            for col in range(len(ORDER)):
                ax.text(col, row, f"{pivot.iloc[row, col]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if pivot.iloc[row, col] > pivot.max().max()*.55 else "black")
        ax.set_title(f"{dataset.upper()} | NRMSE por link | contexto {context}")
        fig.colorbar(heat, ax=ax, label="NRMSE (menor é melhor)")
        fig.savefig(OUT/f"{dataset}_contexto_{context}_nrmse_links.png", dpi=220)
        plt.close(fig)
    print(f"Charts saved: {OUT}")


if __name__ == "__main__":
    main()
