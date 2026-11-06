"""Plot recorded wall-clock times; no additional training or timing runs."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_metrics_by_link import MODELS, NAMES, COLORS, short_link

ROOT = Path(__file__).resolve().parent / "results" / "controlled"
OUT = ROOT / "plots"
FIELDS = [("selection_seconds", "Seleção de parâmetros (s)"),
          ("training_seconds", "Treinamento final (s)"),
          ("inference_seconds", "Predição de todo o teste (s)"),
          ("inference_ms_per_interval", "Predição por intervalo (ms)")]


def load_times():
    rows = []
    for path in sorted(ROOT.glob("*/context_*/*/*/seed_*/run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(path.parent / "predictions.csv", usecols=["Observed"])
        row = {key: record[key] for key in ("dataset", "context", "model", "link", "seed",
                                           "selection_seconds", "training_seconds", "inference_seconds")}
        if len(predictions) == 0 or any(not np.isfinite(row[key]) or row[key] < 0
                                       for key in ("selection_seconds", "training_seconds", "inference_seconds")):
            raise ValueError(f"Invalid timing data: {path}")
        row["test_intervals"] = len(predictions)
        row["observed_test_intervals"] = int(predictions.Observed.astype(str).str.lower().eq("true").sum())
        # Includes missing intervals: the model still issues a prediction for each.
        row["inference_ms_per_interval"] = row["inference_seconds"] * 1000 / len(predictions)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No saved execution times found")
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_times()
    raw.to_csv(OUT / "tempos_por_execucao.csv", index=False)
    keys = ["dataset", "context", "model", "link"]
    means = raw.groupby(keys)[[key for key, _ in FIELDS]].mean()
    # Selection is shared by the final seeds. Use the base seed's measurement
    # once, rather than adding the repeated metadata fields across seeds.
    means["selection_seconds"] = raw.sort_values("seed").groupby(keys).selection_seconds.first()
    means = means.reset_index()
    means.loc[means.model == "Persistence", ["selection_seconds", "training_seconds"]] = np.nan
    means.to_csv(OUT / "tempos_medios_por_link.csv", index=False)
    means.groupby(["dataset", "context", "model"])[[key for key, _ in FIELDS]].mean().to_csv(OUT / "tempos_resumo_por_modelo.csv")

    for dataset, title, filename in (("geant", "GEANT", "geant"), ("sbrc", "RNP", "rnp")):
        group = means[means.dataset == dataset]
        links = sorted(group.link.unique())
        fig, axes = plt.subplots(4, 2, figsize=(28, 19), sharex="col", sharey="row")
        fig.subplots_adjust(left=.06, right=.985, top=.90, bottom=.15, hspace=.27, wspace=.10)
        x, width = np.arange(len(links)), .085
        for row, (field, label) in enumerate(FIELDS):
            positive = group.loc[group[field] > 0, field]
            lower = 10 ** math.floor(math.log10(positive.min())-.1)
            upper = 10 ** math.ceil(math.log10(positive.max())+.1)
            for col, context in enumerate((3, 28)):
                ax = axes[row, col]
                part = group[group.context == context]
                table = part.pivot(index="link", columns="model", values=field).reindex(index=links, columns=MODELS)
                for index, model in enumerate(MODELS):
                    ax.bar(x + (index-4.5)*width, table[model], width,
                           color=COLORS[index], edgecolor="white", linewidth=.3)
                ax.set_yscale("log")
                ax.set_ylim(lower, upper)
                ax.set_xlim(-.6, len(links)-.4)
                if col == 0:
                    ax.set_ylabel(label + "\n(escala logarítmica)", fontsize=12)
                ax.grid(axis="y", alpha=.22, which="major")
                ax.set_axisbelow(True)
                ax.set_xticks(x, [short_link(link, dataset) for link in links], fontsize=11)
                ax.tick_params(axis="y", labelsize=10)
                if row == 0:
                    ax.set_title(f"Contexto {context}: {'18 horas' if context == 3 else '7 dias'} de histórico", fontsize=16, pad=12)
                if row == 3:
                    ax.set_xlabel("Link", fontsize=12, labelpad=10)
        fig.suptitle(f"{title} | tempos de execução por link", fontsize=23, y=.975)
        fig.text(.5, .943, "Tempos registrados nas execuções • treinamento/predição: média das sementes • menor é melhor", ha="center", fontsize=14)
        fig.legend([Patch(facecolor=color) for color in COLORS], NAMES,
                   loc="lower center", bbox_to_anchor=(.5, .067), ncol=5,
                   frameon=False, fontsize=13, columnspacing=2)
        fig.text(.5, .042, "Persistência não requer seleção nem treinamento. Seleção contada uma vez por link/modelo/contexto; predição inclui intervalos ausentes.", ha="center", fontsize=11)
        fig.text(.5, .023, "Medições históricas sob execução concorrente, sem isolamento de desempenho; buscas e épocas variam entre métodos.", ha="center", fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{filename}_tempos_colunas_por_link.{extension}", dpi=200)
        plt.close(fig)

    (OUT / "TEMPOS.md").write_text(
        "# Tempos de execução\n\n"
        "Fonte: selection_seconds, training_seconds e inference_seconds dos run.json.\n\n"
        "- Seleção: busca de configuração, treinamento de candidatos e validação; contada uma vez pelo registro da semente base.\n"
        "- Treinamento: ajuste final após a seleção, incluindo construção do modelo/otimizador.\n"
        "- Predição total: percurso online completo do teste, com realimentação, conversão de escala e atualizações de estado.\n"
        "- Predição por intervalo: tempo total dividido por todos os intervalos previstos, inclusive os ausentes (48 em GEANT e 157 em RNP). É uma média do percurso, não uma medição isolada de latência.\n\n"
        "Treinamento e predição são médias entre sementes por link. Persistência não possui treinamento ou seleção; seus pequenos tempos de preparação registrados não são apresentados como treinamento.\n\n"
        "As execuções originais ocorreram em partições concorrentes na CPU; os valores representam tempos observados nessas condições, não um ensaio de desempenho com hardware isolado. Diferentes buscas, épocas selecionadas e tentativas de otimização também afetam os tempos.\n\n"
        "Não somar selection_seconds dos registros de todas as sementes: o mesmo tempo de seleção foi repetido nos metadados. Tempos de gravação dos arquivos e métricas não estão incluídos.\n",
        encoding="utf-8")
    print(f"Saved timing charts and tables: {OUT}")
    print(f"Loaded {len(raw)} execution records")


if __name__ == "__main__":
    main()
