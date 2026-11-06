"""Build a Portuguese results report directly from the displayed figure data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_metrics_by_link import short_link

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW / "results" / "controlled"
OUT = ROOT / "plots" / "five_models_template_style"
SOURCE = OUT / "metrics.csv"
MODELS = ["SARIMA", "GRU", "LSTM", "ForwardingLSTM", "Transformer"]
NAMES = dict(zip(MODELS, ["SARIMA", "GRU", "LSTM", "Forwarding LSTM", "Transformer"]))
METRICS = ["RMSE", "MAE", "SMAPE", "Cohen_Kappa"]
LABELS = {"RMSE": "RMSE (Mbit/s)", "MAE": "MAE (Mbit/s)",
          "SMAPE": "SMAPE (%)", "Cohen_Kappa": "Cohen’s kappa"}


def number(value):
    return "ND" if pd.isna(value) else format(float(value), ".17g")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows], ""]


def name(link, dataset):
    return short_link(link, dataset).replace("\n", " ")


def main():
    data = pd.read_csv(SOURCE, float_precision="round_trip")
    raw = pd.read_csv(ROOT / "plots" / "metricas_selecionadas" / "metricas_por_execucao.csv", float_precision="round_trip")
    keys = ["dataset", "context", "link", "model"]
    raw = raw.merge(data[keys], on=keys, how="inner", validate="many_to_one")
    assert len(data) == 110 and len(raw) == 462
    assert data.groupby("dataset").link.nunique().to_dict() == {"geant": 6, "sbrc": 5}
    recomputed = raw.groupby(keys)[METRICS].mean().sort_index()
    saved = data.set_index(keys)[METRICS].sort_index()
    np.testing.assert_allclose(recomputed.to_numpy(), saved.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
    counts = raw.groupby(keys).Cohen_Kappa.count()
    macro = data.groupby(["dataset", "context", "model"])[METRICS].mean()
    macro.to_csv(OUT / "medias_relatorio.csv", float_format="%.17g")
    lines = ["# Descrição dos resultados das figuras", "",
             "Este relatório descreve os valores efetivamente utilizados nas 16 figuras da pasta `five_models_template_style`: quatro métricas, dois conjuntos de dados e dois contextos. Cada figura compara SARIMA, GRU, LSTM, Forwarding LSTM e Transformer.", "",
             "As tabelas usam até **17 algarismos significativos**, preservando a representação numérica dos valores de ponto flutuante salvos no CSV. Essa quantidade de dígitos descreve a precisão do arquivo, não a incerteza experimental. `ND` significa não definido/ausente; nunca equivale a zero. O ponto é usado como separador decimal.", "",
             "## 1. Escopo dos dados e das execuções", "",
             "São apresentados **seis links GEANT e cinco links RNP**, em contextos de 3 e 28 intervalos. O conjunto RNP corresponde aos dados BBR salvos sob o identificador `sbrc`. Há **110 combinações de conjunto × contexto × link × modelo**, provenientes de **462 execuções individuais**: as quatro redes usam cinco sementes (42, 43, 44, 45 e 46); SARIMA usa uma execução determinística.", "",
             "GEANT exclui, por solicitação da usuária, Ankara → Lisboa, LUT → Viena, Lisboa → RESTENA e Anella → Poznań. RNP exclui ES → PR, que não possui observações no teste. O subconjunto GEANT foi definido durante a análise dos resultados; suas médias descrevem os seis links exibidos e não representam a avaliação completa dos dez links originais.", ""]
    coverage = []
    for dataset, title in (("geant", "GEANT"), ("sbrc", "RNP")):
        panel = pd.read_csv(ROOT / f"{dataset}_aligned_panel_mbps.csv", float_precision="round_trip")
        split = int(len(panel)*.8)
        lines += [f"### {title}: período e cobertura", "",
                  f"A grade sincronizada contém **{len(panel)} intervalos**, de **{panel.Data.iloc[0]}** a **{panel.Data.iloc[-1]}**. O treinamento externo contém **{split} intervalos**; o teste contém **{len(panel)-split} intervalos**, de **{panel.Data.iloc[split]}** a **{panel.Data.iloc[-1]}**. Todos os modelos usam as mesmas datas de teste para cada link.", ""]
        for link in sorted(data.loc[data.dataset == dataset, "link"].unique()):
            values = panel[link].iloc[split:].replace(-1, np.nan)
            observed = int(values.notna().sum())
            observed_counts = raw.loc[(raw.dataset == dataset) & (raw.link == link), "observed_test_intervals"].unique()
            assert len(observed_counts) == 1 and observed_counts[0] == observed
            coverage.append([title, name(link, dataset), len(values), observed, len(values)-observed])
    lines += table(["Conjunto", "Link", "Intervalos no teste", "Observados avaliados", "Ausentes"], coverage)
    lines += ["## 2. Protocolo de treinamento e previsão", "",
              "- Cada observação representa um intervalo de **seis horas**. Contexto **3** fornece **18 horas** de histórico; contexto **28** fornece **168 horas (sete dias)**. Em ambos, o horizonte é de **um único intervalo de seis horas**.",
              "- Os últimos 20% das datas formam o teste. A seleção usa uma validação cronológica dentro dos primeiros 80%, sem selecionar hiperparâmetros pelo erro de teste. Entradas e escala são ajustadas dentro da respectiva partição de treinamento; alvos ausentes são excluídos da perda e das métricas.",
              "- Cada previsão é emitida antes de ler seu alvo. Uma observação disponível alimenta o próximo histórico; na ausência dela, entra a previsão. Portanto, contexto 28 não corresponde a um horizonte de sete dias sem realimentação.",
              "- A configuração e o número de épocas são selecionados usando a semente 42 e reutilizados nos cinco ajustes finais das redes. As redes têm limite de 100 épocas e paciência 10. Uma média entre sementes não equivale a treinar um ensemble nem a calcular a métrica sobre a média das previsões.",
              "- LSTM e GRU têm uma camada, com busca entre 32/64/128 unidades e taxas Adam de 0.01/0.001/0.0001. Forwarding LSTM tem duas camadas de 64 unidades, Adam 0.001 e seleção de batch 32/64/128. As recorrentes usam Huber com delta 0.25.",
              "- O Transformer padrão usa projeção de dimensão 32, quatro cabeças, uma ou duas camadas, dimensão feed-forward 64, dropout 0.1 e perda MSE. As diferenças de perda e de busca devem ser consideradas na interpretação.",
              "- As redes recorrentes são implementações PyTorch das arquiteturas, sob o protocolo controlado. Esses números não são reproduções numéricas dos antigos pesos Keras.",
              "- SARIMA seleciona ordens (1,0,0), (1,1,0) ou (1,0,1), componente sazonal (1,0,0) e período 4, 12 ou 28. Ele utiliza todo o histórico ajustado para manter seus estados; os resultados nas duas pastas de contexto repetem o protocolo estatístico e não representam SARIMA limitado a três ou 28 amostras.", "",
              "## 3. Definição das métricas e agregação", "",
              "Para os `n` alvos observados do teste, sejam `y` a vazão real, `ŷ` a previsão e `e = ŷ − y`:", "",
              "- **RMSE:** `sqrt(mean(e²))`, em Mbit/s; menor é melhor.",
              "- **MAE:** `mean(abs(e))`, em Mbit/s; menor é melhor.",
              "- **SMAPE:** `mean(200 × abs(e) / (abs(y) + abs(ŷ) + 10⁻¹²))`, em porcentagem; menor é melhor.",
              "- **Cohen’s kappa:** `κ = (pₒ − pₑ)/(1 − pₑ)`, sem ponderação, após discretizar ambos os valores em classes fixas `floor(vazão_Mbit/s / 100)`. As classes estendem-se abaixo de zero e acima de 2000 Mbit/s; pares fora de 0–2000 não são descartados. Maior é melhor. Este kappa não é a acurácia fuzzy do repositório.", "",
              "Kappa é `ND` quando não pode ser definido, por exemplo, quando observações e previsões ocupam uma única classe e `pₑ = 1`. Concordância em uma única classe não deve ser convertida artificialmente em kappa igual a 1. Kappa zero indica concordância equivalente à esperada pelo acaso sob suas marginais; valores negativos indicam concordância inferior à esperada.", "",
              "Primeiro calcula-se a métrica de cada semente; depois calcula-se a média por link. As médias globais abaixo atribuem o mesmo peso a cada link, independentemente de quantos alvos ele possui. **Média dos RMSEs por link não é RMSE calculado após reunir todos os erros.** Para kappa, médias ignoram apenas valores indefinidos; as contagens de sementes e links definidos são informadas porque os denominadores podem diferir entre modelos.", "",
              "## 4. Médias entre os links exibidos", ""]
    for dataset, title in (("geant", "GEANT"), ("sbrc", "RNP")):
        for context in (3, 28):
            lines += [f"### {title}, contexto {context}", ""]
            rows = []
            part = data[(data.dataset == dataset) & (data.context == context)]
            for model in MODELS:
                values = macro.loc[(dataset, context, model)]
                defined = int(part.loc[part.model == model, "Cohen_Kappa"].notna().sum())
                rows.append([NAMES[model], *[number(values[m]) for m in METRICS], f"{defined}/{part.link.nunique()}"])
            lines += table(["Modelo", *[LABELS[m] for m in METRICS], "Links com κ definido"], rows)
            for metric in ("RMSE", "MAE", "SMAPE"):
                scores = part.groupby("model")[metric].mean()
                winner = scores.idxmin()
                forward = float(scores["ForwardingLSTM"])
                competitor = scores.drop("ForwardingLSTM").idxmin()
                rival = float(scores[competitor])
                delta = (forward-rival)/rival*100
                relation = "inferior" if delta < 0 else "superior" if delta > 0 else "igual"
                lines += [f"- **{metric}:** o menor valor médio é de **{NAMES[winner]}**, com **{number(scores[winner])}**. O Forwarding LSTM apresenta **{number(forward)}**, valor **{abs(delta):.6f}% {relation}** ao do melhor concorrente ({NAMES[competitor]}: {number(rival)}).", ""]
            kappa_part = part.set_index(["link", "model"]).Cohen_Kappa.unstack("model").reindex(columns=MODELS)
            complete_kappa_links = kappa_part.dropna()
            if len(complete_kappa_links):
                comparable = part[part.link.isin(complete_kappa_links.index)].groupby("model").Cohen_Kappa.mean()
                leader = comparable.idxmax()
                lines += [f"Para comparar kappa sobre uma base comum, há **{len(complete_kappa_links)} links** com média definida para todos os cinco métodos. Nessa base, o maior kappa médio é de **{NAMES[leader]}**, com **{number(comparable[leader])}**; o Forwarding LSTM obtém **{number(comparable['ForwardingLSTM'])}**. A disponibilidade de sementes ainda pode variar; consulte as tabelas por link.", ""]
            else:
                lines += ["Não há link com kappa definido para todos os cinco métodos; uma ordenação global por kappa teria denominadores distintos e não é apresentada como comparação equivalente.", ""]
    lines += ["## 5. Efeito do contexto no Forwarding LSTM", ""]
    rows = []
    for dataset, title in (("geant", "GEANT"), ("sbrc", "RNP")):
        three = macro.loc[(dataset, 3, "ForwardingLSTM")]
        twenty_eight = macro.loc[(dataset, 28, "ForwardingLSTM")]
        for metric in ("RMSE", "MAE", "SMAPE"):
            delta = twenty_eight[metric]-three[metric]
            percent = delta/three[metric]*100
            rows.append([title, metric, number(three[metric]), number(twenty_eight[metric]), number(delta), number(percent)])
    lines += table(["Conjunto", "Métrica", "Contexto 3", "Contexto 28", "Δ = 28 − 3", "Δ relativo (%)"], rows)
    lines += ["Nas três métricas de erro, diferença negativa favorece contexto 28; diferença positiva favorece contexto 3. O efeito depende do conjunto e da métrica, não apenas da quantidade de histórico.", "",
              "## 6. Valores completos por link", "",
              "A coluna `sementes κ` indica quantas sementes possuem kappa definido, sobre cinco para as redes ou uma para SARIMA. Todos os modelos são apresentados, inclusive quando não vencem determinada comparação.", ""]
    winners = []
    for dataset, title in (("geant", "GEANT"), ("sbrc", "RNP")):
        for context in (3, 28):
            for link in sorted(data.loc[data.dataset == dataset, "link"].unique()):
                part = data[(data.dataset == dataset) & (data.context == context) & (data.link == link)].set_index("model")
                lines += [f"### {title} — {name(link, dataset)} — contexto {context}", ""]
                rows = []
                for model in MODELS:
                    values = part.loc[model]
                    n = int(counts.loc[(dataset, context, link, model)])
                    rows.append([NAMES[model], *[number(values[m]) for m in METRICS], f"{n}/{1 if model == 'SARIMA' else 5}"])
                lines += table(["Modelo", *[LABELS[m] for m in METRICS], "Sementes κ"], rows)
                best = []
                for metric in METRICS:
                    values = part[metric].dropna()
                    if values.empty:
                        leaders = "ND para todos"
                    else:
                        value = values.max() if metric == "Cohen_Kappa" else values.min()
                        tied = values.index[np.isclose(values.to_numpy(), value, rtol=1e-9, atol=1e-12)]
                        leaders = ", ".join(NAMES[m] for m in tied)
                    best.append(f"{metric}: {leaders}")
                    winners.append([title, context, name(link, dataset), metric, leaders])
                lines += ["Melhores valores neste link/contexto: " + "; ".join(best) + ".", ""]
    pd.DataFrame(winners, columns=["conjunto", "contexto", "link", "metrica", "melhores"]).to_csv(OUT / "vencedores_por_link.csv", index=False)
    lines += ["## 7. Variação entre sementes do Forwarding LSTM", "",
              "Desvio-padrão amostral entre as cinco sementes (`ddof=1`), calculado por link. Não é intervalo de confiança nem teste de significância. `ND` ocorre se menos de duas sementes têm kappa definido.", ""]
    sd = raw[raw.model == "ForwardingLSTM"].groupby(["dataset", "context", "link"])[METRICS].std(ddof=1)
    rows = []
    for (dataset, context, link), values in sd.iterrows():
        rows.append(["GEANT" if dataset == "geant" else "RNP", context, name(link, dataset), *[number(values[m]) for m in METRICS]])
    lines += table(["Conjunto", "Contexto", "Link", *["DP " + m for m in METRICS]], rows)
    lines += ["## 8. Limites de interpretação", "",
              "1. As conclusões se restringem aos links selecionados, às datas de teste e às configurações executadas. A remoção de links GEANT após a análise pode alterar médias e posições; não é evidência de superioridade no conjunto GEANT completo.",
              "2. As tabelas não fornecem significância estatística. Há dependência temporal entre intervalos e somente cinco sementes das redes, que compartilham a configuração selecionada na semente 42.",
              "3. Comparações por kappa dependem das faixas de 100 Mbit/s e da distribuição de classes. Baixo erro de vazão pode coexistir com kappa baixo ou indefinido quando a vazão está concentrada em poucas classes. O número de valores definidos precisa acompanhar a média.",
              "4. Os modelos diferem em perda, arquitetura, orçamento de busca e, no caso de SARIMA, representação do histórico. O protocolo controla datas, teste por link e realimentação; não isola exclusivamente o efeito da arquitetura.",
              "5. A comparação é univariada por link. Os resultados anteriores de iTransformer, TimeBridge e TimeMixer++ não integram estas figuras e não devem ser incorporados às mesmas médias sem compatibilizar o protocolo.", "",
              "## 9. Arquivos e rastreabilidade", "",
              "- [Dados numéricos das figuras](metrics.csv).",
              "- [Médias globais deste relatório](medias_relatorio.csv).",
              "- [Melhores modelos por link e métrica](vencedores_por_link.csv).",
              "- [Legendas das figuras](CAPTIONS.md).",
              "- [Manifesto dos gráficos e templates](manifest.json).",
              "- [Protocolo completo](../../../../CONTROLLED.md).",
              "- [Auditoria dos experimentos](../../audit.json): 1.088 execuções originais verificadas, 20 pares modelo/contexto restaurados e nenhuma execução ausente. Este relatório utiliza o subconjunto de 462 execuções que corresponde aos modelos e links exibidos.", "",
              f"SHA-256 do CSV das figuras: `{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}`.", "",
              "### Figuras correspondentes", ""]
    lines += table(["Conjunto", "Contexto", "RMSE", "MAE", "SMAPE", "Cohen’s kappa"],
                   [[title, context, *[f"[{m.upper()}]({prefix}_context_{context}_{m}.png)" for m in ("rmse", "mae", "smape", "cohen_kappa")]]
                    for title, prefix in (("GEANT", "geant"), ("RNP", "rnp")) for context in (3, 28)])
    lines += ["Para regenerar este relatório a partir dos arquivos salvos, execute na raiz do projeto:", "",
              "```powershell", "& .\\ijsm_review\\.venv-experiments\\Scripts\\python.exe .\\ijsm_review\\describe_template_results.py", "```", "",
              "### Identificadores completos dos links exibidos", ""]
    lines += table(["Conjunto", "Rótulo", "Identificador original"],
                   [["GEANT" if d == "geant" else "RNP", name(l, d), l] for d, l in data[["dataset", "link"]].drop_duplicates().sort_values(["dataset", "link"]).itertuples(index=False, name=None)])
    path = OUT / "DESCRICAO_RESULTADOS.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Saved {path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
