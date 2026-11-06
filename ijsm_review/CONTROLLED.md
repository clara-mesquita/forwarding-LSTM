# Comparação por link

Execução completa, na raiz do projeto:

```powershell
& .\ijsm_review\run_controlled.ps1
```

Ou, com o ambiente já preparado:

```powershell
& .\ijsm_review\.venv-experiments\Scripts\python.exe .\ijsm_review\run_controlled.py
& .\ijsm_review\.venv-experiments\Scripts\python.exe .\ijsm_review\plot_controlled.py
```

Dependências adicionais: `requirements-controlled.txt`. Não é necessário baixar
implementações externas para estes modelos.

## Modelos

- Persistência: repete o último valor do histórico, sem treinamento.
- Ridge: quatro valores de alpha, 0.1, 1, 10 e 100.
- Random Forest: 100/300 árvores, profundidade 5/10 e folhas mínimas 2/5.
- MLP: uma camada de 32 neurônios ou duas de 64/32, ReLU, saída linear.
- Transformer padrão: projeção para 32 dimensões, posições sinusoidais,
  quatro cabeças, uma/duas camadas, feed-forward 64 e dropout 0.1.
- LSTM e GRU: uma camada, 32/64/128 unidades, Adam com taxas 0.01/0.001/0.0001.
- Forwarding LSTM: duas camadas LSTM de 64 unidades, Adam 0.001,
  Huber delta 0.25 e seleção de batch 32/64/128.
- ARIMA: ordens (1,0,0), (1,1,0), (1,0,1).
- SARIMA: mesmas ordens com componente sazonal (1,0,0) e períodos 4/12/28.

LSTM, GRU, Random Forest e SARIMA correspondem às famílias encontradas em
`code/forwarding-LSTM/grid` e nos resultados de `results_sbrc_26`.
As redes recorrentes são novas implementações PyTorch das arquiteturas,
com Huber delta 0.25; não reutilizam pesos Keras nem reproduzem exatamente
os experimentos antigos. O Forwarding LSTM mantém a arquitetura e o mecanismo
de realimentação, com ajuste de épocas sob o protocolo comum de validação.
ARIMA/SARIMA usam o histórico completo anterior para estimar seus estados;
a coluna contexto se refere à janela dos modelos supervisionados. Seus dois
resultados de contexto são repetições do mesmo protocolo estatístico, não
modelos ARIMA restritos a três ou 28 observações.

## Protocolo

GEANT (10 links) e SBRC (6 links) usam as mesmas grades temporais sincronizadas
dos experimentos anteriores, separadamente. O BBR de `ijsm_review/data`
é uma cópia dos seis links SBRC e não é contado outra vez.

Cada modelo é treinado por link com contextos 3 e 28. Os últimos 20% das datas
são teste; os últimos 20% do treinamento são validação. Quando não há alvos
observados nessa validação (es-pr), o corte recua para reservar os últimos 20%
dos alvos observados do treinamento. O teste nunca participa dessa escolha.
As datas efetivamente usadas ficam em `run.json`.

Entradas de treinamento são interpoladas somente dentro da respectiva
partição. Limites MinMax vêm somente dessa partição. Alvos ausentes são
excluídos de todos os conjuntos supervisionados e da função de perda.
ARIMA/SARIMA recebem NaN nos alvos ausentes durante a estimação dos estados.
Históricos da validação para early stopping usam apenas preenchimento para
frente, iniciado na última entrada de treinamento.

Cada previsão é emitida antes de ler o respectivo alvo. A observação real,
quando disponível, alimenta o próximo histórico; caso contrário, entra a
previsão. A mesma realimentação é usada para escolher configurações pela
MSE observada de validação. Horizonte: um intervalo de seis horas.

Redes: máximo de 100 épocas, paciência 10, Adam, batches cronológicos e
gradientes limitados a norma 1. MLP/Transformer usam MSE; recorrentes usam
Huber. As épocas são escolhidas na validação e o modelo é reconstruído e
reajustado em todo o treinamento. A seleção usa a semente 42; a configuração
e o número de épocas selecionados são usados nas cinco execuções finais
(42 a 46) de RF e redes. Modelos determinísticos são executados uma vez.
As diferenças de perda e de tamanho das grades são registradas; este é um
protocolo comum de dados e realimentação, não de orçamento de busca idêntico.

Somente candidatos estatísticos convergentes são aceitos. Falhas são
registradas nas configurações avaliadas, sem converter falhas em métricas.
Se o ajuste estatístico final não convergir com LBFGS, há uma tentativa
adicional com Powell, mantendo a configuração selecionada. O uso dessa
tentativa fica registrado em `final_fit_warnings`. Artefatos completos da
versão anterior sem essa tentativa são compatíveis porque seus ajustes
finais já tinham convergido; a retomada preserva seus fingerprints originais.

## Arquivos

`results/controlled/<dataset>/context_<3 ou 28>/<modelo>/<link>/seed_<n>/`:

- `predictions.csv`: datas, observações, máscara e previsões em Mbit/s.
- `metrics.csv`: RMSE, MAE, NRMSE, SMAPE, fuzzy e número de alvos observados.
- `model.pt` ou `model.joblib`: modelo e parâmetros de escala.
- `run.json`: seleção, limites temporais, configuração, épocas, versões,
  tempos e fingerprint de dados/código/parâmetros para retomada.

`metrics_all.csv` reúne as execuções. `metrics_mean_per_link.csv` calcula a
média das sementes por link; `summary.csv` agrega esses links com pesos iguais.
Valores sem observações são ausentes, nunca zero. Fuzzy mantém os limites da
implementação anterior e precisa ser interpretado com sua contagem avaliada.

Resultados das execuções anteriores, incluindo os três benchmarks
multivariados, permanecem em suas pastas originais. Não são misturados a
esta comparação univariada.

Uma nova chamada retoma apenas artefatos completos com fingerprint idêntico.
O log da execução realizada pelo assistente fica em
`results/controlled_training.log` (início) e nos quatro arquivos
`results/controlled_geant3.log`, `controlled_geant28.log`,
`controlled_sbrc3.log` e `controlled_sbrc28.log` (partições independentes).
As figuras e tabelas dos gráficos ficam em
`results/controlled/plots`.

Verificação do protocolo:

```powershell
& .\ijsm_review\.venv-experiments\Scripts\python.exe -m unittest discover -s ijsm_review -p test_controlled.py
```
