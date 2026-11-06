# Descrição dos resultados das figuras

Este relatório descreve os valores efetivamente utilizados nas 16 figuras da pasta `five_models_template_style`: quatro métricas, dois conjuntos de dados e dois contextos. Cada figura compara SARIMA, GRU, LSTM, Forwarding LSTM e Transformer.

As tabelas usam até **17 algarismos significativos**, preservando a representação numérica dos valores de ponto flutuante salvos no CSV. Essa quantidade de dígitos descreve a precisão do arquivo, não a incerteza experimental. `ND` significa não definido/ausente; nunca equivale a zero. O ponto é usado como separador decimal.

## 1. Escopo dos dados e das execuções

São apresentados **seis links GEANT e cinco links RNP**, em contextos de 3 e 28 intervalos. O conjunto RNP corresponde aos dados BBR salvos sob o identificador `sbrc`. Há **110 combinações de conjunto × contexto × link × modelo**, provenientes de **462 execuções individuais**: as quatro redes usam cinco sementes (42, 43, 44, 45 e 46); SARIMA usa uma execução determinística.

GEANT exclui, por solicitação da usuária, Ankara → Lisboa, LUT → Viena, Lisboa → RESTENA e Anella → Poznań. RNP exclui ES → PR, que não possui observações no teste. O subconjunto GEANT foi definido durante a análise dos resultados; suas médias descrevem os seis links exibidos e não representam a avaliação completa dos dez links originais.

### GEANT: período e cobertura

A grade sincronizada contém **238 intervalos**, de **2023-08-25 06:00:00** a **2023-10-23 12:00:00**. O treinamento externo contém **190 intervalos**; o teste contém **48 intervalos**, de **2023-10-11 18:00:00** a **2023-10-23 12:00:00**. Todos os modelos usam as mesmas datas de teste para cada link.

### RNP: período e cobertura

A grade sincronizada contém **783 intervalos**, de **2024-01-15 18:00:00** a **2024-07-29 06:00:00**. O treinamento externo contém **626 intervalos**; o teste contém **157 intervalos**, de **2024-06-20 06:00:00** a **2024-07-29 06:00:00**. Todos os modelos usam as mesmas datas de teste para cada link.

| Conjunto | Link | Intervalos no teste | Observados avaliados | Ausentes |
| --- | --- | --- | --- | --- |
| GEANT | Ankara → Poznań | 48 | 26 | 22 |
| GEANT | RedIRIS → Lisboa | 48 | 32 | 16 |
| GEANT | Lisboa → Ankara | 48 | 30 | 18 |
| GEANT | Poznań → Ankara | 48 | 31 | 17 |
| GEANT | Poznań → Anella | 48 | 30 | 18 |
| GEANT | Viena → LUT | 48 | 26 | 22 |
| RNP | AC → TO | 157 | 35 | 122 |
| RNP | ES → CE | 157 | 123 | 34 |
| RNP | GO → SP | 157 | 113 | 44 |
| RNP | MS → TO | 157 | 37 | 120 |
| RNP | SC → TO | 157 | 30 | 127 |

## 2. Protocolo de treinamento e previsão

- Cada observação representa um intervalo de **seis horas**. Contexto **3** fornece **18 horas** de histórico; contexto **28** fornece **168 horas (sete dias)**. Em ambos, o horizonte é de **um único intervalo de seis horas**.
- Os últimos 20% das datas formam o teste. A seleção usa uma validação cronológica dentro dos primeiros 80%, sem selecionar hiperparâmetros pelo erro de teste. Entradas e escala são ajustadas dentro da respectiva partição de treinamento; alvos ausentes são excluídos da perda e das métricas.
- Cada previsão é emitida antes de ler seu alvo. Uma observação disponível alimenta o próximo histórico; na ausência dela, entra a previsão. Portanto, contexto 28 não corresponde a um horizonte de sete dias sem realimentação.
- A configuração e o número de épocas são selecionados usando a semente 42 e reutilizados nos cinco ajustes finais das redes. As redes têm limite de 100 épocas e paciência 10. Uma média entre sementes não equivale a treinar um ensemble nem a calcular a métrica sobre a média das previsões.
- LSTM e GRU têm uma camada, com busca entre 32/64/128 unidades e taxas Adam de 0.01/0.001/0.0001. Forwarding LSTM tem duas camadas de 64 unidades, Adam 0.001 e seleção de batch 32/64/128. As recorrentes usam Huber com delta 0.25.
- O Transformer padrão usa projeção de dimensão 32, quatro cabeças, uma ou duas camadas, dimensão feed-forward 64, dropout 0.1 e perda MSE. As diferenças de perda e de busca devem ser consideradas na interpretação.
- As redes recorrentes são implementações PyTorch das arquiteturas, sob o protocolo controlado. Esses números não são reproduções numéricas dos antigos pesos Keras.
- SARIMA seleciona ordens (1,0,0), (1,1,0) ou (1,0,1), componente sazonal (1,0,0) e período 4, 12 ou 28. Ele utiliza todo o histórico ajustado para manter seus estados; os resultados nas duas pastas de contexto repetem o protocolo estatístico e não representam SARIMA limitado a três ou 28 amostras.

## 3. Definição das métricas e agregação

Para os `n` alvos observados do teste, sejam `y` a vazão real, `ŷ` a previsão e `e = ŷ − y`:

- **RMSE:** `sqrt(mean(e²))`, em Mbit/s; menor é melhor.
- **MAE:** `mean(abs(e))`, em Mbit/s; menor é melhor.
- **SMAPE:** `mean(200 × abs(e) / (abs(y) + abs(ŷ) + 10⁻¹²))`, em porcentagem; menor é melhor.
- **Cohen’s kappa:** `κ = (pₒ − pₑ)/(1 − pₑ)`, sem ponderação, após discretizar ambos os valores em classes fixas `floor(vazão_Mbit/s / 100)`. As classes estendem-se abaixo de zero e acima de 2000 Mbit/s; pares fora de 0–2000 não são descartados. Maior é melhor. Este kappa não é a acurácia fuzzy do repositório.

Kappa é `ND` quando não pode ser definido, por exemplo, quando observações e previsões ocupam uma única classe e `pₑ = 1`. Concordância em uma única classe não deve ser convertida artificialmente em kappa igual a 1. Kappa zero indica concordância equivalente à esperada pelo acaso sob suas marginais; valores negativos indicam concordância inferior à esperada.

Primeiro calcula-se a métrica de cada semente; depois calcula-se a média por link. As médias globais abaixo atribuem o mesmo peso a cada link, independentemente de quantos alvos ele possui. **Média dos RMSEs por link não é RMSE calculado após reunir todos os erros.** Para kappa, médias ignoram apenas valores indefinidos; as contagens de sementes e links definidos são informadas porque os denominadores podem diferir entre modelos.

## 4. Médias entre os links exibidos

### GEANT, contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Links com κ definido |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 101.80531651486608 | 55.697103977464707 | 8.6973150037848104 | 0.033927289619349245 | 6/6 |
| GRU | 76.652110138336965 | 52.057205125800579 | 9.7906646901079153 | 0.0013068686668666004 | 6/6 |
| LSTM | 137.95958532596538 | 116.04535480006382 | 13.71141551770433 | -0.012192364725758615 | 6/6 |
| Forwarding LSTM | 65.861895895353072 | 41.426069841250857 | 9.0666056020529648 | -0.029575575716573304 | 6/6 |
| Transformer | 78.29363463847919 | 55.142206189870443 | 10.787231990834451 | 0.012737304570613483 | 6/6 |

- **RMSE:** o menor valor médio é de **Forwarding LSTM**, com **65.861895895353072**. O Forwarding LSTM apresenta **65.861895895353072**, valor **14.076865% inferior** ao do melhor concorrente (GRU: 76.652110138336965).

- **MAE:** o menor valor médio é de **Forwarding LSTM**, com **41.426069841250857**. O Forwarding LSTM apresenta **41.426069841250857**, valor **20.422025% inferior** ao do melhor concorrente (GRU: 52.057205125800579).

- **SMAPE:** o menor valor médio é de **SARIMA**, com **8.6973150037848104**. O Forwarding LSTM apresenta **9.0666056020529648**, valor **4.246030% superior** ao do melhor concorrente (SARIMA: 8.6973150037848104).

Para comparar kappa sobre uma base comum, há **6 links** com média definida para todos os cinco métodos. Nessa base, o maior kappa médio é de **SARIMA**, com **0.033927289619349245**; o Forwarding LSTM obtém **-0.029575575716573304**. A disponibilidade de sementes ainda pode variar; consulte as tabelas por link.

### GEANT, contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Links com κ definido |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 101.80531651486608 | 55.697103977464707 | 8.6973150037848104 | 0.033927289619349245 | 6/6 |
| GRU | 146.34735491741137 | 126.0286667620495 | 13.339714974721268 | 0.0067564534231200667 | 6/6 |
| LSTM | 83.035530021179 | 59.676895209701662 | 9.5771964098743965 | 0.0032550554612844162 | 6/6 |
| Forwarding LSTM | 77.69664714237804 | 54.650161308161834 | 8.8061279033527899 | 0.0051256700106314659 | 6/6 |
| Transformer | 78.909786282043655 | 54.446723710887177 | 9.7941103532997928 | -0.018835433249991849 | 6/6 |

- **RMSE:** o menor valor médio é de **Forwarding LSTM**, com **77.69664714237804**. O Forwarding LSTM apresenta **77.69664714237804**, valor **1.537375% inferior** ao do melhor concorrente (Transformer: 78.909786282043655).

- **MAE:** o menor valor médio é de **Transformer**, com **54.446723710887177**. O Forwarding LSTM apresenta **54.650161308161834**, valor **0.373645% superior** ao do melhor concorrente (Transformer: 54.446723710887177).

- **SMAPE:** o menor valor médio é de **SARIMA**, com **8.6973150037848104**. O Forwarding LSTM apresenta **8.8061279033527899**, valor **1.251109% superior** ao do melhor concorrente (SARIMA: 8.6973150037848104).

Para comparar kappa sobre uma base comum, há **6 links** com média definida para todos os cinco métodos. Nessa base, o maior kappa médio é de **SARIMA**, com **0.033927289619349245**; o Forwarding LSTM obtém **0.0051256700106314659**. A disponibilidade de sementes ainda pode variar; consulte as tabelas por link.

### RNP, contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Links com κ definido |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 26.350257799842275 | 17.9798381223376 | 2.1866111467619058 | 0.04973591549295775 | 4/5 |
| GRU | 41.077723107187687 | 36.471837185058774 | 4.3426930345382377 | 0.0090444454687062005 | 4/5 |
| LSTM | 43.473344460669004 | 38.799748363590439 | 4.6242975714936136 | -0.0042746858883554997 | 4/5 |
| Forwarding LSTM | 41.121137650738788 | 36.773811426267116 | 4.4052178739063308 | -0.0024052347610134749 | 4/5 |
| Transformer | 50.094490369425259 | 45.763682414970361 | 5.3536077159719513 | -0.00039571310799670003 | 5/5 |

- **RMSE:** o menor valor médio é de **SARIMA**, com **26.350257799842275**. O Forwarding LSTM apresenta **41.121137650738788**, valor **56.055922% superior** ao do melhor concorrente (SARIMA: 26.350257799842275).

- **MAE:** o menor valor médio é de **SARIMA**, com **17.9798381223376**. O Forwarding LSTM apresenta **36.773811426267116**, valor **104.528045% superior** ao do melhor concorrente (SARIMA: 17.9798381223376).

- **SMAPE:** o menor valor médio é de **SARIMA**, com **2.1866111467619058**. O Forwarding LSTM apresenta **4.4052178739063308**, valor **101.463250% superior** ao do melhor concorrente (SARIMA: 2.1866111467619058).

Para comparar kappa sobre uma base comum, há **4 links** com média definida para todos os cinco métodos. Nessa base, o maior kappa médio é de **SARIMA**, com **0.04973591549295775**; o Forwarding LSTM obtém **-0.0024052347610134749**. A disponibilidade de sementes ainda pode variar; consulte as tabelas por link.

### RNP, contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Links com κ definido |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 26.350257799842275 | 17.9798381223376 | 2.1866111467619058 | 0.04973591549295775 | 4/5 |
| GRU | 36.990950308473302 | 31.764080496939375 | 3.8452484000453553 | -0.0054861865311245 | 4/5 |
| LSTM | 36.962880129035824 | 31.308547173847863 | 3.793703535877019 | -0.0050809385381710252 | 4/5 |
| Forwarding LSTM | 40.841935199772507 | 35.903937482294459 | 4.314181339362241 | -0.0022780398670648001 | 4/5 |
| Transformer | 47.24486564800209 | 42.638882565588148 | 5.0117281319705089 | 0.0030984762886272 | 4/5 |

- **RMSE:** o menor valor médio é de **SARIMA**, com **26.350257799842275**. O Forwarding LSTM apresenta **40.841935199772507**, valor **54.996340% superior** ao do melhor concorrente (SARIMA: 26.350257799842275).

- **MAE:** o menor valor médio é de **SARIMA**, com **17.9798381223376**. O Forwarding LSTM apresenta **35.903937482294459**, valor **99.689993% superior** ao do melhor concorrente (SARIMA: 17.9798381223376).

- **SMAPE:** o menor valor médio é de **SARIMA**, com **2.1866111467619058**. O Forwarding LSTM apresenta **4.314181339362241**, valor **97.299888% superior** ao do melhor concorrente (SARIMA: 2.1866111467619058).

Para comparar kappa sobre uma base comum, há **4 links** com média definida para todos os cinco métodos. Nessa base, o maior kappa médio é de **SARIMA**, com **0.04973591549295775**; o Forwarding LSTM obtém **-0.0022780398670648001**. A disponibilidade de sementes ainda pode variar; consulte as tabelas por link.

## 5. Efeito do contexto no Forwarding LSTM

| Conjunto | Métrica | Contexto 3 | Contexto 28 | Δ = 28 − 3 | Δ relativo (%) |
| --- | --- | --- | --- | --- | --- |
| GEANT | RMSE | 65.861895895353072 | 77.69664714237804 | 11.834751247024968 | 17.969041258437226 |
| GEANT | MAE | 41.426069841250857 | 54.650161308161834 | 13.224091466910977 | 31.922148341822226 |
| GEANT | SMAPE | 9.0666056020529648 | 8.8061279033527899 | -0.2604776987001749 | -2.8729351438998796 |
| RNP | RMSE | 41.121137650738788 | 40.841935199772507 | -0.27920245096628094 | -0.67897550242330107 |
| RNP | MAE | 36.773811426267116 | 35.903937482294459 | -0.8698739439726566 | -2.3654712694570339 |
| RNP | SMAPE | 4.4052178739063308 | 4.314181339362241 | -0.091036534544089776 | -2.0665614539369663 |

Nas três métricas de erro, diferença negativa favorece contexto 28; diferença positiva favorece contexto 3. O efeito depende do conjunto e da métrica, não apenas da quantidade de histórico.

## 6. Valores completos por link

A coluna `sementes κ` indica quantas sementes possuem kappa definido, sobre cinco para as redes ou uma para SARIMA. Todos os modelos são apresentados, inclusive quando não vencem determinada comparação.

### GEANT — Ankara → Poznań — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 114.46502588408129 | 77.413023723193902 | 11.513559379833188 | -0.042979942693409601 | 1/1 |
| GRU | 107.58098408354137 | 67.62089365858364 | 10.103317024416455 | -0.046753246753246797 | 5/5 |
| LSTM | 107.44315499770671 | 63.325911469780927 | 9.5086465287793072 | 0 | 5/5 |
| Forwarding LSTM | 108.34838495550623 | 61.179157455519487 | 9.2073330051652729 | 0 | 5/5 |
| Transformer | 116.3299557572156 | 69.117581388564645 | 10.232875814291427 | -0.0120171621948271 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Forwarding LSTM, LSTM.

### GEANT — RedIRIS → Lisboa — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 300.35402813360025 | 135.16652597926085 | 7.6503377512898716 | -0.063444108761329304 | 1/1 |
| GRU | 164.43956607727085 | 128.75243042262758 | 6.3809756512148317 | 0.012429854264360799 | 5/5 |
| LSTM | 528.55945038454934 | 514.05107749030117 | 28.86287010176072 | 0.0021321961620469 | 5/5 |
| Forwarding LSTM | 99.060700539564763 | 72.411639262670548 | 3.5099081190699493 | -0.0573863465974392 | 5/5 |
| Transformer | 107.69338794778852 | 79.0483003579442 | 3.8023874495730938 | 0.0245975261531521 | 5/5 |

Melhores valores neste link/contexto: RMSE: Forwarding LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Transformer.

### GEANT — Lisboa → Ankara — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 84.984770848636117 | 32.13714914614939 | 4.2057938743148622 | 0 | 1/1 |
| GRU | 84.728052966940211 | 29.129451193897619 | 3.8655539292855439 | 0 | 5/5 |
| LSTM | 85.677377959324403 | 29.299620053407644 | 3.8820673405709361 | 0 | 5/5 |
| Forwarding LSTM | 85.805936737486618 | 28.83513922768207 | 3.827387975919327 | 0 | 5/5 |
| Transformer | 93.054362724726445 | 45.229644600511634 | 5.6318724498569646 | 0.073239436619718296 | 5/5 |

Melhores valores neste link/contexto: RMSE: GRU; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Transformer.

### GEANT — Poznań → Ankara — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 18.660092747470703 | 15.074883872733956 | 1.6800355824526645 | 0 | 1/1 |
| GRU | 17.580678683669497 | 13.650749771874796 | 1.5213220567001735 | 0.042164604490085598 | 5/5 |
| LSTM | 17.741012088938177 | 13.66228165427257 | 1.5226004975646363 | -0.075286384516598595 | 5/5 |
| Forwarding LSTM | 17.85979973273303 | 13.736326769756108 | 1.5308305279183292 | -0.1200671077020006 | 5/5 |
| Transformer | 59.650934619721077 | 56.646777213256961 | 6.124713519175649 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: GRU; MAE: GRU; SMAPE: GRU; Cohen_Kappa: GRU.

### GEANT — Poznań → Anella — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 53.859162556627687 | 46.635677191608558 | 5.4443545458227049 | 0.057591623036649303 | 1/1 |
| GRU | 36.457518194218864 | 29.677766706879208 | 3.428292829341828 | 0 | 5/5 |
| LSTM | 36.470736568195207 | 29.563974537777742 | 3.4157322950493949 | 0 | 5/5 |
| Forwarding LSTM | 35.272725573538445 | 29.774322356767851 | 3.4391768930803552 | 0 | 5/5 |
| Transformer | 39.330867520971182 | 33.162256967932457 | 3.8327174572028961 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: Forwarding LSTM; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: SARIMA.

### GEANT — Viena → LUT — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 38.508818918780399 | 27.755363951841574 | 21.689808888995572 | 0.25239616613418508 | 1/1 |
| GRU | 49.125860824380958 | 43.511939000940593 | 33.444526649688655 | 0 | 5/5 |
| LSTM | 51.865779957078281 | 46.369263594842792 | 35.076576342500985 | 0 | 5/5 |
| Forwarding LSTM | 48.823827833289322 | 42.619833975109053 | 32.884997091164557 | 0 | 5/5 |
| Transformer | 53.702299260452257 | 47.648676611012725 | 35.098825254906679 | -0.0093959731543623998 | 5/5 |

Melhores valores neste link/contexto: RMSE: SARIMA; MAE: SARIMA; SMAPE: SARIMA; Cohen_Kappa: SARIMA.

### GEANT — Ankara → Poznań — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 114.46502588408129 | 77.413023723193902 | 11.513559379833188 | -0.042979942693409601 | 1/1 |
| GRU | 109.4727408813408 | 64.892883824594747 | 9.7052811577969447 | 0 | 5/5 |
| LSTM | 107.92479502935925 | 63.000617232615376 | 9.4577354151233397 | -0.0095238095238095004 | 5/5 |
| Forwarding LSTM | 111.61400873612396 | 62.371889674423095 | 9.3479818852098049 | 0 | 5/5 |
| Transformer | 114.69398752586464 | 68.082446244279112 | 10.127591807185365 | -0.078899004970679198 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Forwarding LSTM, GRU.

### GEANT — RedIRIS → Lisboa — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 300.35402813360025 | 135.16652597926085 | 7.6503377512898716 | -0.063444108761329304 | 1/1 |
| GRU | 588.23944553838771 | 580.13384837033334 | 32.357707617951206 | 0 | 5/5 |
| LSTM | 208.58557572855449 | 184.64415774814188 | 9.2288608295224233 | -0.0041363552432217001 | 5/5 |
| Forwarding LSTM | 176.46387141887101 | 159.74821746366115 | 7.8989213466306207 | 0.0098153919049441003 | 5/5 |
| Transformer | 150.28958935125607 | 127.31133491389797 | 6.2572822091566227 | 0.0049699182223745001 | 5/5 |

Melhores valores neste link/contexto: RMSE: Transformer; MAE: Transformer; SMAPE: Transformer; Cohen_Kappa: Forwarding LSTM.

### GEANT — Lisboa → Ankara — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 84.984770848636117 | 32.13714914614939 | 4.2057938743148622 | 0 | 1/1 |
| GRU | 84.113444304254955 | 30.830197696004131 | 4.0586536868659149 | 0 | 5/5 |
| LSTM | 84.362102213515001 | 29.695096822219401 | 3.9295753609155448 | 0 | 5/5 |
| Forwarding LSTM | 84.667138998484802 | 29.066321469520169 | 3.8565821195988654 | 0 | 5/5 |
| Transformer | 86.185648791553803 | 29.624875627869223 | 3.9151809577234871 | 0.0044109864973284002 | 5/5 |

Melhores valores neste link/contexto: RMSE: GRU; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Transformer.

### GEANT — Poznań → Ankara — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 18.660092747470703 | 15.074883872733956 | 1.6800355824526645 | 0 | 1/1 |
| GRU | 17.861243908592972 | 13.900861645453308 | 1.5491399607059095 | 0.040538720538720402 | 5/5 |
| LSTM | 17.721448806061449 | 13.825807567412076 | 1.5407676835903472 | 0.033190497534737698 | 5/5 |
| Forwarding LSTM | 17.989309316801808 | 13.315399199454712 | 1.4840626411737228 | 0.020938628158844699 | 5/5 |
| Transformer | 31.758805351127439 | 27.576794695375249 | 3.0264080180277211 | -0.0096618357487922007 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: GRU.

### GEANT — Poznań → Anella — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 53.859162556627687 | 46.635677191608558 | 5.4443545458227049 | 0.057591623036649303 | 1/1 |
| GRU | 35.729278658715799 | 30.042395769247509 | 3.4701104897004571 | 0 | 5/5 |
| LSTM | 35.27769911042806 | 28.950138144118107 | 3.3446999351222244 | 0 | 5/5 |
| Forwarding LSTM | 35.70975419720385 | 30.338331767787931 | 3.5044185866247246 | 0 | 5/5 |
| Transformer | 43.619975129509911 | 33.432892933986388 | 3.8395954546450901 | -0.033832663500182603 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: SARIMA.

### GEANT — Viena → LUT — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 38.508818918780399 | 27.755363951841574 | 21.689808888995572 | 0.25239616613418508 | 1/1 |
| GRU | 42.66797621317594 | 36.371813266663921 | 28.897396935307171 | 0 | 5/5 |
| LSTM | 44.341559239155728 | 37.945553743703158 | 29.961539234972498 | 0 | 5/5 |
| Forwarding LSTM | 39.735800186782775 | 33.06080827412395 | 26.744800840879009 | 0 | 5/5 |
| Transformer | 46.910711542950061 | 40.651997849915126 | 31.598603673060467 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: SARIMA; MAE: SARIMA; SMAPE: SARIMA; Cohen_Kappa: SARIMA.

### RNP — AC → TO — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 78.958036376457613 | 55.558596153558533 | 7.244242356681017 | 0.198943661971831 | 1/1 |
| GRU | 138.40700723538953 | 129.33866526848936 | 16.063597009175787 | -0.019778646036020701 | 5/5 |
| LSTM | 146.99588714619182 | 139.30205539928707 | 17.198710369639326 | -0.017098743553421999 | 5/5 |
| Forwarding LSTM | 150.99029779448671 | 143.76089600279207 | 17.700024633491473 | -0.0096209390440538996 | 5/5 |
| Transformer | 168.7183153482942 | 162.86346391304647 | 19.790952296471399 | -0.0019785655399835 | 5/5 |

Melhores valores neste link/contexto: RMSE: SARIMA; MAE: SARIMA; SMAPE: SARIMA; Cohen_Kappa: SARIMA.

### RNP — ES → CE — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 13.363567190574386 | 8.0517230919423639 | 0.88375195747477109 | 0 | 1/1 |
| GRU | 14.000100504346571 | 9.6122658036821687 | 1.0548922672822696 | 0.0634330634248641 | 5/5 |
| LSTM | 28.857540899490481 | 26.072164112367279 | 2.8913182858199238 | 0 | 5/5 |
| Forwarding LSTM | 15.67367979063398 | 12.411859651745836 | 1.3618383534514531 | 0 | 5/5 |
| Transformer | 24.043498864198057 | 19.77460775291134 | 2.139099386345948 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: SARIMA; MAE: SARIMA; SMAPE: SARIMA; Cohen_Kappa: GRU.

### RNP — GO → SP — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 10.939070708400218 | 8.0621951691498808 | 0.83282923841678347 | ND | 0/1 |
| GRU | 18.303755455316622 | 16.622945822074609 | 1.7012180166959241 | ND | 0/5 |
| LSTM | 13.386660801676481 | 11.530232327391031 | 1.1835450051070795 | ND | 0/5 |
| Forwarding LSTM | 9.4183549807684432 | 7.1314628796115329 | 0.73626567939914767 | ND | 0/5 |
| Transformer | 24.231101936213051 | 22.926415306869504 | 2.3357137120375162 | 0 | 2/5 |

Melhores valores neste link/contexto: RMSE: Forwarding LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Transformer.

### RNP — MS → TO — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 13.495426953102752 | 8.243035031678005 | 0.90603047947283921 | 0 | 1/1 |
| GRU | 14.652417646755309 | 9.6504314893183736 | 1.0598211893684009 | -0.0074766355140186 | 5/5 |
| LSTM | 13.326404606525738 | 7.459895704950962 | 0.82007972403432794 | 0 | 5/5 |
| Forwarding LSTM | 14.466410604202826 | 10.662259787158488 | 1.1716460873180652 | 0 | 5/5 |
| Transformer | 15.324904654479088 | 9.7945263210897444 | 1.0720417147750108 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: Forwarding LSTM, LSTM, SARIMA, Transformer.

### RNP — SC → TO — contexto 3

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 14.995187770676392 | 9.9836411653592165 | 1.0662017017641183 | 0 | 1/1 |
| GRU | 20.0253346941304 | 17.134877541729352 | 1.8339366901688059 | 0 | 5/5 |
| LSTM | 14.800228849460511 | 9.6343942739558752 | 1.0278344728674116 | 0 | 5/5 |
| Forwarding LSTM | 15.056945083601978 | 9.9025788100276557 | 1.0563146158715149 | 0 | 5/5 |
| Transformer | 18.154631043941929 | 13.459398780934752 | 1.4302314702298815 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: Forwarding LSTM, GRU, LSTM, SARIMA, Transformer.

### RNP — AC → TO — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 78.958036376457613 | 55.558596153558533 | 7.244242356681017 | 0.198943661971831 | 1/1 |
| GRU | 135.91969408785283 | 126.86500745125312 | 15.784458615857272 | -0.021944746124498 | 5/5 |
| LSTM | 135.24629186342204 | 125.41581529372031 | 15.618985811087118 | -0.020323754152684101 | 5/5 |
| Forwarding LSTM | 155.66872527062569 | 149.0187999193478 | 18.289421268533289 | -0.0091121594682592004 | 5/5 |
| Transformer | 155.09061609865276 | 147.92843928006408 | 18.152399044146609 | 0.0123939051545088 | 5/5 |

Melhores valores neste link/contexto: RMSE: SARIMA; MAE: SARIMA; SMAPE: SARIMA; Cohen_Kappa: SARIMA.

### RNP — ES → CE — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 13.363567190574386 | 8.0517230919423639 | 0.88375195747477109 | 0 | 1/1 |
| GRU | 13.353584482325552 | 8.3045478758847189 | 0.9113777047255226 | 0 | 5/5 |
| LSTM | 13.100912396382318 | 7.699644236651551 | 0.8453733299695122 | 0 | 5/5 |
| Forwarding LSTM | 13.152665176253803 | 7.8555153281677264 | 0.86236212922865829 | 0 | 5/5 |
| Transformer | 27.517673966165169 | 23.803594847255749 | 2.5658393591454471 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: LSTM; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: Forwarding LSTM, GRU, LSTM, SARIMA, Transformer.

### RNP — GO → SP — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 10.939070708400218 | 8.0621951691498808 | 0.83282923841678347 | ND | 0/1 |
| GRU | 7.684188968449865 | 5.8993855552454733 | 0.60932941921293493 | ND | 0/5 |
| LSTM | 7.7312290444658558 | 5.8900823445372694 | 0.60836076603643785 | ND | 0/5 |
| Forwarding LSTM | 7.9279530260257642 | 6.1055247656133549 | 0.63062408391531188 | ND | 0/5 |
| Transformer | 24.438332473972963 | 23.201736627120908 | 2.3674594120602448 | ND | 0/5 |

Melhores valores neste link/contexto: RMSE: GRU; MAE: LSTM; SMAPE: LSTM; Cohen_Kappa: ND para todos.

### RNP — MS → TO — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 13.495426953102752 | 8.243035031678005 | 0.90603047947283921 | 0 | 1/1 |
| GRU | 13.265476321714194 | 8.3922524510661827 | 0.92237282973128765 | 0 | 5/5 |
| LSTM | 13.875998363106737 | 7.9467427232491534 | 0.87264872756629686 | 0 | 5/5 |
| Forwarding LSTM | 12.759052415111594 | 7.2079146828071776 | 0.7927639934656121 | 0 | 5/5 |
| Transformer | 14.123820412173483 | 8.2172469796922467 | 0.90180056147260923 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: Forwarding LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Forwarding LSTM, GRU, LSTM, SARIMA, Transformer.

### RNP — SC → TO — contexto 28

| Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | SMAPE (%) | Cohen’s kappa | Sementes κ |
| --- | --- | --- | --- | --- | --- |
| SARIMA | 14.995187770676392 | 9.9836411653592165 | 1.0662017017641183 | 0 | 1/1 |
| GRU | 14.73180768202406 | 9.3592091512473647 | 0.9987034306997582 | 0 | 5/5 |
| LSTM | 14.859968977802165 | 9.5904512710810259 | 1.0231490447257312 | 0 | 5/5 |
| Forwarding LSTM | 14.701280110845682 | 9.3319327155362259 | 0.99573522166833639 | 0 | 5/5 |
| Transformer | 15.053885289046054 | 10.043395093807758 | 1.0711422830276325 | 0 | 5/5 |

Melhores valores neste link/contexto: RMSE: Forwarding LSTM; MAE: Forwarding LSTM; SMAPE: Forwarding LSTM; Cohen_Kappa: Forwarding LSTM, GRU, LSTM, SARIMA, Transformer.

## 7. Variação entre sementes do Forwarding LSTM

Desvio-padrão amostral entre as cinco sementes (`ddof=1`), calculado por link. Não é intervalo de confiança nem teste de significância. `ND` ocorre se menos de duas sementes têm kappa definido.

| Conjunto | Contexto | Link | DP RMSE | DP MAE | DP SMAPE | DP Cohen_Kappa |
| --- | --- | --- | --- | --- | --- | --- |
| GEANT | 3 | Ankara → Poznań | 0.95343435817345901 | 0.45499917569566628 | 0.057741285910169936 | 0 |
| GEANT | 3 | RedIRIS → Lisboa | 7.0612434863291487 | 4.3521592970174439 | 0.22048198497376556 | 0.052851344105580435 |
| GEANT | 3 | Lisboa → Ankara | 0.27638295251032169 | 0.23476899603568557 | 0.025819233519019484 | 0 |
| GEANT | 3 | Poznań → Ankara | 0.027731554263798291 | 0.025255957811930348 | 0.0028081269534685864 | 0.033110915229804153 |
| GEANT | 3 | Poznań → Anella | 0.12578441746944496 | 0.058666510389102894 | 0.006829235380581466 | 0 |
| GEANT | 3 | Viena → LUT | 2.257035386229755 | 2.635728278977961 | 1.6322122052740149 | 0 |
| GEANT | 28 | Ankara → Poznań | 0.75855495440859388 | 0.49088679813765046 | 0.060912671586903569 | 0 |
| GEANT | 28 | RedIRIS → Lisboa | 11.757439741816144 | 11.535070699164615 | 0.59361193596219419 | 0.0011578702945091682 |
| GEANT | 28 | Lisboa → Ankara | 0.13320787446096283 | 0.17496444161654037 | 0.020157547606675794 | 0 |
| GEANT | 28 | Poznań → Ankara | 0.20313080880707998 | 0.22497891095317363 | 0.024986486552212077 | 0.046820195918768182 |
| GEANT | 28 | Poznań → Anella | 0.67949806137463142 | 1.5278578647223253 | 0.17601980941386339 | 0 |
| GEANT | 28 | Viena → LUT | 1.1125077182356964 | 1.1336017538466976 | 0.76519527989867808 | 0 |
| RNP | 3 | AC → TO | 3.4858615007919549 | 3.9444319100041993 | 0.44254878865408348 | 0.0026476128198109471 |
| RNP | 3 | ES → CE | 0.040513252737515895 | 0.083668050630045149 | 0.0092428974162675186 | 0 |
| RNP | 3 | GO → SP | 0.73365132252344334 | 0.57514008047103959 | 0.059180677620890111 | ND |
| RNP | 3 | MS → TO | 0.3827183497940207 | 0.58421139599858996 | 0.064266336664779711 | 0 |
| RNP | 3 | SC → TO | 0.12141862003742776 | 0.10880378673227036 | 0.011539259319963753 | 0 |
| RNP | 28 | AC → TO | 2.2267922981176427 | 2.4579165193678354 | 0.27314817714473605 | 0.0057231203581538589 |
| RNP | 28 | ES → CE | 0.046177119348835151 | 0.17130661856613336 | 0.018675794135167498 | 0 |
| RNP | 28 | GO → SP | 0.33856606229443076 | 0.21012151645806346 | 0.021725258057440834 | ND |
| RNP | 28 | MS → TO | 0.041454559450202516 | 0.20516487664107419 | 0.022418616913286647 | 0 |
| RNP | 28 | SC → TO | 0.020903015517311014 | 0.030005853874118766 | 0.0031863414844973824 | 0 |

## 8. Limites de interpretação

1. As conclusões se restringem aos links selecionados, às datas de teste e às configurações executadas. A remoção de links GEANT após a análise pode alterar médias e posições; não é evidência de superioridade no conjunto GEANT completo.
2. As tabelas não fornecem significância estatística. Há dependência temporal entre intervalos e somente cinco sementes das redes, que compartilham a configuração selecionada na semente 42.
3. Comparações por kappa dependem das faixas de 100 Mbit/s e da distribuição de classes. Baixo erro de vazão pode coexistir com kappa baixo ou indefinido quando a vazão está concentrada em poucas classes. O número de valores definidos precisa acompanhar a média.
4. Os modelos diferem em perda, arquitetura, orçamento de busca e, no caso de SARIMA, representação do histórico. O protocolo controla datas, teste por link e realimentação; não isola exclusivamente o efeito da arquitetura.
5. A comparação é univariada por link. Os resultados anteriores de iTransformer, TimeBridge e TimeMixer++ não integram estas figuras e não devem ser incorporados às mesmas médias sem compatibilizar o protocolo.

## 9. Arquivos e rastreabilidade

- [Dados numéricos das figuras](metrics.csv).
- [Médias globais deste relatório](medias_relatorio.csv).
- [Melhores modelos por link e métrica](vencedores_por_link.csv).
- [Legendas das figuras](CAPTIONS.md).
- [Manifesto dos gráficos e templates](manifest.json).
- [Protocolo completo](../../../../CONTROLLED.md).
- [Auditoria dos experimentos](../../audit.json): 1.088 execuções originais verificadas, 20 pares modelo/contexto restaurados e nenhuma execução ausente. Este relatório utiliza o subconjunto de 462 execuções que corresponde aos modelos e links exibidos.

SHA-256 do CSV das figuras: `acbef883857364f9771d1546b6d53e630b20ca50fc8d9ea5cc9e863e92e7a7e3`.

### Figuras correspondentes

| Conjunto | Contexto | RMSE | MAE | SMAPE | Cohen’s kappa |
| --- | --- | --- | --- | --- | --- |
| GEANT | 3 | [RMSE](geant_context_3_rmse.png) | [MAE](geant_context_3_mae.png) | [SMAPE](geant_context_3_smape.png) | [COHEN_KAPPA](geant_context_3_cohen_kappa.png) |
| GEANT | 28 | [RMSE](geant_context_28_rmse.png) | [MAE](geant_context_28_mae.png) | [SMAPE](geant_context_28_smape.png) | [COHEN_KAPPA](geant_context_28_cohen_kappa.png) |
| RNP | 3 | [RMSE](rnp_context_3_rmse.png) | [MAE](rnp_context_3_mae.png) | [SMAPE](rnp_context_3_smape.png) | [COHEN_KAPPA](rnp_context_3_cohen_kappa.png) |
| RNP | 28 | [RMSE](rnp_context_28_rmse.png) | [MAE](rnp_context_28_mae.png) | [SMAPE](rnp_context_28_smape.png) | [COHEN_KAPPA](rnp_context_28_cohen_kappa.png) |

Para regenerar este relatório a partir dos arquivos salvos, execute na raiz do projeto:

```powershell
& .\ijsm_review\.venv-experiments\Scripts\python.exe .\ijsm_review\describe_template_results.py
```

### Identificadores completos dos links exibidos

| Conjunto | Rótulo | Identificador original |
| --- | --- | --- |
| GEANT | Ankara → Poznań | esmond data perfsonar-ankara.ulakbim.gov.tr to psmp-gn-bw-poz-pl.geant.org 10-24-2023_6h |
| GEANT | RedIRIS → Lisboa | esmond data perfsonar-sonda.rediris.es to psmp-gn-bw-lis-pt.geant.org 10-24-2023_6h |
| GEANT | Lisboa → Ankara | esmond data psmp-gn-bw-lis-pt.geant.org to perfsonar-ankara.ulakbim.gov.tr 10-24-2023_6h |
| GEANT | Poznań → Ankara | esmond data psmp-gn-bw-poz-pl.geant.org to perfsonar-ankara.ulakbim.gov.tr 10-24-2023_6h |
| GEANT | Poznań → Anella | esmond data psmp-gn-bw-poz-pl.geant.org to pspmp-anella.csuc.cat 10-24-2023_6h |
| GEANT | Viena → LUT | esmond data psmp-gn-bw-vie-at.geant.org to psmall.lut.ac.uk 10-24-2023_6h |
| RNP | AC → TO | ac-to_bbr_interval_0_6h |
| RNP | ES → CE | es-ce_bbr_interval_0_6h |
| RNP | GO → SP | go-sp_bbr_interval_0_6h |
| RNP | MS → TO | ms-to_bbr_interval_0_6h |
| RNP | SC → TO | sc-to_bbr_interval_0_6h |

