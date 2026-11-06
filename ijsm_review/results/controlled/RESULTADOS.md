# Resultados da comparação controlada

Cada linha abaixo é a média entre links avaliáveis, após a média das sementes por link.
GEANT: dez links. SBRC: cinco links avaliáveis; es-pr recebe previsões, mas não tem observações no teste.
Modelos estocásticos usam cinco sementes; os determinísticos, uma execução.

| Conjunto | Contexto | Modelo | RMSE (Mbit/s) | MAE (Mbit/s) | NRMSE | SMAPE (%) |
|---|---:|---|---:|---:|---:|---:|
| GEANT | 3 | ARIMA | 100.587 | 57.867 | 0.1273 | 8.247 |
| GEANT | 3 | Forwarding LSTM | 81.550 | 52.310 | 0.1259 | 8.764 |
| GEANT | 3 | GRU | 87.557 | 57.555 | 0.1285 | 9.048 |
| GEANT | 3 | LSTM | 124.116 | 95.867 | 0.1481 | 11.396 |
| GEANT | 3 | MLP | 191.651 | 166.849 | 0.2207 | 21.787 |
| GEANT | 3 | Persistência | 124.650 | 73.639 | 0.1565 | 10.277 |
| GEANT | 3 | Random Forest | 88.681 | 56.790 | 0.1273 | 8.997 |
| GEANT | 3 | Regressão Ridge | 95.996 | 56.355 | 0.1252 | 8.497 |
| GEANT | 3 | SARIMA | 102.661 | 60.068 | 0.1292 | 8.438 |
| GEANT | 3 | Transformer padrão | 87.677 | 58.598 | 0.1358 | 9.554 |
| GEANT | 28 | ARIMA | 100.587 | 57.867 | 0.1273 | 8.247 |
| GEANT | 28 | Forwarding LSTM | 88.824 | 60.515 | 0.1224 | 8.641 |
| GEANT | 28 | GRU | 142.266 | 114.793 | 0.1588 | 12.382 |
| GEANT | 28 | LSTM | 91.186 | 62.206 | 0.1263 | 8.939 |
| GEANT | 28 | MLP | 116.143 | 89.108 | 0.1460 | 11.380 |
| GEANT | 28 | Persistência | 124.650 | 73.639 | 0.1565 | 10.277 |
| GEANT | 28 | Random Forest | 82.616 | 53.133 | 0.1227 | 8.580 |
| GEANT | 28 | Regressão Ridge | 98.784 | 59.441 | 0.1272 | 8.698 |
| GEANT | 28 | SARIMA | 102.661 | 60.068 | 0.1292 | 8.438 |
| GEANT | 28 | Transformer padrão | 89.373 | 58.492 | 0.1300 | 8.985 |
| SBRC | 3 | ARIMA | 26.647 | 18.499 | 0.0328 | 2.246 |
| SBRC | 3 | Forwarding LSTM | 41.121 | 36.774 | 0.0521 | 4.405 |
| SBRC | 3 | GRU | 41.078 | 36.472 | 0.0513 | 4.343 |
| SBRC | 3 | LSTM | 43.473 | 38.800 | 0.0545 | 4.624 |
| SBRC | 3 | MLP | 35.880 | 30.416 | 0.0447 | 3.670 |
| SBRC | 3 | Persistência | 24.080 | 16.557 | 0.0288 | 2.024 |
| SBRC | 3 | Random Forest | 43.176 | 36.750 | 0.0540 | 4.403 |
| SBRC | 3 | Regressão Ridge | 42.772 | 38.437 | 0.0545 | 4.592 |
| SBRC | 3 | SARIMA | 26.350 | 17.980 | 0.0324 | 2.187 |
| SBRC | 3 | Transformer padrão | 50.094 | 45.764 | 0.0626 | 5.354 |
| SBRC | 28 | ARIMA | 26.647 | 18.499 | 0.0328 | 2.246 |
| SBRC | 28 | Forwarding LSTM | 40.842 | 35.904 | 0.0521 | 4.314 |
| SBRC | 28 | GRU | 36.991 | 31.764 | 0.0469 | 3.845 |
| SBRC | 28 | LSTM | 36.963 | 31.309 | 0.0469 | 3.794 |
| SBRC | 28 | MLP | 27.119 | 20.378 | 0.0334 | 2.487 |
| SBRC | 28 | Persistência | 24.080 | 16.557 | 0.0288 | 2.024 |
| SBRC | 28 | Random Forest | 39.582 | 32.292 | 0.0493 | 3.905 |
| SBRC | 28 | Regressão Ridge | 41.642 | 37.224 | 0.0530 | 4.459 |
| SBRC | 28 | SARIMA | 26.350 | 17.980 | 0.0324 | 2.187 |
| SBRC | 28 | Transformer padrão | 47.245 | 42.639 | 0.0589 | 5.012 |

São execuções únicas de treinamento por semente; diferenças pequenas precisam ser interpretadas com a variação entre sementes e links.
ARIMA/SARIMA usam histórico completo, independentemente da coluna contexto.
LSTM/GRU/Forwarding LSTM são portas PyTorch com validação comum; os resultados antigos ficam em pastas separadas.
A auditoria dos artefatos está em audit.json. As figuras estão em plots/.
