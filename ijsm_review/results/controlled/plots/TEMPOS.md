# Tempos de execução

Fonte: selection_seconds, training_seconds e inference_seconds dos run.json.

- Seleção: busca de configuração, treinamento de candidatos e validação; contada uma vez pelo registro da semente base.
- Treinamento: ajuste final após a seleção, incluindo construção do modelo/otimizador.
- Predição total: percurso online completo do teste, com realimentação, conversão de escala e atualizações de estado.
- Predição por intervalo: tempo total dividido por todos os intervalos previstos, inclusive os ausentes (48 em GEANT e 157 em RNP). É uma média do percurso, não uma medição isolada de latência.

Treinamento e predição são médias entre sementes por link. Persistência não possui treinamento ou seleção; seus pequenos tempos de preparação registrados não são apresentados como treinamento.

As execuções originais ocorreram em partições concorrentes na CPU; os valores representam tempos observados nessas condições, não um ensaio de desempenho com hardware isolado. Diferentes buscas, épocas selecionadas e tentativas de otimização também afetam os tempos.

Não somar selection_seconds dos registros de todas as sementes: o mesmo tempo de seleção foi repetido nos metadados. Tempos de gravação dos arquivos e métricas não estão incluídos.
