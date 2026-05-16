# CLAUDE.md — rinha-with-rust

## Histórico de performance

Após cada execução de `/home/joao/Documents/github-joao/rinha-de-backend-2026/run.sh`, registre os resultados em `performance-history.md` na raiz deste projeto.

**Campos a preencher** (extraídos de `test/results.json`):

| Campo | Caminho no JSON |
|-------|----------------|
| p99 (ms) | `.p99` |
| Score final | `.scoring.final_score` |
| Taxa de erro | `.scoring.failure_rate` |
| TP | `.scoring.breakdown.true_positive_detections` |
| TN | `.scoring.breakdown.true_negative_detections` |
| FP | `.scoring.breakdown.false_positive_detections` |
| FN | `.scoring.breakdown.false_negative_detections` |
| HTTP errors | `.scoring.breakdown.http_errors` |

**Fluxo:**
1. Garanta que o `docker-compose up` está rodando em `rinha-with-rust/`.
2. Execute `cd /home/joao/Documents/github-joao/rinha-de-backend-2026 && bash run.sh`.
3. Extraia os campos acima do JSON impresso.
4. Appende uma nova linha na tabela de `performance-history.md` com a data de hoje e os valores.
5. Adicione notas relevantes (ex.: o que mudou desde o último teste).

**Formatação — obrigatório:**
- Cada resultado é uma **seção `### YYYY-MM-DD · Teste #N`** com sua própria mini-tabela e a nota como parágrafo solto abaixo.
- Não misture linhas de tabelas diferentes em sequência — markdown quebra o parser se houver conteúdo entre linhas de uma mesma tabela.
- Separe cada seção com `---`.
- Colunas numéricas alinhadas à direita com `---------:` no separador.
- Exemplo de entrada bem formatada:
  ```markdown
  ### 2026-05-15 · Teste #2

  | p99 (ms) | Score final | Taxa de erro |   TP |   TN |   FP |   FN | HTTP errors |
  |---------:|------------:|--------------|-----:|-----:|-----:|-----:|------------:|
  |  2002.00 |       -6000 | 26.86%       | 6838 | 9683 | 1975 | 2306 |        1787 |

  Após fix do HAProxy (maxconn 1000). Latência alta e taxa de erro elevada.

  ---
  ```
