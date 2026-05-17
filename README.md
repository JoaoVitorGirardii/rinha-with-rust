# rinha-with-rust

API HTTP em Rust para **detecção de fraude em transações** via busca de vizinhos mais próximos (kNN) sobre um conjunto de 3 milhões de transações de referência rotuladas.

```
POST /fraud-score

{
  "transaction":      { "amount": 150.0, "installments": 1, "requested_at": "..." },
  "customer":         { "avg_amount": 200.0, "tx_count_24h": 3, "known_merchants": [...] },
  "merchant":         { "id": "m1", "mcc": "5411", "avg_amount": 180.0 },
  "terminal":         { "is_online": true, "card_present": true, "km_from_home": 2.5 },
  "last_transaction": { "timestamp": "...", "km_from_current": 1.0 }
}

→ { "approved": true, "fraud_score": 0.2 }
```

`fraud_score` = (vizinhos rotulados como fraude entre os 5 mais próximos) / 5. A transação é aprovada se o score for `< 0.6`.

## Como funciona

Cada transação é convertida em um **vetor de 14 dimensões** (amount normalizado, MCC risk, hora do dia, distância da casa, indicadores binários, etc.). A API responde fazendo uma busca **exata** dos 5 vetores de referência mais próximos pela distância euclidiana.

### Specialist Partitioning + VP-Tree

Uma VP-Tree única sobre 3 M pontos seria custosa de percorrer e perderia localidade de cache. O índice é **particionado em até 256 buckets** por uma chave de 8 bits derivada das features categóricas da transação:

| bit | feature |
|---|---|
| 0 | `last_transaction` presente |
| 1 | `terminal.is_online` |
| 2 | `terminal.card_present` |
| 3 | `unknown_merchant` |
| 4-5 | bucket de MCC risk (4 níveis) |
| 6 | `high_value` (amount/avg_amount > 5) |
| 7 | `frequent_tx` (tx_count_24h > 10) |

Cada partição guarda sua **bounding box** (min/max por dimensão) e tem sua própria VP-Tree. A busca é **key-first**:

1. Explora a partição cuja chave casa com a query.
2. Calcula `lower_bound²(query, bbox)` das outras partições, ordena, e visita só enquanto a cota inferior for menor que a pior distância no top-5 atual.

O resultado é exato — a chave serve para priorizar, não para filtrar.

### Otimizações

- **Quantização para `i16` com `SCALE=10000`** — o vetor cabe em `[i16; 16]` (32 B, alinhado para SIMD). O índice serializado fica em ~144 MB.
- **Distância via `_mm_madd_epi16` (SSE)** — duas cargas de 128 bits cobrem o vetor inteiro; cada instrução `madd` faz `(a-b)*(a-b)` e soma pares de lanes. Evita o downclock de AVX-256 sob limitação de CPU.
- **`mmap` do índice binário** — o `vptree.bin` é gerado em build time e mapeado direto na memória pelo processo, sem parse JSON em runtime. Warmup sequencial das páginas antes de responder `/ready`.
- **Layout BFS** dos nós dentro de cada partição — melhora localidade de cache na travessia.
- **`bytemuck`** para serialização zero-overhead — cast direto de bytes para `&[Node]`.
- **Respostas HTTP pré-computadas** — 6 strings estáticas para `fraud_score ∈ {0.0, 0.2, …, 1.0}`.
- **Lookup O(1) da partição primária** via tabela `key_to_idx[256]`.

## Arquitetura

```
┌─────────┐     ┌─────────┐    Unix    ┌──────────┐
│ client  │────▶│ haproxy │   socket   │  api-1   │
└─────────┘     │  :9999  │───────────▶│  (Axum)  │
                │         │            └──────────┘
                │         │            ┌──────────┐
                │         │───────────▶│  api-2   │
                └─────────┘            │  (Axum)  │
                                       └──────────┘
                                            │
                                            ▼
                                       ┌──────────┐
                                       │ vptree   │
                                       │ .bin     │ (mmap, 144 MB)
                                       └──────────┘
```

- **HAProxy** distribui requests round-robin pelas instâncias da API via Unix sockets (sem overhead de TCP local).
- **api-1** e **api-2** compartilham o mesmo binário e o mesmo `vptree.bin` (cada uma faz seu próprio mmap). Servidor HTTP usa **Axum + Tokio**.

## Pipeline de build (multi-stage Dockerfile)

1. **builder** (`rust:1-bookworm`) compila dois binários em release: `preprocess` e `api`.
2. **preprocessor** (`debian:bookworm-slim`) executa `preprocess references.json.gz vptree.bin` — parseia o JSON, agrupa as 3 M referências por chave de partição, constrói uma VP-Tree por bucket, quantiza, e escreve o `vptree.bin` (header + tabela de partições + nodes).
3. **runtime** (`debian:bookworm-slim`) copia só o binário `api` e o `vptree.bin`. Imagem final ~222 MB.

Builder e runtime ambos em Bookworm para alinhar a versão da glibc (2.36).

## Como rodar

```bash
docker compose build
docker compose up -d

curl http://localhost:9999/ready                  # 200 OK
curl -X POST http://localhost:9999/fraud-score \
     -H "Content-Type: application/json" \
     -d @transaction.json
```

## Estrutura do código

```
src/
├── main.rs              # API: Axum, mmap, key-first kNN, SIMD distance
└── bin/preprocess.rs    # Build do índice: parse JSON → quantize → partition → VP-Tree → bin
```
