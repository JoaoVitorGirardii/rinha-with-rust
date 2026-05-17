# ─── Stage 1: Build both binaries ────────────────────────────────────────────
FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# target-cpu=haswell ativa AVX2/BMI/FMA/POPCNT.
# Mac Mini 2014 (Haswell) e o host de dev (Comet Lake) suportam.
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo build --release --bin preprocess --bin api

# ─── Stage 2: Preprocess — build VP-Tree ─────────────────────────────────────
FROM debian:bookworm-slim AS preprocessor

WORKDIR /data
COPY --from=builder /app/target/release/preprocess /usr/local/bin/preprocess
COPY references.json.gz ./

RUN preprocess references.json.gz vptree.bin

# ─── Stage 3: Runtime image ───────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/api /usr/local/bin/api
COPY --from=preprocessor /data/vptree.bin /data/vptree.bin

ENV VPTREE_PATH=/data/vptree.bin

EXPOSE 8080

CMD ["/usr/local/bin/api"]
