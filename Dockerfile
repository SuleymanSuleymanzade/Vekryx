FROM rust:1.95-bookworm AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY server/Cargo.toml server/Cargo.lock ./server/
COPY server/src ./server/src
RUN cargo build --release --locked --manifest-path server/Cargo.toml

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/lib/vekryx \
    && chown 10001:10001 /var/lib/vekryx
COPY --from=build /build/server/target/release/vekryx-server /usr/local/bin/vekryx-server
USER 10001:10001
ENV VEKRYX_BIND=0.0.0.0:8080 \
    VEKRYX_DATA_DIR=/var/lib/vekryx
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vekryx-server"]
