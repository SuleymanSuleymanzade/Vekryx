# Vekryx HTTP server

To run the API with the browser UI and Ollama, see the
[Docker Compose instructions](../README.md#docker).

Run these commands from the repository root. Start Ollama and pull an embedding
model, then start the standalone Actix server. The server uses Vekryx for
storage and search and Ollama for embeddings.

```sh
ollama pull embeddinggemma
cargo run --manifest-path server/Cargo.toml --locked
```

| Environment variable | Default |
| --- | --- |
| `VEKRYX_BIND` | `127.0.0.1:8080` |
| `VEKRYX_DATA_DIR` | `server/data`, relative to the working directory |
| `VEKRYX_OLLAMA_URL` | `http://127.0.0.1:11434` |

The API assumes a trusted local network or an authenticated proxy. Use one
server process per data directory. Each collection is loaded into memory at
startup. Writes replace the collection snapshot before success is returned.

## Create, ingest, and search

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"notes","model":"embeddinggemma"}'

curl --fail-with-body http://127.0.0.1:8080/v1/collections/notes/documents \
  -H 'Content-Type: application/json' \
  -d '{"id":"storage","text":"Vekryx persists collections using atomic snapshots.","metadata":{"project":"vekryx"}}'

curl --fail-with-body http://127.0.0.1:8080/v1/collections/notes/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How is data persisted?","k":5,"filter":{"project":"vekryx"}}'
```

Collection creation probes Ollama to determine the embedding dimensions. You
can provide `dimensions` as an expected value. Use a stable model revision for
the lifetime of a collection. Choose `cosine` or `euclidean` with `metric` in
each search request. Requests that omit it use the stored default, normally
cosine.

Ingestion splits a document into up to 256 chunks of 800 Unicode characters
with 100 characters of overlap, then embeds them in batches. Posting the same
document ID replaces all of its previous chunks. Collection names and
document IDs accept ASCII letters, digits, hyphens, and underscores, with a
maximum length of 128. The metadata key `document_id` is reserved.

Search returns `mode`, `metric`, and `hits`. Each hit contains `chunk_id`,
`document_id`, `text`, and `metadata`. Hybrid search is the default and adds a
`score` to each hit. It uses reciprocal rank fusion of exact vector search
and BM25, orders by descending score, and resolves ties by chunk ID.

Include an `mmr` object for maximal marginal relevance. For example, send
`{"query":"How is data persisted?","k":5,"metric":"cosine","mmr":{"candidates":30,"lambda":0.35}}`.
`lambda` ranges from 0 (favor diversity) to 1 (favor relevance) and defaults
to 0.5. The candidate pool ranges from `k` to 1,000 and defaults to
`max(20, 4 * k)`. MMR returns the original query `distance` for each hit in
selection order.

The optional filter matches all supplied top-level metadata fields exactly.
It also accepts `document_id`. The default `k` is 10, with an allowed range
of 1 to 100.

| Method | Endpoint |
| --- | --- |
| GET | `/healthz`, `/readyz` |
| GET | `/v1/collections?limit=50&cursor=...` |
| POST | `/v1/collections` |
| GET, DELETE | `/v1/collections/{name}` |
| POST | `/v1/collections/{name}/documents` |
| DELETE | `/v1/collections/{name}/documents/{id}` |
| POST | `/v1/collections/{name}/search` |

The HTTP integration test starts a server and an Ollama-compatible test
provider on loopback. Run it with `cargo test --manifest-path
server/Cargo.toml --locked`.
