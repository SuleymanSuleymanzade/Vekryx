use crate::storage::{valid_name, Collection, Entry, Metric, Storage};
use actix_web::{error::JsonPayloadError, http::StatusCode, web, HttpResponse, ResponseError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    fmt, io,
    path::Path,
    sync::{Arc, RwLock},
    time::Duration,
};
use vekryx::{
    chunking::{ChunkingStrategy, FixedSizeChunker},
    store::Record,
    Vekryx,
};

pub struct State {
    storage: Storage,
    http: reqwest::blocking::Client,
    embed_url: reqwest::Url,
}

impl State {
    pub fn open(directory: &Path, ollama: &str) -> io::Result<Self> {
        let embed_url = reqwest::Url::parse(&format!("{}/api/embed", ollama.trim_end_matches('/')))
            .map_err(io::Error::other)?;
        let storage = Storage::open(directory)?;
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(io::Error::other)?;
        Ok(Self {
            storage,
            http,
            embed_url,
        })
    }

    fn embed(&self, model: &str, input: &[&str]) -> Result<Vec<Vec<f64>>, ApiError> {
        #[derive(Deserialize)]
        struct Embeddings {
            embeddings: Vec<Vec<f64>>,
        }
        let output: Embeddings = self
            .http
            .post(self.embed_url.clone())
            .json(&json!({"model": model, "input": input, "truncate": false}))
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::json)
            .map_err(|error| {
                eprintln!("embedding request failed: {error}");
                ApiError::embedding("Embedding provider request failed")
            })?;
        if output.embeddings.len() != input.len() {
            return Err(ApiError::embedding(
                "Embedding provider returned the wrong number of vectors",
            ));
        }
        Ok(output.embeddings)
    }

    fn entry(&self, name: &str) -> Result<Entry, ApiError> {
        self.storage
            .collections
            .read()
            .map_err(ApiError::internal)?
            .get(name)
            .cloned()
            .ok_or_else(ApiError::missing_collection)
    }

    fn with_current<T>(
        &self,
        name: &str,
        entry: &Entry,
        operation: impl FnOnce() -> Result<T, ApiError>,
    ) -> Result<T, ApiError> {
        let collections = self
            .storage
            .collections
            .read()
            .map_err(ApiError::internal)?;
        if !collections
            .get(name)
            .is_some_and(|current| Arc::ptr_eq(current, entry))
        {
            return Err(ApiError::missing_collection());
        }
        operation()
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_input", message)
    }

    fn missing_collection() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "collection_not_found",
            "Collection not found",
        )
    }

    fn embedding(message: &str) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "embedding_failed", message)
    }

    fn internal(error: impl fmt::Display) -> Self {
        eprintln!("server error: {error}");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "Internal server error",
        )
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl ResponseError for ApiError {
    fn status_code(&self) -> StatusCode {
        self.status
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status).json(json!({
            "error": {"code": self.code, "message": self.message}
        }))
    }
}

async fn run(
    operation: impl FnOnce() -> Result<(StatusCode, Value), ApiError> + Send + 'static,
) -> Result<HttpResponse, ApiError> {
    let (status, body) = web::block(operation).await.map_err(ApiError::internal)??;
    let mut response = HttpResponse::build(status);
    Ok(if status == StatusCode::NO_CONTENT {
        response.finish()
    } else {
        response.json(body)
    })
}

pub fn routes(config: &mut web::ServiceConfig) {
    config
        .app_data(
            web::JsonConfig::default()
                .limit(1024 * 1024)
                .error_handler(|error, _| {
                    match error {
                        JsonPayloadError::Overflow { .. }
                        | JsonPayloadError::OverflowKnownLength { .. } => ApiError::new(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "payload_too_large",
                            "JSON body exceeds 1 MiB",
                        ),
                        JsonPayloadError::ContentType => ApiError::new(
                            StatusCode::UNSUPPORTED_MEDIA_TYPE,
                            "unsupported_media_type",
                            "Expected application/json",
                        ),
                        _ => ApiError::new(
                            StatusCode::BAD_REQUEST,
                            "invalid_json",
                            error.to_string(),
                        ),
                    }
                    .into()
                }),
        )
        .route("/healthz", web::get().to(health))
        .route("/readyz", web::get().to(health))
        .service(
            web::scope("/v1/collections")
                .route("", web::post().to(create))
                .route("", web::get().to(list))
                .route("/{name}", web::get().to(get))
                .route("/{name}", web::delete().to(delete))
                .route("/{name}/documents", web::post().to(ingest))
                .route("/{name}/documents/{id}", web::delete().to(delete_document))
                .route("/{name}/search", web::post().to(search)),
        );
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(json!({"status": "ok"}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Create {
    name: String,
    model: String,
    dimensions: Option<usize>,
    #[serde(default)]
    metric: Metric,
}

#[derive(Serialize)]
struct Summary<'a> {
    name: &'a str,
    model: &'a str,
    dimensions: usize,
    metric: Metric,
    chunk_count: usize,
}

fn summary<'a>(name: &'a str, collection: &'a Collection) -> Summary<'a> {
    Summary {
        name,
        model: &collection.model,
        dimensions: collection.store.dimensions(),
        metric: collection.metric,
        chunk_count: collection.store.len(),
    }
}

async fn create(
    state: web::Data<State>,
    body: web::Json<Create>,
) -> Result<HttpResponse, ApiError> {
    let body = body.into_inner();
    run(move || {
        if !valid_name(&body.name) {
            return Err(ApiError::invalid("Collection names must contain 1–128 ASCII letters, digits, hyphens, or underscores"));
        }
        if body.model.trim().is_empty() || body.model.len() > 256 || body.dimensions == Some(0) {
            return Err(ApiError::invalid("Supply a model name and, optionally, positive dimensions"));
        }
        let duplicate = || ApiError::new(StatusCode::CONFLICT, "collection_exists", "Collection already exists");
        if state.storage.collections.read().map_err(ApiError::internal)?.contains_key(&body.name) {
            return Err(duplicate());
        }
        let probe = state.embed(&body.model, &["embedding dimension probe"])?;
        let dimensions = probe[0].len();
        validate_embedding(&probe[0], dimensions, Metric::Cosine)?;
        if body.dimensions.is_some_and(|expected| expected != dimensions) {
            return Err(ApiError::invalid(format!("Model returned {dimensions} dimensions")));
        }
        let collection = Collection {
            model: body.model,
            metric: body.metric,
            store: Vekryx::new(dimensions).map_err(ApiError::internal)?,
        };
        let mut collections = state.storage.collections.write().map_err(ApiError::internal)?;
        if collections.contains_key(&body.name) {
            return Err(duplicate());
        }
        state.storage.save(&body.name, &collection).map_err(ApiError::internal)?;
        let response = (StatusCode::CREATED, json!(summary(&body.name, &collection)));
        collections.insert(body.name, Arc::new(RwLock::new(collection)));
        Ok(response)
    }).await
}

#[derive(Deserialize)]
struct Page {
    limit: Option<usize>,
    cursor: Option<String>,
}

async fn list(state: web::Data<State>, page: web::Query<Page>) -> Result<HttpResponse, ApiError> {
    let page = page.into_inner();
    run(move || {
        let limit = page.limit.unwrap_or(50);
        if !(1..=100).contains(&limit) {
            return Err(ApiError::invalid("limit must be between 1 and 100"));
        }
        let collections = state
            .storage
            .collections
            .read()
            .map_err(ApiError::internal)?;
        let mut entries = collections
            .iter()
            .filter(|(name, _)| page.cursor.as_ref().is_none_or(|cursor| *name > cursor));
        let mut items = Vec::new();
        let mut last = None;
        for (name, entry) in entries.by_ref().take(limit) {
            let collection = entry.read().map_err(ApiError::internal)?;
            items.push(
                serde_json::to_value(summary(name, &collection)).map_err(ApiError::internal)?,
            );
            last = Some(name);
        }
        let next_cursor = entries.next().and(last);
        Ok((
            StatusCode::OK,
            json!({"collections": items, "next_cursor": next_cursor}),
        ))
    })
    .await
}

async fn get(state: web::Data<State>, name: web::Path<String>) -> Result<HttpResponse, ApiError> {
    run(move || {
        let entry = state.entry(&name)?;
        let collection = entry.read().map_err(ApiError::internal)?;
        Ok((StatusCode::OK, json!(summary(&name, &collection))))
    })
    .await
}

async fn delete(
    state: web::Data<State>,
    name: web::Path<String>,
) -> Result<HttpResponse, ApiError> {
    run(move || {
        let mut collections = state
            .storage
            .collections
            .write()
            .map_err(ApiError::internal)?;
        if !collections.contains_key(name.as_str()) {
            return Err(ApiError::missing_collection());
        }
        state.storage.delete(&name).map_err(ApiError::internal)?;
        collections.remove(name.as_str());
        Ok((StatusCode::NO_CONTENT, Value::Null))
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    id: String,
    text: String,
    #[serde(default)]
    metadata: Map<String, Value>,
}

fn validate_embedding(vector: &[f64], dimensions: usize, metric: Metric) -> Result<(), ApiError> {
    if vector.len() != dimensions || metric.engine().distance(vector, vector).is_err() {
        return Err(ApiError::embedding(
            "Embedding provider returned an invalid vector or unexpected dimensions",
        ));
    }
    Ok(())
}

fn remove_chunks(store: &mut Vekryx, document_id: &str) -> bool {
    let ids: Vec<_> = store
        .records()
        .filter(|(_, record)| {
            record.metadata.get("document_id").and_then(Value::as_str) == Some(document_id)
        })
        .map(|(id, _)| id.clone())
        .collect();
    for id in &ids {
        store.delete(id);
    }
    !ids.is_empty()
}

async fn ingest(
    state: web::Data<State>,
    name: web::Path<String>,
    body: web::Json<Document>,
) -> Result<HttpResponse, ApiError> {
    let body = body.into_inner();
    run(move || {
        if !valid_name(&body.id) || body.text.trim().is_empty() || body.metadata.contains_key("document_id") {
            return Err(ApiError::invalid("Supply a valid document ID and nonempty text. Metadata key document_id is reserved"));
        }
        let entry = state.entry(&name)?;
        let (model, dimensions) = {
            let collection = entry.read().map_err(ApiError::internal)?;
            (collection.model.clone(), collection.store.dimensions())
        };
        let chunks = FixedSizeChunker::new(800, 100).map_err(ApiError::internal)?
            .chunk(&body.text, &body.id).map_err(ApiError::internal)?;
        if chunks.len() > 256 {
            return Err(ApiError::invalid("A document may produce at most 256 chunks"));
        }
        let mut embeddings = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(32) {
            let texts: Vec<_> = batch.iter().map(|chunk| chunk.text.as_str()).collect();
            for vector in state.embed(&model, &texts)? {
                validate_embedding(&vector, dimensions, Metric::Cosine)?;
                embeddings.push(vector);
            }
        }
        state.with_current(&name, &entry, || {
            let mut current = entry.write().map_err(ApiError::internal)?;
            let mut updated = current.clone();
            let replaced = remove_chunks(&mut updated.store, &body.id);
            let chunk_count = chunks.len();
            let mut metadata = body.metadata;
            metadata.insert("document_id".into(), json!(body.id));
            for (chunk, vector) in chunks.into_iter().zip(embeddings) {
                updated.store.insert_record(chunk.id, vector, Record {
                    text: Some(chunk.text),
                    metadata: metadata.clone(),
                }).map_err(ApiError::internal)?;
            }
            state.storage.save(&name, &updated).map_err(ApiError::internal)?;
            *current = updated;
            let status = if replaced { StatusCode::OK } else { StatusCode::CREATED };
            Ok((status, json!({"document_id": body.id, "chunk_count": chunk_count})))
        })
    }).await
}

async fn delete_document(
    state: web::Data<State>,
    path: web::Path<(String, String)>,
) -> Result<HttpResponse, ApiError> {
    let (name, id) = path.into_inner();
    run(move || {
        let entry = state.entry(&name)?;
        state.with_current(&name, &entry, || {
            let mut current = entry.write().map_err(ApiError::internal)?;
            let mut updated = current.clone();
            if !remove_chunks(&mut updated.store, &id) {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    "document_not_found",
                    "Document not found",
                ));
            }
            state
                .storage
                .save(&name, &updated)
                .map_err(ApiError::internal)?;
            *current = updated;
            Ok((StatusCode::NO_CONTENT, Value::Null))
        })
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    query: String,
    k: Option<usize>,
    filter: Option<Map<String, Value>>,
    metric: Option<Metric>,
    mmr: Option<Mmr>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mmr {
    candidates: Option<usize>,
    lambda: Option<f64>,
}

impl Mmr {
    fn parameters(&self, k: usize) -> Result<(usize, f64), ApiError> {
        let candidates = self.candidates.unwrap_or((4 * k).max(20));
        let lambda = self.lambda.unwrap_or(0.5);
        if !(k..=1000).contains(&candidates) {
            return Err(ApiError::invalid(
                "MMR candidates must be between k and 1000",
            ));
        }
        if !(0.0..=1.0).contains(&lambda) {
            return Err(ApiError::invalid("MMR lambda must be between 0 and 1"));
        }
        Ok((candidates, lambda))
    }
}

async fn search(
    state: web::Data<State>,
    name: web::Path<String>,
    body: web::Json<Search>,
) -> Result<HttpResponse, ApiError> {
    let body = body.into_inner();
    run(move || {
        let query = body.query.trim();
        let k = body.k.unwrap_or(10);
        if query.is_empty() || query.chars().count() > 8_000 || !(1..=100).contains(&k) {
            return Err(ApiError::invalid(
                "query must contain 1–8000 characters and k must be between 1 and 100",
            ));
        }
        let mmr = body.mmr.as_ref().map(|mmr| mmr.parameters(k)).transpose()?;
        let entry = state.entry(&name)?;
        let (model, metric, dimensions) = {
            let collection = entry.read().map_err(ApiError::internal)?;
            (
                collection.model.clone(),
                body.metric.unwrap_or(collection.metric),
                collection.store.dimensions(),
            )
        };
        let embeddings = state.embed(&model, &[query])?;
        let vector = &embeddings[0];
        validate_embedding(vector, dimensions, metric)?;
        state.with_current(&name, &entry, || {
            let collection = entry.read().map_err(ApiError::internal)?;
            let corpus = match &body.filter {
                Some(filter) => collection.store.filtered_view(filter),
                None => collection.store.view(),
            };
            let (mode, value_key, hits) = match mmr {
                Some((candidates, lambda)) => (
                    "mmr",
                    "distance",
                    corpus.mmr_search(vector, k, candidates, lambda, metric.engine()),
                ),
                None => (
                    "hybrid",
                    "score",
                    corpus.hybrid_search(query, vector, k, metric.engine(), 0.5, 60),
                ),
            };
            let hits = hits.map_err(ApiError::internal)?;
            let hits = hits
                .into_iter()
                .map(|(id, value)| {
                    let record = collection
                        .store
                        .record(&id)
                        .ok_or_else(|| ApiError::internal("Chunk record is missing"))?;
                    let mut metadata = record.metadata.clone();
                    let document_id = metadata
                        .remove("document_id")
                        .and_then(|value| match value {
                            Value::String(id) => Some(id),
                            _ => None,
                        })
                        .ok_or_else(|| ApiError::internal("Chunk document ID is missing"))?;
                    Ok(json!({
                        "chunk_id": id,
                        "document_id": document_id,
                        "text": record.text,
                        (value_key): value,
                        "metadata": metadata,
                    }))
                })
                .collect::<Result<Vec<_>, ApiError>>()?;
            Ok((
                StatusCode::OK,
                json!({"mode": mode, "metric": metric, "hits": hits}),
            ))
        })
    })
    .await
}
