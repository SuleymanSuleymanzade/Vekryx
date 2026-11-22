mod api;
mod storage;

use actix_web::{web, App, HttpServer};
use std::{env, io, time::Duration};

#[actix_web::main]
async fn main() -> io::Result<()> {
    let bind = env::var("VEKRYX_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let directory = env::var("VEKRYX_DATA_DIR").unwrap_or_else(|_| "server/data".into());
    let ollama = env::var("VEKRYX_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
    // The blocking HTTP client owns its own runtime.
    let state = web::block(move || api::State::open(directory.as_ref(), &ollama))
        .await
        .map_err(io::Error::other)??;
    let state = web::Data::new(state);
    let server_state = state.clone();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(server_state.clone())
            .configure(api::routes)
    })
    .client_request_timeout(Duration::from_secs(10))
    .bind(bind);
    let result = match server {
        Ok(server) => server.run().await,
        Err(error) => Err(error),
    };
    web::block(move || drop(state))
        .await
        .map_err(io::Error::other)?;
    result
}
