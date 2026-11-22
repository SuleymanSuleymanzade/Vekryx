use reqwest::{blocking::Client, StatusCode};
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

fn mock_embeddings() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        for connection in listener.incoming() {
            let mut connection = connection.unwrap();
            let mut request = BufReader::new(&mut connection);
            let mut line = String::new();
            request.read_line(&mut line).unwrap();
            assert!(line.starts_with("POST /api/embed "));
            let mut length = 0;
            loop {
                line.clear();
                request.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            std::io::Read::read_exact(&mut request, &mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["model"], "test-model");
            let embeddings: Vec<_> = body["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|text| {
                    if text.as_str().unwrap().contains("snapshot") {
                        [1, 0]
                    } else {
                        [0, 1]
                    }
                })
                .collect();
            let payload = json!({"embeddings": embeddings}).to_string();
            write!(connection, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len()).unwrap();
        }
    });
    format!("http://{address}")
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start(client: &Client, data: &std::path::Path, provider: &str) -> (Server, String) {
    let socket = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    drop(socket);
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_vekryx-server"))
            .env("VEKRYX_BIND", address.to_string())
            .env("VEKRYX_DATA_DIR", data)
            .env("VEKRYX_OLLAMA_URL", provider)
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let base = format!("http://{address}");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if client
            .get(format!("{base}/readyz"))
            .send()
            .is_ok_and(|response| response.status().is_success())
        {
            return (server, base);
        }
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited at startup"
        );
        assert!(Instant::now() < deadline, "server did not become ready");
        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn documents_are_chunked_ranked_filtered_replaced_and_reloaded() {
    let data = tempfile::tempdir().unwrap();
    let provider = mock_embeddings();
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let (server, base) = start(&client, data.path(), &provider);
    let collections = format!("{base}/v1/collections");
    let response = client
        .post(&collections)
        .json(&json!({"name": "notes", "model": "test-model"}))
        .send()
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.json::<Value>().unwrap()["dimensions"], 2);
    let notes = format!("{collections}/notes");
    let ingest = |body: Value| {
        client
            .post(format!("{notes}/documents"))
            .json(&body)
            .send()
            .unwrap()
    };
    let response = ingest(json!({
        "id": "storage",
        "text": "snapshot persistence ".repeat(60),
        "metadata": {"project": "vekryx"}
    }));
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.json::<Value>().unwrap()["chunk_count"], 2);
    assert_eq!(
        ingest(json!({"id": "other", "text": "Bananas are yellow."})).status(),
        StatusCode::CREATED
    );

    let search = |collection: &str, body: Value| -> Value {
        client
            .post(format!("{collection}/search"))
            .json(&body)
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap()
    };
    let result = search(&notes, json!({"query": "snapshot", "k": 3}));
    let hits = result["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0]["document_id"], "storage");
    assert!(hits[0]["text"].as_str().unwrap().contains("snapshot"));
    assert!(hits
        .windows(2)
        .all(|pair| pair[0]["score"].as_f64().unwrap() >= pair[1]["score"].as_f64().unwrap()));
    assert!(hits[0]["score"].as_f64().unwrap() > hits[2]["score"].as_f64().unwrap());

    let mmr = search(
        &notes,
        json!({
            "query": "snapshot", "k": 3, "metric": "euclidean",
            "mmr": {"candidates": 3, "lambda": 0.0}
        }),
    );
    assert_eq!(mmr["mode"], "mmr");
    assert_eq!(mmr["metric"], "euclidean");
    let hits = mmr["hits"].as_array().unwrap();
    assert_eq!(hits[0]["document_id"], "storage");
    assert_eq!(hits[1]["document_id"], "other");
    assert_eq!(hits[2]["document_id"], "storage");
    assert!((hits[1]["distance"].as_f64().unwrap() - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(hits[2]["distance"], 0.0);
    assert!(hits[1].get("score").is_none());

    let invalid = client
        .post(format!("{notes}/search"))
        .json(&json!({"query": "snapshot", "k": 3, "mmr": {"candidates": 2}}))
        .send()
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);

    assert_eq!(
        ingest(json!({
            "id": "storage", "text": "New snapshot text", "metadata": {"project": "vekryx"}
        }))
        .status(),
        StatusCode::OK
    );
    drop(server);

    let (_server, base) = start(&client, data.path(), &provider);
    let notes = format!("{base}/v1/collections/notes");
    let result = search(
        &notes,
        json!({"query": "snapshot", "filter": {"project": "vekryx"}}),
    );
    let hits = result["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["text"], "New snapshot text");
    assert_eq!(hits[0]["chunk_id"], "storage:chunk:0");
    assert_eq!(
        client
            .delete(format!("{notes}/documents/storage"))
            .send()
            .unwrap()
            .status(),
        StatusCode::NO_CONTENT
    );
    let result = search(
        &notes,
        json!({"query": "snapshot", "filter": {"project": "vekryx"}}),
    );
    assert_eq!(result["hits"], json!([]));
}
