//! Structured operational logs. Never include request bodies or recognized text.
use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use clap::ValueEnum;
use serde_json::{json, Value};
use std::{
    io::Write,
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
#[repr(u8)]
pub(crate) enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
}
static LEVEL: AtomicU8 = AtomicU8::new(3);
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
pub(crate) fn init(level: LogLevel) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}
pub(crate) fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
pub(crate) fn id() -> String {
    format!(
        "{:032x}-{:08x}-{:016x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}
pub(crate) fn emit(level: LogLevel, event: &str, fields: Value) {
    if level as u8 > LEVEL.load(Ordering::Relaxed) || matches!(level, LogLevel::Off) {
        return;
    }
    let mut record = fields;
    record["timestamp_unix_ms"] = json!(timestamp_ms());
    record["level"] = json!(match level {
        LogLevel::Error => "error",
        LogLevel::Warn => "warn",
        LogLevel::Info => "info",
        _ => "debug",
    });
    record["event"] = json!(event);
    if let Ok(mut bytes) = serde_json::to_vec(&record) {
        bytes.push(b'\n');
        // Broken log pipes must not turn successful OCR into a panic.
        let _ = std::io::stderr().lock().write_all(&bytes);
    }
}
#[derive(Clone)]
pub(crate) struct RequestContext {
    pub id: String,
    pub fields: Arc<Mutex<Value>>,
}
impl RequestContext {
    pub fn new() -> Self {
        Self {
            id: id(),
            fields: Arc::new(Mutex::new(json!({}))),
        }
    }
    pub fn set(&self, key: &str, value: Value) {
        if let Ok(mut f) = self.fields.lock() {
            f[key] = value;
        }
    }
}
#[derive(Clone)]
pub(crate) struct ErrorDetail(pub String);
pub(crate) async fn request_log(mut request: Request, next: Next) -> Response {
    let started = Instant::now();
    let context = RequestContext::new();
    let method = request.method().to_string();
    // Log route patterns, never query strings or arbitrary unmatched paths.
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str())
        .unwrap_or("<unmatched>")
        .to_owned();
    request.extensions_mut().insert(context.clone());
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "x-request-id",
        context.id.parse().expect("generated request ID"),
    );
    let status = response.status();
    let mut fields = context
        .fields
        .lock()
        .map(|x| x.clone())
        .unwrap_or(json!({}));
    fields["request_id"] = json!(context.id);
    fields["method"] = json!(method);
    fields["route"] = json!(route);
    fields["status"] = json!(status.as_u16());
    fields["elapsed_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    if let Some(error) = response.extensions().get::<ErrorDetail>() {
        fields["error"] = json!(error.0.chars().take(512).collect::<String>());
    }
    let level = if status.is_server_error() {
        LogLevel::Error
    } else if status.is_client_error() {
        LogLevel::Warn
    } else if route == "/health" || route == "/healthz" || method == "OPTIONS" {
        LogLevel::Debug
    } else {
        LogLevel::Info
    };
    emit(level, "request_complete", fields);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Extension, Router};
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn request_id_is_generated_and_attached_to_success_and_errors() {
        let app = Router::new()
            .route(
                "/ok",
                get(|Extension(ctx): Extension<RequestContext>| async move { ctx.id }),
            )
            .layer(axum::middleware::from_fn(request_log));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let ids=tokio::task::spawn_blocking(move||{
            use std::io::Read;
            let mut ids=vec![];
            for path in ["/ok","/missing?private=data"] {
                let mut stream=std::net::TcpStream::connect(addr).unwrap();
                stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                write!(stream,"GET {path} HTTP/1.1\r\nHost: localhost\r\nx-request-id: untrusted\r\nConnection: close\r\n\r\n").unwrap();
                let mut response=String::new();stream.read_to_string(&mut response).unwrap();
                let id=response.lines().find_map(|l|l.strip_prefix("x-request-id: ")).unwrap().to_owned();
                assert_ne!(id,"untrusted");
                if path=="/ok" {assert!(response.ends_with(&id));}else{assert!(response.contains("404 Not Found"));}
                ids.push(id);
            }ids
        }).await.unwrap();
        assert_ne!(ids[0], ids[1]);
        server.abort();
    }
}
