//! Opt-in bounded case storage. Only the background writer touches the filesystem.
use crate::observability::{emit, timestamp_ms, LogLevel, RequestContext};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, SyncSender},
        Arc,
    },
    thread::JoinHandle,
};
const MAX_QUEUED_BYTES: usize = 32 * 1024 * 1024;
const MAX_CASES: usize = 1024;
const MAX_CASE_BYTES: usize = 16 * 1024 * 1024;
const PER_REQUEST: usize = 16;

struct Job {
    name: String,
    bytes: Vec<u8>,
}
pub(crate) struct Diagnostics {
    sender: Option<SyncSender<Job>>,
    queued_bytes: Arc<AtomicUsize>,
    worker: Option<JoinHandle<()>>,
}
impl Diagnostics {
    pub fn open(path: Option<&Path>, max_bytes: u64) -> Result<Self> {
        let queued_bytes = Arc::new(AtomicUsize::new(0));
        let Some(path) = path else {
            return Ok(Self {
                sender: None,
                queued_bytes,
                worker: None,
            });
        };
        let mut disk = Disk::open(path, max_bytes)?;
        let (sender, receiver) = mpsc::sync_channel::<Job>(32);
        let pending = queued_bytes.clone();
        let worker = std::thread::Builder::new()
            .name("ocr-diagnostics".into())
            .spawn(move || {
                for job in receiver {
                    let size = job.bytes.len();
                    match disk.save(&job.name, &job.bytes) {
                        Ok(()) => emit(
                            LogLevel::Info,
                            "diagnostic_saved",
                            json!({"case_id":job.name,"bytes":size}),
                        ),
                        Err(e) => emit(
                            LogLevel::Warn,
                            "diagnostic_write_failed",
                            json!({"case_id":job.name,"error":e.to_string()}),
                        ),
                    }
                    pending.fetch_sub(size, Ordering::Relaxed);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            queued_bytes,
            worker: Some(worker),
        })
    }
    pub fn enabled(&self) -> bool {
        self.sender.is_some()
    }
    fn submit(&self, name: String, metadata: Value) -> bool {
        let Some(sender) = &self.sender else {
            return false;
        };
        let Ok(bytes) = serde_json::to_vec(&metadata) else {
            return false;
        };
        let size = bytes.len();
        if size > MAX_CASE_BYTES {
            return false;
        }
        if self
            .queued_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                n.checked_add(size).filter(|&n| n <= MAX_QUEUED_BYTES)
            })
            .is_err()
        {
            return false;
        }
        if sender.try_send(Job { name, bytes }).is_err() {
            self.queued_bytes.fetch_sub(size, Ordering::Relaxed);
            return false;
        }
        true
    }
    pub fn observe(
        &self,
        ctx: &RequestContext,
        kind: &str,
        images: &[String],
        result: &Value,
        options: Value,
    ) {
        let predictions = result
            .get(if kind == "page" {
                "lines"
            } else {
                "predictions"
            })
            .and_then(Value::as_array);
        let mut empty = 0;
        let mut recovered = 0;
        let mut cases = Vec::new();
        if let Some(values) = predictions {
            for (index, p) in values.iter().enumerate() {
                let is_empty = p["text"].as_str() == Some("");
                let was_empty = p["original_model"]["text"].as_str() == Some("")
                    || p["empty_fallback"]["applied"].as_bool() == Some(true);
                if is_empty {
                    empty += 1;
                }
                if was_empty && !is_empty {
                    recovered += 1;
                }
                if self.enabled() && (is_empty || was_empty) && cases.len() < PER_REQUEST {
                    cases.push((
                        index,
                        if is_empty { "empty" } else { "recovered" },
                        p.clone(),
                    ));
                }
            }
        }
        let no_detections = kind == "page" && predictions.is_some_and(|p| p.is_empty());
        ctx.set(
            "result_count",
            result.get("count").cloned().unwrap_or(json!(0)),
        );
        ctx.set("empty_predictions", json!(empty));
        ctx.set("recovered_predictions", json!(recovered));
        ctx.set("no_detections", json!(no_detections));
        for key in ["timing", "rust_timings"] {
            if let Some(value) = result.get(key) {
                ctx.set(key, value.clone());
            }
        }
        if let Some(fallback) = result.get("cpu_fallback") {
            ctx.set("cpu_fallback", fallback.clone());
        }
        if !self.enabled() {
            return;
        }
        // Store one original page, with its empty-line metadata, rather than duplicate it.
        if kind == "page" {
            cases = if empty > 0 || no_detections {
                vec![(
                    0,
                    if no_detections {
                        "no_detections"
                    } else {
                        "empty_lines"
                    },
                    json!({"count":result["count"],"empty_lines":cases.iter().map(|(_,_,p)|p).collect::<Vec<_>>()}),
                )]
            } else {
                vec![]
            };
        }
        let selected = cases.len();
        let mut queued = 0;
        for (index, reason, prediction) in cases {
            let Some(image) = images.get(if kind == "page" { 0 } else { index }) else {
                continue;
            };
            if image.len() > MAX_CASE_BYTES {
                continue;
            }
            let name = format!("ocr-case-{}-{index:04x}", ctx.id);
            let record = json!({"schema_version":1,"case_id":name,"request_id":ctx.id,"timestamp_unix_ms":timestamp_ms(),
                "kind":kind,"index":index,"reason":reason,"server_version":env!("CARGO_PKG_VERSION"),
                "request":options,"image":image,"prediction":prediction});
            if self.submit(name, record) {
                queued += 1;
            }
        }
        let eligible = if kind == "page" {
            usize::from(empty > 0 || no_detections)
        } else {
            empty + recovered
        };
        ctx.set("diagnostics_queued", json!(queued));
        ctx.set(
            "diagnostics_dropped",
            json!(eligible.saturating_sub(queued)),
        );
        if queued < selected {
            emit(
                LogLevel::Warn,
                "diagnostics_capture_dropped",
                json!({"request_id":ctx.id,"dropped":selected-queued}),
            );
        }
    }
}
impl Drop for Diagnostics {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct Disk {
    root: PathBuf,
    limit: u64,
    used: u64,
    files: VecDeque<(PathBuf, u64)>,
    _lock: File,
}
fn managed(name: &str, suffix: &str) -> bool {
    name.strip_prefix("ocr-case-")
        .and_then(|s| s.strip_suffix(suffix))
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_hexdigit() || c == b'-'))
}
impl Disk {
    fn open(path: &Path, limit: u64) -> Result<Self> {
        let root = path.join("cases");
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&root)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".writer.lock"))?;
        lock.try_lock()
            .context("diagnostics directory is already in use or cannot be locked")?;
        let mut entries = vec![];
        for entry in fs::read_dir(&root)? {
            let e = entry?;
            if !e.file_type()?.is_file() {
                continue;
            }
            let name = e.file_name();
            let name = name.to_string_lossy();
            if managed(&name, ".tmp") {
                fs::remove_file(e.path())?;
            } else if managed(&name, ".json") {
                let m = e.metadata()?;
                entries.push((m.modified()?, e.path(), m.len()));
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut disk = Self {
            root,
            limit,
            used: entries.iter().map(|e| e.2).sum(),
            files: entries.into_iter().map(|(_, p, n)| (p, n)).collect(),
            _lock: lock,
        };
        disk.prune(0, 0)?;
        Ok(disk)
    }
    fn prune(&mut self, incoming: u64, count: usize) -> Result<()> {
        while self.used.saturating_add(incoming) > self.limit
            || self.files.len() + count > MAX_CASES
        {
            let Some((path, size)) = self.files.front() else {
                break;
            };
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            self.used -= size;
            self.files.pop_front();
        }
        Ok(())
    }
    fn save(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(
            bytes.len() as u64 <= self.limit,
            "case exceeds diagnostics disk budget"
        );
        self.prune(bytes.len() as u64, 1)?;
        let temp = self.root.join(format!("{name}.tmp"));
        let path = self.root.join(format!("{name}.json"));
        let result = (|| -> Result<()> {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result?;
        self.used += bytes.len() as u64;
        self.files.push_back((path, bytes.len() as u64));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "ocr-diagnostics-test-{}",
                crate::observability::id()
            ));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn rotation_restart_lock_and_unrelated_files() {
        let t = Temp::new();
        let mut d = Disk::open(&t.0, 10).unwrap();
        fs::write(d.root.join("keep.txt"), b"unrelated").unwrap();
        d.save("ocr-case-a", b"123456").unwrap();
        d.save("ocr-case-b", b"abcdef").unwrap();
        assert!(!d.root.join("ocr-case-a.json").exists());
        assert_eq!(d.used, 6);
        assert!(d.save("ocr-case-c", b"12345678901").is_err());
        assert!(Disk::open(&t.0, 10).is_err());
        fs::write(d.root.join("ocr-case-d.tmp"), b"partial").unwrap();
        drop(d);
        let d = Disk::open(&t.0, 5).unwrap();
        assert_eq!(d.used, 0);
        assert!(!d.root.join("ocr-case-d.tmp").exists());
        assert!(d.root.join("keep.txt").exists());
    }
    #[test]
    fn writes_only_interesting_cases_and_drains_on_drop() {
        let t = Temp::new();
        let d = Diagnostics::open(Some(&t.0), 1_048_576).unwrap();
        let ctx = RequestContext::new();
        let result = json!({"count":4,"predictions":[{"text":"","score":0.0},
            {"text":"岂","original_model":{"text":""},"cpu_fallback":{"applied":true}},
            {"text":"一","empty_fallback":{"applied":true}}, {"text":"already correct"}]});
        d.observe(
            &ctx,
            "glyph",
            &["YQ==".into(), "Yg==".into(), "Yw==".into(), "ZA==".into()],
            &result,
            json!({"score_mode":"model"}),
        );
        assert_eq!(ctx.fields.lock().unwrap()["empty_predictions"], 1);
        assert_eq!(ctx.fields.lock().unwrap()["recovered_predictions"], 2);
        assert!(!ctx.fields.lock().unwrap().to_string().contains("岂"));
        drop(d);
        let records: Vec<Value> = fs::read_dir(t.0.join("cases"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .map(|p| serde_json::from_slice(&fs::read(p).unwrap()).unwrap())
            .collect();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|r| r["request_id"] == ctx.id));
        assert!(records
            .iter()
            .any(|r| r["reason"] == "recovered" && r["image"] == "Yg=="));
    }
    #[test]
    fn page_saved_once_and_disabled_capture_only_counts() {
        let t = Temp::new();
        let d = Diagnostics::open(Some(&t.0), 1_048_576).unwrap();
        let ctx = RequestContext::new();
        d.observe(
            &ctx,
            "page",
            &["YQ==".into()],
            &json!({"count":2,"lines":[{"text":""},{"text":""}]}),
            json!({}),
        );
        assert_eq!(ctx.fields.lock().unwrap()["diagnostics_queued"], 1);
        drop(d);
        let d = Diagnostics::open(None, 1).unwrap();
        let ctx = RequestContext::new();
        d.observe(
            &ctx,
            "page",
            &["YQ==".into()],
            &json!({"count":0,"lines":[]}),
            json!({}),
        );
        assert_eq!(ctx.fields.lock().unwrap()["no_detections"], true);
        assert!(ctx
            .fields
            .lock()
            .unwrap()
            .get("diagnostics_queued")
            .is_none());
    }
    #[test]
    fn queue_pressure_drops_without_waiting_and_releases_reservation() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        let d = Diagnostics {
            sender: Some(sender),
            queued_bytes: Arc::new(AtomicUsize::new(0)),
            worker: None,
        };
        assert!(d.submit("a".into(), json!({"x":1})));
        let before = d.queued_bytes.load(Ordering::Relaxed);
        assert!(!d.submit("b".into(), json!({"x":2})));
        assert_eq!(d.queued_bytes.load(Ordering::Relaxed), before);
        d.queued_bytes.store(MAX_QUEUED_BYTES, Ordering::Relaxed);
        assert!(!d.submit("c".into(), json!({"x":3})));
    }
    #[test]
    fn write_error_does_not_publish_partial_case_or_poison_writer() {
        let t = Temp::new();
        let mut d = Disk::open(&t.0, 1024).unwrap();
        fs::create_dir(d.root.join("ocr-case-a.json")).unwrap();
        assert!(d.save("ocr-case-a", b"{}").is_err());
        assert!(!d.root.join("ocr-case-a.tmp").exists());
        assert_eq!(d.used, 0);
        d.save("ocr-case-b", b"{}").unwrap();
        assert_eq!(d.used, 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(d.root.join("ocr-case-b.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }
}
