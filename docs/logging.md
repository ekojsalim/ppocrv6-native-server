# Logs and diagnostic cases

The Rust server emits one JSON object per line to **stderr**. Operational logs
never contain input images, recognized text, query strings, or request bodies.
Every response (including malformed JSON, 404s and failures) has a generated
`X-Request-ID`; caller-supplied IDs are not trusted or reused.

`request_complete` events include the request ID, method, route pattern, status,
wall-clock latency, and (when inference was reached) GPU queue wait, result count,
empty/recovered counts and pipeline timings. `cpu_fallback` reports attempted,
applied and skipped work. API errors also include a bounded error message.
Successful health checks and OPTIONS requests are debug-only to avoid probe noise.
Other events include startup/shutdown, diagnostic saves, dropped captures and
storage failures. Timestamps are Unix milliseconds; durations use milliseconds.

## Settings

| Flag | Default | Meaning |
|---|---|---|
| `--log-level` | `info` | `off`, `error`, `warn`, `info`, or `debug` |
| `--diagnostics-dir` | unset | Enable image/result capture in `DIR/cases` |
| `--diagnostics-max-mib` | `256` | Retained case-file byte budget |

Without a diagnostics directory, no image or result data is written to disk.
Use the container's log driver or a service manager for operational log retention;
the server does not maintain a second unbounded application log file.

Example (append the arguments after the image name):

```sh
mkdir -p diagnostics
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -p 127.0.0.1:8184:8184 \
  -v "$PWD/models:/models:ro" -v "$PWD/diagnostics:/diagnostics" \
  ppocrv6-native-server --model-dir /models --listen 0.0.0.0:8184 \
  --diagnostics-dir /diagnostics --diagnostics-max-mib 256 --log-level info
```

## What is saved

Only successful recognition requests produce diagnostic cases:

- **Glyphs:** empty final predictions, and empty model results recovered by the
  CPU dictionary or native one-stroke fallback.
- **Lines:** empty predictions, with their individual original line crop.
- **Pages:** no detected lines, or any line with empty text. Save the original
  page once, with at most 16 empty-line records (including their boxes).

A schema-v1 case is one JSON file containing the original encoded `image`, request
parameters, endpoint kind, original input index, prediction/diagnostics, reason,
server version, timestamp and request ID. Recovered CPU results retain
`original_model`; model probabilities and template similarities remain distinct.
A case is saved asynchronously, so the HTTP response can arrive before the file.
`diagnostic_saved` confirms a completed write; a queued count alone does not.

Cases contain user image data and result text. New case directories use mode
0700 and case files 0600 on Unix. Use a dedicated writable volume. The default
`diagnostics/` location is excluded from Git and container build contexts; keep
custom storage locations outside the source tree too.

## Bounds and failure handling

The writer has a 32-entry, **32 MiB serialized-data queue**, shared across requests.
It never waits for space on an inference thread. Capture selects at most 16 cases
per request, with a **16 MiB maximum serialized case** (including base64 overhead).
Larger cases and queue pressure drop capture, not OCR results. Logs report
`diagnostics_queued` and `diagnostics_dropped`; write errors are separate warnings.
Request parsing and inference retain their existing byte/pixel limits.

Oldest managed cases are removed to keep both the configured byte budget and a
1,024-file limit, including on restart. The budget counts case-file contents,
not filesystem block overhead or unrelated files. Oversized cases are rejected
without evicting all existing cases. Only managed `ocr-case-*.json` and abandoned
matching temporary files are touched. Do not edit files in the managed directory
while the writer is running.

Writes use private temporary files and atomic rename after flushing file data.
An OS file lock prevents two servers from sharing the same case directory;
unwritable or locked directories fail startup. Later I/O failures are logged
without failing recognition. Graceful shutdown drains the queue; a forced kill
can lose pending cases. Startup removes incomplete managed temporary files.

## Replay

The helper uses only Python's standard library:

```sh
python3 tools/replay_diagnostic.py diagnostics/cases/ocr-case-EXAMPLE.json \
  --url http://127.0.0.1:8184 --output replay.json
```

It submits the saved image and parameters to the matching glyph, line or page
endpoint. It does not replace the installed model/dictionary; use the same bundle
when comparing exact results. Invalid input requests and low-confidence nonempty
results are not automatically archived. Operational logs still show their HTTP
status and timings.

## Validation

Validated on the RTX 5070 Ti node with diagnostic capture enabled and a 1 MiB
test budget. HTTP tests cover empty and recovered glyphs, empty lines, pages
without detections, invalid base64/JSON, 404s, queue-timeout 503s, request-ID
correlation, capture limits and replay. The 岂 case replays successfully with
its saved original image and parameters. Operational logs exclude its text and
the test query-string contents. Shutdown drains queued cases, with no temporary
files left behind and retained bytes within budget. A separate run checks that
`warn` suppresses successful-request logs and absent storage configuration
creates no diagnostic directory.

All 27 Rust tests and Clippy pass. Tests include retention across restarts,
exclusive directory ownership, queue backpressure, write failures, private file
permissions and request IDs on successful/error responses. With capture enabled,
10-request warmed medians were 164.1 ms for the page fixture and 74.4 ms for the
1,024-glyph batch, consistent with the earlier CPU-fallback checkpoint. These
measurements do not represent every storage device or input distribution.
