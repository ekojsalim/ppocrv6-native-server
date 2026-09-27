# HTTP API

`GET /health` (also `/healthz`) checks that the server is running.
`GET /v1/ocr/info`, `/v1/lines/info`, and `/v1/glyphs/info` report model and
request limits. All inference requests use JSON and base64 image data.

| POST endpoint | Input | Output |
|---|---|---|
| `/v1/ocr/recognize` | `{"image":"BASE64"}` | Page lines, quadrilaterals, text and scores |
| `/v1/lines/recognize` | `{"images":["BASE64"],"batch_size":8}` | Ordered `predictions` and `count` |
| `/v1/glyphs/recognize` | `{"images":["BASE64"],"batch_size":128}` | Ordered glyph predictions and `count` |

Lines and glyphs also accept `image` for one input. Supply either `image` or
`images`. Batches larger than the inference batch size are processed in chunks.
Line crops must be upright single lines; this endpoint performs no detection or
rotation. Their normalized height is 48, with width buckets 384 and 3200;
crops exceeding normalized width 3200 are rejected. At most 128 line crops or
1024 glyph images are accepted per request. Glyph inference uses width 80.

Glyph requests may set `return_timesteps`, `character_policy` (`all`,
`suppress_ascii`, `cjk_focus`, `cjk_focus_fallback`) and `score_mode` (`model`,
`accepted`). Defaults are `cjk_focus_fallback` and `accepted`; accepted scores
are binary indicators. Lines and pages use the full vocabulary and model scores.

The default image-data budget is 16 MiB per request. Glyphs additionally have
1 MiB and 1,048,576 pixel limits per image. Page images and line batches have a
16 million pixel budget. HTTP body limits allow for base64 overhead. Invalid
inputs return a JSON error; queue timeout returns HTTP 503. Requests share one
GPU inference slot to bound memory usage.

The service has no authentication or TLS termination. Bind to localhost or put
it behind your application's authenticated reverse proxy when exposing it.

An optional [CPU glyph fallback](glyph-fallback.md) handles empty isolated-glyph
results when a dictionary is installed. Recovered predictions expose provenance
and original model output; `score` is nullable in `model` mode, and CTC class IDs
are not synthesized. The `/info` endpoint reports activation and fixed limits.
