# Offline dictionary tools

These tools generate assets and evaluate experiments; none runs in the HTTP
request path. The serving matcher is `server/src/glyph_matcher.rs`, with HTTP
policy in `server/src/glyph_fallback.rs`. The Rust example imports the shared
matcher, so corpus replay tests the same numerical implementation as serving.

- `prepare.py --templates-only`: generate dictionary input from Noto variable SC
  fonts, without any private evaluation data. Requires Pillow and fonttools.
- Rust example `cpu_glyph_fallback build`: normalize/rendered inputs into the v2
  dictionary. `evaluate` performs corpus replay.
- `../package_glyph_dictionary.py`: validate and install a dictionary plus font
  notices into a model bundle using Python's standard library.
- Remaining Python tools and `gpu_bench.cu`: historical experiments and evaluation
  helpers. GPU matching is not part of serving or the model builder.

See `docs/glyph-fallback.md` for production packaging, limits and response fields.
Generated fonts, templates, crops and results belong under ignored `tmp/` paths.
