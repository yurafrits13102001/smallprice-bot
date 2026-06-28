# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Telegram bot that checks whether a product (sent as a marketplace link) already
exists in an internal catalog. Supports AliExpress, Amazon (all TLDs), Temu,
TikTok Shop, eBay, and 1688. User-facing strings are Ukrainian.

## Commands

```bash
# Setup (Windows)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then fill BOT_TOKEN, OPENAI_API_KEY (APIFY_TOKEN optional)

# Build the text/FAISS index from data/products.xlsx (REQUIRED before first run)
python -m scripts.build_index

# Run the bot
python -m bot.main

# Accuracy eval — runs the SAME pipeline as the bot over test_cases.json
python test_accuracy.py

# Docker
docker compose up --build       # mounts ./data, reads .env
```

There is no lint config or unit-test framework. `test_*.py` are standalone ad-hoc
scripts (gitignored) run directly with `python test_xxx.py`; `test_accuracy.py` is
the real eval harness — populate `test_cases.json` (currently a placeholder) with
`{"url", "expected"}` pairs (`expected: null` = should NOT be found) to measure
precision/recall before and after a change.

## Required external state (not in git, see `data/.gitignore`)

- `data/products.xlsx` — the catalog. Headerless: col A = name, col B = primary
  link, cols C–E = supplier links. Loaded by `core/database.py`.
- `data/faiss_index.faiss` + `.meta` — text embedding index (built by `scripts.build_index`).
- `data/faiss_index_clip.faiss` + `.meta` — CLIP image index. **Built only inside
  the bot** when an `.xlsx` is uploaded (`build_clip_index_async`); `scripts.build_index`
  does NOT build it. Until it exists, image search is disabled and the bot falls
  back to text matching.
- `data/clip_image_cache.json` — URL→scraped-image-URL cache from CLIP builds.

## Architecture

There are three entrypoints in `bot/handlers.py`: `handle_message` (`F.text`, a
link), `handle_photo` (`F.photo`, a product photo sent directly — resolves a
Telegram file URL and runs the shared image branch), and `handle_document`
(`F.xlsx` upload → rebuild index). The CLIP→vision image branch is factored into
`_try_image_match`, reused by both the link and photo paths.

`handle_message` is the main orchestrator. A query link flows through up to four
stages, each a fallback for the previous:

1. **Exact URL match** (`ProductMatcher._url_match`) — normalized URL, or extracted
   Amazon ASIN / AliExpress item-id against pre-built maps. Returns 100% instantly.
2. **Image path: CLIP recall → GPT-4o vision precision.** `scrape_product(url)`
   does ONE fetch yielding both title and image. If an image is found and the CLIP
   index is loaded: `search_by_image` returns top-k via cosine on CLIP embeddings;
   everything above `CLIP_RECALL` (0.55) is a candidate. `compare_images_gpt4o`
   then downloads each image itself, inlines as base64, and asks GPT-4o which are
   the *same physical product*. CLIP is deliberately coarse (recall); the vision
   judge is the precision gate.
3. **Text fallback.** If no title yet and `APIFY_TOKEN` is set, retry via Apify
   (residential proxy actor). `normalize_title` (gpt-4o-mini) shrinks the raw
   marketplace title to a 3–7 word product name → `matcher.search` (FAISS text) →
   `verify_matches` (gpt-4o-mini) confirms each candidate is the same function/form.
4. **No match** → reports not found.

### Key design decisions (do not silently undo)

- **Two-tier image judge.** Architecture is CLIP(recall) → GPT-4o vision(precision).
  This split is intentional and confirmed correct — don't collapse it.
- **base64 image inlining** (`_fetch_image_data_url`): we fetch images ourselves so
  one dead/geo-blocked marketplace CDN can't fail the whole vision call. Unfetchable
  candidates with CLIP score ≥ `CLIP_CONFIDENT` (0.85) survive as CLIP-confirmed.
- **`verify_matches` is fail-CLOSED, but retries first**: it wraps the call+parse
  in `_openai_retry` (outer backoff on top of the SDK's own HTTP retries) and only
  returns `[]` after retries are exhausted, so a transient OpenAI blip is less
  likely to surface as "not found". The verdict-parse loop is defensive (skips
  malformed entries) — keep it that way; it runs outside the try/except.
- **One shared `AsyncOpenAI` client** (`matcher.client`, `max_retries=3`) is reused
  for embeddings, normalize, verify, and vision. Do NOT create a new client per
  message — that drops connection reuse and the retry config.
- **`verify_matches` gets the `short_title`, not the raw title.** Raw marketplace
  titles are keyword-stuffed (cup size, neckline, color variants) and make the judge
  over-strict, rejecting genuine matches. If you change this, use a *cleaned* title,
  not the raw one — see commit 71639af.
- **`scrape_product` uses curl_cffi (`impersonate=chrome120`) for ALL domains**, not
  httpx — marketplaces block plain httpx by TLS fingerprint. Extraction priority:
  JSON-LD (schema.org/Product) → og: tags → CSS selectors → URL slug. It does NOT
  defeat IP-based blocks; that's what the Apify fallback is for.

### Layout

- `bot/` — `main.py` (entrypoint, loads index, warms CLIP), `handlers.py` (all
  pipeline logic + rate limiting), `config.py` (pydantic-settings from `.env`).
- `core/` — `matcher.py` (`ProductMatcher`: text FAISS, URL/ASIN maps, CLIP build
  orchestration), `clip_matcher.py` (`CLIPImageIndex`, model load/embed; module-level
  model singleton, `preload_model()` for warmup), `scraper.py` (fetch + parse +
  Apify + `normalize_title`), `database.py` (xlsx → `Product`).
- `scripts/build_index.py` — offline text-index build.

### Config thresholds

`bot/config.py` default `similarity_threshold` is **0.50** but `.env.example` ships
**0.75** — the running value depends on `.env`. CLIP gates live in `core/clip_matcher.py`
(`CLIP_RECALL`=0.55, `CLIP_CONFIDENT`=0.85). Text embeddings use `text-embedding-3-large`
(3072-dim); changing the model requires a full index rebuild.

### Deployment

Server `185.233.44.8` at `/opt/smallprice-bot`; deploy is a manual `git pull` by the
user. `data/` lives outside git so the index and `.env` survive pulls. The Docker
image installs CPU-only torch (CLIP runs on CPU).
