import asyncio
import base64
import json
import logging
import re
import time
from collections import defaultdict, deque
from urllib.parse import urlparse, urlunparse

import httpx
from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ErrorEvent

from bot.config import settings
from core.clip_matcher import CLIP_CONFIDENT, CLIP_RECALL
from core.matcher import ProductMatcher, _extract_url_description
from core.scraper import scrape_product, scrape_product_title_apify, normalize_title

logger = logging.getLogger(__name__)
router = Router()

_URL_RE = re.compile(r"https?://[^\s]+")

matcher: ProductMatcher | None = None

_GENERIC_TITLES = {
    "aliexpress", "amazon", "amazon.com", "ebay", "temu", "etsy",
    "shopee", "1688", "taobao", "walmart", "home", "homepage",
    "page not found", "404", "not found", "access denied",
}

_SCRAPE_ERRORS = {
    "timeout":      "⏱ Сторінка відповідала занадто повільно.",
    "blocked":      "🚫 Сторінка заблокована для автоматичного доступу.",
    "network":      "🌐 Мережева помилка при завантаженні сторінки.",
    "apify_failed": "⚠️ Збій Apify-актора.",
    "apify_empty":  "⚠️ Apify не зміг зчитати назву товару.",
}

_USER_REQUESTS: dict[int, deque] = defaultdict(deque)
_RATE_LIMIT = 10
_RATE_WINDOW = 60


def set_matcher(m: ProductMatcher) -> None:
    global matcher
    matcher = m


async def _openai_retry(factory, *, attempts: int = 2, base_delay: float = 0.6):
    """Run an async OpenAI call (the whole call+parse) with a short backoff.

    The OpenAI SDK already retries transient HTTP errors; this adds an outer retry
    so a malformed/empty response or an error that survives the SDK's own retries
    gets one more shot before a fail-closed path treats it as "no match".
    Re-raises the last exception when all attempts are exhausted.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return await factory()
        except Exception as e:
            last = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                logger.warning(f"OpenAI call failed (attempt {i + 1}/{attempts}), retry in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    raise last


def _check_rate_limit(user_id: int) -> bool:
    now = time.time()
    q = _USER_REQUESTS[user_id]
    while q and now - q[0] > _RATE_WINDOW:
        q.popleft()
    # Bound memory: drop idle users once the table grows. Without this the
    # defaultdict keeps an entry for every user who ever messaged the bot.
    if len(_USER_REQUESTS) > 1000:
        stale = [
            uid for uid, dq in _USER_REQUESTS.items()
            if uid != user_id and (not dq or now - dq[-1] > _RATE_WINDOW)
        ]
        for uid in stale:
            del _USER_REQUESTS[uid]
    if len(q) >= _RATE_LIMIT:
        return False
    q.append(now)
    return True


def _is_generic_title(title: str | None) -> bool:
    if not title:
        return True
    return title.strip().lower() in _GENERIC_TITLES or len(title.strip()) < 5


def _display_url(url: str) -> str:
    """Strip Amazon search params, return canonical /dp/ASIN form."""
    try:
        parsed = urlparse(url)
        if re.search(r'amazon\.[a-z.]+', parsed.netloc):
            asin_match = re.search(r'/(?:dp|gp/product)/([A-Z0-9]{10})', parsed.path)
            if asin_match:
                return f"https://{parsed.netloc}/dp/{asin_match.group(1)}"
        return urlunparse(parsed._replace(query="", fragment=""))
    except Exception:
        return url


async def check_link_alive(url: str) -> bool:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=4,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)
            logger.info(f"check_link_alive: {url[:80]}... -> HTTP {resp.status_code}")
            if resp.status_code in (404, 410):
                return False
            title_match = re.search(r'<title[^>]*>(.*?)</title>', resp.text[:5000], re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip().lower()
                if title in {"page not found", "404 not found", "404", "not found"}:
                    logger.info(f"check_link_alive: dead title: '{title}'")
                    return False
            return True
    except Exception:
        return True


async def _alive(u: str | None) -> bool:
    """True if the link looks live (fail-open). An empty link counts as live."""
    return await check_link_alive(u) if u else True


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привіт! Я бот для пошуку дублікатів товарів.\n\n"
        "Надішліть мені:\n"
        "🔗 посилання на товар (AliExpress, Amazon, Temu, TikTok Shop, 1688), або\n"
        "📷 фото товару\n\n"
        "— і я перевірю, чи є він у нашій базі."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Як користуватися ботом:\n\n"
        "1️⃣ Надішліть посилання на товар з будь-якого маркетплейсу "
        "(AliExpress, Amazon, eBay, Temu, 1688) "
        "або 📷 фото товару\n\n"
        "2️⃣ Зачекайте 5-15 секунд — бот перевірить чи є товар у базі\n\n"
        "3️⃣ Результати:\n"
        "   ✅ Точний збіг (90%+)\n"
        "   🟡 Ймовірний збіг (80-90%)\n"
        "   🔵 Схожий товар (55-80%)\n"
        "   ❌ Не знайдено\n\n"
        "📥 Оновлення бази:\n"
        "Надішліть Excel-файл (.xlsx) — бот автоматично оновить базу\n\n"
        "⏳ Якщо бачите «Поліпшений пошук» — зачекайте до 30 секунд"
    )


@router.error()
async def on_error(event: ErrorEvent) -> None:
    """Catch-all for exceptions that escape a handler.

    Without this, an unexpected error (e.g. a failed edit_text, a None matcher, a
    parse blow-up) is only logged — the user is left staring at the last status
    message ("🔍 Аналізую товар...") forever. We log the full traceback and try to
    tell the user something went wrong so they know to retry. Best-effort: if even
    the notify send fails (e.g. user blocked the bot), we swallow it.
    """
    logger.exception(f"Unhandled error while processing update: {event.exception}")
    update = event.update
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target is None:
        return
    try:
        await target.answer(
            "⚠️ Сталася помилка при обробці запиту. Спробуйте, будь ласка, ще раз."
        )
    except Exception:
        pass


async def verify_matches(
        openai_client,
        query_title: str,
        raw_title: str,
        candidates: list[tuple],
) -> list[tuple]:
    if not candidates:
        return []

    def _product_label(p) -> str:
        desc = _extract_url_description(p.link)
        if not desc:
            for sup in p.supplier_links:
                desc = _extract_url_description(sup)
                if desc:
                    break
        return f"{p.name} — {desc}" if desc else p.name

    candidates_text = "\n".join(
        f"{i + 1}. {_product_label(p)}"
        for i, (p, s) in enumerate(candidates)
    )

    async def _run():
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product matching expert for a dropshipping business.\n"
                        "The same physical product is often sold under different brand names and descriptions on different marketplaces.\n\n"
                        "You receive a query product and candidates from our database. Determine if each candidate COULD BE the same physical item.\n\n"
                        "SAME requires ALL of:\n"
                        "  1. Same specific function/purpose (what the product DOES)\n"
                        "  2. Same general form factor\n"
                        "  Different brand, color, size, or marketing name = still same.\n\n"
                        "NOT SAME examples:\n"
                        "  LED light strip ≠ rubber door protector (different function)\n"
                        "  sports bra ≠ everyday bra (different type)\n"
                        "  tanktop-with-bra ≠ standalone bra (different form)\n"
                        "  knife ≠ keychain (different function)\n"
                        "  car light ≠ car protector strip (different function)\n"
                        "  decoration ≠ protection product (different function)\n\n"
                        "When in doubt about brand/style/color — mark same. "
                        "When in doubt about function/purpose — mark not_same.\n\n"
                        "Respond with ONLY a JSON array:\n"
                        '[{"index": 1, "verdict": "same"}, {"index": 2, "verdict": "not_same"}]\n'
                        "No other text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Full original title: {raw_title}\n"
                        f"Short title: {query_title}\n\n"
                        f"Candidates from database:\n{candidates_text}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    try:
        verdicts = await _openai_retry(_run)
    except Exception as e:
        # Fail closed only after retries are exhausted: verification is the precision
        # gate that rejects same-category but wrong-function items. A transient OpenAI
        # blip shouldn't surface as "not found", so we retry before giving up.
        logger.warning(f"GPT verification failed after retries, dropping {len(candidates)} unverified candidate(s): {e}")
        return []

    verified = []
    for v in verdicts if isinstance(verdicts, list) else []:
        if not isinstance(v, dict) or not isinstance(v.get("index"), int):
            continue
        idx = v["index"] - 1
        if 0 <= idx < len(candidates) and v.get("verdict") == "same":
            verified.append(candidates[idx])
    return verified


_IMG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
}
_MAX_IMG_BYTES = 18 * 1024 * 1024  # stay under OpenAI's 20 MB per-image limit


async def _fetch_image_data_url(client: httpx.AsyncClient, url: str) -> str | None:
    """Download an image and inline it as a base64 data URL.

    Returns None if the URL is unreachable, non-image, or oversized. We fetch the
    image ourselves (instead of handing OpenAI the raw URL) so a stale/geo-blocked
    marketplace CDN can't make the whole vision call fail.
    """
    if not url:
        return None
    try:
        r = await client.get(url)
    except Exception as e:
        logger.debug(f"vision image fetch failed {url[:60]}: {e}")
        return None
    if r.status_code != 200 or not r.content or len(r.content) > _MAX_IMG_BYTES:
        return None
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    if not ctype.startswith("image/"):
        ctype = "image/jpeg"
    b64 = base64.b64encode(r.content).decode("ascii")
    return f"data:{ctype};base64,{b64}"


async def compare_images_gpt4o(
        openai_client,
        query_img_url: str,
        candidates: list[tuple],
) -> list[tuple]:
    """Precision judge: GPT-4o vision decides which candidates are the SAME physical product.

    candidates: list of (product, clip_score, candidate_img_url).
    Returns list of (product, clip_score, confidence) for confirmed matches, sorted by confidence.
    """
    valid = [(p, s, img) for (p, s, img) in candidates if img]
    if not valid:
        return []

    # Download every image ourselves and inline as base64 so OpenAI never has to
    # reach a (possibly dead/blocked) marketplace CDN — one bad URL must not fail
    # the whole vision call.
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=8.0, headers=_IMG_HEADERS
    ) as client:
        fetched = await asyncio.gather(
            _fetch_image_data_url(client, query_img_url),
            *[_fetch_image_data_url(client, img) for _, _, img in valid],
        )
    query_data, cand_data = fetched[0], fetched[1:]

    if not query_data:
        # Without the query image the judge is blind → keep CLIP-confident matches.
        logger.warning("GPT-4o vision: query image unfetchable, falling back to CLIP")
        return [(p, s, s) for (p, s, img) in valid if s >= CLIP_CONFIDENT]

    judged = [(p, s, data) for (p, s, _), data in zip(valid, cand_data) if data]
    dead = [(p, s) for (p, s, _), data in zip(valid, cand_data) if not data]
    if dead:
        logger.warning(
            f"GPT-4o vision: {len(dead)}/{len(valid)} candidate image(s) unfetchable, "
            f"judging {len(judged)}"
        )
    # Candidates we couldn't fetch can't be vision-judged; keep the CLIP-confident
    # ones rather than silently dropping a possible real match.
    fallback = [(p, s, s) for (p, s) in dead if s >= CLIP_CONFIDENT]

    if not judged:
        return sorted(fallback, key=lambda x: x[2], reverse=True)

    content: list[dict] = [
        {"type": "text", "text": "QUERY product image (the item the user sent):"},
        {"type": "image_url", "image_url": {"url": query_data}},
    ]
    for i, (p, s, data) in enumerate(judged, 1):
        content.append({"type": "text", "text": f"Candidate {i}:"})
        content.append({"type": "image_url", "image_url": {"url": data}})
    content.append({
        "type": "text",
        "text": (
            "For each candidate, decide if it is the SAME physical product as the QUERY — "
            "i.e. the identical item that could be sold under a different brand/listing.\n"
            "SAME means it is the very same design/model. Judge by concrete visual details, "
            "not just overall category or silhouette.\n"
            "Treat as DIFFERENT (NOT same) if any of these clearly differ:\n"
            "- pattern/print (e.g. floral print vs solid color)\n"
            "- material/texture (e.g. lace vs plain fabric, mesh vs knit)\n"
            "- construction/cut (e.g. racerback vs crossed-back, wired vs wireless, number of pieces)\n"
            "- function/purpose.\n"
            "ONLY a pure color change of the SAME design counts as SAME. "
            "Two items that are merely the same TYPE of product (e.g. both are bras, both are phone holders) "
            "but differ in the details above are NOT same.\n"
            "Be conservative: if unsure whether it is the identical design, answer same=false.\n"
            'Respond with ONLY a JSON array, no other text: '
            '[{"index": 1, "same": true, "confidence": 0.0}]'
        ),
    })

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        verdicts = json.loads(raw)
    except Exception as e:
        logger.warning(f"GPT-4o vision compare failed: {e}")
        # Fall back to CLIP-only judgement: keep candidates above the confident threshold.
        clip_only = [(p, s, s) for (p, s, _) in judged if s >= CLIP_CONFIDENT]
        return sorted(clip_only + fallback, key=lambda x: x[2], reverse=True)

    by_idx = {i: (p, s) for i, (p, s, data) in enumerate(judged, 1)}
    confirmed = list(fallback)
    for v in verdicts:
        idx = v.get("index")
        if v.get("same") and idx in by_idx:
            p, s = by_idx[idx]
            try:
                conf = float(v.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            confirmed.append((p, s, conf))
    confirmed.sort(key=lambda x: x[2], reverse=True)
    return confirmed


async def _render_matches(status_msg: Message, header: str, rows: list[tuple]) -> None:
    """Render a numbered result list. rows: (product, pct:int, icon:str, label:str)."""
    alive_flags = await asyncio.gather(*[_alive(p.link) for p, *_ in rows])
    lines = [header]
    for i, ((product, pct, icon, label), alive) in enumerate(zip(rows, alive_flags), 1):
        lines.append(f"{i}. {icon} {product.name} ({pct}%)")
        lines.append(f"   {label}")
        if product.link:
            lines.append(f"   🔗 {_display_url(product.link)}")
            if not alive:
                lines.append("   ⚠️ Посилання може бути неактуальне")
        lines.append("")
    await status_msg.edit_text("\n".join(lines))


async def _try_image_match(
    status_msg: Message, query_img: str, openai_client, *, allow_similar: bool = False
) -> bool:
    """CLIP recall → GPT-4o vision precision → render result.

    Shared by the link path (image scraped from the page) and the photo path
    (image sent directly to the bot). Returns True if a result was shown to the
    user, False if the caller should fall through to its own next step.

    With allow_similar=True, when the strict vision judge confirms nothing we still
    surface the top CLIP look-alikes as "🔵 Схожий товар" instead of giving up —
    the photo path's analogue of the text path's "similar" tier (a photo has no
    title to fall back to).
    """
    if not (query_img and matcher.clip_index):
        return False

    await status_msg.edit_text("🔍 Шукаю за зображенням...")
    clip_results = await matcher.search_by_image(query_img, top_k=5)
    logger.info(f"CLIP results: {[(p.name, round(s, 2)) for p, s, _ in clip_results]}")
    # CLIP is only a coarse filter: keep everything above the low recall gate as candidates.
    clip_cand = [(p, s, img) for p, s, img in clip_results if s >= CLIP_RECALL]
    if not clip_cand:
        return False

    confirmed = await compare_images_gpt4o(openai_client, query_img, clip_cand)
    logger.info(f"GPT-4o vision confirmed: {[(p.name, round(c, 2)) for p, _, c in confirmed]}")

    if confirmed:
        rows = []
        for product, clip_score, conf in confirmed:
            pct = int(max(conf, clip_score) * 100)
            icon = "✅" if conf >= 0.80 else "🟡"
            label = "Підтверджено зображенням" if conf >= 0.80 else "Ймовірний збіг"
            rows.append((product, pct, icon, label))
        await _render_matches(status_msg, "🔍 Знайдено за зображенням:\n", rows)
        return True

    if allow_similar:
        # clip_cand is sorted by CLIP score (FAISS returns descending); show the top few.
        rows = [
            (p, int(s * 100), "🔵", "Схожий товар (за зображенням)")
            for p, s, _ in clip_cand[:3]
        ]
        await _render_matches(
            status_msg,
            "🔍 Точного збігу немає, але ось найсхожіші товари за зображенням:\n",
            rows,
        )
        return True

    return False


@router.message(F.text)
async def handle_message(message: Message) -> None:
    if not _check_rate_limit(message.from_user.id):
        await message.answer("⏳ Занадто багато запитів. Зачекайте хвилину.")
        return

    text = message.text.strip()
    url_match = _URL_RE.search(text)
    if not url_match:
        await message.answer("❗ Надішліть посилання на товар для перевірки.")
        return

    url = url_match.group(0)
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status_msg = await message.answer("🔍 Аналізую товар...")

    # 1. URL match
    url_result = matcher._url_match(url)
    if url_result:
        product, score = url_result
        alive = await _alive(product.link)
        link_line = f"   🔗 {_display_url(product.link)}" if product.link else ""
        warning = "\n   ⚠️ Посилання може бути неактуальне" if not alive else ""
        await status_msg.edit_text(
            f"🔍 Знайдено точний збіг за посиланням:\n\n"
            f"1. ✅ {product.name} (100%)\n"
            f"{link_line}{warning}"
        )
        return

    # 2. Single fetch → title + image (shared for CLIP image path and text path)
    raw_title, query_img, scrape_error = await scrape_product(url)
    logger.info(f"CLIP: image={'found' if query_img else 'not found'}, index={'active' if matcher.clip_index else 'inactive'}")

    # 3. CLIP image search (recall) → GPT-4o vision (precision)
    if await _try_image_match(status_msg, query_img, matcher.client):
        return

    # 4. Text fallback
    apify_error = None
    if not raw_title and settings.apify_token:
        await status_msg.edit_text("🔍 Поліпшений пошук, зачекайте трохи довше...")
        raw_title, apify_error = await scrape_product_title_apify(url, settings.apify_token)

    if _is_generic_title(raw_title):
        if raw_title:
            logger.warning(f"Generic/short title detected: '{raw_title}', treating as no title")
        raw_title = None

    if not raw_title:
        error_key = apify_error or scrape_error
        error_detail = _SCRAPE_ERRORS.get(error_key, "")
        retry_hint = "\n🔄 Спробуйте надіслати посилання ще раз." if error_key in ("apify_failed", "apify_empty", "timeout", "network") else ""
        await status_msg.edit_text(
            f"❌ Не вдалось отримати назву товару зі сторінки.\n"
            f"{error_detail}{retry_hint}"
        )
        return

    short_title = await normalize_title(matcher.client, raw_title)

    results = await matcher.search(query=short_title, top_k=10)
    candidates = [(p, s) for p, s in results if s >= settings.similarity_threshold]
    # Verify on the normalized short title, not the raw scraped one. Marketplace
    # titles are keyword-stuffed with hyper-specific detail (cup size, neckline,
    # color/size variants) that makes the judge over-strict and reject genuine
    # "similar" matches, while the short title still filters cross-function junk.
    verified = await verify_matches(matcher.client, short_title, short_title, candidates)
    filtered = [(p, s, False) for p, s in verified]

    if not filtered:
        await status_msg.edit_text(
            f"🔍 Товар: {short_title}\n\n"
            f"❌ Збігів не знайдено — товару немає в базі.\n\n"
            f"📝 Перевірено за назвою товару."
        )
        return

    alive_flags = await asyncio.gather(*[_alive(p.link) for p, _, _ in filtered])
    lines = [f"🔍 Товар: {short_title}\n\nЗнайдено збіги в базі:\n"]
    for i, ((product, score, _), alive) in enumerate(zip(filtered, alive_flags), 1):
        pct = int(score * 100)
        if score >= 0.90:
            icon, label = "✅", "Точний збіг"
        elif score >= 0.80:
            icon, label = "🟡", "Ймовірний збіг"
        else:
            icon, label = "🔵", "Схожий товар"
        lines.append(f"{i}. {icon} {product.name} ({pct}%)")
        lines.append(f"   {label}")
        if product.link:
            lines.append(f"   🔗 {_display_url(product.link)}")
            if not alive:
                lines.append("   ⚠️ Посилання може бути неактуальне")
        lines.append("")

    await status_msg.edit_text("\n".join(lines))


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    if not _check_rate_limit(message.from_user.id):
        await message.answer("⏳ Занадто багато запитів. Зачекайте хвилину.")
        return

    if not matcher.clip_index:
        await message.answer(
            "⚠️ Пошук за фото зараз недоступний (індекс зображень ще не побудовано).\n"
            "Надішліть посилання на товар або оновіть базу Excel-файлом."
        )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status_msg = await message.answer("🔍 Аналізую фото...")

    # Telegram sends several sizes; the last is the largest. Resolve a direct
    # download URL for it — both CLIP and the vision judge fetch the bytes
    # themselves, so the bot token in the URL never leaves this process.
    try:
        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        query_img = f"https://api.telegram.org/file/bot{settings.bot_token}/{file.file_path}"
    except Exception as e:
        logger.warning(f"Failed to resolve Telegram photo: {e}")
        await status_msg.edit_text("❌ Не вдалось завантажити фото. Спробуйте ще раз.")
        return

    if await _try_image_match(status_msg, query_img, matcher.client, allow_similar=True):
        return

    await status_msg.edit_text(
        "❌ Схожих товарів за цим фото не знайдено в базі.\n\n"
        "📷 Порада: надішліть чітке фото товару на однотонному фоні, "
        "або пришліть посилання на товар."
    )


# Serializes index rebuilds: a second .xlsx upload while one is running would
# race on the cache file, double the paid Firecrawl spend, and build a CLIP index
# against a different product ordering than the text index.
_index_build_lock = asyncio.Lock()
# Keep strong references to background tasks — asyncio only holds a weak one, so
# an un-referenced task can be garbage-collected mid-run (silently killing the
# multi-hour CLIP build).
_bg_tasks: set[asyncio.Task] = set()


@router.message(F.document)
async def handle_document(message: Message) -> None:
    # DB replacement + paid scraping — restrict to admins when ADMIN_IDS is set.
    admin_ids = settings.admin_id_set
    if admin_ids and message.from_user.id not in admin_ids:
        logger.warning(f"Unauthorized xlsx upload attempt from user {message.from_user.id}")
        await message.answer("⛔ Оновлювати базу можуть лише адміністратори.")
        return

    if not _check_rate_limit(message.from_user.id):
        await message.answer("⏳ Занадто багато запитів. Зачекайте хвилину.")
        return

    doc = message.document
    if not (doc.file_name or "").endswith((".xlsx", ".xls")):
        await message.answer("❗ Надішліть файл у форматі Excel (.xlsx)")
        return

    if _index_build_lock.locked():
        await message.answer(
            "⏳ Оновлення бази вже виконується. Зачекайте, поки воно завершиться, "
            "і надішліть файл ще раз."
        )
        return

    status_msg = await message.answer("📥 Завантажую файл...")
    await _index_build_lock.acquire()
    release_in_task = False  # the background CLIP task takes over the lock on success

    try:
        file = await message.bot.download(doc)

        from pathlib import Path
        from core.database import load_products

        path = Path(settings.products_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Validate the upload BEFORE overwriting the live base: a corrupt/empty
        # file must not destroy products.xlsx on disk.
        tmp_path = path.with_name(path.name + ".new")
        with open(tmp_path, "wb") as f:
            f.write(file.read())
        products = load_products(str(tmp_path))
        if not products:
            tmp_path.unlink(missing_ok=True)
            await status_msg.edit_text(
                "❌ Файл не містить жодного товару — базу НЕ змінено.\n"
                "Перевірте формат: колонка A — назва, B — посилання, C-E — постачальники."
            )
            return
        tmp_path.replace(path)

        await status_msg.edit_text("🔄 Оновлюю базу та будую індекс...\nТекстовий індекс: 2-3 хвилини.")

        await matcher.build_index(products)
        matcher.save_index(settings.index_path)

        await status_msg.edit_text(
            f"✅ Базу оновлено!\n\n"
            f"📊 Товарів в базі: {len(products)}\n"
            f"🔗 URL у індексі: {len(matcher.url_map)}\n\n"
            f"🔄 CLIP image index будується у фоні — для нових товарів це може "
            f"зайняти до кількох годин. Бот працює, повідомлю, коли фото будуть готові."
        )

        async def _build_clip_and_notify():
            try:
                await matcher.build_clip_index_async(
                    products, settings.index_path, settings.apify_token,
                    concurrency=settings.clip_scrape_concurrency,
                    scrape_timeout=settings.clip_scrape_timeout,
                    firecrawl_api_key=settings.firecrawl_api_key,
                    firecrawl_proxy=settings.firecrawl_proxy,
                    firecrawl_concurrency=settings.firecrawl_concurrency,
                    firecrawl_max_calls=settings.firecrawl_max_calls,
                    firecrawl_skip_1688=settings.firecrawl_skip_1688,
                    use_playwright=settings.use_playwright,
                    playwright_concurrency=settings.playwright_concurrency,
                    playwright_proxy=settings.playwright_proxy,
                )
            finally:
                _index_build_lock.release()
            count = matcher.clip_index.index.ntotal if matcher.clip_index and matcher.clip_index.index else 0
            try:
                if count:
                    await status_msg.edit_text(
                        f"✅ Базу оновлено!\n\n"
                        f"📊 Товарів в базі: {len(products)}\n"
                        f"🔗 URL у індексі: {len(matcher.url_map)}\n\n"
                        f"🖼 CLIP index готовий: {count} фото — пошук за зображенням активний!"
                    )
                else:
                    await status_msg.edit_text(
                        f"✅ Базу оновлено!\n\n"
                        f"📊 Товарів в базі: {len(products)}\n"
                        f"🔗 URL у індексі: {len(matcher.url_map)}\n\n"
                        f"⚠️ CLIP index не побудовано — фото товарів недоступні."
                    )
            except Exception:
                pass

        task = asyncio.create_task(_build_clip_and_notify())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        release_in_task = True

    except Exception as e:
        logger.error(f"Update failed: {e}")
        await status_msg.edit_text(f"❌ Помилка при оновленні бази: {e}")
    finally:
        if not release_in_task:
            _index_build_lock.release()
