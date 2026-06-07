import asyncio
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
from aiogram.types import Message

from bot.config import settings
from core.clip_matcher import CLIP_THRESHOLD, CLIP_CONFIDENT
from core.matcher import ProductMatcher, _extract_url_description
from core.scraper import scrape_product_title_fast, scrape_product_title_apify, normalize_title, scrape_product_image_url

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


def _check_rate_limit(user_id: int) -> bool:
    now = time.time()
    q = _USER_REQUESTS[user_id]
    while q and now - q[0] > _RATE_WINDOW:
        q.popleft()
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


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привіт! Я бот для пошуку дублікатів товарів.\n\n"
        "Надішліть мені посилання на товар з будь-якого маркетплейсу "
        "(AliExpress, Amazon, Temu, TikTok Shop, 1688) — "
        "і я перевірю, чи є він у нашій базі."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 Як користуватися ботом:\n\n"
        "1️⃣ Надішліть посилання на товар з будь-якого маркетплейсу "
        "(AliExpress, Amazon, eBay, Temu, 1688)\n\n"
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

    try:
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
        verdicts = json.loads(raw)
        verified = []
        for v in verdicts:
            idx = v["index"] - 1
            if 0 <= idx < len(candidates) and v["verdict"] == "same":
                verified.append(candidates[idx])
        return verified

    except Exception as e:
        logger.warning(f"GPT verification failed: {e}")
        return candidates


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

    async def _alive(u: str | None) -> bool:
        return await check_link_alive(u) if u else True

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

    # 2. Scrape image and title in parallel — result shared for CLIP and text paths
    query_img, (raw_title, scrape_error) = await asyncio.gather(
        scrape_product_image_url(url),
        scrape_product_title_fast(url),
    )
    logger.info(f"CLIP: image={'found' if query_img else 'not found'}, index={'active' if matcher.clip_index else 'inactive'}")

    # 3. CLIP image search
    if query_img and matcher.clip_index:
        await status_msg.edit_text("🔍 Шукаю за зображенням...")
        clip_results = await matcher.search_by_image(query_img, top_k=5)
        logger.info(f"CLIP results: {[(p.name, round(s,2)) for p,s in clip_results]}")
        clip_good = [(p, s) for p, s in clip_results if s >= CLIP_THRESHOLD]
        if clip_good and not _is_generic_title(raw_title):
            from openai import AsyncOpenAI
            ai = AsyncOpenAI(api_key=settings.openai_api_key)
            short_title_clip = await normalize_title(ai, raw_title)
            clip_good = await verify_matches(ai, short_title_clip, raw_title, clip_good)

        if clip_good:
            alive_flags = await asyncio.gather(*[_alive(p.link) for p, _ in clip_good])
            lines = ["🔍 Знайдено за зображенням:\n"]
            for i, ((product, score), alive) in enumerate(zip(clip_good, alive_flags), 1):
                pct = int(score * 100)
                icon = "✅" if score >= CLIP_CONFIDENT else "🟡"
                label = "Підтверджено зображенням" if score >= CLIP_CONFIDENT else "Ймовірний збіг"
                lines.append(f"{i}. {icon} {product.name} ({pct}%)")
                lines.append(f"   {label}")
                if product.link:
                    lines.append(f"   🔗 {_display_url(product.link)}")
                    if not alive:
                        lines.append("   ⚠️ Посилання може бути неактуальне")
                lines.append("")
            await status_msg.edit_text("\n".join(lines))
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

    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    short_title = await normalize_title(openai_client, raw_title)

    results = await matcher.search(query=short_title, top_k=10)
    candidates = [(p, s) for p, s in results if s >= settings.similarity_threshold]
    verified = await verify_matches(openai_client, short_title, raw_title, candidates)
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


@router.message(F.document)
async def handle_document(message: Message) -> None:
    doc = message.document
    if not doc.file_name.endswith((".xlsx", ".xls")):
        await message.answer("❗ Надішліть файл у форматі Excel (.xlsx)")
        return

    status_msg = await message.answer("📥 Завантажую файл...")

    try:
        file = await message.bot.download(doc)

        from pathlib import Path
        from core.database import load_products

        path = Path(settings.products_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(file.read())

        await status_msg.edit_text("🔄 Оновлюю базу та будую індекс...\nЦе може зайняти 2-3 хвилини.")

        products = load_products(settings.products_path)
        await matcher.build_index(products)
        matcher.save_index(settings.index_path)

        await status_msg.edit_text(
            f"✅ Базу оновлено!\n\n"
            f"📊 Товарів в базі: {len(products)}\n"
            f"🔗 URL у індексі: {len(matcher.url_map)}\n\n"
            f"🔄 CLIP image index будується у фоні..."
        )

        async def _build_clip_and_notify():
            await matcher.build_clip_index_async(products, settings.index_path)
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

        asyncio.create_task(_build_clip_and_notify())

    except Exception as e:
        logger.error(f"Update failed: {e}")
        await status_msg.edit_text(f"❌ Помилка при оновленні бази: {e}")
