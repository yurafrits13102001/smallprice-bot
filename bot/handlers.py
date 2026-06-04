import asyncio
import logging
import re

import httpx
from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from bot.config import settings
from core.matcher import ProductMatcher
from core.scraper import scrape_product_title_fast, scrape_product_title_apify, normalize_title

logger = logging.getLogger(__name__)
router = Router()

_URL_RE = re.compile(r"https?://[^\s]+")

matcher: ProductMatcher | None = None


def set_matcher(m: ProductMatcher) -> None:
    global matcher
    matcher = m


async def check_link_alive(url: str) -> bool:
    """
    Консервативна перевірка посилань:
    1. HTTP 404/410 → мертве
    2. <title> містить "not found" / "page not found" → мертве (працює для Amazon, eBay — серверний рендер)
    3. Все інше → живе (AliExpress safe: у нього title порожній, не тригериться)
    """
    try:
        import re
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        ) as client:
            resp = await client.get(url)
            logger.info(f"check_link_alive: {url[:80]}... -> HTTP {resp.status_code}")

            if resp.status_code in (404, 410):
                return False

            title_match = re.search(r'<title[^>]*>(.*?)</title>', resp.text[:5000], re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip().lower()
                dead_titles = {"page not found", "404 not found", "404", "not found"}
                if title in dead_titles:
                    logger.info(f"check_link_alive: dead title detected: '{title}'")
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

    candidates_text = "\n".join(
        f"{i + 1}. {p.name}"
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
                        "You are a strict product matching expert for an e-commerce database.\n"
                        "You receive a product from a marketplace (with its full original title and short normalized title) and a list of candidates from our database.\n\n"
                        "Determine if each candidate is the SAME PHYSICAL PRODUCT.\n\n"
                        "SAME means: the EXACT same product — same brand, same model, same specific item.\n"
                        "A different brand's version of the same product type is NOT same.\n"
                        "Different name for the exact same branded item = same.\n\n"
                        "NOT SAME means: different brand, different seller item, or different model — even if the product type or form factor is identical.\n\n"
                        "KEY RULE: Two products in the same category (e.g. two bras, two air fresheners, two medical bags) are NOT same unless they are clearly the same branded item. When in doubt, mark as not_same.\n\n"
                        "Use the FULL original title for context — it contains important details about form factor, attachment method, size, and use case that the short title may miss.\n\n"
                        "When in doubt, mark as not_same.\n\n"
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

        import json
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
    text = message.text.strip()
    url_match = _URL_RE.search(text)

    if not url_match:
        await message.answer("❗ Надішліть посилання на товар для перевірки.")
        return

    url = url_match.group(0)

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status_msg = await message.answer("🔍 Аналізую товар...")

    # 1. Try URL matching first
    url_result = matcher._url_match(url)
    if url_result:
        product, score = url_result
        alive = await check_link_alive(product.link) if product.link else True
        link_line = f"   🔗 {product.link}" if product.link else ""
        warning = "\n   ⚠️ Посилання може бути неактуальне" if not alive else ""
        await status_msg.edit_text(
            f"🔍 Знайдено точний збіг за посиланням:\n\n"
            f"1. ✅ {product.name} (100%)\n"
            f"{link_line}{warning}"
        )
        return

    # 2. Scrape product title
    raw_title = await scrape_product_title_fast(url)
    if not raw_title and settings.apify_token:
        await status_msg.edit_text("🔍 Поліпшений пошук, зачекайте трохи довше...")
        raw_title = await scrape_product_title_apify(url, settings.apify_token)

    # Filter out generic/marketplace titles before GPT normalization
    if raw_title:
        GENERIC_TITLES = {
            "aliexpress", "amazon", "amazon.com", "ebay", "temu", "etsy",
            "shopee", "1688", "taobao", "walmart", "home", "homepage",
            "page not found", "404", "not found", "access denied",
        }
        title_lower = raw_title.strip().lower()
        if title_lower in GENERIC_TITLES or len(raw_title.strip()) < 5:
            logger.warning(f"Generic/short title detected: '{raw_title}', treating as no title")
            raw_title = None

    if not raw_title:
        await status_msg.edit_text("❌ Не вдалось отримати назву товару зі сторінки.")
        return

    # 3. Normalize title via GPT
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    short_title = await normalize_title(openai_client, raw_title)

    # 4. Search with both original and normalized title, take best
    results_raw = await matcher.search(query=raw_title, top_k=5)
    results_norm = await matcher.search(query=short_title, top_k=5)

    # Merge: keep best score per product
    seen = {}
    for product, score in results_raw + results_norm:
        if product.name not in seen or score > seen[product.name][1]:
            seen[product.name] = (product, score)

    results = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:5]

    filtered = [
        (product, score) for product, score in results
        if score >= settings.similarity_threshold
    ]

    # 5. GPT verification - filter out false positives
    from openai import AsyncOpenAI
    verify_client = AsyncOpenAI(api_key=settings.openai_api_key)
    filtered = await verify_matches(verify_client, short_title, raw_title, filtered)

    if not filtered:
        await status_msg.edit_text(
            f"🔍 Товар: {short_title}\n"
            f"📝 Оригінал: {raw_title[:100]}\n\n"
            f"❌ Збігів не знайдено — товару немає в базі."
        )
        return

    # 6. Check if product links are still alive (parallel)
    async def _alive(url: str | None) -> bool:
        return await check_link_alive(url) if url else True

    alive_flags = await asyncio.gather(*[_alive(p.link) for p, _ in filtered])

    lines = [f"🔍 Товар: {short_title}\n\nЗнайдено збіги в базі:\n"]
    for i, ((product, score), alive) in enumerate(zip(filtered, alive_flags), 1):
        pct = int(score * 100)
        if score >= 0.90:
            icon = "✅"
            label = "Точний збіг"
        elif score >= 0.80:
            icon = "🟡"
            label = "Ймовірний збіг"
        else:
            icon = "🔵"
            label = "Схожий товар"
        lines.append(f"{i}. {icon} {product.name} ({pct}%)")
        lines.append(f"   {label}")
        if product.link:
            lines.append(f"   🔗 {product.link}")
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

        import os
        from pathlib import Path
        from bot.config import settings
        from core.database import load_products

        # Save file
        path = Path(settings.products_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(file.read())

        await status_msg.edit_text("🔄 Оновлюю базу та будую індекс...\nЦе може зайняти 2-3 хвилини.")

        # Rebuild index
        products = load_products(settings.products_path)
        await matcher.build_index(products)
        matcher.save_index(settings.index_path)

        await status_msg.edit_text(
            f"✅ Базу оновлено!\n\n"
            f"📊 Товарів в базі: {len(products)}\n"
            f"🔗 URL у індексі: {len(matcher.url_map)}"
        )

    except Exception as e:
        logger.error(f"Update failed: {e}")
        await status_msg.edit_text(f"❌ Помилка при оновленні бази: {e}")