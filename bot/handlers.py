import asyncio
import logging
import re

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
                        "SAME if: same product category AND same general type/form factor. Different brand, color, size, or marketing name = still same.\n\n"
                        "NOT SAME only if: clearly different product type within the category "
                        "(e.g. sports bra ≠ everyday bra, tanktop-with-bra ≠ standalone bra, front-closure ≠ standard, knife ≠ keychain).\n\n"
                        "When in doubt within the same category, mark as same.\n\n"
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

    async def _alive(u: str | None) -> bool:
        return await check_link_alive(u) if u else True

    # 1. URL match
    url_result = matcher._url_match(url)
    if url_result:
        product, score = url_result
        alive = await _alive(product.link)
        link_line = f"   🔗 {product.link}" if product.link else ""
        warning = "\n   ⚠️ Посилання може бути неактуальне" if not alive else ""
        await status_msg.edit_text(
            f"🔍 Знайдено точний збіг за посиланням:\n\n"
            f"1. ✅ {product.name} (100%)\n"
            f"{link_line}{warning}"
        )
        return

    # 2. CLIP image search
    query_img = await scrape_product_image_url(url)
    if query_img:
        await status_msg.edit_text("🔍 Шукаю за зображенням...")
        clip_results = await matcher.search_by_image(query_img, top_k=5)
        clip_good = [(p, s) for p, s in clip_results if s >= CLIP_THRESHOLD]
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
                    lines.append(f"   🔗 {product.link}")
                    if not alive:
                        lines.append("   ⚠️ Посилання може бути неактуальне")
                lines.append("")
            await status_msg.edit_text("\n".join(lines))
            return

    # 3. Text fallback
    _SCRAPE_ERRORS = {
        "timeout":      "⏱ Сторінка відповідала занадто повільно.",
        "blocked":      "🚫 Сторінка заблокована для автоматичного доступу.",
        "network":      "🌐 Мережева помилка при завантаженні сторінки.",
        "apify_failed": "⚠️ Збій Apify-актора.",
        "apify_empty":  "⚠️ Apify не зміг зчитати назву товару.",
    }

    raw_title, scrape_error = await scrape_product_title_fast(url)
    apify_error = None
    if not raw_title and settings.apify_token:
        await status_msg.edit_text("🔍 Поліпшений пошук, зачекайте трохи довше...")
        raw_title, apify_error = await scrape_product_title_apify(url, settings.apify_token)

    if raw_title:
        GENERIC_TITLES = {
            "aliexpress", "amazon", "amazon.com", "ebay", "temu", "etsy",
            "shopee", "1688", "taobao", "walmart", "home", "homepage",
            "page not found", "404", "not found", "access denied",
        }
        if raw_title.strip().lower() in GENERIC_TITLES or len(raw_title.strip()) < 5:
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

    results_raw = await matcher.search(query=raw_title, top_k=10)
    results_norm = await matcher.search(query=short_title, top_k=10)

    seen = {}
    for product, score in results_raw + results_norm:
        if product.name not in seen or score > seen[product.name][1]:
            seen[product.name] = (product, score)

    results = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:10]
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