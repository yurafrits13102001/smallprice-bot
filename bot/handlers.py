import logging
import re

from aiogram import Router, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.config import settings
from core.matcher import ProductMatcher
from core.scraper import scrape_product_title, is_supported_url, normalize_title

logger = logging.getLogger(__name__)
router = Router()

_URL_RE = re.compile(r"https?://[^\s]+")

matcher: ProductMatcher | None = None


def set_matcher(m: ProductMatcher) -> None:
    global matcher
    matcher = m


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привіт! Я бот для пошуку дублікатів товарів.\n\n"
        "Надішліть мені посилання на товар з будь-якого маркетплейсу "
        "(AliExpress, Amazon, Temu, TikTok Shop, 1688) — "
        "і я перевірю, чи є він у нашій базі."
    )

async def verify_matches(
    openai_client,
    query_title: str,
    candidates: list[tuple],
) -> list[tuple]:
    if not candidates:
        return []

    candidates_text = "\n".join(
        f"{i+1}. {p.name} (score: {s:.2f})"
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
                        "You compare products to determine if they could be the SAME or a VARIANT of the same product.\n"
                        "Be LENIENT - if products serve the same purpose and have the same form factor, they are likely the same.\n"
                        "Different names for the same thing count as same: 'radiator guard mesh' = 'water tank protection net'.\n"
                        "Different brand but same product = same.\n"
                        "Different color/size = same (variant).\n\n"
                        "Only mark as 'not_same' when products are CLEARLY different things:\n"
                        "- Different form factor (keychain microscope vs phone clip microscope)\n"
                        "- Different product category entirely (phone case vs phone charger)\n"
                        "- Fundamentally different function\n\n"
                        "When in doubt, mark as 'same'.\n\n"
                        "Respond with ONLY a JSON array:\n"
                        '[{"index": 1, "verdict": "same"}, {"index": 2, "verdict": "not_same"}]\n'
                        "No other text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query product: {query_title}\n\n"
                        f"Candidates:\n{candidates_text}"
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

    if not is_supported_url(url):
        await message.answer(
            "⚠️ Непідтримуване джерело. Підтримуються:\n"
            "AliExpress, Amazon, Temu, TikTok Shop, 1688"
        )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    status_msg = await message.answer("🔍 Аналізую товар...")

    # 1. Try URL matching first
    url_result = matcher._url_match(url)
    if url_result:
        product, score = url_result
        await status_msg.edit_text(
            f"🔍 Знайдено точний збіг за посиланням:\n\n"
            f"1. ✅ {product.name} (100%)\n"
            f"   🔗 {product.link}"
        )
        return

    # 2. Scrape product title
    raw_title = await scrape_product_title(url)

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
    filtered = await verify_matches(verify_client, short_title, filtered)

    if not filtered:
        await status_msg.edit_text(
            f"🔍 Товар: {short_title}\n"
            f"📝 Оригінал: {raw_title[:100]}\n\n"
            f"❌ Збігів не знайдено — товару немає в базі."
        )
        return

    lines = [f"🔍 Товар: {short_title}\n\nЗнайдено збіги в базі:\n"]
    for i, (product, score) in enumerate(filtered, 1):
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
        lines.append("")

    await status_msg.edit_text("\n".join(lines))