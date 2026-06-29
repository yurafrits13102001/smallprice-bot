"""Accuracy eval harness.

Runs a list of test links through the SAME matching pipeline as the bot and
reports precision / recall / accuracy. Use it to measure before/after a change.

Test cases live in test_cases.json:
[
  {"url": "https://...", "expected": "Exact product name in DB"},   # should be found
  {"url": "https://...", "expected": null}                          # should NOT be found
]

`expected` = the product `name` from the Excel base that SHOULD be returned,
or null/None if the product is NOT in the base.

Run:  python test_accuracy.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

from core.database import load_products
from core.matcher import ProductMatcher
from core.clip_matcher import CLIP_RECALL
from core.scraper import scrape_product, scrape_product_title_apify, normalize_title
from bot.config import settings
from bot.handlers import verify_matches, compare_images_gpt4o, _is_generic_title

CASES_PATH = Path("test_cases.json")


async def predict(matcher: ProductMatcher, url: str) -> tuple[str | None, str]:
    """Mirror of handle_message logic. Returns (matched_product_name | None, source_stage).

    Must stay faithful to bot.handlers.handle_message so the eval measures what
    actually runs in prod: ONE scrape_product fetch (title+image together) and the
    SAME shared matcher.client for every OpenAI call.
    """
    # 1. URL match
    url_result = matcher._url_match(url)
    if url_result:
        return url_result[0].name, "url"

    # 2. Single fetch → title + image (exactly as the bot does it)
    raw_title, query_img, _scrape_error = await scrape_product(url)

    # 3. CLIP recall → GPT-4o vision precision
    if query_img and matcher.clip_index:
        clip_results = await matcher.search_by_image(query_img, top_k=5)
        clip_cand = [(p, s, img) for p, s, img in clip_results if s >= CLIP_RECALL]
        if clip_cand:
            confirmed = await compare_images_gpt4o(matcher.client, query_img, clip_cand)
            if confirmed:
                return confirmed[0][0].name, "image"

    # 4. text fallback
    if not raw_title and settings.apify_token:
        raw_title, _ = await scrape_product_title_apify(url, settings.apify_token)
    if _is_generic_title(raw_title):
        raw_title = None
    if not raw_title:
        return None, "no_title"

    short_title = await normalize_title(matcher.client, raw_title)
    results = await matcher.search(query=short_title, top_k=10)
    candidates = [(p, s) for p, s in results if s >= settings.similarity_threshold]
    verified = await verify_matches(matcher.client, short_title, short_title, candidates)
    if verified:
        return verified[0][0].name, "text"
    return None, "text_nomatch"


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


async def main():
    if not CASES_PATH.exists():
        print(f"❗ Немає {CASES_PATH}. Створи файл зі списком тестових кейсів (див. докстрінг).")
        return

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    print(f"Завантажено {len(cases)} тестових кейсів")

    products = load_products(settings.products_path)
    matcher = ProductMatcher(api_key=settings.openai_api_key)
    if Path(f"{settings.index_path}.faiss").exists():
        matcher.load_index(settings.index_path)
    else:
        await matcher.build_index(products)
        matcher.save_index(settings.index_path)

    tp = fp = fn = tn = wrong = 0  # wrong = found, but a different product than expected
    rows = []
    for c in cases:
        url = c["url"]
        expected = c.get("expected")
        predicted, source = await predict(matcher, url)

        if expected is None:
            if predicted is None:
                tn += 1; verdict = "✅ TN"
            else:
                fp += 1; verdict = "❌ FP"
        else:
            if predicted is None:
                fn += 1; verdict = "❌ FN"
            elif _norm(predicted) == _norm(expected):
                tp += 1; verdict = "✅ TP"
            else:
                wrong += 1; verdict = "❌ WRONG"
        rows.append((verdict, source, expected, predicted, url))
        print(f"{verdict:10} [{source:11}] exp={expected!r} got={predicted!r}")

    total = len(cases)
    correct = tp + tn
    print("\n" + "=" * 50)
    print(f"Всього:          {total}")
    print(f"✅ Правильних:   {correct} ({correct/total*100:.1f}%)")
    print(f"   TP (знайшов вірно):   {tp}")
    print(f"   TN (вірно немає):     {tn}")
    print(f"❌ Помилок:      {total - correct}")
    print(f"   FN (не знайшов наявний):  {fn}")
    print(f"   FP (знайшов неіснуючий):  {fp}")
    print(f"   WRONG (не той товар):     {wrong}")
    found = tp + wrong + fp
    if found:
        print(f"\nPrecision (з показаних — вірних): {tp/found*100:.1f}%")
    rel = tp + fn + wrong
    if rel:
        print(f"Recall (з наявних — знайдено):    {tp/rel*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
