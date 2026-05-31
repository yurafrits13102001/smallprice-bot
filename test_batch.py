import asyncio
import csv
import re
import logging
import sys
from collections import defaultdict
from urllib.parse import urlparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import httpx

from core.database import load_products
from core.matcher import ProductMatcher
from bot.config import settings

logging.basicConfig(level=logging.WARNING)

DEAD_TITLES = {"page not found", "404 not found", "404", "not found"}


async def check_link_alive(url: str) -> tuple[bool, int]:
    """Returns (alive, status_code). status_code=0 on exception."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code in (403, 404, 410):
                return False, resp.status_code
            title_match = re.search(r'<title[^>]*>(.*?)</title>', resp.text[:5000], re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip().lower()
                if title in DEAD_TITLES:
                    return False, resp.status_code
            return True, resp.status_code
    except Exception:
        return True, 0


def get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return "unknown"


async def run_liveness(products, concurrency: int = 20):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def check_one(product, idx):
        async with sem:
            alive, status = await check_link_alive(product.link)
            results.append((product, alive, status))
            done = len(results)
            if done % 100 == 0:
                print(f"  Liveness progress: {done}/{idx + 1} checked...")

    products_with_links = [(p, i) for i, p in enumerate(products) if p.link]
    total = len(products_with_links)
    print(f"Checking liveness for {total} links (concurrency={concurrency})...")

    tasks = [check_one(p, i) for p, i in products_with_links]
    await asyncio.gather(*tasks)
    return results


async def main():
    # Load products
    print("Loading products...")
    products = load_products(settings.products_path)
    print(f"Loaded {len(products)} products")

    # Load index
    matcher = ProductMatcher(api_key=settings.openai_api_key)
    index_path = settings.index_path
    if Path(f"{index_path}.faiss").exists():
        print("Loading FAISS index...")
        matcher.load_index(index_path)
    else:
        print("Building FAISS index...")
        await matcher.build_index(products)
        matcher.save_index(index_path)

    # === TEST A: URL MATCHING ===
    print("\nRunning URL matching test...")
    match_correct = 0
    match_wrong = 0
    match_none = 0
    no_link = 0

    for product in products:
        if not product.link:
            no_link += 1
            continue
        result = matcher._url_match(product.link)
        if result is None:
            match_none += 1
        elif result[0].name == product.name:
            match_correct += 1
        else:
            match_wrong += 1

    with_links = match_correct + match_wrong + match_none

    # === TEST B: LINK LIVENESS ===
    print("\nRunning liveness test...")
    liveness_results = await run_liveness(products)

    alive_count = sum(1 for _, alive, _ in liveness_results if alive)
    dead_list = [(p, status) for p, alive, status in liveness_results if not alive]
    error_count = sum(1 for _, alive, status in liveness_results if alive and status == 0)

    dead_by_domain = defaultdict(int)
    for product, status in dead_list:
        dead_by_domain[get_domain(product.link)] += 1

    # === REPORT ===
    print("\n" + "=" * 50)
    print("=== URL MATCHING ===")
    print(f"Всього товарів:      {len(products)}")
    print(f"З посиланнями:       {with_links}")
    pct_correct = match_correct / with_links * 100 if with_links else 0
    print(f"✅ Правильний збіг:  {match_correct} ({pct_correct:.1f}%)")
    print(f"❌ Неправильний збіг: {match_wrong}")
    print(f"⚠️  Не знайдено:      {match_none}")
    print(f"🔗 Без посилання:    {no_link}")

    print("\n=== LINK LIVENESS ===")
    total_checked = len(liveness_results)
    pct_alive = alive_count / total_checked * 100 if total_checked else 0
    pct_dead = len(dead_list) / total_checked * 100 if total_checked else 0
    print(f"Перевірено:   {total_checked}")
    print(f"✅ Живі:      {alive_count} ({pct_alive:.1f}%)")
    print(f"💀 Мертві:    {len(dead_list)} ({pct_dead:.1f}%)")
    print(f"❌ Помилки:   {error_count}")

    if dead_by_domain:
        print("\nМертві по доменах:")
        for domain, count in sorted(dead_by_domain.items(), key=lambda x: -x[1]):
            print(f"  {domain}: {count}")

    if dead_list:
        print("\n=== МЕРТВІ ПОСИЛАННЯ ===")
        for i, (product, status) in enumerate(dead_list[:50], 1):
            print(f"{i}. {product.name[:60]} — {product.link} — HTTP {status}")
        if len(dead_list) > 50:
            print(f"  ... і ще {len(dead_list) - 50} (повний список у dead_links.csv)")

    # Save CSV
    csv_path = Path("dead_links.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "link", "status"])
        for product, status in dead_list:
            writer.writerow([product.name, product.link, status])
    print(f"\nМертві посилання збережено: {csv_path} ({len(dead_list)} записів)")


asyncio.run(main())
