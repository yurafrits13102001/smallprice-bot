"""Image-coverage diagnostics for the CLIP build.

Answers the real question: of all catalog PRODUCTS, how many got an image —
and for the ones that did NOT, why (which sources were tried and all failed vs.
never tried). Reads the cache the CLIP build writes (`clip_image_cache.json`):
each catalog URL maps to a scraped image URL (or null), and an "apify::<url>"
key records the residential-proxy fallback result.

A product is "covered" if ANY of its URLs yielded an image via EITHER path —
that mirrors core.matcher.build_clip_index_async, which takes the first hit.

Run:  python -m scripts.diagnose_images
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from bot.config import settings
from core.database import Product  # noqa: F401 — needed to unpickle the index meta

_MARKETPLACES = ("aliexpress", "amazon", "1688", "temu", "tiktok", "ebay", "alibaba")


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "?"
    for m in _MARKETPLACES:
        if m in host:
            return m
    return host or "?"


def _product_urls(p) -> list[str]:
    return [u for u in [p.link] + p.supplier_links if u and str(u).startswith("http")]


def _load_products() -> list:
    meta_path = Path(f"{settings.index_path}.meta")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)["products"]
    from core.database import load_products
    return load_products(settings.products_path)


def main() -> None:
    products = _load_products()
    cache_path = Path(settings.index_path).parent / "clip_image_cache.json"
    if not cache_path.exists():
        print(f"❗ Нема {cache_path}. Спершу збудуйте CLIP-індекс (завантажте xlsx у бота).")
        return
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def direct(url: str):
        """Returns 'hit' | 'miss' | 'untried' for the direct (curl_cffi) scrape."""
        if url not in cache:
            return "untried"
        return "hit" if cache.get(url) else "miss"

    def apify(url: str):
        key = f"apify::{url}"
        if key not in cache:
            return "untried"
        return "hit" if cache.get(key) else "miss"

    def _ns(prefix: str, url: str):
        key = f"{prefix}::{url}"
        if key not in cache:
            return "untried"
        return "hit" if cache.get(key) else "miss"

    def firecrawl(url: str):
        return _ns("fc", url)

    def playwright(url: str):
        return _ns("pw", url)

    # Per-domain URL-level outcome, split by path.
    dom_direct = defaultdict(lambda: defaultdict(int))
    dom_apify = defaultdict(lambda: defaultdict(int))

    covered = 0           # product has an image from ANY path
    covered_direct = 0    # ...from the direct scrape
    covered_firecrawl = 0 # ...recovered by Firecrawl (direct missed)
    covered_playwright = 0
    covered_apify = 0
    no_url = 0
    all_failed = 0        # every url tried on every path and all failed
    some_untried = 0      # missing image but at least one url never fully tried
    missing_dom_share = defaultdict(int)  # domains appearing on products with no image

    for p in products:
        urls = _product_urls(p)
        if not urls:
            no_url += 1
            continue

        has_direct = has_fc = has_pw = has_apify = False
        fully_tried = True
        for u in urls:
            d = direct(u)
            dom_direct[_domain(u)][d] += 1
            dom_apify[_domain(u)][apify(u)] += 1
            if d == "hit":
                has_direct = True
            if firecrawl(u) == "hit":
                has_fc = True
            if playwright(u) == "hit":
                has_pw = True
            if apify(u) == "hit":
                has_apify = True
            # untried if direct never ran, or it missed and no fallback ran for it
            if d == "untried" or (
                d == "miss"
                and firecrawl(u) == "untried"
                and playwright(u) == "untried"
                and apify(u) == "untried"
            ):
                fully_tried = False

        if has_direct or has_fc or has_pw or has_apify:
            covered += 1
            # attribute to the first source that worked (direct is free, then fallbacks)
            if has_direct:
                covered_direct += 1
            elif has_fc:
                covered_firecrawl += 1
            elif has_pw:
                covered_playwright += 1
            else:
                covered_apify += 1
        else:
            for u in urls:
                missing_dom_share[_domain(u)] += 1
            if fully_tried:
                all_failed += 1
            else:
                some_untried += 1

    total = len(products)
    with_url = total - no_url
    print("=" * 60)
    print(f"Товарів усього:            {total}")
    print(f"  без жодного http-URL:    {no_url}")
    print(f"  з ≥1 URL:                {with_url}")
    print("-" * 60)
    pct = lambda n: f"{n / with_url * 100:.1f}%" if with_url else "—"
    print(f"✅ З ФОТО (будь-який шлях): {covered}  ({pct(covered)})")
    print(f"     — прямим скрейпом:    {covered_direct}")
    print(f"     — через Firecrawl:    {covered_firecrawl}")
    if covered_playwright:
        print(f"     — через Playwright:   {covered_playwright}")
    if covered_apify:
        print(f"     — через Apify:        {covered_apify}")
    print(f"❌ БЕЗ ФОТО:               {with_url - covered}")
    print(f"     — всі джерела пробували й усі провалились: {all_failed}")
    print(f"     — є непробувані джерела (білд недокрутив):  {some_untried}")

    print("\n" + "=" * 60)
    print("URL по доменах — ПРЯМИЙ скрейп (curl_cffi):")
    print(f"{'домен':14} {'hit':>6} {'miss':>6} {'untried':>8} {'hit%':>6}")
    for dom in sorted(dom_direct, key=lambda d: -sum(dom_direct[d].values())):
        s = dom_direct[dom]
        tried = s["hit"] + s["miss"]
        rate = f"{s['hit'] / tried * 100:.0f}%" if tried else "—"
        print(f"{dom:14} {s['hit']:6} {s['miss']:6} {s['untried']:8} {rate:>6}")

    print("\nURL по доменах — APIFY fallback (residential):")
    print(f"{'домен':14} {'hit':>6} {'miss':>6} {'untried':>8} {'hit%':>6}")
    for dom in sorted(dom_apify, key=lambda d: -sum(dom_apify[d].values())):
        s = dom_apify[dom]
        tried = s["hit"] + s["miss"]
        rate = f"{s['hit'] / tried * 100:.0f}%" if tried else "—"
        print(f"{dom:14} {s['hit']:6} {s['miss']:6} {s['untried']:8} {rate:>6}")

    if missing_dom_share:
        print("\nДомени на товарах БЕЗ фото (де рвуться всі джерела):")
        for dom, c in sorted(missing_dom_share.items(), key=lambda x: -x[1])[:12]:
            print(f"  {c:6}  {dom}")


if __name__ == "__main__":
    main()
