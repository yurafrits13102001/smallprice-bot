import asyncio
import httpx

async def main():
    url = "https://www.amazon.com/STOBOK-Multitool-Screwdriver-Ballpoint-Combination/dp/B0CDMJ85SF"
    async with httpx.AsyncClient(follow_redirects=True, timeout=10, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }) as client:
        resp = await client.get(url)
        body = resp.text
        print(f"Status: {resp.status_code}")
        print(f"Final URL: {resp.url}")
        print(f"Body length: {len(body)}")

        # Шукаємо title
        import re
        title = re.search(r'<title[^>]*>(.*?)</title>', body, re.DOTALL)
        print(f"Title: {title.group(1).strip() if title else 'NO TITLE'}")

        og = re.search(r'og:title["\s]+content="([^"]*)"', body)
        print(f"OG Title: {og.group(1) if og else 'NO OG:TITLE'}")

        # Маркери мертвої сторінки
        markers = ["notFound", "not-found", "no longer available", "该商品已下架", "rgv587_flag", "page-not-found",
                   "couldn't find that page", "SORRY"]
        for m in markers:
            if m.lower() in body.lower():
                print(f"FOUND MARKER: {m}")

        print(f"\nFirst 500 chars of body:\n{body[:500]}")

asyncio.run(main())
