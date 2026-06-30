from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
    )

    bot_token: str
    openai_api_key: str
    products_path: str = "data/products.xlsx"
    index_path: str = "data/faiss_index"
    similarity_threshold: float = 0.50
    apify_token: str = ""
    # CLIP image-build scraping. Low concurrency + a generous timeout avoids the
    # request burst that AliExpress (and others) throttle. Tune via .env if needed.
    clip_scrape_concurrency: int = 6
    clip_scrape_timeout: float = 15.0
    # Headless-browser image fallback (Playwright) for JS-only pages (1688,
    # AliExpress). Off by default; enable on a host with the browser installed.
    use_playwright: bool = False
    playwright_concurrency: int = 3
    playwright_proxy: str = ""
    # Firecrawl: managed scraper that beats the 403/captcha wall on AliExpress &
    # 1688. proxy: "" (basic) | "auto" (escalate on block) | "stealth" (best for
    # 1688, more credits). Empty key disables it.
    firecrawl_api_key: str = ""
    firecrawl_proxy: str = "auto"
    firecrawl_concurrency: int = 4


settings = Settings()
