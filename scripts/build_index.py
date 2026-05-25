import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import settings
from core.database import load_products
from core.matcher import ProductMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info(f"Loading products from {settings.products_path}...")
    products = load_products(settings.products_path)
    logger.info(f"Loaded {len(products)} products")

    matcher = ProductMatcher(api_key=settings.openai_api_key)
    await matcher.build_index(products)

    Path(settings.index_path).parent.mkdir(parents=True, exist_ok=True)
    matcher.save_index(settings.index_path)

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
