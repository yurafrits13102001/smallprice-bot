import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import settings
from bot.handlers import router, set_matcher
from core.matcher import ProductMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    matcher = ProductMatcher(api_key=settings.openai_api_key)

    try:
        matcher.load_index(settings.index_path)
        logger.info("FAISS index loaded")
    except FileNotFoundError:
        logger.error(
            "Index not found. Run: python -m scripts.build_index"
        )
        return

    set_matcher(matcher)

    if matcher.clip_index:
        logger.info("Warming up CLIP model...")
        from core.clip_matcher import preload_model
        await asyncio.to_thread(preload_model)
        logger.info("CLIP model ready")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Bot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
