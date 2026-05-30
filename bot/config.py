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
    similarity_threshold: float = 0.75
    apify_token: str = ""


settings = Settings()
