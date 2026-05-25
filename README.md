# SmallPrice Bot

Telegram-бот для пошуку дублікатів товарів у внутрішній базі.

## Швидкий старт

```bash
# 1. Створити віртуальне середовище
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Встановити залежності
pip install -r requirements.txt

# 3. Створити .env файл
copy .env.example .env
# Заповнити BOT_TOKEN та OPENAI_API_KEY

# 4. Покласти products.xlsx у data/

# 5. Побудувати індекс
python -m scripts.build_index

# 6. Запустити бота
python -m bot.main
```
