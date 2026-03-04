import os

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# Felo AI
FELO_API_KEY = os.getenv("FELO_API_KEY", "YOUR_FELO_API_KEY")
FELO_API_URL = os.getenv("FELO_API_URL", "https://openapi.felo.ai/v2/chat")

# Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "YOUR_REPLICATE_TOKEN")

# OpenAI (Sora 2)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
