import json
import logging
import asyncio
import io
import os
import tempfile
import time
import base64
import sys
from typing import Optional

import aiohttp
import replicate
import requests
from PIL import Image
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
import telegram

from config import TELEGRAM_TOKEN, FELO_API_KEY, FELO_API_URL, REPLICATE_API_TOKEN, OPENAI_API_KEY

# Устанавливаем токен Replicate
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Универсальный обработчик вывода Replicate
# ------------------------------------------------------------
def get_file_url(output):
    """Универсально получает URL из результата replicate.run()."""
    if isinstance(output, list):
        return output[0] if output else None
    elif hasattr(output, 'url'):
        return output.url
    else:
        return str(output)  # если это уже строка

# ------------------------------------------------------------
# Состояния для ConversationHandler
# ------------------------------------------------------------
(
    CHOOSING_ACTION,           # 0: главное меню
    # Существующие подсостояния
    AWAITING_QUESTION,          # 1: вопрос для Felo
    AWAITING_VIDEO_PROMPT,      # 2: описание для промпта видео
    AWAITING_IMAGE_TEXT,        # 3: текст для генерации картинки (Kandinsky)
    AWAITING_EDIT_IMAGE,        # 4: редактирование фото (Kandinsky)
    AWAITING_VIDEO_PHOTO,       # 5: создание видео из фото (Sora)
    # Новые категории
    CHOOSING_VIDEO_CATEGORY,    # 6: выбор модели для генерации видео
    CHOOSING_IMAGE_CATEGORY,    # 7: выбор модели для генерации изображений
    CHOOSING_UPSCALE_CATEGORY,  # 8: выбор модели для улучшения
    # Состояния для конкретных новых моделей
    AWAITING_MINIMAX_PROMPT,    # 9: minimax/video-01 (ожидаем промпт, опционально фото)
    AWAITING_LUMA_VIDEO,        # 10: luma/reframe-video (ждём видео)
    AWAITING_TOPAZ_VIDEO,       # 11: topazlabs/video-upscale (ждём видео)
    AWAITING_IMAGEN_PROMPT,     # 12: google/imagen-4 (ждём промпт)
    AWAITING_FLUX_KONTEXT_PROMPT, # 13: black-forest-labs/flux-kontext-pro (используем одно состояние с 12)
    AWAITING_IDEOGRAM_PROMPT,   # 14: ideogram-ai/ideogram-v3-turbo
    AWAITING_FLUX_PRO_PROMPT,   # 15: black-forest-labs/flux-1.1-pro
    AWAITING_FLUX_DEV_PROMPT,   # 16: black-forest-labs/flux-dev
    AWAITING_TOPAZ_IMAGE,       # 17: topazlabs/image-upscale (ждём фото)
    AWAITING_CODEFORMER_IMAGE,  # 18: szcho/codeformer (ждём фото)
    AWAITING_GFPGAN_IMAGE,      # 19: tencentarc/gfpgan (ждём фото)
) = range(20)

# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
async def send_typing_indicator(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Фоновая задача: отправляет 'печатает...' каждые 4 секунды."""
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass

async def query_felo_api(user_query: str, system_prompt: str = None) -> str:
    """Отправляет запрос к Felo API и возвращает ответ."""
    headers = {
        "Authorization": f"Bearer {FELO_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "TelegramBot/1.0"
    }
    if system_prompt:
        query = f"{system_prompt}\n\nЗапрос пользователя: {user_query}"
    else:
        query = user_query
    payload = {"query": query}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(FELO_API_URL, json=payload, headers=headers, timeout=30) as resp:
                if resp.status == 403:
                    return "❌ Закончилась квота API Felo. Пополните баланс на сайте felo.ai."
                if resp.status != 200:
                    return f"❌ Ошибка сервера: HTTP {resp.status}"
                data = await resp.json()
                if data.get("status") == 200 and data.get("code") == "OK":
                    answer = data["data"]["answer"]
                    resources = data["data"].get("resources", [])
                    if resources:
                        sources_lines = []
                        for r in resources[:3]:
                            title = r.get('title', 'Ссылка')
                            link = r.get('link')
                            if link:
                                sources_lines.append(f"- [{title}]({link})")
                        if sources_lines:
                            answer += "\n\n📚 **Источники:**\n" + "\n".join(sources_lines)
                    return answer
                else:
                    return f"❌ Ошибка Felo API: {data.get('message', 'Неизвестная ошибка')}"
    except asyncio.TimeoutError:
        return "❌ Превышено время ожидания ответа от Felo."
    except Exception as e:
        logger.exception("Ошибка Felo")
        return f"❌ Ошибка соединения: {str(e)}"

# ------------------------------------------------------------
# Существующие функции (1-5)
# ------------------------------------------------------------
async def generate_video_prompt(user_description: str) -> str:
    """Генерация промпта для видео через Felo."""
    system_prompt = """Ты — профессиональный промпт-инженер для AI-видео моделей.
На основе описания пользователя составь идеальный промпт на английском языке для генерации видео.
Правила: указывай движение, окружение, освещение, ракурс, атмосферу.
Пример: "A cat jumps backward in surprise, slow motion, close-up, detailed fur, natural lighting"
Отвечай ТОЛЬКО готовым промптом, без пояснений."""
    return await query_felo_api(user_description, system_prompt)

async def generate_image_from_text(prompt: str) -> tuple[io.BytesIO, str]:
    """
    Генерация изображения через Kandinsky 2.2.
    Возвращает кортеж: (сжатое изображение в BytesIO, URL оригинала).
    """
    try:
        output = replicate.run(
            "ai-forever/kandinsky-2.2:424befb1eae6af8363edb846ae98a11111a39740988baebd279d73fe3ecc92c2",
            input={
                "prompt": prompt,
                "negative_prompt": "low quality, bad quality, blurry, ugly",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 75,
                "guidance_scale": 7.5,
            }
        )
        original_url = get_file_url(output)
        if not original_url:
            raise Exception("Не удалось получить URL изображения")

        # Скачиваем оригинал
        response = requests.get(original_url)
        response.raise_for_status()

        # Открываем и сжимаем
        img = Image.open(io.BytesIO(response.content))
        max_size = 1024  # ограничим размер по большей стороне
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Сохраняем сжатое изображение в BytesIO
        output_bytes = io.BytesIO()
        img.save(output_bytes, format='JPEG', quality=75, optimize=True)
        output_bytes.seek(0)

        return output_bytes, original_url
    except Exception as e:
        logger.exception("Ошибка Replicate (text2img)")
        raise e

async def edit_image_with_prompt(photo_file_id: str, edit_description: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    """Редактирование изображения через Kandinsky 2.2 (img2img)."""
    file = await context.bot.get_file(photo_file_id)
    image_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # Открываем файл для чтения и передаём в replicate
    f = open(tmp_path, "rb")
    try:
        output = replicate.run(
            "ai-forever/kandinsky-2.2:424befb1eae6af8363edb846ae98a11111a39740988baebd279d73fe3ecc92c2",
            input={
                "image": f,
                "prompt": edit_description,
                "negative_prompt": "low quality, bad quality, blurry, ugly",
                "strength": 0.3,
                "num_inference_steps": 75,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024,
            }
        )
        image_url = get_file_url(output)
        if not image_url:
            raise Exception("Не удалось получить URL изображения")
        response = requests.get(image_url)
        response.raise_for_status()
        # Для редактирования тоже можно применить сжатие, но пока оставим как есть
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

async def poll_video_status(session: aiohttp.ClientSession, video_id: str, headers: dict) -> Optional[str]:
    """Опрашивает статус генерации видео Sora."""
    for _ in range(30):
        await asyncio.sleep(5)
        async with session.get(f"https://api.openai.com/v1/videos/{video_id}", headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data["status"] == "completed":
                    return data["task_result"]["videos"][0]["url"]
                elif data["status"] == "failed":
                    raise Exception(f"Генерация не удалась: {data.get('error', {}).get('message', 'Unknown')}")
    raise Exception("Превышено время ожидания")

async def create_video_from_photo(photo_file_id: str, video_prompt: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    """Создаёт видео из фото через Sora 2 с использованием data URI."""
    # Скачиваем фото
    file = await context.bot.get_file(photo_file_id)
    image_bytes = await file.download_as_bytearray()

    # Определяем MIME-тип изображения с помощью PIL
    img = Image.open(io.BytesIO(image_bytes))
    mime_type = Image.MIME[img.format]  # например, 'image/jpeg' или 'image/png'

    # Конвертируем в base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    data_uri = f"data:{mime_type};base64,{image_base64}"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sora-2",
        "prompt": video_prompt,
        "seconds": "8",
        "size": "1280x720",
        "input_reference": data_uri
    }

    async with aiohttp.ClientSession() as session:
        # Запускаем генерацию видео
        async with session.post("https://api.openai.com/v1/videos", json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Ошибка API Sora: {resp.status} - {error_text}")
            data = await resp.json()
            video_id = data["id"]

        # Опрашиваем статус
        video_url = await poll_video_status(session, video_id, headers)

        # Скачиваем готовое видео
        async with session.get(video_url) as video_resp:
            video_bytes = await video_resp.read()
            return io.BytesIO(video_bytes)

# ------------------------------------------------------------
# НОВЫЕ ФУНКЦИИ для каждой модели Replicate
# ------------------------------------------------------------

# ----- Видео модели -----
async def minimax_generate_video(prompt: str, image_file_id: Optional[str] = None, context: Optional[ContextTypes.DEFAULT_TYPE] = None) -> io.BytesIO:
    """minimax/video-01: генерация видео из текста и опционально изображения."""
    input_data = {"prompt": prompt}
    temp_path = None
    f = None
    if image_file_id and context:
        file = await context.bot.get_file(image_file_id)
        image_bytes = await file.download_as_bytearray()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name
        f = open(temp_path, "rb")
        input_data["image"] = f

    try:
        output = replicate.run("minimax/video-01", input=input_data)
        video_url = get_file_url(output)
        if not video_url:
            raise Exception("Не удалось получить URL видео")
        response = requests.get(video_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        if f:
            f.close()
        if temp_path:
            try:
                os.unlink(temp_path)
            except PermissionError:
                logger.warning(f"Не удалось удалить временный файл {temp_path} (возможно, занят)")

async def luma_reframe_video(video_file_id: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    """luma/reframe-video: изменение кадрирования видео."""
    file = await context.bot.get_file(video_file_id)
    video_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    f = open(tmp_path, "rb")
    try:
        output = replicate.run(
            "luma/reframe-video",
            input={
                "video": f,
                "aspect_ratio": "16:9",
                "resize_mode": "crop"
            }
        )
        video_url = get_file_url(output)
        if not video_url:
            raise Exception("Не удалось получить URL видео")
        response = requests.get(video_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

async def topaz_video_upscale(video_file_id: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    """topazlabs/video-upscale: улучшение и масштабирование видео."""
    file = await context.bot.get_file(video_file_id)
    video_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    f = open(tmp_path, "rb")
    try:
        output = replicate.run("topazlabs/video-upscale", input={"video": f})
        video_url = get_file_url(output)
        if not video_url:
            raise Exception("Не удалось получить URL видео")
        response = requests.get(video_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

# ----- Новые модели изображений -----
async def imagen_generate(prompt: str) -> io.BytesIO:
    output = replicate.run("google/imagen-4", input={"prompt": prompt})
    image_url = get_file_url(output)
    if not image_url:
        raise Exception("Не удалось получить URL изображения")
    response = requests.get(image_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

async def flux_kontext_generate(prompt: str) -> io.BytesIO:
    output = replicate.run("black-forest-labs/flux-kontext-pro", input={"prompt": prompt})
    image_url = get_file_url(output)
    if not image_url:
        raise Exception("Не удалось получить URL изображения")
    response = requests.get(image_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

async def ideogram_generate(prompt: str) -> io.BytesIO:
    output = replicate.run("ideogram-ai/ideogram-v3-turbo", input={"prompt": prompt})
    image_url = get_file_url(output)
    if not image_url:
        raise Exception("Не удалось получить URL изображения")
    response = requests.get(image_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

async def flux_pro_generate(prompt: str) -> io.BytesIO:
    output = replicate.run("black-forest-labs/flux-1.1-pro", input={"prompt": prompt})
    image_url = get_file_url(output)
    if not image_url:
        raise Exception("Не удалось получить URL изображения")
    response = requests.get(image_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

async def flux_dev_generate(prompt: str) -> io.BytesIO:
    output = replicate.run("black-forest-labs/flux-dev", input={"prompt": prompt})
    image_url = get_file_url(output)
    if not image_url:
        raise Exception("Не удалось получить URL изображения")
    response = requests.get(image_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

# ----- Улучшение изображений -----
async def topaz_image_upscale(image_file_id: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    file = await context.bot.get_file(image_file_id)
    image_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    f = open(tmp_path, "rb")
    try:
        output = replicate.run("topazlabs/image-upscale", input={"image": f})
        image_url = get_file_url(output)
        if not image_url:
            raise Exception("Не удалось получить URL изображения")
        response = requests.get(image_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

async def codeformer_restore(image_file_id: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    file = await context.bot.get_file(image_file_id)
    image_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    f = open(tmp_path, "rb")
    try:
        output = replicate.run("szcho/codeformer", input={"image": f})
        image_url = get_file_url(output)
        if not image_url:
            raise Exception("Не удалось получить URL изображения")
        response = requests.get(image_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

async def gfpgan_restore(image_file_id: str, context: ContextTypes.DEFAULT_TYPE) -> io.BytesIO:
    file = await context.bot.get_file(image_file_id)
    image_bytes = await file.download_as_bytearray()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    f = open(tmp_path, "rb")
    try:
        output = replicate.run("tencentarc/gfpgan", input={"image": f})
        image_url = get_file_url(output)
        if not image_url:
            raise Exception("Не удалось получить URL изображения")
        response = requests.get(image_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    finally:
        f.close()
        try:
            os.unlink(tmp_path)
        except PermissionError:
            logger.warning(f"Не удалось удалить временный файл {tmp_path} (возможно, занят)")

# ------------------------------------------------------------
# Клавиатуры (меню)
# ------------------------------------------------------------
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("❓ Задать вопрос на любую тему")],
        [KeyboardButton("🎬 Создать промт для видео")],
        [KeyboardButton("🖼️ Создать картинку по тексту (Kandinsky)")],
        [KeyboardButton("✏️ Изменить картинку по описанию (Kandinsky)")],
        [KeyboardButton("🎥 Создать видео из фото (Sora 2)")],
        [KeyboardButton("🎥 НОВЫЕ МОДЕЛИ ВИДЕО")],
        [KeyboardButton("🖼️ НОВЫЕ МОДЕЛИ ИЗОБРАЖЕНИЙ")],
        [KeyboardButton("🔧 УЛУЧШЕНИЕ ФОТО/ВИДЕО")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)

def get_video_models_keyboard():
    keyboard = [
        [KeyboardButton("minimax/video-01 (текст+опц. фото)")],
        [KeyboardButton("luma/reframe-video (изменение кадрирования)")],
        [KeyboardButton("topazlabs/video-upscale (апскейл видео)")],
        [KeyboardButton("🔙 Назад в главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_image_models_keyboard():
    keyboard = [
        [KeyboardButton("google/imagen-4")],
        [KeyboardButton("black-forest-labs/flux-kontext-pro")],
        [KeyboardButton("ideogram-ai/ideogram-v3-turbo")],
        [KeyboardButton("black-forest-labs/flux-1.1-pro")],
        [KeyboardButton("black-forest-labs/flux-dev")],
        [KeyboardButton("🔙 Назад в главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_upscale_models_keyboard():
    keyboard = [
        [KeyboardButton("topazlabs/image-upscale (апскейл фото)")],
        [KeyboardButton("szcho/codeformer (восстановление лиц)")],
        [KeyboardButton("tencentarc/gfpgan (быстрое восстановление лиц)")],
        [KeyboardButton("🔙 Назад в главное меню")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_cancel_keyboard():
    keyboard = [[KeyboardButton("❌ Отмена")]]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

# ------------------------------------------------------------
# Обработчики команд и главного меню
# ------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🌟 Добро пожаловать в AI-бот! Выбери нужную функцию из меню ниже:",
        reply_markup=get_main_menu_keyboard(),
    )
    return CHOOSING_ACTION

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я AI-бот, который может:\n"
        "❓ Задать вопрос — отвечу на любой вопрос (Felo)\n"
        "🎬 Создать промт для видео — составлю идеальный промпт (Felo)\n"
        "🖼️ Создать картинку по тексту (Kandinsky)\n"
        "✏️ Изменить картинку по описанию (Kandinsky)\n"
        "🎥 Создать видео из фото (Sora 2)\n"
        "🎥 НОВЫЕ МОДЕЛИ ВИДЕО: minimax, luma, topaz\n"
        "🖼️ НОВЫЕ МОДЕЛИ ИЗОБРАЖЕНИЙ: Imagen, Flux, Ideogram\n"
        "🔧 УЛУЧШЕНИЕ: апскейл, восстановление лиц\n\n"
        "Выбери пункт в меню и следуй инструкциям!",
        reply_markup=get_main_menu_keyboard(),
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "❌ Действие отменено. Возвращаюсь в главное меню.",
        reply_markup=get_main_menu_keyboard(),
    )
    return CHOOSING_ACTION

async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    user_id = update.effective_user.id
    logger.info(f"Главное меню, выбор: {text}")

    if text == "❓ Задать вопрос на любую тему":
        await update.message.reply_text("Задай свой вопрос:", reply_markup=get_cancel_keyboard())
        return AWAITING_QUESTION
    elif text == "🎬 Создать промт для видео":
        await update.message.reply_text("Опиши, что должно происходить в видео:", reply_markup=get_cancel_keyboard())
        return AWAITING_VIDEO_PROMPT
    elif text == "🖼️ Создать картинку по тексту (Kandinsky)":
        await update.message.reply_text("Напиши, что изобразить:", reply_markup=get_cancel_keyboard())
        return AWAITING_IMAGE_TEXT
    elif text == "✏️ Изменить картинку по описанию (Kandinsky)":
        await update.message.reply_text("Отправь фото, затем напиши инструкцию:", reply_markup=get_cancel_keyboard())
        return AWAITING_EDIT_IMAGE
    elif text == "🎥 Создать видео из фото (Sora 2)":
        await update.message.reply_text("Отправь фото, затем напиши промт для видео:", reply_markup=get_cancel_keyboard())
        return AWAITING_VIDEO_PHOTO
    elif text == "🎥 НОВЫЕ МОДЕЛИ ВИДЕО":
        await update.message.reply_text("Выбери модель для генерации видео:", reply_markup=get_video_models_keyboard())
        return CHOOSING_VIDEO_CATEGORY
    elif text == "🖼️ НОВЫЕ МОДЕЛИ ИЗОБРАЖЕНИЙ":
        await update.message.reply_text("Выбери модель для генерации изображений:", reply_markup=get_image_models_keyboard())
        return CHOOSING_IMAGE_CATEGORY
    elif text == "🔧 УЛУЧШЕНИЕ ФОТО/ВИДЕО":
        await update.message.reply_text("Выбери модель для улучшения:", reply_markup=get_upscale_models_keyboard())
        return CHOOSING_UPSCALE_CATEGORY
    elif text == "❌ Отмена":
        return await cancel(update, context)
    else:
        await update.message.reply_text("Пожалуйста, выбери пункт из меню.", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION

# ------------------------------------------------------------
# Обработчики для старых функций (1-5)
# ------------------------------------------------------------
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    question = update.message.text
    chat_id = update.effective_chat.id
    typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
    try:
        answer = await query_felo_api(question)
        typing_task.cancel()
        if len(answer) > 4000:
            for part in [answer[i:i+4000] for i in range(0, len(answer), 4000)]:
                await update.message.reply_text(part, parse_mode="Markdown")
        else:
            await update.message.reply_text(answer, parse_mode="Markdown")
    except Exception as e:
        typing_task.cancel()
        logger.exception("Ошибка при обработке вопроса")
        await update.message.reply_text(f"❌ Произошла ошибка: {e}")
    await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
    return CHOOSING_ACTION

async def handle_video_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    description = update.message.text
    chat_id = update.effective_chat.id
    typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
    try:
        prompt = await generate_video_prompt(description)
        typing_task.cancel()
        await update.message.reply_text(prompt)
    except Exception as e:
        typing_task.cancel()
        logger.exception("Ошибка при генерации промпта")
        await update.message.reply_text(f"❌ Ошибка: {e}")
    await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
    return CHOOSING_ACTION

async def handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    prompt = update.message.text
    chat_id = update.effective_chat.id

    typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
    try:
        # Генерируем изображение и получаем сжатую версию + ссылку на оригинал
        compressed_image, original_url = await generate_image_from_text(prompt)
        typing_task.cancel()

        # Отправляем сжатое фото с подписью, содержащей ссылку на оригинал
        caption = f"🖼️ Готово!\n🔗 [Оригинал в полном размере]({original_url})"
        await update.message.reply_photo(
            photo=compressed_image,
            caption=caption,
            parse_mode="Markdown",
            read_timeout=120,
            write_timeout=120,
            connect_timeout=120,
            pool_timeout=120
        )
    except asyncio.TimeoutError:
        logger.error("Тайм-аут при отправке изображения")
        await update.message.reply_text("❌ Превышено время отправки изображения. Попробуйте позже.")
    except telegram.error.TimedOut:
        logger.error("Тайм-аут Telegram при отправке изображения")
        await update.message.reply_text("❌ Соединение с Telegram прервалось. Попробуйте ещё раз.")
    except Exception as e:
        typing_task.cancel()
        logger.exception("Ошибка при генерации изображения")
        await update.message.reply_text(f"❌ Ошибка генерации: {e}")

    await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
    return CHOOSING_ACTION

async def handle_edit_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.photo:
        photo_file_id = update.message.photo[-1].file_id
        context.user_data['edit_photo_id'] = photo_file_id
        await update.message.reply_text("Теперь напиши, как изменить фото:", reply_markup=get_cancel_keyboard())
        return AWAITING_EDIT_IMAGE
    elif 'edit_photo_id' in context.user_data:
        edit_description = update.message.text
        photo_file_id = context.user_data['edit_photo_id']
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            result_bytes = await edit_image_with_prompt(photo_file_id, edit_description, context)
            typing_task.cancel()
            await update.message.reply_photo(photo=result_bytes, caption="✏️ Готово!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка при редактировании изображения")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            del context.user_data['edit_photo_id']
        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Сначала отправь фото.", reply_markup=get_cancel_keyboard())
        return AWAITING_EDIT_IMAGE

async def handle_video_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.photo:
        photo_file_id = update.message.photo[-1].file_id
        context.user_data['sora_photo_id'] = photo_file_id
        await update.message.reply_text("Теперь напиши промт для видео:", reply_markup=get_cancel_keyboard())
        return AWAITING_VIDEO_PHOTO
    elif 'sora_photo_id' in context.user_data:
        video_prompt = update.message.text
        photo_file_id = context.user_data['sora_photo_id']
        chat_id = update.effective_chat.id
        await update.message.reply_text("🎬 Генерирую видео (1-2 минуты)...")
        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            video_bytes = await create_video_from_photo(photo_file_id, video_prompt, context)
            typing_task.cancel()
            await update.message.reply_video(video=video_bytes, caption="🎥 Готово!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка при создании видео")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            del context.user_data['sora_photo_id']
        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Сначала отправь фото.", reply_markup=get_cancel_keyboard())
        return AWAITING_VIDEO_PHOTO

# ------------------------------------------------------------
# Обработчики для новых категорий и моделей
# ------------------------------------------------------------
async def handle_video_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Назад в главное меню":
        await update.message.reply_text("Главное меню:", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION

    if text == "minimax/video-01 (текст+опц. фото)":
        context.user_data['current_model'] = 'minimax'
        await update.message.reply_text(
            "Отправь текстовый промт (и, если хочешь, фото для сохранения образа персонажа).\n"
            "Если хочешь отправить фото, пришли его сейчас. Если нет, просто напиши промт.",
            reply_markup=get_cancel_keyboard()
        )
        return AWAITING_MINIMAX_PROMPT
    elif text == "luma/reframe-video (изменение кадрирования)":
        context.user_data['current_model'] = 'luma'
        await update.message.reply_text("Отправь видео, которое нужно перекадрировать:", reply_markup=get_cancel_keyboard())
        return AWAITING_LUMA_VIDEO
    elif text == "topazlabs/video-upscale (апскейл видео)":
        context.user_data['current_model'] = 'topaz_video'
        await update.message.reply_text("Отправь видео для улучшения качества:", reply_markup=get_cancel_keyboard())
        return AWAITING_TOPAZ_VIDEO
    else:
        await update.message.reply_text("Неизвестный выбор. Попробуй снова.", reply_markup=get_video_models_keyboard())
        return CHOOSING_VIDEO_CATEGORY

async def handle_image_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Назад в главное меню":
        await update.message.reply_text("Главное меню:", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION

    model_map = {
        "google/imagen-4": "imagen",
        "black-forest-labs/flux-kontext-pro": "flux_kontext",
        "ideogram-ai/ideogram-v3-turbo": "ideogram",
        "black-forest-labs/flux-1.1-pro": "flux_pro",
        "black-forest-labs/flux-dev": "flux_dev",
    }
    if text in model_map:
        context.user_data['current_model'] = model_map[text]
        await update.message.reply_text("Напиши промт для генерации изображения:", reply_markup=get_cancel_keyboard())
        return AWAITING_IMAGEN_PROMPT
    else:
        await update.message.reply_text("Неизвестный выбор.", reply_markup=get_image_models_keyboard())
        return CHOOSING_IMAGE_CATEGORY

async def handle_upscale_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == "🔙 Назад в главное меню":
        await update.message.reply_text("Главное меню:", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION

    model_map = {
        "topazlabs/image-upscale (апскейл фото)": "topaz_image",
        "szcho/codeformer (восстановление лиц)": "codeformer",
        "tencentarc/gfpgan (быстрое восстановление лиц)": "gfpgan",
    }
    if text in model_map:
        context.user_data['current_model'] = model_map[text]
        await update.message.reply_text("Отправь фото для обработки:", reply_markup=get_cancel_keyboard())
        return AWAITING_TOPAZ_IMAGE
    else:
        await update.message.reply_text("Неизвестный выбор.", reply_markup=get_upscale_models_keyboard())
        return CHOOSING_UPSCALE_CATEGORY

# ------------------------------------------------------------
# Обработчики для конкретных новых моделей
# ------------------------------------------------------------
async def handle_minimax_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.photo:
        photo_file_id = update.message.photo[-1].file_id
        context.user_data['minimax_image'] = photo_file_id
        await update.message.reply_text("Фото получено. Теперь напиши текстовый промт:", reply_markup=get_cancel_keyboard())
        return AWAITING_MINIMAX_PROMPT
    elif update.message.text:
        prompt = update.message.text
        image_id = context.user_data.get('minimax_image')
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            await update.message.reply_text("🎬 Генерирую видео (до минуты)...")
            video_bytes = await minimax_generate_video(prompt, image_id, context)
            typing_task.cancel()
            await update.message.reply_video(video=video_bytes, caption="✅ Видео готово!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка minimax")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            context.user_data.pop('minimax_image', None)
            context.user_data.pop('current_model', None)
        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Отправь текст или фото.", reply_markup=get_cancel_keyboard())
        return AWAITING_MINIMAX_PROMPT

async def handle_luma_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.video:
        video_file_id = update.message.video.file_id
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            await update.message.reply_text("🔄 Обрабатываю видео...")
            video_bytes = await luma_reframe_video(video_file_id, context)
            typing_task.cancel()
            await update.message.reply_video(video=video_bytes, caption="✅ Видео готово!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка luma")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            context.user_data.pop('current_model', None)
        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Пожалуйста, отправь видео.", reply_markup=get_cancel_keyboard())
        return AWAITING_LUMA_VIDEO

async def handle_topaz_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.video:
        video_file_id = update.message.video.file_id
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            await update.message.reply_text("🔄 Улучшаю видео...")
            video_bytes = await topaz_video_upscale(video_file_id, context)
            typing_task.cancel()
            await update.message.reply_video(video=video_bytes, caption="✅ Видео улучшено!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка topaz video")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            context.user_data.pop('current_model', None)
        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Пожалуйста, отправь видео.", reply_markup=get_cancel_keyboard())
        return AWAITING_TOPAZ_VIDEO

async def handle_image_generation_generic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    prompt = update.message.text
    model = context.user_data.get('current_model')
    chat_id = update.effective_chat.id

    typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
    try:
        await update.message.reply_text("🖼️ Генерирую изображение...")
        if model == 'imagen':
            image_bytes = await imagen_generate(prompt)
        elif model == 'flux_kontext':
            image_bytes = await flux_kontext_generate(prompt)
        elif model == 'ideogram':
            image_bytes = await ideogram_generate(prompt)
        elif model == 'flux_pro':
            image_bytes = await flux_pro_generate(prompt)
        elif model == 'flux_dev':
            image_bytes = await flux_dev_generate(prompt)
        else:
            raise Exception("Неизвестная модель")
        typing_task.cancel()
        await update.message.reply_photo(photo=image_bytes, caption="✅ Изображение готово!")
    except Exception as e:
        typing_task.cancel()
        logger.exception("Ошибка генерации изображения")
        await update.message.reply_text(f"❌ Ошибка: {e}")
    finally:
        context.user_data.pop('current_model', None)

    await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
    return CHOOSING_ACTION

async def handle_upscale_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == "❌ Отмена":
        return await cancel(update, context)
    if update.message.photo:
        photo_file_id = update.message.photo[-1].file_id
        model = context.user_data.get('current_model')
        chat_id = update.effective_chat.id

        typing_task = asyncio.create_task(send_typing_indicator(chat_id, context))
        try:
            await update.message.reply_text("🔧 Обрабатываю фото...")
            if model == 'topaz_image':
                image_bytes = await topaz_image_upscale(photo_file_id, context)
            elif model == 'codeformer':
                image_bytes = await codeformer_restore(photo_file_id, context)
            elif model == 'gfpgan':
                image_bytes = await gfpgan_restore(photo_file_id, context)
            else:
                raise Exception("Неизвестная модель")
            typing_task.cancel()
            await update.message.reply_photo(photo=image_bytes, caption="✅ Фото обработано!")
        except Exception as e:
            typing_task.cancel()
            logger.exception("Ошибка улучшения фото")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            context.user_data.pop('current_model', None)

        await update.message.reply_text("Что ещё?", reply_markup=get_main_menu_keyboard())
        return CHOOSING_ACTION
    else:
        await update.message.reply_text("Пожалуйста, отправь фото.", reply_markup=get_cancel_keyboard())
        return AWAITING_TOPAZ_IMAGE

# ------------------------------------------------------------
# Запуск
# ------------------------------------------------------------
def main():
    # Отладочные сообщения
    print("🟢 Запуск main()...")
    sys.stdout.flush()

    # Проверка наличия ключей
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "ВАШ_ТОКЕН_ТЕЛЕГРАМ":
        logger.error("TELEGRAM_TOKEN не задан!")
        return
    if not FELO_API_KEY or FELO_API_KEY == "ВАШ_FELO_КЛЮЧ":
        logger.error("FELO_API_KEY не задан!")
        return
    if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "r8_ВАШ_REPLICATE_ТОКЕН":
        logger.error("REPLICATE_API_TOKEN не задан!")
        return
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-ВАШ_OPENAI_КЛЮЧ":
        logger.warning("OPENAI_API_KEY не задан. Функция Sora 2 будет недоступна.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING_ACTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice)],
            AWAITING_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)],
            AWAITING_VIDEO_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_prompt)],
            AWAITING_IMAGE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_image_generation)],
            AWAITING_EDIT_IMAGE: [
                MessageHandler(filters.PHOTO, handle_edit_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_image)
            ],
            AWAITING_VIDEO_PHOTO: [
                MessageHandler(filters.PHOTO, handle_video_from_photo),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_from_photo)
            ],
            CHOOSING_VIDEO_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_video_category)],
            CHOOSING_IMAGE_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_image_category)],
            CHOOSING_UPSCALE_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_upscale_category)],
            AWAITING_MINIMAX_PROMPT: [
                MessageHandler(filters.PHOTO, handle_minimax_input),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_minimax_input)
            ],
            AWAITING_LUMA_VIDEO: [MessageHandler(filters.VIDEO, handle_luma_video),
                                 MessageHandler(filters.TEXT & ~filters.COMMAND, handle_luma_video)],
            AWAITING_TOPAZ_VIDEO: [MessageHandler(filters.VIDEO, handle_topaz_video),
                                  MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topaz_video)],
            AWAITING_IMAGEN_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_image_generation_generic)],
            AWAITING_TOPAZ_IMAGE: [MessageHandler(filters.PHOTO, handle_upscale_image),
                                  MessageHandler(filters.TEXT & ~filters.COMMAND, handle_upscale_image)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler('help', help_command))

    logger.info("Бот с новыми моделями запущен...")
    print("✅ Бот запущен, начинаю polling...")
    sys.stdout.flush()
    app.run_polling()

if __name__ == "__main__":
    main()
