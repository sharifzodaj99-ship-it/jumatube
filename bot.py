import os
import json
import logging
import asyncio
import sqlite3
from typing import Dict, Any, List

# Telegram Bot API Imports (python-telegram-bot v20+)
from telegram import Update, User
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Firebase Admin SDK Imports
import firebase_admin
from firebase_admin import credentials, firestore

# OpenAI/DeepSeek API Client
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# SECTION 1 — LOGGING & PRODUCTION CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Пурра тоза кардани тағйирёбандаҳои кӯҳнаи Gemini ва танзими DeepSeek / Telegram
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
ADMIN_CHAT_ID: str = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "8122251511")

# Танзими ягонаи Кленти DeepSeek API
_deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# Идоракунандаи глобалии Firestore
_firestore_client: Any = None

# ---------------------------------------------------------------------------
# SECTION 2 — THREAD-SAFE DATABASE MANAGERS (SQLite)
# ---------------------------------------------------------------------------
def init_booking_db() -> None:
    """Initialises the local SQLite database with the strict production schema."""
    conn = sqlite3.connect("salon_bookings.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salon_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            name TEXT,
            service TEXT,
            time TEXT,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Базаи маълумоти локалии SQLite бомуваффақият омода шуд.")


def save_booking_to_db(user_id: int, username: str, name: str, service: str, time: str, phone: str) -> int:
    """Saves a formal booking record into the local database securely."""
    conn = sqlite3.connect("salon_bookings.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO salon_orders (user_id, username, name, service, time, phone)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, username, name, service, time, phone))
    conn.commit()
    booking_id = cursor.lastrowid or 0
    conn.close()
    return booking_id


async def get_salon_info() -> dict:
    """Static high-grade fallback provider for context-aware injection."""
    return {
        "name": "Beauty Salon Juma",
        "address": "Душанбе, кӯчаи Рудакӣ 55",
        "work_time": "Ҳар рӯз аз 09:00 то 20:00",
        "services": {
            "Маникюр": "80 сомонӣ",
            "Педикюр": "120 сомонӣ",
            "Абру": "50 сомонӣ",
            "Кирпик": "150 сомонӣ"
        }
    }

# ---------------------------------------------------------------------------
# SECTION 3 — SECURE FIREBASE INITIALIZATION
# ---------------------------------------------------------------------------
def _init_firebase() -> None:
    """Initialises Firebase Admin SDK cleanly via raw environment JSON string."""
    global _firestore_client
    raw_json = os.environ.get("FIREBASE_CONFIG_JSON", "")
    if not raw_json:
        logger.warning("FIREBASE_CONFIG_JSON соз карда нашудааст. Хотираи кӯтоҳмуддат фаъол аст.")
        return

    try:
        cred_dict = json.loads(raw_json)
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(cred_dict)
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            # Firebase app already initialised
            pass

        _firestore_client = firestore.client()
        logger.info("Firebase Admin SDK (Cloud Firestore) бомуваффақият фаъол шуд.")
    except Exception as exc:
        logger.error("Хатогии критикӣ дар фаъолкунии Firebase: %s", exc, exc_info=True)

# ---------------------------------------------------------------------------
# SECTION 4 — CONTEXT-AWARE AI HELPERS & MEMORY MANAGERS
# ---------------------------------------------------------------------------
def _format_salon_block(salon: dict) -> str:
    """Converts structured salon metadata into an analytical layout."""
    services_str = "\n".join(
        f"  - {name}: {price}" for name, price in salon.get("services", {}).items()
    )
    return (
        f"Ном: {salon.get('name')}\n"
        f"Суроға: {salon.get('address')}\n"
        f"Вақти кор: {salon.get('work_time')}\n"
        f"Хизматрасониҳо:\n{services_str}"
    )


async def ask_ai(user_id: int, user_text: str) -> str:
    """Dispatches query to DeepSeek by bundling historical states and rules."""
    salon_data = await get_salon_info()
    salon_block = _format_salon_block(salon_data)

    system_prompt = (
        "Ту як ёрдамчии касбӣ ва ҳушманд бо номи 'Beauty AI' барои салони ҳусн ҳастӣ.\n"
        f"Маълумот дар бораи салон:\n{salon_block}\n\n"
        "ҚОИДАҲОИ АСОСӢ:\n"
        "1. Агар муштарӣ аллакай салом дода бошад ё суҳбат давом дошта бошад, ДИГАР САЛОМ НАФИРИСТ ва худро аз нав муаррифӣ накун!\n"
        "2. Танҳо ба саволи муштарӣ кӯтоҳ ва мушаххас ҷавоб деҳ ва марҳила ба марҳила маълумоти намерасидаро (ном, телефон, хизматрасонӣ, вақт) пурс.\n"
        "3. Ҳамин ки ҳамаи 4 маълумот (ном, телефон, вақт, хизматрасонӣ) пурра шуд, АЙНАН дар охири паёми худ ин теги JSON-ро часпон:\n"
        "[BOOKING_DATA:{\"name\": \"Номи клиент\", \"service\": \"Номи хизмат\", \"time\": \"Рӯз ва соат\", \"phone\": \"Телефон\"}]\n"
        "4. Ҳамеша бо забони тоҷикии ширин ва хушмуомила ҷавоб гардон."
    )

    messages = [{"role": "system", "content": system_prompt}]
    user_history_ref = None
    history_data: List[Dict[str, str]] = []
    
    if _firestore_client is not None:
        try:
            user_history_ref = _firestore_client.collection("chat_histories").document(str(user_id))
            doc = await asyncio.to_thread(user_history_ref.get)
            if doc.exists:
                history_data = doc.to_dict().get("messages", [])
                # Extract only last 10 messages context window
                for msg in history_data[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        except Exception as e:
            logger.error("Хатогӣ ҳангоми хондани хотираи Firestore: %s", e)

    messages.append({"role": "user", "content": user_text})

    try:
        response = await _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,  # type: ignore
            temperature=0.3,
            max_tokens=400
        )
        ai_reply = response.choices[0].message.content or ""

        if _firestore_client is not None and user_history_ref is not None:
            try:
                history_data.append({"role": "user", "content": user_text})
                history_data.append({"role": "assistant", "content": ai_reply})
                await asyncio.to_thread(user_history_ref.set, {"messages": history_data})
            except Exception as e:
                logger.error("Хатогӣ ҳангоми сабти хотираи чат: %s", e)

        return ai_reply
    except Exception as exc:
        logger.error("DeepSeek API Exception: %s", exc, exc_info=True)
        return "Бубахшед, дар занҷири коркарди маълумот хатогӣ рух дод. Лутфан қайди худро аз нав нависед."

# ---------------------------------------------------------------------------
# SECTION 5 — HANDLERS & PRODUCTION ROUTING ENTRY POINT
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start boundary entry vector cleanly."""
    if update.message:
        await update.message.reply_text(
            "Салом! 👋 Ман AI ассистенти касбии салони зебоӣ ҳастам.\n"
            "Дар бораи хизматрасониҳо, нархҳо ва вақти кории мо савол диҳед. Чӣ хизмат кунам?",
            parse_mode=ParseMode.MARKDOWN
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Core message processing junction with integrated async context routing."""
    if not update.message or not update.message.text:
        return

    user: User = update.effective_user  # type: ignore
    user_text: str = update.message.text
    logger.info("Message from %s (%s): %r", user.id, user.username, user_text)

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")  # type: ignore
    except TelegramError as exc:
        logger.warning("Could not send typing action to %s: %s", user.id, exc)

    # 🟢 ИСЛОҲИ КРИТИКӢ: Ирсоли дурусти ду аргумент ба аsk_ai
    reply = await ask_ai(user_id=user.id, user_text=user_text)

    if "[BOOKING_DATA:" in reply:
        try:
            parts = reply.split("[BOOKING_DATA:")
            clean_reply = parts[0].strip()
            json_str = parts[1].split("]")[0].strip()
            
            data = json.loads(json_str)
            
            b_id = await asyncio.to_thread(
                save_booking_to_db,
                user_id=user.id,
                username=user.username or "скрыт",
                name=data.get("name"),
                service=data.get("service"),
                time=data.get("time"),
                phone=data.get("phone")
            )
            
            admin_msg = (
                f"🔔 **ЗАПИСИ НАВ АЗ AI (ID: {b_id})**\n\n"
                f"👤 **Клиент:** {data.get('name')}\n"
                f"✨ **Хизматрасонӣ:** {data.get('service')}\n"
                f"📅 **Вақт ва Рӯз:** {data.get('time')}\n"
                f"📞 **Телефон:** {data.get('phone')}\n"
                f"💬 **Телеграм:** @{user.username or 'нест'}"
            )
            
            # Огоҳинома ба админ бо блоки try-except алоҳида барои амнияти суҳбати мизоҷ
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID, 
                    text=admin_msg, 
                    parse_mode=ParseMode.MARKDOWN
                )
            except TelegramError as admin_exc:
                logger.error("Could not dispatch admin notification to %s: %s", ADMIN_CHAT_ID, admin_exc)
            
            await update.message.reply_text(clean_reply, parse_mode=ParseMode.MARKDOWN)
            return

        except Exception as exc:
            logger.error("Хатогӣ дар парсинги JSON-и заказ: %s", exc, exc_info=True)
            reply = reply.split("[BOOKING_DATA:")[0].strip()

    try:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except TelegramError as exc:
        logger.error("Failed to send reply to user %s: %s", user.id, exc, exc_info=True)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global PTB exception interceptor loop."""
    logger.error("PTB unhandled exception: %s", context.error, exc_info=context.error)


def main() -> None:
    """The enterprise execution gateway for the production-ready instance."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN соз карда нашудааст! Бот қатъ мешавад.")
        return
        
    if not DEEPSEEK_API_KEY:
        logger.critical("DEEPSEEK_API_KEY соз карда нашудааст! Бот қатъ мешавад.")
        return

    # Ташаккули базаи SQLite ва Firebase
    init_booking_db()
    _init_firebase()

    # Сохтани занҷири Application-и Telegram
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Пайваст кардани хендлерҳо бе ягон аломати зиёдатии локалӣ
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    logger.info("Алоқа омода шуд! Бот ба раванди Polling оғоз кард...")
    application.run_polling()


if __name__ == "__main__":
    main()