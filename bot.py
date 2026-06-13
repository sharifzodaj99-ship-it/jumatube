"""
bot.py — Production-Ready AI Assistant for Beauty Salon (Telegram Bot)
=======================================================================
Phase 2 — Firebase Firestore Integration (per-field env-var credentials)
------------------------------------------------------------------------
Architecture highlights:
  - Fully async: health-check server uses aiohttp (non-blocking) instead of
    threading + stdlib HTTPServer, so the entire process runs on one event loop.
  - Enterprise error handling: every API call is wrapped; the bot NEVER crashes.
  - Standard logging module with configurable level via LOG_LEVEL env var.
  - DRY constants: all prompts, messages, and config live in one place.
  - Firebase credentials are assembled from individual environment variables,
    one per field. This is the only approach that is completely immune to the
    "Invalid JWT Signature" / private-key escaping bug that affects JSON files
    and single-string env vars: each scalar value is safe in an env var, and
    the private key's \\n sequences are restored with a single .replace() call.
  - get_salon_info() fetches from Firestore asynchronously; on any failure it
    falls back to _LOCAL_SALON_DATA so the bot keeps working uninterrupted.

Required environment variables (Render → Environment):
  TELEGRAM_BOT_TOKEN          — Telegram bot token
  GEMINI_API_KEY              — Google Gemini API key
  FIREBASE_TYPE               — "service_account"
  FIREBASE_PROJECT_ID         — e.g. "my-project-12345"
  FIREBASE_PRIVATE_KEY_ID     — 40-char hex string
  FIREBASE_PRIVATE_KEY        — RSA private key (paste with literal \\n or real newlines)
  FIREBASE_CLIENT_EMAIL       — service account email
  FIREBASE_CLIENT_ID          — numeric string
  FIREBASE_AUTH_URI           — https://accounts.google.com/o/oauth2/auth
  FIREBASE_TOKEN_URI          — https://oauth2.googleapis.com/token
  FIREBASE_AUTH_PROVIDER_CERT — https://www.googleapis.com/oauth2/v1/certs
  FIREBASE_CLIENT_CERT        — service account x509 cert URL

Optional:
  PORT                        — health-check server port (default 10000)
  GEMINI_MODEL                — Gemini model name (default gemini-2.5-flash)
  LOG_LEVEL                   — DEBUG / INFO / WARNING (default INFO)

Dependencies (pip install):
  python-telegram-bot>=20.0
  google-genai
  aiohttp
  firebase-admin
"""

import asyncio
import base64
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

import aiohttp
import firebase_admin  # type: ignore
from aiohttp import web
from firebase_admin import credentials, firestore  # type: ignore
from openai import AsyncOpenAI  # type: ignore
from telegram import Update  # type: ignore
from telegram.error import TelegramError  # type: ignore
from telegram.ext import (  # type: ignore
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# SECTION 1 — LOGGING
# Configure once at module level. Level is overridable via LOG_LEVEL env var
# so you can set DEBUG in development and INFO/WARNING in production.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 2 — CONFIGURATION CONSTANTS
# All magic strings and config values live here. To tune the bot's personality,
# change SYSTEM_INSTRUCTION_TEMPLATE. To add a service, edit SALON_DATA.
# ---------------------------------------------------------------------------

# --- Runtime config (read from environment) --------------------------------
BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
HEALTH_PORT: int = int(os.environ.get("PORT", 10000))
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
SALON_ID: str = os.environ.get("SALON_ID", "main")

# Validate mandatory env vars at startup so the error is obvious.
if not BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN is not set. Exiting.")
    sys.exit(1)
if not DEEPSEEK_API_KEY:
    logger.critical("DEEPSEEK_API_KEY is not set. Exiting.")
    sys.exit(1)

# --- Firebase credential fields (each stored as its own env var) -----------
# WHY individual env vars instead of a JSON file or a single JSON string?
#
# The Firebase private key is a multi-line RSA PEM block. Every other storage
# strategy runs into the same escaping trap:
#
#   • JSON file      → shells, editors, and Render's Secret Files UI all risk
#                      converting the literal \n sequences in the key to real
#                      newlines or back again, silently corrupting the PEM.
#   • Single env var → pasting a JSON blob into a Render env-var field goes
#                      through the browser and Render's own serialisation,
#                      which double-escapes or strips backslashes unpredictably.
#
# Individual env vars sidestep every one of these issues:
#   • Each scalar field (project_id, client_email …) is a plain string —
#     completely safe in any env-var mechanism.
#   • The private key is stored in FIREBASE_PRIVATE_KEY exactly as Render
#     presents it (with literal \n or real newlines — we normalise both).
#   • _build_firebase_credentials() applies one deterministic .replace() call
#     to restore proper newlines before the key is passed to the SDK.
#   • No file I/O, no JSON parsing, no shell quoting — nothing to go wrong.

def _build_firebase_credentials() -> dict | None:
    """
    Assemble the Firebase service-account credential dictionary from
    individual environment variables.

    Returns None (and logs a warning) if any required field is absent,
    so the caller can skip Firebase initialisation gracefully.

    The private-key normalisation step deserves explanation:
      Render stores env var values as-is. When a user pastes a PEM key,
      the newlines may arrive as:
        (a) real newline characters  → the key is already correct
        (b) the two-character sequence \\n → must be replaced with \n
      We call  .replace("\\\\n", "\n")  which handles case (b) without
      breaking case (a), because a real newline is not the string "\\n".
    """
    required_fields = {
        "type":                         "FIREBASE_TYPE",
        "project_id":                   "FIREBASE_PROJECT_ID",
        "private_key_id":               "FIREBASE_PRIVATE_KEY_ID",
        "private_key":                  "FIREBASE_PRIVATE_KEY",
        "client_email":                 "FIREBASE_CLIENT_EMAIL",
        "client_id":                    "FIREBASE_CLIENT_ID",
        "auth_uri":                     "FIREBASE_AUTH_URI",
        "token_uri":                    "FIREBASE_TOKEN_URI",
        "auth_provider_x509_cert_url":  "FIREBASE_AUTH_PROVIDER_CERT",
        "client_x509_cert_url":         "FIREBASE_CLIENT_CERT",
    }

    cred_dict: dict = {}
    missing: list[str] = []

    for json_key, env_var in required_fields.items():
        value = os.environ.get(env_var, "")
        if not value:
            missing.append(env_var)
        else:
            cred_dict[json_key] = value

    if missing:
        logger.warning(
            "Firebase initialisation skipped — the following env vars are not set: %s. "
            "Bot will use local fallback salon data.",
            ", ".join(missing),
        )
        return None

    # -----------------------------------------------------------------------
    # THE CRITICAL FIX — private key newline normalisation
    # -----------------------------------------------------------------------
    # Render (and most CI/CD platforms) may store the private key with the
    # literal two-character sequence \n instead of a real newline character.
    # credentials.Certificate() passes the key to the google-auth RSA parser,
    # which requires real newlines. A broken key causes "Invalid JWT Signature"
    # at the very first Firestore request, often minutes after startup.
    #
    # .replace("\\n", "\n") is safe to call unconditionally:
    #   • If the key already contains real newlines → nothing changes.
    #   • If the key contains literal \n sequences  → they become real newlines.
    # -----------------------------------------------------------------------
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

    return cred_dict

# --- Gemini generation settings --------------------------------------------

    # system_instruction is injected dynamically per-request (see build_ai_config)

# --- User-facing message constants -----------------------------------------
# Keeping every reply string here makes translation/A-B testing trivial.
MSG_WELCOME = (
    "Салом! 👋 Ман мушовири салони зебоӣ ҳастам.\n"
    "Дар бораи хизматрасониҳо, нархҳо ва вақти кории мо савол диҳед. Чӣ хизмат кунам?"
)
MSG_AI_EMPTY_RESPONSE = "Бубахшед, AI дар ин лаҳза ҷавоб дода натавонист. Лутфан дубора кӯшиш кунед."
MSG_AI_ERROR = (
    "Бубахшед, ҳоло мушкилии техникӣ бо сервери AI мавҷуд аст. "
    "Лутфан чанд дақиқа баъд кӯшиш кунед. 🙏"
)
MSG_TELEGRAM_ERROR = "Хатогии дохилӣ рух дод. Лутфан дубора кӯшиш кунед."

# --- System instruction template -------------------------------------------
# Огоҳӣ барои ИИ: Фармон ва қоидаи баргардонидани JSON барои заказҳо
SYSTEM_INSTRUCTION_TEMPLATE = """
Ту ассистенти касбии салони зебоӣ ҳастӣ. Вазифаи ту кӯмак ба мизоҷон дар бораи
хизматрасониҳо, нархҳо ва вақти кор мебошад.

Қоидаҳо:
- Ҷавобҳо кӯтоҳ, равшан ва фаҳмо бошанд.
- Забони хушмуомила ва касбӣ истифода бар.
- Агар саволе берун аз салон бошад, мулоимона гӯй, ки ин берун аз ихтисоси ман аст.

ҚОИДАИ ҚАТЪИИ ЗАПИС КАРДАНИ КЛИЕНТ:
Ҳамин ки клиент хоҳиши сабти ном кардан (запись) кард ва маълумотро дод (Ном, Хидмат, Вақт/Рӯз ва Телефон), ту ба ӯ вежливо тасдиқ мекунӣ.
Дар АМАН КУНҶИ (ОХИРИ) ҷавоби худ, ту МУТЛАҚО ВА ОБЯЗАТЕЛЬНО бояд ин теги техникии JSON-ро ТАНҲО дар як сатр илова кунӣ (агар ягон маълумот набошад, ба ҷояш null гузор):

[BOOKING_DATA:{{"name": "Номи клиент", "service": "Номи хидмат", "time": "Вақт ва рӯз", "phone": "Рақами телефон"}}]

Маълумоти салон:
{salon_block}
""".strip()


# ---------------------------------------------------------------------------
# SECTION 3 — DATA LAYER  (Phase 2: Firebase Firestore)
# ---------------------------------------------------------------------------

# Local fallback — used when Firestore is unreachable or not configured.
# Keeping this here means the bot is always operational even during a DB outage.
_LOCAL_SALON_DATA: dict = {
    "name": "Beauty Salon",
    "address": "Душанбе, марказ",
    "work_time": "09:00 - 20:00",
    "services": {
        "Маникюр": "80 сомонӣ",
        "Педикюр": "120 сомонӣ",
        "Абру": "50 сомонӣ",
        "Кирпик": "150 сомонӣ",
    },
}

# Module-level reference to the Firestore client.
# Populated by _init_firebase() at startup; None if initialisation fails.
_firestore_client = None


def _init_firebase() -> None:
    """Инициализацияи касбӣ ва бехатари Firebase тавассути Base64 Environment Variable."""
    global _firestore_client
    
    # 1. Хондани сатри дарози Base64 аз муҳити сервер (Render)
    b64_creds = os.environ.get("FIREBASE_CREDS_BASE64", "")
    
    if not b64_creds:
        logger.warning("Тағйирёбандаи FIREBASE_CREDS_BASE64 ёфт нашуд. Хотираи локалӣ кор мекунад.")
        return

    try:
        # 2. Декод кардани сатри Base64 ба байтҳо ва табдил ба матни JSON
        decoded_bytes = base64.b64decode(b64_creds)
        creds_dict = json.loads(decoded_bytes)
        
        # 3. Мустақим супоридани маълумот ба SDK-и Google (Бе сохтани ягон файл!)
        cred = credentials.Certificate(creds_dict)
        
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            pass  # Агар аллакай инстансия сохта шуда бошад

        _firestore_client = firestore.client()
        logger.info("Firebase Admin SDK бомуваффақият бо усули Бехатар (Base64) фаъол шуд! 🔐✅")
        
    except Exception as exc:
        logger.error("Хатогии касбӣ дар фаъолкунии Firebase бо Base64: %s", exc, exc_info=True)


async def get_salon_info() -> dict:
    """
    Data-access layer for salon information.

    Attempts to fetch the document from Cloud Firestore dynamically using SALON_ID.
    Falls back to _LOCAL_SALON_DATA on any error so the bot keeps running.

    The Firestore SDK call is synchronous (google-cloud-firestore uses gRPC
    under the hood but exposes a blocking API in the standard client). We wrap
    it in asyncio.to_thread() so it never blocks the event loop.

    Returns:
        dict: Salon metadata including name, address, work_time, and services.
    """
    if _firestore_client is None:
        # Firebase was not initialised (missing env var or init error).
        logger.debug("Firestore client unavailable; returning local salon data.")
        return _LOCAL_SALON_DATA

    try:
        # 🔥 ИСЛОҲИ КАСБӢ 1: Ба ҷои калимаи статикии "main", мо {SALON_ID}-ро мегузорем!
        # (Барои муштарии ҳозираи ту SALON_ID дар Render ба "main" баробар мешавад)
        doc_ref = _firestore_client.collection("salons").document(SALON_ID)
        doc_snapshot = await asyncio.to_thread(doc_ref.get)

        if not doc_snapshot.exists:
            logger.warning(
                f"Firestore document 'salons/{SALON_ID}' does not exist. "
                "Falling back to local salon data."
            )
            return _LOCAL_SALON_DATA

        data: dict = doc_snapshot.to_dict()
        logger.debug(f"Salon data fetched from Firestore for {SALON_ID}: {data}")
        return data

    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"Firestore fetch failed for {SALON_ID} — using local fallback data. Error: {exc}",
            exc_info=True,
        )
        return _LOCAL_SALON_DATA
    # --- ИЛОВАИ НАВ ДАР SECTION 3: БАЗАИ МАЪЛУМОТИ ЗАКАЗҲО ---

def init_booking_db() -> None:
    """Сохтани базаи маълумоти локалии SQLite барои сабти заказҳо."""
    conn = sqlite3.connect("salon_bookings.db", timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salon_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            name TEXT,        -- ✅ Ислоҳ шуд (ба ҷои client_name)
            service TEXT,
            time TEXT,        -- ✅ Ислоҳ шуд (ба ҷои booking_time)
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Базаи маълумоти SQLite барои заказҳо омода аст.")

def save_booking_to_db(user_id: int, username: str, name: str, service: str, time: str, phone: str) -> int:
    """
    Saves a booking to the local SQLite database. 
    Automatically creates the table if it does not exist.
    """
    import sqlite3
    
    # Пайвастшавӣ ба базаи локалӣ
    conn = sqlite3.connect("salon_bookings.db")
    cursor = conn.cursor()
    
    # 🟢 ИСЛОҲИ КРИТИКӢ: Агар таблица набошад, онро автоматӣ месозем
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
    
    # Сабти маълумоти заказ
    cursor.execute("""
        INSERT INTO salon_orders (user_id, username, name, service, time, phone)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, username, name, service, time, phone))
    
    conn.commit()
    booking_id = cursor.lastrowid
    conn.close()
    
    return booking_id
# ---------------------------------------------------------------------------
# SECTION 4 — AI HELPERS & MEMORY
# ---------------------------------------------------------------------------

# Instantiate the Gemini/DeepSeek client once at module level
_deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


def _format_salon_block(salon: dict) -> str:
    """Convert the salon dict into a human-readable block for the system prompt."""
    services_str = "\n".join(
        f"  - {name}: {price}" for name, price in salon.get("services", {}).items()
    )
    return (
        f"Ном: {salon.get('name', '—')}\n"
        f"Суроға: {salon.get('address', '—')}\n"
        f"Вақти кор: {salon.get('work_time', '—')}\n"
        f"Хизматрасониҳо:\n{services_str}"
    )


async def ask_ai(user_id: int, user_text: str) -> str:
    """
    Asks the AI (DeepSeek) by dynamically injecting the system prompt, 
    loading conversation history from Firestore, and managing the context.
    """
    salon_data = await get_salon_info()
    salon_block = _format_salon_block(salon_data)

    system_prompt = (
        "Ту як ёрдамчии касбӣ ва ҳушманд бо номи 'Beauty AI' барои салони ҳусн ҳастӣ.\n"
        f"Маълумот дар бораи saлон:\n{salon_block}\n\n"
        "ҚОИДАҲОИ АСОСӢ:\n"
        "1. Агар муштарӣ аллакай салом дода бошад ё суҳбат давом дошта бошад, ДИГАР САЛОМ НАФИРИСТ ва худро аз нав муаррифӣ накун!\n"
        "2. Танҳо ба саволи муштарӣ кӯтоҳ ва мушаххас ҷавоб деҳ ва марҳила ба марҳила маълумоти намерасидаро (ном, телефон, хизматрасонӣ, вақт) пурс.\n"
        "3. Ҳамин ки ҳамаи 4 маълумот (ном, телефон, вақт, хизматрасонӣ) пурра шуд, АЙНАН дар охири паёми худ ин теги JSON-ро часпон:\n"
        "[BOOKING_DATA:{\"name\": \"Номи клиент\", \"service\": \"Номи хизмат\", \"time\": \"Рӯз ва соат\", \"phone\": \"Телефон\"}]\n"
        "4. Ҳамеша бо забони тоҷикии ширин ва хушмуомила ҷавоб гардон."
    )

    messages = [{"role": "system", "content": system_prompt}]
    user_history_ref = None
    history_data = []
    
    if _firestore_client is not None:
        try:
            user_history_ref = _firestore_client.collection("chat_histories").document(str(user_id))
            doc = await asyncio.to_thread(user_history_ref.get)
            if doc.exists:
                history_data = doc.to_dict().get("messages", [])
                for msg in history_data[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        except Exception as e:
            logger.error(f"Хатогӣ ҳангоми хондани хотираи чат: {e}")

    messages.append({"role": "user", "content": user_text})

    try:
        response = await _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        ai_reply = response.choices[0].message.content

        if _firestore_client is not None and user_history_ref is not None:
            try:
                history_data.append({"role": "user", "content": user_text})
                history_data.append({"role": "assistant", "content": ai_reply})
                history_data = history_data[-20:]
                await asyncio.to_thread(user_history_ref.set, {"messages": history_data})
            except Exception as e:
                logger.error(f"Хатогӣ ҳангоми сабти хотираи чат: {e}")
            

        return ai_reply

    except Exception as exc:
        logger.error(f"DeepSeek API Error: {exc}", exc_info=True)
        return "Бубахшед, дар занҷири коркарди маълумот хатогӣ рух дод. Лутфан қайди худро аз нав нависед."


# ---------------------------------------------------------------------------
# SECTION 5 — HANDLERS & PTB ENTRY POINT
# ---------------------------------------------------------------------------

ADMIN_CHAT_ID = int(
    os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "8122251511")
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Салом! 👋 Ба Beauty AI хуш омадед.\n\n"
        "Барои гирифтани маълумот дар бораи хизматрасониҳо ё сабти ном паём нависед."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_text = update.message.text

    logger.info(
        "Message from %s (%s): %r",
        user.id,
        user.username,
        user_text
    )

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
    except TelegramError as exc:
        logger.warning(
            "Could not send typing action to %s: %s",
            user.id,
            exc
        )

    reply = await ask_ai(
        user_id=user.id,
        user_text=user_text
    )

    if "[BOOKING_DATA:" in reply:
        try:
            parts = reply.split("[BOOKING_DATA:")
            clean_reply = parts[0].strip()
            json_str = parts[1].split("]")[0].strip()

            if not json_str:
                raise ValueError("Empty booking JSON")

            if not json_str.startswith("{"):
                raise ValueError("Invalid booking JSON format")

            data = json.loads(json_str)

            required_fields = [
                "name",
                "service",
                "time",
                "phone"
            ]

            for field in required_fields:
                if not data.get(field):
                    raise ValueError(
                        f"Missing required field: {field}"
                    )

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
                f"🔔 ЗАКАЗИ НАВ (ID: {b_id})\n\n"
                f"👤 Клиент: {data.get('name')}\n"
                f"✨ Хизматрасонӣ: {data.get('service')}\n"
                f"📅 Вақт: {data.get('time')}\n"
                f"📞 Телефон: {data.get('phone')}\n"
                f"💬 Telegram: @{user.username or 'нест'}"
            )

            # ✅ Шакли касбӣ ва бехато:
            await context.bot.send_message(
    chat_id=int(ADMIN_CHAT_ID),  # Адади бутун (Integer) кардани ID
    text=admin_msg
)

            await update.message.reply_text(clean_reply)
            return

        except Exception as exc:
            logger.error("Booking JSON processing error: %s", exc)
            reply = reply.split("[BOOKING_DATA:")[0].strip()
            # Сатри тиллоӣ: Агар хатогӣ шавад, паёми тозаро мефиристем, то бот хомӯш намонад
            await update.message.reply_text(reply)
            return

    # Ин блок барои паёмҳои оддӣ аст (вақте ки теги JSON умуман нест)
    try:
        await update.message.reply_text(reply)
    except TelegramError as exc:
        logger.error("Failed to send reply to user: %s", exc)


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    logger.error(
        "PTB unhandled exception while processing update %s: %s",
        update,
        context.error,
        exc_info=context.error,
    )


# ---------------------------------------------------------------------------
# SECTION 6 — ASYNC HEALTH-CHECK SERVER  (replaces blocking HTTPServer + Thread)
# ---------------------------------------------------------------------------
# Using aiohttp instead of stdlib HTTPServer + threading means:
#   • No OS threads — the health server runs on the same event loop as the bot.
#   • No GIL contention under high message load.
#   • Cleaner shutdown (aiohttp runner's cleanup() is called on exit).

async def _health_handler(request: web.Request) -> web.Response:  # noqa: ARG001
    """Respond 200 OK to Render's /health or root health-check ping."""
    return web.Response(text="Bot is running!", status=200)


async def start_health_server() -> web.AppRunner:
    """
    Create and start the aiohttp health-check server.

    Returns the runner so the caller can shut it down cleanly on exit.
    """
    app = web.Application()
    app.router.add_get("/", _health_handler)
    app.router.add_get("/health", _health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=HEALTH_PORT)
    await site.start()
    logger.info("Health-check server listening on port %d.", HEALTH_PORT)
    return runner


# ---------------------------------------------------------------------------
# SECTION 7 — APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Build and run the Telegram bot together with the async health-check server
    on a single asyncio event loop.

    Lifecycle:
      1. Initialise Firebase (synchronous, fast, done once).
      2. Start aiohttp health server.
      3. Build the PTB Application.
      4. Run polling via the explicit lifecycle API (avoids loop conflicts).
    """
    # --- Firebase -----------------------------------------------------------
    # _init_firebase() is synchronous and completes in milliseconds at startup.
    # It must run before the first get_salon_info() call.
    init_booking_db()
    _init_firebase()
    # --- Health server ------------------------------------------------------
    health_runner = await start_health_server()

    # --- PTB application ----------------------------------------------------
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        # Optionally tune connection pool and timeouts here for high traffic:
        # .connection_pool_size(16)
        # .read_timeout(30)
        # .write_timeout(30)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_error_handler(error_handler) # type: ignore

    logger.info("Bot starting with DeepSeek Model: %s. Press Ctrl+C to stop.", DEEPSEEK_MODEL)

    # PTB's run_polling() calls loop.run_forever() internally, which conflicts
    # with our already-running loop. We use the explicit lifecycle API instead.
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            drop_pending_updates=True,  # ignore messages sent while bot was offline
            allowed_updates=Update.ALL_TYPES,
        )

        # Keep running until SIGINT / SIGTERM
        await asyncio.Event().wait()  # block forever (until cancelled)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Stopping bot and health server…")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await health_runner.cleanup()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())