"""
bot.py — Production-Ready AI Assistant for Beauty Salon (Telegram Bot)
=======================================================================
Architecture highlights:
  - Fully async: health-check server uses aiohttp (non-blocking) instead of
    threading + stdlib HTTPServer, so the entire process runs on one event loop.
  - Enterprise error handling: every API call is wrapped; the bot NEVER crashes.
  - Standard logging module with configurable level via LOG_LEVEL env var.
  - DRY constants: all prompts, messages, and config live in one place.
  - DB-ready data layer: get_salon_info() is the single source of truth and
    can be swapped to a Firestore fetch with a one-line change.

Dependencies (pip install):
  python-telegram-bot>=20.0
  google-genai
  aiohttp
  firebase-admin          # kept as optional import; not required until Phase 2
"""

import asyncio
import logging
import os
import sys

import aiohttp
from aiohttp import web

from google import genai
from google.genai import types

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
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
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
HEALTH_PORT: int = int(os.environ.get("PORT", 10000))
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Validate mandatory env vars at startup so the error is obvious.
if not BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN is not set. Exiting.")
    sys.exit(1)
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY is not set. Exiting.")
    sys.exit(1)

# --- Gemini generation settings --------------------------------------------
GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.7,
    # system_instruction is injected dynamically per-request (see build_ai_config)
)

# --- User-facing message constants -----------------------------------------
# Keeping every reply string here makes translation/A-B testing trivial.
MSG_WELCOME = (
    "Салом! 👋 Ман AI ассистенти касбии салони зебоӣ ҳастам.\n"
    "Дар бораи хизматрасониҳо, нархҳо ва вақти кории мо савол диҳед. Чӣ хизмат кунам?"
)
MSG_AI_EMPTY_RESPONSE = "Бубахшед, AI дар ин лаҳза ҷавоб дода натавонист. Лутфан дубора кӯшиш кунед."
MSG_AI_ERROR = (
    "Бубахшед, ҳоло мушкилии техникӣ бо сервери AI мавҷуд аст. "
    "Лутфан чанд дақиқа баъд кӯшиш кунед. 🙏"
)
MSG_TELEGRAM_ERROR = "Хатогии дохилӣ рух дод. Лутфан дубора кӯшиш кунед."

# --- System instruction template -------------------------------------------
# The {salon_block} placeholder is filled at runtime by build_system_instruction().
SYSTEM_INSTRUCTION_TEMPLATE = """
Ту ассистенти касбии салони зебоӣ ҳастӣ. Вазифаи ту кӯмак ба мизоҷон дар бораи
хизматрасониҳо, нархҳо ва вақти кор мебошад.

Қоидаҳо:
- Ҷавобҳо кӯтоҳ, равшан ва фаҳмо бошанд.
- Забони хушмуомила ва касбӣ истифода бар.
- Агар саволе берун аз салон бошад, мулоимона гӯй, ки ин берун аз ихтисоси ман аст.

Маълумоти салон:
{salon_block}
""".strip()


# ---------------------------------------------------------------------------
# SECTION 3 — DATA LAYER  (swap-ready for Firebase Firestore)
# ---------------------------------------------------------------------------

# Phase 1: local dictionary — the single source of truth for salon data.
# Phase 2: replace _LOCAL_SALON_DATA + get_salon_info() body with a Firestore
#          fetch such as:
#              doc = db.collection("salons").document("main").get()
#              return doc.to_dict()
#          No other code needs to change.

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


async def get_salon_info() -> dict:
    """
    Data-access layer for salon information.

    Returns the salon data dictionary. In Phase 2 this becomes an async
    Firestore call; the callers don't need to change at all.

    Returns:
        dict: Salon metadata including name, address, work_time, and services.
    """
    # --- Phase 2 Firestore snippet (commented out until needed) -------------
    # from firebase_admin import firestore
    # db = firestore.client()
    # doc = await asyncio.to_thread(
    #     db.collection("salons").document("main").get
    # )
    # return doc.to_dict() if doc.exists else _LOCAL_SALON_DATA
    # ------------------------------------------------------------------------
    return _LOCAL_SALON_DATA


# ---------------------------------------------------------------------------
# SECTION 4 — AI HELPERS
# ---------------------------------------------------------------------------

# Instantiate the Gemini client once at module level (thread-safe, reusable).
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _format_salon_block(salon: dict) -> str:
    """
    Convert the salon dict into a human-readable block for the system prompt.

    Separated from the template so it can be unit-tested independently.
    """
    services_str = "\n".join(
        f"  - {name}: {price}" for name, price in salon.get("services", {}).items()
    )
    return (
        f"Ном: {salon.get('name', '—')}\n"
        f"Суроға: {salon.get('address', '—')}\n"
        f"Вақти кор: {salon.get('work_time', '—')}\n"
        f"Хизматрасониҳо:\n{services_str}"
    )


def _build_system_instruction(salon: dict) -> str:
    """Return the fully rendered system instruction string."""
    return SYSTEM_INSTRUCTION_TEMPLATE.format(salon_block=_format_salon_block(salon))


async def ask_ai(user_text: str) -> str:
    """
    Send a user message to Gemini and return the text reply.

    Fetches fresh salon data on every call so that a Firestore-backed
    get_salon_info() will always serve the latest data without a restart.

    Args:
        user_text: The raw message from the Telegram user.

    Returns:
        str: AI-generated reply, or a polite error fallback.
    """
    try:
        salon = await get_salon_info()
        system_instruction = _build_system_instruction(salon)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=GENERATION_CONFIG.temperature,
        )

        # google-genai's generate_content is synchronous under the hood;
        # wrap it in asyncio.to_thread so it doesn't block the event loop.
        response = await asyncio.to_thread(
            _gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=user_text,
            config=config,
        )

        if not response.text:
            logger.warning("Gemini returned an empty response for input: %r", user_text)
            return MSG_AI_EMPTY_RESPONSE

        return response.text

    except Exception as exc:  # noqa: BLE001 — intentionally broad at the boundary
        logger.error("Gemini API error: %s", exc, exc_info=True)
        return MSG_AI_ERROR


# ---------------------------------------------------------------------------
# SECTION 5 — TELEGRAM HANDLERS
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    user = update.effective_user
    logger.info("User %s (%s) started the bot.", user.id, user.username)
    try:
        await update.message.reply_text(MSG_WELCOME)
    except TelegramError as exc:
        logger.error("Failed to send welcome to user %s: %s", user.id, exc)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle every incoming text message.

    Flow:
      1. Send a "typing…" indicator (non-blocking).
      2. Call Gemini async (non-blocking via asyncio.to_thread inside ask_ai).
      3. Reply with the result.

    Each step is independently guarded so a Telegram API hiccup on step 1
    doesn't prevent the user from receiving a reply at step 3.
    """
    user = update.effective_user
    user_text = update.message.text
    logger.info("Message from %s (%s): %r", user.id, user.username, user_text)

    # Step 1 — typing indicator (best-effort; ignore failures)
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
    except TelegramError as exc:
        logger.warning("Could not send typing action to %s: %s", user.id, exc)

    # Step 2 — AI call
    reply = await ask_ai(user_text)

    # Step 3 — send reply
    try:
        await update.message.reply_text(reply)
    except TelegramError as exc:
        logger.error("Failed to send reply to user %s: %s", user.id, exc, exc_info=True)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global PTB error handler — catches anything that slips through individual
    try-excepts (e.g. network timeouts during polling).

    Logs the exception but does NOT re-raise, so the bot keeps running.
    """
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
      1. Start aiohttp health server.
      2. Build the PTB Application.
      3. Run polling (PTB manages its own internal loop via run_polling, but
         because we call it from an already-running loop we use the lower-level
         initialize/start/idle/stop API instead).
    """
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
    application.add_error_handler(error_handler)

    logger.info("Bot starting. Model: %s. Press Ctrl+C to stop.", GEMINI_MODEL)

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