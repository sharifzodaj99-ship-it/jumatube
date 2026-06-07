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
import logging
import os
import sys

import aiohttp
from aiohttp import web

import firebase_admin
from firebase_admin import credentials, firestore

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
    """
    Initialise the Firebase Admin SDK using a credential dictionary assembled
    from individual environment variables by _build_firebase_credentials().

    Flow:
      1. Call _build_firebase_credentials() to collect and validate all fields.
         Returns None if any field is missing → skip init, use local fallback.
      2. Pass the dict directly to credentials.Certificate().
         The SDK accepts a dict, a file path, or a file-like object; a dict is
         the safest choice because it bypasses all file I/O and JSON parsing.
      3. Call firebase_admin.initialize_app() with the credential.
      4. Cache firestore.client() so get_salon_info() can reuse it cheaply.

    Error handling:
      - Missing env vars  → logged as WARNING by _build_firebase_credentials().
      - ValueError        → SDK already initialised (hot-reload / test runner).
      - Any other error   → logged as ERROR with full traceback; bot continues
                            on local fallback data.
    """
    global _firestore_client  # noqa: PLW0603 — intentional module-level state

    cred_dict = _build_firebase_credentials()
    if cred_dict is None:
        # Warning already emitted inside _build_firebase_credentials().
        return

    try:
        # Pass the Python dict directly — zero file I/O, zero JSON parsing.
        cred = credentials.Certificate(cred_dict)

        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            # Already initialised (e.g. hot-reloader triggered main() twice).
            logger.debug("Firebase app already initialised; reusing existing instance.")

        _firestore_client = firestore.client()
        logger.info(
            "Firebase Admin SDK initialised successfully (project: %s).",
            cred_dict.get("project_id", "unknown"),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Firebase initialisation failed — Firestore disabled. "
            "Bot will use local fallback data. Error: %s",
            exc,
            exc_info=True,
        )


async def get_salon_info() -> dict:
    """
    Data-access layer for salon information.

    Attempts to fetch the 'salons/main' document from Cloud Firestore.
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
        # Offload the blocking gRPC call to a thread-pool worker.
        doc_ref = _firestore_client.collection("salons").document("main")
        doc_snapshot = await asyncio.to_thread(doc_ref.get)

        if not doc_snapshot.exists:
            logger.warning(
                "Firestore document 'salons/main' does not exist. "
                "Falling back to local salon data."
            )
            return _LOCAL_SALON_DATA

        data: dict = doc_snapshot.to_dict()
        logger.debug("Salon data fetched from Firestore: %s", data)
        return data

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Firestore fetch failed — using local fallback data. Error: %s",
            exc,
            exc_info=True,
        )
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
      1. Initialise Firebase (synchronous, fast, done once).
      2. Start aiohttp health server.
      3. Build the PTB Application.
      4. Run polling via the explicit lifecycle API (avoids loop conflicts).
    """
    # --- Firebase -----------------------------------------------------------
    # _init_firebase() is synchronous and completes in milliseconds at startup.
    # It must run before the first get_salon_info() call.
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