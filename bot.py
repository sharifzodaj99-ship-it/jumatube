import os
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# 1. Сервери хурди веб барои фиреб додани Render (барои бепул шудан)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Калидҳоро аз Environment Variables-и сервер мехонем
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

SALON_INFO = {
    "name": "Beauty Salon",
    "address": "Душанбе, марказ",
    "work_time": "09:00 - 20:00",
    "services": {"маникюр": "80 сомонӣ", "педикюр": "120 сомонӣ", "абру": "50 сомонӣ", "кирпик": "150 сомонӣ"}
}

def ask_ai(text):
    system_instruction = f"Ту ассистенти салони зебоӣ ҳастӣ. Ҷавобҳо кӯтоҳ бошанд. Маълумот: {SALON_INFO}"
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=text,
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
        )
        return response.text if response.text else "AI ҷавоб дода натавонист."
    except Exception as e:
        print(f"Error: {e}")
        return "Бубахшед, ҳоло мушкилии техникӣ бо сервер мавҷуд аст."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Салом! Ман AI ассистенти салон ҳастам. Чӣ хизмат кунам?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    reply = ask_ai(user_text)
    await update.message.reply_text(reply)

if __name__ == "__main__":
    # Фаъол кардани сервери веб дар замина
    Thread(target=run_health_server, daemon=True).start()
    
    # Иҷрои боти Телеграм
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("AI Bot бо муваффақият ба кор даромад...")
    app.run_polling()