from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

import os  # Инро дар худи сатри якуми код илова кунед, агар набошад

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Инициализатсияи клайент
client = genai.Client(api_key=GEMINI_API_KEY)

SALON_INFO = {
    "name": "Beauty Salon",
    "address": "Душанбе, марказ",
    "work_time": "09:00 - 20:00",
    "services": {
        "маникюр": "80 сомонӣ",
        "педикюр": "120 сомонӣ",
        "абру": "50 сомонӣ",
        "кирпик": "150 сомонӣ"
    }
}

def ask_ai(text):
    system_instruction = f"""
Ту ассистенти салони зебоӣ ҳастӣ.
Ҷавобҳо кӯтоҳ, равшан ва фаҳмо бошанд. Агар муштарӣ бо забони тоҷикӣ нависад, ҳатман бо тоҷикӣ ҷавоб деҳ.

Маълумоти салон:
Ном: {SALON_INFO['name']}
Суроға: {SALON_INFO['address']}
Вақт: {SALON_INFO['work_time']}
Хизматҳо: {SALON_INFO['services']}
"""

    try:
        # Модели насли нави gemini-2.5-flash-ро истифода мебарем, ки дар лоиҳаҳои Enterprise устувор аст
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )
        )
        
        if response.text:
            return response.text
        return "AI ҷавоб дода натавонист."

    except Exception as e:
        print(f"Истиснои Google API: {e}")
        # Роҳи захиравӣ: Агар боз ҳам хатогии модел шавад, кӯшиш бо формати пурраи Vertex
        try:
            print("Кӯшиши дуюм бо формати Vertex...")
            response = client.models.generate_content(
                model='publishers/google/models/gemini-1.5-flash',
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e2:
            print(f"Хатогии дуюм: {e2}")
            return "Бубахшед, ҳоло мушкилии техникӣ бо сервер мавҷуд аст."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Салом! Ман AI ассистенти салон ҳастам. Чӣ хизмат кунам?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    reply = ask_ai(user_text)
    await update.message.reply_text(reply)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("AI Bot бо муваффақият ба кор даромад...")
    app.run_polling()