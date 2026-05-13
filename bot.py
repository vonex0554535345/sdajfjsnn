import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import os
import logging
from pyrogram import Client, filters
from pyrogram.types import Message
from groq import Groq

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION_STRING = os.environ["TELEGRAM_SESSION"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant. Answer concisely.")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "20"))

groq_client = Groq(api_key=GROQ_API_KEY)
histories: dict[int, list[dict]] = {}
is_active = True

app = Client("userbot", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)


@app.on_message(filters.me & filters.command(["on", "off"], prefixes="/"))
async def toggle(client: Client, message: Message) -> None:
    global is_active
    is_active = message.command[0] == "on"
    status = "включён ✅" if is_active else "выключен ❌"
    await message.reply_text(f"Бот {status}")


@app.on_message(filters.private & filters.incoming & ~filters.me)
async def handle_message(client: Client, message: Message) -> None:
    if not is_active or not message.text:
        return

    user_id = message.from_user.id
    history = histories.setdefault(user_id, [])
    history.append({"role": "user", "content": message.text})

    if len(history) > MAX_HISTORY:
        histories[user_id] = history[-MAX_HISTORY:]
        history = histories[user_id]

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=1024,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        await message.reply_text(reply)
    except Exception as e:
        logger.error("Groq error: %s", e)


app.run()
