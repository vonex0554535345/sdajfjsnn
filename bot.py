import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import os
import json
import logging
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.enums import ChatType
from groq import Groq

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION_STRING = os.environ["TELEGRAM_SESSION"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "Ты helpful ассистент. Отвечай коротко на русском.")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "20"))

groq_client = Groq(api_key=GROQ_API_KEY)
histories: dict[int, list[dict]] = {}
is_active = True
me_id: int = None
dialogs_cache: list[dict] = []


async def load_dialogs(client: Client) -> list[dict]:
    global dialogs_cache
    dialogs_cache = []
    async for dialog in client.get_dialogs():
        chat = dialog.chat
        name = (
            chat.title
            or f"{chat.first_name or ''} {chat.last_name or ''}".strip()
            or chat.username
            or str(chat.id)
        )
        dialogs_cache.append({"id": chat.id, "name": name, "username": chat.username or ""})
    return dialogs_cache


def find_chat(name: str, dialogs: list[dict]) -> dict | None:
    name_lower = name.lower()
    for d in dialogs:
        if name_lower == d["name"].lower() or name_lower == d["username"].lower():
            return d
    for d in dialogs:
        if name_lower in d["name"].lower() or name_lower in d["username"].lower():
            return d
    return None


async def execute_task(client: Client, task: str) -> str:
    dialogs = await load_dialogs(client)
    dialog_list = [f"{d['name']} (@{d['username']})" if d["username"] else d["name"] for d in dialogs]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Отправить сообщение контакту или в чат по имени",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chat_name": {"type": "string", "description": "Имя контакта или чата"},
                        "text": {"type": "string", "description": "Текст сообщения"}
                    },
                    "required": ["chat_name", "text"]
                }
            }
        }
    ]

    system = f"""Ты агент-помощник в Telegram. Выполняй задачи пользователя: пиши сообщения нужным людям, задавай вопросы в чатах.
Доступные контакты и чаты:
{chr(10).join(dialog_list[:100])}

Используй инструмент send_message для отправки. Составляй сообщения естественно от лица пользователя."""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task}
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=1024,
    )

    msg = response.choices[0].message
    results = []

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "send_message":
                args = json.loads(tool_call.function.arguments)
                chat = find_chat(args["chat_name"], dialogs)
                if chat:
                    await client.send_message(chat["id"], args["text"])
                    results.append(f"✅ Написал {chat['name']}: «{args['text']}»")
                else:
                    results.append(f"❌ Не нашёл контакт: {args['chat_name']}")

    if results:
        return "\n".join(results)
    return msg.content or "Задача не распознана"


app = Client("userbot", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)


@app.on_message(filters.me & filters.private)
async def handle_saved_message(client: Client, message: Message) -> None:
    global me_id, is_active
    if me_id is None:
        me_id = (await client.get_me()).id
    if message.chat.id != me_id or not message.text:
        return

    text = message.text.strip()

    if text == "/on":
        is_active = True
        await client.send_message("me", "Автоответ включён ✅")
        return
    if text == "/off":
        is_active = False
        await client.send_message("me", "Автоответ выключен ❌")
        return
    if text.startswith("/"):
        return

    await client.send_message("me", "⏳ Выполняю задачу...")
    try:
        result = await execute_task(client, text)
        await client.send_message("me", result)
    except Exception as e:
        logger.error("Task error: %s", e)
        await client.send_message("me", f"❌ Ошибка: {e}")


@app.on_message(filters.private & filters.incoming & ~filters.me)
async def handle_incoming(client: Client, message: Message) -> None:
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
