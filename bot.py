import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import os
import json
import logging
import random
import re
import aiosqlite

from pyrogram import Client, filters, idle
from pyrogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from groq import Groq

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
API_ID         = int(os.environ["TELEGRAM_API_ID"])
API_HASH       = os.environ["TELEGRAM_API_HASH"]
SESSION_STRING = os.environ["TELEGRAM_SESSION"]
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
SYSTEM_PROMPT  = os.environ.get("SYSTEM_PROMPT", "Ты helpful ассистент. Отвечай коротко на русском.")
MODEL          = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY    = int(os.environ.get("MAX_HISTORY", "20"))

# Lead-hunter config
BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")   # from @BotFather
PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "")         # ссылка на портфолио
DB_PATH       = os.environ.get("DB_PATH", "leads.db")

# Список чатов для мониторинга: "id1,id2,@username1,..."
_raw_chats = os.environ.get("MONITOR_CHATS", "")
MONITOR_CHATS: list = []
for _c in _raw_chats.split(","):
    _c = _c.strip()
    if _c:
        try:
            MONITOR_CHATS.append(int(_c))
        except ValueError:
            MONITOR_CHATS.append(_c)

# ── Clients ────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)
app = Client("userbot", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
# Бот нужен для инлайн-кнопок — userbot не может обрабатывать callback_query
notify_bot = Client("notifybot", bot_token=BOT_TOKEN) if BOT_TOKEN else None

# ── State ──────────────────────────────────────────────────────────────────────
histories: dict[int, list[dict]] = {}
is_active = True
blocked_users: set[str] = set()
me_id: int | None = None
dialogs_cache: list[dict] = []
# Хранит данные ожидающих заказов до нажатия кнопки
pending_orders: dict[str, dict] = {}

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_posts (
                chat_id    INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                created_at TEXT    DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, message_id)
            )
        """)
        await db.commit()

async def is_processed(chat_id: int, message_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT 1 FROM processed_posts WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        )
        return await cur.fetchone() is not None

async def mark_processed(chat_id: int, message_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO processed_posts (chat_id, message_id) VALUES (?, ?)",
            (chat_id, message_id),
        )
        await db.commit()

# ══════════════════════════════════════════════════════════════════════════════
#  AI: ORDER FILTER
# ══════════════════════════════════════════════════════════════════════════════

_FILTER_SYSTEM = """\
Ты — строгий фильтр заказов для фриланс-дизайнера. \
Определи: это реальный заказ на дизайн или нет?

ЦЕЛЕВЫЕ ЗАКАЗЫ (is_order: true):
• Веб-дизайн, UI/UX, лендинги, сайты, интерфейсы
• Баннеры, рекламные материалы, посты для соцсетей
• Логотипы, брендинг, фирменный стиль, айдентика
• 3D-графика, 3D-визуализация, рендеры
• Иллюстрации, иконки, персонажи
• Оформление Telegram-каналов/ботов/чатов
• Дизайн презентаций, питч-деков
• Полиграфия: визитки, листовки, упаковка

НЕ ЦЕЛЕВЫЕ (is_order: false):
• Флуд, шутки, оффтоп, спам, реклама
• Резюме/портфолио других дизайнеров ("ищу работу", "мои работы")
• Вакансия в штат или в офис (трудоустройство)
• Вопросы, советы, обсуждения без конкретного ТЗ
• Продажа курсов, обучение
• Любое нетематическое содержимое

Отвечай СТРОГО JSON без лишнего текста:
{"is_order": true/false, "reason": "одно предложение"}"""

async def classify_order(text: str) -> tuple[bool, str]:
    try:
        resp = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user",   "content": text[:1500]},
            ],
            max_tokens=120,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return bool(data.get("is_order")), data.get("reason", "")
    except Exception as e:
        logger.error("classify_order error: %s", e)
    return False, ""

# ══════════════════════════════════════════════════════════════════════════════
#  AI: COVER LETTER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

async def generate_letter(order_text: str) -> str:
    portfolio_line = f"\n\nПортфолио: {PORTFOLIO_URL}" if PORTFOLIO_URL else ""
    prompt = f"""\
Напиши короткий отклик на фриланс-заказ от лица дизайнера. Требования:
— 3–5 предложений, конкретно и без воды
— Вежливый тон, покажи понимание задачи
— Предложи обсудить детали и сроки
— НЕ начинай с "Здравствуйте, меня зовут..."
— Язык: русский{portfolio_line}

Заказ клиента:
{order_text[:1000]}"""
    resp = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.75,
    )
    return resp.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: NOTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

async def send_order_notification(message: Message) -> None:
    sender    = message.from_user
    s_name    = (
        f"{sender.first_name or ''} {sender.last_name or ''}".strip()
        if sender else "Unknown"
    )
    s_mention = f"[{s_name}](tg://user?id={sender.id})" if sender else s_name
    chat_name = getattr(message.chat, "title", None) or str(message.chat.id)
    token     = f"{message.chat.id}_{message.id}"

    pending_orders[token] = {
        "text":      message.text or message.caption or "",
        "sender_id": sender.id if sender else None,
    }

    notification = (
        f"🎯 **Новый заказ на дизайн!**\n\n"
        f"📍 Чат: {chat_name}\n"
        f"👤 Автор: {s_mention}\n\n"
        f"📝 **Текст заказа:**\n{message.text or message.caption or ''}\n\n"
        f"━━━━━━━━━━━━━"
    )
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Написать",   callback_data=f"write_{token}"),
        InlineKeyboardButton("❌ Пропустить", callback_data=f"skip_{token}"),
    ]])

    if notify_bot:
        await notify_bot.send_message(
            me_id,
            notification,
            reply_markup=markup,
            disable_web_page_preview=True,
        )
    else:
        # Режим без бота: шлём в «Избранное» текстом (без инлайн-кнопок)
        await app.send_message(
            "me",
            notification + (
                f"\n\n💡 Нет `TELEGRAM_BOT_TOKEN` — кнопки недоступны.\n"
                f"Чтобы ответить, отправь: `/write {token}`\n"
                f"Чтобы пропустить: `/skip {token}`"
            ),
        )

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: MONITOR HANDLER (userbot слушает чаты)
# ══════════════════════════════════════════════════════════════════════════════

_monitor_filter = (
    filters.chat(MONITOR_CHATS) & filters.incoming & ~filters.me
    if MONITOR_CHATS
    else filters.create(lambda _, __, ___: False)
)

@app.on_message(_monitor_filter)
async def handle_monitored(_: Client, message: Message) -> None:
    text = message.text or message.caption
    if not text or len(text) < 20:
        return

    if await is_processed(message.chat.id, message.id):
        return
    await mark_processed(message.chat.id, message.id)

    is_order, reason = await classify_order(text)
    logger.info(
        "[monitor] chat=%s msg=%s → is_order=%s | %s",
        message.chat.id, message.id, is_order, reason,
    )

    if is_order:
        await send_order_notification(message)

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: CALLBACK HANDLER (notify_bot обрабатывает нажатия кнопок)
# ══════════════════════════════════════════════════════════════════════════════

if notify_bot:
    @notify_bot.on_callback_query()
    async def handle_callback(_: Client, cb: CallbackQuery) -> None:
        data = cb.data or ""

        if data.startswith("skip_"):
            token = data[5:]
            pending_orders.pop(token, None)
            await cb.message.edit_text("❌ Пропущено.", reply_markup=None)
            await cb.answer()

        elif data.startswith("write_"):
            token = data[6:]
            order = pending_orders.get(token)

            if not order:
                await cb.answer("Заказ уже обработан или устарел.", show_alert=True)
                return

            await cb.message.edit_text("⏳ Генерирую отклик...", reply_markup=None)
            await cb.answer()

            try:
                letter = await generate_letter(order["text"])
                # Антифлуд-задержка
                await asyncio.sleep(random.uniform(3, 7))

                if order["sender_id"]:
                    await app.send_message(order["sender_id"], letter)
                    pending_orders.pop(token, None)
                    await cb.message.edit_text(
                        f"✅ Отклик отправлен!\n\n**Текст:**\n{letter}",
                        reply_markup=None,
                    )
                else:
                    await cb.message.edit_text(
                        "❌ Не удалось определить автора заказа.",
                        reply_markup=None,
                    )
            except Exception as e:
                logger.error("Letter send error: %s", e)
                await cb.message.edit_text(f"❌ Ошибка отправки: {e}", reply_markup=None)

# ══════════════════════════════════════════════════════════════════════════════
#  EXISTING: диалоги и execute_task
# ══════════════════════════════════════════════════════════════════════════════

async def load_dialogs(client: Client) -> list[dict]:
    global dialogs_cache
    seen = set()
    dialogs_cache = []

    def add(chat_id, name, username=""):
        if chat_id not in seen:
            seen.add(chat_id)
            dialogs_cache.append({"id": chat_id, "name": name, "username": username or ""})

    contacts = await client.get_contacts()
    for u in contacts:
        name = f"{u.first_name or ''} {u.last_name or ''}".strip() or u.username or str(u.id)
        add(u.id, name, u.username)

    async for dialog in client.get_dialogs():
        chat = dialog.chat
        name = (
            chat.title
            or f"{chat.first_name or ''} {chat.last_name or ''}".strip()
            or chat.username
            or str(chat.id)
        )
        add(chat.id, name, chat.username)

    return dialogs_cache


def find_chat(name: str, dialogs: list[dict]) -> dict | None:
    clean = re.sub(r"\s*\(@[^)]*\)", "", name).strip().lower()
    username_match = re.search(r"@([\w]+)", name)
    extracted_username = username_match.group(1).lower() if username_match else ""

    for d in dialogs:
        d_name = d["name"].lower()
        d_user = d["username"].lower()
        if clean == d_name or (extracted_username and extracted_username == d_user):
            return d
    for d in dialogs:
        d_name = d["name"].lower()
        d_user = d["username"].lower()
        if clean in d_name or d_name in clean or (extracted_username and extracted_username in d_user):
            return d
    return None


async def execute_task(client: Client, task: str) -> str:
    dialogs = await load_dialogs(client)
    dialog_list = [
        f"{d['name']} (@{d['username']})" if d["username"] else d["name"]
        for d in dialogs
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Отправить сообщение контакту или в чат по имени",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_name": {"type": "string", "description": "Имя контакта или чата"},
                    "text":      {"type": "string", "description": "Текст сообщения"},
                },
                "required": ["chat_name", "text"],
            },
        },
    }]

    system = (
        f"Ты агент-помощник в Telegram. Выполняй задачи пользователя: "
        f"пиши сообщения нужным людям, задавай вопросы в чатах.\n"
        f"Доступные контакты и чаты:\n"
        f"{chr(10).join(dialog_list[:100])}\n\n"
        f"Используй инструмент send_message для отправки. "
        f"Составляй сообщения естественно от лица пользователя."
    )

    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": task},
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
                chat_name    = args["chat_name"].strip()
                text_to_send = args["text"]
                logger.info("Tool call: send_message to '%s'", chat_name)
                try:
                    if "@" in chat_name:
                        target = chat_name.lstrip("@")
                    else:
                        found = find_chat(chat_name, dialogs)
                        if not found:
                            results.append(f"❌ Контакт не найден: {chat_name}")
                            continue
                        target = found["id"]
                    await client.send_message(target, text_to_send)
                    results.append(f"✅ Написал {chat_name}: «{text_to_send}»")
                except Exception as e:
                    logger.error("Send error: %s", e)
                    results.append(f"❌ Ошибка отправки {chat_name}: {e}")

    if results:
        return "\n".join(results)
    return msg.content or "Задача не распознана"

# ══════════════════════════════════════════════════════════════════════════════
#  EXISTING: «Избранное» — команды и задачи
# ══════════════════════════════════════════════════════════════════════════════

@app.on_message(filters.me & filters.private)
async def handle_saved_message(client: Client, message: Message) -> None:
    global me_id, is_active
    if me_id is None:
        me_id = (await client.get_me()).id
    if message.chat.id != me_id or not message.text:
        return

    text = message.text.strip()

    # ── /on и /off ──
    if text.startswith("/on") or text.startswith("/off"):
        parts = text.split()
        cmd      = parts[0]
        username = parts[1].lstrip("@").lower() if len(parts) > 1 else None
        if username:
            if cmd == "/on":
                blocked_users.discard(username)
                await client.send_message("me", f"Автоответ для @{username} включён ✅")
            else:
                blocked_users.add(username)
                await client.send_message("me", f"Автоответ для @{username} выключен ❌")
        else:
            if cmd == "/on":
                is_active = True
                blocked_users.clear()
                await client.send_message("me", "Автоответ для всех включён ✅")
            else:
                is_active = False
                await client.send_message("me", "Автоответ для всех выключен ❌")
        return

    # ── Fallback: /write и /skip для режима без BOT_TOKEN ──
    if text.startswith("/write "):
        token = text[7:].strip()
        order = pending_orders.get(token)
        if not order:
            await client.send_message("me", "❌ Заказ не найден или устарел.")
            return
        await client.send_message("me", "⏳ Генерирую отклик...")
        try:
            letter = await generate_letter(order["text"])
            await asyncio.sleep(random.uniform(3, 7))
            if order["sender_id"]:
                await client.send_message(order["sender_id"], letter)
                pending_orders.pop(token, None)
                await client.send_message("me", f"✅ Отклик отправлен!\n\n{letter}")
            else:
                await client.send_message("me", "❌ Автор заказа неизвестен.")
        except Exception as e:
            await client.send_message("me", f"❌ Ошибка: {e}")
        return

    if text.startswith("/skip "):
        token = text[6:].strip()
        pending_orders.pop(token, None)
        await client.send_message("me", "❌ Пропущено.")
        return

    if text.startswith("/"):
        return

    # ── Обычные задачи (AI-агент) ──
    await client.send_message("me", "⏳ Выполняю задачу...")
    try:
        result = await execute_task(client, text)
        await client.send_message("me", result)
    except Exception as e:
        logger.error("Task error: %s", e)
        await client.send_message("me", f"❌ Ошибка: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  EXISTING: входящие личные сообщения (автоответ)
# ══════════════════════════════════════════════════════════════════════════════

@app.on_message(filters.private & filters.incoming & ~filters.me)
async def handle_incoming(_: Client, message: Message) -> None:
    if not is_active or not message.text:
        return
    sender_username = (message.from_user.username or "").lower()
    if sender_username and sender_username in blocked_users:
        return

    user_id = message.from_user.id
    history = histories.setdefault(user_id, [])
    history.append({"role": "user", "content": message.text})
    if len(history) > MAX_HISTORY:
        histories[user_id] = history[-MAX_HISTORY:]
        history = histories[user_id]

    try:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=1024,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        await message.reply_text(reply)
    except Exception as e:
        logger.error("Groq error: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    global me_id
    await init_db()
    logger.info("DB initialised at %s", DB_PATH)

    await app.start()
    me_id = (await app.get_me()).id
    logger.info("Userbot started (me_id=%s). Monitoring %d chat(s).", me_id, len(MONITOR_CHATS))

    if notify_bot:
        await notify_bot.start()
        logger.info("Notify-bot started.")
    else:
        logger.warning(
            "TELEGRAM_BOT_TOKEN not set — inline buttons unavailable. "
            "Use /write <token> and /skip <token> commands in Saved Messages."
        )

    await idle()

    await app.stop()
    if notify_bot:
        await notify_bot.stop()


asyncio.run(main())
