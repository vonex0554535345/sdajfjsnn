import asyncio
import os
import json
import logging
import random
import re
import aiosqlite

from pyrogram import Client, filters, idle
from pyrogram.types import Message

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message as BotMessage,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    LinkPreviewOptions,
)
from aiogram.filters import Command

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
BOT_TOKEN      = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DB_PATH        = os.environ.get("DB_PATH", "leads.db")

# ── Clients ────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)
app = Client("userbot", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)

ai_bot: Bot | None = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()

# ── Runtime state ──────────────────────────────────────────────────────────────
histories:      dict[int, list[dict]] = {}
is_active       = True
blocked_users:  set[str]              = set()
me_id:          int | None            = None
dialogs_cache:  list[dict]            = []
pending_orders: dict[str, dict]       = {}

_cfg: dict[str, str] = {"portfolio_url": "", "intro_text": ""}
_monitor_chats: set  = set()

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS processed_posts (
                chat_id    INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS monitor_chats (
                chat_id INTEGER PRIMARY KEY,
                title   TEXT DEFAULT ''
            );
        """)
        await db.commit()

async def load_settings_from_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        async for row in await db.execute("SELECT key, value FROM settings"):
            _cfg[row[0]] = row[1]
        async for row in await db.execute("SELECT chat_id FROM monitor_chats"):
            _monitor_chats.add(row[0])

async def save_setting(key: str, value: str) -> None:
    _cfg[key] = value
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
        )
        await db.commit()

async def db_add_monitor_chat(chat_id: int, title: str = "") -> None:
    _monitor_chats.add(chat_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO monitor_chats (chat_id, title) VALUES (?, ?)",
            (chat_id, title),
        )
        await db.commit()

async def db_remove_monitor_chat(chat_id: int) -> None:
    _monitor_chats.discard(chat_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM monitor_chats WHERE chat_id=?", (chat_id,))
        await db.commit()

async def db_get_monitor_chats() -> list[tuple[int, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT chat_id, title FROM monitor_chats ORDER BY chat_id")
        return await cur.fetchall()

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
Ты — строгий фильтр заказов для фриланс-дизайнера.
Определи: это реальный заказ на дизайн или нет?

ЦЕЛЕВЫЕ (is_order: true):
Веб-дизайн, UI/UX, лендинги; баннеры, рекламные материалы;
логотипы, брендинг, айдентика; 3D-графика; иллюстрации, иконки;
оформление Telegram-каналов/ботов; презентации; полиграфия.

НЕ ЦЕЛЕВЫЕ (is_order: false):
Флуд, спам, реклама; резюме других дизайнеров; вакансии в штат;
вопросы без конкретного заказа; курсы; нетематический контент.

Ответ СТРОГО JSON: {"is_order": true/false, "reason": "одно предложение"}"""

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
    portfolio = _cfg.get("portfolio_url", "")
    intro     = _cfg.get("intro_text", "")
    intro_block     = f"\nО себе: {intro}" if intro else ""
    portfolio_block = f"\n\nПортфолио: {portfolio}" if portfolio else ""

    prompt = (
        "Напиши короткий отклик на фриланс-заказ от лица дизайнера.\n"
        "Требования: 3-5 предложений, конкретно и без воды; "
        "вежливый и уверенный тон; предложи обсудить детали и сроки; "
        f"НЕ начинай с 'Здравствуйте, меня зовут...';{intro_block} язык: русский{portfolio_block}\n\n"
        f"Заказ клиента:\n{order_text[:1000]}"
    )
    resp = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.75,
    )
    return resp.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def settings_text() -> str:
    chats = await db_get_monitor_chats()
    chats_str = (
        "\n".join(f"  {title or c_id}  |  {c_id}" for c_id, title in chats)
        if chats else "  нет"
    )
    portfolio = _cfg.get("portfolio_url") or "не задано"
    intro     = _cfg.get("intro_text")    or "не задано"
    return (
        "Настройки бота\n\n"
        f"Портфолио:\n  {portfolio}\n\n"
        f"О себе (для откликов):\n  {intro}\n\n"
        f"Мониторинг чатов ({len(chats)}):\n{chats_str}\n\n"
        "Команды:\n"
        "/portfolio  — задать ссылку на портфолио\n"
        "/intro      — текст о себе для откликов\n"
        "/add        — добавить чат в мониторинг\n"
        "/remove     — убрать чат из мониторинга\n"
        "/settings   — показать настройки"
    )

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
        f"🎯 Новый заказ на дизайн!\n\n"
        f"Чат: {chat_name}\n"
        f"Автор: {s_mention}\n\n"
        f"Сообщение:\n{message.text or message.caption or ''}"
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Написать",   callback_data=f"write_{token}"),
        InlineKeyboardButton(text="❌ Пропустить", callback_data=f"skip_{token}"),
    ]])

    if ai_bot and me_id:
        await ai_bot.send_message(
            me_id, notification, reply_markup=keyboard,
            parse_mode="Markdown",
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    else:
        await app.send_message(
            "me",
            notification + f"\n\n/write {token}\n/skip {token}",
        )

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: MONITOR HANDLER (userbot слушает чаты)
# ══════════════════════════════════════════════════════════════════════════════

def _is_monitored(_, __, message: Message) -> bool:
    return bool(_monitor_chats) and message.chat.id in _monitor_chats

@app.on_message(filters.create(_is_monitored) & filters.incoming & ~filters.me)
async def handle_monitored(_: Client, message: Message) -> None:
    text = message.text or message.caption
    if not text or len(text) < 20:
        return
    if await is_processed(message.chat.id, message.id):
        return
    await mark_processed(message.chat.id, message.id)
    is_order, reason = await classify_order(text)
    logger.info("[monitor] chat=%s msg=%s is_order=%s | %s",
                message.chat.id, message.id, is_order, reason)
    if is_order:
        await send_order_notification(message)

# ══════════════════════════════════════════════════════════════════════════════
#  AIOGRAM BOT — команды управления
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(Command("start"))
async def cmd_start(message: BotMessage) -> None:
    await message.answer(
        "Привет! Я бот-охотник за заказами.\n\n"
        "Мониторю фриланс-чаты, фильтрую заказы через AI "
        "и уведомляю тебя с кнопками Написать / Пропустить.\n\n"
        "Команды:\n"
        "/settings — текущие настройки\n"
        "/portfolio — задать портфолио\n"
        "/intro — текст о себе для откликов\n"
        "/add — добавить чат в мониторинг\n"
        "/remove — убрать чат из мониторинга"
    )

@dp.message(Command("settings"))
async def cmd_settings(message: BotMessage) -> None:
    await message.answer(await settings_text())

@dp.message(Command("portfolio"))
async def cmd_portfolio(message: BotMessage) -> None:
    text = message.text or ""
    url = text[10:].strip()
    if not url:
        cur = _cfg.get("portfolio_url") or "не задано"
        await message.answer(
            f"Текущее портфолио: {cur}\n\n"
            "Чтобы задать, напиши:\n"
            "/portfolio https://behance.net/yourname"
        )
        return
    await save_setting("portfolio_url", url)
    await message.answer(f"Портфолио сохранено:\n{url}")

@dp.message(Command("intro"))
async def cmd_intro(message: BotMessage) -> None:
    text = message.text or ""
    intro = text[6:].strip()
    if not intro:
        cur = _cfg.get("intro_text") or "не задано"
        await message.answer(
            f"Текущий текст о себе: {cur}\n\n"
            "AI вставляет его в каждый отклик.\n"
            "Чтобы задать, напиши:\n"
            "/intro Я дизайнер с 5 лет опытом, специализируюсь на веб-дизайне и брендинге"
        )
        return
    await save_setting("intro_text", intro)
    await message.answer(f"Текст о себе сохранён:\n{intro}")

@dp.message(Command("add"))
async def cmd_add(message: BotMessage) -> None:
    text = message.text or ""
    arg = text[4:].strip()
    if not arg:
        await message.answer(
            "Укажи ID или username чата:\n"
            "/add -1001234567890\n"
            "/add @freelance_ru"
        )
        return
    try:
        chat_id = int(arg)
        title = str(chat_id)
        try:
            chat_obj = await app.get_chat(chat_id)
            title = chat_obj.title or chat_obj.username or str(chat_id)
        except Exception:
            pass
        await db_add_monitor_chat(chat_id, title)
        await message.answer(f"Чат добавлен в мониторинг:\n{title}\nID: {chat_id}")
    except ValueError:
        username = arg.lstrip("@")
        try:
            chat_obj = await app.get_chat(username)
            chat_id  = chat_obj.id
            title    = chat_obj.title or chat_obj.username or username
            await db_add_monitor_chat(chat_id, title)
            await message.answer(f"Чат добавлен в мониторинг:\n{title}\nID: {chat_id}")
        except Exception as e:
            await message.answer(f"Чат не найден: @{username}\nОшибка: {e}")

@dp.message(Command("remove"))
async def cmd_remove(message: BotMessage) -> None:
    text = message.text or ""
    arg = text[7:].strip()
    if not arg:
        chats = await db_get_monitor_chats()
        if not chats:
            await message.answer("Список мониторинга пуст.")
            return
        chats_str = "\n".join(f"/remove {c_id}  ({title or c_id})" for c_id, title in chats)
        await message.answer(f"Какой чат убрать?\n\n{chats_str}")
        return
    try:
        chat_id = int(arg)
    except ValueError:
        username = arg.lstrip("@")
        try:
            chat_obj = await app.get_chat(username)
            chat_id  = chat_obj.id
        except Exception as e:
            await message.answer(f"Чат не найден: {e}")
            return
    await db_remove_monitor_chat(chat_id)
    await message.answer(f"Чат {chat_id} удалён из мониторинга.")

@dp.message()
async def cmd_unknown(message: BotMessage) -> None:
    await message.answer(
        "Не понял команду.\n\n"
        "/settings — настройки\n"
        "/add — добавить чат\n"
        "/remove — убрать чат\n"
        "/portfolio — портфолио\n"
        "/intro — текст о себе"
    )

@dp.callback_query(F.data.startswith("skip_"))
async def cb_skip(callback: CallbackQuery) -> None:
    token = (callback.data or "")[5:]
    pending_orders.pop(token, None)
    await callback.message.edit_text("Пропущено.")
    await callback.answer()

@dp.callback_query(F.data.startswith("write_"))
async def cb_write(callback: CallbackQuery) -> None:
    token = (callback.data or "")[6:]
    order = pending_orders.get(token)
    if not order:
        await callback.answer("Заказ уже обработан или устарел.", show_alert=True)
        return
    await callback.message.edit_text("Генерирую отклик...")
    await callback.answer()
    try:
        letter = await generate_letter(order["text"])
        await asyncio.sleep(random.uniform(3, 7))
        if order["sender_id"]:
            await app.send_message(order["sender_id"], letter)
            pending_orders.pop(token, None)
            await callback.message.edit_text(f"Отклик отправлен!\n\n{letter}")
        else:
            await callback.message.edit_text("Не удалось определить автора заказа.")
    except Exception as e:
        logger.error("Letter send error: %s", e)
        await callback.message.edit_text(f"Ошибка отправки: {e}")

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

    for u in await client.get_contacts():
        name = f"{u.first_name or ''} {u.last_name or ''}".strip() or u.username or str(u.id)
        add(u.id, name, u.username)

    async for dialog in client.get_dialogs():
        chat = dialog.chat
        name = (
            chat.title
            or f"{chat.first_name or ''} {chat.last_name or ''}".strip()
            or chat.username or str(chat.id)
        )
        add(chat.id, name, chat.username)

    return dialogs_cache


def find_chat(name: str, dialogs: list[dict]) -> dict | None:
    clean = re.sub(r"\s*\(@[^)]*\)", "", name).strip().lower()
    um = re.search(r"@([\w]+)", name)
    eu = um.group(1).lower() if um else ""
    for d in dialogs:
        dn, du = d["name"].lower(), d["username"].lower()
        if clean == dn or (eu and eu == du):
            return d
    for d in dialogs:
        dn, du = d["name"].lower(), d["username"].lower()
        if clean in dn or dn in clean or (eu and eu in du):
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
                    "chat_name": {"type": "string"},
                    "text":      {"type": "string"},
                },
                "required": ["chat_name", "text"],
            },
        },
    }]
    system = (
        "Ты агент-помощник в Telegram. Выполняй задачи пользователя.\n"
        f"Доступные контакты и чаты:\n{chr(10).join(dialog_list[:100])}\n\n"
        "Используй send_message. Пиши естественно от лица пользователя."
    )
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": task}],
        tools=tools, tool_choice="auto", max_tokens=1024,
    )
    msg = response.choices[0].message
    results = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "send_message":
                args = json.loads(tc.function.arguments)
                chat_name, text_to_send = args["chat_name"].strip(), args["text"]
                try:
                    if "@" in chat_name:
                        target = chat_name.lstrip("@")
                    else:
                        found = find_chat(chat_name, dialogs)
                        if not found:
                            results.append(f"Контакт не найден: {chat_name}")
                            continue
                        target = found["id"]
                    await client.send_message(target, text_to_send)
                    results.append(f"Написал {chat_name}")
                except Exception as e:
                    results.append(f"Ошибка отправки {chat_name}: {e}")
    return "\n".join(results) if results else (msg.content or "Задача не распознана")

# ══════════════════════════════════════════════════════════════════════════════
#  USERBOT: «Избранное» — задачи
# ══════════════════════════════════════════════════════════════════════════════

@app.on_message(filters.me & filters.private)
async def handle_saved_message(client: Client, message: Message) -> None:
    global me_id, is_active
    try:
        if me_id is None:
            me_id = (await client.get_me()).id
        if message.chat.id != me_id or not message.text:
            return

        text = message.text.strip()

        if text.startswith("/on") or text.startswith("/off"):
            parts    = text.split()
            cmd      = parts[0]
            username = parts[1].lstrip("@").lower() if len(parts) > 1 else None
            if username:
                if cmd == "/on":
                    blocked_users.discard(username)
                    await client.send_message("me", f"Автоответ для @{username} включён")
                else:
                    blocked_users.add(username)
                    await client.send_message("me", f"Автоответ для @{username} выключен")
            else:
                if cmd == "/on":
                    is_active = True
                    blocked_users.clear()
                    await client.send_message("me", "Автоответ включён")
                else:
                    is_active = False
                    await client.send_message("me", "Автоответ выключен")
            return

        if text.startswith("/"):
            return

        await client.send_message("me", "Выполняю задачу...")
        try:
            result = await execute_task(client, text)
            await client.send_message("me", result)
        except Exception as e:
            logger.error("Task error: %s", e)
            await client.send_message("me", f"Ошибка задачи: {e}")

    except Exception as e:
        logger.error("handle_saved_message crash: %s", e, exc_info=True)

# ══════════════════════════════════════════════════════════════════════════════
#  USERBOT: входящие личные сообщения (автоответ)
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
    await load_settings_from_db()
    logger.info("DB ready. Monitoring %d chat(s).", len(_monitor_chats))

    async with app:
        me_id = (await app.get_me()).id
        logger.info("Userbot started (me_id=%s)", me_id)

        if ai_bot:
            bot_info = await ai_bot.get_me()
            logger.info("Notify-bot started: @%s (id=%s)", bot_info.username, bot_info.id)
            try:
                await ai_bot.send_message(me_id, f"Бот @{bot_info.username} запущен! Напиши /start")
                logger.info("Startup message sent to owner")
            except Exception as e:
                logger.error("Cannot message owner: %s", e)
            await asyncio.gather(
                idle(),
                dp.start_polling(ai_bot, handle_signals=False),
            )
        else:
            logger.warning("No TELEGRAM_BOT_TOKEN set.")
            await idle()


asyncio.run(main())
