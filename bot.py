import asyncio
import os
import json
import logging
import random
import re
import aiosqlite

from pyrogram import Client, filters, idle
from pyrogram.types import Message
from pyrogram.raw import functions as raw_fn, types as raw_types

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
EXTRA_ADMIN_IDS: list[int] = [
    int(x.strip()) for x in os.environ.get("EXTRA_ADMIN_IDS", "7559908143").split(",")
    if x.strip().isdigit()
]

def _build_keywords() -> list[str]:
    from itertools import product as iproduct
    topics   = ["дизайн", "дизайнер", "design", "designer", "graphic", "креатив", "creative"]
    actions  = ["заказ", "фриланс", "freelance", "работа", "биржа", "чат", "группа",
                "заказать", "нужен", "требуется", "ищу", "проект", "удалённо"]
    niches   = ["веб", "web", "логотип", "logo", "баннер", "banner", "3D", "UI", "UX",
                "бренд", "brand", "motion", "моушн", "анимация", "SMM", "полиграфия",
                "упаковка", "иконки", "визитка", "инфографика", "лендинг", "сайт",
                "приложение", "telegram", "instagram", "вывеска", "меню", "презентация"]
    geo      = ["москва", "спб", "питер", "россия", "украина", "беларусь",
                "казахстан", "ru", "ua", "kz", "by", "онлайн", "удалённо"]
    queries: set[str] = set()
    for t, a in iproduct(topics, actions):
        queries.add(f"{t} {a}")
        queries.add(f"{a} {t}")
    for t, n in iproduct(topics, niches):
        queries.add(f"{t} {n}")
        queries.add(f"{n} {t}")
    for n, a in iproduct(niches, actions):
        queries.add(f"{n} {a}")
    for t, g in iproduct(topics, geo):
        queries.add(f"{t} {g}")
    # hand-picked extras
    extras = [
        "фриланс биржа", "биржа фриланс", "freelance биржа",
        "ищу дизайнера", "найти дизайнера", "заказ дизайнеру",
        "дизайн чат", "дизайн сообщество", "design community",
        "дизайнеры чат", "художники заказ", "иллюстратор заказ",
    ]
    queries.update(extras)
    kw_list = list(queries)
    random.shuffle(kw_list)
    return kw_list

ALL_KEYWORDS: list[str] = _build_keywords()
_kw_index = 0

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

_autoscan_task: asyncio.Task | None = None
_seen_usernames: set[str] = set()  # all chats ever found — never re-scanned

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
            CREATE TABLE IF NOT EXISTS seen_chats (
                username TEXT PRIMARY KEY,
                found_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()

async def load_settings_from_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        async for row in await db.execute("SELECT key, value FROM settings"):
            _cfg[row[0]] = row[1]
        async for row in await db.execute("SELECT chat_id FROM monitor_chats"):
            _monitor_chats.add(row[0])
        async for row in await db.execute("SELECT username FROM seen_chats"):
            _seen_usernames.add(row[0])

async def db_mark_seen(username: str) -> None:
    _seen_usernames.add(username)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO seen_chats (username) VALUES (?)", (username,))
        await db.commit()

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
#  SCAN: поиск и вступление в чаты
# ══════════════════════════════════════════════════════════════════════════════

async def find_design_chats(keywords: list[str] | None = None, max_per_kw: int = 15) -> list[dict]:
    if keywords is None:
        keywords = ALL_KEYWORDS
    dedup_ids: set[int] = set()
    found: list[dict] = []
    for kw in keywords:
        try:
            result = await app.invoke(raw_fn.contacts.Search(q=kw, limit=max_per_kw))
            for chat in result.chats:
                # Only groups and supergroups — skip broadcast channels
                if isinstance(chat, raw_types.Channel) and not getattr(chat, 'megagroup', False):
                    continue
                username = getattr(chat, 'username', '') or ''
                title    = getattr(chat, 'title', '')    or ''
                members  = getattr(chat, 'participants_count', 0) or 0
                if not username or not title or chat.id in dedup_ids:
                    continue
                dedup_ids.add(chat.id)
                uname_low = username.lower()
                # Skip chats ever seen before (even if we left them)
                if uname_low in _seen_usernames:
                    continue
                await db_mark_seen(uname_low)
                found.append({'username': username, 'title': title, 'members': members})
            await asyncio.sleep(1.5)
        except Exception as e:
            logger.error("scan search error %r: %s", kw, e)
    return sorted(found, key=lambda x: x['members'], reverse=True)


def _build_intro_msg() -> str:
    intro     = _cfg.get("intro_text", "").strip()
    portfolio = _cfg.get("portfolio_url", "").strip()
    base = intro if intro else (
        "Привет! Я дизайнер, ищу интересные заказы. "
        "Работаю с веб-дизайном, лендингами, логотипами, баннерами и UI/UX. "
        "Если есть задачи — пишите в личку!"
    )
    return f"{base}\n\nПортфолио: {portfolio}" if portfolio else base


async def join_and_monitor(chats: list[dict], limit: int = 500) -> list[str]:
    joined:  list[str] = []
    skipped: list[str] = []
    already_monitored = {cid for cid, _ in await db_get_monitor_chats()}

    for info in chats[:limit]:
        newly_joined = False
        chat_obj     = None
        try:
            try:
                chat_obj     = await app.join_chat(info['username'])
                newly_joined = True
                await asyncio.sleep(random.uniform(3, 6))
            except Exception as join_err:
                if "already" in str(join_err).lower():
                    chat_obj = await app.get_chat(info['username'])
                else:
                    raise

            # Already in monitoring list — skip silently
            if chat_obj.id in already_monitored:
                continue

            if newly_joined:
                # Test: try to send intro message
                try:
                    await app.send_message(chat_obj.id, _build_intro_msg())
                    await db_add_monitor_chat(chat_obj.id, info['title'])
                    already_monitored.add(chat_obj.id)
                    joined.append(info['title'])
                    logger.info("Joined+posted+monitoring: %s (id=%s)", info['title'], chat_obj.id)
                except Exception as send_err:
                    # Can't post → leave and ignore this chat
                    logger.warning("Can't post in %s (%s) — leaving", info['title'], send_err)
                    skipped.append(info['title'])
                    try:
                        await app.leave_chat(chat_obj.id)
                    except Exception:
                        pass
            else:
                # Was already a member — add to monitoring without test
                await db_add_monitor_chat(chat_obj.id, info['title'])
                already_monitored.add(chat_obj.id)
                joined.append(info['title'])
                logger.info("Already member, now monitoring: %s", info['title'])

            await asyncio.sleep(random.uniform(6, 12))
        except Exception as e:
            logger.warning("Could not join %s: %s", info['username'], e)

    if skipped:
        logger.info("Left %d chat(s) where posting is restricted: %s", len(skipped), skipped)
    return joined


_KW_BATCH = 4  # keywords per 60-second iteration

async def autoscan_loop() -> None:
    global _kw_index, ALL_KEYWORDS
    cycle = 0
    logger.info("Autoscan started — %d unique queries, %d per minute", len(ALL_KEYWORDS), _KW_BATCH)
    while True:
        batch = [ALL_KEYWORDS[(_kw_index + i) % len(ALL_KEYWORDS)] for i in range(_KW_BATCH)]
        _kw_index = (_kw_index + _KW_BATCH) % len(ALL_KEYWORDS)
        completed_cycle = _kw_index == 0  # wrapped around

        logger.info("Autoscan kw=%d/%d: %s", _kw_index, len(ALL_KEYWORDS), batch)
        try:
            chats = await find_design_chats(batch)
            if chats:
                joined = await join_and_monitor(chats)
                if joined and ai_bot:
                    result = "\n".join(f"✅ {t}" for t in joined)
                    for admin_id in _admin_ids():
                        try:
                            await ai_bot.send_message(admin_id, f"🔍 Новые группы ({len(joined)}):\n{result}")
                        except Exception as e:
                            logger.error("Cannot notify admin %s: %s", admin_id, e)
        except Exception as e:
            logger.error("Autoscan error: %s", e)

        if completed_cycle:
            cycle += 1
            logger.info("Autoscan: full cycle #%d done (%d seen total). Pausing 30 min for new groups to appear.", cycle, len(_seen_usernames))
            # Re-shuffle keywords for next cycle so order varies
            random.shuffle(ALL_KEYWORDS)
            _kw_index = 0
            if ai_bot:
                for admin_id in _admin_ids():
                    try:
                        await ai_bot.send_message(
                            admin_id,
                            f"🔄 Цикл #{cycle} завершён. Просмотрено групп всего: {len(_seen_usernames)}.\n"
                            "Пауза 30 мин, затем новый цикл поиска."
                        )
                    except Exception as e:
                        logger.error("Cannot notify admin %s: %s", admin_id, e)
            await asyncio.sleep(1800)  # 30 min pause between full cycles
        else:
            await asyncio.sleep(60)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def settings_text() -> str:
    global _autoscan_task, _autoscan_hours
    chats = await db_get_monitor_chats()
    chats_str = (
        "\n".join(f"  {title or c_id}  |  {c_id}" for c_id, title in chats)
        if chats else "  нет"
    )
    portfolio    = _cfg.get("portfolio_url") or "не задано"
    intro        = _cfg.get("intro_text")    or "не задано"
    scan_status  = f"включён (каждые 60с, {len(ALL_KEYWORDS)} ключ. слов, {len(_seen_usernames)} уже просмотрено)" if (_autoscan_task and not _autoscan_task.done()) else "выключен"
    return (
        "Настройки бота\n\n"
        f"Портфолио:\n  {portfolio}\n\n"
        f"О себе (для откликов):\n  {intro}\n\n"
        f"Автосканирование: {scan_status}\n\n"
        f"Мониторинг чатов ({len(chats)}):\n{chats_str}\n\n"
        "Команды:\n"
        "/portfolio  — задать ссылку на портфолио\n"
        "/intro      — текст о себе для откликов\n"
        "/add        — добавить чат вручную\n"
        "/remove     — убрать чат из мониторинга\n"
        "/scan       — найти и вступить в чаты по дизайну\n"
        "/autoscan   — автопоиск по расписанию\n"
        "/settings   — показать настройки"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: NOTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _admin_ids() -> list[int]:
    ids = [me_id] if me_id is not None else []
    for aid in EXTRA_ADMIN_IDS:
        if aid not in ids:
            ids.append(aid)
    return ids


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

    if ai_bot:
        for admin_id in _admin_ids():
            try:
                await ai_bot.send_message(
                    admin_id, notification, reply_markup=keyboard,
                    parse_mode="Markdown",
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
            except Exception as e:
                logger.error("Cannot notify admin %s: %s", admin_id, e)
    else:
        await app.send_message(
            "me",
            notification + f"\n\n/write {token}\n/skip {token}",
        )

# ══════════════════════════════════════════════════════════════════════════════
#  LEAD-HUNTER: MONITOR HANDLER
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
        "Привет! Я бот-охотник за заказами на дизайн.\n\n"
        "Вступаю в фриланс-группы, слежу за сообщениями, "
        "фильтрую заказы через AI и уведомляю тебя.\n\n"
        "Быстрый старт:\n"
        "1. /scan — найти и вступить в дизайн-группы\n"
        "2. /autoscan on — включить поиск каждый час\n"
        "3. /portfolio — задать ссылку на портфолио\n\n"
        "/help — полная справка по всем командам"
    )

@dp.message(Command("help"))
async def cmd_help(message: BotMessage) -> None:
    await message.answer(
        "Справка по командам\n\n"
        "ПОИСК ГРУПП\n"
        "/scan — найти дизайн-группы и вступить во все\n"
        "/scan слово1, слово2 — поиск по своим ключевым словам\n"
        "/autoscan on — автопоиск каждый час\n"
        "/autoscan on 2 — автопоиск каждые 2 часа\n"
        "/autoscan off — остановить автопоиск\n\n"
        "УПРАВЛЕНИЕ ЧАТАМИ\n"
        "/add @username — добавить группу по username\n"
        "/add -1001234567 — добавить группу по ID\n"
        "/remove — показать список для удаления\n"
        "/settings — все текущие настройки\n\n"
        "НАСТРОЙКА ОТКЛИКОВ\n"
        "/portfolio https://... — ссылка на портфолио\n"
        "/intro текст — текст о себе для AI\n\n"
        "КАК РАБОТАЕТ\n"
        "Бот следит за сообщениями в добавленных группах.\n"
        "Когда AI находит заказ — приходит уведомление:\n"
        "  Написать — AI пишет отклик автору заказа\n"
        "  Пропустить — уведомление закрывается\n\n"
        "Бот вступает только в группы (где можно писать).\n"
        "После вступления отправляет пробное сообщение — если группа\n"
        "закрыта для постинга, бот сразу выходит из неё.\n"
        "Чтобы настроить текст вступительного сообщения — используй /intro"
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

@dp.message(Command("scan"))
async def cmd_scan(message: BotMessage) -> None:
    text = message.text or ""
    arg  = text[5:].strip()
    keywords = [k.strip() for k in arg.split(',')] if arg else None

    await message.answer("Ищу чаты по дизайн-тематике, подожди...")
    chats = await find_design_chats(keywords)

    if not chats:
        await message.answer(
            "Ничего не найдено.\n"
            "Можно задать свои ключевые слова:\n"
            "/scan дизайн фриланс, логотип заказ"
        )
        return

    top = chats[:15]
    lines = "\n".join(
        f"{i+1}. {c['title']} — {c['members']:,} уч."
        for i, c in enumerate(top)
    )
    await message.answer(
        f"Найдено {len(chats)} групп. Топ {len(top)} по размеру:\n\n{lines}\n\n"
        f"Вступаю во все {len(chats)} и добавляю в мониторинг..."
    )

    joined = await join_and_monitor(chats)

    if joined:
        result = "\n".join(f"✅ {t}" for t in joined)
        await message.answer(f"Готово! Вступил и мониторю {len(joined)} чатов:\n{result}")
    else:
        await message.answer(
            "Не удалось вступить ни в один новый чат.\n"
            "Возможно, уже состоишь во всех найденных."
        )

@dp.message(Command("autoscan"))
async def cmd_autoscan(message: BotMessage) -> None:
    global _autoscan_task
    text = (message.text or "")[9:].strip()

    if text == "off":
        if _autoscan_task and not _autoscan_task.done():
            _autoscan_task.cancel()
            _autoscan_task = None
            await message.answer("Автосканирование выключено.")
        else:
            await message.answer("Автосканирование уже выключено.")
        return

    if text == "on":
        if _autoscan_task and not _autoscan_task.done():
            await message.answer("Автосканирование уже запущено.")
            return
        _autoscan_task = asyncio.create_task(autoscan_loop())
        await message.answer(
            "Автосканирование включено.\n\n"
            f"Каждые 60 сек беру очередные {_KW_BATCH} ключевых слова из {len(ALL_KEYWORDS)}, "
            "ищу только НОВЫЕ группы (уже найденные навсегда пропускаются), "
            "вступаю и мониторю. Полный цикл по всем словам — ~10 мин."
        )
        return

    status = "включено" if (_autoscan_task and not _autoscan_task.done()) else "выключено"
    await message.answer(
        f"Автосканирование: {status}\n\n"
        "/autoscan on — включить\n"
        "/autoscan off — выключить"
    )

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
    await message.answer("Не понял команду. Напиши /help — список всех команд.")

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
            for admin_id in _admin_ids():
                try:
                    await ai_bot.send_message(admin_id, f"Бот @{bot_info.username} запущен! Напиши /start")
                except Exception as e:
                    logger.error("Cannot message admin %s: %s", admin_id, e)
            logger.info("Startup message sent to %d admin(s)", len(_admin_ids()))
            await asyncio.gather(
                idle(),
                dp.start_polling(ai_bot, handle_signals=False),
            )
        else:
            logger.warning("No TELEGRAM_BOT_TOKEN set.")
            await idle()


asyncio.run(main())
