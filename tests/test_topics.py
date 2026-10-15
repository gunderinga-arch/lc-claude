"""Chats as Telegram topics: per-topic conversations, models, naming, landing from the main chat."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Chat, Message, Update
from aiogram.types import User as TgUser

from bot.domain import Mode, Role
from bot.services.chat import title_from
from bot.services.conversation import ConversationService
from bot.services.jobs import JobManager
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UserRepo
from bot.telegram import topics
from bot.telegram.middlewares.thread import ThreadMiddleware
from bot.telegram.routers import chat as chat_router


async def _env():
    db = Database(":memory:")
    await db.connect()
    users = UserRepo(db)
    user = await users.upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    repo = ConversationRepo(db)
    service = ConversationService(repo, system_prompt="sys", token_budget=10_000, idle_hours=1)
    return db, user, repo, service


async def test_every_topic_has_its_own_conversation_and_history() -> None:
    db, _, repo, service = await _env()
    main, a, b = await repo.active_id(1), await repo.active_id(1, 10), await repo.active_id(1, 11)
    assert len({main, a, b}) == 3 and await repo.active_id(1, 10) == a
    await service.save_exchange(a, "вопрос А", "ответ А")
    _, in_a = await service.build_context(1, "ещё", thread_id=10)
    _, in_b = await service.build_context(1, "ещё", thread_id=11)
    assert [m.content for m in in_a[1:-1]] == ["вопрос А", "ответ А"] and len(in_b) == 2
    assert sorted(await repo.active_threads(1)) == [10, 11]
    await service.reset(1, 10)  # ending one chat leaves the others alone
    assert await repo.chat(1, 10) is None and await repo.chat(1, 11) is not None
    await db.close()


async def test_topics_do_not_forget_after_silence_but_the_main_chat_does() -> None:
    db, _, repo, service = await _env()
    topic, main = await repo.active_id(1, 10), await repo.active_id(1)
    for cid in (topic, main):
        await db.conn.execute(
            "UPDATE conversations SET started_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(hours=5)).isoformat(), cid),
        )
    await db.conn.commit()
    assert (await service.build_context(1, "q", thread_id=10))[0] == topic
    assert (await service.build_context(1, "q"))[0] != main
    await db.close()


async def test_restart_keeps_the_title_and_the_model_of_the_chat() -> None:
    db, _, _, service = await _env()
    await service.open_chat(1, 10, model_override="claude-opus-4.8")
    await service.set_title(1, 10, "Отчёт")
    fresh = await service.restart(1, 10)
    info = await service.chat(1, 10)
    assert info is not None and info.id == fresh
    assert (info.title, info.model_override) == ("Отчёт", "claude-opus-4.8")
    await service.set_model(1, 10, None)
    assert (await service.chat(1, 10)).model_override is None
    await db.close()


def test_title_from_takes_a_short_first_line() -> None:
    assert title_from("«Привет».\nвторая строка") == "Привет"
    assert title_from("") == "Новый чат"
    long = title_from("слово " * 30)
    assert len(long) <= 41 and long.endswith("…")


def _update(thread: int | None, *, callback: bool = False) -> Update:
    chat, who = Chat(id=1, type="private"), TgUser(id=1, is_bot=False, first_name="u")
    extra = {"message_thread_id": thread, "is_topic_message": True} if thread else {}
    message = Message(message_id=5, date=datetime.now(UTC), chat=chat, from_user=who, text="hi", **extra)
    if callback:
        from aiogram.types import CallbackQuery

        return Update(
            update_id=1,
            callback_query=CallbackQuery(id="c", from_user=who, chat_instance="x", message=message),
        )
    return Update(update_id=1, message=message)


async def test_middleware_gives_the_topic_and_its_model_to_handlers() -> None:
    db, user, _, service = await _env()
    await service.open_chat(1, 10, model_override="claude-opus-4.8")
    middleware = ThreadMiddleware(service)
    seen: dict = {}

    async def handler(event, data):
        seen.update(data)

    await middleware(handler, _update(10, callback=True), {"user": user})
    assert seen["thread_id"] == 10 and seen["user"].model_override == "claude-opus-4.8"
    seen.clear()
    await middleware(handler, _update(None), {"user": user})
    assert (
        seen["thread_id"] == 0 and seen["user"].model_override is None
    )  # the main chat: the user's own pick
    seen.clear()
    await middleware(handler, _update(77), {"user": user})  # a topic the user made: registered, never renamed
    assert (await service.chat(1, 77)).title == ""
    await db.close()


async def test_moved_message_answers_into_the_topic() -> None:
    update = _update(None)
    message = update.message.as_(SimpleNamespace())  # any bot object
    copy = topics.moved(message, 42)
    assert copy.message_thread_id == 42 and copy.is_topic_message and message.message_thread_id is None
    assert copy.bot is message.bot


class _Bot:
    def __init__(self, topics_on: bool) -> None:
        self.topics_on, self.calls = topics_on, []

    async def get_me(self):
        return SimpleNamespace(has_topics_enabled=self.topics_on)

    async def create_forum_topic(self, chat_id, name):
        self.calls.append(("create", name))
        return SimpleNamespace(message_thread_id=500)

    async def send_message(self, chat_id, text, message_thread_id=None, reply_markup=None):
        if message_thread_id in getattr(self, "dead", ()):
            raise TelegramBadRequest(method=None, message="Bad Request: message thread not found")
        self.calls.append(("send", message_thread_id, text))

    async def copy_message(self, chat_id, from_chat, message_id, **kw):
        self.calls.append(("copy", message_id, kw["message_thread_id"]))

    async def delete_message(self, *a):
        pass


class _Msg:
    def __init__(self, bot) -> None:
        self.bot, self.chat, self.message_id, self.text, self.caption = (
            bot,
            SimpleNamespace(id=1),
            9,
            "привет",
            None,
        )
        self.deleted = False
        self.photo = self.document = self.voice = self.audio = self.video_note = None

    async def delete(self) -> None:
        self.deleted = True

    def model_copy(self, update):
        return SimpleNamespace(**{**vars(self), **update})

    async def answer(self, text, **kw):
        pass


async def test_message_in_the_main_chat_opens_a_chat_when_topics_are_on(monkeypatch) -> None:
    db, user, _, service = await _env()
    quota = SimpleNamespace(check=lambda u: _allowed())
    jobs, started = JobManager(), []

    async def job_body(target, thread):
        started.append(thread)

    def make_job(target, thread):
        return job_body(target, thread)

    for on, expected in ((True, 500), (False, 0)):
        topics._seen = (0.0, False)  # forget the cached mode
        bot = _Bot(on)
        started.clear()
        message = _Msg(bot)
        await chat_router._submit(
            message, user, quota, jobs, make_job, bot=bot, conversations=service, thread_id=0
        )
        await asyncio.sleep(0)  # let the job start
        await jobs.shutdown(1)
        jobs = JobManager()
        assert started == [expected]
        assert not message.deleted  # the user's message stays where they wrote it
    assert (await service.chat(1, 500)).title is None  # the bot names it after the first answer
    await db.close()


async def _allowed():
    return SimpleNamespace(allowed=True)


async def test_general_message_uses_the_empty_new_chat_instead_of_opening_another(monkeypatch) -> None:
    db, user, _, service = await _env()
    topics._seen = (0.0, False)
    bot = _Bot(True)
    started = []

    opened = await topics.open_chat(bot, service, user, 1)  # "new chat" pressed
    assert opened == 500 and await service.empty_thread(1) == 500
    quota, jobs = SimpleNamespace(check=lambda u: _allowed()), JobManager()

    async def body(target, thread):
        started.append(thread)

    await chat_router._submit(
        _Msg(bot), user, quota, jobs, lambda t, th: body(t, th), bot=bot, conversations=service, thread_id=0
    )
    await asyncio.sleep(0)
    await jobs.shutdown(1)
    assert started == [500] and [c for c in bot.calls if c[0] == "create"] == [
        ("create", "Новый чат")
    ]  # one topic
    assert ("send", 500, "<blockquote>привет</blockquote>") in bot.calls  # the request, quoted into the topic
    # once someone wrote in it, it is no longer "empty": the next General message opens a new one
    await service.save_exchange(await service.open_chat(1, 500), "q", "a")
    assert await service.empty_thread(1) is None
    await db.close()


async def test_service_message_of_a_new_topic_does_not_register_it_as_user_made() -> None:
    db, user, _, service = await _env()
    from aiogram.types import ForumTopicCreated

    update = _update(88)  # the "topic created" announcement
    update = update.model_copy(
        update={
            "message": update.message.model_copy(
                update={"forum_topic_created": ForumTopicCreated(name="x", icon_color=0)}
            )
        }
    )
    await ThreadMiddleware(service)(lambda e, d: _noop(), update, {"user": user})
    assert await service.chat(1, 88) is None
    await db.close()


async def _noop() -> None:
    return None


async def test_topic_is_named_by_the_fast_model_before_the_answer_appears() -> None:
    from bot.domain import RequestStatus
    from bot.services.chat import ChatReply

    db, user, _, service = await _env()
    await service.open_chat(1, 500)  # a topic the bot opened, nothing written yet
    events: list[str] = []

    class Bot:
        async def edit_forum_topic(self, chat_id, thread_id, name):
            events.append(f"rename:{name}")

        async def send_chat_action(self, *a, **kw):
            pass

    class Chat:
        def model_for(self, user):
            return SimpleNamespace(id="m", title="M", slow=False)

        def is_research(self, text):
            return False

        async def chat_title(self, text):
            await asyncio.sleep(0.05)  # the fast model is slower than a canned answer here
            return "Искусство"

        async def answer(self, user, text, images=(), thread_id=0):
            events.append("answered")
            return ChatReply(RequestStatus.OK, ["ответ"], "ответ", SimpleNamespace(id="m", title="M"), False)

    class Message:
        chat = SimpleNamespace(id=1)

        async def answer(self, text, **kw):
            events.append("shown")

    class Llm:
        def queue_depth(self):
            return 0

    await chat_router._generate(
        Bot(), Message(), user, Chat(), Llm(), SimpleNamespace(add=lambda *a: _noop()),
        text="давай про искусство", thread_id=500, conversations=service,
    )  # fmt: skip
    assert events == ["answered", "rename:Искусство", "shown"]
    assert (await service.chat(1, 500)).title == "Искусство"
    await db.close()


async def test_first_request_is_quoted_with_file_names() -> None:
    doc = SimpleNamespace(
        message_id=1, text=None, caption="разбери", photo=None, voice=None, audio=None, video_note=None,
        document=SimpleNamespace(file_name="отчёт <1>.pdf"),
    )  # fmt: skip
    pic = SimpleNamespace(
        message_id=2,
        text=None,
        caption=None,
        photo=[object()],
        document=None,
        voice=None,
        audio=None,
        video_note=None,
    )
    assert chat_router._quote_html([doc, pic]) == (
        "<blockquote>разбери</blockquote>" + chr(10) + "📎 отчёт &lt;1&gt;.pdf" + chr(10) + "🖼 фото"
    )
    long = SimpleNamespace(message_id=3, text="я" * 2000, caption=None, photo=None, document=None, voice=None,
                           audio=None, video_note=None)  # fmt: skip
    assert chat_router._quote_html([long]).endswith("…</blockquote>")


async def test_a_deleted_empty_topic_is_forgotten_and_a_fresh_one_used() -> None:
    db, user, _, service = await _env()
    bot = _Bot(True)
    await service.open_chat(1, 111)  # an empty chat whose topic the user deleted in Telegram
    bot.dead = {111}
    placed = await topics.place_in_chat(bot, service, user, 1, "hello")
    assert placed == (500, False)  # a new topic, not the dead one
    assert await service.chat(1, 111) is None and await service.chat(1, 500) is not None
    await db.close()
