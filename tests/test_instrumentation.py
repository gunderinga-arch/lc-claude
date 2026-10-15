"""WP-2 instrumentation: uploads log (both directions) and admin-bot access notices."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from bot.config import Settings
from bot.domain import Mode, RequestStatus, Role, UploadKind, UploadStat, User
from bot.files.blocks import OutFile
from bot.services.chat import ChatReply
from bot.storage.db import Database
from bot.storage.repositories import UploadRepo, UserRepo
from bot.telegram.middlewares.access import AccessMiddleware
from bot.telegram.routers import chat as chat_router


@pytest.fixture
async def db():
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _seed(db, *tg_ids: int) -> None:
    """The uploads log has a foreign key to users: the ids it writes about must exist."""
    users = UserRepo(db)
    for tg_id in tg_ids:
        await users.upsert_seen(tg_id, None, default_role=Role.USER, default_mode=Mode.NORMAL)


def _user(tg_id: int = 1) -> User:
    return User(tg_id, "u", Role.USER, Mode.NORMAL, None, datetime.now(UTC))


# ---- _collect: attachments from the user --------------------------------------------------------


class FakeDownloadBot:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    async def download(self, file, destination=None) -> io.BytesIO:
        destination = destination if destination is not None else io.BytesIO()
        destination.write(self.blobs[file.file_id])
        destination.seek(0)
        return destination


def _photo(file_id: str) -> SimpleNamespace:
    sizes = [SimpleNamespace(file_id="small", file_size=1), SimpleNamespace(file_id=file_id, file_size=5)]
    return SimpleNamespace(photo=sizes, document=None, caption=None)


def _doc(file_id: str, name: str, mime: str, size: int = 10) -> SimpleNamespace:
    document = SimpleNamespace(file_id=file_id, file_name=name, mime_type=mime, file_size=size)
    return SimpleNamespace(photo=None, document=document, caption=None)


async def test_collect_logs_photo_and_pdf_with_correct_sizes(db, monkeypatch: pytest.MonkeyPatch) -> None:
    # Extraction itself is covered by test_files_flow.py; here only the upload log matters.
    monkeypatch.setattr(
        chat_router, "extract_text", lambda name, data: SimpleNamespace(text="ok", truncated=False)
    )
    uploads = UploadRepo(db)
    photo_bytes, pdf_bytes = b"\xff\xd8jpeg-bytes", b"%PDF-fake-content"
    bot = FakeDownloadBot({"p1": photo_bytes, "d1": pdf_bytes})
    messages = [_photo("p1"), _doc("d1", "report.pdf", "application/pdf")]
    await _seed(db, 42)

    await chat_router._collect(bot, messages, uploads, user_id=42)

    stats = await uploads.stats(42, None)
    assert stats[UploadKind.PHOTO] == UploadStat(count=1, bytes=len(photo_bytes))
    assert stats[UploadKind.PDF] == UploadStat(count=1, bytes=len(pdf_bytes))


async def test_collect_does_not_log_files_refused_as_too_big(db) -> None:
    uploads = UploadRepo(db)
    bot = FakeDownloadBot({})
    big = _doc("big", "big.pdf", "application/pdf", size=chat_router.TELEGRAM_DOWNLOAD_LIMIT + 1)
    with pytest.raises(chat_router.UnsupportedFile):
        await chat_router._collect(bot, [big], uploads, user_id=42)
    assert await uploads.stats(42, None) == {}


async def test_collect_without_uploads_arg_still_works(db) -> None:
    """Backward compatible default: existing callers that don't pass uploads/user_id are unaffected."""
    bot = FakeDownloadBot({"t": b"hello"})
    text, _ = await chat_router._collect(bot, [_doc("t", "a.txt", "text/plain")])
    assert "hello" in text


# ---- _transcribe: voice/audio/video notes -------------------------------------------------------


class FakeVoiceBot:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def download(self, media, destination=None) -> io.BytesIO:
        destination = destination if destination is not None else io.BytesIO()
        destination.write(self.data)
        destination.seek(0)
        return destination


class FakeMessage:
    def __init__(self, voice=None, audio=None, video_note=None, caption: str | None = None) -> None:
        self.voice, self.audio, self.video_note, self.caption = voice, audio, video_note, caption

    async def reply(self, text: str, **kw) -> None:
        pass

    async def answer(self, text: str, **kw) -> None:
        pass


class FakeSpeech:
    async def transcribe(self, audio: bytes) -> str:
        return "привет"


async def test_transcribe_logs_voice(db) -> None:
    uploads = UploadRepo(db)
    voice = SimpleNamespace(file_id="v", duration=3, file_size=100)
    message = FakeMessage(voice=voice)
    await _seed(db, 7)

    await chat_router._transcribe(FakeVoiceBot(b"voice-bytes"), message, FakeSpeech(), uploads, user_id=7)

    stats = await uploads.stats(7, None)
    assert stats[UploadKind.VOICE] == UploadStat(count=1, bytes=len(b"voice-bytes"))


# ---- generated images/files, logged after chat.answer() -----------------------------------------


class FakeGenBot:
    async def send_chat_action(self, chat_id: int, action) -> None:
        pass


class FakeReplyMessage:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(id=1)
        self.sent: list[str] = []

    async def answer(self, text: str, **kw) -> None:
        self.sent.append("answer")

    async def answer_photo(self, *a, **kw) -> None:
        self.sent.append("photo")

    async def answer_media_group(self, *a, **kw) -> None:
        self.sent.append("media_group")

    async def answer_document(self, *a, **kw) -> None:
        self.sent.append("document")


class FakeLLMClient:
    def queue_depth(self) -> int:
        return 0


class FakeChatService:
    def __init__(self, reply: ChatReply) -> None:
        self._reply = reply

    def model_for(self, user: User):
        return SimpleNamespace(id="m", title="M", slow=False)

    def is_research(self, text: str) -> bool:
        return False

    async def answer(self, user: User, text: str, images=(), thread_id: int = 0) -> ChatReply:
        return self._reply


async def test_generate_logs_generated_images_and_files(db) -> None:
    uploads = UploadRepo(db)
    reply = ChatReply(
        status=RequestStatus.OK,
        parts=["hi"],
        text="hi",
        model=SimpleNamespace(id="m", title="M"),
        fallback_used=False,
        images=[OutFile("cat.jpg", b"img-bytes")],
        files=[OutFile("data.csv", b"csv-bytes")],
    )
    user = _user(99)
    await _seed(db, 99)

    await chat_router._generate(
        FakeGenBot(), FakeReplyMessage(), user, FakeChatService(reply), FakeLLMClient(), uploads, text="hi"
    )

    stats = await uploads.stats(99, None)
    assert stats[UploadKind.GEN_IMAGE] == UploadStat(count=1, bytes=len(b"img-bytes"))
    assert stats[UploadKind.GEN_FILE] == UploadStat(count=1, bytes=len(b"csv-bytes"))


# ---- access-request notices through the admin bot ------------------------------------------------


class FakeNotifyBot:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[int] = []

    async def send_message(self, chat_id: int, text: str, reply_markup=None) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append(chat_id)


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("BUYTOKENS_API_KEY", "k")
    monkeypatch.setenv("ADMIN_IDS", "10")
    return Settings(_env_file=None)


async def test_notice_uses_notifier_when_set(settings) -> None:
    notifier, main_bot = FakeNotifyBot(), FakeNotifyBot()
    mw = AccessMiddleware(users=None, settings=settings, notifier=notifier)  # type: ignore[arg-type]
    await mw._notify_admins(main_bot, 5, "u")
    assert notifier.sent == [10]
    assert main_bot.sent == []


async def test_notice_falls_back_to_main_bot_when_notifier_fails(settings) -> None:
    notifier, main_bot = FakeNotifyBot(fail=True), FakeNotifyBot()
    mw = AccessMiddleware(users=None, settings=settings, notifier=notifier)  # type: ignore[arg-type]
    await mw._notify_admins(main_bot, 5, "u")
    assert main_bot.sent == [10]


async def test_notice_uses_main_bot_when_no_notifier(settings) -> None:
    main_bot = FakeNotifyBot()
    mw = AccessMiddleware(users=None, settings=settings, notifier=None)  # type: ignore[arg-type]
    await mw._notify_admins(main_bot, 5, "u")
    assert main_bot.sent == [10]
