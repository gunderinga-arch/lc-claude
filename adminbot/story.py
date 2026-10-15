"""Story preview channel: the pending twinbot story shown to the owner before it is sent.

twinbot generates the story but does not send it; the pending text waits in twinbot's own data dir
as a preview file (``<request-id>\\n<story text>``). The admin bot's background loop (below, the
same pattern as AlertService / ModelProber) picks a fresh file up, messages every admin the story
with «✅ Отправить» / «✕ Отмена» buttons, and remembers the shown id so the same preview is never
sent twice.

Pure helpers first (no I/O), the loop last.
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import TwinStoryCancel, TwinStorySend
from bot.config import Settings

logger = logging.getLogger(__name__)

#: Name of the preview file in twinbot's data dir (next to twin.db). Duplicated in
#: twinbot/twinbot/story.py — the two packages are deployed separately (no runtime import
#: between them); tests/test_adminbot_twin.py cross-checks them.
STORY_PREVIEW_NAME = "twinbot_story.preview"

#: twinbot's data dir on this server: where the pending story preview waits. Hardcoded rather
#: than taken from .env — it is a deployment fact (the twinbot unit writes only there), and the
#: admin bot process runs on the same host. Not an env file path, no secret inside.
TWINBOT_DATA_DIR = Path("/opt/twinbot/data")

#: How often the loop looks for a fresh preview (seconds). A story is a button press,
#: not an alarm: a few seconds of latency cost nothing, a busy loop would.
STORY_PREVIEW_INTERVAL_S = 3.0

#: Name of the approval file the verdict handler writes next to the switch file: the
#: owner's «✅ Отправить» / «✕ Отмена» under the preview. Duplicated in
#: twinbot/twinbot/story.py — same reason as STORY_PREVIEW_NAME, cross-checked by tests.
STORY_APPROVE_NAME = "twinbot_story.approve"

#: The verdict values written into the approval file; twinbot accepts exactly these.
STORY_ACTION_SEND = "send"
STORY_ACTION_CANCEL = "cancel"

PREVIEW_HEADER = "📖 <b>История от двойника</b>"
PREVIEW_FOOTER = "<i>Отправить её в чат двойника или выбросить? (10 минут на решение)</i>"
#: Cap on the final rendered preview (Telegram rejects a message over 4096 chars).
#: The story is trimmed BEFORE escaping: html.escape can blow each char up ~×4
#: («<» → «&lt;»), so a text-sized cap alone would not keep the message short enough.
PREVIEW_MAX_CHARS = 3500


def preview_path(twinbot_dir: Path) -> Path:
    """Full path of the preview file inside twinbot's data dir."""

    return twinbot_dir / STORY_PREVIEW_NAME


def read_pending_id(settings: Settings) -> str | None:
    """The request id of the preview twinbot currently has on disk, ``None`` if none.

    The verdict must name the story it applies to, and twinbot clears the preview
    file after every decision — so the id comes from the file, not from memory:
    after a restart or a cancelled story there is simply nothing to decide.
    """

    path = preview_path(TWINBOT_DATA_DIR)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    request_id = content.splitlines()[0].strip() if content.splitlines() else ""
    return request_id or None


def story_approve_path(settings: Settings) -> Path:
    """Where twinbot looks for the verdict: next to the switch file, in the writable dir."""

    return settings.db_path.parent / STORY_APPROVE_NAME


def write_story_approve(path: Path, request_id: str, action: str) -> str:
    """Write ``"<iso-ts> <request-id> send|cancel>"``; return the body.

    Same discipline as :func:`adminbot.twin.write_story_request`: unlink first so
    a partial overwrite is never mistaken for a fresh verdict, fresh timestamp so
    twinbot's TTL applies from the press, not from the story's birth.
    """

    body = f"{datetime.now(UTC).isoformat()} {request_id} {action}"
    path.unlink(missing_ok=True)
    path.write_text(body, encoding="utf-8")
    return body


def story_send_markup() -> InlineKeyboardMarkup:
    """The two decision buttons under the previewed story."""

    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Отправить", callback_data=TwinStorySend())
    builder.button(text="✕ Отмена", callback_data=TwinStoryCancel())
    builder.adjust(2)
    return builder.as_markup()


def render_preview(text: str) -> str:
    """The preview message the owner sees: header, story, footer (HTML mode).

    The 4096 cap is applied to the FINAL string: escaping can inflate the text
    several times, so a text-level trim alone would let the message overflow
    Telegram's limit and fail to send.
    """

    body = text.strip() or "<i>(пустой ответ)</i>"
    message = f"{PREVIEW_HEADER}\n\n{html.escape(body)}\n\n{PREVIEW_FOOTER}"
    if len(message) > PREVIEW_MAX_CHARS:
        # Trim the story, not the header/footer, and stay under the cap even in
        # the worst case (every remaining char is an escape surrogate).
        room = PREVIEW_MAX_CHARS - len(PREVIEW_HEADER) - len(PREVIEW_FOOTER) - 4
        room = max(room, 0)
        story = html.escape(body)[: room - 1] + "…" if room else ""
        message = f"{PREVIEW_HEADER}\n\n{story}\n\n{PREVIEW_FOOTER}"
    return message


class StoryPreviewPoller:
    """Watches twinbot's preview file and shows a fresh story to every admin, once per id.

    The shown id is kept in memory only: twinbot deletes the preview file after the verdict
    (send / cancel / timeout), so after a restart the file is simply absent — no id, no message.
    A read that fails (OSError, race with twinbot deleting the file) is logged and skipped;
    the next pass reads whatever is there then.
    """

    def __init__(
        self,
        path: Path,
        send: Callable[[str, InlineKeyboardMarkup], Awaitable[int]],  # like _notify_story
        *,
        interval_s: float = STORY_PREVIEW_INTERVAL_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = path
        self._send = send
        self._interval_s = interval_s
        self._sleep = sleep
        self._clock = clock
        self._shown_id: str | None = None
        self._delivered = 0

    @property
    def shown_id(self) -> str | None:
        """The id of the preview already delivered to the admins (for tests)."""

        return self._shown_id

    @property
    def delivered(self) -> int:
        """How many preview messages have gone out in total (for tests)."""

        return self._delivered

    async def check_once(self) -> bool:
        """One pass: read the file, deliver a *new* story if its id differs from the last shown.

        Returns True when a message went out. The first line of the file is the request id
        (``<iso-ts> <uuid>``, written by twinbot); the rest is the story text. twinbot itself
        never rewrites a preview with the same id, so "id changed" means "a new story".
        """

        try:
            content = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("preview read %s failed: %s", self._path, exc)
            return False
        lines = content.splitlines()
        if not lines:
            return False
        request_id = lines[0].strip()
        if not request_id:
            return False
        if request_id == self._shown_id:
            return False  # already shown, still awaiting the owner's decision
        text = "\n".join(lines[1:]).strip()
        delivered = await self._send(render_preview(text), story_send_markup())
        self._shown_id = request_id
        if delivered:
            self._delivered += 1
            logger.info("story preview %s delivered to %s admins", request_id, delivered)
        else:
            logger.warning("story preview %s reached no admin", request_id)
        return True

    async def run(self) -> None:
        """Poll until cancelled; start it with :func:`start_story_previews`."""

        logger.info("story preview loop started: watching %s", self._path)
        while True:
            try:
                await self.check_once()
            except Exception:  # broad on purpose: the loop must survive anything a pass raises
                logger.exception("story preview pass failed")
            await self._sleep(self._interval_s)


def start_story_previews(poller: StoryPreviewPoller) -> asyncio.Task[None]:
    """Start the loop as a named background task; `task.cancel()` stops it."""

    return asyncio.create_task(poller.run(), name="adminbot-story-preview")
