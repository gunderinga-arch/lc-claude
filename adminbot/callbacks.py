"""Callback data of every admin bot button. The screens link to each other only through these."""

from __future__ import annotations

from aiogram.filters.callback_data import CallbackData

from adminbot.common import Audience, Period
from bot.domain import Role


class Home(CallbackData, prefix="home"):
    """Main menu."""


class Dashboard(CallbackData, prefix="dash"):
    """Usage report over all users."""

    period: Period


class Top(CallbackData, prefix="top"):
    """Leaderboard; `order` is a key of bot.storage.repositories.LEADERBOARD_ORDER."""

    order: str


class UserList(CallbackData, prefix="ul"):
    """Users newest first; role=None shows everyone."""

    role: Role | None = None
    page: int = 0


class FindHint(CallbackData, prefix="find"):
    """Explains the search: any text sent to the bot is a search query."""


class AddUser(CallbackData, prefix="add"):
    """Asks for a Telegram id to add to the whitelist."""


class UserCard(CallbackData, prefix="u"):
    tg_id: int


class SetRole(CallbackData, prefix="role"):
    tg_id: int
    role: Role


class UserStats(CallbackData, prefix="us"):
    tg_id: int
    period: Period


class Limit(CallbackData, prefix="lim"):
    """The context limit screen of one user."""

    tg_id: int


LIMIT_DEFAULT = 0  # SetLimit.value: back to the global HISTORY_TOKEN_BUDGET
LIMIT_CUSTOM = -1  # SetLimit.value: ask the admin to type a number


class SetLimit(CallbackData, prefix="setlim"):
    tg_id: int
    value: int  # tokens, or LIMIT_DEFAULT / LIMIT_CUSTOM


class Dialogs(CallbackData, prefix="dl"):
    """The user's dialogs, newest first."""

    tg_id: int
    page: int = 0


LAST_PAGE = -1  # DialogPage.page: the newest messages


class DialogPage(CallbackData, prefix="dp"):
    conv_id: int
    page: int = LAST_PAGE


class Export(CallbackData, prefix="ex"):
    """The whole dialog as a .txt document."""

    conv_id: int


# ---- second feature set ---------------------------------------------------------------------------


class DailyLimit(CallbackData, prefix="dlim"):
    """The requests-per-day limit screen of one user."""

    tg_id: int


class SetDailyLimit(CallbackData, prefix="setdl"):
    tg_id: int
    value: int  # requests a day, or LIMIT_DEFAULT / LIMIT_CUSTOM


# Unlike LIMIT_DEFAULT, 0 is a real value here ("never reset"), so the sentinels are negative.
IDLE_CUSTOM = -1  # SetContextIdle.value: ask the admin to type a number
IDLE_DEFAULT = -2  # SetContextIdle.value: back to the global CONTEXT_IDLE_HOURS


class ContextIdle(CallbackData, prefix="idle"):
    """The context-reset (idle hours) screen of one user."""

    tg_id: int


class SetContextIdle(CallbackData, prefix="setidle"):
    tg_id: int
    value: float  # hours of silence, 0 = never reset, or IDLE_DEFAULT / IDLE_CUSTOM


class WriteTo(CallbackData, prefix="write"):
    """Start a personal message to one user (sent by the main bot)."""

    tg_id: int


class Broadcast(CallbackData, prefix="bc"):
    """Pick the audience of a broadcast."""


class BroadcastTo(CallbackData, prefix="bcto"):
    audience: Audience


class SendConfirm(CallbackData, prefix="send"):
    """Confirm (ok=True) or drop the message waiting in the FSM data, personal or broadcast."""

    ok: bool


class Errors(CallbackData, prefix="err"):
    """Error journal, newest first; tg_id=None means everyone."""

    page: int = 0
    tg_id: int | None = None


class Models(CallbackData, prefix="mdl"):
    """Model comparison over a period."""

    period: Period


class Activity(CallbackData, prefix="act"):
    """Requests per day, last ACTIVITY_DAYS days; tg_id=None means everyone."""

    tg_id: int | None = None


class Health(CallbackData, prefix="health"):
    """Main bot status and the external services' free quotas."""


class Backup(CallbackData, prefix="bkp"):
    """Database backup: confirm=False shows the warning, True sends the file."""

    confirm: bool = False


class Reserve(CallbackData, prefix="rsv"):
    """Emergency switch and model probe screen."""


class ReserveSet(CallbackData, prefix="rsvset"):
    """Flip the reserve switch; on=True asks to confirm first unless confirm=True."""

    on: bool
    confirm: bool = False


class ProbeNow(CallbackData, prefix="probe"):
    """Probe every model right away, outside the schedule."""


class Twin(CallbackData, prefix="twin"):
    """The owner's style-clone bot (twinbot): its on/off switch."""


class TwinSet(CallbackData, prefix="twinset"):
    """Turn the twin bot on or off."""

    on: bool


class TwinStory(CallbackData, prefix="twinstory"):
    """Ask the twin bot to tell a funny story from the Markus chat, right now."""


class TwinStorySend(CallbackData, prefix="twinstorysend"):
    """Send the previewed story to the twin bot's chat (the verdict file says `send`)."""


class TwinStoryCancel(CallbackData, prefix="twinstorycancel"):
    """Drop the previewed story (the verdict file says `cancel`)."""


class ServerBackup(CallbackData, prefix="srvbkp"):
    """Full server backup: run=False shows the screen, True asks the server to build and send it."""

    run: bool = False


class Balance(CallbackData, prefix="bal"):
    """Tokens left on the BuyTokens keys (asked from the gateway's site on every open)."""
