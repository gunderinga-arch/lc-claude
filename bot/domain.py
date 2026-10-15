"""Domain types shared by every layer. No I/O here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"
    PENDING = "pending"  # asked for access, waiting for an admin
    BLOCKED = "blocked"


class Mode(StrEnum):
    FAST = "fast"
    NORMAL = "normal"
    DEEP = "deep"


class Style(StrEnum):
    """Answer length preference. Works through a prompt instruction: max_tokens would give empty answers."""

    SHORT = "short"
    NORMAL = "normal"
    DETAILED = "detailed"


class ImageModel(StrEnum):
    """Who draws the pictures: the user's pick in the model menu."""

    GEMINI = "gemini"  # Nano Banana through OmniRoute (the owner's Google account); FLUX when it fails
    FLUX = "flux"  # Cloudflare Workers AI, free plan


PERSONA_MAX_CHARS = 2000  # the persona rides along with every request, so it is capped


class RequestStatus(StrEnum):
    OK = "ok"
    STUB = "stub"  # gateway returned its canned political-topic reply
    FAILED = "failed"  # every attempt and fallback exhausted


@dataclass(frozen=True, slots=True)
class User:
    tg_id: int
    username: str | None
    role: Role
    mode: Mode
    model_override: str | None  # explicit model id chosen via /model; None = use the mode's model
    created_at: datetime
    persona: str | None = None  # user's own instructions for the bot's role; None = default behavior
    style: Style = Style.NORMAL
    skills: tuple[str, ...] = ()  # ids of enabled skills from skills.yaml, in the order enabled
    context_limit: int | None = None  # personal history token budget set by an admin; None = global
    daily_limit: int | None = None  # personal requests-per-day limit set by an admin; None = global
    context_idle_hours: float | None = None  # hours of silence before the dialog resets; None = global
    reserve_model: str | None = None  # explicit reserve pick via /model; None = the reserve default
    image_model: ImageModel = ImageModel.GEMINI

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    @property
    def has_access(self) -> bool:
        return self.role in (Role.ADMIN, Role.USER)


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: str  # "user" | "assistant"
    content: str
    approx_tokens: int


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token estimate of one answer (or a sum of them), split by kind.

    `input + output + images + service` is the total; `context` and `tools` are parts of `input`.
    Everything is our own estimate (`approx_tokens`): the gateway's `usage` is partly fabricated.
    """

    input: int = 0  # text sent to the model over all tool rounds: system, history, message, tool results
    output: int = 0  # text the model wrote over all tool rounds
    images: int = 0  # input images, IMAGE_TOKENS each
    service: int = 0  # the hidden prompt rewriter: its input + output
    context: int = 0  # part of `input`: the dialog history
    tools: int = 0  # part of `input`: tool results (web pages, image generation reports)

    @property
    def total(self) -> int:
        return self.input + self.output + self.images + self.service

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input + other.input,
            self.output + other.output,
            self.images + other.images,
            self.service + other.service,
            self.context + other.context,
            self.tools + other.tools,
        )


@dataclass(frozen=True, slots=True)
class RequestRecord:
    user_id: int
    conversation_id: int | None
    model: str  # model that produced the final answer (or the last one tried)
    status: RequestStatus
    attempts: int
    latency_ms: int
    approx_tokens: int  # our own estimate; gateway `usage` is unreliable. Equals usage.total when set
    error: str | None = None
    usage: TokenUsage = TokenUsage()  # the same tokens split by kind
    web_calls: int = 0  # internet tool calls made while answering


@dataclass(frozen=True, slots=True)
class UserUsage:
    """One row of the admin leaderboard: counters only, never message contents."""

    tg_id: int
    username: str | None
    role: Role
    requests: int
    approx_tokens: int
    last_at: datetime | None  # last request; None if the user never asked anything


@dataclass(frozen=True, slots=True)
class UsageStats:
    """Aggregates for the admin /stats command."""

    users_total: int
    users_pending: int
    requests_today: int
    tokens_today: int
    failed_today: int
    avg_latency_ms_today: int


class UploadKind(StrEnum):
    """What went through the bot: files users sent (first group) and files the bot made (GEN_*)."""

    PHOTO = "photo"  # a Telegram photo (compressed)
    IMAGE = "image"  # a picture sent as a file
    PDF = "pdf"
    WORD = "word"  # .docx
    EXCEL = "excel"  # .xlsx
    TEXT = "text"  # plain text, code, csv, json, markdown...
    OTHER = "other"  # any other document (usually refused as unsupported)
    VOICE = "voice"
    AUDIO = "audio"
    VIDEO_NOTE = "video_note"
    GEN_IMAGE = "gen_image"  # a picture the bot generated with FLUX (counts against Cloudflare's quota)
    GEN_IMAGE_GEMINI = "gen_image_gemini"  # a picture drawn by Gemini
    GEN_FILE = "gen_file"  # a file the bot created from its answer


GENERATED_KINDS = frozenset({UploadKind.GEN_IMAGE, UploadKind.GEN_IMAGE_GEMINI, UploadKind.GEN_FILE})

_TEXT_WORDS = (
    ".txt .md .markdown .csv .tsv .json .jsonl .xml .yaml .yml .toml .ini .cfg .conf .log .env "
    ".html .htm .css .js .ts .tsx .jsx .py .java .kt .c .h .cpp .hpp .cs .go .rs .rb .php .swift "
    ".sh .ps1 .bat .sql .svg .tex .rst .srt"
)
_TEXT_EXTENSIONS = frozenset(_TEXT_WORDS.split())
_IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".tif", ".tiff"))


def upload_kind(filename: str | None, mime: str | None) -> UploadKind:
    """Kind of a document users sent, by extension first and MIME type second."""
    name = (filename or "").lower()
    ext = name[name.rfind(".") :] if "." in name else ""
    mime = (mime or "").lower()
    if ext == ".pdf" or mime == "application/pdf":
        return UploadKind.PDF
    if ext == ".docx":
        return UploadKind.WORD
    if ext == ".xlsx":
        return UploadKind.EXCEL
    if ext in _IMAGE_EXTENSIONS or mime.startswith("image/"):
        return UploadKind.IMAGE
    if ext in _TEXT_EXTENSIONS or mime.startswith("text/"):
        return UploadKind.TEXT
    return UploadKind.OTHER


@dataclass(frozen=True, slots=True)
class UploadStat:
    count: int
    bytes: int


@dataclass(frozen=True, slots=True)
class ModelUsage:
    model: str
    requests: int
    tokens: int


@dataclass(frozen=True, slots=True)
class RequestReport:
    """Request counters over a period, for one user or for everyone (admin bot)."""

    requests: int
    ok: int
    stub: int
    failed: int
    avg_latency_ms: int
    total_tokens: int  # SUM(approx_tokens): includes records made before the split by kind existed
    usage: TokenUsage  # the split; records from before it count as zeros here
    web_calls: int
    by_model: tuple[ModelUsage, ...]  # most tokens first
    last_at: datetime | None

    @property
    def unsplit_tokens(self) -> int:
        """Tokens of old records that have no split by kind."""
        return max(0, self.total_tokens - self.usage.total)


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    id: int
    user_id: int
    active: bool  # the dialog the bot currently continues
    started_at: datetime
    messages: int
    last_at: datetime | None  # last message; None for an empty dialog


@dataclass(frozen=True, slots=True)
class ChatInfo:
    """One chat of a user: a Telegram topic (thread_id) with its own history and model."""

    id: int  # conversation id
    thread_id: int
    title: str | None  # None = the topic has not been named yet
    model_override: str | None  # this chat's model; None = the user's own pick / the mode's model


@dataclass(frozen=True, slots=True)
class LoggedMessage:
    """A stored message as the admin bot shows it."""

    id: int
    role: str  # "user" | "assistant"
    content: str
    approx_tokens: int
    created_at: datetime


# Personal context limits an admin may set: below the minimum the bot forgets the question it is
# answering, above the maximum no model has a window that large.
CONTEXT_LIMIT_MIN = 1_000
CONTEXT_LIMIT_MAX = 1_048_576

DAILY_LIMIT_MIN = 1
DAILY_LIMIT_MAX = 100_000


@dataclass(frozen=True, slots=True)
class ErrorEntry:
    """A request that ended without an answer (failed or the gateway's canned stub)."""

    id: int
    user_id: int
    username: str | None
    model: str
    status: RequestStatus
    error: str | None
    attempts: int
    latency_ms: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ModelStats:
    """How one model did over a period."""

    model: str
    requests: int
    ok: int
    stub: int
    failed: int
    avg_ok_latency_ms: int  # answered requests only: timeouts would drown the typical speed
    max_ok_latency_ms: int
    tokens: int  # SUM(approx_tokens)


@dataclass(frozen=True, slots=True)
class DayUsage:
    day: date  # in the admin's time zone
    requests: int
    tokens: int


@dataclass(frozen=True, slots=True)
class BotStatus:
    """The main bot's heartbeat row."""

    started_at: datetime
    heartbeat_at: datetime
    queue_depth: int  # requests waiting for a free gateway slot
    active_jobs: int  # answers being generated right now


HEARTBEAT_S = 30  # how often the main bot rewrites its status row
BOT_STALE_AFTER_S = 120  # no heartbeat for this long: the main bot is down or hung


# One image as the models count it on the gateway (measured 625-1226 prompt tokens).
IMAGE_TOKENS = 1_300

# Tokens per character of each class (Cyrillic and other wide letters, Latin letters, digits and punctuation,
# whitespace), fitted 15.10.2026 to the gateway's prompt_tokens on Russian prose, a Russian markdown doc,
# Python, JSON and English prose, then raised 6%: typical text is overcounted a little, not undercounted.
# Claude (Opus 5, Fable 5.1) spends ~2 chars of Russian a token, ~2.6 of JSON, ~3.5 of code, ~4 of English;
# GPT-6 Sol ~4.4 chars of Russian. A flat 3.5 chars a token undercounted Russian for Claude by ~40%.
TOKEN_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "claude": (0.563, 0.261, 0.776, 0.159),
    "gpt": (0.234, 0.279, 0.591, 0.159),
}
_LATIN = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def count_tokens(text: str, tokenizer: str = "claude") -> int:
    """Tokens of `text` as the `tokenizer` family counts them (unknown families count as Claude, the heavier).

    Fast on megabytes of text: the classes are counted with str methods, not char by char.
    """
    if not text:
        return 0
    wide_w, latin_w, other_w, space_w = TOKEN_WEIGHTS.get(tokenizer, TOKEN_WEIGHTS["claude"])
    size = len(text)
    latin = size - len(text.translate(_LATIN))
    spaces = text.count(" ") + text.count("\n") + text.count("\t")
    # A Cyrillic letter is 2 bytes in UTF-8: the extra bytes count them (a dash or an emoji counts double).
    wide = min(len(text.encode("utf-8", "surrogatepass")) - size, size - latin - spaces)
    other = size - latin - spaces - wide
    return math.ceil(wide * wide_w + latin * latin_w + other * other_w + spaces * space_w)


def approx_tokens(text: str) -> int:
    """Tokens for budgets and accounting: the Claude count, the heaviest of the models here."""
    return count_tokens(text)
