"""Runtime settings, read from environment variables and `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from bot.files.blocks import FILE_INSTRUCTIONS
from bot.files.slides import SLIDES_INSTRUCTIONS

DEFAULT_MODELS_FILE = Path(__file__).with_name("models.yaml")
DEFAULT_SKILLS_FILE = Path(__file__).with_name("skills.yaml")

DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай на языке пользователя, по делу и без воды. "
    "Используй Markdown: **жирный**, `код`, блоки кода с указанием языка, списки. "
    "Не используй таблицы и HTML.\n\n"
    # 15.10.2026 Opus stopped a long translation at 38k chars: "в одно сообщение не влезет, скажи «дальше»".
    "Длина ответа не ограничена. Если задача требует большого ответа (перевод целиком, длинный документ, "
    "весь код), выдай его полностью за один раз: не останавливайся на полпути и не предлагай продолжить "
    "в следующем сообщении. Оборвётся ответ на лимите — бот сам попросит тебя продолжить с места обрыва, "
    "а очень длинный ответ пришлёт пользователю файлом.\n\n"
    # The bot parses these blocks into documents: a SYSTEM_PROMPT override must keep this part.
    f"{FILE_INSTRUCTIONS}\n\n{SLIDES_INSTRUCTIONS}"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: SecretStr
    telegram_api_base: str | None = None
    buytokens_api_key: SecretStr
    # Second BuyTokens key: benchmarks run on it, the bots never use it; the admin bot shows its balance.
    buytokens_api_key_2: SecretStr | None = None
    # The separate admin bot (python -m adminbot); with it set, access requests go there too.
    admin_bot_token: SecretStr | None = None
    admin_utc_offset: int = Field(default=3, ge=-12, le=14)  # hours; the admin bot shows times in it
    # Money: the gateway sells one token counter (~$0.5 per 50M), models may override it in models.yaml.
    token_price: float = Field(default=0.01, ge=0)  # per 1M tokens
    price_currency: str = "$"
    # Free plans of the external services, for the admin bot's quota view and alerts.
    cloudflare_daily_neurons: int = Field(default=10_000, ge=1)
    # Admin alerts (0 / -1 turns one off).
    alert_failed_count: int = Field(default=5, ge=0)  # gateway failures within 15 minutes
    alert_user_tokens: int = Field(default=1_000_000, ge=0)  # tokens one user spends in a day
    daily_report_hour: int = Field(default=9, ge=-1, le=23)  # admin's local hour for the daily summary
    buytokens_base_url: str = "https://api.example.invalid/v1"
    # Reserve gateway (OmniRoute Free Kit, OpenAI-compatible): the admin bot's emergency switch sends
    # every model call there. Without a URL the switch is hidden and the bot always uses BuyTokens.
    reserve_base_url: str | None = None
    reserve_api_key: SecretStr | None = None
    reserve_model: str = "free-coded"  # OmniRoute combo: the strongest free model that answers
    reserve_token_budget: int = Field(default=100_000, ge=1_000)  # free models have smaller contexts
    # Test request to every model this often (admin bot); 0 turns the probe and its alerts off.
    model_probe_minutes: int = Field(default=15, ge=0)

    admin_ids: Annotated[frozenset[int], NoDecode] = frozenset()
    access_mode: Literal["whitelist", "open"] = "whitelist"
    daily_request_limit: int = Field(default=100, ge=1)  # per non-admin user; admins are unlimited
    daily_image_limit: int = Field(default=6, ge=0)  # pictures per non-admin user per day
    max_concurrency: int = Field(default=5, ge=1)  # simultaneous requests to the gateway
    # Tokens of dialog sent with a message, capped anyway by the answering model's window (~1M, models.yaml):
    # the default leaves the window as the only limit. Lower it to save gateway tokens on long chats.
    history_token_budget: int = Field(default=2_000_000, ge=1_000)
    # Hours of silence in a dialog before its history is quietly dropped; 0 turns it off.
    context_idle_hours: float = Field(default=24, ge=0)

    db_path: Path = Path("data/bot.db")
    models_file: Path = DEFAULT_MODELS_FILE
    skills_file: Path = DEFAULT_SKILLS_FILE
    searxng_url: str | None = None  # local SearXNG, http://127.0.0.1:8888; without it web tools are off
    serper_api_key: SecretStr | None = None  # paid Google search, used only when SearXNG fails
    # Image generation (Cloudflare Workers AI); without both, image skills are hidden.
    cloudflare_account_id: str | None = None
    cloudflare_api_token: SecretStr | None = None

    # Voice messages: faster-whisper model size (tiny/base/small/medium); empty turns voice off.
    stt_model: str = "small"
    stt_models_dir: Path = Path("data/models")
    stt_max_seconds: int = Field(default=300, ge=10)  # longer voice messages are refused
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    log_level: str = "INFO"

    @field_validator("telegram_api_base")
    @classmethod
    def _local_api(cls, value: str | None) -> str | None:
        if not value:
            return None
        value = value.rstrip("/")
        url = urlsplit(value)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
        ):
            raise ValueError("TELEGRAM_API_BASE must be a loopback HTTP URL")
        return value

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(int(part) for part in value.replace(";", ",").split(",") if part.strip())
        if isinstance(value, int):
            return frozenset({value})
        return value

    @field_validator("buytokens_base_url", "reserve_base_url")
    @classmethod
    def _strip_slash(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value is not None else None

    @property
    def reserve_configured(self) -> bool:
        """An empty RESERVE_BASE_URL= line in .env counts as "not set"."""
        return bool(self.reserve_base_url)
