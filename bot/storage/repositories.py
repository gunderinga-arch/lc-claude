"""Repositories: thin async wrappers around SQL for each aggregate."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta, timezone

import aiosqlite

from bot.domain import (
    BotStatus,
    ChatInfo,
    ConversationSummary,
    DayUsage,
    ErrorEntry,
    ImageModel,
    LoggedMessage,
    Mode,
    ModelStats,
    ModelUsage,
    RequestRecord,
    RequestReport,
    RequestStatus,
    Role,
    StoredMessage,
    Style,
    TokenUsage,
    UploadKind,
    UploadStat,
    UsageStats,
    User,
    UserUsage,
)
from bot.storage.db import Database

# Leaderboard sort keys -> SQL. A fixed mapping: the order never comes from user input as SQL.
LEADERBOARD_ORDER = {
    "requests": "requests DESC, tokens DESC",
    "tokens": "tokens DESC, requests DESC",
    "last": "last_at DESC",  # SQLite sorts NULL (never active) last in DESC
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _today_start() -> str:
    """Start of the current UTC day, comparable with `_now()` strings (same ISO format)."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _row_to_user(row: aiosqlite.Row) -> User:
    return User(
        tg_id=row["tg_id"],
        username=row["username"],
        role=Role(row["role"]),
        mode=Mode(row["mode"]),
        model_override=row["model_override"],
        created_at=datetime.fromisoformat(row["created_at"]),
        persona=row["persona"],
        style=Style(row["style"]),
        skills=tuple(s for s in row["skills"].split(",") if s),
        context_limit=row["context_limit"],
        daily_limit=row["daily_limit"],
        context_idle_hours=row["context_idle_hours"],
        reserve_model=row["reserve_model"],
        image_model=ImageModel(row["image_model"]),
    )


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _since_clause(column: str, since: datetime | None) -> tuple[str, tuple[str, ...]]:
    """SQL condition and params for "column >= since"; no condition for all time."""
    if since is None:
        return "1 = 1", ()
    return f"{column} >= ?", (since.astimezone(UTC).isoformat(),)


def _like_pattern(text: str) -> str:
    """LIKE pattern for "contains text", with the wildcards in `text` taken literally (ESCAPE '!')."""
    return "%" + text.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


class UserRepo:
    """CRUD for the `users` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, tg_id: int) -> User | None:
        cursor = await self._db.conn.execute("SELECT * FROM users WHERE tg_id = ?", (tg_id,))
        row = await cursor.fetchone()
        return _row_to_user(row) if row else None

    async def upsert_seen(
        self, tg_id: int, username: str | None, *, default_role: Role, default_mode: Mode
    ) -> User:
        """Insert a new user with the given defaults, or refresh an existing user's username."""
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO users (tg_id, username, role, mode, model_override, created_at)
            VALUES (?, ?, ?, ?, NULL, ?)
            ON CONFLICT(tg_id) DO UPDATE SET username = excluded.username
            """,
            (tg_id, username, default_role.value, default_mode.value, _now()),
        )
        await conn.commit()
        user = await self.get(tg_id)
        assert user is not None
        return user

    async def set_role(self, tg_id: int, role: Role) -> bool:
        conn = self._db.conn
        cursor = await conn.execute("UPDATE users SET role = ? WHERE tg_id = ?", (role.value, tg_id))
        await conn.commit()
        return cursor.rowcount > 0

    async def set_mode(self, tg_id: int, mode: Mode) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET mode = ? WHERE tg_id = ?", (mode.value, tg_id))
        await conn.commit()

    async def set_model_override(self, tg_id: int, model_id: str | None) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET model_override = ? WHERE tg_id = ?", (model_id, tg_id))
        await conn.commit()

    async def set_reserve_model(self, tg_id: int, model_id: str | None) -> None:
        """Personal pick from the reserve pool; None returns the user to the reserve's default."""
        conn = self._db.conn
        await conn.execute("UPDATE users SET reserve_model = ? WHERE tg_id = ?", (model_id, tg_id))
        await conn.commit()

    async def set_image_model(self, tg_id: int, model: ImageModel) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET image_model = ? WHERE tg_id = ?", (model.value, tg_id))
        await conn.commit()

    async def list_by_role(self, role: Role) -> list[User]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM users WHERE role = ? ORDER BY created_at", (role.value,)
        )
        rows = await cursor.fetchall()
        return [_row_to_user(row) for row in rows]

    async def set_persona(self, tg_id: int, persona: str | None) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET persona = ? WHERE tg_id = ?", (persona, tg_id))
        await conn.commit()

    async def set_style(self, tg_id: int, style: Style) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET style = ? WHERE tg_id = ?", (style.value, tg_id))
        await conn.commit()

    async def set_skills(self, tg_id: int, skills: tuple[str, ...]) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET skills = ? WHERE tg_id = ?", (",".join(skills), tg_id))
        await conn.commit()

    async def get_chat_floor(self, tg_id: int) -> int | None:
        cursor = await self._db.conn.execute("SELECT chat_floor_id FROM users WHERE tg_id = ?", (tg_id,))
        row = await cursor.fetchone()
        return row["chat_floor_id"] if row else None

    async def set_chat_floor(self, tg_id: int, message_id: int) -> None:
        conn = self._db.conn
        await conn.execute("UPDATE users SET chat_floor_id = ? WHERE tg_id = ?", (message_id, tg_id))
        await conn.commit()

    async def list_all(self, limit: int = 100) -> list[User]:
        cursor = await self._db.conn.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
        return [_row_to_user(row) for row in rows]

    async def set_context_limit(self, tg_id: int, limit: int | None) -> bool:
        """Personal history token budget; None returns the user to the global one."""
        conn = self._db.conn
        cursor = await conn.execute("UPDATE users SET context_limit = ? WHERE tg_id = ?", (limit, tg_id))
        await conn.commit()
        return cursor.rowcount > 0

    async def set_daily_limit(self, tg_id: int, limit: int | None) -> bool:
        """Personal requests-per-day limit; None returns the user to DAILY_REQUEST_LIMIT."""
        conn = self._db.conn
        cursor = await conn.execute("UPDATE users SET daily_limit = ? WHERE tg_id = ?", (limit, tg_id))
        await conn.commit()
        return cursor.rowcount > 0

    async def set_context_idle_hours(self, tg_id: int, hours: float | None) -> bool:
        """Personal hours of silence before the dialog resets; None returns to CONTEXT_IDLE_HOURS."""
        conn = self._db.conn
        cursor = await conn.execute("UPDATE users SET context_idle_hours = ? WHERE tg_id = ?", (hours, tg_id))
        await conn.commit()
        return cursor.rowcount > 0

    async def ids_by_roles(self, roles: Iterable[Role]) -> list[int]:
        """Telegram ids of everyone with one of `roles`, oldest users first (broadcast audience)."""
        values = [r.value for r in roles]
        if not values:
            return []
        marks = ", ".join("?" * len(values))
        cursor = await self._db.conn.execute(
            f"SELECT tg_id FROM users WHERE role IN ({marks}) ORDER BY created_at, tg_id", values
        )
        return [row["tg_id"] for row in await cursor.fetchall()]

    async def list_page(self, role: Role | None, limit: int, offset: int) -> list[User]:
        """Users newest first, optionally only one role."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM users WHERE ? IS NULL OR role = ? "
            "ORDER BY created_at DESC, tg_id LIMIT ? OFFSET ?",
            (role, role, limit, offset),
        )
        return [_row_to_user(row) for row in await cursor.fetchall()]

    async def count(self, role: Role | None = None) -> int:
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) FROM users WHERE ? IS NULL OR role = ?", (role, role)
        )
        (total,) = await cursor.fetchone()
        return int(total)

    async def search(self, query: str, limit: int = 20) -> list[User]:
        """By exact Telegram id, or by a part of the username (with or without @, any case)."""
        query = query.strip().removeprefix("@")
        if not query:
            return []
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM users
            WHERE CAST(tg_id AS TEXT) = ? OR username LIKE ? ESCAPE '!'
            ORDER BY CAST(tg_id AS TEXT) = ? DESC, username
            LIMIT ?
            """,
            (query, _like_pattern(query), query, limit),
        )
        return [_row_to_user(row) for row in await cursor.fetchall()]


class ConversationRepo:
    """CRUD for `conversations` and `messages`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _active_row(self, user_id: int, thread_id: int = 0) -> aiosqlite.Row | None:
        cursor = await self._db.conn.execute(
            "SELECT id, title, model_override FROM conversations "
            "WHERE user_id = ? AND thread_id = ? AND active = 1",
            (user_id, thread_id),
        )
        return await cursor.fetchone()

    async def active_id(
        self,
        user_id: int,
        thread_id: int = 0,
        *,
        model_override: str | None = None,
        title: str | None = None,
    ) -> int:
        """Id of the user's active conversation in the topic (0 = main chat), creating one if needed.

        `model_override` and `title` only matter when a conversation is created here.
        """
        row = await self._active_row(user_id, thread_id)
        if row is not None:
            return int(row["id"])
        conn = self._db.conn
        cursor = await conn.execute(
            "INSERT INTO conversations (user_id, active, started_at, thread_id, title, model_override) "
            "VALUES (?, 1, ?, ?, ?, ?)",
            (user_id, _now(), thread_id, title, model_override),
        )
        await conn.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    async def chat(self, user_id: int, thread_id: int) -> ChatInfo | None:
        """The active chat of a topic; None if the bot has not seen that topic."""
        row = await self._active_row(user_id, thread_id)
        if row is None:
            return None
        return ChatInfo(int(row["id"]), thread_id, row["title"], row["model_override"])

    async def set_title(self, user_id: int, thread_id: int, title: str) -> None:
        conn = self._db.conn
        await conn.execute(
            "UPDATE conversations SET title = ? WHERE user_id = ? AND thread_id = ? AND active = 1",
            (title, user_id, thread_id),
        )
        await conn.commit()

    async def set_model(self, user_id: int, thread_id: int, model_id: str | None) -> None:
        """The model of one chat (None = back to the user's own pick)."""
        await self.active_id(user_id, thread_id)
        conn = self._db.conn
        await conn.execute(
            "UPDATE conversations SET model_override = ? WHERE user_id = ? AND thread_id = ? AND active = 1",
            (model_id, user_id, thread_id),
        )
        await conn.commit()

    async def empty_thread(self, user_id: int) -> int | None:
        """The newest topic the bot opened that nobody has written in yet (still unnamed), if any."""
        cursor = await self._db.conn.execute(
            """
            SELECT thread_id FROM conversations c
            WHERE user_id = ? AND active = 1 AND thread_id != 0 AND title IS NULL
              AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
            ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        return row["thread_id"] if row else None

    async def active_threads(self, user_id: int) -> list[int]:
        """Topics where the user has an active chat (the main chat, 0, is not one of them)."""
        cursor = await self._db.conn.execute(
            "SELECT thread_id FROM conversations WHERE user_id = ? AND active = 1 AND thread_id != 0",
            (user_id,),
        )
        return [row["thread_id"] for row in await cursor.fetchall()]

    async def last_activity(self, conversation_id: int) -> datetime | None:
        """When the last message of the conversation was stored; None if it has no messages."""
        cursor = await self._db.conn.execute(
            "SELECT MAX(created_at) FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        (value,) = await cursor.fetchone()
        return datetime.fromisoformat(value) if value else None

    async def started_at(self, conversation_id: int) -> datetime | None:
        """When the conversation was created; the fallback idle reference for empty dialogs."""
        cursor = await self._db.conn.execute(
            "SELECT started_at FROM conversations WHERE id = ?", (conversation_id,)
        )
        row = await cursor.fetchone()
        return datetime.fromisoformat(row["started_at"]) if row else None

    async def reset(self, user_id: int, thread_id: int = 0) -> None:
        """End the active conversation of the topic: the model stops seeing it, the admin bot still can."""
        conn = self._db.conn
        await conn.execute(
            "UPDATE conversations SET active = 0 WHERE user_id = ? AND thread_id = ? AND active = 1",
            (user_id, thread_id),
        )
        await conn.commit()

    async def add_message(self, conversation_id: int, role: str, content: str, approx_tokens: int) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, approx_tokens, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, approx_tokens, _now()),
        )
        await conn.commit()

    async def recent_messages(self, conversation_id: int, limit: int) -> list[StoredMessage]:
        """Return the last `limit` messages of the conversation, oldest first."""
        cursor = await self._db.conn.execute(
            """
            SELECT role, content, approx_tokens FROM messages
            WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            StoredMessage(role=row["role"], content=row["content"], approx_tokens=row["approx_tokens"])
            for row in reversed(rows)
        ]

    async def list_for_user(self, user_id: int, limit: int, offset: int = 0) -> list[ConversationSummary]:
        """The user's dialogs, newest first."""
        cursor = await self._db.conn.execute(
            """
            SELECT c.id, c.user_id, c.active, c.started_at,
                   COUNT(m.id) AS messages, MAX(m.created_at) AS last_at
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        )
        return [_row_to_summary(row) for row in await cursor.fetchall()]

    async def count_for_user(self, user_id: int) -> int:
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)
        )
        (total,) = await cursor.fetchone()
        return int(total)

    async def summary(self, conversation_id: int) -> ConversationSummary | None:
        cursor = await self._db.conn.execute(
            """
            SELECT c.id, c.user_id, c.active, c.started_at,
                   COUNT(m.id) AS messages, MAX(m.created_at) AS last_at
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return _row_to_summary(row) if row else None

    async def messages_page(self, conversation_id: int, limit: int, offset: int = 0) -> list[LoggedMessage]:
        """Messages in chronological order, `offset` counted from the first one."""
        cursor = await self._db.conn.execute(
            """
            SELECT id, role, content, approx_tokens, created_at FROM messages
            WHERE conversation_id = ? ORDER BY id LIMIT ? OFFSET ?
            """,
            (conversation_id, limit, offset),
        )
        return [
            LoggedMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                approx_tokens=row["approx_tokens"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in await cursor.fetchall()
        ]

    async def purge_older_than(self, days: int) -> int:
        """Delete messages older than `days` days and return the number deleted."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        conn = self._db.conn
        cursor = await conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
        await conn.commit()
        return cursor.rowcount


def _row_to_summary(row: aiosqlite.Row) -> ConversationSummary:
    return ConversationSummary(
        id=row["id"],
        user_id=row["user_id"],
        active=bool(row["active"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        messages=row["messages"],
        last_at=_dt(row["last_at"]),
    )


def _errors_where(user_id: int | None, since: datetime | None, failed_only: bool) -> tuple[str, tuple]:
    where, params = _since_clause("r.created_at", since)
    statuses = (
        [RequestStatus.FAILED.value]
        if failed_only
        else [RequestStatus.FAILED.value, RequestStatus.STUB.value]
    )
    marks = ", ".join("?" * len(statuses))
    return (
        f"{where} AND r.status IN ({marks}) AND (? IS NULL OR r.user_id = ?)",
        (*params, *statuses, user_id, user_id),
    )


class RequestRepo:
    """Append-only log of gateway requests, and today's aggregates."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, record: RequestRecord) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO requests
                (user_id, conversation_id, model, status, attempts, latency_ms, approx_tokens, error,
                 input_tokens, output_tokens, image_tokens, service_tokens, context_tokens, tool_tokens,
                 web_calls, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.user_id,
                record.conversation_id,
                record.model,
                record.status.value,
                record.attempts,
                record.latency_ms,
                record.approx_tokens,
                record.error,
                record.usage.input,
                record.usage.output,
                record.usage.images,
                record.usage.service,
                record.usage.context,
                record.usage.tools,
                record.web_calls,
                _now(),
            ),
        )
        await conn.commit()

    async def stats_today(self) -> UsageStats:
        conn = self._db.conn
        cursor = await conn.execute("SELECT COUNT(*) FROM users")
        (users_total,) = await cursor.fetchone()

        cursor = await conn.execute("SELECT COUNT(*) FROM users WHERE role = ?", (Role.PENDING.value,))
        (users_pending,) = await cursor.fetchone()

        cursor = await conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(approx_tokens), 0),
                   SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), AVG(latency_ms)
            FROM requests WHERE created_at >= ?
            """,
            (RequestStatus.FAILED.value, _today_start()),
        )
        count, tokens, failed, avg_latency = await cursor.fetchone()
        return UsageStats(
            users_total=users_total,
            users_pending=users_pending,
            requests_today=count,
            tokens_today=tokens,
            failed_today=failed or 0,
            avg_latency_ms_today=int(avg_latency) if avg_latency is not None else 0,
        )

    async def report(self, user_id: int | None, since: datetime | None) -> RequestReport:
        """Counters since `since` (None = all time) for one user, or for everyone with user_id=None."""
        where, params = _since_clause("created_at", since)
        where += " AND (? IS NULL OR user_id = ?)"
        params = (*params, user_id, user_id)
        conn = self._db.conn
        cursor = await conn.execute(
            f"""
            SELECT COUNT(*) AS requests,
                   COALESCE(SUM(status = 'ok'), 0) AS ok,
                   COALESCE(SUM(status = 'stub'), 0) AS stub,
                   COALESCE(SUM(status = 'failed'), 0) AS failed,
                   AVG(latency_ms) AS latency,
                   COALESCE(SUM(approx_tokens), 0) AS total,
                   COALESCE(SUM(input_tokens), 0) AS input,
                   COALESCE(SUM(output_tokens), 0) AS output,
                   COALESCE(SUM(image_tokens), 0) AS images,
                   COALESCE(SUM(service_tokens), 0) AS service,
                   COALESCE(SUM(context_tokens), 0) AS context,
                   COALESCE(SUM(tool_tokens), 0) AS tools,
                   COALESCE(SUM(web_calls), 0) AS web_calls,
                   MAX(created_at) AS last_at
            FROM requests WHERE {where}
            """,
            params,
        )
        row = await cursor.fetchone()
        cursor = await conn.execute(
            f"""
            SELECT model, COUNT(*) AS requests, COALESCE(SUM(approx_tokens), 0) AS tokens
            FROM requests WHERE {where}
            GROUP BY model ORDER BY tokens DESC, requests DESC, model
            """,
            params,
        )
        by_model = tuple(
            ModelUsage(model=m["model"], requests=m["requests"], tokens=m["tokens"])
            for m in await cursor.fetchall()
        )
        return RequestReport(
            requests=row["requests"],
            ok=row["ok"],
            stub=row["stub"],
            failed=row["failed"],
            avg_latency_ms=int(row["latency"]) if row["latency"] is not None else 0,
            total_tokens=row["total"],
            usage=TokenUsage(
                input=row["input"],
                output=row["output"],
                images=row["images"],
                service=row["service"],
                context=row["context"],
                tools=row["tools"],
            ),
            web_calls=row["web_calls"],
            by_model=by_model,
            last_at=_dt(row["last_at"]),
        )

    async def errors(
        self,
        limit: int,
        offset: int = 0,
        *,
        user_id: int | None = None,
        since: datetime | None = None,
        failed_only: bool = False,
    ) -> list[ErrorEntry]:
        """Requests without an answer (failed, and stubs unless failed_only), newest first."""
        where, params = _errors_where(user_id, since, failed_only)
        cursor = await self._db.conn.execute(
            f"""
            SELECT r.id, r.user_id, u.username, r.model, r.status, r.error, r.attempts, r.latency_ms,
                   r.created_at
            FROM requests r LEFT JOIN users u ON u.tg_id = r.user_id
            WHERE {where}
            ORDER BY r.id DESC LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
        return [
            ErrorEntry(
                id=row["id"],
                user_id=row["user_id"],
                username=row["username"],
                model=row["model"],
                status=RequestStatus(row["status"]),
                error=row["error"],
                attempts=row["attempts"],
                latency_ms=row["latency_ms"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in await cursor.fetchall()
        ]

    async def count_errors(
        self, *, user_id: int | None = None, since: datetime | None = None, failed_only: bool = False
    ) -> int:
        where, params = _errors_where(user_id, since, failed_only)
        cursor = await self._db.conn.execute(f"SELECT COUNT(*) FROM requests r WHERE {where}", params)
        (total,) = await cursor.fetchone()
        return int(total)

    async def model_stats(self, user_id: int | None, since: datetime | None) -> list[ModelStats]:
        """Per-model counters, most requests first."""
        where, params = _since_clause("created_at", since)
        cursor = await self._db.conn.execute(
            f"""
            SELECT model, COUNT(*) AS requests,
                   COALESCE(SUM(status = 'ok'), 0) AS ok,
                   COALESCE(SUM(status = 'stub'), 0) AS stub,
                   COALESCE(SUM(status = 'failed'), 0) AS failed,
                   AVG(CASE WHEN status = 'ok' THEN latency_ms END) AS avg_ok,
                   MAX(CASE WHEN status = 'ok' THEN latency_ms END) AS max_ok,
                   COALESCE(SUM(approx_tokens), 0) AS tokens
            FROM requests WHERE {where} AND (? IS NULL OR user_id = ?)
            GROUP BY model ORDER BY requests DESC, model
            """,
            (*params, user_id, user_id),
        )
        return [
            ModelStats(
                model=row["model"],
                requests=row["requests"],
                ok=row["ok"],
                stub=row["stub"],
                failed=row["failed"],
                avg_ok_latency_ms=int(row["avg_ok"] or 0),
                max_ok_latency_ms=int(row["max_ok"] or 0),
                tokens=row["tokens"],
            )
            for row in await cursor.fetchall()
        ]

    async def by_day(self, user_id: int | None, days: int, today: date, tz: timezone) -> list[DayUsage]:
        """Requests and tokens of the last `days` local days up to `today`, oldest first, empty days too."""
        first = today - timedelta(days=days - 1)
        offset_minutes = int(tz.utcoffset(None).total_seconds() // 60)
        cursor = await self._db.conn.execute(
            """
            SELECT date(created_at, ?) AS day, COUNT(*) AS requests, COALESCE(SUM(approx_tokens), 0) AS tokens
            FROM requests
            WHERE created_at >= ? AND (? IS NULL OR user_id = ?)
            GROUP BY day
            """,
            (
                f"{offset_minutes:+d} minutes",
                datetime.combine(first, time(), tzinfo=tz).astimezone(UTC).isoformat(),
                user_id,
                user_id,
            ),
        )
        found = {row["day"]: row for row in await cursor.fetchall()}
        result = []
        for n in range(days):
            day = first + timedelta(days=n)
            row = found.get(day.isoformat())
            result.append(DayUsage(day, row["requests"] if row else 0, row["tokens"] if row else 0))
        return result

    async def heavy_users(self, since: datetime, min_tokens: int) -> list[UserUsage]:
        """Users who spent at least `min_tokens` since `since`, most first."""
        cursor = await self._db.conn.execute(
            """
            SELECT u.tg_id, u.username, u.role, COUNT(r.id) AS requests,
                   SUM(r.approx_tokens) AS tokens, MAX(r.created_at) AS last_at
            FROM requests r JOIN users u ON u.tg_id = r.user_id
            WHERE r.created_at >= ?
            GROUP BY u.tg_id HAVING tokens >= ?
            ORDER BY tokens DESC
            """,
            (since.astimezone(UTC).isoformat(), min_tokens),
        )
        return [
            UserUsage(
                tg_id=row["tg_id"],
                username=row["username"],
                role=Role(row["role"]),
                requests=row["requests"],
                approx_tokens=row["tokens"],
                last_at=_dt(row["last_at"]),
            )
            for row in await cursor.fetchall()
        ]

    async def leaderboard(self, order: str, limit: int = 20) -> list[UserUsage]:
        """All-time per-user counters, ranked by `order` (a LEADERBOARD_ORDER key)."""
        cursor = await self._db.conn.execute(
            f"""
            SELECT u.tg_id, u.username, u.role,
                   COUNT(r.id) AS requests,
                   COALESCE(SUM(r.approx_tokens), 0) AS tokens,
                   MAX(r.created_at) AS last_at
            FROM users u LEFT JOIN requests r ON r.user_id = u.tg_id
            GROUP BY u.tg_id
            ORDER BY {LEADERBOARD_ORDER[order]}, u.tg_id
            LIMIT ?
            """,
            (limit,),
        )
        return [
            UserUsage(
                tg_id=row["tg_id"],
                username=row["username"],
                role=Role(row["role"]),
                requests=row["requests"],
                approx_tokens=row["tokens"],
                last_at=datetime.fromisoformat(row["last_at"]) if row["last_at"] else None,
            )
            for row in await cursor.fetchall()
        ]


class UsageRepo:
    """Per-user, per-day request/token counters."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def increment(self, user_id: int, approx_tokens: int) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO daily_usage (user_id, day, requests, approx_tokens)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
                requests = requests + 1,
                approx_tokens = approx_tokens + excluded.approx_tokens
            """,
            (user_id, _today(), approx_tokens),
        )
        await conn.commit()

    async def requests_today(self, user_id: int) -> int:
        cursor = await self._db.conn.execute(
            "SELECT requests FROM daily_usage WHERE user_id = ? AND day = ?",
            (user_id, _today()),
        )
        row = await cursor.fetchone()
        return int(row["requests"]) if row else 0

    async def add_images(self, user_id: int, count: int) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO daily_usage (user_id, day, images) VALUES (?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET images = images + excluded.images
            """,
            (user_id, _today(), count),
        )
        await conn.commit()

    async def images_today(self, user_id: int) -> int:
        cursor = await self._db.conn.execute(
            "SELECT images FROM daily_usage WHERE user_id = ? AND day = ?",
            (user_id, _today()),
        )
        row = await cursor.fetchone()
        return int(row["images"]) if row else 0


class UploadRepo:
    """Log of files that went through the bot, both ways (see UploadKind)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, user_id: int, kind: UploadKind, name: str | None, size_bytes: int) -> None:
        conn = self._db.conn
        await conn.execute(
            "INSERT INTO uploads (user_id, kind, name, size_bytes, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, kind.value, name, size_bytes, _now()),
        )
        await conn.commit()

    async def stats(self, user_id: int | None, since: datetime | None) -> dict[UploadKind, UploadStat]:
        """Count and bytes per kind since `since` (None = all time); user_id=None means everyone.

        Kinds with nothing uploaded are absent; unknown kinds (written by a newer version) are skipped.
        """
        where, params = _since_clause("created_at", since)
        cursor = await self._db.conn.execute(
            f"""
            SELECT kind, COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes FROM uploads
            WHERE {where} AND (? IS NULL OR user_id = ?)
            GROUP BY kind
            """,
            (*params, user_id, user_id),
        )
        known = {k.value: k for k in UploadKind}
        return {
            known[row["kind"]]: UploadStat(count=row["n"], bytes=row["bytes"])
            for row in await cursor.fetchall()
            if row["kind"] in known
        }


class StatusRepo:
    """The main bot's heartbeat, read by the admin bot."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def beat(self, started_at: datetime, queue_depth: int, active_jobs: int) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO bot_status (id, started_at, heartbeat_at, queue_depth, active_jobs)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                started_at = excluded.started_at, heartbeat_at = excluded.heartbeat_at,
                queue_depth = excluded.queue_depth, active_jobs = excluded.active_jobs
            """,
            (started_at.astimezone(UTC).isoformat(), _now(), queue_depth, active_jobs),
        )
        await conn.commit()

    async def get(self) -> BotStatus | None:
        """None until the main bot (a version with the heartbeat) has run once."""
        cursor = await self._db.conn.execute("SELECT * FROM bot_status WHERE id = 1")
        row = await cursor.fetchone()
        if row is None:
            return None
        return BotStatus(
            started_at=datetime.fromisoformat(row["started_at"]),
            heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]),
            queue_depth=row["queue_depth"],
            active_jobs=row["active_jobs"],
        )


class FlagRepo:
    """Named switches shared by both bots: the admin bot writes them, the main bot reads them."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, key: str) -> tuple[str, datetime] | None:
        """(value, when it was last set), or None if the flag was never set."""
        cursor = await self._db.conn.execute("SELECT value, updated_at FROM app_flags WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return None if row is None else (row["value"], datetime.fromisoformat(row["updated_at"]))

    async def set(self, key: str, value: str) -> None:
        conn = self._db.conn
        await conn.execute(
            """
            INSERT INTO app_flags (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, _now()),
        )
        await conn.commit()


class NoticeRepo:
    """Keys of admin alerts already sent."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def claim(self, key: str) -> bool:
        """True exactly once per key: the caller that gets True sends the alert."""
        conn = self._db.conn
        cursor = await conn.execute(
            "INSERT OR IGNORE INTO admin_notices (key, sent_at) VALUES (?, ?)", (key, _now())
        )
        await conn.commit()
        return cursor.rowcount == 1
