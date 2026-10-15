"""Async SQLite connection wrapper and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite

MIGRATIONS: list[str] = [
    # version 1: initial schema
    """
    CREATE TABLE users (
        tg_id INTEGER PRIMARY KEY,
        username TEXT,
        role TEXT NOT NULL,
        mode TEXT NOT NULL,
        model_override TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(tg_id),
        active INTEGER NOT NULL DEFAULT 1,
        started_at TEXT NOT NULL
    );

    CREATE UNIQUE INDEX idx_conversations_one_active_per_user
        ON conversations(user_id) WHERE active = 1;

    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        approx_tokens INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX idx_messages_conversation_id ON messages(conversation_id, id);

    CREATE TABLE requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(tg_id),
        conversation_id INTEGER REFERENCES conversations(id),
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        latency_ms INTEGER NOT NULL,
        approx_tokens INTEGER NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX idx_requests_created_at ON requests(created_at);

    CREATE TABLE daily_usage (
        user_id INTEGER NOT NULL REFERENCES users(tg_id),
        day TEXT NOT NULL,
        requests INTEGER NOT NULL DEFAULT 0,
        approx_tokens INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, day)
    );
    """,
    # version 2: where "clear chat" starts deleting (first message id of the current chat session)
    """
    ALTER TABLE users ADD COLUMN chat_floor_id INTEGER;
    """,
    # version 3: personal answer settings
    """
    ALTER TABLE users ADD COLUMN persona TEXT;
    ALTER TABLE users ADD COLUMN style TEXT NOT NULL DEFAULT 'normal';
    """,
    # version 4: enabled skills, comma-separated ids
    """
    ALTER TABLE users ADD COLUMN skills TEXT NOT NULL DEFAULT '';
    """,
    # version 5: admin bot - personal context limit, tokens by kind, uploads log
    """
    ALTER TABLE users ADD COLUMN context_limit INTEGER;
    ALTER TABLE users ADD COLUMN daily_limit INTEGER;

    ALTER TABLE requests ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN image_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN service_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN context_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN tool_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE requests ADD COLUMN web_calls INTEGER NOT NULL DEFAULT 0;
    CREATE INDEX idx_requests_user ON requests(user_id, created_at);

    CREATE TABLE uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(tg_id),
        kind TEXT NOT NULL,
        name TEXT,
        size_bytes INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_uploads_user ON uploads(user_id, created_at);
    CREATE INDEX idx_conversations_user ON conversations(user_id, id);

    -- The main bot's heartbeat: one row, rewritten every HEARTBEAT_S.
    CREATE TABLE bot_status (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        started_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL,
        queue_depth INTEGER NOT NULL DEFAULT 0,
        active_jobs INTEGER NOT NULL DEFAULT 0
    );

    -- Admin alerts already sent, so a restart or a second process never repeats one.
    CREATE TABLE admin_notices (
        key TEXT PRIMARY KEY,
        sent_at TEXT NOT NULL
    );
    """,
    # version 6: switches the admin bot flips and the main bot reads (e.g. the reserve gateway)
    """
    CREATE TABLE app_flags (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    # version 7: personal reserve-model pick, kept apart from `model_override` for when the switch
    # goes back off (NULL = the reserve's default model)
    """
    ALTER TABLE users ADD COLUMN reserve_model TEXT;
    """,
    # version 8: cap personal daily limits at the new global limit (NULL = the global one, untouched)
    """
    UPDATE users SET daily_limit = 100 WHERE daily_limit IS NOT NULL AND daily_limit > 100;
    """,
    # version 9: who draws the user's pictures (bot.domain.ImageModel)
    """
    ALTER TABLE users ADD COLUMN image_model TEXT NOT NULL DEFAULT 'gemini';
    """,
    # version 10: personal hours of silence before the dialog resets (NULL = the global CONTEXT_IDLE_HOURS)
    """
    ALTER TABLE users ADD COLUMN context_idle_hours REAL;
    """,
    # version 11: pictures drawn per user per day (the daily picture limit)
    """
    ALTER TABLE daily_usage ADD COLUMN images INTEGER NOT NULL DEFAULT 0;
    """,
    # version 12: chats as Telegram topics. `thread_id` is the topic of a private chat (0 = the main
    # chat, "Все"), so a user has one active conversation per topic instead of one in total;
    # `title` is the topic's name (NULL = not named yet), `model_override` the model of this chat.
    """
    ALTER TABLE conversations ADD COLUMN thread_id INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE conversations ADD COLUMN title TEXT;
    ALTER TABLE conversations ADD COLUMN model_override TEXT;
    DROP INDEX idx_conversations_one_active_per_user;
    CREATE UNIQUE INDEX idx_conversations_one_active_per_thread
        ON conversations(user_id, thread_id) WHERE active = 1;
    """,
]


class Database:
    """Owns a single aiosqlite connection and applies migrations on connect."""

    def __init__(self, path: Path | str) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the connection, set pragmas and apply pending migrations."""
        if isinstance(self._path, Path):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        elif isinstance(self._path, str) and self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(self._path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        if self._path != ":memory:":
            await conn.execute("PRAGMA journal_mode = WAL")
        self._conn = conn
        await self._migrate()

    async def _version(self) -> int:
        cursor = await self.conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _migrate(self) -> None:
        conn = self.conn
        current_version = await self._version()
        for index, script in enumerate(MIGRATIONS, start=1):
            if index <= current_version:
                continue
            # One script = one transaction, so a crash never leaves a half-applied version.
            try:
                await conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {index};\nCOMMIT;")
            except sqlite3.OperationalError:
                # The main bot and the admin bot share the file and may start together: if the other
                # process applied this version first, our copy fails ("duplicate column"), which is fine.
                await conn.rollback()
                if await self._version() < index:
                    raise

    async def snapshot(self, path: Path) -> None:
        """A consistent copy of the whole database in a new file (it must not exist yet)."""
        await self.conn.execute("VACUUM INTO ?", (str(path),))

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn
