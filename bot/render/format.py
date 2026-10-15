"""Small formatters for admin screens: user labels, big numbers, relative time."""

from __future__ import annotations

from datetime import datetime

from bot.domain import Role, User, UserUsage

ROLE_MARKS = {Role.ADMIN: " 👑", Role.PENDING: " ⏳", Role.BLOCKED: " 🚫"}


def user_label(user: User | UserUsage) -> str:
    """Username with a role mark ("@name 👑"), or the id. Not HTML-escaped: callers escape."""
    name = f"@{user.username}" if user.username else str(user.tg_id)
    return f"{name}{ROLE_MARKS.get(user.role, '')}"


def short_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def ago(moment: datetime | None, now: datetime) -> str:
    """Relative time: readable whatever time zone the server and the admin are in."""
    if moment is None:
        return "не пользовался"
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 60:
        return "только что"
    for unit_seconds, unit in ((86_400, "дн"), (3_600, "ч"), (60, "мин")):
        if seconds >= unit_seconds:
            return f"{seconds // unit_seconds} {unit} назад"
    return "только что"
