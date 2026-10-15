"""Bound attachment memory, including files with missing size metadata."""

import io
from typing import Any

from aiogram import Bot

from bot.files.extract import UnsupportedFile
from bot.telegram.api import download_limit


def too_big(bot: Bot, name: str) -> str:
    limit = download_limit(bot) // (1024 * 1024)
    if limit == 20:
        return f"Файл «{name}» больше 20 МБ — Telegram не даёт ботам скачивать такие файлы."
    return f"Файл «{name}» больше {limit} МБ — это предел обработки файлов этим ботом."


class _BoundedBuffer(io.BytesIO):
    def __init__(self, limit: int, error: str) -> None:
        super().__init__()
        self.limit, self.error = limit, error

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self.limit:
            raise UnsupportedFile(self.error)
        return super().write(data)


async def download_bytes(bot: Bot, file: Any, name: str) -> bytes:
    limit, error = download_limit(bot), too_big(bot, name)
    if (file.file_size or 0) > limit:
        raise UnsupportedFile(error)
    with _BoundedBuffer(limit, error) as destination:
        await bot.download(file, destination=destination)
        return destination.getvalue()
