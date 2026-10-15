from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.client.telegram import PRODUCTION
from pydantic import ValidationError

from bot.config import Settings
from bot.files.extract import UnsupportedFile
from bot.telegram.api import download_limit, main_bot
from bot.telegram.download import download_bytes


def settings(base=None):
    return Settings(
        _env_file=None, telegram_bot_token="123456:TEST", buytokens_api_key="test", telegram_api_base=base
    )


async def test_cloud_default_and_local_absolute_file_download(tmp_path):
    cloud = main_bot(settings())
    local = main_bot(settings("http://127.0.0.1:8081/"))
    try:
        assert cloud.session.api == PRODUCTION
        assert download_limit(cloud) == 20 * 1024 * 1024
        assert download_limit(local) == 100 * 1024 * 1024
        assert local.session.api.is_local
        assert local.session.api.api_url(local.token, "getMe").startswith("http://127.0.0.1:8081/")
        path = tmp_path / "document.txt"
        path.write_bytes(b"local file")
        # aiogram must read the shared absolute path, not request /file/bot... over HTTP.
        local.get_file = AsyncMock(return_value=SimpleNamespace(file_path=str(path)))
        data = await download_bytes(local, SimpleNamespace(file_id="test", file_size=10), "document.txt")
        assert data == b"local file"
    finally:
        await cloud.session.close()
        await local.session.close()


@pytest.mark.parametrize(
    "base",
    ["https://example.com", "http://10.0.0.2:8081", "http://user:pass@127.0.0.1", "http://127.0.0.1/api"],
)
def test_reject_remote_or_credentialed_api(base):
    with pytest.raises(ValidationError):
        settings(base)


async def test_oversize_stops_before_download_and_missing_metadata_is_bounded(monkeypatch):
    monkeypatch.setattr("bot.telegram.download.download_limit", lambda bot: 8)

    async def chunks(file, destination):
        destination.write(b"123456")
        destination.write(b"789")

    bot = SimpleNamespace(download=AsyncMock(side_effect=chunks))
    with pytest.raises(UnsupportedFile):
        await download_bytes(bot, SimpleNamespace(file_size=9), "big.txt")
    bot.download.assert_not_called()
    with pytest.raises(UnsupportedFile):
        await download_bytes(bot, SimpleNamespace(file_size=None), "unknown.txt")
