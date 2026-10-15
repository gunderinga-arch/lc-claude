from adminbot.balance import info_url, render_balance, render_key
from adminbot.callbacks import Balance, Home

INFO = {
    "active": True,
    "quota": {"total": 1_152_500_000, "used": 552_262_121.5, "remaining": 600_237_878.5},
    "usage": {"today_tokens": 32_093_498, "weekly_tokens": 135_942_777, "month_tokens": 193_746_344},
    "media": {"photo": {"quota": 15, "used": 8, "remaining": 7}},
}


def test_info_url_drops_the_api_suffix():
    assert info_url("https://api.example.invalid/v1") == "https://api.example.invalid/public/key-info"
    assert info_url("http://192.0.2.10:4100/v1/") == "http://192.0.2.10:4100/public/key-info"


def test_key_block_shows_remaining_spend_and_runway():
    text = "\n".join(render_key("Основной — боты", INFO))
    assert "Осталось: <b>600.2M</b> из 1152.5M (52%)" in text
    assert "сегодня 32.1M · 7 дней 135.9M" in text
    assert "Хватит примерно на 31 дн." in text  # 600.2M / (135.9M / 7)
    assert "Фото: 7 из 15" in text and "неактивен" not in text


def test_remaining_is_derived_when_the_site_omits_it():
    info = {"active": False, "quota": {"total": 50_000_000, "used": 7_646_968.5}, "usage": {}}
    text = "\n".join(render_key("Второй — тесты", info))
    assert "неактивен" in text and "Осталось: <b>42.4M</b> из 50.0M (85%)" in text
    assert "Хватит" not in text  # no spend this week, no runway estimate


def test_failed_key_and_buttons():
    text, markup = render_balance([("Основной — боты", INFO), ("Второй — тесты", "сайт не ответил")])
    assert "Не удалось узнать: сайт не ответил" in text and "600.2M" in text
    buttons = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert buttons == [Balance().pack(), Home().pack()]
