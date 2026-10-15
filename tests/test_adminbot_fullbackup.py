from adminbot.callbacks import ServerBackup
from adminbot.fullbackup import is_busy, render_backup


def _buttons(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def test_idle_screen_offers_to_run():
    text, markup = render_backup(requested=False, service="inactive", watcher="active")
    assert "секреты" in text and "уже собирается" not in text
    assert ServerBackup(run=True).pack() in _buttons(markup)


def test_busy_hides_the_run_button():
    for requested, service in ((True, "inactive"), (False, "activating")):
        assert is_busy(requested, service)
        text, markup = render_backup(requested, service, watcher="active")
        assert "уже собирается" in text
        assert ServerBackup(run=True).pack() not in _buttons(markup)
    assert not is_busy(False, "failed")  # a failed run must not block the next one


def test_missing_watcher_says_deploy_is_needed():
    text, markup = render_backup(requested=False, service="inactive", watcher="inactive")
    assert "Нужен деплой" in text
    assert ServerBackup(run=True).pack() not in _buttons(markup)


# The private host's backup script and its service-contract test are not part of this export.
