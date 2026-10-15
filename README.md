# LC Claude

A noncommercial, self-hosted Telegram assistant maintained by Gleb Gunderin.
Written in Python with aiogram, using an OpenAI-compatible model gateway.
The interface and many comments are currently in Russian.

This is an early public source release of a privately developed application.
It is not affiliated with Anthropic or OpenAI. Model names in the included
catalog describe the original gateway configuration; availability and access
depend on the provider you configure.

## Included

- Telegram chat interface with separate conversations and per-chat model selection.
- Context management, validation, retries and fallback handling for model requests.
- Document extraction, voice-transcription and optional search/image integrations.
- Telegram-safe response formatting and SQLite persistence.
- A separate administrator bot with access controls and usage summaries.
- Tests for the exported application code.

This export contains the main `bot/` and `adminbot/` applications. It excludes
the private deployment history, personal twin-bot application, server-management
agent, full-server backup scripts, production data and promotional media.
Some administrator controls are integration points for those separate services:
twin-bot controls, full-server backup requests and host-specific monitoring are
not standalone features of this release. Optional missing welcome media falls
back to a text greeting.

## Local setup

Use Python 3.12 or newer. Create and activate a virtual environment, then install:

```sh
python -m pip install -r requirements.txt
```

Copy `.env.example` to a local `.env` file and edit it before starting the bot:

- Set `TELEGRAM_BOT_TOKEN` to the token for a bot you control.
- Set `BUYTOKENS_API_KEY` and `BUYTOKENS_BASE_URL` for your compatible provider.
  Replace the non-routable example URL before starting the bot.
  These variable names are retained from the original deployment. Compatibility
  does not imply that the application already uses the OpenAI API directly.
- Review `bot/models.yaml`: provider model identifiers, limits and pricing must
  match your provider. The included catalog is a starting configuration, not a
  guarantee that every named model is available.
- Set `ADMIN_IDS` to the numeric Telegram IDs authorized to administer your bot.
- Keep `ACCESS_MODE=whitelist` while configuring access.
- Use a writable `DB_PATH`, normally `data/bot.db`.

Start the assistant from the repository root:

```sh
python -m bot
```

To use the separate administrator interface, set `ADMIN_BOT_TOKEN` to a second
bot's token and run `python -m adminbot` in another process using the same
configuration and database.

Optional integrations need their own credentials and dependencies. Voice
transcription can download a model; image generation and search require
configured providers. Presentation rendering and OCR may need additional
system packages and fonts. Provision these for your environment rather than
assuming the original private server configuration is present.

## Development

```sh
python -m pip install pytest pytest-asyncio ruff
python -m pytest -q
```

Tests for excluded private applications and deployment scripts are not included.
Keep test databases and credentials separate from a running deployment. Review
changes before applying them to a production instance.

## Maintenance and contributions

Gleb Gunderin owns and maintains the project. Bug reports should include a small
reproduction, expected behavior and version details. Never attach bot tokens,
API keys, private conversations or production database files. Small focused
changes with regression tests are welcome.

The private development history began on September 23, 2026. This public export
starts separately; it does not represent the private commits as a pre-existing
public contribution history. Broad external adoption has not been established.

## License

MIT; see [LICENSE](LICENSE). Third-party dependencies retain their own licenses.
