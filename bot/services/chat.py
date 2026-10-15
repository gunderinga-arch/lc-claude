"""The chat pipeline: context -> resilient LLM call -> history, accounting, rendering."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from bot.catalog import Catalog, ModelSpec
from bot.domain import (
    IMAGE_TOKENS,
    ImageModel,
    Mode,
    RequestRecord,
    RequestStatus,
    TokenUsage,
    User,
    approx_tokens,
)
from bot.files.blocks import OutFile, embed_images, split_files
from bot.files.deck_qa import render_sheets, review_prompt
from bot.files.ocr import (
    OCR_MAX_PAGES,
    OCR_PROMPT,
    batches,
    can_read_scans,
    page_count,
    render_pages,
)
from bot.files.slides import DeckError, build_deck, deck_guide, render_decks
from bot.files.store import FileStore
from bot.llm.base import ChatMessage, ChatResult, ChatStatus, LLMClient, ToolCall
from bot.llm.reserve import ReserveSwitch
from bot.llm.window import WINDOW_SLACK
from bot.render.telegram_html import TELEGRAM_LIMIT, render_markdown
from bot.services.conversation import COMPACT_AT, COMPACT_AT_SHARE, COMPACT_KEEP_SHARE, ConversationService
from bot.services.filetools import FILE_TOOL_NAMES, FILE_TOOLS, FileTools, file_guide, fit_files
from bot.services.images import (
    CUTOUT_SUFFIX,
    IMAGE_TOOLS,
    EditRequest,
    ImageError,
    ImageGenerator,
    cut_out_background,
    edit_tool,
    image_ext,
    image_size,
    parse_edit_call,
    parse_image_call,
)
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.services.web import WEB_GUIDE, WEB_TOOLS, WebTools
from bot.storage.repositories import RequestRepo

logger = logging.getLogger(__name__)

# Room left in the first chunk for the header the Telegram layer puts above the answer
# (model name, fallback and context-reset notes).
HEADER_RESERVE = 300

# Messages the prompt rewriter looks at: "спасибо" is not worth a round trip, a long paste is not safe.
REWRITE_MIN_CHARS = 10
REWRITE_MAX_CHARS = 4_000

# Internet tools: each round is one more full model call, and the models here think for a long time.
# 5 since 15.10.2026 (was 3): with ~1M windows every round's pages fit, a deep question may dig further.
MAX_TOOL_ROUNDS = 5
# A chat with big stored files: reading them part by part takes more rounds.
MAX_FILE_TOOL_ROUNDS = 8
# Share of the dialog's budget the files of one message may fill whole; a bigger file goes in cut, in full
# to the file store (read_file / search_file).
FILE_SHARE = 0.8
MAX_WEB_CALLS_PER_ROUND = 4  # each can read up to 100 pages, and its text is resent every later round
# Asking to search turns the internet tools on for that one message, even without the web skill.
WEB_ASK = re.compile(
    r"найд|найти|поищ|поиск|загугл|погугл|гугл|интернет|\bв сети\b|браузер|посмотри|https?://", re.I
)
# Deep research: the bot searches several queries and reads pages itself before the model answers.
DEEP_ASK = re.compile(
    r"исследуй|(?:проведи|сделай|нужно) исследовани|глубок\w* (?:поиск|исследовани)|deep research"
    r"|подробно (?:найди|поищи|изучи|разбер)|(?:изучи|разберись) подробно|найди вс[её] (?:про|о|об)\b"
    r"|собери (?:всю |побольше )?информаци",
    re.I,
)
RESEARCH_QUERIES = 6
RESEARCH_PLANNER = (
    "Ты помогаешь провести исследование в интернете. По вопросу пользователя составь от 4 до "
    f"{RESEARCH_QUERIES} поисковых запросов, которые вместе покроют тему с разных сторон: суть и "
    "определения, свежие данные и новости, цифры и сравнения, мнения экспертов и критику, первоисточники. "
    "Запросы короткие, как их вводят в поисковик. Если тема не только про Россию, часть запросов "
    "пиши по-английски. Ответь только запросами, по одному в строке, без нумерации и пояснений."
)
RESEARCH_BRIEF = (
    "<research>\nЭто глубокое исследование. Бот заранее искал в интернете по запросам: {queries}.\n"
    "Ниже самые подходящие фрагменты прочитанных страниц. Опирайся на них; если чего-то не хватает, "
    "доищи инструментами. Дай развёрнутый структурированный ответ высокого уровня: сверяй источники, "
    "отмечай расхождения и степень уверенности, в конце — список источников ссылками.\n\n"
    "{material}\n</research>"
)
# Room kept in the window besides the dialog: the deck guide, the web guide and a round of tool results.
# More than that still fits: the LLM client then leaves the earliest dialog out of that one request.
EXTRA_ROOM_TOKENS = 40_000
# A pipeline "model" (models.yaml `pipeline:`): the same model plans the answer and picks skills first.
PLAN_CONTEXT_TOKENS = 6_000  # the planner sees the recent dialog only
PLAN_TURN_CHARS = 3_000
PLAN_MAX_SKILLS = 3
PLANNER = (
    "Ты планировщик ответа в Telegram-боте. Ниже диалог, последнее сообщение — текущая задача "
    "пользователя. Не решай её и не пиши ответ пользователю. Составь для исполнителя (сильной модели) "
    "план, по которому он выполнит задачу наилучшим образом:\n"
    "— что на самом деле нужно пользователю и как выглядит отличный результат;\n"
    "— шаги выполнения по порядку; для сложной многоэтапной задачи подробно, с тем, что проверить "
    "после каждого шага;\n"
    "— подводные камни и типичные ошибки; что допустить, если данных не хватает;\n"
    "— формат итогового ответа: структура, объём, нужны ли файлы, код, источники.\n"
    "Для простого вопроса хватит 2–4 строк.\n"
    f"Ещё выбери до {PLAN_MAX_SKILLS} скиллов, которые помогут исполнителю, или ни одного:\n{{skills}}\n"
    "Ответ строго в таком виде:\nSKILLS: id1, id2 (или SKILLS: -)\nPLAN:\n<план>"
)
PLAN_BRIEF = (
    "<plan>\nПлан ответа, составленный заранее (пользователь его не видит, не упоминай его). "
    "Выполни задачу полностью, по этому плану; если план где-то ошибается, отступи от него.\n"
    "{plan}\n</plan>"
)
SUMMARIZER = (
    "Сожми переписку пользователя с ассистентом в конспект, по которому ассистент продолжит разговор, "
    "как будто помнит его целиком. Сохрани: цели и задачи пользователя, что уже сделано и что осталось, "
    "принятые решения и договорённости (стиль, формат, ограничения), важные факты, цифры, имена, ссылки, "
    "названия файлов; ключевые фрагменты кода и текста — дословно. Пропусти приветствия, повторы и "
    "отработанные черновики. Пиши на языке переписки, только конспект, без вступления, до 5000 слов."
)
TITLER = (
    "Придумай короткое название чата по первому сообщению пользователя: 2–4 слова, не больше 40 символов, "
    "на языке сообщения. Только название: без кавычек, точки, эмодзи и пояснений."
)
TITLE_MAX_CHARS = 40  # Telegram caps a topic name at 128; a tab in the topic bar shows far fewer
# Names new chats: ~1.5 s and ~0.002 ₽ a title via RouterAI, ~20 s on any BuyTokens model (14.10.2026).
TITLE_MODEL = "rai/anthropic/claude-haiku-5.5"

# A deck block in an answer, and a request that is about a deck (or its edit: "переделай её в этом стиле"
# has no keyword, but the chat's last decks are in the history).
_DECK_BLOCK = re.compile(
    r'^<<<FILE name="([^"\n]{1,200}\.pptx)">>>\n(.*?)\n?<<<END>>>[ \t]*$', re.S | re.M | re.I
)
_DECK_ASK = re.compile(r"презентаци|слайд|pptx|power\s?point|пауэрпоинт", re.I)

# BuyTokens hands pictures to Opus 5 and GPT-6 Sol through a separate vision service; when it is down the
# model gets "[Image could not be recognized: the vision service did not respond]" instead and says so in
# its own words. Haiku 5.5 (reserve) and Fable 5.1 receive pictures directly (checked 14.10.2026): one of them
# describes the pictures in words and the user's model does the work from that — asking Fable to redo a
# whole heavy answer (a deck) ran past the gateway's 10-minute cut.
VISION_FALLBACK = "claude-fable-5.1"
DESCRIBE_PROMPT = (
    "Другая модель не видит эти картинки. Опиши их для неё так, чтобы она выполнила задачу пользователя, "
    "не видя их: весь текст на картинках дословно; объекты и где они стоят; цвета (HEX); шрифты "
    "(гротеск или антиква, ширина, жирность, капс, разрядка); композиция, сетка, поля, пропорции заголовков "
    "к тексту; фактуры и "
    "приёмы; заметные дефекты. Только описание, без советов и вступления.\n\nЗадача пользователя: {task}"
)
SEEN_NOTE = "[Картинки к этому сообщению до тебя не дошли; их описала другая модель, которая их видит:]"

MAX_JSON_MENDS = 3
JSON_MEND_PROMPT = (
    "В JSON презентации синтаксическая ошибка: {error} (строка {line}, позиция {col}). "
    "Вот строки вокруг неё, слева номера:\n{lines}\n\n"
    "Верни только эти строки, исправленные, без номеров, без пояснений и без ```, в том же порядке. "
    "Меняй только то, что ломает JSON; содержимое не трогай."
)
_LINE_NO = re.compile(r"^\s*\d+:\s?")  # a model may keep the numbers it was shown
_BLIND = re.compile(
    r"vision service|could not be recognized|(служб|сервис)\w*\s+распознавани\w*"
    r"|(изображени|картинк|фото)\w*[^.\n]{0,40}не\s+(дош|получ|виж)",
    re.I,
)


def _is_blind(result: ChatResult) -> bool:
    """The model answered that the pictures never reached it."""
    return result.status is ChatStatus.OK and bool(_BLIND.search(result.text))


def _is_deck_talk(text: str, messages: list[ChatMessage]) -> bool:
    return bool(_DECK_ASK.search(request_words(text))) or any(
        m.role == "assistant" and _DECK_BLOCK.search(m.content) for m in messages[-6:]
    )


def request_words(text: str) -> str:
    """Routing keywords come from the request, never from an attached book's contents."""
    return text.partition("\n\n<file")[0]


def _with_deck_guide(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The deck guide (format, design rules, the font catalog) appended to the system message."""
    if messages and messages[0].role == "system":
        return [replace(messages[0], content=f"{messages[0].content}\n\n{deck_guide()}"), *messages[1:]]
    return [ChatMessage(role="system", content=deck_guide()), *messages]


TOOLS_EXHAUSTED = (
    "Лимит вызовов инструментов исчерпан. Больше их не вызывай, заверши ответ с тем, что уже есть."
)
WEB_TOOL_NAMES = frozenset(t["function"]["name"] for t in WEB_TOOLS)
MAX_IMAGES_PER_ANSWER = 6  # ponytail: ~620 of the 10,000 free Cloudflare neurons a day
DESIGNER_IMAGES_PER_ANSWER = 3  # a site or a deck must not eat the whole daily picture limit at once
RECENT_IMAGES = (
    12  # ponytail: kept in memory only, a restart forgets them (old HTML then shows broken images)
)

_STATUS = {
    ChatStatus.OK: RequestStatus.OK,
    ChatStatus.STUB: RequestStatus.STUB,
    ChatStatus.FAILED: RequestStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class ChatReply:
    status: RequestStatus
    parts: list[str]  # Telegram-HTML chunks of the answer text (file blocks cut out), only for OK
    text: str  # raw markdown answer, for a plain-text resend if Telegram rejects the HTML
    model: ModelSpec  # the model that answered, or the last one tried
    fallback_used: bool
    error: str | None = None  # ChatResult.error for FAILED
    files: list[OutFile] = field(default_factory=list)  # files the model created, sent as documents
    images: list[OutFile] = field(default_factory=list)  # images the model generated, sent as photos
    context_reset: bool = False  # the history was dropped because the gateway's filter blocked it
    served_by: str | None = None  # the reserve gateway's model while the emergency switch is on
    gemini_images: frozenset[str] = frozenset()  # names in `images` drawn by Gemini, the rest by FLUX


@dataclass(slots=True)
class _Draw:
    """The image tool within one answer: the user's pick, and the files Gemini actually drew."""

    model: ImageModel
    cap: int = MAX_IMAGES_PER_ANSWER  # pictures this answer may still draw (per answer and per day)
    daily_left: int | None = None  # what the daily limit allows; None = unlimited
    by_gemini: set[str] = field(default_factory=set)
    generate: bool = True  # the image skill is on: the model may draw new pictures
    sources: tuple[bytes, ...] = ()  # the pictures the user sent with this message
    recent: dict[str, bytes] = field(default_factory=dict)  # pictures drawn earlier in this chat, by name


@dataclass(frozen=True, slots=True)
class _CompleteOutcome:
    """One call to `_complete`: the model result plus everything it cost."""

    result: ChatResult
    usage: TokenUsage
    attempts: int
    latency_ms: int
    web_calls: int


class ChatService:
    def __init__(
        self,
        llm: LLMClient,
        conversations: ConversationService,
        quota: QuotaService,
        router: ModelRouter,
        requests: RequestRepo,
        catalog: Catalog,
        web: WebTools | None = None,
        images: ImageGenerator | None = None,
        *,
        reserve: ReserveSwitch | None = None,  # only when a reserve gateway is configured
        reserve_token_budget: int = 100_000,
        files: FileStore | None = None,  # full texts of big files, for the read_file / search_file tools
    ) -> None:
        self._files = files
        self._reserve = reserve
        self._reserve_token_budget = reserve_token_budget
        self._web = web
        self._images = images
        self._recent_images: dict[tuple[int, int], dict[str, bytes]] = {}  # (user, topic) -> latest images
        self._llm = llm
        self._conversations = conversations
        self._quota = quota
        self._router = router
        self._requests = requests
        self._catalog = catalog
        self._bulk_gate = asyncio.Semaphore(2)  # shared by OCR and all file digests, across users

    def model_for(self, user: User, reserve: bool = False) -> ModelSpec:
        return self._router.resolve(user, reserve=reserve)

    async def _polish_decks(
        self,
        answer: str,
        messages: list[ChatMessage],
        refs: tuple[str, ...],
        model_id: str,
        pictures: dict[str, bytes],
    ) -> str:
        """Each deck in the answer: build it, show the model the rendered slides, take its corrected deck.

        Any failure keeps the deck as the model first wrote it: a review may improve an answer, never lose it.
        """
        for block in list(_DECK_BLOCK.finditer(answer)):
            name, raw = block.group(1), block.group(2)
            try:
                built = build_deck(raw, pictures)
            except DeckError:
                # Broken JSON (a typo, a bad join after a cut): mend the lines around each error only.
                fixed = await self._mend_json(raw, model_id)
                if fixed is None:
                    continue  # render_decks sends it as .json, as before
                answer, raw = answer.replace(raw, fixed, 1), fixed
                try:
                    built = build_deck(raw, pictures)
                except Exception:
                    continue
            except Exception:  # a broken deck is reported by render_decks, not here
                continue
            sheets = await render_sheets(built.pptx)
            if not sheets and not built.warnings:
                continue
            shots = tuple("data:image/png;base64," + base64.b64encode(s).decode() for s in sheets)
            ask = review_prompt(name, built.warnings, bool(refs))
            review = [
                *messages,
                ChatMessage(role="assistant", content=f'<<<FILE name="{name}">>>\n{raw}\n<<<END>>>'),
                ChatMessage(role="user", content=ask, images=shots + refs),
            ]
            result = await self._llm.chat(review, model_id)
            if _is_blind(result) and (seen := await self._describe(shots + refs, ask)):
                # The slides as words, by a model that sees them; the deck's own model does the fixing.
                review[-1] = ChatMessage(role="user", content=f"{ask}\n\n{SEEN_NOTE}\n{seen}")
                result = await self._llm.chat(review, model_id)
            if result.status is not ChatStatus.OK or result.text.strip().upper().startswith("OK"):
                logger.info("deck review: %s kept (%s)", name, result.status.value)
                continue
            fixed = next((f for f in split_files(result.text)[1] if f.name.lower().endswith(".pptx")), None)
            if fixed is None:
                continue
            new_raw = fixed.data.decode("utf-8")
            try:
                build_deck(new_raw, pictures)
            except Exception:
                logger.warning("deck review returned an unusable deck for %s", name, exc_info=True)
                continue
            logger.info(
                "deck review: %s corrected (%d sheets, %d warnings)", name, len(sheets), len(built.warnings)
            )
            answer = answer.replace(raw, new_raw, 1)
        return answer

    async def _mend_json(self, raw: str, model_id: str) -> str | None:
        """Fix a deck's JSON syntax a few lines at a time: the model sees only the lines around each error.

        Cheap even for a 40k-char deck (it is not re-sent whole); Haiku on the reserve does it when it can.
        Returns the repaired JSON, or None if it still does not parse after MAX_JSON_MENDS rounds.
        """
        lines = raw.strip().split("\n")
        if lines and lines[0].startswith("```"):  # a fence inside the block, as parse_deck allows
            lines = [ln for ln in lines if not ln.startswith("```")]
        menders = [m for m in (TITLE_MODEL if self._reserve is not None else None, model_id) if m]
        for _ in range(MAX_JSON_MENDS):
            text = "\n".join(lines)
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError as exc:
                error = exc
            lo, hi = max(0, error.lineno - 3), min(len(lines), error.lineno + 2)
            numbered = "\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi))
            ask = JSON_MEND_PROMPT.format(error=error.msg, line=error.lineno, col=error.colno, lines=numbered)
            mended: list[str] | None = None
            for model in menders:
                if not self._catalog.is_available(model):
                    continue
                result = await self._llm.chat([ChatMessage(role="user", content=ask)], model)
                if result.status is ChatStatus.OK and result.text.strip():
                    body = [ln for ln in result.text.strip().split("\n") if not ln.startswith("```")]
                    mended = [_LINE_NO.sub("", ln, count=1) for ln in body]
                    break
            if not mended:
                return None
            logger.info("deck JSON mended at line %d (%s)", error.lineno, error.msg)
            lines = [*lines[:lo], *mended, *lines[hi:]]
        try:
            json.loads("\n".join(lines))
        except json.JSONDecodeError:
            return None
        return "\n".join(lines)

    async def _describe(self, images: tuple[str, ...], task: str) -> str | None:
        """The pictures in words, by a model that receives them, for a model the gateway keeps them from."""
        describers = [m for m in (TITLE_MODEL if self._reserve is not None else None, VISION_FALLBACK) if m]
        ask = [ChatMessage(role="user", content=DESCRIBE_PROMPT.format(task=task[:2000]), images=images)]
        for model in describers:
            if not self._catalog.is_available(model):
                continue
            result = await self._llm.chat(ask, model)
            if result.status is ChatStatus.OK and result.text.strip() and not _is_blind(result):
                logger.info("pictures described by %s (%d chars)", model, len(result.text))
                return result.text
        return None

    def can_read_scan(self) -> bool:
        return can_read_scans() and self._catalog.is_available(VISION_FALLBACK)

    async def read_scan(self, data: bytes, name: str) -> str | None:
        """A scanned PDF's text, read from its pages as pictures by Fable 5.1; None if it cannot be read.

        Pages go OCR_PAGES_PER_CALL at a time through the shared bulk-work gate, up to OCR_MAX_PAGES.
        ponytail: the vision calls are not counted against the user's daily quota (one scan, rarely).
        """
        total = await asyncio.to_thread(page_count, data)
        if not total or not self.can_read_scan():
            return None
        tmp = tempfile.TemporaryDirectory(prefix="scan-")
        pdf = Path(tmp.name) / "scan.pdf"
        await asyncio.to_thread(pdf.write_bytes, data)

        async def one(first: int, last: int) -> str:
            async with self._bulk_gate:
                shots = await render_pages(pdf, first, last)
                if not shots:
                    return f"--- страницы {first}–{last}: не удалось отрисовать ---"
                uris = tuple("data:image/jpeg;base64," + base64.b64encode(s).decode() for s in shots)
                ask = OCR_PROMPT.format(first=first, last=last, name=name, total=total)
                result = await self._llm.chat(
                    [ChatMessage(role="user", content=ask, images=uris)], VISION_FALLBACK
                )
            vision_failed = re.search(r"vision service|could not be recognized", result.text, re.I)
            if result.status is ChatStatus.OK and result.text.strip() and not vision_failed:
                return result.text.strip()
            return f"--- страницы {first}–{last}: распознать не удалось ---"

        try:
            parts = await asyncio.gather(*(one(first, last) for first, last in batches(total)))
        finally:
            tmp.cleanup()
        if all("распознать не удалось" in p or "не удалось отрисовать" in p for p in parts):
            return None
        read = min(total, OCR_MAX_PAGES)
        tail = "" if read == total else f"\n\n[Распознаны страницы 1–{read} из {total}: дальше не читалось.]"
        logger.info("scan %s read: %d of %d pages", name, read, total)
        return "\n\n".join(parts) + tail

    def can_draw(self, user: User) -> bool:
        """The model may draw pictures for this user (a generator and an image skill such as 🎨 designer)."""
        return self._images is not None and self._conversations.wants_images(user.skills)

    def is_research(self, text: str) -> bool:
        """The message asks for deep research (and there is internet to do it with)."""
        return self._web is not None and DEEP_ASK.search(request_words(text)) is not None

    async def answer(
        self, user: User, text: str, images: tuple[str, ...] = (), thread_id: int = 0
    ) -> ChatReply:
        """Answer one user message in the chat of Telegram topic `thread_id` (0 = the main chat).

        Access/quota checks are the caller's job (done before queueing).
        """
        in_reserve = self._reserve is not None and await self._reserve.active()
        spec = self._router.resolve(user, reserve=in_reserve)
        token_budget = user.context_limit
        if in_reserve:
            # Free models hold far less context, and their rate limits are too tight for the extra
            # rewrite call: the reserve gets the user's own text and a shorter history.
            prompt, service_tokens, latency_ms = text, 0, 0
            token_budget = min(token_budget or self._reserve_token_budget, self._reserve_token_budget)
        else:
            # The model gets the rewritten prompt; the history keeps what the user actually wrote.
            prompt, service_tokens, latency_ms = await self._rewrite(user, text)
        usage = TokenUsage(service=service_tokens)
        skills = user.skills
        planner_model_id = spec.pipeline
        model_id = spec.answer or planner_model_id or spec.id
        if spec.pipeline is not None:
            plan, picked, service, ms = await self._plan(user.tg_id, prompt, planner_model_id, thread_id)
            skills = tuple(dict.fromkeys((*skills, *picked)))
            prompt = f"{prompt}\n\n{PLAN_BRIEF.format(plan=plan)}" if plan else prompt
            usage += TokenUsage(service=service)
            latency_ms += ms
        # The dialog fills the answering model's window (~1M tokens); the reserve keeps its smaller budget.
        window = None if in_reserve else self._history_window(model_id)
        context = dict(
            images=images,
            persona=user.persona,
            style=user.style,
            skills=skills,
            token_budget=token_budget,
            idle_hours=user.context_idle_hours,
            thread_id=thread_id,
            window=window,
        )
        files = None
        keep = {}
        if self._files is not None:
            # A file too big for the window goes in cut (its start and a note); the whole text waits in the
            # store for the file tools, and the history keeps the cut message, not megabytes.
            budget = int((window or self._conversations.history_budget(token_budget)) * FILE_SHARE)
            fitted, keep = await asyncio.to_thread(fit_files, text, budget)
            if fitted != text:
                prompt = (
                    fitted + prompt[len(text) :]
                    if prompt.startswith(text)
                    else (await asyncio.to_thread(fit_files, prompt, budget))[0]
                )
                text = fitted
        web = self._web is not None and (
            self._conversations.wants_web(skills) or WEB_ASK.search(request_words(text)) is not None
        )
        web_calls = 0
        if self.is_research(text) and not in_reserve:  # the reserve's free models hold too little context
            brief, service, web_calls, ms = await self._research(request_words(text))
            web = True
            prompt = f"{prompt}\n\n{brief}" if brief else prompt
            usage += TokenUsage(service=service)
            latency_ms += ms
        wants_images = self._images is not None and self._conversations.wants_images(skills)
        chat_key = (user.tg_id, thread_id)
        recent = dict(self._recent_images.get(chat_key, {}))
        draw = None
        # A sent photo or an earlier picture of the chat can be edited with or without the image skill.
        if self._images is not None and (wants_images or images or recent):
            per_answer = DESIGNER_IMAGES_PER_ANSWER if "designer" in skills else MAX_IMAGES_PER_ANSWER
            left = await self._quota.images_left(user)
            draw = _Draw(
                user.image_model,
                per_answer if left is None else min(per_answer, left),
                left,
                generate=wants_images,
                sources=tuple(_data_uri_bytes(i) for i in images),
                recent=recent,
            )
        generated: dict[str, bytes] = {}
        conversation_id, messages = await self._conversations.build_context(user.tg_id, prompt, **context)
        # build_context may clear an expired chat's files; save this turn's attachments afterwards.
        if self._files is not None:
            for name, body in keep.items():
                await asyncio.to_thread(self._files.save, user.tg_id, thread_id, name, body)
            if await asyncio.to_thread(self._files.names, user.tg_id, thread_id):
                files = FileTools(
                    self._files,
                    user.tg_id,
                    thread_id,
                    None if in_reserve else self._llm,
                    self._catalog.for_mode(Mode.NORMAL).id,
                    bulk_gate=self._bulk_gate,
                )
        if _is_deck_talk(text, messages):  # the full deck guide only where a deck is made or edited
            messages = _with_deck_guide(messages)
        outcome = await self._complete(messages, model_id, images, web, draw, generated, files)
        usage += outcome.usage
        attempts, latency_ms = outcome.attempts, latency_ms + outcome.latency_ms
        web_calls, result = web_calls + outcome.web_calls, outcome.result

        context_reset = False
        if result.status is ChatStatus.STUB and len(messages) > 2:
            # The gateway's topic filter reads the whole history: once a political answer is in it,
            # every later question gets the canned refusal. Ask again without history; if that
            # passes, the history was the problem and it is dropped for good.
            _, fresh = await self._conversations.build_context(
                user.tg_id, prompt, with_history=False, **context
            )
            retry = await self._complete(fresh, model_id, images, web, draw, generated, files)
            usage += retry.usage
            attempts, latency_ms = attempts + retry.attempts, latency_ms + retry.latency_ms
            web_calls += retry.web_calls
            result = retry.result
            if result.status is ChatStatus.OK:
                conversation_id = await self._conversations.restart(user.tg_id, thread_id)
                context_reset = True

        if images and _is_blind(result) and (seen := await self._describe(images, text)):
            # The gateway lost the pictures on the way to this model: it works from their description instead.
            logger.warning("model %s did not get the pictures, answering from their description", model_id)
            last = messages[-1]
            messages = [
                *messages[:-1],
                replace(last, content=f"{last.content}\n\n{SEEN_NOTE}\n{seen}", images=()),
            ]
            retry = await self._complete(messages, model_id, (), web, draw, generated, files)
            usage += retry.usage
            attempts, latency_ms = attempts + retry.attempts, latency_ms + retry.latency_ms
            web_calls += retry.web_calls
            if retry.result.status is ChatStatus.OK:
                result = retry.result

        if result.status is ChatStatus.OK and _DECK_BLOCK.search(result.text):
            # The model sees its slides rendered (and the user's references) and fixes the layout once,
            # before the answer is sent and saved: the history keeps the corrected deck for later edits.
            pictures_now = {**recent, **{name: data for name, data in generated.items() if data}}
            polished = await self._polish_decks(
                result.text, messages, images, result.model_used, pictures_now
            )
            result = replace(result, text=polished)

        status = _STATUS[result.status]
        if status is RequestStatus.OK:
            # Only successful exchanges enter the history: failures and stubs would poison the context.
            # Images are not kept (too heavy); the full answer with its file blocks is, for follow-up edits.
            await self._conversations.save_exchange(conversation_id, text, result.text)
            if not in_reserve:  # the summary needs the main gateway
                budget = self._conversations.history_budget(token_budget, window)
                usage += TokenUsage(
                    service=await self._compact(user.tg_id, conversation_id, thread_id, budget)
                )
        if files is not None:  # digest_file read the user's file with the model: that is spent too
            usage += TokenUsage(service=files.spent)
        # Every call spends gateway balance, so it counts against the daily quota whatever the outcome.
        await self._quota.record(user.tg_id, usage.total)
        await self._requests.add(
            RequestRecord(
                user_id=user.tg_id,
                conversation_id=conversation_id,
                model=result.served_by or result.model_used,  # the reserve's own name keeps stats honest
                status=status,
                attempts=attempts,
                latency_ms=latency_ms,
                approx_tokens=usage.total,
                error=result.error,
                usage=usage,
                web_calls=web_calls,
            )
        )
        # Every drawn picture spent the generator's quota, so it counts whether or not the answer went out.
        await self._quota.record_images(user.tg_id, sum(1 for data in generated.values() if data))
        answer_text, files = split_files(result.text) if status is RequestStatus.OK else ("", [])
        pictures = (
            [OutFile(name, data) for name, data in generated.items()] if status is RequestStatus.OK else []
        )
        # A follow-up edit ("make the title bigger") re-sends the HTML with the old image names,
        # so the latest images of the user stay available for embedding.
        known = {**self._recent_images.get(chat_key, {}), **{p.name: p.data for p in pictures}}
        if pictures:
            self._recent_images[chat_key] = dict(list(known.items())[-RECENT_IMAGES:])
        if known:
            files = [
                OutFile(f.name, embed_images(f.data.decode("utf-8"), known).encode("utf-8"))
                if f.name.lower().endswith((".html", ".htm"))
                else f
                for f in files
            ]
        files = render_decks(files, known)  # a .pptx block is the deck's JSON: build the file and its preview
        return ChatReply(
            status=status,
            parts=render_markdown(answer_text, limit=TELEGRAM_LIMIT - HEADER_RESERVE),
            files=files,
            images=pictures,
            text=result.text,
            # Keep the pipeline badge for its configured answer model; show a fallback's real model.
            model=self._catalog.get(result.model_used)
            if result.attempts and (result.fallback_used or result.model_used != model_id)
            else spec,
            fallback_used=result.fallback_used,
            error=result.error,
            context_reset=context_reset,
            served_by=result.served_by,
            gemini_images=frozenset(draw.by_gemini & {p.name for p in pictures}) if draw else frozenset(),
        )

    async def _complete(
        self,
        messages: list[ChatMessage],
        model_id: str,
        images: tuple[str, ...],
        web: bool,
        draw: _Draw | None,
        generated: dict[str, bytes],
        files: FileTools | None = None,
    ) -> _CompleteOutcome:
        """One answer, running the model's tool calls in between.

        Tool rounds stay out of the history: only the final answer is saved by the caller.
        Images the model generates land in `generated` (file name -> bytes).
        """
        if web:
            today = datetime.now(UTC).strftime("%d.%m.%Y")
            messages = [
                replace(messages[0], content=f"{messages[0].content}\n\n{WEB_GUIDE}\nСегодня {today}."),
                *messages[1:],
            ]
        if files is not None:
            guide = file_guide(files.names())
            messages = [replace(messages[0], content=f"{messages[0].content}\n\n{guide}"), *messages[1:]]
        tools = (*(WEB_TOOLS if web else ()), *(_image_tools(draw)), *(FILE_TOOLS if files else ()))
        rounds = MAX_FILE_TOOL_ROUNDS if files is not None and not web else MAX_TOOL_ROUNDS
        # The dialog history as it stood on entry: resent to the model every round, so it counts each time.
        history_tokens = sum(approx_tokens(m.content) for m in messages[1:-1])
        usage = TokenUsage()
        attempts = latency_ms = web_calls = 0
        for round_no in range(rounds + 1):
            if round_no == rounds:
                # Tools stay declared (the history holds tool calls), the model is just told to stop.
                messages.append(ChatMessage(role="user", content=TOOLS_EXHAUSTED))
            result = await self._llm.chat(messages, model_id, tools)
            usage += TokenUsage(
                input=result.input_tokens
                if result.input_tokens is not None
                else sum(approx_tokens(m.content) for m in messages),
                output=result.output_tokens
                if result.output_tokens is not None
                else approx_tokens(result.text),
                images=IMAGE_TOKENS * len(images) if result.input_tokens is None and round_no == 0 else 0,
                context=history_tokens,
                tools=sum(approx_tokens(m.content) for m in messages if m.role == "tool"),
            )
            attempts, latency_ms = attempts + result.attempts, latency_ms + result.latency_ms
            if result.status is not ChatStatus.OK or not result.tool_calls:
                return _CompleteOutcome(result, usage, attempts, latency_ms, web_calls)
            if not tools or round_no == rounds:
                break
            # Only web calls are capped here: images have their own per-answer limit in `_draw`.
            asked = result.tool_calls
            web = [c for c in asked if c.name in WEB_TOOL_NAMES][:MAX_WEB_CALLS_PER_ROUND]
            calls = tuple(c for c in asked if c.name not in WEB_TOOL_NAMES or c in web)
            web_calls += sum(1 for call in calls if call.name in WEB_TOOL_NAMES)
            outputs = await asyncio.gather(*(self._run_tool(call, generated, draw, files) for call in calls))
            # Names and sizes only: the arguments and results hold the user's text.
            logger.info(
                "tools round %d: %s",
                round_no + 1,
                ", ".join(f"{c.name}->{len(out)}ch" for c, out in zip(calls, outputs, strict=True)),
            )
            messages.append(ChatMessage(role="assistant", content=result.text, tool_calls=calls))
            messages.extend(
                ChatMessage(role="tool", content=out, tool_call_id=call.id)
                for call, out in zip(calls, outputs, strict=True)
            )
        failed = replace(result, status=ChatStatus.FAILED, text="", tool_calls=(), error="tool_loop")
        return _CompleteOutcome(failed, usage, attempts, latency_ms, web_calls)

    async def _run_tool(
        self, call: ToolCall, generated: dict[str, bytes], draw: _Draw | None, files: FileTools | None = None
    ) -> str:
        if call.name in FILE_TOOL_NAMES and files is not None:
            return await files.run(call)
        if call.name == "generate_image" and self._images is not None and draw is not None and draw.generate:
            return await self._draw(call, generated, draw)
        if call.name == "edit_image" and self._images is not None and draw is not None:
            return await self._edit(call, generated, draw)
        if call.name in WEB_TOOL_NAMES and self._web is not None:
            return await self._web.run(call)
        return f"Ошибка: инструмент {call.name} недоступен."

    def _draw_limit(self, generated: dict[str, bytes], draw: _Draw) -> str | None:
        """Why no more pictures may be drawn in this answer (per answer or per day), or None."""
        if len(generated) < draw.cap:
            return None
        if draw.daily_left is not None and len(generated) >= draw.daily_left:
            return (
                f"Лимит картинок на сегодня исчерпан ({self._quota.daily_images} в сутки). "
                "Не рисуй больше, обойдись уже созданными и скажи пользователю, "
                "что новые картинки будут доступны завтра."
            )
        return f"Лимит: не больше {draw.cap} картинок за ответ. Обойдись уже созданными."

    async def _edit(self, call: ToolCall, generated: dict[str, bytes], draw: _Draw) -> str:
        """edit_image: FLUX changes the user's photo or an earlier picture, whatever image model is picked."""
        assert self._images is not None
        try:
            request = parse_edit_call(call)
        except (ValueError, TypeError):
            return "Ошибка: у edit_image обязательны name и prompt."
        source = _edit_source(request, draw)
        if source is None:
            return (
                "Ошибка: такой картинки нет. Источник — номер присланного фото или имя картинки этого чата."
            )
        if limit := self._draw_limit(generated, draw):
            return limit
        stem = _unique_stem(request.name, generated)
        generated[stem] = b""  # reserve the name and the slot before awaiting: calls run concurrently
        try:
            data = await self._images.edit(request.prompt, source)
        except ImageError as exc:
            del generated[stem]
            return f"Ошибка редактирования: {exc}. Скажи об этом пользователю."
        del generated[stem]
        filename = f"{stem}.{image_ext(data)}"
        generated[filename] = data
        return f'Готово: {filename}{_describe_size(data)}. В HTML: <img src="{filename}">.'

    async def _draw(self, call: ToolCall, generated: dict[str, bytes], draw: _Draw) -> str:
        assert self._images is not None
        try:
            request = parse_image_call(call)
        except (ValueError, TypeError):
            return "Ошибка: у generate_image обязательны name и prompt."
        if limit := self._draw_limit(generated, draw):
            return limit
        stem = _unique_stem(request.name, generated)
        generated[stem] = b""  # reserve the name and the slot before awaiting: calls run concurrently
        prompt = request.prompt + (CUTOUT_SUFFIX if request.cutout else "")
        try:
            data, drawn_by = await self._images.generate(prompt, request.size, draw.model)
        except ImageError as exc:
            del generated[stem]
            return f"Ошибка генерации: {exc}. Продолжай без этой картинки и скажи об этом пользователю."
        del generated[stem]
        note = ""
        if request.cutout:
            cut = await asyncio.to_thread(_safe_cutout, data)
            if cut is None:
                note = (
                    " Вырезать фон не удалось, картинка осталась с фоном: вставь её в карточку/рамку "
                    "или на светлом фоне примени mix-blend-mode: multiply."
                )
            else:
                data, note = cut, " Фон прозрачный, картинка обрезана по объекту."
        filename = f"{stem}.{image_ext(data)}"
        generated[filename] = data
        if drawn_by is ImageModel.GEMINI:
            draw.by_gemini.add(filename)
        size = _describe_size(data)
        return f'Готово: {filename}{size}.{note} В HTML: <img src="{filename}">.'

    def _history_window(self, model_id: str) -> int | None:
        """Tokens the dialog may take for `model_id`: its window less the answer and EXTRA_ROOM_TOKENS."""
        try:
            spec = self._catalog.get(model_id)
        except KeyError:
            return None
        return max(1_000, int(spec.context_tokens * (1 - WINDOW_SLACK)) - spec.max_tokens - EXTRA_ROOM_TOKENS)

    async def _compact(
        self, user_id: int, conversation_id: int, thread_id: int = 0, budget: int | None = None
    ) -> int:
        """Compact a long dialog after the answer (see ConversationService.compact): tokens spent.

        With a `budget` (the tokens the dialog may take) it is compacted near that budget, keeping half of it
        word for word: the model sees as much of the chat as its window holds, and the rest as a summary.

        ponytail: runs before the reply is sent, so the rare compacting answer waits for the summary;
        move it to a background task with a per-user lock if that delay starts to matter.
        """
        spent = 0

        async def summarize(transcript: str) -> str | None:
            nonlocal spent
            messages = [
                ChatMessage(role="system", content=SUMMARIZER),
                ChatMessage(role="user", content=transcript),
            ]
            result = await self._llm.chat(messages, self._catalog.for_mode(Mode.FAST).id)
            spent += sum(approx_tokens(m.content) for m in messages) + approx_tokens(result.text)
            return result.text if result.status is ChatStatus.OK else None

        # Small personal history budgets should not summarize away the answer just delivered.
        sizes = (
            {
                "at": max(COMPACT_AT, int(budget * COMPACT_AT_SHARE)),
                "keep": int(budget * COMPACT_KEEP_SHARE),
            }
            if budget and budget >= COMPACT_AT
            else {}
        )
        try:
            if await self._conversations.compact(user_id, conversation_id, summarize, thread_id, **sizes):
                logger.info("dialog of user %s compacted (%s service tokens)", user_id, spent)
        except Exception:  # compaction may save money, it must never cost an answer
            logger.warning("compacting the dialog of user %s failed", user_id, exc_info=True)
        return spent

    async def chat_title(self, text: str) -> str:
        """A short name for a new chat from its first message (the topic's title).

        Haiku on the reserve gateway names it (the fast mode's model without one); any failure falls back to
        the message's own first words.
        ponytail: the call is not counted against the daily quota (~100 tokens per chat).
        """
        fallback = title_from(text)
        model = (
            TITLE_MODEL
            if self._reserve is not None and self._catalog.is_available(TITLE_MODEL)
            else self._catalog.for_mode(Mode.FAST).id
        )
        try:
            result = await self._llm.chat(
                [
                    ChatMessage(role="system", content=TITLER),
                    ChatMessage(role="user", content=text[:REWRITE_MAX_CHARS]),
                ],
                model,
            )
        except Exception:
            logger.warning("naming a chat failed", exc_info=True)
            return fallback
        return title_from(result.text) if result.status is ChatStatus.OK and result.text.strip() else fallback

    async def _rewrite(self, user: User, text: str) -> tuple[str, int, int]:
        """Hidden prompt-improvement step: (prompt for the model, tokens spent, latency ms).

        Any failure falls back to the user's own text: the step may make answers better, never break them.
        """
        system = self._conversations.rewriter_prompt(user.skills)
        # ponytail: file uploads and long pastes are skipped, the rewriter could drop their contents
        if system is None or not REWRITE_MIN_CHARS <= len(text) <= REWRITE_MAX_CHARS or "<file name=" in text:
            return text, 0, 0
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=f"<request>\n{text}\n</request>"),
        ]
        result = await self._llm.chat(messages, self._catalog.for_mode(Mode.FAST).id)
        rewritten = result.text.strip().removeprefix("<request>").removesuffix("</request>").strip()
        if result.status is not ChatStatus.OK or not rewritten:
            rewritten = text
        service = sum(approx_tokens(m.content) for m in messages) + approx_tokens(result.text)
        return rewritten, service, result.latency_ms

    async def _plan(
        self, user_id: int, prompt: str, model_id: str, thread_id: int = 0
    ) -> tuple[str, list[str], int, int]:
        """Pipeline step one: (plan, picked skill ids, service tokens, latency ms).

        A failed planner costs only its time: the model then answers without a plan.
        """
        _, recent = await self._conversations.build_context(
            user_id, prompt, token_budget=PLAN_CONTEXT_TOKENS, thread_id=thread_id
        )
        dialog = "\n\n".join(f"[{m.role}]\n{m.content[:PLAN_TURN_CHARS]}" for m in recent[1:])
        skills = self._conversations.pickable_skills()
        menu = "\n".join(f"- {s.id}: {s.title} — {s.description}".rstrip(" —") for s in skills) or "(нет)"
        messages = [
            ChatMessage(role="system", content=PLANNER.format(skills=menu)),
            ChatMessage(role="user", content=dialog),
        ]
        result = await self._llm.chat(messages, model_id)
        service = sum(approx_tokens(m.content) for m in messages) + approx_tokens(result.text)
        if result.status is not ChatStatus.OK:
            return "", [], service, result.latency_ms
        plan, picked = parse_plan(result.text, {s.id for s in skills})
        return plan, picked, service, result.latency_ms

    async def _research(self, text: str) -> tuple[str, int, int, int]:
        """Deep research before the answer: (brief for the model, service tokens, web calls, latency ms).

        The fast model plans the queries; if it fails, the user's own text is the only query.
        An empty brief means nothing could be read, and the model is left to its tools.
        """
        assert self._web is not None
        messages = [
            ChatMessage(role="system", content=RESEARCH_PLANNER),
            ChatMessage(role="user", content=text[:REWRITE_MAX_CHARS]),
        ]
        result = await self._llm.chat(messages, self._catalog.for_mode(Mode.FAST).id)
        service = sum(approx_tokens(m.content) for m in messages) + approx_tokens(result.text)
        queries = list(dict.fromkeys(parse_queries(result.text) if result.status is ChatStatus.OK else []))
        queries = queries[:RESEARCH_QUERIES] or [text[:300]]
        material, calls = await self._web.research(text, queries)
        brief = RESEARCH_BRIEF.format(queries="; ".join(queries), material=material) if material else ""
        return brief, service, calls, result.latency_ms


def parse_plan(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """The planner's reply as (plan, skill ids it picked among `allowed`); unknown ids are dropped."""
    picked: list[str] = []
    match = re.search(r"^\s*SKILLS:(.*)$", text, re.M | re.I)
    if match:
        ids = dict.fromkeys(s for s in re.split(r"[\s,;]+", match.group(1)) if s in allowed)
        picked = list(ids)[:PLAN_MAX_SKILLS]
        text = text[: match.start()] + text[match.end() :]
    plan = re.sub(r"^[ \t]*PLAN:[ \t]*\n?", "", text.strip(), count=1, flags=re.I | re.M)
    return plan.strip(), picked


def title_from(text: str) -> str:
    """A one-line topic name of at most TITLE_MAX_CHARS: the first line, quotes and dots trimmed."""
    lines = [line.strip().strip("\"'«»`*#. ") for line in text.strip().splitlines()]
    line = next((line for line in lines if line), "")
    if len(line) > TITLE_MAX_CHARS:
        line = line[:TITLE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"
    return line or "Новый чат"


def parse_queries(text: str) -> list[str]:
    """One query per line; list markers, numbering and quotes the model adds anyway are stripped."""
    lines = (
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'«»") for line in text.splitlines()
    )
    return [line for line in lines if 2 < len(line) <= 200]


def _data_uri_bytes(uri: str) -> bytes:
    """The bytes of a `data:...;base64,...` image URI; empty when it is not one."""
    try:
        return base64.b64decode(uri.split(",", 1)[1])
    except (IndexError, ValueError):
        return b""


def _edit_source(request: EditRequest, draw: _Draw) -> bytes | None:
    """The picture to edit: "1".. = the Nth photo of the message, else a picture of the chat by its name."""
    if request.source.isdigit():
        n = int(request.source)
        return draw.sources[n - 1] or None if 1 <= n <= len(draw.sources) else None
    return draw.recent.get(request.source) or next(
        (data for name, data in draw.recent.items() if name.split(".", 1)[0] == request.source), None
    )


def _image_tools(draw: _Draw | None) -> tuple:
    """The picture tools of one answer: drawing needs the image skill, editing needs something to edit."""
    if draw is None:
        return ()
    editable = bool(draw.sources or draw.recent)
    return (*(IMAGE_TOOLS if draw.generate else ()), *((edit_tool(list(draw.recent)),) if editable else ()))


def _unique_stem(name: str, taken: dict[str, bytes]) -> str:
    """Safe ASCII file stem for a generated image, not clashing with the ones made in this answer."""
    stem = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:40] or "image"
    stems = {key.split(".", 1)[0] for key in taken}
    candidate, n = stem, 2
    while candidate in stems:
        candidate, n = f"{stem}-{n}", n + 1
    return candidate


def _safe_cutout(data: bytes) -> bytes | None:
    try:
        return cut_out_background(data)
    except Exception:  # a broken image must not break the answer: the picture just keeps its background
        logger.warning("background cutout failed", exc_info=True)
        return None


def _describe_size(data: bytes) -> str:
    try:
        width, height = image_size(data)
    except Exception:
        return ""
    return f", {width}×{height}"
