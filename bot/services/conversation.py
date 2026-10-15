"""Builds the LLM context for a turn and persists the exchange."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from bot.domain import IMAGE_TOKENS, ChatInfo, Style, approx_tokens
from bot.files.store import FileStore
from bot.llm.base import ChatMessage
from bot.skills import Skill
from bot.storage.repositories import ConversationRepo

# Messages read from the database for one context: a ~1M-token window holds thousands of short turns.
HISTORY_FETCH_LIMIT = 5000
# A long dialog is compacted once its stored history reaches COMPACT_AT tokens: the older part becomes
# one summary, the latest ~COMPACT_KEEP tokens stay word for word. Every answer resends the whole history,
# so this caps what a long chat costs per message. The chat service passes values sized to the model's window
# (COMPACT_AT_SHARE and COMPACT_KEEP_SHARE of the history budget); these are the defaults without one.
COMPACT_AT = 200_000
COMPACT_KEEP = 30_000
COMPACT_AT_SHARE = 0.9
COMPACT_KEEP_SHARE = 0.5
SUMMARY_HEADER = "[Краткое содержание начала этого разговора]\n"
SUMMARY_ACK = "Понял, продолжаем с учётом этого."

STYLE_INSTRUCTIONS = {
    Style.SHORT: "Отвечай кратко: 1–3 предложения или короткий список, без вступлений и повторов вопроса.",
    Style.NORMAL: "",
    Style.DETAILED: "Отвечай подробно и развёрнуто: с объяснениями, примерами и понятной структурой.",
}

PERSONA_HEADER = (
    "Ниже — пожелания пользователя к твоей роли и манере общения. "
    "Следуй им, если они не противоречат правилам выше.\n"
)


def compose_system_prompt(
    base: str, persona: str | None, style: Style, skill_prompts: Sequence[str] = ()
) -> str:
    """Bot rules first, then style, admin skills and the user's persona last: it can't replace the base."""
    parts = [base, STYLE_INSTRUCTIONS[style], *skill_prompts]
    if persona:
        parts.append(PERSONA_HEADER + persona)
    return "\n\n".join(p for p in parts if p)


class ConversationService:
    def __init__(
        self,
        repo: ConversationRepo,
        *,
        system_prompt: str,
        token_budget: int,
        skills: Mapping[str, Skill] | None = None,
        idle_hours: float = 24,
        files: FileStore | None = None,  # the chats' stored big files: a reset forgets them too
    ) -> None:
        self._files = files
        self._repo = repo
        self._system_prompt = system_prompt
        self._token_budget = token_budget
        self._skills = dict(skills or {})
        self._idle_hours = idle_hours

    def rewriter_prompt(self, skills: tuple[str, ...]) -> str | None:
        """System prompt of the first enabled rewrite skill, if any."""
        return next(
            (self._skills[s].prompt for s in skills if s in self._skills and self._skills[s].rewrite), None
        )

    def wants_web(self, skills: tuple[str, ...]) -> bool:
        return any(s in self._skills and self._skills[s].web for s in skills)

    def wants_images(self, skills: tuple[str, ...]) -> bool:
        return any(s in self._skills and self._skills[s].images for s in skills)

    def pickable_skills(self) -> list[Skill]:
        """Skills the answer planner may switch on for one message (the hidden rewriters are not)."""
        return [s for s in self._skills.values() if not s.rewrite]

    async def build_context(
        self,
        user_id: int,
        user_text: str,
        *,
        images: tuple[str, ...] = (),
        persona: str | None = None,
        style: Style = Style.NORMAL,
        skills: tuple[str, ...] = (),
        with_history: bool = True,
        token_budget: int | None = None,
        idle_hours: float | None = None,
        thread_id: int = 0,
        window: int | None = None,
    ) -> tuple[int, list[ChatMessage]]:
        """Return (conversation_id, [system] + kept history + [user]) within the token budget.

        `images` (data: URIs) go only to the current message; history keeps text only.
        Skill ids no longer in skills.yaml are skipped. `token_budget` overrides the service's
        global one (a user's personal context limit); None uses the global budget.
        `idle_hours` likewise overrides the global idle timeout (a user's personal value,
        0 = never reset); None uses the global `CONTEXT_IDLE_HOURS`. `thread_id` is the chat's Telegram
        topic (0 = the main chat); only the main chat forgets after silence, a topic is a chat the user
        keeps and ends on purpose. `window` caps the budget at what the answering model can take in.
        History messages come marked `droppable`: the LLM client leaves the earliest out if a request still
        outgrows a model's window (tool results, a fallback with a smaller window).
        """
        budget = self.history_budget(token_budget, window)
        conversation_id = await self._active_after_idle(user_id, idle_hours, thread_id)
        enabled = [self._skills[s] for s in skills if s in self._skills]
        skill_prompts = [s.prompt for s in enabled if not s.rewrite]
        system = compose_system_prompt(self._system_prompt, persona, style, skill_prompts)
        used = approx_tokens(system) + approx_tokens(user_text) + IMAGE_TOKENS * len(images)

        history = (
            await self._repo.recent_messages(conversation_id, HISTORY_FETCH_LIMIT) if with_history else []
        )
        kept: list[ChatMessage] = []
        for message in reversed(history):
            used += message.approx_tokens
            if used > budget:
                break
            kept.append(ChatMessage(role=message.role, content=message.content, droppable=True))
        kept.reverse()

        while kept and kept[0].role == "assistant":
            kept.pop(0)

        messages = [ChatMessage(role="system", content=system)]
        messages.extend(kept)
        messages.append(ChatMessage(role="user", content=user_text, images=images))
        return conversation_id, messages

    def history_budget(self, token_budget: int | None = None, window: int | None = None) -> int:
        """Tokens a context may take: the personal budget (else the global one), capped by `window`."""
        budget = token_budget if token_budget is not None else self._token_budget
        return min(budget, window) if window is not None else budget

    async def _active_after_idle(
        self, user_id: int, idle_hours: float | None = None, thread_id: int = 0
    ) -> int:
        """The active conversation id, quietly reset after an idle timeout.

        The timeout counts from the last stored message (the user's or the bot's; an empty
        dialog counts from `started_at`). A reset is invisible to the user: no flag, no notice.
        `idle_hours` is the user's personal value (0 = never reset); None uses the global one.
        """
        timeout = self._idle_hours if idle_hours is None else idle_hours
        conversation_id = await self._repo.active_id(user_id, thread_id)
        if timeout <= 0 or thread_id:
            return conversation_id
        last = await self._repo.last_activity(conversation_id)
        if last is None:
            last = await self._repo.started_at(conversation_id)
        if last is None:
            return conversation_id
        if datetime.now(UTC) - last > timedelta(hours=timeout):
            await self._repo.reset(user_id)
            if self._files is not None:
                self._files.clear(user_id, 0)
            return await self._repo.active_id(user_id)
        return conversation_id

    async def save_exchange(self, conversation_id: int, user_text: str, assistant_text: str) -> None:
        await self._repo.add_message(conversation_id, "user", user_text, approx_tokens(user_text))
        await self._repo.add_message(
            conversation_id, "assistant", assistant_text, approx_tokens(assistant_text)
        )

    async def reset(self, user_id: int, thread_id: int = 0) -> None:
        await self._repo.reset(user_id, thread_id)
        if self._files is not None:
            self._files.clear(user_id, thread_id)

    async def open_chat(
        self, user_id: int, thread_id: int, *, model_override: str | None = None, title: str | None = None
    ) -> int:
        """Register a topic as a chat (it inherits `model_override`); returns the conversation id.

        A `title` marks a topic as already named, so the bot never renames it.
        """
        return await self._repo.active_id(user_id, thread_id, model_override=model_override, title=title)

    async def chat(self, user_id: int, thread_id: int) -> ChatInfo | None:
        return await self._repo.chat(user_id, thread_id)

    async def set_title(self, user_id: int, thread_id: int, title: str) -> None:
        await self._repo.set_title(user_id, thread_id, title)

    async def set_model(self, user_id: int, thread_id: int, model_id: str | None) -> None:
        await self._repo.set_model(user_id, thread_id, model_id)

    async def threads(self, user_id: int) -> list[int]:
        return await self._repo.active_threads(user_id)

    async def empty_thread(self, user_id: int) -> int | None:
        return await self._repo.empty_thread(user_id)

    async def compact(
        self,
        user_id: int,
        conversation_id: int,
        summarize: Callable[[str], Awaitable[str | None]],
        thread_id: int = 0,
        *,
        at: int = COMPACT_AT,
        keep: int = COMPACT_KEEP,
    ) -> bool:
        """Quietly squeeze a long dialog once it reaches `at` tokens, keeping the latest `keep`; True if done.

        The older messages go to `summarize` as one transcript; the dialog then continues in a fresh
        conversation holding the summary and the latest messages (the admin bot keeps the old one).
        No summary (the model failed) leaves everything as it was: the budget window still applies.
        """
        history = await self._repo.recent_messages(conversation_id, HISTORY_FETCH_LIMIT)
        if sum(m.approx_tokens for m in history) < at:
            return False
        kept, used = [], 0
        for message in reversed(history):
            if used + message.approx_tokens > keep and len(kept) >= 2:
                break
            used += message.approx_tokens
            kept.append(message)
        kept.reverse()
        while kept and kept[0].role == "assistant":
            kept.pop(0)
        older = history[: len(history) - len(kept)]
        if not older:
            return False
        summary = await summarize("\n\n".join(f"[{m.role}]\n{m.content}" for m in older))
        if not summary:
            return False
        fresh = await self.restart(user_id, thread_id)
        text = SUMMARY_HEADER + summary.strip()
        await self._repo.add_message(fresh, "user", text, approx_tokens(text))
        await self._repo.add_message(fresh, "assistant", SUMMARY_ACK, approx_tokens(SUMMARY_ACK))
        for message in kept:
            await self._repo.add_message(fresh, message.role, message.content, message.approx_tokens)
        return True

    async def restart(self, user_id: int, thread_id: int = 0) -> int:
        """Drop the current history and return the id of a fresh conversation of the same chat.

        The chat keeps its title and model: only what the model remembers is dropped.
        """
        info = await self._repo.chat(user_id, thread_id)
        await self._repo.reset(user_id, thread_id)
        return await self._repo.active_id(
            user_id,
            thread_id,
            model_override=info.model_override if info else None,
            title=info.title if info else None,
        )
