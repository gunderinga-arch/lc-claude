from bot.domain import Mode, Role
from bot.services import conversation as conv
from bot.services.conversation import ConversationService
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UserRepo


async def test_long_dialog_is_compacted_into_summary_plus_latest_messages() -> None:
    db = Database(":memory:")
    await db.connect()
    await UserRepo(db).upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    repo = ConversationRepo(db)
    service = ConversationService(repo, system_prompt="sys", token_budget=350_000, skills={})
    cid = await repo.active_id(1)
    for i in range(10):  # 20 messages of 12k tokens: 240k, over COMPACT_AT
        await repo.add_message(cid, "user", f"q{i}", 12_000)
        await repo.add_message(cid, "assistant", f"a{i}", 12_000)

    async def failed(_: str) -> None:
        return None

    assert not await service.compact(1, cid, failed)  # no summary: nothing changes
    assert await repo.active_id(1) == cid

    seen: list[str] = []

    async def summarize(transcript: str) -> str:
        seen.append(transcript)
        return "конспект"

    assert await service.compact(1, cid, summarize)
    fresh = await repo.active_id(1)
    assert fresh != cid
    messages = await repo.recent_messages(fresh, 100)
    assert messages[0].content == conv.SUMMARY_HEADER + "конспект"
    assert [m.content for m in messages[2:]] == ["q9", "a9"]  # the latest ~COMPACT_KEEP tokens, verbatim
    assert "q0" in seen[0] and "q8" in seen[0] and "q9" not in seen[0]
    assert not await service.compact(1, fresh, summarize)  # now small: left alone
    await db.close()


async def test_compaction_keeps_latest_pair_even_if_answer_exceeds_keep():
    db = Database(":memory:")
    await db.connect()
    try:
        await UserRepo(db).upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
        repo = ConversationRepo(db)
        service = ConversationService(repo, system_prompt="s", token_budget=32_000)
        cid = await repo.active_id(1)
        for role, text, tokens in [
            ("user", "old q", 5000),
            ("assistant", "old a", 5000),
            ("user", "new q", 1000),
            ("assistant", "code", 17_000),
        ]:
            await repo.add_message(cid, role, text, tokens)

        async def summarize(text):
            assert "code" not in text and "old q" in text
            return "summary"

        assert await service.compact(1, cid, summarize, at=20_000, keep=16_000)
        messages = await repo.recent_messages(await repo.active_id(1), 100)
        assert [m.content for m in messages[-2:]] == ["new q", "code"]
    finally:
        await db.close()
