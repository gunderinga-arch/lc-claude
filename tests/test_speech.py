import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.services.speech import SpeechRecognizer, SpeechUnavailable
from bot.telegram.routers import chat as chat_router


class FakeModel:
    def __init__(self, texts: list[str], delay: float = 0.0) -> None:
        self.texts = texts
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def transcribe(self, audio, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        import time

        time.sleep(self.delay)
        self.active -= 1
        segments = [SimpleNamespace(text=t) for t in self.texts]
        return iter(segments), SimpleNamespace(duration=3.0, language="ru")


def _recognizer(model: FakeModel, tmp_path: Path, loads: list[int] | None = None) -> SpeechRecognizer:
    def loader(size, path):
        if loads is not None:
            loads.append(1)
        return model

    return SpeechRecognizer("small", tmp_path / "models", loader=loader)


async def test_transcribe_joins_segments_and_loads_once(tmp_path) -> None:
    loads: list[int] = []
    rec = _recognizer(FakeModel([" Привет, ", "как дела? "]), tmp_path, loads)
    assert await rec.transcribe(b"ogg") == "Привет, как дела?"
    assert await rec.transcribe(b"ogg") == "Привет, как дела?"
    assert loads == [1]
    assert (tmp_path / "models").is_dir()


async def test_silence_gives_empty_text(tmp_path) -> None:
    assert await _recognizer(FakeModel([]), tmp_path).transcribe(b"ogg") == ""


async def test_one_transcription_at_a_time(tmp_path) -> None:
    model = FakeModel(["x"], delay=0.05)
    rec = _recognizer(model, tmp_path)
    await asyncio.gather(*(rec.transcribe(b"ogg") for _ in range(3)))
    assert model.max_active == 1  # single core: parallel runs would only be slower


async def test_broken_model_is_reported_and_warm_up_survives(tmp_path) -> None:
    def broken(size, path):
        raise SpeechUnavailable("faster-whisper is not installed")

    rec = SpeechRecognizer("small", tmp_path, loader=broken)
    await rec.warm_up()  # logs, does not raise: the bot must still start
    with pytest.raises(SpeechUnavailable):
        await rec.transcribe(b"ogg")


# ---- the chat side: echo the transcript, then answer it ----------------------------------------


class FakeBot:
    async def download(self, media, destination=None):
        import io

        destination = destination if destination is not None else io.BytesIO()
        destination.write(b"ogg")
        destination.seek(0)
        return destination


class FakeMessage:
    def __init__(self, caption: str | None = None) -> None:
        self.voice = SimpleNamespace(file_id="v", duration=5, file_size=1000)
        self.audio = self.video_note = None
        self.caption = caption
        self.replies: list[str] = []
        self.answers: list[str] = []

    async def reply(self, text, **kw):
        self.replies.append(text)

    async def answer(self, text, **kw):
        self.answers.append(text)


class FakeSpeech:
    def __init__(self, result: str | Exception) -> None:
        self.result = result

    async def transcribe(self, audio: bytes) -> str:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_transcript_is_echoed_escaped_and_returned() -> None:
    msg = FakeMessage(caption="Ответь кратко")
    text = await chat_router._transcribe(FakeBot(), msg, FakeSpeech("сравни <b>A</b> и B"))
    assert text == "Ответь кратко\n\nсравни <b>A</b> и B"
    assert msg.replies == ["🎤 <i>сравни &lt;b&gt;A&lt;/b&gt; и B</i>"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [("", chat_router.VOICE_EMPTY_TEXT), (SpeechUnavailable("no model"), chat_router.VOICE_FAIL_TEXT)],
)
async def test_nothing_to_answer(result, expected) -> None:
    msg = FakeMessage()
    assert await chat_router._transcribe(FakeBot(), msg, FakeSpeech(result)) is None
    assert msg.answers == [expected] and msg.replies == []
