"""Speech to text for voice messages: faster-whisper on the server's CPU.

The gateway has no audio support (no /audio/transcriptions, `input_audio` is rejected), so voice is
transcribed locally and the model gets plain text. Benchmark on the 1-vCPU server, 20 s of Russian
speech: small int8 took ~10 s with 3% word errors (base ~4 s / 6%, tiny ~2 s / 12-26%).
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SpeechUnavailable(Exception):
    """Speech recognition can't run here (package missing or the model failed to load)."""


def _load_whisper(model_size: str, models_dir: Path) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # optional dependency: the bot works without voice support
        raise SpeechUnavailable("faster-whisper is not installed") from exc
    return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(models_dir))


class SpeechRecognizer:
    """Loads the model once (lazily or via warm_up) and transcribes one clip at a time.

    One at a time on purpose: the server has a single core, parallel runs would only be slower.
    """

    def __init__(
        self,
        model_size: str,
        models_dir: Path,
        *,
        loader: Callable[[str, Path], Any] = _load_whisper,
    ) -> None:
        self._model_size = model_size
        self._models_dir = models_dir
        self._loader = loader
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            self._model = self._loader(self._model_size, self._models_dir)
            logger.info("speech model %s loaded", self._model_size)
        return self._model

    def _run(self, audio: bytes) -> str:
        model = self._ensure_model()
        segments, info = model.transcribe(io.BytesIO(audio), beam_size=1, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "voice transcribed: %.0f s of %s audio -> %d chars", info.duration, info.language, len(text)
        )
        return text

    async def warm_up(self) -> None:
        """Load the model in the background at startup, so the first voice message doesn't wait for it."""
        try:
            async with self._lock:
                await asyncio.to_thread(self._ensure_model)
        except Exception:
            logger.exception("speech model failed to load; voice messages will get an error reply")

    async def transcribe(self, audio: bytes) -> str:
        """Text of the clip ("" if no speech was found). Raises SpeechUnavailable."""
        async with self._lock:
            try:
                return await asyncio.to_thread(self._run, audio)
            except SpeechUnavailable:
                raise
            except Exception as exc:
                logger.exception("speech recognition failed")
                raise SpeechUnavailable(str(exc)) from exc
