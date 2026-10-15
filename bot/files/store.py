"""Full texts of the big files of a chat, for the model's read_file / search_file tools.

A file too big for the model's window goes into the message cut (its start, with a note); the whole text
waits here, per chat (user + Telegram topic), so the model can read any part or search it — in that answer
and in later ones, after the history has dropped the file or squeezed it into a summary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

KEEP_DAYS = 30  # a chat's files are deleted this long after they were last saved or read


class FileStore:
    def __init__(self, root: Path, keep_days: float = KEEP_DAYS) -> None:
        self._root = root
        self._keep_s = keep_days * 86_400

    def _dir(self, user_id: int, thread_id: int) -> Path:
        return self._root / str(user_id) / str(thread_id)

    @staticmethod
    def _stem(name: str) -> str:
        return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]

    def save(self, user_id: int, thread_id: int, name: str, text: str) -> None:
        """Keep `text` as file `name` of the chat (a file of the same name is replaced)."""
        folder = self._dir(user_id, thread_id)
        folder.mkdir(parents=True, exist_ok=True)
        stem = self._stem(name)
        tmp = folder / f"{stem}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, folder / f"{stem}.txt")
        (folder / f"{stem}.name").write_text(name, encoding="utf-8")
        self.prune()

    def names(self, user_id: int, thread_id: int) -> list[str]:
        folder = self._dir(user_id, thread_id)
        if not folder.is_dir():
            return []
        found = [(p.stat().st_mtime, p) for p in folder.glob("*.name") if p.with_suffix(".txt").exists()]
        return [p.read_text(encoding="utf-8") for _, p in sorted(found)]

    def read(self, user_id: int, thread_id: int, name: str) -> tuple[str, str] | None:
        """(stored name, text) of the chat's file `name`; a name differing in case or path also matches."""
        names = self.names(user_id, thread_id)
        wanted = name.strip().lower()
        match = next((n for n in names if n == name), None) or next(
            (
                n
                for n in names
                if n.lower() == wanted or n.lower().rsplit("/", 1)[-1] == wanted.rsplit("/", 1)[-1]
            ),
            None,
        )
        if match is None and len(names) == 1:
            match = names[0]  # one file in the chat: a garbled name can only mean it
        if match is None:
            return None
        path = self._dir(user_id, thread_id) / f"{self._stem(match)}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        os.utime(path)  # read = still in use: pruning counts from the last use
        return match, text

    def clear(self, user_id: int, thread_id: int) -> None:
        """Forget the chat's files ("new dialog", "clear chat")."""
        shutil.rmtree(self._dir(user_id, thread_id), ignore_errors=True)

    def prune(self) -> None:
        """Delete files unused for KEEP_DAYS and the folders they leave empty."""
        if not self._root.is_dir():
            return
        cutoff = time.time() - self._keep_s
        for path in self._root.glob("*/*/*.txt"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    path.with_suffix(".name").unlink(missing_ok=True)
            except OSError:
                logger.warning("pruning stored file %s failed", path, exc_info=True)
        for folder in sorted(self._root.glob("*/*"), reverse=True):
            try:
                if folder.is_dir():
                    folder.rmdir()  # atomic: a concurrent save may have filled it since the scan
            except OSError:
                pass  # nonempty or removed concurrently
