"""Skills: admin-defined instruction packs a user can switch on in settings."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

# The id rides in callback_data ("set:skill:<id>", 64 bytes max), so it is short and ASCII.
_ID_RE = re.compile(r"^[a-z0-9_-]{1,40}$")
SKILL_PROMPT_MAX_CHARS = 8000  # every enabled skill rides along with every request


class SkillsError(ValueError):
    """The skills file is inconsistent."""


@dataclass(frozen=True, slots=True)
class Skill:
    id: str
    title: str  # shown on the button, emoji included
    prompt: str  # appended to the system prompt; for a rewrite skill, the rewriter's system prompt
    description: str = ""
    rewrite: bool = False  # hidden pre-step: a fast model rewrites the user's message before the answer
    web: bool = False  # gives the model internet tools on every message (needs SEARXNG_URL)
    images: bool = False  # gives the model image generation (needs CLOUDFLARE_ACCOUNT_ID and _API_TOKEN)


def toggle_skill(enabled: tuple[str, ...], skill_id: str, skills: Mapping[str, Skill]) -> tuple[str, ...]:
    """Flip one skill; ids that vanished from the file are dropped on the way."""
    kept = tuple(s for s in enabled if s in skills and s != skill_id)
    return kept if skill_id in enabled else (*kept, skill_id)


def load_skills(path: Path) -> dict[str, Skill]:
    """Skills in file order. No file = no skills, the settings button just stays hidden."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    skills: dict[str, Skill] = {}
    for skill_id, item in (raw.get("skills") or {}).items():
        skill_id = str(skill_id)
        if not _ID_RE.match(skill_id):
            raise SkillsError(f"{path}: skill id {skill_id!r} must match {_ID_RE.pattern}")
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            raise SkillsError(f"{path}: skill {skill_id} needs a title")
        prompt = str(item.get("prompt", "")).strip()
        if not prompt or len(prompt) > SKILL_PROMPT_MAX_CHARS:
            raise SkillsError(f"{path}: skill {skill_id} needs a prompt of 1..{SKILL_PROMPT_MAX_CHARS} chars")
        skills[skill_id] = Skill(
            id=skill_id,
            title=str(item["title"]).strip(),
            prompt=prompt,
            description=str(item.get("description", "")).strip(),
            rewrite=bool(item.get("rewrite", False)),
            web=bool(item.get("web", False)),
            images=bool(item.get("images", False)),
        )
    return skills
