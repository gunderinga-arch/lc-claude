"""Image generation for the models, exposed as a function tool: FLUX or Gemini, the user's pick.
Editing a picture the user sent (or one drawn earlier) is always FLUX: Gemini's gateway takes no input image.

FLUX: Cloudflare Workers AI (FLUX.2 klein). Free plan: 10,000 neurons a day, ~104 per 1024x1024 image.
Over the limit Cloudflare answers with an error until 00:00 UTC and charges nothing.

Gemini: Nano Banana through OmniRoute (the reserve gateway), on the owner's Google account. Models are
tried in GEMINI_MODELS order; when all fail (no cookies, weekly quota spent) the picture comes from FLUX.

Cutouts: the object is generated on plain white and the white is keyed out (flood fill from the edges),
so a PNG with transparency comes out without a background-removal network on a 1.7 GB server.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from bot.domain import ImageModel
from bot.llm.base import ToolCall, ToolSpec

logger = logging.getLogger(__name__)

CLOUDFLARE_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/"
DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
TIMEOUT_S = 120.0
PROMPT_MAX_CHARS = 2_000

# OmniRoute image models, best first: the owner's Gemini subscription (gemini-web, cookies refreshed by
# the server's gemini-refresh timer), then Antigravity (the same Google account, a small weekly quota).
GEMINI_MODELS = ("gemini-web/nano-banana-web", "antigravity/gemini-3.1-flash-image")
# gemini-web drives the Gemini page in OmniRoute's Chromium, which gives it 20 s to load; from this
# server the first load often misses that and a second one fits, so a page timeout gets one retry.
PAGE_TIMEOUT = "страница Gemini не загрузилась"
GEMINI_PAGE_TRIES = 2
# Antigravity takes aspect_ratio (size is ignored there), gemini-web only knows three sizes.
GEMINI_SHAPES: dict[str, tuple[str, str]] = {
    "square": ("1:1", "1024x1024"),
    "landscape": ("16:9", "1536x1024"),
    "portrait": ("9:16", "1024x1536"),
    "banner": ("21:9", "1536x1024"),
}

# Appended to a cutout prompt: a plain white field around one whole object is what the keying needs.
# "Floating" matters: an object standing on something gets a contact shadow that survives the keying.
CUTOUT_SUFFIX = (
    ", single isolated object floating in mid-air, levitating, centered, whole object fully inside the frame "
    "with margin around it, on a pure plain white background (#FFFFFF), no surface, no table, no floor, "
    "no cast shadow, no contact shadow, no reflection, clean studio cutout"
)
BG_MIN_LIGHTNESS = 225  # a corner darker than this means the background is not plain white
BG_THRESH = 60  # flood-fill tolerance: summed RGB difference from the seed pixel (JPEG noise, soft light)
CUTOUT_PADDING = 8  # px kept around the object after trimming

# Presets keep the model from asking for sizes the API rejects; every side is a multiple of 64.
SIZES: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "banner": (1536, 640),
}

IMAGE_TOOLS: tuple[ToolSpec, ...] = (
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Сгенерировать картинку нейросетью. Промт пиши подробно и по-английски: объект, "
                "композиция, стиль, свет, палитра. Надписи на картинке выходят ненадёжно — "
                "в макетах верстай их в HTML. "
                "cutout=true — вырезать отдельный объект (чашку, фрукт, товар, растение, 3D-фигуру): бот "
                "сгенерирует его на белом фоне, уберёт фон и вернёт PNG с прозрачностью, обрезанный по "
                'объекту. Возвращает имя файла и размеры; в HTML: <img src="имя">.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Имя файла латиницей без расширения, например hero",
                    },
                    "prompt": {"type": "string", "description": "Подробный промт на английском"},
                    "size": {"type": "string", "enum": list(SIZES), "description": "Формат картинки"},
                    "cutout": {
                        "type": "boolean",
                        "description": "Вырезать объект с прозрачным фоном. Для фонов и сцен — false",
                    },
                },
                "required": ["name", "prompt"],
            },
        },
    },
)

EDIT_MAX_SIDE = 1024  # the source goes to FLUX scaled to fit this, sides in multiples of 64
EDIT_MIN_SIDE = 256


def edit_tool(recent: Sequence[str] = ()) -> ToolSpec:
    """The edit_image tool; `recent` are the names of pictures drawn earlier in this chat (editable too)."""
    earlier = f" Или имя ранее нарисованной картинки этого чата: {', '.join(recent)}." if recent else ""
    return {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": (
                "Отредактировать готовую картинку по просьбе пользователя: сменить фон, цвет, одежду, стиль, "
                "добавить или убрать объект. Исходник — фото, которое пользователь прислал в этом сообщении, "
                "или картинка, нарисованная раньше. Редактирует нейросеть FLUX (сама подбирает размер по "
                "исходнику). Промт — что изменить, по-английски, остальное должно остаться как есть. "
                'Возвращает имя файла и размеры; в HTML: <img src="имя">.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя нового файла латиницей без расширения"},
                    "prompt": {
                        "type": "string",
                        "description": "Что изменить, по-английски; всё остальное оставить как есть",
                    },
                    "source": {
                        "type": "string",
                        "description": "Номер присланного фото, 1 — первое (по умолчанию)." + earlier,
                    },
                },
                "required": ["name", "prompt"],
            },
        },
    }


class ImageError(Exception):
    """Generation failed; the message is safe to show the model."""


class ImageGenerator:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str = DEFAULT_MODEL,
        gemini_url: str | None = None,
        gemini_key: str = "",
        gemini_cookies: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._gemini_cookies = gemini_cookies
        self._transport = transport
        self._client = httpx.AsyncClient(
            base_url=CLOUDFLARE_URL.format(account=account_id),
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=TIMEOUT_S,
            transport=transport,
        )
        self._gemini = (
            httpx.AsyncClient(
                base_url=gemini_url,
                headers={"Authorization": f"Bearer {gemini_key}"},
                timeout=TIMEOUT_S,
                transport=transport,
            )
            if gemini_url
            else None
        )

    @property
    def has_gemini(self) -> bool:
        return self._gemini is not None

    async def generate(
        self, prompt: str, size: str = "square", model: ImageModel = ImageModel.FLUX
    ) -> tuple[bytes, ImageModel]:
        """The picture and who actually drew it: Gemini falls back to FLUX on any failure."""
        if model is ImageModel.GEMINI and self._gemini is not None:
            for name in GEMINI_MODELS:
                for _attempt in range(GEMINI_PAGE_TRIES):
                    try:
                        return await self._draw_gemini(name, prompt, size), ImageModel.GEMINI
                    except ImageError as exc:
                        logger.info("gemini image via %s failed (%s)", name, exc)
                        if str(exc) != PAGE_TIMEOUT:
                            break
        return await self._draw_flux(prompt, size), ImageModel.FLUX

    async def _draw_gemini(self, name: str, prompt: str, size: str) -> bytes:
        assert self._gemini is not None
        aspect, dims = GEMINI_SHAPES.get(size, GEMINI_SHAPES["square"])
        body = {
            "model": name,
            "prompt": prompt[:PROMPT_MAX_CHARS],
            "n": 1,
            "aspect_ratio": aspect,
            "size": dims,
        }
        try:
            response = await self._gemini.post("images/generations", json=body)
        except httpx.HTTPError as exc:
            raise ImageError(f"шлюз недоступен ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            logger.warning("gemini image %s HTTP %d: %s", name, response.status_code, response.text[:200])
            raise ImageError(
                PAGE_TIMEOUT if "page.goto: Timeout" in response.text else f"HTTP {response.status_code}"
            )
        try:
            item = response.json()["data"][0]
            if "b64_json" in item:
                return base64.b64decode(item["b64_json"], validate=True)
            url = str(item["url"])
        except (ValueError, KeyError, IndexError, TypeError, binascii.Error) as exc:
            raise ImageError("непонятный ответ") from exc
        return await self._download_gemini(url)

    async def _download_gemini(self, url: str) -> bytes:
        """gemini-web answers with a googleusercontent link that opens only with the account's cookies.

        The cookies file is written by the server's gemini-refresh timer (all Google cookies of the
        signed-in profile); it is read on every call, so a refresh needs no restart.
        """
        if self._gemini_cookies is None:
            raise ImageError("ссылка на картинку без файла cookie")
        try:
            saved = json.loads(self._gemini_cookies.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ImageError("нет файла cookie Gemini") from exc
        jar = httpx.Cookies()
        for c in saved:
            jar.set(c["name"], c["value"], domain=c["domain"], path=c.get("path", "/"))
        try:
            async with httpx.AsyncClient(
                cookies=jar, follow_redirects=True, timeout=TIMEOUT_S, transport=self._transport
            ) as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            raise ImageError(f"картинка не скачалась ({type(exc).__name__})") from exc
        if response.status_code >= 400 or not response.headers.get("content-type", "").startswith("image/"):
            raise ImageError(f"картинка не скачалась (HTTP {response.status_code})")
        return response.content

    async def edit(self, prompt: str, source: bytes) -> bytes:
        """The source picture changed as the prompt says; always FLUX (it keeps the source's proportions)."""
        try:
            with Image.open(io.BytesIO(source)) as src:
                rgb = src.convert("RGB")
        except OSError as exc:
            raise ImageError("исходник не открылся как картинка") from exc
        width, height = _fit_sides(*rgb.size)
        buf = io.BytesIO()
        rgb.resize((width, height)).save(buf, "PNG")
        form = {
            "prompt": (None, prompt[:PROMPT_MAX_CHARS]),
            "width": (None, str(width)),
            "height": (None, str(height)),
            "input_image_0": ("source.png", buf.getvalue(), "image/png"),
        }
        return await self._flux(form)

    async def _draw_flux(self, prompt: str, size: str) -> bytes:
        width, height = SIZES.get(size, SIZES["square"])
        # FLUX.2 models take multipart/form-data; (None, value) makes a plain field, not a file.
        form = {
            "prompt": (None, prompt[:PROMPT_MAX_CHARS]),
            "width": (None, str(width)),
            "height": (None, str(height)),
        }
        return await self._flux(form)

    async def _flux(self, form: dict) -> bytes:
        try:
            response = await self._client.post(self._model, files=form)
        except httpx.HTTPError as exc:
            logger.warning("cloudflare image request failed: %s", type(exc).__name__)
            raise ImageError("сервис генерации картинок недоступен") from exc
        if response.status_code == 429 or _is_quota_error(response):
            raise ImageError("дневной лимит генерации картинок исчерпан, он обновится в 03:00 по Москве")
        if response.status_code >= 400:
            logger.warning("cloudflare image HTTP %d: %s", response.status_code, response.text[:200])
            raise ImageError(f"сервис генерации ответил ошибкой {response.status_code}")
        try:
            body = response.json()
            result = body.get("result", body)
            return base64.b64decode(result["image"], validate=True)
        except (ValueError, KeyError, TypeError, AttributeError, binascii.Error) as exc:
            raise ImageError("сервис генерации вернул непонятный ответ") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._gemini is not None:
            await self._gemini.aclose()


def _fit_sides(width: int, height: int) -> tuple[int, int]:
    """The source's proportions, EDIT_MAX_SIDE at most, each side a multiple of 64 in the allowed range."""
    scale = min(1.0, EDIT_MAX_SIDE / max(width, height))

    def side(value: int) -> int:
        return min(EDIT_MAX_SIDE, max(EDIT_MIN_SIDE, round(value * scale / 64) * 64))

    return side(width), side(height)


def _is_quota_error(response: httpx.Response) -> bool:
    text = response.text.lower()
    return response.status_code >= 400 and ("neuron" in text or "daily" in text or "quota" in text)


@dataclass(frozen=True, slots=True)
class ImageRequest:
    name: str
    prompt: str
    size: str = "square"
    cutout: bool = False


def parse_image_call(call: ToolCall) -> ImageRequest:
    """Arguments of a generate_image call; ValueError on bad arguments."""
    args = json.loads(call.arguments or "{}")
    if not isinstance(args, dict) or not str(args.get("prompt", "")).strip():
        raise ValueError("prompt is required")
    cutout = args.get("cutout") in (True, "true", "True", 1)
    return ImageRequest(
        name=str(args.get("name") or "image"),
        prompt=str(args["prompt"]).strip(),
        size=str(args.get("size") or "square"),
        cutout=cutout,
    )


@dataclass(frozen=True, slots=True)
class EditRequest:
    name: str
    prompt: str
    source: str = "1"


def parse_edit_call(call: ToolCall) -> EditRequest:
    """Arguments of an edit_image call; ValueError on bad arguments."""
    args = json.loads(call.arguments or "{}")
    if not isinstance(args, dict) or not str(args.get("prompt", "")).strip():
        raise ValueError("prompt is required")
    return EditRequest(
        name=str(args.get("name") or "edited"),
        prompt=str(args["prompt"]).strip(),
        source=str(args.get("source") or "1").strip(),
    )


def cut_out_background(data: bytes) -> bytes | None:
    """Key out a plain light background touching the edges: trimmed PNG with alpha, or None.

    None when there is no such background (dark or busy corners) or when keying would eat almost
    everything (a white object on white): the caller then keeps the original picture.
    CPU-bound (~0.2 s for 1 MP): run it in a thread.
    """
    with Image.open(io.BytesIO(data)) as src:
        rgb = src.convert("RGB")
    width, height = rgb.size
    # Seeds all along the border: an object touching one edge must not cut the background in two.
    seeds = [(x, y) for x in range(0, width, 32) for y in (0, height - 1)]
    seeds += [(x, y) for y in range(0, height, 32) for x in (0, width - 1)]
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    if any(min(rgb.getpixel(c)) < BG_MIN_LIGHTNESS for c in corners):  # type: ignore[arg-type]
        return None

    marker = (
        255,
        0,
        254,
    )  # fill colour that marks "background"; a real photo pixel of it is vanishingly rare
    keyed = rgb.copy()
    for seed in seeds:
        pixel = keyed.getpixel(seed)
        if pixel != marker and min(pixel) >= BG_MIN_LIGHTNESS:  # type: ignore[arg-type]
            ImageDraw.floodfill(keyed, seed, marker, thresh=BG_THRESH)

    diff = ImageChops.difference(keyed, Image.new("RGB", keyed.size, marker))
    mask = diff.convert("L").point(lambda v: 255 if v == 0 else 0)  # 255 = background
    share = mask.histogram()[255] / (width * height)
    if not 0.05 <= share <= 0.95:
        return None

    alpha = mask.point(lambda v: 0 if v else 255)
    # Erode 1 px to drop the light halo of JPEG edges, then soften the cut line.
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    box = alpha.point(lambda v: 255 if v > 16 else 0).getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    rgba = rgba.crop(
        (
            max(0, left - CUTOUT_PADDING),
            max(0, top - CUTOUT_PADDING),
            min(width, right + CUTOUT_PADDING),
            min(height, bottom + CUTOUT_PADDING),
        )
    )
    out = io.BytesIO()
    rgba.save(out, "PNG", optimize=True)
    return out.getvalue()


def image_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def image_ext(data: bytes) -> str:
    return "png" if data.startswith(b"\x89PNG") else "jpg"
