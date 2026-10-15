"""Image generation: the Cloudflare client, HTML assembly and the designer's tool loop."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image, ImageDraw

from bot.catalog import load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import ImageModel, Mode, RequestStatus, Role
from bot.files.blocks import embed_images, split_files
from bot.llm.base import ChatResult, ChatStatus, ToolCall
from bot.services.chat import DESIGNER_IMAGES_PER_ANSWER, MAX_IMAGES_PER_ANSWER, ChatService
from bot.services.conversation import ConversationService
from bot.services.images import (
    CUTOUT_SUFFIX,
    IMAGE_TOOLS,
    ImageError,
    ImageGenerator,
    ImageRequest,
    cut_out_background,
    parse_image_call,
)
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.skills import Skill
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo
from tests.test_chat_service import FakeLLM, ok

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
PNG = b"\x89PNG\r\n\x1a\nfake-png"


# ---- Cloudflare client -------------------------------------------------------------------------


def cloudflare(handler) -> ImageGenerator:
    return ImageGenerator("acc123", "cf-token", transport=httpx.MockTransport(handler))


async def test_generate_sends_multipart_and_decodes_image() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"result": {"image": base64.b64encode(JPEG).decode()}, "success": True}
        )

    gen = cloudflare(handler)
    assert await gen.generate("a cat", "landscape") == (JPEG, ImageModel.FLUX)
    await gen.aclose()
    request = seen[0]
    assert request.url.path == "/client/v4/accounts/acc123/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    assert request.headers["Authorization"] == "Bearer cf-token"
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    body = request.content.decode()
    assert 'name="prompt"\r\n\r\na cat' in body
    assert 'name="width"\r\n\r\n1344' in body and 'name="height"\r\n\r\n768' in body
    assert "filename" not in body  # plain fields, not files


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(429, json={"errors": [{"message": "rate"}]}), "лимит"),
        (httpx.Response(400, json={"errors": [{"message": "daily free allocation of neurons"}]}), "лимит"),
        (httpx.Response(500, text="boom"), "ошибкой 500"),
        (httpx.Response(200, json={"result": {}}), "непонятный"),
        (httpx.Response(200, json={"result": {"image": "!!not base64!!"}}), "непонятный"),
    ],
)
async def test_generate_errors_are_readable(response: httpx.Response, message: str) -> None:
    gen = cloudflare(lambda _r: response)
    with pytest.raises(ImageError, match=message):
        await gen.generate("x")
    await gen.aclose()


async def test_gemini_web_page_timeout_gets_one_retry() -> None:
    seen: list[str] = []
    timeouts = iter([True, False])

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model.startswith("gemini-web") and next(timeouts):
            return httpx.Response(400, json={"error": {"message": "page.goto: Timeout 20000ms exceeded."}})
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    gen = ImageGenerator("acc", "cf", gemini_url="http://omni/v1", transport=httpx.MockTransport(handler))
    assert await gen.generate("cat", "square", ImageModel.GEMINI) == (PNG, ImageModel.GEMINI)
    assert seen == ["gemini-web/nano-banana-web", "gemini-web/nano-banana-web"]  # not Antigravity's quota
    await gen.aclose()


async def test_gemini_web_link_is_downloaded_with_the_saved_cookies(tmp_path) -> None:
    link = "https://lh3.googleusercontent.com/gg-dl/abc"
    cookies = tmp_path / "gemini-cookies.json"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "omni":
            model = json.loads(request.content)["model"]
            if model.startswith("gemini-web"):
                return httpx.Response(200, json={"data": [{"url": link}]})
            return httpx.Response(429, json={"error": {"message": "quota"}})
        if str(request.url) == link:
            if "SID=s1" not in request.headers.get("cookie", ""):
                return httpx.Response(403, text="login")
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
        return httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()}})

    gen = ImageGenerator(
        "acc",
        "cf",
        gemini_url="http://omni/v1",
        gemini_cookies=cookies,
        transport=httpx.MockTransport(handler),
    )
    # No cookies file yet: the link cannot be opened, Antigravity is out of quota, FLUX draws it.
    assert await gen.generate("cat", "square", ImageModel.GEMINI) == (JPEG, ImageModel.FLUX)
    cookies.write_text(json.dumps([{"name": "SID", "value": "s1", "domain": ".googleusercontent.com"}]))
    assert await gen.generate("cat", "square", ImageModel.GEMINI) == (PNG, ImageModel.GEMINI)
    await gen.aclose()


async def test_gemini_tries_models_in_order_then_falls_back_to_flux() -> None:
    seen: list[str] = []
    gemini_ok = {"antigravity/gemini-3.1-flash-image"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "omni":
            body = json.loads(request.content)
            seen.append(body["model"])
            assert request.url.path == "/v1/images/generations"
            assert request.headers["Authorization"] == "Bearer omni-key"
            assert (body["aspect_ratio"], body["size"]) == ("9:16", "1024x1536")
            if body["model"] not in gemini_ok:
                return httpx.Response(429, json={"error": {"message": "quota"}})
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        seen.append("flux")
        return httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()}})

    gen = ImageGenerator(
        "acc",
        "cf",
        gemini_url="http://omni/v1",
        gemini_key="omni-key",
        transport=httpx.MockTransport(handler),
    )
    assert await gen.generate("a cat", "portrait", ImageModel.GEMINI) == (PNG, ImageModel.GEMINI)
    assert seen == ["gemini-web/nano-banana-web", "antigravity/gemini-3.1-flash-image"]

    gemini_ok.clear()  # every Gemini model out of quota: FLUX draws it
    seen.clear()
    assert await gen.generate("a cat", "portrait", ImageModel.GEMINI) == (JPEG, ImageModel.FLUX)
    assert seen == ["gemini-web/nano-banana-web", "antigravity/gemini-3.1-flash-image", "flux"]
    await gen.aclose()


def test_parse_image_call() -> None:
    call = ToolCall("c", "generate_image", '{"name": "hero", "prompt": " sea ", "size": "banner"}')
    assert parse_image_call(call) == ImageRequest("hero", "sea", "banner", cutout=False)
    with_cutout = ToolCall("c", "generate_image", '{"prompt": "x", "cutout": true}')
    assert parse_image_call(with_cutout) == ImageRequest("image", "x", "square", cutout=True)
    with pytest.raises(ValueError):
        parse_image_call(ToolCall("c", "generate_image", '{"name": "a"}'))


# ---- HTML assembly -----------------------------------------------------------------------------


def test_full_html_page_in_code_fence_becomes_a_file() -> None:
    page = "<!DOCTYPE html>\n<html><body>hi</body></html>"
    text, files = split_files(f"Вот макет:\n```html\n{page}\n```\nГотово.")
    assert text == "Вот макет:\n📎 page.html\nГотово."
    assert files[0].name == "page.html" and files[0].data.decode() == page

    snippet = "```html\n<div>кнопка</div>\n```"
    assert split_files(snippet) == (snippet, [])  # a snippet stays in the chat

    inside = f'<<<FILE name="a.md">>>\n```html\n{page}\n```\n<<<END>>>'
    _, files = split_files(inside)
    assert [f.name for f in files] == ["a.md"]  # no nested promotion


def test_embed_images_replaces_only_image_references() -> None:
    html = (
        '<img src="hero.jpg"><img src=\'hero\'><div style="background:url(hero.jpg)"></div>'
        '<a href="hero.jpg">x</a><img src="hero-2.jpg"><img src="other.jpg"><p>hero.jpg</p>'
    )
    out = embed_images(html, {"hero.jpg": JPEG, "icon.png": PNG})
    uri = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    assert out.count(uri) == 4
    assert 'src="hero-2.jpg"' in out and 'src="other.jpg"' in out and "<p>hero.jpg</p>" in out


# ---- designer tool loop ------------------------------------------------------------------------


class FakeImages:
    def __init__(self, fail: bool = False) -> None:
        self.prompts: list[tuple[str, str]] = []
        self.fail = fail

    async def generate(
        self, prompt: str, size: str = "square", model: ImageModel = ImageModel.FLUX
    ) -> tuple[bytes, ImageModel]:
        self.prompts.append((prompt, size))
        if self.fail:
            raise ImageError("дневной лимит генерации картинок исчерпан")
        return PNG if "icon" in prompt else JPEG, model


def draw(call_id: str, name: str, prompt: str) -> ToolCall:
    return ToolCall(call_id, "generate_image", json.dumps({"name": name, "prompt": prompt}))


def tool_turn(*calls: ToolCall) -> ChatResult:
    return ChatResult(ChatStatus.OK, "", "claude-opus-4.8", 1, 10, False, tool_calls=calls)


async def _service(llm: FakeLLM, images: FakeImages):
    db = Database(":memory:")
    await db.connect()
    user = await UserRepo(db).upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    skills = {"design": Skill("design", "🎨", "DESIGN", images=True)}
    conversations = ConversationService(
        ConversationRepo(db), system_prompt="sys", token_budget=50_000, skills=skills
    )
    service = ChatService(
        llm,
        conversations,
        QuotaService(UsageRepo(db), daily_limit=10),
        ModelRouter(catalog),
        RequestRepo(db),
        catalog,
        None,
        images,  # type: ignore[arg-type]
    )
    return db, service, replace(user, skills=("design",))


HTML_ANSWER = (
    'Макет готов.\n<<<FILE name="design.html">>>\n'
    '<html><img src="hero.jpg"><img src="icon.png"></html>\n<<<END>>>'
)


async def test_designer_generates_images_and_embeds_them_into_html() -> None:
    llm = FakeLLM(
        tool_turn(draw("c1", "Hero", "sea at dawn"), draw("c2", "icon", "coffee icon")),
        ok(HTML_ANSWER),
        ok('<<<FILE name="design.html">>>\n<html><img src="hero.jpg"></html>\n<<<END>>>'),
    )
    images = FakeImages()
    db, service, user = await _service(llm, images)

    reply = await service.answer(user, "сделай лендинг кофейни")
    assert reply.status is RequestStatus.OK
    assert llm.tools[0] == IMAGE_TOOLS
    assert [(p.name, p.data) for p in reply.images] == [("hero.jpg", JPEG), ("icon.png", PNG)]
    tool_results = [m.content for m in llm.calls[1][0] if m.role == "tool"]
    assert tool_results[0].startswith("Готово: hero.jpg")
    html = reply.files[0].data.decode()
    assert "data:image/jpeg;base64," in html and "data:image/png;base64," in html
    assert "hero.jpg" not in html

    # The follow-up edit re-uses the old names: they are still embedded, and history holds no base64.
    follow = await service.answer(user, "сделай заголовок крупнее")
    assert follow.images == [] and "data:image/jpeg;base64," in follow.files[0].data.decode()
    history = " ".join(m.content for m in llm.calls[2][0])
    assert "base64" not in history and 'src="hero.jpg"' in history
    await db.close()


async def test_users_image_model_reaches_the_generator_and_marks_gemini_files() -> None:
    llm = FakeLLM(
        tool_turn(draw("c1", "cat", "a cat")), ok("готово"), tool_turn(draw("c2", "dog", "a dog")), ok("ок")
    )
    images = FakeImages()
    db, service, user = await _service(llm, images)
    assert user.image_model is ImageModel.GEMINI  # the column's default

    reply = await service.answer(user, "нарисуй кота")
    assert reply.gemini_images == {"cat.jpg"}

    await UserRepo(db).set_image_model(1, ImageModel.FLUX)
    stored = await UserRepo(db).get(1)
    assert stored is not None and stored.image_model is ImageModel.FLUX
    reply = await service.answer(replace(stored, skills=user.skills), "нарисуй собаку")
    assert [p.name for p in reply.images] == ["dog.jpg"] and reply.gemini_images == frozenset()
    await db.close()


async def test_six_images_in_one_round_all_run() -> None:
    # The designer prompt allows 6 images per batch: the per-round cap is for web calls only.
    llm = FakeLLM(tool_turn(*(draw(f"c{i}", f"img{i}", f"object {i}") for i in range(6))), ok("готово"))
    images = FakeImages()
    db, service, user = await _service(llm, images)

    reply = await service.answer(user, "сделай лендинг")
    assert len(images.prompts) == 6
    assert len(reply.images) == 6
    await db.close()


async def test_image_limit_names_and_failures() -> None:
    calls = [draw(f"c{i}", "Фон!", f"bg {i}") for i in range(4)]
    more = [draw(f"d{i}", "bg", f"more {i}") for i in range(4)]
    llm = FakeLLM(tool_turn(*calls), tool_turn(*more), ok("готово"))
    images = FakeImages()
    db, service, user = await _service(llm, images)
    reply = await service.answer(user, "много фонов")
    assert len(reply.images) == MAX_IMAGES_PER_ANSWER == len(images.prompts)
    assert [p.name for p in reply.images][:3] == ["image.jpg", "image-2.jpg", "image-3.jpg"]
    last_round = [m.content for m in llm.calls[2][0] if m.role == "tool"]
    assert sum("Лимит" in c for c in last_round) == 2
    await db.close()

    llm = FakeLLM(tool_turn(draw("c1", "hero", "x")), ok("без картинки"))
    db, service, user = await _service(llm, FakeImages(fail=True))
    reply = await service.answer(user, "картинку")
    assert reply.status is RequestStatus.OK and reply.images == []
    assert "Ошибка генерации: дневной лимит" in next(m.content for m in llm.calls[1][0] if m.role == "tool")
    await db.close()


# ---- cutouts -----------------------------------------------------------------------------------


def picture(background: tuple[int, int, int], fmt: str = "JPEG", size: int = 200) -> bytes:
    """A red disc on a plain background, JPEG-compressed like a generator's output."""
    img = Image.new("RGB", (size, size), background)
    ImageDraw.Draw(img).ellipse((60, 50, 140, 150), fill=(200, 40, 30))
    out = io.BytesIO()
    img.save(out, fmt, quality=85)
    return out.getvalue()


def test_cutout_removes_white_background_and_trims() -> None:
    png = cut_out_background(picture((255, 255, 255)))
    assert png is not None and png.startswith(b"\x89PNG")
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA"
    assert img.size[0] < 110 and img.size[1] < 120  # trimmed to the disc (80x100) plus padding
    assert img.getpixel((0, 0))[3] == 0  # corner is transparent
    centre = img.getpixel((img.size[0] // 2, img.size[1] // 2))
    assert centre[3] == 255 and centre[0] > 150  # the red object is intact


def test_cutout_touching_edge_keeps_background_connected() -> None:
    img = Image.new("RGB", (200, 200), (250, 250, 250))
    ImageDraw.Draw(img).rectangle((80, 0, 120, 199), fill=(20, 90, 200))  # splits the frame in two
    out = io.BytesIO()
    img.save(out, "PNG")
    cut = Image.open(io.BytesIO(cut_out_background(out.getvalue())))
    assert cut.getpixel((0, 100))[3] == 0 and cut.getpixel((cut.size[0] - 1, 100))[3] == 0


def test_cutout_gives_up_when_keying_makes_no_sense() -> None:
    assert cut_out_background(picture((30, 30, 30))) is None  # dark background: nothing to key out
    blank = io.BytesIO()
    Image.new("RGB", (100, 100), (255, 255, 255)).save(blank, "PNG")
    assert cut_out_background(blank.getvalue()) is None  # only background: the object would vanish


async def test_designer_cutout_call_returns_transparent_png() -> None:
    class RealImages(FakeImages):
        async def generate(
            self, prompt: str, size: str = "square", model: ImageModel = ImageModel.FLUX
        ) -> tuple[bytes, ImageModel]:
            self.prompts.append((prompt, size))
            return picture((255, 255, 255)), model

    args = {"name": "cup", "prompt": "a red cup", "cutout": True}
    call = ToolCall("c1", "generate_image", json.dumps(args))
    llm = FakeLLM(tool_turn(call), ok("готово"))
    images = RealImages()
    db, service, user = await _service(llm, images)
    reply = await service.answer(user, "чашку без фона")
    assert images.prompts[0][0] == "a red cup" + CUTOUT_SUFFIX
    assert [p.name for p in reply.images] == ["cup.png"]
    result = next(m.content for m in llm.calls[1][0] if m.role == "tool")
    assert result.startswith("Готово: cup.png, ") and "прозрачный" in result
    await db.close()


async def test_designer_draws_three_per_answer_and_six_a_day() -> None:
    four = [draw(f"c{i}", f"a{i}", f"x {i}") for i in range(4)]
    llm = FakeLLM(
        tool_turn(*four), ok("готово"), tool_turn(*four), ok("готово"), tool_turn(*four), ok("готово")
    )
    images = FakeImages()
    db, service, user = await _service(llm, images)
    user = replace(user, skills=("designer",))
    service._conversations._skills["designer"] = Skill("designer", "🎨", "DESIGN", images=True)

    first = await service.answer(user, "сайт кофейни")
    assert len(first.images) == DESIGNER_IMAGES_PER_ANSWER == 3
    second = await service.answer(user, "презентация")
    assert len(second.images) == 3  # 6 today: the daily limit is reached
    third = await service.answer(user, "ещё баннер")
    assert third.images == []
    refusals = [m.content for m in llm.calls[5][0] if m.role == "tool"]
    assert all("на сегодня исчерпан" in c for c in refusals)
    assert await UsageRepo(db).images_today(1) == 6

    admin = replace(user, role=Role.ADMIN)  # admins draw without a daily limit
    assert await service._quota.images_left(admin) is None
    await db.close()
