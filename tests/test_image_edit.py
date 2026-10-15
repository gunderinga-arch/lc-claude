"""Editing a picture: always FLUX with the source image, whatever image model the user picked."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace

import httpx
from PIL import Image

from bot.domain import ImageModel
from bot.llm.base import ToolCall
from bot.services.images import ImageError, ImageGenerator, _fit_sides, parse_edit_call
from tests.test_chat_service import FakeLLM, ok
from tests.test_images import JPEG, PNG, FakeImages, _service, tool_turn


def _photo(size=(1600, 900)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, "PNG")
    return buf.getvalue()


def _uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def test_fit_sides_keeps_proportions_in_multiples_of_64() -> None:
    assert _fit_sides(1600, 900) == (1024, 576)
    assert _fit_sides(640, 480) == (640, 512)
    assert _fit_sides(100, 100) == (256, 256)  # never below the minimum
    assert _fit_sides(4000, 100) == (1024, 256)


async def test_edit_posts_the_source_as_a_reference_image_to_flux() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["type"] = request.headers["content-type"]
        return httpx.Response(200, json={"result": {"image": base64.b64encode(JPEG).decode()}})

    gen = ImageGenerator("acc", "cf", gemini_url="http://omni/v1", transport=httpx.MockTransport(handler))
    assert await gen.edit("make the walls yellow", _photo()) == JPEG  # Gemini is configured, FLUX still edits
    body = seen["body"]
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="input_image_0"' in body and b"make the walls yellow" in body
    assert b'name="width"\r\n\r\n1024' in body and b'name="height"\r\n\r\n576' in body
    await gen.aclose()


async def test_edit_of_something_that_is_not_an_image_is_an_error() -> None:
    gen = ImageGenerator("acc", "cf", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    try:
        await gen.edit("x", b"not an image")
    except ImageError as exc:
        assert "картинк" in str(exc)
    else:
        raise AssertionError("expected ImageError")
    await gen.aclose()


def test_parse_edit_call() -> None:
    call = ToolCall("c", "edit_image", '{"name": "bg", "prompt": " blue sky ", "source": " 2 "}')
    request = parse_edit_call(call)
    assert (request.name, request.prompt, request.source) == ("bg", "blue sky", "2")
    assert parse_edit_call(ToolCall("c", "edit_image", '{"prompt": "x"}')).source == "1"


class EditImages(FakeImages):
    def __init__(self) -> None:
        super().__init__()
        self.edits: list[tuple[str, bytes]] = []

    async def edit(self, prompt: str, source: bytes) -> bytes:
        self.edits.append((prompt, source))
        return PNG


def edit(call_id: str, source: str | None = None, prompt: str = "blue background") -> ToolCall:
    args = {"name": "edited", "prompt": prompt, **({"source": source} if source else {})}
    return ToolCall(call_id, "edit_image", json.dumps(args))


async def test_sent_photo_is_edited_by_flux_even_without_the_image_skill() -> None:
    llm = FakeLLM(tool_turn(edit("c1")), ok("Готово, фон синий"))
    images = EditImages()
    db, service, user = await _service(llm, images)
    user = replace(user, skills=(), image_model=ImageModel.GEMINI)  # no skill, Gemini picked
    photo = _photo()

    reply = await service.answer(user, "сделай фон синим", images=(_uri(photo),))

    names = [t["function"]["name"] for t in llm.tools[0]]
    assert names == ["edit_image"]  # nothing to draw from scratch without the skill, but editing is offered
    assert images.edits == [("blue background", photo)]
    assert [(p.name, p.data) for p in reply.images] == [("edited.png", PNG)]
    assert reply.gemini_images == frozenset()  # FLUX did it, never marked as Gemini
    await db.close()


async def test_edit_is_not_offered_without_anything_to_edit() -> None:
    llm = FakeLLM(ok("привет"))
    db, service, user = await _service(llm, EditImages())
    await service.answer(replace(user, skills=()), "привет")
    assert llm.tools[0] == ()
    await db.close()


async def test_picture_drawn_earlier_can_be_edited_by_its_name() -> None:
    llm = FakeLLM(
        tool_turn(ToolCall("c1", "generate_image", json.dumps({"name": "cat", "prompt": "a cat"}))),
        ok("вот кот"),
        tool_turn(edit("c2", source="cat", prompt="add a hat")),
        ok("в шляпе"),
    )
    images = EditImages()
    db, service, user = await _service(llm, images)

    await service.answer(user, "нарисуй кота")
    reply = await service.answer(user, "добавь шляпу")

    edit_spec = next(t for t in llm.tools[2] if t["function"]["name"] == "edit_image")
    assert "cat.jpg" in edit_spec["function"]["parameters"]["properties"]["source"]["description"]
    assert images.edits == [("add a hat", JPEG)]
    assert [p.name for p in reply.images] == ["edited.png"]
    await db.close()


async def test_unknown_source_is_reported_to_the_model() -> None:
    llm = FakeLLM(tool_turn(edit("c1", source="7")), ok("не вышло"))
    images = EditImages()
    db, service, user = await _service(llm, images)
    await service.answer(user, "поправь", images=(_uri(_photo()),))
    result = next(m.content for m in llm.calls[1][0] if m.role == "tool")
    assert result.startswith("Ошибка: такой картинки нет") and images.edits == []
    await db.close()
