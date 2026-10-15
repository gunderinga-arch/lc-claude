"""Internet access for the models: search through the local SearXNG, pages read by the bot itself.

The gateway only relays tool calls; the bot runs them. Every failure becomes a tool result text,
so the model can say "the search is down" instead of the whole answer failing.
SearXNG runs on the same server and pages are plain HTTP downloads, so neither is metered;
Serper (paid Google results) is only the fallback when SearXNG is down or finds nothing.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import re
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import lxml.etree
import lxml.html
import trafilatura

from bot.llm.base import ToolCall, ToolSpec

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev"
SEARCH_TIMEOUT_S = 20.0
SEARCH_RESULTS = 10
MAX_URLS_PER_CALL = 10
MAX_CRAWL_PAGES = 100
DEFAULT_CRAWL_PAGES = 30
CALL_MAX_CHARS = (
    60_000  # one tool result, ~15k tokens: every round resends it, and it counts against the quota
)
PAGE_MAX_BYTES = 3_000_000
FETCH_TIMEOUT_S = 15.0
FETCH_CONCURRENCY = 8  # per bot process: one crawl must not flood a site or the server's uplink
CRAWL_TIMEOUT_S = 90.0
MAX_REDIRECTS = 5
CHUNK_CHARS = 1_500
# Pages deep research reads (FETCH_CONCURRENCY at a time): more sides of the topic for the bigger budget.
RESEARCH_PAGES = 20
# Material in front of the model in deep research, on top of what it digs itself: ~50-100k tokens of a ~1M
# window (80k chars before 15.10.2026, when the history was capped at 128k tokens).
RESEARCH_MAX_CHARS = 200_000
# An honest bot name: a Chrome string with Python's TLS fingerprint gets 403 from Wikipedia and others.
USER_AGENT = "Mozilla/5.0 (compatible; LCClaudeBot/1.0)"
_SKIP_EXT = re.compile(
    r"\.(?:jpe?g|png|gif|webp|svg|ico|bmp|pdf|zip|rar|7z|gz|tar|mp[34]|avi|mov|webm|css|js|json|xml|exe|dmg|apk)$",
    re.I,
)

WEB_GUIDE = (
    "У тебя есть интернет: web_search (поиск, до 10 результатов), open_url (прочитать до 10 страниц целиком) "
    "и crawl_site (обойти сайт до 100 страниц и получить самые подходящие к вопросу фрагменты).\n"
    "Ищи, когда ответ зависит от свежих или проверяемых данных, когда пользователь просит поискать "
    "или прислал ссылку. На общие вопросы, код и творческие задачи отвечай сам, без поиска.\n"
    "На серьёзный вопрос ищи широко: несколько запросов с разных сторон (для тем про Россию по-русски, "
    "для остального можно и по-английски), потом открой 3–8 лучших страниц одним вызовом open_url "
    "с параметром query. Опирайся на найденное, а не на память; если источники расходятся, скажи об этом.\n"
    "В конце ответа, где использовал интернет, дай список источников ссылками вида [название](url). "
    "Не выдумывай ссылки — только те, что реально вернули инструменты."
)

WEB_TOOLS: tuple[ToolSpec, ...] = (
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Поиск в интернете. Возвращает до 10 результатов: заголовок, ссылка, короткий фрагмент. "
                "Для полного текста открой нужные страницы через open_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "description": "news — свежие новости, иначе general",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                f"Открыть до {MAX_URLS_PER_CALL} страниц и получить их текст. С query из длинных страниц "
                "вернутся самые подходящие к нему фрагменты, без query — начало каждой страницы."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "query": {"type": "string", "description": "Что ищешь на этих страницах (необязательно)"},
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_site",
            "description": (
                f"Обойти сайт по ссылкам с начальной страницы (до {MAX_CRAWL_PAGES} страниц того же сайта) "
                "и вернуть самые подходящие к query фрагменты. Для документации, каталогов, больших "
                "сайтов, когда нужного нет на одной странице. Занимает до полутора минут."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Начальная страница"},
                    "query": {"type": "string", "description": "Что ищешь на сайте"},
                    "max_pages": {"type": "integer", "description": f"По умолчанию {DEFAULT_CRAWL_PAGES}"},
                },
                "required": ["url", "query"],
            },
        },
    },
)


class PageError(Exception):
    """A page that could not be read; the message goes to the model as is."""


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    text: str
    links: list[str] = field(default_factory=list)


Resolver = Callable[[str, int], Awaitable[list[str]]]


async def _resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port)
    return [info[4][0] for info in infos]


class WebTools:
    def __init__(
        self,
        searxng_url: str,
        *,
        serper_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        resolve: Resolver = _resolve,
    ) -> None:
        self._search_client = httpx.AsyncClient(
            base_url=searxng_url, timeout=SEARCH_TIMEOUT_S, transport=transport
        )
        self._serper = (
            httpx.AsyncClient(
                base_url=SERPER_URL,
                headers={"X-API-KEY": serper_key},
                timeout=SEARCH_TIMEOUT_S,
                transport=transport,
            )
            if serper_key
            else None
        )
        # Redirects are followed by hand: every hop must pass the public-address check.
        self._pages = httpx.AsyncClient(
            timeout=FETCH_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
            transport=transport,
        )
        self._resolve = resolve
        self._slots = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def run(self, call: ToolCall) -> str:
        """Execute one tool call and return the text for the model. Never raises (except cancellation)."""
        try:
            args = json.loads(call.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments are not an object")
            if call.name == "web_search":
                return await self._search(str(args["query"]), str(args.get("topic") or "general"))
            if call.name == "open_url":
                urls = args["urls"]
                urls = [urls] if isinstance(urls, str) else [str(u) for u in urls]
                return await self._open(urls[:MAX_URLS_PER_CALL], str(args.get("query") or ""))
            if call.name == "crawl_site":
                pages = int(args.get("max_pages") or DEFAULT_CRAWL_PAGES)
                return await self._crawl(
                    str(args["url"]), str(args["query"]), max(1, min(pages, MAX_CRAWL_PAGES))
                )
            return f"Ошибка: неизвестный инструмент {call.name}"
        except (ValueError, KeyError, TypeError) as exc:
            return f"Ошибка: неверные аргументы инструмента ({type(exc).__name__})"
        except httpx.HTTPError as exc:
            # No URL or body in the log: the query is the user's text.
            logger.warning("web %s failed: %s", call.name, type(exc).__name__)
            return "Ошибка: интернет-поиск сейчас недоступен. Ответь без него и скажи об этом пользователю."

    # ---- search -----------------------------------------------------------------------------------

    async def _search(self, query: str, topic: str) -> str:
        lines = [
            f"[{i}] {r['title']}\n{r['url']}\n{r['text']}".rstrip()
            for i, r in enumerate(await self._rows(query, topic), start=1)
        ]
        return "\n\n".join(lines) or "Ничего не найдено."

    async def research(self, question: str, queries: list[str]) -> tuple[str, int]:
        """Deep research, run by the bot before the answer: every query searched at once, the best
        RESEARCH_PAGES results read (taken in turn from each query, so every side of the topic gets in),
        and the chunks closest to the question kept. Returns (material for the model, web calls made).
        """
        found = await asyncio.gather(*(self._rows(q, "general") for q in queries), return_exceptions=True)
        lists = [r for r in found if isinstance(r, list)]
        urls: list[str] = []
        for rank in range(SEARCH_RESULTS):
            for rows in lists:
                if rank < len(rows) and rows[rank]["url"] not in urls and len(urls) < RESEARCH_PAGES:
                    urls.append(rows[rank]["url"])
        pages = [p for p in await asyncio.gather(*(self._read(u) for u in urls)) if isinstance(p, Page)]
        calls = len(queries) + len(urls)
        if not pages:
            return "", calls
        best = rank_chunks(" ".join([question, *queries]), pages, RESEARCH_MAX_CHARS)
        return f"Прочитано страниц: {len(pages)} из {len(urls)}.\n\n{best}", calls

    async def _rows(self, query: str, topic: str) -> list[dict[str, str]]:
        """Up to SEARCH_RESULTS unique results as {title, url, text}.

        SearXNG first; Serper (paid Google, $1 per 1000) only when SearXNG is down or found nothing.
        """
        rows: list[dict[str, Any]] = []
        try:
            params = {"q": query, "format": "json"}
            if topic == "news":
                params["categories"] = "news"
            body = await _json(await self._search_client.get("/search", params=params))
            rows = [
                {"title": r.get("title"), "url": r.get("url"), "text": r.get("content")}
                for r in _items(body, "results")
            ]
        except httpx.HTTPError as exc:
            if self._serper is None:
                raise
            logger.warning("searxng failed, falling back to serper: %s", type(exc).__name__)
        if not rows and self._serper is not None:
            payload: dict[str, Any] = {"q": query, "num": SEARCH_RESULTS}
            if re.search(r"[а-яё]", query, re.I):
                payload |= {"gl": "ru", "hl": "ru"}
            path, key = ("/news", "news") if topic == "news" else ("/search", "organic")
            body = await _json(await self._serper.post(path, json=payload))
            rows = [
                {"title": r.get("title"), "url": r.get("link"), "text": r.get("snippet")}
                for r in _items(body, key)
            ]
        unique: dict[str, dict[str, str]] = {}
        for r in rows:
            url = r["url"]
            if isinstance(url, str) and url and url not in unique:
                unique[url] = {"title": str(r["title"] or ""), "url": url, "text": str(r["text"] or "")}
        return list(unique.values())[:SEARCH_RESULTS]

    # ---- reading pages ----------------------------------------------------------------------------

    async def _open(self, urls: list[str], query: str) -> str:
        if not urls:
            return "Ошибка: не передано ни одной ссылки."
        results = await asyncio.gather(*(self._read(u) for u in urls))
        pages = [r for r in results if isinstance(r, Page)]
        failed = [
            f"{u}\nНе удалось открыть страницу: {r}"
            for u, r in zip(urls, results, strict=True)
            if isinstance(r, str)
        ]
        if query and pages:
            body = rank_chunks(query, pages, CALL_MAX_CHARS)
        else:
            share = CALL_MAX_CHARS // max(1, len(pages))
            body = "\n\n---\n\n".join(
                f"{p.url}\n{p.text[:share]}" + ("\n[…текст обрезан]" if len(p.text) > share else "")
                for p in pages
            )
        return "\n\n---\n\n".join(x for x in (body, *failed) if x) or "Не удалось открыть страницы."

    async def _crawl(self, start: str, query: str, max_pages: int) -> str:
        host = _site(start)
        if not host:
            return "Ошибка: неверная ссылка."
        prefix = httpx.URL(start).path.rsplit("/", 1)[0] + "/"
        # Links under the start page's folder go first: /docs/guide/ should not drift into the blog.
        inside, outside = deque([start]), deque[str]()
        seen, pages, failed = {start}, list[Page](), 0
        deadline = asyncio.get_running_loop().time() + CRAWL_TIMEOUT_S
        while (inside or outside) and len(pages) < max_pages and asyncio.get_running_loop().time() < deadline:
            batch = []
            while (inside or outside) and len(batch) < min(FETCH_CONCURRENCY, max_pages - len(pages)):
                batch.append((inside or outside).popleft())
            for result in await asyncio.gather(*(self._read(u, links=True) for u in batch)):
                if isinstance(result, str):
                    failed += 1
                    continue
                pages.append(result)
                for link in result.links:
                    link = link.split("#", 1)[0]
                    url = httpx.URL(link) if link.startswith(("http://", "https://")) else None
                    if url is None or link in seen or _site(link) != host or _SKIP_EXT.search(url.path):
                        continue
                    seen.add(link)
                    (inside if url.path.startswith(prefix) else outside).append(link)
        if not pages:
            return "Не удалось открыть ни одной страницы сайта."
        head = f"Обойдено страниц: {len(pages)}" + (f", не открылось: {failed}" if failed else "")
        best = rank_chunks(query, pages, CALL_MAX_CHARS)
        return f"{head}. Ниже самые подходящие к запросу фрагменты.\n\n{best}"

    async def _read(self, url: str, *, links: bool = False) -> Page | str:
        """One page as text, or the reason it failed."""
        try:
            async with self._slots:
                final, ctype, body = await self._fetch(url)
            text, found = await asyncio.to_thread(_parse, final, ctype, body, links)
        except PageError as exc:
            return str(exc)
        except httpx.HTTPStatusError as exc:
            return f"ответ {exc.response.status_code}"
        except httpx.HTTPError as exc:
            return f"сетевая ошибка ({type(exc).__name__})"
        if not text.strip():
            return "на странице нет текста (возможно, она собирается скриптами)"
        return Page(final, text, found)

    async def _fetch(self, url: str) -> tuple[str, str, bytes]:
        for _ in range(MAX_REDIRECTS + 1):
            if not await self._public(url):
                raise PageError("адрес недоступен")
            async with self._pages.stream("GET", url) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers.get("location", "")))
                    continue
                response.raise_for_status()
                ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and "html" not in ctype and ctype != "text/plain":
                    raise PageError(f"это не страница ({ctype})")  # ponytail: PDFs skipped, pypdf is at hand
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) >= PAGE_MAX_BYTES:
                        break
                return str(response.url), ctype, bytes(body)
        raise PageError("слишком много перенаправлений")

    async def _public(self, url: str) -> bool:
        """Only public internet addresses: the model must not reach OmniRoute, SearXNG or the LAN.

        ponytail: checked before the request, so DNS rebinding between check and connect gets through;
        pin the resolved IP in the transport if that ever matters.
        """
        try:
            parsed = httpx.URL(url)
            if parsed.scheme not in ("http", "https") or not parsed.host:
                return False
            ips = await self._resolve(parsed.host, parsed.port or (443 if parsed.scheme == "https" else 80))
            return bool(ips) and all(ipaddress.ip_address(ip.split("%", 1)[0]).is_global for ip in ips)
        except (OSError, ValueError, httpx.InvalidURL):
            return False

    async def aclose(self) -> None:
        await self._search_client.aclose()
        await self._pages.aclose()
        if self._serper is not None:
            await self._serper.aclose()


async def _json(response: httpx.Response) -> Any:
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise httpx.DecodingError("search body is not JSON") from exc


def _items(body: Any, key: str) -> list[dict[str, Any]]:
    items = body.get(key) if isinstance(body, dict) else None
    return [i for i in items or [] if isinstance(i, dict)]


def _site(url: str) -> str:
    try:
        host = httpx.URL(url).host.lower()
    except httpx.InvalidURL:
        return ""
    return host.removeprefix("www.")


def _parse(url: str, ctype: str, body: bytes, links: bool) -> tuple[str, list[str]]:
    """CPU-bound, runs in a thread: main text as markdown, plus absolute links when crawling."""
    if ctype == "text/plain":
        return body.decode("utf-8", errors="replace"), []
    text = trafilatura.extract(
        body, url=url, output_format="markdown", include_tables=True, include_links=False, favor_recall=True
    )
    found: list[str] = []
    if links:
        try:
            doc = lxml.html.fromstring(body)
            doc.make_links_absolute(url, handle_failures="discard")
            found = [link for _el, attr, link, _pos in doc.iterlinks() if attr == "href"]
        except (lxml.etree.LxmlError, ValueError):
            pass
    return text or "", found


# ---- ranking -------------------------------------------------------------------------------------

_WORD = re.compile(r"\w{3,}")


def _terms(text: str) -> list[str]:
    # ponytail: a 6-letter prefix stands in for a stemmer (Russian endings); snowball if ranking misses
    return [w[:6] for w in _WORD.findall(text.lower())]


def _chunks(text: str) -> list[str]:
    """Paragraphs glued up to CHUNK_CHARS; an oversized paragraph is cut hard."""
    out, current = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        while len(para) > CHUNK_CHARS:
            out.append(para[:CHUNK_CHARS])
            para = para[CHUNK_CHARS:]
        if current and len(current) + len(para) > CHUNK_CHARS:
            out.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    return [*out, current] if current else out


def rank_chunks(query: str, pages: list[Page], budget: int) -> str:
    """BM25 over page chunks: the best ones that fit `budget`, shown per page in their original order."""
    chunks = [(p.url, c) for p in pages for c in _chunks(p.text) if c]
    if not chunks:
        return ""
    words = set(_terms(query))
    docs = [Counter(_terms(c)) for _, c in chunks]
    avg = sum(sum(d.values()) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in words & d.keys())
    idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(d: Counter[str]) -> float:
        size = sum(d.values())
        return sum(idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * size / avg)) for t in idf if t in d)

    order = sorted(range(len(chunks)), key=lambda i: -score(docs[i]))  # stable: ties keep page order
    picked, used = set(), 0
    for i in order:
        if used + len(chunks[i][1]) <= budget:
            picked.add(i)
            used += len(chunks[i][1])
    parts, last = [], None
    for i in sorted(picked):
        url, text = chunks[i]
        # Not "### url": the extracted markdown has its own headings, the source line must not look like one.
        parts.append(text if url == last else f"[Источник: {url}]\n{text}")
        last = url
    return "\n\n".join(parts)
