"""Polite, same-site, breadth-first crawler with optional JavaScript rendering."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .config import USER_AGENT
from .extractor import extract_content, extract_links, extract_title, looks_js_rendered

SKIP_EXT = re.compile(
    r"\.(png|jpe?g|gif|svg|webp|ico|bmp|pdf|zip|gz|tgz|tar|rar|7z|mp4|mp3|wav|avi|mov|webm|"
    r"css|js|mjs|json|xml|rss|atom|woff2?|ttf|eot|otf|exe|dmg|pkg|apk|csv|xlsx?|docx?|pptx?)$",
    re.I,
)
SKIP_PATH = re.compile(r"/(login|signin|sign-in|signup|sign-up|logout|cart|checkout|account|wp-admin|wp-json)(/|$)", re.I)
TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src$)", re.I)

Emit = Callable[..., Awaitable[None]]


@dataclass
class Page:
    url: str
    title: str
    text: str  # extracted markdown
    depth: int
    rendered: bool = False
    links: list[str] = field(default_factory=list)


def normalize_url(url: str) -> str:
    p = urlparse(url)
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not TRACKING_PARAMS.match(k)])
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def url_key(url: str) -> str:
    """Identity used for de-duplication: `/docs` and `/docs/` are the same page."""
    url = normalize_url(url)
    return url[:-1] if url.endswith("/") and urlparse(url).path != "/" else url


def _host(netloc: str) -> str:
    return netloc.lower().removeprefix("www.")


class JSRenderer:
    """Lazily starts a headless Chromium via Playwright (optional dependency)."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._lock = asyncio.Lock()
        self.available: bool | None = None
        self.error = ""

    async def _start(self):
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch()
            self.available = True
        except Exception as e:  # not installed, or browsers not downloaded
            self.available = False
            self.error = str(e).splitlines()[0][:200]

    async def render(self, url: str) -> str | None:
        async with self._lock:
            if self.available is None:
                await self._start()
        if not self.available:
            return None
        page = await self._browser.new_page(user_agent=USER_AGENT)
        try:
            await page.goto(url, wait_until="networkidle", timeout=25_000)
            return await page.content()
        except Exception:
            return None
        finally:
            await page.close()

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()


class Crawler:
    def __init__(self, start_url: str, *, max_pages: int = 25, max_depth: int = 2,
                 same_path: bool = False, js_mode: str = "auto", concurrency: int = 6,
                 emit: Emit | None = None):
        if not urlparse(start_url).scheme:
            start_url = "https://" + start_url
        self.start_url = normalize_url(start_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.js_mode = js_mode  # "off" | "auto" | "always"
        self.concurrency = concurrency
        self.emit = emit or (lambda *a, **k: asyncio.sleep(0))
        start = urlparse(self.start_url)
        self.host = _host(start.netloc)
        # "Stay within path": /docs/guide/intro -> only crawl under /docs/guide/
        self.path_prefix = start.path[: start.path.rfind("/") + 1] if same_path else ""
        self.pages: list[Page] = []
        self.seen: set[str] = set()
        self.failed = 0
        self.robots: robotparser.RobotFileParser | None = None
        self.renderer = JSRenderer()
        self._js_warned = False

    def in_scope(self, url: str) -> bool:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or _host(p.netloc) != self.host:
            return False
        if SKIP_EXT.search(p.path) or SKIP_PATH.search(p.path):
            return False
        if self.path_prefix and not (p.path + "/").startswith(self.path_prefix):
            return False
        if self.robots and not self.robots.can_fetch(USER_AGENT, url):
            return False
        return True

    async def _load_robots(self, client: httpx.AsyncClient):
        p = urlparse(self.start_url)
        try:
            r = await client.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=10)
            if r.status_code == 200:
                rp = robotparser.RobotFileParser()
                rp.parse(r.text.splitlines())
                self.robots = rp
        except httpx.HTTPError:
            pass

    async def _sitemap_urls(self, client: httpx.AsyncClient) -> list[str]:
        p = urlparse(self.start_url)
        try:
            r = await client.get(f"{p.scheme}://{p.netloc}/sitemap.xml", timeout=10)
            if r.status_code != 200 or "<urlset" not in r.text:
                return []
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
            return [normalize_url(u) for u in locs[: self.max_pages * 4]]
        except httpx.HTTPError:
            return []

    async def _fetch_page(self, client: httpx.AsyncClient, url: str, depth: int) -> Page | None:
        html, final_url, rendered = "", url, False
        if self.js_mode != "always":
            r = await client.get(url)
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            if "html" not in r.headers.get("content-type", "html"):
                return None
            html, final_url = r.text, normalize_url(str(r.url))
            if url_key(final_url) != url_key(url):
                if url_key(final_url) in self.seen or not self.in_scope(final_url):
                    return None  # redirected to a page we already have, or off-site
                self.seen.add(url_key(final_url))

        text = extract_content(html, final_url) if html else ""
        if self.js_mode == "always" or (self.js_mode == "auto" and looks_js_rendered(html, text)):
            js_html = await self.renderer.render(final_url)
            if js_html:
                js_text = extract_content(js_html, final_url)
                if len(js_text) > len(text):
                    html, text, rendered = js_html, js_text, True
            elif self.renderer.available is False and not self._js_warned:
                self._js_warned = True
                await self.emit("log", level="warn",
                                message="JavaScript rendering unavailable (install Playwright to enable). "
                                        "Continuing with static HTML.")
        if not html:
            return None
        return Page(url=final_url, title=extract_title(html) or final_url, text=text, depth=depth,
                    rendered=rendered, links=extract_links(html, final_url))

    async def run(self) -> list[Page]:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        limits = httpx.Limits(max_connections=self.concurrency)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=20, limits=limits) as client:
            await self._load_robots(client)
            queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
            self.seen.add(url_key(self.start_url))
            queue.put_nowait((self.start_url, 0))
            if self.max_depth > 0:
                sitemap = [u for u in await self._sitemap_urls(client)
                           if url_key(u) not in self.seen and self.in_scope(u)]
                if sitemap:
                    await self.emit("log", level="info", message=f"Found sitemap.xml with {len(sitemap)} in-scope URLs")
                for u in sitemap:
                    self.seen.add(url_key(u))
                    queue.put_nowait((u, 1))

            async def worker():
                while True:
                    url, depth = await queue.get()
                    try:
                        if len(self.pages) >= self.max_pages:
                            continue
                        page = await self._fetch_page(client, url, depth)
                        if not page or len(self.pages) >= self.max_pages:
                            continue
                        self.pages.append(page)
                        await self.emit("page", url=page.url, title=page.title, words=len(page.text.split()),
                                        rendered=page.rendered, count=len(self.pages), max=self.max_pages)
                        if depth < self.max_depth:
                            for link in page.links:
                                link = normalize_url(link)
                                if url_key(link) not in self.seen and self.in_scope(link):
                                    self.seen.add(url_key(link))
                                    queue.put_nowait((link, depth + 1))
                    except Exception as e:
                        self.failed += 1
                        await self.emit("log", level="warn", message=f"Skipped {url} — {str(e)[:120]}")
                    finally:
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
            try:
                await queue.join()
            finally:
                for w in workers:
                    w.cancel()
                await self.renderer.close()
        return self.pages
