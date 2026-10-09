"""Turn raw HTML into clean, heading-aware Markdown.

Strategy:
  1. trafilatura — a battle-tested main-content extractor that drops menus,
     footers, sidebars and ads.
  2. A BeautifulSoup fallback that strips known chrome (nav/header/footer,
     cookie banners, share widgets...) and keeps only link-light text blocks.
The longer, more useful result wins.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

import trafilatura
from bs4 import BeautifulSoup

CHROME_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "nav", "header",
               "footer", "aside", "button", "input", "select", "template", "canvas"]
CHROME_ROLES = ["navigation", "banner", "contentinfo", "search", "dialog", "menu", "menubar"]
CHROME_HINT = re.compile(
    r"(^|[-_\s])(nav|navbar|menu|footer|header|sidebar|breadcrumbs?|cookie|consent|banner|"
    r"share|social|advert|ads?|promo|popup|modal|newsletter|subscribe|toc|skip)([-_\s]|$)",
    re.I,
)
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "dt", "dd"]
SPA_MARKERS = re.compile(r'id=["\'](root|app|__next|__nuxt|svelte)["\']|ng-version=|data-reactroot', re.I)


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    base = soup.find("base", href=True)
    base_url = urljoin(base_url, base["href"]) if base else base_url
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("mailto:", "tel:", "javascript:", "#")):
            links.append(urljoin(base_url, href))
    return links


def extract_title(html: str) -> str:
    meta = trafilatura.extract_metadata(html)
    if meta and meta.title:
        return meta.title.strip()
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _with_trafilatura(html: str, url: str) -> str:
    text = trafilatura.extract(
        html, url=url, output_format="markdown", include_tables=True, include_links=False,
        include_comments=False, include_images=False, favor_recall=True, deduplicate=True,
    )
    return text or ""


def _with_soup(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(CHROME_TAGS):
        tag.decompose()
    doomed = soup.find_all(attrs={"role": CHROME_ROLES})
    for el in soup.find_all(True):
        if el.name in ("html", "body", "main", "article"):
            continue
        hint = " ".join(el.get("class", [])) + " " + (el.get("id") or "")
        if CHROME_HINT.search(hint):
            doomed.append(el)
    for el in doomed:
        if not getattr(el, "decomposed", False):
            el.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all(BLOCK_TAGS):
        # Skip containers whose text will be picked up by a nested block anyway.
        if el.name not in ("pre",) and el.find(BLOCK_TAGS):
            continue
        if el.find_parent("pre"):
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("p", "li", "td", "dd"):
            link_chars = sum(len(a.get_text(strip=True)) for a in el.find_all("a"))
            if len(text) < 200 and link_chars / max(len(text), 1) > 0.6:
                continue  # link-heavy -> almost certainly navigation
        if el.name.startswith("h"):
            lines.append("#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        elif el.name == "pre":
            lines.append("```\n" + el.get_text().strip("\n") + "\n```")
        elif el.name == "blockquote":
            lines.append("> " + text)
        else:
            lines.append(text)
    return "\n\n".join(lines)


def extract_content(html: str, url: str) -> str:
    primary = _with_trafilatura(html, url)
    if len(primary) >= 400:
        return primary
    fallback = _with_soup(html)
    return fallback if len(fallback) > len(primary) else primary


def looks_js_rendered(html: str, text: str) -> bool:
    """Heuristic: the static HTML holds little readable text, or is a JS app shell."""
    return len(text) < 400 or (len(text) < 1500 and bool(SPA_MARKERS.search(html)))
