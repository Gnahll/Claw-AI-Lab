"""Resolve publisher landing pages using their machine-readable citation metadata."""
from __future__ import annotations

import hashlib
import gzip
import io
from html import unescape
from html.parser import HTMLParser
import logging
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)


class CitationMetadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        data = dict(attrs)
        name = (data.get("name") or data.get("property") or "").lower()
        content = " ".join(unescape(data.get("content") or "").split())
        if name and content:
            self.values.setdefault(name, []).append(content)


def paper_from_html(url: str, html: str) -> Paper | None:
    parser = CitationMetadata()
    parser.feed(html)
    values = parser.values

    def first(*names: str) -> str:
        return next((values[name][0] for name in names if values.get(name)), "")

    # A generic HTML <title> (login/challenge/404) is not a paper title.
    title = first("citation_title", "dc.title", "dc.title.article")
    if not title or not any(c.isalpha() for c in title):
        return None
    date = first("citation_publication_date", "citation_date", "dc.date.issued", "dc.date")
    year = re.search(r"\b(?:19|20)\d{2}\b", date)
    authors = values.get("citation_author") or values.get("dc.creator.personalname") or values.get("dc.creator") or []
    abstract = first("citation_abstract", "dc.description", "description", "og:description")
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    return Paper(
        paper_id="user-ref-" + hashlib.sha256(url.encode()).hexdigest()[:16],
        title=title, authors=tuple(Author(name=name) for name in authors),
        year=int(year.group()) if year else 0,
        abstract=" ".join(abstract.split()),
        venue=first("citation_conference_title", "citation_journal_title", "dc.source"),
        doi=first("citation_doi", "dc.identifier.doi"), url=url, source="user_reference",
    )


def resolve_publisher_reference(url: str) -> Paper | None:
    if urlsplit(url).scheme not in {"http", "https"}:
        return None
    try:
        request = Request(url, headers={"User-Agent": "ResearchClaw/0.3 (academic reference metadata)"})
        with urlopen(request, timeout=12) as response:
            raw = response.read(2 * 1024 * 1024)
            # Some publisher/CDN responses are gzip even without negotiation;
            # urllib does not decode content encodings automatically.
            if raw.startswith(b"\x1f\x8b"):
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                    raw = compressed.read(2 * 1024 * 1024)
            html = raw.decode("utf-8", errors="replace")
        return paper_from_html(url, html)
    except (OSError, ValueError) as exc:
        logger.warning("Publisher reference metadata unavailable for %s (%s)", url, type(exc).__name__)
        return None
