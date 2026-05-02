"""Wikipedia API client and text-file helpers.

This module fetches plain-text Wikipedia extracts through the MediaWiki API.
It does not perform chunking, embedding, vector storage, or answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import quote

import requests


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "LocalWikipediaRAGAssistant/1.0 "
    "(AI Aided Computer Engineering HW3)"
)
UNWANTED_SECTION_HEADINGS = {
    "see also",
    "notes",
    "references",
    "works cited",
    "further reading",
    "external links",
    "bibliography",
    "sources",
    "citations",
    "publications",
    "selected publications",
    "film and television",
    "in popular culture",
    "popular culture",
    "commemoration",
    "honours",
    "honors",
    "awards and honours",
    "awards and honors",
}


@dataclass(frozen=True)
class WikipediaPage:
    """Plain-text content and metadata for one Wikipedia page."""

    title: str
    page_id: int | None
    source_url: str | None
    extract: str


class WikipediaClientError(Exception):
    """Base exception for Wikipedia client failures."""


class WikipediaPageNotFoundError(WikipediaClientError):
    """Raised when Wikipedia returns a missing or unusable page."""


class WikipediaClient:
    """Small MediaWiki API client for fetching page extracts."""

    def __init__(
        self,
        base_url: str = WIKIPEDIA_API_URL,
        timeout_seconds: int = 20,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        """Create a Wikipedia client.

        Args:
            base_url: MediaWiki API endpoint.
            timeout_seconds: HTTP request timeout.
            user_agent: User-Agent header sent to Wikipedia.
        """

        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def fetch_page(self, title: str) -> WikipediaPage:
        """Fetch a full plain-text Wikipedia extract by page title."""

        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title must not be blank.")

        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "inprop": "url",
            "titles": clean_title,
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
        }
        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WikipediaClientError(f"Wikipedia request failed: {exc}") from exc

        if response.status_code != 200:
            raise WikipediaClientError(
                "Wikipedia request returned "
                f"HTTP {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WikipediaClientError("Wikipedia response was not valid JSON.") from exc

        page = self._extract_page(payload, clean_title)
        extract = str(page.get("extract") or "").strip()

        if page.get("missing") or not extract:
            raise WikipediaPageNotFoundError(
                f"Wikipedia page not found or empty: {clean_title}"
            )

        page_id = page.get("pageid")
        if page_id is not None:
            try:
                page_id = int(page_id)
            except (TypeError, ValueError) as exc:
                raise WikipediaClientError("Wikipedia page id was invalid.") from exc

        return WikipediaPage(
            title=str(page.get("title") or clean_title).strip(),
            page_id=page_id,
            source_url=page.get("fullurl"),
            extract=extract,
        )

    def _extract_page(self, payload: object, requested_title: str) -> dict:
        """Extract the first page object from a MediaWiki API payload."""

        if not isinstance(payload, dict):
            raise WikipediaClientError("Wikipedia response JSON had invalid shape.")

        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            info = error.get("info", "No additional error information.")
            raise WikipediaClientError(f"Wikipedia API error {code}: {info}")

        query = payload.get("query")
        if not isinstance(query, dict):
            raise WikipediaClientError("Wikipedia response did not contain query data.")

        pages = query.get("pages")
        if not isinstance(pages, list) or not pages:
            raise WikipediaClientError("Wikipedia response did not contain page data.")

        page = pages[0]
        if not isinstance(page, dict):
            raise WikipediaClientError("Wikipedia page data had invalid shape.")

        if "title" not in page:
            page["title"] = requested_title

        return page


def fetch_wikipedia_page(title: str) -> WikipediaPage:
    """Fetch a Wikipedia page using the default client."""

    return WikipediaClient().fetch_page(title)


def build_wikipedia_page_url(title: str) -> str:
    """Build a normal English Wikipedia page URL for a title."""

    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title must not be blank.")

    page_title = clean_title.replace(" ", "_")
    encoded_title = quote(page_title, safe="._-()")
    return f"https://en.wikipedia.org/wiki/{encoded_title}"


def safe_filename(name: str) -> str:
    """Convert an entity name into a stable, simple filename stem."""

    lowered_name = name.strip().lower()
    filename_parts: list[str] = []

    for character in lowered_name:
        if character.isalnum():
            filename_parts.append(character)
        elif character.isspace() or character in {"-", "_"}:
            filename_parts.append("_")

    filename = "".join(filename_parts)
    filename = re.sub(r"_+", "_", filename).strip("_")
    return filename or "entity"


def normalize_wikipedia_text(text: str) -> str:
    """Normalize line endings, remove footers, and collapse blank lines."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = remove_unwanted_wikipedia_sections(normalized)
    output_lines: list[str] = []
    previous_blank = False

    for line in normalized.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            if not previous_blank:
                output_lines.append("")
            previous_blank = True
            continue

        output_lines.append(clean_line)
        previous_blank = False

    return "\n".join(output_lines).strip()


def remove_unwanted_wikipedia_sections(text: str) -> str:
    """Remove trailing Wikipedia footer/reference sections from plain text."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    kept_lines: list[str] = []

    for line in normalized.split("\n"):
        clean_line = line.strip()
        if clean_line.casefold() in UNWANTED_SECTION_HEADINGS:
            break
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def write_text_file(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text_file(path: Path) -> str:
    """Read UTF-8 text from a local file."""

    return path.read_text(encoding="utf-8")
