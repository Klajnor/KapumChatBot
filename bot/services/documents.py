"""Text extraction from uploaded documents and linked web pages.

Heavy parsers are optional dependencies: ``pdfplumber`` (PDF), ``python-docx``
(DOCX) and ``trafilatura`` (high-quality article extraction). Each import is
guarded, so the bot runs without them and degrades gracefully:

* a PDF/DOCX upload is rejected with an explanation when its parser is
  missing (plain text never needs one);
* link extraction falls back to a built-in stdlib HTML parser.

URL fetching is SSRF-guarded: only ``http(s)`` URLs whose host resolves to a
public IP are fetched, redirects are followed manually and validated hop by
hop, and downloads are size-capped. Each hop is resolved once, screened, and
the request is then pinned to exactly the validated addresses (a custom
resolver), so a rebinding DNS answer between the check and the request cannot
redirect it to an internal host.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp

from bot.config import config

logger = logging.getLogger(__name__)

# --- File type detection ----------------------------------------------------

# Extensions treated as plain text (decoded and forwarded to the model as is).
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".log", ".rst", ".tex",
    ".csv", ".tsv", ".json", ".xml", ".yml", ".yaml", ".ini", ".cfg", ".toml",
    ".html", ".htm", ".css", ".sql",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bat", ".ps1",
})

_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def detect_file_type(file_name: str | None, mime_type: str | None = None) -> str | None:
    """Classify an upload as ``"pdf"``, ``"docx"``, ``"txt"`` or None.

    The extension wins; the MIME type is the fallback for nameless files.
    Legacy ``.doc``/``.docx``-in-``.doc``, spreadsheets and archives are not
    supported and return None.
    """
    name = (file_name or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    for extension in _TEXT_EXTENSIONS:
        if name.endswith(extension):
            return "txt"
    if mime_type:
        mime = mime_type.lower().split(";")[0].strip()
        if mime in _MIME_TYPES:
            return _MIME_TYPES[mime]
        if mime.startswith("text/"):
            return "txt"
    return None


# --- Text decoding ----------------------------------------------------------


def decode_text_bytes(data: bytes) -> str:
    """Decode text bytes: BOM/UTF-16, UTF-8, Cyrillic, then lossy latin-1."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    for encoding in ("utf-8", "cp1251", "koi8-r"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def clip_extracted(text: str) -> str:
    """Clip extracted text to ``MAX_EXTRACTED_CHARS`` with a visible marker."""
    limit = config.max_extracted_chars
    text = text.strip()
    if len(text) <= limit:
        return text
    return (
        text[:limit].rstrip()
        + f"\n\n[… text truncated to the first {limit} characters …]"
    )


# --- Document extraction ----------------------------------------------------

# Decompression / expansion guards. Uploads are capped at 20 MB *compressed*,
# which says nothing about how much memory parsing them needs.
_MAX_EXPANDED_BYTES = 100 * 1024 * 1024  # DOCX: declared uncompressed total
_MAX_PDF_PAGES = 200  # PDF: pages parsed before giving up


def _extract_pdf(data: bytes) -> str | None:
    """Extract text from PDF bytes with pdfplumber (optional dependency)."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber is not installed; PDF analysis is unavailable")
        return None
    try:
        parts: list[str] = []
        collected = 0
        with pdfplumber.open(BytesIO(data)) as pdf:
            # Both caps matter: a 20 MB PDF can hold thousands of dense pages,
            # and the text is only clipped *after* extraction, so without a
            # stop condition the whole document is expanded into memory first.
            for page in pdf.pages[:_MAX_PDF_PAGES]:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
                    collected += len(text)
                if collected >= config.max_extracted_chars:
                    break
        return "\n\n".join(parts) or None
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not kill a reply
        logger.error("PDF text extraction failed: %s", exc)
        return None


def _docx_expanded_size(data: bytes) -> int | None:
    """Declared uncompressed size of a DOCX, or None when it is not a zip."""
    try:
        import zipfile

        with zipfile.ZipFile(BytesIO(data)) as archive:
            return sum(info.file_size for info in archive.infolist())
    except Exception:  # noqa: BLE001 - let the parser report the real problem
        return None


def _extract_docx(data: bytes) -> str | None:
    """Extract text (paragraphs + tables) from DOCX bytes (optional dep)."""
    try:
        import docx
    except ImportError:
        logger.warning("python-docx is not installed; DOCX analysis is unavailable")
        return None
    # A DOCX is a zip: a few megabytes of compressed XML can expand to
    # gigabytes, and python-docx builds the whole tree in memory. Reject the
    # archive on its *declared* size before parsing anything.
    expanded = _docx_expanded_size(data)
    if expanded is not None and expanded > _MAX_EXPANDED_BYTES:
        logger.error(
            "DOCX rejected: declares %d bytes uncompressed (limit %d)",
            expanded,
            _MAX_EXPANDED_BYTES,
        )
        return None
    try:
        document = docx.Document(BytesIO(data))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts) or None
    except Exception as exc:  # noqa: BLE001 - a broken DOCX must not kill a reply
        logger.error("DOCX text extraction failed: %s", exc)
        return None


async def extract_document_text(file_type: str, data: bytes) -> str | None:
    """Extract readable text from document bytes (None when impossible).

    Parsing runs in a worker thread: pdfplumber on a large PDF is CPU-bound
    and must not block the event loop.
    """
    if file_type == "txt":
        text: str | None = decode_text_bytes(data)
    elif file_type == "pdf":
        text = await asyncio.to_thread(_extract_pdf, data)
    elif file_type == "docx":
        text = await asyncio.to_thread(_extract_docx, data)
    else:
        return None
    if not text or not text.strip():
        return None
    return clip_extracted(text)


# --- URL safety (SSRF guard) -------------------------------------------------


def _ip_is_public(ip: Any) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_public_url(url: str) -> bool:
    """True when ``url`` is an http(s) URL that resolves to public IPs only.

    Loopback, private, link-local (cloud metadata!) and reserved addresses are
    refused. Blocking DNS resolution makes this function synchronous-CPU, so
    async callers should use ``asyncio.to_thread``.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    return _resolve_public_ips(host) is not None


def _resolve_public_ips(host: str) -> list[str] | None:
    """All public IP addresses of ``host``, or None when any answer is unsafe.

    One private, loopback, link-local or otherwise reserved address among the
    answers rejects the whole host. Blocking DNS resolution makes this
    synchronous-CPU, so async callers should use ``asyncio.to_thread``.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return None
    if not infos:
        return None
    ips: list[str] = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except (ValueError, IndexError):
            return None
        if not _ip_is_public(ip):
            return None
        if str(ip) not in ips:
            ips.append(str(ip))
    return ips


# --- HTML extraction ---------------------------------------------------------


class _HTMLText(HTMLParser):
    """Minimal readability fallback: page title plus visible text.

    Used when trafilatura is not installed or fails. Scripts, styles and
    other non-visible elements are dropped; block tags become line breaks.
    """

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head"})
    _BREAK = frozenset({
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section",
        "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "blockquote", "pre", "hr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title" and not self.title:
            self._in_title = True
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth and data.strip():
            self.parts.append(data)


def _naive_html_extract(body: bytes) -> tuple[str, str]:
    """(title, visible text) from HTML bytes using only the stdlib."""
    parser = _HTMLText()
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not kill a reply
        return "", ""
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return parser.title.strip(), text.strip()


def _extract_page_text(body: bytes, content_type: str) -> tuple[str, str]:
    """(title, text) from a fetched response body.

    HTML goes through trafilatura when available (main content only, ads and
    navigation dropped) and through the naive stdlib parser otherwise. PDFs
    found behind a link are parsed like uploads; anything else text-like is
    simply decoded.
    """
    if "pdf" in content_type:
        return "", _extract_pdf(body) or ""
    if "html" not in content_type:
        return "", decode_text_bytes(body)
    try:
        import trafilatura
    except ImportError:
        trafilatura = None  # type: ignore[assignment]
    if trafilatura is not None:
        try:
            html_text = body.decode("utf-8", errors="replace")
            extracted = trafilatura.extract(
                html_text,
                include_links=False,
                include_images=False,
                include_tables=True,
                favor_recall=True,
            )
            title = ""
            try:
                metadata = trafilatura.extract_metadata(html_text)
                title = (getattr(metadata, "title", "") or "").strip()
            except Exception as meta_exc:  # noqa: BLE001 - metadata is best-effort
                logger.debug("Metadata extraction failed: %s", meta_exc)
            if extracted and extracted.strip():
                return title, extracted.strip()
        except Exception as exc:  # noqa: BLE001 - fall back to the naive parser
            logger.warning("trafilatura extraction failed, using fallback: %s", exc)
    return _naive_html_extract(body)


# --- Fetching ----------------------------------------------------------------

# A fetched page is capped so one huge HTML cannot eat the model's context
# (the extracted text is additionally clipped by MAX_EXTRACTED_CHARS).
_MAX_FETCH_BYTES = 5 * 1024 * 1024
_MAX_REDIRECTS = 5
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KapumChatBot/1.0; +https://github.com/)"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,text/plain;q=0.9,"
        "application/pdf;q=0.8,*/*;q=0.5"
    ),
    "Accept-Language": "en,ru;q=0.8",
}


@dataclass(slots=True)
class FetchedPage:
    """Readable content fetched from a URL."""

    url: str
    title: str
    text: str


class _PinnedResolver(aiohttp.resolver.AbstractResolver):
    """DNS resolver that answers with the pre-validated addresses only.

    aiohttp would resolve the hostname again on its own, and a malicious
    authoritative DNS server could answer that second lookup with an internal
    address (DNS rebinding). Pinning the addresses that passed the safety
    check closes the window; TLS still validates the certificate against the
    hostname, which the URL keeps.
    """

    def __init__(self, ips: list[str]) -> None:
        self._ips = ips

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        del family  # per-address: v6 answers must carry AF_INET6
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": (
                    socket.AF_INET6
                    if ipaddress.ip_address(ip).version == 6
                    else socket.AF_INET
                ),
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
            for ip in self._ips
        ]

    async def close(self) -> None:
        return None


async def _fetch_body(url: str) -> tuple[bytes | None, str, str]:
    """GET ``url`` following redirects manually; validate every hop.

    Returns ``(body, final_url, content_type)``; body is None on any refusal
    or failure. ``allow_redirects=False`` is deliberate: aiohttp would follow
    a redirect to an internal address without giving us a chance to check it.
    Every hop (including the first URL) is resolved once, screened for public
    addresses, and the request is pinned to exactly those addresses.
    """
    timeout = aiohttp.ClientTimeout(total=config.link_fetch_timeout)
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                parsed = urlsplit(url)
            except ValueError:
                return None, url, ""
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                logger.warning(
                    "Refusing a non-http(s) or hostless URL: %s",
                    parsed.netloc or url,
                )
                return None, url, ""
            ips = await asyncio.to_thread(_resolve_public_ips, parsed.hostname)
            if not ips:
                logger.warning(
                    "Refusing a host without public addresses: %s", parsed.netloc
                )
                return None, url, ""
            connector = aiohttp.TCPConnector(resolver=_PinnedResolver(ips))
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout, connector=connector
                ) as session:
                    async with session.get(
                        url, headers=_FETCH_HEADERS, allow_redirects=False
                    ) as resp:
                        if resp.status in (301, 302, 303, 307, 308):
                            location = resp.headers.get("Location")
                            if not location:
                                return None, url, ""
                            url = urljoin(url, location)
                            continue
                        if resp.status != 200:
                            logger.info("URL fetch got HTTP %s", resp.status)
                            return None, url, ""
                        content_type = (
                            (resp.headers.get("Content-Type") or "").lower()
                        )
                        allowed = (
                            content_type.startswith(
                                (
                                    "text/",
                                    "application/json",
                                    "application/xml",
                                    "application/pdf",
                                )
                            )
                            or "xhtml" in content_type
                            or not content_type
                        )
                        if not allowed:
                            logger.info(
                                "Skipping non-text content type %r", content_type
                            )
                            return None, url, ""
                        declared = resp.content_length
                        if declared is not None and declared > _MAX_FETCH_BYTES:
                            logger.info("Page is too large (%d bytes)", declared)
                            return None, url, ""
                        body = await resp.content.read(_MAX_FETCH_BYTES + 1)
                        if len(body) > _MAX_FETCH_BYTES:
                            return None, url, ""
                        return body, str(resp.url), content_type
            finally:
                await connector.close()
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        logger.warning("URL fetch failed for %s: %s", urlsplit(url).netloc, exc)
        return None, url, ""
    return None, url, ""


async def fetch_url_text(url: str) -> FetchedPage | None:
    """Fetch ``url`` and extract its readable text (None on any failure).

    The caller gets a final (possibly redirected) URL, the page title and the
    main content, clipped to ``MAX_EXTRACTED_CHARS``.
    """
    if not await asyncio.to_thread(is_safe_public_url, url):
        logger.warning("Refusing to fetch a non-public URL: %s", urlsplit(url).netloc)
        return None
    body, final_url, content_type = await _fetch_body(url)
    if body is None:
        return None
    try:
        title, text = await asyncio.to_thread(_extract_page_text, body, content_type)
    except Exception as exc:  # noqa: BLE001 - extraction must never propagate
        logger.error("Page text extraction failed: %s", exc)
        return None
    if not text or not text.strip():
        return None
    return FetchedPage(url=final_url or url, title=title, text=clip_extracted(text))
