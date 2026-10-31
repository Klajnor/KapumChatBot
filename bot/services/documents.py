"""Text extraction from uploaded documents and linked web pages.

Heavy parsers are optional dependencies: ``pdfplumber`` (PDF), ``python-docx``
(DOCX), ``openpyxl`` (XLSX), ``python-pptx`` (PPTX) and ``trafilatura`` (high-quality article extraction). Each import is
guarded, so the bot runs without them and degrades gracefully:

* a PDF/DOCX/XLSX/PPTX upload is rejected with an explanation when its parser is
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp

from bot.config import config
from bot.services.operations import stage
from bot.services.pdf_limits import PdfResourceLimit, pdf_stream_limits

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
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}


def detect_file_type(file_name: str | None, mime_type: str | None = None) -> str | None:
    """Classify an upload as ``"pdf"``, ``"docx"``, ``"xlsx"``, ``"pptx"``, ``"txt"`` or None.

    The extension wins; the MIME type is the fallback for nameless files.
    Legacy ``.doc``/``.xls``/``.ppt`` formats and archives are not
    supported and return None.
    """
    name = (file_name or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".xlsx"):
        return "xlsx"
    if name.endswith(".pptx"):
        return "pptx"
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
_MAX_EXPANDED_BYTES = 100 * 1024 * 1024  # DOCX/XLSX/PPTX: declared uncompressed total
_MAX_PDF_PAGES = 200  # PDF: pages parsed before giving up
_MAX_SHEETS = 20  # XLSX: sheets read
_MAX_SHEET_ROWS = 5000  # XLSX: rows read per sheet
_MAX_SLIDES = 200  # PPTX: slides read


class DocumentParseTimeout(Exception):
    """Parsing a document took longer than ``DOCUMENT_PARSE_TIMEOUT``."""


# Разбор документов идёт в собственном небольшом пуле: зависший или очень
# тяжёлый файл не должен отнимать потоки у общего пула (DNS-резолв SSRF-проверки,
# экспорт, формулы). Потоки нельзя убить, поэтому разборщики сами проверяют
# дедлайн между страницами/строками/слайдами и выходят с DocumentParseTimeout.
_PARSE_WORKERS = 2
_parse_pool = ThreadPoolExecutor(
    max_workers=_PARSE_WORKERS, thread_name_prefix="docparse"
)
# Слот принадлежит реальному потоку, а не ожидающей корутине. Таймаут/отмена
# запроса не освобождают его: иначе зависшие парсеры создают неограниченную
# очередь с байтами новых документов. Семафор общий для всех event loop.
_parse_slots = threading.BoundedSemaphore(_PARSE_WORKERS)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise DocumentParseTimeout


async def _run_parser(func: Any, *args: Any) -> Any:
    """Run a blocking parser in the dedicated pool under ``DOCUMENT_PARSE_TIMEOUT``.

    При занятом пуле сразу возвращает ``DocumentParseTimeout`` без очереди.
    Слот освобождается только после фактического завершения разбора.
    """
    timeout = config.document_parse_timeout
    loop = asyncio.get_running_loop()
    slots = _parse_slots
    if not slots.acquire(blocking=False):
        raise DocumentParseTimeout

    def job() -> Any:
        try:
            return func(*args, deadline=time.monotonic() + timeout)
        finally:
            slots.release()

    try:
        future = loop.run_in_executor(_parse_pool, job)
    except BaseException:
        slots.release()
        raise
    # После таймаута исключение фонового парсера тоже должно быть прочитано.
    future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout + 2)
    except TimeoutError:
        raise DocumentParseTimeout from None


def _extract_pdf(data: bytes, deadline: float | None = None) -> str | None:
    """Extract text from PDF bytes with pdfplumber (optional dependency)."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber is not installed; PDF analysis is unavailable")
        return None
    try:
        parts: list[str] = []
        collected = 0
        with pdf_stream_limits(), pdfplumber.open(BytesIO(data)) as pdf:
            # Both caps matter: a 20 MB PDF can hold thousands of dense pages,
            # and the text is only clipped *after* extraction, so without a
            # stop condition the whole document is expanded into memory first.
            for page in pdf.pages[:_MAX_PDF_PAGES]:
                _check_deadline(deadline)
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
                    collected += len(text)
                if collected >= config.max_extracted_chars:
                    break
        return "\n\n".join(parts) or None
    except (DocumentParseTimeout, PdfResourceLimit):
        raise
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not kill a reply
        logger.error("PDF text extraction failed: %s", exc)
        return None


def _docx_expanded_size(data: bytes) -> int | None:
    """Declared uncompressed size of an OOXML file, or None when it is not a zip."""
    try:
        import zipfile

        with zipfile.ZipFile(BytesIO(data)) as archive:
            return sum(info.file_size for info in archive.infolist())
    except Exception:  # noqa: BLE001 - let the parser report the real problem
        return None


def _extract_docx(data: bytes, deadline: float | None = None) -> str | None:
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
        parts = []
        for paragraph in document.paragraphs:
            _check_deadline(deadline)
            if paragraph.text.strip():
                parts.append(paragraph.text)
        for table in document.tables:
            for row in table.rows:
                _check_deadline(deadline)
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts) or None
    except DocumentParseTimeout:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken DOCX must not kill a reply
        logger.error("DOCX text extraction failed: %s", exc)
        return None


def _ooxml_too_large(data: bytes, kind: str) -> bool:
    """True when an OOXML zip declares an unreasonable uncompressed size."""
    expanded = _docx_expanded_size(data)
    if expanded is not None and expanded > _MAX_EXPANDED_BYTES:
        logger.error(
            "%s rejected: declares %d bytes uncompressed (limit %d)",
            kind,
            expanded,
            _MAX_EXPANDED_BYTES,
        )
        return True
    return False


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("\n", " ").strip()


def _extract_xlsx(data: bytes, deadline: float | None = None) -> str | None:
    """Extract sheets as ``a | b | c`` rows from XLSX bytes (optional dep).

    Cached formula results are read (``data_only``); formulas that were never
    calculated by a spreadsheet application come out empty.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl is not installed; XLSX analysis is unavailable")
        return None
    if _ooxml_too_large(data, "XLSX"):
        return None
    try:
        # Первый проход — кэшированные значения. Если пусто (файл создан
        # библиотекой без пересчёта формул или с неверным <dimension>), второй
        # проход сбрасывает размеры листа и показывает формулы как текст.
        text = _read_xlsx_sheets(openpyxl, data, data_only=True, deadline=deadline)
        if not text:
            text = _read_xlsx_sheets(openpyxl, data, data_only=False, deadline=deadline)
        if not text:
            logger.warning("XLSX contains no readable cells")
        return text
    except DocumentParseTimeout:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken XLSX must not kill a reply
        logger.error("XLSX text extraction failed: %s", exc)
        return None


def _read_xlsx_sheets(
    openpyxl: Any, data: bytes, *, data_only: bool, deadline: float | None = None
) -> str | None:
    """Rows of every sheet as text; ``data_only=False`` keeps formulas as ``=...``."""
    workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=data_only)
    try:
        parts: list[str] = []
        collected = 0
        for sheet in workbook.worksheets[:_MAX_SHEETS]:
            if hasattr(sheet, "reset_dimensions"):
                # Неверный <dimension> в read-only режиме обрезает чтение.
                sheet.reset_dimensions()
            lines = [f"## Sheet: {sheet.title}"]
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                _check_deadline(deadline)
                if index >= _MAX_SHEET_ROWS or collected >= config.max_extracted_chars:
                    break
                cells = [_cell_text(value) for value in row]
                while cells and not cells[-1]:
                    cells.pop()
                if not any(cells):
                    continue
                line = " | ".join(cells)
                lines.append(line)
                collected += len(line)
            if len(lines) > 1:
                parts.append("\n".join(lines))
            if collected >= config.max_extracted_chars:
                break
        return "\n\n".join(parts) or None
    finally:
        workbook.close()


def _shape_texts(shapes: Any) -> list[str]:
    """Text of shapes, tables and grouped shapes on a slide, in order."""
    texts: list[str] = []
    for shape in shapes:
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text:
                texts.append(text)
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    texts.append(" | ".join(cells))
        children = getattr(shape, "shapes", None)
        if children is not None and getattr(shape, "shape_type", None) == 6:  # group
            texts.extend(_shape_texts(children))
    return texts


def _extract_pptx(data: bytes, deadline: float | None = None) -> str | None:
    """Extract slide text, tables and speaker notes from PPTX bytes (optional dep)."""
    try:
        import pptx
    except ImportError:
        logger.warning("python-pptx is not installed; PPTX analysis is unavailable")
        return None
    if _ooxml_too_large(data, "PPTX"):
        return None
    try:
        presentation = pptx.Presentation(BytesIO(data))
        parts: list[str] = []
        collected = 0
        for number, slide in enumerate(presentation.slides, 1):
            _check_deadline(deadline)
            if number > _MAX_SLIDES or collected >= config.max_extracted_chars:
                break
            texts = _shape_texts(slide.shapes)
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    texts.append(f"[Notes] {notes}")
            if texts:
                block = f"## Slide {number}\n" + "\n".join(texts)
                parts.append(block)
                collected += len(block)
        return "\n\n".join(parts) or None
    except DocumentParseTimeout:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken PPTX must not kill a reply
        logger.error("PPTX text extraction failed: %s", exc)
        return None


@stage("Extracting document text")
async def extract_document_text(file_type: str, data: bytes) -> str | None:
    """Extract readable text from document bytes (None when impossible).

    Parsing runs in a dedicated worker pool under ``DOCUMENT_PARSE_TIMEOUT``:
    pdfplumber on a large PDF is CPU-bound and must not block the event loop.
    Raises ``DocumentParseTimeout`` when the parser does not finish in time —
    callers tell the user and refund the quota.
    Raises ``PdfResourceLimit`` on excessive PDF Flate expansion; callers must
    reject the document rather than retry it via OCR.
    """
    if file_type == "txt":
        text: str | None = decode_text_bytes(data)
    elif file_type == "pdf":
        text = await _run_parser(_extract_pdf, data)
    elif file_type == "docx":
        text = await _run_parser(_extract_docx, data)
    elif file_type == "xlsx":
        text = await _run_parser(_extract_xlsx, data)
    elif file_type == "pptx":
        text = await _run_parser(_extract_pptx, data)
    else:
        return None
    if not text or not text.strip():
        return None
    return clip_extracted(text)


# --- URL safety (SSRF guard) -------------------------------------------------


def _ip_is_public(ip: Any) -> bool:
    """True only for globally routable addresses.

    ``is_global`` (not a denylist of private ranges) also refuses CGNAT
    100.64.0.0/10 — home of the Alibaba Cloud metadata service
    100.100.100.200 and of Tailscale — as well as 192.0.0.0/24, 198.18.0.0/15
    and documentation ranges.
    """
    return ip.is_global and not (
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
    if not host or not _port_allowed(parsed):
        return False
    return _resolve_public_ips(host) is not None


# Только стандартные веб-порты: иначе ссылка превращается в сканер портов
# и доступ к «внутренним» сервисам на публичных адресах (Redis, админки…).
_ALLOWED_PORTS = frozenset({80, 443})


def _port_allowed(parsed: Any) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return False
    return port is None or port in _ALLOWED_PORTS


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


class _LinkCollector(HTMLParser):
    """Collects ``<a href>`` targets of a page in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value.strip())


# Targets that are never readable pages.
_SKIP_LINK_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogg", ".wav",
    ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".msi", ".dmg", ".apk",
    ".css", ".js", ".woff", ".woff2", ".ttf",
)


def extract_page_links(body: bytes, base_url: str, limit: int = 50) -> list[str]:
    """Absolute, de-duplicated http(s) links found in an HTML page.

    Fragments are dropped, the page itself is excluded, and links to media,
    archives and static assets are skipped. Same-host links come first: they
    are the most likely continuation of the page the user pointed at.
    """
    collector = _LinkCollector()
    try:
        collector.feed(body.decode("utf-8", errors="replace"))
        collector.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not kill a reply
        return []
    base = urlsplit(base_url)
    own = base._replace(fragment="").geturl()
    seen: set[str] = {own}
    same_host: list[str] = []
    other: list[str] = []
    for href in collector.hrefs:
        if href.lower().startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        try:
            absolute = urljoin(base_url, href)
            parts = urlsplit(absolute)
        except ValueError:
            continue
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if parts.path.lower().endswith(_SKIP_LINK_EXTENSIONS):
            continue
        absolute = parts._replace(fragment="").geturl()
        if absolute in seen:
            continue
        seen.add(absolute)
        (same_host if parts.hostname == base.hostname else other).append(absolute)
    return (same_host + other)[:limit]


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


def _extract_page_text(
    body: bytes, content_type: str, deadline: float | None = None
) -> tuple[str, str]:
    """(title, text) from a fetched response body.

    HTML goes through trafilatura when available (main content only, ads and
    navigation dropped) and through the naive stdlib parser otherwise. PDFs
    found behind a link are parsed like uploads; anything else text-like is
    simply decoded.
    """
    if "pdf" in content_type:
        return "", _extract_pdf(body, deadline) or ""
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
    links: list[str] = field(default_factory=list)


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


async def _fetch_body(
    url: str,
    *,
    max_bytes: int | None = None,
    headers: dict[str, str] | None = None,
    text_only: bool = True,
    timeout_seconds: float | None = None,
) -> tuple[bytes | None, str, str | None]:
    """GET ``url`` following redirects manually; validate every hop.

    Returns ``(body, final_url, content_type)``. On any refusal or failure
    ``body`` and ``content_type`` are ``None``; ``content_type`` is the bare
    media type (``text/plain``, without parameters like ``charset``).
    ``allow_redirects=False`` is deliberate: aiohttp would follow a redirect to
    an internal address without giving us a chance to check it. Every hop
    (including the first URL) is resolved once, screened for public addresses,
    and the request is pinned to exactly those addresses.

    Для изображений используется тот же сетевой шлюз с ``text_only=False``
    и собственным лимитом. Явные заголовки не передаются на другой origin.
    """
    timeout = aiohttp.ClientTimeout(
        total=config.link_fetch_timeout if timeout_seconds is None else timeout_seconds
    )
    limit = _MAX_FETCH_BYTES if max_bytes is None else max_bytes
    try:
        try:
            initial = urlsplit(url)
        except ValueError:
            return None, url, None
        origin = (initial.scheme, initial.netloc)
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                parsed = urlsplit(url)
            except ValueError:
                return None, url, None
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                logger.warning(
                    "Refusing a non-http(s) or hostless URL: %s",
                    parsed.netloc or url,
                )
                return None, url, None
            if not _port_allowed(parsed):
                logger.warning("Refusing a non-web port: %s", parsed.netloc)
                return None, url, None
            ips = await asyncio.to_thread(_resolve_public_ips, parsed.hostname)
            if not ips:
                logger.warning(
                    "Refusing a host without public addresses: %s", parsed.netloc
                )
                return None, url, None
            connector = aiohttp.TCPConnector(resolver=_PinnedResolver(ips))
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout, connector=connector
                ) as session, session.get(
                    url,
                    # Авторизация не должна следовать на другой origin.
                    headers=(
                        headers if headers is not None
                        and (parsed.scheme, parsed.netloc) == origin
                        else _FETCH_HEADERS
                    ),
                    allow_redirects=False,
                ) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            return None, url, None
                        url = urljoin(url, location)
                        continue
                    if resp.status != 200:
                        logger.info("URL fetch got HTTP %s", resp.status)
                        return None, url, None
                    content_type = (
                        (resp.headers.get("Content-Type") or "")
                        .lower()
                        .split(";", 1)[0]
                        .strip()
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
                    if text_only and not allowed:
                        logger.info(
                            "Skipping non-text content type %r", content_type
                        )
                        return None, url, None
                    declared = resp.content_length
                    if declared is not None and declared > limit:
                        logger.info("Page is too large (%d bytes)", declared)
                        return None, url, None
                    # StreamReader.read(n) отдаёт только то, что уже в буфере,
                    # а не n байт: читаем чанками до EOF или до лимита.
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.content.iter_chunked(65536):
                        total += len(chunk)
                        if total > limit:
                            logger.info("Download exceeds %d bytes", limit)
                            return None, url, None
                        chunks.append(chunk)
                    return b"".join(chunks), str(resp.url), content_type
            finally:
                await connector.close()
    except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
        logger.warning("URL fetch failed (%s)", type(exc).__name__)
        return None, url, None
    return None, url, None


async def fetch_url_text(url: str, *, with_links: bool = False) -> FetchedPage | None:
    """Fetch ``url`` and extract its readable text (None on any failure).

    The caller gets a final (possibly redirected) URL, the page title and the
    main content, clipped to ``MAX_EXTRACTED_CHARS``. With ``with_links`` the
    page's outgoing links are collected too, for crawling one level deeper.
    """
    if not await asyncio.to_thread(is_safe_public_url, url):
        logger.warning("Refusing to fetch a non-public URL: %s", urlsplit(url).netloc)
        return None
    body, final_url, content_type = await _fetch_body(url)
    if body is None:
        return None
    try:
        title, text = await _run_parser(
            _extract_page_text, body, content_type or ""
        )
    except DocumentParseTimeout:
        logger.warning("Page parsing timed out: %s", urlsplit(url).netloc)
        return None
    except Exception as exc:  # noqa: BLE001 - extraction must never propagate
        logger.error("Page text extraction failed: %s", exc)
        return None
    if not text or not text.strip():
        return None
    links: list[str] = []
    if with_links and "html" in (content_type or ""):
        links = await asyncio.to_thread(extract_page_links, body, final_url or url)
    return FetchedPage(
        url=final_url or url, title=title, text=clip_extracted(text), links=links
    )
