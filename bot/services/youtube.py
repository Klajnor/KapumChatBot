"""YouTube captions for video summaries (yt-dlp, optional dependency).

Only the captions are read — nothing is downloaded and no speech recognition
runs, so a video without captions is reported as such. The URL handed to
yt-dlp is rebuilt from a validated 11-character video id, never taken from the
message, so a crafted link cannot steer the extractor to another host.

YouTube changes often and yt-dlp breaks with it: every failure becomes a
``YouTubeError`` with a user-facing text, and the fix is almost always
``pip install -U yt-dlp``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from bot.config import config

logger = logging.getLogger(__name__)

_VIDEO_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_BUCKET_SECONDS = 60  # one [m:ss] timestamp per minute of speech


class YouTubeError(Exception):
    """A failure whose message is safe to show to the user."""


@dataclass(slots=True)
class Transcript:
    video_id: str
    title: str
    channel: str
    duration: int  # seconds, 0 when unknown
    language: str
    automatic: bool
    text: str


def youtube_video_id(url: str) -> str | None:
    """The video id of a YouTube watch/shorts/live/embed/youtu.be URL, else None."""
    match = _VIDEO_URL_RE.match(url.strip())
    return match.group(1) if match else None


def _format_clock(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_json3(raw: bytes | str) -> str:
    """Caption text with a ``[m:ss]`` mark at the start of every minute."""
    data = json.loads(raw)
    lines: list[str] = []
    bucket = -1
    for event in data.get("events", []):
        segments = event.get("segs")
        if not segments:
            continue
        text = " ".join("".join(seg.get("utf8", "") for seg in segments).split())
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000
        index = int(start // _BUCKET_SECONDS)
        if index != bucket:
            bucket = index
            lines.append(f"[{_format_clock(start)}] {text}")
        else:
            lines[-1] += " " + text
    return "\n".join(lines)


def _language_matches(track: str, wanted: str) -> bool:
    track, wanted = track.lower(), wanted.lower()
    return track == wanted or track.startswith(wanted + "-")


def _pick_track(info: dict[str, Any], preferred: list[str]) -> tuple[str, bool, list[dict]] | None:
    """Best caption track as (language, automatic, formats).

    Manual captions beat automatic ones; within a kind the user's preferred
    languages come first, then the video's own language.
    """
    own = (info.get("language") or "").lower()
    for automatic, tracks in (
        (False, info.get("subtitles") or {}),
        (True, info.get("automatic_captions") or {}),
    ):
        if not tracks:
            continue
        candidates: list[str] = []
        for wanted in preferred:
            candidates += [t for t in tracks if _language_matches(t, wanted)]
        if own:
            candidates += [t for t in tracks if _language_matches(t, own)]
        # Automatic captions list every translation; "-orig" marks the spoken one.
        candidates += [t for t in tracks if t.endswith("-orig")]
        if not automatic:
            candidates += list(tracks)
        for lang in candidates:
            formats = [f for f in tracks[lang] if f.get("ext") == "json3" and f.get("url")]
            if formats:
                return lang, automatic, formats
    return None


def _load_transcript(video_id: str) -> Transcript:
    """Blocking work: extract metadata and download the caption track."""
    try:
        import yt_dlp
    except ImportError:
        raise YouTubeError(
            "YouTube support is not installed on this bot (pip install yt-dlp)."
        ) from None

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreconfig": True,
        "socket_timeout": 15,
        "retries": 1,
        "extractor_retries": 1,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise YouTubeError("Could not load the video.")
            if info.get("is_live"):
                raise YouTubeError("Live streams cannot be summarized.")
            preferred = [x.strip() for x in config.youtube_langs.split(",") if x.strip()]
            picked = _pick_track(info, preferred)
            if picked is None:
                raise YouTubeError("This video has no captions, so there is nothing to read.")
            language, automatic, formats = picked
            body = ydl.urlopen(formats[0]["url"]).read()
    except YouTubeError:
        raise
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many unrelated types
        logger.warning("YouTube extraction failed for %s: %s", video_id, exc)
        raise YouTubeError(
            "Could not load the video (private, age-restricted, region-locked, "
            "or YouTube blocked the request)."
        ) from exc

    try:
        text = parse_json3(body)
    except (ValueError, AttributeError) as exc:
        logger.warning("Unreadable caption track for %s: %s", video_id, exc)
        raise YouTubeError("The captions of this video could not be read.") from exc
    if not text.strip():
        raise YouTubeError("The captions of this video are empty.")

    limit = config.youtube_max_chars
    if len(text) > limit:
        text = (
            text[:limit].rstrip()
            + f"\n\n[… transcript truncated to the first {limit} characters …]"
        )
    return Transcript(
        video_id=video_id,
        title=str(info.get("title") or ""),
        channel=str(info.get("uploader") or info.get("channel") or ""),
        duration=int(info.get("duration") or 0),
        language=language,
        automatic=automatic,
        text=text,
    )


async def fetch_transcript(video_id: str) -> Transcript:
    """Captions of one video; raises ``YouTubeError`` with a user-facing text."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_load_transcript, video_id), config.youtube_timeout
        )
    except asyncio.TimeoutError:
        raise YouTubeError("YouTube took too long to answer. Try again later.") from None


def describe(transcript: Transcript) -> str:
    """Header placed above the transcript in the model request."""
    kind = "auto-generated" if transcript.automatic else "author's"
    lines = [f"[YouTube video: https://www.youtube.com/watch?v={transcript.video_id}]"]
    if transcript.title:
        lines.append(f"Title: {transcript.title}")
    if transcript.channel:
        lines.append(f"Channel: {transcript.channel}")
    if transcript.duration:
        lines.append(f"Duration: {_format_clock(transcript.duration)}")
    lines.append(f"Captions: {kind}, language {transcript.language}")
    return "\n".join(lines)
