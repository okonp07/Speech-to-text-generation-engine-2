"""Fetch an already-transcribed caption track from YouTube.

The captions endpoint is far less aggressively bot-checked than the
video-stream endpoint, so this path works from datacenter IPs (like
Streamlit Community Cloud) where ``yt-dlp`` typically fails. It is also
dramatically faster than downloading audio and running Whisper: we get the
author's captions (or YouTube's auto-generated ones) back as structured
segments in under a second.

When no captions exist for the URL (e.g. many music videos, private
streams, some Shorts), this module raises :class:`CaptionsUnavailableError`
and the caller is expected to tell the user to try a different URL or use
the video-file upload tab.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .transcriber import TranscriptionResult, TranscriptionSegment


class CaptionsError(RuntimeError):
    """Base class for caption-fetching failures."""


class CaptionsUnavailableError(CaptionsError):
    """Raised when the video has no retrievable captions at all."""


class InvalidYouTubeUrlError(CaptionsError):
    """Raised when we cannot extract a video ID from the given URL."""


# Matches the common YouTube URL shapes and captures the 11-char video ID.
_VIDEO_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/watch\?[^ ]*?\bv=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/live/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/v/([A-Za-z0-9_-]{11})"),
)


def extract_video_id(url: str) -> str:
    """Return the YouTube video ID embedded in *url*, or raise."""

    if not url:
        raise InvalidYouTubeUrlError("No URL was provided.")
    trimmed = url.strip()
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(trimmed)
        if match:
            return match.group(1)
    raise InvalidYouTubeUrlError(
        "Could not find a YouTube video ID in the URL. "
        "Paste a link like https://www.youtube.com/watch?v=… or https://youtu.be/…"
    )


def fetch_youtube_captions(
    url: str,
    language_hint: Optional[str] = None,
) -> TranscriptionResult:
    """Fetch captions for *url* and return them as a :class:`TranscriptionResult`.

    Works transparently across the two incompatible versions of
    ``youtube-transcript-api``: the 0.6.x class-method API and the 1.x
    instance-based API.

    Parameters
    ----------
    url:
        A YouTube URL (watch, shorts, youtu.be, embed, live).
    language_hint:
        Preferred ISO language code (e.g. ``"en"``). When supplied, the
        corresponding track is tried first; we otherwise fall back to
        whatever track YouTube has. ``None`` means "let YouTube pick".
    """

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise CaptionsError(
            "youtube-transcript-api is not installed. Add it to requirements.txt."
        ) from exc

    known_errors = _load_known_error_classes()
    video_id = extract_video_id(url)

    # The language priority order we try. We always fall through to the
    # generated/auto tracks at the end so we get *something* when we can.
    preferred_languages: list[str] = []
    if language_hint:
        preferred_languages.append(language_hint)
    for fallback in ("en", "en-US", "en-GB"):
        if fallback not in preferred_languages:
            preferred_languages.append(fallback)

    try:
        entries, chosen_language = _get_transcript_entries(
            YouTubeTranscriptApi, video_id, preferred_languages
        )
    except known_errors["TranscriptsDisabled"] as exc:
        raise CaptionsUnavailableError(
            "Captions are disabled on this video."
        ) from exc
    except known_errors["NoTranscriptFound"] as exc:
        raise CaptionsUnavailableError(
            "No captions are available for this video."
        ) from exc
    except known_errors["VideoUnavailable"] as exc:
        raise CaptionsUnavailableError(
            "YouTube reports this video as unavailable."
        ) from exc
    except CaptionsError:
        raise
    except Exception as exc:  # pragma: no cover - defensive / network
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        if (
            "could not retrieve" in lowered
            or "no element found" in lowered
            or "transcripts are disabled" in lowered
            or "no transcripts were found" in lowered
        ):
            raise CaptionsUnavailableError(
                "YouTube returned no captions for this video."
            ) from exc
        raise CaptionsError(
            f"Captions fetch failed. Details: {message}"
        ) from exc

    segments = _entries_to_segments(entries)
    if not segments:
        raise CaptionsUnavailableError(
            "Captions track exists but is empty."
        )

    transcript_text = " ".join(seg.text for seg in segments).strip()
    duration = max((seg.end_seconds for seg in segments), default=0.0)

    return TranscriptionResult(
        text=transcript_text or "No speech detected.",
        # Captions are an official track; confidence doesn't map cleanly
        # onto a probability. Report a neutral 1.0 to distinguish from a
        # Whisper run, and surface the "source" via other UI affordances.
        confidence=1.0,
        language=chosen_language,
        language_confidence=None,
        duration_seconds=duration,
        segments=tuple(segments),
    )


def _load_known_error_classes() -> dict:
    """Return the library's named error classes, with safe fallbacks.

    The 1.x release reorganized the error module paths slightly. We try a
    few locations and substitute a private marker class when one is missing
    so the ``except`` blocks in :func:`fetch_youtube_captions` still work.
    """

    result: dict = {}
    for name in ("TranscriptsDisabled", "NoTranscriptFound", "VideoUnavailable"):
        cls = None
        for module_path in (
            "youtube_transcript_api._errors",
            "youtube_transcript_api",
        ):
            try:
                module = __import__(module_path, fromlist=[name])
                cls = getattr(module, name, None)
                if cls is not None:
                    break
            except Exception:
                continue
        if cls is None:

            class _Missing(Exception):  # noqa: D401 - sentinel
                """Placeholder so except-clause matches nothing."""

            cls = _Missing
        result[name] = cls
    return result


def _get_transcript_entries(
    api_class,
    video_id: str,
    preferred_languages: list[str],
):
    """Version-agnostic adapter returning ``(entries, language_code)``."""

    # --- 1.x: instance-based API (``YouTubeTranscriptApi().fetch/.list``) ---
    # Some 1.x builds still have to be instantiated before ``fetch``/``list``
    # work; others expose them as methods on the class object directly. Try
    # an instance first — this is the common 1.x shape.
    try:
        instance = api_class()
    except Exception:  # pragma: no cover - defensive
        instance = None

    if instance is not None and (
        hasattr(instance, "fetch") or hasattr(instance, "list")
    ):
        transcript_list = None
        if hasattr(instance, "list"):
            try:
                transcript_list = instance.list(video_id)
            except Exception:
                transcript_list = None

        if transcript_list is not None:
            transcript = _pick_best_transcript(transcript_list, preferred_languages)
            fetched = transcript.fetch()
            return _unpack_fetched(fetched, getattr(transcript, "language_code", None))

        if hasattr(instance, "fetch"):
            fetched = instance.fetch(video_id, languages=preferred_languages)
            return _unpack_fetched(
                fetched,
                preferred_languages[0] if preferred_languages else None,
            )

    # --- 0.6.x: class-method API (``YouTubeTranscriptApi.list_transcripts``) ---
    if hasattr(api_class, "list_transcripts"):
        transcript_list = api_class.list_transcripts(video_id)
        transcript = _pick_best_transcript(transcript_list, preferred_languages)
        entries = transcript.fetch()
        return _unpack_fetched(entries, getattr(transcript, "language_code", None))

    if hasattr(api_class, "get_transcript"):
        entries = api_class.get_transcript(
            video_id, languages=preferred_languages
        )
        return _unpack_fetched(
            entries,
            preferred_languages[0] if preferred_languages else None,
        )

    raise CaptionsError(
        "Incompatible youtube-transcript-api version: no known entry points "
        "(fetch/list or get_transcript/list_transcripts). Upgrade the library."
    )


def _pick_best_transcript(transcript_list, preferred_languages: list[str]):
    """Prefer a human-authored caption over an auto-generated one."""

    for finder in ("find_manually_created_transcript", "find_generated_transcript"):
        fn = getattr(transcript_list, finder, None)
        if fn is None:
            continue
        try:
            return fn(preferred_languages)
        except Exception:
            continue
    try:
        return next(iter(transcript_list))
    except StopIteration as exc:
        raise CaptionsUnavailableError(
            "No captions are available for this video."
        ) from exc


def _unpack_fetched(fetched, fallback_language: Optional[str]):
    """Normalize the many shapes ``youtube-transcript-api`` returns.

    Returns ``(entries_iterable, language_code)``.
    """

    language = None

    # v1.x FetchedTranscript: has ``.snippets`` and ``.language_code``.
    if hasattr(fetched, "snippets"):
        entries = list(fetched.snippets)
        language = getattr(fetched, "language_code", None)
    elif hasattr(fetched, "__iter__") and not isinstance(fetched, (str, bytes)):
        entries = list(fetched)
        language = getattr(fetched, "language_code", None)
    else:
        entries = [fetched]

    return entries, (language or fallback_language or "unknown")


def _entries_to_segments(
    entries: Sequence[object],
) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for raw in entries:
        # The library has shipped two shapes: dicts in older versions, and
        # objects (``FetchedTranscriptSnippet``) in newer ones. Handle both.
        if isinstance(raw, dict):
            text = str(raw.get("text", "") or "").strip()
            start = float(raw.get("start", 0.0) or 0.0)
            duration = float(raw.get("duration", 0.0) or 0.0)
        else:
            text = str(getattr(raw, "text", "") or "").strip()
            start = float(getattr(raw, "start", 0.0) or 0.0)
            duration = float(getattr(raw, "duration", 0.0) or 0.0)

        if not text or text == "[Music]":
            continue
        segments.append(
            TranscriptionSegment(
                start_seconds=start,
                end_seconds=start + max(duration, 0.0),
                text=text,
                confidence=1.0,
            )
        )
    return segments


__all__ = [
    "CaptionsError",
    "CaptionsUnavailableError",
    "InvalidYouTubeUrlError",
    "extract_video_id",
    "fetch_youtube_captions",
]
