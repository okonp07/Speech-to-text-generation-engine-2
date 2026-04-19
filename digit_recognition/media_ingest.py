"""Helpers that bring external media (YouTube URLs, uploaded video files)
into a local audio file that :class:`SpeechTranscriber` can transcribe.

This module isolates all of the "ingest" concerns so the Streamlit app can
stay focused on UI. It intentionally has no Streamlit imports.

Two public entry points:

* :func:`fetch_youtube_audio` — download the audio track of a YouTube URL
  using ``yt-dlp`` and return the path to the resulting audio file.
* :func:`extract_audio_from_video` — use ``ffmpeg`` to strip the audio
  track out of an uploaded video file and return the path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Optional


YOUTUBE_URL_PATTERN = re.compile(
    r"^(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"[\w\-]+",
    re.IGNORECASE,
)


SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mpeg",
    ".mpg",
)


class MediaIngestError(RuntimeError):
    """Raised when audio cannot be produced from the requested source."""


@dataclass(frozen=True)
class IngestedMedia:
    """Represents a locally-available audio file ready for transcription."""

    audio_path: Path
    display_name: str
    source_description: str


def is_valid_youtube_url(url: str) -> bool:
    """Return ``True`` if *url* looks like a YouTube video URL."""

    if not url:
        return False
    return bool(YOUTUBE_URL_PATTERN.match(url.strip()))


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
    return cleaned or "audio"


def _require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise MediaIngestError(
            "ffmpeg is not available on this machine. Install ffmpeg to extract "
            "audio from video files or YouTube downloads. On Streamlit Community "
            "Cloud, add 'ffmpeg' to packages.txt."
        )
    return ffmpeg_path


def fetch_youtube_audio(
    url: str,
    output_dir: str | os.PathLike[str] | None = None,
    preferred_codec: str = "mp3",
    preferred_quality: str = "192",
) -> IngestedMedia:
    """Download the audio stream of *url* and return the local file path.

    Parameters
    ----------
    url:
        A YouTube video URL (youtube.com/watch, youtu.be/, shorts, etc.).
    output_dir:
        Directory to write the audio file to. A fresh temporary directory is
        used when ``None`` is supplied.
    preferred_codec / preferred_quality:
        Forwarded to the ``FFmpegExtractAudio`` post-processor used by
        ``yt-dlp``.
    """

    if not is_valid_youtube_url(url):
        raise MediaIngestError(
            "That does not look like a YouTube URL. Paste a link that starts "
            "with https://www.youtube.com/ or https://youtu.be/."
        )

    try:
        from yt_dlp import YoutubeDL  # type: ignore
        from yt_dlp.utils import DownloadError  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MediaIngestError(
            "yt-dlp is required to download YouTube audio. Install "
            "requirements.txt (it is pinned there)."
        ) from exc

    # yt-dlp shells out to ffmpeg for the audio post-processor.
    _require_ffmpeg()

    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="yt_audio_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "%(title).120s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": preferred_codec,
                "preferredquality": preferred_quality,
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": False,
        # Retries help with flaky connections and transient 403s.
        "retries": 3,
        "fragment_retries": 3,
        # Default UA some YouTube "bot check" paths accept.
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        },
    }

    # YouTube often rotates which "player client" works at a given moment.
    # Try a few in order; the first one that succeeds wins. This avoids a
    # single transient client failure killing the whole request.
    client_fallbacks: tuple[tuple[str, ...], ...] = (
        ("android",),
        ("web",),
        ("tv_embedded",),
        ("ios",),
    )

    last_error: Exception | None = None
    info = None
    for clients in client_fallbacks:
        opts = {
            **base_opts,
            "extractor_args": {"youtube": {"player_client": list(clients)}},
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url.strip(), download=True)
            break
        except DownloadError as exc:
            last_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

    if info is None:
        raise MediaIngestError(_format_ytdlp_error(last_error)) from last_error

    # yt-dlp renames the post-processed file. Find whatever audio file is
    # now newest in the work directory rather than reconstructing its name.
    audio_path = _newest_audio_file(work_dir)
    if audio_path is None:
        raise MediaIngestError(
            "yt-dlp finished but no audio file was produced. Check ffmpeg "
            "installation and disk space."
        )

    return IngestedMedia(
        audio_path=audio_path,
        display_name=_sanitize_filename(title) + audio_path.suffix,
        source_description=f"YouTube: {title}",
    )


def extract_audio_from_video(
    video_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    display_name: Optional[str] = None,
) -> IngestedMedia:
    """Extract the audio track from *video_path* into a new ``.wav`` file.

    The audio is written as 16 kHz mono PCM — a safe, lossless input for the
    Whisper model and easy for the rest of the app (librosa/soundfile) to
    load.
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise MediaIngestError(f"Video file not found: {video_path}")

    ffmpeg_path = _require_ffmpeg()

    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="video_audio_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / (video_path.stem + ".wav")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",              # no video
        "-ac",
        "1",                # mono
        "-ar",
        "16000",            # 16 kHz
        "-f",
        "wav",
        str(audio_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 30,  # 30-minute ceiling for long videos
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaIngestError(
            "Audio extraction timed out after 30 minutes. Try a shorter clip "
            "or extract audio externally first."
        ) from exc
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise MediaIngestError("ffmpeg executable disappeared mid-run.") from exc

    if completed.returncode != 0 or not audio_path.exists():
        stderr_tail = (completed.stderr or "").strip().splitlines()[-5:]
        raise MediaIngestError(
            "ffmpeg could not extract audio from that video. "
            + ("Details: " + " | ".join(stderr_tail) if stderr_tail else "")
        )

    return IngestedMedia(
        audio_path=audio_path,
        display_name=(display_name or video_path.name),
        source_description=f"Video: {display_name or video_path.name}",
    )


def _format_ytdlp_error(exc: Exception | None) -> str:
    """Turn a raw yt-dlp error into something actionable for the UI."""

    if exc is None:
        return "YouTube download failed for an unknown reason."

    raw = str(exc).strip()
    # yt-dlp messages often have ANSI color and "[youtube] VIDEOID:" prefixes;
    # strip those so the UI shows a clean message.
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    # Keep the most informative tail of the message.
    clean = raw.splitlines()[-1] if raw else ""

    lowered = clean.lower()
    # Order matters: match the most specific messages first.
    if "sign in to confirm you" in lowered and "bot" in lowered:
        return (
            "YouTube is challenging this request with a bot check. This usually "
            "happens from cloud IPs (e.g. Streamlit Community Cloud). Try a "
            "different video, run the app locally, or supply YouTube cookies. "
            f"Original error: {clean}"
        )
    if "age-restricted" in lowered or "age restricted" in lowered:
        return (
            "This video is age-restricted and requires sign-in. "
            "yt-dlp cannot fetch it anonymously."
        )
    if "private video" in lowered:
        return "This YouTube video is private and cannot be downloaded."
    if "members-only" in lowered or "members only" in lowered:
        return "This video is members-only and cannot be downloaded anonymously."
    if "not available in your country" in lowered or "geo-restrict" in lowered:
        return "This video is region-restricted for your server's location."
    if "proxy" in lowered and "403" in lowered:
        return (
            "Network layer refused the connection (proxy 403). The host running "
            "this app cannot reach YouTube. Check outbound network / firewall "
            f"rules. Details: {clean}"
        )
    if "video unavailable" in lowered:
        return f"YouTube reports this video as unavailable. Details: {clean}"

    # Fallback: include the real yt-dlp message so the user can act on it.
    return f"YouTube download failed. Details: {clean or raw[:300]}"


def _newest_audio_file(directory: Path) -> Optional[Path]:
    audio_suffixes = {".mp3", ".m4a", ".wav", ".webm", ".opus", ".flac", ".ogg"}
    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in audio_suffixes]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


__all__ = [
    "IngestedMedia",
    "MediaIngestError",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "extract_audio_from_video",
    "fetch_youtube_audio",
    "is_valid_youtube_url",
]
