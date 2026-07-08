"""Small REST client for the local Voicebox API.

Voicebox is a separate local voice studio. This module keeps integration
logic out of the Streamlit UI and talks to Voicebox over its HTTP API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
from urllib.parse import urljoin

import requests


DEFAULT_VOICEBOX_API_URL = "http://127.0.0.1:17493"

VOICEBOX_TTS_ENGINES = (
    "qwen",
    "qwen_custom_voice",
    "luxtts",
    "chatterbox",
    "chatterbox_turbo",
    "tada",
    "kokoro",
)

VOICEBOX_LANGUAGES = (
    "en",
    "zh",
    "ja",
    "ko",
    "de",
    "fr",
    "ru",
    "pt",
    "es",
    "it",
    "he",
    "ar",
    "da",
    "el",
    "fi",
    "hi",
    "ms",
    "nl",
    "no",
    "pl",
    "sv",
    "sw",
    "tr",
)


class VoiceboxError(RuntimeError):
    """Raised when Voicebox cannot complete a requested operation."""


@dataclass(frozen=True)
class VoiceboxProfile:
    id: str
    name: str
    language: str = "en"
    voice_type: str = "cloned"
    default_engine: str | None = None
    preset_engine: str | None = None
    sample_count: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VoiceboxProfile":
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name") or payload["id"]),
            language=str(payload.get("language") or "en"),
            voice_type=str(payload.get("voice_type") or "cloned"),
            default_engine=payload.get("default_engine"),
            preset_engine=payload.get("preset_engine"),
            sample_count=int(payload.get("sample_count") or 0),
        )

    @property
    def label(self) -> str:
        engine = self.default_engine or self.preset_engine
        suffix_parts = [self.voice_type]
        if engine:
            suffix_parts.append(engine)
        suffix = ", ".join(suffix_parts)
        return f"{self.name} ({suffix})"


@dataclass(frozen=True)
class VoiceboxGeneration:
    id: str
    status: str
    text: str
    language: str
    duration: float | None = None
    error: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VoiceboxGeneration":
        return cls(
            id=str(payload["id"]),
            status=str(payload.get("status") or "completed"),
            text=str(payload.get("text") or ""),
            language=str(payload.get("language") or "en"),
            duration=payload.get("duration"),
            error=payload.get("error"),
        )


class VoiceboxClient:
    """HTTP wrapper for Voicebox's profile, generation, and audio endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_VOICEBOX_API_URL,
        *,
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "health")

    def list_profiles(self) -> list[VoiceboxProfile]:
        payload = self._request_json("GET", "profiles")
        if not isinstance(payload, list):
            raise VoiceboxError("Voicebox returned an unexpected profiles response.")
        return [VoiceboxProfile.from_payload(item) for item in payload]

    def generate_speech(
        self,
        *,
        profile_id: str,
        text: str,
        language: str = "en",
        engine: str | None = None,
        model_size: str | None = None,
        instruct: str | None = None,
        max_chunk_chars: int = 800,
        crossfade_ms: int = 50,
        normalize: bool = True,
    ) -> VoiceboxGeneration:
        data: dict[str, Any] = {
            "profile_id": profile_id,
            "text": text,
            "language": language,
            "max_chunk_chars": max_chunk_chars,
            "crossfade_ms": crossfade_ms,
            "normalize": normalize,
        }
        if engine:
            data["engine"] = engine
        if model_size:
            data["model_size"] = model_size
        if instruct:
            data["instruct"] = instruct
        payload = self._request_json("POST", "generate", json=data)
        return VoiceboxGeneration.from_payload(payload)

    def get_generation(self, generation_id: str) -> VoiceboxGeneration:
        payload = self._request_json("GET", f"history/{generation_id}")
        return VoiceboxGeneration.from_payload(payload)

    def wait_for_generation(
        self,
        generation_id: str,
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 1.5,
    ) -> VoiceboxGeneration:
        deadline = time.monotonic() + timeout_seconds
        last_generation: VoiceboxGeneration | None = None
        while time.monotonic() < deadline:
            last_generation = self.get_generation(generation_id)
            if last_generation.status == "completed":
                return last_generation
            if last_generation.status == "failed":
                detail = last_generation.error or "Voicebox generation failed."
                raise VoiceboxError(detail)
            time.sleep(poll_interval_seconds)
        status = last_generation.status if last_generation else "unknown"
        raise VoiceboxError(f"Voicebox generation timed out while status was '{status}'.")

    def fetch_audio(self, generation_id: str) -> bytes:
        response = self._request("GET", f"audio/{generation_id}")
        if not response.content:
            raise VoiceboxError("Voicebox returned an empty audio file.")
        return bytes(response.content)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise VoiceboxError("Voicebox returned a non-JSON response.") from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = urljoin(self.base_url, path.lstrip("/"))
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            message = _response_error_message(exc)
            raise VoiceboxError(message) from exc


def _response_error_message(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        request_url = getattr(exc.request, "url", "the configured URL")
        return f"Could not reach Voicebox. Is it running at {request_url}?"
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if detail:
        return f"Voicebox error: {detail}"
    return f"Voicebox returned HTTP {response.status_code}."


__all__ = [
    "DEFAULT_VOICEBOX_API_URL",
    "VOICEBOX_LANGUAGES",
    "VOICEBOX_TTS_ENGINES",
    "VoiceboxClient",
    "VoiceboxError",
    "VoiceboxGeneration",
    "VoiceboxProfile",
]
