from __future__ import annotations

import pytest
import requests

from digit_recognition.voicebox_client import VoiceboxClient, VoiceboxError


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", status_code=200):
        self.payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError("boom")
            error.response = self
            raise error


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_list_profiles_normalizes_payload():
    session = FakeSession(
        [
            FakeResponse(
                [
                    {
                        "id": "profile-1",
                        "name": "Morgan",
                        "language": "en",
                        "voice_type": "preset",
                        "preset_engine": "kokoro",
                    }
                ]
            )
        ]
    )
    client = VoiceboxClient("http://voicebox.local", session=session)

    profiles = client.list_profiles()

    assert profiles[0].id == "profile-1"
    assert profiles[0].label == "Morgan (preset, kokoro)"
    assert session.calls[0][1] == "http://voicebox.local/profiles"


def test_generate_speech_sends_voicebox_payload():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "id": "gen-1",
                    "status": "generating",
                    "text": "Hello",
                    "language": "en",
                }
            )
        ]
    )
    client = VoiceboxClient("http://voicebox.local/", session=session)

    generation = client.generate_speech(
        profile_id="profile-1",
        text="Hello",
        language="en",
        engine="kokoro",
        model_size="1.7B",
        instruct="warm",
    )

    assert generation.id == "gen-1"
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "http://voicebox.local/generate"
    assert kwargs["json"]["profile_id"] == "profile-1"
    assert kwargs["json"]["engine"] == "kokoro"
    assert kwargs["json"]["instruct"] == "warm"


def test_fetch_audio_rejects_empty_response():
    session = FakeSession([FakeResponse(content=b"")])
    client = VoiceboxClient("http://voicebox.local", session=session)

    with pytest.raises(VoiceboxError, match="empty audio"):
        client.fetch_audio("gen-1")


def test_http_error_uses_voicebox_detail():
    session = FakeSession([FakeResponse({"detail": "Profile not found"}, status_code=404)])
    client = VoiceboxClient("http://voicebox.local", session=session)

    with pytest.raises(VoiceboxError, match="Profile not found"):
        client.list_profiles()
