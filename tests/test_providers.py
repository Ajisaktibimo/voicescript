from pathlib import Path
import sys
import types

import pytest

from voicescript.config import Settings
from voicescript.providers import separation
from voicescript.providers.separation import DisabledSourceSeparator, create_source_separator
from voicescript.providers.speech import (
    DisabledDiarizer,
    DisabledTranscriber,
    LocalPyannoteDiarizer,
    SpeechAnalyzer,
    create_speech_analyzer,
)


def test_settings_loads_provider_selection_from_env_file(runtime_dir, monkeypatch):
    (runtime_dir / ".env").write_text(
        "\n".join(
            [
                "VOICESCRIPT_API_KEY=from-env-file",
                "VOICESCRIPT_TRANSCRIPTION_PROVIDER=disabled",
                "VOICESCRIPT_DIARIZATION_PROVIDER=disabled",
                "VOICESCRIPT_SOURCE_SEPARATION_PROVIDER=disabled",
                "VOICESCRIPT_WHISPER_DEVICE=cpu",
                "VOICESCRIPT_WHISPER_COMPUTE_TYPE=int8",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(runtime_dir)

    settings = Settings.from_env()

    assert settings.transcription_provider == "disabled"
    assert settings.diarization_provider == "disabled"
    assert settings.source_separation_provider == "disabled"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"


def test_speech_analyzer_can_be_composed_from_disabled_providers():
    analyzer = SpeechAnalyzer(
        transcriber=DisabledTranscriber(),
        diarizer=DisabledDiarizer(),
    )

    readiness = analyzer.readiness()
    result = analyzer.analyze(Path("court.wav"))

    assert readiness["transcription"]["provider"] == "disabled"
    assert readiness["diarization"]["provider"] == "disabled"
    assert result.transcript_text == ""
    assert result.speaker_segments == []
    assert "Transcription disabled" in result.limitations[0]
    assert "Diarization disabled" in result.limitations[1]


@pytest.mark.parametrize(
    ("provider_name", "expected_transcriber", "expected_diarizer", "expected_separator"),
    [
        ("local", "local-faster-whisper", "local-pyannote", "local-demucs"),
        ("disabled", "disabled", "disabled", "disabled"),
        ("huggingface", "huggingface", "huggingface", "huggingface"),
        ("api", "api", "api", "api"),
        ("other", "other", "other", "other"),
    ],
)
def test_provider_selection_accepts_supported_names(
    provider_name,
    expected_transcriber,
    expected_diarizer,
    expected_separator,
):
    settings = Settings(
        transcription_provider=provider_name,
        diarization_provider=provider_name,
        source_separation_provider=provider_name,
    )

    speech = create_speech_analyzer(settings)
    separator = create_source_separator(settings)

    assert speech.transcriber.provider_name == expected_transcriber
    assert speech.diarizer.provider_name == expected_diarizer
    assert separator.provider_name == expected_separator


@pytest.mark.parametrize("provider_name", ["huggingface", "api", "other"])
def test_nonlocal_provider_selection_is_configured_without_network_calls(provider_name, runtime_dir):
    settings = Settings(
        transcription_provider=provider_name,
        diarization_provider=provider_name,
        source_separation_provider=provider_name,
    )

    speech = create_speech_analyzer(settings)
    separator = create_source_separator(settings)
    speech_result = speech.analyze(Path("court.wav"))
    separation_result = separator.separate_vocals(Path("court.wav"), runtime_dir)

    assert provider_name in speech.readiness()["transcription"]["provider"]
    assert provider_name in speech.readiness()["diarization"]["provider"]
    assert provider_name in separator.readiness()["source_separation"]["provider"]
    assert speech_result.transcript_text == ""
    assert any("no network call was attempted" in item for item in speech_result.limitations)
    assert separation_result.vocals_path is None
    assert any("no network call was attempted" in item for item in separation_result.limitations)


def test_provider_selection_rejects_other_names():
    allowed = "Allowed values: api, disabled, huggingface, local, other"
    with pytest.raises(ValueError, match=allowed):
        create_speech_analyzer(Settings(transcription_provider="openai"))

    with pytest.raises(ValueError, match=allowed):
        create_speech_analyzer(Settings(diarization_provider="cloudish"))

    with pytest.raises(ValueError, match=allowed):
        create_source_separator(Settings(source_separation_provider="demuc"))


def test_local_demucs_uses_current_python_interpreter(monkeypatch, runtime_dir):
    calls = []

    monkeypatch.setattr(separation, "_demucs_available", lambda: True)
    monkeypatch.setattr(separation.subprocess, "run", lambda command, **kwargs: calls.append(command) or _Completed())

    separator = create_source_separator(Settings(source_separation_provider="local"))
    separator.separate_vocals(Path("court.wav"), runtime_dir)

    assert calls
    assert calls[0][0] == separation.sys.executable


def test_local_pyannote_diarizer_returns_limitation_when_audio_loading_fails(monkeypatch):
    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _FakePipeline

    monkeypatch.setattr("voicescript.providers.speech._module_status", lambda *args, **kwargs: {"available": True, "detail": "available"})
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    diarizer = LocalPyannoteDiarizer(Settings(pyannote_auth_token="token"))

    segments, limitations = diarizer.diarize(Path("court.wav"))

    assert segments == []
    assert any("Diarization failed" in limitation for limitation in limitations)


class _Completed:
    returncode = 0
    stdout = ""
    stderr = ""


class _FakePipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, payload):
        raise TypeError("'>' not supported between instances of 'MagicMock' and 'MagicMock'")
