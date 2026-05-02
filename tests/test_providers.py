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
    LocalOnnxDiarizer,
    LocalOnnxWhisperTranscriber,
    LocalPyannoteDiarizer,
    SpeechAnalyzer,
    create_speech_analyzer,
    _resolve_onnx_model,
)
from voicescript.schemas import DiarizationResult, TranscriptionResult


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


def test_disabled_speech_providers_return_normalized_pydantic_results():
    transcription = DisabledTranscriber().transcribe(Path("court.wav"))
    diarization = DisabledDiarizer().diarize(Path("court.wav"))

    assert isinstance(transcription, TranscriptionResult)
    assert transcription.provider == "disabled"
    assert transcription.output_type == "none"
    assert transcription.transcript_segments == []
    assert transcription.limitations == ["Transcription disabled by provider configuration."]

    assert isinstance(diarization, DiarizationResult)
    assert diarization.provider == "disabled"
    assert diarization.output_type == "none"
    assert diarization.speaker_segments == []
    assert diarization.estimated_speaker_count is None
    assert diarization.limitations == ["Diarization disabled by provider configuration."]


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


def test_onnx_provider_selection_is_scoped_to_speech():
    settings = Settings(
        transcription_provider="onnx",
        diarization_provider="onnx",
        source_separation_provider="disabled",
    )

    speech = create_speech_analyzer(settings)

    assert speech.transcriber.provider_name == "local-onnx-whisper"
    assert speech.diarizer.provider_name == "local-onnx-diarization"


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
    speech_allowed = "Allowed values: api, disabled, huggingface, local, local-onnx, onnx, other"
    separation_allowed = "Allowed values: api, disabled, huggingface, local, other"
    with pytest.raises(ValueError, match=speech_allowed):
        create_speech_analyzer(Settings(transcription_provider="openai"))

    with pytest.raises(ValueError, match=speech_allowed):
        create_speech_analyzer(Settings(diarization_provider="cloudish"))

    with pytest.raises(ValueError, match=separation_allowed):
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

    result = diarizer.diarize(Path("court.wav"))

    assert isinstance(result, DiarizationResult)
    assert result.provider == "local-pyannote"
    assert result.output_type == "error"
    assert result.speaker_segments == []
    assert any("Diarization failed" in limitation for limitation in result.limitations)


def test_local_pyannote_diarizer_returns_limitation_for_unsupported_output(monkeypatch):
    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _UnsupportedOutputPipeline

    monkeypatch.setattr("voicescript.providers.speech._module_status", lambda *args, **kwargs: {"available": True, "detail": "available"})
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    diarizer = LocalPyannoteDiarizer(Settings(pyannote_auth_token="token"))

    result = diarizer.diarize(Path("court.wav"))

    assert isinstance(result, DiarizationResult)
    assert result.provider == "local-pyannote"
    assert result.output_type == "DiarizeOutput"
    assert result.speaker_segments == []
    assert result.estimated_speaker_count is None
    assert "Raw diarization output type: DiarizeOutput." in result.evidence
    assert any("unsupported output type DiarizeOutput" in limitation for limitation in result.limitations)


def test_local_pyannote_diarizer_normalizes_segment_sequence_output(monkeypatch):
    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _SequenceOutputPipeline

    monkeypatch.setattr("voicescript.providers.speech._module_status", lambda *args, **kwargs: {"available": True, "detail": "available"})
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    diarizer = LocalPyannoteDiarizer(Settings(pyannote_auth_token="token"))

    result = diarizer.diarize(Path("court.wav"))

    assert isinstance(result, DiarizationResult)
    assert result.output_type == "SequenceOutput"
    assert result.estimated_speaker_count == 2
    assert [segment.speaker for segment in result.speaker_segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment.start_seconds for segment in result.speaker_segments] == [0.0, 2.0]
    assert "Normalized 2 diarization segment(s)." in result.evidence


def test_onnx_speech_providers_report_missing_local_models_without_network(monkeypatch, runtime_dir):
    monkeypatch.setattr("voicescript.providers.speech._module_status", lambda *args, **kwargs: {"available": True, "detail": "available"})
    settings = Settings(
        model_cache_dir=runtime_dir / "models",
        transcription_provider="onnx",
        diarization_provider="onnx",
        whisper_onnx_model="missing-whisper.onnx",
        pyannote_onnx_model="missing-pyannote.onnx",
        model_fetch_policy="local_only",
    )

    speech = create_speech_analyzer(settings)
    result = speech.analyze(Path("court.wav"))

    assert isinstance(speech.transcriber, LocalOnnxWhisperTranscriber)
    assert isinstance(speech.diarizer, LocalOnnxDiarizer)
    assert speech.readiness()["onnxruntime"]["available"] is True
    assert result.transcript_text == ""
    assert result.speaker_segments == []
    assert any("missing-whisper.onnx" in limitation for limitation in result.limitations)
    assert any("missing-pyannote.onnx" in limitation for limitation in result.limitations)


def test_onnx_speech_providers_use_existing_model_files(monkeypatch, runtime_dir):
    monkeypatch.setattr("voicescript.providers.speech._module_status", lambda *args, **kwargs: {"available": True, "detail": "available"})
    whisper_model = runtime_dir / "models" / "whisper.onnx"
    diarization_model = runtime_dir / "models" / "diarization.onnx"
    whisper_model.parent.mkdir(parents=True)
    whisper_model.write_bytes(b"onnx")
    diarization_model.write_bytes(b"onnx")
    settings = Settings(
        model_cache_dir=runtime_dir / "models",
        transcription_provider="onnx",
        diarization_provider="onnx",
        whisper_onnx_model="whisper.onnx",
        pyannote_onnx_model="diarization.onnx",
        model_fetch_policy="local_only",
    )

    speech = create_speech_analyzer(settings)
    readiness = speech.readiness()
    result = speech.analyze(Path("court.wav"))

    assert readiness["transcription"]["available"] is True
    assert readiness["diarization"]["available"] is True
    assert readiness["transcription"]["model_path"] == str(whisper_model)
    assert readiness["diarization"]["model_path"] == str(diarization_model)
    assert any("ONNX transcription inference is not implemented" in item for item in result.limitations)
    assert any("ONNX diarization inference is not implemented" in item for item in result.limitations)


def test_onnx_model_resolution_fetches_only_when_enabled(runtime_dir):
    calls = []

    def fake_fetcher(**kwargs):
        calls.append(kwargs)
        model_dir = runtime_dir / "fetched"
        model_dir.mkdir()
        (model_dir / "whisper.onnx").write_bytes(b"onnx")
        return str(model_dir)

    disabled_settings = Settings(
        model_cache_dir=runtime_dir / "models-disabled",
        model_fetch_policy="local_only",
        onnx_fetch_enabled=False,
    )
    missing_path, limitations = _resolve_onnx_model(
        disabled_settings,
        "whisper.onnx",
        "voicescript/whisper-onnx",
        kind="whisper",
        fetcher=fake_fetcher,
    )
    assert missing_path is None
    assert calls == []
    assert any("fetch is disabled" in limitation for limitation in limitations)

    enabled_settings = Settings(
        model_cache_dir=runtime_dir / "models-enabled",
        model_fetch_policy="allow_download",
        onnx_fetch_enabled=True,
    )
    fetched_path, fetch_limitations = _resolve_onnx_model(
        enabled_settings,
        "whisper.onnx",
        "voicescript/whisper-onnx",
        kind="whisper",
        fetcher=fake_fetcher,
    )
    assert fetched_path == runtime_dir / "fetched" / "whisper.onnx"
    assert fetch_limitations == []
    assert calls and calls[0]["repo_id"] == "voicescript/whisper-onnx"


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


class DiarizeOutput:
    pass


class _UnsupportedOutputPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, payload):
        return DiarizeOutput()


class SequenceOutput:
    speaker_diarization = [
        {"start": 2, "end": 3, "speaker": " SPEAKER_01 "},
        {"start": 0, "end": 1.5, "speaker": "SPEAKER_00"},
    ]


class _SequenceOutputPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, payload):
        return SequenceOutput()
