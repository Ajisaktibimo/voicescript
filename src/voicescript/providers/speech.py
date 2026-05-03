from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from voicescript.config import Settings
from voicescript.schemas import SpeakerSegment, SpeechAnalysisResult, TranscriptSegment


ALLOWED_PROVIDER_NAMES = ("api", "disabled", "huggingface", "local", "other")


class TranscriptionProvider(Protocol):
    provider_name: str

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        ...

    def transcribe(self, input_file: Path) -> tuple[list[TranscriptSegment], list[str]]:
        ...


class DiarizationProvider(Protocol):
    provider_name: str

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        ...

    def diarize(self, input_file: Path) -> tuple[list[SpeakerSegment], list[str]]:
        ...


class DisabledTranscriber:
    provider_name = "disabled"

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "transcription": {"available": False, "provider": self.provider_name, "detail": "disabled"},
            "faster_whisper": {"available": False, "detail": "disabled"},
        }

    def transcribe(self, input_file: Path) -> tuple[list[TranscriptSegment], list[str]]:
        return [], ["Transcription disabled by provider configuration."]


class DisabledDiarizer:
    provider_name = "disabled"

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "diarization": {"available": False, "provider": self.provider_name, "detail": "disabled"},
            "pyannote.audio": {"available": False, "detail": "disabled"},
        }

    def diarize(self, input_file: Path) -> tuple[list[SpeakerSegment], list[str]]:
        return [], ["Diarization disabled by provider configuration."]


class UnavailableTranscriber:
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "transcription": {
                "available": False,
                "provider": self.provider_name,
                "detail": "selected but no offline adapter is configured",
            }
        }

    def transcribe(self, input_file: Path) -> tuple[list[TranscriptSegment], list[str]]:
        return [], [
            f"Provider '{self.provider_name}' is selected but no offline transcription adapter is configured; "
            "no network call was attempted."
        ]


class UnavailableDiarizer:
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "diarization": {
                "available": False,
                "provider": self.provider_name,
                "detail": "selected but no offline adapter is configured",
            }
        }

    def diarize(self, input_file: Path) -> tuple[list[SpeakerSegment], list[str]]:
        return [], [
            f"Provider '{self.provider_name}' is selected but no offline diarization adapter is configured; "
            "no network call was attempted."
        ]


class LocalWhisperTranscriber:
    provider_name = "local-faster-whisper"

    def __init__(self, settings: Settings):
        self.settings = settings

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        status = _module_status("faster_whisper")
        return {
            "transcription": {
                "available": status["available"],
                "provider": self.provider_name,
                "detail": status["detail"],
            },
            "faster_whisper": status,
        }

    def transcribe(self, input_file: Path) -> tuple[list[TranscriptSegment], list[str]]:
        status = _module_status("faster_whisper")
        if not status["available"]:
            return [], ["Transcription skipped because faster-whisper is not installed."]

        from faster_whisper import WhisperModel

        model = WhisperModel(
            self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
        )
        segments, _info = model.transcribe(str(input_file), vad_filter=True)
        return [
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
        ], []


class LocalPyannoteDiarizer:
    provider_name = "local-pyannote"

    def __init__(self, settings: Settings):
        self.settings = settings

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        status = _module_status("pyannote.audio", needs_token=True, token=self.settings.pyannote_auth_token)
        return {
            "diarization": {
                "available": status["available"] and bool(self.settings.pyannote_auth_token),
                "provider": self.provider_name,
                "detail": status["detail"],
            },
            "pyannote.audio": status,
        }

    def diarize(self, input_file: Path) -> tuple[list[SpeakerSegment], list[str]]:
        status = _module_status("pyannote.audio", needs_token=True, token=self.settings.pyannote_auth_token)
        if not status["available"]:
            return [], ["Diarization skipped because pyannote.audio is not installed."]
        if not self.settings.pyannote_auth_token:
            return [], ["Diarization skipped because PYANNOTE_AUTH_TOKEN is not configured."]

        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                self.settings.pyannote_model,
                token=self.settings.pyannote_auth_token,
            )
            diarization = pipeline(str(input_file))
        except Exception as exc:
            return [], [f"Diarization failed in local pyannote provider: {exc}"]

        segments: list[SpeakerSegment] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=str(speaker),
                    start_seconds=float(turn.start),
                    end_seconds=float(turn.end),
                    text="",
                )
            )
        return segments, []


class SpeechAnalyzer:
    def __init__(self, transcriber: TranscriptionProvider, diarizer: DiarizationProvider):
        self.transcriber = transcriber
        self.diarizer = diarizer

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {**self.transcriber.readiness(), **self.diarizer.readiness()}

    def analyze(self, input_file: Path) -> SpeechAnalysisResult:
        transcript_segments, transcription_limits = self.transcriber.transcribe(input_file)
        diarization_segments, diarization_limits = self.diarizer.diarize(input_file)
        speaker_segments = _align_transcript_to_speakers(transcript_segments, diarization_segments)
        if diarization_segments and not transcript_segments:
            speaker_segments = diarization_segments
        transcript_text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
        return SpeechAnalysisResult(
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            speaker_segments=speaker_segments,
            limitations=[*transcription_limits, *diarization_limits],
            readiness=self.readiness(),
        )


def create_speech_analyzer(settings: Settings) -> SpeechAnalyzer:
    return SpeechAnalyzer(
        transcriber=_create_transcriber(settings),
        diarizer=_create_diarizer(settings),
    )


def _create_transcriber(settings: Settings) -> TranscriptionProvider:
    provider = _normalise_provider(settings.transcription_provider)
    if provider == "disabled":
        return DisabledTranscriber()
    if provider == "local":
        return LocalWhisperTranscriber(settings)
    if provider in {"api", "huggingface", "other"}:
        return UnavailableTranscriber(provider)
    raise ValueError(
        f"Unsupported transcription provider '{settings.transcription_provider}'. {_allowed_provider_message()}"
    )


def _create_diarizer(settings: Settings) -> DiarizationProvider:
    provider = _normalise_provider(settings.diarization_provider)
    if provider == "disabled":
        return DisabledDiarizer()
    if provider == "local":
        return LocalPyannoteDiarizer(settings)
    if provider in {"api", "huggingface", "other"}:
        return UnavailableDiarizer(provider)
    raise ValueError(
        f"Unsupported diarization provider '{settings.diarization_provider}'. {_allowed_provider_message()}"
    )


def _normalise_provider(provider: str) -> str:
    return provider.strip().lower()


def _allowed_provider_message() -> str:
    return f"Allowed values: {', '.join(ALLOWED_PROVIDER_NAMES)}."


def _module_status(module_name: str, *, needs_token: bool = False, token: str | None = None) -> dict[str, str | bool]:
    try:
        installed = find_spec(module_name) is not None
    except ModuleNotFoundError:
        installed = False
    if not installed:
        return {"available": False, "detail": "not installed"}
    if needs_token and not token:
        return {"available": True, "detail": "installed; token not configured"}
    return {"available": True, "detail": "available"}


def _align_transcript_to_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[SpeakerSegment],
) -> list[SpeakerSegment]:
    if not diarization_segments:
        return []
    if not transcript_segments:
        return diarization_segments

    aligned: list[SpeakerSegment] = []
    for transcript in transcript_segments:
        best = _best_overlap(transcript, diarization_segments)
        aligned.append(
            SpeakerSegment(
                speaker=best.speaker if best else "UNKNOWN",
                start_seconds=transcript.start_seconds,
                end_seconds=transcript.end_seconds,
                text=transcript.text,
            )
        )
    return aligned


def _best_overlap(transcript: TranscriptSegment, diarization_segments: list[SpeakerSegment]) -> SpeakerSegment | None:
    best_segment: SpeakerSegment | None = None
    best_overlap = 0.0
    for segment in diarization_segments:
        overlap = max(
            0.0,
            min(transcript.end_seconds, segment.end_seconds) - max(transcript.start_seconds, segment.start_seconds),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment
    return best_segment
