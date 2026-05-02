from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from voicescript.config import Settings
from voicescript.schemas import SpeakerSegment, SpeechAnalysisResult, TranscriptSegment


ALLOWED_PROVIDER_NAMES = ("api", "disabled", "huggingface", "local", "local-onnx", "onnx", "other")


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


class LocalOnnxWhisperTranscriber:
    provider_name = "local-onnx-whisper"

    def __init__(self, settings: Settings):
        self.settings = settings

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        runtime_status = _module_status("onnxruntime")
        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.whisper_onnx_model,
            self.settings.whisper_onnx_repo_id,
            kind="whisper",
        )
        return {
            "transcription": {
                "available": bool(runtime_status["available"] and model_path),
                "provider": self.provider_name,
                "detail": "available" if model_path else "; ".join(limitations),
                "model_path": str(model_path) if model_path else "",
            },
            "onnxruntime": runtime_status,
        }

    def transcribe(self, input_file: Path) -> tuple[list[TranscriptSegment], list[str]]:
        runtime_status = _module_status("onnxruntime")
        if not runtime_status["available"]:
            return [], ["ONNX transcription skipped because onnxruntime is not installed."]

        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.whisper_onnx_model,
            self.settings.whisper_onnx_repo_id,
            kind="whisper",
        )
        if not model_path:
            return [], limitations

        return [], [
            f"ONNX transcription inference is not implemented yet for model '{model_path}'. "
            "Model resolution succeeded; add a Whisper ONNX tokenizer/preprocessor adapter before relying on transcript output."
        ]


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


class LocalOnnxDiarizer:
    provider_name = "local-onnx-diarization"

    def __init__(self, settings: Settings):
        self.settings = settings

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        runtime_status = _module_status("onnxruntime")
        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.pyannote_onnx_model,
            self.settings.pyannote_onnx_repo_id,
            kind="pyannote",
        )
        return {
            "diarization": {
                "available": bool(runtime_status["available"] and model_path),
                "provider": self.provider_name,
                "detail": "available" if model_path else "; ".join(limitations),
                "model_path": str(model_path) if model_path else "",
            },
            "onnxruntime": runtime_status,
        }

    def diarize(self, input_file: Path) -> tuple[list[SpeakerSegment], list[str]]:
        runtime_status = _module_status("onnxruntime")
        if not runtime_status["available"]:
            return [], ["ONNX diarization skipped because onnxruntime is not installed."]

        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.pyannote_onnx_model,
            self.settings.pyannote_onnx_repo_id,
            kind="pyannote",
        )
        if not model_path:
            return [], limitations

        return [], [
            f"ONNX diarization inference is not implemented yet for model '{model_path}'. "
            "Pyannote-style ONNX diarization still needs segmentation, embedding, and clustering glue."
        ]


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
    if provider in {"onnx", "local-onnx"}:
        return LocalOnnxWhisperTranscriber(settings)
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
    if provider in {"onnx", "local-onnx"}:
        return LocalOnnxDiarizer(settings)
    if provider in {"api", "huggingface", "other"}:
        return UnavailableDiarizer(provider)
    raise ValueError(
        f"Unsupported diarization provider '{settings.diarization_provider}'. {_allowed_provider_message()}"
    )


def _normalise_provider(provider: str) -> str:
    return provider.strip().lower()


def _allowed_provider_message() -> str:
    return f"Allowed values: {', '.join(ALLOWED_PROVIDER_NAMES)}."


def _resolve_onnx_model(
    settings: Settings,
    model_reference: str,
    repo_id: str | None,
    *,
    kind: str,
    fetcher=None,
) -> tuple[Path | None, list[str]]:
    model_dir = _onnx_model_dir(settings)
    candidate = Path(model_reference)
    if not candidate.is_absolute():
        candidate = model_dir / candidate
    if candidate.exists():
        return candidate, []

    fetch_enabled = settings.onnx_fetch_enabled or settings.model_fetch_policy == "allow_download"
    if not fetch_enabled:
        return None, [
            f"ONNX {kind} model is missing at {candidate}; model fetch is disabled "
            f"(policy={settings.model_fetch_policy})."
        ]
    if not repo_id:
        return None, [f"ONNX {kind} model fetch is enabled, but no Hugging Face repo id is configured."]

    try:
        fetched_path = _fetch_onnx_model(repo_id, model_dir, fetcher=fetcher)
    except Exception as exc:
        return None, [f"ONNX {kind} model fetch failed for repo '{repo_id}': {exc}"]

    fetched_candidate = Path(fetched_path)
    if fetched_candidate.is_file():
        return fetched_candidate, []
    nested_candidate = fetched_candidate / Path(model_reference).name
    if nested_candidate.exists():
        return nested_candidate, []
    return None, [f"ONNX {kind} model fetch completed, but '{Path(model_reference).name}' was not found."]


def _fetch_onnx_model(repo_id: str, cache_dir: Path, *, fetcher=None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if fetcher is None:
        from huggingface_hub import snapshot_download

        fetcher = snapshot_download
    return Path(
        fetcher(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir / repo_id.replace("/", "__")),
        )
    )


def _onnx_model_dir(settings: Settings) -> Path:
    default = Path("data/models")
    if settings.onnx_model_dir == default and settings.model_cache_dir != default:
        return settings.model_cache_dir
    return settings.onnx_model_dir


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
