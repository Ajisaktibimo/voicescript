from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from voicescript.config import Settings
from voicescript.schemas import AudioMetadata, DiarizationResult, SpeakerSegment, SpeechAnalysisResult, TranscriptionResult, TranscriptSegment


ALLOWED_PROVIDER_NAMES = ("api", "disabled", "huggingface", "local", "local-onnx", "onnx", "other")


class TranscriptionProvider(Protocol):
    provider_name: str

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        ...

    def transcribe(self, input_file: Path) -> TranscriptionResult:
        ...


class DiarizationProvider(Protocol):
    provider_name: str

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        ...

    def diarize(self, input_file: Path, *, audio_metadata: AudioMetadata | None = None) -> DiarizationResult:
        ...


class DisabledTranscriber:
    provider_name = "disabled"

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "transcription": {"available": False, "provider": self.provider_name, "detail": "disabled"},
            "faster_whisper": {"available": False, "detail": "disabled"},
        }

    def transcribe(self, input_file: Path) -> TranscriptionResult:
        return _transcription_result(
            provider=self.provider_name,
            input_path=input_file,
            output_type="none",
            limitations=["Transcription disabled by provider configuration."],
        )


class DisabledDiarizer:
    provider_name = "disabled"

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "diarization": {"available": False, "provider": self.provider_name, "detail": "disabled"},
            "pyannote.audio": {"available": False, "detail": "disabled"},
        }

    def diarize(self, input_file: Path, *, audio_metadata: AudioMetadata | None = None) -> DiarizationResult:
        return _diarization_result(
            provider=self.provider_name,
            input_path=input_file,
            output_type="none",
            limitations=["Diarization disabled by provider configuration."],
        )


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

    def transcribe(self, input_file: Path) -> TranscriptionResult:
        return _transcription_result(
            provider=self.provider_name,
            input_path=input_file,
            output_type="none",
            limitations=[
                f"Provider '{self.provider_name}' is selected but no offline transcription adapter is configured; "
                "no network call was attempted."
            ],
        )


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

    def diarize(self, input_file: Path, *, audio_metadata: AudioMetadata | None = None) -> DiarizationResult:
        return _diarization_result(
            provider=self.provider_name,
            input_path=input_file,
            output_type="none",
            limitations=[
                f"Provider '{self.provider_name}' is selected but no offline diarization adapter is configured; "
                "no network call was attempted."
            ],
        )


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

    def transcribe(self, input_file: Path) -> TranscriptionResult:
        status = _module_status("faster_whisper")
        if not status["available"]:
            return _transcription_result(
                provider=self.provider_name,
                model=self.settings.whisper_model,
                input_path=input_file,
                output_type="none",
                limitations=["Transcription skipped because faster-whisper is not installed."],
            )

        from faster_whisper import WhisperModel

        model = WhisperModel(
            self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
        )
        segments, _info = model.transcribe(str(input_file), vad_filter=True)
        return _transcription_result(
            provider=self.provider_name,
            model=self.settings.whisper_model,
            input_path=input_file,
            output_type=type(segments).__name__,
            transcript_segments=[
                TranscriptSegment(
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments
            ],
        )


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

    def transcribe(self, input_file: Path) -> TranscriptionResult:
        runtime_status = _module_status("onnxruntime")
        if not runtime_status["available"]:
            return _transcription_result(
                provider=self.provider_name,
                model=self.settings.whisper_onnx_model,
                input_path=input_file,
                output_type="none",
                limitations=["ONNX transcription skipped because onnxruntime is not installed."],
            )

        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.whisper_onnx_model,
            self.settings.whisper_onnx_repo_id,
            kind="whisper",
        )
        if not model_path:
            return _transcription_result(
                provider=self.provider_name,
                model=self.settings.whisper_onnx_model,
                input_path=input_file,
                output_type="none",
                limitations=limitations,
            )

        return _transcription_result(
            provider=self.provider_name,
            model=str(model_path),
            input_path=input_file,
            output_type="onnx",
            limitations=[
                f"ONNX transcription inference is not implemented yet for model '{model_path}'. "
                "Model resolution succeeded; add a Whisper ONNX tokenizer/preprocessor adapter before relying on transcript output."
            ],
            evidence=[f"Resolved ONNX transcription model at {model_path}."],
        )


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

    def diarize(
        self,
        input_file: Path,
        *,
        audio_metadata: AudioMetadata | None = None,
    ) -> DiarizationResult:
        status = _module_status("pyannote.audio", needs_token=True, token=self.settings.pyannote_auth_token)
        if not status["available"]:
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_model,
                input_path=input_file,
                output_type="none",
                limitations=["Diarization skipped because pyannote.audio is not installed."],
            )
        if not self.settings.pyannote_auth_token:
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_model,
                input_path=input_file,
                output_type="none",
                limitations=["Diarization skipped because PYANNOTE_AUTH_TOKEN is not configured."],
            )

        if audio_metadata and audio_metadata.duration_seconds and audio_metadata.duration_seconds < 0.5:
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_model,
                input_path=input_file,
                output_type="none",
                limitations=["Audio duration is too short for reliable diarization (minimum 0.5s required)."],
            )

        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                self.settings.pyannote_model,
                token=self.settings.pyannote_auth_token,
            )
            
            clustering_params = self._resolve_clustering_params(audio_metadata)
            if clustering_params:
                pipeline.instantiate({"clustering": clustering_params})

            diarization = pipeline(str(input_file))
        except Exception as exc:
            err_msg = str(exc)
            limitations = [f"Diarization failed in local pyannote provider: {exc}"]
            if "degrees of freedom" in err_msg or "reduction factor" in err_msg:
                limitations.append("PyTorch internal error: input window too small for standard deviation calculation.")
            
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_model,
                input_path=input_file,
                output_type="error",
                limitations=limitations,
            )

        output_type = type(diarization).__name__
        segments, limitations = _parse_pyannote_diarization(diarization)
        evidence = [f"Raw diarization output type: {output_type}."]
        if clustering_params:
            evidence.append(f"Pyannote clustering parameters overrides: {clustering_params}.")
        return _diarization_result(
            provider=self.provider_name,
            model=self.settings.pyannote_model,
            input_path=input_file,
            output_type=output_type,
            speaker_segments=segments,
            limitations=limitations,
            evidence=evidence,
        )

    def _adaptive_clustering_params(self, metadata: AudioMetadata | None) -> dict | None:
        if metadata is None:
            return None

        base_threshold = 0.7045
        base_cluster_size = 12
        
        threshold_degradation = 0.0
        cluster_size_reduction = 0

        if metadata.bitrate and metadata.bitrate < 128000:
            threshold_degradation += 0.15
            cluster_size_reduction += 4
        if metadata.bitrate and metadata.bitrate < 64000:
            threshold_degradation += 0.10
            cluster_size_reduction += 4

        if metadata.sample_rate and metadata.sample_rate <= 16000:
            threshold_degradation += 0.05

        if metadata.channels and metadata.channels == 1:
            threshold_degradation += 0.02

        if threshold_degradation == 0.0 and cluster_size_reduction == 0:
            return None

        return {
            "threshold": max(0.30, base_threshold - threshold_degradation),
            "min_cluster_size": max(2, base_cluster_size - cluster_size_reduction)
        }

    def _resolve_clustering_params(self, metadata: AudioMetadata | None = None) -> dict | None:
        params: dict = {}
        
        adaptive_params = self._adaptive_clustering_params(metadata) or {}

        if self.settings.pyannote_clustering_threshold is not None:
            params["threshold"] = self.settings.pyannote_clustering_threshold
        elif "threshold" in adaptive_params:
            params["threshold"] = adaptive_params["threshold"]
                
        if self.settings.pyannote_min_cluster_size is not None:
            params["min_cluster_size"] = self.settings.pyannote_min_cluster_size
        elif "min_cluster_size" in adaptive_params:
            params["min_cluster_size"] = adaptive_params["min_cluster_size"]
            
        if params:
            params.setdefault("method", "centroid")
        return params or None


def _parse_pyannote_diarization(diarization: object) -> tuple[list[SpeakerSegment], list[str]]:
    itertracks = getattr(diarization, "itertracks", None)
    if callable(itertracks):
        return _parse_pyannote_itertracks(itertracks)

    for field_name in ("speaker_diarization", "diarization", "segments"):
        nested_output = getattr(diarization, field_name, None)
        if nested_output is not None and nested_output is not diarization:
            segments, limitations = _parse_pyannote_diarization(nested_output)
            if segments or not _is_unsupported_output_limitation(limitations):
                return segments, limitations

    if isinstance(diarization, (list, tuple)):
        return _parse_speaker_segment_items(diarization, source="sequence")

    return [], [
        f"Diarization produced unsupported output type {type(diarization).__name__}; "
        "speaker segments were not parsed."
    ]


def _parse_pyannote_itertracks(itertracks) -> tuple[list[SpeakerSegment], list[str]]:
    segments: list[SpeakerSegment] = []
    try:
        for turn, _track, speaker in itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=str(speaker),
                    start_seconds=float(turn.start),
                    end_seconds=float(turn.end),
                    text="",
                )
            )
    except Exception as exc:
        return [], [f"Diarization output parsing failed in local pyannote provider: {exc}"]
    return segments, []


def _parse_speaker_segment_items(items: object, *, source: str) -> tuple[list[SpeakerSegment], list[str]]:
    segments: list[SpeakerSegment] = []
    try:
        for item in items:
            segment = _speaker_segment_from_item(item)
            if segment is not None:
                segments.append(segment)
    except TypeError:
        return [], [f"Diarization output field '{source}' is not iterable."]
    except Exception as exc:
        return [], [f"Diarization output parsing failed from '{source}': {exc}"]

    if not segments:
        return [], [f"Diarization output field '{source}' had no parseable speaker segments."]
    return segments, []


def _speaker_segment_from_item(item: object) -> SpeakerSegment | None:
    if isinstance(item, dict):
        start = _first_mapping_value(item, ("start", "start_seconds"))
        end = _first_mapping_value(item, ("end", "end_seconds"))
        speaker = _first_mapping_value(item, ("speaker", "label"))
    else:
        start = _first_attribute_value(item, ("start", "start_seconds"))
        end = _first_attribute_value(item, ("end", "end_seconds"))
        speaker = _first_attribute_value(item, ("speaker", "label"))

    if start is None or end is None or speaker is None:
        return None
    return SpeakerSegment(
        speaker=str(speaker),
        start_seconds=float(start),
        end_seconds=float(end),
        text="",
    )


def _first_mapping_value(item: dict, names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in item:
            return item[name]
    return None


def _first_attribute_value(item: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _is_unsupported_output_limitation(limitations: list[str]) -> bool:
    return bool(limitations) and "unsupported output type" in limitations[0]


def _transcription_result(
    *,
    provider: str,
    model: str | None = None,
    input_source: str = "provider_input",
    input_path: Path | str | None = None,
    output_type: str,
    transcript_segments: list[TranscriptSegment] | None = None,
    limitations: list[str] | None = None,
    evidence: list[str] | None = None,
) -> TranscriptionResult:
    normalized_segments = _normalize_transcript_segments(transcript_segments or [])
    transcript_text = " ".join(segment.text for segment in normalized_segments if segment.text)
    normalized_evidence = list(evidence or [])
    if normalized_segments:
        normalized_evidence.append(f"Normalized {len(normalized_segments)} transcript segment(s).")
    return TranscriptionResult(
        provider=provider,
        model=model,
        input_source=input_source,
        input_path=str(input_path) if input_path is not None else None,
        output_type=output_type,
        transcript_text=transcript_text,
        transcript_segments=normalized_segments,
        confidence="medium" if normalized_segments else "unknown",
        limitations=limitations or [],
        evidence=normalized_evidence,
    )


def _diarization_result(
    *,
    provider: str,
    model: str | None = None,
    input_source: str = "provider_input",
    input_path: Path | str | None = None,
    output_type: str,
    speaker_segments: list[SpeakerSegment] | None = None,
    limitations: list[str] | None = None,
    evidence: list[str] | None = None,
) -> DiarizationResult:
    normalized_segments = _normalize_speaker_segments(speaker_segments or [])
    speaker_count = _estimate_normalized_speaker_count(normalized_segments)
    normalized_evidence = list(evidence or [])
    if normalized_segments:
        normalized_evidence.append(f"Normalized {len(normalized_segments)} diarization segment(s).")
    return DiarizationResult(
        provider=provider,
        model=model,
        input_source=input_source,
        input_path=str(input_path) if input_path is not None else None,
        output_type=output_type,
        speaker_segments=normalized_segments,
        estimated_speaker_count=speaker_count,
        confidence="medium" if speaker_count else "unknown",
        limitations=limitations or [],
        evidence=normalized_evidence,
    )


def _tag_transcription_input(
    result: TranscriptionResult,
    input_file: Path,
    input_source: str,
) -> TranscriptionResult:
    return result.model_copy(
        update={
            "input_source": input_source,
            "input_path": str(input_file),
            "evidence": _append_input_evidence(result.evidence, "Transcription", input_source, input_file),
        }
    )


def _tag_diarization_input(
    result: DiarizationResult,
    input_file: Path,
    input_source: str,
) -> DiarizationResult:
    return result.model_copy(
        update={
            "input_source": input_source,
            "input_path": str(input_file),
            "evidence": _append_input_evidence(result.evidence, "Diarization", input_source, input_file),
        }
    )


def _append_input_evidence(
    evidence: list[str],
    label: str,
    input_source: str,
    input_file: Path,
) -> list[str]:
    item = f"{label} input source: {input_source} ({input_file})."
    if item in evidence:
        return evidence
    return [*evidence, item]


def _normalize_transcript_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    normalized: list[TranscriptSegment] = []
    for segment in segments:
        start = max(0.0, float(segment.start_seconds))
        end = max(start, float(segment.end_seconds))
        text = segment.text.strip()
        speaker = segment.speaker.strip() if segment.speaker else None
        normalized.append(
            TranscriptSegment(
                start_seconds=start,
                end_seconds=end,
                text=text,
                speaker=speaker,
            )
        )
    return sorted(normalized, key=lambda item: (item.start_seconds, item.end_seconds, item.text))


def _normalize_speaker_segments(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    normalized: list[SpeakerSegment] = []
    for segment in segments:
        start = max(0.0, float(segment.start_seconds))
        end = max(start, float(segment.end_seconds))
        speaker = segment.speaker.strip() or "UNKNOWN"
        normalized.append(
            SpeakerSegment(
                speaker=speaker,
                start_seconds=start,
                end_seconds=end,
                text=segment.text.strip(),
            )
        )
    return sorted(normalized, key=lambda item: (item.start_seconds, item.end_seconds, item.speaker))


def _estimate_normalized_speaker_count(segments: list[SpeakerSegment]) -> int | None:
    speakers = {
        segment.speaker
        for segment in segments
        if segment.speaker.strip() and segment.speaker.strip().upper() != "UNKNOWN"
    }
    return len(speakers) if speakers else None


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

    def diarize(self, input_file: Path, *, audio_metadata: AudioMetadata | None = None) -> DiarizationResult:
        runtime_status = _module_status("onnxruntime")
        if not runtime_status["available"]:
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_onnx_model,
                input_path=input_file,
                output_type="none",
                limitations=["ONNX diarization skipped because onnxruntime is not installed."],
            )

        model_path, limitations = _resolve_onnx_model(
            self.settings,
            self.settings.pyannote_onnx_model,
            self.settings.pyannote_onnx_repo_id,
            kind="pyannote",
        )
        if not model_path:
            return _diarization_result(
                provider=self.provider_name,
                model=self.settings.pyannote_onnx_model,
                input_path=input_file,
                output_type="none",
                limitations=limitations,
            )

        return _diarization_result(
            provider=self.provider_name,
            model=str(model_path),
            input_path=input_file,
            output_type="onnx",
            limitations=[
                f"ONNX diarization inference is not implemented yet for model '{model_path}'. "
                "Pyannote-style ONNX diarization still needs segmentation, embedding, and clustering glue."
            ],
            evidence=[f"Resolved ONNX diarization model at {model_path}."],
        )


class SpeechAnalyzer:
    def __init__(self, transcriber: TranscriptionProvider, diarizer: DiarizationProvider):
        self.transcriber = transcriber
        self.diarizer = diarizer

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {**self.transcriber.readiness(), **self.diarizer.readiness()}

    def _call_diarizer(self, diarization_input: Path, *, audio_metadata: AudioMetadata | None = None) -> DiarizationResult:
        try:
            return self.diarizer.diarize(diarization_input, audio_metadata=audio_metadata)
        except TypeError:
            # Fallback for providers that don't support the new signature
            return self.diarizer.diarize(diarization_input)

    def analyze(
        self,
        transcription_input: Path,
        *,
        diarization_input: Path | None = None,
        transcription_source: str = "analysis_input",
        diarization_source: str = "analysis_input",
        audio_metadata: AudioMetadata | None = None,
    ) -> SpeechAnalysisResult:
        diarization_input = diarization_input or transcription_input
        transcription = _tag_transcription_input(
            self.transcriber.transcribe(transcription_input),
            transcription_input,
            transcription_source,
        )
        diarization = _tag_diarization_input(
            self._call_diarizer(diarization_input, audio_metadata=audio_metadata),
            diarization_input,
            diarization_source,
        )
        speaker_segments = _align_transcript_to_speakers(
            transcription.transcript_segments,
            diarization.speaker_segments,
        )
        if diarization.speaker_segments and not transcription.transcript_segments:
            speaker_segments = diarization.speaker_segments
        transcript_text = transcription.transcript_text or " ".join(
            segment.text.strip() for segment in transcription.transcript_segments if segment.text.strip()
        )
        return SpeechAnalysisResult(
            transcript_text=transcript_text,
            transcript_segments=transcription.transcript_segments,
            speaker_segments=speaker_segments,
            transcription=transcription,
            diarization=diarization,
            limitations=[*transcription.limitations, *diarization.limitations],
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

    aligned: list[SpeakerSegment] = []
    for diar_seg in diarization_segments:
        # Collect all transcripts that have their 'best overlap' with this diarization segment.
        # This prevents duplicating transcript text across multiple speaker segments
        # while ensuring each diarization interval is represented in the output.
        overlapping_texts = []
        for trans_seg in transcript_segments:
            if _best_overlap(trans_seg, diarization_segments) == diar_seg:
                overlapping_texts.append(trans_seg.text)

        text = " ".join(overlapping_texts).strip()
        if not text:
            text = "[Inaudible / No Transcript]"

        aligned.append(
            SpeakerSegment(
                speaker=diar_seg.speaker,
                start_seconds=diar_seg.start_seconds,
                end_seconds=diar_seg.end_seconds,
                text=text,
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
