from __future__ import annotations

import hashlib
import inspect
import logging
from pathlib import Path
from typing import Iterable

from .analysis import build_report_from_measurements
from .config import Settings
from .ffmpeg_tools import FFmpegTools
from .providers.separation import SourceSeparator, create_source_separator
from .providers.speech import SpeechAnalyzer, create_speech_analyzer
from .schemas import (
    AudioMetadata,
    BatchAggregate,
    ChannelAnalysis,
    CommandProvenance,
    ForensicReport,
    SilenceSummary,
    SourceSeparationResult,
    VolumeStats,
    to_jsonable,
)
from .parsers import parse_ffprobe_metadata, parse_silencedetect_output, parse_volumedetect_output


logger = logging.getLogger("uvicorn.error")


class ForensicAnalyzer:
    def __init__(
        self,
        settings: Settings | None = None,
        ffmpeg_tools: FFmpegTools | None = None,
        speech_analyzer: SpeechAnalyzer | None = None,
        demucs_separator: SourceSeparator | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.ffmpeg_tools = ffmpeg_tools or FFmpegTools(self.settings)
        self.speech_analyzer = speech_analyzer or create_speech_analyzer(self.settings)
        self.demucs_separator = demucs_separator or create_source_separator(self.settings)

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            **self.ffmpeg_tools.readiness(),
            **self.speech_analyzer.readiness(),
            **self.demucs_separator.readiness(),
        }

    def analyze_file(self, path: Path, *, run_id: str | None = None) -> ForensicReport:
        path = Path(path)
        run_id = run_id or f"direct:{path.name}"
        self._log_stage(run_id, "input", "received", f"file={path.name} path={path} bytes={_file_size(path)}")
        metadata, silence, volume, channel_analysis, provenance = self._measure_core_audio(path, run_id=run_id)

        self._log_stage(
            run_id,
            "source_separation",
            "start",
            f"provider={getattr(self.demucs_separator, 'provider_name', self.demucs_separator.__class__.__name__)}",
        )
        source_separation = self.demucs_separator.separate_vocals(
            path,
            self.settings.data_dir / "demucs" / path.stem,
        )
        self._log_stage(
            run_id,
            "source_separation",
            "done",
            f"available={source_separation.available} enabled={source_separation.enabled} "
            f"vocals_path={source_separation.vocals_path or ''} limitations={len(source_separation.limitations)}",
        )

        transcription_input = Path(source_separation.vocals_path) if source_separation.vocals_path else path
        transcription_source = "demucs_vocals" if source_separation.vocals_path else "original_audio"
        diarization_input, diarization_provenance = self._prepare_diarization_input(
            path,
            run_id=run_id,
            channels=metadata.channels,
        )
        provenance.append(diarization_provenance)
        self._log_stage(
            run_id,
            "speech",
            "start",
            f"transcription_input={transcription_input} transcription_source={transcription_source} "
            f"diarization_input={diarization_input} diarization_source=normalized_original_audio",
        )
        speech = self.speech_analyzer.analyze(
            transcription_input,
            diarization_input=diarization_input,
            transcription_source=transcription_source,
            diarization_source="normalized_original_audio",
            diarization_hints=self._diarization_hints_from_settings(),
        )
        self._log_stage(
            run_id,
            "speech",
            "done",
            f"transcript_chars={len(speech.transcript_text)} speaker_segments={len(speech.speaker_segments)} "
            f"limitations={len(speech.limitations)}",
        )

        self._log_stage(run_id, "hash", "start", f"file={path.name}")
        sha256 = sha256_file(path)
        self._log_stage(run_id, "hash", "done", f"sha256={sha256[:12]}...")

        self._log_stage(run_id, "report", "start", "building forensic report")
        report = build_report_from_measurements(
            file_name=metadata.file_name or path.name,
            source_path=str(path),
            sha256=sha256,
            metadata=metadata,
            silence=silence,
            volume=volume,
            channel_analysis=channel_analysis,
            speaker_segments=speech.speaker_segments,
            transcript_text=speech.transcript_text,
            provenance=provenance,
            extra_limitations=speech.limitations,
            source_separation=source_separation,
            transcription=speech.transcription,
            diarization=speech.diarization,
        )
        self._log_stage(
            run_id,
            "report",
            "done",
            f"issues={len(report.issues)} indicators={len(report.tamper_indicators)} limitations={len(report.limitations)}",
        )
        return report

    def analyze_batch(self, paths: Iterable[Path]) -> dict[str, object]:
        reports: list[ForensicReport] = []
        failures: list[dict[str, str]] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                reports.append(_analyze_file_with_run_id(self, path, run_id=f"batch:{path.name}"))
            except Exception as exc:
                failures.append({"file_name": path.name, "path": str(path), "error": str(exc)})
        return {
            "aggregate": to_jsonable(aggregate_reports(reports, failed_count=len(failures))),
            "reports": to_jsonable(reports),
            "failures": failures,
        }

    def get_audio_metadata(self, path: Path) -> dict[str, object]:
        result = self.ffmpeg_tools.ffprobe_metadata(Path(path))
        return to_jsonable(parse_ffprobe_metadata(result.stdout))

    def analyze_audio_quality(self, path: Path) -> dict[str, object]:
        metadata_result = self.ffmpeg_tools.ffprobe_metadata(Path(path))
        metadata = parse_ffprobe_metadata(metadata_result.stdout)
        silence_result = self.ffmpeg_tools.detect_silence(Path(path))
        volume_result = self.ffmpeg_tools.detect_volume(Path(path))
        silence = parse_silencedetect_output(silence_result.stderr, metadata.duration_seconds)
        volume = parse_volumedetect_output(volume_result.stderr, self.settings.low_volume_threshold_db)
        return {
            "silence_ratio": silence.silence_ratio,
            "silence_segments": to_jsonable(silence.segments),
            "clipping_detected": volume.clipping_detected,
            "avg_volume_db": volume.avg_volume_db,
            "max_volume_db": volume.max_volume_db,
            "low_volume_detected": volume.low_volume_detected,
        }

    def analyze_channels(self, path: Path) -> dict[str, object]:
        result = self.ffmpeg_tools.ffprobe_metadata(Path(path))
        metadata = parse_ffprobe_metadata(result.stdout)
        return to_jsonable(estimate_channel_setup(metadata))

    def estimate_speakers(self, path: Path) -> dict[str, object]:
        path = Path(path)
        run_id = f"speakers:{path.name}"
        diarization_input, diarization_provenance = self._prepare_diarization_input(path, run_id=run_id)
        speech = self.speech_analyzer.analyze(
            path,
            diarization_input=diarization_input,
            transcription_source="original_audio",
            diarization_source="normalized_original_audio",
        )
        report = build_report_from_measurements(
            file_name=path.name,
            sha256=sha256_file(path),
            metadata=_unknown_metadata(path.name),
            silence=SilenceSummary(),
            volume=VolumeStats(
                avg_volume_db=None,
                max_volume_db=None,
                clipping_detected=False,
                low_volume_detected=False,
            ),
            channel_analysis=ChannelAnalysis(
                measured_channels=None,
                audio_streams=0,
                channel_layout=None,
                duplicated_channels_likely=False,
                channel_imbalance_db=None,
                estimated_microphone_count=None,
                confidence="low",
                evidence=["Microphone setup was not analyzed for this speaker-only estimate."],
            ),
            speaker_segments=speech.speaker_segments,
            transcript_text=speech.transcript_text,
            provenance=[diarization_provenance],
            extra_limitations=speech.limitations,
            transcription=speech.transcription,
            diarization=speech.diarization,
        )
        return to_jsonable(report.estimated_speaker_count)

    def estimate_microphone_setup(self, path: Path) -> dict[str, object]:
        channel = self.analyze_channels(path)
        return {
            "value": channel["estimated_microphone_count"],
            "confidence": channel["confidence"],
            "evidence": channel["evidence"],
        }

    def detect_forensic_indicators(self, path: Path) -> list[dict[str, object]]:
        path = Path(path)
        metadata, silence, volume, channel_analysis, provenance = self._measure_core_audio(
            path,
            run_id=f"indicators:{path.name}",
        )
        report = build_report_from_measurements(
            file_name=metadata.file_name or path.name,
            source_path=str(path),
            sha256=sha256_file(path),
            metadata=metadata,
            silence=silence,
            volume=volume,
            channel_analysis=channel_analysis,
            speaker_segments=[],
            transcript_text="",
            provenance=provenance,
            extra_limitations=["Speech analysis was not run for indicator-only analysis."],
            source_separation=SourceSeparationResult(
                available=False,
                enabled=False,
                limitations=["Source separation was not run for indicator-only analysis."],
            ),
        )
        return to_jsonable(report.tamper_indicators)

    def generate_spectrogram(self, input_file: Path, output_image: Path) -> dict[str, object]:
        result = self.ffmpeg_tools.generate_spectrogram(Path(input_file), Path(output_image))
        return to_jsonable(self.ffmpeg_tools.provenance("ffmpeg showspectrumpic", result))

    def isolate_vocals(self, input_file: Path, output_dir: Path | None = None) -> dict[str, object]:
        target_dir = output_dir or (self.settings.data_dir / "demucs" / Path(input_file).stem)
        return to_jsonable(self.demucs_separator.separate_vocals(Path(input_file), Path(target_dir)))

    def _measure_core_audio(
        self,
        path: Path,
        *,
        run_id: str,
    ) -> tuple[AudioMetadata, SilenceSummary, VolumeStats, ChannelAnalysis, list[CommandProvenance]]:
        provenance: list[CommandProvenance] = []

        self._log_stage(run_id, "ffprobe", "start", f"file={path.name}")
        metadata_result = self.ffmpeg_tools.ffprobe_metadata(path)
        provenance.append(self.ffmpeg_tools.provenance("ffprobe", metadata_result))
        metadata = parse_ffprobe_metadata(metadata_result.stdout)
        self._log_stage(
            run_id,
            "ffprobe",
            "done",
            f"duration={metadata.duration_seconds} sample_rate={metadata.sample_rate} "
            f"channels={metadata.channels} streams={metadata.audio_streams}",
        )

        self._log_stage(run_id, "silence", "start", f"threshold={self.settings.silence_noise_db}")
        silence_result = self.ffmpeg_tools.detect_silence(path)
        provenance.append(self.ffmpeg_tools.provenance("ffmpeg silencedetect", silence_result))
        silence = parse_silencedetect_output(
            silence_result.stderr,
            duration_seconds=metadata.duration_seconds,
        )
        self._log_stage(
            run_id,
            "silence",
            "done",
            f"segments={len(silence.segments)} ratio={silence.silence_ratio}",
        )

        self._log_stage(run_id, "volume", "start", "running volumedetect")
        volume_result = self.ffmpeg_tools.detect_volume(path)
        provenance.append(self.ffmpeg_tools.provenance("ffmpeg volumedetect", volume_result))
        volume = parse_volumedetect_output(
            volume_result.stderr,
            low_volume_threshold_db=self.settings.low_volume_threshold_db,
        )
        self._log_stage(
            run_id,
            "volume",
            "done",
            f"avg_db={volume.avg_volume_db} max_db={volume.max_volume_db} clipping={volume.clipping_detected}",
        )

        self._log_stage(run_id, "channels", "start", "estimating channel and microphone indicators")
        channel_analysis = estimate_channel_setup(metadata)
        self._log_stage(
            run_id,
            "channels",
            "done",
            f"measured={channel_analysis.measured_channels} mic_estimate={channel_analysis.estimated_microphone_count} "
            f"confidence={channel_analysis.confidence}",
        )

        return metadata, silence, volume, channel_analysis, provenance

    def _prepare_diarization_input(
        self,
        path: Path,
        *,
        run_id: str,
        channels: int | None = None,
    ) -> tuple[Path, CommandProvenance]:
        output_file = self.settings.data_dir / "diarization" / _safe_artifact_name(run_id, path) / "mono16k.wav"
        self._log_stage(
            run_id,
            "diarization_normalize",
            "start",
            f"input={path} output={output_file} channels={channels}",
        )
        result = self.ffmpeg_tools.normalize_for_speech(path, output_file, channels=channels)
        self._log_stage(
            run_id,
            "diarization_normalize",
            "done",
            f"output={output_file}",
        )
        return output_file, self.ffmpeg_tools.provenance("ffmpeg normalize diarization", result)

    def _log_stage(self, run_id: str, stage: str, status: str, detail: str) -> None:
        logger.info("pipeline run_id=%s stage=%s status=%s %s", run_id, stage, status, detail)

    def _diarization_hints_from_settings(self) -> dict[str, int]:
        hints: dict[str, int] = {}
        if getattr(self.settings, "pyannote_min_speakers", None) is not None:
            hints["min_speakers"] = int(self.settings.pyannote_min_speakers)
        if getattr(self.settings, "pyannote_max_speakers", None) is not None:
            hints["max_speakers"] = int(self.settings.pyannote_max_speakers)
        return hints


def estimate_channel_setup(metadata: AudioMetadata) -> ChannelAnalysis:
    evidence: list[str] = []
    estimated_microphone_count: int | None
    confidence = "low"
    duplicated = False
    imbalance = None

    if metadata.channels is None:
        estimated_microphone_count = None
        evidence.append("ffprobe did not report channel count.")
    elif metadata.channels == 1:
        estimated_microphone_count = 1
        confidence = "medium"
        evidence.append("Mono stream has one measured channel; this supports, but does not prove, one microphone source.")
    elif metadata.channels == 2 and (metadata.channel_layout or "").lower() == "stereo":
        estimated_microphone_count = None
        confidence = "low"
        evidence.append("Stereo layout has two measured channels; microphone count cannot be confirmed from metadata alone.")
    else:
        estimated_microphone_count = None
        confidence = "low"
        evidence.append(
            f"{metadata.channels} measured channel(s); microphone count cannot be confirmed from metadata alone."
        )

    if metadata.audio_streams > 1:
        evidence.append(f"Container has {metadata.audio_streams} audio streams; each stream should be reviewed separately.")

    encoder = str(metadata.raw.get("format", {}).get("tags", {}).get("encoder", "")).lower()
    if "lavf" in encoder or "ffmpeg" in encoder:
        evidence.append("Container metadata references ffmpeg/libav; this may indicate an exported or transcoded file.")

    return ChannelAnalysis(
        measured_channels=metadata.channels,
        audio_streams=metadata.audio_streams,
        channel_layout=metadata.channel_layout,
        duplicated_channels_likely=duplicated,
        channel_imbalance_db=imbalance,
        estimated_microphone_count=estimated_microphone_count,
        confidence=confidence,
        evidence=evidence,
    )


def aggregate_reports(reports: list[ForensicReport], failed_count: int) -> BatchAggregate:
    completed_count = len(reports)
    total_duration = sum(report.duration_seconds or 0.0 for report in reports)
    if total_duration:
        weighted_silence = sum(
            (report.duration_seconds or 0.0) * report.audio_quality.silence_ratio for report in reports
        ) / total_duration
    else:
        weighted_silence = 0.0

    issue_rollup: dict[str, int] = {}
    for report in reports:
        for issue in report.issues:
            issue_rollup[issue] = issue_rollup.get(issue, 0) + 1

    worst_quality_files = [
        report.file_name
        for report in sorted(
            reports,
            key=lambda item: (len(item.tamper_indicators), item.audio_quality.silence_ratio),
            reverse=True,
        )[:5]
        if report.tamper_indicators
    ]

    return BatchAggregate(
        file_count=completed_count + failed_count,
        completed_count=completed_count,
        failed_count=failed_count,
        weighted_silence_ratio=round(weighted_silence, 6),
        issue_rollup=issue_rollup,
        worst_quality_files=worst_quality_files,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _safe_artifact_name(run_id: str, path: Path) -> str:
    raw_name = f"{run_id}_{Path(path).stem}"
    safe_name = "".join(character if character.isalnum() else "_" for character in raw_name)
    return safe_name.strip("_") or "audio"


def _analyze_file_with_run_id(analyzer: ForensicAnalyzer, path: Path, *, run_id: str) -> ForensicReport:
    try:
        parameters = inspect.signature(analyzer.analyze_file).parameters
    except (TypeError, ValueError):
        return analyzer.analyze_file(path)
    if "run_id" in parameters:
        return analyzer.analyze_file(path, run_id=run_id)
    return analyzer.analyze_file(path)


def _unknown_metadata(file_name: str) -> AudioMetadata:
    return AudioMetadata(
        file_name=file_name,
        duration_seconds=None,
        bitrate=None,
        sample_rate=None,
        channels=None,
        audio_streams=0,
        channel_layout=None,
        codec_name=None,
        container_format=None,
        raw={},
    )
