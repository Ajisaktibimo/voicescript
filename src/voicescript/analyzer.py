from __future__ import annotations

import hashlib
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

    def analyze_file(self, path: Path) -> ForensicReport:
        path = Path(path)
        metadata, silence, volume, channel_analysis, provenance = self._measure_core_audio(path)
        source_separation = self.demucs_separator.separate_vocals(
            path,
            self.settings.data_dir / "demucs" / path.stem,
        )
        speech_input = Path(source_separation.vocals_path) if source_separation.vocals_path else path
        speech = self.speech_analyzer.analyze(speech_input)

        return build_report_from_measurements(
            file_name=metadata.file_name or path.name,
            source_path=str(path),
            sha256=sha256_file(path),
            metadata=metadata,
            silence=silence,
            volume=volume,
            channel_analysis=channel_analysis,
            speaker_segments=speech.speaker_segments,
            transcript_text=speech.transcript_text,
            provenance=provenance,
            extra_limitations=speech.limitations,
            source_separation=source_separation,
        )

    def analyze_batch(self, paths: Iterable[Path]) -> dict[str, object]:
        reports: list[ForensicReport] = []
        failures: list[dict[str, str]] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                reports.append(self.analyze_file(path))
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
        speech = self.speech_analyzer.analyze(Path(path))
        report = build_report_from_measurements(
            file_name=Path(path).name,
            sha256=sha256_file(Path(path)),
            metadata=_unknown_metadata(Path(path).name),
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
            provenance=[],
            extra_limitations=speech.limitations,
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
        metadata, silence, volume, channel_analysis, provenance = self._measure_core_audio(path)
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
    ) -> tuple[AudioMetadata, SilenceSummary, VolumeStats, ChannelAnalysis, list[CommandProvenance]]:
        provenance: list[CommandProvenance] = []

        metadata_result = self.ffmpeg_tools.ffprobe_metadata(path)
        provenance.append(self.ffmpeg_tools.provenance("ffprobe", metadata_result))
        metadata = parse_ffprobe_metadata(metadata_result.stdout)

        silence_result = self.ffmpeg_tools.detect_silence(path)
        provenance.append(self.ffmpeg_tools.provenance("ffmpeg silencedetect", silence_result))
        silence = parse_silencedetect_output(
            silence_result.stderr,
            duration_seconds=metadata.duration_seconds,
        )

        volume_result = self.ffmpeg_tools.detect_volume(path)
        provenance.append(self.ffmpeg_tools.provenance("ffmpeg volumedetect", volume_result))
        volume = parse_volumedetect_output(
            volume_result.stderr,
            low_volume_threshold_db=self.settings.low_volume_threshold_db,
        )

        return metadata, silence, volume, estimate_channel_setup(metadata), provenance


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
