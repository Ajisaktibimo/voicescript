import json
import logging
from pathlib import Path

import pytest

from voicescript.analyzer import ForensicAnalyzer, aggregate_reports, estimate_channel_setup
from voicescript.config import Settings
from voicescript.schemas import (
    AudioMetadata,
    ChannelAnalysis,
    CommandProvenance,
    ForensicReport,
    SilenceSummary,
    SourceSeparationResult,
    SpeechAnalysisResult,
    SpeakerSegment,
    ToolResult,
    DiarizationResult,
    TranscriptionResult,
    VolumeStats,
)
from voicescript.analysis import build_report_from_measurements


def test_detect_forensic_indicators_uses_rules_only(runtime_dir):
    audio = runtime_dir / "court.wav"
    audio.write_bytes(b"audio")
    analyzer = ForensicAnalyzer(
        settings=_settings(runtime_dir),
        ffmpeg_tools=FakeFFmpegTools(),
        speech_analyzer=RaisingSpeechAnalyzer(),
        demucs_separator=RaisingSourceSeparator(),
    )

    indicators = analyzer.detect_forensic_indicators(audio)

    assert [item["finding"] for item in indicators] == ["Extended silence or missing-audio span detected"]


def test_demucs_vocals_path_is_used_for_speech_when_available(runtime_dir):
    audio = runtime_dir / "court.wav"
    audio.write_bytes(b"audio")
    vocals = runtime_dir / "vocals.wav"
    vocals.write_bytes(b"vocals")
    speech = RecordingSpeechAnalyzer()
    analyzer = ForensicAnalyzer(
        settings=_settings(runtime_dir),
        ffmpeg_tools=FakeFFmpegTools(),
        speech_analyzer=speech,
        demucs_separator=FixedSourceSeparator(
            SourceSeparationResult(available=True, enabled=True, vocals_path=str(vocals))
        ),
    )

    report = analyzer.analyze_file(audio)

    assert speech.paths == [vocals]
    assert report.source_separation.vocals_path == str(vocals)
    assert report.transcription.provider == "test-transcriber"
    assert report.diarization.provider == "test-diarizer"


def test_demucs_unavailable_falls_back_to_original_for_speech(runtime_dir):
    audio = runtime_dir / "court.wav"
    audio.write_bytes(b"audio")
    speech = RecordingSpeechAnalyzer()
    analyzer = ForensicAnalyzer(
        settings=_settings(runtime_dir),
        ffmpeg_tools=FakeFFmpegTools(),
        speech_analyzer=speech,
        demucs_separator=FixedSourceSeparator(
            SourceSeparationResult(
                available=False,
                enabled=True,
                limitations=["Demucs is not installed."],
            )
        ),
    )

    report = analyzer.analyze_file(audio)

    assert speech.paths == [audio]
    assert "Demucs is not installed." in report.limitations


def test_stereo_channel_count_does_not_claim_two_microphones():
    channel = estimate_channel_setup(
        AudioMetadata(
            file_name="stereo.wav",
            duration_seconds=10.0,
            bitrate=None,
            sample_rate=48000,
            channels=2,
            audio_streams=1,
            channel_layout="stereo",
            codec_name="pcm_s16le",
            container_format="wav",
            raw={},
        )
    )

    assert channel.estimated_microphone_count is None
    assert channel.confidence == "low"
    assert any("cannot be confirmed" in item for item in channel.evidence)


def test_analyze_batch_collects_per_file_failures(runtime_dir):
    bad = runtime_dir / "bad.wav"
    good = runtime_dir / "good.wav"
    analyzer = PartiallyFailingAnalyzer()

    result = analyzer.analyze_batch([bad, good])

    assert result["aggregate"]["file_count"] == 2
    assert result["aggregate"]["completed_count"] == 1
    assert result["aggregate"]["failed_count"] == 1
    assert [report["file_name"] for report in result["reports"]] == ["good.wav"]
    assert result["failures"] == [
        {"file_name": "bad.wav", "path": str(bad), "error": "decode failed"}
    ]


def test_analyze_file_logs_pipeline_stages(runtime_dir, caplog):
    audio = runtime_dir / "court.wav"
    audio.write_bytes(b"audio")
    analyzer = ForensicAnalyzer(
        settings=_settings(runtime_dir),
        ffmpeg_tools=FakeFFmpegTools(),
        speech_analyzer=RecordingSpeechAnalyzer(),
        demucs_separator=FixedSourceSeparator(
            SourceSeparationResult(available=False, enabled=True, limitations=["Demucs is not installed."])
        ),
    )

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        analyzer.analyze_file(audio, run_id="analysis-1")

    messages = [record.getMessage() for record in caplog.records]
    expected_stages = [
        "stage=input",
        "stage=ffprobe",
        "stage=silence",
        "stage=volume",
        "stage=channels",
        "stage=source_separation",
        "stage=speech",
        "stage=hash",
        "stage=report",
    ]
    for stage in expected_stages:
        assert any("pipeline run_id=analysis-1" in message and stage in message for message in messages)


class FakeFFmpegTools:
    def ffprobe_metadata(self, path: Path) -> ToolResult:
        return ToolResult(
            stdout=json.dumps(
                {
                    "format": {
                        "filename": str(path),
                        "duration": "30.0",
                        "bit_rate": "128000",
                        "format_name": "wav",
                    },
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "pcm_s16le",
                            "sample_rate": "16000",
                            "channels": 1,
                            "channel_layout": "mono",
                        }
                    ],
                }
            ),
            stderr="",
            exit_code=0,
            command=["ffprobe", str(path)],
        )

    def detect_silence(self, path: Path) -> ToolResult:
        return ToolResult(
            stdout="",
            stderr=(
                "[silencedetect] silence_start: 5\n"
                "[silencedetect] silence_end: 20 | silence_duration: 15\n"
            ),
            exit_code=0,
            command=["ffmpeg", "silencedetect", str(path)],
        )

    def detect_volume(self, path: Path) -> ToolResult:
        return ToolResult(
            stdout="",
            stderr="[volumedetect] mean_volume: -18.0 dB\n[volumedetect] max_volume: -6.0 dB\n",
            exit_code=0,
            command=["ffmpeg", "volumedetect", str(path)],
        )

    def provenance(self, tool: str, result: ToolResult) -> CommandProvenance:
        return CommandProvenance(tool=tool, command=result.command, exit_code=result.exit_code)


class RaisingSpeechAnalyzer:
    def readiness(self):
        return {}

    def analyze(self, path: Path):
        raise AssertionError("speech analysis should not be called")


class RaisingSourceSeparator:
    def readiness(self):
        return {}

    def separate_vocals(self, input_file: Path, output_dir: Path):
        raise AssertionError("source separation should not be called")


class RecordingSpeechAnalyzer:
    def __init__(self):
        self.paths: list[Path] = []

    def readiness(self):
        return {}

    def analyze(self, path: Path) -> SpeechAnalysisResult:
        self.paths.append(Path(path))
        return SpeechAnalysisResult(
            transcript_text="hello",
            transcription=TranscriptionResult(
                provider="test-transcriber",
                output_type="fixture",
                transcript_text="hello",
            ),
            diarization=DiarizationResult(
                provider="test-diarizer",
                output_type="fixture",
                speaker_segments=[
                    SpeakerSegment(speaker="SPEAKER_00", start_seconds=0, end_seconds=1, text="")
                ],
                estimated_speaker_count=1,
                confidence="medium",
            ),
            speaker_segments=[
                SpeakerSegment(speaker="SPEAKER_00", start_seconds=0, end_seconds=1, text="hello")
            ],
        )


class FixedSourceSeparator:
    provider_name = "fixed"

    def __init__(self, result: SourceSeparationResult):
        self.result = result

    def readiness(self):
        return {}

    def separate_vocals(self, input_file: Path, output_dir: Path) -> SourceSeparationResult:
        return self.result


class PartiallyFailingAnalyzer(ForensicAnalyzer):
    def __init__(self):
        pass

    def analyze_file(self, path: Path) -> ForensicReport:
        if path.name == "bad.wav":
            raise RuntimeError("decode failed")
        return _minimal_report(path.name)


def _settings(runtime_dir: Path) -> Settings:
    return Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        transcription_provider="disabled",
        diarization_provider="disabled",
        source_separation_provider="disabled",
    )


def _minimal_report(file_name: str) -> ForensicReport:
    return build_report_from_measurements(
        file_name=file_name,
        sha256="feedface",
        metadata=AudioMetadata(
            file_name=file_name,
            duration_seconds=2.0,
            bitrate=128000,
            sample_rate=16000,
            channels=1,
            audio_streams=1,
            channel_layout="mono",
            codec_name="pcm_s16le",
            container_format="wav",
            raw={},
        ),
        silence=SilenceSummary(segments=[], silence_ratio=0.0),
        volume=VolumeStats(
            avg_volume_db=-18.0,
            max_volume_db=-5.0,
            clipping_detected=False,
            low_volume_detected=False,
            histogram_0db=0,
        ),
        channel_analysis=ChannelAnalysis(
            measured_channels=1,
            audio_streams=1,
            channel_layout="mono",
            duplicated_channels_likely=False,
            channel_imbalance_db=None,
            estimated_microphone_count=1,
            confidence="medium",
            evidence=["Mono stream has one measured channel"],
        ),
        speaker_segments=[],
        transcript_text="",
        provenance=[],
    )
