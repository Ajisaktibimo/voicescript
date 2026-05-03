from voicescript.analysis import build_report_from_measurements
from voicescript.models import (
    AudioMetadata,
    ChannelAnalysis,
    CommandProvenance,
    SilenceSegment,
    SilenceSummary,
    SpeakerSegment,
    VolumeStats,
)
from voicescript.schemas import DiarizationResult, TranscriptionResult, TranscriptSegment


def test_forensic_report_keeps_estimates_confidence_and_evidence():
    report = build_report_from_measurements(
        file_name="deposition_001.wav",
        sha256="abc123",
        metadata=AudioMetadata(
            file_name="deposition_001.wav",
            duration_seconds=100.0,
            bitrate=256000,
            sample_rate=48000,
            channels=2,
            audio_streams=1,
            channel_layout="stereo",
            codec_name="pcm_s16le",
            container_format="wav",
            raw={},
        ),
        silence=SilenceSummary(
            segments=[SilenceSegment(start_seconds=10, end_seconds=30, duration_seconds=20)],
            silence_ratio=0.2,
        ),
        volume=VolumeStats(
            avg_volume_db=-34.0,
            max_volume_db=-0.1,
            clipping_detected=True,
            low_volume_detected=True,
            histogram_0db=3,
        ),
        channel_analysis=ChannelAnalysis(
            measured_channels=2,
            audio_streams=1,
            channel_layout="stereo",
            duplicated_channels_likely=True,
            channel_imbalance_db=8.0,
            estimated_microphone_count=1,
            confidence="medium",
            evidence=["Stereo channels are highly correlated", "Channel energy differs by 8.0 dB"],
        ),
        speaker_segments=[
            SpeakerSegment(speaker="SPEAKER_00", start_seconds=0, end_seconds=6, text="Good morning."),
            SpeakerSegment(speaker="SPEAKER_01", start_seconds=7, end_seconds=12, text="Good morning."),
        ],
        transcript_text="Good morning. Good morning.",
        provenance=[
            CommandProvenance(tool="ffprobe", command=["ffprobe", "-show_streams"], exit_code=0)
        ],
    )

    assert report.channels == 2
    assert report.audio_streams == 1
    assert report.estimated_speaker_count.value == 2
    assert report.estimated_speaker_count.confidence == "medium"
    assert report.estimated_microphone_count.value == 1
    assert report.estimated_microphone_count.confidence == "medium"
    assert report.audio_quality.silence_ratio == 0.2

    findings = {finding.finding: finding for finding in report.tamper_indicators}
    assert "Extended silence or missing-audio span detected" in findings
    assert "Potential clipping or hard limiting detected" in findings
    assert "Unusually low average volume detected" in findings
    assert "Possible duplicated stereo channels" in findings
    assert all(f.confidence in {"low", "medium", "high"} for f in report.tamper_indicators)
    assert all(f.evidence for f in report.tamper_indicators)
    assert report.limitations
    assert report.limitations
    assert "triage" in report.summary_text.lower()


def test_report_does_not_infer_speakers_when_diarization_unavailable():
    report = build_report_from_measurements(
        file_name="mono.wav",
        sha256="def456",
        metadata=AudioMetadata(
            file_name="mono.wav",
            duration_seconds=10.0,
            bitrate=None,
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
            max_volume_db=-4.0,
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

    assert report.estimated_speaker_count.value is None
    assert report.estimated_speaker_count.confidence == "low"
    assert "Diarization unavailable" in report.estimated_speaker_count.evidence[0]


def test_report_fusion_downgrades_one_speaker_when_transcript_implies_more():
    diar = DiarizationResult(
        provider="local-pyannote",
        output_type="fixture",
        speaker_segments=[
            SpeakerSegment(speaker="SPEAKER_00", start_seconds=0, end_seconds=2, text="")
        ],
        estimated_speaker_count=1,
        confidence="medium",
    )
    trans = TranscriptionResult(
        provider="local-faster-whisper",
        output_type="fixture",
        transcript_text=(
            "Q: Officer, what did the witness say? "
            "A: She replied that the suspect was uncooperative."
        ),
        transcript_segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=10.0,
                text=(
                    "Q: Officer, what did the witness say? "
                    "A: She replied that the suspect was uncooperative."
                ),
            )
        ],
    )

    report = build_report_from_measurements(
        file_name="court.wav",
        sha256="cafebabe",
        metadata=AudioMetadata(
            file_name="court.wav",
            duration_seconds=120.0,
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
            avg_volume_db=-20.0,
            max_volume_db=-3.0,
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
            evidence=[],
        ),
        speaker_segments=[
            SpeakerSegment(speaker="SPEAKER_00", start_seconds=0, end_seconds=2, text="")
        ],
        transcript_text=trans.transcript_text,
        provenance=[],
        transcription=trans,
        diarization=diar,
    )

    assert report.estimated_speaker_count.value >= 2
    assert report.estimated_speaker_count.confidence == "low"
    assert any(
        finding.finding == "Diarization-transcript speaker count mismatch"
        for finding in report.tamper_indicators
    )
    # raw diarization unchanged
    assert report.diarization.estimated_speaker_count == 1
    assert report.diarization.confidence == "medium"


def test_unknown_speaker_label_not_counted():
    report = build_report_from_measurements(
        file_name="mixed.wav",
        sha256="abc123",
        metadata=AudioMetadata(
            file_name="mixed.wav",
            duration_seconds=10.0,
            bitrate=None,
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
            max_volume_db=-4.0,
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
        speaker_segments=[
            SpeakerSegment(speaker="UNKNOWN", start_seconds=0, end_seconds=2, text="Unaligned."),
            SpeakerSegment(speaker="SPEAKER_00", start_seconds=2, end_seconds=4, text="Aligned."),
        ],
        transcript_text="Unaligned. Aligned.",
        provenance=[],
    )

    assert report.estimated_speaker_count.value == 1
    assert "UNKNOWN" not in report.estimated_speaker_count.evidence[0]

    rebuilt = build_report_from_measurements(
        file_name="unknown.wav",
        sha256="def456",
        metadata=AudioMetadata(
            file_name="unknown.wav",
            duration_seconds=10.0,
            bitrate=None,
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
            max_volume_db=-4.0,
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
        speaker_segments=[
            SpeakerSegment(speaker="UNKNOWN", start_seconds=0, end_seconds=2, text="Unaligned."),
        ],
        transcript_text="Unaligned.",
        provenance=[],
    )
    assert rebuilt.estimated_speaker_count.value is None
