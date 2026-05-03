from voicescript.analysis import build_report_from_measurements
from voicescript.evaluation import (
    compute_speaker_attribution_error,
    compute_wer,
    evaluate_report,
    evaluate_speaker_count,
)
from voicescript.schemas import (
    AudioMetadata,
    ChannelAnalysis,
    Estimate,
    EvaluationReference,
    SilenceSummary,
    SpeakerReferenceSegment,
    SpeakerSegment,
    TranscriptReference,
    VolumeStats,
)


def test_compute_wer_exact_match_is_zero():
    assert compute_wer("Q: State your name.", "q state your name") == 0.0


def test_compute_wer_counts_insertions_deletions_and_substitutions():
    # reference: "state your full name" = 4 words
    # hypothesis: "state the name" has one substitution, one deletion = 2 edits
    assert compute_wer("state your full name", "state the name") == 0.5


def test_evaluate_speaker_count_matches_distinct_reference_speakers():
    reference = [
        SpeakerReferenceSegment(speaker="ATTORNEY", start_seconds=0, end_seconds=1),
        SpeakerReferenceSegment(speaker="WITNESS", start_seconds=1, end_seconds=2),
    ]
    predicted = Estimate(value=2, confidence="medium", evidence=[])

    result = evaluate_speaker_count(reference, predicted)

    assert result == (2, 2, True)


def test_evaluate_speaker_count_reports_mismatch():
    reference = [
        SpeakerReferenceSegment(speaker="ATTORNEY", start_seconds=0, end_seconds=1),
        SpeakerReferenceSegment(speaker="WITNESS", start_seconds=1, end_seconds=2),
        SpeakerReferenceSegment(speaker="COURT_REPORTER", start_seconds=2, end_seconds=3),
    ]
    predicted = Estimate(value=1, confidence="high", evidence=[])

    result = evaluate_speaker_count(reference, predicted)

    assert result == (3, 1, False)


def test_speaker_attribution_error_is_zero_for_matching_overlaps():
    reference = [
        SpeakerReferenceSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=2.0),
        SpeakerReferenceSegment(speaker="WITNESS", start_seconds=2.0, end_seconds=4.0),
    ]
    predicted = [
        SpeakerSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=2.0),
        SpeakerSegment(speaker="WITNESS", start_seconds=2.0, end_seconds=4.0),
    ]

    assert compute_speaker_attribution_error(reference, predicted) == 0.0


def test_speaker_attribution_error_counts_wrong_overlap_duration():
    reference = [
        SpeakerReferenceSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=2.0),
        SpeakerReferenceSegment(speaker="WITNESS", start_seconds=2.0, end_seconds=4.0),
    ]
    predicted = [
        SpeakerSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=4.0),
    ]

    assert compute_speaker_attribution_error(reference, predicted) == 0.5


def test_evaluate_report_returns_combined_metrics():
    report = build_report_from_measurements(
        file_name="court.wav",
        sha256="abc",
        metadata=AudioMetadata(
            file_name="court.wav",
            duration_seconds=60.0,
            bitrate=128000,
            sample_rate=16000,
            channels=1,
            audio_streams=1,
            channel_layout="mono",
            codec_name="pcm_s16le",
            container_format="wav",
            raw={},
        ),
        silence=SilenceSummary(),
        volume=VolumeStats(
            avg_volume_db=-18.0,
            max_volume_db=-3.0,
            clipping_detected=False,
            low_volume_detected=False,
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
            SpeakerSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=2.0, text="State your name."),
            SpeakerSegment(speaker="WITNESS", start_seconds=2.0, end_seconds=4.0, text="Jane Doe."),
        ],
        transcript_text="State your name. Jane Doe.",
        provenance=[],
    )
    reference = EvaluationReference(
        file_name="court.wav",
        transcript=TranscriptReference(text="State your name. Jane Doe."),
        speaker_segments=[
            SpeakerReferenceSegment(speaker="ATTORNEY", start_seconds=0.0, end_seconds=2.0),
            SpeakerReferenceSegment(speaker="WITNESS", start_seconds=2.0, end_seconds=4.0),
        ],
    )

    result = evaluate_report(report, reference)

    assert result.file_name == "court.wav"
    assert result.wer == 0.0
    assert result.speaker_count_match is True
    assert result.speaker_attribution_error_rate == 0.0
    assert result.calibration_bucket == "medium"
