from voicescript import models
from voicescript.schemas import (
    AudioMetadata,
    EvidenceArtifact,
    EvidenceManifest,
    EvaluationReference,
    EvaluationResult,
    ForensicReport,
    SpeakerReferenceSegment,
    TranscriptReference,
    ToolVersion,
    to_jsonable,
)
from voicescript.schemas.audio import SilenceSummary


def test_schemas_package_is_public_source_for_pydantic_models():
    assert AudioMetadata is models.AudioMetadata
    assert EvidenceArtifact is models.EvidenceArtifact
    assert EvidenceManifest is models.EvidenceManifest
    assert ForensicReport is models.ForensicReport
    assert ToolVersion is models.ToolVersion
    assert to_jsonable(SilenceSummary()) == {"segments": [], "silence_ratio": 0.0}


def test_evidence_manifest_schema_round_trips_artifacts_and_tool_versions():
    manifest = EvidenceManifest(
        run_id="run-1",
        generated_at="2026-05-02T12:00:00+00:00",
        source=EvidenceArtifact(
            kind="source",
            path="uploads/run-1/court.wav",
            sha256="abc123",
            size_bytes=42,
            recorded_at="2026-05-02T11:59:00+00:00",
        ),
        derived_artifacts=[
            EvidenceArtifact(
                kind="demucs_vocals",
                path="demucs/run-1/vocals.wav",
                sha256="def456",
                size_bytes=21,
            )
        ],
        tool_versions=[
            ToolVersion(tool="ffmpeg", available=True, path="ffmpeg", version="ffmpeg version 6.1")
        ],
    )

    payload = to_jsonable(manifest)

    assert payload["source"]["kind"] == "source"
    assert payload["derived_artifacts"][0]["kind"] == "demucs_vocals"
    assert payload["tool_versions"][0]["tool"] == "ffmpeg"


def test_evaluation_schemas_are_public_exports():
    reference = EvaluationReference(
        file_name="court.wav",
        transcript=TranscriptReference(text="Q: State your name. A: Jane Doe."),
        speaker_segments=[
            SpeakerReferenceSegment(
                speaker="ATTORNEY",
                start_seconds=0.0,
                end_seconds=1.5,
                text="State your name.",
            ),
            SpeakerReferenceSegment(
                speaker="WITNESS",
                start_seconds=1.5,
                end_seconds=3.0,
                text="Jane Doe.",
            ),
        ],
    )
    result = EvaluationResult(
        file_name="court.wav",
        wer=0.0,
        word_count=7,
        speaker_count_expected=2,
        speaker_count_predicted=2,
        speaker_count_match=True,
        speaker_attribution_error_rate=0.0,
        calibration_bucket="medium",
        notes=[],
    )

    assert reference.file_name == "court.wav"
    assert result.speaker_count_match is True
