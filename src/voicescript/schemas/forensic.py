from __future__ import annotations

from typing import Literal

from pydantic import Field

from .audio import AudioQuality, ChannelAnalysis, SilenceSegment
from .base import Confidence, JsonModel, Severity
from .speech import DiarizationResult, SpeakerSegment, TranscriptionResult
from .tooling import CommandProvenance


class Estimate(JsonModel):
    value: int | None
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)


class ForensicFinding(JsonModel):
    finding: str
    severity: Severity
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)
    time_range_seconds: tuple[float, float] | None = None
    recommended_follow_up: str



class EvidenceArtifact(JsonModel):
    kind: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    recorded_at: str | None = None


class ToolVersion(JsonModel):
    tool: str
    available: bool
    path: str | None = None
    version: str | None = None


class EvidenceManifest(JsonModel):
    run_id: str
    generated_at: str
    source: EvidenceArtifact
    derived_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    tool_versions: list[ToolVersion] = Field(default_factory=list)


class ForensicProfile(JsonModel):
    classification: Literal["forensic_triage"] = "forensic_triage"
    claim_level: str = "Indicators only; not a certified forensic authentication opinion."
    generated_by: str = "voicescript deterministic rules"


class ForensicReport(JsonModel):
    file_name: str
    source_path: str | None = None
    sha256: str
    duration_seconds: float | None
    bitrate: int | None
    sample_rate: int | None
    channels: int | None
    audio_streams: int
    codec_name: str | None
    container_format: str | None
    forensic_profile: ForensicProfile = Field(default_factory=ForensicProfile)
    estimated_speaker_count: Estimate
    estimated_microphone_count: Estimate
    channel_analysis: ChannelAnalysis
    audio_quality: AudioQuality
    transcript_text: str = ""
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
    transcription: TranscriptionResult = Field(default_factory=TranscriptionResult)
    diarization: DiarizationResult = Field(default_factory=DiarizationResult)
    tamper_indicators: list[ForensicFinding] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence_scores: dict[str, Confidence] = Field(default_factory=dict)
    evidence_trail: list[str] = Field(default_factory=list)
    evidence_manifest: EvidenceManifest | None = None
    limitations: list[str] = Field(default_factory=list)
    summary_text: str
    provenance: list[CommandProvenance] = Field(default_factory=list)


class BatchAggregate(JsonModel):
    file_count: int
    completed_count: int
    failed_count: int
    weighted_silence_ratio: float
    issue_rollup: dict[str, int] = Field(default_factory=dict)
    worst_quality_files: list[str] = Field(default_factory=list)
