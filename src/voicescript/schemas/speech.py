from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Confidence
from .base import JsonModel


class SpeakerSegment(JsonModel):
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str = ""


class TranscriptSegment(JsonModel):
    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None


ProviderConfidence = Confidence | Literal["unknown"]


class TranscriptionResult(JsonModel):
    provider: str = "unknown"
    model: str | None = None
    input_source: str = "unknown"
    input_path: str | None = None
    output_type: str = "unknown"
    transcript_text: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    confidence: ProviderConfidence = "unknown"
    limitations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class DiarizationResult(JsonModel):
    provider: str = "unknown"
    model: str | None = None
    input_source: str = "unknown"
    input_path: str | None = None
    output_type: str = "unknown"
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
    estimated_speaker_count: int | None = None
    confidence: ProviderConfidence = "unknown"
    limitations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SpeechAnalysisResult(JsonModel):
    transcript_text: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
    transcription: TranscriptionResult = Field(default_factory=TranscriptionResult)
    diarization: DiarizationResult = Field(default_factory=DiarizationResult)
    limitations: list[str] = Field(default_factory=list)
    readiness: dict[str, dict[str, str | bool]] = Field(default_factory=dict)
