from __future__ import annotations

from pydantic import Field

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


class SpeechAnalysisResult(JsonModel):
    transcript_text: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    readiness: dict[str, dict[str, str | bool]] = Field(default_factory=dict)
