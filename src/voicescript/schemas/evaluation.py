from __future__ import annotations

from pydantic import Field

from .base import JsonModel


class TranscriptReference(JsonModel):
    text: str


class SpeakerReferenceSegment(JsonModel):
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str = ""


class EvaluationReference(JsonModel):
    file_name: str
    transcript: TranscriptReference
    speaker_segments: list[SpeakerReferenceSegment] = Field(default_factory=list)


class EvaluationResult(JsonModel):
    file_name: str
    wer: float
    word_count: int
    speaker_count_expected: int | None
    speaker_count_predicted: int | None
    speaker_count_match: bool | None
    speaker_attribution_error_rate: float | None
    calibration_bucket: str
    notes: list[str] = Field(default_factory=list)
