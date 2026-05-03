from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import Confidence, JsonModel


class SilenceSegment(JsonModel):
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class SilenceSummary(JsonModel):
    segments: list[SilenceSegment] = Field(default_factory=list)
    silence_ratio: float = 0.0


class VolumeStats(JsonModel):
    avg_volume_db: float | None
    max_volume_db: float | None
    clipping_detected: bool
    low_volume_detected: bool
    histogram_0db: int = 0


class AudioMetadata(JsonModel):
    file_name: str
    duration_seconds: float | None
    bitrate: int | None
    sample_rate: int | None
    channels: int | None
    audio_streams: int
    channel_layout: str | None
    codec_name: str | None
    container_format: str | None
    raw: dict[str, Any] = Field(default_factory=dict)


class ChannelAnalysis(JsonModel):
    measured_channels: int | None
    audio_streams: int
    channel_layout: str | None
    duplicated_channels_likely: bool
    channel_imbalance_db: float | None
    estimated_microphone_count: int | None
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)


class AudioQuality(JsonModel):
    silence_ratio: float
    silence_segments: list[SilenceSegment] = Field(default_factory=list)
    clipping_detected: bool
    avg_volume_db: float | None
    max_volume_db: float | None
    low_volume_detected: bool
