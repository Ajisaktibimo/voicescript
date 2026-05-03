from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import AudioMetadata, SilenceSegment, SilenceSummary, VolumeStats


_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<start>-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(?P<end>-?\d+(?:\.\d+)?).*?silence_duration:\s*(?P<duration>\d+(?:\.\d+)?)"
)
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(?P<value>-?\d+(?:\.\d+)?)\s*dB")
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(?P<value>-?\d+(?:\.\d+)?)\s*dB")
_HISTOGRAM_0DB_RE = re.compile(r"histogram_0db:\s*(?P<count>\d+)")


def _to_float(value: Any) -> float | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_ffprobe_metadata(raw_json: str) -> AudioMetadata:
    payload = json.loads(raw_json)
    streams = payload.get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    first_audio = audio_streams[0] if audio_streams else {}
    format_info = payload.get("format", {})

    duration_seconds = _to_float(format_info.get("duration"))
    if duration_seconds is None:
        duration_seconds = _to_float(first_audio.get("duration"))

    file_name = Path(format_info.get("filename") or "unknown").name

    return AudioMetadata(
        file_name=file_name,
        duration_seconds=duration_seconds,
        bitrate=_to_int(format_info.get("bit_rate") or first_audio.get("bit_rate")),
        sample_rate=_to_int(first_audio.get("sample_rate")),
        channels=_to_int(first_audio.get("channels")),
        audio_streams=len(audio_streams),
        channel_layout=first_audio.get("channel_layout"),
        codec_name=first_audio.get("codec_name"),
        container_format=format_info.get("format_name"),
        raw=payload,
    )


def parse_silencedetect_output(output: str, duration_seconds: float | None) -> SilenceSummary:
    pending_start: float | None = None
    segments: list[SilenceSegment] = []

    for line in output.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group("start"))
            continue

        end_match = _SILENCE_END_RE.search(line)
        if end_match:
            end = float(end_match.group("end"))
            duration = float(end_match.group("duration"))
            start = pending_start if pending_start is not None else max(0.0, end - duration)
            segments.append(
                SilenceSegment(
                    start_seconds=round(start, 3),
                    end_seconds=round(end, 3),
                    duration_seconds=round(duration, 3),
                )
            )
            pending_start = None

    total_silence = sum(segment.duration_seconds for segment in segments)
    ratio = total_silence / duration_seconds if duration_seconds and duration_seconds > 0 else 0.0
    return SilenceSummary(segments=segments, silence_ratio=round(ratio, 6))


def parse_volumedetect_output(output: str, low_volume_threshold_db: float = -30.0) -> VolumeStats:
    mean_match = _MEAN_VOLUME_RE.search(output)
    max_match = _MAX_VOLUME_RE.search(output)
    histogram_match = _HISTOGRAM_0DB_RE.search(output)

    avg_volume = float(mean_match.group("value")) if mean_match else None
    max_volume = float(max_match.group("value")) if max_match else None
    histogram_0db = int(histogram_match.group("count")) if histogram_match else 0

    clipping_detected = histogram_0db > 0 or (max_volume is not None and max_volume >= -0.5)
    low_volume_detected = avg_volume is not None and avg_volume <= low_volume_threshold_db

    return VolumeStats(
        avg_volume_db=avg_volume,
        max_volume_db=max_volume,
        clipping_detected=clipping_detected,
        low_volume_detected=low_volume_detected,
        histogram_0db=histogram_0db,
    )
