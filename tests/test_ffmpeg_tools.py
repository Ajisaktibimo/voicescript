from __future__ import annotations

from pathlib import Path

import pytest

from voicescript.config import Settings
from voicescript.ffmpeg_tools import FFmpegTools
from voicescript.models import ToolResult


@pytest.fixture()
def ffmpeg(monkeypatch, runtime_dir):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="test",
        transcription_provider="disabled",
        diarization_provider="disabled",
        source_separation_provider="disabled",
    )
    tools = FFmpegTools(settings)
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, required_binary: str) -> ToolResult:
        captured.append(command)
        return ToolResult(stdout="", stderr="", exit_code=0, command=command)

    monkeypatch.setattr(tools, "run", fake_run)
    return tools, captured


def test_normalize_for_speech_forces_mono_when_channels_omitted(ffmpeg, runtime_dir):
    tools, captured = ffmpeg
    output = runtime_dir / "out.wav"

    tools.normalize_for_speech(runtime_dir / "input.wav", output)

    assert captured, "ffmpeg command was not invoked"
    command = captured[0]
    assert "-ar" in command and "16000" in command
    assert "-ac" in command
    assert command[command.index("-ac") + 1] == "1"


def test_normalize_for_speech_preserves_stereo_when_channels_two(ffmpeg, runtime_dir):
    tools, captured = ffmpeg
    output = runtime_dir / "out.wav"

    tools.normalize_for_speech(runtime_dir / "input.wav", output, channels=2)

    command = captured[0]
    assert "-ar" in command and "16000" in command
    assert "-ac" in command
    assert command[command.index("-ac") + 1] == "2"


def test_normalize_for_speech_forces_mono_when_channels_one(ffmpeg, runtime_dir):
    tools, captured = ffmpeg
    output = runtime_dir / "out.wav"

    tools.normalize_for_speech(runtime_dir / "input.wav", output, channels=1)

    command = captured[0]
    assert command[command.index("-ac") + 1] == "1"


def test_normalize_for_speech_preserves_multichannel_for_multi_mic_sources(ffmpeg, runtime_dir):
    """Forensic captures (e.g. 4-mic dictaphone, 6-track interview rig) carry per-mic
    speaker cues. Preserve the source channel count rather than collapsing it."""
    tools, captured = ffmpeg
    output = runtime_dir / "out.wav"

    tools.normalize_for_speech(runtime_dir / "input.wav", output, channels=6)

    command = captured[0]
    assert command[command.index("-ac") + 1] == "6"


def test_normalize_for_speech_force_mono_overrides_source_channels(ffmpeg, runtime_dir):
    """Operator can opt in to mono collapse for old pipelines or single-stream tools."""
    tools, captured = ffmpeg
    output = runtime_dir / "out.wav"

    tools.normalize_for_speech(runtime_dir / "input.wav", output, channels=4, force_mono=True)

    command = captured[0]
    assert command[command.index("-ac") + 1] == "1"
