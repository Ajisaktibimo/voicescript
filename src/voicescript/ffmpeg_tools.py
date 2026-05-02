from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .models import CommandProvenance, ToolResult


class DependencyUnavailable(RuntimeError):
    """Raised when an external binary required for analysis is unavailable."""


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, result: ToolResult):
        super().__init__(message)
        self.result = result


class FFmpegTools:
    def __init__(self, settings: Settings):
        self.settings = settings

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "ffmpeg": self._binary_status(self.settings.ffmpeg_binary),
            "ffprobe": self._binary_status(self.settings.ffprobe_binary),
        }

    def ffprobe_metadata(self, input_file: Path) -> ToolResult:
        return self.run(
            [
                self.settings.ffprobe_binary,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(input_file),
            ],
            required_binary=self.settings.ffprobe_binary,
        )

    def detect_silence(self, input_file: Path) -> ToolResult:
        return self.run(
            [
                self.settings.ffmpeg_binary,
                "-i",
                str(input_file),
                "-af",
                f"silencedetect=noise={self.settings.silence_noise_db}:d={self.settings.silence_min_duration_seconds:g}",
                "-f",
                "null",
                "-",
            ],
            required_binary=self.settings.ffmpeg_binary,
        )

    def detect_volume(self, input_file: Path) -> ToolResult:
        return self.run(
            [
                self.settings.ffmpeg_binary,
                "-i",
                str(input_file),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            required_binary=self.settings.ffmpeg_binary,
        )

    def generate_spectrogram(self, input_file: Path, output_image: Path) -> ToolResult:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        return self.run(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(input_file),
                "-lavfi",
                "showspectrumpic=s=1920x1080",
                str(output_image),
            ],
            required_binary=self.settings.ffmpeg_binary,
        )

    def normalize_for_speech(
        self,
        input_file: Path,
        output_file: Path,
        *,
        channels: int | None = None,
        force_mono: bool = False,
    ) -> ToolResult:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if force_mono:
            target_channels = 1
        elif channels is None:
            target_channels = 1
        else:
            target_channels = max(1, int(channels))
        return self.run(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(input_file),
                "-ar",
                "16000",
                "-ac",
                str(target_channels),
                str(output_file),
            ],
            required_binary=self.settings.ffmpeg_binary,
        )

    def run(self, command: list[str], *, required_binary: str) -> ToolResult:
        if shutil.which(required_binary) is None:
            raise DependencyUnavailable(f"Required binary is not available on PATH: {required_binary}")

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result = ToolResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            command=command,
        )
        if completed.returncode != 0:
            raise ToolExecutionError(f"Command failed with exit code {completed.returncode}", result)
        return result

    def provenance(self, tool: str, result: ToolResult) -> CommandProvenance:
        return CommandProvenance(
            tool=tool,
            command=result.command,
            exit_code=result.exit_code,
            stdout_excerpt=result.stdout[:1000],
            stderr_excerpt=result.stderr[:1000],
        )

    def _binary_status(self, binary: str) -> dict[str, str | bool]:
        resolved = shutil.which(binary)
        if not resolved:
            return {"available": False, "path": "", "version": ""}
        version = self._version_line(binary)
        return {"available": True, "path": resolved, "version": version}

    def _version_line(self, binary: str) -> str:
        try:
            completed = subprocess.run(
                [binary, "-version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except OSError:
            return ""
        return (completed.stdout or completed.stderr).splitlines()[0] if (completed.stdout or completed.stderr) else ""
