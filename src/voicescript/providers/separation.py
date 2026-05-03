from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from voicescript.config import Settings
from voicescript.schemas import CommandProvenance, SourceSeparationResult


ALLOWED_PROVIDER_NAMES = ("api", "disabled", "huggingface", "local", "other")


class SourceSeparator(Protocol):
    provider_name: str

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        ...

    def separate_vocals(self, input_file: Path, output_dir: Path) -> SourceSeparationResult:
        ...


class DisabledSourceSeparator:
    provider_name = "disabled"

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        return {
            "source_separation": {"available": False, "provider": self.provider_name, "detail": "disabled"},
            "demucs": {"available": False, "enabled": False, "detail": "disabled"},
        }

    def separate_vocals(self, input_file: Path, output_dir: Path) -> SourceSeparationResult:
        return SourceSeparationResult(
            available=False,
            enabled=False,
            limitations=["Source separation disabled by provider configuration."],
        )


class UnavailableSourceSeparator:
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        detail = "selected but no offline adapter is configured"
        return {
            "source_separation": {
                "available": False,
                "provider": self.provider_name,
                "detail": detail,
            },
            "demucs": {"available": False, "enabled": False, "detail": detail},
        }

    def separate_vocals(self, input_file: Path, output_dir: Path) -> SourceSeparationResult:
        return SourceSeparationResult(
            engine=self.provider_name,
            available=False,
            enabled=False,
            limitations=[
                f"Provider '{self.provider_name}' is selected but no offline source-separation adapter is configured; "
                "no network call was attempted."
            ],
        )


class LocalDemucsSeparator:
    provider_name = "local-demucs"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env(require_api_key=False)

    def readiness(self) -> dict[str, dict[str, str | bool]]:
        installed = _demucs_available()
        detail = "available" if installed else "not installed"
        if installed and not self.settings.demucs_enabled:
            detail = "installed; disabled by configuration"
        return {
            "source_separation": {
                "available": installed and self.settings.demucs_enabled,
                "provider": self.provider_name,
                "detail": detail,
            },
            "demucs": {"available": installed, "enabled": self.settings.demucs_enabled, "detail": detail},
        }

    def separate_vocals(self, input_file: Path, output_dir: Path) -> SourceSeparationResult:
        if not self.settings.demucs_enabled:
            return SourceSeparationResult(
                available=_demucs_available(),
                enabled=False,
                limitations=["Demucs source separation is disabled by configuration."],
            )
        if not _demucs_available():
            return SourceSeparationResult(
                available=False,
                enabled=True,
                limitations=["Demucs is not installed."],
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            self.settings.demucs_model,
            "-o",
            str(output_dir),
            str(input_file),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        provenance = CommandProvenance(
            tool="demucs",
            command=command,
            exit_code=completed.returncode,
            stdout_excerpt=completed.stdout[:1000],
            stderr_excerpt=completed.stderr[:1000],
        )
        if completed.returncode != 0:
            return SourceSeparationResult(
                available=True,
                enabled=True,
                limitations=[f"Demucs failed with exit code {completed.returncode}."],
                provenance=[provenance],
            )

        vocals_path = _find_vocals_file(output_dir, input_file)
        limitations = [] if vocals_path else ["Demucs completed but no vocals.wav output was found."]
        return SourceSeparationResult(
            available=True,
            enabled=True,
            vocals_path=str(vocals_path) if vocals_path else None,
            limitations=limitations,
            provenance=[provenance],
        )


def create_source_separator(settings: Settings) -> SourceSeparator:
    provider = _normalise_provider(settings.source_separation_provider)
    if provider == "disabled":
        return DisabledSourceSeparator()
    if provider == "local":
        return LocalDemucsSeparator(settings)
    if provider in {"api", "huggingface", "other"}:
        return UnavailableSourceSeparator(provider)
    raise ValueError(
        f"Unsupported source separation provider '{settings.source_separation_provider}'. {_allowed_provider_message()}"
    )


def _demucs_available() -> bool:
    return find_spec("demucs") is not None


def _normalise_provider(provider: str) -> str:
    return provider.strip().lower()


def _allowed_provider_message() -> str:
    return f"Allowed values: {', '.join(ALLOWED_PROVIDER_NAMES)}."


def _find_vocals_file(output_dir: Path, input_file: Path) -> Path | None:
    stem = input_file.stem
    candidates = list(output_dir.glob(f"**/{stem}/vocals.wav")) + list(output_dir.glob("**/vocals.wav"))
    return candidates[0] if candidates else None
