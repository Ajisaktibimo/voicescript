from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field


class Settings(BaseModel):
    data_dir: Path = Field(default=Path("data"))
    api_key: str = ""
    inline_jobs: bool = False
    max_upload_size_bytes: int = 500 * 1024 * 1024
    max_batch_files: int = 25
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    silence_noise_db: str = "-50dB"
    silence_min_duration_seconds: float = 2.0
    low_volume_threshold_db: float = -30.0
    transcription_provider: str = "local"
    diarization_provider: str = "local"
    source_separation_provider: str = "local"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    pyannote_auth_token: str | None = None
    demucs_enabled: bool = True
    demucs_model: str = "htdemucs"

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> "Settings":
        values = _load_env_values(Path(".env"))

        def get(name: str, default: str | None = None) -> str | None:
            return os.getenv(name) or values.get(name) or default

        data_dir = Path(get("VOICESCRIPT_DATA_DIR", "data") or "data")
        api_key = get("VOICESCRIPT_API_KEY")
        if require_api_key and not api_key:
            raise ValueError("VOICESCRIPT_API_KEY must be set; refusing to use a built-in default API key.")
        inline_jobs = _truthy(get("VOICESCRIPT_INLINE_JOBS", "false"))
        settings = cls(
            data_dir=data_dir,
            api_key=api_key or "",
            inline_jobs=inline_jobs,
            max_upload_size_bytes=int(get("VOICESCRIPT_MAX_UPLOAD_SIZE_BYTES", str(500 * 1024 * 1024)) or "0"),
            max_batch_files=int(get("VOICESCRIPT_MAX_BATCH_FILES", "25") or "25"),
            ffmpeg_binary=get("VOICESCRIPT_FFMPEG", "ffmpeg") or "ffmpeg",
            ffprobe_binary=get("VOICESCRIPT_FFPROBE", "ffprobe") or "ffprobe",
            silence_noise_db=get("VOICESCRIPT_SILENCE_NOISE_DB", "-50dB") or "-50dB",
            silence_min_duration_seconds=float(get("VOICESCRIPT_SILENCE_MIN_DURATION_SECONDS", "2") or "2"),
            low_volume_threshold_db=float(get("VOICESCRIPT_LOW_VOLUME_THRESHOLD_DB", "-30") or "-30"),
            transcription_provider=(get("VOICESCRIPT_TRANSCRIPTION_PROVIDER", "local") or "local").lower(),
            diarization_provider=(get("VOICESCRIPT_DIARIZATION_PROVIDER", "local") or "local").lower(),
            source_separation_provider=(get("VOICESCRIPT_SOURCE_SEPARATION_PROVIDER", "local") or "local").lower(),
            whisper_model=get("VOICESCRIPT_WHISPER_MODEL", "small") or "small",
            whisper_device=get("VOICESCRIPT_WHISPER_DEVICE", "auto") or "auto",
            whisper_compute_type=get("VOICESCRIPT_WHISPER_COMPUTE_TYPE", "int8") or "int8",
            pyannote_model=get("VOICESCRIPT_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
            or "pyannote/speaker-diarization-3.1",
            pyannote_auth_token=get("PYANNOTE_AUTH_TOKEN"),
            demucs_enabled=_truthy(get("VOICESCRIPT_DEMUCS_ENABLED", "true")),
            demucs_model=get("VOICESCRIPT_DEMUCS_MODEL", "htdemucs") or "htdemucs",
        )
        _setup_windows_dlls(settings)
        return settings


def _setup_windows_dlls(settings: Settings) -> None:
    """Ensure FFmpeg DLLs are discoverable on Windows."""
    if os.name != "nt":
        return

    # If the user provided an absolute path to FFmpeg, add its directory to the DLL search path
    ffmpeg_path = Path(settings.ffmpeg_binary)
    if ffmpeg_path.is_absolute():
        bin_dir = ffmpeg_path.parent
        if bin_dir.exists():
            # Add to PATH for subprocesses
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
            # Add to DLL search path for Python DLL loading (required for torchcodec)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except Exception:
                    # Some paths might fail, we can ignore and fallback
                    pass


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def _load_env_values(path: Path) -> Mapping[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values
