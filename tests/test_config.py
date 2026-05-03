from pathlib import Path

import pytest

from voicescript.config import Settings


def test_settings_loads_values_from_env_file(runtime_dir, monkeypatch):
    env_file = runtime_dir / ".env"
    env_file.write_text(
        "\n".join(
            [
                "VOICESCRIPT_API_KEY=from-env-file",
                "VOICESCRIPT_DATA_DIR=custom-data",
                "VOICESCRIPT_FFMPEG=C:/tools/ffmpeg/bin/ffmpeg.exe",
                "VOICESCRIPT_FFPROBE=C:/tools/ffmpeg/bin/ffprobe.exe",
                "VOICESCRIPT_INLINE_JOBS=true",
                "VOICESCRIPT_DEMUCS_ENABLED=false",
                "PYANNOTE_AUTH_TOKEN=local-token",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(runtime_dir)
    monkeypatch.delenv("VOICESCRIPT_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.api_key == "from-env-file"
    assert settings.data_dir == Path("custom-data")
    assert settings.ffmpeg_binary == "C:/tools/ffmpeg/bin/ffmpeg.exe"
    assert settings.ffprobe_binary == "C:/tools/ffmpeg/bin/ffprobe.exe"
    assert settings.inline_jobs is True
    assert settings.demucs_enabled is False
    assert settings.pyannote_auth_token == "local-token"


def test_settings_requires_explicit_api_key(runtime_dir, monkeypatch):
    monkeypatch.chdir(runtime_dir)
    monkeypatch.delenv("VOICESCRIPT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="VOICESCRIPT_API_KEY"):
        Settings.from_env()


def test_env_example_does_not_ship_runnable_api_key():
    lines = Path(".env.example").read_text(encoding="utf-8").splitlines()
    api_key_lines = [line for line in lines if line.startswith("VOICESCRIPT_API_KEY=")]

    assert api_key_lines == ["VOICESCRIPT_API_KEY="]
