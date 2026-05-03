from pathlib import Path

import voicescript.providers.separation as separation
from voicescript.demucs_tools import DemucsSeparator


def test_demucs_readiness_reports_missing_dependency_without_failing(runtime_dir, monkeypatch):
    monkeypatch.setattr(separation, "_demucs_available", lambda: False)
    separator = DemucsSeparator()

    readiness = separator.readiness()
    result = separator.separate_vocals(Path("court.wav"), runtime_dir)

    assert readiness["demucs"]["available"] is False
    assert result.engine == "demucs"
    assert result.available is False
    assert result.enabled is True
    assert result.vocals_path is None
    assert "not installed" in result.limitations[0].lower()
