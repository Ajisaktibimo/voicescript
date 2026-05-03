from pathlib import Path

from voicescript.mcp_tools import McpToolset


class FakeAnalyzer:
    def analyze_file(self, path: Path):
        return {"file_name": path.name, "forensic_profile": {"classification": "forensic_triage"}}

    def analyze_batch(self, paths):
        return {"file_count": len(paths), "reports": [self.analyze_file(Path(p)) for p in paths]}

    def get_audio_metadata(self, path: Path):
        return {"file_name": path.name, "channels": 2, "audio_streams": 1}

    def analyze_audio_quality(self, path: Path):
        return {"silence_ratio": 0.0, "clipping_detected": False}

    def analyze_channels(self, path: Path):
        return {"measured_channels": 2, "estimated_microphone_count": 1}

    def estimate_speakers(self, path: Path):
        return {"value": None, "confidence": "low"}

    def estimate_microphone_setup(self, path: Path):
        return {"value": 1, "confidence": "medium"}

    def detect_forensic_indicators(self, path: Path):
        return []


    def evaluate_report(self, report, reference):
        return {"file_name": report["file_name"], "wer": 0.0}


def test_mcp_toolset_delegates_to_analyzer_and_returns_json_ready_data():
    toolset = McpToolset(analyzer=FakeAnalyzer())

    assert toolset.get_audio_metadata("court.wav")["channels"] == 2
    assert toolset.analyze_audio_quality("court.wav")["clipping_detected"] is False
    assert toolset.analyze_channels("court.wav")["estimated_microphone_count"] == 1
    assert toolset.estimate_speakers("court.wav")["confidence"] == "low"
    assert toolset.estimate_microphone_setup("court.wav")["value"] == 1
    assert toolset.detect_forensic_indicators("court.wav") == []
    assert toolset.analyze_audio_file("court.wav")["file_name"] == "court.wav"
    assert toolset.analyze_audio_batch(["a.wav", "b.wav"])["file_count"] == 2


def test_evaluate_report_tool_delegates_to_analyzer():
    toolset = McpToolset(analyzer=FakeAnalyzer())

    result = toolset.evaluate_report(
        {"file_name": "court.wav", "sha256": "abc"},
        {"file_name": "court.wav", "transcript": {"text": ""}, "speaker_segments": []},
    )

    assert result["file_name"] == "court.wav"
    assert result["wer"] == 0.0
