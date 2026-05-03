from __future__ import annotations

from pathlib import Path

from .analyzer import ForensicAnalyzer
from .models import to_jsonable


class McpToolset:
    def __init__(self, analyzer: ForensicAnalyzer | object | None = None):
        self.analyzer = analyzer or ForensicAnalyzer()

    def get_audio_metadata(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.get_audio_metadata(input_file))

    def analyze_audio_quality(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.analyze_audio_quality(input_file))

    def detect_silence(self, input_file: str) -> dict[str, object]:
        return self.analyze_audio_quality(input_file)

    def detect_clipping(self, input_file: str) -> dict[str, object]:
        quality = self.analyze_audio_quality(input_file)
        return {
            "clipping_detected": quality["clipping_detected"],
            "avg_volume_db": quality["avg_volume_db"],
            "max_volume_db": quality["max_volume_db"],
        }

    def analyze_channels(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.analyze_channels(input_file))

    def estimate_speakers(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.estimate_speakers(input_file))

    def estimate_microphone_setup(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.estimate_microphone_setup(input_file))

    def detect_forensic_indicators(self, input_file: str) -> list[dict[str, object]]:
        return to_jsonable(self.analyzer.detect_forensic_indicators(input_file))

    def generate_spectrogram(self, input_file: str, output_image: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.generate_spectrogram(Path(input_file), Path(output_image)))

    def analyze_audio_file(self, input_file: str) -> dict[str, object]:
        return to_jsonable(self.analyzer.analyze_file(input_file))

    def analyze_audio_batch(self, input_files: list[str]) -> dict[str, object]:
        return to_jsonable(self.analyzer.analyze_batch([Path(input_file) for input_file in input_files]))

    def evaluate_report(self, report: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
        return to_jsonable(self.analyzer.evaluate_report(report, reference))
