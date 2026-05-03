"""Backward-compatible speech provider exports.

New code should import from ``voicescript.providers.speech``.
"""

from .config import Settings
from .providers.speech import (  # noqa: F401
    DiarizationProvider,
    DisabledDiarizer,
    DisabledTranscriber,
    LocalPyannoteDiarizer,
    LocalWhisperTranscriber,
    SpeechAnalyzer,
    SpeechAnalysisResult as SpeechResult,
    TranscriptionProvider,
    create_speech_analyzer,
)


class LocalSpeechAnalyzer(SpeechAnalyzer):
    """Compatibility wrapper for the old settings-based local analyzer."""

    def __init__(self, settings: Settings):
        analyzer = create_speech_analyzer(settings)
        super().__init__(transcriber=analyzer.transcriber, diarizer=analyzer.diarizer)
