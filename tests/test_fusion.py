from __future__ import annotations

import pytest

from voicescript.fusion import apply_speaker_fusion
from voicescript.schemas import (
    AudioMetadata,
    ChannelAnalysis,
    DiarizationResult,
    Estimate,
    ForensicFinding,
    SpeakerSegment,
    TranscriptSegment,
    TranscriptionResult,
)


def _meta(*, duration: float = 60.0, channels: int = 1, layout: str = "mono") -> AudioMetadata:
    return AudioMetadata(
        file_name="court.wav",
        duration_seconds=duration,
        bitrate=128000,
        sample_rate=16000,
        channels=channels,
        audio_streams=1,
        channel_layout=layout,
        codec_name="pcm_s16le",
        container_format="wav",
        raw={},
    )


def _channels(*, channels: int = 1, layout: str = "mono") -> ChannelAnalysis:
    return ChannelAnalysis(
        measured_channels=channels,
        audio_streams=1,
        channel_layout=layout,
        duplicated_channels_likely=False,
        channel_imbalance_db=None,
        estimated_microphone_count=channels if channels == 1 else None,
        confidence="medium" if channels == 1 else "low",
        evidence=[],
    )


def _diar(*, value: int | None, confidence: str = "medium") -> DiarizationResult:
    segments: list[SpeakerSegment] = []
    if value:
        for i in range(value):
            segments.append(
                SpeakerSegment(
                    speaker=f"SPEAKER_{i:02d}",
                    start_seconds=float(i),
                    end_seconds=float(i + 1),
                    text="",
                )
            )
    return DiarizationResult(
        provider="local-pyannote",
        output_type="fixture",
        speaker_segments=segments,
        estimated_speaker_count=value,
        confidence=confidence,
    )


def _transcription(text: str) -> TranscriptionResult:
    return TranscriptionResult(
        provider="local-faster-whisper",
        output_type="fixture",
        transcript_text=text,
        transcript_segments=[
            TranscriptSegment(start_seconds=0.0, end_seconds=10.0, text=text)
        ],
    )


def _fired_rules(findings: list[ForensicFinding]) -> set[str]:
    rules: set[str] = set()
    for finding in findings:
        for line in finding.evidence:
            if line.startswith("rule="):
                rules.add(line.split("=", 1)[1].split(":")[0].strip())
    return rules


def test_diarization_one_speaker_with_multi_party_pronouns_downgrades_and_flags_mismatch():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription(
            "I told you yesterday. You said it was fine. No, I did not say that."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value == 2
    assert estimate.confidence == "low"
    assert any("speaker count mismatch" in f.finding.lower() for f in findings)
    fired = _fired_rules(findings)
    assert "multi_party_pronouns" in fired


def test_diarization_one_speaker_with_interrogation_qna_pattern_flags_mismatch():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription(
            "Q: Where were you that night? A: I was at home. Q: Alone? A: Yes."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value == 2
    assert estimate.confidence == "low"
    assert any("speaker count mismatch" in f.finding.lower() for f in findings)
    assert "interrogation_pattern" in _fired_rules(findings)


def test_diarization_one_speaker_with_monologue_transcript_keeps_one_speaker():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription(
            "I went to the store. I bought milk. I came back home and made tea."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value == 1
    assert not any("speaker count mismatch" in f.finding.lower() for f in findings)


def test_diarization_one_speaker_with_single_cue_does_not_override():
    """One cue alone is not enough — avoid false positives. Need >= 2 cues."""
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription("The officer arrived at the scene."),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value == 1
    assert not any("speaker count mismatch" in f.finding.lower() for f in findings)


def test_diarization_one_speaker_with_legal_roles_and_reported_speech_flags_mismatch():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription(
            "The officer said the suspect was uncooperative. The witness replied that she heard nothing."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value >= 2
    assert estimate.confidence == "low"
    fired = _fired_rules(findings)
    assert "reported_speech" in fired
    assert "legal_role_names" in fired


def test_diarization_two_speakers_with_concurring_transcript_promotes_to_high():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=2, confidence="medium"),
        transcription=_transcription(
            "Q: Officer, what did you observe? A: I observed the suspect leaving. The witness said she saw him too."
        ),
        channel_analysis=_channels(channels=2, layout="stereo"),
        metadata=_meta(channels=2, layout="stereo"),
    )

    assert estimate.value == 2
    assert estimate.confidence == "high"


def test_short_audio_caps_confidence_to_low_even_with_strong_cues():
    estimate, _findings = apply_speaker_fusion(
        diarization=_diar(value=2, confidence="high"),
        transcription=_transcription(
            "Q: Officer? A: Yes. Q: Witness? A: Yes."
        ),
        channel_analysis=_channels(channels=2, layout="stereo"),
        metadata=_meta(duration=4.0, channels=2, layout="stereo"),
    )

    assert estimate.confidence == "low"


def test_empty_diarization_with_strong_cues_promotes_only_to_medium():
    estimate, _findings = apply_speaker_fusion(
        diarization=_diar(value=None, confidence="low"),
        transcription=_transcription(
            "Q: Officer? A: Yes. The witness said she heard the suspect arguing."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert estimate.value is not None and estimate.value >= 2
    assert estimate.confidence in {"low", "medium"}


def test_fusion_does_not_mutate_diarization_result():
    diar = _diar(value=1)
    apply_speaker_fusion(
        diarization=diar,
        transcription=_transcription(
            "Q: Officer? A: Yes. The witness replied that she saw nothing."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )

    assert diar.estimated_speaker_count == 1
    assert diar.confidence == "medium"


def test_fusion_evidence_names_each_rule_that_fired():
    estimate, findings = apply_speaker_fusion(
        diarization=_diar(value=1),
        transcription=_transcription(
            "Q: Officer, did the witness say anything? A: He replied that he saw the suspect leave."
        ),
        channel_analysis=_channels(),
        metadata=_meta(),
    )
    assert estimate.evidence, "fused estimate should carry evidence describing why it changed"
    assert findings, "a forensic finding should be emitted on mismatch"
    finding_evidence = " ".join(f for finding in findings for f in finding.evidence)
    assert "rule=" in finding_evidence
