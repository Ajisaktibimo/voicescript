"""Forensic fusion layer.

Combines acoustic diarization output with transcript / channel cues so that
overconfident single-speaker results are downgraded when other evidence
implies multiple participants. Rules are deterministic and individually
auditable: each cue that fires is named in the evidence so the analyst can
inspect why a downgrade happened.

Design constraints:
- Never mutate the raw DiarizationResult. The diarization stays as evidence;
  only the top-level ForensicReport.estimated_speaker_count Estimate is fused.
- Never silently overwrite a count without naming the cues responsible.
- English-only cue lists for now (limitation flagged in evidence).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import (
    AudioMetadata,
    ChannelAnalysis,
    DiarizationResult,
    Estimate,
    ForensicFinding,
    TranscriptionResult,
)


SHORT_AUDIO_SECONDS = 8.0
# Each cue carries a numeric strength. Override fires when total >= OVERRIDE_STRENGTH.
# Strong cues (Q:/A: structure, pronoun-turn alternation) are sufficient alone;
# supporting cues (reported speech, legal roles, multi-channel) need to combine.
OVERRIDE_STRENGTH = 2
HIGH_CONFIDENCE_STRENGTH = 4
STRENGTH_STRONG = 2
STRENGTH_SUPPORTING = 1


_LEGAL_ROLES = (
    "officer",
    "detective",
    "counsel",
    "attorney",
    "witness",
    "suspect",
    "defendant",
    "prosecutor",
    "judge",
    "interviewer",
    "interviewee",
)

_REPORTED_SPEECH_VERBS = (
    "said",
    "replied",
    "answered",
    "responded",
    "asked",
    "told",
    "stated",
    "declared",
    "testified",
    "claimed",
)

_PRONOUN_TURN_PATTERNS = (
    re.compile(r"\bI\b[^.?!]{1,80}?\byou\b", re.IGNORECASE),
    re.compile(r"\byou\b[^.?!]{1,80}?\bI\b", re.IGNORECASE),
    re.compile(r"\bno,?\s+I\b", re.IGNORECASE),
    re.compile(r"\bI did not\b", re.IGNORECASE),
)


class _Cue:
    __slots__ = ("name", "snippet", "min_speakers", "strength")

    def __init__(self, name: str, snippet: str, min_speakers: int, strength: int):
        self.name = name
        self.snippet = snippet
        self.min_speakers = min_speakers
        self.strength = strength

    def evidence_line(self) -> str:
        return f"rule={self.name}: {self.snippet}"


def apply_speaker_fusion(
    *,
    diarization: DiarizationResult,
    transcription: TranscriptionResult,
    channel_analysis: ChannelAnalysis,
    metadata: AudioMetadata,
) -> tuple[Estimate, list[ForensicFinding]]:
    text = transcription.transcript_text or ""
    cues = _collect_cues(text=text, channel_analysis=channel_analysis)
    inferred_min = max((c.min_speakers for c in cues), default=0)
    total_strength = sum(c.strength for c in cues)
    raw_value = diarization.estimated_speaker_count
    raw_confidence = diarization.confidence if diarization.confidence in {"low", "medium", "high"} else "low"

    short_audio = bool(metadata.duration_seconds and metadata.duration_seconds < SHORT_AUDIO_SECONDS)

    fused_value = raw_value
    fused_confidence = raw_confidence
    findings: list[ForensicFinding] = []
    evidence: list[str] = []

    if cues:
        evidence.append(
            f"Fusion cues fired ({len(cues)}, strength={total_strength}): "
            f"{', '.join(sorted({c.name for c in cues}))}."
        )

    if (raw_value or 0) <= 1 and inferred_min >= 2 and total_strength >= OVERRIDE_STRENGTH:
        fused_value = max(raw_value or 0, inferred_min)
        fused_confidence = "low"
        evidence.append(
            f"Raw diarization speaker count: {raw_value} ({raw_confidence}); "
            f"fused estimate: {fused_value} (low) due to {len(cues)} multi-speaker cue(s)."
        )
        findings.append(
            ForensicFinding(
                finding="Diarization-transcript speaker count mismatch",
                severity="medium",
                confidence="medium",
                evidence=[c.evidence_line() for c in cues],
                recommended_follow_up=(
                    "Review the native pyannote diarization output and have an analyst verify the speaker count manually."
                ),
            )
        )
    elif (raw_value or 0) >= 2 and total_strength >= HIGH_CONFIDENCE_STRENGTH:
        fused_value = raw_value
        fused_confidence = "high"
        evidence.append(
            f"Diarization count {raw_value} corroborated by {len(cues)} transcript / channel cue(s); "
            "confidence promoted to high."
        )
    elif raw_value is None and inferred_min >= 2 and total_strength >= OVERRIDE_STRENGTH:
        fused_value = inferred_min
        fused_confidence = "medium"
        evidence.append(
            f"Diarization unavailable; inferring {inferred_min} speaker(s) from "
            f"{len(cues)} transcript cue(s) (medium confidence — transcript-only)."
        )

    if short_audio:
        if fused_confidence == "high":
            fused_confidence = "medium"
        if fused_confidence == "medium":
            fused_confidence = "low"
        evidence.append(
            f"Audio duration {metadata.duration_seconds:.1f}s is short; capping fused confidence."
        )

    final_evidence = list(evidence) if evidence else _baseline_evidence(raw_value, raw_confidence)
    return (
        Estimate(value=fused_value, confidence=fused_confidence, evidence=final_evidence),
        findings,
    )


def _baseline_evidence(value: int | None, confidence: str) -> list[str]:
    if value is None:
        return ["No fusion cues fired; diarization produced no speaker count."]
    return [f"No fusion cues fired; preserving diarization count {value} ({confidence})."]


def _collect_cues(*, text: str, channel_analysis: ChannelAnalysis) -> list[_Cue]:
    cues: list[_Cue] = []
    if text:
        cues.extend(_pronoun_turn_cues(text))
        cues.extend(_reported_speech_cues(text))
        cues.extend(_interrogation_cues(text))
        cues.extend(_legal_role_cues(text))
    cues.extend(_channel_cues(channel_analysis))
    return _dedupe_cues(cues)


def _pronoun_turn_cues(text: str) -> Iterable[_Cue]:
    for pattern in _PRONOUN_TURN_PATTERNS:
        match = pattern.search(text)
        if match:
            yield _Cue(
                "multi_party_pronouns",
                _snippet(text, match.start(), match.end()),
                min_speakers=2,
                strength=STRENGTH_STRONG,
            )
            return


def _reported_speech_cues(text: str) -> Iterable[_Cue]:
    pattern = re.compile(
        r"\b(he|she|they|the\s+\w+)\s+(" + "|".join(_REPORTED_SPEECH_VERBS) + r")\b",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        yield _Cue(
            "reported_speech",
            _snippet(text, match.start(), match.end()),
            min_speakers=2,
            strength=STRENGTH_SUPPORTING,
        )


def _interrogation_cues(text: str) -> Iterable[_Cue]:
    if re.search(r"(?:^|[\.\?!\s])Q[:.]\s", text) and re.search(r"(?:^|[\.\?!\s])A[:.]\s", text):
        yield _Cue(
            "interrogation_pattern",
            "Q:/A: turn markers found in transcript",
            min_speakers=2,
            strength=STRENGTH_STRONG,
        )


def _legal_role_cues(text: str) -> Iterable[_Cue]:
    lowered = text.lower()
    distinct_roles = sorted({role for role in _LEGAL_ROLES if re.search(rf"\b{role}\b", lowered)})
    if len(distinct_roles) >= 2:
        yield _Cue(
            "legal_role_names",
            f"distinct role tokens detected: {', '.join(distinct_roles)}",
            min_speakers=len(distinct_roles),
            strength=STRENGTH_SUPPORTING,
        )


def _channel_cues(channel_analysis: ChannelAnalysis) -> Iterable[_Cue]:
    measured = channel_analysis.measured_channels or 0
    if measured >= 2:
        yield _Cue(
            "multi_channel_capture",
            f"input has {measured} measured channel(s); per-mic separation supports >1 speaker",
            min_speakers=2,
            strength=STRENGTH_SUPPORTING,
        )


def _dedupe_cues(cues: list[_Cue]) -> list[_Cue]:
    seen: set[str] = set()
    unique: list[_Cue] = []
    for cue in cues:
        if cue.name in seen:
            continue
        seen.add(cue.name)
        unique.append(cue)
    return unique


def _snippet(text: str, start: int, end: int, *, pad: int = 30) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    excerpt = text[lo:hi].strip().replace("\n", " ")
    return f"... {excerpt} ..."
