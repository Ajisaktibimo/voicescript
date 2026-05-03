from __future__ import annotations

import re

from voicescript.schemas import (
    EvaluationReference,
    EvaluationResult,
    Estimate,
    ForensicReport,
    SpeakerReferenceSegment,
    SpeakerSegment,
)


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))

    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            substitution_cost = 0 if left_word == right_word else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    reference_words = _normalize_words(reference)
    hypothesis_words = _normalize_words(hypothesis)

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    return round(_edit_distance(reference_words, hypothesis_words) / len(reference_words), 6)


def evaluate_speaker_count(
    reference_segments: list[SpeakerReferenceSegment],
    predicted: Estimate,
) -> tuple[int | None, int | None, bool | None]:
    reference_speakers = {
        segment.speaker.strip()
        for segment in reference_segments
        if segment.speaker.strip()
    }
    expected = len(reference_speakers) if reference_speakers else None
    actual = predicted.value
    if expected is None or actual is None:
        return expected, actual, None
    return expected, actual, expected == actual


def compute_speaker_attribution_error(
    reference_segments: list[SpeakerReferenceSegment],
    predicted_segments: list[SpeakerSegment],
) -> float | None:
    total_overlap = 0.0
    wrong_overlap = 0.0
    for reference in reference_segments:
        for predicted in predicted_segments:
            overlap = _overlap_seconds(
                reference.start_seconds,
                reference.end_seconds,
                predicted.start_seconds,
                predicted.end_seconds,
            )
            if overlap <= 0:
                continue
            total_overlap += overlap
            if reference.speaker.strip() != predicted.speaker.strip():
                wrong_overlap += overlap
    if total_overlap == 0:
        return None
    return round(wrong_overlap / total_overlap, 6)


def _overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def evaluate_report(report: ForensicReport, reference: EvaluationReference) -> EvaluationResult:
    expected, predicted, count_match = evaluate_speaker_count(
        reference.speaker_segments,
        report.estimated_speaker_count,
    )
    return EvaluationResult(
        file_name=report.file_name,
        wer=compute_wer(reference.transcript.text, report.transcript_text),
        word_count=len(_normalize_words(reference.transcript.text)),
        speaker_count_expected=expected,
        speaker_count_predicted=predicted,
        speaker_count_match=count_match,
        speaker_attribution_error_rate=compute_speaker_attribution_error(
            reference.speaker_segments,
            report.speaker_segments,
        ),
        calibration_bucket=report.estimated_speaker_count.confidence,
        notes=[],
    )
