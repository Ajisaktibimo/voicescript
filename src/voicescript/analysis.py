from __future__ import annotations

from collections import Counter

from .fusion import apply_speaker_fusion
from .models import (
    AudioMetadata,
    AudioQuality,
    ChannelAnalysis,
    CommandProvenance,
    DiarizationResult,
    Estimate,
    EvidenceManifest,
    ForensicFinding,
    ForensicReport,
    SilenceSummary,
    SpeakerSegment,
    TranscriptionResult,
    VolumeStats,
)


LONG_SILENCE_SECONDS = 10.0
CHANNEL_IMBALANCE_DB = 6.0


def build_report_from_measurements(
    *,
    file_name: str,
    sha256: str,
    metadata: AudioMetadata,
    silence: SilenceSummary,
    volume: VolumeStats,
    channel_analysis: ChannelAnalysis,
    speaker_segments: list[SpeakerSegment],
    transcript_text: str,
    provenance: list[CommandProvenance],
    source_path: str | None = None,
    extra_limitations: list[str] | None = None,
    transcription: TranscriptionResult | dict[str, object] | None = None,
    diarization: DiarizationResult | dict[str, object] | None = None,
    evidence_manifest: EvidenceManifest | dict[str, object] | None = None,
) -> ForensicReport:
    raw_estimate = _estimate_speaker_count(speaker_segments)
    estimated_microphone_count = Estimate(
        value=channel_analysis.estimated_microphone_count,
        confidence=channel_analysis.confidence,
        evidence=channel_analysis.evidence,
    )

    findings = _build_findings(metadata, silence, volume, channel_analysis)
    transcription_result = _coerce_transcription(transcription)
    diarization_result = _coerce_diarization(diarization)

    estimated_speaker_count, fusion_findings = _fuse_speaker_estimate(
        raw_estimate=raw_estimate,
        diarization=diarization_result,
        transcription=transcription_result,
        channel_analysis=channel_analysis,
        metadata=metadata,
    )
    findings.extend(fusion_findings)

    issues = [finding.finding for finding in findings]
    recommendations = _build_recommendations(findings, estimated_speaker_count)
    limitations = [
        "This report is forensic triage only and is not a certified authentication opinion.",
        "Speaker and microphone counts are estimates derived from measurable signals.",
    ]
    if estimated_speaker_count.value is None:
        limitations.append("Speaker count is unknown because diarization was unavailable or inconclusive.")
    if extra_limitations:
        limitations.extend(extra_limitations)

    evidence_trail = [
        f"ffprobe measured {metadata.audio_streams} audio stream(s), {metadata.channels} channel(s), "
        f"sample rate {metadata.sample_rate} Hz.",
        f"silencedetect found {len(silence.segments)} silence segment(s), ratio {silence.silence_ratio:.3f}.",
        f"volumedetect measured average {volume.avg_volume_db} dB and max {volume.max_volume_db} dB.",
        *channel_analysis.evidence,
        *estimated_speaker_count.evidence,
    ]

    summary_text = _build_summary_text(findings, estimated_speaker_count, estimated_microphone_count)

    return ForensicReport(
        file_name=file_name,
        source_path=source_path,
        sha256=sha256,
        duration_seconds=metadata.duration_seconds,
        bitrate=metadata.bitrate,
        sample_rate=metadata.sample_rate,
        channels=metadata.channels,
        audio_streams=metadata.audio_streams,
        codec_name=metadata.codec_name,
        container_format=metadata.container_format,
        estimated_speaker_count=estimated_speaker_count,
        estimated_microphone_count=estimated_microphone_count,
        channel_analysis=channel_analysis,
        audio_quality=AudioQuality(
            silence_ratio=silence.silence_ratio,
            silence_segments=silence.segments,
            clipping_detected=volume.clipping_detected,
            avg_volume_db=volume.avg_volume_db,
            max_volume_db=volume.max_volume_db,
            low_volume_detected=volume.low_volume_detected,
        ),
        transcript_text=transcript_text,
        speaker_segments=speaker_segments,
        transcription=transcription_result,
        diarization=diarization_result,
        tamper_indicators=findings,
        issues=issues,
        recommendations=recommendations,
        confidence_scores={
            "speaker_count": estimated_speaker_count.confidence,
            "microphone_count": estimated_microphone_count.confidence,
            "channel_analysis": channel_analysis.confidence,
        },
        evidence_trail=evidence_trail,
        evidence_manifest=_coerce_evidence_manifest(evidence_manifest),
        limitations=limitations,
        summary_text=summary_text,
        provenance=provenance,
    )


def _fuse_speaker_estimate(
    *,
    raw_estimate: Estimate,
    diarization: DiarizationResult,
    transcription: TranscriptionResult,
    channel_analysis: ChannelAnalysis,
    metadata: AudioMetadata,
) -> tuple[Estimate, list[ForensicFinding]]:
    diarization_for_fusion = diarization
    if diarization.estimated_speaker_count is None and raw_estimate.value is not None:
        diarization_for_fusion = diarization.model_copy(
            update={
                "estimated_speaker_count": raw_estimate.value,
                "confidence": raw_estimate.confidence,
            }
        )

    fused, fusion_findings = apply_speaker_fusion(
        diarization=diarization_for_fusion,
        transcription=transcription,
        channel_analysis=channel_analysis,
        metadata=metadata,
    )

    if not fusion_findings and fused.value == raw_estimate.value and fused.confidence == raw_estimate.confidence:
        return raw_estimate, []

    merged_evidence = list(raw_estimate.evidence)
    for line in fused.evidence:
        if line not in merged_evidence:
            merged_evidence.append(line)
    return (
        Estimate(value=fused.value, confidence=fused.confidence, evidence=merged_evidence),
        fusion_findings,
    )



def _coerce_transcription(value: TranscriptionResult | dict[str, object] | None) -> TranscriptionResult:
    if isinstance(value, TranscriptionResult):
        return value
    if isinstance(value, dict):
        return TranscriptionResult.model_validate(value)
    return TranscriptionResult()


def _coerce_diarization(value: DiarizationResult | dict[str, object] | None) -> DiarizationResult:
    if isinstance(value, DiarizationResult):
        return value
    if isinstance(value, dict):
        return DiarizationResult.model_validate(value)
    return DiarizationResult()


def _coerce_evidence_manifest(value: EvidenceManifest | dict[str, object] | None) -> EvidenceManifest | None:
    if isinstance(value, EvidenceManifest):
        return value
    if isinstance(value, dict):
        return EvidenceManifest.model_validate(value)
    return None


def _estimate_speaker_count(segments: list[SpeakerSegment]) -> Estimate:
    speakers = sorted({segment.speaker for segment in segments if _is_real_speaker_label(segment.speaker)})
    if not speakers:
        return Estimate(
            value=None,
            confidence="low",
            evidence=["Diarization unavailable or produced no speaker-labelled segments."],
        )

    counts = Counter(segment.speaker for segment in segments if _is_real_speaker_label(segment.speaker))
    confidence = "high" if len(segments) >= 6 and min(counts.values()) >= 2 else "medium"
    return Estimate(
        value=len(speakers),
        confidence=confidence,
        evidence=[f"Diarization produced {len(speakers)} distinct speaker label(s): {', '.join(speakers)}."],
    )


def _is_real_speaker_label(label: str | None) -> bool:
    if not label:
        return False
    return label.strip().upper() not in {"UNKNOWN", "UNASSIGNED", "NO_SPEAKER"}


def _build_findings(
    metadata: AudioMetadata,
    silence: SilenceSummary,
    volume: VolumeStats,
    channel_analysis: ChannelAnalysis,
) -> list[ForensicFinding]:
    findings: list[ForensicFinding] = []

    for segment in silence.segments:
        if segment.duration_seconds >= LONG_SILENCE_SECONDS:
            findings.append(
                ForensicFinding(
                    finding="Extended silence or missing-audio span detected",
                    severity="medium",
                    confidence="high",
                    evidence=[f"Silence lasted {segment.duration_seconds:.1f}s."],
                    time_range_seconds=(segment.start_seconds, segment.end_seconds),
                    recommended_follow_up="Review the source recording around this time range and confirm whether silence is expected.",
                )
            )

    if volume.clipping_detected:
        findings.append(
            ForensicFinding(
                finding="Potential clipping or hard limiting detected",
                severity="medium",
                confidence="medium",
                evidence=[
                    f"Maximum volume was {volume.max_volume_db} dB; histogram_0db={volume.histogram_0db}."
                ],
                recommended_follow_up="Inspect waveform peaks and consider obtaining the original recorder export.",
            )
        )

    if volume.low_volume_detected:
        findings.append(
            ForensicFinding(
                finding="Unusually low average volume detected",
                severity="low",
                confidence="high",
                evidence=[f"Average volume was {volume.avg_volume_db} dB."],
                recommended_follow_up="Check microphone gain and consider level-normalized review copies.",
            )
        )

    if channel_analysis.duplicated_channels_likely:
        findings.append(
            ForensicFinding(
                finding="Possible duplicated stereo channels",
                severity="low",
                confidence=channel_analysis.confidence,
                evidence=channel_analysis.evidence,
                recommended_follow_up="Compare channels visually and by correlation before assuming two independent microphones.",
            )
        )

    if (
        channel_analysis.channel_imbalance_db is not None
        and channel_analysis.channel_imbalance_db >= CHANNEL_IMBALANCE_DB
    ):
        findings.append(
            ForensicFinding(
                finding="Channel imbalance detected",
                severity="low",
                confidence="medium",
                evidence=[f"Channel energy differs by {channel_analysis.channel_imbalance_db:.1f} dB."],
                recommended_follow_up="Inspect left/right channel assignments and microphone placement.",
            )
        )

    if metadata.audio_streams > 1:
        findings.append(
            ForensicFinding(
                finding="Multiple audio streams detected",
                severity="info",
                confidence="high",
                evidence=[f"Container includes {metadata.audio_streams} audio streams."],
                recommended_follow_up="Analyze each stream independently and preserve stream mapping in exhibits.",
            )
        )

    return findings


def _build_recommendations(findings: list[ForensicFinding], speaker_count: Estimate) -> list[str]:
    recommendations = [
        "Preserve the original source file and review copies separately.",
        "Use findings as triage indicators and confirm material issues with specialist review.",
    ]
    if any("silence" in finding.finding.lower() for finding in findings):
        recommendations.append("Review long silence ranges against the deposition timeline.")
    if speaker_count.value is None:
        recommendations.append("Run diarization before relying on speaker-count estimates.")
    return recommendations


def _build_summary_text(
    findings: list[ForensicFinding],
    speaker_count: Estimate,
    microphone_count: Estimate,
) -> str:
    speaker_phrase = (
        f"{speaker_count.value} estimated speaker(s)"
        if speaker_count.value is not None
        else "unknown speaker count"
    )
    mic_phrase = (
        f"{microphone_count.value} estimated microphone source(s)"
        if microphone_count.value is not None
        else "unknown microphone setup"
    )
    issue_phrase = f"{len(findings)} forensic indicator(s)" if findings else "no major forensic indicators"
    return (
        f"Forensic triage found {issue_phrase}, with {speaker_phrase} and {mic_phrase}. "
        "These are evidence-backed indicators, not courtroom-grade conclusions."
    )
