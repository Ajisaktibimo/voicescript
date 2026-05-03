import json

from voicescript.parsers import (
    parse_ffprobe_metadata,
    parse_silencedetect_output,
    parse_volumedetect_output,
)


def test_parse_ffprobe_metadata_preserves_stream_channel_distinctions():
    payload = {
        "format": {
            "filename": "deposition_001.wav",
            "duration": "12.5",
            "bit_rate": "256000",
            "format_name": "wav",
            "tags": {"encoder": "Lavf forensic fixture"},
        },
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "bit_rate": "256000",
            },
            {"codec_type": "video", "codec_name": "mjpeg"},
        ],
    }

    metadata = parse_ffprobe_metadata(json.dumps(payload))

    assert metadata.file_name == "deposition_001.wav"
    assert metadata.duration_seconds == 12.5
    assert metadata.bitrate == 256000
    assert metadata.sample_rate == 48000
    assert metadata.channels == 2
    assert metadata.audio_streams == 1
    assert metadata.channel_layout == "stereo"
    assert metadata.codec_name == "pcm_s16le"
    assert metadata.container_format == "wav"
    assert metadata.raw["format"]["tags"]["encoder"] == "Lavf forensic fixture"


def test_parse_silencedetect_output_builds_segments_and_ratio():
    output = """
    [silencedetect @ 000001] silence_start: 1.25
    [silencedetect @ 000001] silence_end: 3.75 | silence_duration: 2.5
    [silencedetect @ 000001] silence_start: 8
    [silencedetect @ 000001] silence_end: 10 | silence_duration: 2
    """

    summary = parse_silencedetect_output(output, duration_seconds=20.0)

    assert summary.silence_ratio == 0.225
    assert [(s.start_seconds, s.end_seconds, s.duration_seconds) for s in summary.segments] == [
        (1.25, 3.75, 2.5),
        (8.0, 10.0, 2.0),
    ]


def test_parse_volumedetect_output_flags_near_zero_db_clipping():
    output = """
    [Parsed_volumedetect_0 @ 000001] mean_volume: -31.4 dB
    [Parsed_volumedetect_0 @ 000001] max_volume: -0.1 dB
    [Parsed_volumedetect_0 @ 000001] histogram_0db: 17
    """

    stats = parse_volumedetect_output(output)

    assert stats.avg_volume_db == -31.4
    assert stats.max_volume_db == -0.1
    assert stats.clipping_detected is True
    assert stats.low_volume_detected is True
    assert stats.histogram_0db == 17
