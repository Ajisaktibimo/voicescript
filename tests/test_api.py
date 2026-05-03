from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient
import pytest

from voicescript.api import create_app
from voicescript.config import Settings
from voicescript.models import (
    AudioMetadata,
    ChannelAnalysis,
    SilenceSummary,
    VolumeStats,
)
from voicescript.analysis import build_report_from_measurements


@pytest.fixture()
def runtime_dir():
    path = Path("test-runtime") / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class FakeAnalyzer:
    def analyze_file(self, path: Path):
        return build_report_from_measurements(
            file_name=path.name,
            sha256="feedface",
            metadata=AudioMetadata(
                file_name=path.name,
                duration_seconds=2.0,
                bitrate=128000,
                sample_rate=16000,
                channels=1,
                audio_streams=1,
                channel_layout="mono",
                codec_name="pcm_s16le",
                container_format="wav",
                raw={},
            ),
            silence=SilenceSummary(segments=[], silence_ratio=0.0),
            volume=VolumeStats(
                avg_volume_db=-18.0,
                max_volume_db=-5.0,
                clipping_detected=False,
                low_volume_detected=False,
                histogram_0db=0,
            ),
            channel_analysis=ChannelAnalysis(
                measured_channels=1,
                audio_streams=1,
                channel_layout="mono",
                duplicated_channels_likely=False,
                channel_imbalance_db=None,
                estimated_microphone_count=1,
                confidence="medium",
                evidence=["Mono stream has one measured channel"],
            ),
            speaker_segments=[],
            transcript_text="",
            provenance=[],
        )


def make_client(runtime_dir):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    return TestClient(create_app(settings=settings, analyzer=FakeAnalyzer()))


@pytest.fixture()
def client(runtime_dir):
    test_client = make_client(runtime_dir)
    yield test_client
    test_client.close()


def test_health_is_public_and_reports_dependency_readiness(client):

    response = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "voicescript"
    assert "ffmpeg" in payload["dependencies"]
    assert "ffprobe" in payload["dependencies"]
    assert "demucs" in payload["dependencies"]


def test_api_requires_key_for_job_creation(client):
    response = client.post(
        "/v1/audio-jobs",
        files={"file": ("sample.wav", b"audio-bytes", "audio/wav")},
    )

    assert response.status_code == 401


def test_openapi_docs_include_x_api_key_security_scheme(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    security_scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
    assert security_scheme == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert {"ApiKeyAuth": []} in schema["paths"]["/v1/audio-jobs"]["post"]["security"]
    assert "security" not in schema["paths"]["/v1/health"]["get"]


def test_analyze_endpoint_returns_report_without_polling_job_id(client):
    response = client.post(
        "/v1/analyze",
        headers={"X-API-Key": "secret"},
        files={"file": ("instant.wav", b"audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_name"] == "instant.wav"
    assert payload["sha256"] == "feedface"
    assert "job_id" not in payload


def test_analyze_endpoint_requires_api_key(client):
    response = client.post(
        "/v1/analyze",
        files={"file": ("instant.wav", b"audio-bytes", "audio/wav")},
    )

    assert response.status_code == 401


def test_upload_job_runs_and_report_can_be_retrieved(client):
    create_response = client.post(
        "/v1/audio-jobs",
        headers={"X-API-Key": "secret"},
        files={"file": ("sample.wav", b"audio-bytes", "audio/wav")},
    )

    assert create_response.status_code == 200
    job_id = create_response.json()["job_id"]

    status_response = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "secret"})
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    report_response = client.get(f"/v1/reports/{job_id}", headers={"X-API-Key": "secret"})
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["file_name"] == "sample.wav"
    assert report["estimated_microphone_count"]["value"] == 1
    assert report["forensic_profile"]["classification"] == "forensic_triage"


def test_batch_job_aggregates_completed_reports(client):
    response = client.post(
        "/v1/batches",
        headers={"X-API-Key": "secret"},
        files=[
            ("files", ("a.wav", b"a", "audio/wav")),
            ("files", ("b.wav", b"b", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    batch_id = response.json()["batch_id"]

    batch_response = client.get(f"/v1/batches/{batch_id}", headers={"X-API-Key": "secret"})
    assert batch_response.status_code == 200
    payload = batch_response.json()
    assert payload["status"] == "completed"
    assert payload["aggregate"]["file_count"] == 2
    assert payload["aggregate"]["completed_count"] == 2
    assert payload["aggregate"]["weighted_silence_ratio"] == 0.0


@pytest.mark.parametrize(
    ("endpoint", "files"),
    [
        ("/v1/audio-jobs", {"file": ("too-big.wav", b"12345", "audio/wav")}),
        ("/v1/analyze", {"file": ("too-big.wav", b"12345", "audio/wav")}),
        ("/v1/batches", [("files", ("too-big.wav", b"12345", "audio/wav"))]),
    ],
)
def test_upload_rejects_file_over_configured_limit(runtime_dir, endpoint, files):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
        max_upload_size_bytes=4,
    )
    test_client = TestClient(create_app(settings=settings, analyzer=FakeAnalyzer()))

    response = test_client.post(endpoint, headers={"X-API-Key": "secret"}, files=files)

    assert response.status_code == 413
    assert not list((runtime_dir / "data" / "uploads").glob("*/*"))
    test_client.close()


def test_batch_job_returns_partial_when_one_file_fails(runtime_dir):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    test_client = TestClient(create_app(settings=settings, analyzer=SometimesFailingAnalyzer()))

    response = test_client.post(
        "/v1/batches",
        headers={"X-API-Key": "secret"},
        files=[
            ("files", ("fail.wav", b"bad", "audio/wav")),
            ("files", ("ok.wav", b"ok", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    batch_id = response.json()["batch_id"]
    assert response.json()["status"] == "partial"

    batch_response = test_client.get(f"/v1/batches/{batch_id}", headers={"X-API-Key": "secret"})
    payload = batch_response.json()
    assert payload["status"] == "partial"
    assert payload["aggregate"]["completed_count"] == 1
    assert payload["aggregate"]["failed_count"] == 1
    statuses = {job["file_name"]: job["status"] for job in payload["jobs"]}
    assert statuses == {"fail.wav": "failed", "ok.wav": "completed"}
    errors = {job["file_name"]: job["error"] for job in payload["jobs"]}
    assert errors["fail.wav"] == "decode failed"
    test_client.close()


def test_analyze_endpoint_returns_controlled_error_when_analysis_fails(runtime_dir):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    test_client = TestClient(create_app(settings=settings, analyzer=AlwaysFailingAnalyzer()))

    response = test_client.post(
        "/v1/analyze",
        headers={"X-API-Key": "secret"},
        files={"file": ("fail.wav", b"bad", "audio/wav")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Analysis failed: decode failed"
    test_client.close()


class SometimesFailingAnalyzer(FakeAnalyzer):
    def analyze_file(self, path: Path):
        if path.name == "fail.wav":
            raise RuntimeError("decode failed")
        return super().analyze_file(path)


class AlwaysFailingAnalyzer(FakeAnalyzer):
    def analyze_file(self, path: Path):
        raise RuntimeError("decode failed")
