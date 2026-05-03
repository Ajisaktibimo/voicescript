from pathlib import Path
import asyncio
import logging
import shutil
import threading
import time
import uuid

from fastapi.testclient import TestClient
import anyio
import httpx
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


class FakeAgent:
    def analyze_file(self, path: Path):
        return {
            "file_name": path.name,
            "duration_seconds": 120,
            "audio_quality": {"silence_ratio": 0.05, "clipping_detected": False, "avg_volume_db": -18},
            "issues": ["Minor hiss"],
            "recommendations": "High quality audio."
        }


def make_client(runtime_dir, analyzer=None, agent=None):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    return TestClient(create_app(
        settings=settings, 
        analyzer=analyzer or FakeAnalyzer(),
        agent=agent or FakeAgent()
    ))


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
    assert "Upload Files With Picker Slots For Inline Analysis" in schema["paths"]["/v1/analyze/files"]["post"]["summary"]
    assert "Upload Files With Picker Slots For Agentic Analysis" in schema["paths"]["/v1/analyze/llm/files"]["post"]["summary"]
    assert "List Saved Report Table" in schema["paths"]["/v1/reports"]["get"]["summary"]
    inline_body_ref = schema["paths"]["/v1/analyze/files"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    inline_schema = schema["components"]["schemas"][inline_body_ref.removeprefix("#/components/schemas/")]
    assert inline_schema["properties"]["file_1"] == {"type": "string", "format": "binary", "title": "File 1"}
    assert inline_schema["properties"]["file_2"] == {
        "anyOf": [{"type": "string", "format": "binary"}, {"type": "null"}],
        "title": "File 2",
    }
    assert "files" not in inline_schema["properties"]
    for path in ("/v1/analyze", "/v1/analyze/llm", "/v1/batches"):
        multipart = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]
        assert multipart["encoding"]["files"] == {"style": "form", "explode": True}


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


def test_analyze_endpoint_accepts_multiple_files_and_returns_aggregate(client):
    response = client.post(
        "/v1/analyze",
        headers={"X-API-Key": "secret"},
        files=[
            ("files", ("first.wav", b"first-audio", "audio/wav")),
            ("files", ("second.wav", b"second-audio", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["file_count"] == 2
    assert payload["aggregate"]["completed_count"] == 2
    assert payload["aggregate"]["failed_count"] == 0
    assert [report["file_name"] for report in payload["reports"]] == ["first.wav", "second.wav"]
    assert payload["failures"] == []
    assert "job_id" not in payload


def test_analyze_files_endpoint_is_docs_friendly_multiple_upload(client):
    response = client.post(
        "/v1/analyze/files",
        headers={"X-API-Key": "secret"},
        files=[
            ("file_1", ("first.wav", b"first-audio", "audio/wav")),
            ("file_2", ("second.wav", b"second-audio", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["file_count"] == 2
    assert [report["file_name"] for report in payload["reports"]] == ["first.wav", "second.wav"]


def test_analyze_llm_endpoint_returns_agentic_report(client):
    response = client.post(
        "/v1/analyze/llm",
        headers={"X-API-Key": "secret"},
        files={"file": ("instant.wav", b"audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_name"] == "instant.wav"
    assert "recommendations" in payload
    assert payload["recommendations"] == "High quality audio."


def test_analyze_llm_endpoint_accepts_multiple_files_and_returns_aggregate(client):
    response = client.post(
        "/v1/analyze/llm",
        headers={"X-API-Key": "secret"},
        files=[
            ("files", ("first.wav", b"first-audio", "audio/wav")),
            ("files", ("second.wav", b"second-audio", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"] == {
        "file_count": 2,
        "completed_count": 2,
        "failed_count": 0,
        "weighted_silence_ratio": 0.05,
        "issue_rollup": {"Minor hiss": 2},
    }
    assert [report["file_name"] for report in payload["reports"]] == ["first.wav", "second.wav"]
    assert payload["failures"] == []


def test_analyze_llm_files_endpoint_is_docs_friendly_multiple_upload(client):
    response = client.post(
        "/v1/analyze/llm/files",
        headers={"X-API-Key": "secret"},
        files=[
            ("file_1", ("first.wav", b"first-audio", "audio/wav")),
            ("file_2", ("second.wav", b"second-audio", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["file_count"] == 2
    assert [report["file_name"] for report in payload["reports"]] == ["first.wav", "second.wav"]


def test_analyze_endpoint_logs_upload_and_report_pipeline_stages(runtime_dir, caplog):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    analyzer = RunIdRecordingAnalyzer()
    test_client = TestClient(create_app(settings=settings, analyzer=analyzer))

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = test_client.post(
            "/v1/analyze",
            headers={"X-API-Key": "secret"},
            files={"file": ("instant.wav", b"audio-bytes", "audio/wav")},
        )

    assert response.status_code == 200
    assert analyzer.run_ids and analyzer.run_ids[0]
    run_id = analyzer.run_ids[0]
    messages = [record.getMessage() for record in caplog.records]
    assert any(f"pipeline run_id={run_id} stage=upload status=done" in message for message in messages)
    assert any(f"pipeline run_id={run_id} stage=report_persist status=done" in message for message in messages)
    test_client.close()


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


def test_reports_endpoint_returns_table_rows_for_saved_reports(client):
    deterministic_response = client.post(
        "/v1/analyze",
        headers={"X-API-Key": "secret"},
        files={"file": ("deterministic.wav", b"audio-bytes", "audio/wav")},
    )
    llm_response = client.post(
        "/v1/analyze/llm",
        headers={"X-API-Key": "secret"},
        files={"file": ("agentic.wav", b"audio-bytes", "audio/wav")},
    )

    response = client.get("/v1/reports", headers={"X-API-Key": "secret"})

    assert deterministic_response.status_code == 200
    assert llm_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    rows_by_name = {row["file_name"]: row for row in payload["reports"]}
    assert set(rows_by_name) == {"deterministic.wav", "agentic.wav"}
    assert rows_by_name["agentic.wav"]["llm_report"] == "High quality audio."
    assert rows_by_name["agentic.wav"]["is_llm_report"] is True
    assert rows_by_name["deterministic.wav"]["is_llm_report"] is False
    assert rows_by_name["agentic.wav"]["date_time"]
    assert rows_by_name["agentic.wav"]["report_id"].endswith("_llm")


def test_report_table_responds_while_inline_analysis_is_running(runtime_dir):
    started = threading.Event()
    analyzer = SlowAnalyzer(started, delay_seconds=0.6)
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    app = create_app(settings=settings, analyzer=analyzer)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            async def post_analyze():
                response = await async_client.post(
                    "/v1/analyze",
                    headers={"X-API-Key": "secret"},
                    files={"file": ("slow.wav", b"audio-bytes", "audio/wav")},
                )
                assert response.status_code == 200

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(post_analyze)
                assert await anyio.to_thread.run_sync(started.wait, 1)

                began_at = time.perf_counter()
                response = await async_client.get("/v1/reports", headers={"X-API-Key": "secret"})
                elapsed = time.perf_counter() - began_at

                assert response.status_code == 200
                assert elapsed < 0.3

    anyio.run(scenario)


def test_inline_analysis_offloads_model_work_from_event_loop(runtime_dir):
    settings = Settings(
        data_dir=runtime_dir / "data",
        api_key="secret",
        inline_jobs=True,
    )
    app = create_app(settings=settings, analyzer=EventLoopGuardAnalyzer(), agent=EventLoopGuardAgent())

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            analyze_response = await async_client.post(
                "/v1/analyze",
                headers={"X-API-Key": "secret"},
                files={"file": ("inline.wav", b"audio-bytes", "audio/wav")},
            )
            llm_response = await async_client.post(
                "/v1/analyze/llm",
                headers={"X-API-Key": "secret"},
                files={"file": ("agent.wav", b"audio-bytes", "audio/wav")},
            )

        assert analyze_response.status_code == 200
        assert llm_response.status_code == 200

    anyio.run(scenario)


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


class SlowAnalyzer(FakeAnalyzer):
    def __init__(self, started: threading.Event, *, delay_seconds: float):
        self.started = started
        self.delay_seconds = delay_seconds

    def analyze_file(self, path: Path):
        self.started.set()
        time.sleep(self.delay_seconds)
        return super().analyze_file(path)


class EventLoopGuardAnalyzer(FakeAnalyzer):
    def analyze_file(self, path: Path):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return super().analyze_file(path)
        raise RuntimeError("analyzer ran on the request event loop")


class EventLoopGuardAgent(FakeAgent):
    def analyze_file(self, path: Path):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return super().analyze_file(path)
        raise RuntimeError("agent ran on the request event loop")


class RunIdRecordingAnalyzer(FakeAnalyzer):
    def __init__(self):
        self.run_ids: list[str] = []

    def analyze_file(self, path: Path, *, run_id: str | None = None):
        self.run_ids.append(run_id or "")
        return super().analyze_file(path)
