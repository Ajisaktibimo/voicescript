from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from .analyzer import ForensicAnalyzer, aggregate_reports
from .config import Settings
from .models import ForensicReport, to_jsonable
from .storage import JobStore, LocalStorage


api_key_header = APIKeyHeader(name="X-API-Key", scheme_name="ApiKeyAuth", auto_error=False)
logger = logging.getLogger("uvicorn.error")


def create_app(
    *,
    settings: Settings | None = None,
    analyzer: ForensicAnalyzer | None = None,
    storage: LocalStorage | None = None,
    job_store: JobStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = storage or LocalStorage(settings)
    job_store = job_store or JobStore(settings)
    analyzer = analyzer or ForensicAnalyzer(settings=settings)

    app = FastAPI(title="VoiceScript Forensic Audio API", version="0.1.0")
    app.state.settings = settings
    app.state.storage = storage
    app.state.job_store = job_store
    app.state.analyzer = analyzer

    def require_api_key(api_key: Annotated[str | None, Security(api_key_header)] = None) -> None:
        if not settings.api_key:
            raise HTTPException(status_code=500, detail="API key is not configured")
        if api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        readiness = analyzer.readiness() if hasattr(analyzer, "readiness") else {}
        readiness.setdefault("ffmpeg", {"available": False, "detail": "not reported by analyzer"})
        readiness.setdefault("ffprobe", {"available": False, "detail": "not reported by analyzer"})
        readiness.setdefault("demucs", {"available": False, "detail": "not reported by analyzer"})
        return {
            "service": "voicescript",
            "status": "ok",
            "dependencies": readiness,
            "storage": {"data_dir": str(settings.data_dir), "available": True},
        }

    @app.post("/v1/audio-jobs", dependencies=[Depends(require_api_key)])
    async def create_audio_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict[str, str]:
        job_id = uuid.uuid4().hex
        final_source_path = await _save_upload_with_limit(storage, job_id, file, settings.max_upload_size_bytes)
        job_store.create_job(file.filename or final_source_path.name, final_source_path, job_id=job_id)
        job_store.update_job(job_id, stage="queued")

        if settings.inline_jobs:
            _process_job(job_id, final_source_path, analyzer, storage, job_store)
        else:
            background_tasks.add_task(_process_job, job_id, final_source_path, analyzer, storage, job_store)
        return {"job_id": job_id, "status": job_store.get_job(job_id)["status"]}

    @app.post("/v1/analyze", dependencies=[Depends(require_api_key)])
    async def analyze_now(file: UploadFile = File(...)) -> dict[str, object]:
        analysis_id = uuid.uuid4().hex
        source_path = await _save_upload_with_limit(storage, analysis_id, file, settings.max_upload_size_bytes)
        logger.info("Starting inline analysis %s for %s", analysis_id, file.filename or source_path.name)
        try:
            report = analyzer.analyze_file(source_path)
            storage.write_report(analysis_id, report)
        except Exception as exc:
            logger.exception("Inline analysis %s failed for %s", analysis_id, file.filename or source_path.name)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
        logger.info("Completed inline analysis %s for %s", analysis_id, file.filename or source_path.name)
        return to_jsonable(report)

    @app.post("/v1/batches", dependencies=[Depends(require_api_key)])
    async def create_batch(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
    ) -> dict[str, object]:
        if len(files) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"Batch contains {len(files)} file(s); maximum is {settings.max_batch_files}.",
            )

        job_paths: list[tuple[str, str, Path]] = []
        try:
            for upload in files:
                job_id = uuid.uuid4().hex
                final_path = await _save_upload_with_limit(
                    storage,
                    job_id,
                    upload,
                    settings.max_upload_size_bytes,
                )
                job_paths.append((job_id, upload.filename or final_path.name, final_path))
        except HTTPException:
            for _job_id, _file_name, path in job_paths:
                _remove_upload_path(path)
            raise

        for job_id, file_name, final_path in job_paths:
            job_store.create_job(file_name, final_path, job_id=job_id)

        batch_id = job_store.create_batch([job_id for job_id, _file_name, _path in job_paths])
        for job_id, _file_name, final_path in job_paths:
            if settings.inline_jobs:
                _process_job(job_id, final_path, analyzer, storage, job_store)
            else:
                background_tasks.add_task(_process_job, job_id, final_path, analyzer, storage, job_store)

        jobs = job_store.list_jobs_for_batch(batch_id)
        return {
            "batch_id": batch_id,
            "job_ids": [job_id for job_id, _file_name, _path in job_paths],
            "status": _batch_status(jobs),
        }

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def get_job(job_id: str) -> dict[str, object]:
        job = job_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job["job_id"],
            "batch_id": job["batch_id"],
            "file_name": job["file_name"],
            "status": job["status"],
            "stage": job["stage"],
            "error": job["error"],
            "report_available": bool(job["report_path"]),
        }

    @app.get("/v1/reports/{job_id}", dependencies=[Depends(require_api_key)])
    def get_report(job_id: str) -> dict[str, object]:
        job = job_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job["report_path"]:
            raise HTTPException(status_code=409, detail="Report is not available yet")
        return storage.read_report(job["report_path"])

    @app.get("/v1/batches/{batch_id}", dependencies=[Depends(require_api_key)])
    def get_batch(batch_id: str) -> dict[str, object]:
        batch = job_store.get_batch(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        jobs = job_store.list_jobs_for_batch(batch_id)

        reports: list[ForensicReport] = []
        for job in jobs:
            if job["report_path"]:
                reports.append(ForensicReport.model_validate(storage.read_report(job["report_path"])))

        aggregate = aggregate_reports(
            reports,
            failed_count=sum(1 for job in jobs if job["status"] == "failed"),
        )
        return {
            "batch_id": batch_id,
            "status": _batch_status(jobs),
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "file_name": job["file_name"],
                    "status": job["status"],
                    "stage": job["stage"],
                    "error": job["error"],
                    "report_available": bool(job["report_path"]),
                }
                for job in jobs
            ],
            "aggregate": to_jsonable(aggregate),
        }

    return app


async def _save_upload_with_limit(
    storage: LocalStorage,
    job_id: str,
    file: UploadFile,
    max_upload_size_bytes: int,
) -> Path:
    safe_name = Path(file.filename or "upload.bin").name or "upload.bin"
    target_dir = storage.upload_dir / job_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    bytes_written = 0
    chunk_size = 1024 * 1024

    try:
        with target.open("wb") as handle:
            while chunk := await file.read(chunk_size):
                bytes_written += len(chunk)
                if bytes_written > max_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {max_upload_size_bytes} bytes.",
                    )
                handle.write(chunk)
    except HTTPException:
        _remove_upload_path(target)
        raise
    return target


def _remove_upload_path(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _batch_status(jobs: list[dict[str, object]]) -> str:
    statuses = {job["status"] for job in jobs}
    if any(status == "failed" for status in statuses):
        return "failed" if all(job["status"] == "failed" for job in jobs) else "partial"
    if jobs and all(job["status"] == "completed" for job in jobs):
        return "completed"
    if any(job["status"] == "running" for job in jobs):
        return "running"
    return "queued"


def _process_job(
    job_id: str,
    source_path: Path,
    analyzer: ForensicAnalyzer,
    storage: LocalStorage,
    job_store: JobStore,
) -> None:
    try:
        logger.info("Starting job %s for %s", job_id, source_path.name)
        job_store.update_job(job_id, status="running", stage="analyzing")
        report = analyzer.analyze_file(source_path)
        report_path = storage.write_report(job_id, report)
        job_store.update_job(job_id, status="completed", stage="completed", report_path=report_path)
        logger.info("Completed job %s for %s", job_id, source_path.name)
    except Exception as exc:
        logger.exception("Job %s failed for %s", job_id, source_path.name)
        job_store.update_job(job_id, status="failed", stage="failed", error=str(exc))
