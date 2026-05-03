from __future__ import annotations

import json
import logging
import inspect
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

from .analyzer import ForensicAnalyzer, aggregate_reports
from .agent import ForensicAgent
from .config import Settings
from .models import ForensicReport, to_jsonable
from .storage import JobStore, LocalStorage


api_key_header = APIKeyHeader(name="X-API-Key", scheme_name="ApiKeyAuth", auto_error=False)
logger = logging.getLogger("uvicorn.error")
MULTI_FILE_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "encoding": {
                    "files": {"style": "form", "explode": True},
                }
            }
        }
    }
}


def create_app(
    *,
    settings: Settings | None = None,
    analyzer: ForensicAnalyzer | None = None,
    agent: ForensicAgent | None = None,
    storage: LocalStorage | None = None,
    job_store: JobStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = storage or LocalStorage(settings)
    job_store = job_store or JobStore(settings)
    analyzer = analyzer or ForensicAnalyzer(settings=settings)
    agent = agent or ForensicAgent(settings=settings)

    app = FastAPI(title="VoiceScript Forensic Audio API", version="0.1.0")
    app.state.settings = settings
    app.state.storage = storage
    app.state.job_store = job_store
    app.state.analyzer = analyzer
    app.state.agent = agent

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
        file: UploadFile | None = File(None),
        input_url: str | None = Form(None),
    ) -> dict[str, str]:
        if not file and not input_url:
            raise HTTPException(status_code=400, detail="Either 'file' or 'input_url' must be provided.")

        job_id = uuid.uuid4().hex
        if file:
            final_source_path = await _save_upload_with_limit(storage, job_id, file, settings.max_upload_size_bytes)
            file_name = file.filename or final_source_path.name
        else:
            final_source_path = input_url
            file_name = input_url

        job_store.create_job(file_name, final_source_path, job_id=job_id)
        job_store.update_job(job_id, stage="queued")

        if settings.inline_jobs:
            await run_in_threadpool(_process_job, job_id, final_source_path, analyzer, storage, job_store)
        else:
            background_tasks.add_task(_process_job, job_id, final_source_path, analyzer, storage, job_store)
        return {"job_id": job_id, "status": job_store.get_job(job_id)["status"]}

    @app.post("/v1/analyze", dependencies=[Depends(require_api_key)], openapi_extra=MULTI_FILE_OPENAPI_EXTRA)
    async def analyze_now(
        file: UploadFile | None = File(None),
        files: list[UploadFile] | None = File(None),
        input_url: str | None = Form(None),
    ) -> dict[str, object]:
        uploads = [upload for upload in ([file] if file else []) + (files or []) if upload is not None]
        if not uploads and not input_url:
            raise HTTPException(status_code=400, detail="Either 'file', 'files', or 'input_url' must be provided.")
        if input_url and uploads:
            raise HTTPException(status_code=400, detail="Use either uploaded files or 'input_url', not both.")
        if len(uploads) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"Request contains {len(uploads)} file(s); maximum is {settings.max_batch_files}.",
            )
        if len(uploads) > 1:
            return await _analyze_uploads_now(
                uploads,
                analyzer=analyzer,
                storage=storage,
                max_upload_size_bytes=settings.max_upload_size_bytes,
            )

        analysis_id = uuid.uuid4().hex
        if uploads:
            source_path = await _save_upload_with_limit(storage, analysis_id, uploads[0], settings.max_upload_size_bytes)
            input_identifier = uploads[0].filename or source_path.name
        else:
            source_path = input_url
            input_identifier = input_url

        logger.info("Starting inline analysis %s for %s", analysis_id, input_identifier)
        if uploads:
            logger.info(
                "pipeline run_id=%s stage=upload status=done file=%s path=%s bytes=%s",
                analysis_id,
                input_identifier,
                source_path,
                _file_size(source_path),
            )

        try:
            report = await run_in_threadpool(_analyze_file, analyzer, source_path, run_id=analysis_id)
            logger.info("pipeline run_id=%s stage=report_persist status=start", analysis_id)
            report_path = storage.write_report(analysis_id, report)
            logger.info("pipeline run_id=%s stage=report_persist status=done path=%s", analysis_id, report_path)
        except Exception as exc:
            logger.exception("Inline analysis %s failed for %s", analysis_id, input_identifier)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
        return to_jsonable(report)

    @app.post(
        "/v1/analyze/files",
        dependencies=[Depends(require_api_key)],
        summary="Upload Files With Picker Slots For Inline Analysis",
        description=(
            "Swagger-friendly endpoint for uploading multiple audio files at once. "
            "Use the numbered file picker controls in /docs for an immediate aggregate response."
        ),
    )
    async def analyze_files_now(
        file_1: UploadFile = File(...),
        file_2: UploadFile | None = File(None),
        file_3: UploadFile | None = File(None),
        file_4: UploadFile | None = File(None),
        file_5: UploadFile | None = File(None),
    ) -> dict[str, object]:
        files = _upload_slots(file_1, file_2, file_3, file_4, file_5)
        if len(files) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"Request contains {len(files)} file(s); maximum is {settings.max_batch_files}.",
            )
        return await _analyze_uploads_now(
            files,
            analyzer=analyzer,
            storage=storage,
            max_upload_size_bytes=settings.max_upload_size_bytes,
        )

    @app.post("/v1/analyze/llm", dependencies=[Depends(require_api_key)], openapi_extra=MULTI_FILE_OPENAPI_EXTRA)
    async def analyze_llm(
        file: UploadFile | None = File(None),
        files: list[UploadFile] | None = File(None),
        input_url: str | None = Form(None),
    ) -> dict[str, object]:
        uploads = [upload for upload in ([file] if file else []) + (files or []) if upload is not None]
        if not uploads and not input_url:
            raise HTTPException(status_code=400, detail="Either 'file', 'files', or 'input_url' must be provided.")
        if input_url and uploads:
            raise HTTPException(status_code=400, detail="Use either uploaded files or 'input_url', not both.")
        if len(uploads) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"Request contains {len(uploads)} file(s); maximum is {settings.max_batch_files}.",
            )
        if len(uploads) > 1:
            return await _analyze_uploads_with_agent_now(
                uploads,
                agent=agent,
                storage=storage,
                max_upload_size_bytes=settings.max_upload_size_bytes,
            )

        analysis_id = uuid.uuid4().hex
        if uploads:
            source_path = await _save_upload_with_limit(storage, analysis_id, uploads[0], settings.max_upload_size_bytes)
            input_identifier = uploads[0].filename or source_path.name
        else:
            source_path = input_url
            input_identifier = input_url

        logger.info("Starting agentic analysis %s for %s", analysis_id, input_identifier)
        if uploads:
            logger.info(
                "pipeline run_id=%s stage=upload status=done file=%s path=%s bytes=%s",
                analysis_id,
                input_identifier,
                source_path,
                _file_size(source_path),
            )

        try:
            report = await run_in_threadpool(agent.analyze_file, source_path)
            logger.info("pipeline run_id=%s stage=report_persist status=start", analysis_id)
            report_path = storage.write_report(f"{analysis_id}_llm", report)
            logger.info("pipeline run_id=%s stage=report_persist status=done path=%s", analysis_id, report_path)
        except Exception as exc:
            logger.exception("Agentic analysis %s failed for %s", analysis_id, input_identifier)
            raise HTTPException(status_code=500, detail=f"LLM Analysis failed: {exc}") from exc
        return to_jsonable(report)

    @app.post(
        "/v1/analyze/llm/files",
        dependencies=[Depends(require_api_key)],
        summary="Upload Files With Picker Slots For Agentic Analysis",
        description=(
            "Swagger-friendly endpoint for uploading multiple audio files to the LangGraph/LLM agent. "
            "Use the numbered file picker controls in /docs for LLM-reviewed reports plus an aggregate."
        ),
    )
    async def analyze_llm_files_now(
        file_1: UploadFile = File(...),
        file_2: UploadFile | None = File(None),
        file_3: UploadFile | None = File(None),
        file_4: UploadFile | None = File(None),
        file_5: UploadFile | None = File(None),
    ) -> dict[str, object]:
        files = _upload_slots(file_1, file_2, file_3, file_4, file_5)
        if len(files) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"Request contains {len(files)} file(s); maximum is {settings.max_batch_files}.",
            )
        return await _analyze_uploads_with_agent_now(
            files,
            agent=agent,
            storage=storage,
            max_upload_size_bytes=settings.max_upload_size_bytes,
        )

    @app.post("/v1/batches", dependencies=[Depends(require_api_key)], openapi_extra=MULTI_FILE_OPENAPI_EXTRA)
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
                await run_in_threadpool(_process_job, job_id, final_path, analyzer, storage, job_store)
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

    @app.get(
        "/v1/reports",
        dependencies=[Depends(require_api_key)],
        summary="List Saved Report Table",
        description=(
            "Return saved reports as table-friendly JSON rows with file name, report timestamp, "
            "and the LLM/human-readable report text when available."
        ),
    )
    def list_reports() -> dict[str, object]:
        return {"reports": _report_table_rows(storage)}

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


async def _analyze_uploads_now(
    uploads: list[UploadFile],
    *,
    analyzer: ForensicAnalyzer,
    storage: LocalStorage,
    max_upload_size_bytes: int,
) -> dict[str, object]:
    reports: list[ForensicReport] = []
    failures: list[dict[str, str]] = []
    for upload in uploads:
        analysis_id = uuid.uuid4().hex
        source_path: Path | None = None
        input_identifier = upload.filename or "upload.bin"
        try:
            source_path = await _save_upload_with_limit(storage, analysis_id, upload, max_upload_size_bytes)
            logger.info("Starting inline analysis %s for %s", analysis_id, input_identifier)
            logger.info(
                "pipeline run_id=%s stage=upload status=done file=%s path=%s bytes=%s",
                analysis_id,
                input_identifier,
                source_path,
                _file_size(source_path),
            )
            report = await run_in_threadpool(_analyze_file, analyzer, source_path, run_id=analysis_id)
            logger.info("pipeline run_id=%s stage=report_persist status=start", analysis_id)
            report_path = storage.write_report(analysis_id, report)
            logger.info("pipeline run_id=%s stage=report_persist status=done path=%s", analysis_id, report_path)
            reports.append(report)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Inline analysis %s failed for %s", analysis_id, input_identifier)
            failures.append(
                {
                    "file_name": input_identifier,
                    "path": str(source_path) if source_path else "",
                    "error": str(exc),
                }
            )

    return {
        "aggregate": to_jsonable(aggregate_reports(reports, failed_count=len(failures))),
        "reports": to_jsonable(reports),
        "failures": failures,
    }


async def _analyze_uploads_with_agent_now(
    uploads: list[UploadFile],
    *,
    agent: ForensicAgent,
    storage: LocalStorage,
    max_upload_size_bytes: int,
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for upload in uploads:
        analysis_id = uuid.uuid4().hex
        source_path: Path | None = None
        input_identifier = upload.filename or "upload.bin"
        try:
            source_path = await _save_upload_with_limit(storage, analysis_id, upload, max_upload_size_bytes)
            logger.info("Starting agentic analysis %s for %s", analysis_id, input_identifier)
            logger.info(
                "pipeline run_id=%s stage=upload status=done file=%s path=%s bytes=%s",
                analysis_id,
                input_identifier,
                source_path,
                _file_size(source_path),
            )
            report = await run_in_threadpool(agent.analyze_file, source_path)
            logger.info("pipeline run_id=%s stage=report_persist status=start", analysis_id)
            report_path = storage.write_report(f"{analysis_id}_llm", report)
            logger.info("pipeline run_id=%s stage=report_persist status=done path=%s", analysis_id, report_path)
            reports.append(dict(report))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Agentic analysis %s failed for %s", analysis_id, input_identifier)
            failures.append(
                {
                    "file_name": input_identifier,
                    "path": str(source_path) if source_path else "",
                    "error": str(exc),
                }
            )

    return {
        "aggregate": _aggregate_agent_reports(reports, failed_count=len(failures)),
        "reports": to_jsonable(reports),
        "failures": failures,
    }


def _aggregate_agent_reports(reports: list[dict[str, object]], failed_count: int) -> dict[str, object]:
    total_duration = sum(_float_value(report.get("duration_seconds")) for report in reports)
    if total_duration:
        weighted_silence = sum(
            _float_value(report.get("duration_seconds"))
            * _float_value(_audio_quality(report).get("silence_ratio"))
            for report in reports
        ) / total_duration
    else:
        weighted_silence = 0.0

    issue_rollup: dict[str, int] = {}
    for report in reports:
        for issue in report.get("issues") or []:
            if isinstance(issue, str):
                issue_rollup[issue] = issue_rollup.get(issue, 0) + 1

    return {
        "file_count": len(reports) + failed_count,
        "completed_count": len(reports),
        "failed_count": failed_count,
        "weighted_silence_ratio": round(weighted_silence, 6),
        "issue_rollup": issue_rollup,
    }


def _audio_quality(report: dict[str, object]) -> dict[str, object]:
    value = report.get("audio_quality")
    return value if isinstance(value, dict) else {}


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _upload_slots(*files: UploadFile | None) -> list[UploadFile]:
    return [file for file in files if file is not None]


def _report_table_rows(storage: LocalStorage) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    paths = sorted(
        storage.report_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            report = storage.read_report(path)
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable report file %s", path)
            continue

        report_id = path.stem
        rows.append(
            {
                "report_id": report_id,
                "file_name": _string_value(report.get("file_name"), fallback=path.name),
                "date_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "llm_report": _llm_report_text(report),
                "is_llm_report": _is_llm_report(report_id, report),
            }
        )
    return rows


def _llm_report_text(report: dict[str, object]) -> str:
    recommendations = report.get("recommendations")
    if isinstance(recommendations, str):
        return recommendations
    if isinstance(recommendations, list):
        return " ".join(str(item) for item in recommendations if item)
    return _string_value(report.get("summary_text"))


def _is_llm_report(report_id: str, report: dict[str, object]) -> bool:
    return report_id.endswith("_llm") or "llm_usage" in report


def _string_value(value: object, *, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _remove_upload_path(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _file_size(path: Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _analyze_file(analyzer: ForensicAnalyzer, source_path: Path, *, run_id: str) -> ForensicReport:
    try:
        parameters = inspect.signature(analyzer.analyze_file).parameters
    except (TypeError, ValueError):
        return analyzer.analyze_file(source_path)
    if "run_id" in parameters:
        return analyzer.analyze_file(source_path, run_id=run_id)
    return analyzer.analyze_file(source_path)


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
        logger.info(
            "pipeline run_id=%s stage=upload status=done file=%s path=%s bytes=%s",
            job_id,
            source_path.name,
            source_path,
            _file_size(source_path),
        )
        job_store.update_job(job_id, status="running", stage="analyzing")
        report = _analyze_file(analyzer, source_path, run_id=job_id)
        logger.info("pipeline run_id=%s stage=report_persist status=start", job_id)
        report_path = storage.write_report(job_id, report)
        logger.info("pipeline run_id=%s stage=report_persist status=done path=%s", job_id, report_path)
        job_store.update_job(job_id, status="completed", stage="completed", report_path=report_path)
        logger.info("Completed job %s for %s", job_id, source_path.name)
    except Exception as exc:
        logger.exception("Job %s failed for %s", job_id, source_path.name)
        job_store.update_job(job_id, status="failed", stage="failed", error=str(exc))
