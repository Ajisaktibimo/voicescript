from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .models import ForensicReport, to_jsonable


class LocalStorage:
    def __init__(self, settings: Settings):
        self.root = settings.data_dir
        self.upload_dir = self.root / "uploads"
        self.report_dir = self.root / "reports"
        self.spectrogram_dir = self.root / "spectrograms"
        for directory in (self.root, self.upload_dir, self.report_dir, self.spectrogram_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def save_upload(self, job_id: str, file_name: str, content: bytes) -> Path:
        safe_name = Path(file_name).name or "upload.bin"
        target_dir = self.upload_dir / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / safe_name
        target.write_bytes(content)
        return target

    def write_report(self, job_id: str, report: ForensicReport | dict[str, Any]) -> Path:
        target = self.report_dir / f"{job_id}.json"
        target.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
        return target

    def read_report(self, report_path: str | Path) -> dict[str, Any]:
        return json.loads(Path(report_path).read_text(encoding="utf-8"))


class JobStore:
    def __init__(self, settings: Settings):
        self.db_path = settings.data_dir / "voicescript.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_job(self, file_name: str, source_path: Path, batch_id: str | None = None, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, batch_id, file_name, source_path, status, stage, error, report_path)
                VALUES (?, ?, ?, ?, 'queued', 'queued', NULL, NULL)
                """,
                (job_id, batch_id, file_name, str(source_path)),
            )
        return job_id

    def create_batch(self, job_ids: list[str]) -> str:
        batch_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batches (batch_id, job_ids) VALUES (?, ?)",
                (batch_id, json.dumps(job_ids)),
            )
            for job_id in job_ids:
                conn.execute("UPDATE jobs SET batch_id = ? WHERE job_id = ?", (batch_id, job_id))
        return batch_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        error: str | None = None,
        report_path: Path | None = None,
    ) -> None:
        current = self.get_job(job_id)
        if not current:
            raise KeyError(job_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, error = ?, report_path = ?
                WHERE job_id = ?
                """,
                (
                    status or current["status"],
                    stage or current["stage"],
                    error,
                    str(report_path) if report_path is not None else current.get("report_path"),
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["job_ids"] = json.loads(payload["job_ids"])
        return payload

    def list_jobs_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE batch_id = ? ORDER BY rowid", (batch_id,)).fetchall()
        return [dict(row) for row in rows]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    batch_id TEXT,
                    file_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT,
                    report_path TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    job_ids TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
