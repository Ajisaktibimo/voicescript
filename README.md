# VoiceScript — Forensic Audio Analysis Agent

AI-powered system that analyses court deposition recordings, extracts measurable audio evidence, and produces structured forensic reports with LLM-generated insights.

---

## Architecture

VoiceScript is organised around a small FastAPI control plane and a heavier audio intelligence pipeline. The **REST API** accepts uploads, persists source artifacts, exposes job/report endpoints, and chooses the execution path. Inline requests are supported for local demos and quick checks, but sync model work is pushed into FastAPI's threadpool so report/status endpoints can still respond while Demucs, Pyannote, Whisper, or LLM review is running. FastAPI **BackgroundTasks** are the local async mode for `job_id` workflows. For production legal workloads, the same job contract should be backed by Celery workers so long model runs happen outside the API process.

The **LangGraph agent** decides which ffmpeg/audio tools to invoke, collects their results, and composes a human-readable summary via OpenAI. The **ML pipeline** runs FFmpeg quality analysis, Pyannote 3.1 diarisation, Faster-Whisper transcription, alignment, and a forensic fusion stage with deterministic confidence rules. The **FastMCP server** exposes 12 tools that are directly callable from Claude Desktop or any MCP-compatible client.

> See [architecture.mmd](architecture.mmd) for the full system diagram.

---

## Design Decisions

- **Original audio to Pyannote** — forensic evidence must be unmodified; Demucs output loses room tone and channel cues that diarisation relies on
- **Deterministic rules before LLM** — the fusion layer uses pattern matching and confidence rules to correct overconfident 1-speaker results; LLM summarises, it does not decide
- **Separate Pyannote and Whisper responsibilities** — Pyannote is used for speaker diarisation because it is designed to answer "who spoke when" from acoustic speaker embeddings and turn boundaries. Whisper/Faster-Whisper is used for transcription because it is designed to answer "what was said" with strong ASR accuracy and timestamps. Keeping them separate lets the pipeline tune, evaluate, retry, or replace diarisation and transcription independently, then align the outputs with explicit confidence and limitations instead of hiding speaker attribution inside one opaque model.
- **Evaluation harness over guessing** — WER, speaker count accuracy, and attribution error rate are computed against human-labelled references rather than tuning parameters blind
- **Pydantic + `to_jsonable()` for all outputs** — every report is schema-validated at creation time; `to_jsonable()` in `models.py` handles serialisation consistently across API, MCP, and storage layers
- **Threadpool for API responsiveness** — local inline analysis is convenient when testing a single deposition, but Pyannote, Whisper, Demucs, ffmpeg, and LLM calls are mostly synchronous from FastAPI's point of view. Running that work directly inside an `async` route can block the event loop and make `/v1/reports` feel frozen. The API therefore offloads heavy sync calls to the FastAPI threadpool. This is not a scaling strategy; it is a guardrail so the server can keep breathing while one request is busy.
- **BackgroundTasks for local job mode** — FastAPI `BackgroundTasks` are useful for this project stage because they let `/v1/audio-jobs` and `/v1/batches` return a `job_id` quickly without adding Redis, RabbitMQ, or a worker deployment. They are intentionally modest: the task still belongs to the API process, so if the process dies, the job dies with it. That is acceptable for local forensic triage and assessment demos, not enough for production legal operations.
- **Celery is the production queue** — legal/deposition audio can be long, expensive, and operationally important. A real deployment needs retries, cancellation, worker isolation, per-model resource pools, progress updates, and the ability to run CPU/GPU-heavy jobs without tying up the API process. Celery with Redis or RabbitMQ is the planned extension point: FastAPI should persist the source artifact, enqueue a Celery task, return a `job_id`, and let workers update `/v1/jobs/{job_id}`, `/v1/reports/{job_id}`, and `/v1/batches/{batch_id}` as artifacts and reports are produced.
- **Multiprocessing when model isolation matters** — some audio workloads benefit from true process isolation rather than more threads: CPU-heavy post-processing, GPU memory cleanup, model crashes, or running different model stacks side by side. Celery workers give that process boundary naturally. If running without Celery, Python multiprocessing can be used for isolated local workers, but it should sit behind the same job/report contract so the public API does not change.

---

## Quick Start

This project does not use a `requirements.txt` file. Dependencies are declared in `pyproject.toml`.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
pip install -e .

# Install test dependencies
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[dev]"
```

Configure the service:

```bash
cp .env.example .env
```

Set at least:

```env
VOICESCRIPT_API_KEY=your-key
OPENAI_API_KEY=your-openai-key
VOICESCRIPT_FFMPEG=ffmpeg
VOICESCRIPT_FFPROBE=ffprobe
```

If FFmpeg is bundled locally on Windows, point to the binaries:

```env
VOICESCRIPT_FFMPEG=C:\path\to\ffmpeg.exe
VOICESCRIPT_FFPROBE=C:\path\to\ffprobe.exe
```

Start the API server:

```bash
python api_server.py
```

Check readiness:

```bash
curl http://localhost:8000/v1/health
```

Analyse a file:

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H "X-API-Key: your-key" \
  -F "file=@court_recording.mp3"
```

The API port defaults to 8000; override with `VOICESCRIPT_PORT=9000`.

---

## Report Table

Use `GET /v1/reports` to list saved report JSON files as table-friendly rows. This is useful from `/docs` after running `/v1/analyze`, `/v1/analyze/files`, `/v1/analyze/llm`, or `/v1/analyze/llm/files`.

```bash
curl -H "X-API-Key: your-key" \
  http://localhost:8000/v1/reports
```

Example response:

```json
{
  "reports": [
    {
      "report_id": "10c0f6c1f8bb431ebf795499e3fc3569_llm",
      "file_name": "bad_audio.mp3",
      "date_time": "2026-05-03T06:45:21.123456+00:00",
      "llm_report": "Review the recording for clipped sections and obtain the original source if possible.",
      "is_llm_report": true
    }
  ]
}
```

---

## Batch Processing

For a quick inline batch from Swagger UI, use `POST /v1/analyze/files`. It exposes numbered file picker slots in `/docs` (`file_1`, `file_2`, etc.) so you can upload several recordings without the confusing Swagger `Add string item` control. The response contains per-file reports plus an aggregate insight block and does not require polling a job id.

Swagger/OpenAPI describes uploaded file bytes as `string($binary)` internally. Use **Try it out** and the numbered file controls. For API clients and curl, repeated multipart `files` parts are still supported on `POST /v1/analyze`.

```bash
curl -X POST http://localhost:8000/v1/analyze/files \
  -H "X-API-Key: your-key" \
  -F "file_1=@bad_audio.mp3" \
  -F "file_2=@moonlight-plaza.mp3"
```

For longer deposition sets, use the async batch workflow:

```bash
curl -X POST http://localhost:8000/v1/batches \
  -H "X-API-Key: your-key" \
  -F "files=@bad_audio.mp3" \
  -F "files=@moonlight-plaza.mp3"

curl -H "X-API-Key: your-key" \
  http://localhost:8000/v1/batches/{batch_id}
```

Example aggregate response, abbreviated from local `data/reports` results:

```json
{
  "aggregate": {
    "file_count": 2,
    "completed_count": 2,
    "failed_count": 0,
    "weighted_silence_ratio": 0.02249,
    "issue_rollup": {
      "Potential clipping or hard limiting detected": 1
    },
    "worst_quality_files": ["bad_audio.mp3"]
  },
  "reports": [
    {
      "file_name": "bad_audio.mp3",
      "duration_seconds": 121.107625,
      "estimated_speaker_count": {"value": 6, "confidence": "medium"},
      "audio_quality": {"silence_ratio": 0.10072, "clipping_detected": true}
    },
    {
      "file_name": "moonlight-plaza.mp3",
      "duration_seconds": 854.532,
      "estimated_speaker_count": {"value": 4, "confidence": "medium"},
      "audio_quality": {"silence_ratio": 0.011402, "clipping_detected": false}
    }
  ],
  "failures": []
}
```

---

## LangGraph Agent (Agentic Analysis)

```bash
# Analyse using the agentic workflow (LLM decides which tools to call)
curl -X POST http://localhost:8000/v1/analyze/llm \
  -H "X-API-Key: your-key" \
  -F "file=@court_recording.mp3"
```

Or via MCP tool `analyze_with_agent`.

For multiple files in Swagger UI, use `POST /v1/analyze/llm/files`. It exposes numbered file picker slots in `/docs`. The endpoint returns LLM-reviewed reports plus a lightweight aggregate with completion counts, weighted silence ratio, and issue rollup.

```bash
curl -X POST http://localhost:8000/v1/analyze/llm/files \
  -H "X-API-Key: your-key" \
  -F "file_1=@bad_audio.mp3" \
  -F "file_2=@moonlight-plaza.mp3"
```

### Legal Workflow Quality

The current agent reviews structured tool outputs before producing its LLM-generated report. The next workflow-quality enhancement is to have the agent review the complete deterministic `ForensicReport` object after analysis, including:

- `evidence_trail` and command provenance
- report `limitations` and claim-level wording
- diarization provider confidence and speaker-count evidence
- transcription and diarization evaluation metrics, including WER, speaker-count accuracy, and attribution error
- legal workflow markers such as Q/A structure, objections, speaker roles, attorney/witness labels, and reviewer correction notes

This keeps the deterministic report as the source of truth while using the LLM as a reviewer and legal-workflow summarizer rather than as the forensic decision maker.

---

## MCP Server

```bash
# Start the MCP server
python mcp_server.py
```

### Available Tools

**Metadata & Quality:**
- `get_audio_metadata` — ffprobe metadata (duration, bitrate, sample rate, channels)
- `detect_silence` — silence segments and silence ratio
- `detect_clipping` — hard-limiting / clipping indicators
- `analyze_channels` — channel layout and microphone count estimate

**Speaker Analysis:**
- `estimate_speakers` — speaker count from diarisation
- `estimate_microphone_setup` — microphone count from channel evidence
- `detect_forensic_indicators` — tamper indicators with confidence levels
- `generate_spectrogram` — spectrogram image for visual review

**Full Analysis:**
- `analyze_audio_file` — complete forensic triage in one call
- `analyze_audio_batch` — forensic analysis across multiple files
- `analyze_with_agent` — agentic LLM-driven analysis
- `evaluate_report` — score a report against a human-labelled reference

---

## Example Output

LLM-generated report from `bad_audio.mp3`:

```json
{
  "file_name": "bad_audio.mp3",
  "duration_seconds": 121.107625,
  "audio_quality": {
    "silence_ratio": 0.10072,
    "clipping_detected": true,
    "avg_volume_db": -16.6
  },
  "issues": [
    "Low-bitrate mono MP3 encoding (~47 kbps) may limit audio fidelity.",
    "Clipping was detected, which can distort peaks and reduce intelligibility.",
    "Approximately 10.1% of the file consists of silence.",
    "Multiple silence segments are present, including a 6.448-second gap between 32.976 s and 39.423 s."
  ],
  "recommendations": "Review the recording for clipped sections and, if possible, obtain the original source or a higher-quality export. If the file must be used, apply restoration tools cautiously (de-clipping, noise reduction, and level normalization) and verify the relevant passages manually. The audio appears usable overall, but fidelity limitations and clipping may affect forensic interpretation.",
  "llm_usage": {
    "prompt_tokens": 2572,
    "completion_tokens": 603,
    "total_tokens": 3175
  }
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VOICESCRIPT_API_KEY` | — | Required. API key for REST endpoints |
| `OPENAI_API_KEY` | — | Required for LangGraph agent |
| `VOICESCRIPT_LLM_MODEL` | `gpt-5.4-mini` | OpenAI model for summarisation |
| `VOICESCRIPT_PORT` | `8000` | REST API port |
| `VOICESCRIPT_PYANNOTE_AUTH_TOKEN` | — | HuggingFace token for Pyannote diarisation |
| `VOICESCRIPT_TRANSCRIPTION_PROVIDER` | `disabled` | `local-faster-whisper` to enable |
| `VOICESCRIPT_DIARIZATION_PROVIDER` | `disabled` | `local-pyannote` to enable |
