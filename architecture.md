# VoiceScript Architecture

VoiceScript is a local-first forensic audio triage system for legal and deposition recordings. The current implementation is intentionally practical: it can run as a FastAPI service, expose the same analysis layer through MCP, process one file or batches, and produce structured JSON reports that are careful about confidence and limitations.

The system is not trying to pretend that AI can certify authenticity on its own. It measures what can be measured, estimates what must be inferred, and keeps the evidence trail close to every finding.

## Current Implementation

The application is built around these runtime pieces:

- **FastAPI REST API** in `src/voicescript/api.py`
- **FastMCP server** in `mcp_server.py`
- **Core analyzer service** in `src/voicescript/analyzer.py`
- **FFmpeg/ffprobe tooling** in `src/voicescript/ffmpeg_tools.py`
- **Speech providers** for Whisper/Faster-Whisper and Pyannote in `src/voicescript/providers/`
- **LangGraph/OpenAI agent layer** in `src/voicescript/agent.py`
- **Pydantic schemas** in `src/voicescript/schemas/` and compatibility models in `src/voicescript/models.py`
- **Local storage and job store** in `src/voicescript/storage.py`
- **Evaluation harness** in `src/voicescript/evaluation.py`

Storage is local by default under `data/`. The architecture leaves room for S3 or another artifact store later, but the current code writes uploads, reports, spectrograms, and job metadata locally.

## API Surface

The REST API exposes both immediate analysis and job-based workflows.

Public:

- `GET /v1/health`

Protected with `X-API-Key`:

- `POST /v1/analyze`
- `POST /v1/analyze/files`
- `POST /v1/analyze/llm`
- `POST /v1/analyze/llm/files`
- `POST /v1/audio-jobs`
- `POST /v1/batches`
- `GET /v1/jobs/{job_id}`
- `GET /v1/reports`
- `GET /v1/reports/{job_id}`
- `GET /v1/batches/{batch_id}`

The `/v1/analyze` endpoints are convenient for quick local checks because they return the report directly. The `/v1/audio-jobs` and `/v1/batches` endpoints are the better shape for long recordings because they return a `job_id` and let the client poll status and retrieve reports separately.

For Swagger UI, `/v1/analyze/files` and `/v1/analyze/llm/files` use numbered upload fields (`file_1`, `file_2`, etc.) because Swagger can render arrays of binary files as confusing string controls. For API clients, repeated multipart `files` parts are still supported on `/v1/analyze`, `/v1/analyze/llm`, and `/v1/batches`.

## Execution Model

There are three execution modes to understand.

### Inline Analysis

`POST /v1/analyze` and `POST /v1/analyze/llm` run analysis during the request and return the report directly. This is useful for demos, short audio, and manual testing.

The important implementation detail is that heavy synchronous work is offloaded through FastAPI's threadpool. Without that, a Pyannote, Whisper, Demucs, ffmpeg, or LLM call could block the ASGI event loop and make unrelated endpoints, such as `GET /v1/reports`, feel hung. The threadpool is a responsiveness guardrail, not a production scaling plan.

### FastAPI BackgroundTasks

`POST /v1/audio-jobs` and `POST /v1/batches` use FastAPI `BackgroundTasks` when `VOICESCRIPT_INLINE_JOBS=false`. This lets the API return a `job_id` quickly while the local API process continues the work after the response.

This is enough for local development and assessment demos. It is not enough for production legal operations because the task still lives inside the API process. If the process restarts, crashes, or is deployed with limited worker capacity, the job can be lost or starved.

### Celery Extension Path

For production, Celery should own long-running work. The API contract already points in that direction:

1. FastAPI receives the upload.
2. FastAPI persists the source artifact.
3. FastAPI creates a job record.
4. FastAPI enqueues a Celery task and returns `job_id`.
5. Celery workers run FFmpeg, Demucs, Pyannote, Whisper, fusion, evaluation, and optional LLM review.
6. Workers persist reports and artifacts.
7. Clients poll `/v1/jobs/{job_id}`, `/v1/reports/{job_id}`, and `/v1/batches/{batch_id}`.

Celery matters here because legal audio jobs can be long, expensive, and operationally important. A production queue gives retries, process isolation, cancellation, progress tracking, worker scaling, and separate CPU/GPU resource pools. In plain terms: the API should stay calm and responsive while the heavy audio work happens somewhere built for heavy audio work.

## Multiprocessing Reasoning

Threadpool execution keeps the API event loop responsive, but it does not isolate model memory or crashes. Some workloads need process boundaries:

- GPU memory cleanup after diarization or separation
- CPU-heavy evaluation or post-processing
- provider-specific dependency conflicts
- model crashes that should not take down the API
- running separate worker pools for Whisper, Pyannote, Demucs, and LLM review

Celery workers naturally provide that process boundary. A local multiprocessing worker can also be used for experiments, but it should sit behind the same job/report contract so the API does not need to change.

## Audio Analysis Pipeline

The deterministic analysis pipeline produces the baseline forensic report:

1. **Ingest**
   - Store source file.
   - Compute source hash.
   - Preserve paths and provenance.

2. **Metadata**
   - Use `ffprobe` to extract duration, bitrate, sample rate, channels, streams, codec, and container format.

3. **Quality Analysis**
   - Use `ffmpeg` silence detection.
   - Use `ffmpeg` volume detection.
   - Detect low volume and clipping indicators.

4. **Channel and Microphone Indicators**
   - Treat channel count, stream count, speaker count, and microphone count as separate concepts.
   - Estimate microphone setup from channel layout and channel evidence without overstating certainty.

5. **Source Separation**
   - Demucs can isolate vocals for transcription support.
   - The original audio remains the forensic evidence source.

6. **Speech Intelligence**
   - Whisper/Faster-Whisper handles transcription: what was said.
   - Pyannote handles diarization: who spoke when.
   - The alignment layer merges transcript segments and diarization turns.

7. **Forensic Fusion**
   - Generate findings with severity, confidence, evidence, limitations, and recommended follow-up.
   - Add tamper indicators, audio quality issues, evidence trail, confidence scores, and summary text.

8. **Structured Output**
   - Validate report objects with Pydantic.
   - Serialize through `to_jsonable()` for API, MCP, and storage consistency.

## Why Whisper and Pyannote Are Separate

Whisper and Pyannote solve different problems.

Whisper/Faster-Whisper is strong at transcription. It answers: "What words were spoken, and roughly when?"

Pyannote is built for speaker diarization. It answers: "Which speaker was active during this time range?"

Keeping them separate gives the project more control. If diarization is wrong, we can tune or swap Pyannote without changing transcription. If transcription is weak, we can change Whisper model size, device, or provider without rewriting speaker logic. Most importantly for legal work, the report can state confidence and limitations for transcription and diarization independently instead of blending them into one opaque AI output.

## LLM and Agent Layer

The LangGraph/OpenAI layer is used as a reviewer and summarizer, not as the source of forensic truth.

The deterministic report remains the evidence-backed object. The LLM can review the complete report, including:

- evidence trail
- command provenance
- limitations
- diarization confidence
- audio quality issues
- evaluation metrics
- legal workflow markers

The goal is to produce clearer human-readable legal workflow insights while keeping measurable evidence and rule-based findings auditable.

## MCP Surface

The FastMCP server exposes the same core service layer used by the API. This avoids having one behavior through REST and a different behavior through MCP.

Current MCP tools include:

- `get_audio_metadata`
- `detect_silence`
- `detect_clipping`
- `analyze_channels`
- `estimate_speakers`
- `estimate_microphone_setup`
- `detect_forensic_indicators`
- `generate_spectrogram`
- `analyze_audio_file`
- `analyze_audio_batch`
- `analyze_with_agent`
- `evaluate_report`

## Evaluation Harness

The evaluation harness compares reports against human-labeled references. It focuses on the things that matter for legal transcription quality:

- WER
- speaker count accuracy
- speaker attribution error
- diarization confidence checks
- regression fixtures for known audio cases

This is the right place to improve the system over time. Guessing that a parameter is better is not enough; the system needs repeatable measurements.

## Current Feature Set

Implemented features include:

- API-key protected REST endpoints
- public health endpoint with dependency readiness
- inline single-file analysis
- inline multi-file analysis
- agentic LLM analysis
- job-based analysis
- batch job creation and aggregation
- saved report table endpoint
- per-job report retrieval
- upload size and batch count limits
- FFmpeg/ffprobe metadata extraction
- silence detection
- volume and clipping detection
- channel and microphone estimation indicators
- Demucs source separation support
- Whisper/Faster-Whisper transcription provider path
- Pyannote diarization provider path
- ONNX model configuration hooks
- structured forensic report schema
- provenance and evidence trail fields
- tamper/audio-quality indicators
- spectrogram generation
- MCP tool exposure
- evaluation harness
- Swagger-friendly file upload endpoints

## Known Boundaries

The system is forensic triage, not certified forensic authentication.

The current local BackgroundTasks mode is useful but not durable enough for production. Celery or another real queue should be added before using this as a production legal transcription service.

The current local storage layer is practical for development. Production should use durable artifact storage and stronger chain-of-custody controls.

The LLM layer should review and explain report objects, not silently override deterministic forensic findings.
