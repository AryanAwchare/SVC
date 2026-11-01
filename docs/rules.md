# Development Rules, Engineering Standards & Guardrails
## Singing Voice Conversion Website (Seed-VC MVP)

---

### 1. Code Quality & Type Safety

#### 1.1 Python Standards (Backend & Worker)
- **Strict Type Annotations:** Every function, method parameter, and return value must include comprehensive Python type hints (`typing.Optional`, `typing.List`, `typing.Dict`, `typing.Union`, `typing.Callable`, etc.).
- **Modern Python:** Use Python 3.11+ idioms. Use Pydantic v2 schemas for all external data boundaries, request/response models, and environment configuration.
- **Code Style:** Follow PEP 8 style guidelines. Code should be clean, readable, and free of commented-out dead code or extraneous print debugging statements (use the standard `logging` library).
- **Docstrings & Comments:** Every public module, class, and service function must have a concise docstring explaining its purpose, inputs, return value, and potential exceptions raised. Maintain existing comments and docstrings.

#### 1.2 TypeScript Standards (Frontend)
- **Strict TypeScript:** `tsconfig.json` must enforce `"strict": true`, `"noImplicitAny": true`, and `"strictNullChecks": true`. Never use `any` unless dealing with untyped third-party browser events, and even then, wrap in a typed boundary.
- **Component Design:** React components must be modular, single-responsibility, and typed with explicit prop interfaces (`interface ComponentProps { ... }`).
- **State Management:** Keep state close to where it is used. Avoid bloated global stores for state that only affects a single sub-tree.

---

### 2. Dependency Management & Version Pinning
- **Always Pin Exact Versions:** Never leave open-ended versions (e.g., `fastapi>=0.100.0` is forbidden; write `fastapi==0.110.0`).
- **Verified Compatibility:** Pin dependencies only after verifying that they install cleanly and resolve without dependency conflicts on Python 3.11.
- **Platform Compatibility Matrix:** Explicitly document the verified runtime environment:
  - Python: `3.11.x`
  - PyTorch: `2.2.0+cu121` (or latest verified CUDA 12 wheel on Modal)
  - CUDA: `12.1`
  - Node.js: `18.x` or `20.x` LTS
- **Dependency Minimization:** Do not add heavy dependencies unless strictly required. When introducing a new package, justify the trade-off in 2–3 sentences.

---

### 3. Modularity & Interface Abstraction
- **No Direct Vendor Lock-in:** All external services (GPU compute, persistent storage, audio separation, voice conversion) must sit behind an abstract base class / interface.
  - Storage must use `StorageBackend` (`LocalStorageBackend` vs `S3StorageBackend`).
  - GPU execution must use `JobExecutor` (`MockJobExecutor` vs `ModalJobExecutor`).
  - Voice conversion must use `VoiceConverter` (`SeedVCConverter` vs future `RVCConverter`).
- **Zero-Cost Local Dev:** The system must run end-to-end on a student's CPU laptop without requiring an active cloud subscription or GPU hardware, using `USE_MOCK_GPU=true`.

---

### 4. Audio Engineering Guardrails & Math Safety
- **Chunking & Crossfade Math:**
  When splitting vocals into chunks for diffusion inference, always use an equal-power or linear crossfade transition to prevent audible phase clicks or amplitude dips at chunk seams:
  $$w_1(t) = 1 - \frac{t}{L_{\text{fade}}}, \quad w_2(t) = \frac{t}{L_{\text{fade}}} \quad \text{such that} \quad w_1(t) + w_2(t) = 1.0$$
- **Clipping Prevention:** Every final audio summation (vocal + instrumental) must pass through a limiter or peak normalizer. The final output must never exceed $-1.0\text{ dBTP}$ (true peak) to prevent clipping distortion when converted to lossy MP3.
- **Mastering Target:** The final song output must be mastered to $-14.0 \pm 1.0\text{ LUFS}$ integrated loudness (EBU R128 standard) using `pyloudnorm`.
- **Sample Rate Uniformity:** All audio entering the neural pipeline must be strictly resampled to $44,100\text{ Hz}$ to prevent pitch drift, temporal stretching, or sample-rate mismatch artifacts.

---

### 5. Security & System Hardening
- **No Secrets in Git:** Never commit API keys, tokens, or credentials. All secrets are loaded through `.env` files using Pydantic `BaseSettings`. Maintain `.env.example` in sync with all configuration options.
- **Safe Subprocess Execution:** Never invoke `ffmpeg` or system binaries using shell strings (`shell=True` is strictly banned). Always pass arguments as an explicit sanitized list of strings (`subprocess.run(["ffmpeg", "-i", input_path, ...], shell=False)`).
- **File Validation Before Processing:** Never trust file extensions sent from the browser. Verify actual container format and streams using `ffprobe` JSON output before opening files with audio libraries.
- **Rate Limiting & DoS Protection:** Limit conversion job starts to 5 per IP per hour. Enforce maximum file size limits (50 MB for songs, 10 MB for voice samples) and duration caps (6 minutes for songs, 60 seconds for samples).

---

### 6. Error Handling & State Resilience
- **Explicit Exceptions:** Never swallow exceptions with bare `except: pass` or catch-all handlers without logging. Catch specific exception classes (`AudioValidationError`, `GPUWorkerTimeoutError`, `StorageError`).
- **No Stuck Jobs:** Every job must have a guaranteed terminal state (`DONE` or `FAILED`). Wrap pipeline stages in structured `try ... except` blocks that automatically mark the job as `FAILED` in the database with a user-friendly error message if an unhandled failure occurs.
- **No Infinite Retries:** Cloud GPU calls must have a strict timeout (max 10 minutes). Do not implement unbounded retry loops that burn through serverless compute credits.

---

### 7. Testing Requirements
Before advancing past any phase gate, write and run automated tests for:
1. **Audio Validation:** Tests verifying that corrupted audio files, non-audio files, and oversized files are correctly rejected with structured 400 responses.
2. **State Machine Transitions:** Unit tests verifying that job states transition sequentially (`CREATED` $\to$ `UPLOADED` $\to$ `QUEUED` $\to$ `CONVERTING` $\to$ `DONE`).
3. **Crossfade Math:** Numerical tests verifying that chunk overlap windows sum to unity gain ($1.0$) across the fade region.
4. **Mastering & Loudness:** Tests verifying that final mixed audio does not clip and meets the $-14\text{ LUFS}$ target.
5. **API Contracts:** FastAPI test client tests verifying that all endpoints adhere to the schemas defined in `docs/architecture.md`.

---

### 8. Scope Discipline & AI Verification
- **Scope Discipline:** Build only what is defined in the current phase. Do not add unapproved features (e.g., user authentication, RVC training, payments). If an improvement is identified, propose it first.
- **Fact-Checking & Verification:** Never invent model names, Hugging Face checkpoint paths, or CLI flags. Verify parameters against the official repository (`Plachtaa/seed-vc`, `facebookresearch/demucs`, Modal documentation). When a choice is made, state what source was verified.
- **Clear Commit Units:** Work in small, logical increments. Every commit must have a descriptive message stating *what* was changed and *why*.
