# Phased Delivery Plan & Acceptance Gates
## Singing Voice Conversion Website (Seed-VC MVP)

---

### Phase Roadmap Overview

```
Phase 0: Notebook Prototype (Colab/Kaggle) ────▶ [GATE 0: Audio Quality Sign-Off]
                                                            │
Phase 1: Worker Package & Modal GPU CLI   ────▶ [GATE 1: Cloud GPU CLI Verified]
                                                            │
Phase 2: Backend API, JobStore & Storage  ────▶ [GATE 2: API & Storage Tests Pass]
                                                            │
Phase 3: Frontend Web UI & A/B Player     ────▶ [GATE 3: End-to-End Demo Ready]
                                                            │
Phase 4: Hardening, Eval & Audio Polish   ────▶ [GATE 4: All Section 12 Criteria Met]
                                                            │
Phase 5: Production Deployment & Docs     ────▶ [GATE 5: Live App & Video Demo]
                                                            │
Phase 6: [ON HOLD] RVC High-Quality Engine ───▶ (Deferred until explicit instruction)
```

---

### Phase 0: Prototype Notebook (Colab / Kaggle GPU)

**Objective:** Validate the audio quality of Seed-VC singing voice conversion coupled with vocal stem separation on a single GPU notebook before writing production infrastructure. Tune hyperparameter defaults.

#### Deliverables & Tasks
- [ ] Create `notebooks/phase0_prototype.ipynb`.
- [ ] Step 1: Install Demucs (`htdemucs`) and Seed-VC singing dependencies on a Colab/Kaggle free T4 GPU instance.
- [ ] Step 2: Download the singing checkpoint `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth`, CAMPPlus, and RMVPE.
- [ ] Step 3: Run separation on a sample track (3–4 minutes). Verify separation bleed and isolation quality.
- [ ] Step 4: Run Seed-VC F0-conditioned conversion using a 15–20s user voice reference.
- [ ] Step 5: Test hyperparameter sweeps:
  - Diffusion steps: 25 vs 35 vs 50.
  - CFG scale: 0.5 vs 0.7 vs 0.9.
  - Auto-F0 adjustment: On vs Off.
  - Semitone shifts: -12, 0, +12.
- [ ] Step 6: Test audio chunking (15s chunks with 2s linear crossfades) to prevent GPU memory spikes.
- [ ] Step 7: Sum vocal stem + instrumental, apply limiter, and measure integrated loudness with `pyloudnorm`.
- [ ] Step 8: Document timings per stage and qualitative observations in the notebook.

#### Acceptance Gate 0
> [!IMPORTANT]
> **GATE 0 EXIT CRITERIA:**
> - [ ] Converted singing voice distinctly matches the user's vocal timbre.
> - [ ] Melody, pitch accuracy, rhythm, and lyrical intelligibility are intact.
> - [ ] Summed mix has no audible clicks, distortion, or clipping.
> - [ ] User listens to resulting `.wav` output and explicitly grants: **"Approved Audio Quality"**.

---

### Phase 1: Worker Package & Modal Deployment

**Objective:** Package the prototype pipeline into a modular, production-ready Python package (`worker/`), define the `VoiceConverter` abstraction, and deploy the serverless container to Modal.

#### Deliverables & Tasks
- [ ] Define abstract base class `worker/engines/base.py` (`VoiceConverter` interface).
- [ ] Implement `worker/engines/seed_vc.py` adhering to `VoiceConverter`.
- [ ] Create `worker/engines/rvc_stub.py` (explicitly raises 400 "on hold" error).
- [ ] Implement `worker/engines/registry.py` with engine factory.
- [ ] Implement audio processing modules:
  - `worker/pipeline/separation.py` (Demucs stem separation).
  - `worker/pipeline/chunking.py` (Overlap-add and equal-power crossfading).
  - `worker/pipeline/postprocess.py` (De-essing, high-pass filter, subtle reverb).
  - `worker/pipeline/mastering.py` (Limiting and EBU-R128 -14 LUFS normalization).
  - `worker/pipeline/runner.py` (Full 7-stage coordinator).
- [ ] Write `worker/cache_weights.py` to populate Modal persistent volume `seed-vc-weights`.
- [ ] Implement `worker/gpu_app.py` with Modal serverless entry points and volume mounting.
- [ ] Write CLI runner `scripts/run_cli.py` to trigger an end-to-end cloud GPU conversion directly from the terminal.

#### Acceptance Gate 1
> [!IMPORTANT]
> **GATE 1 EXIT CRITERIA:**
> - [ ] Running `python scripts/run_cli.py --song test.mp3 --sample voice.wav` dispatches to Modal GPU and returns a mastered `final_mix.mp3`.
> - [ ] Worker container spins down to zero after 60 seconds of idle time.
> - [ ] Total run duration on cloud GPU logged and verified under 3.5 minutes for a 3.5-minute track.

---

### Phase 2: Backend API, JobStore & Storage Layer

**Objective:** Build the robust, asynchronous FastAPI backend orchestrator with local mock mode, job queue lifecycle, storage abstraction, and automatic 24-hour cleanup.

#### Deliverables & Tasks
- [ ] Setup FastAPI app in `backend/app/main.py` with CORS, exception handlers, and security headers.
- [ ] Create Pydantic v2 schemas in `backend/app/schemas/`:
  - `JobCreate`, `JobStart`, `JobResponse`, `JobStatus`, `ErrorResponse`.
- [ ] Implement `StorageBackend` abstraction in `backend/app/storage/`:
  - `LocalStorageBackend` (disk storage for dev).
  - `S3StorageBackend` (Cloudflare R2 / AWS S3 pre-signed URLs for prod).
- [ ] Implement `JobStore` in `backend/app/jobs/store.py` with SQLite / SQLModel.
- [ ] Implement `USE_MOCK_GPU=true` mode in `backend/app/jobs/service.py`:
  - CPU-only, zero-cost audio pitching simulation for instant local development.
- [ ] Connect `USE_MOCK_GPU=false` mode to call deployed Modal worker function asynchronously.
- [ ] Implement background 24-hour TTL auto-deletion task in `backend/app/jobs/cleanup.py`.
- [ ] Implement REST endpoints:
  - `POST /api/jobs`, `POST /api/jobs/{id}/upload/{type}`, `POST /api/jobs/{id}/start`
  - `GET /api/jobs/{id}`, `GET /api/jobs/{id}/download/{kind}`, `DELETE /api/jobs/{id}`, `GET /api/health`
- [ ] Write PyTest test suite in `backend/tests/` covering validation, state transitions, and API routes.

#### Acceptance Gate 2
> [!IMPORTANT]
> **GATE 2 EXIT CRITERIA:**
> - [ ] 100% of PyTest test cases pass.
> - [ ] API successfully handles job creation, upload, progress tracking, and file retrieval in mock mode.
> - [ ] API successfully handles full job execution when toggled to Modal GPU worker.
> - [ ] `DELETE /api/jobs/{id}` purges all files from storage and database.

---

### Phase 3: Frontend Web Application & Player

**Objective:** Build the modern Next.js single-page application with in-browser voice recording, real-time stage progress polling, and synchronized A/B comparison audio player.

#### Deliverables & Tasks
- [ ] Initialize Next.js 14 (App Router) with TypeScript and Tailwind CSS in `frontend/`.
- [ ] Implement dark-mode aesthetic theme with responsive layout.
- [ ] Build `AudioUploader.tsx`: drag-and-drop song picker with duration inspection and preview.
- [ ] Build `AudioRecorder.tsx`: in-browser microphone capture using MediaRecorder API, HTML5 Canvas audio level meter, reading prompt, and file upload fallback tab.
- [ ] Build `SettingsModal.tsx`: semitone shift slider (-12 to +12), quality segmented control (Fast/Balanced/Best), and format selection.
- [ ] Build `ConsentCheckbox.tsx`: mandatory legal & ethics confirmation.
- [ ] Build `ProgressCard.tsx`: stage-by-stage timeline with animated progress bar, real-time status message, and estimated ETA countdown.
- [ ] Build `ABPlayer.tsx`: synchronized dual-audio player with single-click hot-swap between original and converted tracks, timeline scrub bar, and duration timer.
- [ ] Build `DownloadButtonGroup.tsx`: download full mix, download vocal stem, and immediate "Delete My Audio" button.
- [ ] Connect frontend to backend API with automatic retry and error handling.

#### Acceptance Gate 3
> [!IMPORTANT]
> **GATE 3 EXIT CRITERIA:**
> - [ ] Complete user flow operates smoothly in local mock mode from the web browser.
> - [ ] In-browser microphone recording works seamlessly with live volume meter feedback.
> - [ ] A/B player hot-swaps between audio sources without desyncing or resetting playback position.
> - [ ] Mobile-responsive layout tested on desktop and mobile viewports.

---

### Phase 4: Hardening, Audio Polish & Evaluation

**Objective:** Eliminate edge-case bugs, tune crossfade math, add safety rate limiting and metadata watermarks, run the evaluation benchmark suite, and generate the project report deliverables.

#### Deliverables & Tasks
- [ ] Audit chunk overlap-add math in `worker/pipeline/chunking.py` for strictly seamless energy preservation ($w_1(t) + w_2(t) = 1$).
- [ ] Implement F0 key suggestion helper: compare median pitch of source vocal vs target sample to suggest optimal semitone shift.
- [ ] Add IP-based rate limiting on conversion starts (max 5/hour per IP).
- [ ] Embed ID3 / Vorbis comment metadata disclosure tag in exported audio: *"Synthesized with Seed-VC (Zero-Shot Singing Voice Conversion) for personal educational use"*.
- [ ] Implement quantitative evaluation runner in `scripts/eval/run_eval.py`:
  - Benchmark 5 diverse songs (different genres, vocal registers, tempos).
  - Record execution time per stage: validation, separation, conversion, mixing.
  - Calculate estimated cloud cost per song (Modal GPU seconds $\times$ hourly rate).
- [ ] Create `scripts/eval/mos_template.csv` for Subjective Mean Opinion Score (MOS) listening surveys (evaluating naturalness, similarity, and intelligibility).

#### Acceptance Gate 4
> [!IMPORTANT]
> **GATE 4 EXIT CRITERIA:**
> - [ ] All Section 12 acceptance criteria met and documented in evaluation report.
> - [ ] Zero audio clicks, pops, or clipping across all benchmark tracks.
> - [ ] Rate limiter blocks abusive requests with clean `429 Too Many Requests`.
> - [ ] Evaluation report and MOS template generated and populated.

---

### Phase 5: Production Deployment & Documentation

**Objective:** Deploy the full-stack system to production hosting, configure Cloudflare R2 / S3 storage, document end-to-end operational procedures, and prepare project demonstration materials.

#### Deliverables & Tasks
- [ ] Deploy Next.js frontend to Vercel or Cloudflare Pages.
- [ ] Deploy FastAPI backend to a serverless/container host (Render, Railway, Fly.io, or VPS).
- [ ] Configure production storage (Cloudflare R2 or AWS S3) with pre-signed upload/download URLs.
- [ ] Verify production Modal GPU webhook integration and environment secrets.
- [ ] Finalize root `README.md` with system architecture diagrams, quickstart steps, and ethical AI usage statement.
- [ ] Draft demo video walkthrough script and project report appendix for the AI/DS academic submission.

#### Acceptance Gate 5
> [!IMPORTANT]
> **GATE 5 EXIT CRITERIA:**
> - [ ] Production website publicly accessible via HTTPS.
> - [ ] Full live conversion test performed over the web using a real song and real voice sample.
> - [ ] Auto-cleanup cron verified on production database and storage bucket.
> - [ ] All documentation updated and complete.

---

### Phase 6: [ON HOLD] RVC High-Quality Engine

**Status:** ON HOLD (Deferred).
**Prerequisite:** Explicit user instruction to begin Phase 6 after Phase 5 is fully delivered.
*Refer to `docs/rvc-future.md` for architectural design and prerequisites.*
