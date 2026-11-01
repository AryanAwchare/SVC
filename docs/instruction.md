# Setup & Execution Instructions
## Singing Voice Conversion Website (Seed-VC MVP)

This document provides exact, reproducible setup and run commands for running both the frontend and backend locally on a CPU-only laptop, configuring serverless cloud GPU execution with Modal, and troubleshooting common developer friction points.

---

### 1. Prerequisites

Before starting, ensure your local development system has:
1. **Python 3.11+** installed (`python --version` should output 3.11.x or 3.12.x).
2. **Node.js 18+ & npm** (or `pnpm`) installed (`node --version` $\ge$ v18.17.0).
3. **FFmpeg** installed and accessible on your system PATH (`ffmpeg -version`).
   - *Windows (via winget):* `winget install Gyan.FFmpeg`
   - *Windows (via Chocolatey):* `choco install ffmpeg`
   - *Mac (via Homebrew):* `brew install ffmpeg`
   - *Linux (Ubuntu/Debian):* `sudo apt update && sudo apt install -y ffmpeg`
4. **Git** installed.

---

### 2. Environment Configuration (`.env`)

In the project root, create a `.env` file copied from `.env.example`:

```bash
cp .env.example .env
```

#### `.env` Key Reference

```ini
# Environment Mode
ENVIRONMENT=development
DEBUG=true

# Server Configuration
API_HOST=127.0.0.1
API_PORT=8000
CLIENT_ORIGIN=http://localhost:3000

# GPU Execution Mode: set to 'true' for free local CPU testing!
# When 'true', simulated pitch-shift audio is returned without touching cloud GPU.
USE_MOCK_GPU=true

# Modal Cloud GPU Configuration (Required only when USE_MOCK_GPU=false)
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
MODAL_APP_NAME=vocal-morph-gpu

# Storage Configuration
# 'local' writes to ./backend/storage_data; 's3' uses Cloudflare R2 or AWS S3
STORAGE_BACKEND=local
LOCAL_STORAGE_DIR=./backend/storage_data
S3_BUCKET_NAME=
S3_ENDPOINT_URL=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_REGION=auto

# Security & Constraints
MAX_SONG_DURATION_SEC=360
MAX_SAMPLE_DURATION_SEC=60
JOB_RETENTION_HOURS=24
RATE_LIMIT_PER_HOUR=10
```

---

### 3. Local Development (CPU Only - Mock GPU Mode)

You can build, test, and run the entire web application and API on your laptop with **zero GPU requirements and zero cloud costs** using `USE_MOCK_GPU=true`. In this mode, the worker executes a lightweight CPU audio simulation that separates and gently pitch-shifts the original vocal so you can verify the entire UI, audio recorder, polling states, and A/B comparison player.

#### 3.1 Step A: Setup & Run Backend (FastAPI)

1. Open a terminal in the project root:
   ```bash
   cd backend
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```
3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Start the FastAPI backend server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
5. Verify health:
   Open `http://localhost:8000/api/health` in your browser. You should see:
   ```json
   {
     "status": "healthy",
     "gpu_mode": "mock",
     "storage_backend": "local"
   }
   ```
   Interactive OpenAPI docs are at `http://localhost:8000/docs`.

---

#### 3.2 Step B: Setup & Run Frontend (Next.js)

1. Open a second terminal in the project root:
   ```bash
   cd frontend
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` in your web browser.

---

#### 3.3 Step C: Verification Walkthrough (Mock Mode)

1. On `http://localhost:3000`, drag and drop any `.mp3` or `.wav` song.
2. Grant browser microphone permission and record a 10-second vocal sample (or upload a short voice clip).
3. Check the consent box.
4. Click **Convert My Song**.
5. Watch the real-time stage progression:
   - `Validating Audio` $\to$ `Separating Vocals` $\to$ `Converting Voice` $\to$ `Mixing` $\to$ `DONE`.
6. Play the track in the A/B comparison player, toggle between Original and Converted, and download both output files.
7. Click **Delete Data** and verify that local files in `backend/storage_data` are wiped cleanly.

---

### 4. Running the GPU Backend on Google Colab (with ngrok)

For students who do not want to set up paid cloud accounts or manage cloud credentials, you can run the entire backend and GPU inference pipeline inside a free **Google Colab (T4 GPU)** instance, connected to your local Next.js frontend via **ngrok**.

#### 4.1 Step-by-Step Colab Backend Launch
1. Open Google Colab and upload [`notebooks/singing_voice_conversion_colab.ipynb`](../notebooks/singing_voice_conversion_colab.ipynb).
2. Set runtime type to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Get your free ngrok authtoken from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).
4. Paste your token in Cell 7: `NGROK_AUTH_TOKEN = "your_token_here"`.
5. Run all cells (`Runtime -> Run all`).
6. Colab will print your public API endpoint:
   ```
   🚀 PUBLIC API URL: https://xxxx-xx-xx-xx-xx.ngrok-free.app
   ```
7. In your local frontend directory, create or edit `frontend/.env.local`:
   ```ini
   NEXT_PUBLIC_API_URL=https://xxxx-xx-xx-xx-xx.ngrok-free.app
   ```
8. Start your local frontend:
   ```bash
   cd frontend
   npm run dev
   ```
   *Your local website will now stream all conversions directly through your free Google Colab GPU!*

---

### 5. Cloud GPU Setup & Deployment (Modal Option)

When you are ready to run real neural inference using Seed-VC and Demucs, configure Modal.

#### 4.1 Create Modal Account & Authenticate
1. Create a free account at [modal.com](https://modal.com). (Modal provides $30/month in free compute credits, which is more than enough for hundreds of test conversions).
2. Install the Modal CLI into your Python environment:
   ```bash
   pip install modal
   ```
3. Authenticate your machine with your Modal account:
   ```bash
   modal setup
   ```
   *This automatically creates your `~/.modal.toml` credentials.*

---

#### 4.2 Create Model Cache Volume

To avoid re-downloading 3+ GB of neural weights on every serverless container spin-up, we cache weights in a persistent Modal Network Volume:

1. Create the persistent volume:
   ```bash
   modal volume create seed-vc-weights
   ```
2. Run the caching script to pre-download the Seed-VC singing checkpoint, CAMPPlus speaker encoder, RMVPE pitch extractor, and Demucs separation weights:
   ```bash
   modal run worker/cache_weights.py
   ```
   *This populates `/root/models` inside the volume.*

---

#### 4.3 Deploy the GPU Worker Function

Deploy the serverless GPU app to Modal:

```bash
modal deploy worker/gpu_app.py
```

Modal will print your deployed app status and endpoint. The worker uses an A10G or T4 GPU with `container_idle_timeout=60`, meaning the GPU will shut down after 60 seconds of inactivity to incur **$0.00** idle costs.

---

#### 4.4 Switch Backend to Real Cloud GPU

In your root `.env` file, change:
```ini
USE_MOCK_GPU=false
```
Restart the FastAPI server (`uvicorn app.main:app --reload`). All conversion jobs submitted from the web UI will now be processed by your cloud GPU on Modal!

---

### 5. Running End-to-End Tests

#### Backend Automated Test Suite
Run unit tests, audio normalization tests, and API contract tests:
```bash
cd backend
pytest -v
```

#### End-to-End Evaluation Benchmark
Run the evaluation harness on a test song and reference clip:
```bash
python scripts/eval/run_eval.py \
  --song tests/fixtures/test_song.mp3 \
  --sample tests/fixtures/test_voice.wav \
  --output_dir ./eval_results
```
This logs stage-by-stage timings, audio metrics, and estimated GPU cost to `eval_results/eval_report.json`.

---

### 6. Troubleshooting & Common Pitfalls

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`** | FFmpeg is not installed or not in system PATH. | Install FFmpeg via `winget install Gyan.FFmpeg` (Windows) or `brew install ffmpeg` (Mac). Restart your terminal. |
| **CORS error in browser console (`Access-Control-Allow-Origin`)** | Frontend running on a port different from `CLIENT_ORIGIN`. | Ensure `.env` has `CLIENT_ORIGIN=http://localhost:3000` (or match whatever port `npm run dev` outputs). |
| **PowerShell script execution disabled (`Execution of scripts is disabled on this system`)** | Default Windows PowerShell security policy. | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then retry activating `venv`. |
| **CUDA Out of Memory (OOM) on Modal Worker** | Long audio chunk size during diffusion pass. | Ensure chunking is enabled in `worker/pipeline/chunking.py`. Default chunk size is 15 seconds with 2 seconds crossfade. |
| **Modal Cold Start Latency (> 40 seconds)** | First run downloading weights on the fly. | Make sure you ran `modal run worker/cache_weights.py` to persist all checkpoints in the `seed-vc-weights` volume. |
| **Browser Microphone Record Permission Denied** | Browser blocked mic access or localhost not treated as secure. | Click the lock/tune icon in the browser URL bar, set Microphone to "Allow", and reload. (Localhost is always treated as a secure context). |
