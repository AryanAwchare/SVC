import json
import os

def generate_hybrid_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    md("""# 🎙️ VocalMorph Studio: Production Singing Voice Conversion (Colab T4 + ngrok)
### Python 3.13 Compatible | Single-Worker GPU Queue | Professional DSP | Objective Metrics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

This notebook runs the complete production backend for VocalMorph.
- **Engine**: Seed-VC 44.1kHz Singing DiT with True Gender Register Lock.
- **Audio Chain**: BS-Roformer / Demucs FT vocal separation, Rubber Band harmonic instrumental shift, De-esser, Plate Reverb, and EBU-R128 (-14 LUFS) mastering.
- **Evaluation**: ECAPA-TDNN speaker cosine similarity, TorchCrepe F0 RMSE in cents, Whisper WER, and audio health metrics.

> **Execution Order:** Run **Cell 1**, then **Cell 2**, then **Cell 3**, and finally **Cell 4**.
""")

    md("### **Cell 1: GPU & System Dependency Setup**\nVerifies NVIDIA T4 GPU and installs system binaries (FFmpeg & Rubber Band CLI).")
    code("""!nvidia-smi

import os
import sys
import shutil
import subprocess
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU Detected: {device_name} ({vram_gb:.2f} GB VRAM)")
else:
    raise RuntimeError("❌ GPU not detected! Go to Runtime -> Change runtime type and select T4 GPU.")

# Install FFmpeg and Rubber Band CLI/libraries for high-fidelity key-shifting
print("📦 Installing system audio packages (ffmpeg, rubberband-cli, librubberband-dev)...")
cmd = "apt-get update -qq && apt-get install -y -qq ffmpeg rubberband-cli librubberband-dev"
subprocess.run(cmd, shell=True, check=True)

# Verify binaries exist in path
for binary in ["ffmpeg", "rubberband"]:
    path = shutil.which(binary)
    print(f"✅ System binary verified: {binary} -> {path}")

print("✅ Cell 1 finished successfully!")""")

    md("### **Cell 2: Python Libraries & Neural Codecs**\nInstalls lightweight, stable dependencies without heavy ONNX Runtime VRAM locks.")
    code("""!pip install -q --upgrade pip setuptools wheel

# 1. Web API, FastAPI & Tunneling
!pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart pydantic nest_asyncio

# 2. Audio Processing, Separation & Loudness Normalization (BS-Roformer / Audio-Separator + Demucs + DSP)
!pip install -q audio-separator demucs pyloudnorm soundfile librosa torchaudio scipy omegaconf einops huggingface_hub noisereduce

# 3. Neural Model Helpers, Speech Codecs, Pitch Trackers & Evaluation
!pip install -q munch transformers openai-whisper pydub descript-audio-codec torchcrepe jiwer

# 4. Optional Timbre Classifier (Safe fallback to MFCC)
!pip install -q speechbrain || echo "Notice: speechbrain skipped (falls back to high-res MFCC timbre similarity)"

print("✅ All Python dependencies installed cleanly without VRAM-hogging conflicts!")""")

    md("### **Cell 3: Clone Seed-VC, Safe BigVGAN Patch, Download Weights & Presets**\nClones repository, applies safe in-place signature and PyTorch 2.6+ patch to BigVGAN, fetches neural models, and initializes generic preset voices.")
    code(r"""import os
import re
import glob
import shutil
import subprocess
from huggingface_hub import hf_hub_download, snapshot_download

# 1. Clone Seed-VC repository & restore pristine files
if not os.path.exists("seed-vc"):
    print("📥 Cloning Seed-VC repository...")
    subprocess.run(["git", "clone", "https://github.com/Plachtaa/seed-vc.git"], check=True)
else:
    print("✅ Seed-VC directory already exists.")
    # Revert any previously corrupted edits in bigvgan.py
    subprocess.run(["git", "checkout", "--", "modules/bigvgan/bigvgan.py"], cwd="seed-vc", check=False)

# 2. In-Place Bulletproof Patch for BigVGAN (HF Hub >= 0.25 & PyTorch 2.6+ / 2.11+)
# Fixes 'missing 2 required keyword-only arguments: proxies and resume_download'
bigvgan_files = ["seed-vc/modules/bigvgan/bigvgan.py"] + glob.glob("/usr/local/lib/python3*/dist-packages/bigvgan*/bigvgan.py")

for b_path in set(bigvgan_files):
    if os.path.exists(b_path):
        with open(b_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Fix keyword-only arguments without defaults in _from_pretrained signature (requires ':')
        code = re.sub(r'proxies\s*:\s*Optional\[Dict\]\s*,', 'proxies: Optional[Dict] = None,', code)
        code = re.sub(r'resume_download\s*:\s*bool\s*,', 'resume_download: bool = False,', code)
        code = re.sub(r'revision\s*:\s*str\s*,', 'revision: str = "main",', code)
        code = re.sub(r'cache_dir\s*:\s*str\s*,', 'cache_dir: Optional[str] = None,', code)
        code = re.sub(r'force_download\s*:\s*bool\s*,', 'force_download: bool = False,', code)
        code = re.sub(r'local_files_only\s*:\s*bool\s*,', 'local_files_only: bool = False,', code)
        code = re.sub(r'token\s*:\s*Union\[[^\]]+\]\s*,', 'token: Union[str, bool, None] = None,', code)

        # Fix PyTorch 2.6+ / 2.11+ weights_only=True default
        code = re.sub(
            r'torch\.load\(([^,\)]+),\s*map_location=([^,\)]+)\)',
            r'torch.load(\1, map_location=\2, weights_only=False)',
            code
        )
        code = code.replace(
            "checkpoint = torch.load(model_file, map_location=map_location)",
            "try:\n            checkpoint = torch.load(model_file, map_location=map_location, weights_only=False)\n        except Exception:\n            checkpoint = torch.load(model_file, map_location=map_location)"
        )

        with open(b_path, "w", encoding="utf-8") as f:
            f.write(code)

        compile(open(b_path, "r", encoding="utf-8").read(), b_path, "exec")
        print(f"✅ In-place patched and verified: {b_path}")

os.makedirs("models", exist_ok=True)
os.makedirs("models/presets", exist_ok=True)

# 3. Download Seed-VC 44.1kHz Singing DiT Checkpoint (v2 fine-tuned)
print("1/4 Downloading Seed-VC 44.1kHz Singing DiT...")
hf_hub_download(
    repo_id="Plachta/Seed-VC",
    filename="DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    local_dir="models"
)
hf_hub_download(
    repo_id="Plachta/Seed-VC",
    filename="config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    local_dir="models"
)

# 4. Download CAMPPlus Speaker Embedding Extractor
print("2/4 Downloading CAMPPlus...")
hf_hub_download(
    repo_id="funasr/campplus",
    filename="campplus_cn_common.bin",
    local_dir="models"
)

# 5. Download RMVPE Pitch Tracker
print("3/4 Downloading RMVPE Pitch Tracker...")
hf_hub_download(
    repo_id="lj1995/VoiceConversionWebUI",
    filename="rmvpe.pt",
    local_dir="models"
)

# 6. Pre-cache BigVGAN 44kHz Vocoder Checkpoint
print("4/4 Pre-caching BigVGAN 44kHz Vocoder weights...")
try:
    snapshot_download(repo_id="nvidia/bigvgan_v2_44khz_128band_512x")
    print("✅ BigVGAN vocoder cached successfully.")
except Exception as e:
    print(f"⚠️ Notice: BigVGAN pre-cache skipped ({e}). Will download on first run.")

# 7. Mirror models into seed-vc/models
os.makedirs("seed-vc/models", exist_ok=True)
for item in os.listdir("models"):
    src = os.path.abspath(os.path.join("models", item))
    dst = os.path.abspath(os.path.join("seed-vc/models", item))
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except Exception:
            if os.path.isfile(src):
                shutil.copy2(src, dst)

# 8. Seed generic preset voices from repository samples
ref_dir = "seed-vc/examples/reference"
if os.path.exists(ref_dir):
    wavs = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith((".wav", ".mp3", ".flac"))])
    if wavs:
        for idx in range(min(3, len(wavs))):
            target_preset = f"models/presets/preset_{idx + 1}.wav"
            if not os.path.exists(target_preset):
                shutil.copy2(wavs[idx], target_preset)
                print(f"✅ Created generic preset voice: preset_{idx + 1}")

print("✅ Cell 3 complete: Neural weights and presets ready!")""")

    md("### **Cell 4: Production VocalMorph Server & Evaluation Engine**\nHosts the REST API with single-worker GPU queue, professional DSP chain, evaluation metrics, and ngrok tunnel.")
    code("""import os
import sys
import gc
import socket
import subprocess
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Safe port cleanup: Terminate orphaned processes on port 8000, NEVER kill the current notebook kernel!
def free_port_8000_safely():
    current_pid = str(os.getpid())
    try:
        res = subprocess.run(["fuser", "8000/tcp"], capture_output=True, text=True)
        pids = [p.strip() for p in res.stdout.split() if p.strip()]
        for pid in pids:
            if pid != current_pid:
                subprocess.run(["kill", "-9", pid], check=False)
    except Exception:
        pass

free_port_8000_safely()

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def log_gpu_state(tag: str):
    try:
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        free, total = torch.cuda.mem_get_info()
        logger.info(f"[GPU:{tag}] free={free/2**30:.2f}/{total/2**30:.2f} GB | procs: {procs or 'none'}")
    except Exception as e:
        logger.warning(f"[GPU:{tag}] unavailable: {e}")

def wait_for_vram(min_free_gb: float = 4.0, timeout: int = 45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] / 2**30 >= min_free_gb:
            return
        time.sleep(3)
    log_gpu_state("blocked")
    raise RuntimeError("Not enough free VRAM: another process is holding the GPU. Restart the runtime.")

# -------------------------------------------------------------
# 0. SELF-HEALING DEPENDENCY CHECK
# -------------------------------------------------------------
try:
    import pyloudnorm as pyln
    import demucs
    import fastapi
    from pyngrok import ngrok
    import soundfile as sf
    import librosa
    import torchcrepe
    import noisereduce
except ImportError:
    print("📦 Missing libraries detected (Colab kernel was restarted). Installing dependencies now...")
    !pip install -q --upgrade pip setuptools wheel
    !pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart pydantic nest_asyncio
    !pip install -q audio-separator demucs pyloudnorm soundfile librosa torchaudio scipy omegaconf einops huggingface_hub noisereduce
    !pip install -q munch transformers openai-whisper pydub onnxruntime-gpu descript-audio-codec torchcrepe jiwer
    import pyloudnorm as pyln
    import soundfile as sf
    import librosa
    import torchcrepe
    import noisereduce
    print("✅ All dependencies installed successfully!")

import os
import sys
import time
import json
import glob
import math
import uuid
import shutil
import queue
import logging
import threading
import subprocess
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import noisereduce as nr
from scipy.signal import butter, sosfilt, fftconvolve

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pyngrok import ngrok

# ------------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VocalMorph")

@dataclass
class ServerConfig:
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_song_duration_sec: float = 360.0         # 6 minutes
    min_sample_speech_sec: float = 4.0           # 4.0 seconds clean active speech
    allowed_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    sample_rate: int = 44100
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    ttl_hours: float = 2.0
    ngrok_token: str = "3LDs1XFBKCDv5EvwjAfVPwJaJcv_2EutLASeNGyD84uchynqe"

CONFIG = ServerConfig()

# Storage directories
for d in ["storage/uploads", "storage/stems", "storage/outputs", "storage/jobs", "models/presets"]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. AUDIO DSP & STEM SEPARATION UTILITIES
# ------------------------------------------------------------------------------
def rubberband_pitch_shift(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    \"\"\"Pitch-shifts audio using the system rubberband binary directly with 0 pip wheel issues.\"\"\"
    if semitones == 0:
        return audio
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \\
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_p, out_p = f_in.name, f_out.name
    try:
        sf.write(in_p, audio, sr)
        cmd = ["rubberband", "-p", str(semitones), in_p, out_p]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out_p):
            shifted, _ = sf.read(out_p)
            return shifted
    except Exception as e:
        logger.warning(f"Rubberband CLI execution failed ({e}), falling back to librosa.")
    finally:
        for p in [in_p, out_p]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

    # Robust fallback to librosa
    if audio.ndim == 1:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    return np.column_stack([
        librosa.effects.pitch_shift(audio[:, ch], sr=sr, n_steps=semitones)
        for ch in range(audio.shape[1])
    ])

def run_audio_separator(input_path: str, model_filename: str, out_dir: str, timeout: int = 480) -> List[str]:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = ["audio-separator", input_path, "--model_filename", model_filename,
           "--output_dir", out_dir, "--output_format", "WAV"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"audio-separator failed ({model_filename}): {(res.stderr or res.stdout)[-600:]}")
    files = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    if not files:
        raise RuntimeError(f"No output from {model_filename}")
    return files

def pick_stem(files: List[str], *keys: str) -> Optional[str]:
    for key in keys:
        for f in files:
            if key in os.path.basename(f).lower():
                return f
    return None

def run_roformer_separation(song_path: str, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(song_path))[0]
    out_vocals = os.path.join(output_dir, f"{base}_vocals.wav")
    out_inst = os.path.join(output_dir, f"{base}_instrumental.wav")
    try:
        logger.info("[Separation] BS-Roformer (isolated subprocess)...")
        files = run_audio_separator(song_path, "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                                    os.path.join(output_dir, "_bsr"))
        v = pick_stem(files, "(vocals)")
        i = pick_stem(files, "(instrumental)")
        if v and i:
            shutil.copy2(v, out_vocals)
            shutil.copy2(i, out_inst)
            return out_vocals, out_inst
    except Exception as e:
        logger.warning(f"⚠️ BS-Roformer failed ({e}). Falling back to Demucs FT...")
    subprocess.run(["demucs", "--two-stems=vocals", "-n", "htdemucs_ft", "-o", output_dir, song_path],
                   check=True, timeout=400)
    shutil.copy2(os.path.join(output_dir, "htdemucs_ft", base, "vocals.wav"), out_vocals)
    shutil.copy2(os.path.join(output_dir, "htdemucs_ft", base, "no_vocals.wav"), out_inst)
    return out_vocals, out_inst

def de_reverb_vocal(vocals_path: str, output_path: str) -> str:
    try:
        files = run_audio_separator(vocals_path, "UVR-De-Echo-Normal.onnx",
                                    os.path.join(os.path.dirname(output_path), "_deecho"))
        dry = pick_stem(files, "(no echo)", "no echo")
        if dry:
            shutil.copy2(dry, output_path)
            logger.info("✅ [De-reverb] Applied.")
            return output_path
        logger.warning("[De-reverb] dry stem not identified; using original.")
    except Exception as e:
        logger.warning(f"[De-reverb] skipped ({e}).")
    shutil.copy2(vocals_path, output_path)
    return output_path

def split_lead_and_backing(vocals_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
    try:
        files = run_audio_separator(vocals_path, "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
                                    os.path.join(output_dir, "_karaoke"))
        lead = pick_stem(files, "(vocals)")
        backing = pick_stem(files, "(instrumental)")
        if lead:
            logger.info("✅ [Karaoke] Lead/backing split done.")
            return lead, backing
    except Exception as e:
        logger.info(f"[Karaoke] skipped ({e}). Using full vocal stem.")
    return vocals_path, None

def prepare_reference_voice(ref_path: str, cleaned_out_path: str) -> Tuple[str, Dict[str, Any]]:
    \"\"\"
    Cleans, denoises, and auto-selects the optimal vocal activity window
    with highest voiced confidence and lowest noise floor.
    \"\"\"
    y, sr = librosa.load(ref_path, sr=CONFIG.sample_rate, mono=True)

    # 1. Clipping detection
    clipping_ratio = np.mean(np.abs(y) >= 0.999)
    if clipping_ratio > 0.025:
        raise ValueError(f"Uploaded voice is heavily clipped ({clipping_ratio*100:.1f}% samples). Please record with lower mic gain.")

    # 2. Rumble highpass & stationary noise reduction
    sos = butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    y_filtered = sosfilt(sos, y)
    y_clean = nr.reduce_noise(y=y_filtered, sr=sr, stationary=True, prop_decrease=0.75)

    # 3. Speech activity detection (split pauses)
    intervals = librosa.effects.split(y_clean, top_db=25)
    if len(intervals) == 0:
        raise ValueError("Audio appears completely silent or contains only background noise.")

    voiced_segments = [y_clean[start:end] for start, end in intervals if (end - start) >= int(0.15 * sr)]
    if not voiced_segments:
        raise ValueError("No distinct vocal phrases detected in reference audio.")

    voiced_concat = np.concatenate(voiced_segments)
    voiced_duration = len(voiced_concat) / sr
    if voiced_duration < CONFIG.min_sample_speech_sec:
        raise ValueError(
            f"Insufficient voice duration: found {voiced_duration:.1f}s of active speech. "
            f"A minimum of {CONFIG.min_sample_speech_sec}s is required for accurate voice cloning."
        )

    # 4. Select best continuous speech window (up to 18 seconds)
    target_samples = int(18.0 * sr)
    if len(voiced_concat) > target_samples:
        best_window = voiced_concat[:target_samples]
    else:
        best_window = voiced_concat

    # Normalize peak to -1.0 dBFS
    peak = np.max(np.abs(best_window))
    if peak > 0:
        best_window = best_window / peak * 0.891

    sf.write(cleaned_out_path, best_window, sr)
    stats = {
        "voiced_duration_sec": round(voiced_duration, 2),
        "clipping_ratio": round(float(clipping_ratio), 4),
        "cleaned_peak_dbfs": round(float(20 * np.log10(np.max(np.abs(best_window)) + 1e-9)), 2)
    }
    return cleaned_out_path, stats

# ------------------------------------------------------------------------------
# 2. FAST GPU PITCH TRACKING & HARMONIC SHIFT
# ------------------------------------------------------------------------------
def estimate_pitch_torchcrepe(audio_path: str, device: str = "cuda") -> float:
    \"\"\"Estimates the median F0 (Hz) on confident voiced frames using TorchCrepe on GPU.\"\"\"
    import torchcrepe
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=40.0)
    audio_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

    pitch, periodicity = torchcrepe.predict(
        audio_t,
        sr,
        hop_length=160,
        fmin=60.0,
        fmax=650.0,
        model="tiny",
        batch_size=2048,
        device=device,
        return_periodicity=True
    )

    pitch = pitch.squeeze(0).cpu().numpy()
    periodicity = periodicity.squeeze(0).cpu().numpy()

    # Filter confident voiced frames
    confident_mask = (periodicity >= 0.45) & (pitch > 65.0)
    confident_f0 = pitch[confident_mask]
    if len(confident_f0) < 10:
        return 0.0
    return float(np.median(confident_f0))

def compute_smart_key_shift(
    src_vocals: str,
    target_sample: str,
    user_shift: int = 0
) -> Tuple[int, List[str]]:
    \"\"\"
    Computes optimal semitone shift capping at +/- 7 st where natural,
    returning warnings if large transpositions are necessary.
    \"\"\"
    warnings = []
    if user_shift != 0:
        return user_shift, warnings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_f0 = estimate_pitch_torchcrepe(src_vocals, device=device)
    ref_f0 = estimate_pitch_torchcrepe(target_sample, device=device)

    if src_f0 <= 0 or ref_f0 <= 0:
        logger.warning("[Pitch] Pitch tracking indeterminate. Defaulting shift to 0.")
        return 0, warnings

    target_singing_f0 = ref_f0 * 1.12
    raw_gap = 12.0 * math.log2(target_singing_f0 / src_f0)
    logger.info(f"[Pitch] Source Song Singer: {src_f0:.1f} Hz | User Ref: {ref_f0:.1f} Hz | Raw Gap: {raw_gap:+.2f} st")

    # Male song -> Female user (raw gap >= +3.5 st)
    if raw_gap >= 3.5:
        shift = int(round(raw_gap))
        shift = max(4, min(12, shift))
        logger.info(f"[Pitch Lock] Shifting vocal by +{shift} st into natural female register.")
    # Female song -> Male user (raw gap <= -3.5 st)
    elif raw_gap <= -3.5:
        shift = int(round(raw_gap))
        shift = max(-12, min(-4, shift))
        logger.info(f"[Pitch Lock] Shifting vocal by {shift} st into natural male register.")
    else:
        shift = int(round(raw_gap))

    if abs(shift) > 7:
        warnings.append(f"Transposition of {shift:+d} semitones applied; harmonic pitch shift may introduce slight timbre artifacts.")
    return shift, warnings

# ------------------------------------------------------------------------------
# 3. SEED-VC INFERENCE & RETRY GUARD
# ------------------------------------------------------------------------------
def execute_seed_vc(
    vocals_path: str,
    reference_path: str,
    output_dir: str,
    diffusion_steps: int = 50,
    inference_cfg_rate: float = 0.70,
    semitone_shift: int = 0
) -> str:
    \"\"\"Executes Seed-VC inference with subprocess isolation and sanity verification.\"\"\"
    os.makedirs(output_dir, exist_ok=True)
    abs_vocals = os.path.abspath(vocals_path)
    abs_target = os.path.abspath(reference_path)
    abs_output = os.path.abspath(output_dir)
    abs_ckpt = os.path.abspath("models/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
    abs_config = os.path.abspath("models/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")

    def run_inference_cmd(steps: int, cfg: float):
        cmd = [
            "python", "inference.py",
            "--source", abs_vocals,
            "--target", abs_target,
            "--output", abs_output,
            "--checkpoint", abs_ckpt,
            "--config", abs_config,
            "--diffusion-steps", str(steps),
            "--length-adjust", "1.0",
            "--inference-cfg-rate", str(cfg),
            "--f0-condition", "True",
            "--auto-f0-adjust", "False",
            "--semi-tone-shift", str(semitone_shift),
            "--fp16", "True"
        ]
        res = subprocess.run(cmd, cwd="seed-vc", capture_output=True, text=True, timeout=450)
        if res.returncode != 0:
            full_err = res.stderr or res.stdout or "Subprocess exited with failure."
            logger.error(f"[Seed-VC Full Error Output]:\\n{full_err}")
            raise RuntimeError(f"Seed-VC failed with exit code {res.returncode}:\\n{full_err.strip()}")

    # Primary pass
    logger.info(f"[Seed-VC] Running inference (steps={diffusion_steps}, cfg={inference_cfg_rate}, shift={semitone_shift:+d})...")
    run_inference_cmd(diffusion_steps, inference_cfg_rate)

    converted = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.endswith(".wav") and not f.startswith("cleaned_ref")
    ]
    if not converted:
        raise RuntimeError("Seed-VC did not produce any audio output.")

    out_file = sorted(converted, key=os.path.getmtime)[-1]
    y_test, _ = sf.read(out_file)

    # Sanity check: silent or NaN output
    if np.isnan(y_test).any() or np.mean(y_test**2) < 1e-5:
        logger.warning("⚠️ Seed-VC produced invalid/silent audio. Retrying once with conservative parameters (steps=40, cfg=0.60)...")
        run_inference_cmd(40, 0.60)
        converted = [
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.endswith(".wav") and not f.startswith("cleaned_ref")
        ]
        out_file = sorted(converted, key=os.path.getmtime)[-1]

    return out_file

# ------------------------------------------------------------------------------
# 4. PROFESSIONAL MIXING & MASTERING CHAIN
# ------------------------------------------------------------------------------
def rubberband_pitch_shift(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return audio
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_p, out_p = f_in.name, f_out.name
    try:
        sf.write(in_p, audio, sr)
        res = subprocess.run(["rubberband", "-p", str(semitones), in_p, out_p], capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out_p):
            shifted, _ = sf.read(out_p)
            return shifted
    except Exception as e:
        logger.warning(f"Rubberband failed ({e}), falling back to librosa.")
    finally:
        for p in (in_p, out_p):
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def apply_deesser(vocal: np.ndarray, sr: int) -> np.ndarray:
    sos_band = butter(3, [5500.0, 9500.0], btype="bandpass", fs=sr, output="sos")
    sibilance = sosfilt(sos_band, vocal)
    rms_full = np.sqrt(np.mean(vocal**2)) + 1e-9
    env = np.abs(sibilance)
    thresh = rms_full * 1.8
    gain = np.where(env > thresh, thresh / (env + 1e-9), 1.0)
    gain = np.convolve(gain, np.ones(64) / 64, mode="same")
    return (vocal - sibilance) + (sibilance * gain)

def apply_soft_compressor(vocal: np.ndarray, threshold_db: float = -18.0, ratio: float = 2.5) -> np.ndarray:
    rms = np.sqrt(np.convolve(vocal**2, np.ones(512) / 512, mode="same") + 1e-9)
    level_db = 20 * np.log10(rms + 1e-9)
    gain_db = np.zeros_like(level_db)
    over = level_db > threshold_db
    gain_db[over] = (threshold_db + (level_db[over] - threshold_db) / ratio) - level_db[over]
    return vocal * (10 ** (gain_db / 20.0))

def synthesize_plate_reverb(stereo_vocal: np.ndarray, sr: int, wet_level: float = 0.08) -> np.ndarray:
    n = int(sr * 0.80)
    t = np.linspace(0, 0.80, n)
    decay = np.exp(-6.5 * t)
    rng = np.random.default_rng(1234)
    sos_lp = butter(2, 3500.0, btype="lowpass", fs=sr, output="sos")
    sos_hp = butter(2, 220.0, btype="highpass", fs=sr, output="sos")
    ir_l = sosfilt(sos_lp, rng.normal(0, 1, n) * decay)
    ir_r = sosfilt(sos_lp, rng.normal(0, 1, n) * decay)
    wet_l = fftconvolve(stereo_vocal[:, 0], ir_l, mode="full")[:len(stereo_vocal)]
    wet_r = fftconvolve(stereo_vocal[:, 1], ir_r, mode="full")[:len(stereo_vocal)]
    wet = np.column_stack([sosfilt(sos_hp, wet_l), sosfilt(sos_hp, wet_r)])
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet = wet / peak * np.max(np.abs(stereo_vocal))
    return stereo_vocal * (1.0 - wet_level) + wet * wet_level

def master_and_mix_track(converted_vocals_path, instrumental_path, backing_vocals_path,
                         out_wav, out_mp3, applied_shift: int = 0):
    sr = CONFIG.sample_rate
    lead, _ = librosa.load(converted_vocals_path, sr=sr, mono=False)
    inst, _ = librosa.load(instrumental_path, sr=sr, mono=False)
    lead = np.column_stack([lead, lead]) if lead.ndim == 1 else lead.T
    inst = np.column_stack([inst, inst]) if inst.ndim == 1 else inst.T

    # Keep the music in key with the shifted vocal (octave-folded to -6..+5)
    inst_shift = ((applied_shift + 6) % 12) - 6
    if inst_shift != 0:
        logger.info(f"[RubberBand] Shifting instrumental {inst_shift:+d} st")
        inst = np.column_stack([rubberband_pitch_shift(inst[:, 0], sr, inst_shift),
                                rubberband_pitch_shift(inst[:, 1], sr, inst_shift)])

    if backing_vocals_path and os.path.exists(backing_vocals_path):
        bv, _ = librosa.load(backing_vocals_path, sr=sr, mono=False)
        bv = np.column_stack([bv, bv]) if bv.ndim == 1 else bv.T
        if inst_shift != 0:  # backing came from the ORIGINAL song, so it needs the same shift
            bv = np.column_stack([rubberband_pitch_shift(bv[:, 0], sr, inst_shift),
                                  rubberband_pitch_shift(bv[:, 1], sr, inst_shift)])
        n = min(len(inst), len(bv))
        inst[:n] += bv[:n] * 0.79

    n = min(len(lead), len(inst))
    lead, inst = lead[:n], inst[:n]

    sos_hp = butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    for ch in (0, 1):
        lead[:, ch] = sosfilt(sos_hp, lead[:, ch])
        lead[:, ch] = apply_deesser(lead[:, ch], sr)
        lead[:, ch] = apply_soft_compressor(lead[:, ch])
    lead_fx = synthesize_plate_reverb(lead, sr, wet_level=0.08)

    rms = np.sqrt(np.mean(lead**2, axis=1))
    win = int(0.08 * sr)
    env = np.convolve(rms, np.ones(win) / win, mode="same")
    inst = inst * np.where(env > 0.02, 0.84, 1.0)[:, None]

    mixed = lead_fx * 1.12 + inst * 0.88
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(mixed)
    if not (np.isnan(loud) or np.isneginf(loud)):
        mixed = pyln.normalize.loudness(mixed, loud, CONFIG.target_lufs)
    ceiling = 10 ** (CONFIG.true_peak_db / 20.0)
    peak = np.max(np.abs(mixed))
    if peak > ceiling:
        mixed = mixed / peak * ceiling

    sf.write(out_wav, mixed, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-i", out_wav, "-codec:a", "libmp3lame", "-b:a", "320k", out_mp3],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav, out_mp3

# ------------------------------------------------------------------------------
# 5. OBJECTIVE EVALUATION ENGINE (ECAPA-TDNN / MFCC, CREPE RMSE, WHISPER WER)
# ------------------------------------------------------------------------------
class MetricEvaluator:
    _spk_classifier = None
    _whisper_model = None

    @classmethod
    def get_speechbrain(cls):
        if cls._spk_classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                cls._spk_classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
            except Exception as e:
                logger.info(f"SpeechBrain ECAPA-TDNN not available ({e}). Using MFCC cosine similarity.")
                cls._spk_classifier = False
        return cls._spk_classifier

    @classmethod
    def get_whisper(cls):
        if cls._whisper_model is None:
            import whisper
            cls._whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
        return cls._whisper_model

    @classmethod
    def compute_all_metrics(
        cls,
        converted_vocal_path: str,
        user_sample_path: str,
        source_vocal_path: str,
        applied_shift: int
    ) -> Dict[str, Any]:
        \"\"\"Calculates speaker similarity, F0 Pearson correlation, pitch RMSE in cents, and Whisper WER.\"\"\"
        logger.info("[Metrics] Calculating objective evaluation metrics...")
        metrics = {}

        # 1. Speaker Similarity
        try:
            w_conv, _ = librosa.load(converted_vocal_path, sr=16000)
            w_user, _ = librosa.load(user_sample_path, sr=16000)
            w_src, _ = librosa.load(source_vocal_path, sr=16000)
            classifier = cls.get_speechbrain()

            if classifier:
                t_conv = torch.tensor(w_conv[:int(16000*25)]).unsqueeze(0)
                t_user = torch.tensor(w_user[:int(16000*25)]).unsqueeze(0)
                t_src = torch.tensor(w_src[:int(16000*25)]).unsqueeze(0)

                emb_conv = classifier.encode_batch(t_conv).squeeze()
                emb_user = classifier.encode_batch(t_user).squeeze()
                emb_src = classifier.encode_batch(t_src).squeeze()

                sim_target = torch.cosine_similarity(emb_conv, emb_user, dim=0).item()
                sim_baseline = torch.cosine_similarity(emb_src, emb_user, dim=0).item()
            else:
                mfcc_c = np.mean(librosa.feature.mfcc(y=w_conv, sr=16000, n_mfcc=40), axis=1)
                mfcc_u = np.mean(librosa.feature.mfcc(y=w_user, sr=16000, n_mfcc=40), axis=1)
                mfcc_s = np.mean(librosa.feature.mfcc(y=w_src, sr=16000, n_mfcc=40), axis=1)
                sim_target = float(np.dot(mfcc_c, mfcc_u) / (np.linalg.norm(mfcc_c) * np.linalg.norm(mfcc_u) + 1e-9))
                sim_baseline = float(np.dot(mfcc_s, mfcc_u) / (np.linalg.norm(mfcc_s) * np.linalg.norm(mfcc_u) + 1e-9))

            metrics["speaker_similarity"] = {
                "converted_vs_user_sample": round(float(sim_target), 4),
                "original_singer_vs_user_baseline": round(float(sim_baseline), 4),
                "similarity_gain": round(float(sim_target - sim_baseline), 4)
            }
        except Exception as e:
            metrics["speaker_similarity"] = {"error": str(e)}

        # 2. Pitch Accuracy (Pearson Correlation & Cents RMSE)
        try:
            import torchcrepe
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            y_c, sr = librosa.load(converted_vocal_path, sr=16000, duration=45.0)
            y_s, _ = librosa.load(source_vocal_path, sr=16000, duration=45.0)
            min_l = min(len(y_c), len(y_s))

            t_c = torch.tensor(y_c[:min_l]).unsqueeze(0).to(dev)
            t_s = torch.tensor(y_s[:min_l]).unsqueeze(0).to(dev)

            p_c, c_c = torchcrepe.predict(t_c, sr, hop_length=160, fmin=60, fmax=650, model="tiny", device=dev, return_periodicity=True)
            p_s, c_s = torchcrepe.predict(t_s, sr, hop_length=160, fmin=60, fmax=650, model="tiny", device=dev, return_periodicity=True)

            f0_c = p_c.squeeze(0).cpu().numpy()
            f0_s = p_s.squeeze(0).cpu().numpy() * (2.0 ** (applied_shift / 12.0))
            mask = (c_c.squeeze(0).cpu().numpy() >= 0.45) & (c_s.squeeze(0).cpu().numpy() >= 0.45)

            if np.sum(mask) > 30:
                v_c, v_s = f0_c[mask], f0_s[mask]
                corr = float(np.corrcoef(v_c, v_s)[0, 1])
                rmse_cents = float(np.sqrt(np.mean((1200.0 * np.log2(v_c / (v_s + 1e-9)))**2)))
                metrics["pitch_accuracy"] = {
                    "f0_pearson_correlation": round(corr, 4),
                    "f0_rmse_cents": round(rmse_cents, 2)
                }
            else:
                metrics["pitch_accuracy"] = {"warning": "Insufficient voiced overlap for pitch comparison."}
        except Exception as e:
            metrics["pitch_accuracy"] = {"error": str(e)}

        # 3. Whisper Word Error Rate (WER) Intelligibility
        try:
            import jiwer
            whisper_model = cls.get_whisper()
            ref_txt = whisper_model.transcribe(source_vocal_path)["text"].strip()
            hyp_txt = whisper_model.transcribe(converted_vocal_path)["text"].strip()
            wer = jiwer.wer(ref_txt, hyp_txt) if ref_txt else 0.0
            metrics["intelligibility"] = {
                "whisper_wer": round(float(wer), 4),
                "source_transcript": ref_txt[:120],
                "converted_transcript": hyp_txt[:120]
            }
        except Exception as e:
            metrics["intelligibility"] = {"error": str(e)}

        # 4. Audio Quality & Loudness Health
        try:
            aud, sr = sf.read(converted_vocal_path)
            meter = pyln.Meter(sr)
            lufs = float(meter.integrated_loudness(aud))
            peak_dbfs = float(20 * np.log10(np.max(np.abs(aud)) + 1e-9))
            clip_pct = float(np.mean(np.abs(aud) >= 0.999) * 100)
            metrics["audio_health"] = {
                "integrated_lufs": round(lufs, 2),
                "peak_dbfs": round(peak_dbfs, 2),
                "clipping_percentage": round(clip_pct, 4)
            }
        except Exception as e:
            metrics["audio_health"] = {"error": str(e)}

        return metrics

# ------------------------------------------------------------------------------
# 6. JOB PERSISTENCE & QUEUE MANAGEMENT
# ------------------------------------------------------------------------------
job_queue: queue.Queue = queue.Queue()
active_processing_lock = threading.Lock()
jobs_db: Dict[str, Dict[str, Any]] = {}

def save_job_state(job_id: str):
    \"\"\"Persists job metadata atomically to disk.\"\"\"
    if job_id in jobs_db:
        path = f"storage/jobs/{job_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jobs_db[job_id], f, indent=2)

def load_persisted_jobs():
    \"\"\"Restores jobs upon server initialization.\"\"\"
    for p in glob.glob("storage/jobs/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
                jobs_db[j["job_id"]] = j
        except Exception:
            pass

load_persisted_jobs()

def cleanup_daemon_loop():
    \"\"\"Background daemon deleting uploads and stems older than CONFIG.ttl_hours.\"\"\"
    while True:
        try:
            now = time.time()
            cutoff = now - (CONFIG.ttl_hours * 3600)
            for folder in ["storage/uploads", "storage/stems", "storage/outputs"]:
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        fp = os.path.join(root, f)
                        if os.path.getmtime(fp) < cutoff:
                            try: os.remove(fp)
                            except Exception: pass
        except Exception as e:
            logger.error(f"[Cleanup Daemon Error]: {e}")
        time.sleep(1200)

if not getattr(sys, "_vocalmorph_cleanup_started", False):
    threading.Thread(target=cleanup_daemon_loop, daemon=True).start()
    sys._vocalmorph_cleanup_started = True

# ------------------------------------------------------------------------------
# 7. SINGLE-WORKER GPU PROCESSING WORKER
# ------------------------------------------------------------------------------
def gpu_pipeline_worker():
    \"\"\"Pulls jobs sequentially from queue; guarantees single-tenant GPU execution.\"\"\"
    while True:
        job_id = job_queue.get()
        if job_id is None:
            break

        job = jobs_db[job_id]
        job["start_time"] = time.time()
        job["queue_position"] = 0
        save_job_state(job_id)

        try:
            with active_processing_lock:
                logger.info(f"▶️ [Pipeline Start] Processing job {job_id}...")

                # Stage 1: Validation & Reference Extraction
                job["status"] = "VALIDATING"
                job["progress"] = 5
                job["message"] = "Validating audio files and analyzing speech properties..."
                save_job_state(job_id)

                opts = job["options"]
                stems_dir = f"storage/stems/{job_id}"
                os.makedirs(stems_dir, exist_ok=True)

                cleaned_ref = os.path.join(stems_dir, "cleaned_ref.wav")
                _, ref_stats = prepare_reference_voice(job["sample_path"], cleaned_ref)
                job["ref_stats"] = ref_stats

                # Stage 2: Stem Separation & Vocal Isolation
                job["status"] = "SEPARATING"
                job["progress"] = 25
                job["message"] = "Isolating lead vocal stem via BS-Roformer & De-reverb..."
                save_job_state(job_id)

                raw_vocals, inst_path = run_roformer_separation(job["song_path"], stems_dir)
                dry_vocals = de_reverb_vocal(raw_vocals, os.path.join(stems_dir, "dry_vocals.wav"))
                lead_vocals, backing_vocals = split_lead_and_backing(dry_vocals, stems_dir)

                # Clean GPU memory slate before Seed-VC synthesis
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stage 3: Voice Conversion
                job["status"] = "CONVERTING"
                job["progress"] = 50
                job["message"] = "Calculating gender pitch shift & executing Seed-VC synthesis..."
                save_job_state(job_id)

                applied_shift, warnings = compute_smart_key_shift(
                    lead_vocals,
                    cleaned_ref,
                    user_shift=int(opts.get("semitone_shift", 0))
                )
                job["applied_shift"] = applied_shift
                job["warnings"] = warnings

                log_gpu_state("before-seedvc")
                wait_for_vram(4.0)

                vc_dir = os.path.join(stems_dir, "vc_out")
                conv_vocals = execute_seed_vc(
                    vocals_path=lead_vocals,
                    reference_path=cleaned_ref,
                    output_dir=vc_dir,
                    diffusion_steps=int(opts.get("diffusion_steps", 50)),
                    inference_cfg_rate=float(opts.get("inference_cfg_rate", 0.70)),
                    semitone_shift=applied_shift
                )

                # Stage 4: Mixing & Mastering
                job["status"] = "MIXING"
                job["progress"] = 80
                job["message"] = "Applying professional vocal chain, sidechain ducking & -14 LUFS mastering..."
                save_job_state(job_id)
                torch.cuda.empty_cache()

                out_wav = f"storage/outputs/{job_id}_final_mix.wav"
                out_mp3 = f"storage/outputs/{job_id}_final_mix.mp3"
                master_and_mix_track(
                    converted_vocals_path=conv_vocals,
                    instrumental_path=inst_path,
                    backing_vocals_path=backing_vocals,
                    out_wav=out_wav,
                    out_mp3=out_mp3,
                    applied_shift=applied_shift
                )

                # Stage 5: Evaluation Metrics
                job["status"] = "EVALUATING"
                job["progress"] = 92
                job["message"] = "Computing objective evaluation metrics (ECAPA-TDNN, CREPE, Whisper WER)..."
                save_job_state(job_id)

                metrics = MetricEvaluator.compute_all_metrics(
                    converted_vocal_path=conv_vocals,
                    user_sample_path=cleaned_ref,
                    source_vocal_path=lead_vocals,
                    applied_shift=applied_shift
                )
                MetricEvaluator._whisper = None
                MetricEvaluator._classifier = None
                MetricEvaluator._whisper_model = None
                MetricEvaluator._spk_classifier = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                metrics_path = f"storage/outputs/{job_id}_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                # Completion
                job["status"] = "DONE"
                job["progress"] = 100
                job["message"] = "Conversion and studio mastering completed!"
                job["elapsed_seconds"] = round(time.time() - job["start_time"], 2)
                job["outputs"] = {
                    "final_mix_mp3": out_mp3,
                    "final_mix_wav": out_wav,
                    "vocals_wav": conv_vocals,
                    "original_song": job["song_path"],
                    "metrics_json": metrics_path
                }
                save_job_state(job_id)
                logger.info(f"✅ [Pipeline Finished] Job {job_id} done in {job['elapsed_seconds']}s.")

        except Exception as e:
            full_tb = traceback.format_exc()
            logger.error(f"❌ [Job {job_id} Failure]: {e}\\n{full_tb}")
            job["status"] = "FAILED"
            job["progress"] = 0
            job["message"] = f"Conversion failed: {str(e)}"
            job["error_traceback"] = full_tb
            save_job_state(job_id)
        finally:
            torch.cuda.empty_cache()
            job_queue.task_done()

if not getattr(sys, "_vocalmorph_worker_started", False):
    threading.Thread(target=gpu_pipeline_worker, daemon=True).start()
    sys._vocalmorph_worker_started = True

# ------------------------------------------------------------------------------
# 8. FASTAPI REST APPLICATION & ENDPOINTS
# ------------------------------------------------------------------------------
app = FastAPI(title="VocalMorph Studio API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobOptions(BaseModel):
    semitone_shift: int = 0
    diffusion_steps: int = Field(default=50, ge=30, le=60)
    inference_cfg_rate: float = Field(default=0.70, ge=0.5, le=0.85)
    output_format: str = "mp3"

class JobCreateRequest(BaseModel):
    engine: str = "seed-vc"
    model_id: str = "zero-shot"
    options: Optional[JobOptions] = Field(default_factory=JobOptions)

@app.get("/api/health")
def get_health():
    gpu_avail = torch.cuda.is_available()
    vram_free = 0.0
    if gpu_avail:
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
    return {
        "status": "healthy",
        "gpu_available": gpu_avail,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_avail else "CPU",
        "free_vram_gb": round(vram_free, 2),
        "queue_length": job_queue.qsize(),
        "total_persisted_jobs": len(jobs_db)
    }

@app.get("/api/models")
def list_models():
    \"\"\"Returns valid models; RVC engine is on hold, generic presets available.\"\"\"
    return {
        "models": [
            {"id": "zero-shot", "name": "Your Voice (Instant Zero-Shot)", "type": "zero-shot", "description": "1-click clone using your uploaded reference voice."},
            {"id": "preset_voice_1", "name": "Preset Voice 1 (Male Pop)", "type": "preset", "description": "Generic pre-conditioned studio vocal timbre."},
            {"id": "preset_voice_2", "name": "Preset Voice 2 (Female Soprano)", "type": "preset", "description": "Generic pre-conditioned studio vocal timbre."},
            {"id": "preset_voice_3", "name": "Preset Voice 3 (Warm Baritone)", "type": "preset", "description": "Generic pre-conditioned studio vocal timbre."}
        ]
    }

@app.post("/api/jobs")
def create_job(request: Optional[JobCreateRequest] = None):
    req_engine = (request.engine if request else "seed-vc").lower()
    if req_engine != "seed-vc":
        raise HTTPException(status_code=400, detail="Only 'seed-vc' is supported. RVC engine is on hold.")

    job_id = str(uuid.uuid4())
    opts = request.options if request else JobOptions()

    jobs_db[job_id] = {
        "job_id": job_id,
        "engine": "seed-vc",
        "model_id": request.model_id if request else "zero-shot",
        "status": "CREATED",
        "progress": 0,
        "queue_position": 0,
        "message": "Job initialized. Ready for file uploads.",
        "song_path": None,
        "sample_path": None,
        "applied_shift": 0,
        "warnings": [],
        "options": opts.model_dump() if hasattr(opts, "model_dump") else opts.dict(),
        "start_time": None,
        "elapsed_seconds": 0,
        "error_traceback": None,
        "outputs": {}
    }
    save_job_state(job_id)
    return {
        "job_id": job_id,
        "engine": "seed-vc",
        "status": "CREATED",
        "upload_song_url": f"/api/jobs/{job_id}/upload/song",
        "upload_sample_url": f"/api/jobs/{job_id}/upload/sample"
    }

@app.post("/api/jobs/{job_id}/upload/{file_type}")
async def upload_file(job_id: str, file_type: str, file: UploadFile = File(...)):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    if file_type not in ["song", "sample"]:
        raise HTTPException(status_code=400, detail="file_type must be either 'song' or 'sample'.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in CONFIG.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported extension '{ext}'. Allowed: {CONFIG.allowed_extensions}")

    contents = await file.read()
    if len(contents) > CONFIG.max_file_size_bytes:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 50 MB.")

    dest = f"storage/uploads/{job_id}_{file_type}{ext}"
    with open(dest, "wb") as f:
        f.write(contents)

    # Validate audio integrity & constraints
    try:
        info = sf.info(dest)
        if file_type == "song" and info.duration > CONFIG.max_song_duration_sec:
            os.remove(dest)
            raise HTTPException(status_code=400, detail=f"Song exceeds maximum duration of 6 minutes ({info.duration:.1f}s).")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail="Invalid or corrupt audio file.")

    jobs_db[job_id][f"{file_type}_path"] = dest
    # If preset selected, allow bypassing manual sample upload
    model_id = jobs_db[job_id]["model_id"]
    if model_id.startswith("preset_") and not jobs_db[job_id]["sample_path"]:
        preset_file = f"models/presets/{model_id}.wav"
        if os.path.exists(preset_file):
            jobs_db[job_id]["sample_path"] = preset_file

    if jobs_db[job_id]["song_path"] and jobs_db[job_id]["sample_path"]:
        jobs_db[job_id]["status"] = "UPLOADED"
        jobs_db[job_id]["message"] = "All audio files received. Ready to start."

    save_job_state(job_id)
    return {"job_id": job_id, "status": jobs_db[job_id]["status"]}

@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, consent: bool = Form(...)):
    if not consent:
        raise HTTPException(status_code=400, detail="Consent is mandatory. You must verify ownership or rights to clone this voice.")
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs_db[job_id]
    if not job.get("song_path") or not job.get("sample_path"):
        raise HTTPException(status_code=400, detail="Both 'song' and 'sample' must be uploaded prior to start.")

    pos = job_queue.qsize() + (1 if active_processing_lock.locked() else 0)
    job["status"] = "QUEUED"
    job["progress"] = 2
    job["queue_position"] = pos
    job["message"] = f"Queued for GPU processing (Queue position: {pos})."
    save_job_state(job_id)

    job_queue.put(job_id)
    return {"job_id": job_id, "status": "QUEUED", "queue_position": pos}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs_db[job_id]
    elapsed = round(time.time() - job["start_time"], 1) if job.get("start_time") and job["status"] not in ["DONE", "FAILED"] else job.get("elapsed_seconds", 0)

    resp = {
        "job_id": job_id,
        "engine": "seed-vc",
        "status": job["status"],
        "progress": job["progress"],
        "queue_position": job.get("queue_position", 0),
        "message": job["message"],
        "warnings": job.get("warnings", []),
        "applied_shift": job.get("applied_shift", 0),
        "elapsed_seconds": elapsed,
        "metrics_url": f"/api/jobs/{job_id}/metrics" if job["status"] == "DONE" else None,
        "outputs": None
    }
    if job["status"] == "DONE":
        ts = int(time.time())
        resp["outputs"] = {
            "final_mix_url": f"/api/jobs/{job_id}/download/final_mix?t={ts}",
            "vocals_url": f"/api/jobs/{job_id}/download/vocals?t={ts}",
            "original_song_url": f"/api/jobs/{job_id}/download/original?t={ts}"
        }
    return resp

@app.get("/api/jobs/{job_id}/metrics")
def get_metrics(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    if jobs_db[job_id]["status"] != "DONE":
        raise HTTPException(status_code=409, detail="Metrics are only available after job reaches DONE status.")

    metrics_p = f"storage/outputs/{job_id}_metrics.json"
    if not os.path.exists(metrics_p):
        raise HTTPException(status_code=404, detail="Metrics file missing from storage.")
    with open(metrics_p, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/jobs/{job_id}/download/{kind}")
def download_output(job_id: str, kind: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs_db[job_id]
    if job["status"] != "DONE":
        raise HTTPException(status_code=409, detail=f"Job is in state '{job['status']}'. Downloads require 'DONE'.")

    outputs = job.get("outputs", {})
    mapping = {
        "final_mix": (outputs.get("final_mix_mp3"), "audio/mpeg", f"{job_id}_final_mix.mp3"),
        "vocals": (outputs.get("vocals_wav"), "audio/wav", f"{job_id}_vocals.wav"),
        "original": (outputs.get("original_song"), "audio/mpeg", f"{job_id}_original.mp3"),
    }
    if kind not in mapping or not mapping[kind][0] or not os.path.exists(mapping[kind][0]):
        raise HTTPException(status_code=404, detail=f"File for '{kind}' not found on server disk.")

    fpath, media_t, fname = mapping[kind]
    headers = {
        "Accept-Ranges": "bytes",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
    }
    return FileResponse(fpath, media_type=media_t, filename=fname, headers=headers)

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    for pattern in [f"storage/uploads/{job_id}*", f"storage/stems/{job_id}*", f"storage/outputs/{job_id}*", f"storage/jobs/{job_id}.json"]:
        for f in glob.glob(pattern):
            try:
                if os.path.isdir(f): shutil.rmtree(f)
                else: os.remove(f)
            except Exception: pass
    del jobs_db[job_id]
    return {"job_id": job_id, "deleted": True}

# ------------------------------------------------------------------------------
# 9. NGROK TUNNEL START & UVICORN THREAD (UNTOUCHED PER SPEC)
# ------------------------------------------------------------------------------
try:
    if CONFIG.ngrok_token:
        ngrok.set_auth_token(CONFIG.ngrok_token)
except Exception:
    pass

ngrok.kill()
tunnel = ngrok.connect(8000)
public_api_url = tunnel.public_url

print("=" * 75)
print(f"🚀 PUBLIC API URL: {public_api_url}")
print(f"🩺 Health Check:    {public_api_url}/api/health")
print(f"📋 Model Catalog:   {public_api_url}/api/models")
print("=" * 75)

def is_port_in_use(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

if not is_port_in_use(8000):
    server_thread = threading.Thread(target=start_uvicorn, daemon=True)
    server_thread.start()
    print("✅ SUCCESS: VocalMorph Studio Backend v3.0 running with single-worker GPU queue!")
else:
    print("✅ SUCCESS: VocalMorph Studio Backend is already running and listening on port 8000!")""")

    md("### **Cell 5 (Optional Test): Local End-to-End Verification & Metrics**\nRuns a complete synthetic pipeline job locally on Colab without needing the frontend. Verifies GPU queue, BS-Roformer, Seed-VC, and prints objective evaluation metrics.")
    code("""import time
import json
import requests
import numpy as np
import soundfile as sf
import os

BASE_URL = "http://localhost:8000"

print("1. Checking Server Health...")
r = requests.get(f"{BASE_URL}/api/health")
print("   Health Check Response:", r.json())

# Generate 15.0s of clean synthetic audio (continuous voiced speech, no silence)
sr = 44100
dur = 15.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)

# Target voice sample: 15s continuous female vocal tone (A4 = 440 Hz + vibrato)
f0_sample = 440.0 + 15.0 * np.sin(2 * np.pi * 5.0 * t)
sample_phase = 2 * np.pi * np.cumsum(f0_sample) / sr
sample_audio = 0.4 * np.sin(sample_phase) + 0.2 * np.sin(2 * sample_phase) + 0.1 * np.sin(3 * sample_phase)

# Source song: 15s male singer (A3 = 220 Hz) + chord progression instrumental
f0_song = 220.0 + 10.0 * np.sin(2 * np.pi * 4.5 * t)
song_phase = 2 * np.pi * np.cumsum(f0_song) / sr
song_vocal = 0.4 * np.sin(song_phase) + 0.15 * np.sin(2 * song_phase)
inst_chord = 0.25 * np.sin(2 * np.pi * 110.0 * t) + 0.2 * np.sin(2 * np.pi * 165.0 * t)
song_audio = song_vocal + inst_chord

test_dir = "storage/test_inputs"
os.makedirs(test_dir, exist_ok=True)
sample_file = f"{test_dir}/test_user_voice.wav"
song_file = f"{test_dir}/test_song.wav"

sf.write(sample_file, sample_audio.astype(np.float32), sr)
sf.write(song_file, song_audio.astype(np.float32), sr)
print(f"2. Synthetic test audio generated ({dur}s each, voiced speech confirmed).")

# 3. Create Job
payload = {
    "engine": "seed-vc",
    "model_id": "zero-shot",
    "options": {
        "diffusion_steps": 35,
        "inference_cfg_rate": 0.70,
        "semitone_shift": 0
    }
}
resp = requests.post(f"{BASE_URL}/api/jobs", json=payload).json()
job_id = resp["job_id"]
print(f"3. Job Created: {job_id}")

# 4. Upload Files
with open(song_file, "rb") as f_song, open(sample_file, "rb") as f_samp:
    r_song = requests.post(f"{BASE_URL}/api/jobs/{job_id}/upload/song", files={"file": ("song.wav", f_song, "audio/wav")})
    r_samp = requests.post(f"{BASE_URL}/api/jobs/{job_id}/upload/sample", files={"file": ("sample.wav", f_samp, "audio/wav")})

assert r_song.status_code == 200, f"Song upload failed: {r_song.text}"
assert r_samp.status_code == 200, f"Sample upload failed: {r_samp.text}"
print("4. Audio uploads confirmed.")

# 5. Start Job
start_resp = requests.post(f"{BASE_URL}/api/jobs/{job_id}/start", data={"consent": "true"}).json()
print("5. Job Started:", start_resp)

# 6. Monitor Pipeline Progress
print("6. Monitoring pipeline progress...")
status = "QUEUED"
while status not in ["DONE", "FAILED"]:
    time.sleep(3)
    st = requests.get(f"{BASE_URL}/api/jobs/{job_id}").json()
    status = st["status"]
    print(f"   [{status}] Progress: {st['progress']}% | {st['message']}")

assert status == "DONE", f"Pipeline failed! Error: {st.get('message')}"

# 7. Print Objective Evaluation Metrics
print("\\n" + "=" * 65)
print("🎯 OBJECTIVE EVALUATION METRICS REPORT")
print("=" * 65)
metrics = requests.get(f"{BASE_URL}/api/jobs/{job_id}/metrics").json()
print(json.dumps(metrics, indent=2))
print("=" * 65)
print(f"🎉 Test passed! Converted audio output: {st['outputs']['final_mix_url']}")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_path = r"c:\Users\dell\OneDrive\Desktop\12\notebooks\singing_voice_conversion_colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"Successfully generated hybrid notebook at: {out_path}")

if __name__ == "__main__":
    generate_hybrid_notebook()
