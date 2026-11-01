import os
import sys
import gc
import re
import glob
import time
import json
import math
import uuid
import shutil
import queue
import socket
import logging
import threading
import subprocess
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

# Ensure we are always in /content on Google Colab or root workspace
if os.path.exists("/content"):
    try:
        os.chdir("/content")
    except Exception:
        pass

import torch
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import noisereduce as nr
import torchcrepe
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.ndimage import uniform_filter1d

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from pyngrok import ngrok

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Safe port cleanup: Terminate orphaned processes on port 8000, NEVER kill current notebook kernel!
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
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
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
    raise RuntimeError("Not enough free VRAM: another process is holding the GPU. Please restart Colab session.")

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
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_song_duration_sec: float = 600.0         # 10 minutes
    min_sample_speech_sec: float = 3.0           # 3.0 seconds clean active speech
    sample_rate: int = 44100
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    ttl_hours: float = 24.0
    ngrok_token: str = ""

CONFIG = ServerConfig()

for d in ["storage/uploads", "storage/stems", "storage/outputs", "storage/jobs", "models/presets"]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. AUDIO SEPARATION (ISOLATED SUBPROCESSES) & REFERENCE PREPARATION
# ------------------------------------------------------------------------------
def run_audio_separator(input_path: str, model_filename: str, out_dir: str, timeout: int = 1200) -> List[str]:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [
        "audio-separator", input_path,
        "--model_filename", model_filename,
        "--output_dir", out_dir,
        "--output_format", "WAV"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if res.returncode != 0:
        err_msg = (res.stderr or res.stdout or "Unknown error")[-600:]
        raise RuntimeError(f"audio-separator failed ({model_filename}): {err_msg}")
    files = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    if not files:
        raise RuntimeError(f"No output produced by {model_filename}")
    return files

def pick_stem(files: List[str], *keys: str) -> Optional[str]:
    for key in keys:
        key_lower = key.lower()
        for f in files:
            if key_lower in os.path.basename(f).lower():
                return f
    logger.warning(f"[pick_stem] Keys {keys} not found in output files: {[os.path.basename(f) for f in files]}")
    return None

def run_roformer_separation(song_path: str, output_dir: str, timeout: int = 1200) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(song_path))[0]
    out_vocals = os.path.join(output_dir, f"{base}_vocals.wav")
    out_inst = os.path.join(output_dir, f"{base}_instrumental.wav")
    try:
        logger.info("[Separation] Running BS-Roformer (isolated subprocess)...")
        files = run_audio_separator(
            song_path,
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            os.path.join(output_dir, "_bsr"),
            timeout=timeout
        )
        v = pick_stem(files, "(vocals)", "vocals")
        i = pick_stem(files, "(instrumental)", "instrumental")
        if v and i:
            shutil.copy2(v, out_vocals)
            shutil.copy2(i, out_inst)
            logger.info("✅ [Separation] BS-Roformer vocal isolation successful.")
            return out_vocals, out_inst
        logger.warning("[Separation] Could not identify BS-Roformer stems; falling back to Demucs FT...")
    except Exception as e:
        logger.warning(f"⚠️ BS-Roformer execution failed ({e}). Falling back to Demucs FT...")

    return run_demucs_separation(song_path, output_dir, timeout=timeout)

def run_demucs_separation(song_path: str, output_dir: str, timeout: int = 600) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(song_path))[0]
    out_vocals = os.path.join(output_dir, f"{base}_vocals.wav")
    out_inst = os.path.join(output_dir, f"{base}_instrumental.wav")
    logger.info("[Separation] Running Demucs FT (htdemucs_ft)...")
    cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs_ft", "-o", output_dir, song_path]
    subprocess.run(cmd, check=True, timeout=timeout)
    demucs_vocals = os.path.join(output_dir, "htdemucs_ft", base, "vocals.wav")
    demucs_inst = os.path.join(output_dir, "htdemucs_ft", base, "no_vocals.wav")
    shutil.copy2(demucs_vocals, out_vocals)
    shutil.copy2(demucs_inst, out_inst)
    logger.info("✅ [Separation] Demucs stem separation successful.")
    return out_vocals, out_inst

def de_reverb_vocal(vocals_path: str, output_path: str, timeout: int = 600) -> str:
    try:
        files = run_audio_separator(
            vocals_path,
            "UVR-De-Echo-Normal.onnx",
            os.path.join(os.path.dirname(output_path), "_deecho"),
            timeout=timeout
        )
        dry = pick_stem(files, "(no echo)", "no echo", "dry")
        if dry:
            shutil.copy2(dry, output_path)
            logger.info("✅ [De-reverb] UVR De-Echo applied successfully.")
            return output_path
        logger.warning("[De-reverb] Dry stem not identified from UVR-De-Echo output; retaining original.")
    except Exception as e:
        logger.warning(f"[De-reverb] Skipped due to error: {e}")
    shutil.copy2(vocals_path, output_path)
    return output_path

def split_lead_and_backing(vocals_path: str, output_dir: str, timeout: int = 800) -> Tuple[str, Optional[str]]:
    try:
        files = run_audio_separator(
            vocals_path,
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            os.path.join(output_dir, "_karaoke"),
            timeout=timeout
        )
        lead = pick_stem(files, "(vocals)", "lead")
        backing = pick_stem(files, "(instrumental)", "backing", "harmony")
        if lead:
            logger.info("✅ [Karaoke] Lead and backing harmony split successfully.")
            return lead, backing
    except Exception as e:
        logger.info(f"[Karaoke] Skipped due to error: {e}. Using full vocal stem as lead.")
    return vocals_path, None

def transcode_to_clean_wav(input_path: str, output_wav_path: str) -> None:
    """Uses FFmpeg to decode any uploaded audio format (webm, mp3, m4a, ogg, etc.) to 44.1kHz 16-bit WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_s16le",
        output_wav_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed: {res.stderr[-300:]}")

def prepare_reference_voice(ref_path: str, cleaned_out_path: str) -> Tuple[str, Dict[str, Any]]:
    y, sr = librosa.load(ref_path, sr=CONFIG.sample_rate, mono=True)

    clipping_ratio = np.mean(np.abs(y) >= 0.999)
    if clipping_ratio > 0.05:
        logger.warning(f"Voice reference has high clipping ({clipping_ratio*100:.1f}%). Proceeding with normalization.")

    sos = butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    y_filtered = sosfilt(sos, y)
    y_clean = nr.reduce_noise(y=y_filtered, sr=sr, stationary=True, prop_decrease=0.75)

    intervals = librosa.effects.split(y_clean, top_db=25)
    if len(intervals) == 0:
        raise ValueError("Audio appears completely silent or contains only background noise.")

    voiced_segments = [y_clean[start:end] for start, end in intervals if (end - start) >= int(0.12 * sr)]
    if not voiced_segments:
        raise ValueError("No distinct vocal phrases detected in reference audio.")

    voiced_concat = np.concatenate(voiced_segments)
    voiced_duration = len(voiced_concat) / sr
    if voiced_duration < CONFIG.min_sample_speech_sec:
        raise ValueError(
            f"Insufficient voice duration: found {voiced_duration:.1f}s of active speech. "
            f"A minimum of {CONFIG.min_sample_speech_sec}s is required for accurate voice cloning."
        )

    target_samples = int(18.0 * sr)
    best_window = voiced_concat[:target_samples] if len(voiced_concat) > target_samples else voiced_concat

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
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=45.0)
        audio_tensor = torch.tensor(y).unsqueeze(0).to(device)

        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
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

        confident_mask = (periodicity >= 0.45) & (pitch > 65.0)
        confident_f0 = pitch[confident_mask]
        if len(confident_f0) < 10:
            return 0.0
        return float(np.median(confident_f0))
    except Exception as e:
        logger.warning(f"[Pitch Track] torchcrepe pitch tracking failed ({e}); returning 0.0")
        return 0.0

def compute_smart_key_shift(
    src_vocals: str,
    target_sample: str,
    user_shift: int = 0
) -> Tuple[int, List[str]]:
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

    if raw_gap >= 3.5:
        shift = int(round(raw_gap))
        shift = max(4, min(12, shift))
        logger.info(f"[Pitch Lock] Shifting vocal by +{shift} st into natural female register.")
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
# 3. SEED-VC ENVIRONMENT & INFERENCE EXECUTION
# ------------------------------------------------------------------------------
def get_seed_vc_environment() -> Tuple[str, str, str]:
    """
    Locates seed-vc directory, checkpoint file, and config file.
    Auto-clones seed-vc and auto-downloads model weights if missing.
    Returns: (seed_vc_dir, abs_ckpt, abs_config)
    """
    if os.path.exists("/content"):
        try:
            os.chdir("/content")
        except Exception:
            pass

    # 1. Locate seed-vc directory
    seed_vc_dir = None
    candidates = [
        "/content/seed-vc",
        os.path.abspath("seed-vc"),
        os.path.abspath(".")
    ]
    for cand in candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "inference.py")):
            seed_vc_dir = os.path.abspath(cand)
            break

    if seed_vc_dir is None:
        target = "/content/seed-vc" if os.path.exists("/content") else os.path.abspath("seed-vc")
        logger.info(f"📥 Seed-VC directory not found. Auto-cloning to {target}...")
        subprocess.run(["git", "clone", "https://github.com/Plachtaa/seed-vc.git", target], check=True)
        seed_vc_dir = os.path.abspath(target)

    # 2. In-place BigVGAN patch check
    bigvgan_path = os.path.join(seed_vc_dir, "modules", "bigvgan", "bigvgan.py")
    if os.path.isfile(bigvgan_path):
        try:
            with open(bigvgan_path, "r", encoding="utf-8") as f:
                bcode = f.read()
            if "proxies: Optional[Dict] = None" not in bcode:
                logger.info("Applying safe BigVGAN signature patch...")
                bcode = re.sub(r'proxies\s*:\s*Optional\[Dict\]\s*,', 'proxies: Optional[Dict] = None,', bcode)
                bcode = re.sub(r'resume_download\s*:\s*bool\s*,', 'resume_download: bool = False,', bcode)
                bcode = re.sub(r'revision\s*:\s*str\s*,', 'revision: str = "main",', bcode)
                bcode = re.sub(r'cache_dir\s*:\s*str\s*,', 'cache_dir: Optional[str] = None,', bcode)
                bcode = re.sub(r'force_download\s*:\s*bool\s*,', 'force_download: bool = False,', bcode)
                bcode = re.sub(r'local_files_only\s*:\s*bool\s*,', 'local_files_only: bool = False,', bcode)
                bcode = re.sub(r'token\s*:\s*Union\[[^\]]+\]\s*,', 'token: Union[str, bool, None] = None,', bcode)
                bcode = re.sub(r'torch\.load\(([^,\)]+),\s*map_location=([^,\)]+)\)', r'torch.load(\1, map_location=\2, weights_only=False)', bcode)
                with open(bigvgan_path, "w", encoding="utf-8") as f:
                    f.write(bcode)
        except Exception as e:
            logger.warning(f"BigVGAN patch check notice: {e}")

    # 3. Locate checkpoints & config
    ckpt_name = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
    config_name = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

    ckpt_candidates = [
        os.path.join(seed_vc_dir, "models", ckpt_name),
        os.path.abspath(os.path.join("models", ckpt_name)),
        f"/content/models/{ckpt_name}",
        f"/content/seed-vc/models/{ckpt_name}",
    ]
    ckpt_path = next((os.path.abspath(p) for p in ckpt_candidates if os.path.isfile(p)), None)

    config_candidates = [
        os.path.join(seed_vc_dir, "models", config_name),
        os.path.abspath(os.path.join("models", config_name)),
        f"/content/models/{config_name}",
        f"/content/seed-vc/models/{config_name}",
    ]
    config_path = next((os.path.abspath(p) for p in config_candidates if os.path.isfile(p)), None)

    if ckpt_path is None or config_path is None:
        logger.info("📥 DiT weights or config missing. Auto-downloading from HuggingFace...")
        os.makedirs("models", exist_ok=True)
        from huggingface_hub import hf_hub_download
        ckpt_path = os.path.abspath(hf_hub_download(repo_id="Plachta/Seed-VC", filename=ckpt_name, local_dir="models"))
        config_path = os.path.abspath(hf_hub_download(repo_id="Plachta/Seed-VC", filename=config_name, local_dir="models"))
        for aux_repo, aux_file in [("funasr/campplus", "campplus_cn_common.bin"), ("lj1995/VoiceConversionWebUI", "rmvpe.pt")]:
            try:
                hf_hub_download(repo_id=aux_repo, filename=aux_file, local_dir="models")
            except Exception:
                pass

        # Mirror models into seed_vc_dir/models
        os.makedirs(os.path.join(seed_vc_dir, "models"), exist_ok=True)
        for fn in [ckpt_name, config_name, "campplus_cn_common.bin", "rmvpe.pt"]:
            src = os.path.abspath(os.path.join("models", fn))
            dst = os.path.abspath(os.path.join(seed_vc_dir, "models", fn))
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

    return seed_vc_dir, ckpt_path, config_path

class SeedVCOutput(str):
    """Hybrid return object that acts as a string filepath or unpacks as (filepath, applied_shift, ref_path)."""
    def __new__(cls, val, shift=0, ref=""):
        obj = str.__new__(cls, val)
        obj.applied_shift = shift
        obj.reference_path = ref
        return obj

    def __iter__(self):
        return iter([str(self), self.applied_shift, self.reference_path])

def execute_seed_vc(
    vocals_path: str,
    reference_path: Optional[str] = None,
    output_dir: str = "",
    diffusion_steps: int = 50,
    inference_cfg_rate: float = 0.70,
    semitone_shift: int = 0,
    reference_sample_path: Optional[str] = None,
    **kwargs
) -> SeedVCOutput:
    actual_ref = reference_path or reference_sample_path
    if not actual_ref:
        raise ValueError("execute_seed_vc requires a valid reference audio path (reference_path or reference_sample_path).")

    seed_vc_dir, abs_ckpt, abs_config = get_seed_vc_environment()
    os.makedirs(output_dir, exist_ok=True)
    abs_vocals = os.path.abspath(vocals_path)
    abs_target = os.path.abspath(actual_ref)
    abs_output = os.path.abspath(output_dir)

    # Dynamic timeout based on audio duration: max(900, 6 * duration)
    try:
        dur = sf.info(abs_vocals).duration
        timeout_sec = max(900, int(6.0 * dur))
    except Exception:
        timeout_sec = 1200

    def run_inference_cmd(steps: int, cfg: float):
        cmd = [
            sys.executable, "inference.py",
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
        sub_env = os.environ.copy()
        sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        sub_env["PYTHONPATH"] = f"{seed_vc_dir}:{sub_env.get('PYTHONPATH', '')}"

        res = subprocess.run(cmd, cwd=seed_vc_dir, capture_output=True, text=True, timeout=timeout_sec, env=sub_env)
        if res.returncode != 0:
            full_err = res.stderr or res.stdout or "Subprocess exited with failure."
            logger.error(f"[Seed-VC Full Error Output]:\n{full_err}")
            raise RuntimeError(f"Seed-VC failed with exit code {res.returncode}:\n{full_err.strip()}")

    logger.info(f"[Seed-VC] Running inference (steps={diffusion_steps}, cfg={inference_cfg_rate}, shift={semitone_shift:+d}, timeout={timeout_sec}s)...")
    run_inference_cmd(diffusion_steps, inference_cfg_rate)

    converted = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.endswith(".wav") and not f.startswith("cleaned_ref")
    ]
    if not converted:
        raise RuntimeError("Seed-VC did not produce any audio output.")

    out_file = sorted(converted, key=os.path.getmtime)[-1]
    y_test, _ = sf.read(out_file)

    if np.isnan(y_test).any() or np.mean(y_test**2) < 1e-5:
        logger.warning("⚠️ Seed-VC produced invalid/silent audio. Retrying once with conservative parameters (steps=40, cfg=0.60)...")
        run_inference_cmd(40, 0.60)
        converted = [
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.endswith(".wav") and not f.startswith("cleaned_ref")
        ]
        out_file = sorted(converted, key=os.path.getmtime)[-1]

    return SeedVCOutput(out_file, semitone_shift, abs_target)

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
        logger.warning(f"Rubberband pitch shift failed ({e}), falling back to librosa.")
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
    # Fast O(N) smoothing instead of O(N*M) np.convolve
    gain = uniform_filter1d(gain, size=64, mode="constant")
    return (vocal - sibilance) + (sibilance * gain)

def apply_soft_compressor(vocal: np.ndarray, threshold_db: float = -18.0, ratio: float = 2.5) -> np.ndarray:
    # Fast O(N) running RMS filter
    rms = np.sqrt(uniform_filter1d(vocal**2, size=512, mode="constant") + 1e-9)
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

def master_and_mix_track(
    converted_vocals_path: str,
    instrumental_path: str,
    backing_vocals_path: Optional[str] = None,
    out_wav: Optional[str] = None,
    out_mp3: Optional[str] = None,
    applied_shift: int = 0,
    output_mix_wav: Optional[str] = None,
    output_mix_mp3: Optional[str] = None,
    **kwargs
) -> Tuple[str, str]:
    target_wav = out_wav or output_mix_wav
    target_mp3 = out_mp3 or output_mix_mp3
    if not target_wav or not target_mp3:
        raise ValueError("master_and_mix requires valid output paths for WAV and MP3.")

    sr = CONFIG.sample_rate
    lead, _ = librosa.load(converted_vocals_path, sr=sr, mono=False)
    inst, _ = librosa.load(instrumental_path, sr=sr, mono=False)
    lead = np.column_stack([lead, lead]) if lead.ndim == 1 else lead.T
    inst = np.column_stack([inst, inst]) if inst.ndim == 1 else inst.T

    # Keep music in harmonic alignment with shifted vocals (octave-folded to -6..+5)
    inst_shift = ((applied_shift + 6) % 12) - 6
    if inst_shift != 0:
        logger.info(f"[RubberBand] Harmonically shifting instrumental by {inst_shift:+d} st")
        inst = np.column_stack([
            rubberband_pitch_shift(inst[:, 0], sr, inst_shift),
            rubberband_pitch_shift(inst[:, 1], sr, inst_shift)
        ])

    if backing_vocals_path and os.path.exists(backing_vocals_path):
        bv, _ = librosa.load(backing_vocals_path, sr=sr, mono=False)
        bv = np.column_stack([bv, bv]) if bv.ndim == 1 else bv.T
        if inst_shift != 0:
            bv = np.column_stack([
                rubberband_pitch_shift(bv[:, 0], sr, inst_shift),
                rubberband_pitch_shift(bv[:, 1], sr, inst_shift)
            ])
        n_bv = min(len(inst), len(bv))
        inst[:n_bv] += bv[:n_bv] * 0.79

    n = min(len(lead), len(inst))
    lead, inst = lead[:n], inst[:n]

    sos_hp = butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    for ch in (0, 1):
        lead[:, ch] = sosfilt(sos_hp, lead[:, ch])
        lead[:, ch] = apply_deesser(lead[:, ch], sr)
        lead[:, ch] = apply_soft_compressor(lead[:, ch])
    lead_fx = synthesize_plate_reverb(lead, sr, wet_level=0.08)

    # Dynamic sidechain ducking with fast O(N) uniform filter
    rms = np.sqrt(np.mean(lead**2, axis=1))
    win = int(0.08 * sr)
    env = uniform_filter1d(rms, size=win, mode="constant")
    inst = inst * np.where(env > 0.02, 0.84, 1.0)[:, None]

    mixed = lead_fx * 1.12 + inst * 0.88
    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(mixed)
        if not (np.isnan(loud) or np.isneginf(loud)):
            mixed = pyln.normalize.loudness(mixed, loud, CONFIG.target_lufs)
    except Exception as e:
        logger.warning(f"[Mastering] Loudness normalization warning: {e}")

    ceiling = 10 ** (CONFIG.true_peak_db / 20.0)
    peak = np.max(np.abs(mixed))
    if peak > ceiling:
        mixed = mixed / peak * ceiling

    sf.write(target_wav, mixed, sr, subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-y", "-i", target_wav, "-codec:a", "libmp3lame", "-b:a", "320k", target_mp3],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return target_wav, target_mp3

master_and_mix = master_and_mix_track

def separate_vocals_pipeline(
    song_path: str,
    output_dir: str,
    separator_type: str = "bs-roformer",
    use_karaoke: bool = False,
    **kwargs
) -> Tuple[str, str]:
    if "demucs" in str(separator_type).lower():
        return run_demucs_separation(song_path, output_dir)
    return run_roformer_separation(song_path, output_dir)

separate_vocals = separate_vocals_pipeline

def calculate_exact_gender_and_key_shift(vocals_path: str, ref_path: str, user_shift: int = 0) -> int:
    shift, _ = compute_smart_key_shift(vocals_path, ref_path, user_shift=user_shift)
    return shift

estimate_voice_pitch_hz = estimate_pitch_torchcrepe

# ------------------------------------------------------------------------------
# 5. OBJECTIVE EVALUATION METRICS ENGINE
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
                logger.info(f"SpeechBrain ECAPA-TDNN not available ({e}). Using MFCC fallback.")
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
        metrics = {}

        # 1. Speaker Cosine Similarity
        try:
            w_conv, _ = librosa.load(converted_vocal_path, sr=16000, mono=True)
            w_user, _ = librosa.load(user_sample_path, sr=16000, mono=True)
            w_src, _ = librosa.load(source_vocal_path, sr=16000, mono=True)

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

        # 2. Pitch Accuracy
        try:
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

        # 3. Whisper Word Error Rate
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

        # 4. Audio Quality & Loudness
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
    if job_id in jobs_db:
        path = f"storage/jobs/{job_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jobs_db[job_id], f, indent=2)

def load_persisted_jobs():
    for p in glob.glob("storage/jobs/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
                jobs_db[j["job_id"]] = j
        except Exception:
            pass

load_persisted_jobs()

def cleanup_daemon_loop():
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
def async_compute_metrics(job_id: str, conv_vocals: str, cleaned_ref: str, lead_vocals: str, applied_shift: int):
    """Computes evaluation metrics in background without blocking DONE status or audio playback."""
    try:
        logger.info(f"[Metrics:{job_id}] Computing objective evaluation metrics in background...")
        metrics = MetricEvaluator.compute_all_metrics(
            converted_vocal_path=conv_vocals,
            user_sample_path=cleaned_ref,
            source_vocal_path=lead_vocals,
            applied_shift=applied_shift
        )
        metrics["status"] = "ready"
        metrics_path = f"storage/outputs/{job_id}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if job_id in jobs_db:
            jobs_db[job_id]["metrics"] = metrics
            save_job_state(job_id)
        logger.info(f"✅ [Metrics:{job_id}] Metrics calculation completed and saved.")
    except Exception as e:
        logger.warning(f"[Metrics:{job_id}] Failed: {e}")
        err_metrics = {"status": "failed", "error": str(e)}
        with open(f"storage/outputs/{job_id}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(err_metrics, f)
    finally:
        MetricEvaluator._whisper_model = None
        MetricEvaluator._spk_classifier = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def gpu_pipeline_worker(stop_event: threading.Event):
    logger.info("👷 [GPU Worker] Pipeline worker started and waiting for jobs.")
    while not stop_event.is_set():
        try:
            job_id = job_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        if job_id is None:
            break

        job = jobs_db.get(job_id)
        if not job:
            job_queue.task_done()
            continue

        job["start_time"] = time.time()
        job["queue_position"] = 0
        job["stage_log"] = []
        job["failed_stage"] = None
        save_job_state(job_id)

        def record_stage(stage: str, msg: str, progress: int):
            t_elapsed = round(time.time() - job["start_time"], 1)
            job["status"] = stage
            job["progress"] = progress
            job["message"] = msg
            job["stage_log"].append({"stage": stage, "elapsed_sec": t_elapsed, "message": msg})
            save_job_state(job_id)

        try:
            with active_processing_lock:
                logger.info(f"▶️ [Pipeline Start] Processing job {job_id}...")
                opts = job.get("options", {})
                stems_dir = f"storage/stems/{job_id}"
                os.makedirs(stems_dir, exist_ok=True)

                # Stage 1: Validation
                record_stage("VALIDATING", "Validating reference audio and analyzing speech properties...", 5)
                cleaned_ref = os.path.join(stems_dir, "cleaned_ref.wav")
                _, ref_stats = prepare_reference_voice(job["sample_path"], cleaned_ref)
                job["ref_stats"] = ref_stats
                save_job_state(job_id)

                # Stage 2: Stem Separation
                separator_choice = str(opts.get("separator", "bs-roformer")).lower()
                record_stage("SEPARATING", f"Isolating lead vocal stem via {separator_choice}...", 20)

                if "demucs" in separator_choice:
                    raw_vocals, inst_path = run_demucs_separation(job["song_path"], stems_dir)
                else:
                    raw_vocals, inst_path = run_roformer_separation(job["song_path"], stems_dir)

                # De-reverb option
                if opts.get("use_dereverb", False):
                    record_stage("SEPARATING", "Applying UVR De-Echo room de-reverberation...", 35)
                    dry_vocals = de_reverb_vocal(raw_vocals, os.path.join(stems_dir, "dry_vocals.wav"))
                else:
                    dry_vocals = raw_vocals

                # Lead / Backing separation option
                if opts.get("use_karaoke", True):
                    record_stage("SEPARATING", "Splitting lead vocals and backing harmonies...", 45)
                    lead_vocals, backing_vocals = split_lead_and_backing(dry_vocals, stems_dir)
                else:
                    lead_vocals = dry_vocals
                    backing_vocals = None

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stage 3: Voice Conversion
                record_stage("CONVERTING", "Calculating gender pitch shift & executing Seed-VC synthesis...", 50)
                applied_shift, warnings = compute_smart_key_shift(
                    lead_vocals,
                    cleaned_ref,
                    user_shift=int(opts.get("semitone_shift", 0))
                )
                job["applied_shift"] = applied_shift
                job["warnings"] = warnings
                save_job_state(job_id)

                log_gpu_state("before-seedvc")
                wait_for_vram(min_free_gb=3.5, timeout=45)

                vc_dir = os.path.join(stems_dir, "vc_out")
                diff_steps = max(10, min(100, int(opts.get("diffusion_steps", 50))))
                cfg_rate = max(0.1, min(1.5, float(opts.get("inference_cfg_rate", 0.70))))

                conv_vocals = execute_seed_vc(
                    vocals_path=lead_vocals,
                    reference_path=cleaned_ref,
                    output_dir=vc_dir,
                    diffusion_steps=diff_steps,
                    inference_cfg_rate=cfg_rate,
                    semitone_shift=applied_shift
                )

                # Stage 4: Mixing & Studio Mastering
                record_stage("MIXING", "Applying professional vocal chain, sidechain ducking & -14 LUFS mastering...", 80)
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

                # Stage 5: Mark Job DONE Immediately!
                job["status"] = "DONE"
                job["progress"] = 100
                job["message"] = "Conversion and studio mastering completed!"
                job["elapsed_seconds"] = round(time.time() - job["start_time"], 2)
                job["outputs"] = {
                    "final_mix_url": f"/api/jobs/{job_id}/download/final_mix?t={int(time.time())}",
                    "vocals_url": f"/api/jobs/{job_id}/download/vocals?t={int(time.time())}",
                    "original_song_url": f"/api/jobs/{job_id}/download/original?t={int(time.time())}"
                }
                job["disk_outputs"] = {
                    "final_mix_mp3": out_mp3,
                    "final_mix_wav": out_wav,
                    "vocals_wav": conv_vocals,
                    "original_song": job["song_path"]
                }
                record_stage("DONE", "Conversion and studio mastering completed!", 100)
                save_job_state(job_id)
                logger.info(f"✅ [Pipeline Finished] Job {job_id} done in {job['elapsed_seconds']}s.")

                # Launch metrics in non-blocking daemon thread
                m_thread = threading.Thread(
                    target=async_compute_metrics,
                    args=(job_id, conv_vocals, cleaned_ref, lead_vocals, applied_shift),
                    daemon=True
                )
                m_thread.start()

        except Exception as e:
            full_tb = traceback.format_exc()
            logger.error(f"❌ [Job {job_id} Failure]: {e}\n{full_tb}")
            job["status"] = "FAILED"
            job["progress"] = 0
            job["failed_stage"] = job.get("status", "UNKNOWN")
            job["message"] = f"Conversion failed: {str(e)}"
            job["error"] = str(e)
            job["error_traceback"] = full_tb
            save_job_state(job_id)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            job_queue.task_done()

# Handle thread hot-reloading when Cell 4 is re-run in Jupyter/Colab
if hasattr(sys, "_vocalmorph_worker_stop_event"):
    sys._vocalmorph_worker_stop_event.set()
    job_queue.put(None)
    time.sleep(0.5)

worker_stop_event = threading.Event()
sys._vocalmorph_worker_stop_event = worker_stop_event
worker_thread = threading.Thread(target=gpu_pipeline_worker, args=(worker_stop_event,), daemon=True)
worker_thread.start()
logger.info("✅ SUCCESS: Clean GPU pipeline worker started with generation management.")

# ------------------------------------------------------------------------------
# 8. FASTAPI REST APPLICATION & ENDPOINTS
# ------------------------------------------------------------------------------
app = FastAPI(title="VocalMorph Studio API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    semitone_shift: int = 0
    diffusion_steps: int = 50
    inference_cfg_rate: float = 0.70
    output_format: str = "mp3"
    use_dereverb: bool = False
    use_karaoke: bool = True
    separator: str = "bs-roformer"

class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
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
    return {
        "models": [
            {"id": "zero-shot", "name": "Your Voice (Instant Zero-Shot)", "type": "zero-shot", "description": "Instant clone using your uploaded reference voice."},
            {"id": "rvc-studio", "name": "Custom Studio Voice (RVC v2)", "type": "zero-shot", "description": "Trained studio voice model fallback mapped to Seed-VC."},
            {"id": "preset_voice_1", "name": "Preset Voice 1 (Male Pop)", "type": "preset", "description": "Pre-conditioned studio vocal timbre."},
            {"id": "preset_voice_2", "name": "Preset Voice 2 (Female Soprano)", "type": "preset", "description": "Pre-conditioned studio vocal timbre."},
            {"id": "preset_voice_3", "name": "Preset Voice 3 (Warm Baritone)", "type": "preset", "description": "Pre-conditioned studio vocal timbre."}
        ]
    }

@app.post("/api/jobs")
def create_job(request: Optional[JobCreateRequest] = None):
    # Tolerate frontend differences: accept whatever engine or model_id is provided
    req_engine = (request.engine if request and request.engine else "seed-vc").lower()
    if req_engine != "seed-vc":
        logger.warning(f"[API:Jobs] Unsupported engine '{req_engine}' requested. Gracefully routing to Seed-VC.")

    model_id = request.model_id if request and request.model_id else "zero-shot"
    job_id = str(uuid.uuid4())
    raw_opts = request.options.model_dump() if request and request.options else {}

    # Clamp options to safe bounds
    safe_opts = {
        "semitone_shift": int(raw_opts.get("semitone_shift", 0)),
        "diffusion_steps": max(10, min(100, int(raw_opts.get("diffusion_steps", 50)))),
        "inference_cfg_rate": max(0.1, min(1.5, float(raw_opts.get("inference_cfg_rate", 0.70)))),
        "output_format": str(raw_opts.get("output_format", "mp3")),
        "use_dereverb": bool(raw_opts.get("use_dereverb", False)),
        "use_karaoke": bool(raw_opts.get("use_karaoke", True)),
        "separator": str(raw_opts.get("separator", "bs-roformer")),
    }

    jobs_db[job_id] = {
        "job_id": job_id,
        "engine": "seed-vc",
        "model_id": model_id,
        "status": "CREATED",
        "progress": 0,
        "queue_position": 0,
        "message": "Job initialized. Ready for file uploads.",
        "song_path": None,
        "sample_path": None,
        "applied_shift": 0,
        "warnings": [],
        "options": safe_opts,
        "start_time": None,
        "elapsed_seconds": 0,
        "error": None,
        "error_traceback": None,
        "stage_log": [],
        "failed_stage": None,
        "outputs": None,
        "disk_outputs": {}
    }
    save_job_state(job_id)
    return {
        "job_id": job_id,
        "engine": "seed-vc",
        "status": "CREATED",
        "upload_song_url": f"/api/jobs/{job_id}/upload/song",
        "upload_sample_url": f"/api/jobs/{job_id}/upload/sample"
    }

def resolve_preset_path(model_id: str) -> Optional[str]:
    if not model_id or model_id in ["zero-shot", "custom"]:
        return None
    candidates = [
        f"models/presets/{model_id}.wav",
        f"models/presets/{model_id.replace('voice_', '')}.wav",
        f"models/presets/{model_id.replace('preset_', 'preset_voice_')}.wav",
        f"/content/models/presets/{model_id}.wav",
        f"/content/models/presets/{model_id.replace('voice_', '')}.wav",
        f"/content/models/presets/{model_id.replace('preset_', 'preset_voice_')}.wav",
        "/content/seed-vc/examples/reference/casual_female.wav",
        "seed-vc/examples/reference/casual_female.wav",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    presets = glob.glob("models/presets/*.wav") + glob.glob("/content/models/presets/*.wav")
    if presets:
        return os.path.abspath(presets[0])
    return None

@app.post("/api/jobs/{job_id}/upload/{file_type}")
async def upload_file(job_id: str, file_type: str, file: UploadFile = File(...)):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    if file_type not in ["song", "sample"]:
        raise HTTPException(status_code=400, detail="file_type must be either 'song' or 'sample'.")

    contents = await file.read()
    if len(contents) > CONFIG.max_file_size_bytes:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 100 MB.")

    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".tmp"
    raw_dest = f"storage/uploads/{job_id}_{file_type}_raw{orig_ext}"
    wav_dest = f"storage/uploads/{job_id}_{file_type}.wav"

    with open(raw_dest, "wb") as f:
        f.write(contents)

    # Transcode ANY uploaded file directly with FFmpeg to 44.1kHz 16-bit stereo WAV
    try:
        transcode_to_clean_wav(raw_dest, wav_dest)
        info = sf.info(wav_dest)
        if file_type == "song" and info.duration > CONFIG.max_song_duration_sec:
            os.remove(wav_dest)
            raise HTTPException(status_code=400, detail=f"Song duration ({info.duration:.1f}s) exceeds maximum allowed {CONFIG.max_song_duration_sec}s.")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio file: {e}")
    finally:
        if os.path.exists(raw_dest):
            try: os.remove(raw_dest)
            except Exception: pass

    jobs_db[job_id][f"{file_type}_path"] = wav_dest
    model_id = jobs_db[job_id]["model_id"]
    if not jobs_db[job_id]["sample_path"]:
        p_path = resolve_preset_path(model_id)
        if p_path:
            jobs_db[job_id]["sample_path"] = p_path

    if jobs_db[job_id]["song_path"] and jobs_db[job_id]["sample_path"]:
        jobs_db[job_id]["status"] = "UPLOADED"
        jobs_db[job_id]["message"] = "All audio files received. Ready to start."

    save_job_state(job_id)
    return {"job_id": job_id, "status": jobs_db[job_id]["status"]}

@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, consent: Optional[bool] = Form(True)):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs_db[job_id]
    if not job.get("sample_path"):
        p_path = resolve_preset_path(job.get("model_id", ""))
        if p_path:
            job["sample_path"] = p_path

    if not job.get("song_path") or not job.get("sample_path"):
        raise HTTPException(status_code=400, detail="Both 'song' and 'sample' files must be uploaded before starting.")

    pos = job_queue.qsize() + 1
    job["status"] = "QUEUED"
    job["queue_position"] = pos
    job["message"] = f"Job queued at position #{pos}."
    save_job_state(job_id)

    job_queue.put(job_id)
    return {"job_id": job_id, "status": "QUEUED", "queue_position": pos}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs_db[job_id]
    elapsed = round(time.time() - job["start_time"], 1) if job.get("start_time") and job["status"] not in ["DONE", "FAILED"] else job.get("elapsed_seconds", 0)

    return {
        "job_id": job_id,
        "engine": "seed-vc",
        "status": job["status"],
        "progress": job["progress"],
        "queue_position": job.get("queue_position", 0),
        "message": job["message"],
        "warnings": job.get("warnings", []),
        "applied_shift": job.get("applied_shift", 0),
        "elapsed_seconds": elapsed,
        "error": job.get("error"),
        "outputs": job.get("outputs")
    }

@app.get("/api/jobs/{job_id}/debug")
def get_job_debug(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs_db[job_id]
    stems_dir = f"storage/stems/{job_id}"
    stem_files = os.listdir(stems_dir) if os.path.exists(stems_dir) else []
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "failed_stage": job.get("failed_stage"),
        "stage_log": job.get("stage_log", []),
        "ref_stats": job.get("ref_stats"),
        "applied_shift": job.get("applied_shift"),
        "warnings": job.get("warnings", []),
        "stems": stem_files,
        "options": job.get("options", {}),
        "error": job.get("error"),
        "error_traceback": job.get("error_traceback")
    }

@app.get("/api/jobs/{job_id}/metrics")
def get_metrics(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    metrics_p = f"storage/outputs/{job_id}_metrics.json"
    if os.path.exists(metrics_p):
        with open(metrics_p, "r", encoding="utf-8") as f:
            return json.load(f)
    if jobs_db[job_id]["status"] == "DONE":
        return JSONResponse(status_code=202, content={"status": "pending", "message": "Metrics calculation in progress..."})
    raise HTTPException(status_code=409, detail=f"Job is in state '{jobs_db[job_id]['status']}'. Metrics require 'DONE'.")

@app.get("/api/jobs/{job_id}/download/{kind}")
def download_output(job_id: str, kind: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs_db[job_id]
    if job["status"] != "DONE":
        raise HTTPException(status_code=409, detail=f"Job is in state '{job['status']}'. Downloads require 'DONE'.")

    outputs = job.get("disk_outputs", {})
    mapping = {
        "final_mix": (outputs.get("final_mix_mp3"), "audio/mpeg", f"{job_id}_final_mix.mp3"),
        "vocals": (outputs.get("vocals_wav"), "audio/wav", f"{job_id}_vocals.wav"),
        "original": (outputs.get("original_song"), "audio/wav", f"{job_id}_original.wav"),
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
# 9. UVICORN SERVER & SAFE NGROK TUNNEL
# ------------------------------------------------------------------------------
MANUAL_NGROK_TOKEN = "3LDs1XFBKCDv5EvwjAfVPwJaJcv_2EutLASeNGyD84uchynqe"

def is_port_in_use(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

if not is_port_in_use(8000):
    server_thread = threading.Thread(target=start_uvicorn, daemon=True)
    server_thread.start()
    time.sleep(1.5)
    print("✅ SUCCESS: VocalMorph Studio Backend running on port 8000!")
else:
    print("✅ SUCCESS: VocalMorph Studio Backend is actively listening on port 8000!")

def get_ngrok_token():
    try:
        from google.colab import userdata
        token = userdata.get('NGROK_AUTHTOKEN') or userdata.get('ngrok_token')
        if token and token.strip():
            return token.strip()
    except Exception:
        pass
    if MANUAL_NGROK_TOKEN and MANUAL_NGROK_TOKEN.strip():
        return MANUAL_NGROK_TOKEN.strip()
    return os.environ.get("NGROK_AUTHTOKEN", "").strip()

public_api_url = "http://localhost:8000"
resolved_token = get_ngrok_token()

if resolved_token:
    try:
        ngrok.kill()
        time.sleep(1.0)

        # Clear old config or ignore if already set
        for cfg in ["/root/.config/ngrok/ngrok.yml", os.path.expanduser("~/.config/ngrok/ngrok.yml")]:
            if os.path.exists(cfg):
                try: os.remove(cfg)
                except Exception: pass

        try:
            ngrok.set_auth_token(resolved_token)
        except Exception as e:
            if "already set" not in str(e).lower():
                logger.warning(f"Notice setting auth token: {e}")

        tunnel = ngrok.connect(8000)
        public_api_url = tunnel.public_url
        print("=" * 75)
        print(f"🚀 PUBLIC NGROK URL: {public_api_url}")
        print(f"🩺 Health Check:    {public_api_url}/api/health")
        print(f"📋 Model Catalog:   {public_api_url}/api/models")
        print("=" * 75)
    except Exception as e:
        print(f"⚠️ Ngrok tunnel failed to connect: {e}")
        print("   Running locally on http://localhost:8000")
else:
    print("=" * 75)
    print("ℹ️ Running in Local Mode: http://localhost:8000")
    print("👉 Set MANUAL_NGROK_TOKEN above or add 'NGROK_AUTHTOKEN' to Colab Secrets (🔑).")
    print("=" * 75)
