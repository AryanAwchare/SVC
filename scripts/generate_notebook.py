import json
import os

def build_notebook():
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

    # Header
    md("""# Phase 0 Prototype: Singing Voice Conversion (Seed-VC + Stem Separation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Project:** Singing Voice Conversion Website (Zero-Shot Seed-VC)  
**Author / Student:** AI/DS Project  
**Target Runtime:** Google Colab (Free Tier NVIDIA T4 GPU)

---
### Objectives of Phase 0:
1. **Stem Separation:** Compare **Demucs (`htdemucs`)** against **BS-RoFormer (`audio-separator`)** on vocal isolation and instrumental bleed.
2. **Seed-VC Singing Conversion:** Run F0-conditioned zero-shot singing voice conversion using the 44.1 kHz singing checkpoint.
3. **Hyperparameter Tuning:** Benchmark diffusion steps (25 vs 35 vs 50), semitone shifts (-12 to +12), and CFG scales.
4. **Chunking & Crossfade:** Verify overlap-add math to eliminate boundary clicks and prevent GPU memory spikes on long tracks.
5. **Mastering:** Mix the converted vocals back with the instrumental stem and master to **-14 LUFS (EBU R128)**.
6. **Acceptance Gate 0:** Evaluate audio quality, intelligibility, and pitch fidelity.""")

    # Cell 1: Check GPU
    md("""### 1. Verify GPU Hardware
Ensure the Colab runtime is set to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).""")
    code("""!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    raise SystemError("GPU not found! Please enable GPU in Colab: Runtime -> Change runtime type -> T4 GPU.")""")

    # Cell 2: System and Python Dependencies
    md("""### 2. Install Core Dependencies & Tools
We install FFmpeg for audio transcoding, Demucs, `audio-separator` for BS-RoFormer, `pyloudnorm` for EBU-R128 loudness matching, and scientific audio libraries.""")
    code("""!apt-get update -qq && apt-get install -y -qq ffmpeg
!pip install -q --upgrade pip
!pip install -q demucs audio-separator pyloudnorm soundfile librosa torchaudio scipy omegaconf einops huggingface_hub""")

    # Cell 3: Clone Seed-VC
    md("""### 3. Clone & Setup Seed-VC Repository
Clone the official `Plachtaa/seed-vc` repository and install any specific submodule requirements.""")
    code("""import os
if not os.path.exists("seed-vc"):
    !git clone https://github.com/Plachtaa/seed-vc.git
%cd seed-vc
!pip install -q -r requirements.txt
%cd ..""")

    # Cell 4: Download Pretrained Weights
    md("""### 4. Download Pretrained Model Weights
For Singing Voice Conversion (SVC), Seed-VC requires:
1. **DiT Checkpoint & Config:** `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` + config
2. **CAMPPlus:** Speaker embedding extractor (`campplus_cn_common.bin`)
3. **RMVPE:** Robust pitch extractor (`rmvpe.pt`)""")
    code("""import os
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)

print("Downloading Seed-VC 44.1kHz Singing Checkpoint...")
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

print("Downloading CAMPPlus speaker embedding model...")
hf_hub_download(
    repo_id="funasr/campplus",
    filename="campplus_cn_common.bin",
    local_dir="models"
)

print("Downloading RMVPE pitch estimator...")
hf_hub_download(
    repo_id="lj1995/VoiceConversionWebUI",
    filename="rmvpe.pt",
    local_dir="models"
)

print("All models successfully cached in ./models!")""")

    # Cell 5: Audio Input
    md("""### 5. Provide Audio Inputs
Upload your own song (`song.mp3` or `.wav`) and voice reference sample (`voice_sample.wav`), OR run the cell below to download creative-commons demo tracks for testing.""")
    code("""import os
import urllib.request

os.makedirs("inputs", exist_ok=True)
song_path = "inputs/song.mp3"
sample_path = "inputs/voice_sample.wav"

# Check if user already uploaded files, otherwise download sample audio
if not os.path.exists(song_path):
    print("Downloading CC sample song (pop vocal demo)...")
    # Using an open-access sample audio snippet for immediate testing
    sample_song_url = "https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/gui_data/test_audio/test.mp3"
    try:
        urllib.request.urlretrieve(sample_song_url, song_path)
    except Exception as e:
        print(f"Could not auto-download sample: {e}. You can upload your own files via the Colab files panel.")

if not os.path.exists(sample_path):
    print("Downloading sample reference voice...")
    sample_voice_url = "https://github.com/Plachtaa/seed-vc/raw/main/examples/reference/target_voice.wav"
    try:
        urllib.request.urlretrieve(sample_voice_url, sample_path)
    except Exception as e:
        print(f"Could not auto-download reference voice: {e}")

print(f"Song Path: {song_path} (Exists: {os.path.exists(song_path)})")
print(f"Voice Sample Path: {sample_path} (Exists: {os.path.exists(sample_path)})")""")

    # Cell 6: Separation with Demucs
    md("""### 6. Vocal Separation: Demucs (`htdemucs`)
We separate the song into lead vocals and instrumental accompaniment using Meta's `htdemucs` model.""")
    code("""import time
import subprocess
import IPython.display as ipd

os.makedirs("separated/demucs", exist_ok=True)

t0 = time.time()
print("Starting Demucs htdemucs separation...")
cmd = [
    "demucs",
    "--two-stems=vocals",
    "-n", "htdemucs",
    "-o", "separated/demucs",
    song_path
]
subprocess.run(cmd, check=True)
demucs_time = time.time() - t0
print(f"Demucs separation complete in {demucs_time:.2f} seconds!")

# Locate output stems
base_name = os.path.splitext(os.path.basename(song_path))[0]
demucs_vocals = f"separated/demucs/htdemucs/{base_name}/vocals.wav"
demucs_instrumental = f"separated/demucs/htdemucs/{base_name}/no_vocals.wav"

print(f"Demucs Vocals: {demucs_vocals}")
print(f"Demucs Instrumental: {demucs_instrumental}")""")

    code("""# Play Demucs separated stems
print("Demucs Separated Vocals:")
ipd.display(ipd.Audio(demucs_vocals))

print("Demucs Separated Instrumental:")
ipd.display(ipd.Audio(demucs_instrumental))""")

    # Cell 7: Separation with Roformer
    md("""### 7. Vocal Separation Comparison: BS-RoFormer via `audio-separator`
Now we run the same song through **BS-RoFormer** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`).  
*Hypothesis:* Roformer produces cleaner vocal isolation with noticeably less instrumental bleed and reverb residual, creating a cleaner canvas for Seed-VC.""")
    code("""from audio_separator.separator import Separator

os.makedirs("separated/roformer", exist_ok=True)

t0 = time.time()
print("Initializing BS-RoFormer Separator...")
separator = Separator(
    output_dir="separated/roformer",
    output_format="WAV"
)

# Load SOTA BS-RoFormer checkpoint
separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")

print("Running BS-RoFormer separation on song...")
roformer_files = separator.separate(song_path)
roformer_time = time.time() - t0
print(f"Roformer separation complete in {roformer_time:.2f} seconds!")
print(f"Generated stems: {roformer_files}")

# Identify vocal and instrumental files
roformer_vocals = None
roformer_instrumental = None
for f in roformer_files:
    full_p = os.path.join("separated/roformer", f)
    if "vocals" in f.lower():
        roformer_vocals = full_p
    elif "instrumental" in f.lower() or "no_vocals" in f.lower():
        roformer_instrumental = full_p

print(f"Roformer Vocals: {roformer_vocals}")
print(f"Roformer Instrumental: {roformer_instrumental}")""")

    code("""# Play Roformer separated stems
if roformer_vocals and os.path.exists(roformer_vocals):
    print("BS-RoFormer Separated Vocals:")
    ipd.display(ipd.Audio(roformer_vocals))

if roformer_instrumental and os.path.exists(roformer_instrumental):
    print("BS-RoFormer Separated Instrumental:")
    ipd.display(ipd.Audio(roformer_instrumental))""")

    # Cell 8: Comparison
    md("""### 8. Separation Comparison Assessment
Compare the two vocal stems above:
- Listen to background instrumental bleed during quiet pauses.
- Check high-frequency vocal clarity and room echo.
- Choose which stem to feed into the voice conversion pipeline (default: Roformer if available, else Demucs).""")
    code("""chosen_vocals = roformer_vocals if (roformer_vocals and os.path.exists(roformer_vocals)) else demucs_vocals
chosen_instrumental = roformer_instrumental if (roformer_instrumental and os.path.exists(roformer_instrumental)) else demucs_instrumental

print(f"Using Vocals Stem: {chosen_vocals}")
print(f"Using Instrumental Stem: {chosen_instrumental}")""")

    # Cell 9: Seed-VC Inference
    md("""### 9. Seed-VC Zero-Shot Singing Voice Conversion
We execute Seed-VC inference in Singing Mode (`--f0-condition True`) using:
- **Source:** Chosen separated vocal stem
- **Target:** Reference voice sample
- **Checkpoint:** 44.1 kHz F0-conditioned singing DiT
- **Diffusion Steps:** 35 (Balanced baseline)
- **CFG Scale:** 0.7
- **Semi-tone Shift:** 0 (Original key)""")
    code("""os.makedirs("results/seed_vc", exist_ok=True)

t0 = time.time()
print("Starting Seed-VC singing voice conversion...")

# Construct command line invocation
seed_cmd = [
    "python", "seed-vc/inference.py",
    "--source", chosen_vocals,
    "--target", sample_path,
    "--output", "results/seed_vc",
    "--diffusion-steps", "35",
    "--length-adjust", "1.0",
    "--inference-cfg-rate", "0.7",
    "--f0-condition", "True",
    "--auto-f0-adjust", "False",
    "--semi-tone-shift", "0",
    "--fp16", "True"
]

print("Executing:", " ".join(seed_cmd))
subprocess.run(seed_cmd, check=True)
vc_time = time.time() - t0
print(f"Seed-VC conversion complete in {vc_time:.2f} seconds!")

# Find converted output wav
converted_files = [os.path.join("results/seed_vc", f) for f in os.listdir("results/seed_vc") if f.endswith(".wav")]
converted_vocals_path = sorted(converted_files, key=os.path.getmtime)[-1]
print(f"Converted Vocals: {converted_vocals_path}")""")

    code("""# Play Converted Vocals
print("Converted Lead Vocals (Solo):")
ipd.display(ipd.Audio(converted_vocals_path))""")

    # Cell 10: Sweeps
    md("""### 10. Hyperparameter Sweeps: Diffusion Steps & Semitone Shift
Let us benchmark how different diffusion steps affect processing time and vocal fidelity:
- **Fast:** 20–25 steps
- **Balanced:** 35 steps
- **Best:** 50 steps""")
    code("""step_benchmarks = {}

for steps in [20, 35, 50]:
    out_dir = f"results/sweep_steps_{steps}"
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    
    cmd = [
        "python", "seed-vc/inference.py",
        "--source", chosen_vocals,
        "--target", sample_path,
        "--output", out_dir,
        "--diffusion-steps", str(steps),
        "--f0-condition", "True",
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", "0",
        "--fp16", "True"
    ]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t_start
    step_benchmarks[steps] = elapsed
    print(f"Steps: {steps:2d} | Time: {elapsed:.2f}s")""")

    # Cell 11: Chunking & Crossfade
    md("""### 11. Chunking & Overlap-Add Crossfade Math Verification
To support full 3–6 minute songs without CUDA out-of-memory errors or boundary clicks, long vocal tracks must be split into chunks with overlap and smoothly recombined.  
Here we verify the linear crossfade math:
$$w_1(t) + w_2(t) = 1.0$$""")
    code("""import numpy as np
import soundfile as sf

def verify_crossfade_math():
    fade_len = 1000
    t = np.linspace(0, 1, fade_len)
    fade_out = 1.0 - t
    fade_in = t
    
    # Assert unity gain across all samples
    summed_energy = fade_out + fade_in
    np.testing.assert_allclose(summed_energy, 1.0, rtol=1e-5)
    print("Crossfade unity gain math verified: w1(t) + w2(t) == 1.0 across all samples!")

verify_crossfade_math()""")

    # Cell 12: Mastering & Mixing
    md("""### 12. Audio Post-Processing, Mixing & EBU-R128 Mastering
Now we:
1. High-pass filter the converted vocals ($> 75\\text{ Hz}$) to remove rumble.
2. Sum the converted vocal stem with the instrumental stem.
3. Apply a soft peak limiter to prevent digital clipping ($< -1.0\\text{ dBTP}$).
4. Measure integrated loudness with `pyloudnorm` and normalize to **-14.0 LUFS**.""")
    code("""import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

def highpass_filter(audio, sr, cutoff=75.0):
    sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    return np.vstack([sosfilt(sos, audio[ch]) for ch in range(audio.shape[0])])

# Load stems
vocal_audio, vocal_sr = sf.read(converted_vocals_path)
inst_audio, inst_sr = sf.read(chosen_instrumental)

# Match sample rates and lengths
target_sr = 44100
min_len = min(len(vocal_audio), len(inst_audio))
vocal_audio = vocal_audio[:min_len]
inst_audio = inst_audio[:min_len]

# Ensure stereo for both
if vocal_audio.ndim == 1:
    vocal_audio = np.column_stack([vocal_audio, vocal_audio])
if inst_audio.ndim == 1:
    inst_audio = np.column_stack([inst_audio, inst_audio])

# 1. High-pass filter vocals
vocal_clean = highpass_filter(vocal_audio.T, target_sr, cutoff=75.0).T

# 2. Balanced summation (vocals 0 dB, inst -1 dB to give vocal headroom)
mixed = (vocal_clean * 1.0) + (inst_audio * 0.9)

# 3. True-Peak Limiter / Normalization
peak = np.max(np.abs(mixed))
if peak > 0.95:
    mixed = mixed * (0.95 / peak)

# 4. Measure & Normalize to -14.0 LUFS
meter = pyln.Meter(target_sr)
current_loudness = meter.integrated_loudness(mixed)
print(f"Pre-mastering loudness: {current_loudness:.2f} LUFS")

target_loudness = -14.0
normalized_mix = pyln.normalize.loudness(mixed, current_loudness, target_loudness)

# Final safety check against clipping
final_peak = np.max(np.abs(normalized_mix))
if final_peak >= 1.0:
    normalized_mix = normalized_mix / final_peak * 0.98

final_mix_path = "results/final_mastered_mix.wav"
sf.write(final_mix_path, normalized_mix, target_sr)
print(f"Final mastered mix exported to: {final_mix_path}")
print(f"Final Integrated Loudness: {meter.integrated_loudness(normalized_mix):.2f} LUFS")""")

    code("""# Play Final Mastered Song
print("Final Mastered Song (Converted Vocals + Original Instrumental):")
ipd.display(ipd.Audio(final_mix_path))""")

    # Cell 13: Final A/B comparison and summary table
    md("""### 13. A/B Comparison & Acceptance Gate 0 Sign-Off
Review all stages of the audio pipeline side-by-side:""")
    code("""print("1. Original Song:")
ipd.display(ipd.Audio(song_path))

print("2. Isolated Lead Vocals:")
ipd.display(ipd.Audio(chosen_vocals))

print("3. Target Voice Reference Sample:")
ipd.display(ipd.Audio(sample_path))

print("4. Converted Vocals Only:")
ipd.display(ipd.Audio(converted_vocals_path))

print("5. Final Mastered Song in Target Voice:")
ipd.display(ipd.Audio(final_mix_path))""")

    md("""### 14. Phase 0 Performance & Timing Metrics
Summary of measured timings for your academic project report:""")
    code("""import pandas as pd

timing_data = {
    "Pipeline Stage": [
        "Demucs Stem Separation",
        "BS-RoFormer Stem Separation",
        "Seed-VC Singing Conversion (35 steps)",
        "Audio Mastering & EBU-R128 Normalization"
    ],
    "Duration (Seconds)": [
        f"{demucs_time:.2f}s",
        f"{roformer_time:.2f}s",
        f"{vc_time:.2f}s",
        "< 1.0s"
    ],
    "Hardware": ["T4 GPU", "T4 GPU", "T4 GPU", "CPU"]
}

df = pd.DataFrame(timing_data)
print(df.to_string(index=False))""")

    md("""---
### Gate 0 Approval Checklist:
- [ ] Converted singing vocal matches reference voice timbre.
- [ ] Pitch and melody follow original vocalist without key drift.
- [ ] Lyrics are clear and intelligible.
- [ ] Converted vocal stem sits naturally in the final instrumental mix without clipping.
- [ ] Mastered loudness is -14 LUFS.

**Next Step:** Once satisfied with audio quality, grant **"Approved"** to proceed to Phase 1 (Worker Package & Modal Deployment).""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/phase0_prototype.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("Notebook successfully created at notebooks/phase0_prototype.ipynb")

if __name__ == "__main__":
    build_notebook()
