import os
import time
import json
import glob
import requests
import subprocess
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

# Ensure we are in /content on Google Colab or root workspace
if os.path.exists("/content"):
    try:
        os.chdir("/content")
    except Exception:
        pass

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🎙️ VocalMorph Studio: Stage-by-Stage Real-Audio Test Runner")
print("=" * 70)

# 1. Health & VRAM Check
print("\n[Step 1/6] Probing Backend Health & VRAM...")
res = requests.get(f"{BASE_URL}/api/health").json()
print("   Health Check Response:", res)
assert res["gpu_available"], "CUDA GPU is not available!"
assert res["free_vram_gb"] > 3.0, f"Insufficient VRAM: {res['free_vram_gb']} GB available"
print(f"   ✅ GPU Online: {res['gpu_name']} ({res['free_vram_gb']} GB free VRAM)")

# 2. Prepare Real 75-second Song & Real Voice Sample
print("\n[Step 2/6] Preparing Real Audio Inputs (75s Song Slice + Real Speech)...")
test_song = "test_song_75s.wav"
test_sample = "test_voice_sample.wav"

def get_candidate_wavs(subpath):
    patterns = [
        f"{subpath}/*.wav",
        f"{subpath}/**/*.wav",
        f"seed-vc/{subpath}/*.wav",
        f"/content/seed-vc/{subpath}/*.wav",
        f"seed-vc/{subpath}/**/*.wav",
        f"/content/seed-vc/{subpath}/**/*.wav",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(list(set(files)))

# Find a real human speech sample
ref_candidates = get_candidate_wavs("examples/reference") + glob.glob("models/presets/*.wav") + glob.glob("/content/models/presets/*.wav")
if ref_candidates:
    src_ref = ref_candidates[0]
    cmd = ["ffmpeg", "-y", "-i", src_ref, "-t", "12", "-ar", "44100", "-ac", "1", test_sample]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   ✅ Real voice reference extracted from {src_ref} (12s human speech).")
else:
    # Safe download from GitHub raw
    print("   📥 Downloading voice reference from GitHub raw...")
    try:
        url = "https://raw.githubusercontent.com/Plachtaa/seed-vc/main/examples/reference/casual_female.wav"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        with open(test_sample, "wb") as f:
            f.write(r.content)
        print("   ✅ Real voice reference downloaded from repository.")
    except Exception as e:
        print(f"   ⚠️ Falling back to synthetic voice reference: {e}")
        t = np.linspace(0, 10.0, 44100 * 10, endpoint=False)
        voice = 0.4 * np.sin(2 * np.pi * 175.0 * t) + 0.2 * np.sin(2 * np.pi * 350.0 * t) + 0.1 * np.sin(2 * np.pi * 700.0 * t)
        sf.write(test_sample, voice, 44100)

# Find or download real song audio and create 75s slice
song_candidates = get_candidate_wavs("examples/source") + get_candidate_wavs("examples")
if song_candidates:
    src_song = song_candidates[0]
    cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", src_song, "-t", "75", "-ar", "44100", "-ac", "2", test_song]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   ✅ Real song slice created from {src_song} (75.0s audio).")
else:
    print("   📥 Fetching song audio from GitHub raw...")
    download_success = False
    try:
        url = "https://raw.githubusercontent.com/Plachtaa/seed-vc/main/examples/source/yae_0.wav"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        with open("raw_source.wav", "wb") as f:
            f.write(r.content)
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", "raw_source.wav", "-t", "75", "-ar", "44100", "-ac", "2", test_song]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Real song slice created from downloaded sample (75.0s audio).")
        download_success = True
    except Exception as e:
        print(f"   ⚠️ Online fetch failed ({e}), synthesizing rich harmonic 75s song...")

    if not download_success:
        sr = 44100
        t = np.linspace(0, 75.0, sr * 75, endpoint=False)
        melody_f0 = 220.0 + 35.0 * np.sin(2 * np.pi * 0.3 * t)
        vocal = 0.4 * np.sin(2 * np.pi * melody_f0 * t) + 0.2 * np.sin(4 * np.pi * melody_f0 * t)
        bass = 0.3 * np.sin(2 * np.pi * 110.0 * t)
        chord = 0.15 * (np.sin(2 * np.pi * 330.0 * t) + np.sin(2 * np.pi * 440.0 * t))
        mix_l = vocal * 0.7 + bass * 0.5 + chord
        mix_r = vocal * 0.7 + bass * 0.5 + chord * 1.1
        stereo = np.column_stack([mix_l, mix_r])
        sf.write(test_song, stereo, sr)
        print("   ✅ Harmonic polyphonic 75s song synthesized.")

print("\n   Previewing Input Reference Voice:")
display(Audio(test_sample))

# 3. Create Conversion Job with Testable Quality Options
print("\n[Step 3/6] Initializing Conversion Job...")
payload = {
    "engine": "seed-vc",
    "model_id": "zero-shot",
    "options": {
        "semitone_shift": 0,
        "diffusion_steps": 35,
        "inference_cfg_rate": 0.70,
        "separator": "bs-roformer",
        "use_dereverb": False,
        "use_karaoke": True,
        "output_format": "mp3"
    }
}
resp = requests.post(f"{BASE_URL}/api/jobs", json=payload).json()
job_id = resp["job_id"]
print(f"   ✅ Job Created: {job_id}")

# 4. Upload Real Audio Files
print("\n[Step 4/6] Uploading Audio Files to Server...")
with open(test_song, "rb") as fsong, open(test_sample, "rb") as fsample:
    u1 = requests.post(f"{BASE_URL}/api/jobs/{job_id}/upload/song", files={"file": fsong}).json()
    u2 = requests.post(f"{BASE_URL}/api/jobs/{job_id}/upload/sample", files={"file": fsample}).json()
print("   Upload Responses:", u1, u2)

# 5. Start Pipeline
print("\n[Step 5/6] Triggering Conversion Pipeline...")
start_res = requests.post(f"{BASE_URL}/api/jobs/{job_id}/start", data={"consent": "true"}).json()
print("   Pipeline Start Response:", start_res)

# 6. Stage-by-Stage Monitoring
print("\n[Step 6/6] Live Pipeline Stage Tracking...")
last_stage = ""
t_start = time.time()

while True:
    st = requests.get(f"{BASE_URL}/api/jobs/{job_id}").json()
    stage = st["status"]
    elapsed = st.get("elapsed_seconds", round(time.time() - t_start, 1))

    if stage != last_stage:
        print(f"   ⏱️ [{elapsed}s] Stage: {stage} ({st.get('progress', 0)}%) | {st.get('message', '')}")
        last_stage = stage

    if stage == "DONE":
        print(f"\n🎉 Conversion Completed Successfully in {elapsed}s!")
        print("\n" + "=" * 70)
        print("🔊 AUDIO PLAYBACK & COMPARISON")
        print("=" * 70)

        mix_url = f"{BASE_URL}/api/jobs/{job_id}/download/final_mix"
        vocals_url = f"{BASE_URL}/api/jobs/{job_id}/download/vocals"

        print("1. Converted Solo Vocals (Seed-VC 44.1kHz):")
        display(Audio(url=vocals_url))

        print("2. Final Studio Master (Key-Matched Instrumental + Sidechain + -14 LUFS):")
        display(Audio(url=mix_url))

        # Query metrics (non-blocking)
        print("\n" + "=" * 70)
        print("📊 OBJECTIVE EVALUATION METRICS")
        print("=" * 70)
        for _ in range(30):
            m_res = requests.get(f"{BASE_URL}/api/jobs/{job_id}/metrics")
            if m_res.status_code == 200:
                print(json.dumps(m_res.json(), indent=2))
                break
            time.sleep(2)
        break

    elif stage == "FAILED":
        print(f"\n❌ Pipeline Failed at stage: {st.get('message')}")
        dbg = requests.get(f"{BASE_URL}/api/jobs/{job_id}/debug").json()
        print("Debug Dump:", json.dumps(dbg, indent=2))
        break

    time.sleep(2.5)
