import os
import re
import glob
import shutil
import subprocess
from huggingface_hub import hf_hub_download, snapshot_download

# Ensure we are always in /content on Google Colab or root workspace
if os.path.exists("/content"):
    os.chdir("/content")

seed_vc_dir = "/content/seed-vc" if os.path.exists("/content") else os.path.abspath("seed-vc")

# 1. Clone Seed-VC repository & restore pristine files
if not os.path.isdir(seed_vc_dir):
    print(f"📥 Cloning Seed-VC repository into {seed_vc_dir}...")
    subprocess.run(["git", "clone", "https://github.com/Plachtaa/seed-vc.git", seed_vc_dir], check=True)
else:
    print(f"✅ Seed-VC directory verified at {seed_vc_dir}.")
    if os.path.isfile(os.path.join(seed_vc_dir, "modules", "bigvgan", "bigvgan.py")):
        # Revert any previously corrupted edits in bigvgan.py
        subprocess.run(["git", "checkout", "--", "modules/bigvgan/bigvgan.py"], cwd=seed_vc_dir, check=False)

# 2. In-Place Bulletproof Patch for BigVGAN (HF Hub >= 0.25 & PyTorch 2.6+ / 2.11+)
# Fixes 'missing 2 required keyword-only arguments: proxies and resume_download'
bigvgan_files = [
    os.path.join(seed_vc_dir, "modules", "bigvgan", "bigvgan.py")
] + glob.glob("/usr/local/lib/python3*/dist-packages/bigvgan*/bigvgan.py")

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

# 7. Mirror models into seed_vc_dir/models
target_models_dir = os.path.join(seed_vc_dir, "models")
os.makedirs(target_models_dir, exist_ok=True)
for item in os.listdir("models"):
    src = os.path.abspath(os.path.join("models", item))
    dst = os.path.abspath(os.path.join(target_models_dir, item))
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except Exception:
            if os.path.isfile(src):
                shutil.copy2(src, dst)

# 8. Seed generic preset voices from repository samples
ref_dir = os.path.join(seed_vc_dir, "examples", "reference")
if os.path.exists(ref_dir):
    wavs = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith((".wav", ".mp3", ".flac"))])
    if wavs:
        for idx in range(min(3, len(wavs))):
            target_preset = f"models/presets/preset_{idx + 1}.wav"
            target_preset_voice = f"models/presets/preset_voice_{idx + 1}.wav"
            if not os.path.exists(target_preset):
                shutil.copy2(wavs[idx], target_preset)
            if not os.path.exists(target_preset_voice):
                shutil.copy2(wavs[idx], target_preset_voice)
            print(f"✅ Created generic preset voice: preset_{idx + 1} & preset_voice_{idx + 1}")

# 9. Pre-cache Whisper & BS-Roformer weights so the first job starts instantly
print("5/5 Pre-caching Whisper and Stem Separator weights...")
try:
    from transformers import WhisperModel
    WhisperModel.from_pretrained("openai/whisper-base")
    print("✅ Whisper-base speech encoder pre-cached.")
except Exception as e:
    print(f"⚠️ Whisper pre-cache notice: {e}")

try:
    from audio_separator.separator import Separator
    sep = Separator(model_file_dir="/tmp/audio-separator-models")
    sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    del sep
    print("✅ BS-Roformer vocal separator pre-cached.")
except Exception as e:
    print(f"⚠️ Separator pre-cache notice: {e}")

print("✅ Cell 3 complete: Neural weights, separators, and presets ready!")