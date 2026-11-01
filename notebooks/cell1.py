!nvidia-smi

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

print("✅ Cell 1 finished successfully!")
