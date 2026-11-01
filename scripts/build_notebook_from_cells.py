import nbformat as nbf

nb = nbf.v4.new_notebook()

header_md = """# 🎙️ VocalMorph Studio: Production Singing Voice Conversion (Colab T4 + ngrok)
### Python 3.13 Compatible | Seed-VC 44.1kHz Singing DiT + BigVGAN + RMVPE | Audio-Separator BS-Roformer

---

## 📌 Architecture Highlights
1. **Isolated Subprocess Separation**: `BS-Roformer`, `UVR-De-Echo`, and `Karaoke Split` run in dedicated CLI subprocesses, releasing 100% of GPU memory on completion.
2. **Harmonic Key Matching**: `Rubber Band` pitch transposition octave-folded to [-6, +5] semitones applied to instrumental and backing harmonies.
3. **Studio DSP Mastering**: Band-subtraction de-esser (5.5 - 9.5 kHz), fast $O(N)$ running RMS compressor, seeded stereo plate reverb, dynamic sidechain ducking (-1.5 dB), and -14.0 LUFS EBU-R128 broadcast mastering.
4. **Non-Blocking Objective Evaluation**: ECAPA-TDNN speaker similarity, torchcrepe F0 Pearson & RMSE cents, Whisper WER, and ITU-R BS.1770 audio health.
5. **Robust API Contract**: Graceful tolerance for frontend parameters, FFmpeg transcode for any audio format, and Generation-managed worker threads.
"""

cell1_md = "### **Cell 1: GPU & System Dependency Setup**\nVerifies NVIDIA T4 GPU and installs system binaries (FFmpeg, libsndfile, rubberband-cli)."
cell2_md = "### **Cell 2: Python Libraries & Neural Codecs**\nInstalls audio-separator, demucs, PyTorch audio tools, and evaluation frameworks."
cell3_md = "### **Cell 3: Clone Seed-VC, Safe BigVGAN Patch, Download Weights & Presets**\nClones repository, applies safe in-place signature and PyTorch 2.6+ patch to BigVGAN, fetches neural models, and initializes generic preset voices."
cell4_md = "### **Cell 4: Production VocalMorph Server & Evaluation Engine**\nHosts the REST API with single-worker GPU queue, professional DSP chain, non-blocking evaluation metrics, and ngrok tunnel."
cell5_md = "### **Cell 5: Stage-by-Stage Real Audio Verification & Playback**\nRuns a stage-by-stage pipeline on real 75-second audio, displaying waveform audio players after each stage and reporting metrics."

def read_code(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

nb.cells = [
    nbf.v4.new_markdown_cell(header_md),
    nbf.v4.new_markdown_cell(cell1_md),
    nbf.v4.new_code_cell(read_code("notebooks/cell1.py")),
    nbf.v4.new_markdown_cell(cell2_md),
    nbf.v4.new_code_cell(read_code("notebooks/cell2.py")),
    nbf.v4.new_markdown_cell(cell3_md),
    nbf.v4.new_code_cell(read_code("notebooks/cell3.py")),
    nbf.v4.new_markdown_cell(cell4_md),
    nbf.v4.new_code_cell(read_code("notebooks/cell4.py")),
    nbf.v4.new_markdown_cell(cell5_md),
    nbf.v4.new_code_cell(read_code("notebooks/cell5.py")),
]

out_path = "notebooks/singing_voice_conversion_colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully built {out_path} from cell1.py..cell5.py")
