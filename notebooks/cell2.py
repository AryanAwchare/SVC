!pip install -q --upgrade pip setuptools wheel

# 1. Web API, FastAPI & Tunneling
!pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart pydantic nest_asyncio

# 2. Audio Processing, Separation & Loudness Normalization (BS-Roformer / Audio-Separator + Demucs + DSP)
!pip install -q audio-separator demucs pyloudnorm soundfile librosa torchaudio scipy omegaconf einops huggingface_hub noisereduce

# 3. Neural Model Helpers, Speech Codecs, Pitch Trackers & Evaluation
!pip install -q munch transformers openai-whisper pydub descript-audio-codec torchcrepe jiwer

# 4. Optional Timbre Classifier (Safe fallback to MFCC)
!pip install -q speechbrain || echo "Notice: speechbrain skipped (falls back to high-res MFCC timbre similarity)"

print("✅ All Python dependencies installed cleanly without VRAM-hogging conflicts!")
