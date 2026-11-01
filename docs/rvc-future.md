# Architectural Design: Future RVC Integration (On Hold)
## Retrieval-based Voice Conversion (RVC) Engine Extension

---

### 1. Executive Summary & Status
**Current Status:** **ON HOLD (DO NOT IMPLEMENT IN MVP)**  
The MVP focuses exclusively on **Seed-VC**, which is a *zero-shot* voice conversion engine. Zero-shot means the user needs only 10–30 seconds of audio and no model training takes place.

**RVC (Retrieval-based Voice Conversion)** represents a future "High-Quality Studio Mode" upgrade. RVC trains an individual acoustic model (`.pth` checkpoint + `.index` feature retrieval file) for each specific voice. This document specifies how the system is pre-architected so that RVC can be plugged into the existing backend and worker pipelines without refactoring or rewriting any existing code.

---

### 2. Architectural Differences: Seed-VC vs RVC

| Architectural Dimension | Seed-VC (MVP Engine) | RVC (Future Engine) |
| :--- | :--- | :--- |
| **Model Paradigm** | Zero-Shot Diffusion Transformer | One-Shot / Few-Shot Fine-Tuned VITS |
| **User Data Requirement** | 10 to 30 seconds of clean speech or singing | 5 to 15 minutes of dry vocal recordings |
| **Setup Latency** | Instant (0 seconds setup; runtime feature extraction) | Requires training job: 10 to 25 minutes on GPU |
| **Inference Latency** | Moderate (~20–45s per chunk on GPU) | Very Fast (~5–10s per minute of audio) |
| **Artifact Footprint** | None stored per user (stateless reference audio) | ~50 MB `.pth` checkpoint + ~10 MB `.index` file per user |
| **Pitch Tracking (SVC)** | Conditioned with RMVPE during diffusion | Conditioned with RMVPE or Crepe in VITS generator |
| **Serverless Fit** | Highly suitable for stateless serverless invocations | Requires persistent training queues and model storage |

---

### 3. Engine Abstraction Compliance

In `worker/engines/base.py`, the `VoiceConverter` interface was specifically designed to handle both stateless zero-shot engines and stateful trained engines:

```python
class VoiceConverter(ABC):
    name: str
    requires_training: bool

    @abstractmethod
    def prepare(self, reference_audio_path: str, options: Dict[str, Any]) -> Any:
        """
        For Seed-VC:
            Extracts CAMPPlus embedding and validates audio -> returns embedding handle.
        For RVC (Future):
            If user already has a model_id: loads model from persistent storage.
            If training requested: triggers RVC training pipeline and yields model handle.
        """
        pass

    @abstractmethod
    def convert(
        self,
        source_vocals_path: str,
        handle: Any,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Runs conversion using the prepared handle.
        Returns the output path of the converted vocal stem.
        """
        pass
```

#### Current MVP Stub Implementation
In `worker/engines/rvc_stub.py`:
```python
from worker.engines.base import VoiceConverter

class RVCConverter(VoiceConverter):
    name = "rvc"
    requires_training = True

    def prepare(self, reference_audio_path: str, options: dict):
        raise NotImplementedError(
            "The RVC engine is currently ON HOLD for Phase 6. "
            "Please use the default 'seed-vc' engine for zero-shot conversion."
        )

    def convert(self, source_vocals_path: str, handle: any, options: dict, progress_callback=None):
        raise NotImplementedError("RVC conversion is not active in MVP.")
```

When a client submits `"engine": "rvc"` to `POST /api/jobs`, the backend API queries `get_engine("rvc")` which returns a clean `400 Bad Request` with the message:
`"The RVC engine is on hold and will be enabled in a future release. Please select 'seed-vc'."`

---

### 4. What Will Be Needed When Unblocking Phase 6 (RVC)

When Phase 6 is greenlit, the following subsystems must be built:

#### 4.1 Audio Training Dataset Collection & Preprocessing
- **User Audio Input:** Instead of a single 15-second snippet, the user uploads or records 5–10 minutes of clean, isolated vocal clips (or multiple shorter takes).
- **Vocal Slicing:** Automated silence splitting (using `pydub.silence` or `librosa`) into 2–4 second segments.
- **Pitch & Feature Extraction:**
  - RMVPE pitch extraction (`rmvpe.pt`) to generate pitch `.npy` arrays.
  - ContentVec / HuBERT feature extraction (`checkpoint_best_legacy_500.pt`) to generate 768-dimensional content representations.

#### 4.2 Modal GPU Training Job
- **Training Pipeline:** Fine-tuning an RVC v2 base generator (typically 200–300 epochs for high-quality singing).
- **Compute Time:** Approximately 10–18 minutes on an NVIDIA A10G (or ~25 minutes on a T4).
- **Cost Estimate:**
  - Modal A10G pricing: $\approx \$1.10/\text{hour}$.
  - Cost per training run: $\approx \$0.20 - \$0.35$ per voice model.
- **Index Generation:** Running FAISS on the extracted features to produce the `.index` file for retrieval-based timbre enhancement.

#### 4.3 Storage & Persistence Layer
- **Model Storage:** User voice models must be persisted in an S3/R2 bucket under `models/{user_or_voice_id}/`:
  - `model.pth` (~50 MB)
  - `model.index` (~10 MB)
- **Lifecycle Management:** Because models are expensive to train, a retention policy (e.g., 30-day retention or user account binding) will replace the 24-hour auto-purge for model files.

#### 4.4 UI & UX Enhancements
- **Mode Toggle on Home Screen:**
  - `Zero-Shot (Instant, 15s sample, powered by Seed-VC)` [Current Default]
  - `Studio Model (Requires 5-min training, powered by RVC)` [Phase 6]
- **Training Progress Screen:** An asynchronous training status view tracking epoch progress, loss metrics, and estimated time remaining.
- **Saved Voice Library:** A dropdown letting returning users select their previously trained voice models for one-click song conversions.
