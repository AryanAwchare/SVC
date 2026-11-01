# Software Licenses & Compliance Matrix
## Singing Voice Conversion Website (Seed-VC MVP)

This document catalogs the open-source licenses governing every core dependency and model weight utilized in this project, evaluating their legal implications for both **student / academic use** and potential **future commercial hosting**.

---

### 1. Model & Machine Learning Components

| Component / Model | Repository / Creator | License | Commercial Use Permitted? | Copyleft / Attribution Requirements |
| :--- | :--- | :--- | :---: | :--- |
| **Seed-VC** (Voice Conversion Engine) | `Plachtaa/seed-vc` | **GNU GPL v3.0** | Yes (with strong copyleft) | Any derivative software or service incorporating/distributing Seed-VC must be open-sourced under the GPL-3.0 license. |
| **Demucs** (`htdemucs` Separation) | Meta AI Research (`facebookresearch/demucs`) | **MIT License** | Yes | Permissive; requires preservation of copyright and license notice. |
| **python-audio-separator** / Roformer Stems | `nomadkaraoke/python-audio-separator` | **MIT License** | Yes | Permissive; wrapper and core separation code are MIT. |
| **CAMPPlus** (Speaker Embedding) | Alibaba DAMO Academy / FunASR | **Apache 2.0** | Yes | Permissive; requires attribution and license notice. |
| **RMVPE** (Pitch Extractor) | `lj1995/VoiceConversionWebUI` | **MIT License** | Yes | Permissive open-source pitch estimation model. |
| **Whisper** (Feature Representation) | OpenAI (`openai/whisper`) | **MIT License** | Yes | Permissive; audio feature representations extracted freely. |
| **BigVGAN** (Neural Vocoder) | NVIDIA Research (`NVIDIA/BigVGAN`) | **MIT License** | Yes | Fully permissive under modern BigVGAN v2 release. |

---

### 2. Software Frameworks & Libraries

| Library / Tool | Primary Use | License | Notes |
| :--- | :--- | :--- | :--- |
| **FastAPI** | Backend Web API | MIT | Highly permissive. |
| **Next.js** | Frontend Web Framework | MIT | Highly permissive. |
| **Tailwind CSS** | UI Styling | MIT | Highly permissive. |
| **SQLModel / SQLAlchemy** | Database ORM | MIT | Highly permissive. |
| **Modal Client** | Serverless Cloud GPU SDK | Apache 2.0 | Standard commercial SDK. |
| **PyTorch & Torchaudio** | Tensor & Neural Network Compute | BSD-3-Clause | Highly permissive. |
| **FFmpeg** | Audio decoding, transcoding, resampling | LGPL v2.1+ / GPL v2+ | Standard CLI invocation via `subprocess` avoids linking conflicts. |
| **Librosa / SoundFile / Scipy** | Audio feature & DSP utilities | ISC / BSD-3-Clause | Highly permissive scientific Python stack. |
| **pyloudnorm** | EBU-R128 Loudness Normalization | MIT | Permissive audio mastering utility. |

---

### 3. Implications for Student / Academic Use

For an AI/DS university coursework submission, academic portfolio, or open-source research demonstration:
1. **Full Compliance:** All tools, models, and libraries are $100\%$ free and open-source.
2. **Academic Freedom:** Student research, demonstration videos, and GitHub public repositories fall cleanly within the permitted educational scope of every dependency.
3. **Attribution:** Citations and acknowledgments should be included in the project's final report and `README.md` for:
   - Seed-VC (Plachta et al.)
   - Demucs (Défossez et al., Meta AI)
   - RMVPE (Harvest/Crepe/RMVPE contributors)

---

### 4. Implications for Future Commercial SaaS Hosting

If this project is ever commercialized or hosted as a paid online service:
1. **The GPL-3.0 Copyleft Impact (Seed-VC):**
   - Because Seed-VC is released under **GNU General Public License v3.0**, if you modify or link Seed-VC directly into your application distribution, your code must also be licensed under GPL-3.0.
   - Running Seed-VC as an isolated backend microservice or serverless worker accessed via network API (the "SaaS loophole" in standard GPL-3.0 vs AGPL-3.0) allows the proprietary frontend or billing layer to remain private under standard interpretations. However, the worker package itself should remain open-source to respect the upstream authors.
2. **Copyrighted Songs & Fair Use:**
   - In a commercial SaaS environment, processing copyrighted pop songs uploaded by users incurs digital copyright and DMCA liability. A commercial platform would require registered DMCA safe harbor agents, terms of service indemnification, and potential licensing arrangements with performing rights organizations (ASCAP/BMI) or mechanical rights holders.
   - For the student MVP, this risk is mitigated by enforcing personal, private, non-commercial use, mandatory consent checkboxes, and automated 24-hour file deletion.
3. **Voice Rights & Deepfake Regulations:**
   - Emerging voice privacy laws (e.g., California ELVIS-style legislation, EU AI Act transparency rules) require explicit consent before cloning or converting an individual's vocal likeness.
   - The consent tracking system and metadata watermark tags implemented in this architecture are designed to meet these regulatory guidelines proactively.
