# Product Requirements Document (PRD)
## Singing Voice Conversion Website (Seed-VC MVP)

---

### 1. Overview & Problem Statement
Singing voice conversion (SVC) enables a user to hear a song performed in their own unique voice while preserving the song's original melody, tempo, lyrical pronunciation, and emotional dynamics. Traditional SVC systems (like older versions of so-vits-svc or RVC) require users to collect 5–15 minutes of pristine, isolated vocal data and train a per-user neural network for 15–45 minutes on a GPU. This presents an insurmountable friction point for casual users and non-technical students.

This project delivers a **zero-shot singing voice conversion web application**. By leveraging **Seed-VC** (an F0-conditioned diffusion transformer architecture) paired with a high-fidelity vocal separation model (Demucs / Roformer), users can upload any commercial song and a single 10–30 second voice sample (uploaded or recorded in-browser) to receive a full studio-quality remix in their voice within minutes.

Because the system is developed on a CPU-only client machine, all computational heavy-lifting is offloaded to a serverless cloud GPU runtime (Modal) that scales to zero when idle, minimizing hosting costs for a student project.

---

### 2. User Personas & Core Use Cases
- **The AI/DS Student / Evaluator:** Wants to understand zero-shot neural audio architectures, evaluate pitch-conditioning trade-offs, inspect pipeline latency and cost, and run reproducible subjective/objective benchmarks.
- **The Casual Creator / Music Enthusiast:** Wants to hear themselves sing their favorite track, test what their voice would sound like in different musical genres, or generate fun personalized audio clips without technical configuration.
- **The Indie Vocalist / Songwriter:** Wants to draft vocal demos and sketch harmonies across different vocal keys using semitone adjustments before recording studio takes.

---

### 3. User Flow & Experience

```
[1. Upload Song] ----> [2. Voice Reference] ----> [3. Options & Consent]
  (mp3/wav/flac)        (Record or Upload)          (Semitone, Quality, Checkbox)
                                                              |
                                                              v
[5. Results & A/B Player] <--- [4. Real-Time Job Progress] <--+ (Click "Convert")
  - A/B Compare Player           - Separating Vocals (25%)
  - Download Mix & Vocals        - Pitch Tracking & Seed-VC (65%)
  - Convert Another              - Mastering & Remux (90%)
```

1. **Step 1: Upload Song:** User provides a song file (mp3, wav, m4a, flac). The browser inspects the file duration and displays the audio title, file size, and duration.
2. **Step 2: Voice Reference:** User provides a clean 10–30 second voice sample by either:
   - Recording directly through their microphone using the built-in MediaRecorder tool (with live input volume meter and a prompt sentence to read).
   - Uploading an existing clean speech or singing file.
3. **Step 3: Conversion Settings & Consent:**
   - *Semitone Shift:* Optional pitch adjustment (-12 to +12 semitones) for fitting vocal ranges (e.g., male to female or female to male).
   - *Diffusion Quality:* Fast (25 steps), Balanced (35 steps), or Best (50 steps).
   - *Output Format:* MP3 (320 kbps) or WAV (16-bit PCM).
   - *Consent Checkbox (Mandatory):* Explicit user confirmation of voice ownership/permission and personal usage rights.
4. **Step 4: Real-time Job Progress:** User sees an animated stage-by-stage progress bar showing current status (`VALIDATING` $\to$ `SEPARATING` $\to$ `CONVERTING` $\to$ `MIXING` $\to$ `DONE`) with realistic elapsed time and ETA.
5. **Step 5: Playback & Export:**
   - A synchronized A/B audio player lets the user seamlessly switch between the original song and the converted track at any point during playback.
   - Download options for the final full mix and the isolated converted vocal stem.
   - An immediate "Delete My Audio Now" button, plus automatic 24-hour expiration.

---

### 4. Project Scope

#### In Scope (MVP)
- **Zero-shot Voice Conversion Engine:** Seed-VC singing checkpoint (`DiT_seed_v2_uvit_whisper_base_f0_44k`) with F0 conditioning.
- **Audio Stem Separation:** High-accuracy separation of vocals and instrumental accompaniment (Demucs / Roformer).
- **Audio Engineering & Mastering:** Chunked inference with smooth crossfading, vocal de-essing, dynamic range control, vocal-instrumental balance, and loudness normalization to -14 LUFS (EBU R128).
- **Stateful Job Queue:** Asynchronous job processing with progress percentage and human-readable state logs.
- **Full-Stack Web App:** Next.js (App Router, Tailwind CSS, TypeScript) + FastAPI (Python 3.11).
- **Serverless Cloud GPU Integration:** Modal worker container that scales to zero, caching weights in a persistent network volume.
- **Safety, Privacy & Consent:** IP-hashed consent logging, 24-hour auto-cleanup, immediate purge on demand, and audio file metadata tagging.
- **Mock GPU Mode:** Local CPU development mode simulating conversion for zero-cost UI/API testing.
- **Evaluation Toolkit:** Quantitative benchmark script logging stage timings and estimated cost, plus a subjective Mean Opinion Score (MOS) survey sheet.

#### Out of Scope (On Hold for Future Phases)
- **RVC Training Engine:** Multi-epoch training of individual RVC models (designed for in architecture, but deferred).
- **User Accounts & Authentication:** No user logins, passwords, or persistent profiles in MVP.
- **Monetization & Billing:** No Stripe, payment gates, or credit tokens.
- **Public Feed & Social Sharing:** No public community gallery or shared link repository.
- **Real-time Streaming Conversion:** No live microphone-to-microphone streaming.
- **Native Mobile Apps:** Web responsive only.

---

### 5. Quality Bar & Acceptance Criteria (MVP)

| Criteria | Threshold / Target | Verification Method |
| :--- | :--- | :--- |
| **End-to-End Latency** | $< 3.5$ minutes for a 3.5-minute song on an A10G/T4 GPU | Timing logs recorded in evaluation script (`scripts/eval.py`). |
| **Pitch & Melody Accuracy** | Zero perceptible key drift; melody contour matches original vocalist within $\pm 10$ cents. | RMVPE F0 tracking comparison in Phase 0 prototype. |
| **Lyrical Intelligibility** | Words clearly distinguishable; no phoneme collapse or slurring. | Subjective listening test with lyrics comparison. |
| **Audio Artifacts** | Zero clicks/pops at chunk boundaries; no phase cancellation or robotic flutter. | Equal-power linear crossfade validation on chunk seams. |
| **Mix Balance** | Converted vocal sits naturally on the instrumental; master integrated loudness at $-14 \pm 1$ LUFS, true peak $< -1.0$ dBTP. | `pyloudnorm` measurement in export pipeline. |
| **Reliability & State Safety** | $100\%$ of failed jobs transition cleanly to `FAILED` with actionable error messages; no orphaned/stuck jobs. | Pytest test suite with corrupted audio, oversized files, and simulated worker timeouts. |
| **Cost Efficiency** | Worker containers spin down to zero replicas after 60 seconds of inactivity; model weights cached in Modal volume. | Modal dashboard inspection (zero idle compute cost). |

---

### 6. Safety, Privacy & Ethical Guardrails
1. **Consent Gate:** Processing will not start unless the user certifies:
   > *"This is my own voice, or I have explicit permission from the voice owner. I have the right to use this song for personal processing."*
   The server records a cryptographic hash of the client's IP address, the timestamp, and the job ID in the audit record.
2. **Data Retention & Auto-Purge:**
   - All input audio files, intermediate stems, and exported tracks are automatically deleted 24 hours after job creation via a background cleanup worker.
   - Users can trigger immediate file deletion via a `DELETE /api/jobs/{id}` action on the results page.
3. **No Public Indexing:** Audio output URLs are temporary, non-guessable UUID-based links or pre-signed URLs. No search engine indexing or public directory exists.
4. **Watermarking & AI Attribution:** Exported audio files include standard ID3 / Vorbis comment metadata fields:
   - `COMMENT`: `"Synthesized with Seed-VC (Zero-Shot Singing Voice Conversion) for personal educational use."`
   - `ENCODED_BY`: `"Singing Voice Conversion WebApp"`
5. **Input Validation:** Strict MIME type and `ffprobe` format verification to prevent malicious payload uploads. Maximum song duration capped at 6 minutes; sample capped at 60 seconds.
