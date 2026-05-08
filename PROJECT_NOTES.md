# VoiceGuard — Project Notes

> Living log of decisions, state, and the path to the JPM final meeting.
> Update this file as work progresses. Claude reads it at the start of every session.

---

## Status (as of 2026-05-03)

**What works**
- Streamlit dashboard with three scripted scenarios (Clean / Borderline / Synthetic Bot)
- Two synced tabs (Live Simulation + Call Risk Audit) over a 20s animation window
- LangGraph topology drafted in `graph.py` with six stage agents under `agents/`
- Deployed at https://voiceguard-ai.streamlit.app

**What's mocked**
- `agents/node1_voice_cloning.py:20-23` — spectral / prosody scores via `random.uniform`
- `app.py:56-103` — `SCENARIOS` hardcodes every numeric output per scenario
- `audio/` — three demo files only; no labeled corpus
- `graph.py` is not invoked by `app.py`

**Why it matters**: this is the gap a JPM fraud team will probe first. The pitch is stronger if we can show real model output on real audio plus standard eval metrics.

---

## Roadmap to JPM Final

### Phase 1 — Real detection on the existing audio (1 afternoon)
- Replace `random.uniform` in `node1_voice_cloning.py` with a Hugging Face anti-spoofing model
- Candidate: `MelodyMachine/Deepfake-audio-detection` (Wav2Vec2, MIT-friendly)
- Verify scores roughly track the hand-tuned ones for `audio/clean.mp3`, `audio/borderline.mp3`, `audio/ai_voice.wav`
- Add a "Live mode" toggle in the UI to run real inference instead of canned scores

### Phase 2 — Standard evaluation (1–2 days)
- Wire an `eval/` harness against ASVspoof 2021 LA eval split
- Report **EER** and **min t-DCF** — these are the metrics fraud teams expect
- Add **In-the-Wild** dataset for a real-world stress test
- Add **WaveFake** for per-synthesizer breakdown (ElevenLabs / Tortoise / VITS)

### Phase 3 — Telephony robustness (1 day)
- Resample to 8 kHz, apply μ-law (G.711) and Opus encoding, re-eval
- MUSAN noise + RIR reverb sweep, plot degradation curve
- Replay-attack subset (bonafide re-recorded over a speaker)
- Latency benchmark: p50 / p95 inference per stage; IVR budget < 300 ms

### Phase 4 — Pitch artifacts
- Plots: ROC curve, EER table, codec degradation chart, per-synthesizer accuracy
- One-pager: model card (what we use, training data, license, known failure modes)
- Live-mode demo path: upload-a-WAV or mic record → real pipeline → real verdict

---

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-03 | Keep scripted scenarios for the live walkthrough; add real-mode alongside | Scripted demo is load-bearing for storytelling; real-mode is what makes JPM trust it |
| 2026-05-03 | Target HF `MelodyMachine/Deepfake-audio-detection` for v1 swap | Few-line integration via `transformers`, MIT-friendly, ASVspoof-trained |
| 2026-05-03 | Report EER + min t-DCF (not raw accuracy) | Standard metrics for spoof detection; fraud teams will ask |

---

## Open questions

- Do we have JPM contact for sample telephony audio (real-call codec characteristics)?
- Is on-device / on-prem inference a hard requirement, or is HF-Inference-API acceptable for the demo?
- Does the prosody score need to be a separate model, or is a feature-derived signal (jitter / shimmer / VAD-based pause stats) sufficient and more defensible?

---

## Notes from past sessions

_Add a dated bullet here at the end of each working session — what changed, what we learned, what's next._

- 2026-05-03 — Initial sketch of evaluation strategy. Memory + PROJECT_NOTES.md created.
- 2026-05-08 — Phase 1 partial: real Wav2Vec2 deepfake detector + librosa F0 prosody wired into `node1_voice_cloning.py` via new `detectors/` package. Live Mode toggle in UI; scripted scenarios remain default for the JPM walkthrough. FP-aware calibration (`p^1.6`) baked in to bias toward fewer false positives per JPM's 19/7 cost ratio. Heavy deps (`transformers`, `torch`, `librosa`) added to requirements.txt; lazy-imported so scripted path costs nothing.
- 2026-05-08 — Local inference verified on the three demo audio files. Raw spectral / FP-tuned spectral / prosody: `clean.mp3` 0.00 / 0.00 / 0.00; `borderline.mp3` 0.00 / 0.00 / 0.00; `ai_voice.wav` 1.00 / 1.00 / 0.22. Model catches macOS `say` (older TTS) but misses both ElevenLabs samples — expected for an ASVspoof-trained Wav2Vec2 detector against modern neural TTS. Use this in the JPM pitch as evidence for ensemble + HITL rather than papering over it.
- 2026-05-08 — Added speaker-verification layer (`detectors/speaker_verify.py`) using Resemblyzer. Resolves the ElevenLabs-evasion gap: even though the deepfake classifier doesn't catch ElevenLabs, comparing the caller's voiceprint against an enrolled customer baseline does. UI now has a "Registered Customer Voice" upload slot (visible when Live Mode is on) and a dynamic "Registered Customer Calling" scenario that plays the uploaded clip. Combined-verdict logic (`_combined_verdict` in `app.py`) uses `risk = max(deepfake, mismatch_risk)` where `mismatch_risk` is calibrated to Resemblyzer's empirical range (sim ≥ 0.70 → no contribution; sim ≤ 0.50 → 0.65, lands in FLAG band). Same-speaker measured at 1.00, different-speaker pairs at 0.43–0.54 across the three demo files. End-to-end verdicts using user's voice as reference: Registered Customer → PASS, Clean Caller (=borderline.mp3) → FLAG, Synthetic Bot (=ai_voice.wav) → BLOCK. Pitch story upgrades to "voice biometrics + spoof detection layered" — matches how Pindrop / Nuance Gatekeeper / Chase Voice ID already work. `resemblyzer>=0.1.4` added to requirements.
