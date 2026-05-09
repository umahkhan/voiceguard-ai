# VoiceGuard AI

> Real-time voice-fraud detection dashboard with human-in-the-loop review.
> Two layered ML signals (speaker biometrics + deepfake classifier) feeding
> a LangGraph pipeline that pauses for a human reviewer before deciding.

---

## Live Demo

**▶ https://voiceguard-ai.streamlit.app**

Open the link, pick a scenario from **Caller Audio Under Test**, click **Connect Incoming Call**, watch the Speaker Match and Voice Risk meters populate, then click one of the three reviewer-action buttons (Approve / Step-Up Auth / Block) to resume the pipeline.

---

## What It Does

VoiceGuard AI is a fraud-operations dashboard for a contact-center reviewer. When a call comes in, the system runs **two complementary ML models in parallel** against the audio:

1. **Speaker Match (ECAPA-TDNN)** — voiceprint similarity between the caller and the enrolled customer. Catches *who* the caller is, regardless of whether the audio is real or synthetic.
2. **Voice Risk (Wav2Vec2 deepfake classifier)** — synthesis-detection probability on the caller's audio. Catches *what kind* of audio it is, regardless of who's claimed to be calling.

The two signals are combined into a single verdict (PASS / FLAG / BLOCK), and the call is routed through a **LangGraph pipeline** that pauses for a human reviewer's decision before taking any downstream action (auth challenge, block, or clear).

This architecture closes a gap that single-model approaches miss: modern AI voice clones (e.g. ElevenLabs) can fool synthesis classifiers but cannot easily fool a speaker-verification model trained on the customer's actual voice.

---

## The Architecture

### Two ML signals, both real

```
Caller audio (m4a / mp3 / wav)
        │
        ├──→ ECAPA-TDNN (SpeechBrain)         ──→ Speaker Match score
        │      • 192-dim voice embedding
        │      • cosine similarity vs. customer voiceprint
        │      • same speaker: 0.5–0.9
        │      • different speaker: -0.1 to 0.3
        │
        └──→ Wav2Vec2 deepfake classifier      ──→ Voice Risk score
              • motheecreator/Deepfake-audio-detection
              • 94.5M-parameter network
              • binary {fake, real} probability
              • plus librosa F0-CV anomaly as a secondary signal
```

Both models run on every Connect — no scripted scores, no hardcoded values. The dashboard's meter values are the actual model outputs.

### LangGraph pipeline with human-in-the-loop

```
START
  │
  ▼
[1. Voice Cloning Detector]   ← ECAPA + Wav2Vec2 inference, populates state
  │
  ▼
[2. IVR Entry — Defense 1]    ← combines voice signals into entry confidence
  │
  ├── ivr_entry_confidence > 0.8 ─────────────┐
  │                                           │
  └── ≤ 0.8 → [3. IVR Nav — Defense 2]        │
                       │                      │
                       ▼                      ▼
              [4. Agent Handoff — Defense 3]
                  • computes agent confidence
                  • generates alert message for the live reviewer
                       │
                       ▼
              ⏸ ─────── INTERRUPT_AFTER ─────── ⏸
                  Graph pauses here. Reviewer sees the
                  HITL panel in the dashboard, picks one
                  of Approve / Step-Up Auth / Block.
                  Their decision is written to state, the
                  graph resumes via invoke(None, config).
                       │
              ┌────────┼─────────────┐
              │ stepup │ approve     │ block
              ▼        ▼             ▼
       [5. Auth     │             │
        Challenge] │             │
              │     ▼             ▼
              └─►[6. Intelligence — leadership log]
                       │
                       ▼
                      END
```

The pause is implemented with LangGraph's `interrupt_after=["agent_handoff"]` plus a `MemorySaver` checkpointer. Each call gets a unique `thread_id` so multiple paused calls don't collide.

When the reviewer clicks an action button:

```python
g.update_state(config, {"human_decision": "stepup"})
g.invoke(None, config=config)
```

The conditional edge from `agent_handoff` reads `human_decision` and routes accordingly:

| Decision | Path |
|---|---|
| `approve` | → intelligence (case cleared) |
| `stepup` | → auth_challenge (OTP fires) → intelligence |
| `block` | → intelligence (transaction_blocked=True, case logged) |

This isn't cosmetic — clicking a button in the dashboard actually advances a real graph that runs downstream nodes.

---

## Models

| Component | Model | Why this one |
|---|---|---|
| Speaker verification | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Industry-standard ECAPA-TDNN trained on VoxCeleb; same architecture used by production speaker-recognition systems. Replaces an earlier Resemblyzer-based prototype that gave too much overlap between same/different speaker pairs. |
| Deepfake detection | [`motheecreator/Deepfake-audio-detection`](https://huggingface.co/motheecreator/Deepfake-audio-detection) | Wav2Vec2 fine-tuned on a corpus that includes modern neural TTS samples. The fine-tuned successor (`MelodyMachine/Deepfake-audio-detection`) reports higher in-distribution accuracy but generalizes worse to current ElevenLabs — exactly the threat we care about. See PROJECT_NOTES.md for the head-to-head numbers. |
| Pitch anomaly | `librosa.yin` | Classical YIN pitch tracker; F0 coefficient-of-variation as a secondary signal. Catches older/cruder synthesis even when the Wav2Vec2 model misses. |

### Combined verdict logic

```python
# Speaker similarity → mismatch risk (flips direction so high = bad)
mismatch_norm = clip((0.35 - similarity) / 0.10, 0, 1)
mismatch_risk = mismatch_norm * 0.65    # caps at FLAG band

# Synthesis is already on the right axis
deepfake = max(spectral, prosody)

# Both on "high = bad" scale; worst signal wins
risk = max(deepfake, mismatch_risk)
verdict = PASS if risk < 0.50 else FLAG if risk < 0.75 else BLOCK
```

The 0.65 cap on `mismatch_risk` means speaker mismatch alone never blocks (caps at FLAG); only synthesis detection (or both signals together) can push to BLOCK. Keeps the FP rate down — speaker mismatch on a borderline match shouldn't auto-block; reviewer should eye the call first.

---

## Demo Scenarios

The dashboard has **one customer voiceprint** (Umair Khan, account ****0042) and **five caller-audio scenarios**:

| Scenario | Source | Verdict | What it teaches |
|---|---|---|---|
| Real Call · Example 1 | Customer's enrollment recording (replay) | PASS | Sanity check — same audio as voiceprint. |
| Real Call · Example 2 | Different recording, same speaker | PASS | Cross-content speaker matching — the realistic case. |
| AI Clone · Neutral Script | ElevenLabs clone of customer reading a neutral script | BLOCK | Modern deepfake of the actual customer. **Headline test case.** |
| AI Clone · Urgent Script | ElevenLabs clone reading a fraud-style urgent script | BLOCK | Same threat, different content. Detection is content-agnostic. |
| Robocall | macOS `say -v Albert` reading the customer's script | BLOCK | Crude TTS — easy case, clearly synthetic. Both signals trip cleanly. |

---

## Quick Start (Local)

Requires Python 3.10+.

```bash
git clone https://github.com/umahkhan/voiceguard-ai.git
cd voiceguard-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`. **First call has a cold-start of ~15–25 s** while the two models download (~720 MB Wav2Vec2 + ~80 MB ECAPA). Subsequent calls are sub-second.

No API keys required for the core dashboard. The optional `_anthropic.py` helper adds Claude-generated alert text inside the LangGraph pipeline — set `ANTHROPIC_API_KEY` if you want that, otherwise it falls back to canned strings.

---

## Project Structure

```
.
├── app.py                          # Streamlit dashboard (the UI + HITL handlers)
├── graph.py                        # LangGraph wiring with interrupt_after + MemorySaver
├── state.py                        # Typed state schema for the pipeline
├── detectors/
│   ├── voice_clone.py              # Wav2Vec2 + librosa F0 (Voice Risk)
│   └── speaker_verify.py           # SpeechBrain ECAPA-TDNN (Speaker Match)
├── agents/
│   ├── node1_voice_cloning.py      # graph node — populates Stage 1 signals
│   ├── node2_ivr_entry.py          # IVR entry analyzer
│   ├── node3_ivr_navigation.py     # IVR navigation anomaly
│   ├── node4_agent_handoff.py     # alert generation + the HITL pause point
│   ├── node5_auth_challenge.py     # OTP / step-up auth (fires only on stepup)
│   └── node7_intelligence.py       # leadership log + loss-avoidance summary
├── audio/                          # 5 scenario clips + customer voiceprint
├── requirements.txt
├── PROJECT_NOTES.md                # Living log of architecture decisions
├── JPM_DEMO_SETUP.md               # Walkthrough script for the JPM pitch
└── README.md
```

---

## Tech Stack

- **Streamlit** — dashboard UI + animation
- **LangGraph + MemorySaver** — pipeline orchestration with durable pause/resume
- **transformers / torch** — Wav2Vec2 deepfake classifier
- **SpeechBrain** — ECAPA-TDNN speaker verification
- **librosa** — audio I/O + classical pitch tracking
- **Anthropic Claude** *(optional)* — alert-text generation inside graph nodes

---

## What's Real vs What's Mocked

For full transparency:

| Component | Status |
|---|---|
| Speaker Match meter | Real — ECAPA on the actual audio |
| Voice Risk meter | Real — Wav2Vec2 + librosa F0 on the actual audio |
| Combined verdict | Real — derived from the two signals via the formula above |
| LangGraph pipeline pause/resume | Real — `interrupt_after` + `MemorySaver`, button click drives `update_state` |
| Reviewer audit log | Real — session-scoped, persists button clicks |
| Behavior Risk + Agent Suspicion meters | Scripted per scenario (downstream pipeline stages not yet wired to real signals) |
| Caller context tile (claimed identity, account, transaction) | Scripted — there's no real CRM behind it |
| Telephony / SIP / RTP integration | Not present — calls are file playback, not live phone |

Voice analysis is fully model-driven; the pipeline orchestration around it (IVR analysis, CRM integration, telephony) is the production-engineering work that follows the pitch.

---

## Disclaimer

VoiceGuard AI is a research demo. The voice-analysis path is real and reproducible, but production deployment would need a labeled corpus for cost-aware threshold calibration, telephony integration, a real customer-data system, and continuous model evaluation as attackers update their tools.
