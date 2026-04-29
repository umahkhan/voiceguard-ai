# VoiceGuard AI

> Real-time voice fraud defense dashboard for contact-center call flows.
> JPMorgan-aligned design system. Built with Streamlit.

---

## Live Demo

**Try it here:** _coming soon — deploy to Streamlit Community Cloud (instructions below)_

After deployment, paste your URL here, e.g. `https://umahkhan-voiceguard.streamlit.app`.

---

## What It Does

VoiceGuard AI simulates how a bank's contact center would catch a synthetic-voice ("deepfake") fraud call as it moves through the IVR and onto a live agent. The dashboard walks through a 6-stage defense pipeline and shows, in real time:

- Whether the call is **cleared, flagged, or blocked**
- Voice biometric scores (spectral + prosody)
- Agent suspicion + behavioral risk
- Which defense stages **ran** vs. were **skipped** based on confidence thresholds
- A play-by-play stage detail with live status (`pending → running → executed/skipped`)
- Estimated loss avoided per intercepted call

Three pre-built scenarios let you compare outcomes:

| Scenario | What it represents |
|---|---|
| **Clean Caller** | Normal customer — all signals within baseline. PASS. |
| **Borderline Suspicious** | Mixed signals — rehearsed prosody, hesitant navigation. FLAG. |
| **Synthetic Bot — High Confidence** | All three indicators flag clear synthesis. BLOCK. |

---

## Two Tabs, One Synced Animation

- **Live Simulation** — incoming-call card, scenario selector, place-call button, verdict bar, three risk metrics, recommended action, system note.
- **Call Risk Audit** — the same call's audit trail: meta strip, path-taken pills, and a stage-by-stage breakdown.

When you place a call, **both tabs animate in sync** — meta, path pills, and stage rows tick together over the 20-second analysis window. Switch tabs mid-call and you'll see the same moment from each perspective.

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

The app will open at `http://localhost:8501`. **No API keys required** — the demo uses bundled audio (`audio/`) and hardcoded scenario data.

---

## Deploy Your Own (Streamlit Community Cloud — Free)

The fastest way to share VoiceGuard with anyone:

1. Push your fork to GitHub.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app** → pick this repo → set the main file to `app.py` → **Deploy**.
4. You'll get a public URL like `https://<your-handle>-voiceguard.streamlit.app`. Share it.

Deploys take about 60 seconds. Re-deploys happen automatically on every push to `main`.

> **Note on secrets:** the bundled demo needs none. If you later wire up the optional LangGraph pipeline (see below), add `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY` via the Streamlit Cloud **Secrets** pane — they get injected as environment variables at runtime and never enter the repo.

### Other hosting options

| Platform | Notes |
|---|---|
| **Hugging Face Spaces** | Pick the Streamlit SDK, paste the repo, add secrets in the UI. |
| **Render / Railway / Fly.io** | Set start command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. |

---

## Project Structure

```
.
├── app.py              # Streamlit dashboard (the demo)
├── audio/              # Pre-rendered scenario audio (mp3 + wav)
├── graph.py            # Optional LangGraph pipeline wiring (not used by app.py)
├── state.py            # Typed state schema for the pipeline
├── agents/             # 6 stage agents — voice cloning, IVR, handoff, auth, intel
├── requirements.txt
├── .env.example        # Template for ANTHROPIC_API_KEY (only needed for graph.py)
└── README.md
```

`app.py` is fully self-contained — it imports `graph.py` only as a soft dependency and falls back gracefully if missing.

---

## Optional: The LangGraph Pipeline

`graph.py` and `agents/` implement the same 6-stage pipeline as a real LangGraph flow with Anthropic-powered agents, mapped to the Future-State Journey:

```
Voice Cloning → IVR Entry → IVR Navigation → Agent Handoff → Auth Challenge → Intelligence
```

To experiment with it locally, copy `.env.example → .env`, set `ANTHROPIC_API_KEY`, and import `build_graph()` from `graph.py`. The current `app.py` doesn't call it — the dashboard runs entirely on hardcoded scenario data so the demo is fast and offline.

---

## Tech Stack

- **Streamlit** — UI + animation loop
- **LangGraph + Anthropic** — optional pipeline (not exercised by the demo)
- **ElevenLabs** — used once to render scenario audio (not needed at runtime)

---

## Disclaimer

VoiceGuard AI is a **prototype / educational demo**. Scenario data, scores, and verdicts are hand-tuned for illustration. Do not use it as-is for production fraud decisions.
