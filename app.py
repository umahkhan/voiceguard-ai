"""VoiceGuard AI — Streamlit dashboard for voice fraud detection.

JPMorgan-aligned design: deep navy + white card system, two tabs that fit
in a 900px viewport without scrolling. All scenario data is hardcoded so
the app runs without any model calls.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _components
from dotenv import load_dotenv

load_dotenv()

# Optional pipeline import — kept for backward compat, not required.
try:
    from graph import build_graph  # noqa: F401
    from state import VoiceGuardState  # noqa: F401
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


AUDIO_DIR = Path(__file__).parent / "audio"

# Customer voiceprint is pinned to a file in the project — no upload UI.
# Drop the enrollment recording at this path.
REGISTERED_VOICE_FILE = AUDIO_DIR / "customer_voiceprint_umair.m4a"
SPOOFED_VOICE_FILE    = AUDIO_DIR / "customer_voiceprint_umair_spoofed.mp3"

# Available customer voiceprints — the baseline pool the dashboard's
# voiceprint dropdown picks from. Add new entries here to expand it.
VOICEPRINTS: dict[str, str] = {
    "Umair Khan": "customer_voiceprint_umair.m4a",
}


def _audio_mime(path: Path) -> str:
    """Map audio file extension → MIME type. Streamlit Cloud occasionally
    serves files with the wrong Content-Type when given a path string,
    causing the browser to not render the audio control. Passing bytes
    plus an explicit format avoids the issue."""
    return {
        ".m4a":  "audio/mp4",
        ".mp4":  "audio/mp4",
        ".aac":  "audio/aac",
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
        ".flac": "audio/flac",
        ".ogg":  "audio/ogg",
    }.get(path.suffix.lower(), "audio/mpeg")


# ---------------------------------------------------------------------------
# Design tokens — Chase brand palette
# ---------------------------------------------------------------------------
NAVY        = "#003087"   # Chase primary blue
NAVY_DEEP   = "#001a4d"   # Chase deep blue
ACCENT      = "#0072cf"   # Chase interactive blue
INK         = "#1a1a1a"   # Chase body text
PAPER       = "#ffffff"
CANVAS      = "#f5f6f8"   # Chase page background
BORDER      = "#d8d8d8"   # Chase border
MUTED       = "#616c7d"   # Chase secondary text

PASS_BG,  PASS_TEXT  = "#0a3d1e", "#4ade80"
FLAG_BG,  FLAG_TEXT  = "#3d2500", "#f59e0b"
BLOCK_BG, BLOCK_TEXT = "#3d0a0a", "#f87171"

LOW_FG,  LOW_BG  = "#1a7a3c", "#e0f5e9"
MED_FG,  MED_BG  = "#b45309", "#fef3c7"
HIGH_FG, HIGH_BG = "#c81e1e", "#fee2e2"


# ---------------------------------------------------------------------------
# Scenarios — four fraud-typology use cases plus an on-demand live mic
# scenario. Each maps to a specific JPM-relevant fraud pattern; together
# they cover the four failure modes a JPM reviewer would expect to
# defend against.
#
#  1. Authenticated Customer — enrolled voice playback → PASS
#     "no friction for legit calls" (the FP-cost story)
#  2. AI Voice Clone — ElevenLabs sample (JPM hears what a deepfake
#     actually sounds like) → FLAG via speaker mismatch
#  3. Real-Person Impersonator — different real human → FLAG via
#     speaker mismatch (the case a synthesis-only detector misses)
#  4. Robocall / Crude Bot — older TTS → BLOCK (both signals trip)
#
# Audio resolution:
#   - requires_registered_voice=True → uses the uploaded enrollment clip
#   - otherwise looks for `audio` filename in audio/, falls back to
#     `fallback_audio` if the preferred file isn't recorded yet
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "Real Call · Example 1": {
        "spectral": 0.05, "prosody": 0.05, "behavior": 0.10, "conf": 0.10,
        "audio": "customer_voiceprint_umair_real_call.m4a",
        "fallback_audio": "customer_voiceprint_umair.m4a",
        "caller_id":      "+1 212-555-0199",
        "claimed_name":   "Umair (registered customer)",
        "account_suffix": "0042",
        "txn_type":       "Statement question · address update",
        "txn_amount":     0,
        "txn_destination":"—",
        "prior_calls_30d": 6,
        "ivr_path":       "Self-service first, then agent (typical pattern)",
        "loss_avoidance": 0,
        "narrative": (
            "Legitimate inbound call from the registered customer. "
            "Voiceprint match on the same speaker, different content. "
            "Sanity check that the system passes legitimate calls "
            "cleanly — the false-positive cost story for JPM."
        ),
        "expected": "PASS",
    },
    "Real Call · Example 2": {
        "spectral": 0.05, "prosody": 0.05, "behavior": 0.10, "conf": 0.10,
        "audio": "real_umair.m4a",
        "caller_id":      "+1 212-555-0199",
        "claimed_name":   "Umair (registered customer)",
        "account_suffix": "0042",
        "txn_type":       "Routine inbound call",
        "txn_amount":     0,
        "txn_destination":"—",
        "prior_calls_30d": 6,
        "ivr_path":       "Self-service first, then agent (typical pattern)",
        "loss_avoidance": 0,
        "narrative": (
            "Second real-Umair recording. Same speaker as the enrolled "
            "voiceprint but different content — exercises the speaker "
            "model's ability to match across utterances, which is the "
            "actual production case (you don't always have the same "
            "phrase to compare)."
        ),
        "expected": "PASS",
    },
    "AI Clone · Neutral Script": {
        "spectral": 0.55, "prosody": 0.45, "behavior": 0.50, "conf": 0.60,
        "audio": "customer_voiceprint_umair_spoofed.mp3",
        "caller_id":      "+1 415-555-0144",
        "claimed_name":   "Umair (claimed)",
        "account_suffix": "0042",
        "txn_type":       "Wire transfer",
        "txn_amount":     27500,
        "txn_destination":"new external beneficiary (first time)",
        "prior_calls_30d": 0,
        "ivr_path":       "Direct-to-agent (skipped self-service)",
        "loss_avoidance": 27500,
        "narrative": (
            "AI clone of Umair's enrolled voice. Both signals trip — "
            "speaker match catches the voiceprint mismatch and the "
            "synthesis classifier catches the deepfake artifacts. "
            "Layered defense in action."
        ),
        "expected": "BLOCK",
    },
    "AI Clone · Urgent Script": {
        "spectral": 0.55, "prosody": 0.45, "behavior": 0.65, "conf": 0.65,
        "audio": "customer_voiceprint_urgentcall_spoof_umair.mp3",
        "caller_id":      "+1 415-555-0144",
        "claimed_name":   "Umair (claimed)",
        "account_suffix": "0042",
        "txn_type":       "Emergency wire transfer",
        "txn_amount":     50000,
        "txn_destination":"new external beneficiary (first time)",
        "prior_calls_30d": 0,
        "ivr_path":       "Direct-to-agent (skipped self-service); urgent tone",
        "loss_avoidance": 50000,
        "narrative": (
            "Same AI-clone threat as Umair Spoofed, but the attacker "
            "had the clone read an urgent fraud script — large wire to "
            "a new beneficiary, time pressure, willingness to authorize "
            "anything. Both signals catch it; urgency markers don't "
            "change the underlying detectability."
        ),
        "expected": "BLOCK",
    },
    "Robocall": {
        "spectral": 0.94, "prosody": 0.91, "behavior": 0.89, "conf": 0.94,
        "audio": "robocall_umair.wav",
        "caller_id":      "+1 800-555-0123",
        "claimed_name":   "Umair (claimed by bot)",
        "account_suffix": "0042",
        "txn_type":       "Account verification request",
        "txn_amount":     0,
        "txn_destination":"—",
        "prior_calls_30d": 4,
        "ivr_path":       "Sub-second key presses, no human pauses",
        "loss_avoidance": 87000,
        "narrative": (
            "Crude TTS robocall reading Umair's enrollment script. "
            "Both signals trip — speaker match catches the voiceprint "
            "mismatch, voice-risk catches the obvious synthesis. The "
            "easy case, but real and worth showing alongside the "
            "harder ElevenLabs spoof."
        ),
        "expected": "BLOCK",
    },
}

CALL_DURATION_S = 5.0  # short, real-time-feel analysis window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verdict(conf: float) -> str:
    if conf < 0.50:
        return "PASS"
    if conf < 0.75:
        return "FLAG"
    return "BLOCK"


def _verdict_label(v: str) -> str:
    return {
        "PASS":  "CALL CLEARED",
        "FLAG":  "FLAGGED FOR REVIEW",
        "BLOCK": "TRANSACTION BLOCKED",
    }[v]


def _level(score: float) -> str:
    if score < 0.50:
        return "LOW"
    if score < 0.75:
        return "MED"
    return "HIGH"


def _level_palette(level: str) -> tuple[str, str]:
    return {
        "LOW":  (LOW_FG, LOW_BG),
        "MED":  (MED_FG, MED_BG),
        "HIGH": (HIGH_FG, HIGH_BG),
    }[level]


def _action_for(verdict: str) -> str:
    return {
        "PASS":  "Continue with the caller's request as normal. No extra verification needed.",
        "FLAG":  "Verify the caller's identity using the one-time code that was sent before doing anything on the account.",
        "BLOCK": "Stop the call and route to Fraud Recovery. Do not process any account requests.",
    }[verdict]


@st.cache_resource(show_spinner=False)
def _get_graph():
    """LangGraph pipeline (built once, shared via Streamlit's resource cache).
    The graph compiles with a MemorySaver checkpointer so each call's
    thread_id keeps its own pause/resume state."""
    from graph import build_graph
    return build_graph()


def _invoke_pipeline(initial_state: dict, thread_id: str) -> dict:
    """Run the graph from START until it pauses at human_review.
    Returns the paused-state values dict."""
    g = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    g.invoke(initial_state, config=config)
    snap = g.get_state(config)
    return dict(snap.values)


def _resume_pipeline(decision: str, thread_id: str) -> dict:
    """Resume the paused graph with the reviewer's decision and run to END."""
    g = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    g.update_state(config, {"human_decision": decision})
    g.invoke(None, config=config)
    snap = g.get_state(config)
    return dict(snap.values)


def _pipeline_position(graph_state: dict | None) -> str:
    """Plain-language pipeline status for the dashboard banner.
    Pipeline pauses *after* agent_handoff; resume is driven by the
    reviewer's button click. Status keys off observable state fields."""
    if not graph_state:
        return "idle"
    intel = graph_state.get("intelligence_log") or {}
    if intel.get("leadership_summary"):
        if graph_state.get("transaction_blocked"):
            return "complete (blocked)"
        if graph_state.get("otp_sent"):
            return "complete (auth challenge fired)"
        return "complete"
    if graph_state.get("agent_alert_fired"):
        return "paused — awaiting reviewer decision"
    if graph_state.get("ivr_entry_confidence") is not None:
        return "in progress"
    return "started"


@st.cache_data(show_spinner=False)
def _live_scores_for(audio_path: str) -> dict:
    """Run the real deepfake detector on an audio path. Cached across reruns."""
    from detectors import detect
    return detect(audio_path)


@st.cache_data(show_spinner=False)
def _speaker_similarity(reference_path: str, test_path: str) -> float:
    """Cosine similarity between two voice clips, [-1, 1]. Cached."""
    from detectors import speaker_similarity as _sim
    return float(_sim(reference_path, test_path))


LIVE_MIC_SCENARIO = "Live Microphone Input"


def _scenario_names() -> list[str]:
    """Caller Audio Under Test options."""
    return list(SCENARIOS.keys())


def _scenario_data(name: str) -> dict:
    """Resolve a scenario name to its data dict. Live-mic is synthesized;
    everything else lives in the SCENARIOS dict."""
    if name == LIVE_MIC_SCENARIO:
        return {
            "spectral": 0.30, "prosody": 0.30, "behavior": 0.30, "conf": 0.30,
            "audio": None,  # path comes from live_mic_audio_path
            "caller_id":      "Microphone (local capture)",
            "claimed_name":   "Live Caller",
            "account_suffix": "—",
            "txn_type":       "Live recording — real-time analysis",
            "txn_amount":     0,
            "txn_destination":"—",
            "prior_calls_30d": 0,
            "ivr_path":       "Direct microphone input",
            "loss_avoidance": 0,
            "narrative": (
                "Audio captured live from the operator's microphone. "
                "Speaker match and synthesis-detection signals are "
                "computed against the registered customer voiceprint."
            ),
            "expected": "—",
        }
    return SCENARIOS[name]


def _scenario_audio_path(name: str, sc: dict) -> str | None:
    """Resolve the audio file path for a scenario.
      - Live mic → live_mic_audio_path
      - Else → looks for sc['audio'] in AUDIO_DIR; falls back to
        sc['fallback_audio'] if the preferred file isn't on disk yet.
    """
    if name == LIVE_MIC_SCENARIO:
        return st.session_state.get("live_mic_audio_path") or None
    preferred = sc.get("audio")
    if preferred:
        p = AUDIO_DIR / preferred
        if p.exists():
            return str(p)
    fallback = sc.get("fallback_audio")
    if fallback:
        f = AUDIO_DIR / fallback
        if f.exists():
            return str(f)
    return None


def _is_using_fallback_audio(name: str, sc: dict) -> bool:
    """True when the scenario is playing fallback audio because the
    preferred recording isn't on disk yet."""
    if name == LIVE_MIC_SCENARIO:
        return False
    preferred = sc.get("audio")
    fallback = sc.get("fallback_audio")
    if not preferred or not fallback:
        return False
    return not (AUDIO_DIR / preferred).exists() and (AUDIO_DIR / fallback).exists()


def _combined_verdict(
    spectral: float,
    prosody: float,
    similarity: float | None,
    scripted_conf: float,
) -> tuple[str, float]:
    """Decision logic when real signals are available.

    Calibration is anchored to ECAPA-TDNN's empirical ranges on VoxCeleb:
    same speaker typically 0.50–0.85, different speakers typically
    -0.10–0.30. The model's own verification threshold sits around 0.25.

      sim ≥ 0.35 → 0    (high-confidence match)
      sim ≤ 0.25 → 0.65 (clear mismatch, lands in FLAG band)
      between   → linear

    Deepfake score (spectral or prosody) operates independently — a clear
    synthesis pushes to BLOCK regardless of speaker match.
    """
    if similarity is None:
        return _verdict(scripted_conf), scripted_conf
    deepfake = max(spectral, prosody)
    mismatch_norm = max(0.0, min(1.0, (0.35 - similarity) / 0.10))
    mismatch_risk = mismatch_norm * 0.65  # caps mismatch alone at FLAG, not BLOCK
    risk = max(deepfake, mismatch_risk)
    return _verdict(risk), risk


def current_state() -> dict:
    """Compose the live scenario view from session state.

    Live Mode replaces scripted spectral/prosody with real Wav2Vec2 + F0
    output. When a registered voice is also uploaded, speaker similarity
    is computed and the verdict is recalculated using combined-signal
    logic (deepfake score OR speaker mismatch can trigger).
    """
    valid_names = (*SCENARIOS.keys(), LIVE_MIC_SCENARIO)
    name = st.session_state.get("preset_choice", "")
    if name not in valid_names:
        name = next(iter(SCENARIOS.keys()))
        st.session_state["preset_choice"] = name
    sc = _scenario_data(name)
    live = bool(st.session_state.get("live_mode", False))
    registered_path = st.session_state.get("registered_voice_path") or None
    audio_path = _scenario_audio_path(name, sc)

    spectral = sc["spectral"]
    prosody = sc["prosody"]
    similarity: float | None = None
    model_used = "Scripted scenario"
    if live and audio_path:
        try:
            scores = _live_scores_for(audio_path)
            spectral = scores["spectral_score"]
            prosody = scores["prosody_score"]
            model_used = scores["model_used"]
        except Exception as exc:  # noqa: BLE001
            st.session_state["live_mode_error"] = str(exc)
            live = False

    if live and registered_path and audio_path:
        try:
            if str(Path(audio_path).resolve()) == str(Path(registered_path).resolve()):
                similarity = 1.0
            else:
                similarity = _speaker_similarity(registered_path, audio_path)
        except Exception as exc:  # noqa: BLE001
            st.session_state["live_mode_error"] = (
                f"Speaker verification failed: {exc}"
            )

    if live and similarity is not None:
        verdict, conf = _combined_verdict(spectral, prosody, similarity, sc["conf"])
    else:
        verdict, conf = _verdict(sc["conf"]), sc["conf"]

    voice_risk = max(spectral, prosody)
    return {
        "name":           name,
        "spectral":       spectral,
        "prosody":        prosody,
        "behavior":       sc["behavior"],
        "voice_risk":     voice_risk,
        "agent_susp":     sc["conf"],
        "conf":           conf,
        "verdict":        verdict,
        "speaker_similarity": similarity,
        "caller_id":      sc.get("caller_id", st.session_state.get("caller_id", "+1 212-555-0199")),
        "claimed_name":   sc.get("claimed_name", "Unknown"),
        "account_suffix": sc.get("account_suffix", "0000"),
        "txn_type":       sc.get("txn_type", "—"),
        "txn_amount":     sc.get("txn_amount", 0),
        "txn_destination":sc.get("txn_destination", "—"),
        "prior_calls_30d":sc.get("prior_calls_30d", 0),
        "ivr_path":       sc.get("ivr_path", "—"),
        "narrative":      sc["narrative"],
        "audio":          sc.get("audio"),
        "audio_path":     audio_path,
        "loss_avoidance": sc["loss_avoidance"],
        "live_mode":      live,
        "model_used":     model_used,
        "registered":     bool(registered_path),
        "expected":       sc.get("expected", verdict),
    }


def _reset_call() -> None:
    """Reset to idle state — clears verdict, timer, path progress, and the
    reviewer audit log. Called whenever the scenario changes or a new voice
    recording is captured so each fresh setup starts with a clean log."""
    st.session_state["call_state"] = "idle"
    st.session_state["timer_value"] = 0.0
    st.session_state["call_progress"] = 0.0
    st.session_state["reviewer_log"] = []


# Stage breakpoints over a 0-1 progress curve. Six equal slices of the call.
_STAGE_BREAKS: list[float] = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]


def _step_class(i: int, runs: list[bool], call_state: str, progress: float) -> str:
    """Decide the CSS class for path step i (0-based)."""
    start, end = _STAGE_BREAKS[i], _STAGE_BREAKS[i + 1]
    if call_state == "complete":
        return "run" if runs[i] else "skip"
    if call_state == "idle":
        return "pending"
    # running — progressive
    if progress < start:
        return "pending"
    if progress >= end:
        return "run" if runs[i] else "skip"
    return "active" if runs[i] else "skip"


def path_taken_html(state: dict, call_state: str, progress: float) -> str:
    runs = _stage_runs(state)
    pills = "".join(
        f'<div class="vg-step {_step_class(i, runs, call_state, progress)}">'
        f'{name}</div>'
        for i, (name, _t) in enumerate(STAGE_PIPELINE)
    )
    return (
        '<div class="vg-card" style="margin-top:6px;">'
        '<div class="vg-meta-l" style="margin-bottom:6px;">Path Taken</div>'
        f'<div class="vg-path">{pills}</div>'
        '</div>'
    )


def meta_strip_html(state: dict, call_state: str, progress: float) -> str:
    session_id = st.session_state.get("session_id", 0)
    if call_state == "complete":
        verdict      = state["verdict"]
        verdict_sub  = _verdict_label(state["verdict"])
        conf_v       = f"{state['conf']*100:0.0f}%"
        conf_sub     = f"{_level(state['conf'])} risk"
        audio_v      = f"{CALL_DURATION_S:0.1f}s"
        audio_sub    = (
            f"spectral {state['spectral']:0.2f} · prosody {state['prosody']:0.2f}"
        )
    elif call_state == "running":
        verdict      = "..."
        verdict_sub  = "ANALYZING"
        conf_v       = "..."
        conf_sub     = "in progress"
        audio_v      = f"{progress * CALL_DURATION_S:0.1f}s"
        audio_sub    = "capturing voice"
    else:  # idle
        verdict      = "—"
        verdict_sub  = "AWAITING CALL"
        conf_v       = "—"
        conf_sub     = "no call yet"
        audio_v      = "0.0s"
        audio_sub    = "no audio captured"
    if call_state == "complete" and state.get("live_mode"):
        audio_sub = f"{audio_sub} · live detection"
    return (
        '<div class="vg-meta-row">'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">Call ID</div>'
        f'<div class="vg-meta-v" style="color:#1a1a1a;">#VG-{session_id:05d}</div>'
        f'<div class="vg-meta-s" style="color:#616c7d;">{state["caller_id"]}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">Final Verdict</div>'
        f'<div class="vg-meta-v" style="color:#1a1a1a;">{verdict}</div>'
        f'<div class="vg-meta-s" style="color:#616c7d;">{verdict_sub}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">AI Confidence</div>'
        f'<div class="vg-meta-v" style="color:#1a1a1a;">{conf_v}</div>'
        f'<div class="vg-meta-s" style="color:#616c7d;">{conf_sub}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">Audio + Scores</div>'
        f'<div class="vg-meta-v" style="color:#1a1a1a;">{audio_v}</div>'
        f'<div class="vg-meta-s" style="color:#616c7d;">{audio_sub}</div>'
        '</div>'
        '</div>'
    )


def placeholder_html(call_state: str) -> str:
    if call_state == "running":
        title = "Analyzing Call..."
        body = (
            "Listening to caller voice. Verdict and risk metrics will "
            "populate here when analysis completes."
        )
        dots = (
            '<div class="vg-p-dots">'
            '<span class="vg-p-dot"></span>'
            '<span class="vg-p-dot"></span>'
            '<span class="vg-p-dot"></span>'
            '</div>'
        )
        return (
            f'<div class="vg-placeholder running">'
            f'<div class="vg-p-title">{title}</div>'
            f'<div class="vg-p-body">{body}</div>'
            f'{dots}'
            '</div>'
        )
    return (
        '<div class="vg-placeholder idle">'
        '<div class="vg-p-title">Awaiting Call</div>'
        '<div class="vg-p-body">'
        'Click <b>Place Incoming Call</b> to begin analysis. Verdict, '
        'risk indicators, and the call advisory will populate here.'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def inject_css() -> None:
    css = (
        "<style>"
        # Page chrome — Chase uses a light gray canvas
        ".block-container{padding-top:0.45rem!important;padding-bottom:0.45rem!important;"
        "padding-left:1.4rem!important;padding-right:1.4rem!important;max-width:100%;}"
        "header[data-testid='stHeader']{display:none;height:0;}"
        "footer{visibility:hidden;}"
        "#MainMenu{visibility:hidden;}"
        "[data-testid='stToolbar']{display:none;}"
        # Chase uses their own font stack; -apple-system is the closest safe fallback
        f"html,body,[class*='css']{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:16px;color:{INK};}}"
        f".stApp{{background:{CANVAS};}}"

        # Header banner — solid Chase primary blue, no gradient
        f".vg-header{{background:{NAVY};"
        "padding:14px 24px;border-radius:0;color:#fff;margin-bottom:0;"
        "display:flex;align-items:center;justify-content:space-between;"
        f"border-bottom:3px solid {ACCENT};}}"
        ".vg-header .vg-h-title{font-size:22px;font-weight:700;letter-spacing:-0.3px;line-height:1.2;}"
        ".vg-header .vg-h-sub{font-size:13px;color:#a8c4e8;font-weight:400;margin-top:2px;letter-spacing:0.1px;}"
        f".vg-header .vg-h-meta{{font-size:12px;color:#93b9df;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;background:rgba(255,255,255,0.12);padding:4px 10px;border-radius:3px;}}"

        # Tabs — Chase-style: underline active, no filled pill
        f".stTabs [data-baseweb='tab-list']{{gap:0;border-bottom:2px solid {BORDER};background:#fff;padding:0 4px;}}"
        f".stTabs [data-baseweb='tab']{{background:transparent;color:{MUTED};font-weight:600;"
        f"font-size:15px;padding:10px 18px;border-radius:0;letter-spacing:0.2px;border-bottom:3px solid transparent;margin-bottom:-2px;}}"
        f".stTabs [aria-selected='true']{{color:{NAVY}!important;border-bottom:3px solid {NAVY}!important;font-weight:700!important;background:transparent!important;}}"

        # Generic card — Chase white card with subtle shadow
        f".vg-card{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;"
        "padding:12px 16px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);}}"

        # Incoming call card — Chase blue, flat/solid
        f".vg-incoming{{background:{NAVY};"
        "color:#fff;border-radius:4px;padding:14px 18px;position:relative;"
        "margin-bottom:8px;border-left:4px solid #4ade80;}}"
        ".vg-incoming .vg-i-label{font-size:11px;letter-spacing:1.8px;text-transform:uppercase;"
        "color:#93c5fd;font-weight:700;}"
        ".vg-incoming .vg-i-num{font-size:26px;font-weight:700;letter-spacing:-0.3px;"
        "margin-top:4px;font-variant-numeric:tabular-nums;}"
        ".vg-incoming .vg-i-sub{font-size:14px;color:#cbd5e1;margin-top:2px;}"
        ".vg-incoming .vg-i-live{position:absolute;top:14px;right:14px;"
        "background:rgba(74,222,128,0.2);color:#4ade80;"
        "border:1px solid rgba(74,222,128,0.5);padding:3px 10px;"
        "border-radius:3px;font-size:12px;font-weight:700;letter-spacing:0.8px;}"
        ".vg-incoming .vg-i-timer-label{font-size:11px;letter-spacing:1.6px;text-transform:uppercase;"
        "color:#93c5fd;font-weight:700;margin-top:8px;}"
        ".vg-incoming .vg-i-timer{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums;"
        "color:#fff;letter-spacing:-0.4px;line-height:1.1;}"

        # Waveform — Chase accent blue bars
        ".vg-wave{display:flex;align-items:center;gap:3px;margin-top:8px;height:28px;}"
        f".vg-wave .vg-bar{{width:3px;background:{ACCENT};border-radius:1px;opacity:0.85;"
        "animation:wave-anim 1s ease-in-out infinite;}"
        "@keyframes wave-anim{0%,100%{height:4px;}50%{height:24px;}}"

        # Verdict bar — Chase flat card style
        ".vg-verdict{border-radius:4px;padding:12px 18px;margin-bottom:8px;"
        "display:flex;justify-content:space-between;align-items:center;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.10);}"
        f".vg-verdict.pass{{background:{PASS_BG};color:{PASS_TEXT};border-left:5px solid #4ade80;}}"
        f".vg-verdict.flag{{background:{FLAG_BG};color:{FLAG_TEXT};border-left:5px solid #f59e0b;}}"
        f".vg-verdict.block{{background:{BLOCK_BG};color:{BLOCK_TEXT};border-left:5px solid #f87171;}}"
        ".vg-verdict .vg-v-label{font-size:11px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,0.9);font-weight:700;}"
        ".vg-verdict .vg-v-text{font-size:30px;font-weight:700;letter-spacing:-0.3px;color:#fff;line-height:1.1;}"
        ".vg-verdict .vg-v-sub{font-size:14px;font-weight:600;color:rgba(255,255,255,0.9);}"
        ".vg-verdict .vg-v-conf-label{font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:rgba(255,255,255,0.9);font-weight:700;text-align:right;}"
        ".vg-verdict .vg-v-conf{font-size:32px;font-weight:700;color:#fff;text-align:right;font-variant-numeric:tabular-nums;line-height:1.1;}"

        # Risk metric card — Chase flat white card
        f".vg-metric{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:12px 14px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.05);}}"
        f".vg-metric .vg-m-label{{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};}}"
        f".vg-metric .vg-m-value{{font-size:30px;font-weight:700;color:{INK};font-variant-numeric:tabular-nums;line-height:1;}}"
        f".vg-metric .vg-m-desc{{font-size:13px;color:{MUTED};line-height:1.4;margin-top:4px;font-weight:400;}}"
        ".vg-metric .vg-m-bar{height:3px;border-radius:2px;background:#e8e8e8;margin-top:8px;overflow:hidden;}"
        ".vg-metric .vg-m-bar>span{display:block;height:100%;border-radius:2px;}"

        # Badges — Chase pill-style labels
        ".vg-badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;}"
        f".vg-badge.low{{color:{LOW_FG};background:{LOW_BG};}}"
        f".vg-badge.med{{color:{MED_FG};background:{MED_BG};}}"
        f".vg-badge.high{{color:{HIGH_FG};background:{HIGH_BG};}}"

        # Advisory cards — Chase white card with left accent
        f".vg-advisory{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:12px 16px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.05);}}"
        ".vg-advisory.pass-acc{border-left:4px solid #22c55e;}"
        ".vg-advisory.flag-acc{border-left:4px solid #f59e0b;}"
        ".vg-advisory.block-acc{border-left:4px solid #ef4444;}"
        f".vg-advisory .vg-a-label{{font-size:13px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;color:{NAVY};}}"
        # Recommended action — colored per verdict
        ".vg-advisory.pass-acc .vg-a-action{background:#dcfce7;color:#14532d;border:1.5px solid #4ade80;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        ".vg-advisory.flag-acc .vg-a-action{background:#fef3c7;color:#78350f;border:1.5px solid #f59e0b;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        ".vg-advisory.block-acc .vg-a-action{background:#fee2e2;color:#7f1d1d;border:1.5px solid #ef4444;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        # OK to Do items — bold card chips
        f".vg-advisory .vg-ok-item{{display:flex;align-items:flex-start;gap:9px;background:#f8fafc;border:1px solid {BORDER};border-radius:8px;padding:8px 11px;margin-top:5px;font-size:15px;font-weight:700;color:{INK};line-height:1.4;}}"
        ".vg-advisory .vg-ok-icon{flex-shrink:0;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900;margin-top:1px;}"
        ".vg-advisory.pass-acc .vg-ok-icon{background:#dcfce7;color:#16a34a;}"
        ".vg-advisory.flag-acc .vg-ok-icon{background:#fef3c7;color:#d97706;}"
        ".vg-advisory.block-acc .vg-ok-icon{background:#fee2e2;color:#dc2626;}"
        f".vg-advisory .vg-a-narrative{{font-size:14px;color:{INK};line-height:1.5;margin-top:6px;font-weight:400;}}"
        ".vg-advisory .vg-a-pills{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;}"
        f".vg-advisory .vg-pill{{background:{CANVAS};border:1px solid {BORDER};border-radius:3px;padding:3px 10px;font-size:13px;font-weight:600;color:{INK};}}"
        f".vg-advisory .vg-pill .vg-pill-k{{color:{MUTED};font-weight:500;}}"

        # Force Chase ink on Streamlit-generated text — overrides dark-mode leakage
        f"[data-testid='stMarkdownContainer'] h1,"
        f"[data-testid='stMarkdownContainer'] h2,"
        f"[data-testid='stMarkdownContainer'] h3,"
        f"[data-testid='stMarkdownContainer']>div>p{{color:{INK}!important;}}"
        f"[data-testid='stWidgetLabel'] p,"
        f"[data-testid='stWidgetLabel'] label{{color:{INK}!important;}}"
        f"label{{color:{INK}!important;}}"

        # Inputs — Chase square-ish style
        f".stTextInput input,.stSelectbox > div > div{{border-radius:3px!important;border:1px solid {BORDER}!important;font-size:15px!important;background:#fff!important;color:{INK}!important;}}"
        f".stTextInput input:focus{{border-color:{ACCENT}!important;box-shadow:0 0 0 2px rgba(0,114,207,0.15)!important;outline:none!important;}}"
        f".stSelectbox [data-baseweb='select'] *{{color:{INK}!important;}}"
        f"""
        .stTextInput input:disabled {{
            color: {INK} !important;
            -webkit-text-fill-color: {INK} !important;
            opacity: 1 !important;
            font-weight: 600 !important;
            background-color: #f5f6f8 !important;
            border: 1px solid {BORDER} !important;
        }}
        """

        # Button — Chase primary: square corners, solid blue
        f".stButton > button{{background:{NAVY}!important;color:#fff!important;border:none!important;border-radius:4px!important;padding:11px 20px!important;font-size:15px!important;font-weight:700!important;letter-spacing:0.3px!important;width:100%;}}"
        f".stButton > button:hover{{background:{ACCENT}!important;transition:background 0.15s;}}"

        # Path Taken — Chase flat chips
        ".vg-path{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}"
        ".vg-path .vg-step{flex:1;min-width:120px;padding:7px 10px;border-radius:3px;"
        "font-size:14px;font-weight:600;text-align:center;letter-spacing:0.2px;"
        "line-height:1.3;transition:all 0.15s ease;}"
        f".vg-path .vg-step.run{{background:{NAVY};color:#fff;}}"
        f".vg-path .vg-step.skip{{background:#f0f0f0;color:{MUTED};border:1px dashed {BORDER};opacity:0.75;}}"
        f".vg-path .vg-step.active{{background:{ACCENT};color:#fff;"
        "animation:stage-pulse 1.1s ease-in-out infinite;}"
        f".vg-path .vg-step.pending{{background:{CANVAS};color:{MUTED};border:1px solid {BORDER};}}"
        "@keyframes stage-pulse{"
        "0%,100%{box-shadow:0 0 0 0 rgba(0,114,207,0.5);}"
        "50%{box-shadow:0 0 0 6px rgba(0,114,207,0);}"
        "}"

        # Audit metadata row
        f".vg-meta-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px;}}"

        # Placeholder
        f".vg-placeholder{{background:#fff;border-radius:4px;padding:36px 28px;text-align:center;border:1px solid {BORDER};}}"
        f".vg-placeholder.running{{border-color:{ACCENT};box-shadow:0 0 0 2px rgba(0,114,207,0.12);}}"
        f".vg-placeholder .vg-p-title{{font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;}}"
        f".vg-placeholder.running .vg-p-title{{color:{NAVY};}}"
        f".vg-placeholder .vg-p-body{{font-size:14px;color:{MUTED};line-height:1.5;}}"
        ".vg-placeholder .vg-p-dots{display:inline-flex;gap:5px;margin-top:12px;}"
        f".vg-placeholder .vg-p-dot{{width:7px;height:7px;border-radius:50%;background:{ACCENT};"
        "animation:dot-bounce 1s ease-in-out infinite;}"
        ".vg-placeholder .vg-p-dot:nth-child(2){animation-delay:0.15s;}"
        ".vg-placeholder .vg-p-dot:nth-child(3){animation-delay:0.30s;}"
        "@keyframes dot-bounce{"
        "0%,80%,100%{transform:translateY(0);opacity:0.4;}"
        "40%{transform:translateY(-5px);opacity:1;}"
        "}"

        # Stage rows — Chase flat card rows
        f".vg-stage-row{{display:flex;align-items:center;gap:12px;padding:9px 14px;margin-bottom:5px;border-radius:3px;border:1px solid {BORDER};background:{PAPER};box-shadow:0 1px 2px rgba(0,0,0,0.04);}}"
        f".vg-stage-row.skip{{background:{CANVAS};border:1px solid {BORDER};opacity:0.75;}}"
        ".vg-stage-row .vg-s-num{width:32px;height:32px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;flex-shrink:0;}"
        f".vg-stage-row.run .vg-s-num{{background:{NAVY};color:#fff;}}"
        f".vg-stage-row.skip .vg-s-num{{background:#e8e8e8;color:{MUTED};}}"
        ".vg-stage-row .vg-s-body{flex:1;min-width:0;}"
        ".vg-stage-row .vg-s-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}"
        f".vg-stage-row .vg-s-name{{font-size:15px;font-weight:700;color:{INK};}}"
        f".vg-stage-row .vg-s-tag{{font-size:11px;color:{MUTED};letter-spacing:1.2px;text-transform:uppercase;font-weight:600;}}"
        ".vg-stage-row .vg-s-status{font-size:11px;padding:2px 7px;border-radius:3px;letter-spacing:1px;text-transform:uppercase;font-weight:700;}"
        f".vg-stage-row.run .vg-s-status{{background:{LOW_BG};color:{LOW_FG};}}"
        f".vg-stage-row.skip .vg-s-status{{background:#e8e8e8;color:{MUTED};}}"
        f".vg-stage-row .vg-s-desc{{font-size:14px;color:{INK};margin-top:2px;line-height:1.4;font-weight:400;}}"
        f".vg-stage-row.skip .vg-s-desc{{color:{MUTED};}}"
        f".vg-stage-row .vg-s-score{{font-size:14px;font-weight:700;color:{INK};font-variant-numeric:tabular-nums;flex-shrink:0;min-width:140px;text-align:right;}}"
        f".vg-stage-row.skip .vg-s-score{{color:{MUTED};}}"
        f".vg-stage-row.pending{{background:{CANVAS};border:1px solid {BORDER};opacity:0.65;}}"
        f".vg-stage-row.pending .vg-s-num{{background:#e0e0e0;color:{MUTED};}}"
        f".vg-stage-row.pending .vg-s-status{{background:#e0e0e0;color:{MUTED};}}"
        f".vg-stage-row.pending .vg-s-desc{{color:{MUTED};}}"
        f".vg-stage-row.pending .vg-s-score{{color:{MUTED};}}"
        f".vg-stage-row.active{{background:{PAPER};border:1px solid {ACCENT};box-shadow:0 0 0 2px rgba(0,114,207,0.12);}}"
        f".vg-stage-row.active .vg-s-num{{background:{ACCENT};color:#fff;animation:stage-pulse 1.1s ease-in-out infinite;}}"
        ".vg-stage-row.active .vg-s-status{background:#dbeafe;color:#1a56c4;}"

        # Audit metadata tiles
        f".vg-meta{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:10px 14px;height:100%;box-shadow:0 1px 2px rgba(0,0,0,0.04);}}"
        # .vg-meta-l is used both inside .vg-meta tiles and inside .vg-card headers — needs a standalone rule
        f".vg-meta-l{{font-size:11px;letter-spacing:1.4px;text-transform:uppercase;color:{MUTED}!important;font-weight:700;}}"
        f".vg-meta .vg-meta-v{{font-size:22px;font-weight:700;color:{INK}!important;font-variant-numeric:tabular-nums;margin-top:4px;line-height:1.1;}}"
        f".vg-meta .vg-meta-s{{font-size:13px;color:{MUTED}!important;margin-top:3px;font-weight:500;}}"

        # HITL action panel — bigger and bolder for the demo
        f".vg-hitl{{background:{PAPER};border:1px solid {BORDER};border-radius:6px;padding:18px 22px;margin-top:10px;box-shadow:0 2px 6px rgba(0,0,0,0.10);}}"
        ".vg-hitl.flag{border-left:6px solid #f59e0b;}"
        ".vg-hitl.block{border-left:6px solid #ef4444;}"
        ".vg-hitl.pass{border-left:6px solid #22c55e;}"
        ".vg-hitl-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}"
        f".vg-hitl-title{{font-size:16px;font-weight:900;letter-spacing:1.6px;text-transform:uppercase;color:{NAVY};}}"
        ".vg-hitl-rec{font-size:14px;font-weight:800;letter-spacing:1.0px;text-transform:uppercase;padding:6px 14px;border-radius:4px;}"
        ".vg-hitl-rec.flag{background:#fef3c7;color:#78350f;border:1.5px solid #f59e0b;}"
        ".vg-hitl-rec.block{background:#fee2e2;color:#7f1d1d;border:1.5px solid #ef4444;}"
        ".vg-hitl-rec.pass{background:#dcfce7;color:#14532d;border:1.5px solid #4ade80;}"
        # Plain-language reasoning row — much more readable
        f".vg-hitl-reason{{font-size:17px;font-weight:600;color:{INK};line-height:1.55;background:{CANVAS};border:1.5px solid {BORDER};border-radius:5px;padding:14px 16px;margin-bottom:10px;}}"

        # Caller-context tile — bigger type, more breathing room
        f".vg-ctx{{background:{PAPER};border:1px solid {BORDER};border-radius:6px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);}}"
        f".vg-ctx-title{{font-size:14px;font-weight:900;letter-spacing:1.6px;text-transform:uppercase;color:{NAVY};margin-bottom:10px;}}"
        ".vg-ctx-row{display:flex;justify-content:space-between;align-items:baseline;padding:8px 0;border-bottom:1px solid #eef0f3;font-size:16px;}"
        ".vg-ctx-row:last-child{border-bottom:none;}"
        f".vg-ctx-k{{color:{MUTED};font-weight:600;}}"
        f".vg-ctx-v{{color:{INK};font-weight:800;font-variant-numeric:tabular-nums;}}"
        ".vg-ctx-v.high{color:#c81e1e;}"
        ".vg-ctx-v.med{color:#b45309;}"

        # Reviewer audit log — readable at presentation distance
        f".vg-log{{background:{PAPER};border:1px solid {BORDER};border-radius:6px;padding:14px 18px;max-height:180px;overflow-y:auto;box-shadow:0 1px 3px rgba(0,0,0,0.06);}}"
        f".vg-log-title{{font-size:14px;font-weight:900;letter-spacing:1.6px;text-transform:uppercase;color:{NAVY};margin-bottom:10px;}}"
        ".vg-log-row{display:flex;gap:10px;align-items:center;font-size:15px;padding:7px 0;border-bottom:1px solid #eef0f3;}"
        ".vg-log-row:last-child{border-bottom:none;}"
        f".vg-log-ts{{color:{MUTED};font-variant-numeric:tabular-nums;font-size:13px;font-weight:600;}}"
        ".vg-log-decision{font-weight:800;padding:3px 10px;border-radius:3px;font-size:12px;letter-spacing:0.7px;}"
        ".vg-log-decision.approve{background:#dcfce7;color:#14532d;}"
        ".vg-log-decision.stepup{background:#fef3c7;color:#78350f;}"
        ".vg-log-decision.block{background:#fee2e2;color:#7f1d1d;}"
        f".vg-log-name{{color:{INK};font-weight:700;}}"
        f".vg-log-empty{{color:{MUTED};font-size:15px;font-style:italic;padding:6px 0;}}"

        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_header() -> None:
    session_label = f"VG-{st.session_state.get('session_id', 0):05d}"
    html = (
        '<div class="vg-header">'
        '<div>'
        '<div class="vg-h-title">VoiceGuard AI · Fraud Operations Dashboard</div>'
        '</div>'
        f'<div class="vg-h-meta">Session #{session_label}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def incoming_call_html(caller_id: str, _scenario_name: str, timer_value: float) -> str:
    bars = "".join(
        f'<div class="vg-bar" style="animation-delay:{i*0.08}s;"></div>'
        for i in range(9)
    )
    return (
        f'<div class="vg-incoming" style="background:#003087;color:#fff;border-radius:4px;'
        f'padding:14px 18px;position:relative;margin-bottom:8px;border-left:4px solid #4ade80;">'
        '<div class="vg-i-live">● LIVE</div>'
        '<div class="vg-i-label" style="color:#93c5fd;">Incoming Call</div>'
        f'<div class="vg-i-num" style="color:#fff;">{caller_id}</div>'
        '<div class="vg-i-sub" style="color:#cbd5e1;">Chase Fraud Helpline · IVR</div>'
        '<div class="vg-i-timer-label" style="color:#93c5fd;">Audio Duration</div>'
        f'<div class="vg-i-timer" style="color:#fff;">{timer_value:0.1f}s</div>'
        f'<div class="vg-wave">{bars}</div>'
        '</div>'
    )


def render_verdict_bar(verdict: str, conf: float) -> None:
    klass = verdict.lower()
    label = _verdict_label(verdict)
    html = (
        f'<div class="vg-verdict {klass}">'
        '<div>'
        '<div class="vg-v-label">VoiceGuard Verdict</div>'
        f'<div class="vg-v-text">{verdict}</div>'
        f'<div class="vg-v-sub">{label}</div>'
        '</div>'
        '<div>'
        '<div class="vg-v-conf-label">AI Confidence</div>'
        f'<div class="vg-v-conf">{conf*100:0.0f}%</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metric(label: str, score: float, description: str, inverted: bool = False) -> None:
    """Render a metric tile. inverted=True means higher=better (e.g. speaker
    match) — risk badging flips to LOW for high values, and the badge text
    switches to MATCH / PARTIAL / MISMATCH so the label reads correctly."""
    risk_score = (1.0 - score) if inverted else score
    level = _level(risk_score)
    fg, _bg = _level_palette(level)
    pct = max(2, score * 100)
    if inverted:
        badge_text = {"LOW": "MATCH", "MED": "PARTIAL", "HIGH": "MISMATCH"}[level]
    else:
        badge_text = level
    html = (
        '<div class="vg-metric">'
        f'<div class="vg-m-label">{label}</div>'
        '<div style="display:flex;align-items:baseline;gap:8px;margin-top:3px;">'
        f'<div class="vg-m-value">{score:0.2f}</div>'
        f'<span class="vg-badge {level.lower()}">{badge_text}</span>'
        '</div>'
        f'<div class="vg-m-desc">{description}</div>'
        f'<div class="vg-m-bar"><span style="width:{pct:0.0f}%;background:{fg};"></span></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _plain_reason(state: dict) -> str:
    """Plain-language one-liner explaining what the signals mean."""
    sim = state.get("speaker_similarity")
    deepfake = max(state.get("spectral", 0.0), state.get("prosody", 0.0))
    verdict = state["verdict"]
    parts: list[str] = []
    if sim is not None:
        if sim >= 0.40:
            parts.append("voiceprint matches the enrolled customer")
        elif sim >= 0.25:
            parts.append("voiceprint is a borderline match")
        else:
            parts.append("voiceprint does NOT match the enrolled customer")
    if deepfake >= 0.75:
        parts.append("audio shows clear synthesis artifacts")
    elif deepfake >= 0.40:
        parts.append("some synthesis artifacts present")
    else:
        parts.append("no synthesis artifacts detected")
    if state.get("behavior", 0) >= 0.75:
        parts.append("IVR navigation looks bot-like")
    summary = "; ".join(parts).capitalize() + "."
    if verdict == "PASS":
        return summary + " No action needed beyond standard call handling."
    if verdict == "FLAG":
        return summary + " Recommend step-up authentication before processing the request."
    return summary + " Recommend immediate block and routing to Fraud Recovery."


def caller_context_html(state: dict) -> str:
    amount = state.get("txn_amount", 0)
    amount_str = f"${amount:,}" if amount else "—"
    risk_class = "high" if amount >= 25000 else ("med" if amount >= 5000 else "")
    prior = int(state.get("prior_calls_30d", 0))
    prior_class = "high" if prior == 0 else ""
    rows = [
        ("Claimed customer",  f"{state.get('claimed_name','—')} · ****{state.get('account_suffix','0000')}", ""),
        ("Caller ID",         state.get("caller_id", "—"), ""),
        ("Transaction",       f"{state.get('txn_type','—')}", ""),
        ("Amount in flight",  amount_str, risk_class),
        ("Destination",       state.get("txn_destination", "—"), ""),
        ("Prior calls (30d)", f"{prior}", prior_class),
        ("IVR path",          state.get("ivr_path", "—"), ""),
    ]
    rows_html = "".join(
        f'<div class="vg-ctx-row">'
        f'<span class="vg-ctx-k">{k}</span>'
        f'<span class="vg-ctx-v {cls}">{v}</span>'
        f'</div>'
        for k, v, cls in rows
    )
    return (
        '<div class="vg-ctx">'
        '<div class="vg-ctx-title">Caller Context</div>'
        f'{rows_html}'
        '</div>'
    )


def hitl_header_html(state: dict) -> str:
    verdict = state["verdict"]
    rec = {"PASS": "Approve", "FLAG": "Step-Up Auth", "BLOCK": "Block & Escalate"}[verdict]
    klass = verdict.lower()
    return (
        f'<div class="vg-hitl {klass}">'
        '<div class="vg-hitl-head">'
        '<div class="vg-hitl-title">Reviewer Action</div>'
        f'<div class="vg-hitl-rec {klass}">Recommended: {rec}</div>'
        '</div>'
        f'<div class="vg-hitl-reason">{_plain_reason(state)}</div>'
        '</div>'
    )


def audit_log_html() -> str:
    decisions = st.session_state.get("reviewer_log", [])
    if not decisions:
        body = '<div class="vg-log-empty">No reviewer decisions in this session yet.</div>'
    else:
        rows = []
        for d in reversed(decisions[-20:]):
            cls = {"approve": "approve", "stepup": "stepup", "block": "block"}[d["decision"]]
            label = {"approve": "APPROVED", "stepup": "STEP-UP", "block": "BLOCKED"}[d["decision"]]
            rows.append(
                '<div class="vg-log-row">'
                f'<span class="vg-log-ts">{d["ts"]}</span>'
                f'<span class="vg-log-decision {cls}">{label}</span>'
                f'<span class="vg-log-name">{d["scenario"]}</span>'
                '</div>'
            )
        body = "".join(rows)
    return (
        '<div class="vg-log">'
        '<div class="vg-log-title">Reviewer Audit Log (this session)</div>'
        f'{body}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Tab 1 — Live Simulation
# ---------------------------------------------------------------------------
def _render_pending_metric(label: str, hint: str = "Analyzing…") -> None:
    """Skeleton tile shown for a signal that hasn't streamed in yet."""
    html = (
        '<div class="vg-metric" style="opacity:0.65;">'
        f'<div class="vg-m-label">{label}</div>'
        '<div style="display:flex;align-items:baseline;gap:8px;margin-top:3px;">'
        f'<div class="vg-m-value" style="color:{MUTED};">—</div>'
        '<span class="vg-badge" style="background:#e5e7eb;color:#6b7280;">PENDING</span>'
        '</div>'
        f'<div class="vg-m-desc">{hint}</div>'
        '<div class="vg-m-bar"><span style="width:0%;"></span></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_analyzing_verdict_bar() -> None:
    html = (
        '<div class="vg-verdict" style="background:#1f2937;color:#fff;border-left:5px solid #6b7280;">'
        '<div>'
        '<div class="vg-v-label">VoiceGuard Verdict</div>'
        '<div class="vg-v-text">ANALYZING</div>'
        '<div class="vg-v-sub">Signals streaming in…</div>'
        '</div>'
        '<div>'
        '<div class="vg-v-conf-label">AI Confidence</div>'
        '<div class="vg-v-conf">…</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_streaming_right(slot, state: dict, stage: int) -> None:
    """Render the right column at a partial reveal level.

    stage = 0  → all placeholders (Analyzing)
    stage = 1  → Speaker Match revealed
    stage = 2  → + Voice Risk
    stage = 3  → + Agent Suspicion + Behavior Risk
    stage = 4  → + Verdict bar (replaces the analyzing placeholder)
    stage = 5  → + HITL panel + Caller Context (final)
    """
    sim = state.get("speaker_similarity")
    n_cols = 4 if sim is not None else 3
    with slot.container():
        if stage >= 4:
            render_verdict_bar(state["verdict"], state["conf"])
        else:
            _render_analyzing_verdict_bar()

        cols = st.columns(n_cols)
        i = 0
        if sim is not None:
            with cols[i]:
                if stage >= 1:
                    render_metric(
                        "Speaker Match", float(sim),
                        "Voiceprint similarity to enrolled customer.",
                        inverted=True,
                    )
                else:
                    _render_pending_metric("Speaker Match", "Comparing voiceprint…")
            i += 1
        with cols[i]:
            if stage >= 2:
                render_metric(
                    "Voice Risk", state["voice_risk"],
                    "Synthesis artifacts in the audio.",
                )
            else:
                _render_pending_metric("Voice Risk", "Detecting synthesis…")
        i += 1
        with cols[i]:
            if stage >= 3:
                render_metric(
                    "Agent Suspicion", state["agent_susp"],
                    "Combined signal for the live agent.",
                )
            else:
                _render_pending_metric("Agent Suspicion")
        i += 1
        with cols[i]:
            if stage >= 3:
                render_metric(
                    "Behavior Risk", state["behavior"],
                    "IVR navigation pattern anomaly.",
                )
            else:
                _render_pending_metric("Behavior Risk")

        if stage >= 5:
            st.markdown(hitl_header_html(state), unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            verdict = state["verdict"]
            with b1:
                approve = st.button(
                    "✓ Approve & Continue",
                    key=f"hitl_approve_{st.session_state.get('call_seq', 0)}",
                    disabled=(verdict == "BLOCK"),
                    use_container_width=True,
                )
            with b2:
                stepup = st.button(
                    "⚠ Step-Up Auth",
                    key=f"hitl_stepup_{st.session_state.get('call_seq', 0)}",
                    use_container_width=True,
                )
            with b3:
                block = st.button(
                    "✕ Block & Escalate",
                    key=f"hitl_block_{st.session_state.get('call_seq', 0)}",
                    use_container_width=True,
                )
            if approve or stepup or block:
                decision = "approve" if approve else ("stepup" if stepup else "block")
                st.session_state.setdefault("reviewer_log", []).append({
                    "ts":       time.strftime("%H:%M:%S"),
                    "scenario": state["name"],
                    "decision": decision,
                    "verdict":  verdict,
                })
                st.session_state["call_seq"] = st.session_state.get("call_seq", 0) + 1
                st.rerun()

            ctx_col, log_col = st.columns([6, 4])
            with ctx_col:
                st.markdown(caller_context_html(state), unsafe_allow_html=True)
            with log_col:
                st.markdown(audit_log_html(), unsafe_allow_html=True)


def _render_complete_right(slot, state: dict) -> None:
    """Render the right-side stack: verdict, signals, HITL panel + context."""
    with slot.container():
        render_verdict_bar(state["verdict"], state["conf"])

        sim = state.get("speaker_similarity")
        n_cols = 4 if sim is not None else 3
        cols = st.columns(n_cols)
        i = 0
        if sim is not None:
            with cols[i]:
                render_metric(
                    "Speaker Match", float(sim),
                    "Voiceprint similarity to enrolled customer.",
                    inverted=True,
                )
            i += 1
        with cols[i]:
            render_metric(
                "Voice Risk", state["voice_risk"],
                "Synthesis artifacts in the audio.",
            )
        i += 1
        with cols[i]:
            render_metric(
                "Agent Suspicion", state["agent_susp"],
                "Combined signal for the live agent.",
            )
        i += 1
        with cols[i]:
            render_metric(
                "Behavior Risk", state["behavior"],
                "IVR navigation pattern anomaly.",
            )

        st.markdown(hitl_header_html(state), unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        verdict = state["verdict"]
        with b1:
            approve = st.button(
                "✓ Approve & Continue",
                key=f"hitl_approve_{st.session_state.get('call_seq', 0)}",
                disabled=(verdict == "BLOCK"),
                use_container_width=True,
            )
        with b2:
            stepup = st.button(
                "⚠ Step-Up Auth",
                key=f"hitl_stepup_{st.session_state.get('call_seq', 0)}",
                use_container_width=True,
            )
        with b3:
            block = st.button(
                "✕ Block & Escalate",
                key=f"hitl_block_{st.session_state.get('call_seq', 0)}",
                use_container_width=True,
            )
        if approve or stepup or block:
            decision = "approve" if approve else ("stepup" if stepup else "block")
            # Resume the LangGraph pipeline with this decision. The graph
            # routes to auth_challenge (stepup) or straight to intelligence
            # (approve / block) and runs to END.
            tid = st.session_state.get("graph_thread_id")
            try:
                if tid:
                    final_state = _resume_pipeline(decision, tid)
                    st.session_state["graph_state"] = final_state
            except Exception as exc:  # noqa: BLE001
                st.session_state["graph_error"] = str(exc)
            st.session_state.setdefault("reviewer_log", []).append({
                "ts":       time.strftime("%H:%M:%S"),
                "scenario": state["name"],
                "decision": decision,
                "verdict":  verdict,
            })
            st.session_state["call_seq"] = st.session_state.get("call_seq", 0) + 1
            st.rerun()

        # Pipeline status banner — shows where the call sits in the graph
        gs = st.session_state.get("graph_state")
        pos = _pipeline_position(gs)
        if gs:
            st.markdown(_pipeline_status_html(pos, gs), unsafe_allow_html=True)

        ctx_col, log_col = st.columns([6, 4])
        with ctx_col:
            st.markdown(caller_context_html(state), unsafe_allow_html=True)
        with log_col:
            st.markdown(audit_log_html(), unsafe_allow_html=True)


def _pipeline_status_html(position: str, graph_state: dict) -> str:
    decision = graph_state.get("human_decision", "pending")
    intel = graph_state.get("intelligence_log") or {}
    review_complete = bool(intel.get("leadership_summary"))
    pieces = ["Voice Cloning ✓", "IVR Entry ✓"]
    if graph_state.get("agent_alert_fired"):
        if review_complete:
            pieces.append(f"Agent Handoff ✓ — Reviewer: {decision}")
        else:
            pieces.append("⏸ Agent Handoff — awaiting reviewer")
    if graph_state.get("otp_sent"):
        pieces.append("Auth Challenge ✓")
    if review_complete:
        pieces.append("Intelligence ✓")
    chips = "".join(
        f'<span style="background:#003087;color:#fff;padding:4px 10px;border-radius:3px;'
        f'font-size:13px;font-weight:700;letter-spacing:0.4px;margin-right:6px;'
        f'display:inline-block;margin-bottom:4px;">{p}</span>'
        for p in pieces
    )
    return (
        f'<div class="vg-card" style="margin-top:10px;padding:14px 18px;">'
        f'{chips}'
        f'</div>'
    )


def render_live_simulation() -> dict:
    left, right = st.columns([3, 7], gap="medium")

    with left:
        incoming_slot = st.empty()

        # ─── Customer Voiceprint — dropdown + audio player ───────────
        st.markdown("##### 🟢  Customer Voiceprint")
        st.caption("Customer: **Umair Khan** · Account ****0042")
        vp_choice = st.selectbox(
            "Voiceprint",
            list(VOICEPRINTS.keys()),
            key="voiceprint_choice",
            on_change=_reset_call,
            label_visibility="collapsed",
        )
        vp_path = AUDIO_DIR / VOICEPRINTS[vp_choice]
        if vp_path.exists():
            if st.session_state.get("registered_voice_path") != str(vp_path):
                st.session_state["registered_voice_path"] = str(vp_path)
                st.session_state["registered_voice_name"] = vp_path.name
                _reset_call()
            # Pass bytes + explicit MIME so Streamlit Cloud serves the file
            # with the right Content-Type. Path-only call fails silently for
            # .m4a on some runners (the browser receives octet-stream and
            # never renders the audio control).
            st.audio(vp_path.read_bytes(), format=_audio_mime(vp_path))
            has_reg = True
        else:
            st.error(f"⚠ Missing audio file: `audio/{vp_path.name}`")
            has_reg = False

        # ─── Caller Audio Under Test — orange "under test" dot ───────
        st.markdown("##### 🟠  Caller Audio Under Test")
        st.selectbox(
            "Caller audio",
            _scenario_names() if has_reg else ["— voiceprint missing —"],
            key="preset_choice",
            on_change=_reset_call,
            label_visibility="collapsed",
            disabled=not has_reg,
        )

        if st.session_state.get("live_mode_error"):
            st.caption(f"⚠ {st.session_state['live_mode_error']}")

        # ─── Connect ─────────────────────────────────────────────────
        connect_disabled = not has_reg
        place_call = st.button(
            "▶  Connect Incoming Call",
            type="primary",
            disabled=connect_disabled,
            help="Voiceprint missing" if not has_reg else None,
        )
        audio_slot = st.empty()

    with right:
        result_slot = st.empty()

    # Full-width meta strip + pipeline path
    meta_slot = st.empty()
    path_slot = st.empty()

    state = current_state()
    call_state = st.session_state.get("call_state", "idle")
    if place_call:
        call_state = "running"
        st.session_state["call_state"] = "running"

    if call_state == "complete":
        timer_value = CALL_DURATION_S
        progress = 1.0
    else:
        timer_value = 0.0
        progress = 0.0

    incoming_slot.markdown(
        incoming_call_html(state["caller_id"], state["name"], timer_value),
        unsafe_allow_html=True,
    )
    meta_slot.markdown(
        meta_strip_html(state, call_state, progress), unsafe_allow_html=True
    )
    path_slot.markdown(
        path_taken_html(state, call_state, progress), unsafe_allow_html=True
    )

    if call_state == "complete":
        _render_complete_right(result_slot, state)
    else:
        result_slot.markdown(placeholder_html(call_state), unsafe_allow_html=True)

    return {
        "place_call":    place_call,
        "state":         state,
        "incoming_slot": incoming_slot,
        "audio_slot":    audio_slot,
        "result_slot":   result_slot,
        "meta_slot":     meta_slot,
        "path_slot":     path_slot,
    }


def render_stage_audit() -> dict:
    """Tab 2 — stage-by-stage technical audit. Shares slot pattern with the
    main dashboard so animation updates can target both tabs."""
    state = current_state()
    call_state = st.session_state.get("call_state", "idle")
    progress = 1.0 if call_state == "complete" else 0.0
    meta_slot = st.empty()
    path_slot = st.empty()
    stage_slot = st.empty()
    meta_slot.markdown(
        meta_strip_html(state, call_state, progress), unsafe_allow_html=True
    )
    path_slot.markdown(
        path_taken_html(state, call_state, progress), unsafe_allow_html=True
    )
    stage_slot.markdown(
        stage_detail_html(state, call_state, progress), unsafe_allow_html=True
    )
    return {
        "meta_slot":  meta_slot,
        "path_slot":  path_slot,
        "stage_slot": stage_slot,
    }


def run_call_animation(sim: dict, audit: dict) -> None:
    """Run the real-time-feel analysis animation. Invokes the LangGraph
    pipeline at the start (it pauses at human_review for the reviewer),
    plays audio, and animates signal reveal in parallel."""
    state = sim["state"]
    audio_path = Path(state["audio_path"]) if state.get("audio_path") else None
    if audio_path and audio_path.exists():
        fmt = "audio/mp3" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
        sim["audio_slot"].audio(str(audio_path), format=fmt, autoplay=True)

    # Drive the LangGraph pipeline through to its human-review pause.
    # Thread ID is timestamp-unique so consecutive Connect clicks don't
    # collide on a stale checkpoint. The dashboard's already-computed
    # scores are seeded into initial state so node1's cached inference
    # is the only model work; downstream nodes (IVR Entry, Navigation,
    # Agent Handoff) populate the journey trace + alert message.
    thread_id = f"call-{int(time.time() * 1000)}"
    initial_graph_state = {
        "caller_id":             state["caller_id"],
        "audio_path":            str(audio_path) if audio_path else "",
        "live_mode":             True,
        "registered_voice_path": st.session_state.get("registered_voice_path", ""),
        "spectral_score":        float(state["spectral"]),
        "prosody_score":         float(state["prosody"]),
        "speaker_similarity":    (
            float(state["speaker_similarity"])
            if state.get("speaker_similarity") is not None else 0.0
        ),
        "verdict":               state["verdict"],
        "human_decision":        "pending",
        "journey_trace":         [],
    }
    try:
        paused_state = _invoke_pipeline(initial_graph_state, thread_id)
        st.session_state["graph_state"] = paused_state
        st.session_state["graph_thread_id"] = thread_id
    except Exception as exc:  # noqa: BLE001
        st.session_state["graph_state"] = None
        st.session_state["graph_error"] = str(exc)

    st.session_state["session_calls"].append({
        "name":      state["name"],
        "caller_id": state["caller_id"],
        "verdict":   state["verdict"],
        "conf":      state["conf"],
        "ts":        time.time(),
    })

    reveal_at = {
        0:  0,  # placeholders
        10: 1,  # Speaker Match
        20: 2,  # + Voice Risk
        30: 3,  # + Agent Suspicion / Behavior Risk
        40: 4,  # + Verdict bar
    }
    current_stage = -1

    steps = int(CALL_DURATION_S * 10) + 1
    for i in range(steps):
        t = i / 10.0
        progress = t / CALL_DURATION_S
        st.session_state["timer_value"] = t
        st.session_state["call_progress"] = progress
        sim["incoming_slot"].markdown(
            incoming_call_html(state["caller_id"], state["name"], t),
            unsafe_allow_html=True,
        )
        meta = meta_strip_html(state, "running", progress)
        path = path_taken_html(state, "running", progress)
        sim["meta_slot"].markdown(meta, unsafe_allow_html=True)
        sim["path_slot"].markdown(path, unsafe_allow_html=True)
        audit["meta_slot"].markdown(meta, unsafe_allow_html=True)
        audit["path_slot"].markdown(path, unsafe_allow_html=True)
        audit["stage_slot"].markdown(
            stage_detail_html(state, "running", progress),
            unsafe_allow_html=True,
        )
        if i in reveal_at and reveal_at[i] != current_stage:
            current_stage = reveal_at[i]
            _render_streaming_right(sim["result_slot"], state, current_stage)
        time.sleep(0.1)

    st.session_state["call_state"] = "complete"
    st.session_state["call_progress"] = 1.0
    _render_complete_right(sim["result_slot"], state)
    final_meta = meta_strip_html(state, "complete", 1.0)
    final_path = path_taken_html(state, "complete", 1.0)
    sim["meta_slot"].markdown(final_meta, unsafe_allow_html=True)
    sim["path_slot"].markdown(final_path, unsafe_allow_html=True)
    audit["meta_slot"].markdown(final_meta, unsafe_allow_html=True)
    audit["path_slot"].markdown(final_path, unsafe_allow_html=True)
    audit["stage_slot"].markdown(
        stage_detail_html(state, "complete", 1.0), unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Tab 2 — Call Risk Audit
# ---------------------------------------------------------------------------
STAGE_PIPELINE: list[tuple[str, str]] = [
    ("Voice Cloning Detector", "Baseline"),
    ("IVR Entry",              "Defense 1"),
    ("IVR Navigation",         "Defense 2"),
    ("Agent Handoff",          "Defense 3"),
    ("Auth Challenge",         "Defense 4"),
    ("Intelligence",           "Leadership"),
]


def _stage_runs(state: dict) -> list[bool]:
    return [
        True,
        True,
        state["voice_risk"] <= 0.80,
        True,
        state["agent_susp"] >= 0.50,
        True,
    ]


def _stage_detail(idx: int, state: dict) -> tuple[str, str]:
    """Return (description, score_text) for stage idx (1..6)."""
    if idx == 1:
        sim = state.get("speaker_similarity")
        score = (
            f"spectral {state['spectral']:0.2f} · prosody {state['prosody']:0.2f}"
        )
        if sim is not None:
            score = f"speaker {float(sim):0.2f} · {score}"
        return ("Captured voice biometrics from the call audio.", score)
    if idx == 2:
        return (
            "Logged entry confidence at the start of the call.",
            f"{state['voice_risk']:0.2f}",
        )
    if idx == 3:
        if state["voice_risk"] <= 0.80:
            return (
                f"Entry confidence {state['voice_risk']:0.2f} ≤ 0.80, ran the navigation check.",
                f"{state['voice_risk']:0.2f} ≤ 0.80",
            )
        return (
            f"Entry confidence {state['voice_risk']:0.2f} > 0.80, fast tracked. No navigation check.",
            f"{state['voice_risk']:0.2f} > 0.80",
        )
    if idx == 4:
        return (
            "Sent the alert packet with caller scores to the live agent.",
            "alert sent",
        )
    if idx == 5:
        if state["agent_susp"] >= 0.50:
            return (
                f"Agent suspicion {state['agent_susp']:0.2f} ≥ 0.50, OTP challenge required.",
                f"{state['agent_susp']:0.2f} ≥ 0.50",
            )
        return (
            f"Agent suspicion {state['agent_susp']:0.2f} < 0.50, no extra check needed.",
            f"{state['agent_susp']:0.2f} < 0.50",
        )
    return (
        "Logged session metrics and P&L impact.",
        f"loss avoided ${state['loss_avoidance']:,}",
    )


def stage_detail_html(state: dict, call_state: str, progress: float) -> str:
    runs = _stage_runs(state)
    rows = []
    for i, (name, tag) in enumerate(STAGE_PIPELINE, start=1):
        cls = _step_class(i - 1, runs, call_state, progress)
        if cls == "pending":
            row_cls, status = "pending", "Pending"
            desc, score = "Awaiting analysis.", "—"
        elif cls == "active":
            row_cls, status = "active", "Running"
            desc, score = _stage_detail(i, state)
        elif cls == "run":
            row_cls, status = "run", "Executed"
            desc, score = _stage_detail(i, state)
        else:  # skip
            row_cls, status = "skip", "Skipped"
            desc, score = _stage_detail(i, state)
        rows.append(
            f'<div class="vg-stage-row {row_cls}">'
            f'<div class="vg-s-num">{i}</div>'
            '<div class="vg-s-body">'
            '<div class="vg-s-head">'
            f'<span class="vg-s-name">{name}</span>'
            f'<span class="vg-s-tag">{tag.upper()}</span>'
            f'<span class="vg-s-status">{status}</span>'
            '</div>'
            f'<div class="vg-s-desc">{desc}</div>'
            '</div>'
            f'<div class="vg-s-score">{score}</div>'
            '</div>'
        )
    return (
        '<div class="vg-card">'
        '<div class="vg-meta-l" style="margin-bottom:6px;">Stage Detail</div>'
        f'{"".join(rows)}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Init + main
# ---------------------------------------------------------------------------
def init_session() -> None:
    ss = st.session_state
    ss.setdefault("preset_choice",  next(iter(SCENARIOS.keys())))
    ss.setdefault("caller_id",      "+1 212-555-0199")
    ss.setdefault("timer_value",    0.0)
    ss.setdefault("call_state",     "idle")          # idle | running | complete
    ss.setdefault("call_progress",  0.0)
    ss.setdefault("session_calls", [])
    ss.setdefault("session_id",     int(time.time()) % 100000)
    ss["live_mode"] = True  # detection is always on; flag retained for legacy paths
    ss.setdefault("live_mode_error", "")
    ss.setdefault("live_mic_audio_path", "")
    ss.setdefault("reviewer_log",  [])
    ss.setdefault("call_seq",      0)
    ss.setdefault("graph_state",   None)
    ss.setdefault("graph_thread_id", "")
    ss.setdefault("graph_error",   "")

    # Auto-pin the customer voiceprint to the file on disk. No upload UI.
    if REGISTERED_VOICE_FILE.exists():
        ss["registered_voice_path"] = str(REGISTERED_VOICE_FILE)
        ss["registered_voice_name"] = REGISTERED_VOICE_FILE.name
    else:
        ss["registered_voice_path"] = ""
        ss["registered_voice_name"] = ""


def render_business_case() -> None:
    """Pre-demo "set the scene" tab.

    Tight analytical framing of the business problem, why current tools
    fail, our two-signal + HITL approach, comparison vs alternatives,
    and a current-state journey map. Single scrollable page; not
    interactive. Reviewer reads, then clicks over to the Live Dashboard.
    """
    NAVY_TOP = "#001a4d"
    RED      = "#c81e1e"
    AMBER    = "#b45309"

    # ─────────────────────────────────────────────────────────────────
    # Hero — title + subtitle + 4 KPI tiles
    # ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:{NAVY_TOP};color:#fff;padding:32px 36px;
        border-radius:6px;margin-top:8px;margin-bottom:14px;
        border-left:6px solid {RED};'>
          <div style='font-size:38px;font-weight:900;line-height:1.1;
          letter-spacing:-0.5px;'>A New Kind of Attacker Is Calling Your Bank.</div>
          <div style='font-size:18px;color:#cbd5e1;margin-top:14px;
          line-height:1.5;font-weight:400;max-width:920px;'>
            AI-generated voice agents are impersonating real customers —
            bypassing IVR, fooling biometrics, and manipulating live agents
            in real time.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpis = [
        ("$12.5B",  "in contact-center fraud losses · 2024 alone",      RED),
        ("1,210%",  "increase in AI-enabled contact-center fraud",       NAVY),
        ("1 in 127","retail calls fraudulent right now",                 ACCENT),
        ("24.5%",   "human detection rate · agents miss 3 of every 4",  NAVY_TOP),
    ]
    cols = st.columns(4)
    for col, (val, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div style='background:{color};color:#fff;border-radius:6px;
                padding:22px 18px;height:100%;text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);'>
                  <div style='font-size:36px;font-weight:900;letter-spacing:-0.5px;
                  font-family:Georgia,serif;'>{val}</div>
                  <div style='font-size:12px;color:rgba(255,255,255,0.85);
                  margin-top:8px;line-height:1.4;font-weight:500;'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────
    # Section 1 — The Problem (single column, analytical)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{ACCENT};letter-spacing:2.2px;"
        f"text-transform:uppercase;font-weight:900;border-left:5px solid {ACCENT};"
        f"padding-left:12px;'>01 · The Problem</div>"
        f"<h2 style='font-size:42px;font-weight:900;color:{NAVY_TOP};"
        f"font-family:Georgia,serif;margin:8px 0 18px 0;line-height:1.15;'>"
        f"Banks built contact centers to serve people. That assumption is now the vulnerability.</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style='background:{NAVY_TOP};color:#fff;padding:20px 24px;
        border-radius:6px;margin-bottom:14px;'>
          <div style='font-size:18px;line-height:1.55;font-weight:500;'>
            AI voice agents call bank contact centers and impersonate real
            customers to bypass authentication. The asymmetry is brutal:
            an attacker needs <b>3–5 seconds of audio</b> to clone a
            customer's voice — and human agents catch only <b>24.5%</b> of
            those calls. Fraudsters succeed on <b>three out of every four
            attempts</b>, while the volume of attempts has grown
            <b>1,210%</b> in a year.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 attack vectors as a strip — kept tight
    attacks = [
        ("①", "IVR Bypass", "Synthetic voices pass automated phone systems before any human is involved.", NAVY),
        ("②", "Biometric Defeat", "Voice clones built from 3–5 sec of audio fool dedicated voice authentication systems.", RED),
        ("③", "Agent Manipulation", "AI bots reach live agents and socially engineer account changes — resets, transfers, unlocks.", NAVY),
    ]
    cols = st.columns(3, gap="medium")
    for col, (num, title, desc, accent) in zip(cols, attacks):
        with col:
            st.markdown(
                f"""
                <div style='background:#fff;border:1px solid {BORDER};
                border-left:4px solid {accent};border-radius:4px;
                padding:14px 16px;height:100%;
                display:grid;grid-template-columns:auto 1fr;gap:14px;
                align-items:start;'>
                  <div style='font-size:24px;font-weight:900;color:{accent};line-height:1;'>{num}</div>
                  <div>
                    <div style='font-size:15px;font-weight:800;color:{INK};'>{title}</div>
                    <div style='font-size:13.5px;color:{INK};margin-top:5px;line-height:1.45;'>{desc}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div style='background:#fef2f2;border-left:4px solid {RED};
        border-radius:4px;padding:13px 18px;margin-top:12px;
        font-size:14px;color:{INK};line-height:1.55;'>
          <b style='color:{RED};'>Evidence:</b> MSUFCU disclosed
          <b>$2.57M</b> in deepfake-driven exposure over 14 months —
          <i>discovered only after</i> deploying AI voice screening. By the
          time conventional fraud tools flagged the pattern, the attacks
          were already inside.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────
    # Section 2 — Why Today's Tools Fail (analytical)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{ACCENT};letter-spacing:2.2px;"
        f"text-transform:uppercase;font-weight:900;border-left:5px solid {ACCENT};"
        f"padding-left:12px;'>02 · Gap Analysis</div>"
        f"<h2 style='font-size:42px;font-weight:900;color:{NAVY_TOP};"
        f"font-family:Georgia,serif;margin:8px 0 18px 0;line-height:1.15;'>"
        f"Why Today's Tools Fail</h2>",
        unsafe_allow_html=True,
    )

    gaps = [
        ("Voice Biometrics",
         "Pindrop · Nuance Gatekeeper",
         "Designed pre-deepfake. Modern voice clones (e.g. ElevenLabs) match the customer's voiceprint closely enough to pass — by design. No liveness layer.",
         "Defeated by clones."),
        ("Manual Agent Training",
         "Human escalation",
         "Agents are the last line of defense, but human detection of synthetic voices is empirically <b>24.5%</b>. Inconsistent across shifts and not scalable to 100M+ calls.",
         "75% miss rate."),
        ("Post-Call Forensics",
         "After-the-fact analytics",
         "By the time a forensic flag triggers, funds have moved and the customer has been notified. Forensics inform next quarter's policy — not this call.",
         "Too late."),
    ]
    cols = st.columns(3, gap="medium")
    for col, (name, vendors, body, headline) in zip(cols, gaps):
        with col:
            st.markdown(
                f"""
                <div style='background:#fff;border:1px solid {BORDER};
                border-top:4px solid {RED};border-radius:6px;
                padding:18px 20px;height:100%;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                  <div style='font-size:17px;font-weight:800;color:{NAVY_TOP};
                  font-family:Georgia,serif;'>{name}</div>
                  <div style='font-size:12px;color:{MUTED};font-style:italic;
                  margin-top:2px;'>{vendors}</div>
                  <div style='font-size:13.5px;color:{INK};margin-top:10px;
                  line-height:1.55;'>{body}</div>
                  <div style='font-size:13px;font-weight:800;color:{RED};
                  margin-top:11px;border-top:1px dashed {BORDER};
                  padding-top:9px;'>✗ {headline}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='text-align:center;font-size:14px;color:{NAVY};"
        f"font-style:italic;margin-top:14px;'>"
        f"<b>The unmet gap:</b> a real-time, layered, voice-aware decision "
        f"layer that catches what each existing tool misses.</div>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────
    # Section 3 — Our Approach (specific, dashboard-aligned)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{ACCENT};letter-spacing:2.2px;"
        f"text-transform:uppercase;font-weight:900;border-left:5px solid {ACCENT};"
        f"padding-left:12px;'>03 · Our Approach</div>"
        f"<h2 style='font-size:42px;font-weight:900;color:{NAVY_TOP};"
        f"font-family:Georgia,serif;margin:8px 0 18px 0;line-height:1.15;'>"
        f"Two ML Signals + Human-in-the-Loop</h2>",
        unsafe_allow_html=True,
    )

    pillars = [
        ("01", "Speaker Verification",
         "ECAPA-TDNN voiceprint comparison",
         "Cosine similarity between the caller's voiceprint and the enrolled "
         "customer baseline. Catches <i>who</i> the caller is regardless of "
         "whether the audio is real or synthetic — including AI clones.",
         ["SpeechBrain ECAPA-TDNN (VoxCeleb-trained)",
          "Industry-standard speaker verification",
          "Same-speaker similarity 0.50–0.95",
          "Different-speaker similarity -0.10–0.30"]),
        ("02", "Synthesis Detection",
         "Wav2Vec2 deepfake classifier + F0 prosody",
         "Wav2Vec2 binary classifier returns P(synthetic) directly, plus "
         "librosa F0 anomaly as a secondary signal. Catches <i>what kind</i> "
         "of audio it is regardless of who's claimed to be calling.",
         ["motheecreator/Deepfake-audio-detection",
          "94.5M-parameter Wav2Vec2 network",
          "Catches modern ElevenLabs at 99%+",
          "Librosa YIN pitch anomaly as backstop"]),
        ("03", "Human-in-the-Loop Pipeline",
         "LangGraph with interrupt_after",
         "Every call routes through a LangGraph pipeline that pauses at the "
         "Agent Handoff stage. The reviewer's decision (Approve / Step-Up / "
         "Block) is written to graph state and resumes execution — the click "
         "literally drives the next half of the pipeline.",
         ["interrupt_after + MemorySaver checkpointer",
          "Reviewer audit log per session",
          "Step-Up routes to OTP auth_challenge node",
          "Block routes to intelligence with case logged"]),
    ]
    cols = st.columns(3, gap="medium")
    for col, (num, title, sub, body, checks) in zip(cols, pillars):
        with col:
            checks_html = "".join(
                f"<div style='font-size:13px;color:{NAVY};margin-top:5px;"
                f"font-weight:600;'>✓ {c}</div>"
                for c in checks
            )
            st.markdown(
                f"""
                <div style='background:#fff;border:1px solid {BORDER};
                border-radius:6px;padding:18px 20px;height:100%;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                  <div style='display:inline-block;background:{NAVY_TOP};
                  color:#fff;border-radius:50%;width:32px;height:32px;
                  line-height:32px;text-align:center;font-size:13px;
                  font-weight:900;'>{num}</div>
                  <div style='font-size:18px;font-weight:800;color:{NAVY_TOP};
                  margin-top:10px;line-height:1.25;font-family:Georgia,serif;'>{title}</div>
                  <div style='font-size:12px;color:{MUTED};font-style:italic;
                  margin-top:3px;'>{sub}</div>
                  <div style='font-size:13.5px;color:{INK};margin-top:9px;
                  line-height:1.5;'>{body}</div>
                  <div style='border-top:1px solid {BORDER};margin:11px 0 4px 0;'></div>
                  {checks_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Empirical evidence callout from our testing
    st.markdown(
        f"""
        <div style='background:#eff6ff;border-left:4px solid {ACCENT};
        border-radius:4px;padding:14px 18px;margin-top:14px;
        font-size:14px;color:{INK};line-height:1.55;'>
          <b style='color:{NAVY};'>Why layered matters — measured on our own test set.</b>
          An ElevenLabs clone of the customer's voice scored
          <code>fake_prob = 0.99</code> on the synthesis classifier <i>and</i>
          <code>cosine = +0.06</code> on the voiceprint check (vs. +1.00 for the
          customer's real voice). Either signal alone catches it; together
          they make false-negatives genuinely difficult — even against the
          state-of-the-art consumer voice cloners that defeat single-vendor
          biometrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────
    # Section 4 — VoiceGuard Pipeline (LangGraph + HITL)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:14px;color:{ACCENT};letter-spacing:2.2px;"
        f"text-transform:uppercase;font-weight:900;border-left:5px solid {ACCENT};"
        f"padding-left:12px;'>04 · The Pipeline</div>"
        f"<h2 style='font-size:42px;font-weight:900;color:{NAVY_TOP};"
        f"font-family:Georgia,serif;margin:8px 0 6px 0;line-height:1.15;'>"
        f"What Happens When a Call Comes In</h2>"
        f"<div style='font-size:15px;color:{MUTED};font-style:italic;"
        f"margin-bottom:18px;'>"
        f"Six LangGraph stages. The reviewer is the decision authority — their click "
        f"resumes the graph and routes one of three ways."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Pre-human flow: stages 1–4 ─────────────────────────────────
    pre = [
        ("1", "Voice Cloning Detector",
         "ML inference",
         "<b>ECAPA-TDNN</b> speaker match · "
         "<b>Wav2Vec2</b> deepfake classifier · "
         "<b>librosa F0</b> prosody",
         "Outputs: <code>Speaker Match</code> · <code>Voice Risk</code>",
         NAVY),
        ("2", "IVR Entry",
         "Defense 1 · automated",
         "Combines voice signals into <code>entry_confidence</code>. "
         "Routes high-confidence calls fast-track.",
         "",
         NAVY),
        ("3", "IVR Navigation",
         "Defense 2 · conditional",
         "Runs only when entry_confidence ≤ 0.8. Checks for "
         "bot-like timing patterns.",
         "",
         NAVY),
        ("4", "Agent Handoff",
         "Defense 3 · prepares alert",
         "Computes <code>agent_confidence</code>, generates the "
         "alert message for the reviewer.",
         "<b style='color:#b45309;'>⏸ Pipeline pauses here · "
         "<code>interrupt_after</code></b>",
         AMBER),
    ]
    cols = st.columns(4, gap="small")
    for col, (num, title, sub, body, foot, accent) in zip(cols, pre):
        with col:
            foot_html = (
                f"<div style='font-size:12px;margin-top:9px;line-height:1.4;"
                f"border-top:1px dashed {BORDER};padding-top:8px;color:{INK};'>{foot}</div>"
                if foot else ""
            )
            st.markdown(
                f"""
                <div style='background:#fff;border:1px solid {BORDER};
                border-top:4px solid {accent};border-radius:6px;
                padding:14px 16px;height:100%;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
                  <div style='display:flex;align-items:center;gap:8px;'>
                    <div style='background:{accent};color:#fff;
                    border-radius:50%;width:26px;height:26px;
                    line-height:26px;text-align:center;font-size:12px;
                    font-weight:900;'>{num}</div>
                    <div style='font-size:14px;font-weight:800;color:{NAVY_TOP};
                    line-height:1.2;'>{title}</div>
                  </div>
                  <div style='font-size:11px;color:{MUTED};
                  letter-spacing:0.6px;text-transform:uppercase;
                  font-weight:700;margin-top:6px;'>{sub}</div>
                  <div style='font-size:13px;color:{INK};margin-top:8px;
                  line-height:1.5;'>{body}</div>
                  {foot_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Down arrow into the human review card ─────────────────────
    st.markdown(
        f"<div style='text-align:center;font-size:24px;color:{AMBER};"
        f"line-height:1;margin:8px 0 4px 0;'>↓</div>",
        unsafe_allow_html=True,
    )

    # ── Human reviewer decision card ──────────────────────────────
    st.markdown(
        f"""
        <div style='background:#fffbeb;border:2px solid {AMBER};
        border-radius:8px;padding:18px 22px;
        box-shadow:0 2px 8px rgba(180,83,9,0.15);'>
          <div style='display:flex;align-items:center;gap:14px;
          flex-wrap:wrap;'>
            <div style='background:{AMBER};color:#fff;
            font-size:18px;font-weight:900;padding:8px 14px;
            border-radius:6px;letter-spacing:0.5px;'>👤 HUMAN REVIEWER</div>
            <div style='font-size:14px;color:{INK};line-height:1.5;flex:1;
            min-width:280px;'>
              Sees the alert, signals, and caller context. Picks one of three
              actions — the click writes <code>human_decision</code> to graph
              state and resumes execution via <code>invoke(None, config)</code>.
            </div>
          </div>
          <div style='display:grid;grid-template-columns:repeat(3,1fr);
          gap:10px;margin-top:14px;'>
            <div style='background:#dcfce7;border:1.5px solid #4ade80;
            border-radius:5px;padding:10px 14px;'>
              <div style='font-size:12px;color:#14532d;font-weight:900;
              letter-spacing:0.7px;'>✓ APPROVE</div>
              <div style='font-size:12.5px;color:#14532d;margin-top:4px;
              line-height:1.45;'>case cleared, routes directly to Stage 6 (Intelligence)</div>
            </div>
            <div style='background:#fef3c7;border:1.5px solid #f59e0b;
            border-radius:5px;padding:10px 14px;'>
              <div style='font-size:12px;color:#78350f;font-weight:900;
              letter-spacing:0.7px;'>⚠ STEP-UP AUTH</div>
              <div style='font-size:12.5px;color:#78350f;margin-top:4px;
              line-height:1.45;'>routes through Stage 5 (Auth Challenge) before clearing</div>
            </div>
            <div style='background:#fee2e2;border:1.5px solid #ef4444;
            border-radius:5px;padding:10px 14px;'>
              <div style='font-size:12px;color:#7f1d1d;font-weight:900;
              letter-spacing:0.7px;'>✗ BLOCK & ESCALATE</div>
              <div style='font-size:12.5px;color:#7f1d1d;margin-top:4px;
              line-height:1.45;'>transaction blocked, case logged at Stage 6</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='text-align:center;font-size:24px;color:{NAVY};"
        f"line-height:1;margin:8px 0 4px 0;'>↓</div>",
        unsafe_allow_html=True,
    )

    # ── Post-human: stages 5 and 6 ────────────────────────────────
    post = [
        ("5", "Auth Challenge",
         "Defense 4 · conditional",
         "Runs only on <b>Step-Up</b>. Sends OTP to the customer's "
         "registered device. Attacker can't complete it.",
         NAVY),
        ("6", "Intelligence",
         "Leadership log · always runs",
         "Loss avoidance + attack-vector classification + executive "
         "summary written to <code>intelligence_log</code>.",
         NAVY),
    ]
    cols = st.columns(2, gap="medium")
    for col, (num, title, sub, body, accent) in zip(cols, post):
        with col:
            st.markdown(
                f"""
                <div style='background:#fff;border:1px solid {BORDER};
                border-top:4px solid {accent};border-radius:6px;
                padding:14px 16px;height:100%;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
                  <div style='display:flex;align-items:center;gap:8px;'>
                    <div style='background:{accent};color:#fff;
                    border-radius:50%;width:26px;height:26px;
                    line-height:26px;text-align:center;font-size:12px;
                    font-weight:900;'>{num}</div>
                    <div style='font-size:14px;font-weight:800;color:{NAVY_TOP};
                    line-height:1.2;'>{title}</div>
                  </div>
                  <div style='font-size:11px;color:{MUTED};
                  letter-spacing:0.6px;text-transform:uppercase;
                  font-weight:700;margin-top:6px;'>{sub}</div>
                  <div style='font-size:13px;color:{INK};margin-top:8px;
                  line-height:1.5;'>{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div style='background:#eff6ff;border-left:4px solid {ACCENT};
        border-radius:4px;padding:12px 18px;margin-top:14px;
        font-size:13.5px;color:{INK};line-height:1.55;'>
          <b style='color:{NAVY};'>Why this matters:</b> today's bank stack has
          no real-time AI-voice detection at <i>any</i> of these stages. VoiceGuard
          adds the ML signals at Stage 1 and the explicit human checkpoint at
          Stage 4 — <b>before</b> any account action is processed. The reviewer's
          decision drives the rest of the graph; the click isn't cosmetic.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────
    # Transition CTA
    # ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{NAVY},{ACCENT});
        color:#fff;border-radius:6px;padding:18px 24px;text-align:center;'>
          <div style='font-size:18px;font-weight:800;font-family:Georgia,serif;'>
            See it run on real audio →
            <span style='background:rgba(255,255,255,0.18);padding:3px 10px;
            border-radius:4px;margin-left:8px;font-size:15px;'>Live Dashboard</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Warming up ML models — this takes ~20 seconds the first time, then every call is instant…")
def _warm_up_pipeline() -> bool:
    """Pre-load both ML models and pre-compute scores for every scenario.

    Without this, the first Connect Call eats a 15–25 second cold start
    while Wav2Vec2 + ECAPA load and the first audio file is decoded
    through librosa's audioread fallback (m4a is slow on Streamlit Cloud
    because the bundled libsndfile lacks AAC). With this, the first
    visitor pays the cold-start cost once at app load (with a clear
    spinner), and every Connect Call after that is sub-second.

    Cached via @st.cache_resource so it runs exactly once per app
    container lifetime.
    """
    from detectors import detect, speaker_similarity

    voiceprint = REGISTERED_VOICE_FILE
    # Warm the speaker-verification model first (it's the lighter of the two)
    if voiceprint.exists():
        try:
            speaker_similarity(str(voiceprint), str(voiceprint))
        except Exception:
            pass

    # Warm the deepfake classifier + librosa F0
    if voiceprint.exists():
        try:
            detect(str(voiceprint))
        except Exception:
            pass

    # Pre-compute scores for every scenario so the first click is instant
    for sc in SCENARIOS.values():
        audio_filename = sc.get("audio")
        if not audio_filename:
            continue
        audio_path = AUDIO_DIR / audio_filename
        if not audio_path.exists():
            continue
        try:
            detect(str(audio_path))
            if voiceprint.exists():
                speaker_similarity(str(voiceprint), str(audio_path))
        except Exception:
            pass

    return True


def main() -> None:
    st.set_page_config(
        page_title="VoiceGuard AI · Fraud Operations Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session()
    inject_css()
    render_header()

    # Pre-warm models + cache scenario scores. Spinner shows once per
    # container; subsequent visits hit the cached resource and skip it.
    _warm_up_pipeline()

    tab_business, tab_dashboard, tab_audit = st.tabs([
        "  Business Case  ",
        "  Live Dashboard  ",
        "  Stage-by-Stage Audit  ",
    ])
    with tab_business:
        render_business_case()
    with tab_dashboard:
        sim_slots = render_live_simulation()
    with tab_audit:
        audit_slots = render_stage_audit()

    if sim_slots["place_call"]:
        # Auto-switch to "Live Dashboard" tab (index 1) on button press.
        # Streamlit reruns from the top on every interaction, which resets the
        # active tab to the first one (Business Case). This JS click fixes it.
        _components.html(
            """<script>
            setTimeout(function() {
                var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > 1) tabs[1].click();
            }, 50);
            </script>""",
            height=0,
        )
        run_call_animation(sim_slots, audit_slots)


if __name__ == "__main__":
    main()
