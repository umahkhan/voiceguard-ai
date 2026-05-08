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
# Scenarios — fraud-attack typology. Each entry includes the caller context
# a human reviewer would see in production (claimed identity, transaction in
# flight, prior call history). Scripted spectral/prosody/conf values are
# overridden by real model output when Live Mode is on.
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "Voice Clone Attempt — Female Caller": {
        "spectral": 0.55, "prosody": 0.40, "behavior": 0.62, "conf": 0.62,
        "audio": "clean.mp3",
        "caller_id":      "+1 415-555-0144",
        "claimed_name":   "Sarah Chen",
        "account_suffix": "8421",
        "txn_type":       "Wire transfer",
        "txn_amount":     27500,
        "txn_destination":"new external beneficiary (first time)",
        "prior_calls_30d": 0,
        "ivr_path":       "Direct-to-agent (skipped self-service)",
        "loss_avoidance": 27500,
        "narrative": (
            "Caller claims to be account holder. Voice sounds natural but "
            "does NOT match the enrolled voiceprint. Most likely a "
            "synthetic clone of a different speaker, or a real impersonator. "
            "High-value wire request to a new beneficiary raises the "
            "transaction risk."
        ),
        "expected": "FLAG",
    },
    "Voice Clone Attempt — Male Caller": {
        "spectral": 0.55, "prosody": 0.40, "behavior": 0.58, "conf": 0.62,
        "audio": "borderline.mp3",
        "caller_id":      "+1 646-555-0177",
        "claimed_name":   "Marcus Reilly",
        "account_suffix": "3309",
        "txn_type":       "Add payee + transfer",
        "txn_amount":     12800,
        "txn_destination":"newly-added Zelle recipient",
        "prior_calls_30d": 1,
        "ivr_path":       "Selected 'lost card' before pivoting to transfers",
        "loss_avoidance": 12800,
        "narrative": (
            "Caller's voice does not match the enrolled voiceprint for "
            "this account. IVR pattern is unusual — pivoted from card "
            "support to a transfer request mid-call, a known social-"
            "engineering pretext."
        ),
        "expected": "FLAG",
    },
    "Robocall Bot (Crude Synthesizer)": {
        "spectral": 0.94, "prosody": 0.91, "behavior": 0.89, "conf": 0.94,
        "audio": "ai_voice.wav",
        "caller_id":      "+1 800-555-0123",
        "claimed_name":   "Daniel Park",
        "account_suffix": "5562",
        "txn_type":       "Account verification request",
        "txn_amount":     0,
        "txn_destination":"—",
        "prior_calls_30d": 4,
        "ivr_path":       "Sub-second key presses, no human pauses",
        "loss_avoidance": 87000,
        "narrative": (
            "Audio is unmistakably synthesized. Robotic prosody, no breath "
            "sounds, machine-cadenced IVR navigation. Pattern matches "
            "credential-harvesting bot rings observed across the network "
            "in the last 30 days."
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


@st.cache_data(show_spinner=False)
def _live_scores_for(audio_path: str) -> dict:
    """Run the real deepfake detector on an audio path. Cached across reruns."""
    from detectors import detect
    return detect(audio_path, fp_tuned=True)


@st.cache_data(show_spinner=False)
def _speaker_similarity(reference_path: str, test_path: str) -> float:
    """Cosine similarity between two voice clips, [-1, 1]. Cached."""
    from detectors import speaker_similarity as _sim
    return float(_sim(reference_path, test_path))


REGISTERED_SCENARIO = "Authenticated Customer (Your Voice)"
LIVE_MIC_SCENARIO   = "Live Microphone Input"


def _scenario_names() -> list[str]:
    """Available scenarios. The registered-customer entry appears once a
    reference voice has been uploaded; the live-mic entry appears whenever
    Live Mode is on."""
    base = list(SCENARIOS.keys())
    extras: list[str] = []
    if st.session_state.get("registered_voice_path"):
        extras.append(REGISTERED_SCENARIO)
    if st.session_state.get("live_mode"):
        extras.append(LIVE_MIC_SCENARIO)
    return extras + base


def _scenario_data(name: str) -> dict:
    """Resolve a scenario name to its data dict. The registered-customer
    and live-mic scenarios are synthesized at runtime."""
    if name == REGISTERED_SCENARIO:
        return {
            "spectral": 0.05, "prosody": 0.05, "behavior": 0.10, "conf": 0.10,
            "audio": None,  # path comes from registered_voice_path
            "caller_id":      "+1 212-555-0199",
            "claimed_name":   "Registered Customer",
            "account_suffix": "0042",
            "txn_type":       "Balance inquiry · routine transfer",
            "txn_amount":     1850,
            "txn_destination":"existing internal account",
            "prior_calls_30d": 6,
            "ivr_path":       "Self-service first, then agent (typical pattern)",
            "loss_avoidance": 0,
            "narrative": (
                "Voiceprint matches the enrolled customer baseline within "
                "expected tolerance. No synthesis artifacts. Caller's IVR "
                "navigation matches their historical pattern. Authenticated."
            ),
            "expected": "PASS",
        }
    if name == LIVE_MIC_SCENARIO:
        return {
            "spectral": 0.30, "prosody": 0.30, "behavior": 0.30, "conf": 0.30,
            "audio": None,  # path comes from live_mic_audio_path
            "caller_id":      "Microphone (local capture)",
            "claimed_name":   "Live Caller",
            "account_suffix": "—",
            "txn_type":       "Live recording — verdict from real-time analysis",
            "txn_amount":     0,
            "txn_destination":"—",
            "prior_calls_30d": 0,
            "ivr_path":       "Direct microphone input",
            "loss_avoidance": 0,
            "narrative": (
                "Audio captured live from the operator's microphone. Speaker "
                "match and synthesis-detection signals are computed against "
                "the registered customer voiceprint and shown below."
            ),
            "expected": "—",
        }
    return SCENARIOS[name]


def _scenario_audio_path(name: str, sc: dict) -> str | None:
    if name == REGISTERED_SCENARIO:
        return st.session_state.get("registered_voice_path") or None
    if name == LIVE_MIC_SCENARIO:
        return st.session_state.get("live_mic_audio_path") or None
    return str(AUDIO_DIR / sc["audio"]) if sc.get("audio") else None


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
    valid_names = (*SCENARIOS.keys(), REGISTERED_SCENARIO, LIVE_MIC_SCENARIO)
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
        model_short = str(state.get("model_used", "")).split("/")[-1]
        if model_short:
            audio_sub = f"{audio_sub} · {model_short}"
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

        # HITL action panel — appears when verdict is FLAG/BLOCK
        f".vg-hitl{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:14px 16px;margin-top:8px;box-shadow:0 1px 4px rgba(0,0,0,0.08);}}"
        ".vg-hitl.flag{border-left:4px solid #f59e0b;}"
        ".vg-hitl.block{border-left:4px solid #ef4444;}"
        ".vg-hitl.pass{border-left:4px solid #22c55e;}"
        ".vg-hitl-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}"
        f".vg-hitl-title{{font-size:13px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:{NAVY};}}"
        ".vg-hitl-rec{font-size:12px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;padding:3px 10px;border-radius:3px;}"
        ".vg-hitl-rec.flag{background:#fef3c7;color:#78350f;border:1px solid #f59e0b;}"
        ".vg-hitl-rec.block{background:#fee2e2;color:#7f1d1d;border:1px solid #ef4444;}"
        ".vg-hitl-rec.pass{background:#dcfce7;color:#14532d;border:1px solid #4ade80;}"
        # Plain-language reasoning row
        f".vg-hitl-reason{{font-size:14px;color:{INK};line-height:1.45;background:{CANVAS};border:1px solid {BORDER};border-radius:4px;padding:9px 12px;margin-bottom:8px;}}"

        # Caller-context tile — key/value grid
        f".vg-ctx{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:12px 14px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}}"
        f".vg-ctx-title{{font-size:11px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;color:{MUTED};margin-bottom:6px;}}"
        ".vg-ctx-row{display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;border-bottom:1px solid #eef0f3;font-size:14px;}"
        ".vg-ctx-row:last-child{border-bottom:none;}"
        f".vg-ctx-k{{color:{MUTED};font-weight:500;}}"
        f".vg-ctx-v{{color:{INK};font-weight:700;font-variant-numeric:tabular-nums;}}"
        ".vg-ctx-v.high{color:#c81e1e;}"
        ".vg-ctx-v.med{color:#b45309;}"

        # Reviewer audit log
        f".vg-log{{background:{PAPER};border:1px solid {BORDER};border-radius:4px;padding:10px 14px;max-height:140px;overflow-y:auto;}}"
        f".vg-log-title{{font-size:11px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;color:{MUTED};margin-bottom:6px;}}"
        ".vg-log-row{display:flex;gap:8px;align-items:center;font-size:13px;padding:4px 0;border-bottom:1px solid #eef0f3;}"
        ".vg-log-row:last-child{border-bottom:none;}"
        f".vg-log-ts{{color:{MUTED};font-variant-numeric:tabular-nums;font-size:12px;}}"
        ".vg-log-decision{font-weight:700;padding:1px 7px;border-radius:3px;font-size:11px;letter-spacing:0.5px;}"
        ".vg-log-decision.approve{background:#dcfce7;color:#14532d;}"
        ".vg-log-decision.stepup{background:#fef3c7;color:#78350f;}"
        ".vg-log-decision.block{background:#fee2e2;color:#7f1d1d;}"
        f".vg-log-name{{color:{INK};font-weight:600;}}"
        f".vg-log-empty{{color:{MUTED};font-size:13px;font-style:italic;}}"

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
        '<div class="vg-h-sub">Real-time deepfake &amp; voice-clone detection · Human-in-the-loop reviewer console</div>'
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
    match) — risk badging flips to LOW for high values."""
    risk_score = (1.0 - score) if inverted else score
    level = _level(risk_score)
    fg, _bg = _level_palette(level)
    pct = max(2, score * 100)
    html = (
        '<div class="vg-metric">'
        f'<div class="vg-m-label">{label}</div>'
        '<div style="display:flex;align-items:baseline;gap:8px;margin-top:3px;">'
        f'<div class="vg-m-value">{score:0.2f}</div>'
        f'<span class="vg-badge {level.lower()}">{level}</span>'
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


def render_live_simulation() -> dict:
    left, right = st.columns([3, 7], gap="medium")

    with left:
        incoming_slot = st.empty()
        st.markdown("### Scenario")
        st.selectbox(
            "Scenario",
            _scenario_names(),
            key="preset_choice",
            on_change=_reset_call,
            label_visibility="collapsed",
        )
        st.markdown("### Detection Mode")
        st.toggle(
            "Live Model (Wav2Vec2 + speaker verification)",
            key="live_mode",
            on_change=_reset_call,
            help=(
                "Off: scripted scenario scores (instant, used for the "
                "walkthrough). On: runs a real Wav2Vec2 deepfake "
                "classifier + F0 prosody analysis on the scenario "
                "audio. If a registered voice is uploaded, also "
                "computes speaker similarity. First run takes "
                "~10–20s to load the models."
            ),
        )
        if st.session_state.get("live_mode"):
            st.markdown("### Registered Customer Voice")
            method = st.radio(
                "Enrollment method",
                ["Record", "Upload file"],
                key="enrollment_method",
                horizontal=True,
                label_visibility="collapsed",
                help=(
                    "Recording with the browser mic gives the best "
                    "live-call match because both clips use the same "
                    "device and codec. Upload only if you want to "
                    "enroll from an existing recording — but expect "
                    "lower same-speaker scores when conditions differ."
                ),
            )
            if method == "Record":
                ref_audio = st.audio_input(
                    "Record customer voice",
                    key="registered_voice_recorder",
                    label_visibility="collapsed",
                    help=(
                        "Speak continuously for 5–10 seconds. The "
                        "browser captures audio with the same mic and "
                        "codec the live-call scenario uses, so "
                        "same-speaker similarity will be high."
                    ),
                )
                if ref_audio is not None:
                    content = ref_audio.getvalue()
                    h = hashlib.md5(content).hexdigest()[:12]
                    temp_path = Path(tempfile.gettempdir()) / f"voiceguard_ref_{h}.wav"
                    if not temp_path.exists():
                        temp_path.write_bytes(content)
                    if st.session_state.get("registered_voice_path") != str(temp_path):
                        st.session_state["registered_voice_path"] = str(temp_path)
                        _reset_call()
                    st.caption(f"Enrolled (recorded) · {len(content) / 1024:0.0f} KB")
                elif st.session_state.get("registered_voice_path"):
                    st.caption("Enrolled voiceprint loaded.")
            else:
                uploaded = st.file_uploader(
                    "Registered Customer Voice",
                    type=["wav", "mp3", "m4a", "flac"],
                    key="registered_voice_uploader",
                    label_visibility="collapsed",
                    help=(
                        "Upload a 5–10 second clip. Note: if the upload "
                        "uses a different mic/codec than the live call "
                        "(e.g. Voice Memos m4a vs browser mic), "
                        "same-speaker similarity may drop ~0.20 below "
                        "what you'd see with matched conditions."
                    ),
                )
                if uploaded is not None:
                    content = uploaded.read()
                    h = hashlib.md5(content).hexdigest()[:12]
                    suffix = Path(uploaded.name).suffix.lower() or ".wav"
                    temp_path = Path(tempfile.gettempdir()) / f"voiceguard_ref_{h}{suffix}"
                    if not temp_path.exists():
                        temp_path.write_bytes(content)
                    if st.session_state.get("registered_voice_path") != str(temp_path):
                        st.session_state["registered_voice_path"] = str(temp_path)
                        _reset_call()
                    st.caption(
                        f"Enrolled (uploaded) · {Path(uploaded.name).name} · "
                        f"{len(content) / 1024:0.0f} KB"
                    )
                elif st.session_state.get("registered_voice_path"):
                    st.caption("Enrolled voiceprint loaded.")
        if st.session_state.get("live_mode_error"):
            st.caption(f"⚠ {st.session_state['live_mode_error']}")

        # Live mic capture — only meaningful when the live-mic scenario is selected
        if st.session_state.get("preset_choice") == LIVE_MIC_SCENARIO:
            st.markdown("### Live Microphone")
            mic_audio = st.audio_input(
                "Record caller audio",
                key="live_mic_recorder",
                label_visibility="collapsed",
                help=(
                    "Click to record live audio. The dashboard will run "
                    "speaker verification + deepfake detection on whatever "
                    "you record, against the enrolled voice."
                ),
            )
            if mic_audio is not None:
                content = mic_audio.getvalue()
                h = hashlib.md5(content).hexdigest()[:12]
                temp_path = Path(tempfile.gettempdir()) / f"voiceguard_mic_{h}.wav"
                if not temp_path.exists():
                    temp_path.write_bytes(content)
                if st.session_state.get("live_mic_audio_path") != str(temp_path):
                    st.session_state["live_mic_audio_path"] = str(temp_path)
                    _reset_call()
                st.caption(f"Recording captured · {len(content) / 1024:0.0f} KB")
            elif st.session_state.get("live_mic_audio_path"):
                # Streamlit clears the widget when the user re-records.
                # Keep the previous capture available until a new one comes in.
                st.caption("Previous recording loaded — record again to refresh.")

        st.markdown("### Connect")
        connect_disabled = (
            st.session_state.get("preset_choice") == LIVE_MIC_SCENARIO
            and not st.session_state.get("live_mic_audio_path")
        )
        place_call = st.button(
            "▶  Connect Incoming Call",
            type="primary",
            disabled=connect_disabled,
            help="Record audio first" if connect_disabled else None,
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

    # Stage-by-stage audit pinned beneath the dashboard, collapsible.
    with st.expander("Stage-by-stage audit (technical detail)", expanded=False):
        stage_slot = st.empty()
        stage_slot.markdown(
            stage_detail_html(state, call_state, progress),
            unsafe_allow_html=True,
        )

    return {
        "place_call":    place_call,
        "state":         state,
        "incoming_slot": incoming_slot,
        "audio_slot":    audio_slot,
        "result_slot":   result_slot,
        "meta_slot":     meta_slot,
        "path_slot":     path_slot,
        "stage_slot":    stage_slot,
    }


def run_call_animation(sim: dict) -> None:
    """Run a short, real-time-feel analysis animation. Audio autoplays in
    parallel; signals populate after a brief 'analyzing' window so the
    reviewer experiences the flow as a real call coming in."""
    state = sim["state"]
    audio_path = Path(state["audio_path"]) if state.get("audio_path") else None
    if audio_path and audio_path.exists():
        fmt = "audio/mp3" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
        sim["audio_slot"].audio(str(audio_path), format=fmt, autoplay=True)

    st.session_state["session_calls"].append({
        "name":      state["name"],
        "caller_id": state["caller_id"],
        "verdict":   state["verdict"],
        "conf":      state["conf"],
        "ts":        time.time(),
    })

    # Reveal milestones: which signals are visible at each tick during the
    # animation. Mimics streaming inference — speaker match settles fastest,
    # synthesis detection a moment later, behavior signals last.
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
        sim["stage_slot"].markdown(
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
    sim["stage_slot"].markdown(
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
    ss.setdefault("live_mode",     False)
    ss.setdefault("live_mode_error", "")
    ss.setdefault("registered_voice_path", "")
    ss.setdefault("live_mic_audio_path",   "")
    ss.setdefault("reviewer_log",  [])
    ss.setdefault("call_seq",      0)


def main() -> None:
    st.set_page_config(
        page_title="VoiceGuard AI · Fraud Operations Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session()
    inject_css()
    render_header()

    sim_slots = render_live_simulation()

    if sim_slots["place_call"]:
        run_call_animation(sim_slots)


if __name__ == "__main__":
    main()
