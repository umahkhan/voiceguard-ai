"""VoiceGuard AI — Streamlit dashboard for voice fraud detection.

JPMorgan-aligned design: deep navy + white card system, two tabs that fit
in a 900px viewport without scrolling. All scenario data is hardcoded so
the app runs without any model calls.
"""

from __future__ import annotations

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
# Scenarios — all numeric data hardcoded. Audio paths point at files
# already rendered into ./audio (ElevenLabs MP3s for clean / borderline,
# macOS `say` WAV for the synthetic-voice scenarios).
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "Clean Caller": {
        "spectral": 0.12, "prosody": 0.18, "behavior": 0.14, "conf": 0.15,
        "audio": "clean.mp3",
        "loss_avoidance": 0,
        "narrative": (
            "Caller's voice biometrics, prosody, and IVR navigation pattern "
            "all sit well within the normal baseline. Stress markers and "
            "breath sounds match a human speaker."
        ),
        "ok_actions": [
            "Handle balance, transaction, and routine questions normally.",
            "Process customer requested changes using the standard call script.",
            "Close the call with the usual wrap up and CRM note.",
        ],
    },
    "Borderline Suspicious": {
        "spectral": 0.55, "prosody": 0.62, "behavior": 0.49, "conf": 0.59,
        "audio": "borderline.mp3",
        "loss_avoidance": 18000,
        "narrative": (
            "Mixed signals. Some prosody markers look rehearsed but voice "
            "biometrics are within range. IVR navigation is hesitant rather "
            "than bot-like. Treat as elevated risk pending verification."
        ),
        "ok_actions": [
            "Ask the caller to read back the one-time code. Do not read it to them.",
            "Talk about public info such as branch hours or general products.",
            "Continue with the original request only after you confirm identity.",
        ],
    },
    "Synthetic Bot — High Confidence": {
        "spectral": 0.94, "prosody": 0.91, "behavior": 0.89, "conf": 0.94,
        "audio": "ai_voice.wav",
        "loss_avoidance": 87000,
        "narrative": (
            "All three indicators flag clear synthesis. Spectral, prosody, "
            "and navigation patterns match high-confidence bot signatures "
            "from recent fraud rings. Recommend immediate hand off to Fraud "
            "Recovery."
        ),
        "ok_actions": [
            "Put the caller on hold and transfer to Fraud Recovery (extension 4400).",
            "Tell the caller only that another team is reviewing the call.",
            "Note the case ID in the customer record before ending the call.",
        ],
    },
}

CALL_DURATION_S = 20.0


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


def current_state() -> dict:
    """Compose the live scenario view from session state."""
    name = st.session_state.get("preset_choice", "Clean Caller")
    sc = SCENARIOS[name]
    voice_risk = max(sc["spectral"], sc["prosody"])
    return {
        "name":           name,
        "spectral":       sc["spectral"],
        "prosody":        sc["prosody"],
        "behavior":       sc["behavior"],
        "voice_risk":     voice_risk,
        "agent_susp":     sc["conf"],
        "conf":           sc["conf"],
        "verdict":        _verdict(sc["conf"]),
        "caller_id":      st.session_state.get("caller_id", "+1 212-555-0199"),
        "ok_actions":     sc["ok_actions"],
        "narrative":      sc["narrative"],
        "audio":          sc["audio"],
        "loss_avoidance": sc["loss_avoidance"],
    }


def _reset_call() -> None:
    """Reset to idle state — clears verdict, timer, and path progress."""
    st.session_state["call_state"] = "idle"
    st.session_state["timer_value"] = 0.0
    st.session_state["call_progress"] = 0.0


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
        '<div class="vg-h-title">VoiceGuard AI</div>'
        '<div class="vg-h-sub">JPMorgan Chase Contact Center · Voice Fraud Defense</div>'
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


def render_metric(label: str, score: float, description: str) -> None:
    level = _level(score)
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


def render_advisory_left(state: dict) -> None:
    klass = {"PASS": "pass-acc", "FLAG": "flag-acc", "BLOCK": "block-acc"}[state["verdict"]]
    icon = {"PASS": "✓", "FLAG": "!", "BLOCK": "✓"}[state["verdict"]]
    ok_items = "".join(
        f'<div class="vg-ok-item">'
        f'<span class="vg-ok-icon">{icon}</span>'
        f'<span>{a}</span>'
        f'</div>'
        for a in state["ok_actions"]
    )
    html = (
        f'<div class="vg-advisory {klass}">'
        '<div class="vg-a-label">Recommended Action</div>'
        f'<div class="vg-a-action">{_action_for(state["verdict"])}</div>'
        '<div class="vg-a-label" style="margin-top:9px;">OK to Do</div>'
        f'{ok_items}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_advisory_right(state: dict) -> None:
    ivr = "Suspicious" if state["behavior"] >= 0.50 else "Normal"
    otp = "Triggered" if state["agent_susp"] > 0.60 else "Not triggered"
    html = (
        '<div class="vg-advisory">'
        '<div class="vg-a-label">System Note</div>'
        f'<div class="vg-a-narrative">{state["narrative"]}</div>'
        '<div class="vg-a-pills">'
        f'<span class="vg-pill"><span class="vg-pill-k">IVR Path:</span> {ivr}</span>'
        f'<span class="vg-pill"><span class="vg-pill-k">OTP:</span> {otp}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1 — Live Simulation
# ---------------------------------------------------------------------------
def _render_complete_right(slot, state: dict) -> None:
    """Fill the right-column slot with verdict + risk + advisory."""
    with slot.container():
        render_verdict_bar(state["verdict"], state["conf"])
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric(
                "Voice Risk", state["voice_risk"],
                "How synthetic or unusual the caller's voice sounded.",
            )
        with c2:
            render_metric(
                "Agent Suspicion", state["agent_susp"],
                "How suspicious the call looked to the agent layer.",
            )
        with c3:
            render_metric(
                "Behavior Risk", state["behavior"],
                "How unusual the caller's path through the menu was.",
            )
        a1, a2 = st.columns(2)
        with a1:
            render_advisory_left(state)
        with a2:
            render_advisory_right(state)


def render_live_simulation() -> dict:
    left, right = st.columns([3, 7], gap="medium")

    with left:
        incoming_slot = st.empty()
        st.markdown("### Scenario Preset")
        st.selectbox(
            "Scenario Preset",
            list(SCENARIOS.keys()),
            key="preset_choice",
            on_change=_reset_call,
            label_visibility="collapsed",
        )
        st.markdown("### Caller Phone Number")
        st.text_input("Caller ID", key="caller_id", disabled=True, label_visibility="collapsed")
        st.markdown("### Start Call")
        place_call = st.button("▶  Place Incoming Call", type="primary")
        audio_slot = st.empty()

    with right:
        result_slot = st.empty()

    # Bottom row — full width, animates as the call progresses.
    meta_slot = st.empty()
    path_slot = st.empty()

    state = current_state()
    call_state = st.session_state.get("call_state", "idle")
    if place_call:
        # Switch to running for this script execution; the shared animation
        # driver will mark call_state "complete" once the loop finishes.
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


def run_call_animation(sim: dict, audit: dict) -> None:
    """Drive the running -> complete animation across both tabs."""
    state = sim["state"]
    audio_path = AUDIO_DIR / state["audio"]
    if audio_path.exists():
        fmt = "audio/mp3" if audio_path.suffix == ".mp3" else "audio/wav"
        sim["audio_slot"].audio(str(audio_path), format=fmt, autoplay=True)

    st.session_state["session_calls"].append({
        "name":      state["name"],
        "caller_id": state["caller_id"],
        "verdict":   state["verdict"],
        "conf":      state["conf"],
        "ts":        time.time(),
    })

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
        state["agent_susp"] > 0.60,
        True,
    ]


def _stage_detail(idx: int, state: dict) -> tuple[str, str]:
    """Return (description, score_text) for stage idx (1..6)."""
    if idx == 1:
        return (
            "Captured voice biometrics from the call audio.",
            f"spectral {state['spectral']:0.2f} · prosody {state['prosody']:0.2f}",
        )
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
        if state["agent_susp"] > 0.60:
            return (
                f"Agent suspicion {state['agent_susp']:0.2f} > 0.60, OTP challenge required.",
                f"{state['agent_susp']:0.2f} > 0.60",
            )
        return (
            f"Agent suspicion {state['agent_susp']:0.2f} ≤ 0.60, no extra check needed.",
            f"{state['agent_susp']:0.2f} ≤ 0.60",
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


def render_call_risk_audit() -> dict:
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


# ---------------------------------------------------------------------------
# Init + main
# ---------------------------------------------------------------------------
def init_session() -> None:
    ss = st.session_state
    ss.setdefault("preset_choice",  "Clean Caller")
    ss.setdefault("caller_id",      "+1 212-555-0199")
    ss.setdefault("timer_value",    0.0)
    ss.setdefault("call_state",     "idle")          # idle | running | complete
    ss.setdefault("call_progress",  0.0)
    ss.setdefault("session_calls", [])
    ss.setdefault("session_id",     int(time.time()) % 100000)


def main() -> None:
    st.set_page_config(
        page_title="VoiceGuard AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session()
    inject_css()
    render_header()

    tab1, tab2 = st.tabs(["  Live Simulation  ", "  Call Risk Audit  "])
    with tab1:
        sim_slots = render_live_simulation()
    with tab2:
        audit_slots = render_call_risk_audit()

    if sim_slots["place_call"]:
        run_call_animation(sim_slots, audit_slots)


if __name__ == "__main__":
    main()
