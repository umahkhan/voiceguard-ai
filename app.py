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
# Design tokens
# ---------------------------------------------------------------------------
NAVY        = "#0a3d7a"
NAVY_DEEP   = "#06285a"
ACCENT      = "#378add"
INK         = "#0f172a"
PAPER       = "#ffffff"
CANVAS      = "#ffffff"
BORDER      = "#e0e0e0"
MUTED       = "#64748b"

PASS_BG,  PASS_TEXT  = "#14532d", "#4ade80"
FLAG_BG,  FLAG_TEXT  = "#451a03", "#f59e0b"
BLOCK_BG, BLOCK_TEXT = "#450a0a", "#f87171"

LOW_FG,  LOW_BG  = "#16a34a", "#dcfce7"
MED_FG,  MED_BG  = "#d97706", "#fef3c7"
HIGH_FG, HIGH_BG = "#dc2626", "#fee2e2"


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
        f'<div class="vg-meta-v">#VG-{session_id:05d}</div>'
        f'<div class="vg-meta-s">{state["caller_id"]}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">Final Verdict</div>'
        f'<div class="vg-meta-v">{verdict}</div>'
        f'<div class="vg-meta-s">{verdict_sub}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">AI Confidence</div>'
        f'<div class="vg-meta-v">{conf_v}</div>'
        f'<div class="vg-meta-s">{conf_sub}</div>'
        '</div>'
        '<div class="vg-meta">'
        '<div class="vg-meta-l">Audio + Scores</div>'
        f'<div class="vg-meta-v">{audio_v}</div>'
        f'<div class="vg-meta-s">{audio_sub}</div>'
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
        # Page chrome
        ".block-container{padding-top:0.45rem!important;padding-bottom:0.45rem!important;"
        "padding-left:1.4rem!important;padding-right:1.4rem!important;max-width:100%;}"
        "header[data-testid='stHeader']{display:none;height:0;}"
        "footer{visibility:hidden;}"
        "#MainMenu{visibility:hidden;}"
        "[data-testid='stToolbar']{display:none;}"
        f"html,body,[class*='css']{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:16px;color:{INK};}}"
        f".stApp{{background:{CANVAS};}}"

        # Header banner
        f".vg-header{{background:linear-gradient(135deg,{NAVY_DEEP} 0%,{NAVY} 60%,{ACCENT} 130%);"
        "padding:10px 22px;border-radius:12px;color:#fff;margin-bottom:7px;"
        "display:flex;align-items:center;justify-content:space-between;"
        "box-shadow:0 4px 14px rgba(10,61,122,0.18);}"
        ".vg-header .vg-h-title{font-size:26px;font-weight:800;letter-spacing:-0.2px;}"
        ".vg-header .vg-h-sub{font-size:15px;opacity:0.88;font-weight:500;margin-top:1px;}"
        ".vg-header .vg-h-meta{font-size:14px;opacity:0.85;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;}"

        # Tabs
        f".stTabs [data-baseweb='tab-list']{{gap:4px;border-bottom:1px solid {BORDER};}}"
        f".stTabs [data-baseweb='tab']{{background:transparent;color:{MUTED};font-weight:700;"
        "font-size:16px;padding:5px 14px;border-radius:8px 8px 0 0;letter-spacing:0.4px;}"
        f".stTabs [aria-selected='true']{{background:{NAVY}!important;color:#fff!important;}}"

        # Generic card
        f".vg-card{{background:{PAPER};border:0.5px solid {BORDER};border-radius:12px;"
        "padding:9px 14px;margin-bottom:7px;}"

        # Incoming call card
        f".vg-incoming{{background:linear-gradient(135deg,{NAVY_DEEP} 0%,{NAVY} 100%);"
        "color:#fff;border-radius:12px;padding:11px 16px;position:relative;"
        "margin-bottom:7px;box-shadow:0 4px 14px rgba(10,61,122,0.2);}"
        ".vg-incoming .vg-i-label{font-size:13px;letter-spacing:1.6px;text-transform:uppercase;"
        "color:#93c5fd;font-weight:800;}"
        ".vg-incoming .vg-i-num{font-size:27px;font-weight:800;letter-spacing:-0.3px;"
        "margin-top:3px;font-variant-numeric:tabular-nums;}"
        ".vg-incoming .vg-i-sub{font-size:15px;color:#cbd5e1;margin-top:1px;}"
        ".vg-incoming .vg-i-live{position:absolute;top:12px;right:12px;"
        "background:rgba(74,222,128,0.18);color:#4ade80;"
        "border:1px solid rgba(74,222,128,0.4);padding:2px 8px;"
        "border-radius:999px;font-size:13px;font-weight:800;letter-spacing:0.6px;}"
        ".vg-incoming .vg-i-timer-label{font-size:13px;letter-spacing:1.4px;text-transform:uppercase;"
        "color:#93c5fd;font-weight:800;margin-top:6px;}"
        ".vg-incoming .vg-i-timer{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;"
        "color:#fff;letter-spacing:-0.4px;line-height:1.1;}"

        # Waveform
        ".vg-wave{display:flex;align-items:center;gap:3px;margin-top:6px;height:26px;}"
        f".vg-wave .vg-bar{{width:4px;background:{ACCENT};border-radius:2px;"
        "animation:wave-anim 1s ease-in-out infinite;}"
        "@keyframes wave-anim{0%,100%{height:5px;}50%{height:22px;}}"

        # Verdict bar
        ".vg-verdict{border-radius:12px;padding:10px 18px;margin-bottom:7px;"
        "display:flex;justify-content:space-between;align-items:center;"
        "box-shadow:0 4px 14px rgba(15,23,42,0.08);}"
        f".vg-verdict.pass{{background:{PASS_BG};color:{PASS_TEXT};border:1px solid rgba(74,222,128,0.35);}}"
        f".vg-verdict.flag{{background:{FLAG_BG};color:{FLAG_TEXT};border:1px solid rgba(245,158,11,0.35);}}"
        f".vg-verdict.block{{background:{BLOCK_BG};color:{BLOCK_TEXT};border:1px solid rgba(248,113,113,0.4);}}"
        ".vg-verdict .vg-v-label{font-size:13px;letter-spacing:1.7px;text-transform:uppercase;opacity:0.85;font-weight:800;}"
        ".vg-verdict .vg-v-text{font-size:32px;font-weight:800;letter-spacing:-0.4px;color:#fff;line-height:1.1;}"
        ".vg-verdict .vg-v-sub{font-size:15px;font-weight:700;opacity:0.9;letter-spacing:0.3px;}"
        ".vg-verdict .vg-v-conf-label{font-size:13px;letter-spacing:1.6px;text-transform:uppercase;opacity:0.8;font-weight:800;text-align:right;}"
        ".vg-verdict .vg-v-conf{font-size:34px;font-weight:800;color:#fff;text-align:right;font-variant-numeric:tabular-nums;line-height:1.1;}"

        # Risk metric card
        f".vg-metric{{background:{PAPER};border:0.5px solid {BORDER};border-radius:12px;padding:9px 13px;height:100%;}}"
        f".vg-metric .vg-m-label{{font-size:13px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;color:{MUTED};}}"
        f".vg-metric .vg-m-value{{font-size:32px;font-weight:800;color:{INK};font-variant-numeric:tabular-nums;line-height:1;}}"
        f".vg-metric .vg-m-desc{{font-size:14px;color:{MUTED};line-height:1.4;margin-top:4px;font-weight:500;}}"
        ".vg-metric .vg-m-bar{height:4px;border-radius:999px;background:#e5e7eb;margin-top:6px;overflow:hidden;}"
        ".vg-metric .vg-m-bar>span{display:block;height:100%;border-radius:999px;}"

        # Badges
        ".vg-badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:1.1px;text-transform:uppercase;}"
        f".vg-badge.low{{color:{LOW_FG};background:{LOW_BG};}}"
        f".vg-badge.med{{color:{MED_FG};background:{MED_BG};}}"
        f".vg-badge.high{{color:{HIGH_FG};background:{HIGH_BG};}}"

        # Advisory cards
        f".vg-advisory{{background:{PAPER};border:0.5px solid {BORDER};border-radius:12px;padding:9px 14px;height:100%;}}"
        ".vg-advisory.pass-acc{border-left:4px solid #4ade80;}"
        ".vg-advisory.flag-acc{border-left:4px solid #f59e0b;}"
        ".vg-advisory.block-acc{border-left:4px solid #f87171;}"
        f".vg-advisory .vg-a-label{{font-size:13px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;color:{NAVY};}}"
        # Recommended action — colored per verdict
        ".vg-advisory.pass-acc .vg-a-action{background:#dcfce7;color:#14532d;border:1.5px solid #4ade80;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        ".vg-advisory.flag-acc .vg-a-action{background:#fef3c7;color:#78350f;border:1.5px solid #f59e0b;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        ".vg-advisory.block-acc .vg-a-action{background:#fee2e2;color:#7f1d1d;border:1.5px solid #f87171;font-weight:800;font-size:17px;border-radius:8px;padding:9px 13px;margin-top:4px;line-height:1.4;}"
        # OK to Do items — bold card chips
        f".vg-advisory .vg-ok-item{{display:flex;align-items:flex-start;gap:9px;background:#f8fafc;border:1px solid {BORDER};border-radius:8px;padding:8px 11px;margin-top:5px;font-size:15px;font-weight:700;color:{INK};line-height:1.4;}}"
        ".vg-advisory .vg-ok-icon{flex-shrink:0;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900;margin-top:1px;}"
        ".vg-advisory.pass-acc .vg-ok-icon{background:#dcfce7;color:#16a34a;}"
        ".vg-advisory.flag-acc .vg-ok-icon{background:#fef3c7;color:#d97706;}"
        ".vg-advisory.block-acc .vg-ok-icon{background:#fee2e2;color:#dc2626;}"
        f".vg-advisory .vg-a-narrative{{font-size:15px;color:{INK};line-height:1.5;margin-top:5px;font-weight:500;}}"
        ".vg-advisory .vg-a-pills{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;}"
        f".vg-advisory .vg-pill{{background:#f1f5f9;border-radius:999px;padding:3px 9px;font-size:13px;font-weight:700;color:{INK};letter-spacing:0.3px;}}"
        f".vg-advisory .vg-pill .vg-pill-k{{color:{MUTED};font-weight:600;}}"

        # Inputs
        f".stTextInput input,.stSelectbox > div > div{{border-radius:8px!important;border:1px solid {BORDER}!important;font-size:16px!important;}}"
        f".stTextInput input:focus{{border-color:{ACCENT}!important;box-shadow:0 0 0 2px rgba(55,138,221,0.18)!important;}}"

        # Button
        f".stButton > button{{background:{NAVY}!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:10px 18px!important;font-size:16px!important;font-weight:700!important;letter-spacing:0.4px!important;width:100%;}}"
        f".stButton > button:hover{{background:{NAVY_DEEP}!important;}}"

        # Path Taken pills
        ".vg-path{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}"
        ".vg-path .vg-step{flex:1;min-width:130px;padding:8px 10px;border-radius:999px;"
        "font-size:15px;font-weight:800;text-align:center;letter-spacing:0.3px;"
        "line-height:1.3;transition:all 0.2s ease;}"
        f".vg-path .vg-step.run{{background:{NAVY};color:#fff;border:1px solid {NAVY};}}"
        f".vg-path .vg-step.skip{{background:#f8fafc;color:{MUTED};border:1.5px dashed #cbd5e1;opacity:0.7;}}"
        f".vg-path .vg-step.active{{background:{ACCENT};color:#fff;border:1px solid {ACCENT};"
        "animation:stage-pulse 1.1s ease-in-out infinite;}"
        ".vg-path .vg-step.pending{background:#f1f5f9;color:#cbd5e1;border:1px solid #e2e8f0;}"
        "@keyframes stage-pulse{"
        "0%,100%{box-shadow:0 0 0 0 rgba(55,138,221,0.55);}"
        "50%{box-shadow:0 0 0 7px rgba(55,138,221,0);}"
        "}"

        # Audit metadata row (rendered as one HTML grid so it can update during animation)
        f".vg-meta-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:6px;}}"

        # Placeholder card (right column when call hasn't completed)
        ".vg-placeholder{background:#fff;border-radius:12px;padding:32px 28px;"
        "text-align:center;}"
        f".vg-placeholder.idle{{border:1px dashed #cbd5e1;}}"
        f".vg-placeholder.running{{border:1px dashed {ACCENT};}}"
        f".vg-placeholder .vg-p-title{{font-size:17px;font-weight:800;letter-spacing:0.2px;"
        f"color:{INK};margin-bottom:6px;}}"
        f".vg-placeholder.running .vg-p-title{{color:{NAVY};}}"
        f".vg-placeholder .vg-p-body{{font-size:15px;color:{MUTED};line-height:1.5;}}"
        ".vg-placeholder .vg-p-dots{display:inline-flex;gap:4px;margin-top:10px;}"
        f".vg-placeholder .vg-p-dot{{width:6px;height:6px;border-radius:50%;background:{ACCENT};"
        "animation:dot-bounce 1s ease-in-out infinite;}"
        ".vg-placeholder .vg-p-dot:nth-child(2){animation-delay:0.15s;}"
        ".vg-placeholder .vg-p-dot:nth-child(3){animation-delay:0.30s;}"
        "@keyframes dot-bounce{"
        "0%,80%,100%{transform:translateY(0);opacity:0.4;}"
        "40%{transform:translateY(-4px);opacity:1;}"
        "}"

        # Stage rows
        f".vg-stage-row{{display:flex;align-items:center;gap:12px;padding:7px 12px;margin-bottom:4px;border-radius:8px;border:1px solid {BORDER};background:{PAPER};}}"
        ".vg-stage-row.skip{background:#f8fafc;border:1px dashed #cbd5e1;opacity:0.78;}"
        ".vg-stage-row .vg-s-num{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:800;flex-shrink:0;}"
        f".vg-stage-row.run .vg-s-num{{background:{NAVY};color:#fff;}}"
        f".vg-stage-row.skip .vg-s-num{{background:#e2e8f0;color:{MUTED};}}"
        ".vg-stage-row .vg-s-body{flex:1;min-width:0;}"
        ".vg-stage-row .vg-s-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}"
        f".vg-stage-row .vg-s-name{{font-size:16px;font-weight:800;color:{INK};}}"
        f".vg-stage-row .vg-s-tag{{font-size:12px;color:{MUTED};letter-spacing:1px;text-transform:uppercase;font-weight:700;}}"
        ".vg-stage-row .vg-s-status{font-size:12px;padding:2px 7px;border-radius:999px;letter-spacing:1px;text-transform:uppercase;font-weight:800;}"
        f".vg-stage-row.run .vg-s-status{{background:{LOW_BG};color:{LOW_FG};}}"
        f".vg-stage-row.skip .vg-s-status{{background:#e2e8f0;color:{MUTED};}}"
        f".vg-stage-row .vg-s-desc{{font-size:15px;color:{INK};margin-top:2px;line-height:1.4;font-weight:500;}}"
        f".vg-stage-row.skip .vg-s-desc{{color:{MUTED};}}"
        f".vg-stage-row .vg-s-score{{font-size:15px;font-weight:800;color:{INK};font-variant-numeric:tabular-nums;flex-shrink:0;min-width:140px;text-align:right;}}"
        f".vg-stage-row.skip .vg-s-score{{color:{MUTED};}}"
        ".vg-stage-row.pending{background:#f8fafc;border:1px solid #e2e8f0;opacity:0.7;}"
        ".vg-stage-row.pending .vg-s-num{background:#e2e8f0;color:#cbd5e1;}"
        ".vg-stage-row.pending .vg-s-status{background:#e2e8f0;color:#94a3b8;}"
        f".vg-stage-row.pending .vg-s-desc{{color:{MUTED};}}"
        f".vg-stage-row.pending .vg-s-score{{color:#cbd5e1;}}"
        f".vg-stage-row.active{{background:{PAPER};border:1px solid {ACCENT};box-shadow:0 0 0 3px rgba(55,138,221,0.15);}}"
        f".vg-stage-row.active .vg-s-num{{background:{ACCENT};color:#fff;animation:stage-pulse 1.1s ease-in-out infinite;}}"
        ".vg-stage-row.active .vg-s-status{background:#dbeafe;color:#1d4ed8;}"

        # Audit metadata
        f".vg-meta{{background:{PAPER};border:0.5px solid {BORDER};border-radius:12px;padding:8px 13px;height:100%;}}"
        f".vg-meta .vg-meta-l{{font-size:12px;letter-spacing:1.3px;text-transform:uppercase;color:{MUTED};font-weight:800;}}"
        f".vg-meta .vg-meta-v{{font-size:24px;font-weight:800;color:{INK};font-variant-numeric:tabular-nums;margin-top:3px;line-height:1.1;}}"
        f".vg-meta .vg-meta-s{{font-size:14px;color:{MUTED};margin-top:3px;font-weight:600;}}"

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
        '<div class="vg-incoming">'
        '<div class="vg-i-live">● LIVE</div>'
        '<div class="vg-i-label">Incoming Call</div>'
        f'<div class="vg-i-num">{caller_id}</div>'
        '<div class="vg-i-sub">Chase Fraud Helpline · IVR</div>'
        '<div class="vg-i-timer-label">Audio Duration</div>'
        f'<div class="vg-i-timer">{timer_value:0.1f}s</div>'
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
