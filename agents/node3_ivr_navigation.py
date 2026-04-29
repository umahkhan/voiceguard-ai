"""Node 3 — IVR Navigation Agent (Defense 2).

Journey Stage 3: bot attempts account access via IVR menu navigation. Defense
2 fires: behavioral pattern analysis detects anomalies in navigation timing
and phrasing.
"""

from __future__ import annotations

import random

from state import VoiceGuardState

from ._anthropic import call_claude


def ivr_navigation_agent(state: VoiceGuardState) -> VoiceGuardState:
    entry_conf = float(state.get("ivr_entry_confidence", 0.0))

    # Correlate with entry confidence + small noise so the signals are coherent.
    noise = random.uniform(-0.1, 0.1)
    anomaly_score = round(max(0.0, min(1.0, entry_conf + noise)), 3)

    # Bot-like timing: sub-200ms menu responses.
    avg_response_ms = max(80, int(800 * (1.0 - anomaly_score) + random.randint(-50, 50)))
    timing_flag = avg_response_ms < 200 or anomaly_score > 0.55

    prompt = (
        "You are VoiceGuard's behavioral analyzer for IVR navigation. In ONE sentence "
        "(max 35 words) summarize the navigation anomaly. Compare timing to a 600ms human "
        f"baseline and note phrasing cadence. Context: avg_response_ms={avg_response_ms}, "
        f"anomaly_score={anomaly_score:.2f}, timing_flagged={timing_flag}."
    )
    try:
        behavioral_summary = call_claude(prompt, max_tokens=160)
    except Exception as exc:  # pragma: no cover
        behavioral_summary = f"[nav summary failed: {exc}]"

    log = (
        f"[Stage 3 — Defense 2] Nav anomaly score: {anomaly_score:.3f}. "
        f"Timing flagged: {timing_flag} ({avg_response_ms}ms avg). "
        f"Phrasing anomalies: {behavioral_summary}"
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "ivr_nav_anomaly_score": anomaly_score,
        "navigation_timing_flag": timing_flag,
        "journey_trace": trace,
    }
