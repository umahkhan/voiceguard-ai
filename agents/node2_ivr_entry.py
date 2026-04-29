"""Node 2 — IVR Entry Agent (Defense 1).

Journey Stage 2: synthetic voice enters the IVR system. Defense 1 fires:
first 50ms voice analysis runs and a confidence score is computed. Calls with
high entry confidence are flagged and routed to a specialist queue; everyone
else goes to standard with continued monitoring.
"""

from __future__ import annotations

from state import VoiceGuardState

from ._anthropic import call_claude


def ivr_entry_agent(state: VoiceGuardState) -> VoiceGuardState:
    spectral = float(state.get("spectral_score", 0.0))
    prosody = float(state.get("prosody_score", 0.0))
    confidence = round(0.6 * spectral + 0.4 * prosody, 3)

    prompt = (
        "You are VoiceGuard's IVR entry analyzer. Write ONE short internal system note "
        "(max 25 words) describing the AI-voice probability and the next routing action. "
        f"Context: AI voice probability={confidence:.2f}. "
        "Example tone: 'IVR Entry: AI voice probability 0.74. Flagging for behavioral analysis.'"
    )
    try:
        system_note = call_claude(prompt, max_tokens=120)
    except Exception as exc:  # pragma: no cover — surface failure in trace
        system_note = f"[ivr entry note failed: {exc}]"

    if confidence > 0.5:
        verdict = "FLAG"
        queue = "specialist"
    else:
        verdict = state.get("verdict", "PASS")
        queue = "standard"

    log = (
        f"[Stage 2 — Defense 1] IVR entry confidence: {confidence:.3f}. "
        f"Routed to: {queue}. Note: {system_note}"
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "ivr_entry_confidence": confidence,
        "verdict": verdict,
        "journey_trace": trace,
    }
