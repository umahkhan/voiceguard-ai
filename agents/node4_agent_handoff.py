"""Node 4 — Agent Handoff Agent (Defense 3).

Journey Stage 4: call reaches a live agent. Defense 3 fires: agent alert
fires with a confidence score and the agent is trained with a step-up auth
requirement.
"""

from __future__ import annotations

from state import VoiceGuardState

from ._anthropic import call_claude


def agent_handoff_agent(state: VoiceGuardState) -> VoiceGuardState:
    entry_conf = float(state.get("ivr_entry_confidence", 0.0))
    nav_score = float(state.get("ivr_nav_anomaly_score", entry_conf))

    # If Stage 3 was skipped (fast-track from high-confidence Stage 2) we still
    # weight the entry confidence heavier — it is the strongest signal we have.
    agent_confidence = round(0.5 * entry_conf + 0.5 * nav_score, 3)
    step_up = agent_confidence >= 0.5

    caller = state.get("caller_id", "UNKNOWN")
    prompt = (
        "You are VoiceGuard generating a JPMorgan Chase contact-center alert for the "
        "live human agent. Write ONE short, action-oriented alert (max 45 words). "
        "Lead with the ⚠️ VOICEGUARD ALERT prefix. Include AI confidence and whether "
        "step-up authentication is required. Context: "
        f"caller_id={caller}, ai_confidence={agent_confidence:.2f}, step_up_required={step_up}."
    )
    try:
        alert_message = call_claude(prompt, max_tokens=200)
    except Exception as exc:  # pragma: no cover
        alert_message = f"[alert generation failed: {exc}]"

    log = (
        f"[Stage 4 — Defense 3] Agent alerted. Confidence: {agent_confidence:.3f}. "
        f"Step-up auth: {step_up}."
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "agent_alert_fired": True,
        "agent_confidence_score": agent_confidence,
        "step_up_auth_triggered": step_up,
        "alert_message": alert_message,
        "journey_trace": trace,
    }
