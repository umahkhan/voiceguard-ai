"""Node 5 — Auth Challenge Agent (Defense 4).

Journey Stage 5: OTP sent to the customer's real device. Defense 4 fires: the
attacker cannot complete the OTP, so the transaction is blocked.
"""

from __future__ import annotations

from state import VoiceGuardState

from ._anthropic import call_claude


def auth_challenge_agent(state: VoiceGuardState) -> VoiceGuardState:
    agent_confidence = float(state.get("agent_confidence_score", 0.0))

    otp_sent = True
    # High-confidence synthetic callers fail OTP because they don't possess the
    # real customer's device. Below threshold: legitimate caller passes.
    otp_completed = agent_confidence <= 0.65

    if not otp_completed:
        verdict = "BLOCK"
        transaction_blocked = True
    else:
        verdict = "PASS"
        transaction_blocked = False

    prompt = (
        "You are VoiceGuard's intelligence layer. In ONE sentence (max 30 words) summarize "
        "the auth-challenge outcome for the case log. Context: "
        f"otp_completed={otp_completed}, transaction_blocked={transaction_blocked}, "
        f"agent_confidence={agent_confidence:.2f}."
    )
    try:
        outcome_summary = call_claude(prompt, max_tokens=140)
    except Exception as exc:  # pragma: no cover
        outcome_summary = f"[outcome summary failed: {exc}]"

    intelligence_log = dict(state.get("intelligence_log", {}))
    intelligence_log["auth_outcome_summary"] = outcome_summary

    log = (
        f"[Stage 5 — Defense 4] OTP sent. Completed: {otp_completed}. "
        f"Transaction blocked: {transaction_blocked}."
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "otp_sent": otp_sent,
        "otp_completed": otp_completed,
        "verdict": verdict,
        "transaction_blocked": transaction_blocked,
        "intelligence_log": intelligence_log,
        "journey_trace": trace,
    }
