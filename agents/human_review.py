"""Human-in-the-loop review node.

The LangGraph pipeline pauses *before* this node via `interrupt_before`.
The reviewer makes a decision in the dashboard (Approve / Step-Up / Block),
which writes `human_decision` into state. When the graph resumes, this
node executes — recording the decision in the journey trace — and
downstream conditional routing branches based on `human_decision`:

  approve → intelligence (case cleared, no further checks)
  stepup  → auth_challenge → intelligence
  block   → intelligence (case marked blocked)
"""

from __future__ import annotations

from state import VoiceGuardState


def human_review_agent(state: VoiceGuardState) -> VoiceGuardState:
    decision = state.get("human_decision", "pending")
    if decision == "block":
        verdict = "BLOCK"
        transaction_blocked = True
    elif decision == "approve":
        verdict = state.get("verdict", "PASS")
        transaction_blocked = False
    else:  # stepup — verdict updated by auth_challenge based on OTP outcome
        verdict = state.get("verdict", "FLAG")
        transaction_blocked = False

    log = f"[Human Review] Reviewer decision: {decision}."
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "human_review_completed": True,
        "verdict": verdict,
        "transaction_blocked": transaction_blocked,
        "journey_trace": trace,
    }
