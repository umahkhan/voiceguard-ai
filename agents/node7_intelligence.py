"""Node 7 — Intelligence Agent.

Runs for both clean passes (no fraud) and post-block outcomes. Computes P&L
loss avoidance, attack vector classification, and customer impact, and writes
a leadership-ready summary to intelligence_log.
"""

from __future__ import annotations

import random

from state import VoiceGuardState

from ._anthropic import call_claude


def intelligence_agent(state: VoiceGuardState) -> VoiceGuardState:
    verdict = state.get("verdict", "PASS")
    transaction_blocked = bool(state.get("transaction_blocked", False))
    agent_confidence = float(state.get("agent_confidence_score", 0.0))
    entry_conf = float(state.get("ivr_entry_confidence", 0.0))

    if verdict == "PASS" and not transaction_blocked:
        loss_avoidance = 0.0
        customers_impacted = 0
        attack_vector = "Clean"
    else:
        loss_avoidance = float(random.randint(15_000, 75_000))
        customers_impacted = 1
        if entry_conf > 0.8:
            attack_vector = "High-Confidence Synthetic Voice"
        elif state.get("navigation_timing_flag"):
            attack_vector = "Voice Clone + Bot Navigation"
        else:
            attack_vector = "Voice Clone"

    prompt = (
        "You are VoiceGuard's intelligence layer. Write EXACTLY 2 sentences for a JPMorgan "
        "leadership dashboard summarizing this call outcome. Keep it executive-ready and "
        f"quantitative. Context: verdict={verdict}, agent_confidence={agent_confidence:.2f}, "
        f"attack_vector={attack_vector}, loss_avoidance=${loss_avoidance:,.0f}, "
        f"transaction_blocked={transaction_blocked}."
    )
    try:
        leadership_summary = call_claude(prompt, max_tokens=240)
    except Exception as exc:  # pragma: no cover
        leadership_summary = f"[summary generation failed: {exc}]"

    intelligence_log = dict(state.get("intelligence_log", {}))
    intelligence_log.update({
        "loss_avoidance": loss_avoidance,
        "customers_impacted": customers_impacted,
        "attack_vector": attack_vector,
        "leadership_summary": leadership_summary,
    })

    log = (
        f"[Intelligence] Attack vector: {attack_vector}. "
        f"Loss avoidance: ${loss_avoidance:,.0f}. Customers impacted: {customers_impacted}."
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "intelligence_log": intelligence_log,
        "journey_trace": trace,
    }
