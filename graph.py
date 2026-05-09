"""LangGraph wiring for the VoiceGuard pipeline with human-in-the-loop.

Topology:

    START
      │
      ▼
    [Node 1: Voice Cloning Detector — Wav2Vec2 + ECAPA + librosa F0]
      │
      ▼
    [Node 2: IVR Entry — Defense 1]
      │
      ├── ivr_entry_confidence > 0.8 ─────────────────────────┐
      │                                                       │
      └── ≤ 0.8 → [Node 3: IVR Nav — Defense 2]               │
                       │                                      │
                       ▼                                      ▼
              [Node 4: Agent Handoff — Defense 3] ◄──────────┘
                  • computes agent_confidence
                  • generates alert message
                  • THIS IS the human-review pause point
                       │
                       │  ┌── INTERRUPT_AFTER ──┐
                       │  │  Graph pauses here. │
                       │  │  Dashboard reads    │
                       │  │  state, reviewer    │
                       │  │  picks a decision,  │
                       │  │  resume fires.      │
                       │  └─────────────────────┘
                       │
              ┌────────┼─────────────────┐
              │ stepup │ approve / block │
              ▼        ▼                 │
       [Node 5: Auth   │                 │
        Challenge — D4]│                 │
              │        ▼                 ▼
              └─►  [Node 7: Intelligence — leadership summary]
                       │
                       ▼
                      END

`agent_handoff` was previously paired with a separate `human_review` node;
they did the same job (one fired the alert, the other was a pause point).
We merge them by using LangGraph's `interrupt_after` rather than
`interrupt_before`: agent_handoff runs first (so the alert message and
agent_confidence are populated and visible to the reviewer), THEN the
graph pauses, awaiting the reviewer's decision.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.node1_voice_cloning import voice_cloning_detector
from agents.node2_ivr_entry import ivr_entry_agent
from agents.node3_ivr_navigation import ivr_navigation_agent
from agents.node4_agent_handoff import agent_handoff_agent
from agents.node5_auth_challenge import auth_challenge_agent
from agents.node7_intelligence import intelligence_agent
from state import VoiceGuardState


def _route_after_ivr_entry(state: VoiceGuardState) -> str:
    if float(state.get("ivr_entry_confidence", 0.0)) > 0.8:
        return "agent_handoff"
    return "nav_analysis"


def _route_after_agent_handoff(state: VoiceGuardState) -> str:
    """Reviewer's decision (set by the dashboard via update_state before
    resuming) controls the downstream path."""
    decision = state.get("human_decision", "approve")
    if decision == "stepup":
        return "auth_challenge"
    return "intelligence"  # both approve and block terminate via intelligence


def build_graph():
    g = StateGraph(VoiceGuardState)

    g.add_node("voice_cloning", voice_cloning_detector)
    g.add_node("ivr_entry", ivr_entry_agent)
    g.add_node("nav_analysis", ivr_navigation_agent)
    g.add_node("agent_handoff", agent_handoff_agent)
    g.add_node("auth_challenge", auth_challenge_agent)
    g.add_node("intelligence", intelligence_agent)

    g.add_edge(START, "voice_cloning")
    g.add_edge("voice_cloning", "ivr_entry")
    g.add_conditional_edges(
        "ivr_entry",
        _route_after_ivr_entry,
        {"nav_analysis": "nav_analysis", "agent_handoff": "agent_handoff"},
    )
    g.add_edge("nav_analysis", "agent_handoff")
    g.add_conditional_edges(
        "agent_handoff",
        _route_after_agent_handoff,
        {"auth_challenge": "auth_challenge", "intelligence": "intelligence"},
    )
    g.add_edge("auth_challenge", "intelligence")
    g.add_edge("intelligence", END)

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_after=["agent_handoff"],  # pause AFTER alert is computed
    )
