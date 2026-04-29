"""LangGraph wiring for the 5-stage VoiceGuard journey-mapped pipeline.

Topology (matches the Future-State Journey Map slide):

    START
      │
      ▼
    [Node 1: Voice Cloning Detector]
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
                       │
              ┌────────┴────────┐
              │ step_up=True    │ step_up=False
              ▼                 ▼
       [Node 5: Auth        [Node 7: Intelligence]
        Challenge — D4]
              │                 │
              ▼                 │
       [Node 7: Intel] ◄────────┘
              │
              ▼
             END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.node1_voice_cloning import voice_cloning_detector
from agents.node2_ivr_entry import ivr_entry_agent
from agents.node3_ivr_navigation import ivr_navigation_agent
from agents.node4_agent_handoff import agent_handoff_agent
from agents.node5_auth_challenge import auth_challenge_agent
from agents.node7_intelligence import intelligence_agent
from state import VoiceGuardState


def _route_after_ivr_entry(state: VoiceGuardState) -> str:
    """Fast-track high-confidence synthetic callers straight to the agent alert."""
    if float(state.get("ivr_entry_confidence", 0.0)) > 0.8:
        return "agent_handoff"
    return "nav_analysis"


def _route_after_handoff(state: VoiceGuardState) -> str:
    """Step-up auth required → OTP challenge. Else → clean pass to intelligence."""
    if state.get("step_up_auth_triggered"):
        return "auth_challenge"
    return "intelligence"


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
        {
            "nav_analysis": "nav_analysis",
            "agent_handoff": "agent_handoff",
        },
    )
    g.add_edge("nav_analysis", "agent_handoff")

    g.add_conditional_edges(
        "agent_handoff",
        _route_after_handoff,
        {
            "auth_challenge": "auth_challenge",
            "intelligence": "intelligence",
        },
    )
    g.add_edge("auth_challenge", "intelligence")
    g.add_edge("intelligence", END)

    return g.compile()
