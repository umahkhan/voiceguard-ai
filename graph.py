"""LangGraph wiring for the VoiceGuard pipeline with human-in-the-loop.

Topology:

    START
      │
      ▼
    [Node 1: Voice Cloning Detector — synthesis + speaker signals]
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
                       ▼
              [Human Review]  ◄──── INTERRUPT (graph pauses here)
                       │
              ┌────────┼────────────┐
              │        │            │
         approve     stepup        block
              │        │            │
              │        ▼            │
              │  [Auth Challenge]   │
              │        │            │
              ▼        ▼            ▼
              [Node 7: Intelligence — leadership summary]
                       │
                       ▼
                      END

The graph compiles with `interrupt_before=["human_review"]` and a
MemorySaver checkpointer, so each call gets a thread_id and the
pause/resume cycle is durable across the dashboard's reruns.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.human_review import human_review_agent
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


def _route_after_human_review(state: VoiceGuardState) -> str:
    """Reviewer's decision controls the downstream path."""
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
    g.add_node("human_review", human_review_agent)
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
    g.add_edge("agent_handoff", "human_review")
    g.add_conditional_edges(
        "human_review",
        _route_after_human_review,
        {"auth_challenge": "auth_challenge", "intelligence": "intelligence"},
    )
    g.add_edge("auth_challenge", "intelligence")
    g.add_edge("intelligence", END)

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"],
    )
