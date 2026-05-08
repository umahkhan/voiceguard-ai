"""VoiceGuardState — single typed state dict passed through the LangGraph pipeline.

Mirrors the 5-stage Future-State Journey Map: every field corresponds to data
captured or computed at one of the attacker's touchpoints (baseline capture,
IVR entry, IVR navigation, agent handoff, auth challenge) plus the final
intelligence layer.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class VoiceGuardState(TypedDict, total=False):
    # Identifiers
    caller_id: str
    account_number: str

    # Stage 1 — Voice Cloning Detector / baseline capture
    raw_audio_duration_sec: float
    baseline_collected: bool
    spectral_score: float
    prosody_score: float
    audio_path: str
    live_mode: bool
    model_used: str
    registered_voice_path: str
    speaker_similarity: float

    # Stage 2 — IVR Entry Agent (Defense 1)
    ivr_entry_confidence: float

    # Stage 3 — IVR Navigation Agent (Defense 2)
    ivr_nav_anomaly_score: float
    navigation_timing_flag: bool

    # Stage 4 — Agent Handoff Agent (Defense 3)
    agent_alert_fired: bool
    agent_confidence_score: float
    step_up_auth_triggered: bool
    alert_message: str

    # Stage 5 — Auth Challenge Agent (Defense 4)
    otp_sent: bool
    otp_completed: bool

    # Final outcome
    verdict: Literal["PASS", "FLAG", "BLOCK"]
    transaction_blocked: bool

    # Intelligence layer
    intelligence_log: dict[str, Any]

    # Cross-stage observability
    journey_trace: list[str]
