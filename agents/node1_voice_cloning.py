"""Node 1 — Voice Cloning Detector (Baseline Collection).

Journey Stage 1: attacker acquires 3–5 seconds of audio. VoiceGuard layer
begins baseline data collection. No routing decision is made here — this node
captures the signals that downstream defenses will use.
"""

from __future__ import annotations

import random

from state import VoiceGuardState


def voice_cloning_detector(state: VoiceGuardState) -> VoiceGuardState:
    duration = float(state.get("raw_audio_duration_sec") or round(random.uniform(3.0, 5.0), 2))

    spectral = state.get("spectral_score")
    prosody = state.get("prosody_score")
    if spectral is None:
        spectral = round(random.uniform(0.2, 0.9), 3)
    if prosody is None:
        prosody = round(random.uniform(0.2, 0.9), 3)

    log = (
        f"[Stage 1] Baseline collected. Audio: {duration:.2f}s. "
        f"Spectral: {spectral:.3f}. Prosody: {prosody:.3f}."
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "raw_audio_duration_sec": duration,
        "baseline_collected": True,
        "spectral_score": float(spectral),
        "prosody_score": float(prosody),
        "journey_trace": trace,
    }
