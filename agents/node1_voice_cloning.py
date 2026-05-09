"""Node 1 — Voice Cloning Detector (Baseline Collection).

Journey Stage 1: attacker acquires 3–5 seconds of audio. VoiceGuard layer
begins baseline data collection. No routing decision is made here — this node
captures the signals that downstream defenses will use.

Two scoring paths:
  * Live mode: real Wav2Vec2 deepfake detector + librosa F0 prosody (see
    `detectors.voice_clone`). Used when an audio file is supplied and
    `live_mode` is True in state.
  * Scripted mode: hand-tuned scores already on state (default for the
    JPM walkthrough scenarios). If neither is available, fall back to a
    randomized baseline so the pipeline still produces a verdict.
"""

from __future__ import annotations

import random

from state import VoiceGuardState


def voice_cloning_detector(state: VoiceGuardState) -> VoiceGuardState:
    audio_path = state.get("audio_path")
    live_mode = bool(state.get("live_mode"))

    spectral = state.get("spectral_score")
    prosody = state.get("prosody_score")
    duration = state.get("raw_audio_duration_sec")
    model_used = "scripted"

    if live_mode and audio_path:
        from detectors import detect

        result = detect(audio_path)
        spectral = result["spectral_score"]
        prosody = result["prosody_score"]
        duration = result["raw_audio_duration_sec"]
        model_used = result["model_used"]

    if duration is None:
        duration = round(random.uniform(3.0, 5.0), 2)
    if spectral is None:
        spectral = round(random.uniform(0.2, 0.9), 3)
    if prosody is None:
        prosody = round(random.uniform(0.2, 0.9), 3)

    log = (
        f"[Stage 1] Baseline collected ({model_used}). "
        f"Audio: {float(duration):.2f}s. "
        f"Spectral: {float(spectral):.3f}. Prosody: {float(prosody):.3f}."
    )
    trace = list(state.get("journey_trace", []))
    trace.append(log)

    return {
        "raw_audio_duration_sec": float(duration),
        "baseline_collected": True,
        "spectral_score": float(spectral),
        "prosody_score": float(prosody),
        "model_used": model_used,
        "journey_trace": trace,
    }
