"""Wav2Vec2-based voice-cloning detector.

Two honest signals, both computed fresh per audio file:

  * `spectral_score` — P(synthetic) directly from a Wav2Vec2 deepfake
    classifier. Raw model probability, no calibration applied — what
    you see is what the model says. Production deployments may want
    a cost-aware calibration on top (the FP/FN cost asymmetry is
    workload-specific) but for the demo we surface the model's actual
    output so the scores are interpretable as probabilities.
  * `prosody_score`  — F0-contour anomaly from librosa.yin. Synthetic
    speech tends to have unnaturally regular pitch; we measure the
    coefficient of variation of voiced-frame F0 and invert.
"""

from __future__ import annotations

import functools
from pathlib import Path

# Switched from MelodyMachine/Deepfake-audio-detection to motheecreator's
# variant — same Wav2Vec2 architecture but trained on a more diverse corpus
# that includes modern neural TTS samples. The MelodyMachine checkpoint
# (ASVspoof2021-trained) returned fake_prob=0.00 for current ElevenLabs
# clones; motheecreator returns fake_prob=0.99+ on the same files while
# still scoring real human speech as fake_prob=0.00.
MODEL_NAME = "motheecreator/Deepfake-audio-detection"

_FAKE_LABEL_KEYS = ("fake", "spoof", "deepfake", "synthetic", "ai", "bonafide")


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline
    # Force CPU. MPS (Apple Silicon) hits a conv1d output-channel cap on
    # this model; CUDA isn't expected on Streamlit Cloud either.
    return pipeline("audio-classification", model=MODEL_NAME, device=-1)


def _extract_fake_prob(results: list[dict]) -> float:
    fake_score = None
    real_score = None
    for r in results:
        label = str(r.get("label", "")).lower()
        score = float(r.get("score", 0.0))
        if "bonafide" in label or "real" in label or "human" in label:
            real_score = score
        elif any(k in label for k in ("fake", "spoof", "deepfake", "synthetic", "ai")):
            fake_score = score
    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return float(results[0].get("score", 0.5))


@functools.lru_cache(maxsize=64)
def _spectral_score(audio_path: str) -> float:
    import librosa

    pipe = _load_pipeline()
    sr = int(pipe.feature_extractor.sampling_rate)
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    out = pipe({"raw": y, "sampling_rate": sr}, top_k=5)
    return _extract_fake_prob(out)


@functools.lru_cache(maxsize=64)
def _prosody_score(audio_path: str) -> float:
    """F0-contour anomaly: flatter pitch contour → higher score.

    Conversational human speech typically lands in CV 0.20–0.40 on natural
    recordings (mp3 codec adds jitter that nudges this up). Older TTS
    (concatenative, formant) is much flatter (CV ~0.10–0.25). Modern
    neural TTS (ElevenLabs, Tortoise, XTTS) deliberately injects
    natural-looking variation, so this signal alone won't catch them —
    that's what the Wav2Vec2 spectral model is for. Prosody is a
    secondary check that flags obvious / older synthesis cheaply.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    voiced = f0[(f0 > 60) & (f0 < 400)]
    if len(voiced) < 20:
        return 0.5
    cv = float(np.std(voiced) / (np.mean(voiced) + 1e-6))
    # Map: CV ≤ 0.08 → 1.0 (very flat). CV ≥ 0.30 → 0.0 (natural).
    return float(np.clip((0.30 - cv) / 0.22, 0.0, 1.0))


@functools.lru_cache(maxsize=64)
def _duration(audio_path: str) -> float:
    import librosa
    return float(librosa.get_duration(path=audio_path))


def detect(audio_path: str | Path) -> dict:
    """Run real-model detection on a single audio file.

    Returns the raw model probability for spectral_score (no calibration
    applied) plus the librosa F0 prosody score. What you see is what the
    model says.
    """
    p = str(audio_path)
    spectral = _spectral_score(p)
    return {
        "raw_audio_duration_sec": round(_duration(p), 2),
        "spectral_score": round(spectral, 3),
        "prosody_score": round(_prosody_score(p), 3),
        "model_used": MODEL_NAME,
    }
